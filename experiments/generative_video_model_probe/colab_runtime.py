"""在 Colab GPU 环境中运行 generative_video_model_probe 生成式视频模型探测。"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import time
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

from external_baseline.baseline_registry import audit_external_baseline_records
from experiments.generative_video_model_probe.external_baseline_runner import run_external_baseline_status
from main.methods.state_space_watermark.endpoint_latent_detector import compute_endpoint_latent_evidence
from main.methods.state_space_watermark.authenticated_trajectory_sketch import (
    build_authenticated_trajectory_sketch_payload,
    sign_authenticated_trajectory_sketch,
)
from main.methods.state_space_watermark.flow_velocity_runtime import FlowVelocityConstraintRuntime
from experiments.generative_video_model_probe.formal_method_variants import (
    FORMAL_METHOD_VARIANTS,
    GENERATION_METHOD_VARIANTS,
    velocity_runtime_mechanism_for_method_variant,
)
from main.methods.state_space_watermark.flow_latent_layout import FiveDimensionalFlowLatentLayout
from main.methods.state_space_watermark.ltx_flow_replay_backend import build_ltx_latent_layout
from main.methods.state_space_watermark.path_observation import aggregate_path_observations
from main.methods.state_space_watermark.watermark_key_derivation import (
    WATERMARK_KEY_DERIVATION_ID,
    derive_watermark_key_text,
)
from evaluation.protocol.flow_evidence_fields import (
    with_flow_evidence_protocol_defaults,
    with_flow_evidence_protocol_defaults_many,
)
from runtime.core.progress import (
    ProgressReporter,
    configure_noisy_library_progress,
    configure_pipeline_progress_bar,
    emit_progress_event,
    suppress_third_party_progress_output,
)
from evaluation.protocol.record_writer import write_json, write_jsonl
from evaluation.protocol.table_builder import write_csv


WAN21_PRIMARY_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
LTX_VIDEO_CROSS_MODEL_ID = "Lightricks/LTX-Video"

REGISTERED_GENERATION_MODEL_FAMILIES: dict[str, dict[str, str]] = {
    WAN21_PRIMARY_MODEL_ID: {
        "generation_model_family": "diffusers_wan21_flow_matching_dit",
        "pipeline_class_name": "WanPipeline",
        "scheduler_id": "wan21_flow_matching_pipeline_default_scheduler",
        "trajectory_time_grid_id": "wan_flow_scheduler_runtime_step_grid",
    },
    LTX_VIDEO_CROSS_MODEL_ID: {
        "generation_model_family": "diffusers_ltx_video",
        "pipeline_class_name": "LTXPipeline",
        "scheduler_id": "ltx_pipeline_default_scheduler",
        "trajectory_time_grid_id": (
            "ltx_sequence_length_shifted_flow_scheduler_runtime_step_grid"
        ),
    },
}

_HUGGINGFACE_COMMIT_PATTERN = re.compile(r"^[0-9a-fA-F]{40,64}$")

PAPER_RESULT_PROFILES = {"probe_paper", "pilot_paper", "full_paper"}

PAPER_FORMAL_METHOD_VARIANTS = FORMAL_METHOD_VARIANTS
PROFILE_SETTINGS = {
    "pilot_paper": {
        "prompt_limit": None,
        "seed_limit": None,
        "num_inference_steps": 16,
        "num_frames": 49,
        "height": 320,
        "width": 512,
        "run_cross_model": True,
        "cross_model_prompt_limit": 10,
        "cross_model_seed_limit": 6,
        "prompt_suite_roles": ["pilot_paper"],
        "seed_suite_roles": ["pilot_paper"],
    },
    "probe_paper": {
        "prompt_limit": None,
        "seed_limit": None,
        "num_inference_steps": 16,
        "num_frames": 49,
        "height": 320,
        "width": 512,
        "run_cross_model": True,
        "cross_model_prompt_limit": 5,
        "cross_model_seed_limit": 4,
        "prompt_suite_roles": ["probe_paper"],
        "seed_suite_roles": ["probe_paper"],
    },
    "full_paper": {
        "prompt_limit": None,
        "seed_limit": None,
        "num_inference_steps": 16,
        "num_frames": 49,
        "height": 320,
        "width": 512,
        "run_cross_model": True,
        "cross_model_prompt_limit": 20,
        "cross_model_seed_limit": 10,
        "prompt_suite_roles": ["full_paper"],
        "seed_suite_roles": ["full_paper"],
    },
    "motion_calibration": {
        "prompt_limit": None,
        "seed_limit": None,
        "num_inference_steps": 16,
        "num_frames": 49,
        "height": 320,
        "width": 512,
        "run_cross_model": False,
        "prompt_suite_roles": [
            "motion_calibration_negative_static",
            "motion_calibration_positive_motion",
            "motion_calibration_ambiguous_low_motion",
        ],
        "seed_suite_roles": ["motion_calibration"],
    },
}


def _read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _select_dtype(torch_module: Any) -> Any:
    """根据 Colab GPU 能力选择 dtype, T4 默认使用 float16。"""
    major, _minor = torch_module.cuda.get_device_capability(0)
    return torch_module.bfloat16 if major >= 8 else torch_module.float16


def _tensor_stats(value: Any) -> dict:
    """从 latent tensor 中提取轻量 trajectory 统计量。"""
    try:
        detached = value.detach().float()
        return {
            "latent_norm": round(float(detached.norm().item()), 6),
            "latent_mean": round(float(detached.mean().item()), 6),
            "latent_std": round(float(detached.std().item()), 6),
        }
    except Exception as exc:  # pragma: no cover - 仅在 Colab pipeline callback 中触发
        return {"latent_norm": None, "latent_mean": None, "latent_std": None, "trajectory_capture_failure_reason": str(exc)}


def _model_family_from_id(model_id: str) -> str:
    """返回显式注册的生成模型家族, 未知 ID 必须立即失败。"""

    normalized = str(model_id).strip()
    try:
        return REGISTERED_GENERATION_MODEL_FAMILIES[normalized][
            "generation_model_family"
        ]
    except KeyError as exc:
        registered = ", ".join(sorted(REGISTERED_GENERATION_MODEL_FAMILIES))
        raise ValueError(
            f"未注册的 generation model ID: {normalized!r}; 仅允许: {registered}"
        ) from exc


def _registered_generation_model(model_id: str) -> dict[str, str]:
    """读取一个不可变副本, 供 pipeline、scheduler 和时间网格共同复用。"""

    _model_family_from_id(model_id)
    return dict(REGISTERED_GENERATION_MODEL_FAMILIES[str(model_id).strip()])


def _resolve_generation_model_commit(
    model_id: str,
    *,
    requested_revision: str | None,
    hf_token: str | None,
) -> tuple[str, str]:
    """把配置 revision 解析为 Hugging Face 不可变 commit。

    正式生成始终把解析后的 commit 传给 `from_pretrained`, 防止生成与 replay
    在不同时间解析到不同模型内容。只有显式给出的不可变 commit 可以在 Hub
    元数据暂时不可用时离线使用; branch 或 tag 无法证明不可变性, 因而失败。
    """

    _registered_generation_model(model_id)
    configured_revision = str(requested_revision or "").strip() or None
    try:
        from huggingface_hub import model_info

        info = model_info(
            model_id,
            revision=configured_revision,
            token=hf_token,
        )
        resolved_commit = str(getattr(info, "sha", "") or "").strip()
    except Exception as exc:
        if configured_revision and _HUGGINGFACE_COMMIT_PATTERN.fullmatch(
            configured_revision
        ):
            return configured_revision.lower(), "configured_immutable_commit_offline"
        raise RuntimeError(
            f"无法把 generation model revision 解析为不可变 commit: {model_id}"
        ) from exc
    if not _HUGGINGFACE_COMMIT_PATTERN.fullmatch(resolved_commit):
        raise RuntimeError(
            f"Hugging Face 未返回合法 generation model commit: {model_id}"
        )
    source = (
        "configured_revision_huggingface_resolved_commit"
        if configured_revision
        else "huggingface_default_revision_resolved_commit"
    )
    return resolved_commit.lower(), source


def _generation_model_provenance_from_pipeline(
    pipeline: Any,
    *,
    expected_model_id: str,
) -> dict[str, str | None]:
    """读取 pipeline 加载时冻结的模型 provenance, 缺失时拒绝写正式记录。"""

    provenance = getattr(pipeline, "_sstw_generation_model_provenance", None)
    if not isinstance(provenance, dict):
        raise RuntimeError("生成 pipeline 缺少 SSTW 冻结模型 provenance")
    if provenance.get("generation_model_id") != expected_model_id:
        raise RuntimeError("生成 pipeline provenance 与请求的 generation model ID 不一致")
    commit = str(provenance.get("generation_model_commit_or_hash") or "")
    if not _HUGGINGFACE_COMMIT_PATTERN.fullmatch(commit):
        raise RuntimeError("生成 pipeline provenance 缺少不可变 commit")
    return dict(provenance)


def validate_generation_model_provenance(record: Mapping[str, Any]) -> str:
    """校验正式 record 的模型家族和不可变 revision, 并返回 commit。"""

    model_id = str(record.get("generation_model_id") or "").strip()
    expected_family = _model_family_from_id(model_id)
    observed_family = str(record.get("generation_model_family") or "").strip()
    if observed_family != expected_family:
        raise ValueError(
            f"generation model family 与注册表不一致: {model_id}"
        )
    commit = str(record.get("generation_model_commit_or_hash") or "").strip()
    if not _HUGGINGFACE_COMMIT_PATTERN.fullmatch(commit):
        raise ValueError(f"正式 generation record 缺少不可变模型 commit: {model_id}")
    if record.get("generation_model_revision_resolution_status") != "resolved_and_frozen":
        raise ValueError(f"generation model revision 尚未冻结: {model_id}")
    allowed_sources = {
        "configured_revision_huggingface_resolved_commit",
        "huggingface_default_revision_resolved_commit",
        "configured_immutable_commit_offline",
    }
    if record.get("generation_model_revision_source") not in allowed_sources:
        raise ValueError(f"generation model revision 来源不受支持: {model_id}")
    return commit.lower()


def _scheduler_id_for_model(model_id: str) -> str:
    """根据模型家族登记 pipeline 默认 scheduler 语义。"""

    return _registered_generation_model(model_id)["scheduler_id"]


def _load_video_generation_pipeline(
    model_id: str,
    torch_dtype: Any,
    *,
    revision: str | None = None,
) -> Any:
    """加载真实视频生成 pipeline。

    该函数属于 GPU runtime 路径。Wan2.1 是主结果模型, LTX-Video 是预注册的
    小规模跨模型泛化模型; 两者都必须进入真实 Flow scheduler 约束和 replay 路径。
    """
    configure_noisy_library_progress()
    hf_token = os.environ.get("HF_TOKEN") or None
    registration = _registered_generation_model(model_id)
    resolved_commit, revision_source = _resolve_generation_model_commit(
        model_id,
        requested_revision=revision,
        hf_token=hf_token,
    )
    emit_progress_event("video_generation_model_load", f"start | model={model_id}")
    if registration["pipeline_class_name"] == "WanPipeline":
        with suppress_third_party_progress_output("video_generation_model_import"):
            from diffusers import WanPipeline

        with suppress_third_party_progress_output("video_generation_model_load"):
            pipe = WanPipeline.from_pretrained(
                model_id,
                revision=resolved_commit,
                torch_dtype=torch_dtype,
                token=hf_token,
            )
    elif registration["pipeline_class_name"] == "LTXPipeline":
        with suppress_third_party_progress_output("video_generation_model_import"):
            from diffusers import LTXPipeline

        with suppress_third_party_progress_output("video_generation_model_load"):
            pipe = LTXPipeline.from_pretrained(
                model_id,
                revision=resolved_commit,
                torch_dtype=torch_dtype,
                token=hf_token,
            )
    else:  # pragma: no cover - 注册表常量已限制所有分支
        raise RuntimeError("注册模型缺少受支持的 pipeline class")
    setattr(
        pipe,
        "_sstw_generation_model_provenance",
        {
            "generation_model_id": model_id,
            "generation_model_requested_revision": str(revision or "") or None,
            "generation_model_commit_or_hash": resolved_commit,
            "generation_model_revision_source": revision_source,
            "generation_model_revision_resolution_status": "resolved_and_frozen",
        },
    )
    progress_bar_status = configure_pipeline_progress_bar(pipe)
    if hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    emit_progress_event("video_generation_model_load", f"finish | model={model_id} | pipeline_progress_bar={progress_bar_status}")
    return pipe


def _flow_latent_layout_for_pipeline(
    pipeline: Any,
    *,
    model_id: str,
    num_frames: int,
    height: int,
    width: int,
) -> Any:
    """返回模型原生 latent 到 SSTW 五维 tubelet 坐标的可逆适配器。"""

    if _model_family_from_id(model_id) == "diffusers_ltx_video":
        return build_ltx_latent_layout(
            pipeline,
            num_frames=num_frames,
            height=height,
            width=width,
        )
    return FiveDimensionalFlowLatentLayout(layout_id="wan_five_dimensional_flow_latent")


def _generation_kwargs_for_model(model_id: str, settings: dict[str, Any]) -> dict[str, Any]:
    """构造与模型官方 pipeline 参数语义一致的生成参数。"""

    common = {
        "width": settings["width"],
        "height": settings["height"],
        "num_frames": settings["num_frames"],
        "num_inference_steps": settings["num_inference_steps"],
    }
    if _model_family_from_id(model_id) == "diffusers_ltx_video":
        return {
            **common,
            "frame_rate": 8,
            "guidance_scale": 3.0,
            "decode_timestep": 0.05,
            "decode_noise_scale": 0.025,
        }
    return {**common, "guidance_scale": 5.0}


def _trajectory_time_grid_id_for_model(model_id: str) -> str:
    """登记生成阶段真实使用的模型家族 Flow 时间网格。"""

    return _registered_generation_model(model_id)["trajectory_time_grid_id"]


def _export_video(frames: Any, path: Path, fps: int) -> None:
    """使用 Diffusers 工具导出视频文件。"""
    from diffusers.utils import export_to_video

    path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(path), fps=fps)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _formalize_paper_trajectory_record(record: dict[str, Any], profile: str) -> dict[str, Any]:
    """保持真实 scheduler velocity 记录原样, 禁止把代理字段改名后进入正式包。"""

    forbidden = [key for key in record if "proxy" in key or "placeholder" in key]
    if forbidden:
        raise ValueError(f"正式 trajectory record 包含禁止字段: {forbidden}")
    return dict(record)


def _select_profile_items(items: list[dict], limit: int | None, allowed_roles: list[str] | None = None) -> list[dict]:
    """根据 profile 选择 prompt 或 seed。

    通用工程写法是先按显式 role 过滤, 再按 limit 截断。项目特定写法是让 motion_calibration profile
    只读取 calibration split, 防止 calibration 阈值使用 pilot main 或 evaluation 样本。
    """
    allowed_role_set = set(allowed_roles or [])
    selected = [item for item in items if not allowed_role_set or item.get("prompt_suite_role") in allowed_role_set]
    return selected if limit is None else selected[:limit]


def _select_cross_model_seeds(items: list[dict], limit: int, allowed_roles: list[str]) -> list[dict]:
    """按 calibration/test 等量抽取跨模型 seed, 避免小样本只落入单一 split。"""

    selected = _select_profile_items(items, None, allowed_roles)
    calibration = [item for item in selected if item.get("split") == "calibration"]
    test = [item for item in selected if item.get("split") == "test"]
    if len(calibration) < limit // 2 or len(test) < limit // 2:
        raise ValueError("跨模型 seed 子集无法同时覆盖 calibration 与 test split")
    calibration_count = limit // 2
    test_count = limit - calibration_count
    return calibration[:calibration_count] + test[:test_count]


def _build_generation_plan(prompt_suite: dict, profile: str, model_id: str, cross_model_id: str | None) -> list[dict]:
    """根据 profile 构建主模型与可选跨模型运行计划。"""
    settings = PROFILE_SETTINGS[profile]
    prompts = _select_profile_items(prompt_suite["prompts"], settings["prompt_limit"], settings.get("prompt_suite_roles"))
    seeds = _select_profile_items(prompt_suite["seeds"], settings["seed_limit"], settings.get("seed_suite_roles"))
    model_items = [{
        "generation_model_id": model_id,
        "cross_model_role": "main_generation_model",
        "model_prompts": prompts,
        "model_seeds": seeds,
    }]
    if settings["run_cross_model"] and cross_model_id:
        cross_prompts = _select_profile_items(
            prompt_suite["prompts"],
            int(settings["cross_model_prompt_limit"]),
            settings.get("prompt_suite_roles"),
        )
        cross_seeds = _select_cross_model_seeds(
            prompt_suite["seeds"],
            int(settings["cross_model_seed_limit"]),
            settings.get("seed_suite_roles") or [],
        )
        model_items.append({
            "generation_model_id": cross_model_id,
            "cross_model_role": "cross_model_validation_model",
            "model_prompts": cross_prompts,
            "model_seeds": cross_seeds,
        })
    plan = []
    include_clean_negative_references = profile in {"probe_paper", "pilot_paper", "full_paper"}
    for model_item in model_items:
        for prompt in model_item["model_prompts"]:
            for seed in model_item["model_seeds"]:
                base_item = {
                    "generation_model_id": model_item["generation_model_id"],
                    "cross_model_role": model_item["cross_model_role"],
                    **prompt,
                    **seed,
                }
                base_item["prompt_suite_role"] = prompt.get("prompt_suite_role")
                base_item["seed_suite_role"] = seed.get("seed_suite_role") or seed.get("prompt_suite_role")
                method_variants = (
                    ("sstw_full_method",)
                    if profile in PAPER_RESULT_PROFILES
                    else ("key_conditioned_state_space_with_trajectory",)
                )
                for method_variant in method_variants:
                    item = dict(base_item)
                    item["sample_role"] = "attacked_positive_source"
                    item["generation_sample_role"] = "attacked_positive_source"
                    item["method_variant"] = method_variant
                    item["watermark_embedding_status"] = (
                        "flow_scheduler_velocity_constraint"
                        if method_variant not in {"endpoint_only_control", "without_velocity_constraint"}
                        else "endpoint_latent_only_control"
                        if method_variant == "endpoint_only_control"
                        else "velocity_constraint_disabled_control"
                    )
                    item["formal_method_variant_execution"] = profile in PAPER_RESULT_PROFILES
                    plan.append(item)
                if not include_clean_negative_references:
                    continue
                clean_item = dict(base_item)
                clean_item["sample_role"] = "clean_negative"
                clean_item["generation_sample_role"] = "clean_negative"
                clean_item["method_variant"] = "sstw_clean_unwatermarked_reference"
                clean_item["watermark_embedding_status"] = "clean_unwatermarked_reference"
                clean_item["clean_negative_pair_role"] = "same_prompt_seed_unwatermarked_reference"
                clean_item["formal_method_variant_execution"] = True
                plan.append(clean_item)
    return plan


def _build_internal_ablation_generation_plan(main_plan: list[dict], profile: str) -> list[dict]:
    """在全部独立视频上展开会改变嵌入轨迹的 component-removal 变体。

    主计划仍保持每个 prompt/seed 一条 full-method positive 和一条 clean negative.
    只有改变 scheduler 运行机制的变体需要重新生成视频。仅改变观测变换或
    检测器结构的变体会在正式 Flow 评分阶段复用 full-method 的同一视频和同一
    replay, 从而避免浪费 GPU, 也避免把第二次生成的随机性混入消融效应。
    """

    if profile not in PAPER_RESULT_PROFILES:
        return []
    candidates = [
        item for item in main_plan
        if item.get("sample_role") == "attacked_positive_source"
        and item.get("method_variant") == "sstw_full_method"
    ]
    sources = [item for item in candidates if item.get("split") in {"calibration", "test"}]
    records: list[dict] = []
    for source in sources:
        for method_variant in GENERATION_METHOD_VARIANTS:
            if method_variant == "sstw_full_method":
                continue
            item = dict(source)
            item["method_variant"] = method_variant
            item["internal_ablation_source"] = True
            item["watermark_embedding_status"] = (
                "endpoint_latent_only_control"
                if method_variant == "endpoint_only_control"
                else "velocity_constraint_disabled_control"
                if method_variant == "without_velocity_constraint"
                else "flow_scheduler_velocity_constraint"
            )
            records.append(item)
    return records


def run_colab_probe(
    output_root: str | Path,
    prompt_suite_path: str | Path,
    profile: str,
    model_id: str,
    cross_model_id: str | None = None,
    *,
    model_revision: str | None = None,
    cross_model_revision: str | None = None,
) -> dict:
    """执行真实 Colab GPU 生成测试并写出 governed records。"""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Colab runtime does not expose CUDA GPU")
    gpu_name = str(torch.cuda.get_device_name(0))
    gpu_memory_mb = int(torch.cuda.get_device_properties(0).total_memory // (1024 * 1024))

    output_root = Path(output_root)
    prompt_suite = _read_json(prompt_suite_path)
    settings = PROFILE_SETTINGS[profile]
    dtype = _select_dtype(torch)
    hf_token_status = "provided" if os.environ.get("HF_TOKEN") else "not_provided"
    authentication_key_text = os.environ.get("SSTW_TRAJECTORY_AUTHENTICATION_KEY") or ""
    authentication_key_id = os.environ.get("SSTW_TRAJECTORY_AUTHENTICATION_KEY_ID") or ""
    if (
        len(authentication_key_text.encode("utf-8")) < 32
        or not authentication_key_id.strip()
    ):
        raise RuntimeError(
            "SSTW 嵌入运行要求至少32字节的所有者密钥和非空 key ID"
        )
    authentication_key = authentication_key_text.encode("utf-8")
    _registered_generation_model(model_id)
    if cross_model_id:
        _registered_generation_model(cross_model_id)
    if cross_model_id == model_id and (
        str(model_revision or "") != str(cross_model_revision or "")
    ):
        raise ValueError("同一 generation model ID 不能同时请求两个不同 revision")
    requested_revision_by_model = {
        model_id: model_revision,
        **({cross_model_id: cross_model_revision} if cross_model_id else {}),
    }
    main_plan = _build_generation_plan(prompt_suite, profile, model_id, cross_model_id)
    plan = main_plan + _build_internal_ablation_generation_plan(main_plan, profile)
    progress = ProgressReporter("flow_model_runtime_generation", len(plan), "video")

    generation_records: list[dict] = []
    trajectory_records: list[dict] = []
    trajectory_sketch_records: list[dict] = []
    quality_records: list[dict] = []
    active_pipe_by_model: dict[str, Any] = {}

    for index, item in enumerate(plan):
        progress.update(
            index + 1,
            (
                f"profile={profile} model={item['generation_model_id']} "
                f"prompt={item['prompt_id']} seed={item['seed_id']}"
            ),
        )
        model_for_item = item["generation_model_id"]
        if model_for_item not in active_pipe_by_model:
            active_pipe_by_model[model_for_item] = _load_video_generation_pipeline(
                model_for_item,
                dtype,
                revision=requested_revision_by_model.get(model_for_item),
            )
        pipe = active_pipe_by_model[model_for_item]
        model_provenance = _generation_model_provenance_from_pipeline(
            pipe,
            expected_model_id=model_for_item,
        )
        latent_layout = _flow_latent_layout_for_pipeline(
            pipe,
            model_id=model_for_item,
            num_frames=int(settings["num_frames"]),
            height=int(settings["height"]),
            width=int(settings["width"]),
        )
        generation_kwargs = _generation_kwargs_for_model(model_for_item, settings)
        generator = torch.Generator(device="cuda").manual_seed(int(item["seed_value"]))
        generator_state_digest_random = sha256(
            generator.get_state().cpu().numpy().tobytes()
        ).hexdigest()
        velocity_causal_pair_id = sha256(
            json.dumps(
                {
                    "generation_model_id": item["generation_model_id"],
                    "prompt_id": item["prompt_id"],
                    "seed_id": item["seed_id"],
                    "split": item.get("split", "main"),
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        trace_id = f"trace_{index:04d}_{item['sample_role']}_{item['method_variant']}"
        step_stats: list[dict] = []
        applied_step_count = 0
        path_summary: dict[str, Any] = {}
        endpoint_summary: dict[str, Any] = {}

        video_path = output_root / "videos" / (
            f"{item['generation_model_id'].replace('/', '_')}_{item['prompt_id']}_{item['seed_id']}_"
            f"{item['sample_role']}_{item['method_variant']}.mp4"
        )
        started = time.time()
        try:
            key_text = derive_watermark_key_text(
                authentication_key,
                key_id=authentication_key_id,
                generation_model_id=str(item["generation_model_id"]),
                prompt_id=str(item["prompt_id"]),
                seed_id=str(item["seed_id"]),
            )
            with FlowVelocityConstraintRuntime(
                pipe.scheduler,
                key_text=key_text,
                total_steps=int(settings["num_inference_steps"]),
                mechanism_config=velocity_runtime_mechanism_for_method_variant(
                    str(item["method_variant"])
                ),
                latent_layout=latent_layout,
            ) as velocity_runtime:
                with suppress_third_party_progress_output("flow_model_runtime_single_video_generation"):
                    result = pipe(
                        prompt=item["prompt_text"],
                        negative_prompt=item.get("prompt_negative_text"),
                        generator=generator,
                        **generation_kwargs,
                    )
            step_stats = [
                {
                    "trajectory_trace_id": trace_id,
                    "sample_role": item["sample_role"],
                    "method_variant": item["method_variant"],
                    "watermark_embedding_status": item["watermark_embedding_status"],
                    **record,
                }
                for record in velocity_runtime.step_records
            ]
            applied_step_count = sum(
                record.get("velocity_field_constraint_status") == "applied"
                for record in step_stats
            )
            path_summary = aggregate_path_observations(step_stats)
            if velocity_runtime.canonical_endpoint_latent is not None:
                endpoint_summary = compute_endpoint_latent_evidence(
                    velocity_runtime.canonical_endpoint_latent,
                    key_text=key_text,
                ).as_dict()
                endpoint_summary["endpoint_evidence_source"] = "generation_scheduler_endpoint_latent"
                endpoint_summary.update(latent_layout.as_dict())
            frames = result.frames[0]
            _export_video(frames, video_path, fps=8)
            generation_status = "success"
            failure_reason = "none"
            video_sha256 = _sha256_file(video_path)
        except Exception as exc:  # pragma: no cover - Colab GPU path
            generation_status = "failed"
            failure_reason = str(exc)
            video_sha256 = None
        runtime_sec = round(time.time() - started, 3)

        sketch_status = "not_required_nonpaper_profile"
        if profile in PAPER_RESULT_PROFILES:
            if generation_status == "success" and step_stats and authentication_key and authentication_key_id:
                sketch_payload = build_authenticated_trajectory_sketch_payload(
                    step_stats,
                    key_id=authentication_key_id,
                    prompt_digest=sha256(item["prompt_text"].encode("utf-8")).hexdigest(),
                    seed_id=str(item["seed_id"]),
                    model_signature=str(item["generation_model_id"]),
                    sampler_signature=(
                        f"{type(pipe.scheduler).__name__}:"
                        f"{sha256(json.dumps(dict(pipe.scheduler.config), sort_keys=True, default=str).encode('utf-8')).hexdigest()}"
                    ),
                    time_grid_id=_trajectory_time_grid_id_for_model(model_for_item),
                    generation_nonce_random=secrets.token_hex(16),
                )
                signed_sketch = sign_authenticated_trajectory_sketch(
                    sketch_payload,
                    authentication_key=authentication_key,
                )
                trajectory_sketch_records.append({
                    "record_version": "authenticated_trajectory_sketch_v1",
                    "trajectory_trace_id": trace_id,
                    "generation_model_id": item["generation_model_id"],
                    "prompt_id": item["prompt_id"],
                    "seed_id": item["seed_id"],
                    "method_variant": item["method_variant"],
                    "authenticated_trajectory_sketch_status": "signed",
                    **signed_sketch,
                })
                sketch_status = "signed"
            else:
                missing_reasons = []
                if not authentication_key:
                    missing_reasons.append("missing_SSTW_TRAJECTORY_AUTHENTICATION_KEY")
                if not authentication_key_id:
                    missing_reasons.append("missing_SSTW_TRAJECTORY_AUTHENTICATION_KEY_ID")
                if not step_stats:
                    missing_reasons.append("missing_trajectory_steps")
                trajectory_sketch_records.append({
                    "record_version": "authenticated_trajectory_sketch_v1",
                    "trajectory_trace_id": trace_id,
                    "generation_model_id": item["generation_model_id"],
                    "prompt_id": item["prompt_id"],
                    "seed_id": item["seed_id"],
                    "method_variant": item["method_variant"],
                    "authenticated_trajectory_sketch_status": "missing",
                    "trajectory_sketch_failure_reason": ";".join(missing_reasons) or failure_reason,
                })
                sketch_status = "missing"

        generation_records.append({
            "prompt_suite_id": prompt_suite["prompt_suite_id"],
            "colab_runtime_profile": profile,
            "generation_model_id": item["generation_model_id"],
            "generation_model_name": item["generation_model_id"],
            "generation_model_family": _model_family_from_id(item["generation_model_id"]),
            "generation_model_version": model_provenance[
                "generation_model_commit_or_hash"
            ],
            **model_provenance,
            "generation_model_license_status": "model_card_required",
            "hf_token_status": hf_token_status,
            "gpu_name": gpu_name,
            "gpu_memory_mb": gpu_memory_mb,
            "cross_model_role": item["cross_model_role"],
            "sample_role": item["sample_role"],
            "generation_sample_role": item["sample_role"],
            "method_variant": item["method_variant"],
            "watermark_embedding_status": item["watermark_embedding_status"],
            "velocity_constraint_config_id": "flow_velocity_constraint_default",
            "watermark_key_derivation_id": WATERMARK_KEY_DERIVATION_ID,
            "watermark_key_id": authentication_key_id,
            "flow_phase_schedule_id": "sin_squared_middle_flow_phase",
            "sampling_constraint_applied_step_count": applied_step_count,
            "formal_generation_watermark_embedding_level": (
                "clean_unwatermarked_reference"
                if item["sample_role"] == "clean_negative"
                else "endpoint_latent_only_control"
                if item["method_variant"] == "endpoint_only_control"
                else "velocity_constraint_disabled_control"
                if item["method_variant"] == "without_velocity_constraint"
                else "flow_scheduler_model_output_velocity_constraint"
            ),
            "formal_method_variant_execution": item.get("formal_method_variant_execution", False),
            "prompt_id": item["prompt_id"],
            "prompt_text_hash": sha256(item["prompt_text"].encode("utf-8")).hexdigest()[:16],
            "prompt_category": item["prompt_category"],
            "prompt_suite_role": item["prompt_suite_role"],
            "seed_suite_role": item.get("seed_suite_role"),
            "motion_pattern_id": item["motion_pattern_id"],
            "motion_claim_role": item.get("motion_claim_role"),
            "motion_calibration_role": item.get("motion_calibration_role"),
            "split": item.get("split", "main"),
            "seed_id": item["seed_id"],
            "generation_seed_random": int(item["seed_value"]),
            "generation_generator_state_digest_random": generator_state_digest_random,
            "velocity_causal_pair_id": velocity_causal_pair_id,
            "velocity_causal_intervention_status": (
                "velocity_constraint_enabled"
                if item["method_variant"] == "sstw_full_method"
                else "velocity_constraint_disabled"
                if item["method_variant"] == "without_velocity_constraint"
                else "not_in_velocity_causal_pair"
            ),
            "scheduler_id": _scheduler_id_for_model(item["generation_model_id"]),
            "trajectory_scheduler_id": _scheduler_id_for_model(item["generation_model_id"]),
            "trajectory_time_grid_id": _trajectory_time_grid_id_for_model(item["generation_model_id"]),
            "num_inference_steps": settings["num_inference_steps"],
            "guidance_scale": generation_kwargs["guidance_scale"],
            "video_length_frames": settings["num_frames"],
            "video_resolution": f"{settings['width']}x{settings['height']}",
            "fps": 8,
            "generation_status": generation_status,
            "generation_failure_reason": failure_reason,
            "generation_runtime_sec": runtime_sec,
            "video_path": str(video_path),
            "video_sha256": video_sha256,
            "trajectory_capture_status": "captured" if step_stats else "not_captured",
            "trajectory_capture_failure_reason": "none" if step_stats else failure_reason,
            "trajectory_trace_id": trace_id,
            "trajectory_num_steps": len(step_stats),
            "authenticated_trajectory_sketch_status": sketch_status,
            **path_summary,
            **endpoint_summary,
        })
        trajectory_records.extend(_formalize_paper_trajectory_record(record, profile) for record in step_stats)
        quality_records.append({
            "generation_model_id": item["generation_model_id"],
            "prompt_id": item["prompt_id"],
            "seed_id": item["seed_id"],
            "quality_metric_status": "not_run",
            "motion_metric_status": "not_run",
            "semantic_metric_status": "not_run",
            "metric_failure_reason": "optional_metric_dependencies_not_configured",
        })

    success_count = sum(1 for record in generation_records if record["generation_status"] == "success")
    cross_model_records = [
        record
        for record in generation_records
        if record.get("cross_model_role") == "cross_model_validation_model"
    ]
    cross_model_success_count = sum(
        record.get("generation_status") == "success"
        for record in cross_model_records
    )
    progress.finish(f"success={success_count} failed={len(generation_records) - success_count}")

    generation_records = [
        with_flow_evidence_protocol_defaults(
            record,
            trajectory_source_level="flow_scheduler_model_output_and_state_update"
            if profile in PAPER_RESULT_PROFILES
            else "callback_latent_trace"
            if record.get("trajectory_capture_status") == "captured"
            else "not_captured",
            claim_support_status="generation_evidence_only",
        )
        for record in generation_records
    ]
    trajectory_records = with_flow_evidence_protocol_defaults_many(
        trajectory_records,
        trajectory_source_level="flow_scheduler_model_output_step"
        if profile in PAPER_RESULT_PROFILES
        else "callback_latent_step",
        claim_support_status="trajectory_trace_evidence_only",
    )
    quality_records = with_flow_evidence_protocol_defaults_many(
        quality_records,
        trajectory_source_level="not_applicable",
        claim_support_status="optional_quality_metric_status_only",
    )
    external_records = run_external_baseline_status("configs/external_baselines/external_baselines.json")
    external_baseline_audit = audit_external_baseline_records(external_records)
    write_jsonl(output_root / "records" / "generation_records.jsonl", generation_records)
    write_jsonl(output_root / "records" / "trajectory_trace.jsonl", trajectory_records)
    write_jsonl(output_root / "records" / "trajectory_sketch_records.jsonl", trajectory_sketch_records)
    write_jsonl(output_root / "records" / "quality_motion_semantic_records.jsonl", quality_records)
    write_jsonl(output_root / "records" / "external_baseline_records.jsonl", external_records)
    write_csv(output_root / "tables" / "generation_runtime_table.csv", generation_records)
    write_csv(output_root / "tables" / "external_baseline_status_table.csv", external_records)
    write_json(output_root / "artifacts" / "external_baseline_status_decision.json", external_baseline_audit)
    decision = {
        "stage_id": "generative_video_generation",
        "implementation_decision": "PASS" if any(record["generation_status"] == "success" for record in generation_records) else "FAIL",
        "mechanism_decision": "PENDING_FORMAL_DETECTION_AND_REPLAY",
        "details": {
            "formal_claim_status": "velocity_path_generation_evidence_ready_detection_and_replay_pending",
            "generation_model_main_table_ready": any(record["generation_status"] == "success" for record in generation_records),
            "trajectory_observation_gain_confirmed": False,
            "fixed_low_fpr_audit_pass": False,
            "quality_motion_semantic_consistency_pass": False,
            "cross_model_validation_status": (
                "run_success"
                if cross_model_records and cross_model_success_count == len(cross_model_records)
                else "run_failed"
                if cross_model_records
                else "not_configured"
            ),
            "cross_model_validation_record_count": len(cross_model_records),
            "cross_model_validation_success_count": cross_model_success_count,
            "external_baseline_comparison_status": "limitation_records_written",
            "gpu_name": gpu_name,
            "gpu_memory_mb": gpu_memory_mb,
        },
    }
    write_json(output_root / "artifacts" / "generative_video_colab_runtime_decision.json", decision)
    write_json(output_root / "artifacts" / "generation_manifest.json", {
        "artifact_id": "generative_video_colab_runtime_manifest",
        "artifact_type": "manifest",
        "input_paths": [str(prompt_suite_path)],
        "output_paths": [
            str(output_root / "records" / "generation_records.jsonl"),
            str(output_root / "records" / "trajectory_trace.jsonl"),
            str(output_root / "records" / "trajectory_sketch_records.jsonl"),
        ],
        "rebuild_command": "python -m experiments.generative_video_model_probe.colab_runtime",
        "colab_runtime_profile": profile,
        "gpu_name": gpu_name,
        "gpu_memory_mb": gpu_memory_mb,
    })
    return {"output_root": str(output_root), "generation_record_count": len(generation_records), "trajectory_record_count": len(trajectory_records), "decision": decision}


def main() -> None:
    parser = argparse.ArgumentParser(description="在 Colab GPU 环境中运行 generative_video_model_probe 生成式视频模型探测。")
    parser.add_argument("--output-root", default="outputs/runs/generative_video_generation")
    parser.add_argument("--prompt-suite-path", default="outputs/datasets/generative_video_prompt_suite/prompt_seed_suite.json")
    parser.add_argument("--profile", choices=sorted(PROFILE_SETTINGS), default="pilot")
    parser.add_argument("--model-id", default=WAN21_PRIMARY_MODEL_ID)
    parser.add_argument("--cross-model-id", default=LTX_VIDEO_CROSS_MODEL_ID)
    parser.add_argument("--model-revision", default="")
    parser.add_argument("--cross-model-revision", default="")
    args = parser.parse_args()
    cross_model_id = args.cross_model_id or None
    print(json.dumps(run_colab_probe(
        args.output_root,
        args.prompt_suite_path,
        args.profile,
        args.model_id,
        cross_model_id,
        model_revision=args.model_revision or None,
        cross_model_revision=args.cross_model_revision or None,
    ), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
