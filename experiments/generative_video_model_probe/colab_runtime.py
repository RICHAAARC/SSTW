"""在 Colab GPU 环境中运行 generative_video_model_probe 生成式视频模型探测。"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import time
from hashlib import sha256
from pathlib import Path
from typing import Any

from external_baseline.baseline_registry import audit_external_baseline_records
from experiments.generative_video_model_probe.external_baseline_runner import run_external_baseline_status
from main.methods.state_space_watermark.endpoint_latent_detector import compute_endpoint_latent_evidence
from main.methods.state_space_watermark.authenticated_trajectory_sketch import (
    build_authenticated_trajectory_sketch_payload,
    sign_authenticated_trajectory_sketch,
)
from main.methods.state_space_watermark.flow_velocity_runtime import FlowVelocityConstraintRuntime
from main.methods.state_space_watermark.path_observation import aggregate_path_observations
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

PAPER_RESULT_PROFILES = {"probe_paper", "pilot_paper", "full_paper"}

PAPER_FORMAL_METHOD_VARIANTS = (
    "sstw_full_method",
    "endpoint_only_control",
    "trajectory_only_score",
    "without_velocity_constraint",
    "without_endpoint_aware_control",
    "without_replay_uncertainty_weighting",
    "without_flow_state_admissibility",
    "generic_ssm_baseline",
)
PROFILE_SETTINGS = {
    "pilot_paper": {
        "prompt_limit": None,
        "seed_limit": None,
        "num_inference_steps": 16,
        "num_frames": 49,
        "height": 320,
        "width": 512,
        "run_cross_model": False,
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
        "run_cross_model": False,
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
        "run_cross_model": False,
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
    """根据模型 ID 判断当前生成模型家族。"""
    normalized = model_id.lower()
    if "wan2.1" in normalized or "wan2_1" in normalized or "wan-ai" in normalized:
        return "diffusers_wan21_flow_matching_dit"
    if "ltx" in normalized:
        return "diffusers_ltx_video"
    return "diffusers_video_generation"


def _scheduler_id_for_model(model_id: str) -> str:
    """根据模型家族登记 pipeline 默认 scheduler 语义。"""
    if _model_family_from_id(model_id) == "diffusers_wan21_flow_matching_dit":
        return "wan21_flow_matching_pipeline_default_scheduler"
    return "ltx_pipeline_default_scheduler"


def _load_video_generation_pipeline(model_id: str, torch_dtype: Any) -> Any:
    """加载真实视频生成 pipeline。

    该函数属于 Colab runtime 路径。项目主线必须使用 Wan2.1 Flow Matching DiT;
    LTX-Video 只保留为工程 fallback, 不能替代主论文证据。
    """
    configure_noisy_library_progress()
    hf_token = os.environ.get("HF_TOKEN") or None
    emit_progress_event("video_generation_model_load", f"start | model={model_id}")
    if _model_family_from_id(model_id) == "diffusers_wan21_flow_matching_dit":
        with suppress_third_party_progress_output("video_generation_model_import"):
            from diffusers import WanPipeline

        with suppress_third_party_progress_output("video_generation_model_load"):
            pipe = WanPipeline.from_pretrained(model_id, torch_dtype=torch_dtype, token=hf_token)
    else:
        with suppress_third_party_progress_output("video_generation_model_import"):
            from diffusers import LTXPipeline

        with suppress_third_party_progress_output("video_generation_model_load"):
            pipe = LTXPipeline.from_pretrained(model_id, torch_dtype=torch_dtype, token=hf_token)
    progress_bar_status = configure_pipeline_progress_bar(pipe)
    if hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    emit_progress_event("video_generation_model_load", f"finish | model={model_id} | pipeline_progress_bar={progress_bar_status}")
    return pipe


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


def _build_generation_plan(prompt_suite: dict, profile: str, model_id: str, cross_model_id: str | None) -> list[dict]:
    """根据 profile 构建主模型与可选跨模型运行计划。"""
    settings = PROFILE_SETTINGS[profile]
    prompts = _select_profile_items(prompt_suite["prompts"], settings["prompt_limit"], settings.get("prompt_suite_roles"))
    seeds = _select_profile_items(prompt_suite["seeds"], settings["seed_limit"], settings.get("seed_suite_roles"))
    model_items = [{"generation_model_id": model_id, "cross_model_role": "main_generation_model"}]
    if settings["run_cross_model"] and cross_model_id:
        model_items.append({"generation_model_id": cross_model_id, "cross_model_role": "cross_model_validation_model"})
    plan = []
    include_clean_negative_references = profile in {"probe_paper", "pilot_paper", "full_paper"}
    for model_item in model_items:
        for prompt in prompts:
            for seed in seeds:
                base_item = {**model_item, **prompt, **seed}
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
    """在全部独立视频上展开 component-removal 变体, 不污染主方法样本数.

    主计划仍保持每个 prompt/seed 一条 full-method positive 和一条 clean negative.
    内部消融完整复用相同 prompt/seed/split 身份, 从而既能拟合每个变体的冻结
    后验, 又能在 held-out 视频上进行同源配对因果比较.
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
        for method_variant in PAPER_FORMAL_METHOD_VARIANTS:
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


def run_colab_probe(output_root: str | Path, prompt_suite_path: str | Path, profile: str, model_id: str, cross_model_id: str | None = None) -> dict:
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
    main_plan = _build_generation_plan(prompt_suite, profile, model_id, cross_model_id)
    plan = main_plan + _build_internal_ablation_generation_plan(main_plan, profile)
    progress = ProgressReporter("wan21_runtime_generation", len(plan), "video")

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
            active_pipe_by_model[model_for_item] = _load_video_generation_pipeline(model_for_item, dtype)
        pipe = active_pipe_by_model[model_for_item]
        generator = torch.Generator(device="cuda").manual_seed(int(item["seed_value"]))
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
            key_text = f"{item['generation_model_id']}::{item['prompt_id']}::{item['seed_id']}"
            with FlowVelocityConstraintRuntime(
                pipe.scheduler,
                key_text=key_text,
                total_steps=int(settings["num_inference_steps"]),
                method_variant=str(item["method_variant"]),
            ) as velocity_runtime:
                with suppress_third_party_progress_output("wan21_runtime_single_video_generation"):
                    result = pipe(
                        prompt=item["prompt_text"],
                        negative_prompt=item.get("prompt_negative_text"),
                        width=settings["width"],
                        height=settings["height"],
                        num_frames=settings["num_frames"],
                        num_inference_steps=settings["num_inference_steps"],
                        guidance_scale=5.0,
                        generator=generator,
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
            if velocity_runtime.endpoint_latent is not None:
                endpoint_summary = compute_endpoint_latent_evidence(
                    velocity_runtime.endpoint_latent,
                    key_text=key_text,
                ).as_dict()
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
            authentication_key = os.environ.get("SSTW_TRAJECTORY_AUTHENTICATION_KEY")
            authentication_key_id = os.environ.get("SSTW_TRAJECTORY_AUTHENTICATION_KEY_ID")
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
                    time_grid_id="wan_flow_scheduler_runtime_step_grid",
                    generation_nonce_random=secrets.token_hex(16),
                )
                signed_sketch = sign_authenticated_trajectory_sketch(
                    sketch_payload,
                    authentication_key=authentication_key.encode("utf-8"),
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
            "generation_model_version": "from_pretrained_runtime_resolution",
            "generation_model_commit_or_hash": None,
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
            "scheduler_id": _scheduler_id_for_model(item["generation_model_id"]),
            "trajectory_scheduler_id": _scheduler_id_for_model(item["generation_model_id"]),
            "trajectory_time_grid_id": "wan_flow_scheduler_runtime_step_grid",
            "num_inference_steps": settings["num_inference_steps"],
            "guidance_scale": 5.0,
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
            "cross_model_validation_status": "run" if cross_model_id else "not_configured",
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
    parser.add_argument("--cross-model-id", default="")
    args = parser.parse_args()
    cross_model_id = args.cross_model_id or None
    print(json.dumps(run_colab_probe(args.output_root, args.prompt_suite_path, args.profile, args.model_id, cross_model_id), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
