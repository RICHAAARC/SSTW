"""在 Colab GPU 环境中运行 B6 sampling-time constraint probe。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from hashlib import sha256
from pathlib import Path
from typing import Any

from experiments.generative_video_model_probe.colab_runtime import _export_video, _select_dtype, _sha256_file, _tensor_stats
from main.generation.sampling_constraint_adapter import apply_latent_sampling_constraint
from main.protocol.flow_evidence_fields import conservative_flow_score, flow_evidence_protocol_defaults
from main.protocol.record_writer import write_json, write_jsonl
from main.protocol.table_builder import write_csv

PROFILE_SETTINGS = {
    "smoke": {"prompt_limit": 1, "seed_limit": 1, "num_inference_steps": 8, "num_frames": 33, "height": 320, "width": 512},
    "recommended": {"prompt_limit": 2, "seed_limit": 2, "num_inference_steps": 16, "num_frames": 49, "height": 320, "width": 512},
}

WAN21_PRIMARY_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

METHOD_VARIANTS = (
    "key_conditioned_state_space_with_trajectory",
    "keyed_state_trajectory_constraint",
    "trajectory_constraint_without_admissibility",
    "trajectory_constraint_without_key_condition",
    "trajectory_constraint_wrong_key_control",
)


def _read_json(path: str | Path) -> dict:
    """读取 JSON 文件。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _schedule_by_id(schedule_config: dict, schedule_id: str) -> dict:
    """根据 schedule ID 读取配置。"""
    for item in schedule_config["schedules"]:
        if item["lambda_schedule_id"] == schedule_id:
            return item
    raise KeyError(schedule_id)


def _variant_schedule_id(method_variant: str) -> str:
    """为 B6 probe 方法变体选择 schedule。"""
    if method_variant == "key_conditioned_state_space_with_trajectory":
        return "mid_window_weak_constraint"
    return "mid_window_weak_constraint"


def _build_generation_plan(prompt_suite: dict, profile: str, model_id: str) -> list[dict]:
    """构造 B6 probe 的 prompt / seed / method 运行计划。"""
    settings = PROFILE_SETTINGS[profile]
    prompts = prompt_suite["prompts"][: settings["prompt_limit"]]
    seeds = prompt_suite["seeds"][: settings["seed_limit"]]
    plan = []
    for prompt in prompts:
        for seed in seeds:
            for method_variant in METHOD_VARIANTS:
                plan.append({
                    "generation_model_id": model_id,
                    "method_variant": method_variant,
                    **prompt,
                    **seed,
                })
    return plan


def _model_family_from_id(model_id: str) -> str:
    """根据模型 ID 选择生成主干家族标签。"""
    normalized = model_id.lower()
    if "wan2.1" in normalized or "wan2_1" in normalized or "wan-ai" in normalized:
        return "diffusers_wan21_flow_matching_dit"
    if "ltx" in normalized:
        return "diffusers_ltx_video"
    return "diffusers_video_generation"


def _scheduler_id_for_model(model_id: str) -> str:
    """为不同视频生成 pipeline 记录 scheduler 语义。"""
    if _model_family_from_id(model_id) == "diffusers_wan21_flow_matching_dit":
        return "wan21_flow_matching_pipeline_default_scheduler"
    return "ltx_pipeline_default_scheduler"


def _clean_run_output_dirs(output_root: Path) -> None:
    """清理当前 Colab run_root 下会被本次运行重建的输出子目录。

    该函数属于通用工程写法。Colab 和 Google Drive 会保留上一次运行的视频和表格,
    因此 runtime 在写入新 records 前必须清理本 stage 自己管理的子目录, 避免旧文件混入 package。
    它只删除 `output_root` 下的固定 governed output 子目录, 不递归删除任意用户路径。
    """
    for subdir_name in ("records", "tables", "reports", "artifacts", "videos"):
        subdir = output_root / subdir_name
        if subdir.exists():
            shutil.rmtree(subdir)


def _jsonable_scheduler_config(pipe: Any) -> dict[str, Any]:
    """提取 scheduler 中可稳定序列化的配置。"""
    scheduler = getattr(pipe, "scheduler", None)
    config = getattr(scheduler, "config", {})
    if hasattr(config, "to_dict"):
        config = config.to_dict()
    if not isinstance(config, dict):
        config = dict(config) if config else {}
    return {str(key): value for key, value in config.items() if isinstance(value, (str, int, float, bool, type(None), list, tuple))}


def _sampler_signature_record(pipe: Any, model_id: str, scheduler_id: str) -> dict[str, str | None]:
    """记录真实 sampler signature 摘要。

    该函数属于项目特定写法。它和 preflight 使用同类 hash 逻辑, 只记录 scheduler
    元数据摘要, 不记录模型权重或访问 token。这样 sampling-time records 可以和 preflight
    records 对齐, 并为 wrong-sampler replay 检查保留证据。
    """
    scheduler_name = type(getattr(pipe, "scheduler", None)).__name__
    payload = {
        "generation_model_id": model_id,
        "scheduler_id": scheduler_id,
        "scheduler_class": scheduler_name,
        "scheduler_config": _jsonable_scheduler_config(pipe),
    }
    digest = sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    return {
        "sampler_signature_id": f"sampler_signature_{digest[:16]}",
        "sampler_signature_sha256": digest,
        "sampler_class_name": scheduler_name,
        "sampler_signature_placeholder": None,
    }


def _load_video_generation_pipeline(model_id: str, torch_dtype: Any) -> Any:
    """加载 B6 / SSTW-TC 主线视频生成 pipeline。

    该函数属于通用工程封装。Wan2.1 被设置为 SSTW-TC 主线 Flow Matching DiT 模型;
    LTX-Video 保留为前置机制验证和回退测试入口。Notebook 仍然只调用仓库模块,
    不在 Notebook 中手写正式 records。
    """
    hf_token = os.environ.get("HF_TOKEN") or None
    if _model_family_from_id(model_id) == "diffusers_wan21_flow_matching_dit":
        from diffusers import WanPipeline

        pipe = WanPipeline.from_pretrained(model_id, torch_dtype=torch_dtype, token=hf_token)
    else:
        from diffusers import LTXPipeline

        pipe = LTXPipeline.from_pretrained(model_id, torch_dtype=torch_dtype, token=hf_token)
    if hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    return pipe


def run_sampling_constraint_colab_probe(
    output_root: str | Path,
    prompt_suite_path: str | Path,
    profile: str,
    model_id: str,
    constraint_config_path: str | Path = "configs/generation/sampling_constraint.json",
    lambda_schedules_path: str | Path = "configs/generation/lambda_schedules.json",
) -> dict:
    """执行 B6 sampling-time constraint Colab probe 并写出 governed records。"""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Colab runtime does not expose CUDA GPU")
    gpu_name = str(torch.cuda.get_device_name(0))
    gpu_memory_mb = int(torch.cuda.get_device_properties(0).total_memory // (1024 * 1024))

    output_root = Path(output_root)
    _clean_run_output_dirs(output_root)
    prompt_suite = _read_json(prompt_suite_path)
    constraint_config = _read_json(constraint_config_path)
    lambda_schedules = _read_json(lambda_schedules_path)
    settings = PROFILE_SETTINGS[profile]
    dtype = _select_dtype(torch)
    hf_token_status = "provided" if os.environ.get("HF_TOKEN") else "not_provided"
    pipe = _load_video_generation_pipeline(model_id, dtype)
    plan = _build_generation_plan(prompt_suite, profile, model_id)
    model_family = _model_family_from_id(model_id)
    scheduler_id = _scheduler_id_for_model(model_id)
    sampler_signature = _sampler_signature_record(pipe, model_id, scheduler_id)

    generation_records: list[dict] = []
    trajectory_records: list[dict] = []
    constraint_records: list[dict] = []

    for index, item in enumerate(plan):
        generator = torch.Generator(device="cuda").manual_seed(int(item["seed_value"]))
        trace_id = f"b6_trace_{index:04d}"
        constraint_trace_id = f"b6_constraint_{index:04d}"
        step_stats: list[dict] = []
        step_constraint_records: list[dict] = []
        previous_callback_latents: Any | None = None
        schedule = _schedule_by_id(lambda_schedules, _variant_schedule_id(item["method_variant"]))
        key_text = f"{constraint_config['constraint_key_id']}::{item['prompt_id']}::{item['seed_id']}"

        def constraint_callback(pipe_instance: Any, step_index: int, timestep: Any, callback_kwargs: dict) -> dict:  # pragma: no cover - Colab GPU path
            nonlocal previous_callback_latents
            latents = callback_kwargs.get("latents")
            stats = _tensor_stats(latents) if latents is not None else {"latent_norm": None, "latent_mean": None, "latent_std": None}
            if latents is not None:
                constrained_latents, constraint_record = apply_latent_sampling_constraint(
                    latents,
                    int(step_index),
                    int(settings["num_inference_steps"]),
                    constraint_config,
                    schedule,
                    item["method_variant"],
                    key_text,
                    previous_callback_latents,
                )
                callback_kwargs["latents"] = constrained_latents
                previous_callback_latents = constrained_latents.detach().clone()
            else:
                constraint_record = {
                    "constraint_apply_status": "not_applied",
                    "constraint_apply_reason": "missing_latents",
                    "latent_alignment_gain": 0.0,
                    "flow_velocity_proxy_available": False,
                    "flow_velocity_proxy_source": "missing_current_callback_latents",
                }
            step_stats.append({
                "trajectory_trace_id": trace_id,
                "trajectory_step_index": int(step_index),
                "trajectory_timestep": float(timestep) if hasattr(timestep, "__float__") else str(timestep),
                **flow_evidence_protocol_defaults(
                    negative_family="not_applicable_callback_trace",
                    trajectory_source_level="callback_latent_trace",
                    sampler_signature_placeholder=sampler_signature.get("sampler_signature_placeholder"),
                    flow_state_admissibility_status="callback_trace_captured_not_scored",
                    claim_support_status="not_supported_step_trace_only",
                ),
                **stats,
                **sampler_signature,
            })
            constraint_output_record = {
                "record_version": "sampling_time_constraint_colab_probe_v1",
                "stage_id": "sampling_time_constraint_colab_probe",
                "constraint_trace_id": constraint_trace_id,
                "trajectory_trace_id": trace_id,
                "trajectory_step_index": int(step_index),
                "generation_model_id": item["generation_model_id"],
                "prompt_id": item["prompt_id"],
                "seed_id": item["seed_id"],
                "method_variant": item["method_variant"],
                "sampling_constraint_config_id": constraint_config["sampling_constraint_config_id"],
                "constraint_projection_operator_id": constraint_config["constraint_projection_operator_id"],
                "constraint_key_id": constraint_config["constraint_key_id"],
                "constraint_payload_code_id": constraint_config["constraint_payload_code_id"],
                "constraint_tubelet_selector_id": constraint_config["constraint_tubelet_selector_id"],
                "constraint_main_claim_status": "real_sampling_probe_pending_full_claim",
                "flow_matching_backbone_claim_status": "wan21_primary_flow_matching_claim" if model_family == "diffusers_wan21_flow_matching_dit" else "non_primary_mechanism_probe",
                **flow_evidence_protocol_defaults(
                    negative_family="not_applicable_positive_generation",
                    trajectory_source_level="callback_latent_trace",
                    sampler_signature_placeholder=sampler_signature.get("sampler_signature_placeholder"),
                    flow_state_admissibility_status="enabled" if constraint_record.get("constraint_admissibility_enabled") else "disabled_by_method_variant",
                    claim_support_status="not_supported_until_postprocess_and_quality_guards_pass",
                ),
                **constraint_record,
                **sampler_signature,
            }
            constraint_output_record["S_path_inv"] = constraint_output_record.get("latent_alignment_after_constraint")
            constraint_output_record["S_velocity"] = constraint_output_record.get("flow_velocity_alignment_after_constraint")
            constraint_output_record["S_final_conservative"] = conservative_flow_score(constraint_output_record)
            step_constraint_records.append(constraint_output_record)
            return callback_kwargs

        video_path = output_root / "videos" / f"{item['generation_model_id'].replace('/', '_')}_{item['method_variant']}_{item['prompt_id']}_{item['seed_id']}.mp4"
        started = time.time()
        try:
            result = pipe(
                prompt=item["prompt_text"],
                negative_prompt=item.get("prompt_negative_text"),
                width=settings["width"],
                height=settings["height"],
                num_frames=settings["num_frames"],
                num_inference_steps=settings["num_inference_steps"],
                generator=generator,
                callback_on_step_end=constraint_callback,
                callback_on_step_end_tensor_inputs=["latents"],
            )
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

        generation_records.append({
            "prompt_suite_id": prompt_suite["prompt_suite_id"],
            "colab_runtime_profile": profile,
            "generation_model_id": item["generation_model_id"],
            "generation_model_name": item["generation_model_id"],
            "generation_model_family": model_family,
            "generation_model_version": "from_pretrained_runtime_resolution",
            "generation_model_license_status": "model_card_required",
            "hf_token_status": hf_token_status,
            "gpu_name": gpu_name,
            "gpu_memory_mb": gpu_memory_mb,
            "method_variant": item["method_variant"],
            "prompt_id": item["prompt_id"],
            "prompt_text_hash": sha256(item["prompt_text"].encode("utf-8")).hexdigest()[:16],
            "prompt_category": item["prompt_category"],
            "prompt_suite_role": item["prompt_suite_role"],
            "motion_pattern_id": item["motion_pattern_id"],
            "seed_id": item["seed_id"],
            "scheduler_id": scheduler_id,
            "trajectory_scheduler_id": scheduler_id,
            "trajectory_time_grid_id": "flow_matching_callback_on_step_end_grid" if model_family == "diffusers_wan21_flow_matching_dit" else "callback_on_step_end_grid",
            "flow_matching_backbone_claim_status": "wan21_primary_flow_matching_claim" if model_family == "diffusers_wan21_flow_matching_dit" else "non_primary_mechanism_probe",
            **flow_evidence_protocol_defaults(
                negative_family="not_applicable_positive_generation",
                trajectory_source_level="callback_latent_trace" if step_stats else "not_captured",
                sampler_signature_placeholder=sampler_signature.get("sampler_signature_placeholder"),
                flow_state_admissibility_status="runtime_records_pending_postprocess",
                claim_support_status="not_supported_runtime_record_only",
            ),
            "num_inference_steps": settings["num_inference_steps"],
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
            "constraint_trace_id": constraint_trace_id,
            **sampler_signature,
        })
        trajectory_records.extend(step_stats)
        constraint_records.extend(step_constraint_records)

    successful_count = sum(1 for record in generation_records if record["generation_status"] == "success")
    write_jsonl(output_root / "records" / "generation_records.jsonl", generation_records)
    write_jsonl(output_root / "records" / "trajectory_trace.jsonl", trajectory_records)
    write_jsonl(output_root / "records" / "constraint_records.jsonl", constraint_records)
    write_csv(output_root / "tables" / "generation_runtime_table.csv", generation_records)
    write_csv(output_root / "tables" / "constraint_step_table.csv", constraint_records)
    decision = {
        "stage_id": "sampling_time_constraint_colab_probe",
        "implementation_decision": "PASS" if successful_count > 0 and constraint_records else "FAIL",
        "mechanism_decision": "FAIL",
        "details": {
            "formal_claim_status": "real_sampling_probe_pending_postprocess_and_formal_metrics",
            "successful_generation_count": successful_count,
            "constraint_record_count": len(constraint_records),
            "constraint_main_claim_status": "real_sampling_probe_pending_full_claim",
            "primary_sstw_tc_model_id": WAN21_PRIMARY_MODEL_ID,
            "primary_sstw_tc_model_status": "matched" if model_id == WAN21_PRIMARY_MODEL_ID else "not_matched",
            "gpu_name": gpu_name,
            "gpu_memory_mb": gpu_memory_mb,
        },
    }
    write_json(output_root / "artifacts" / "sampling_time_constraint_colab_runtime_decision.json", decision)
    write_json(output_root / "artifacts" / "generation_manifest.json", {
        "artifact_id": "sampling_time_constraint_colab_manifest",
        "artifact_type": "manifest",
        "input_paths": [str(prompt_suite_path), str(constraint_config_path), str(lambda_schedules_path)],
        "output_paths": [str(output_root / "records" / "generation_records.jsonl"), str(output_root / "records" / "constraint_records.jsonl")],
        "rebuild_command": "python -m experiments.sampling_time_constraint.colab_runtime",
        "primary_sstw_tc_model_id": WAN21_PRIMARY_MODEL_ID,
        "generation_model_family": model_family,
        "colab_runtime_profile": profile,
        "gpu_name": gpu_name,
        "gpu_memory_mb": gpu_memory_mb,
        **sampler_signature,
    })
    return {
        "output_root": str(output_root),
        "generation_record_count": len(generation_records),
        "successful_generation_count": successful_count,
        "trajectory_record_count": len(trajectory_records),
        "constraint_record_count": len(constraint_records),
        "decision": decision,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="在 Colab GPU 环境中运行 B6 sampling-time constraint probe。")
    parser.add_argument("--output-root", default="outputs/runs/sampling_time_constraint_colab")
    parser.add_argument("--prompt-suite-path", default="outputs/datasets/generative_video_prompt_suite/prompt_seed_suite.json")
    parser.add_argument("--profile", choices=sorted(PROFILE_SETTINGS), default="smoke")
    parser.add_argument("--model-id", default=WAN21_PRIMARY_MODEL_ID)
    parser.add_argument("--constraint-config-path", default="configs/generation/sampling_constraint.json")
    parser.add_argument("--lambda-schedules-path", default="configs/generation/lambda_schedules.json")
    args = parser.parse_args()
    print(json.dumps(run_sampling_constraint_colab_probe(
        args.output_root,
        args.prompt_suite_path,
        args.profile,
        args.model_id,
        args.constraint_config_path,
        args.lambda_schedules_path,
    ), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
