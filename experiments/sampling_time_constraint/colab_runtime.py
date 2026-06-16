"""在 Colab GPU 环境中运行 B6 sampling-time constraint probe。"""

from __future__ import annotations

import argparse
import json
import os
import time
from hashlib import sha256
from pathlib import Path
from typing import Any

from experiments.generative_video_model_probe.colab_runtime import _export_video, _load_ltx_pipeline, _select_dtype, _sha256_file, _tensor_stats
from main.generation.sampling_constraint_adapter import apply_latent_sampling_constraint
from main.protocol.record_writer import write_json, write_jsonl
from main.protocol.table_builder import write_csv

PROFILE_SETTINGS = {
    "smoke": {"prompt_limit": 1, "seed_limit": 1, "num_inference_steps": 8, "num_frames": 33, "height": 320, "width": 512},
    "recommended": {"prompt_limit": 2, "seed_limit": 2, "num_inference_steps": 16, "num_frames": 49, "height": 320, "width": 512},
}

METHOD_VARIANTS = (
    "key_conditioned_state_space_with_trajectory",
    "keyed_state_trajectory_constraint",
    "trajectory_constraint_without_admissibility",
    "trajectory_constraint_without_key_condition",
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
    prompt_suite = _read_json(prompt_suite_path)
    constraint_config = _read_json(constraint_config_path)
    lambda_schedules = _read_json(lambda_schedules_path)
    settings = PROFILE_SETTINGS[profile]
    dtype = _select_dtype(torch)
    hf_token_status = "provided" if os.environ.get("HF_TOKEN") else "not_provided"
    pipe = _load_ltx_pipeline(model_id, dtype)
    plan = _build_generation_plan(prompt_suite, profile, model_id)

    generation_records: list[dict] = []
    trajectory_records: list[dict] = []
    constraint_records: list[dict] = []

    for index, item in enumerate(plan):
        generator = torch.Generator(device="cuda").manual_seed(int(item["seed_value"]))
        trace_id = f"b6_trace_{index:04d}"
        constraint_trace_id = f"b6_constraint_{index:04d}"
        step_stats: list[dict] = []
        step_constraint_records: list[dict] = []
        schedule = _schedule_by_id(lambda_schedules, _variant_schedule_id(item["method_variant"]))
        key_text = f"{constraint_config['constraint_key_id']}::{item['prompt_id']}::{item['seed_id']}::{item['method_variant']}"

        def constraint_callback(pipe_instance: Any, step_index: int, timestep: Any, callback_kwargs: dict) -> dict:  # pragma: no cover - Colab GPU path
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
                )
                callback_kwargs["latents"] = constrained_latents
            else:
                constraint_record = {
                    "constraint_apply_status": "not_applied",
                    "constraint_apply_reason": "missing_latents",
                    "latent_alignment_gain": 0.0,
                }
            step_stats.append({
                "trajectory_trace_id": trace_id,
                "trajectory_step_index": int(step_index),
                "trajectory_timestep": float(timestep) if hasattr(timestep, "__float__") else str(timestep),
                **stats,
            })
            step_constraint_records.append({
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
                **constraint_record,
            })
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
            "generation_model_family": "diffusers_ltx_video",
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
            "scheduler_id": "ltx_pipeline_default_scheduler",
            "trajectory_scheduler_id": "ltx_pipeline_default_scheduler",
            "trajectory_time_grid_id": "callback_on_step_end_grid",
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
        "colab_runtime_profile": profile,
        "gpu_name": gpu_name,
        "gpu_memory_mb": gpu_memory_mb,
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
    parser.add_argument("--model-id", default="Lightricks/LTX-Video")
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
