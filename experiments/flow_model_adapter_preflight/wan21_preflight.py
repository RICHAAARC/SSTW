"""Wan2.1 Flow adapter preflight, 只验证接口能力, 不运行完整实验。"""

from __future__ import annotations

import argparse
import json
import os
import time
from hashlib import sha256
from pathlib import Path
from typing import Any

from main.protocol.flow_evidence_fields import conservative_flow_score, flow_evidence_protocol_defaults
from main.protocol.record_writer import write_json, write_jsonl

WAN21_PRIMARY_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


def _jsonable_scheduler_config(pipe: Any) -> dict[str, Any]:
    """提取 scheduler 中可序列化的配置。"""
    scheduler = getattr(pipe, "scheduler", None)
    config = getattr(scheduler, "config", {})
    if hasattr(config, "to_dict"):
        config = config.to_dict()
    if not isinstance(config, dict):
        config = dict(config) if config else {}
    return {str(key): value for key, value in config.items() if isinstance(value, (str, int, float, bool, type(None), list, tuple))}


def _sampler_signature(pipe: Any, model_id: str) -> dict[str, str]:
    """构造受治理的 sampler signature 摘要。

    该函数属于项目特定写法。它不会记录完整模型权重或 token, 只记录 scheduler
    类型、模型 ID 和可序列化 scheduler 配置的 hash, 用于判断 replay 或 wrong sampler
    是否可能伪造正确轨迹。
    """
    scheduler = getattr(pipe, "scheduler", None)
    scheduler_class = type(scheduler).__name__
    scheduler_config = _jsonable_scheduler_config(pipe)
    payload = {
        "generation_model_id": model_id,
        "scheduler_class": scheduler_class,
        "scheduler_config": scheduler_config,
    }
    digest = sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    return {
        "sampler_signature_id": f"sampler_signature_{digest[:16]}",
        "sampler_signature_sha256": digest,
        "sampler_class_name": scheduler_class,
    }


def _tensor_stats(latents: Any) -> dict[str, float | None]:
    """记录 callback latent 的轻量统计值。"""
    if latents is None:
        return {"latent_norm": None, "latent_mean": None, "latent_std": None}
    detached = latents.detach().float()
    return {
        "latent_norm": round(float(detached.norm().item()), 6),
        "latent_mean": round(float(detached.mean().item()), 6),
        "latent_std": round(float(detached.std().item()), 6),
    }


def _velocity_proxy(previous_latents: Any | None, latents: Any) -> dict[str, Any]:
    """用相邻 callback latent 位移记录 velocity proxy。"""
    if previous_latents is None or latents is None:
        return {
            "flow_velocity_proxy_available": False,
            "flow_velocity_proxy_source": "missing_previous_or_current_callback_latents",
            "S_velocity": None,
        }
    displacement = latents.detach().float() - previous_latents.to(device=latents.device).detach().float()
    return {
        "flow_velocity_proxy_available": True,
        "flow_velocity_proxy_source": "adjacent_callback_latent_displacement",
        "S_velocity": round(float(displacement.norm().item()), 6),
    }


def _load_wan21_pipeline(model_id: str, torch_dtype: Any) -> Any:
    """加载 Wan2.1 Diffusers pipeline。"""
    from diffusers import WanPipeline

    hf_token = os.environ.get("HF_TOKEN") or None
    pipe = WanPipeline.from_pretrained(model_id, torch_dtype=torch_dtype, token=hf_token)
    if hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    return pipe


def run_wan21_flow_adapter_preflight(
    output_root: str | Path,
    model_id: str = WAN21_PRIMARY_MODEL_ID,
    num_inference_steps: int = 4,
    num_frames: int = 33,
    height: int = 320,
    width: int = 512,
) -> dict[str, Any]:
    """执行 Wan2.1 adapter preflight 并写出 records。

    该函数只验证接口: 模型加载、callback latent 捕获、time grid、sampler signature、
    velocity proxy 以及 L4 显存是否满足最小 smoke 运行。它不生成主实验结论。
    """
    import torch

    output_root = Path(output_root)
    records: list[dict[str, Any]] = []
    trajectory_records: list[dict[str, Any]] = []
    started = time.time()
    cuda_available = torch.cuda.is_available()
    gpu_name = str(torch.cuda.get_device_name(0)) if cuda_available else "cuda_unavailable"
    gpu_memory_mb = int(torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)) if cuda_available else 0
    l4_memory_sufficient = gpu_memory_mb >= 20_000

    base_record = {
        "record_version": "wan21_flow_adapter_preflight_v1",
        "stage_id": "flow_model_adapter_preflight",
        "generation_model_id": model_id,
        "generation_model_family": "diffusers_wan21_flow_matching_dit",
        "gpu_name": gpu_name,
        "gpu_memory_mb": gpu_memory_mb,
        "l4_memory_sufficient": l4_memory_sufficient,
        **flow_evidence_protocol_defaults(
            negative_family="not_applicable_preflight",
            trajectory_source_level="not_captured",
            sampler_signature_placeholder="pending_until_pipeline_load",
            flow_state_admissibility_status="not_evaluated_preflight",
            claim_support_status="not_supported_preflight_only",
        ),
    }
    if not cuda_available:
        decision = {
            **base_record,
            "adapter_preflight_decision": "FAIL",
            "adapter_preflight_failure_reason": "cuda_unavailable",
            "model_load_status": "not_run",
            "callback_latent_capture_status": "not_run",
            "time_grid_capture_status": "not_run",
            "sampler_signature_status": "not_run",
            "velocity_proxy_status": "not_run",
            "runtime_sec": round(time.time() - started, 3),
        }
        write_jsonl(output_root / "records" / "wan21_flow_adapter_preflight_records.jsonl", [decision])
        write_json(output_root / "artifacts" / "wan21_flow_adapter_preflight_decision.json", decision)
        return decision

    previous_latents: Any | None = None
    callback_latent_count = 0
    time_grid: list[Any] = []
    velocity_proxy_count = 0
    try:
        dtype = torch.float16
        pipe = _load_wan21_pipeline(model_id, dtype)
        sampler_signature = _sampler_signature(pipe, model_id)

        def callback_on_step_end(pipe_instance: Any, step_index: int, timestep: Any, callback_kwargs: dict) -> dict:  # pragma: no cover - GPU preflight path
            nonlocal previous_latents, callback_latent_count, velocity_proxy_count
            latents = callback_kwargs.get("latents")
            callback_latent_count += 1 if latents is not None else 0
            velocity_record = _velocity_proxy(previous_latents, latents)
            if velocity_record["flow_velocity_proxy_available"]:
                velocity_proxy_count += 1
            if latents is not None:
                previous_latents = latents.detach().clone()
            time_value = float(timestep) if hasattr(timestep, "__float__") else str(timestep)
            time_grid.append(time_value)
            trajectory_record = {
                **base_record,
                **sampler_signature,
                "sampler_signature_placeholder": None,
                "trajectory_source_level": "callback_latent_trace",
                "trajectory_step_index": int(step_index),
                "trajectory_timestep": time_value,
                "flow_state_admissibility_status": "callback_trace_observed",
                **_tensor_stats(latents),
                **velocity_record,
            }
            trajectory_record["S_path_inv"] = trajectory_record["latent_norm"]
            trajectory_record["S_final_conservative"] = conservative_flow_score(trajectory_record)
            trajectory_records.append(trajectory_record)
            return callback_kwargs

        generator = torch.Generator(device="cuda").manual_seed(17)
        pipe(
            prompt="A small paper boat moving slowly on a calm lake.",
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            generator=generator,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=["latents"],
        )
        model_load_status = "loaded"
        failure_reason = "none"
    except Exception as exc:  # pragma: no cover - GPU preflight path
        sampler_signature = {}
        model_load_status = "failed"
        failure_reason = f"{type(exc).__name__}: {exc}"

    decision_pass = all([
        model_load_status == "loaded",
        callback_latent_count > 0,
        bool(time_grid),
        bool(sampler_signature),
        velocity_proxy_count > 0,
        l4_memory_sufficient,
    ])
    decision = {
        **base_record,
        **sampler_signature,
        "sampler_signature_placeholder": None if sampler_signature else base_record["sampler_signature_placeholder"],
        "trajectory_source_level": "callback_latent_trace" if callback_latent_count else "not_captured",
        "adapter_preflight_decision": "PASS" if decision_pass else "FAIL",
        "adapter_preflight_failure_reason": failure_reason,
        "model_load_status": model_load_status,
        "callback_latent_capture_status": "captured" if callback_latent_count else "not_captured",
        "callback_latent_count": callback_latent_count,
        "time_grid_capture_status": "captured" if time_grid else "not_captured",
        "time_grid": time_grid,
        "sampler_signature_status": "captured" if sampler_signature else "not_captured",
        "velocity_proxy_status": "captured" if velocity_proxy_count else "not_captured",
        "velocity_proxy_count": velocity_proxy_count,
        "runtime_sec": round(time.time() - started, 3),
    }
    records.append(decision)
    write_jsonl(output_root / "records" / "wan21_flow_adapter_preflight_records.jsonl", records)
    write_jsonl(output_root / "records" / "wan21_flow_adapter_trajectory_records.jsonl", trajectory_records)
    write_json(output_root / "artifacts" / "wan21_flow_adapter_preflight_decision.json", decision)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 Wan2.1 Flow adapter preflight。")
    parser.add_argument("--output-root", default="outputs/runs/wan21_flow_adapter_preflight")
    parser.add_argument("--model-id", default=WAN21_PRIMARY_MODEL_ID)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--num-frames", type=int, default=33)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=512)
    args = parser.parse_args()
    print(json.dumps(run_wan21_flow_adapter_preflight(
        output_root=args.output_root,
        model_id=args.model_id,
        num_inference_steps=args.num_inference_steps,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
    ), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
