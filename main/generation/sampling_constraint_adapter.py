"""将 sampling-time weak constraint 接入真实生成采样 callback。"""

from __future__ import annotations

from hashlib import sha256
from typing import Any

from main.generation.lambda_schedule import build_lambda_schedule


def _stable_phase_from_key(key_text: str) -> float:
    """从 key 文本生成确定性相位, 避免在 callback 中引入不可追踪随机性。"""
    digest = sha256(key_text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _deterministic_direction_like(tensor: Any, key_text: str) -> Any:
    """构造与 latent 同形状的确定性方向张量。

    该函数属于项目特定写法。真实生成模型 callback 中没有显式 tubelet code 张量时, 使用 key 派生的确定性方向作为弱约束方向, 并把方向构造过程记录到 governed records。
    """
    import torch

    phase = _stable_phase_from_key(key_text)
    flat_index = torch.arange(tensor.numel(), device=tensor.device, dtype=tensor.float().dtype)
    direction = torch.sin(flat_index * 0.017 + phase).reshape(tensor.shape)
    norm = direction.norm().clamp_min(1e-8)
    return direction / norm


def _cosine_alignment(left: Any, right: Any) -> float:
    """计算两个张量展平后的余弦对齐度。"""
    left_flat = left.detach().float().reshape(-1)
    right_flat = right.detach().float().reshape(-1)
    denom = left_flat.norm().clamp_min(1e-8) * right_flat.norm().clamp_min(1e-8)
    return round(float((left_flat @ right_flat / denom).item()), 6)


def apply_latent_sampling_constraint(
    latents: Any,
    step_index: int,
    num_steps: int,
    constraint_config: dict,
    schedule_config: dict,
    method_variant: str,
    key_text: str,
) -> tuple[Any, dict]:
    """在采样 callback 中对 latent 施加弱约束并返回记录字段。

    该函数属于项目特定写法。它只在范数预算内对 latent 添加 key-conditioned 方向偏置, 目的是检验 sampling-time weak constraint 是否能增强 trajectory observation, 而不是强制重写生成结果。
    """
    enabled = method_variant != "key_conditioned_state_space_with_trajectory"
    key_conditioned = method_variant != "trajectory_constraint_without_key_condition"
    admissibility_enabled = method_variant != "trajectory_constraint_without_admissibility"
    direction_key = key_text if key_conditioned else f"key_agnostic::{key_text}"
    direction = _deterministic_direction_like(latents, direction_key)
    lambda_values = build_lambda_schedule(
        str(schedule_config["lambda_schedule_id"]),
        num_steps,
        float(schedule_config["lambda_max"]),
        (float(schedule_config["lambda_time_window"][0]), float(schedule_config["lambda_time_window"][1])),
    )
    lambda_value = float(lambda_values[min(step_index, len(lambda_values) - 1)]) if lambda_values else 0.0
    before_alignment = _cosine_alignment(latents, direction)
    before_norm = float(latents.detach().float().norm().item())
    applied = bool(enabled and admissibility_enabled and lambda_value > 0.0)
    if applied:
        # 约束强度按 latent norm 的小比例缩放, 避免在真实生成链路中产生强制覆盖式扰动。
        norm_budget = float(constraint_config["constraint_norm_budget"])
        delta_norm = before_norm * norm_budget * lambda_value
        delta = direction.to(dtype=latents.dtype) * delta_norm
        constrained_latents = latents + delta
    else:
        constrained_latents = latents
    after_alignment = _cosine_alignment(constrained_latents, direction)
    after_norm = float(constrained_latents.detach().float().norm().item())
    record = {
        "sampling_constraint_enabled": enabled,
        "constraint_apply_status": "applied" if applied else "not_applied",
        "constraint_apply_reason": "active_lambda_step" if applied else "disabled_or_inactive_lambda_step",
        "constraint_key_condition_enabled": key_conditioned,
        "constraint_admissibility_enabled": admissibility_enabled,
        "lambda_schedule_id": schedule_config["lambda_schedule_id"],
        "lambda_max": float(schedule_config["lambda_max"]),
        "lambda_time_window": f"{schedule_config['lambda_time_window'][0]}:{schedule_config['lambda_time_window'][1]}",
        "lambda_value": round(lambda_value, 6),
        "constraint_norm_budget": float(constraint_config["constraint_norm_budget"]),
        "latent_alignment_before_constraint": before_alignment,
        "latent_alignment_after_constraint": after_alignment,
        "latent_alignment_gain": round(after_alignment - before_alignment, 6),
        "latent_norm_before_constraint": round(before_norm, 6),
        "latent_norm_after_constraint": round(after_norm, 6),
        "latent_constraint_delta_norm": round(abs(after_norm - before_norm), 6),
    }
    return constrained_latents, record
