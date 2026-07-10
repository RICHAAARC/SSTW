"""在 Flow scheduler 消费模型输出前施加密钥条件速度场弱约束。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from main.methods.state_space_watermark.flow_tubelet_key_code import (
    FlowTubeletKeyCodeConfig,
    flow_phase_weight,
)


@dataclass(frozen=True)
class VelocityFieldConstraintConfig:
    """定义速度场约束的能量预算与 endpoint-aware 调节参数。"""

    lambda_max: float = 0.12
    velocity_norm_ratio_budget: float = 0.02
    endpoint_target_margin: float = 0.01
    endpoint_control_minimum: float = 0.25
    endpoint_control_maximum: float = 1.5
    scheduler_velocity_sign: float = -1.0


def _cosine_alignment(left: Any, right: Any) -> float:
    """计算两个同形张量的余弦对齐度。"""

    left_flat = left.detach().float().reshape(-1)
    right_flat = right.detach().float().reshape(-1)
    denominator = left_flat.norm().clamp_min(1e-8) * right_flat.norm().clamp_min(1e-8)
    return float((left_flat @ right_flat / denominator).item())


def _endpoint_control_multiplier(
    sample: Any,
    key_direction: Any,
    config: VelocityFieldConstraintConfig,
) -> tuple[float, float]:
    """根据当前 latent 的 endpoint margin 分配剩余控制强度。"""

    response = _cosine_alignment(sample, key_direction)
    target = max(float(config.endpoint_target_margin), 1e-8)
    deficit_ratio = max(0.0, (target - response) / target)
    multiplier = max(
        float(config.endpoint_control_minimum),
        min(float(config.endpoint_control_maximum), deficit_ratio),
    )
    return multiplier, response


def apply_velocity_field_constraint(
    model_output: Any,
    sample: Any,
    key_direction: Any,
    *,
    flow_phase: float,
    config: VelocityFieldConstraintConfig | None = None,
    tubelet_config: FlowTubeletKeyCodeConfig | None = None,
    endpoint_control_enabled: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """修改 scheduler 即将消费的模型输出并返回真实约束统计。

    该函数接收的是 Transformer 经过 classifier-free guidance 后、传入 Flow
    scheduler 之前的 `model_output`。因此记录中的 velocity 证据来自真实采样
    更新输入, 不再由 callback 后置 latent 位移冒充速度场。
    """

    config = config or VelocityFieldConstraintConfig()
    tubelet_config = tubelet_config or FlowTubeletKeyCodeConfig()
    if model_output.shape != sample.shape or model_output.shape != key_direction.shape:
        raise ValueError("model_output、sample 和 key_direction 必须同形")
    if config.lambda_max < 0.0 or config.velocity_norm_ratio_budget < 0.0:
        raise ValueError("速度场约束强度和范数预算不能为负数")

    schedule_weight = flow_phase_weight(flow_phase, tubelet_config)
    lambda_value = float(config.lambda_max) * schedule_weight
    base_norm = float(model_output.detach().float().norm().item())
    control_multiplier, endpoint_response_before = _endpoint_control_multiplier(sample, key_direction, config)
    if not endpoint_control_enabled:
        control_multiplier = 1.0
    delta_norm = base_norm * float(config.velocity_norm_ratio_budget) * lambda_value * control_multiplier
    signed_direction = key_direction * float(config.scheduler_velocity_sign)
    delta = signed_direction.to(dtype=model_output.dtype) * delta_norm
    constrained = model_output + delta

    before_alignment = _cosine_alignment(model_output, key_direction)
    after_alignment = _cosine_alignment(constrained, key_direction)
    actual_delta_norm = float((constrained - model_output).detach().float().norm().item())
    ratio = actual_delta_norm / max(base_norm, 1e-8)
    record = {
        "velocity_field_constraint_status": "applied" if lambda_value > 0.0 else "inactive_flow_phase",
        "velocity_field_source": "scheduler_model_output_before_flow_match_step",
        "flow_phase": round(float(flow_phase), 8),
        "flow_phase_weight": round(schedule_weight, 8),
        "velocity_constraint_lambda": round(lambda_value, 8),
        "velocity_norm_before_constraint": round(base_norm, 6),
        "velocity_norm_after_constraint": round(float(constrained.detach().float().norm().item()), 6),
        "velocity_constraint_delta_norm": round(actual_delta_norm, 6),
        "velocity_constraint_delta_ratio": round(ratio, 8),
        "velocity_alignment_before_constraint": round(before_alignment, 8),
        "velocity_alignment_after_constraint": round(after_alignment, 8),
        "velocity_alignment_gain": round(after_alignment - before_alignment, 8),
        "endpoint_control_enabled": bool(endpoint_control_enabled),
        "endpoint_control_multiplier": round(control_multiplier, 8),
        "endpoint_response_before_step": round(endpoint_response_before, 8),
        "scheduler_velocity_sign": float(config.scheduler_velocity_sign),
        "velocity_norm_ratio_budget": float(config.velocity_norm_ratio_budget),
    }
    return constrained, record
