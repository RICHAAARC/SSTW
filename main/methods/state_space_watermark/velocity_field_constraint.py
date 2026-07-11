"""在 Flow scheduler 消费模型输出前施加密钥条件速度场弱约束。"""

from __future__ import annotations

from dataclasses import dataclass
import math
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
    flow_energy_budget_ratio: float = 0.000015
    finite_difference_probe_ratio: float = 0.25
    minimum_controllability_gain: float = 1e-8
    prompt_velocity_sensitivity_gate_enabled: bool = True
    semantic_tangent_projection_enabled: bool = True
    semantic_tangent_minimum_retained_ratio: float = 0.05


@dataclass(frozen=True)
class VelocityControlContext:
    """提供 endpoint 最小能量控制所需的真实时间步和累计能量状态。"""

    delta_sigma: float
    cumulative_control_energy: float
    cumulative_reference_energy: float
    remaining_step_count: int


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


def _prompt_velocity_safe_direction(
    key_direction: Any,
    model_output: Any,
    config: VelocityFieldConstraintConfig,
) -> tuple[Any, str, float]:
    """构造对 prompt-conditioned 主速度一阶低敏的密钥方向。

    该投影不把简单像素或 latent 启发式冒充语义模型。它使用生成器自身的
    prompt-conditioned velocity 作为一阶语义切向，并将水印方向在该切向上
    正交化。由此得到的是明确的局部一阶保护，而不是不可验证的“语义无损”。
    """

    direction = key_direction.detach().float()
    velocity = model_output.detach().float()
    original_norm = direction.norm().clamp_min(1e-12)
    if config.prompt_velocity_sensitivity_gate_enabled:
        velocity_rms = velocity.square().mean().sqrt().clamp_min(1e-8)
        sensitivity = velocity.abs() / velocity_rms
        direction = direction / (1.0 + sensitivity.square())
    if config.semantic_tangent_projection_enabled:
        velocity_norm_squared = velocity.square().sum().clamp_min(1e-12)
        projection_coefficient = (direction * velocity).sum() / velocity_norm_squared
        direction = direction - projection_coefficient * velocity
    retained_ratio = float((direction.norm() / original_norm).item())
    if retained_ratio < float(config.semantic_tangent_minimum_retained_ratio):
        return (
            key_direction.detach().float() * 0.0,
            "rejected_low_retained_key_energy",
            retained_ratio,
        )
    safe_direction = direction / direction.norm().clamp_min(1e-12) * original_norm
    return safe_direction, "prompt_velocity_tangent_projection_applied", retained_ratio


def _finite_difference_endpoint_control(
    model_output: Any,
    sample: Any,
    key_direction: Any,
    safe_direction: Any,
    *,
    step_norm_budget: float,
    context: VelocityControlContext,
    config: VelocityFieldConstraintConfig,
) -> dict[str, float | str | bool]:
    """用当前 Flow Euler 局部模型估计 endpoint 响应并求最小标量控制量。"""

    delta_sigma = float(context.delta_sigma)
    signed_direction = safe_direction * float(config.scheduler_velocity_sign)
    baseline_next = sample.detach().float() + delta_sigma * model_output.detach().float()
    response_without_control = _cosine_alignment(baseline_next, key_direction)
    target = max(float(config.endpoint_target_margin), 1e-8)
    deficit = max(0.0, target - response_without_control)
    probe_norm = max(
        step_norm_budget * float(config.finite_difference_probe_ratio),
        float(model_output.detach().float().norm().item()) * 1e-8,
        1e-10,
    )
    probe_next = baseline_next + delta_sigma * signed_direction * probe_norm
    probe_response = _cosine_alignment(probe_next, key_direction)
    controllability_gain = (probe_response - response_without_control) / probe_norm
    reference_energy_increment = (
        delta_sigma * delta_sigma
        * float(model_output.detach().float().square().sum().item())
    )
    projected_reference_energy = (
        max(0.0, float(context.cumulative_reference_energy))
        + reference_energy_increment * max(1, int(context.remaining_step_count))
    )
    total_energy_budget = (
        float(config.flow_energy_budget_ratio) * projected_reference_energy
    )
    remaining_energy_budget = max(
        0.0,
        total_energy_budget - max(0.0, float(context.cumulative_control_energy)),
    )
    energy_limited_delta_norm = math.sqrt(remaining_energy_budget) / max(
        abs(delta_sigma),
        1e-12,
    )
    admissible_delta_norm = min(step_norm_budget, energy_limited_delta_norm)
    if deficit <= 0.0:
        required_delta_norm = 0.0
        selected_delta_norm = 0.0
        status = "endpoint_target_already_reached"
    elif controllability_gain < float(config.minimum_controllability_gain):
        required_delta_norm = math.inf
        selected_delta_norm = 0.0
        status = "rejected_insufficient_finite_difference_controllability"
    else:
        required_delta_norm = deficit / controllability_gain
        selected_delta_norm = min(required_delta_norm, admissible_delta_norm)
        status = (
            "minimum_energy_control_applied"
            if selected_delta_norm + 1e-12 >= required_delta_norm
            else "minimum_energy_control_budget_limited"
        )
    predicted_response = _cosine_alignment(
        baseline_next + delta_sigma * signed_direction * selected_delta_norm,
        key_direction,
    )
    control_energy_increment = (
        delta_sigma * delta_sigma * selected_delta_norm * selected_delta_norm
    )
    return {
        "selected_delta_norm": float(selected_delta_norm),
        "required_delta_norm": float(required_delta_norm),
        "endpoint_response_without_control": float(response_without_control),
        "endpoint_response_finite_difference_probe": float(probe_response),
        "endpoint_response_predicted_after_step": float(predicted_response),
        "endpoint_controllability_gain": float(controllability_gain),
        "endpoint_margin_deficit_before_control": float(deficit),
        "endpoint_control_energy_increment": float(control_energy_increment),
        "endpoint_control_cumulative_energy_after": (
            float(context.cumulative_control_energy) + control_energy_increment
        ),
        "endpoint_reference_energy_increment": float(reference_energy_increment),
        "endpoint_reference_cumulative_energy_after": (
            float(context.cumulative_reference_energy) + reference_energy_increment
        ),
        "endpoint_projected_total_energy_budget": float(total_energy_budget),
        "endpoint_remaining_energy_budget_before_step": float(remaining_energy_budget),
        "endpoint_minimum_energy_control_status": status,
        "endpoint_quality_energy_guard_passed": (
            control_energy_increment <= remaining_energy_budget + 1e-10
        ),
    }


def apply_velocity_field_constraint(
    model_output: Any,
    sample: Any,
    key_direction: Any,
    *,
    flow_phase: float,
    config: VelocityFieldConstraintConfig | None = None,
    tubelet_config: FlowTubeletKeyCodeConfig | None = None,
    endpoint_control_enabled: bool = True,
    control_context: VelocityControlContext | None = None,
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
    if (
        config.lambda_max < 0.0
        or config.velocity_norm_ratio_budget < 0.0
        or config.flow_energy_budget_ratio < 0.0
    ):
        raise ValueError("速度场约束强度和范数预算不能为负数")

    schedule_weight = flow_phase_weight(flow_phase, tubelet_config)
    lambda_value = float(config.lambda_max) * schedule_weight
    base_norm = float(model_output.detach().float().norm().item())
    step_norm_budget = (
        base_norm * float(config.velocity_norm_ratio_budget) * lambda_value
    )
    safe_direction, semantic_projection_status, retained_key_energy_ratio = (
        _prompt_velocity_safe_direction(key_direction, model_output, config)
    )
    endpoint_response_before = _cosine_alignment(sample, key_direction)
    minimum_energy_record: dict[str, Any] = {}
    formal_control_context_complete = bool(
        control_context is not None
        and math.isfinite(float(control_context.delta_sigma))
        and abs(float(control_context.delta_sigma)) > 1e-12
        and int(control_context.remaining_step_count) > 0
        and float(control_context.cumulative_control_energy) >= 0.0
        and float(control_context.cumulative_reference_energy) >= 0.0
        and semantic_projection_status
        == "prompt_velocity_tangent_projection_applied"
    )
    if endpoint_control_enabled and formal_control_context_complete:
        minimum_energy_record = _finite_difference_endpoint_control(
            model_output,
            sample,
            key_direction,
            safe_direction,
            step_norm_budget=step_norm_budget,
            context=control_context,
            config=config,
        )
        delta_norm = float(minimum_energy_record["selected_delta_norm"])
        control_multiplier = delta_norm / max(step_norm_budget, 1e-12)
    elif endpoint_control_enabled:
        # 旧调用只能保持数值兼容，明确标记为不完整，不能支持正式 P3 Claim。
        control_multiplier, endpoint_response_before = _endpoint_control_multiplier(
            sample,
            key_direction,
            config,
        )
        delta_norm = step_norm_budget * control_multiplier
        minimum_energy_record = {
            "endpoint_minimum_energy_control_status": (
                "compatibility_heuristic_missing_time_and_energy_context"
            ),
            "endpoint_quality_energy_guard_passed": False,
        }
    else:
        control_multiplier = 1.0
        delta_norm = step_norm_budget
        minimum_energy_record = {
            "endpoint_minimum_energy_control_status": "disabled_endpoint_agnostic_ablation",
            "endpoint_quality_energy_guard_passed": True,
        }
    signed_direction = safe_direction * float(config.scheduler_velocity_sign)
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
        "endpoint_control_formal_context_complete": formal_control_context_complete,
        "endpoint_control_policy": (
            "finite_difference_endpoint_minimum_energy_approximation"
            if formal_control_context_complete
            else "compatibility_or_ablation"
        ),
        "semantic_projection_status": semantic_projection_status,
        "semantic_projection_retained_key_energy_ratio": round(
            retained_key_energy_ratio,
            8,
        ),
        "scheduler_velocity_sign": float(config.scheduler_velocity_sign),
        "velocity_norm_ratio_budget": float(config.velocity_norm_ratio_budget),
        "flow_energy_budget_ratio": float(config.flow_energy_budget_ratio),
        **{
            key: (
                round(value, 10)
                if isinstance(value, float) and math.isfinite(value)
                else value
            )
            for key, value in minimum_energy_record.items()
        },
    }
    return constrained, record
