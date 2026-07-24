"""Budgeted injection for prompt-orthogonal state-trajectory directions."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from main.methods.state_space_watermark.flow_tubelet_key_code import (
    FlowTubeletKeyCodeConfig,
    flow_phase_weight,
)
from main.methods.state_space_watermark.state_rotation_operator import (
    PromptOrthogonalDirection,
)
from main.methods.state_space_watermark.velocity_field_constraint import (
    VelocityControlContext,
    VelocityFieldConstraintConfig,
)


PROMPT_ORTHOGONAL_STATE_TRAJECTORY_CARRIER_ID = (
    "prompt_independent_rank_two_state_rotation_continuous_code"
)
PROMPT_ORTHOGONAL_LAMBDA_MAX = 0.12
PROMPT_ORTHOGONAL_VELOCITY_NORM_RATIO_BUDGET = 0.02
PROMPT_ORTHOGONAL_FLOW_ENERGY_BUDGET_RATIO = 0.000015
PROMPT_ORTHOGONAL_DIRECTION_COSINE_MINIMUM = 0.999
PROMPT_ORTHOGONAL_BUDGET_RELATIVE_TOLERANCE = 1e-5
PROMPT_ORTHOGONAL_SCHEDULER_CONTROL_DTYPE = "float32"


@dataclass(frozen=True)
class PromptOrthogonalInjectionConfig:
    """Freeze the first mechanism-smoke injection rather than tune on results."""

    carrier_id: str = PROMPT_ORTHOGONAL_STATE_TRAJECTORY_CARRIER_ID
    lambda_max: float = PROMPT_ORTHOGONAL_LAMBDA_MAX
    velocity_norm_ratio_budget: float = (
        PROMPT_ORTHOGONAL_VELOCITY_NORM_RATIO_BUDGET
    )
    flow_energy_budget_ratio: float = (
        PROMPT_ORTHOGONAL_FLOW_ENERGY_BUDGET_RATIO
    )
    minimum_direction_cosine: float = (
        PROMPT_ORTHOGONAL_DIRECTION_COSINE_MINIMUM
    )

    def __post_init__(self) -> None:
        if self.carrier_id != PROMPT_ORTHOGONAL_STATE_TRAJECTORY_CARRIER_ID:
            raise ValueError("prompt-orthogonal carrier id 未冻结")
        numeric = (
            (self.lambda_max, PROMPT_ORTHOGONAL_LAMBDA_MAX),
            (
                self.velocity_norm_ratio_budget,
                PROMPT_ORTHOGONAL_VELOCITY_NORM_RATIO_BUDGET,
            ),
            (
                self.flow_energy_budget_ratio,
                PROMPT_ORTHOGONAL_FLOW_ENERGY_BUDGET_RATIO,
            ),
            (
                self.minimum_direction_cosine,
                PROMPT_ORTHOGONAL_DIRECTION_COSINE_MINIMUM,
            ),
        )
        if any(
            not math.isclose(
                float(observed),
                float(required),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for observed, required in numeric
        ):
            raise ValueError("prompt-orthogonal injection 数值配置未冻结")


@dataclass(frozen=True)
class StateTrajectoryInjectionRecord:
    """Diagnostics derived from the original and final scheduler velocities."""

    status: str
    flow_phase: float
    flow_phase_weight: float
    continuous_code: float
    velocity_norm_before: float
    velocity_norm_after: float
    delta_norm: float
    joint_norm_budget: float
    energy_limited_delta_norm: float
    joint_scale: float
    direction_cosine: float | None
    norm_guard_passed: bool | None
    energy_guard_passed: bool | None
    direction_guard_passed: bool | None
    control_energy_increment: float
    control_cumulative_energy_after: float
    reference_energy_increment: float
    reference_cumulative_energy_after: float
    inactive_phase_noop: bool
    endpoint_control_enabled: bool


def _budget_guard_passed(observed: float, budget: float) -> bool:
    actual = float(observed)
    limit = float(budget)
    return bool(
        math.isfinite(actual)
        and math.isfinite(limit)
        and actual >= 0.0
        and limit >= 0.0
        and (
            actual <= limit
            or math.isclose(
                actual,
                limit,
                rel_tol=PROMPT_ORTHOGONAL_BUDGET_RELATIVE_TOLERANCE,
                abs_tol=1e-10,
            )
        )
    )


def apply_prompt_orthogonal_state_trajectory_injection(
    model_output: Any,
    state: Any,
    keyed_direction: PromptOrthogonalDirection,
    *,
    continuous_code: float,
    flow_phase: float,
    control_context: VelocityControlContext,
    injection_config: PromptOrthogonalInjectionConfig | None = None,
    tubelet_config: FlowTubeletKeyCodeConfig | None = None,
    velocity_config: VelocityFieldConstraintConfig | None = None,
) -> tuple[Any, StateTrajectoryInjectionRecord]:
    """Apply one AC-only delta with one joint norm/energy scale."""

    injection_config = injection_config or PromptOrthogonalInjectionConfig()
    tubelet_config = tubelet_config or FlowTubeletKeyCodeConfig()
    velocity_config = velocity_config or VelocityFieldConstraintConfig()
    if model_output.shape != state.shape or (
        model_output.shape != keyed_direction.direction.shape
    ):
        raise ValueError("model output、state 与 keyed direction 必须同形")
    expected_velocity_values = (
        (
            velocity_config.lambda_max,
            injection_config.lambda_max,
            "lambda_max",
        ),
        (
            velocity_config.velocity_norm_ratio_budget,
            injection_config.velocity_norm_ratio_budget,
            "velocity_norm_ratio_budget",
        ),
        (
            velocity_config.flow_energy_budget_ratio,
            injection_config.flow_energy_budget_ratio,
            "flow_energy_budget_ratio",
        ),
    )
    for observed, expected, name in expected_velocity_values:
        if not math.isclose(
            float(observed),
            float(expected),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"prompt-orthogonal {name} 与冻结配置不一致")
    code = float(continuous_code)
    if not math.isfinite(code) or abs(code) > 1.0 + 1e-12:
        raise ValueError("continuous code 必须有限且位于[-1,1]")
    interval = float(control_context.delta_sigma)
    if (
        not math.isfinite(interval)
        or abs(interval) <= 1e-12
        or control_context.cumulative_control_energy < 0.0
        or control_context.cumulative_reference_energy < 0.0
        or int(control_context.remaining_step_count) <= 0
    ):
        raise ValueError("prompt-orthogonal control context 不完整")
    schedule_weight = flow_phase_weight(flow_phase, tubelet_config)
    base = model_output.detach().float()
    base_norm = float(base.norm().item())
    reference_increment = interval**2 * float(base.square().sum().item())
    reference_cumulative_after = (
        float(control_context.cumulative_reference_energy)
        + reference_increment
    )
    projected_reference = (
        float(control_context.cumulative_reference_energy)
        + reference_increment
        * max(1, int(control_context.remaining_step_count))
    )
    total_energy_budget = (
        injection_config.flow_energy_budget_ratio * projected_reference
    )
    remaining_energy = max(
        0.0,
        total_energy_budget
        - float(control_context.cumulative_control_energy),
    )
    if schedule_weight <= 1e-12:
        if abs(code) > 1e-12:
            raise ValueError("inactive flow phase 的 continuous code 必须为零")
        return model_output, StateTrajectoryInjectionRecord(
            status="inactive_flow_phase",
            flow_phase=float(flow_phase),
            flow_phase_weight=0.0,
            continuous_code=0.0,
            velocity_norm_before=base_norm,
            velocity_norm_after=base_norm,
            delta_norm=0.0,
            joint_norm_budget=0.0,
            energy_limited_delta_norm=(
                math.sqrt(remaining_energy) / abs(interval)
            ),
            joint_scale=0.0,
            direction_cosine=None,
            norm_guard_passed=None,
            energy_guard_passed=None,
            direction_guard_passed=None,
            control_energy_increment=0.0,
            control_cumulative_energy_after=float(
                control_context.cumulative_control_energy
            ),
            reference_energy_increment=reference_increment,
            reference_cumulative_energy_after=reference_cumulative_after,
            inactive_phase_noop=True,
            endpoint_control_enabled=False,
        )
    if abs(code) <= 1e-12:
        raise ValueError("active flow phase 的 continuous code 不得为零")
    if not keyed_direction.active:
        raise RuntimeError("active continuous code 缺少合法 state tangent direction")
    direction = keyed_direction.direction.detach().float()
    direction_norm = float(direction.norm().item())
    if not math.isfinite(direction_norm) or direction_norm <= 1e-12:
        raise RuntimeError("state trajectory direction norm 退化")
    signed_unit_direction = direction * (
        math.copysign(1.0, code) / direction_norm
    )
    joint_norm_budget = (
        base_norm
        * injection_config.velocity_norm_ratio_budget
        * injection_config.lambda_max
        * schedule_weight
    )
    candidate_delta = (
        signed_unit_direction * joint_norm_budget * abs(code)
    )
    candidate_norm = float(candidate_delta.norm().item())
    energy_limited_norm = math.sqrt(remaining_energy) / abs(interval)
    admissible_norm = min(joint_norm_budget, energy_limited_norm)
    joint_scale = min(
        1.0,
        admissible_norm / max(candidate_norm, 1e-12),
    )
    delta = candidate_delta * joint_scale
    # The FlowMatch scheduler consumes FP32 control for both watermarked and
    # clean variants.  Casting the small delta back to bf16 here can erase it.
    constrained = base + delta
    applied_delta = constrained.detach().float() - base
    actual_delta_norm = float(applied_delta.norm().item())
    energy_increment = interval**2 * actual_delta_norm**2
    cosine = float(
        (
            applied_delta.reshape(-1)
            @ signed_unit_direction.reshape(-1)
            / (
                applied_delta.norm().clamp_min(1e-12)
                * signed_unit_direction.norm().clamp_min(1e-12)
            )
        ).item()
    )
    norm_guard = _budget_guard_passed(
        actual_delta_norm,
        joint_norm_budget,
    )
    energy_guard = _budget_guard_passed(
        energy_increment,
        remaining_energy,
    )
    direction_guard = bool(
        math.isfinite(cosine)
        and cosine + 1e-12
        >= injection_config.minimum_direction_cosine
    )
    if not norm_guard or not energy_guard or not direction_guard:
        raise RuntimeError("prompt-orthogonal 最终 direction/norm/energy guard 失败")
    return constrained, StateTrajectoryInjectionRecord(
        status="prompt_orthogonal_state_trajectory_applied",
        flow_phase=float(flow_phase),
        flow_phase_weight=float(schedule_weight),
        continuous_code=code,
        velocity_norm_before=base_norm,
        velocity_norm_after=float(
            constrained.detach().float().norm().item()
        ),
        delta_norm=actual_delta_norm,
        joint_norm_budget=joint_norm_budget,
        energy_limited_delta_norm=energy_limited_norm,
        joint_scale=joint_scale,
        direction_cosine=cosine,
        norm_guard_passed=norm_guard,
        energy_guard_passed=energy_guard,
        direction_guard_passed=direction_guard,
        control_energy_increment=energy_increment,
        control_cumulative_energy_after=(
            float(control_context.cumulative_control_energy)
            + energy_increment
        ),
        reference_energy_increment=reference_increment,
        reference_cumulative_energy_after=reference_cumulative_after,
        inactive_phase_noop=False,
        endpoint_control_enabled=False,
    )
