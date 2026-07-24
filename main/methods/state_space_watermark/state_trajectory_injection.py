"""Budgeted injection for prompt-orthogonal state-trajectory directions."""

from __future__ import annotations

from dataclasses import dataclass
import json
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
PROMPT_ORTHOGONAL_FINITE_PRECISION_BACKOFF_SAFETY_FACTOR = 0.999
PROMPT_ORTHOGONAL_FINITE_PRECISION_MAX_BACKOFF_COUNT = 8
PROMPT_ORTHOGONAL_FINITE_PRECISION_REFINEMENT_COUNT = 12
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
    candidate_delta_norm: float
    intended_delta_norm_before_projection: float
    finite_precision_projection_scale: float
    finite_precision_projection_attempt_count: int
    finite_precision_backoff_count: int
    finite_precision_projection_status: str
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


@dataclass(frozen=True)
class _FinitePrecisionDeltaEvaluation:
    constrained: Any
    projection_scale: float
    actual_delta_norm: float
    energy_increment: float
    direction_cosine: float | None
    norm_guard_passed: bool
    energy_guard_passed: bool
    direction_guard_passed: bool

    @property
    def all_guards_passed(self) -> bool:
        return bool(
            self.actual_delta_norm > 0.0
            and self.norm_guard_passed
            and self.energy_guard_passed
            and self.direction_guard_passed
        )


@dataclass(frozen=True)
class _FinitePrecisionDeltaSelection:
    evaluation: _FinitePrecisionDeltaEvaluation
    attempt_count: int
    backoff_count: int
    status: str


class PromptOrthogonalInjectionGuardError(RuntimeError):
    """Fail closed while retaining only safe scalar diagnostics."""

    def __init__(self, diagnostics: dict[str, object]) -> None:
        self.diagnostics = dict(diagnostics)
        super().__init__(
            "prompt-orthogonal 最终 direction/norm/energy guard 失败: "
            + json.dumps(
                self.diagnostics,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )


def _budget_guard_passed(observed: float, budget: float) -> bool:
    actual = float(observed)
    limit = float(budget)
    return bool(
        math.isfinite(actual)
        and math.isfinite(limit)
        and actual >= 0.0
        and limit >= 0.0
        and actual <= limit
    )


def _evaluate_finite_precision_delta(
    *,
    base: Any,
    intended_delta: Any,
    signed_unit_direction: Any,
    projection_scale: float,
    interval: float,
    joint_norm_budget: float,
    remaining_energy: float,
    minimum_direction_cosine: float,
) -> _FinitePrecisionDeltaEvaluation:
    scale = float(projection_scale)
    constrained = base + intended_delta * scale
    applied_delta = constrained.detach().float() - base
    actual_delta_norm = float(applied_delta.norm().item())
    energy_increment = float(interval) ** 2 * actual_delta_norm**2
    if actual_delta_norm > 0.0 and math.isfinite(actual_delta_norm):
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
    else:
        cosine = None
    norm_guard = _budget_guard_passed(
        actual_delta_norm,
        joint_norm_budget,
    )
    energy_guard = _budget_guard_passed(
        energy_increment,
        remaining_energy,
    )
    direction_guard = bool(
        cosine is not None
        and math.isfinite(cosine)
        and cosine + 1e-12 >= float(minimum_direction_cosine)
    )
    return _FinitePrecisionDeltaEvaluation(
        constrained=constrained,
        projection_scale=scale,
        actual_delta_norm=actual_delta_norm,
        energy_increment=energy_increment,
        direction_cosine=cosine,
        norm_guard_passed=norm_guard,
        energy_guard_passed=energy_guard,
        direction_guard_passed=direction_guard,
    )


def _guard_failure_diagnostics(
    *,
    evaluation: _FinitePrecisionDeltaEvaluation,
    initial_evaluation: _FinitePrecisionDeltaEvaluation,
    attempt_count: int,
    backoff_count: int,
    candidate_delta_norm: float,
    intended_delta_norm_before_projection: float,
    joint_norm_budget: float,
    remaining_energy: float,
    flow_phase: float,
    schedule_weight: float,
    continuous_code: float,
) -> dict[str, object]:
    return {
        "actual_delta_norm": evaluation.actual_delta_norm,
        "candidate_delta_norm": float(candidate_delta_norm),
        "continuous_code": float(continuous_code),
        "direction_cosine": evaluation.direction_cosine,
        "direction_guard_passed": evaluation.direction_guard_passed,
        "energy_guard_passed": evaluation.energy_guard_passed,
        "energy_increment": evaluation.energy_increment,
        "finite_precision_backoff_count": int(backoff_count),
        "finite_precision_projection_attempt_count": int(attempt_count),
        "finite_precision_projection_scale": evaluation.projection_scale,
        "flow_phase": float(flow_phase),
        "flow_phase_weight": float(schedule_weight),
        "initial_actual_delta_norm": (
            initial_evaluation.actual_delta_norm
        ),
        "initial_direction_cosine": initial_evaluation.direction_cosine,
        "initial_energy_guard_passed": (
            initial_evaluation.energy_guard_passed
        ),
        "initial_norm_guard_passed": initial_evaluation.norm_guard_passed,
        "intended_delta_norm_before_projection": float(
            intended_delta_norm_before_projection
        ),
        "joint_norm_budget": float(joint_norm_budget),
        "norm_guard_passed": evaluation.norm_guard_passed,
        "remaining_energy": float(remaining_energy),
    }


def _select_finite_precision_delta(
    *,
    base: Any,
    intended_delta: Any,
    signed_unit_direction: Any,
    interval: float,
    joint_norm_budget: float,
    remaining_energy: float,
    minimum_direction_cosine: float,
    candidate_delta_norm: float,
    intended_delta_norm_before_projection: float,
    flow_phase: float,
    schedule_weight: float,
    continuous_code: float,
) -> _FinitePrecisionDeltaSelection:
    """Select the largest observed feasible FP32 control on a fixed search."""

    attempt_count = 1
    backoff_count = 0
    initial = _evaluate_finite_precision_delta(
        base=base,
        intended_delta=intended_delta,
        signed_unit_direction=signed_unit_direction,
        projection_scale=1.0,
        interval=interval,
        joint_norm_budget=joint_norm_budget,
        remaining_energy=remaining_energy,
        minimum_direction_cosine=minimum_direction_cosine,
    )
    if initial.all_guards_passed:
        return _FinitePrecisionDeltaSelection(
            evaluation=initial,
            attempt_count=attempt_count,
            backoff_count=backoff_count,
            status="direct_actual_delta_pass",
        )
    if not initial.direction_guard_passed:
        raise PromptOrthogonalInjectionGuardError(
            _guard_failure_diagnostics(
                evaluation=initial,
                initial_evaluation=initial,
                attempt_count=attempt_count,
                backoff_count=backoff_count,
                candidate_delta_norm=candidate_delta_norm,
                intended_delta_norm_before_projection=(
                    intended_delta_norm_before_projection
                ),
                joint_norm_budget=joint_norm_budget,
                remaining_energy=remaining_energy,
                flow_phase=flow_phase,
                schedule_weight=schedule_weight,
                continuous_code=continuous_code,
            )
        )

    previous_infeasible = initial
    feasible: _FinitePrecisionDeltaEvaluation | None = None
    upper_infeasible_scale = 1.0
    for _ in range(PROMPT_ORTHOGONAL_FINITE_PRECISION_MAX_BACKOFF_COUNT):
        correction_candidates: list[float] = []
        if not previous_infeasible.norm_guard_passed:
            correction_candidates.append(
                float(joint_norm_budget)
                / max(previous_infeasible.actual_delta_norm, 1e-30)
            )
        if not previous_infeasible.energy_guard_passed:
            correction_candidates.append(
                math.sqrt(
                    float(remaining_energy)
                    / max(previous_infeasible.energy_increment, 1e-30)
                )
            )
        if not correction_candidates:
            break
        correction = min(correction_candidates)
        step_factor = min(
            PROMPT_ORTHOGONAL_FINITE_PRECISION_BACKOFF_SAFETY_FACTOR,
            correction
            * PROMPT_ORTHOGONAL_FINITE_PRECISION_BACKOFF_SAFETY_FACTOR,
        )
        next_scale = previous_infeasible.projection_scale * step_factor
        if (
            not math.isfinite(next_scale)
            or next_scale <= 0.0
            or next_scale >= previous_infeasible.projection_scale
        ):
            break
        upper_infeasible_scale = previous_infeasible.projection_scale
        backoff_count += 1
        attempt_count += 1
        trial = _evaluate_finite_precision_delta(
            base=base,
            intended_delta=intended_delta,
            signed_unit_direction=signed_unit_direction,
            projection_scale=next_scale,
            interval=interval,
            joint_norm_budget=joint_norm_budget,
            remaining_energy=remaining_energy,
            minimum_direction_cosine=minimum_direction_cosine,
        )
        if trial.all_guards_passed:
            feasible = trial
            break
        previous_infeasible = trial
        if not trial.direction_guard_passed:
            break

    if feasible is None:
        raise PromptOrthogonalInjectionGuardError(
            _guard_failure_diagnostics(
                evaluation=previous_infeasible,
                initial_evaluation=initial,
                attempt_count=attempt_count,
                backoff_count=backoff_count,
                candidate_delta_norm=candidate_delta_norm,
                intended_delta_norm_before_projection=(
                    intended_delta_norm_before_projection
                ),
                joint_norm_budget=joint_norm_budget,
                remaining_energy=remaining_energy,
                flow_phase=flow_phase,
                schedule_weight=schedule_weight,
                continuous_code=continuous_code,
            )
        )

    best = feasible
    for _ in range(PROMPT_ORTHOGONAL_FINITE_PRECISION_REFINEMENT_COUNT):
        midpoint = 0.5 * (
            best.projection_scale + upper_infeasible_scale
        )
        if midpoint <= best.projection_scale:
            break
        attempt_count += 1
        trial = _evaluate_finite_precision_delta(
            base=base,
            intended_delta=intended_delta,
            signed_unit_direction=signed_unit_direction,
            projection_scale=midpoint,
            interval=interval,
            joint_norm_budget=joint_norm_budget,
            remaining_energy=remaining_energy,
            minimum_direction_cosine=minimum_direction_cosine,
        )
        if trial.all_guards_passed:
            best = trial
        else:
            upper_infeasible_scale = midpoint
    return _FinitePrecisionDeltaSelection(
        evaluation=best,
        attempt_count=attempt_count,
        backoff_count=backoff_count,
        status="bounded_actual_delta_backoff_pass",
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
            candidate_delta_norm=0.0,
            intended_delta_norm_before_projection=0.0,
            finite_precision_projection_scale=0.0,
            finite_precision_projection_attempt_count=0,
            finite_precision_backoff_count=0,
            finite_precision_projection_status="inactive_flow_phase_noop",
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
    intended_delta_norm = float(delta.norm().item())
    # The scheduler consumes the representable FP32 control.  At a very small
    # edge-of-window budget, ``(base + delta) - base`` can exceed the analytic
    # norm by many FP32 reduction ulps.  Project the actual representable delta
    # back inside the unchanged budgets rather than weakening any guard.
    selection = _select_finite_precision_delta(
        base=base,
        intended_delta=delta,
        signed_unit_direction=signed_unit_direction,
        interval=interval,
        joint_norm_budget=joint_norm_budget,
        remaining_energy=remaining_energy,
        minimum_direction_cosine=(
            injection_config.minimum_direction_cosine
        ),
        candidate_delta_norm=candidate_norm,
        intended_delta_norm_before_projection=intended_delta_norm,
        flow_phase=flow_phase,
        schedule_weight=schedule_weight,
        continuous_code=code,
    )
    selected = selection.evaluation
    constrained = selected.constrained
    return constrained, StateTrajectoryInjectionRecord(
        status="prompt_orthogonal_state_trajectory_applied",
        flow_phase=float(flow_phase),
        flow_phase_weight=float(schedule_weight),
        continuous_code=code,
        velocity_norm_before=base_norm,
        velocity_norm_after=float(
            constrained.detach().float().norm().item()
        ),
        delta_norm=selected.actual_delta_norm,
        joint_norm_budget=joint_norm_budget,
        energy_limited_delta_norm=energy_limited_norm,
        joint_scale=joint_scale,
        candidate_delta_norm=candidate_norm,
        intended_delta_norm_before_projection=intended_delta_norm,
        finite_precision_projection_scale=(
            selected.projection_scale
        ),
        finite_precision_projection_attempt_count=selection.attempt_count,
        finite_precision_backoff_count=selection.backoff_count,
        finite_precision_projection_status=selection.status,
        direction_cosine=selected.direction_cosine,
        norm_guard_passed=selected.norm_guard_passed,
        energy_guard_passed=selected.energy_guard_passed,
        direction_guard_passed=selected.direction_guard_passed,
        control_energy_increment=selected.energy_increment,
        control_cumulative_energy_after=(
            float(control_context.cumulative_control_energy)
            + selected.energy_increment
        ),
        reference_energy_increment=reference_increment,
        reference_cumulative_energy_after=reference_cumulative_after,
        inactive_phase_noop=False,
        endpoint_control_enabled=False,
    )
