"""构造用于 keyed forward replay 同步检测的时变轨迹载体。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from itertools import combinations
import json
import math
from typing import Any, Sequence

from main.methods.state_space_watermark.flow_tubelet_key_code import (
    flow_phase_weight,
)
from main.methods.state_space_watermark.velocity_field_constraint import (
    VelocityControlContext,
    VelocityFieldConstraintConfig,
    apply_velocity_field_constraint,
)


PREDICTIVE_TRAJECTORY_CARRIER_ID = (
    "key_context_balanced_multisegment_predictive_carrier"
)
PREDICTIVE_TRAJECTORY_PHASE_SEGMENT_COUNT = 8
PREDICTIVE_TRAJECTORY_MINIMUM_ACTIVE_PHASE_COUNT = 4
PREDICTIVE_TRAJECTORY_MINIMUM_ACTIVE_CODE_MAGNITUDE = 0.25
PREDICTIVE_TRAJECTORY_MINIMUM_WEIGHTED_CODE_ENERGY = 0.20
PREDICTIVE_TRAJECTORY_MAXIMUM_ABSOLUTE_CODE_CORRELATION = 0.75
PREDICTIVE_TRAJECTORY_BUDGET_GUARD_RELATIVE_TOLERANCE = 1e-5


def _predictive_budget_guard_passed(
    observed: float,
    budget: float,
) -> bool:
    """容纳 GPU 浮点范数归约误差，同时拒绝实质预算超限。"""

    actual = float(observed)
    limit = float(budget)
    if (
        not math.isfinite(actual)
        or not math.isfinite(limit)
        or actual < 0.0
        or limit < 0.0
    ):
        return False
    return actual <= limit or math.isclose(
        actual,
        limit,
        rel_tol=PREDICTIVE_TRAJECTORY_BUDGET_GUARD_RELATIVE_TOLERANCE,
        abs_tol=1e-10,
    )


@dataclass(frozen=True)
class PredictiveTrajectoryCarrierConfig:
    """冻结无独立 DC 通道的预测同步轨迹载体。"""

    carrier_id: str = PREDICTIVE_TRAJECTORY_CARRIER_ID
    phase_segment_count: int = PREDICTIVE_TRAJECTORY_PHASE_SEGMENT_COUNT
    minimum_active_phase_count: int = (
        PREDICTIVE_TRAJECTORY_MINIMUM_ACTIVE_PHASE_COUNT
    )
    minimum_active_code_magnitude: float = (
        PREDICTIVE_TRAJECTORY_MINIMUM_ACTIVE_CODE_MAGNITUDE
    )
    minimum_weighted_code_energy: float = (
        PREDICTIVE_TRAJECTORY_MINIMUM_WEIGHTED_CODE_ENERGY
    )
    maximum_absolute_code_correlation: float = (
        PREDICTIVE_TRAJECTORY_MAXIMUM_ABSOLUTE_CODE_CORRELATION
    )

    def __post_init__(self) -> None:
        expected = {
            "carrier_id": PREDICTIVE_TRAJECTORY_CARRIER_ID,
            "phase_segment_count": PREDICTIVE_TRAJECTORY_PHASE_SEGMENT_COUNT,
            "minimum_active_phase_count": (
                PREDICTIVE_TRAJECTORY_MINIMUM_ACTIVE_PHASE_COUNT
            ),
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"predictive trajectory {name} 未冻结")
        numeric = (
            (
                self.minimum_active_code_magnitude,
                PREDICTIVE_TRAJECTORY_MINIMUM_ACTIVE_CODE_MAGNITUDE,
            ),
            (
                self.minimum_weighted_code_energy,
                PREDICTIVE_TRAJECTORY_MINIMUM_WEIGHTED_CODE_ENERGY,
            ),
            (
                self.maximum_absolute_code_correlation,
                PREDICTIVE_TRAJECTORY_MAXIMUM_ABSOLUTE_CODE_CORRELATION,
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
            raise ValueError("predictive trajectory 数值门槛未冻结")


@dataclass(frozen=True)
class PredictiveTrajectorySchedule:
    """保存可在冻结的 generation/replay 同网格重建的平衡多段时间码。"""

    codes: tuple[float, ...]
    raw_signs: tuple[int, ...]
    active_weights: tuple[float, ...]
    phase_segments: tuple[int, ...]
    phase_codebook_signs: tuple[int, ...]
    phase_function_digest: str
    schedule_digest: str
    weighted_mean: float
    weighted_residual: float
    active_phase_count: int
    minimum_active_code_magnitude: float
    weighted_code_energy: float

    def metadata_for_step(self, step_index: int) -> dict[str, Any]:
        index = int(step_index)
        return {
            "predictive_trajectory_carrier_id": (
                PREDICTIVE_TRAJECTORY_CARRIER_ID
            ),
            "predictive_trajectory_schedule_digest": self.schedule_digest,
            "predictive_trajectory_phase_function_digest": (
                self.phase_function_digest
            ),
            "predictive_trajectory_phase_segment": int(
                self.phase_segments[index]
            ),
            "predictive_trajectory_ac_code": round(
                float(self.codes[index]),
                10,
            ),
            "predictive_trajectory_ac_raw_sign": int(
                self.raw_signs[index]
            ),
            "predictive_trajectory_ac_weight": round(
                float(self.active_weights[index]),
                10,
            ),
            "predictive_trajectory_ac_weighted_mean": round(
                float(self.weighted_mean),
                12,
            ),
            "predictive_trajectory_ac_weighted_residual": round(
                float(self.weighted_residual),
                12,
            ),
            "predictive_trajectory_active_phase_count": int(
                self.active_phase_count
            ),
            "predictive_trajectory_minimum_active_code_magnitude": round(
                float(self.minimum_active_code_magnitude),
                10,
            ),
            "predictive_trajectory_weighted_code_energy": round(
                float(self.weighted_code_energy),
                10,
            ),
            "predictive_trajectory_noncollapse_verified": bool(
                self.active_phase_count
                >= PREDICTIVE_TRAJECTORY_MINIMUM_ACTIVE_PHASE_COUNT
                and self.minimum_active_code_magnitude + 1e-12
                >= PREDICTIVE_TRAJECTORY_MINIMUM_ACTIVE_CODE_MAGNITUDE
                and self.weighted_code_energy + 1e-12
                >= PREDICTIVE_TRAJECTORY_MINIMUM_WEIGHTED_CODE_ENERGY
            ),
        }


def _balanced_phase_codebook(
    *,
    key_text: str,
    key_context_digest: str,
    phase_segments: Sequence[int],
    active_weights: Sequence[float],
) -> tuple[int, ...]:
    """在真实 active segments 上选一个不塌缩的 keyed 四正四负码。"""

    key_digest = sha256(
        (
            f"{PREDICTIVE_TRAJECTORY_CARRIER_ID}::phase_codebook::"
            f"{key_text}::{key_context_digest}"
        ).encode("utf-8")
    ).digest()
    segment_weights = [
        sum(
            float(weight)
            for segment, weight in zip(
                phase_segments,
                active_weights,
                strict=True,
            )
            if int(segment) == segment_index
        )
        for segment_index in range(PREDICTIVE_TRAJECTORY_PHASE_SEGMENT_COUNT)
    ]
    active_segments = tuple(
        index for index, weight in enumerate(segment_weights) if weight > 0.0
    )
    total_weight = sum(segment_weights)
    valid_positive_sets: list[tuple[int, ...]] = []
    for positive_count in range(1, len(active_segments)):
        if positive_count > 4 or len(active_segments) - positive_count > 4:
            continue
        for positive_tuple in combinations(active_segments, positive_count):
            positive = set(positive_tuple)
            weighted_mean = sum(
                weight * (1.0 if index in positive else -1.0)
                for index, weight in enumerate(segment_weights)
            ) / total_weight
            centered = [
                (1.0 if index in positive else -1.0) - weighted_mean
                for index in active_segments
            ]
            maximum_magnitude = max(abs(value) for value in centered)
            normalized = [value / maximum_magnitude for value in centered]
            minimum_magnitude = min(abs(value) for value in normalized)
            energy = sum(
                segment_weights[index] * value * value
                for index, value in zip(
                    active_segments,
                    normalized,
                    strict=True,
                )
            ) / total_weight
            if (
                minimum_magnitude + 1e-12
                >= PREDICTIVE_TRAJECTORY_MINIMUM_ACTIVE_CODE_MAGNITUDE
                and energy + 1e-12
                >= PREDICTIVE_TRAJECTORY_MINIMUM_WEIGHTED_CODE_ENERGY
            ):
                valid_positive_sets.append(tuple(sorted(positive)))
    if not valid_positive_sets:
        raise RuntimeError(
            "predictive trajectory active schedule 不存在非塌缩平衡码"
        )
    selected_active_positive = min(
        valid_positive_sets,
        key=lambda values: sha256(
            key_digest
            + json.dumps(
                list(values),
                separators=(",", ":"),
            ).encode("utf-8")
        ).digest(),
    )
    positive = set(selected_active_positive)
    inactive_segments = [
        index
        for index in range(PREDICTIVE_TRAJECTORY_PHASE_SEGMENT_COUNT)
        if index not in active_segments
    ]
    inactive_ranking = sorted(
        inactive_segments,
        key=lambda index: (
            sha256(key_digest + bytes([index])).digest(),
            index,
        ),
    )
    positive.update(inactive_ranking[: 4 - len(positive)])
    if len(positive) != 4:
        raise RuntimeError("predictive trajectory 无法补全四正四负码")
    return tuple(
        1 if index in positive else -1
        for index in range(PREDICTIVE_TRAJECTORY_PHASE_SEGMENT_COUNT)
    )


def build_predictive_trajectory_schedule(
    *,
    key_text: str,
    key_context_digest: str,
    flow_phases: Sequence[float],
    active_weights: Sequence[float],
) -> PredictiveTrajectorySchedule:
    """在真实 schedule 上实例化平衡多段时间码并 fail-closed。"""

    phases = tuple(float(value) for value in flow_phases)
    weights = tuple(max(0.0, float(value)) for value in active_weights)
    if not key_text:
        raise ValueError("predictive trajectory schedule 缺少 key")
    if len(key_context_digest) != 64:
        raise ValueError("predictive trajectory schedule 缺少64位 context digest")
    if not phases or len(phases) != len(weights):
        raise ValueError("predictive trajectory phase/weight 必须等长非空")
    if any(not 0.0 <= phase <= 1.0 for phase in phases):
        raise ValueError("predictive trajectory phase 必须位于 [0,1]")
    active_indices = [
        index for index, weight in enumerate(weights) if weight > 0.0
    ]
    if len(active_indices) < PREDICTIVE_TRAJECTORY_MINIMUM_ACTIVE_PHASE_COUNT:
        raise RuntimeError("predictive trajectory 有效 active phase 数不足")

    segment_count = PREDICTIVE_TRAJECTORY_PHASE_SEGMENT_COUNT
    phase_segments: list[int] = []
    for phase, weight in zip(phases, weights, strict=True):
        normalized = max(0.0, min(1.0, (phase - 0.25) / 0.50))
        segment = min(segment_count - 1, int(math.floor(normalized * segment_count)))
        phase_segments.append(segment)
    codebook = _balanced_phase_codebook(
        key_text=key_text,
        key_context_digest=key_context_digest,
        phase_segments=phase_segments,
        active_weights=weights,
    )
    raw_signs = [
        codebook[segment] if weight > 0.0 else 0
        for segment, weight in zip(phase_segments, weights, strict=True)
    ]
    if len({raw_signs[index] for index in active_indices}) != 2:
        raise RuntimeError("predictive trajectory active schedule 未覆盖正负时间码")

    total_weight = sum(weights)
    weighted_mean = sum(
        weight * sign
        for weight, sign in zip(weights, raw_signs, strict=True)
    ) / total_weight
    centered = [
        float(sign) - weighted_mean if weight > 0.0 else 0.0
        for sign, weight in zip(raw_signs, weights, strict=True)
    ]
    maximum_magnitude = max(abs(value) for value in centered)
    if maximum_magnitude <= 1e-12:
        raise RuntimeError("predictive trajectory centered code 退化")
    codes = tuple(value / maximum_magnitude for value in centered)
    residual = sum(
        weight * code
        for weight, code in zip(weights, codes, strict=True)
    )
    minimum_magnitude = min(abs(codes[index]) for index in active_indices)
    energy = sum(
        weight * code * code
        for weight, code in zip(weights, codes, strict=True)
    ) / total_weight
    if abs(residual) > 1e-10:
        raise RuntimeError("predictive trajectory AC code 未达到加权零均值")
    if (
        minimum_magnitude + 1e-12
        < PREDICTIVE_TRAJECTORY_MINIMUM_ACTIVE_CODE_MAGNITUDE
    ):
        raise RuntimeError("predictive trajectory active code magnitude 退化")
    if (
        energy + 1e-12
        < PREDICTIVE_TRAJECTORY_MINIMUM_WEIGHTED_CODE_ENERGY
    ):
        raise RuntimeError("predictive trajectory weighted code energy 退化")

    phase_function_payload = {
        "carrier_id": PREDICTIVE_TRAJECTORY_CARRIER_ID,
        "key_context_digest": key_context_digest,
        "key_binding_digest": sha256(key_text.encode("utf-8")).hexdigest(),
        "phase_codebook_signs": list(codebook),
        "phase_segment_count": segment_count,
    }
    phase_function_digest = sha256(
        json.dumps(
            phase_function_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    schedule_payload = {
        **phase_function_payload,
        "flow_phases": [format(value, ".17g") for value in phases],
        "active_weights": [format(value, ".17g") for value in weights],
        "phase_segments": phase_segments,
        "codes": [format(value, ".17g") for value in codes],
    }
    schedule_digest = sha256(
        json.dumps(
            schedule_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return PredictiveTrajectorySchedule(
        codes=codes,
        raw_signs=tuple(raw_signs),
        active_weights=weights,
        phase_segments=tuple(phase_segments),
        phase_codebook_signs=codebook,
        phase_function_digest=phase_function_digest,
        schedule_digest=schedule_digest,
        weighted_mean=weighted_mean,
        weighted_residual=residual,
        active_phase_count=len(active_indices),
        minimum_active_code_magnitude=minimum_magnitude,
        weighted_code_energy=energy,
    )


def predictive_trajectory_weighted_code_correlation(
    left: PredictiveTrajectorySchedule,
    right: PredictiveTrajectorySchedule,
) -> float:
    """计算同一真实 schedule 上两候选时间码的加权相关系数。"""

    if left.active_weights != right.active_weights:
        raise ValueError("predictive trajectory correlation 要求同一权重网格")
    weights = left.active_weights
    numerator = sum(
        weight * left_code * right_code
        for weight, left_code, right_code in zip(
            weights,
            left.codes,
            right.codes,
            strict=True,
        )
    )
    left_energy = sum(
        weight * code * code
        for weight, code in zip(weights, left.codes, strict=True)
    )
    right_energy = sum(
        weight * code * code
        for weight, code in zip(weights, right.codes, strict=True)
    )
    denominator = math.sqrt(left_energy * right_energy)
    if denominator <= 1e-12:
        raise RuntimeError("predictive trajectory correlation 能量退化")
    return numerator / denominator


def apply_predictive_trajectory_constraint(
    model_output: Any,
    sample: Any,
    signed_phase_direction: Any,
    *,
    ac_code: float,
    flow_phase: float,
    config: VelocityFieldConstraintConfig,
    tubelet_config: Any,
    carrier_config: PredictiveTrajectoryCarrierConfig,
    control_context: VelocityControlContext,
) -> tuple[Any, dict[str, Any]]:
    """仅用时变 AC 载体修改 velocity，并维持原 norm/Flow-energy 预算。"""

    code = float(ac_code)
    if not math.isfinite(code):
        raise ValueError("predictive trajectory AC code 必须有限")
    schedule_weight = flow_phase_weight(flow_phase, tubelet_config)
    delta_sigma = float(control_context.delta_sigma)
    base = model_output.detach().float()
    base_norm = float(base.norm().item())
    reference_increment = delta_sigma**2 * float(base.square().sum().item())
    projected_reference = (
        max(0.0, float(control_context.cumulative_reference_energy))
        + reference_increment
        * max(1, int(control_context.remaining_step_count))
    )
    total_energy_budget = (
        float(config.flow_energy_budget_ratio) * projected_reference
    )
    remaining_energy = max(
        0.0,
        total_energy_budget
        - max(0.0, float(control_context.cumulative_control_energy)),
    )
    if abs(code) <= 1e-12:
        if schedule_weight > 1e-12:
            raise ValueError("predictive trajectory active phase code 不得为零")
        return model_output, {
            "velocity_field_constraint_status": "inactive_flow_phase",
            "velocity_field_source": (
                "scheduler_model_output_before_flow_match_step"
            ),
            "flow_phase": round(float(flow_phase), 8),
            "flow_phase_weight": 0.0,
            "velocity_constraint_lambda": 0.0,
            "velocity_norm_before_constraint": round(base_norm, 6),
            "velocity_norm_after_constraint": round(base_norm, 6),
            "velocity_constraint_delta_norm": 0.0,
            "velocity_constraint_delta_ratio": 0.0,
            "endpoint_control_enabled": False,
            "endpoint_control_formal_context_complete": False,
            "predictive_trajectory_inactive_phase_noop": True,
            "predictive_trajectory_norm_guard_passed": None,
            "predictive_trajectory_energy_guard_passed": None,
            "predictive_trajectory_control_energy_increment": 0.0,
            "predictive_trajectory_control_cumulative_energy_after": float(
                control_context.cumulative_control_energy
            ),
            "predictive_trajectory_reference_energy_increment": (
                reference_increment
            ),
            "predictive_trajectory_reference_cumulative_energy_after": (
                float(control_context.cumulative_reference_energy)
                + reference_increment
            ),
            "predictive_trajectory_observability_mode": (
                "bounded_terminal_residual_from_phase_conditioned_carrier"
            ),
            "endpoint_quality_energy_guard_passed": True,
        }

    candidate, record = apply_velocity_field_constraint(
        model_output,
        sample,
        signed_phase_direction,
        flow_phase=flow_phase,
        config=replace(
            config,
            lambda_max=float(config.lambda_max) * abs(code),
        ),
        tubelet_config=tubelet_config,
        endpoint_control_enabled=False,
        control_context=control_context,
    )
    delta = candidate.detach().float() - base
    observed_delta_norm = float(delta.norm().item())
    joint_norm_budget = (
        base_norm
        * float(config.velocity_norm_ratio_budget)
        * float(config.lambda_max)
        * schedule_weight
    )
    energy_limited_norm = math.sqrt(remaining_energy) / max(
        abs(delta_sigma),
        1e-12,
    )
    admissible_norm = min(joint_norm_budget, energy_limited_norm)
    scale = min(
        1.0,
        admissible_norm / max(observed_delta_norm, 1e-12),
    )
    delta = delta * scale
    constrained = model_output + delta.to(dtype=model_output.dtype)
    actual_delta_norm = float(delta.norm().item())
    energy_increment = delta_sigma**2 * actual_delta_norm**2
    norm_guard = _predictive_budget_guard_passed(
        actual_delta_norm,
        joint_norm_budget,
    )
    energy_guard = _predictive_budget_guard_passed(
        energy_increment,
        remaining_energy,
    )
    if not norm_guard or not energy_guard:
        raise RuntimeError(
            "predictive trajectory 最终预算 guard 失败: "
            f"delta_norm={actual_delta_norm:.12g}, "
            f"norm_budget={joint_norm_budget:.12g}, "
            f"energy_increment={energy_increment:.12g}, "
            f"remaining_energy={remaining_energy:.12g}, "
            f"relative_tolerance="
            f"{PREDICTIVE_TRAJECTORY_BUDGET_GUARD_RELATIVE_TOLERANCE:.12g}"
        )
    return constrained, {
        **record,
        "velocity_constraint_lambda": round(
            float(config.lambda_max) * schedule_weight,
            8,
        ),
        "velocity_norm_after_constraint": round(
            float(constrained.detach().float().norm().item()),
            6,
        ),
        "velocity_constraint_delta_norm": round(actual_delta_norm, 6),
        "velocity_constraint_delta_ratio": round(
            actual_delta_norm / max(base_norm, 1e-8),
            8,
        ),
        "endpoint_control_enabled": False,
        "endpoint_control_formal_context_complete": False,
        "predictive_trajectory_inactive_phase_noop": False,
        "predictive_trajectory_ac_code": round(code, 10),
        "predictive_trajectory_joint_scale": round(scale, 10),
        "predictive_trajectory_joint_norm_budget": joint_norm_budget,
        "predictive_trajectory_energy_limited_delta_norm": (
            energy_limited_norm
        ),
        "predictive_trajectory_norm_guard_passed": norm_guard,
        "predictive_trajectory_energy_guard_passed": energy_guard,
        "predictive_trajectory_control_energy_increment": energy_increment,
        "predictive_trajectory_control_cumulative_energy_after": (
            float(control_context.cumulative_control_energy)
            + energy_increment
        ),
        "predictive_trajectory_reference_energy_increment": (
            reference_increment
        ),
        "predictive_trajectory_reference_cumulative_energy_after": (
            float(control_context.cumulative_reference_energy)
            + reference_increment
        ),
        "predictive_trajectory_observability_mode": (
            "bounded_terminal_residual_from_phase_conditioned_carrier"
        ),
        "endpoint_quality_energy_guard_passed": energy_guard,
        "predictive_trajectory_carrier_config_complete": bool(
            carrier_config.carrier_id == PREDICTIVE_TRAJECTORY_CARRIER_ID
        ),
    }
