"""构造 schedule-bound 的带符号零均值轨迹载体。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
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


SIGNED_BALANCED_AC_CARRIER_ID = (
    "key_context_schedule_bound_centered_binary_ac_v1"
)
_PHASE_BIN_BASE_PATTERN = (1, 1, -1, 1, -1, -1, 1, -1)


@dataclass(frozen=True)
class SignedTrajectoryCarrierConfig:
    """冻结 AC 轨迹通道与小 DC endpoint 通道的预算分配。"""

    carrier_id: str = SIGNED_BALANCED_AC_CARRIER_ID
    ac_allocation: float = 0.75
    dc_allocation: float = 0.25
    phase_bin_count: int = 8
    minimum_ac_direction_retained_cosine: float = 0.25

    def __post_init__(self) -> None:
        if self.carrier_id != SIGNED_BALANCED_AC_CARRIER_ID:
            raise ValueError("不支持的 signed trajectory carrier")
        if not 0.0 < float(self.ac_allocation) < 1.0:
            raise ValueError("signed trajectory AC allocation 必须位于 (0,1)")
        if not 0.0 < float(self.dc_allocation) < 1.0:
            raise ValueError("signed trajectory DC allocation 必须位于 (0,1)")
        if not math.isclose(
            float(self.ac_allocation) + float(self.dc_allocation),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("signed trajectory AC/DC allocation 总和必须为1")
        if int(self.phase_bin_count) != 8:
            raise ValueError("signed trajectory phase bin count 必须冻结为8")
        if not math.isclose(
            float(self.minimum_ac_direction_retained_cosine),
            0.25,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "signed trajectory 最小 AC direction retained cosine 必须冻结为0.25"
            )


@dataclass(frozen=True)
class SignedTrajectorySchedule:
    """保存可由 key、context 与真实 schedule 重建的严格零均值 AC code。"""

    codes: tuple[float, ...]
    raw_signs: tuple[int, ...]
    active_weights: tuple[float, ...]
    phase_bins: tuple[int, ...]
    phase_offset: int
    phase_pattern_reversed: bool
    phase_pattern_polarity: int
    phase_function_digest: str
    schedule_digest: str
    weighted_mean: float
    weighted_residual: float

    def metadata_for_step(self, step_index: int) -> dict[str, Any]:
        index = int(step_index)
        return {
            "signed_trajectory_carrier_id": SIGNED_BALANCED_AC_CARRIER_ID,
            "signed_trajectory_schedule_digest": self.schedule_digest,
            "signed_trajectory_ac_code": round(float(self.codes[index]), 10),
            "signed_trajectory_ac_raw_sign": int(self.raw_signs[index]),
            "signed_trajectory_ac_weight": round(
                float(self.active_weights[index]),
                10,
            ),
            "signed_trajectory_phase_bin": int(self.phase_bins[index]),
            "signed_trajectory_phase_offset": int(self.phase_offset),
            "signed_trajectory_phase_pattern_reversed": bool(
                self.phase_pattern_reversed
            ),
            "signed_trajectory_phase_pattern_polarity": int(
                self.phase_pattern_polarity
            ),
            "signed_trajectory_phase_function_digest": (
                self.phase_function_digest
            ),
            "signed_trajectory_ac_weighted_mean": round(
                float(self.weighted_mean),
                12,
            ),
            "signed_trajectory_ac_weighted_residual": round(
                float(self.weighted_residual),
                12,
            ),
            "signed_trajectory_ac_zero_mean_verified": (
                abs(float(self.weighted_residual)) <= 1e-10
            ),
        }


def build_signed_trajectory_schedule(
    *,
    key_text: str,
    key_context_digest: str,
    flow_phases: Sequence[float],
    active_weights: Sequence[float],
    phase_bin_count: int = 8,
) -> SignedTrajectorySchedule:
    """把密钥二值原码中心化为真实 schedule 权重下严格零均值的 AC code。

    二值原码由连续规范 Flow phase 的8个固定 bin 与 key/context phase offset
    决定，不依赖离散 schedule 的 step index。因此 8-step generation 与
    20-step replay 会在同一 phase bin 重建同一符号。每个真实 schedule 再减去
    自身加权均值并统一缩放，不改变正负符号，同时使
    ``sum(weight_t * code_t) == 0``。这避免 AC 轨迹码退化为 endpoint 累积
    能量；独立 DC 通道由运行时另行分配。
    """

    phases = tuple(float(value) for value in flow_phases)
    weights = tuple(max(0.0, float(value)) for value in active_weights)
    if not key_text:
        raise ValueError("signed trajectory schedule 缺少 key")
    if len(key_context_digest) != 64:
        raise ValueError("signed trajectory schedule 缺少64位 context digest")
    if not phases or len(phases) != len(weights):
        raise ValueError("signed trajectory schedule 需要等长非空 phase/weight")
    if any(not 0.0 <= phase <= 1.0 for phase in phases):
        raise ValueError("signed trajectory phase 必须位于 [0,1]")
    if int(phase_bin_count) != 8:
        raise ValueError("signed trajectory phase bin count 必须冻结为8")
    active_indices = [index for index, weight in enumerate(weights) if weight > 0.0]
    if len(active_indices) < 2:
        raise ValueError("signed trajectory schedule 至少需要两个 active phase")

    schedule_payload = {
        "carrier_id": SIGNED_BALANCED_AC_CARRIER_ID,
        "key_context_digest": key_context_digest,
        "flow_phases": [format(value, ".17g") for value in phases],
        "active_weights": [format(value, ".17g") for value in weights],
    }
    public_schedule_digest = sha256(
        json.dumps(
            schedule_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    phase_key_digest = sha256(
        (
            f"{SIGNED_BALANCED_AC_CARRIER_ID}::phase_function::"
            f"{key_text}::{key_context_digest}"
        ).encode("utf-8")
    ).digest()
    phase_offset = int.from_bytes(phase_key_digest[:8], "big") % int(
        phase_bin_count
    )
    phase_pattern_reversed = bool(phase_key_digest[8] & 1)
    phase_pattern_polarity = 1 if phase_key_digest[9] & 1 else -1
    phase_bins = [
        min(
            int(phase_bin_count) - 1,
            int(math.floor(phase * int(phase_bin_count))),
        )
        for phase in phases
    ]
    raw_signs = [0 for _ in phases]
    for index in active_indices:
        pattern_index = (
            phase_offset - phase_bins[index]
            if phase_pattern_reversed
            else phase_bins[index] + phase_offset
        ) % int(phase_bin_count)
        raw_signs[index] = (
            phase_pattern_polarity
            * int(_PHASE_BIN_BASE_PATTERN[pattern_index])
        )
    if len({raw_signs[index] for index in active_indices}) != 2:
        raise ValueError(
            "signed trajectory active schedule 未跨越正负 phase bin"
        )
    phase_function_payload = {
        "carrier_id": SIGNED_BALANCED_AC_CARRIER_ID,
        "key_context_digest": key_context_digest,
        "key_binding_digest": sha256(key_text.encode("utf-8")).hexdigest(),
        "phase_bin_count": int(phase_bin_count),
        "phase_offset": phase_offset,
        "phase_pattern_reversed": phase_pattern_reversed,
        "phase_pattern_polarity": phase_pattern_polarity,
    }
    phase_function_digest = sha256(
        json.dumps(
            phase_function_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    total_weight = sum(weights)
    weighted_mean = sum(
        weight * sign for weight, sign in zip(weights, raw_signs, strict=True)
    ) / total_weight
    centered = [
        (float(sign) - weighted_mean) if weight > 0.0 else 0.0
        for sign, weight in zip(raw_signs, weights, strict=True)
    ]
    maximum_magnitude = max(abs(value) for value in centered)
    if maximum_magnitude <= 1e-12:
        raise RuntimeError("signed trajectory centered code 退化")
    codes = tuple(value / maximum_magnitude for value in centered)
    residual = sum(
        weight * code for weight, code in zip(weights, codes, strict=True)
    )
    if abs(residual) > 1e-10:
        raise RuntimeError("signed trajectory AC code 未达到加权零均值")
    if not any(value > 0.0 for value in codes) or not any(
        value < 0.0 for value in codes
    ):
        raise RuntimeError("signed trajectory AC code 必须同时包含正负 phase")

    binding_payload = {
        **schedule_payload,
        "key_binding_digest": sha256(key_text.encode("utf-8")).hexdigest(),
        "raw_signs": raw_signs,
        "phase_bins": phase_bins,
        "phase_offset": phase_offset,
        "phase_pattern_reversed": phase_pattern_reversed,
        "phase_pattern_polarity": phase_pattern_polarity,
        "phase_function_digest": phase_function_digest,
        "codes": [format(value, ".17g") for value in codes],
    }
    binding_digest = sha256(
        json.dumps(
            binding_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return SignedTrajectorySchedule(
        codes=codes,
        raw_signs=tuple(raw_signs),
        active_weights=weights,
        phase_bins=tuple(phase_bins),
        phase_offset=phase_offset,
        phase_pattern_reversed=phase_pattern_reversed,
        phase_pattern_polarity=phase_pattern_polarity,
        phase_function_digest=phase_function_digest,
        schedule_digest=binding_digest,
        weighted_mean=weighted_mean,
        weighted_residual=residual,
    )


def select_signed_trajectory_joint_scale(
    *,
    observed_delta_norm: float,
    joint_norm_budget: float,
    energy_limited_delta_norm: float,
) -> dict[str, float]:
    """选择同时遵守原始 velocity norm 与 Flow energy 预算的统一比例。"""

    observed = float(observed_delta_norm)
    norm_budget = float(joint_norm_budget)
    energy_budget = float(energy_limited_delta_norm)
    if (
        not math.isfinite(observed)
        or not math.isfinite(norm_budget)
        or not math.isfinite(energy_budget)
        or observed < 0.0
        or norm_budget < 0.0
        or energy_budget < 0.0
    ):
        raise ValueError("signed trajectory joint budget 必须是有限非负数")
    denominator = max(observed, 1e-12)
    norm_scale = min(1.0, norm_budget / denominator)
    energy_scale = min(1.0, energy_budget / denominator)
    return {
        "norm_scale": norm_scale,
        "energy_scale": energy_scale,
        "joint_scale": min(norm_scale, energy_scale),
    }


def select_dc_scale_for_ac_direction_retention(
    *,
    ac_delta_norm: float,
    dc_delta_norm: float,
    ac_dc_dot: float,
    minimum_retained_cosine: float,
) -> dict[str, float]:
    """限制同 basis DC，使 joint delta 不会覆盖 signed AC 方向。

    返回可保留的最大 DC 比例。该 guard 在 joint norm/energy guard 前执行；
    后者只会用正比例统一缩放，因此不会改变这里保证的方向余弦。
    """

    ac_norm = float(ac_delta_norm)
    dc_norm = float(dc_delta_norm)
    dot = float(ac_dc_dot)
    minimum = float(minimum_retained_cosine)
    if (
        not math.isfinite(ac_norm)
        or not math.isfinite(dc_norm)
        or not math.isfinite(dot)
        or ac_norm <= 1e-12
        or dc_norm < 0.0
        or not 0.0 < minimum < 1.0
    ):
        raise ValueError("signed trajectory AC/DC direction guard 输入无效")

    ac_norm_squared = ac_norm * ac_norm

    def cosine_for_scale(scale: float) -> float:
        joint_norm_squared = (
            ac_norm_squared
            + 2.0 * scale * dot
            + scale * scale * dc_norm * dc_norm
        )
        if joint_norm_squared <= 1e-24:
            return -1.0
        projection = ac_norm_squared + scale * dot
        return projection / (ac_norm * math.sqrt(joint_norm_squared))

    numerical_target = min(1.0, minimum + 1e-8)
    candidate_cosine = cosine_for_scale(1.0)
    if candidate_cosine >= numerical_target:
        selected_scale = 1.0
    else:
        lower = 0.0
        upper = 1.0
        for _ in range(64):
            midpoint = 0.5 * (lower + upper)
            if cosine_for_scale(midpoint) >= numerical_target:
                lower = midpoint
            else:
                upper = midpoint
        selected_scale = lower
    selected_cosine = cosine_for_scale(selected_scale)
    if selected_cosine + 1e-10 < minimum:
        raise RuntimeError("signed trajectory DC cap 未保留 AC direction")
    return {
        "dc_scale": selected_scale,
        "candidate_joint_ac_cosine": candidate_cosine,
        "selected_joint_ac_cosine": selected_cosine,
    }


def apply_signed_trajectory_two_channel_constraint(
    model_output: Any,
    sample: Any,
    spatial_key_direction: Any,
    *,
    ac_code: float,
    flow_phase: float,
    config: VelocityFieldConstraintConfig,
    tubelet_config: Any,
    carrier_config: SignedTrajectoryCarrierConfig,
    control_context: VelocityControlContext,
) -> tuple[Any, dict[str, Any]]:
    """在原总 lambda/能量预算内组合 AC trajectory 与 DC endpoint 控制。

    AC 和 DC 两次调用只用于复用现有投影、语义切向与 endpoint controller。
    最终合成 delta 会再次按同一 Flow 能量预算裁剪，因此两通道不会把总预算
    扩大到默认 ``lambda_max`` 之外。
    """

    def cosine_alignment(left: Any, right: Any) -> float:
        left_flat = left.detach().float().reshape(-1)
        right_flat = right.detach().float().reshape(-1)
        denominator = (
            left_flat.norm().clamp_min(1e-8)
            * right_flat.norm().clamp_min(1e-8)
        )
        return float((left_flat @ right_flat / denominator).item())

    code = float(ac_code)
    if not math.isfinite(code):
        raise ValueError("signed trajectory AC code 必须有限")
    schedule_weight = flow_phase_weight(flow_phase, tubelet_config)
    if abs(code) <= 1e-12:
        if schedule_weight > 1e-12:
            raise ValueError("signed trajectory active phase 的 AC code 不得为零")
        delta_sigma = float(control_context.delta_sigma)
        if not math.isfinite(delta_sigma) or abs(delta_sigma) <= 1e-12:
            raise ValueError("signed trajectory inactive no-op 缺少有效 delta_sigma")
        base = model_output.detach().float()
        base_norm = float(base.norm().item())
        reference_increment = (
            delta_sigma**2 * float(base.square().sum().item())
        )
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
        velocity_alignment = cosine_alignment(
            model_output,
            spatial_key_direction,
        )
        endpoint_response = cosine_alignment(
            sample,
            spatial_key_direction,
        )
        baseline_next = (
            sample.detach().float() + delta_sigma * base
        )
        endpoint_response_after_noop = cosine_alignment(
            baseline_next,
            spatial_key_direction,
        )
        inactive_context_complete = bool(
            int(control_context.remaining_step_count) > 0
            and float(control_context.cumulative_control_energy) >= 0.0
            and float(control_context.cumulative_reference_energy) >= 0.0
        )
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
            "velocity_alignment_before_constraint": round(
                velocity_alignment,
                8,
            ),
            "velocity_alignment_after_constraint": round(
                velocity_alignment,
                8,
            ),
            "velocity_alignment_gain": 0.0,
            "velocity_alignment_reference": (
                "dc_spatial_key_direction_for_joint_ac_dc_record"
            ),
            "endpoint_control_enabled": True,
            "endpoint_control_policy": (
                "inactive_signed_trajectory_phase_noop"
            ),
            "endpoint_control_formal_context_complete": False,
            "endpoint_response_before_step": round(endpoint_response, 8),
            "endpoint_response_before_constraint": round(
                endpoint_response,
                8,
            ),
            "endpoint_response_without_control": (
                endpoint_response_after_noop
            ),
            "endpoint_response_predicted_after_step": (
                endpoint_response_after_noop
            ),
            "endpoint_margin_deficit_before_control": max(
                0.0,
                float(config.endpoint_target_margin)
                - endpoint_response_after_noop,
            ),
            "scheduler_velocity_sign": float(
                config.scheduler_velocity_sign
            ),
            "velocity_norm_ratio_budget": float(
                config.velocity_norm_ratio_budget
            ),
            "flow_energy_budget_ratio": float(
                config.flow_energy_budget_ratio
            ),
            "signed_trajectory_ac_allocation": float(
                carrier_config.ac_allocation
            ),
            "signed_trajectory_dc_allocation": float(
                carrier_config.dc_allocation
            ),
            "signed_trajectory_ac_code": 0.0,
            "signed_trajectory_minimum_ac_direction_retained_cosine": float(
                carrier_config.minimum_ac_direction_retained_cosine
            ),
            "signed_trajectory_inactive_phase_noop": True,
            "signed_trajectory_inactive_phase_noop_context_complete": (
                inactive_context_complete
            ),
            "signed_trajectory_ac_direction_guard_applicable": False,
            "signed_trajectory_ac_direction_guard_passed": None,
            "signed_trajectory_joint_norm_guard_passed": None,
            "signed_trajectory_joint_energy_guard_passed": None,
            "signed_trajectory_joint_norm_budget": 0.0,
            "endpoint_control_energy_increment": 0.0,
            "endpoint_control_cumulative_energy_after": float(
                control_context.cumulative_control_energy
            ),
            "endpoint_reference_energy_increment": reference_increment,
            "endpoint_reference_cumulative_energy_after": (
                float(control_context.cumulative_reference_energy)
                + reference_increment
            ),
            "endpoint_projected_total_energy_budget": total_energy_budget,
            "endpoint_remaining_energy_budget_before_step": (
                remaining_energy
            ),
            "endpoint_quality_energy_guard_passed": True,
            "endpoint_minimum_energy_control_status": (
                "inactive_signed_trajectory_phase_noop"
            ),
        }
    ac_direction = spatial_key_direction * (1.0 if code >= 0.0 else -1.0)
    ac_config = replace(
        config,
        lambda_max=(
            float(config.lambda_max)
            * float(carrier_config.ac_allocation)
            * abs(code)
        ),
    )
    after_ac, ac_record = apply_velocity_field_constraint(
        model_output,
        sample,
        ac_direction,
        flow_phase=flow_phase,
        config=ac_config,
        tubelet_config=tubelet_config,
        endpoint_control_enabled=False,
        control_context=control_context,
    )
    dc_config = replace(
        config,
        lambda_max=float(config.lambda_max) * float(carrier_config.dc_allocation),
    )
    after_both, dc_record = apply_velocity_field_constraint(
        after_ac,
        sample,
        spatial_key_direction,
        flow_phase=flow_phase,
        config=dc_config,
        tubelet_config=tubelet_config,
        endpoint_control_enabled=True,
        control_context=control_context,
    )

    ac_delta = after_ac.detach().float() - model_output.detach().float()
    dc_delta_candidate = (
        after_both.detach().float() - after_ac.detach().float()
    )
    ac_delta_norm = float(ac_delta.norm().item())
    dc_delta_candidate_norm = float(dc_delta_candidate.norm().item())
    if ac_delta_norm <= 1e-12:
        raise RuntimeError("signed trajectory AC channel 未产生可辨识增量")
    ac_dc_dot = float(
        (
            ac_delta.reshape(-1)
            @ dc_delta_candidate.reshape(-1)
        ).item()
    )
    dc_direction_guard = select_dc_scale_for_ac_direction_retention(
        ac_delta_norm=ac_delta_norm,
        dc_delta_norm=dc_delta_candidate_norm,
        ac_dc_dot=ac_dc_dot,
        minimum_retained_cosine=(
            carrier_config.minimum_ac_direction_retained_cosine
        ),
    )
    dc_direction_scale = dc_direction_guard["dc_scale"]
    dc_delta = dc_delta_candidate * dc_direction_scale
    after_direction_guard = after_ac.detach().float() + dc_delta
    delta = after_direction_guard - model_output.detach().float()
    delta_sigma = abs(float(control_context.delta_sigma))
    base_norm = float(model_output.detach().float().norm().item())
    joint_norm_budget = (
        base_norm
        * float(config.velocity_norm_ratio_budget)
        * float(config.lambda_max)
        * schedule_weight
    )
    reference_increment = (
        float(control_context.delta_sigma) ** 2
        * float(model_output.detach().float().square().sum().item())
    )
    projected_reference = (
        max(0.0, float(control_context.cumulative_reference_energy))
        + reference_increment * max(1, int(control_context.remaining_step_count))
    )
    total_energy_budget = float(config.flow_energy_budget_ratio) * projected_reference
    remaining_energy = max(
        0.0,
        total_energy_budget
        - max(0.0, float(control_context.cumulative_control_energy)),
    )
    energy_limited_delta_norm = math.sqrt(remaining_energy) / max(
        delta_sigma,
        1e-12,
    )
    observed_delta_norm = float(delta.norm().item())
    joint_scale_selection = select_signed_trajectory_joint_scale(
        observed_delta_norm=observed_delta_norm,
        joint_norm_budget=joint_norm_budget,
        energy_limited_delta_norm=energy_limited_delta_norm,
    )
    norm_scale = joint_scale_selection["norm_scale"]
    energy_scale = joint_scale_selection["energy_scale"]
    joint_scale = joint_scale_selection["joint_scale"]
    delta = delta * joint_scale
    constrained = model_output + delta.to(dtype=model_output.dtype)
    actual_delta_norm = float(delta.norm().item())
    energy_increment = (
        float(control_context.delta_sigma) ** 2 * actual_delta_norm**2
    )

    baseline_next = (
        sample.detach().float()
        + float(control_context.delta_sigma)
        * model_output.detach().float()
    )
    constrained_next = (
        sample.detach().float()
        + float(control_context.delta_sigma)
        * constrained.detach().float()
    )
    response_without_control = cosine_alignment(
        baseline_next,
        spatial_key_direction,
    )
    response_after_control = cosine_alignment(
        constrained_next,
        spatial_key_direction,
    )
    velocity_alignment_before = cosine_alignment(
        model_output,
        spatial_key_direction,
    )
    velocity_alignment_after = cosine_alignment(
        constrained,
        spatial_key_direction,
    )
    final_joint_ac_direction_cosine = cosine_alignment(delta, ac_delta)
    ac_direction_guard_passed = bool(
        actual_delta_norm > 1e-12
        and final_joint_ac_direction_cosine + 1e-10
        >= carrier_config.minimum_ac_direction_retained_cosine
    )
    record = {
        "velocity_field_constraint_status": (
            "applied" if actual_delta_norm > 0.0 else "inactive_flow_phase"
        ),
        "velocity_field_source": (
            "scheduler_model_output_before_flow_match_step"
        ),
        "flow_phase": round(float(flow_phase), 8),
        "flow_phase_weight": round(schedule_weight, 8),
        "velocity_constraint_lambda": round(
            float(config.lambda_max) * schedule_weight,
            8,
        ),
        "velocity_norm_before_constraint": round(base_norm, 6),
        "velocity_norm_after_constraint": round(
            float(constrained.detach().float().norm().item()),
            6,
        ),
        "velocity_constraint_delta_norm": round(actual_delta_norm, 6),
        "velocity_constraint_delta_ratio": round(
            actual_delta_norm / max(base_norm, 1e-8),
            8,
        ),
        "velocity_alignment_before_constraint": round(
            velocity_alignment_before,
            8,
        ),
        "velocity_alignment_after_constraint": round(
            velocity_alignment_after,
            8,
        ),
        "velocity_alignment_gain": round(
            velocity_alignment_after - velocity_alignment_before,
            8,
        ),
        "velocity_alignment_reference": (
            "dc_spatial_key_direction_for_joint_ac_dc_record"
        ),
        "endpoint_control_enabled": True,
        "endpoint_control_policy": (
            "signed_ac_plus_independent_small_dc_joint_budget_control"
        ),
        "endpoint_control_formal_context_complete": bool(
            dc_record.get("endpoint_control_formal_context_complete") is True
            and actual_delta_norm <= joint_norm_budget + 1e-10
            and energy_increment <= remaining_energy + 1e-10
            and ac_direction_guard_passed
        ),
        "endpoint_response_before_step": round(
            cosine_alignment(sample, spatial_key_direction),
            8,
        ),
        "endpoint_response_before_constraint": round(
            cosine_alignment(sample, spatial_key_direction),
            8,
        ),
        "endpoint_response_without_control": response_without_control,
        "endpoint_response_predicted_after_step": response_after_control,
        "endpoint_margin_deficit_before_control": max(
            0.0,
            float(config.endpoint_target_margin) - response_without_control,
        ),
        "signed_trajectory_ac_semantic_projection_status": ac_record.get(
            "semantic_projection_status"
        ),
        "signed_trajectory_dc_semantic_projection_status_before_joint_guard": (
            dc_record.get("semantic_projection_status")
        ),
        "signed_trajectory_ac_semantic_projection_retained_key_energy_ratio": (
            ac_record.get("semantic_projection_retained_key_energy_ratio")
        ),
        "signed_trajectory_dc_semantic_projection_retained_key_energy_ratio_before_joint_guard": dc_record.get(
            "semantic_projection_retained_key_energy_ratio"
        ),
        "signed_trajectory_joint_delta_velocity_alignment": round(
            cosine_alignment(delta, model_output),
            8,
        ),
        "scheduler_velocity_sign": float(config.scheduler_velocity_sign),
        "velocity_norm_ratio_budget": float(
            config.velocity_norm_ratio_budget
        ),
        "flow_energy_budget_ratio": float(config.flow_energy_budget_ratio),
        "signed_trajectory_ac_allocation": float(carrier_config.ac_allocation),
        "signed_trajectory_dc_allocation": float(carrier_config.dc_allocation),
        "signed_trajectory_inactive_phase_noop": False,
        "signed_trajectory_inactive_phase_noop_context_complete": False,
        "signed_trajectory_ac_direction_guard_applicable": True,
        "signed_trajectory_minimum_ac_direction_retained_cosine": float(
            carrier_config.minimum_ac_direction_retained_cosine
        ),
        "signed_trajectory_ac_code": round(code, 10),
        "signed_trajectory_ac_delta_norm_before_joint_guard": float(
            (
                after_ac.detach().float()
                - model_output.detach().float()
            ).norm().item()
        ),
        "signed_trajectory_dc_delta_norm_before_joint_guard": float(
            dc_delta.norm().item()
        ),
        "signed_trajectory_dc_delta_norm_before_direction_guard": (
            dc_delta_candidate_norm
        ),
        "signed_trajectory_dc_direction_guard_scale": round(
            dc_direction_scale,
            10,
        ),
        "signed_trajectory_candidate_joint_ac_direction_cosine": round(
            dc_direction_guard["candidate_joint_ac_cosine"],
            10,
        ),
        "signed_trajectory_final_joint_ac_direction_cosine": round(
            final_joint_ac_direction_cosine,
            10,
        ),
        "signed_trajectory_ac_direction_guard_passed": (
            ac_direction_guard_passed
        ),
        "signed_trajectory_joint_norm_budget": joint_norm_budget,
        "signed_trajectory_joint_energy_limited_delta_norm": (
            energy_limited_delta_norm
        ),
        "signed_trajectory_joint_norm_scale": round(norm_scale, 10),
        "signed_trajectory_joint_energy_scale": round(energy_scale, 10),
        "signed_trajectory_joint_scale": round(joint_scale, 10),
        "endpoint_control_energy_increment": energy_increment,
        "endpoint_control_cumulative_energy_after": (
            float(control_context.cumulative_control_energy) + energy_increment
        ),
        "endpoint_reference_energy_increment": reference_increment,
        "endpoint_reference_cumulative_energy_after": (
            float(control_context.cumulative_reference_energy)
            + reference_increment
        ),
        "endpoint_projected_total_energy_budget": total_energy_budget,
        "endpoint_remaining_energy_budget_before_step": remaining_energy,
        "endpoint_quality_energy_guard_passed": (
            energy_increment <= remaining_energy + 1e-10
        ),
        "signed_trajectory_joint_energy_guard_passed": (
            energy_increment <= remaining_energy + 1e-10
        ),
        "signed_trajectory_joint_norm_guard_passed": (
            actual_delta_norm <= joint_norm_budget + 1e-10
        ),
        "endpoint_minimum_energy_control_status": (
            "signed_ac_plus_independent_small_dc_joint_budget_control"
        ),
        "signed_trajectory_joint_norm_budget_utilization": (
            actual_delta_norm / max(joint_norm_budget, 1e-12)
        ),
    }
    return constrained, record
