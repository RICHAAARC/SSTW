"""Continuous balanced time code for prompt-orthogonal state trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Sequence


PROMPT_ORTHOGONAL_TRAJECTORY_CODE_ID = (
    "first_harmonic_two_channel_weighted_centered"
)
PROMPT_ORTHOGONAL_TRAJECTORY_CODE_DIMENSION = 2


@dataclass(frozen=True)
class ContinuousTrajectorySchedule:
    """A grid projection of one schedule-independent continuous codeword."""

    codes: tuple[float, ...]
    raw_values: tuple[float, ...]
    basis_values: tuple[tuple[float, float], ...]
    centered_basis_values: tuple[tuple[float, float], ...]
    active_weights: tuple[float, ...]
    codeword: tuple[float, float]
    continuous_function_digest: str
    schedule_projection_digest: str
    weighted_mean: float
    weighted_residual: float
    weighted_code_energy: float
    minimum_active_code_magnitude: float
    active_phase_count: int


def derive_continuous_trajectory_codeword(
    trajectory_code_subkey: str,
) -> tuple[float, float]:
    """Map a secret subkey to one unit vector on the frozen 2-D basis."""

    if not str(trajectory_code_subkey).strip():
        raise ValueError("trajectory code subkey 不能为空")
    digest = sha256(
        (
            f"{PROMPT_ORTHOGONAL_TRAJECTORY_CODE_ID}::"
            f"{trajectory_code_subkey}"
        ).encode("utf-8")
    ).digest()
    numerator = int.from_bytes(digest[:8], "big")
    phase = 2.0 * math.pi * numerator / float(2**64)
    return math.cos(phase), math.sin(phase)


def continuous_trajectory_basis(
    flow_phase: float,
) -> tuple[float, float]:
    """Evaluate the frozen first-harmonic sine/cosine basis."""

    phase = float(flow_phase)
    if not math.isfinite(phase) or not 0.0 <= phase <= 1.0:
        raise ValueError("continuous trajectory phase 必须位于[0,1]")
    return math.sin(2.0 * math.pi * phase), math.cos(
        2.0 * math.pi * phase
    )


def build_continuous_trajectory_schedule(
    *,
    trajectory_code_subkey: str,
    flow_phases: Sequence[float],
    active_weights: Sequence[float],
) -> ContinuousTrajectorySchedule:
    """Project the same continuous function onto a real scheduler grid."""

    phases = tuple(float(value) for value in flow_phases)
    weights = tuple(float(value) for value in active_weights)
    if not phases or len(phases) != len(weights):
        raise ValueError("flow phases 与 active weights 必须非空且等长")
    if any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0
        for value in phases
    ):
        raise ValueError("flow phases 必须有限且位于[0,1]")
    if any(not math.isfinite(value) or value < 0.0 for value in weights):
        raise ValueError("active weights 必须有限且非负")
    positive_weight = sum(weights)
    active_count = sum(value > 0.0 for value in weights)
    if positive_weight <= 0.0 or active_count < 2:
        raise ValueError("continuous trajectory 至少需要两个 active phase")
    codeword = derive_continuous_trajectory_codeword(
        trajectory_code_subkey
    )
    basis = tuple(continuous_trajectory_basis(phase) for phase in phases)
    basis_means = tuple(
        sum(
            weight * values[channel]
            for weight, values in zip(weights, basis, strict=True)
        )
        / positive_weight
        for channel in range(PROMPT_ORTHOGONAL_TRAJECTORY_CODE_DIMENSION)
    )
    centered_basis = tuple(
        (
            values[0] - basis_means[0],
            values[1] - basis_means[1],
        )
        if weight > 0.0
        else (0.0, 0.0)
        for values, weight in zip(basis, weights, strict=True)
    )
    raw = tuple(
        codeword[0] * values[0] + codeword[1] * values[1]
        for values in basis
    )
    weighted_mean = sum(
        weight * value for weight, value in zip(weights, raw, strict=True)
    ) / positive_weight
    centered = tuple(
        value - weighted_mean if weight > 0.0 else 0.0
        for value, weight in zip(raw, weights, strict=True)
    )
    maximum = max(abs(value) for value in centered)
    if maximum <= 1e-12:
        raise RuntimeError("continuous trajectory code 加权中心化后塌缩")
    codes = tuple(value / maximum for value in centered)
    weighted_residual = sum(
        weight * value for weight, value in zip(weights, codes, strict=True)
    )
    weighted_energy = sum(
        weight * value * value
        for weight, value in zip(weights, codes, strict=True)
    ) / positive_weight
    minimum_magnitude = min(
        abs(value)
        for value, weight in zip(codes, weights, strict=True)
        if weight > 0.0
    )
    function_payload = {
        "basis_family": "first_harmonic_sine_cosine",
        "code_dimension": PROMPT_ORTHOGONAL_TRAJECTORY_CODE_DIMENSION,
        "code_id": PROMPT_ORTHOGONAL_TRAJECTORY_CODE_ID,
        "codeword": [round(value, 15) for value in codeword],
    }
    function_digest = sha256(
        json.dumps(
            function_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    schedule_payload = {
        "active_weights": [round(value, 15) for value in weights],
        "centered_basis_values": [
            [round(value, 15) for value in values]
            for values in centered_basis
        ],
        "codes": [round(value, 15) for value in codes],
        "continuous_function_digest": function_digest,
        "flow_phases": [round(value, 15) for value in phases],
    }
    schedule_digest = sha256(
        json.dumps(
            schedule_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return ContinuousTrajectorySchedule(
        codes=codes,
        raw_values=raw,
        basis_values=basis,
        centered_basis_values=centered_basis,
        active_weights=weights,
        codeword=codeword,
        continuous_function_digest=function_digest,
        schedule_projection_digest=schedule_digest,
        weighted_mean=weighted_mean,
        weighted_residual=weighted_residual,
        weighted_code_energy=weighted_energy,
        minimum_active_code_magnitude=minimum_magnitude,
        active_phase_count=active_count,
    )


def weighted_continuous_code_correlation(
    left: ContinuousTrajectorySchedule,
    right: ContinuousTrajectorySchedule,
) -> float:
    """Compare two code projections on an identical real schedule."""

    if left.active_weights != right.active_weights:
        raise ValueError("trajectory correlation 要求相同 active weights")
    numerator = sum(
        weight * left_code * right_code
        for weight, left_code, right_code in zip(
            left.active_weights,
            left.codes,
            right.codes,
            strict=True,
        )
    )
    left_energy = sum(
        weight * value * value
        for weight, value in zip(
            left.active_weights,
            left.codes,
            strict=True,
        )
    )
    right_energy = sum(
        weight * value * value
        for weight, value in zip(
            right.active_weights,
            right.codes,
            strict=True,
        )
    )
    denominator = math.sqrt(left_energy * right_energy)
    if denominator <= 1e-12:
        raise RuntimeError("trajectory correlation 能量退化")
    return numerator / denominator
