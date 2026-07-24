"""Synchronous vector demodulation for replay innovation responses."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


PROMPT_ORTHOGONAL_SMOKE_WHITENING_MODE = (
    "identity_whitening_not_fitted_smoke_only"
)
PROMPT_ORTHOGONAL_DEMODULATION_SOLVER_ID = (
    "candidate_independent_weighted_two_channel_least_squares"
)


@dataclass(frozen=True)
class TrajectoryVectorDemodulation:
    """Preserve the two-channel response before candidate scoring."""

    demodulation_vector: tuple[float, float]
    candidate_codeword: tuple[float, float]
    matched_amplitude: float
    orthogonal_amplitude: float
    matched_cosine_score: float
    effective_weight_sum: float
    demodulation_solver_id: str
    whitening_mode: str
    vector_context_complete: bool


def demodulate_trajectory_responses(
    *,
    step_responses: Sequence[float],
    basis_values: Sequence[tuple[float, float]],
    scheduler_weights: Sequence[float],
    reliability_weights: Sequence[float],
    candidate_codeword: tuple[float, float],
    whitening_mode: str = PROMPT_ORTHOGONAL_SMOKE_WHITENING_MODE,
) -> TrajectoryVectorDemodulation:
    """Aggregate both temporal basis channels with candidate-independent weights."""

    responses = tuple(float(value) for value in step_responses)
    basis = tuple(
        (float(values[0]), float(values[1])) for values in basis_values
    )
    scheduler = tuple(float(value) for value in scheduler_weights)
    reliability = tuple(float(value) for value in reliability_weights)
    if not responses or not (
        len(responses)
        == len(basis)
        == len(scheduler)
        == len(reliability)
    ):
        raise ValueError("trajectory demodulation 输入必须非空且等长")
    if whitening_mode != PROMPT_ORTHOGONAL_SMOKE_WHITENING_MODE:
        raise ValueError("首轮 smoke whitening mode 未冻结")
    all_values = (
        responses
        + tuple(value for pair in basis for value in pair)
        + scheduler
        + reliability
        + tuple(float(value) for value in candidate_codeword)
    )
    if any(not math.isfinite(value) for value in all_values):
        raise ValueError("trajectory demodulation 输入必须有限")
    if any(value < 0.0 for value in scheduler + reliability):
        raise ValueError("scheduler/reliability weights 必须非负")
    code_norm = math.hypot(*candidate_codeword)
    if not math.isclose(code_norm, 1.0, rel_tol=0.0, abs_tol=1e-10):
        raise ValueError("candidate codeword 必须为二维单位向量")
    effective_weights = tuple(
        left * right
        for left, right in zip(scheduler, reliability, strict=True)
    )
    total_weight = sum(effective_weights)
    if total_weight <= 1e-12:
        raise ValueError("trajectory demodulation 有效权重退化")
    raw_vector = tuple(
        sum(
            weight * values[channel] * response
            for weight, values, response in zip(
                effective_weights,
                basis,
                responses,
                strict=True,
            )
        )
        / total_weight
        for channel in range(2)
    )
    gram_00 = sum(
        weight * values[0] * values[0]
        for weight, values in zip(effective_weights, basis, strict=True)
    ) / total_weight
    gram_01 = sum(
        weight * values[0] * values[1]
        for weight, values in zip(effective_weights, basis, strict=True)
    ) / total_weight
    gram_11 = sum(
        weight * values[1] * values[1]
        for weight, values in zip(effective_weights, basis, strict=True)
    ) / total_weight
    determinant = gram_00 * gram_11 - gram_01 * gram_01
    if determinant <= 1e-12:
        raise ValueError("trajectory demodulation basis Gram matrix 退化")
    vector = (
        (gram_11 * raw_vector[0] - gram_01 * raw_vector[1])
        / determinant,
        (-gram_01 * raw_vector[0] + gram_00 * raw_vector[1])
        / determinant,
    )
    vector_norm = math.hypot(*vector)
    matched = (
        vector[0] * float(candidate_codeword[0])
        + vector[1] * float(candidate_codeword[1])
    )
    orthogonal = (
        -vector[0] * float(candidate_codeword[1])
        + vector[1] * float(candidate_codeword[0])
    )
    cosine = matched / max(vector_norm, 1e-12)
    return TrajectoryVectorDemodulation(
        demodulation_vector=(float(vector[0]), float(vector[1])),
        candidate_codeword=(
            float(candidate_codeword[0]),
            float(candidate_codeword[1]),
        ),
        matched_amplitude=float(matched),
        orthogonal_amplitude=float(orthogonal),
        matched_cosine_score=float(cosine),
        effective_weight_sum=float(total_weight),
        demodulation_solver_id=PROMPT_ORTHOGONAL_DEMODULATION_SOLVER_ID,
        whitening_mode=whitening_mode,
        vector_context_complete=True,
    )
