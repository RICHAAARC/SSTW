"""CPU-only primitives for the first Patch-relation Gate 0 contract.

The module reconstructs one public, candidate-key-independent Patch pair,
applies one signed phase perturbation to the temporal component of Wan's 3D
RoPE tuple, and extracts one time-preserving saved-video relation feature.
It contains no model hook, runner, observer, or execution authorization.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import hmac
import math
from typing import Mapping

import numpy as np


TOKEN_GRID_SHAPE = (9, 20, 32)
TOKEN_COUNT = math.prod(TOKEN_GRID_SHAPE)
ATTENTION_HEAD_DIM = 128
TEMPORAL_ROPE_DIMENSION = 44
TEMPORAL_ROPE_PAIR_INDEX = 0
ROPE_TUPLE_SHAPE = (1, TOKEN_COUNT, 1, ATTENTION_HEAD_DIM)
VIDEO_SHAPE_RGB24 = (33, 320, 512, 3)
VIDEO_WINDOW_FRAME_INDICES = tuple(range(11, 22))
TOKEN_WINDOW_TIME_INDICES = (3, 4, 5)
PATCH_SIZE_PIXELS = 16
PATCH_A_TOKEN_COORDINATE = (9, 13)
PATCH_B_TOKEN_COORDINATE = (9, 18)
FEATURE_SHAPE = (11, 6)
PHASE_BUDGET_RADIANS = 0.015625
RELATION_DOMAIN = b"sstw.patch_relation_gate0.coefficient.v1"
RELATION_DESCRIPTOR_ID = "wan21_temporal_rope_patch_pair_relation_001"
FEATURE_SCHEMA_ID = "sstw_saved_rgb24_patch_pair_dct_relation_v1"


@dataclass(frozen=True)
class PublicPatchRelationDescriptor:
    """One frozen public Patch-pair relation."""

    relation_id: str
    coefficients: np.ndarray
    descriptor_digest: str


@dataclass(frozen=True)
class DerivedSignedRelationCoefficient:
    """A key/context-derived sign and its auditable domain-separated digest."""

    signed_coefficient: int
    derivation_digest: str


@dataclass(frozen=True)
class ConstructionWhitening:
    """C0-only fixed center and elementwise scale."""

    center: np.ndarray
    scale: np.ndarray
    whitening_digest: str


@dataclass(frozen=True)
class SignedRelationStatistics:
    """Signed odd/common observability statistics in frozen coordinates."""

    clean_intercept: np.ndarray
    observed_odd: np.ndarray
    observed_common: np.ndarray
    clean_noise_norm: float
    odd_norm: float
    common_norm: float
    antisymmetry_cosine: float
    antisymmetry_residual: float
    common_odd_ratio: float
    odd_clean_noise_ratio: float


@dataclass(frozen=True)
class C0RelationConstruction:
    """Frozen output of C0; this is T_rel, not a separately identified H_e2e."""

    descriptor_digest: str
    feature_schema_id: str
    whitening: ConstructionWhitening
    transfer_values: np.ndarray
    positive_exposure: float
    negative_exposure: float
    statistics: SignedRelationStatistics
    construction_ready: bool
    construction_digest: str


@dataclass(frozen=True)
class GateZeroRelationEvaluation:
    """Held-out identity-A apply-only result."""

    gate_zero_ready: bool
    signed_gate_ready: bool
    transfer_direction_cosine: float
    transfer_relative_error: float
    formal_result: bool = False
    stage_progression_allowed: bool = False
    observer_implementation_allowed: bool = False


def _require_exact_array(
    values: np.ndarray,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype,
    label: str,
) -> np.ndarray:
    array = np.asarray(values)
    if array.shape != shape:
        raise ValueError(f"{label} shape 不匹配: {array.shape}")
    if array.dtype != dtype:
        raise ValueError(f"{label} dtype 必须精确为 {dtype}")
    if not array.flags.c_contiguous:
        raise ValueError(f"{label} 必须为 C-contiguous")
    if np.issubdtype(dtype, np.floating) and not np.all(np.isfinite(array)):
        raise ValueError(f"{label} 必须全部有限")
    return array


def _canonical_array_digest(label: str, values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    payload = (
        label.encode("utf-8")
        + b"\x00"
        + str(array.shape).encode("ascii")
        + b"\x00"
        + array.dtype.str.encode("ascii")
        + b"\x00"
        + array.tobytes(order="C")
    )
    return sha256(payload).hexdigest()


def _token_index(time_index: int, row_index: int, column_index: int) -> int:
    frames, rows, columns = TOKEN_GRID_SHAPE
    if not (
        0 <= time_index < frames
        and 0 <= row_index < rows
        and 0 <= column_index < columns
    ):
        raise ValueError("Patch token coordinate 越界")
    return (time_index * rows + row_index) * columns + column_index


def build_public_patch_relation_descriptor() -> PublicPatchRelationDescriptor:
    """Reconstruct the only public zero-sum Patch-pair relation."""

    coefficients = np.zeros(TOKEN_GRID_SHAPE, dtype="<f4", order="C")
    row_a, column_a = PATCH_A_TOKEN_COORDINATE
    row_b, column_b = PATCH_B_TOKEN_COORDINATE
    for time_index in TOKEN_WINDOW_TIME_INDICES:
        coefficients[time_index, row_a, column_a] = np.float32(1.0)
        coefficients[time_index, row_b, column_b] = np.float32(-1.0)
    descriptor_digest = _canonical_array_digest(
        RELATION_DESCRIPTOR_ID,
        coefficients,
    )
    descriptor = PublicPatchRelationDescriptor(
        relation_id=RELATION_DESCRIPTOR_ID,
        coefficients=coefficients,
        descriptor_digest=descriptor_digest,
    )
    validate_public_patch_relation_descriptor(descriptor)
    return descriptor


def validate_public_patch_relation_descriptor(
    descriptor: PublicPatchRelationDescriptor,
) -> PublicPatchRelationDescriptor:
    """Reject any descriptor that is not the exact public reconstruction."""

    if descriptor.relation_id != RELATION_DESCRIPTOR_ID:
        raise ValueError("relation_id 不匹配")
    coefficients = _require_exact_array(
        descriptor.coefficients,
        shape=TOKEN_GRID_SHAPE,
        dtype=np.dtype("<f4"),
        label="relation coefficients",
    )
    expected = np.zeros(TOKEN_GRID_SHAPE, dtype="<f4", order="C")
    row_a, column_a = PATCH_A_TOKEN_COORDINATE
    row_b, column_b = PATCH_B_TOKEN_COORDINATE
    for time_index in TOKEN_WINDOW_TIME_INDICES:
        expected[time_index, row_a, column_a] = np.float32(1.0)
        expected[time_index, row_b, column_b] = np.float32(-1.0)
    if not np.array_equal(coefficients, expected):
        raise ValueError("relation coefficients 不是冻结的公开 Patch pair")
    for time_index in TOKEN_WINDOW_TIME_INDICES:
        if float(np.sum(coefficients[time_index], dtype=np.float64)) != 0.0:
            raise ValueError("relation coefficients 必须逐窗口严格 zero-sum")
    active = coefficients.reshape(-1)[np.flatnonzero(coefficients)]
    if active.size == 0 or float(active[0]) != 1.0:
        raise ValueError("relation sign canonicalization 不匹配")
    observed_digest = _canonical_array_digest(
        RELATION_DESCRIPTOR_ID,
        coefficients,
    )
    if descriptor.descriptor_digest != observed_digest:
        raise ValueError("relation descriptor digest 不匹配")
    return descriptor


def derive_signed_relation_coefficient(
    *,
    master_key: bytes,
    context_digest: str,
    descriptor: PublicPatchRelationDescriptor,
) -> DerivedSignedRelationCoefficient:
    """Derive one sign without changing the public dictionary.

    Different keys are domain separated by the full derivation digest.  Because
    this first contract has only one signed dimension, two keys may legitimately
    map to the same sign; no wrong-key selectivity claim is made at Gate 0.
    """

    validate_public_patch_relation_descriptor(descriptor)
    if not isinstance(master_key, bytes) or len(master_key) < 16:
        raise ValueError("master_key 必须至少16 bytes")
    if (
        not isinstance(context_digest, str)
        or len(context_digest) != 64
        or any(character not in "0123456789abcdef" for character in context_digest)
    ):
        raise ValueError("context_digest 必须为64字符小写SHA-256")
    message = (
        RELATION_DOMAIN
        + b"\x00"
        + bytes.fromhex(context_digest)
        + bytes.fromhex(descriptor.descriptor_digest)
    )
    digest = hmac.new(master_key, message, "sha256").digest()
    return DerivedSignedRelationCoefficient(
        signed_coefficient=1 if (digest[0] & 1) else -1,
        derivation_digest=digest.hex(),
    )


def build_relation_phase_delta(
    descriptor: PublicPatchRelationDescriptor,
    *,
    signed_coefficient: int,
) -> np.ndarray:
    """Build the compact full-token/head-dimension phase delta.

    Wan v0.35.2 flattens Conv3D tokens with width fastest.  The selected
    temporal RoPE pair occupies head-dimension entries 0 and 1.
    """

    validate_public_patch_relation_descriptor(descriptor)
    if type(signed_coefficient) is not int or signed_coefficient not in (-1, 0, 1):
        raise ValueError("signed_coefficient 只允许 -1/0/+1")
    phase_delta = np.zeros(ROPE_TUPLE_SHAPE, dtype="<f8", order="C")
    if signed_coefficient == 0:
        return phase_delta
    flattened = descriptor.coefficients.reshape(TOKEN_COUNT, order="C")
    signed_phase = (
        flattened.astype(np.float64)
        * float(signed_coefficient)
        * PHASE_BUDGET_RADIANS
    )
    pair_start = 2 * TEMPORAL_ROPE_PAIR_INDEX
    phase_delta[0, :, 0, pair_start] = signed_phase
    phase_delta[0, :, 0, pair_start + 1] = signed_phase
    return phase_delta


def apply_wan_rotary_phase_numpy(
    freqs_cos: np.ndarray,
    freqs_sin: np.ndarray,
    *,
    descriptor: PublicPatchRelationDescriptor,
    signed_coefficient: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the frozen phase shift to Wan's real `(cos, sin)` RoPE tuple."""

    cosine = _require_exact_array(
        freqs_cos,
        shape=ROPE_TUPLE_SHAPE,
        dtype=np.dtype("<f8"),
        label="Wan freqs_cos",
    )
    sine = _require_exact_array(
        freqs_sin,
        shape=ROPE_TUPLE_SHAPE,
        dtype=np.dtype("<f8"),
        label="Wan freqs_sin",
    )
    delta = build_relation_phase_delta(
        descriptor,
        signed_coefficient=signed_coefficient,
    )
    if signed_coefficient == 0:
        return cosine.copy(order="C"), sine.copy(order="C")
    delta_cos = np.cos(delta)
    delta_sin = np.sin(delta)
    shifted_cosine = np.ascontiguousarray(
        cosine * delta_cos - sine * delta_sin,
        dtype="<f8",
    )
    shifted_sine = np.ascontiguousarray(
        sine * delta_cos + cosine * delta_sin,
        dtype="<f8",
    )
    if not np.all(np.isfinite(shifted_cosine)) or not np.all(
        np.isfinite(shifted_sine)
    ):
        raise ValueError("shifted Wan RoPE tuple 出现非有限值")
    return shifted_cosine, shifted_sine


def _dct_vector(size: int, frequency: int) -> np.ndarray:
    if not 0 <= frequency < size:
        raise ValueError("DCT frequency 越界")
    alpha = math.sqrt(1.0 / size) if frequency == 0 else math.sqrt(2.0 / size)
    indices = np.arange(size, dtype=np.float64)
    return alpha * np.cos(
        math.pi * (2.0 * indices + 1.0) * frequency / (2.0 * size)
    )


def _patch_dct_feature(patch: np.ndarray) -> np.ndarray:
    normalized = patch.astype(np.float64) / 255.0
    constant = _dct_vector(PATCH_SIZE_PIXELS, 0)
    first = _dct_vector(PATCH_SIZE_PIXELS, 1)
    horizontal = np.einsum(
        "y,x,yxc->c",
        constant,
        first,
        normalized,
        optimize=False,
    )
    vertical = np.einsum(
        "y,x,yxc->c",
        first,
        constant,
        normalized,
        optimize=False,
    )
    return np.concatenate([horizontal, vertical]).astype("<f8", copy=False)


def extract_saved_rgb24_patch_relation_feature(
    rgb24_video: np.ndarray,
) -> np.ndarray:
    """Extract an 11x6 signed Patch-pair DCT relation without time averaging."""

    video = _require_exact_array(
        rgb24_video,
        shape=VIDEO_SHAPE_RGB24,
        dtype=np.dtype("uint8"),
        label="saved RGB24 video",
    )
    row_a, column_a = PATCH_A_TOKEN_COORDINATE
    row_b, column_b = PATCH_B_TOKEN_COORDINATE
    features = np.empty(FEATURE_SHAPE, dtype="<f8", order="C")
    for output_index, frame_index in enumerate(VIDEO_WINDOW_FRAME_INDICES):
        frame = video[frame_index]
        patch_a = frame[
            row_a * PATCH_SIZE_PIXELS : (row_a + 1) * PATCH_SIZE_PIXELS,
            column_a * PATCH_SIZE_PIXELS : (column_a + 1) * PATCH_SIZE_PIXELS,
            :,
        ]
        patch_b = frame[
            row_b * PATCH_SIZE_PIXELS : (row_b + 1) * PATCH_SIZE_PIXELS,
            column_b * PATCH_SIZE_PIXELS : (column_b + 1) * PATCH_SIZE_PIXELS,
            :,
        ]
        features[output_index] = _patch_dct_feature(patch_a) - _patch_dct_feature(
            patch_b
        )
    if not np.all(np.isfinite(features)):
        raise ValueError("Patch-relation feature 出现非有限值")
    return features


def _require_feature(values: np.ndarray, label: str) -> np.ndarray:
    return _require_exact_array(
        values,
        shape=FEATURE_SHAPE,
        dtype=np.dtype("<f8"),
        label=label,
    )


def _whitening_digest(center: np.ndarray, scale: np.ndarray) -> str:
    return sha256(
        b"sstw.patch_relation_gate0.whitening.v1\x00"
        + center.tobytes(order="C")
        + scale.tobytes(order="C")
    ).hexdigest()


def fit_c0_whitening(
    clean_a: np.ndarray,
    clean_b: np.ndarray,
) -> ConstructionWhitening:
    first = _require_feature(clean_a, "C0 clean_a")
    second = _require_feature(clean_b, "C0 clean_b")
    center = np.ascontiguousarray(0.5 * (first + second), dtype="<f8")
    scale = np.maximum(
        np.abs(first - second) / math.sqrt(2.0),
        1e-6,
    )
    scale = np.ascontiguousarray(scale, dtype="<f8")
    return ConstructionWhitening(
        center=center,
        scale=scale,
        whitening_digest=_whitening_digest(center, scale),
    )


def _validate_whitening(
    whitening: ConstructionWhitening,
) -> ConstructionWhitening:
    center = _require_feature(whitening.center, "whitening center")
    scale = _require_feature(whitening.scale, "whitening scale")
    if np.any(scale < 1e-6):
        raise ValueError("whitening scale 低于冻结 floor")
    if whitening.whitening_digest != _whitening_digest(center, scale):
        raise ValueError("whitening digest 不匹配")
    return whitening


def _apply_whitening(
    values: np.ndarray,
    whitening: ConstructionWhitening,
) -> np.ndarray:
    _validate_whitening(whitening)
    feature = _require_feature(values, "relation feature")
    return np.ascontiguousarray(
        (feature - whitening.center) / whitening.scale,
        dtype="<f8",
    )


def _safe_cosine(left: np.ndarray, right: np.ndarray) -> float:
    left_flat = left.reshape(-1)
    right_flat = right.reshape(-1)
    left_norm = float(np.linalg.norm(left_flat))
    right_norm = float(np.linalg.norm(right_flat))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    value = float(np.dot(left_flat, right_flat) / (left_norm * right_norm))
    if not math.isfinite(value) or value < -1.0 - 1e-12 or value > 1.0 + 1e-12:
        raise ValueError("antisymmetry cosine 超出数值边界")
    return min(1.0, max(-1.0, value))


def compute_signed_relation_statistics(
    *,
    clean_a: np.ndarray,
    clean_b: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
    whitening: ConstructionWhitening,
) -> SignedRelationStatistics:
    whitened_clean_a = _apply_whitening(clean_a, whitening)
    whitened_clean_b = _apply_whitening(clean_b, whitening)
    whitened_positive = _apply_whitening(positive, whitening)
    whitened_negative = _apply_whitening(negative, whitening)
    intercept = 0.5 * (whitened_clean_a + whitened_clean_b)
    positive_delta = whitened_positive - intercept
    negative_delta = whitened_negative - intercept
    odd = 0.5 * (positive_delta - negative_delta)
    common = 0.5 * (positive_delta + negative_delta)
    clean_noise = float(np.linalg.norm(whitened_clean_a - whitened_clean_b))
    clean_noise = max(clean_noise, 1e-6)
    odd_norm = float(np.linalg.norm(odd))
    common_norm = float(np.linalg.norm(common))
    positive_norm = float(np.linalg.norm(positive_delta))
    negative_norm = float(np.linalg.norm(negative_delta))
    denominator = max(positive_norm + negative_norm, 1e-12)
    return SignedRelationStatistics(
        clean_intercept=np.ascontiguousarray(intercept, dtype="<f8"),
        observed_odd=np.ascontiguousarray(odd, dtype="<f8"),
        observed_common=np.ascontiguousarray(common, dtype="<f8"),
        clean_noise_norm=clean_noise,
        odd_norm=odd_norm,
        common_norm=common_norm,
        antisymmetry_cosine=_safe_cosine(positive_delta, -negative_delta),
        antisymmetry_residual=float(
            np.linalg.norm(positive_delta + negative_delta) / denominator
        ),
        common_odd_ratio=common_norm / max(odd_norm, 1e-12),
        odd_clean_noise_ratio=odd_norm / clean_noise,
    )


def signed_gate_ready(statistics: SignedRelationStatistics) -> bool:
    scalar_values = (
        statistics.clean_noise_norm,
        statistics.odd_norm,
        statistics.common_norm,
        statistics.antisymmetry_cosine,
        statistics.antisymmetry_residual,
        statistics.common_odd_ratio,
        statistics.odd_clean_noise_ratio,
    )
    if not all(math.isfinite(value) for value in scalar_values):
        raise ValueError("signed statistics 包含非有限值")
    if statistics.clean_noise_norm <= 0.0:
        raise ValueError("signed statistics clean noise 必须严格为正")
    if (
        statistics.clean_intercept.shape != FEATURE_SHAPE
        or statistics.observed_odd.shape != FEATURE_SHAPE
        or statistics.observed_common.shape != FEATURE_SHAPE
    ):
        raise ValueError("signed statistics vector shape 不匹配")
    for label, array in (
        ("clean_intercept", statistics.clean_intercept),
        ("observed_odd", statistics.observed_odd),
        ("observed_common", statistics.observed_common),
    ):
        _require_exact_array(
            array,
            shape=FEATURE_SHAPE,
            dtype=np.dtype("<f8"),
            label=f"signed statistics {label}",
        )
    recomputed_odd_norm = float(np.linalg.norm(statistics.observed_odd))
    recomputed_common_norm = float(np.linalg.norm(statistics.observed_common))
    if not math.isclose(
        statistics.odd_norm,
        recomputed_odd_norm,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ) or not math.isclose(
        statistics.common_norm,
        recomputed_common_norm,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("signed statistics norm 与 raw vectors 不一致")
    positive_delta = statistics.observed_odd + statistics.observed_common
    negative_delta = statistics.observed_common - statistics.observed_odd
    recomputed_cosine = _safe_cosine(positive_delta, -negative_delta)
    recomputed_residual = float(
        np.linalg.norm(positive_delta + negative_delta)
        / max(
            float(np.linalg.norm(positive_delta))
            + float(np.linalg.norm(negative_delta)),
            1e-12,
        )
    )
    recomputed_common_ratio = recomputed_common_norm / max(
        recomputed_odd_norm,
        1e-12,
    )
    recomputed_noise_ratio = recomputed_odd_norm / statistics.clean_noise_norm
    for label, observed, recomputed in (
        ("antisymmetry_cosine", statistics.antisymmetry_cosine, recomputed_cosine),
        (
            "antisymmetry_residual",
            statistics.antisymmetry_residual,
            recomputed_residual,
        ),
        ("common_odd_ratio", statistics.common_odd_ratio, recomputed_common_ratio),
        (
            "odd_clean_noise_ratio",
            statistics.odd_clean_noise_ratio,
            recomputed_noise_ratio,
        ),
    ):
        if not math.isclose(observed, recomputed, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"signed statistics {label} 与 raw vectors 不一致")
    return bool(
        statistics.antisymmetry_cosine >= 0.9
        and statistics.antisymmetry_residual <= 0.25
        and statistics.common_odd_ratio <= 0.5
        and statistics.odd_clean_noise_ratio >= 3.0
    )


def _construction_digest(
    descriptor_digest: str,
    whitening_digest: str,
    transfer: np.ndarray,
    positive_exposure: float,
    negative_exposure: float,
    statistics: SignedRelationStatistics,
    construction_ready: bool,
) -> str:
    statistic_scalars = np.asarray(
        [
            statistics.clean_noise_norm,
            statistics.odd_norm,
            statistics.common_norm,
            statistics.antisymmetry_cosine,
            statistics.antisymmetry_residual,
            statistics.common_odd_ratio,
            statistics.odd_clean_noise_ratio,
        ],
        dtype="<f8",
    )
    return sha256(
        b"sstw.patch_relation_gate0.c0.v1\x00"
        + bytes.fromhex(descriptor_digest)
        + bytes.fromhex(whitening_digest)
        + transfer.tobytes(order="C")
        + np.asarray(
            [positive_exposure, negative_exposure],
            dtype="<f8",
        ).tobytes()
        + statistics.clean_intercept.tobytes(order="C")
        + statistics.observed_odd.tobytes(order="C")
        + statistics.observed_common.tobytes(order="C")
        + statistic_scalars.tobytes(order="C")
        + (b"\x01" if construction_ready else b"\x00")
    ).hexdigest()


def _validate_local_signed_exposure_pair(
    positive_exposure: float,
    negative_exposure: float,
    *,
    label: str,
) -> float:
    """Validate future-adapter scalars for local formula tests only.

    These caller values are not governed runtime exposure records and cannot
    support execution evidence until a separately reviewed runtime adapter
    measures and binds the realized relation control.
    """

    if (
        not math.isfinite(positive_exposure)
        or not math.isfinite(negative_exposure)
        or positive_exposure <= 0.0
        or negative_exposure >= 0.0
    ):
        raise ValueError(f"{label} signed exposures 必须有限且正负方向正确")
    exposure_span = positive_exposure - negative_exposure
    if not math.isfinite(exposure_span) or exposure_span <= 0.0:
        raise ValueError(f"{label} exposure span 退化")
    return exposure_span


def construct_c0_relation_transfer(
    *,
    descriptor: PublicPatchRelationDescriptor,
    clean_a: np.ndarray,
    clean_b: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
    positive_exposure: float,
    negative_exposure: float,
) -> C0RelationConstruction:
    """Freeze C0 whitening and the restricted end-to-end T_rel."""

    validate_public_patch_relation_descriptor(descriptor)
    exposure_span = _validate_local_signed_exposure_pair(
        positive_exposure,
        negative_exposure,
        label="C0 local primitive",
    )
    whitening = fit_c0_whitening(clean_a, clean_b)
    statistics = compute_signed_relation_statistics(
        clean_a=clean_a,
        clean_b=clean_b,
        positive=positive,
        negative=negative,
        whitening=whitening,
    )
    whitened_positive = _apply_whitening(positive, whitening)
    whitened_negative = _apply_whitening(negative, whitening)
    transfer = np.ascontiguousarray(
        (whitened_positive - whitened_negative) / exposure_span,
        dtype="<f8",
    )
    ready = signed_gate_ready(statistics)
    digest = _construction_digest(
        descriptor.descriptor_digest,
        whitening.whitening_digest,
        transfer,
        positive_exposure,
        negative_exposure,
        statistics,
        ready,
    )
    return C0RelationConstruction(
        descriptor_digest=descriptor.descriptor_digest,
        feature_schema_id=FEATURE_SCHEMA_ID,
        whitening=whitening,
        transfer_values=transfer,
        positive_exposure=float(positive_exposure),
        negative_exposure=float(negative_exposure),
        statistics=statistics,
        construction_ready=ready,
        construction_digest=digest,
    )


def _validate_c0_construction(
    construction: C0RelationConstruction,
    descriptor: PublicPatchRelationDescriptor,
) -> C0RelationConstruction:
    validate_public_patch_relation_descriptor(descriptor)
    if construction.descriptor_digest != descriptor.descriptor_digest:
        raise ValueError("C0 descriptor binding 不匹配")
    if construction.feature_schema_id != FEATURE_SCHEMA_ID:
        raise ValueError("C0 feature schema 不匹配")
    _validate_whitening(construction.whitening)
    transfer = _require_feature(construction.transfer_values, "C0 T_rel")
    if (
        not math.isfinite(construction.positive_exposure)
        or not math.isfinite(construction.negative_exposure)
        or construction.positive_exposure <= 0.0
        or construction.negative_exposure >= 0.0
    ):
        raise ValueError("C0 construction exposure binding 不匹配")
    observed_ready = signed_gate_ready(construction.statistics)
    expected_digest = _construction_digest(
        construction.descriptor_digest,
        construction.whitening.whitening_digest,
        transfer,
        construction.positive_exposure,
        construction.negative_exposure,
        construction.statistics,
        construction.construction_ready,
    )
    if construction.construction_digest != expected_digest:
        raise ValueError("C0 construction digest 不匹配")
    if construction.construction_ready != observed_ready:
        raise ValueError("C0 readiness 与统计不一致")
    return construction


def evaluate_gate0_apply_only(
    *,
    descriptor: PublicPatchRelationDescriptor,
    construction: C0RelationConstruction,
    clean_a: np.ndarray,
    clean_b: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
    positive_exposure: float,
    negative_exposure: float,
) -> GateZeroRelationEvaluation:
    """Evaluate identity A without refitting dictionary, feature, whitening, or T."""

    _validate_c0_construction(construction, descriptor)
    if not construction.construction_ready:
        raise ValueError("C0 construction 未通过，禁止 Gate 0 apply-only")
    exposure_span = _validate_local_signed_exposure_pair(
        positive_exposure,
        negative_exposure,
        label="Gate 0 local primitive",
    )
    statistics = compute_signed_relation_statistics(
        clean_a=clean_a,
        clean_b=clean_b,
        positive=positive,
        negative=negative,
        whitening=construction.whitening,
    )
    exposure_half_span = 0.5 * exposure_span
    predicted_odd = construction.transfer_values * exposure_half_span
    observed_odd = statistics.observed_odd
    direction_cosine = _safe_cosine(predicted_odd, observed_odd)
    relative_error = float(
        np.linalg.norm(observed_odd - predicted_odd)
        / max(float(np.linalg.norm(observed_odd)), 1e-12)
    )
    if not math.isfinite(relative_error):
        raise ValueError("Gate 0 transfer relative error 非有限")
    signed_ready = signed_gate_ready(statistics)
    transfer_ready = direction_cosine >= 0.9 and relative_error <= 0.5
    return GateZeroRelationEvaluation(
        gate_zero_ready=bool(signed_ready and transfer_ready),
        signed_gate_ready=signed_ready,
        transfer_direction_cosine=direction_cosine,
        transfer_relative_error=relative_error,
    )


def frozen_method_boundary() -> Mapping[str, bool]:
    """Return the local primitive boundary; every execution/claim flag is false."""

    return {
        "runtime_implementation_authorized": False,
        "gpu_execution_allowed": False,
        "colab_execution_allowed": False,
        "runner_implementation_allowed": False,
        "notebook_handler_implementation_allowed": False,
        "drive_update_allowed": False,
        "observer_implementation_allowed": False,
        "attack_execution_allowed": False,
        "fixed_fpr_execution_allowed": False,
        "baseline_execution_allowed": False,
        "paper_claim_allowed": False,
        "formal_result": False,
        "stage_progression_allowed": False,
    }
