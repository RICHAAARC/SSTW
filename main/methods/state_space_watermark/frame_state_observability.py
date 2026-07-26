"""NumPy core primitives for frame-state Gate 0 construction.

This module contains no model, GPU, runner, observer, or detection runtime.
It implements only the frozen public atom, scheduler-state exposure, local
output feature, and construction/apply-only Gate 0 mathematics.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from main.methods.state_space_watermark.state_trajectory_injection import (
    _budget_guard_passed,
    search_strict_finite_precision_projection,
)


LATENT_SHAPE = (1, 16, 9, 40, 64)
RESTRICTED_TIME_INDICES = (3, 4, 5)
RESTRICTED_TIME_WEIGHTS = (0.25, 0.5, 0.25)
ATOM_DOMAIN = b"sstw_frame_state_public_decoder_jacobian_atom"
ATOM_ARRAY_KEY = "frame_state_public_atom"
POWER_ITERATION_COUNT = 8
FEATURE_FRAME_INDICES = tuple(range(11, 22))
FEATURE_DIMENSION = 528
MINIMUM_DIRECTION_COSINE = 0.999
LAMBDA_MAX = 0.12
VELOCITY_NORM_RATIO_BUDGET = 0.02
FLOW_ENERGY_BUDGET_RATIO = 0.000015
FROZEN_SIGMA_GRID = (
    1.0,
    0.9475425481796265,
    0.8827877640724182,
    0.8008373379707336,
    0.6937931180000305,
    0.5480455756187439,
    0.33797216415405273,
    0.008928571827709675,
    0.0,
)
FROZEN_WAVEFORM = (0.0, 0.0, 0.0, 0.0, 0.25, 1.0, 0.5, 0.0)
FROZEN_ACTIVE_STEP_INDICES = (4, 5, 6)
CLEAN_NOISE_FLOOR = 1e-6


@dataclass(frozen=True)
class PublicFrameStateAtom:
    values: np.ndarray
    array_digest: str


@dataclass(frozen=True)
class FlowScheduleStep:
    step_index: int
    sigma: float
    next_sigma: float
    delta_sigma: float
    waveform: float
    active: bool


@dataclass(frozen=True)
class FrameStateControlResult:
    step_index: int
    delta_sigma: float
    waveform: float
    constrained_velocity: np.ndarray
    actual_delta_velocity: np.ndarray
    signed_state_update_exposure: float
    actual_delta_norm: float
    intended_delta_norm: float
    joint_norm_budget: float
    remaining_energy: float
    energy_increment: float
    direction_cosine: float | None
    projection_scale: float
    projection_attempt_count: int
    projection_status: str
    inactive_exact_noop: bool


@dataclass(frozen=True)
class _NumpyProjectionEvaluation:
    constrained: np.ndarray
    applied_delta: np.ndarray
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
class SignedResponseStatistics:
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
class CheckpointResponses:
    clean_a: np.ndarray
    clean_b: np.ndarray
    positive: np.ndarray
    negative: np.ndarray


@dataclass(frozen=True)
class GateZeroEvaluation:
    gate_zero_ready: bool
    checkpoint_signed_gate_ready: Mapping[str, bool]
    transfer_direction_cosine: float
    transfer_relative_error: float
    primary_transfer_gate_ready: bool
    formal_result: bool = False
    stage_progression_allowed: bool = False


def _as_little_endian_float32_c(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(values, dtype="<f4"))


def _require_exact_float32_array(
    values: np.ndarray,
    *,
    shape: tuple[int, ...],
    label: str,
) -> np.ndarray:
    array = np.asarray(values)
    if array.shape != shape:
        raise ValueError(f"{label} shape 不匹配")
    if array.dtype != np.dtype("<f4"):
        raise ValueError(f"{label} dtype 必须精确为 float32")
    if not array.flags.c_contiguous:
        raise ValueError(f"{label} 必须为 C-contiguous")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} 必须全部有限")
    return array


def _float32_l2_norm(values: np.ndarray) -> float:
    array = _as_little_endian_float32_c(values)
    return float(np.linalg.norm(array.reshape(-1)))


def clamp_machine_roundoff_cosine(value: float) -> float:
    """Clamp only Cauchy roundoff within the frozen 1e-12 boundary."""

    cosine = float(value)
    if (
        not math.isfinite(cosine)
        or cosine < -1.0 - 1e-12
        or cosine > 1.0 + 1e-12
    ):
        raise ValueError("direction cosine 非有限或超出 Cauchy 边界")
    return min(1.0, max(-1.0, cosine))


def public_atom_digest(values: np.ndarray) -> str:
    array = _as_little_endian_float32_c(values)
    if array.shape != LATENT_SHAPE:
        raise ValueError("public atom shape 必须为冻结 Wan latent layout")
    return sha256(array.tobytes(order="C")).hexdigest()


def _restricted_mask() -> np.ndarray:
    mask = np.zeros(LATENT_SHAPE, dtype=bool)
    mask[:, :, RESTRICTED_TIME_INDICES, :, :] = True
    return mask


def _normalize_restricted(values: np.ndarray) -> np.ndarray:
    observed = np.asarray(values)
    if observed.dtype != np.dtype("<f4"):
        raise ValueError("public atom iteration dtype 必须精确为 float32")
    array = np.ascontiguousarray(observed)
    array = np.where(_restricted_mask(), array, np.float32(0.0))
    norm = _float32_l2_norm(array)
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("public atom restricted direction norm 退化")
    return _as_little_endian_float32_c(array / np.float32(norm))


def build_public_rademacher_initialization() -> np.ndarray:
    """Build the frozen compact-support SHA256 Rademacher initialization."""

    support_count = 1 * 16 * len(RESTRICTED_TIME_INDICES) * 40 * 64
    signs = np.empty(support_count, dtype="<f4")
    filled = 0
    counter = 0
    while filled < support_count:
        digest = sha256(
            ATOM_DOMAIN + b"\x00" + counter.to_bytes(8, "big")
        ).digest()
        for byte in digest:
            for bit_index in range(7, -1, -1):
                if filled >= support_count:
                    break
                signs[filled] = (
                    np.float32(1.0)
                    if ((byte >> bit_index) & 1)
                    else np.float32(-1.0)
                )
                filled += 1
        counter += 1
    result = np.zeros(LATENT_SHAPE, dtype="<f4")
    result[:, :, RESTRICTED_TIME_INDICES, :, :] = signs.reshape(
        (1, 16, len(RESTRICTED_TIME_INDICES), 40, 64),
        order="C",
    )
    return _normalize_restricted(result)


def build_public_frame_state_atom(
    jacobian_gram_callback: Callable[[np.ndarray], np.ndarray],
) -> PublicFrameStateAtom:
    """Run exactly eight generic restricted J^T J iterations."""

    direction = build_public_rademacher_initialization()
    for _ in range(POWER_ITERATION_COUNT):
        candidate = np.asarray(jacobian_gram_callback(direction.copy()))
        if candidate.shape != LATENT_SHAPE:
            raise ValueError("Jacobian callback 输出 shape 不匹配")
        if candidate.dtype != np.dtype("<f4"):
            raise ValueError("Jacobian callback 输出 dtype 必须精确为 float32")
        if not np.all(np.isfinite(candidate)):
            raise ValueError("Jacobian callback 输出非有限")
        direction = _normalize_restricted(candidate)
    weighted = np.zeros(LATENT_SHAPE, dtype="<f4")
    for time_index, weight in zip(
        RESTRICTED_TIME_INDICES,
        RESTRICTED_TIME_WEIGHTS,
        strict=True,
    ):
        weighted[:, :, time_index, :, :] = (
            direction[:, :, time_index, :, :] * np.float32(weight)
        )
    atom = _normalize_restricted(weighted)
    flat = atom.reshape(-1)
    maximum = float(np.max(np.abs(flat)))
    first = int(np.flatnonzero(np.abs(flat) == maximum)[0])
    if float(flat[first]) < 0.0:
        atom = _as_little_endian_float32_c(-atom)
    return validate_public_frame_state_atom(atom)


def validate_public_frame_state_atom(values: np.ndarray) -> PublicFrameStateAtom:
    array = np.asarray(values)
    if array.shape != LATENT_SHAPE:
        raise ValueError("public atom shape 不匹配")
    if array.dtype != np.dtype("<f4"):
        raise ValueError("public atom dtype/byte order 必须为 little-endian float32")
    if not array.flags.c_contiguous or not np.all(np.isfinite(array)):
        raise ValueError("public atom 必须为有限 C-contiguous array")
    mask = _restricted_mask()
    if np.any(array[~mask] != 0.0):
        raise ValueError("public atom restricted support 外必须严格为零")
    norm = _float32_l2_norm(array)
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=20e-6):
        raise ValueError("public atom 必须为冻结 float32 unit direction")
    flat = array.reshape(-1)
    first = int(np.flatnonzero(np.abs(flat) == np.max(np.abs(flat)))[0])
    if float(flat[first]) <= 0.0:
        raise ValueError("public atom sign canonicalization 不匹配")
    return PublicFrameStateAtom(
        values=np.ascontiguousarray(array),
        array_digest=public_atom_digest(array),
    )


def write_public_frame_state_atom(
    path: str | Path,
    atom: PublicFrameStateAtom,
) -> None:
    validated = validate_public_frame_state_atom(atom.values)
    if validated.array_digest != atom.array_digest:
        raise ValueError("public atom digest 不匹配")
    target = Path(path)
    if target.exists():
        raise FileExistsError("public atom artifact 已存在")
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(target, **{ATOM_ARRAY_KEY: validated.values})


def read_public_frame_state_atom(path: str | Path) -> PublicFrameStateAtom:
    with np.load(Path(path), allow_pickle=False) as archive:
        if archive.files != [ATOM_ARRAY_KEY]:
            raise ValueError("public atom NPZ 必须精确包含唯一冻结 array key")
        values = np.array(archive[ATOM_ARRAY_KEY], copy=True)
    return validate_public_frame_state_atom(values)


def build_flow_schedule(
    sigma_grid: Sequence[float] = FROZEN_SIGMA_GRID,
    waveform: Sequence[float] = FROZEN_WAVEFORM,
) -> tuple[FlowScheduleStep, ...]:
    if len(sigma_grid) != 9 or len(waveform) != 8:
        raise ValueError("frame-state Flow schedule 必须精确为8步")
    sigmas = tuple(float(np.float32(value)) for value in sigma_grid)
    waves = tuple(float(value) for value in waveform)
    if (
        any(not math.isfinite(value) for value in sigmas + waves)
        or any(not 0.0 <= value <= 1.0 for value in waves)
        or any(sigmas[index + 1] >= sigmas[index] for index in range(8))
        or sigmas[-1] != 0.0
    ):
        raise ValueError("frame-state sigma/waveform 不满足冻结单调边界")
    frozen_sigmas = tuple(float(np.float32(value)) for value in FROZEN_SIGMA_GRID)
    if sigmas != frozen_sigmas or waves != FROZEN_WAVEFORM:
        raise ValueError("frame-state sigma/waveform 必须与冻结8步合同精确一致")
    return tuple(
        FlowScheduleStep(
            step_index=index,
            sigma=sigmas[index],
            next_sigma=sigmas[index + 1],
            delta_sigma=float(
                np.float32(sigmas[index + 1]) - np.float32(sigmas[index])
            ),
            waveform=waves[index],
            active=waves[index] > 0.0,
        )
        for index in range(8)
    )


def apply_frame_state_control_numpy(
    base_velocity: np.ndarray,
    atom: np.ndarray,
    *,
    signed_state_coefficient: int,
    step: FlowScheduleStep,
    cumulative_control_energy: float,
    cumulative_reference_energy: float,
    remaining_step_count: int,
) -> FrameStateControlResult:
    """Apply one strict representable FP32 velocity control."""

    base = _require_exact_float32_array(
        base_velocity,
        shape=LATENT_SHAPE,
        label="base velocity",
    )
    direction = validate_public_frame_state_atom(np.asarray(atom)).values
    if signed_state_coefficient not in (-1, 0, 1):
        raise ValueError("signed state coefficient 必须为-1/0/1")
    frozen_schedule = build_flow_schedule()
    if (
        not math.isfinite(step.delta_sigma)
        or abs(step.delta_sigma) <= 1e-12
        or step.active is not (step.waveform > 0.0)
        or not math.isfinite(float(cumulative_control_energy))
        or not math.isfinite(float(cumulative_reference_energy))
        or float(cumulative_control_energy) < 0.0
        or float(cumulative_reference_energy) < 0.0
        or int(remaining_step_count) != 8 - int(step.step_index)
        or step.step_index not in range(8)
        or step != frozen_schedule[step.step_index]
    ):
        raise ValueError("step/remaining count 必须绑定冻结8步 schedule")
    base_norm = _float32_l2_norm(base)
    reference_increment = step.delta_sigma**2 * base_norm**2
    projected_reference = (
        float(cumulative_reference_energy)
        + reference_increment * int(remaining_step_count)
    )
    total_energy_budget = FLOW_ENERGY_BUDGET_RATIO * projected_reference
    remaining_energy = max(
        0.0, total_energy_budget - float(cumulative_control_energy)
    )
    if signed_state_coefficient == 0 or not step.active:
        if signed_state_coefficient != 0 and step.waveform != 0.0:
            raise RuntimeError("active signed control 不得静默 no-op")
        return FrameStateControlResult(
            step_index=step.step_index,
            delta_sigma=step.delta_sigma,
            waveform=step.waveform,
            constrained_velocity=base_velocity,
            actual_delta_velocity=np.zeros_like(base),
            signed_state_update_exposure=0.0,
            actual_delta_norm=0.0,
            intended_delta_norm=0.0,
            joint_norm_budget=0.0,
            remaining_energy=remaining_energy,
            energy_increment=0.0,
            direction_cosine=None,
            projection_scale=0.0,
            projection_attempt_count=0,
            projection_status="clean_or_inactive_exact_noop",
            inactive_exact_noop=True,
        )
    if base_norm <= 0.0 or remaining_energy <= 0.0:
        raise RuntimeError("active control 缺少非零 norm/energy budget")
    velocity_sign = signed_state_coefficient * (
        1 if step.delta_sigma > 0.0 else -1
    )
    signed_direction = direction * np.float32(velocity_sign)
    norm_budget = (
        base_norm
        * VELOCITY_NORM_RATIO_BUDGET
        * LAMBDA_MAX
        * step.waveform
    )
    energy_limited = math.sqrt(remaining_energy) / abs(step.delta_sigma)
    intended_norm = min(norm_budget, energy_limited)
    if intended_norm <= 0.0:
        raise RuntimeError("active control intended delta 退化")
    intended_delta = signed_direction * np.float32(intended_norm)

    def evaluate(scale: float) -> _NumpyProjectionEvaluation:
        constrained = np.asarray(
            base + intended_delta * np.float32(scale),
            dtype="<f4",
        )
        applied = np.asarray(constrained - base, dtype="<f4")
        actual_norm = _float32_l2_norm(applied)
        energy = step.delta_sigma**2 * actual_norm**2
        cosine = None
        if actual_norm > 0.0:
            applied64 = applied.astype(np.float64).reshape(-1)
            direction64 = signed_direction.astype(np.float64).reshape(-1)
            raw_cosine = float(
                np.dot(applied64, direction64)
                / (
                    np.linalg.norm(applied64)
                    * np.linalg.norm(direction64)
                )
            )
            cosine = clamp_machine_roundoff_cosine(raw_cosine)
        return _NumpyProjectionEvaluation(
            constrained=constrained,
            applied_delta=applied,
            projection_scale=float(scale),
            actual_delta_norm=actual_norm,
            energy_increment=energy,
            direction_cosine=cosine,
            norm_guard_passed=_budget_guard_passed(
                actual_norm, norm_budget
            ),
            energy_guard_passed=_budget_guard_passed(
                energy, remaining_energy
            ),
            direction_guard_passed=bool(
                cosine is not None
                and math.isfinite(cosine)
                and cosine + 1e-12 >= MINIMUM_DIRECTION_COSINE
            ),
        )

    search = search_strict_finite_precision_projection(
        evaluate,
        joint_norm_budget=norm_budget,
        remaining_energy=remaining_energy,
    )
    if not search.feasible:
        raise RuntimeError(
            "frame-state active FP32 direction/norm/energy guard 失败: "
            f"status={search.status},"
            f"actual_norm={search.evaluation.actual_delta_norm},"
            f"norm_budget={norm_budget},"
            f"energy={search.evaluation.energy_increment},"
            f"remaining_energy={remaining_energy},"
            f"cosine={getattr(search.evaluation, 'direction_cosine', None)}"
        )
    selected = search.evaluation
    actual_coordinate = float(
        np.dot(
            selected.applied_delta.astype(np.float64).reshape(-1),
            direction.astype(np.float64).reshape(-1),
        )
    )
    return FrameStateControlResult(
        step_index=step.step_index,
        delta_sigma=step.delta_sigma,
        waveform=step.waveform,
        constrained_velocity=selected.constrained,
        actual_delta_velocity=selected.applied_delta,
        signed_state_update_exposure=(
            step.delta_sigma * actual_coordinate
        ),
        actual_delta_norm=selected.actual_delta_norm,
        intended_delta_norm=intended_norm,
        joint_norm_budget=norm_budget,
        remaining_energy=remaining_energy,
        energy_increment=selected.energy_increment,
        direction_cosine=selected.direction_cosine,
        projection_scale=selected.projection_scale,
        projection_attempt_count=search.attempt_count,
        projection_status=search.status,
        inactive_exact_noop=False,
    )


def accumulate_actual_signed_exposure(
    step_results: Sequence[FrameStateControlResult],
    atom: np.ndarray,
) -> float:
    """Recompute the exact ordered exposure from governed actual deltas."""

    if len(step_results) != 8 or [
        result.step_index for result in step_results
    ] != list(range(8)):
        raise ValueError("actual exposure 必须绑定完整有序8步")
    direction = validate_public_frame_state_atom(np.asarray(atom)).values
    schedule = build_flow_schedule()
    recomputed_values: list[float] = []
    active_nonzero: list[bool] = []
    for result, step in zip(step_results, schedule, strict=True):
        if (
            result.delta_sigma != step.delta_sigma
            or result.waveform != step.waveform
        ):
            raise ValueError("actual exposure result schedule binding 不匹配")
        actual_delta = _require_exact_float32_array(
            result.actual_delta_velocity,
            shape=LATENT_SHAPE,
            label="actual delta velocity",
        )
        actual_norm = _float32_l2_norm(actual_delta)
        nonzero = actual_norm > 0.0
        if step.active:
            active_nonzero.append(nonzero)
        else:
            if nonzero or not result.inactive_exact_noop:
                raise ValueError("inactive step actual delta 必须 exact-zero no-op")
        if result.actual_delta_norm != actual_norm:
            raise ValueError("actual delta norm 自报值与重算值不一致")
        actual_coordinate = float(
            np.dot(
                actual_delta.astype(np.float64).reshape(-1),
                direction.astype(np.float64).reshape(-1),
            )
        )
        recomputed = float(step.delta_sigma * actual_coordinate)
        if (
            not math.isfinite(result.signed_state_update_exposure)
            or result.signed_state_update_exposure != recomputed
        ):
            raise ValueError("actual signed exposure 自报值与重算值不一致")
        recomputed_values.append(recomputed)
    if active_nonzero and any(active_nonzero) and not all(active_nonzero):
        raise ValueError("signed active steps 必须全部为非零 actual control")
    clean_path = not any(active_nonzero)
    for result, step in zip(step_results, schedule, strict=True):
        if step.active:
            if clean_path:
                if (
                    not result.inactive_exact_noop
                    or result.projection_status
                    != "clean_or_inactive_exact_noop"
                    or result.projection_scale != 0.0
                    or result.projection_attempt_count != 0
                ):
                    raise ValueError("clean active step 必须保持 exact no-op")
            else:
                if (
                    result.inactive_exact_noop
                    or result.projection_status not in {
                        "direct_actual_delta_pass",
                        "bounded_actual_delta_backoff_pass",
                    }
                    or not 0.0 < result.projection_scale <= 1.0
                    or result.projection_attempt_count <= 0
                    or result.direction_cosine is None
                    or not math.isfinite(result.direction_cosine)
                    or not -1.0 <= result.direction_cosine <= 1.0
                    or result.direction_cosine < MINIMUM_DIRECTION_COSINE
                ):
                    raise ValueError("signed active step projection record 不一致")
    return float(math.fsum(recomputed_values))


def extract_local_temporal_feature(
    frames: np.ndarray,
    *,
    rgb24: bool,
) -> np.ndarray:
    """Extract the frozen 11-frame x 4 x 4 x RGB raw 528-D feature."""

    array = np.asarray(frames)
    if array.shape != (33, 320, 512, 3):
        raise ValueError("frame-state feature 输入必须为33x320x512 RGB")
    if rgb24:
        if array.dtype != np.uint8:
            raise ValueError("saved-video feature 必须来自 RGB24 uint8")
        normalized = array.astype(np.float64) / 255.0
    else:
        if array.dtype != np.float32:
            raise ValueError("decoded feature 必须来自 float32 [0,1]")
        if not np.all(np.isfinite(array)) or np.any((array < 0) | (array > 1)):
            raise ValueError("decoded feature 值域必须为闭区间[0,1]")
        normalized = array.astype(np.float64)
    height, width = array.shape[1:3]
    values: list[float] = []
    for frame_index in FEATURE_FRAME_INDICES:
        frame = normalized[frame_index]
        for row in range(4):
            row_start, row_end = row * height // 4, (row + 1) * height // 4
            for column in range(4):
                col_start = column * width // 4
                col_end = (column + 1) * width // 4
                values.extend(
                    np.mean(
                        frame[row_start:row_end, col_start:col_end],
                        axis=(0, 1),
                        dtype=np.float64,
                    ).tolist()
                )
    feature = np.asarray(values, dtype=np.float64)
    if feature.shape != (FEATURE_DIMENSION,) or not np.all(np.isfinite(feature)):
        raise RuntimeError("frame-state local temporal feature 构造失败")
    return feature


def final_latent_carrier_projection(
    final_latent: np.ndarray,
    atom: np.ndarray,
) -> np.ndarray:
    latent = _require_exact_float32_array(
        final_latent,
        shape=LATENT_SHAPE,
        label="final latent",
    )
    direction = validate_public_frame_state_atom(np.asarray(atom)).values
    return np.asarray(
        [
            np.dot(
                latent.astype(np.float64).reshape(-1),
                direction.astype(np.float64).reshape(-1),
            )
        ],
        dtype=np.float64,
    )


def compute_signed_response_statistics(
    responses: CheckpointResponses,
) -> SignedResponseStatistics:
    raw_vectors = [
        np.asarray(value)
        for value in (
            responses.clean_a,
            responses.clean_b,
            responses.positive,
            responses.negative,
        )
    ]
    if any(value.dtype != np.dtype("float64") for value in raw_vectors):
        raise ValueError("signed checkpoint raw vectors 必须精确为 float64")
    vectors = [value.reshape(-1) for value in raw_vectors]
    if len({value.shape for value in vectors}) != 1 or any(
        not np.all(np.isfinite(value)) for value in vectors
    ):
        raise ValueError("signed checkpoint vectors 必须同形且有限")
    clean_intercept = 0.5 * (vectors[0] + vectors[1])
    positive_centered = vectors[2] - clean_intercept
    negative_centered = vectors[3] - clean_intercept
    odd = 0.5 * (vectors[2] - vectors[3])
    common = 0.5 * (vectors[2] + vectors[3]) - clean_intercept
    odd_norm = float(np.linalg.norm(odd))
    common_norm = float(np.linalg.norm(common))
    positive_norm = float(np.linalg.norm(positive_centered))
    negative_norm = float(np.linalg.norm(negative_centered))
    denominator = positive_norm * negative_norm
    cosine = (
        float(
            np.dot(positive_centered, -negative_centered) / denominator
        )
        if denominator > 0.0
        else float("-inf")
    )
    residual_denominator = positive_norm + negative_norm
    residual = (
        float(np.linalg.norm(positive_centered + negative_centered))
        / residual_denominator
        if residual_denominator > 0.0
        else float("inf")
    )
    noise = 0.5 * float(np.linalg.norm(vectors[0] - vectors[1]))
    return SignedResponseStatistics(
        clean_intercept=clean_intercept,
        observed_odd=odd,
        observed_common=common,
        clean_noise_norm=noise,
        odd_norm=odd_norm,
        common_norm=common_norm,
        antisymmetry_cosine=cosine,
        antisymmetry_residual=residual,
        common_odd_ratio=(
            common_norm / odd_norm if odd_norm > 0.0 else float("inf")
        ),
        odd_clean_noise_ratio=(
            odd_norm / max(noise, CLEAN_NOISE_FLOOR)
        ),
    )


def estimate_construction_t0(
    responses: CheckpointResponses,
    *,
    positive_actual_exposure: float,
    negative_actual_exposure: float,
) -> np.ndarray:
    denominator = float(positive_actual_exposure) - float(
        negative_actual_exposure
    )
    if not math.isfinite(denominator) or abs(denominator) <= 1e-12:
        raise ValueError("C0 actual exposure difference 退化")
    statistics = compute_signed_response_statistics(responses)
    return np.asarray(
        2.0 * statistics.observed_odd / denominator,
        dtype=np.float64,
    )


def predict_apply_only_odd(
    t0: np.ndarray,
    *,
    positive_actual_exposure: float,
    negative_actual_exposure: float,
) -> np.ndarray:
    transfer = np.asarray(t0, dtype=np.float64).reshape(-1)
    if transfer.shape != (FEATURE_DIMENSION,) or not np.all(
        np.isfinite(transfer)
    ):
        raise ValueError("apply-only T0 必须为有限528维向量")
    exposure_half_difference = 0.5 * (
        float(positive_actual_exposure)
        - float(negative_actual_exposure)
    )
    if not math.isfinite(exposure_half_difference):
        raise ValueError("identity A exposure 非有限")
    return transfer * exposure_half_difference


def evaluate_gate_zero(
    checkpoint_responses: Mapping[str, CheckpointResponses],
    *,
    predicted_primary_odd: np.ndarray,
) -> GateZeroEvaluation:
    required = {
        "final_latent_carrier_projection",
        "decoded_local_temporal_feature",
        "saved_video_local_temporal_feature",
    }
    if set(checkpoint_responses) != required:
        raise ValueError("Gate 0 checkpoint coverage 不完整")
    expected_dimensions = {
        "final_latent_carrier_projection": 1,
        "decoded_local_temporal_feature": FEATURE_DIMENSION,
        "saved_video_local_temporal_feature": FEATURE_DIMENSION,
    }
    for checkpoint_id, dimension in expected_dimensions.items():
        responses = checkpoint_responses[checkpoint_id]
        if not isinstance(responses, CheckpointResponses):
            raise TypeError(
                f"Gate 0 {checkpoint_id} 必须提供 raw CheckpointResponses"
            )
        if any(
            np.asarray(value).reshape(-1).shape != (dimension,)
            for value in (
                responses.clean_a,
                responses.clean_b,
                responses.positive,
                responses.negative,
            )
        ):
            raise ValueError(
                f"Gate 0 {checkpoint_id} representation dimension 不匹配"
            )
    checkpoint_statistics = {
        checkpoint_id: compute_signed_response_statistics(responses)
        for checkpoint_id, responses in checkpoint_responses.items()
    }
    for checkpoint_id, statistics in checkpoint_statistics.items():
        derived = (
            statistics.clean_noise_norm,
            statistics.odd_norm,
            statistics.common_norm,
            statistics.antisymmetry_cosine,
            statistics.antisymmetry_residual,
            statistics.common_odd_ratio,
            statistics.odd_clean_noise_ratio,
        )
        if any(not math.isfinite(value) for value in derived):
            raise ValueError(
                f"Gate 0 {checkpoint_id} 派生统计必须有限"
            )
    readiness = {
        checkpoint_id: bool(
            value.antisymmetry_cosine >= 0.9
            and value.antisymmetry_residual <= 0.25
            and value.common_odd_ratio <= 0.5
            and value.odd_clean_noise_ratio >= 3.0
        )
        for checkpoint_id, value in checkpoint_statistics.items()
    }
    observed = checkpoint_statistics[
        "saved_video_local_temporal_feature"
    ].observed_odd
    predicted = np.asarray(predicted_primary_odd, dtype=np.float64).reshape(-1)
    if observed.shape != (FEATURE_DIMENSION,) or predicted.shape != (
        FEATURE_DIMENSION,
    ):
        raise ValueError("primary T0 prediction shape 不匹配")
    if not np.all(np.isfinite(predicted)):
        raise ValueError("primary T0 prediction 必须有限")
    denominator = float(np.linalg.norm(observed) * np.linalg.norm(predicted))
    direction_cosine = (
        clamp_machine_roundoff_cosine(
            float(np.dot(observed, predicted) / denominator)
        )
        if denominator > 0.0
        else float("-inf")
    )
    if not math.isfinite(direction_cosine):
        raise ValueError("primary T0 direction cosine 必须有限")
    relative_error = float(np.linalg.norm(observed - predicted)) / max(
        float(np.linalg.norm(observed)),
        1e-6,
    )
    transfer_ready = bool(
        direction_cosine >= 0.9 and relative_error <= 0.5
    )
    return GateZeroEvaluation(
        gate_zero_ready=bool(all(readiness.values()) and transfer_ready),
        checkpoint_signed_gate_ready=readiness,
        transfer_direction_cosine=direction_cosine,
        transfer_relative_error=relative_error,
        primary_transfer_gate_ready=transfer_ready,
    )
