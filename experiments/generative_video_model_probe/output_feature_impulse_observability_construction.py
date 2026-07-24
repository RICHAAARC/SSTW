"""Run the single-identity output-feature impulse observability construction.

This module is deliberately construction-only.  It generates the frozen
14-video impulse plan, records actual FP32 state-update exposure, builds the
audited ``A_actual`` matrix, measures five end-to-end checkpoints, and evaluates
Gate A.  It does not implement an observer, replay, attacks, calibration, or
stage progression.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from evaluation.protocol.impulse_observability_contract import (
    CONSTRUCTION_FEATURE_OUTPUT_DIMENSION,
    CONSTRUCTION_FLOW_STEP_COUNT,
    CONSTRUCTION_LATENT_LAYOUT_SHAPE,
    IMPULSE_CLEAN_REPEAT_COUNT,
    IMPULSE_OBSERVABILITY_PROFILE_ID,
    IMPULSE_PROBE_COUNT,
    IMPULSE_TRIAGE_VIDEO_COUNT,
    STAGE_BASIS_RANK,
    ActualDesignMatrix,
    ActualImpulseExposureTrace,
    ConstructionStageBasis,
    GateAStatistics,
    ImpulseProbePlanRecord,
    assemble_actual_design_matrix,
    build_construction_stage_basis,
    build_impulse_triage_plan,
    canonical_json_digest,
    compute_intended_impulse_control,
    construction_feature_row_binding_digest,
    effective_construction_basis_directions,
    evaluate_gate_a_statistics,
    extract_construction_output_feature_from_normalized_latent,
    extract_construction_reencoded_summary_from_normalized_latent,
    load_impulse_observability_config,
    validate_construction_output_features,
)
from evaluation.protocol.record_writer import write_json, write_jsonl
from main.methods.state_space_watermark.endpoint_latent_detector import (
    configure_wan_vae_encode_memory,
    encode_video_to_wan_endpoint_latent,
)


DEFAULT_CONFIG_PATH = (
    "configs/protocol/"
    "sstw_output_feature_impulse_observability_construction.json"
)
TEST_ID = "output_feature_impulse_observability_construction"
RECORD_VERSION = "output_feature_impulse_observability_construction_v1"
PRIMARY_CHECKPOINT_IDS = (
    "T_latent",
    "T_decoded",
    "T_saved_video",
    "T_reencoded",
    "T_output_feature",
)
CHECKPOINT_SOURCE_STATUS = {
    "T_latent": "ready_actual_final_latent",
    "T_decoded": "ready_pre_save_rgb_float32",
    "T_saved_video": "ready_rgb24_exact_shape_readback",
    "T_reencoded": "ready_streaming_vae_reencode",
    "T_output_feature": "ready_governed_per_video_feature_record",
}
PROJECTION_BACKOFF_SAFETY_FACTOR = 0.999
PROJECTION_MAX_BACKOFF_COUNT = 16
PROJECTION_REFINEMENT_COUNT = 12


@dataclass(frozen=True)
class ProjectionEvaluation:
    """One actual FP32 projection evaluation."""

    constrained: Any
    scale: float
    actual_delta_norm: float
    energy_increment: float
    direction_cosine: float | None
    norm_guard_passed: bool
    energy_guard_passed: bool
    direction_guard_passed: bool
    norm_scale_hint: float | None = None
    energy_scale_hint: float | None = None

    @property
    def all_guards_passed(self) -> bool:
        return bool(
            self.norm_guard_passed
            and self.energy_guard_passed
            and self.direction_guard_passed
            and self.actual_delta_norm > 0.0
        )


@dataclass(frozen=True)
class ProjectionSelection:
    """Largest observed feasible non-zero FP32 projection."""

    evaluation: ProjectionEvaluation
    attempt_count: int
    backoff_count: int
    status: str


@dataclass(frozen=True)
class NumpyImpulseApplication:
    """No-torch reference result used by FP32 rounding regressions."""

    constrained: np.ndarray
    selection: ProjectionSelection | None
    inactive_noop: bool


@dataclass(frozen=True)
class ConstructionGenerationBatch:
    """Generation-side records before output VAE re-encoding."""

    generation_records: tuple[Mapping[str, Any], ...]
    trajectory_step_records: tuple[Mapping[str, Any], ...]
    exposure_traces: tuple[ActualImpulseExposureTrace, ...]
    checkpoint_records: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class ConstructionFeatureBatch:
    """Output-side VAE checkpoint and governed feature records."""

    checkpoint_records: tuple[Mapping[str, Any], ...]
    feature_records: tuple[Mapping[str, Any], ...]


def _stable_digest(payload: Mapping[str, Any]) -> str:
    return sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repository_commit() -> str:
    root = Path(__file__).resolve().parents[2]
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
    ).strip()


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not_installed"


def _strict_budget_guard(actual: float, budget: float) -> bool:
    return bool(
        math.isfinite(float(actual))
        and math.isfinite(float(budget))
        and float(actual) >= 0.0
        and float(budget) >= 0.0
        and float(actual) <= float(budget)
    )


def _stable_direction_cosine(
    *,
    desired_velocity_sign: int,
    target_coordinate: float,
    actual_delta_norm: float,
) -> float:
    """Compute one contract coordinate cosine with roundoff-only clamping."""

    norm = float(actual_delta_norm)
    coordinate = float(target_coordinate)
    sign = int(desired_velocity_sign)
    if (
        sign not in (-1, 1)
        or not math.isfinite(norm)
        or not math.isfinite(coordinate)
        or norm <= 0.0
    ):
        raise RuntimeError("impulse direction cosine context 非法")
    raw = sign * coordinate / norm
    if not math.isfinite(raw) or raw < -1.0 - 1e-12 or raw > 1.0 + 1e-12:
        raise RuntimeError(
            "impulse direction coordinate 违反 Cauchy guard: "
            f"coordinate={coordinate} actual_norm={norm} raw_cosine={raw}"
        )
    return max(-1.0, min(1.0, raw))


def measure_numpy_actual_delta_coordinates(
    actual_delta: np.ndarray,
    basis: ConstructionStageBasis,
    *,
    target_coordinate: int,
    desired_velocity_sign: int,
) -> tuple[float, tuple[float, ...], float]:
    """No-torch reference for the frozen effective-basis measurement."""

    actual = np.asarray(actual_delta, dtype=np.float32).reshape(-1)
    effective = effective_construction_basis_directions(basis)
    directions = effective.astype(np.float64)
    directions /= np.linalg.norm(directions, axis=0, keepdims=True)
    actual64 = actual.astype(np.float64)
    norm = float(np.linalg.norm(actual64))
    coordinates = np.asarray(actual64 @ directions, dtype=np.float64)
    cosine = _stable_direction_cosine(
        desired_velocity_sign=desired_velocity_sign,
        target_coordinate=float(coordinates[target_coordinate]),
        actual_delta_norm=norm,
    )
    coordinates[target_coordinate] = (
        int(desired_velocity_sign) * cosine * norm
    )
    return norm, tuple(float(value) for value in coordinates), cosine


def _require_active_control_feasible(
    *,
    waveform: float,
    intended_delta_norm: float,
    remaining_energy: float,
    base_velocity_norm: float,
    step_index: int,
) -> bool:
    """Reject an active scheduled impulse that collapses to a zero control."""

    active = abs(float(waveform)) > 1e-15
    if not active:
        return False
    numeric = (
        float(intended_delta_norm),
        float(remaining_energy),
        float(base_velocity_norm),
    )
    if (
        any(not math.isfinite(value) for value in numeric)
        or float(intended_delta_norm) <= 1e-15
        or float(remaining_energy) <= 0.0
        or float(base_velocity_norm) <= 0.0
    ):
        raise RuntimeError(
            "active impulse waveform 不存在可行非零 control: "
            f"step={int(step_index)} waveform={float(waveform)} "
            f"base_norm={float(base_velocity_norm)} "
            f"intended_norm={float(intended_delta_norm)} "
            f"remaining_energy={float(remaining_energy)}"
        )
    return True


def _select_largest_feasible_projection(
    evaluator: Callable[[float], ProjectionEvaluation],
) -> ProjectionSelection:
    """Use a deterministic bounded search without relaxing any guard."""

    attempt_count = 1
    backoff_count = 0
    initial = evaluator(1.0)
    if initial.all_guards_passed:
        return ProjectionSelection(
            evaluation=initial,
            attempt_count=attempt_count,
            backoff_count=backoff_count,
            status="direct_actual_delta_pass",
        )
    if not initial.direction_guard_passed:
        raise RuntimeError(
            "impulse projection direction guard 失败: "
            f"scale=1 actual_norm={initial.actual_delta_norm} "
            f"energy={initial.energy_increment} "
            f"cosine={initial.direction_cosine}"
        )

    previous = initial
    feasible: ProjectionEvaluation | None = None
    upper_infeasible_scale = 1.0
    for _ in range(PROJECTION_MAX_BACKOFF_COUNT):
        if (
            previous.norm_guard_passed
            and previous.energy_guard_passed
        ):
            break
        # Evaluators normalize each correction by its frozen budget before
        # returning it through the two optional scale hints below.
        norm_hint = previous.norm_scale_hint
        energy_hint = previous.energy_scale_hint
        hints = [
            float(value)
            for value in (norm_hint, energy_hint)
            if value is not None
        ]
        next_scale = (
            min(hints)
            if hints
            else previous.scale * 0.5
        )
        next_scale = min(
            previous.scale * PROJECTION_BACKOFF_SAFETY_FACTOR,
            next_scale * PROJECTION_BACKOFF_SAFETY_FACTOR,
        )
        if (
            not math.isfinite(next_scale)
            or next_scale <= 0.0
            or next_scale >= previous.scale
        ):
            break
        upper_infeasible_scale = previous.scale
        attempt_count += 1
        backoff_count += 1
        trial = evaluator(next_scale)
        if trial.all_guards_passed:
            feasible = trial
            break
        if not trial.direction_guard_passed:
            break
        previous = trial

    if feasible is None:
        raise RuntimeError(
            "impulse projection 不存在满足 direction/norm/energy 的非零 FP32 delta: "
            f"attempts={attempt_count} scale={previous.scale} "
            f"actual_norm={previous.actual_delta_norm} "
            f"energy={previous.energy_increment} "
            f"cosine={previous.direction_cosine}"
        )

    best = feasible
    for _ in range(PROJECTION_REFINEMENT_COUNT):
        midpoint = 0.5 * (best.scale + upper_infeasible_scale)
        if midpoint <= best.scale:
            break
        attempt_count += 1
        trial = evaluator(midpoint)
        if trial.all_guards_passed:
            best = trial
        else:
            upper_infeasible_scale = midpoint
    return ProjectionSelection(
        evaluation=best,
        attempt_count=attempt_count,
        backoff_count=backoff_count,
        status="bounded_actual_delta_backoff_pass",
    )


def _projection_evaluation(
    *,
    constrained: Any,
    scale: float,
    actual_delta_norm: float,
    delta_sigma: float,
    norm_budget: float,
    remaining_energy: float,
    direction_cosine: float | None,
    minimum_direction_cosine: float,
) -> ProjectionEvaluation:
    energy_increment = float(delta_sigma) ** 2 * float(actual_delta_norm) ** 2
    norm_guard = _strict_budget_guard(actual_delta_norm, norm_budget)
    energy_guard = _strict_budget_guard(
        energy_increment,
        remaining_energy,
    )
    return ProjectionEvaluation(
        constrained=constrained,
        scale=float(scale),
        actual_delta_norm=float(actual_delta_norm),
        energy_increment=energy_increment,
        direction_cosine=direction_cosine,
        norm_guard_passed=norm_guard,
        energy_guard_passed=energy_guard,
        direction_guard_passed=bool(
            direction_cosine is not None
            and math.isfinite(float(direction_cosine))
            and float(direction_cosine) + 1e-12
            >= float(minimum_direction_cosine)
        ),
        norm_scale_hint=(
            float(scale) * float(norm_budget) / max(actual_delta_norm, 1e-30)
            if not norm_guard
            else None
        ),
        energy_scale_hint=(
            float(scale)
            * math.sqrt(
                float(remaining_energy)
                / max(energy_increment, 1e-30)
            )
            if not energy_guard
            else None
        ),
    )


def apply_numpy_float32_impulse(
    base: np.ndarray,
    unit_direction: np.ndarray,
    *,
    signed_delta_norm: float,
    delta_sigma: float,
    norm_budget: float,
    remaining_energy: float,
    minimum_direction_cosine: float,
) -> NumpyImpulseApplication:
    """Reference actual-delta projection using real NumPy FP32 rounding."""

    base32 = np.asarray(base, dtype=np.float32)
    direction32 = np.asarray(unit_direction, dtype=np.float32)
    if base32.shape != direction32.shape:
        raise ValueError("NumPy impulse base/direction shape 不一致")
    direction_norm = float(
        np.linalg.norm(direction32.reshape(-1).astype(np.float64))
    )
    if not math.isfinite(direction_norm) or direction_norm <= 0.0:
        raise ValueError("NumPy impulse direction 必须非零有限")
    direction32 = np.asarray(
        direction32 / np.float32(direction_norm),
        dtype=np.float32,
    )
    signed_norm = float(signed_delta_norm)
    if signed_norm == 0.0:
        return NumpyImpulseApplication(
            constrained=base,
            selection=None,
            inactive_noop=True,
        )
    signed_direction = (
        direction32 if signed_norm > 0.0 else -direction32
    )
    measurement_direction64 = (
        signed_direction.reshape(-1).astype(np.float64)
    )
    measurement_direction64 /= np.linalg.norm(measurement_direction64)
    intended = direction32 * np.float32(signed_norm)

    def evaluator(scale: float) -> ProjectionEvaluation:
        constrained = np.asarray(
            base32 + intended * np.float32(scale),
            dtype=np.float32,
        )
        actual = np.asarray(constrained - base32, dtype=np.float32)
        actual64 = actual.reshape(-1).astype(np.float64)
        norm = float(np.linalg.norm(actual64))
        cosine = (
            None
            if norm <= 0.0
            else _stable_direction_cosine(
                desired_velocity_sign=1,
                target_coordinate=float(
                    actual64 @ measurement_direction64
                ),
                actual_delta_norm=norm,
            )
        )
        return _projection_evaluation(
            constrained=constrained,
            scale=scale,
            actual_delta_norm=norm,
            delta_sigma=delta_sigma,
            norm_budget=norm_budget,
            remaining_energy=remaining_energy,
            direction_cosine=cosine,
            minimum_direction_cosine=minimum_direction_cosine,
        )

    selection = _select_largest_feasible_projection(evaluator)
    return NumpyImpulseApplication(
        constrained=selection.evaluation.constrained,
        selection=selection,
        inactive_noop=False,
    )


def _equal_area_rgb_summary(frames: Any, *, require_rgb24: bool) -> np.ndarray:
    array = np.asarray(frames)
    if (
        array.shape != (33, 320, 512, 3)
        or not np.all(np.isfinite(array))
    ):
        raise ValueError("RGB checkpoint 必须精确为33×320×512×3且有限")
    if require_rgb24:
        if array.dtype != np.uint8:
            raise ValueError("saved video readback 必须为 RGB24 uint8")
        values = array.astype(np.float32) / np.float32(255.0)
    else:
        values = array.astype(np.float32)
        if float(np.min(values)) < 0.0 or float(np.max(values)) > 1.0:
            raise ValueError("pre-save decoded frames 必须位于[0,1]")
    pooled: list[float] = []
    for channel_index in range(3):
        for row_index in range(4):
            row_start = math.floor(row_index * 320 / 4)
            row_end = math.floor((row_index + 1) * 320 / 4)
            for column_index in range(4):
                column_start = math.floor(column_index * 512 / 4)
                column_end = math.floor((column_index + 1) * 512 / 4)
                pooled.append(
                    float(
                        np.mean(
                            values[
                                :,
                                row_start:row_end,
                                column_start:column_end,
                                channel_index,
                            ],
                            dtype=np.float64,
                        )
                    )
                )
    result = np.asarray(pooled, dtype=np.float64)
    if result.shape != (48,) or not np.all(np.isfinite(result)):
        raise RuntimeError("RGB checkpoint 48维 summary 组装失败")
    return result


def read_saved_video_rgb24_summary(
    video_path: str | Path,
    *,
    frame_iterator: Callable[[Path], Any] | None = None,
) -> np.ndarray:
    """Decode the saved MP4 and require the frozen RGB24 boundary."""

    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if frame_iterator is None:
        import imageio.v3 as iio

        frame_iterator = iio.imiter
    frames = [np.asarray(frame) for frame in frame_iterator(path)]
    if not frames:
        raise RuntimeError("saved video 没有可回读帧")
    return _equal_area_rgb_summary(
        np.stack(frames, axis=0),
        require_rgb24=True,
    )


def _checkpoint_record(
    *,
    probe_id: str,
    plan_index: int,
    checkpoint_id: str,
    values: Sequence[float],
    source_path: str,
    source_status: str,
) -> dict[str, Any]:
    numeric = np.asarray(values, dtype=np.float64)
    record = {
        "record_version": RECORD_VERSION,
        "impulse_probe_id": probe_id,
        "impulse_probe_plan_index": int(plan_index),
        "impulse_transfer_checkpoint_id": checkpoint_id,
        "impulse_transfer_checkpoint_dimension": int(numeric.size),
        "impulse_transfer_checkpoint_values": numeric.tolist(),
        "impulse_transfer_checkpoint_source_path": source_path,
        "impulse_transfer_checkpoint_source_status": source_status,
        "formal_result": False,
        "stage_progression_allowed": False,
        "claim_support_status": (
            "output_feature_impulse_observability_construction_only_"
            "not_method_evidence"
        ),
    }
    record["impulse_transfer_checkpoint_record_id"] = _stable_digest(record)
    return record


def _normalized_latent_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    array = np.asarray(value, dtype=np.float32)
    if array.shape != CONSTRUCTION_LATENT_LAYOUT_SHAPE:
        raise ValueError("normalized Wan latent shape 与 construction 合同不一致")
    if not np.all(np.isfinite(array)):
        raise ValueError("normalized Wan latent 含非有限值")
    return array


def build_output_feature_records(
    config: Mapping[str, Any],
    *,
    probe_id: str,
    plan_index: int,
    video_path: str | Path,
    video_sha256: str,
    normalized_latent: Any,
    encoder_metadata: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build T_reencoded, T_output_feature, and its governed row record."""

    latent = _normalized_latent_numpy(normalized_latent)
    reencoded = extract_construction_reencoded_summary_from_normalized_latent(
        latent
    )
    feature = extract_construction_output_feature_from_normalized_latent(
        latent,
        zero_rejection_epsilon=float(
            config["construction_feature_schema"][
                "output_normalization_epsilon"
            ]
        ),
    )
    schema_digest = str(
        config["construction_feature_schema"]["feature_schema_digest"]
    )
    row_binding = construction_feature_row_binding_digest(
        probe_id=probe_id,
        feature_schema_digest=schema_digest,
        feature_values=feature,
    )
    reencoded_record = _checkpoint_record(
        probe_id=probe_id,
        plan_index=plan_index,
        checkpoint_id="T_reencoded",
        values=reencoded,
        source_path=str(video_path),
        source_status="ready_streaming_vae_reencode",
    )
    output_record = _checkpoint_record(
        probe_id=probe_id,
        plan_index=plan_index,
        checkpoint_id="T_output_feature",
        values=feature,
        source_path=str(video_path),
        source_status="ready_governed_per_video_feature_record",
    )
    feature_record = {
        "record_version": RECORD_VERSION,
        "impulse_probe_id": probe_id,
        "impulse_probe_plan_index": int(plan_index),
        "construction_feature_values": feature.tolist(),
        "construction_feature_schema_digest": schema_digest,
        "construction_feature_row_binding_digest": row_binding,
        "construction_feature_row_identity_binding_status": "ready",
        "video_path": str(video_path),
        "video_sha256": str(video_sha256),
        "encoder_metadata": dict(encoder_metadata),
        "formal_result": False,
        "stage_progression_allowed": False,
        "claim_support_status": (
            "output_feature_impulse_observability_construction_only_"
            "not_method_evidence"
        ),
    }
    feature_record["construction_feature_record_id"] = _stable_digest(
        feature_record
    )
    return reencoded_record, output_record, feature_record


def validate_output_vae_metadata(
    config: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> None:
    """Bind the real streaming encode result to the frozen feature schema."""

    schema = config["construction_feature_schema"]
    memory = schema["streaming_memory_config"]
    exact = {
        "endpoint_vae_encode_strategy": (
            "cpu_resident_spatiotemporal_streaming"
        ),
        "endpoint_vae_temporal_chunk_frame_count": memory[
            "temporal_chunk_frame_count"
        ],
        "endpoint_vae_tile_sample_height": memory["tile_sample_height"],
        "endpoint_vae_tile_sample_width": memory["tile_sample_width"],
        "endpoint_vae_tile_sample_stride_height": memory[
            "tile_sample_stride_height"
        ],
        "endpoint_vae_tile_sample_stride_width": memory[
            "tile_sample_stride_width"
        ],
        "endpoint_vae_maximum_incremental_cuda_peak_gib": memory[
            "maximum_incremental_cuda_peak_gib"
        ],
        "endpoint_vae_minimum_cuda_free_gib": memory[
            "minimum_cuda_free_gib"
        ],
        "endpoint_video_frame_count": schema["input_frame_count"],
        "endpoint_vae_model_class": schema["encoder_class"],
        "endpoint_vae_encode_status": "ready",
        "endpoint_latent_shape": list(CONSTRUCTION_LATENT_LAYOUT_SHAPE),
    }
    mismatches = {
        key: {"expected": expected, "observed": metadata.get(key)}
        for key, expected in exact.items()
        if metadata.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(
            "output-side VAE metadata 与冻结 schema 不一致: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )


def validate_checkpoint_and_feature_records(
    config: Mapping[str, Any],
    plan: Sequence[ImpulseProbePlanRecord],
    checkpoint_records: Sequence[Mapping[str, Any]],
    feature_records: Sequence[Mapping[str, Any]],
) -> tuple[Any, dict[str, bool]]:
    """Derive checkpoint readiness from measured records, never booleans."""

    expected_probe_ids = tuple(record.probe_id for record in plan)
    expected_sequence = tuple(
        (probe_id, checkpoint_id)
        for probe_id in expected_probe_ids
        for checkpoint_id in PRIMARY_CHECKPOINT_IDS
    )
    observed_sequence = tuple(
        (
            str(record.get("impulse_probe_id") or ""),
            str(record.get("impulse_transfer_checkpoint_id") or ""),
        )
        for record in checkpoint_records
    )
    if observed_sequence != expected_sequence:
        raise ValueError(
            "five-checkpoint records 必须精确覆盖冻结14-video顺序"
        )
    contracts = config["transfer_checkpoint_contract"]
    checkpoint_values: dict[tuple[str, str], np.ndarray] = {}
    for plan_index, probe_id in enumerate(expected_probe_ids):
        for checkpoint_id in PRIMARY_CHECKPOINT_IDS:
            index = plan_index * len(PRIMARY_CHECKPOINT_IDS) + (
                PRIMARY_CHECKPOINT_IDS.index(checkpoint_id)
            )
            record = checkpoint_records[index]
            values = np.asarray(
                record.get("impulse_transfer_checkpoint_values"),
                dtype=np.float64,
            )
            expected_dimension = int(contracts[checkpoint_id]["dimension"])
            expected_record_id = _stable_digest(
                {
                    key: value
                    for key, value in record.items()
                    if key != "impulse_transfer_checkpoint_record_id"
                }
            )
            if (
                record.get("impulse_probe_plan_index") != plan_index
                or values.shape != (expected_dimension,)
                or not np.all(np.isfinite(values))
                or record.get("impulse_transfer_checkpoint_source_status")
                != CHECKPOINT_SOURCE_STATUS[checkpoint_id]
                or record.get("impulse_transfer_checkpoint_record_id")
                != expected_record_id
            ):
                raise ValueError(
                    f"{probe_id}/{checkpoint_id} checkpoint 不完整"
                )
            checkpoint_values[(probe_id, checkpoint_id)] = values

    if len(feature_records) != IMPULSE_TRIAGE_VIDEO_COUNT:
        raise ValueError("feature records 必须精确为14条")
    feature_probe_ids = tuple(
        str(record.get("impulse_probe_id") or "")
        for record in feature_records
    )
    if feature_probe_ids != expected_probe_ids:
        raise ValueError("feature record probe identity/order 不一致")
    feature_values = tuple(
        record.get("construction_feature_values") or ()
        for record in feature_records
    )
    row_bindings = tuple(
        str(record.get("construction_feature_row_binding_digest") or "")
        for record in feature_records
    )
    schema_digests = {
        str(record.get("construction_feature_schema_digest") or "")
        for record in feature_records
    }
    if schema_digests != {
        str(config["construction_feature_schema"]["feature_schema_digest"])
    }:
        raise ValueError("feature records schema digest 不一致")
    validated_features = validate_construction_output_features(
        config,
        feature_values,
        feature_schema_digest=next(iter(schema_digests)),
        probe_ids=feature_probe_ids,
        row_binding_digests=row_bindings,
    )
    for probe_id, feature in zip(
        expected_probe_ids,
        validated_features.values,
        strict=True,
    ):
        if not np.array_equal(
            feature,
            checkpoint_values[(probe_id, "T_output_feature")],
        ):
            raise ValueError(
                f"{probe_id} feature record 与 T_output_feature 不一致"
            )
    for plan_index, record in enumerate(feature_records):
        expected_record_id = _stable_digest(
            {
                key: value
                for key, value in record.items()
                if key != "construction_feature_record_id"
            }
        )
        if (
            record.get("impulse_probe_plan_index") != plan_index
            or record.get(
                "construction_feature_row_identity_binding_status"
            )
            != "ready"
            or record.get("construction_feature_record_id")
            != expected_record_id
        ):
            raise ValueError("feature governed record identity 未完整绑定")
    return validated_features, {
        checkpoint_id: all(
            (probe_id, checkpoint_id) in checkpoint_values
            for probe_id in expected_probe_ids
        )
        for checkpoint_id in PRIMARY_CHECKPOINT_IDS
    }


def _validate_prompt_seed_source(
    source_root: Path,
    config: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    candidates = list(source_root.rglob("prompt_seed_suite.json"))
    if len(candidates) != 1:
        raise RuntimeError(
            "Gate A source package 必须唯一包含 prompt_seed_suite.json"
        )
    path = candidates[0]
    suite = json.loads(path.read_text(encoding="utf-8-sig"))
    identity = config["execution_identity"]
    if suite.get("prompt_suite_id") != identity["prompt_suite_id"]:
        raise ValueError("prompt suite ID 与冻结 construction identity 不一致")
    prompts = [
        item
        for item in suite.get("prompts", [])
        if item.get("prompt_id") == identity["prompt_id"]
    ]
    seeds = [
        item
        for item in suite.get("seeds", [])
        if item.get("seed_id") == identity["seed_id"]
    ]
    if len(prompts) != 1 or len(seeds) != 1:
        raise ValueError("prompt003/seed2201 在 source suite 中必须唯一")
    prompt = dict(prompts[0])
    seed = dict(seeds[0])
    positive_prompt_digest = sha256(
        str(prompt.get("prompt_text") or "").encode("utf-8")
    ).hexdigest()
    negative_prompt_digest = sha256(
        str(prompt.get("prompt_negative_text") or "").encode("utf-8")
    ).hexdigest()
    if (
        int(seed.get("seed_value", -1)) != int(identity["seed_value"])
        or positive_prompt_digest
        != identity["positive_prompt_text_sha256"]
        or negative_prompt_digest != identity["negative_prompt_text_sha256"]
    ):
        raise ValueError("prompt/seed 内容与冻结 execution identity 不一致")
    return path, prompt, seed


class ImpulseObservabilitySchedulerRuntime:
    """Patch one Flow scheduler and record actual FP32 impulse exposure."""

    def __init__(
        self,
        scheduler: Any,
        *,
        config: Mapping[str, Any],
        plan_record: ImpulseProbePlanRecord,
        basis: ConstructionStageBasis,
    ) -> None:
        self.scheduler = scheduler
        self.config = config
        self.plan_record = plan_record
        self.basis = basis
        self.step_records: list[dict[str, Any]] = []
        self.final_latent: Any | None = None
        self._original_step: Any | None = None
        self._step_index = 0
        self._cumulative_control_energy = 0.0
        self._cumulative_reference_energy = 0.0
        self._basis_tensor: Any | None = None

    def __enter__(self) -> "ImpulseObservabilitySchedulerRuntime":
        if "FlowMatch" not in type(self.scheduler).__name__:
            raise RuntimeError("impulse construction 要求 FlowMatch scheduler")
        self._original_step = self.scheduler.step

        def constrained_step(
            model_output: Any,
            timestep: Any,
            sample: Any,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            return self._run_step(
                model_output,
                timestep,
                sample,
                *args,
                **kwargs,
            )

        self.scheduler.step = constrained_step
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._original_step is not None:
            self.scheduler.step = self._original_step

    def _validate_schedule(self) -> None:
        sigma_values = tuple(
            float(value.detach().float().item())
            if hasattr(value, "detach")
            else float(value)
            for value in self.scheduler.sigmas
        )
        expected = tuple(
            float(value)
            for value in self.config["flow_schedule_contract"]["sigma_grid"]
        )
        if len(sigma_values) != len(expected) or any(
            not math.isclose(
                observed,
                target,
                rel_tol=1e-7,
                abs_tol=1e-8,
            )
            for observed, target in zip(sigma_values, expected, strict=True)
        ):
            raise RuntimeError("真实 scheduler sigma grid 与冻结8-step合同不一致")

    def _torch_basis(self, base: Any) -> Any:
        if self._basis_tensor is None:
            import torch

            matrix = torch.from_numpy(
                effective_construction_basis_directions(self.basis)
            )
            self._basis_tensor = matrix.to(
                device=base.device,
                dtype=torch.float32,
            )
        return self._basis_tensor

    def _apply_active_control(
        self,
        base: Any,
        *,
        target_coordinate: int,
        signed_delta_norm: float,
        delta_sigma: float,
        norm_budget: float,
        remaining_energy: float,
        minimum_direction_cosine: float,
    ) -> tuple[Any, ProjectionSelection, tuple[float, ...]]:
        import torch

        basis = self._torch_basis(base)
        direction = basis[:, target_coordinate].reshape(base.shape)
        measurement_basis = basis.to(dtype=torch.float64)
        measurement_basis = measurement_basis / torch.linalg.vector_norm(
            measurement_basis,
            dim=0,
            keepdim=True,
        )
        desired_velocity_sign = 1 if signed_delta_norm > 0.0 else -1
        intended = direction * float(signed_delta_norm)

        def evaluator(scale: float) -> ProjectionEvaluation:
            constrained = (base + intended * float(scale)).float()
            actual = (constrained - base).float()
            actual64 = actual.reshape(-1).to(dtype=torch.float64)
            norm = float(torch.linalg.vector_norm(actual64).item())
            cosine = (
                None
                if norm <= 0.0
                else _stable_direction_cosine(
                    desired_velocity_sign=desired_velocity_sign,
                    target_coordinate=float(
                        torch.dot(
                            actual64,
                            measurement_basis[:, target_coordinate],
                        ).item()
                    ),
                    actual_delta_norm=norm,
                )
            )
            return _projection_evaluation(
                constrained=constrained,
                scale=scale,
                actual_delta_norm=norm,
                delta_sigma=delta_sigma,
                norm_budget=norm_budget,
                remaining_energy=remaining_energy,
                direction_cosine=cosine,
                minimum_direction_cosine=minimum_direction_cosine,
            )

        selection = _select_largest_feasible_projection(evaluator)
        actual = (selection.evaluation.constrained - base).float()
        actual64 = actual.reshape(-1).to(dtype=torch.float64)
        coordinates_array = (
            actual64 @ measurement_basis
        ).detach().cpu().numpy().astype(np.float64)
        stable_cosine = _stable_direction_cosine(
            desired_velocity_sign=desired_velocity_sign,
            target_coordinate=float(coordinates_array[target_coordinate]),
            actual_delta_norm=selection.evaluation.actual_delta_norm,
        )
        coordinates_array[target_coordinate] = (
            desired_velocity_sign
            * stable_cosine
            * selection.evaluation.actual_delta_norm
        )
        coordinates = tuple(float(value) for value in coordinates_array)
        return selection.evaluation.constrained, selection, coordinates

    def _run_step(
        self,
        model_output: Any,
        timestep: Any,
        sample: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if self._original_step is None:
            raise RuntimeError("impulse scheduler runtime 未进入 context")
        if self._step_index == 0:
            self._validate_schedule()
        if self._step_index >= CONSTRUCTION_FLOW_STEP_COUNT:
            raise RuntimeError("impulse scheduler 调用超过冻结8步")

        schedule = self.config["flow_schedule_contract"]
        exposure = self.config["actual_exposure_contract"]
        index = self._step_index
        phase = float(schedule["flow_phase_by_step"][index])
        delta_sigma = float(schedule["delta_sigma_by_step"][index])
        macro_interval = int(
            schedule["macro_interval_index_by_step"][index]
        )
        base = model_output.detach().float()
        if tuple(base.shape) != CONSTRUCTION_LATENT_LAYOUT_SHAPE:
            raise RuntimeError("Wan model output latent shape 与 construction 不一致")
        base_norm = float(base.norm().item())
        waveform = (
            0.0
            if self.plan_record.stage_index is None
            else float(
                schedule["temporal_waveform_by_macro_interval"][
                    self.plan_record.stage_index
                ][index]
            )
        )
        control = None
        if self.plan_record.polarity:
            control = compute_intended_impulse_control(
                probe_state_update_polarity=self.plan_record.polarity,
                temporal_waveform=waveform,
                delta_sigma=delta_sigma,
                base_velocity_norm=base_norm,
                cumulative_control_energy=self._cumulative_control_energy,
                cumulative_reference_energy=self._cumulative_reference_energy,
                remaining_step_count=CONSTRUCTION_FLOW_STEP_COUNT - index,
                lambda_max=float(exposure["lambda_max"]),
                velocity_norm_ratio_budget=float(
                    exposure["velocity_norm_ratio_budget"]
                ),
                flow_energy_budget_ratio=float(
                    exposure["flow_energy_budget_ratio"]
                ),
            )
        reference_increment = delta_sigma**2 * base_norm**2
        remaining_energy = (
            0.0 if control is None else control.remaining_control_energy
        )
        intended_norm = 0.0 if control is None else control.intended_delta_norm
        intended_exposure = (
            0.0 if control is None else control.signed_state_update_exposure
        )
        target_coordinate = (
            -1
            if self.plan_record.stage_index is None
            else self.plan_record.stage_index * 2
            + int(self.plan_record.channel_index)
        )
        selection: ProjectionSelection | None = None
        coordinates = (0.0,) * STAGE_BASIS_RANK
        constrained = model_output
        actual_norm = 0.0
        energy_increment = 0.0
        direction_cosine = 1.0
        scheduled_active = _require_active_control_feasible(
            waveform=waveform,
            intended_delta_norm=intended_norm,
            remaining_energy=remaining_energy,
            base_velocity_norm=base_norm,
            step_index=index,
        ) if control is not None else False
        if scheduled_active:
            constrained, selection, coordinates = self._apply_active_control(
                base,
                target_coordinate=target_coordinate,
                signed_delta_norm=control.signed_velocity_coordinate,
                delta_sigma=delta_sigma,
                norm_budget=intended_norm,
                remaining_energy=remaining_energy,
                minimum_direction_cosine=float(
                    exposure["minimum_direction_cosine"]
                ),
            )
            actual_norm = selection.evaluation.actual_delta_norm
            energy_increment = selection.evaluation.energy_increment
            direction_cosine = float(
                selection.evaluation.direction_cosine
            )

        actual_exposures = tuple(
            delta_sigma * value for value in coordinates
        )
        self._cumulative_reference_energy += reference_increment
        self._cumulative_control_energy += energy_increment
        record = {
            "record_version": RECORD_VERSION,
            "impulse_probe_id": self.plan_record.probe_id,
            "impulse_stage_index": self.plan_record.stage_index,
            "impulse_state_channel_index": self.plan_record.channel_index,
            "impulse_polarity": self.plan_record.polarity,
            "impulse_flow_step_index": index,
            "impulse_flow_phase": phase,
            "impulse_delta_sigma": delta_sigma,
            "impulse_macro_interval_index": macro_interval,
            "impulse_intended_velocity_waveform": waveform,
            "impulse_reference_base_velocity_norm": base_norm,
            "impulse_remaining_control_energy_before_step": remaining_energy,
            "impulse_reference_energy_increment": reference_increment,
            "impulse_reference_cumulative_energy": (
                self._cumulative_reference_energy
            ),
            "impulse_intended_delta_norm": intended_norm,
            "impulse_actual_velocity_basis_coordinate": (
                0.0
                if target_coordinate < 0
                else coordinates[target_coordinate]
            ),
            "impulse_actual_channel_velocity_coordinate": list(coordinates),
            "impulse_intended_signed_exposure": intended_exposure,
            "impulse_actual_signed_exposure": (
                0.0
                if target_coordinate < 0
                else actual_exposures[target_coordinate]
            ),
            "impulse_actual_channel_exposure": list(actual_exposures),
            "impulse_actual_delta_norm": actual_norm,
            "impulse_actual_projection_scale": (
                0.0
                if intended_norm <= 0.0
                else actual_norm / intended_norm
            ),
            "impulse_finite_precision_projection_scale": (
                0.0 if selection is None else selection.evaluation.scale
            ),
            "impulse_finite_precision_projection_attempt_count": (
                0 if selection is None else selection.attempt_count
            ),
            "impulse_finite_precision_backoff_count": (
                0 if selection is None else selection.backoff_count
            ),
            "impulse_finite_precision_projection_status": (
                "inactive_exact_noop"
                if selection is None
                else selection.status
            ),
            "impulse_cumulative_control_energy": (
                self._cumulative_control_energy
            ),
            "impulse_actual_direction_cosine": direction_cosine,
            "impulse_norm_guard_passed": _strict_budget_guard(
                actual_norm,
                intended_norm,
            ),
            "impulse_energy_guard_passed": _strict_budget_guard(
                energy_increment,
                remaining_energy,
            ),
            "impulse_direction_guard_passed": bool(
                not scheduled_active
                or direction_cosine + 1e-12
                >= float(exposure["minimum_direction_cosine"])
            ),
            "impulse_inactive_exact_noop": not scheduled_active,
            "formal_result": False,
            "stage_progression_allowed": False,
        }
        self.step_records.append(record)
        result = self._original_step(
            constrained,
            timestep,
            sample,
            *args,
            **kwargs,
        )
        previous_sample = (
            result[0]
            if isinstance(result, tuple)
            else getattr(result, "prev_sample", None)
        )
        if previous_sample is None:
            raise RuntimeError("Flow scheduler output 缺少 prev_sample")
        self.final_latent = previous_sample.detach().float()
        self._step_index += 1
        return result

    def build_exposure_trace(self) -> ActualImpulseExposureTrace | None:
        if self.plan_record.polarity == 0:
            return None
        if len(self.step_records) != CONSTRUCTION_FLOW_STEP_COUNT:
            raise RuntimeError("signed impulse trajectory 未精确覆盖8步")
        rows = self.step_records
        actual_vector = np.asarray(
            [
                row["impulse_actual_channel_exposure"]
                for row in rows
            ],
            dtype=np.float64,
        ).sum(axis=0)
        return ActualImpulseExposureTrace(
            probe_id=self.plan_record.probe_id,
            stage_index=int(self.plan_record.stage_index),
            channel_index=int(self.plan_record.channel_index),
            polarity=int(self.plan_record.polarity),
            step_indices=tuple(
                int(row["impulse_flow_step_index"]) for row in rows
            ),
            flow_phase_by_step=tuple(
                float(row["impulse_flow_phase"]) for row in rows
            ),
            delta_sigma_by_step=tuple(
                float(row["impulse_delta_sigma"]) for row in rows
            ),
            macro_interval_index_by_step=tuple(
                int(row["impulse_macro_interval_index"]) for row in rows
            ),
            intended_velocity_waveform_by_step=tuple(
                float(row["impulse_intended_velocity_waveform"])
                for row in rows
            ),
            reference_base_velocity_norm_by_step=tuple(
                float(row["impulse_reference_base_velocity_norm"])
                for row in rows
            ),
            remaining_control_energy_before_step_by_step=tuple(
                float(
                    row[
                        "impulse_remaining_control_energy_before_step"
                    ]
                )
                for row in rows
            ),
            reference_energy_increment_by_step=tuple(
                float(row["impulse_reference_energy_increment"])
                for row in rows
            ),
            reference_cumulative_energy_by_step=tuple(
                float(row["impulse_reference_cumulative_energy"])
                for row in rows
            ),
            intended_delta_norm_by_step=tuple(
                float(row["impulse_intended_delta_norm"]) for row in rows
            ),
            actual_velocity_basis_coordinate_by_step=tuple(
                float(row["impulse_actual_velocity_basis_coordinate"])
                for row in rows
            ),
            actual_channel_velocity_coordinate_by_step=tuple(
                tuple(
                    float(value)
                    for value in row[
                        "impulse_actual_channel_velocity_coordinate"
                    ]
                )
                for row in rows
            ),
            intended_signed_exposure_by_step=tuple(
                float(row["impulse_intended_signed_exposure"])
                for row in rows
            ),
            actual_signed_exposure_by_step=tuple(
                float(row["impulse_actual_signed_exposure"])
                for row in rows
            ),
            actual_channel_exposure_by_step=tuple(
                tuple(
                    float(value)
                    for value in row["impulse_actual_channel_exposure"]
                )
                for row in rows
            ),
            actual_exposure_vector=tuple(
                float(value) for value in actual_vector
            ),
            delta_norm_by_step=tuple(
                float(row["impulse_actual_delta_norm"]) for row in rows
            ),
            projection_scale_by_step=tuple(
                float(row["impulse_actual_projection_scale"])
                for row in rows
            ),
            cumulative_energy_by_step=tuple(
                float(row["impulse_cumulative_control_energy"])
                for row in rows
            ),
            direction_cosine_by_step=tuple(
                float(row["impulse_actual_direction_cosine"])
                for row in rows
            ),
            norm_guard_passed_by_step=tuple(
                bool(row["impulse_norm_guard_passed"]) for row in rows
            ),
            energy_guard_passed_by_step=tuple(
                bool(row["impulse_energy_guard_passed"]) for row in rows
            ),
            waveform_schema_digest=str(
                self.config["flow_schedule_contract"][
                    "waveform_schema_digest"
                ]
            ),
            runtime_adapter_schema_digest=str(
                self.config["runtime_adapter_contract"][
                    "adapter_schema_digest"
                ]
            ),
            basis_digest=self.basis.basis_digest,
        )


def _latent_checkpoint_summary(
    latent: Any,
    basis: ConstructionStageBasis,
) -> np.ndarray:
    import torch

    value = latent.detach().float()
    if tuple(value.shape) != CONSTRUCTION_LATENT_LAYOUT_SHAPE:
        raise ValueError("T_latent shape 与 construction 合同不一致")
    matrix = torch.from_numpy(
        np.asarray(basis.values, dtype=np.float32)
    ).to(device=value.device)
    flattened = value.reshape(-1)
    result = [
        float(torch.dot(flattened, matrix[:, index]).item())
        for index in range(STAGE_BASIS_RANK)
    ]
    result.append(float(value.norm().item()))
    array = np.asarray(result, dtype=np.float64)
    if array.shape != (7,) or not np.all(np.isfinite(array)):
        raise RuntimeError("T_latent 7维 summary 组装失败")
    return array


def _write_generation_snapshots(
    output_root: Path,
    generation_records: Sequence[Mapping[str, Any]],
    step_records: Sequence[Mapping[str, Any]],
    exposure_traces: Sequence[ActualImpulseExposureTrace],
    checkpoint_records: Sequence[Mapping[str, Any]],
) -> None:
    write_jsonl(
        output_root / "records" / "impulse_generation_records.jsonl",
        list(generation_records),
    )
    write_jsonl(
        output_root / "records" / "impulse_trajectory_step_records.jsonl",
        list(step_records),
    )
    write_jsonl(
        output_root / "records" / "impulse_actual_exposure_traces.jsonl",
        [asdict(trace) for trace in exposure_traces],
    )
    write_jsonl(
        output_root / "records" / "impulse_checkpoint_records.jsonl",
        list(checkpoint_records),
    )


def execute_real_impulse_generation(
    config: Mapping[str, Any],
    *,
    prompt: Mapping[str, Any],
    seed: Mapping[str, Any],
    output_root: Path,
    plan: Sequence[ImpulseProbePlanRecord],
    basis: ConstructionStageBasis,
) -> ConstructionGenerationBatch:
    """Generate the frozen 14 videos with one cached Wan pipeline."""

    import torch

    from experiments.generative_video_model_probe.colab_runtime import (
        _export_video,
        _generation_model_provenance_from_pipeline,
        _load_video_generation_pipeline,
        _scheduler_signature,
        _select_dtype,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("Gate A construction 需要可用 Colab CUDA GPU")
    identity = config["execution_identity"]
    positive_prompt_digest = sha256(
        str(prompt.get("prompt_text") or "").encode("utf-8")
    ).hexdigest()
    negative_prompt_digest = sha256(
        str(prompt.get("prompt_negative_text") or "").encode("utf-8")
    ).hexdigest()
    if (
        prompt.get("prompt_id") != identity["prompt_id"]
        or seed.get("seed_id") != identity["seed_id"]
        or int(seed.get("seed_value", -1)) != int(identity["seed_value"])
        or positive_prompt_digest
        != identity["positive_prompt_text_sha256"]
        or negative_prompt_digest != identity["negative_prompt_text_sha256"]
    ):
        raise RuntimeError("generation prompt/seed identity 未绑定冻结 source")
    if _package_version("diffusers") != identity["diffusers_version"]:
        raise RuntimeError("diffusers runtime version 与冻结 construction 不一致")
    pipe = _load_video_generation_pipeline(
        identity["generation_model_id"],
        _select_dtype(torch),
        revision=identity["generation_model_revision"],
    )
    provenance = _generation_model_provenance_from_pipeline(
        pipe,
        expected_model_id=identity["generation_model_id"],
    )
    if (
        provenance["generation_model_commit_or_hash"]
        != identity["generation_model_revision"]
        or _scheduler_signature(pipe.scheduler)
        != identity["scheduler_signature"]
    ):
        raise RuntimeError("generation model/scheduler provenance 未冻结")
    generation_records: list[dict[str, Any]] = []
    step_records: list[dict[str, Any]] = []
    exposure_traces: list[ActualImpulseExposureTrace] = []
    checkpoint_records: list[dict[str, Any]] = []
    videos_root = output_root / "videos"
    videos_root.mkdir(parents=True, exist_ok=True)

    for plan_index, probe in enumerate(plan):
        generator = torch.Generator(device="cuda").manual_seed(
            int(seed["seed_value"])
        )
        generator_digest = sha256(
            generator.get_state().cpu().numpy().tobytes()
        ).hexdigest()
        video_path = videos_root / f"{plan_index:02d}_{probe.probe_id}.mp4"
        started = time.time()
        runtime = ImpulseObservabilitySchedulerRuntime(
            pipe.scheduler,
            config=config,
            plan_record=probe,
            basis=basis,
        )
        try:
            with runtime:
                result = pipe(
                    prompt=prompt["prompt_text"],
                    negative_prompt=prompt.get("prompt_negative_text"),
                    generator=generator,
                    height=int(identity["height"]),
                    width=int(identity["width"]),
                    num_frames=int(identity["num_frames"]),
                    num_inference_steps=int(
                        identity["num_inference_steps"]
                    ),
                    guidance_scale=float(identity["guidance_scale"]),
                    output_type="np",
                )
            if runtime.final_latent is None:
                raise RuntimeError("generation 未捕获 final latent")
            decoded_frames = np.asarray(result.frames[0])
            decoded_summary = _equal_area_rgb_summary(
                decoded_frames,
                require_rgb24=False,
            )
            _export_video(
                result.frames[0],
                video_path,
                fps=int(identity["fps"]),
            )
            video_digest = _sha256_file(video_path)
            saved_summary = read_saved_video_rgb24_summary(video_path)
            latent_summary = _latent_checkpoint_summary(
                runtime.final_latent,
                basis,
            )
            exposure_trace = runtime.build_exposure_trace()
            if exposure_trace is not None:
                exposure_traces.append(exposure_trace)
            step_records.extend(runtime.step_records)
            checkpoint_records.extend(
                (
                    _checkpoint_record(
                        probe_id=probe.probe_id,
                        plan_index=plan_index,
                        checkpoint_id="T_latent",
                        values=latent_summary,
                        source_path="generation_scheduler_final_latent",
                        source_status="ready_actual_final_latent",
                    ),
                    _checkpoint_record(
                        probe_id=probe.probe_id,
                        plan_index=plan_index,
                        checkpoint_id="T_decoded",
                        values=decoded_summary,
                        source_path=str(video_path),
                        source_status="ready_pre_save_rgb_float32",
                    ),
                    _checkpoint_record(
                        probe_id=probe.probe_id,
                        plan_index=plan_index,
                        checkpoint_id="T_saved_video",
                        values=saved_summary,
                        source_path=str(video_path),
                        source_status="ready_rgb24_exact_shape_readback",
                    ),
                )
            )
            generation_record = {
                "record_version": RECORD_VERSION,
                "impulse_probe_id": probe.probe_id,
                "impulse_probe_plan_index": plan_index,
                "impulse_probe_role": probe.probe_role,
                "impulse_stage_index": probe.stage_index,
                "impulse_state_channel_index": probe.channel_index,
                "impulse_polarity": probe.polarity,
                "impulse_nominal_signed_amplitude": (
                    probe.nominal_signed_amplitude
                ),
                "generation_status": "success",
                "generation_failure_reason": "none",
                "generation_runtime_sec": round(time.time() - started, 3),
                "generation_model_id": identity["generation_model_id"],
                "generation_model_revision": (
                    identity["generation_model_revision"]
                ),
                "scheduler_signature": identity["scheduler_signature"],
                "prompt_id": identity["prompt_id"],
                "positive_prompt_text_sha256": positive_prompt_digest,
                "negative_prompt_text_sha256": negative_prompt_digest,
                "seed_id": identity["seed_id"],
                "generation_seed_random": int(seed["seed_value"]),
                "generation_generator_state_digest_random": generator_digest,
                "video_path": str(video_path),
                "video_sha256": video_digest,
                "trajectory_step_count": len(runtime.step_records),
                "endpoint_control_enabled": False,
                "formal_result": False,
                "stage_progression_allowed": False,
                "claim_support_status": (
                    "output_feature_impulse_observability_construction_only_"
                    "not_method_evidence"
                ),
            }
            generation_record["impulse_generation_record_id"] = (
                _stable_digest(generation_record)
            )
            generation_records.append(generation_record)
            _write_generation_snapshots(
                output_root,
                generation_records,
                step_records,
                exposure_traces,
                checkpoint_records,
            )
        except Exception as exc:
            failure_record = {
                "record_version": RECORD_VERSION,
                "impulse_probe_id": probe.probe_id,
                "impulse_probe_plan_index": plan_index,
                "generation_status": "failed",
                "generation_failure_reason": str(exc),
                "generation_runtime_sec": round(time.time() - started, 3),
                "prompt_id": identity["prompt_id"],
                "positive_prompt_text_sha256": positive_prompt_digest,
                "negative_prompt_text_sha256": negative_prompt_digest,
                "seed_id": identity["seed_id"],
                "generation_seed_random": int(seed["seed_value"]),
                "video_path": str(video_path),
                "formal_result": False,
                "stage_progression_allowed": False,
            }
            generation_records.append(failure_record)
            _write_generation_snapshots(
                output_root,
                generation_records,
                step_records,
                exposure_traces,
                checkpoint_records,
            )
            raise
    batch = ConstructionGenerationBatch(
        generation_records=tuple(generation_records),
        trajectory_step_records=tuple(step_records),
        exposure_traces=tuple(exposure_traces),
        checkpoint_records=tuple(checkpoint_records),
    )
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return batch


def extract_real_output_features(
    config: Mapping[str, Any],
    *,
    output_root: Path,
    plan: Sequence[ImpulseProbePlanRecord],
    generation_records: Sequence[Mapping[str, Any]],
) -> ConstructionFeatureBatch:
    """Load the frozen output-side VAE once and re-encode all saved videos."""

    import torch
    from diffusers import AutoencoderKLWan

    feature_schema = config["construction_feature_schema"]
    identity = config["execution_identity"]
    vae = AutoencoderKLWan.from_pretrained(
        identity["generation_model_id"],
        subfolder=feature_schema["encoder_subfolder"],
        revision=feature_schema["encoder_revision"],
        torch_dtype=torch.bfloat16,
        token=os.environ.get("HF_TOKEN"),
    )
    vae.to("cuda")
    vae.eval()
    if (
        type(vae).__name__ != feature_schema["encoder_class"]
        or vae.dtype != torch.bfloat16
        or getattr(vae.config, "patch_size", None)
        != feature_schema["encode_patch_size_required"]
    ):
        raise RuntimeError("output-side VAE class/dtype/patch_size 未冻结")
    configure_wan_vae_encode_memory(vae)
    checkpoints: list[dict[str, Any]] = []
    features: list[dict[str, Any]] = []
    for plan_index, (probe, generation) in enumerate(
        zip(plan, generation_records, strict=True)
    ):
        video_path = Path(str(generation["video_path"]))
        normalized, metadata = encode_video_to_wan_endpoint_latent(
            vae,
            video_path,
        )
        validate_output_vae_metadata(config, metadata)
        governed_metadata = {
            **metadata,
            "construction_feature_encoder_id": (
                feature_schema["encoder_id"]
            ),
            "construction_feature_encoder_revision": (
                feature_schema["encoder_revision"]
            ),
            "construction_feature_encode_execution_dtype": "bfloat16",
            "diffusers_version": _package_version("diffusers"),
        }
        reencoded, output, feature = build_output_feature_records(
            config,
            probe_id=probe.probe_id,
            plan_index=plan_index,
            video_path=video_path,
            video_sha256=str(generation["video_sha256"]),
            normalized_latent=normalized,
            encoder_metadata=governed_metadata,
        )
        checkpoints.extend((reencoded, output))
        features.append(feature)
        write_jsonl(
            output_root / "records" / "impulse_feature_records.jsonl",
            features,
        )
    batch = ConstructionFeatureBatch(
        checkpoint_records=tuple(checkpoints),
        feature_records=tuple(features),
    )
    del vae
    gc.collect()
    torch.cuda.empty_cache()
    return batch


def _validate_generation_batch(
    config: Mapping[str, Any],
    plan: Sequence[ImpulseProbePlanRecord],
    batch: ConstructionGenerationBatch,
) -> None:
    expected_ids = tuple(record.probe_id for record in plan)
    observed_ids = tuple(
        str(record.get("impulse_probe_id") or "")
        for record in batch.generation_records
    )
    generator_digests = {
        str(
            record.get("generation_generator_state_digest_random")
            or ""
        )
        for record in batch.generation_records
    }
    video_paths = tuple(
        str(record.get("video_path") or "")
        for record in batch.generation_records
    )
    identity = config["execution_identity"]
    identity_ready = all(
        record.get("impulse_probe_plan_index") == index
        and record.get("impulse_probe_role") == plan[index].probe_role
        and record.get("impulse_stage_index") == plan[index].stage_index
        and record.get("impulse_state_channel_index")
        == plan[index].channel_index
        and record.get("impulse_polarity") == plan[index].polarity
        and record.get("endpoint_control_enabled") is False
        and record.get("generation_model_id")
        == identity["generation_model_id"]
        and record.get("generation_model_revision")
        == identity["generation_model_revision"]
        and record.get("scheduler_signature")
        == identity["scheduler_signature"]
        and record.get("prompt_id") == identity["prompt_id"]
        and record.get("positive_prompt_text_sha256")
        == identity["positive_prompt_text_sha256"]
        and record.get("negative_prompt_text_sha256")
        == identity["negative_prompt_text_sha256"]
        and record.get("seed_id") == identity["seed_id"]
        and record.get("generation_seed_random") == identity["seed_value"]
        and int(record.get("trajectory_step_count") or 0)
        == CONSTRUCTION_FLOW_STEP_COUNT
        for index, record in enumerate(batch.generation_records)
    )
    video_files_ready = all(
        Path(path).is_file()
        and _sha256_file(Path(path))
        == str(record.get("video_sha256") or "")
        for path, record in zip(
            video_paths,
            batch.generation_records,
            strict=True,
        )
    )
    expected_step_sequence = tuple(
        (probe_id, step_index)
        for probe_id in expected_ids
        for step_index in range(CONSTRUCTION_FLOW_STEP_COUNT)
    )
    observed_step_sequence = tuple(
        (
            str(record.get("impulse_probe_id") or ""),
            int(record.get("impulse_flow_step_index", -1)),
        )
        for record in batch.trajectory_step_records
    )
    clean_steps = batch.trajectory_step_records[
        : IMPULSE_CLEAN_REPEAT_COUNT * CONSTRUCTION_FLOW_STEP_COUNT
    ]
    clean_noop_ready = all(
        float(record.get("impulse_actual_delta_norm") or 0.0) == 0.0
        and record.get("impulse_inactive_exact_noop") is True
        for record in clean_steps
    )
    expected_trace_ids = tuple(
        record.probe_id
        for record in plan
        if record.probe_role == "signed_interval_impulse"
    )
    if (
        observed_ids != expected_ids
        or len(batch.generation_records) != IMPULSE_TRIAGE_VIDEO_COUNT
        or any(
            record.get("generation_status") != "success"
            for record in batch.generation_records
        )
        or generator_digests == {""}
        or len(generator_digests) != 1
        or len(set(video_paths)) != IMPULSE_TRIAGE_VIDEO_COUNT
        or any(not path for path in video_paths)
        or not identity_ready
        or not video_files_ready
        or observed_step_sequence != expected_step_sequence
        or not clean_noop_ready
        or len(batch.exposure_traces) != IMPULSE_PROBE_COUNT
        or tuple(
            trace.probe_id for trace in batch.exposure_traces
        )
        != expected_trace_ids
        or len(batch.trajectory_step_records)
        != IMPULSE_TRIAGE_VIDEO_COUNT * CONSTRUCTION_FLOW_STEP_COUNT
    ):
        raise RuntimeError(
            "14-video generation identity/success/RNG/exposure coverage 未就绪"
        )


def _actual_design_record(
    design: ActualDesignMatrix,
    *,
    basis_digest: str,
) -> dict[str, Any]:
    return {
        "record_version": RECORD_VERSION,
        "impulse_actual_design_probe_ids": list(design.probe_ids),
        "impulse_actual_design_values": design.values.tolist(),
        "impulse_actual_design_rank": design.rank,
        "impulse_actual_design_condition_number": design.condition_number,
        "impulse_actual_design_compression_allowed": (
            design.compression_allowed
        ),
        "impulse_waveform_cosine_by_probe": dict(
            design.waveform_cosine_by_probe
        ),
        "impulse_intended_actual_ratio_by_probe": {
            key: list(value)
            for key, value in design.intended_actual_ratio_by_probe.items()
        },
        "impulse_positive_negative_waveform_symmetry": dict(
            design.positive_negative_symmetry_by_channel
        ),
        "impulse_positive_negative_amplitude_asymmetry": dict(
            design.positive_negative_amplitude_asymmetry_by_channel
        ),
        "impulse_cross_channel_leakage_ratio": dict(
            design.cross_channel_leakage_by_probe
        ),
        "construction_basis_digest": basis_digest,
        "formal_result": False,
        "stage_progression_allowed": False,
    }


def _gate_statistics_dict(stats: GateAStatistics) -> dict[str, Any]:
    return {
        **asdict(stats),
        "impulse_sample_internal_observability_gate_ready": (
            stats.gate_a_ready
        ),
    }


def run_output_feature_impulse_observability_construction(
    source_root: str | Path,
    output_root: str | Path,
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    generation_executor: Callable[..., ConstructionGenerationBatch] = (
        execute_real_impulse_generation
    ),
    feature_executor: Callable[..., ConstructionFeatureBatch] = (
        extract_real_output_features
    ),
    basis_builder: Callable[[str], ConstructionStageBasis] = (
        build_construction_stage_basis
    ),
) -> dict[str, Any]:
    """Run or locally simulate the authorized, construction-only Gate A."""

    source = Path(source_root).resolve()
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = load_impulse_observability_config(config_path)
    authorization = config["authorization_state_machine"]
    if (
        authorization["impulse_triage_execution_allowed"] is not True
        or authorization["current_state"]
        != "impulse_triage_execution_authorized_pending_user_colab_run"
    ):
        raise RuntimeError("14-video impulse triage 尚未获授权")
    plan = build_impulse_triage_plan(config)
    prompt_suite_path, prompt, seed = _validate_prompt_seed_source(
        source,
        config,
    )
    master_key = os.environ.get("SSTW_TRAJECTORY_AUTHENTICATION_KEY") or ""
    if len(master_key.encode("utf-8")) < int(
        config["construction_basis"]["master_key_minimum_utf8_bytes"]
    ):
        raise RuntimeError("construction owner master key 缺失或过短")
    basis = basis_builder(master_key)
    repository_commit = _repository_commit()
    config_digest = canonical_json_digest(config)
    runtime_versions = {
        "python_version": sys.version.split()[0],
        "torch_version": _package_version("torch"),
        "diffusers_version": _package_version("diffusers"),
    }
    write_jsonl(
        output / "records" / "impulse_generation_plan.jsonl",
        [asdict(record) for record in plan],
    )

    try:
        generation = generation_executor(
            config,
            prompt=prompt,
            seed=seed,
            output_root=output,
            plan=plan,
            basis=basis,
        )
        _validate_generation_batch(config, plan, generation)
        _write_generation_snapshots(
            output,
            generation.generation_records,
            generation.trajectory_step_records,
            generation.exposure_traces,
            generation.checkpoint_records,
        )
        design = assemble_actual_design_matrix(
            config,
            generation.exposure_traces,
        )
        feature_batch = feature_executor(
            config,
            output_root=output,
            plan=plan,
            generation_records=generation.generation_records,
        )
        ordered_checkpoints: list[Mapping[str, Any]] = []
        generation_checkpoint_map = {
            (
                str(record["impulse_probe_id"]),
                str(record["impulse_transfer_checkpoint_id"]),
            ): record
            for record in generation.checkpoint_records
        }
        feature_checkpoint_map = {
            (
                str(record["impulse_probe_id"]),
                str(record["impulse_transfer_checkpoint_id"]),
            ): record
            for record in feature_batch.checkpoint_records
        }
        for probe in plan:
            for checkpoint_id in PRIMARY_CHECKPOINT_IDS:
                mapping = (
                    generation_checkpoint_map
                    if checkpoint_id
                    in {"T_latent", "T_decoded", "T_saved_video"}
                    else feature_checkpoint_map
                )
                ordered_checkpoints.append(
                    mapping[(probe.probe_id, checkpoint_id)]
                )
        validated_features, checkpoint_ready = (
            validate_checkpoint_and_feature_records(
                config,
                plan,
                ordered_checkpoints,
                feature_batch.feature_records,
            )
        )
        stats = evaluate_gate_a_statistics(
            config,
            validated_features,
            design,
            primary_checkpoint_ready=checkpoint_ready,
        )
    except Exception as exc:
        failure = {
            "record_version": RECORD_VERSION,
            "profile_id": IMPULSE_OBSERVABILITY_PROFILE_ID,
            "impulse_observability_construction_decision": (
                "runtime_or_construction_validation_failure_stop"
            ),
            "impulse_observability_failure_reason": str(exc),
            "formal_result": False,
            "stage_progression_allowed": False,
            "cross_identity_confirmation_design_allowed": False,
            "gate_b_execution_allowed": False,
            "gate_c_execution_allowed": False,
            "observer_execution_allowed": False,
            "claim_support_status": (
                "failure_recovery_only_not_claim_evidence"
            ),
            "repository_commit": repository_commit,
            "config_digest": config_digest,
            "construction_feature_schema_digest": config[
                "construction_feature_schema"
            ]["feature_schema_digest"],
            "prompt_id": config["execution_identity"]["prompt_id"],
            "positive_prompt_text_sha256": config["execution_identity"][
                "positive_prompt_text_sha256"
            ],
            "seed_id": config["execution_identity"]["seed_id"],
            "generation_seed_random": config["execution_identity"][
                "seed_value"
            ],
            "construction_basis_digest": basis.basis_digest,
            "impulse_waveform_schema_digest": config[
                "flow_schedule_contract"
            ]["waveform_schema_digest"],
            "impulse_runtime_adapter_schema_digest": config[
                "runtime_adapter_contract"
            ]["adapter_schema_digest"],
            **runtime_versions,
        }
        write_json(
            output
            / "artifacts"
            / "output_feature_impulse_observability_decision.json",
            failure,
        )
        write_jsonl(
            output
            / "records"
            / "output_feature_impulse_observability_failure_records.jsonl",
            [failure],
        )
        raise

    actual_design_record = _actual_design_record(
        design,
        basis_digest=basis.basis_digest,
    )
    gate_ready = bool(stats.gate_a_ready)
    decision = {
        "record_version": RECORD_VERSION,
        "profile_id": IMPULSE_OBSERVABILITY_PROFILE_ID,
        "impulse_observability_construction_decision": (
            "sample_internal_causal_observability_gate_pass"
            if gate_ready
            else "sample_internal_causal_observability_gate_failed_stop"
        ),
        "impulse_sample_internal_observability_gate_ready": gate_ready,
        "cross_identity_confirmation_design_allowed": gate_ready,
        "gate_b_execution_allowed": False,
        "gate_c_execution_allowed": False,
        "key_selectivity_construction_allowed": False,
        "state_dynamics_design_allowed": False,
        "observer_execution_allowed": False,
        "attack_execution_allowed": False,
        "fixed_fpr_execution_allowed": False,
        "external_baseline_execution_allowed": False,
        "formal_result": False,
        "stage_progression_allowed": False,
        "generation_record_count": len(generation.generation_records),
        "trajectory_step_record_count": len(
            generation.trajectory_step_records
        ),
        "actual_exposure_trace_count": len(generation.exposure_traces),
        "checkpoint_record_count": len(ordered_checkpoints),
        "feature_record_count": len(feature_batch.feature_records),
        "gate_a_statistics": _gate_statistics_dict(stats),
        "claim_support_status": (
            "output_feature_impulse_observability_construction_only_"
            "not_method_evidence"
        ),
        "repository_commit": repository_commit,
        "config_digest": config_digest,
        "construction_feature_schema_digest": config[
            "construction_feature_schema"
        ]["feature_schema_digest"],
        "positive_prompt_text_sha256": config["execution_identity"][
            "positive_prompt_text_sha256"
        ],
        "prompt_id": config["execution_identity"]["prompt_id"],
        "seed_id": config["execution_identity"]["seed_id"],
        "generation_seed_random": config["execution_identity"][
            "seed_value"
        ],
        "construction_basis_digest": basis.basis_digest,
        "impulse_waveform_schema_digest": config[
            "flow_schedule_contract"
        ]["waveform_schema_digest"],
        "impulse_runtime_adapter_schema_digest": config[
            "runtime_adapter_contract"
        ]["adapter_schema_digest"],
        **runtime_versions,
    }
    write_jsonl(
        output / "records" / "impulse_checkpoint_records.jsonl",
        ordered_checkpoints,
    )
    write_jsonl(
        output / "records" / "impulse_feature_records.jsonl",
        feature_batch.feature_records,
    )
    write_json(
        output / "artifacts" / "impulse_actual_design_matrix.json",
        actual_design_record,
    )
    write_json(
        output
        / "artifacts"
        / "output_feature_impulse_observability_decision.json",
        decision,
    )
    generator_digests = sorted(
        {
            str(record["generation_generator_state_digest_random"])
            for record in generation.generation_records
        }
    )
    manifest = {
        "manifest_kind": (
            "output_feature_impulse_observability_construction_manifest"
        ),
        "profile_id": IMPULSE_OBSERVABILITY_PROFILE_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit": repository_commit,
        **runtime_versions,
        "config_path": str(Path(config_path)),
        "config_digest": config_digest,
        "construction_feature_schema_digest": config[
            "construction_feature_schema"
        ]["feature_schema_digest"],
        "construction_basis_digest": basis.basis_digest,
        "impulse_waveform_schema_digest": config[
            "flow_schedule_contract"
        ]["waveform_schema_digest"],
        "impulse_runtime_adapter_schema_digest": config[
            "runtime_adapter_contract"
        ]["adapter_schema_digest"],
        "prompt_suite_path": str(prompt_suite_path),
        "prompt_id": config["execution_identity"]["prompt_id"],
        "positive_prompt_text_sha256": config["execution_identity"][
            "positive_prompt_text_sha256"
        ],
        "seed_id": config["execution_identity"]["seed_id"],
        "generation_seed_random": config["execution_identity"][
            "seed_value"
        ],
        "generation_generator_state_digest_random": (
            generator_digests[0] if len(generator_digests) == 1 else None
        ),
        "same_initial_generator_state_verified": len(generator_digests) == 1,
        "video_records": [
            {
                "impulse_probe_id": record["impulse_probe_id"],
                "video_path": record["video_path"],
                "video_sha256": record["video_sha256"],
            }
            for record in generation.generation_records
        ],
        "output_paths": [
            "records/impulse_generation_plan.jsonl",
            "records/impulse_generation_records.jsonl",
            "records/impulse_trajectory_step_records.jsonl",
            "records/impulse_actual_exposure_traces.jsonl",
            "records/impulse_checkpoint_records.jsonl",
            "records/impulse_feature_records.jsonl",
            "artifacts/impulse_actual_design_matrix.json",
            "artifacts/output_feature_impulse_observability_decision.json",
        ],
        "formal_result": False,
        "stage_progression_allowed": False,
        "claim_support_status": (
            "output_feature_impulse_observability_construction_only_"
            "not_method_evidence"
        ),
    }
    write_json(
        output
        / "artifacts"
        / "output_feature_impulse_observability_manifest.json",
        manifest,
    )
    return decision


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run output-feature impulse observability Gate A triage"
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    print(
        json.dumps(
            run_output_feature_impulse_observability_construction(
                args.source_root,
                args.output_root,
                config_path=args.config_path,
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
