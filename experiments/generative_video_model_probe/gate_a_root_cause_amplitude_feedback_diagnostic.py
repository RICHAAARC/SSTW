"""Run the six-video post-Gate-A amplitude/feedback root-cause diagnostic.

The experiment is not a Gate A retry.  It applies one predeclared half
amplitude to early/late channel zero, validates the immutable 0.12 Gate A FAIL
as an apply-only baseline, and reports overlapping causal candidates.  It
cannot return Gate A PASS, stage progression, or method evidence.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from evaluation.protocol.gate_a_root_cause_amplitude_feedback_contract import (
    COMPARABLE_CHECKPOINT_IDS,
    DEFAULT_DIAGNOSTIC_CONFIG_PATH,
    DIAGNOSTIC_PLAN_IDS,
    DIAGNOSTIC_PROFILE_ID,
    DIAGNOSTIC_RECORD_VERSION,
    PAIR_IDS,
    HistoricalGateAFailureSource,
    PairedResponseStatistics,
    ScalingComparison,
    build_gate_a_root_cause_diagnostic_plan,
    classify_root_cause_candidates,
    compare_half_to_historical_full,
    compute_paired_response_statistics,
    compute_paired_response_statistics_from_gram,
    load_gate_a_root_cause_diagnostic_config,
    validate_historical_gate_a_failure_source,
)
from evaluation.protocol.impulse_observability_contract import (
    CONSTRUCTION_FLOW_STEP_COUNT,
    CONSTRUCTION_LATENT_LAYOUT_SHAPE,
    ConstructionStageBasis,
    ImpulseProbePlanRecord,
    _validate_trace_schedule_and_budget,
    _validate_trace_shape,
    build_construction_stage_basis,
    canonical_json_digest,
    construction_feature_row_binding_digest,
    load_impulse_observability_config,
)
from evaluation.protocol.record_writer import write_json, write_jsonl
from experiments.generative_video_model_probe.output_feature_impulse_observability_construction import (
    CHECKPOINT_SOURCE_STATUS,
    PRIMARY_CHECKPOINT_IDS,
    ConstructionFeatureBatch,
    ConstructionGenerationBatch,
    _package_version,
    _sha256_file,
    execute_real_impulse_generation,
    extract_real_output_features,
)


TEST_ID = "gate_a_root_cause_amplitude_feedback_diagnostic"
CLAIM_SUPPORT_STATUS = (
    "gate_a_failure_root_cause_diagnostic_only_not_method_evidence"
)
CAPTURE_ARRAY_CHUNK_SIZE = 1_048_576
PAIRED_RESPONSE_RECORD_FIELDS = frozenset(
    {
        "record_version",
        "profile_id",
        "root_cause_pair_id",
        "root_cause_checkpoint_id",
        "root_cause_clean_distance",
        "root_cause_positive_centered_norm",
        "root_cause_negative_centered_norm",
        "root_cause_odd_norm",
        "root_cause_common_norm",
        "root_cause_common_odd_ratio",
        "root_cause_antisymmetry_cosine",
        "root_cause_antisymmetry_residual",
        "root_cause_statistics_finite",
        "formal_result",
        "stage_progression_allowed",
        "claim_support_status",
    }
)
SCALING_COMPARISON_RECORD_FIELDS = frozenset(
    {
        "record_version",
        "profile_id",
        "root_cause_pair_id",
        "root_cause_checkpoint_id",
        "root_cause_actual_amplitude_ratio",
        "root_cause_actual_amplitude_ratio_ready",
        "root_cause_odd_ratio_to_full",
        "root_cause_common_ratio_to_full",
        "root_cause_normalized_odd_scaling",
        "root_cause_normalized_common_scaling",
        "root_cause_antisymmetry_cosine_improvement",
        "root_cause_antisymmetry_residual_improvement",
        "root_cause_local_linear_scaling_ready",
        "root_cause_comparison_status",
        "formal_result",
        "stage_progression_allowed",
        "claim_support_status",
    }
)


def _paired_response_record(
    value: PairedResponseStatistics,
) -> dict[str, Any]:
    record = {
        "record_version": DIAGNOSTIC_RECORD_VERSION,
        "profile_id": DIAGNOSTIC_PROFILE_ID,
        "root_cause_pair_id": value.pair_id,
        "root_cause_checkpoint_id": value.checkpoint_id,
        "root_cause_clean_distance": value.clean_distance,
        "root_cause_positive_centered_norm": (
            value.positive_centered_norm
        ),
        "root_cause_negative_centered_norm": (
            value.negative_centered_norm
        ),
        "root_cause_odd_norm": value.odd_norm,
        "root_cause_common_norm": value.common_norm,
        "root_cause_common_odd_ratio": value.common_odd_ratio,
        "root_cause_antisymmetry_cosine": (
            value.antisymmetry_cosine
        ),
        "root_cause_antisymmetry_residual": (
            value.antisymmetry_residual
        ),
        "root_cause_statistics_finite": value.finite,
        "formal_result": False,
        "stage_progression_allowed": False,
        "claim_support_status": CLAIM_SUPPORT_STATUS,
    }
    if set(record) != PAIRED_RESPONSE_RECORD_FIELDS:
        raise AssertionError("root-cause paired response record schema 漂移")
    return record


def _scaling_comparison_record(
    value: ScalingComparison,
) -> dict[str, Any]:
    record = {
        "record_version": DIAGNOSTIC_RECORD_VERSION,
        "profile_id": DIAGNOSTIC_PROFILE_ID,
        "root_cause_pair_id": value.pair_id,
        "root_cause_checkpoint_id": value.checkpoint_id,
        "root_cause_actual_amplitude_ratio": (
            value.actual_amplitude_ratio
        ),
        "root_cause_actual_amplitude_ratio_ready": (
            value.actual_amplitude_ratio_ready
        ),
        "root_cause_odd_ratio_to_full": value.odd_ratio_to_full,
        "root_cause_common_ratio_to_full": value.common_ratio_to_full,
        "root_cause_normalized_odd_scaling": (
            value.normalized_odd_scaling
        ),
        "root_cause_normalized_common_scaling": (
            value.normalized_common_scaling
        ),
        "root_cause_antisymmetry_cosine_improvement": (
            value.antisymmetry_cosine_improvement
        ),
        "root_cause_antisymmetry_residual_improvement": (
            value.antisymmetry_residual_improvement
        ),
        "root_cause_local_linear_scaling_ready": (
            value.local_linear_scaling_ready
        ),
        "root_cause_comparison_status": value.comparison_status,
        "formal_result": False,
        "stage_progression_allowed": False,
        "claim_support_status": CLAIM_SUPPORT_STATUS,
    }
    if set(record) != SCALING_COMPARISON_RECORD_FIELDS:
        raise AssertionError("root-cause scaling comparison record schema 漂移")
    return record


def _write_root_cause_metric_records(
    output_root: Path,
    *,
    half_statistics: Mapping[
        tuple[str, str],
        PairedResponseStatistics,
    ],
    historical_statistics: Mapping[
        tuple[str, str],
        PairedResponseStatistics,
    ],
    scaling: Mapping[tuple[str, str], ScalingComparison],
) -> None:
    write_jsonl(
        output_root
        / "records"
        / "root_cause_paired_response_statistics.jsonl",
        [
            _paired_response_record(value)
            for _, value in sorted(half_statistics.items())
        ],
    )
    write_jsonl(
        output_root
        / "records"
        / "root_cause_historical_response_statistics.jsonl",
        [
            _paired_response_record(value)
            for _, value in sorted(historical_statistics.items())
        ],
    )
    write_jsonl(
        output_root / "records" / "root_cause_scaling_comparisons.jsonl",
        [
            _scaling_comparison_record(value)
            for _, value in sorted(scaling.items())
        ],
    )


def _repository_commit() -> str:
    root = Path(__file__).resolve().parents[2]
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
    ).strip()


def _stable_record(payload: Mapping[str, Any], id_field: str) -> dict[str, Any]:
    record = dict(payload)
    record[id_field] = canonical_json_digest(record)
    return record


def _sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    return sha256(value.tobytes(order="C")).hexdigest()


def _diagnostic_runtime_config(
    diagnostic_config: Mapping[str, Any],
) -> dict[str, Any]:
    base = load_impulse_observability_config(
        diagnostic_config["base_construction_contract"]["config_path"]
    )
    runtime = json.loads(json.dumps(base))
    half = float(
        diagnostic_config["diagnostic_plan"]["diagnostic_lambda_max"]
    )
    runtime["actual_exposure_contract"]["lambda_max"] = half
    runtime["impulse_probe"]["nominal_signed_amplitude"] = half
    return runtime


def _prompt_and_seed(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = config["execution_identity"]
    return (
        {
            "prompt_id": identity["prompt_id"],
            "prompt_text": identity["prompt_text"],
            "prompt_negative_text": identity["negative_prompt_text"],
            "prompt_suite_id": identity["prompt_suite_id"],
        },
        {
            "seed_id": identity["seed_id"],
            "seed_value": identity["seed_value"],
        },
    )


class DiagnosticArrayCapture:
    """Persist small final latents and temporary pre-save frames on CPU."""

    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root
        self.latent_root = output_root / "artifacts" / "final_latents"
        self.working_root = output_root / ".diagnostic_working"
        self.latent_root.mkdir(parents=True, exist_ok=True)
        self.working_root.mkdir(parents=True, exist_ok=True)
        self.decoded_paths: dict[str, Path] = {}
        self.latent_paths: dict[str, Path] = {}

    def __call__(
        self,
        *,
        probe: ImpulseProbePlanRecord,
        plan_index: int,
        final_latent: Any,
        decoded_frames: Any,
        video_path: Path,
        video_sha256: str,
        basis: ConstructionStageBasis,
    ) -> Sequence[Mapping[str, Any]]:
        latent = (
            final_latent.detach().float().cpu().numpy()
            if hasattr(final_latent, "detach")
            else np.asarray(final_latent, dtype=np.float32)
        )
        latent = np.ascontiguousarray(latent, dtype=np.float32)
        if latent.shape != CONSTRUCTION_LATENT_LAYOUT_SHAPE:
            raise RuntimeError("diagnostic full final latent shape 未冻结")
        decoded = np.ascontiguousarray(decoded_frames, dtype=np.float32)
        if decoded.shape != (33, 320, 512, 3) or not np.all(
            np.isfinite(decoded)
        ):
            raise RuntimeError("diagnostic pre-save decoded array 未冻结")
        if float(np.min(decoded)) < 0.0 or float(np.max(decoded)) > 1.0:
            raise RuntimeError("diagnostic pre-save decoded range 非法")

        latent_path = (
            self.latent_root / f"{plan_index:02d}_{probe.probe_id}.npz"
        )
        np.savez_compressed(latent_path, final_latent=latent)
        decoded_path = (
            self.working_root
            / f"{plan_index:02d}_{probe.probe_id}_decoded.npy"
        )
        np.save(decoded_path, decoded, allow_pickle=False)
        self.latent_paths[probe.probe_id] = latent_path
        self.decoded_paths[probe.probe_id] = decoded_path
        record = {
            "record_version": DIAGNOSTIC_RECORD_VERSION,
            "profile_id": DIAGNOSTIC_PROFILE_ID,
            "impulse_probe_id": probe.probe_id,
            "impulse_probe_plan_index": plan_index,
            "final_latent_artifact_path": str(latent_path),
            "final_latent_artifact_sha256": _sha256_file(latent_path),
            "final_latent_array_sha256": _sha256_array(latent),
            "final_latent_shape": list(latent.shape),
            "final_latent_dtype": str(latent.dtype),
            "pre_save_decoded_array_sha256": _sha256_array(decoded),
            "pre_save_decoded_shape": list(decoded.shape),
            "pre_save_decoded_dtype": str(decoded.dtype),
            "video_path": str(video_path),
            "video_sha256": video_sha256,
            "construction_basis_digest": basis.basis_digest,
            "temporary_pre_save_array_retention": (
                "local_only_until_gram_statistics_complete"
            ),
            "formal_result": False,
            "stage_progression_allowed": False,
            "claim_support_status": CLAIM_SUPPORT_STATUS,
        }
        return (
            _stable_record(
                record,
                "root_cause_array_capture_record_id",
            ),
        )

    def remove_temporary_decoded_arrays(self) -> None:
        for path in self.decoded_paths.values():
            path.unlink(missing_ok=True)
        if self.working_root.exists() and not any(
            self.working_root.iterdir()
        ):
            self.working_root.rmdir()


def _load_latent_artifact(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        value = np.asarray(archive["final_latent"], dtype=np.float32)
    if value.shape != CONSTRUCTION_LATENT_LAYOUT_SHAPE:
        raise ValueError("diagnostic latent artifact shape 漂移")
    return value


def _read_saved_rgb24(video_path: Path) -> np.ndarray:
    import imageio.v3 as iio

    frames = [np.asarray(frame) for frame in iio.imiter(video_path)]
    if not frames:
        raise RuntimeError(f"saved video 没有可回读帧: {video_path}")
    value = np.ascontiguousarray(np.stack(frames, axis=0))
    if value.shape != (33, 320, 512, 3) or value.dtype != np.uint8:
        raise RuntimeError("saved video full RGB24 shape/dtype 未冻结")
    return value


def _materialize_saved_rgb24_arrays(
    *,
    video_paths: Mapping[str, Path],
    working_root: Path,
) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    working_root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    provenance: dict[str, dict[str, Any]] = {}
    for plan_index, probe_id in enumerate(DIAGNOSTIC_PLAN_IDS):
        array = _read_saved_rgb24(video_paths[probe_id])
        path = working_root / f"{plan_index:02d}_{probe_id}_saved_rgb24.npy"
        np.save(path, array, allow_pickle=False)
        paths[probe_id] = path
        provenance[probe_id] = {
            "array_sha256": _sha256_array(array),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "video_sha256": _sha256_file(video_paths[probe_id]),
        }
        del array
    return paths, provenance


def _gram_matrix_from_array_paths(
    paths: Mapping[str, Path],
    *,
    loader: Callable[[Path], np.ndarray],
) -> np.ndarray:
    if tuple(paths) != DIAGNOSTIC_PLAN_IDS:
        raise ValueError("full-representation Gram row identity/order 不一致")
    gram = np.zeros(
        (len(DIAGNOSTIC_PLAN_IDS), len(DIAGNOSTIC_PLAN_IDS)),
        dtype=np.float64,
    )
    for left_index, left_id in enumerate(DIAGNOSTIC_PLAN_IDS):
        left = np.asarray(loader(paths[left_id])).reshape(-1)
        if not np.all(np.isfinite(left)):
            raise ValueError(f"{left_id} full representation 含非有限值")
        for right_index in range(left_index, len(DIAGNOSTIC_PLAN_IDS)):
            right_id = DIAGNOSTIC_PLAN_IDS[right_index]
            right = np.asarray(loader(paths[right_id])).reshape(-1)
            if left.shape != right.shape:
                raise ValueError("full-representation flattened shape 不一致")
            total = 0.0
            for start in range(0, left.size, CAPTURE_ARRAY_CHUNK_SIZE):
                stop = min(start + CAPTURE_ARRAY_CHUNK_SIZE, left.size)
                total += float(
                    np.dot(
                        left[start:stop].astype(np.float64, copy=False),
                        right[start:stop].astype(np.float64, copy=False),
                    )
                )
            gram[left_index, right_index] = total
            gram[right_index, left_index] = total
            del right
        del left
    return gram


def _load_npy_memmap(path: Path) -> np.ndarray:
    return np.load(path, mmap_mode="r", allow_pickle=False)


def _relabel_record(
    record: Mapping[str, Any],
    *,
    id_field: str | None = None,
) -> dict[str, Any]:
    value = {
        **record,
        "record_version": DIAGNOSTIC_RECORD_VERSION,
        "profile_id": DIAGNOSTIC_PROFILE_ID,
        "formal_result": False,
        "stage_progression_allowed": False,
        "claim_support_status": CLAIM_SUPPORT_STATUS,
    }
    if id_field is not None:
        value.pop(id_field, None)
        value[id_field] = canonical_json_digest(value)
    return value


def _validate_diagnostic_generation_batch(
    runtime_config: Mapping[str, Any],
    diagnostic_config: Mapping[str, Any],
    plan: Sequence[ImpulseProbePlanRecord],
    batch: ConstructionGenerationBatch,
    *,
    historical: HistoricalGateAFailureSource,
    basis: ConstructionStageBasis,
) -> dict[str, float]:
    expected_ids = tuple(record.probe_id for record in plan)
    identity = diagnostic_config["execution_identity"]
    observed_ids = tuple(
        str(record.get("impulse_probe_id") or "")
        for record in batch.generation_records
    )
    generator_digests = {
        str(record.get("generation_generator_state_digest_random") or "")
        for record in batch.generation_records
    }
    video_paths = tuple(
        str(record.get("video_path") or "")
        for record in batch.generation_records
    )
    if (
        expected_ids != DIAGNOSTIC_PLAN_IDS
        or observed_ids != expected_ids
        or len(batch.generation_records) != 6
        or len(batch.trajectory_step_records) != 48
        or len(batch.exposure_traces) != 4
        or len(batch.checkpoint_records) != 18
        or len(batch.diagnostic_capture_records) != 6
        or len(generator_digests) != 1
        or generator_digests == {""}
        or len(set(video_paths)) != 6
    ):
        raise RuntimeError("6-video diagnostic generation coverage 未就绪")
    for plan_index, (probe, record) in enumerate(
        zip(plan, batch.generation_records, strict=True)
    ):
        expected_prompt_digest = diagnostic_config["execution_identity"][
            "prompt_text_sha256"
        ]
        if (
            record.get("generation_status") != "success"
            or record.get("impulse_probe_plan_index") != plan_index
            or record.get("impulse_probe_role") != probe.probe_role
            or record.get("impulse_stage_index") != probe.stage_index
            or record.get("impulse_state_channel_index")
            != probe.channel_index
            or record.get("impulse_polarity") != probe.polarity
            or record.get("impulse_nominal_signed_amplitude")
            != probe.nominal_signed_amplitude
            or record.get("generation_model_id")
            != identity["generation_model_id"]
            or record.get("generation_model_revision")
            != identity["generation_model_revision"]
            or record.get("scheduler_signature")
            != identity["scheduler_signature"]
            or record.get("prompt_id") != identity["prompt_id"]
            or record.get("positive_prompt_text_sha256")
            != expected_prompt_digest
            or record.get("negative_prompt_text_sha256")
            != identity["negative_prompt_text_sha256"]
            or record.get("seed_id") != identity["seed_id"]
            or record.get("generation_seed_random") != identity["seed_value"]
            or record.get("trajectory_step_count") != 8
            or record.get("endpoint_control_enabled") is not False
            or not Path(str(record.get("video_path") or "")).is_file()
            or _sha256_file(Path(str(record["video_path"])))
            != record.get("video_sha256")
        ):
            raise RuntimeError(f"{probe.probe_id} generation identity 未冻结")
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
    if observed_step_sequence != expected_step_sequence:
        raise RuntimeError("diagnostic step identity/order 不一致")
    clean_ids = {"clean_start", "clean_end"}
    for record in batch.trajectory_step_records:
        if record["impulse_probe_id"] in clean_ids and (
            record.get("impulse_inactive_exact_noop") is not True
            or float(record.get("impulse_actual_delta_norm") or 0.0) != 0.0
            or any(
                float(value) != 0.0
                for value in record.get(
                    "impulse_actual_channel_exposure",
                    (),
                )
            )
        ):
            raise RuntimeError("diagnostic clean step 不是 exact no-op")
    expected_trace_ids = tuple(
        record.probe_id for record in plan if record.polarity != 0
    )
    if (
        tuple(trace.probe_id for trace in batch.exposure_traces)
        != expected_trace_ids
        or basis.basis_digest
        != historical.decision["construction_basis_digest"]
    ):
        raise RuntimeError("diagnostic exposure/basis 未绑定 historical source")
    trace_map = {trace.probe_id: trace for trace in batch.exposure_traces}
    actual_amplitudes: dict[str, float] = {}
    for trace in batch.exposure_traces:
        _validate_trace_shape(trace)
        _validate_trace_schedule_and_budget(runtime_config, trace)
        if trace.basis_digest != basis.basis_digest:
            raise RuntimeError(f"{trace.probe_id} basis digest 不一致")
    for pair_id in PAIR_IDS:
        positive = trace_map[f"positive_{pair_id}"]
        negative = trace_map[f"negative_{pair_id}"]
        coordinate = positive.stage_index * 2 + positive.channel_index
        plus = float(positive.actual_exposure_vector[coordinate])
        minus = float(negative.actual_exposure_vector[coordinate])
        if plus <= 0.0 or minus >= 0.0:
            raise RuntimeError(f"{pair_id} half-amplitude exposure polarity 错误")
        actual_amplitudes[pair_id] = 0.5 * (abs(plus) + abs(minus))
    return actual_amplitudes


def _ordered_checkpoint_records(
    plan: Sequence[ImpulseProbePlanRecord],
    generation: ConstructionGenerationBatch,
    features: ConstructionFeatureBatch,
) -> tuple[dict[str, Any], ...]:
    generation_map = {
        (
            str(record["impulse_probe_id"]),
            str(record["impulse_transfer_checkpoint_id"]),
        ): record
        for record in generation.checkpoint_records
    }
    feature_map = {
        (
            str(record["impulse_probe_id"]),
            str(record["impulse_transfer_checkpoint_id"]),
        ): record
        for record in features.checkpoint_records
    }
    ordered: list[dict[str, Any]] = []
    for probe in plan:
        for checkpoint_id in PRIMARY_CHECKPOINT_IDS:
            source = (
                generation_map
                if checkpoint_id in {"T_latent", "T_decoded", "T_saved_video"}
                else feature_map
            )
            ordered.append(
                _relabel_record(
                    source[(probe.probe_id, checkpoint_id)],
                    id_field="impulse_transfer_checkpoint_record_id",
                )
            )
    return tuple(ordered)


def _validate_diagnostic_checkpoint_records(
    runtime_config: Mapping[str, Any],
    plan: Sequence[ImpulseProbePlanRecord],
    checkpoints: Sequence[Mapping[str, Any]],
    feature_records: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], np.ndarray]:
    expected_ids = tuple(record.probe_id for record in plan)
    expected_sequence = tuple(
        (probe_id, checkpoint_id)
        for probe_id in expected_ids
        for checkpoint_id in PRIMARY_CHECKPOINT_IDS
    )
    observed_sequence = tuple(
        (
            str(record.get("impulse_probe_id") or ""),
            str(record.get("impulse_transfer_checkpoint_id") or ""),
        )
        for record in checkpoints
    )
    if observed_sequence != expected_sequence or len(checkpoints) != 30:
        raise RuntimeError("diagnostic five-checkpoint identity/order 未就绪")
    result: dict[tuple[str, str], np.ndarray] = {}
    contracts = runtime_config["transfer_checkpoint_contract"]
    for plan_index, probe_id in enumerate(expected_ids):
        for offset, checkpoint_id in enumerate(PRIMARY_CHECKPOINT_IDS):
            record = checkpoints[plan_index * 5 + offset]
            values = np.asarray(
                record.get("impulse_transfer_checkpoint_values"),
                dtype=np.float64,
            )
            expected_id = canonical_json_digest(
                {
                    key: value
                    for key, value in record.items()
                    if key != "impulse_transfer_checkpoint_record_id"
                }
            )
            if (
                record.get("impulse_probe_plan_index") != plan_index
                or values.shape
                != (int(contracts[checkpoint_id]["dimension"]),)
                or not np.all(np.isfinite(values))
                or record.get("impulse_transfer_checkpoint_source_status")
                != CHECKPOINT_SOURCE_STATUS[checkpoint_id]
                or record.get("impulse_transfer_checkpoint_record_id")
                != expected_id
            ):
                raise RuntimeError(
                    f"{probe_id}/{checkpoint_id} diagnostic checkpoint 非法"
                )
            result[(probe_id, checkpoint_id)] = values
    if (
        len(feature_records) != 6
        or tuple(
            str(record.get("impulse_probe_id") or "")
            for record in feature_records
        )
        != expected_ids
    ):
        raise RuntimeError("diagnostic feature identity/order 未就绪")
    schema_digest = runtime_config["construction_feature_schema"][
        "feature_schema_digest"
    ]
    tolerance = float(
        runtime_config["construction_feature_schema"][
            "output_normalization_norm_tolerance"
        ]
    )
    for plan_index, record in enumerate(feature_records):
        values = np.asarray(
            record.get("construction_feature_values"),
            dtype=np.float64,
        )
        expected_binding = construction_feature_row_binding_digest(
            probe_id=expected_ids[plan_index],
            feature_schema_digest=schema_digest,
            feature_values=values,
        )
        expected_id = canonical_json_digest(
            {
                key: value
                for key, value in record.items()
                if key != "construction_feature_record_id"
            }
        )
        if (
            values.shape != (256,)
            or not np.all(np.isfinite(values))
            or not math.isclose(
                float(np.linalg.norm(values)),
                1.0,
                rel_tol=0.0,
                abs_tol=tolerance,
            )
            or record.get("impulse_probe_plan_index") != plan_index
            or record.get("construction_feature_schema_digest")
            != schema_digest
            or record.get("construction_feature_row_binding_digest")
            != expected_binding
            or record.get("construction_feature_record_id") != expected_id
            or not np.array_equal(
                values,
                result[(expected_ids[plan_index], "T_output_feature")],
            )
        ):
            raise RuntimeError(
                f"{expected_ids[plan_index]} diagnostic feature 非法"
            )
    return result


def _checkpoint_view(
    checkpoint_values: Mapping[tuple[str, str], np.ndarray],
    *,
    historical: bool,
) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    source_ids = (
        (
            "clean_a",
            "positive_early_flow_channel_0",
            "negative_early_flow_channel_0",
            "positive_late_flow_channel_0",
            "negative_late_flow_channel_0",
            "clean_b",
        )
        if historical
        else DIAGNOSTIC_PLAN_IDS
    )
    output_ids = DIAGNOSTIC_PLAN_IDS if historical else source_ids
    mapping = dict(zip(output_ids, source_ids, strict=True))
    checkpoint_mapping = {
        "T_latent_six_basis": "T_latent",
        "T_decoded_48d": "T_decoded",
        "T_saved_video_48d": "T_saved_video",
        "T_reencoded_256d": "T_reencoded",
        "T_output_feature_256d": "T_output_feature",
    }
    for public_id, internal_id in checkpoint_mapping.items():
        rows: dict[str, np.ndarray] = {}
        for output_id, source_id in mapping.items():
            value = checkpoint_values[(source_id, internal_id)]
            rows[output_id] = (
                value[:6].copy()
                if internal_id == "T_latent"
                else value.copy()
            )
        result[public_id] = rows
    return result


def _pair_statistics_from_views(
    config: Mapping[str, Any],
    views: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[tuple[str, str], PairedResponseStatistics]:
    epsilon = float(config["checkpoint_contract"]["norm_denominator_epsilon"])
    result: dict[tuple[str, str], PairedResponseStatistics] = {}
    for checkpoint_id, rows in views.items():
        if tuple(rows) != DIAGNOSTIC_PLAN_IDS:
            raise RuntimeError(f"{checkpoint_id} row identity/order 不一致")
        for pair_id in PAIR_IDS:
            result[(pair_id, checkpoint_id)] = (
                compute_paired_response_statistics(
                    pair_id=pair_id,
                    checkpoint_id=checkpoint_id,
                    clean_start=rows["clean_start"],
                    clean_end=rows["clean_end"],
                    positive=rows[f"positive_{pair_id}"],
                    negative=rows[f"negative_{pair_id}"],
                    denominator_epsilon=epsilon,
                )
            )
    return result


def _historical_saved_video_paths(
    historical: HistoricalGateAFailureSource,
) -> dict[str, Path]:
    return {
        "clean_start": historical.video_paths["clean_a"],
        "clean_end": historical.video_paths["clean_b"],
        "positive_early_flow_channel_0": historical.video_paths[
            "positive_early_flow_channel_0"
        ],
        "negative_early_flow_channel_0": historical.video_paths[
            "negative_early_flow_channel_0"
        ],
        "positive_late_flow_channel_0": historical.video_paths[
            "positive_late_flow_channel_0"
        ],
        "negative_late_flow_channel_0": historical.video_paths[
            "negative_late_flow_channel_0"
        ],
    }


def _full_representation_gram_record(
    *,
    checkpoint_id: str,
    gram: np.ndarray,
    source_provenance: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    record = {
        "record_version": DIAGNOSTIC_RECORD_VERSION,
        "profile_id": DIAGNOSTIC_PROFILE_ID,
        "checkpoint_id": checkpoint_id,
        "row_probe_ids": list(DIAGNOSTIC_PLAN_IDS),
        "gram_matrix": gram.tolist(),
        "source_provenance": dict(source_provenance),
        "gram_accumulator_dtype": "float64",
        "gram_chunk_size": CAPTURE_ARRAY_CHUNK_SIZE,
        "formal_result": False,
        "stage_progression_allowed": False,
        "claim_support_status": CLAIM_SUPPORT_STATUS,
    }
    return _stable_record(record, "root_cause_full_gram_record_id")


def _apply_clean_pollution_fail_closed(
    classification: Mapping[str, Any],
    *,
    contaminated: bool,
) -> dict[str, Any]:
    if not contaminated:
        return {
            **classification,
            "clean_intercept_and_pair_classification_valid": True,
            "contamination_reason": "",
        }
    return {
        **classification,
        "candidate_classifications": [
            "generation_order_state_pollution_candidate",
            "indeterminate",
        ],
        "local_linear_support_candidate": False,
        "quantization_or_observation_floor_candidate": False,
        "feedback_nonlinearity_candidate": False,
        "carrier_decoder_feature_mismatch_candidate": False,
        "classification_status": "contaminated_indeterminate",
        "clean_intercept_and_pair_classification_valid": False,
        "contamination_reason": (
            "clean_start_end_mismatch_confounds_ordered_pair_"
            "odd_common_and_scaling"
        ),
    }


def run_gate_a_root_cause_amplitude_feedback_diagnostic(
    source_root: str | Path,
    output_root: str | Path,
    *,
    config_path: str | Path = DEFAULT_DIAGNOSTIC_CONFIG_PATH,
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
    """Run the authorized six-video diagnostic and preserve Gate A FAIL."""

    source = Path(source_root).resolve()
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = load_gate_a_root_cause_diagnostic_config(config_path)
    runtime_config = _diagnostic_runtime_config(config)
    plan = build_gate_a_root_cause_diagnostic_plan(config)
    prompt, seed = _prompt_and_seed(config)
    capture = DiagnosticArrayCapture(output)
    repository_commit = _repository_commit()
    config_digest = canonical_json_digest(config)
    runtime_versions = {
        "python_version": sys.version.split()[0],
        "torch_version": _package_version("torch"),
        "diffusers_version": _package_version("diffusers"),
    }
    write_jsonl(
        output / "records" / "root_cause_generation_plan.jsonl",
        [
            {
                **asdict(record),
                "record_version": DIAGNOSTIC_RECORD_VERSION,
                "profile_id": DIAGNOSTIC_PROFILE_ID,
                "claim_support_status": CLAIM_SUPPORT_STATUS,
            }
            for record in plan
        ],
    )

    historical: HistoricalGateAFailureSource | None = None
    basis: ConstructionStageBasis | None = None
    try:
        historical = validate_historical_gate_a_failure_source(source, config)
        master_key = (
            os.environ.get("SSTW_TRAJECTORY_AUTHENTICATION_KEY") or ""
        )
        minimum_key_bytes = int(
            runtime_config["construction_basis"][
                "master_key_minimum_utf8_bytes"
            ]
        )
        if len(master_key.encode("utf-8")) < minimum_key_bytes:
            raise RuntimeError(
                "root-cause diagnostic owner master key 缺失或过短"
            )
        basis = basis_builder(master_key)
        if basis.basis_digest != historical.decision[
            "construction_basis_digest"
        ]:
            raise RuntimeError(
                "当前 owner key/basis 与 historical Gate A 不一致"
            )
        generation = generation_executor(
            runtime_config,
            prompt=prompt,
            seed=seed,
            output_root=output,
            plan=plan,
            basis=basis,
            diagnostic_capture=capture,
        )
        actual_amplitudes = _validate_diagnostic_generation_batch(
            runtime_config,
            config,
            plan,
            generation,
            historical=historical,
            basis=basis,
        )
        features = feature_executor(
            runtime_config,
            output_root=output,
            plan=plan,
            generation_records=generation.generation_records,
        )
        feature_records = tuple(
            _relabel_record(
                record,
                id_field="construction_feature_record_id",
            )
            for record in features.feature_records
        )
        checkpoints = _ordered_checkpoint_records(plan, generation, features)
        checkpoint_values = _validate_diagnostic_checkpoint_records(
            runtime_config,
            plan,
            checkpoints,
            feature_records,
        )
        half_views = _checkpoint_view(
            checkpoint_values,
            historical=False,
        )
        full_views = _checkpoint_view(
            historical.checkpoint_values,
            historical=True,
        )
        half_statistics = _pair_statistics_from_views(config, half_views)
        full_statistics = _pair_statistics_from_views(config, full_views)
        half_l2_rows = {
            probe_id: checkpoint_values[(probe_id, "T_latent")][6:7]
            for probe_id in DIAGNOSTIC_PLAN_IDS
        }
        historical_l2_source = {
            "clean_start": "clean_a",
            "positive_early_flow_channel_0": (
                "positive_early_flow_channel_0"
            ),
            "negative_early_flow_channel_0": (
                "negative_early_flow_channel_0"
            ),
            "positive_late_flow_channel_0": (
                "positive_late_flow_channel_0"
            ),
            "negative_late_flow_channel_0": (
                "negative_late_flow_channel_0"
            ),
            "clean_end": "clean_b",
        }
        historical_l2_rows = {
            output_id: historical.checkpoint_values[
                (source_id, "T_latent")
            ][6:7]
            for output_id, source_id in historical_l2_source.items()
        }
        epsilon = float(
            config["checkpoint_contract"]["norm_denominator_epsilon"]
        )
        for pair_id in PAIR_IDS:
            half_statistics[
                (pair_id, "T_latent_l2_norm")
            ] = compute_paired_response_statistics(
                pair_id=pair_id,
                checkpoint_id="T_latent_l2_norm",
                clean_start=half_l2_rows["clean_start"],
                clean_end=half_l2_rows["clean_end"],
                positive=half_l2_rows[f"positive_{pair_id}"],
                negative=half_l2_rows[f"negative_{pair_id}"],
                denominator_epsilon=epsilon,
            )
            full_statistics[
                (pair_id, "T_latent_l2_norm")
            ] = compute_paired_response_statistics(
                pair_id=pair_id,
                checkpoint_id="T_latent_l2_norm",
                clean_start=historical_l2_rows["clean_start"],
                clean_end=historical_l2_rows["clean_end"],
                positive=historical_l2_rows[f"positive_{pair_id}"],
                negative=historical_l2_rows[f"negative_{pair_id}"],
                denominator_epsilon=epsilon,
            )

        full_gram_records: list[dict[str, Any]] = []
        latent_gram = _gram_matrix_from_array_paths(
            capture.latent_paths,
            loader=_load_latent_artifact,
        )
        latent_provenance = {
            probe_id: {
                "artifact_path": str(path),
                "artifact_sha256": _sha256_file(path),
            }
            for probe_id, path in capture.latent_paths.items()
        }
        full_gram_records.append(
            _full_representation_gram_record(
                checkpoint_id="T_final_latent_full",
                gram=latent_gram,
                source_provenance=latent_provenance,
            )
        )
        decoded_gram = _gram_matrix_from_array_paths(
            capture.decoded_paths,
            loader=_load_npy_memmap,
        )
        decoded_provenance = {
            str(record["impulse_probe_id"]): {
                "array_sha256": record["pre_save_decoded_array_sha256"],
                "shape": record["pre_save_decoded_shape"],
                "dtype": record["pre_save_decoded_dtype"],
            }
            for record in generation.diagnostic_capture_records
        }
        full_gram_records.append(
            _full_representation_gram_record(
                checkpoint_id="T_decoded_full_rgb_float32",
                gram=decoded_gram,
                source_provenance=decoded_provenance,
            )
        )
        half_video_paths = {
            str(record["impulse_probe_id"]): Path(str(record["video_path"]))
            for record in generation.generation_records
        }
        half_saved_paths, half_saved_provenance = (
            _materialize_saved_rgb24_arrays(
                video_paths=half_video_paths,
                working_root=output / ".diagnostic_working" / "half_saved",
            )
        )
        half_saved_gram = _gram_matrix_from_array_paths(
            half_saved_paths,
            loader=_load_npy_memmap,
        )
        full_gram_records.append(
            _full_representation_gram_record(
                checkpoint_id="T_saved_video_full_rgb24",
                gram=half_saved_gram,
                source_provenance=half_saved_provenance,
            )
        )
        historical_saved_paths, historical_saved_provenance = (
            _materialize_saved_rgb24_arrays(
                video_paths=_historical_saved_video_paths(historical),
                working_root=(
                    output
                    / ".diagnostic_working"
                    / "historical_saved"
                ),
            )
        )
        historical_saved_gram = _gram_matrix_from_array_paths(
            historical_saved_paths,
            loader=_load_npy_memmap,
        )
        full_gram_records.append(
            _full_representation_gram_record(
                checkpoint_id=(
                    "T_historical_saved_video_full_rgb24_apply_only"
                ),
                gram=historical_saved_gram,
                source_provenance=historical_saved_provenance,
            )
        )
        for pair_id in PAIR_IDS:
            half_statistics[
                (pair_id, "T_final_latent_full")
            ] = compute_paired_response_statistics_from_gram(
                pair_id=pair_id,
                checkpoint_id="T_final_latent_full",
                gram_matrix=latent_gram,
                row_ids=DIAGNOSTIC_PLAN_IDS,
                denominator_epsilon=epsilon,
            )
            half_statistics[
                (pair_id, "T_decoded_full_rgb_float32")
            ] = compute_paired_response_statistics_from_gram(
                pair_id=pair_id,
                checkpoint_id="T_decoded_full_rgb_float32",
                gram_matrix=decoded_gram,
                row_ids=DIAGNOSTIC_PLAN_IDS,
                denominator_epsilon=epsilon,
            )
            half_statistics[
                (pair_id, "T_saved_video_full_rgb24")
            ] = compute_paired_response_statistics_from_gram(
                pair_id=pair_id,
                checkpoint_id="T_saved_video_full_rgb24",
                gram_matrix=half_saved_gram,
                row_ids=DIAGNOSTIC_PLAN_IDS,
                denominator_epsilon=epsilon,
            )
            full_statistics[
                (pair_id, "T_saved_video_full_rgb24")
            ] = compute_paired_response_statistics_from_gram(
                pair_id=pair_id,
                checkpoint_id="T_saved_video_full_rgb24",
                gram_matrix=historical_saved_gram,
                row_ids=DIAGNOSTIC_PLAN_IDS,
                denominator_epsilon=epsilon,
            )
        scaling: dict[tuple[str, str], ScalingComparison] = {}
        for pair_id in PAIR_IDS:
            for checkpoint_id in COMPARABLE_CHECKPOINT_IDS:
                scaling[(pair_id, checkpoint_id)] = (
                    compare_half_to_historical_full(
                        config,
                        half=half_statistics[(pair_id, checkpoint_id)],
                        full=full_statistics[(pair_id, checkpoint_id)],
                        half_actual_amplitude=actual_amplitudes[pair_id],
                        full_actual_amplitude=(
                            historical.actual_exposure_by_pair[pair_id]
                        ),
                    )
                )
        classification = classify_root_cause_candidates(
            config,
            half_statistics=half_statistics,
            scaling=scaling,
            half_actual_amplitude_by_pair=actual_amplitudes,
        )
        clean_start_generation = generation.generation_records[0]
        clean_end_generation = generation.generation_records[-1]
        clean_state = {
            "clean_start_end_video_sha256_equal": (
                clean_start_generation["video_sha256"]
                == clean_end_generation["video_sha256"]
            ),
            "clean_start_end_final_latent_array_sha256_equal": (
                generation.diagnostic_capture_records[0][
                    "final_latent_array_sha256"
                ]
                == generation.diagnostic_capture_records[-1][
                    "final_latent_array_sha256"
                ]
            ),
            "clean_start_end_pre_save_decoded_array_sha256_equal": (
                generation.diagnostic_capture_records[0][
                    "pre_save_decoded_array_sha256"
                ]
                == generation.diagnostic_capture_records[-1][
                    "pre_save_decoded_array_sha256"
                ]
            ),
            "clean_start_end_reencoded_checkpoint_equal": np.array_equal(
                checkpoint_values[("clean_start", "T_reencoded")],
                checkpoint_values[("clean_end", "T_reencoded")],
            ),
            "clean_start_end_output_feature_equal": np.array_equal(
                checkpoint_values[("clean_start", "T_output_feature")],
                checkpoint_values[("clean_end", "T_output_feature")],
            ),
            "clean_repeat_role": (
                "order_drift_and_state_pollution_check_only"
            ),
            "clean_repeat_formal_noise_distribution": False,
        }
        clean_state["generation_order_state_pollution_candidate"] = not all(
            value
            for key, value in clean_state.items()
            if key.endswith("_equal")
        )
        classification = _apply_clean_pollution_fail_closed(
            classification,
            contaminated=clean_state[
                "generation_order_state_pollution_candidate"
            ],
        )
        capture.remove_temporary_decoded_arrays()
        for path in tuple(half_saved_paths.values()) + tuple(
            historical_saved_paths.values()
        ):
            path.unlink(missing_ok=True)
        working_root = output / ".diagnostic_working"
        if working_root.exists():
            shutil.rmtree(working_root)
    except Exception as exc:
        # Keep compact governed records/final-latent artifacts for recovery,
        # but never upload large temporary decoded/RGB arrays to Drive.
        capture.remove_temporary_decoded_arrays()
        working_root = output / ".diagnostic_working"
        if working_root.exists():
            shutil.rmtree(working_root)
        failure = {
            "record_version": DIAGNOSTIC_RECORD_VERSION,
            "profile_id": DIAGNOSTIC_PROFILE_ID,
            "gate_a_root_cause_diagnostic_decision": (
                "runtime_or_input_validation_failure_stop"
            ),
            "gate_a_root_cause_diagnostic_failure_reason": str(exc),
            "historical_gate_a_decision_preserved": True,
            "gate_a_retry": False,
            "gate_a_pass": False,
            "cross_identity_confirmation_allowed": False,
            "formal_result": False,
            "stage_progression_allowed": False,
            "gate_b_execution_allowed": False,
            "gate_c_execution_allowed": False,
            "wrong_key_execution_allowed": False,
            "observer_execution_allowed": False,
            "state_dynamics_design_allowed": False,
            "attack_execution_allowed": False,
            "pilot_execution_allowed": False,
            "fixed_fpr_execution_allowed": False,
            "external_baseline_execution_allowed": False,
            "paper_claim_allowed": False,
            "frozen_feedback_diagnostic_design_allowed": False,
            "carrier_feature_redesign_allowed": False,
            "automatic_followup_execution_allowed": False,
            "claim_support_status": (
                "failure_recovery_only_not_claim_evidence"
            ),
            "repository_commit": repository_commit,
            "config_digest": config_digest,
            "historical_source_snapshot_digest": (
                ""
                if historical is None
                else historical.source_snapshot_digest
            ),
            **runtime_versions,
        }
        write_json(
            output
            / "artifacts"
            / "gate_a_root_cause_amplitude_feedback_decision.json",
            failure,
        )
        write_jsonl(
            output
            / "records"
            / "gate_a_root_cause_amplitude_feedback_failure_records.jsonl",
            [failure],
        )
        raise

    if historical is None or basis is None:
        raise AssertionError("root-cause diagnostic validated inputs 丢失")
    generation_records = tuple(
        _relabel_record(
            record,
            id_field="impulse_generation_record_id",
        )
        for record in generation.generation_records
    )
    step_records = tuple(
        _relabel_record(record)
        for record in generation.trajectory_step_records
    )
    capture_records = tuple(
        _relabel_record(
            record,
            id_field="root_cause_array_capture_record_id",
        )
        for record in generation.diagnostic_capture_records
    )
    write_jsonl(
        output / "records" / "impulse_generation_records.jsonl",
        generation_records,
    )
    write_jsonl(
        output / "records" / "impulse_trajectory_step_records.jsonl",
        step_records,
    )
    write_jsonl(
        output / "records" / "impulse_actual_exposure_traces.jsonl",
        [asdict(trace) for trace in generation.exposure_traces],
    )
    write_jsonl(
        output / "records" / "impulse_checkpoint_records.jsonl",
        checkpoints,
    )
    write_jsonl(
        output / "records" / "impulse_feature_records.jsonl",
        feature_records,
    )
    write_jsonl(
        output / "records" / "root_cause_array_capture_records.jsonl",
        capture_records,
    )
    write_jsonl(
        output / "records" / "root_cause_full_gram_records.jsonl",
        full_gram_records,
    )
    _write_root_cause_metric_records(
        output,
        half_statistics=half_statistics,
        historical_statistics=full_statistics,
        scaling=scaling,
    )
    historical_validation = {
        "record_version": DIAGNOSTIC_RECORD_VERSION,
        "profile_id": DIAGNOSTIC_PROFILE_ID,
        "historical_gate_a_source_status": "ready_complete_fail_result",
        "historical_source_snapshot_digest": (
            historical.source_snapshot_digest
        ),
        "historical_repository_commit": historical.decision[
            "repository_commit"
        ],
        "historical_gate_a_decision": historical.decision[
            "impulse_observability_construction_decision"
        ],
        "historical_gate_a_pass": False,
        "historical_formal_result": False,
        "historical_stage_progression_allowed": False,
        "historical_full_final_latent_comparison_available": False,
        "historical_full_pre_save_decoded_comparison_available": False,
        "historical_saved_video_full_rgb24_comparison_available": True,
        "formal_result": False,
        "stage_progression_allowed": False,
        "claim_support_status": CLAIM_SUPPORT_STATUS,
    }
    write_json(
        output
        / "artifacts"
        / "historical_gate_a_failure_source_validation.json",
        historical_validation,
    )
    authorization = config["authorization_boundary"]
    frozen_feedback_allowed = bool(
        classification["feedback_nonlinearity_candidate"]
        or classification["classification_status"] == "indeterminate"
    )
    redesign_allowed = bool(
        classification["carrier_decoder_feature_mismatch_candidate"]
    )
    decision = {
        "record_version": DIAGNOSTIC_RECORD_VERSION,
        "profile_id": DIAGNOSTIC_PROFILE_ID,
        "gate_a_root_cause_diagnostic_decision": (
            "candidate_explanations_recorded_no_gate_change"
        ),
        "candidate_classification": classification,
        "clean_order_drift_diagnostic": clean_state,
        "historical_gate_a_decision_preserved": True,
        "historical_gate_a_pass": False,
        "gate_a_retry": False,
        "gate_a_pass": False,
        "cross_identity_confirmation_allowed": False,
        "gate_b_execution_allowed": False,
        "gate_c_execution_allowed": False,
        "wrong_key_execution_allowed": False,
        "observer_execution_allowed": False,
        "state_dynamics_design_allowed": False,
        "attack_execution_allowed": False,
        "pilot_execution_allowed": False,
        "fixed_fpr_execution_allowed": False,
        "external_baseline_execution_allowed": False,
        "paper_claim_allowed": False,
        "frozen_feedback_diagnostic_design_allowed": bool(
            authorization[
                "frozen_feedback_diagnostic_design_may_be_allowed"
            ]
            and frozen_feedback_allowed
        ),
        "carrier_feature_redesign_allowed": bool(
            authorization[
                "carrier_feature_redesign_may_be_allowed"
            ]
            and redesign_allowed
        ),
        "automatic_followup_execution_allowed": False,
        "generation_record_count": len(generation_records),
        "trajectory_step_record_count": len(step_records),
        "actual_exposure_trace_count": len(generation.exposure_traces),
        "checkpoint_record_count": len(checkpoints),
        "feature_record_count": len(feature_records),
        "scaling_comparison_count": len(scaling),
        "historical_source_snapshot_digest": (
            historical.source_snapshot_digest
        ),
        "historical_repository_commit": historical.decision[
            "repository_commit"
        ],
        "diagnostic_lambda_max": config["diagnostic_plan"][
            "diagnostic_lambda_max"
        ],
        "historical_lambda_max": config["diagnostic_plan"][
            "historical_lambda_max"
        ],
        "historical_lambda_rerun_executed": False,
        "construction_basis_digest": basis.basis_digest,
        "construction_feature_schema_digest": runtime_config[
            "construction_feature_schema"
        ]["feature_schema_digest"],
        "impulse_waveform_schema_digest": runtime_config[
            "flow_schedule_contract"
        ]["waveform_schema_digest"],
        "impulse_runtime_adapter_schema_digest": runtime_config[
            "runtime_adapter_contract"
        ]["adapter_schema_digest"],
        "repository_commit": repository_commit,
        "config_digest": config_digest,
        "formal_result": False,
        "stage_progression_allowed": False,
        "claim_support_status": CLAIM_SUPPORT_STATUS,
        **runtime_versions,
    }
    write_json(
        output
        / "artifacts"
        / "gate_a_root_cause_amplitude_feedback_decision.json",
        decision,
    )
    manifest = {
        "manifest_kind": (
            "gate_a_root_cause_amplitude_feedback_diagnostic_manifest"
        ),
        "profile_id": DIAGNOSTIC_PROFILE_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit": repository_commit,
        "config_path": str(Path(config_path)),
        "config_digest": config_digest,
        "historical_source_snapshot_digest": (
            historical.source_snapshot_digest
        ),
        "historical_repository_commit": historical.decision[
            "repository_commit"
        ],
        "historical_gate_a_decision": historical.decision[
            "impulse_observability_construction_decision"
        ],
        "diagnostic_plan_ids": list(DIAGNOSTIC_PLAN_IDS),
        "diagnostic_lambda_max": 0.06,
        "historical_lambda_max": 0.12,
        "historical_lambda_rerun_executed": False,
        "same_initial_generator_state_verified": len(
            {
                record["generation_generator_state_digest_random"]
                for record in generation_records
            }
        )
        == 1,
        "construction_basis_digest": basis.basis_digest,
        "video_records": [
            {
                "impulse_probe_id": record["impulse_probe_id"],
                "video_path": record["video_path"],
                "video_sha256": record["video_sha256"],
            }
            for record in generation_records
        ],
        "output_paths": [
            "records/root_cause_generation_plan.jsonl",
            "records/impulse_generation_records.jsonl",
            "records/impulse_trajectory_step_records.jsonl",
            "records/impulse_actual_exposure_traces.jsonl",
            "records/impulse_checkpoint_records.jsonl",
            "records/impulse_feature_records.jsonl",
            "records/root_cause_array_capture_records.jsonl",
            "records/root_cause_full_gram_records.jsonl",
            "records/root_cause_paired_response_statistics.jsonl",
            "records/root_cause_historical_response_statistics.jsonl",
            "records/root_cause_scaling_comparisons.jsonl",
            "artifacts/historical_gate_a_failure_source_validation.json",
            "artifacts/gate_a_root_cause_amplitude_feedback_decision.json",
        ],
        "formal_result": False,
        "stage_progression_allowed": False,
        "claim_support_status": CLAIM_SUPPORT_STATUS,
        **runtime_versions,
    }
    write_json(
        output
        / "artifacts"
        / "gate_a_root_cause_amplitude_feedback_manifest.json",
        manifest,
    )
    return decision


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the six-video Gate A failure root-cause diagnostic"
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--config-path",
        default=DEFAULT_DIAGNOSTIC_CONFIG_PATH,
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run_gate_a_root_cause_amplitude_feedback_diagnostic(
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
