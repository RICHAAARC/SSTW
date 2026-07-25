"""CPU-only analysis contract for the existing six-video Gate A diagnostic.

This module never regenerates video, selects a feature, or changes Gate A.
It validates the immutable f06a0934 result and measures predeclared
video-time signed odd/common responses directly from saved RGB24 frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from evaluation.protocol.gate_a_root_cause_amplitude_feedback_contract import (
    compute_paired_response_statistics,
)
from evaluation.protocol.impulse_observability_contract import (
    canonical_json_digest,
)


DEFAULT_CONFIG_PATH = (
    "configs/protocol/"
    "sstw_existing_six_video_spatiotemporal_signed_response_diagnostic.json"
)
FROZEN_CONFIG_DIGEST = (
    "855ab6a5c734630f147b2e4d1520a8a6134360251e4aee1f3a3b8b797d540fd4"
)
PROFILE_ID = (
    "sstw_existing_six_video_spatiotemporal_signed_response_diagnostic"
)
RECORD_VERSION = (
    "existing_six_video_spatiotemporal_signed_response_diagnostic_v1"
)
CLAIM_SUPPORT_STATUS = (
    "existing_six_video_spatiotemporal_diagnostic_only_not_method_evidence"
)
PLAN_IDS = (
    "clean_start",
    "positive_early_flow_channel_0",
    "negative_early_flow_channel_0",
    "positive_late_flow_channel_0",
    "negative_late_flow_channel_0",
    "clean_end",
)
PAIR_IDS = ("early_flow_channel_0", "late_flow_channel_0")
VIDEO_NAMES = {
    "clean_start": "00_clean_start.mp4",
    "positive_early_flow_channel_0": (
        "01_positive_early_flow_channel_0.mp4"
    ),
    "negative_early_flow_channel_0": (
        "02_negative_early_flow_channel_0.mp4"
    ),
    "positive_late_flow_channel_0": (
        "03_positive_late_flow_channel_0.mp4"
    ),
    "negative_late_flow_channel_0": (
        "04_negative_late_flow_channel_0.mp4"
    ),
    "clean_end": "05_clean_end.mp4",
}
REQUIRED_RELATIVE_PATHS = (
    "artifacts/final_latents/00_clean_start.npz",
    "artifacts/final_latents/01_positive_early_flow_channel_0.npz",
    "artifacts/final_latents/02_negative_early_flow_channel_0.npz",
    "artifacts/final_latents/03_positive_late_flow_channel_0.npz",
    "artifacts/final_latents/04_negative_late_flow_channel_0.npz",
    "artifacts/final_latents/05_clean_end.npz",
    "artifacts/gate_a_root_cause_amplitude_feedback_decision.json",
    "artifacts/gate_a_root_cause_amplitude_feedback_manifest.json",
    "artifacts/historical_gate_a_failure_source_validation.json",
    "records/impulse_actual_exposure_traces.jsonl",
    "records/impulse_checkpoint_records.jsonl",
    "records/impulse_feature_records.jsonl",
    "records/impulse_generation_records.jsonl",
    "records/impulse_trajectory_step_records.jsonl",
    "records/root_cause_array_capture_records.jsonl",
    "records/root_cause_full_gram_records.jsonl",
    "records/root_cause_generation_plan.jsonl",
    "records/root_cause_historical_response_statistics.jsonl",
    "records/root_cause_paired_response_statistics.jsonl",
    "records/root_cause_scaling_comparisons.jsonl",
    "videos/00_clean_start.mp4",
    "videos/01_positive_early_flow_channel_0.mp4",
    "videos/02_negative_early_flow_channel_0.mp4",
    "videos/03_positive_late_flow_channel_0.mp4",
    "videos/04_negative_late_flow_channel_0.mp4",
    "videos/05_clean_end.mp4",
)
RECORD_COUNTS = {
    "records/root_cause_generation_plan.jsonl": 6,
    "records/impulse_generation_records.jsonl": 6,
    "records/impulse_trajectory_step_records.jsonl": 48,
    "records/impulse_actual_exposure_traces.jsonl": 4,
    "records/impulse_checkpoint_records.jsonl": 30,
    "records/impulse_feature_records.jsonl": 6,
    "records/root_cause_array_capture_records.jsonl": 6,
    "records/root_cause_full_gram_records.jsonl": 4,
    "records/root_cause_paired_response_statistics.jsonl": 18,
    "records/root_cause_historical_response_statistics.jsonl": 14,
    "records/root_cause_scaling_comparisons.jsonl": 12,
}
INTERVAL_REPRESENTATIONS = (
    "frame_interval_full_rgb24",
    "adjacent_difference_interval_full_rgb24",
)


@dataclass(frozen=True)
class ValidatedExistingSixVideoSource:
    root: Path
    source_snapshot_digest: str
    decision: Mapping[str, Any]
    manifest: Mapping[str, Any]
    video_paths: Mapping[str, Path]


@dataclass(frozen=True)
class SpatiotemporalSignedResponseRecord:
    pair_id: str
    representation_id: str
    segment_id: str
    sample_start: int
    sample_stop: int
    source_frame_start: int
    source_frame_stop: int
    positive_centered_norm: float
    negative_centered_norm: float
    odd_norm: float
    common_norm: float
    common_odd_ratio: float | None
    antisymmetry_cosine: float | None
    antisymmetry_residual: float | None
    finite: bool
    signed_response_gate_passed: bool


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    rows = tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    )
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"expected JSONL objects: {path}")
    return rows


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_snapshot_digest(root: Path) -> str:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    rows = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": _sha256_file(path),
            "size": path.stat().st_size,
        }
        for path in files
    ]
    return canonical_json_digest({"files": rows})


def load_spatiotemporal_diagnostic_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    config = _read_json(Path(path))
    if canonical_json_digest(config) != FROZEN_CONFIG_DIGEST:
        raise ValueError(
            "existing-six-video spatiotemporal config 未匹配预冻结 exact digest"
        )
    if (
        config.get("profile_id") != PROFILE_ID
        or config.get("analysis_role")
        != "existing_six_video_read_only_root_cause_triage"
        or config.get("execution_authorized") is not True
        or config.get("cpu_only") is not True
        or config.get("formal_result") is not False
        or config.get("stage_progression_allowed") is not False
        or config.get("claim_support_status") != CLAIM_SUPPORT_STATUS
    ):
        raise ValueError("existing-six-video 顶层合同未冻结")
    source = config["source_binding"]
    if (
        tuple(source["plan_ids"]) != PLAN_IDS
        or source["source_file_count"] != len(REQUIRED_RELATIVE_PATHS)
        or source["gate_a_pass_required"] is not False
        or source["formal_result_required"] is not False
        or source["stage_progression_allowed_required"] is not False
        or source["same_initial_generator_state_required"] is not True
        or source["clean_all_frozen_representations_equal_required"]
        is not True
    ):
        raise ValueError("existing-six-video source binding 未冻结")
    plan = config["analysis_plan"]
    intervals = tuple(
        (
            row["interval_id"],
            row["frame_start"],
            row["frame_stop"],
        )
        for row in plan["video_frame_intervals"]
    )
    difference_intervals = tuple(
        (
            row["interval_id"],
            row["difference_start"],
            row["difference_stop"],
        )
        for row in plan["adjacent_difference_intervals"]
    )
    if (
        plan["video_count"] != 6
        or plan["frame_count"] != 33
        or (plan["height"], plan["width"], plan["channel_count"])
        != (320, 512, 3)
        or plan["pixel_dtype"] != "uint8"
        or plan["clean_intercept"]
        != "arithmetic_mean_clean_start_clean_end"
        or intervals
        != (
            ("early_video_time", 0, 11),
            ("middle_video_time", 11, 22),
            ("late_video_time", 22, 33),
        )
        or difference_intervals
        != (
            ("early_video_time", 0, 10),
            ("middle_video_time", 11, 21),
            ("late_video_time", 22, 32),
        )
        or plan["spatial_block_analysis_enabled"] is not False
        or plan["candidate_key_independent"] is not True
        or plan["runtime_frame_block_channel_band_selection_allowed"]
        is not False
        or not math.isclose(
            float(plan["norm_denominator_epsilon"]),
            1e-15,
            rel_tol=0.0,
            abs_tol=0.0,
        )
    ):
        raise ValueError("existing-six-video analysis plan 未冻结")
    gate = config["signed_response_gate"]
    if gate != {
        "minimum_antisymmetry_cosine": 0.9,
        "maximum_antisymmetry_residual": 0.25,
        "maximum_common_odd_ratio": 0.5,
        "minimum_odd_norm": 1e-12,
    }:
        raise ValueError("existing-six-video signed gate 未冻结")
    classification = config["classification_contract"]
    if (
        tuple(classification["representation_families"])
        != INTERVAL_REPRESENTATIONS
        or classification["all_three_intervals_required_per_family"]
        is not True
        or classification["both_flow_stage_pairs_required"] is not True
        or classification["single_frame_or_single_transition_pass_insufficient"]
        is not True
        or classification["automatic_feature_selection_allowed"] is not False
        or classification["unique_root_cause_claim_allowed"] is not False
        or classification["historical_gate_a_failure_override_allowed"]
        is not False
    ):
        raise ValueError("existing-six-video classification contract 未冻结")
    if any(
        value is not False
        for value in config["authorization_boundary"].values()
    ):
        raise ValueError("existing-six-video diagnostic 越权")
    return config


def validate_existing_six_video_source(
    source_root: str | Path,
    config: Mapping[str, Any],
) -> ValidatedExistingSixVideoSource:
    root = Path(source_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("existing-six-video source 不允许 symlink")
    if any(
        "recovery" in path.name.lower() or path.name.startswith("partial_")
        for path in root.rglob("*")
    ):
        raise ValueError("existing-six-video source 不得是 recovery/partial")
    observed_paths = tuple(
        path.relative_to(root).as_posix()
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    )
    if observed_paths != tuple(sorted(REQUIRED_RELATIVE_PATHS)):
        raise ValueError("existing-six-video source 文件集合不完整或含额外文件")
    binding = config["source_binding"]
    snapshot_digest = _source_snapshot_digest(root)
    if snapshot_digest != binding["source_snapshot_digest"]:
        raise ValueError("existing-six-video source snapshot digest 不一致")

    decision = _read_json(
        root
        / "artifacts"
        / "gate_a_root_cause_amplitude_feedback_decision.json"
    )
    manifest = _read_json(
        root
        / "artifacts"
        / "gate_a_root_cause_amplitude_feedback_manifest.json"
    )
    exact_decision = {
        "record_version": binding["record_version"],
        "profile_id": binding["profile_id"],
        "gate_a_root_cause_diagnostic_decision": binding["decision"],
        "repository_commit": binding["repository_commit"],
        "config_digest": binding["config_digest"],
        "historical_source_snapshot_digest": binding[
            "historical_gate_a_source_snapshot_digest"
        ],
        "historical_repository_commit": binding[
            "historical_repository_commit"
        ],
        "construction_basis_digest": binding["construction_basis_digest"],
        "construction_feature_schema_digest": binding[
            "construction_feature_schema_digest"
        ],
        "impulse_waveform_schema_digest": binding[
            "impulse_waveform_schema_digest"
        ],
        "impulse_runtime_adapter_schema_digest": binding[
            "impulse_runtime_adapter_schema_digest"
        ],
        "generation_record_count": binding["generation_record_count"],
        "trajectory_step_record_count": binding[
            "trajectory_step_record_count"
        ],
        "actual_exposure_trace_count": binding[
            "actual_exposure_trace_count"
        ],
        "checkpoint_record_count": binding["checkpoint_record_count"],
        "feature_record_count": binding["feature_record_count"],
        "historical_gate_a_pass": False,
        "gate_a_pass": False,
        "formal_result": False,
        "stage_progression_allowed": False,
    }
    mismatches = {
        key: {"expected": expected, "observed": decision.get(key)}
        for key, expected in exact_decision.items()
        if decision.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            "existing-six-video decision binding 不一致: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )
    clean = decision.get("clean_order_drift_diagnostic")
    required_clean_fields = (
        "clean_start_end_video_sha256_equal",
        "clean_start_end_final_latent_array_sha256_equal",
        "clean_start_end_pre_save_decoded_array_sha256_equal",
        "clean_start_end_reencoded_checkpoint_equal",
        "clean_start_end_output_feature_equal",
    )
    if (
        not isinstance(clean, dict)
        or any(clean.get(field) is not True for field in required_clean_fields)
        or clean.get("generation_order_state_pollution_candidate") is not False
    ):
        raise ValueError("existing-six-video clean equality 未冻结")
    if any(
        decision.get(field) is not False
        for field in (
            "gate_a_retry",
            "cross_identity_confirmation_allowed",
            "gate_b_execution_allowed",
            "gate_c_execution_allowed",
            "wrong_key_execution_allowed",
            "observer_execution_allowed",
            "state_dynamics_design_allowed",
            "attack_execution_allowed",
            "fixed_fpr_execution_allowed",
            "external_baseline_execution_allowed",
            "paper_claim_allowed",
            "automatic_followup_execution_allowed",
        )
    ):
        raise ValueError("existing-six-video source decision 越权")
    if (
        manifest.get("manifest_kind") != binding["manifest_kind"]
        or manifest.get("profile_id") != binding["profile_id"]
        or manifest.get("repository_commit") != binding["repository_commit"]
        or manifest.get("config_digest") != binding["config_digest"]
        or manifest.get("historical_source_snapshot_digest")
        != binding["historical_gate_a_source_snapshot_digest"]
        or manifest.get("construction_basis_digest")
        != binding["construction_basis_digest"]
        or tuple(manifest.get("diagnostic_plan_ids") or ()) != PLAN_IDS
        or manifest.get("same_initial_generator_state_verified") is not True
        or manifest.get("formal_result") is not False
        or manifest.get("stage_progression_allowed") is not False
    ):
        raise ValueError("existing-six-video manifest binding 不一致")

    for relative_path, expected_count in RECORD_COUNTS.items():
        rows = _read_jsonl(root / relative_path)
        if len(rows) != expected_count:
            raise ValueError(
                f"existing-six-video record count 不一致: {relative_path}"
            )
    plan_rows = _read_jsonl(
        root / "records" / "root_cause_generation_plan.jsonl"
    )
    if tuple(row.get("probe_id") for row in plan_rows) != PLAN_IDS:
        raise ValueError("existing-six-video plan identity/order 不一致")
    generation_rows = _read_jsonl(
        root / "records" / "impulse_generation_records.jsonl"
    )
    if tuple(
        row.get("impulse_probe_id") for row in generation_rows
    ) != PLAN_IDS:
        raise ValueError("existing-six-video generation identity/order 不一致")
    generator_digests = {
        row.get("generation_generator_state_digest_random")
        for row in generation_rows
    }
    if len(generator_digests) != 1 or generator_digests == {None}:
        raise ValueError("existing-six-video initial generator state 不一致")

    expected_video_sha = binding["video_sha256_by_probe"]
    manifest_video_records = manifest.get("video_records")
    if not isinstance(manifest_video_records, list) or tuple(
        row.get("impulse_probe_id") for row in manifest_video_records
    ) != PLAN_IDS:
        raise ValueError("existing-six-video manifest video order 不一致")
    video_paths: dict[str, Path] = {}
    for index, (probe_id, generation, video_record) in enumerate(
        zip(PLAN_IDS, generation_rows, manifest_video_records, strict=True)
    ):
        video_path = root / "videos" / VIDEO_NAMES[probe_id]
        observed_sha = _sha256_file(video_path)
        if (
            generation.get("generation_status") != "success"
            or generation.get("impulse_probe_plan_index") != index
            or generation.get("video_sha256") != expected_video_sha[probe_id]
            or video_record.get("video_sha256") != expected_video_sha[probe_id]
            or observed_sha != expected_video_sha[probe_id]
            or generation.get("formal_result") is not False
            or generation.get("stage_progression_allowed") is not False
        ):
            raise ValueError(
                f"existing-six-video video/generation binding 失败: {probe_id}"
            )
        video_paths[probe_id] = video_path
    if expected_video_sha["clean_start"] != expected_video_sha["clean_end"]:
        raise ValueError("existing-six-video clean video digest 不一致")
    return ValidatedExistingSixVideoSource(
        root=root,
        source_snapshot_digest=snapshot_digest,
        decision=decision,
        manifest=manifest,
        video_paths=video_paths,
    )


def _record_from_statistics(
    config: Mapping[str, Any],
    *,
    pair_id: str,
    representation_id: str,
    segment_id: str,
    sample_start: int,
    sample_stop: int,
    source_frame_start: int,
    source_frame_stop: int,
    clean_start: np.ndarray,
    clean_end: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
) -> SpatiotemporalSignedResponseRecord:
    statistics = compute_paired_response_statistics(
        pair_id=pair_id,
        checkpoint_id=representation_id,
        clean_start=np.asarray(clean_start, dtype=np.float64).reshape(-1),
        clean_end=np.asarray(clean_end, dtype=np.float64).reshape(-1),
        positive=np.asarray(positive, dtype=np.float64).reshape(-1),
        negative=np.asarray(negative, dtype=np.float64).reshape(-1),
        denominator_epsilon=float(
            config["analysis_plan"]["norm_denominator_epsilon"]
        ),
    )
    gate = config["signed_response_gate"]
    passed = bool(
        statistics.finite
        and statistics.odd_norm >= float(gate["minimum_odd_norm"])
        and statistics.antisymmetry_cosine is not None
        and statistics.antisymmetry_cosine
        >= float(gate["minimum_antisymmetry_cosine"])
        and statistics.antisymmetry_residual is not None
        and statistics.antisymmetry_residual
        <= float(gate["maximum_antisymmetry_residual"])
        and statistics.common_odd_ratio is not None
        and statistics.common_odd_ratio
        <= float(gate["maximum_common_odd_ratio"])
    )
    return SpatiotemporalSignedResponseRecord(
        pair_id=pair_id,
        representation_id=representation_id,
        segment_id=segment_id,
        sample_start=sample_start,
        sample_stop=sample_stop,
        source_frame_start=source_frame_start,
        source_frame_stop=source_frame_stop,
        positive_centered_norm=statistics.positive_centered_norm,
        negative_centered_norm=statistics.negative_centered_norm,
        odd_norm=statistics.odd_norm,
        common_norm=statistics.common_norm,
        common_odd_ratio=statistics.common_odd_ratio,
        antisymmetry_cosine=statistics.antisymmetry_cosine,
        antisymmetry_residual=statistics.antisymmetry_residual,
        finite=statistics.finite,
        signed_response_gate_passed=passed,
    )


def compute_spatiotemporal_signed_response(
    config: Mapping[str, Any],
    frames_by_probe: Mapping[str, np.ndarray],
) -> tuple[SpatiotemporalSignedResponseRecord, ...]:
    if tuple(frames_by_probe) != PLAN_IDS:
        raise ValueError("spatiotemporal frame identity/order 不一致")
    plan = config["analysis_plan"]
    expected_shape = (
        int(plan["frame_count"]),
        int(plan["height"]),
        int(plan["width"]),
        int(plan["channel_count"]),
    )
    frames: dict[str, np.ndarray] = {}
    for probe_id, value in frames_by_probe.items():
        observed = np.asarray(value)
        if observed.shape != expected_shape or observed.dtype != np.uint8:
            raise ValueError(
                f"{probe_id} RGB24 shape/dtype 未冻结: "
                f"{observed.shape}/{observed.dtype}"
            )
        frames[probe_id] = observed.astype(np.float64) / 255.0
    if not np.array_equal(
        frames_by_probe["clean_start"],
        frames_by_probe["clean_end"],
    ):
        raise ValueError(
            "clean_start/end RGB24 不一致，所有时空 signed 结论 fail-closed"
        )

    records: list[SpatiotemporalSignedResponseRecord] = []
    for pair_id in PAIR_IDS:
        values = {
            "clean_start": frames["clean_start"],
            "clean_end": frames["clean_end"],
            "positive": frames[f"positive_{pair_id}"],
            "negative": frames[f"negative_{pair_id}"],
        }
        for frame_index in range(expected_shape[0]):
            records.append(
                _record_from_statistics(
                    config,
                    pair_id=pair_id,
                    representation_id="per_frame_full_rgb24",
                    segment_id=f"frame_{frame_index:02d}",
                    sample_start=frame_index,
                    sample_stop=frame_index + 1,
                    source_frame_start=frame_index,
                    source_frame_stop=frame_index + 1,
                    **{
                        key: value[frame_index]
                        for key, value in values.items()
                    },
                )
            )
        for interval in plan["video_frame_intervals"]:
            start = int(interval["frame_start"])
            stop = int(interval["frame_stop"])
            records.append(
                _record_from_statistics(
                    config,
                    pair_id=pair_id,
                    representation_id="frame_interval_full_rgb24",
                    segment_id=str(interval["interval_id"]),
                    sample_start=start,
                    sample_stop=stop,
                    source_frame_start=start,
                    source_frame_stop=stop,
                    **{key: value[start:stop] for key, value in values.items()},
                )
            )
        differences = {
            key: np.diff(value, axis=0) for key, value in values.items()
        }
        for difference_index in range(expected_shape[0] - 1):
            records.append(
                _record_from_statistics(
                    config,
                    pair_id=pair_id,
                    representation_id=(
                        "adjacent_frame_difference_full_rgb24"
                    ),
                    segment_id=(
                        f"difference_{difference_index:02d}_"
                        f"{difference_index + 1:02d}"
                    ),
                    sample_start=difference_index,
                    sample_stop=difference_index + 1,
                    source_frame_start=difference_index,
                    source_frame_stop=difference_index + 2,
                    **{
                        key: value[difference_index]
                        for key, value in differences.items()
                    },
                )
            )
        for interval in plan["adjacent_difference_intervals"]:
            start = int(interval["difference_start"])
            stop = int(interval["difference_stop"])
            records.append(
                _record_from_statistics(
                    config,
                    pair_id=pair_id,
                    representation_id=(
                        "adjacent_difference_interval_full_rgb24"
                    ),
                    segment_id=str(interval["interval_id"]),
                    sample_start=start,
                    sample_stop=stop,
                    source_frame_start=start,
                    source_frame_stop=stop + 1,
                    **{
                        key: value[start:stop]
                        for key, value in differences.items()
                    },
                )
            )
    if len(records) != 142:
        raise AssertionError("spatiotemporal signed response record count 漂移")
    return tuple(records)


def classify_spatiotemporal_signed_response(
    config: Mapping[str, Any],
    records: Sequence[SpatiotemporalSignedResponseRecord],
) -> dict[str, Any]:
    expected_identities: list[tuple[Any, ...]] = []
    plan = config["analysis_plan"]
    for pair_id in PAIR_IDS:
        for frame_index in range(33):
            expected_identities.append(
                (
                    pair_id,
                    "per_frame_full_rgb24",
                    f"frame_{frame_index:02d}",
                    frame_index,
                    frame_index + 1,
                    frame_index,
                    frame_index + 1,
                )
            )
        for interval in plan["video_frame_intervals"]:
            start = int(interval["frame_start"])
            stop = int(interval["frame_stop"])
            expected_identities.append(
                (
                    pair_id,
                    "frame_interval_full_rgb24",
                    str(interval["interval_id"]),
                    start,
                    stop,
                    start,
                    stop,
                )
            )
        for difference_index in range(32):
            expected_identities.append(
                (
                    pair_id,
                    "adjacent_frame_difference_full_rgb24",
                    (
                        f"difference_{difference_index:02d}_"
                        f"{difference_index + 1:02d}"
                    ),
                    difference_index,
                    difference_index + 1,
                    difference_index,
                    difference_index + 2,
                )
            )
        for interval in plan["adjacent_difference_intervals"]:
            start = int(interval["difference_start"])
            stop = int(interval["difference_stop"])
            expected_identities.append(
                (
                    pair_id,
                    "adjacent_difference_interval_full_rgb24",
                    str(interval["interval_id"]),
                    start,
                    stop,
                    start,
                    stop + 1,
                )
            )
    observed_identities = [
        (
            record.pair_id,
            record.representation_id,
            record.segment_id,
            record.sample_start,
            record.sample_stop,
            record.source_frame_start,
            record.source_frame_stop,
        )
        for record in records
    ]
    if (
        observed_identities != expected_identities
        or any(not record.finite for record in records)
    ):
        return {
            "candidate_classifications": ["indeterminate"],
            "temporal_feature_salvage_candidate": False,
            "carrier_redesign_required_candidate": False,
            "multiple_candidates": False,
            "classification_status": "indeterminate",
            "stable_interval_representation_families": [],
            "family_gate_summaries": {},
            "single_frame_or_single_transition_pass_sufficient": False,
            "automatic_feature_selection_allowed": False,
            "unique_root_cause_claim_allowed": False,
        }
    stable_families = []
    family_summaries: dict[str, dict[str, Any]] = {}
    for representation_id in INTERVAL_REPRESENTATIONS:
        family_records = [
            record
            for record in records
            if record.representation_id == representation_id
        ]
        expected_segments = {
            (pair_id, interval_id)
            for pair_id in PAIR_IDS
            for interval_id in (
                "early_video_time",
                "middle_video_time",
                "late_video_time",
            )
        }
        observed_segments = {
            (record.pair_id, record.segment_id)
            for record in family_records
        }
        ready = bool(
            observed_segments == expected_segments
            and len(family_records) == 6
            and all(
                record.signed_response_gate_passed
                for record in family_records
            )
        )
        if ready:
            stable_families.append(representation_id)
        family_summaries[representation_id] = {
            "interval_record_count": len(family_records),
            "signed_gate_pass_count": sum(
                record.signed_response_gate_passed
                for record in family_records
            ),
            "all_three_intervals_both_flow_stages_ready": ready,
        }
    salvage = bool(stable_families)
    redesign = not salvage
    candidates = [
        (
            "temporal_feature_salvage_candidate"
            if salvage
            else "carrier_redesign_required_candidate"
        )
    ]
    return {
        "candidate_classifications": candidates,
        "temporal_feature_salvage_candidate": salvage,
        "carrier_redesign_required_candidate": redesign,
        "multiple_candidates": False,
        "classification_status": candidates[0],
        "stable_interval_representation_families": stable_families,
        "family_gate_summaries": family_summaries,
        "single_frame_or_single_transition_pass_sufficient": False,
        "automatic_feature_selection_allowed": False,
        "unique_root_cause_claim_allowed": False,
    }


def classify_frozen_feedback_signed_response_design(
    *,
    clean_coverage_and_guards_ready: bool,
    early_full_final_latent_signed: bool,
    late_full_final_latent_signed: bool,
    all_post_latent_checkpoints_signed: bool,
) -> dict[str, Any]:
    """Apply the mutually exclusive design-only frozen-feedback table."""

    inputs = (
        clean_coverage_and_guards_ready,
        early_full_final_latent_signed,
        late_full_final_latent_signed,
        all_post_latent_checkpoints_signed,
    )
    if any(type(value) is not bool for value in inputs):
        raise TypeError("frozen-feedback design predicates 必须是 bool")

    latent_all_signed = bool(
        early_full_final_latent_signed
        and late_full_final_latent_signed
    )
    latent_all_failed = bool(
        not early_full_final_latent_signed
        and not late_full_final_latent_signed
    )
    if not clean_coverage_and_guards_ready:
        candidates = ["indeterminate_stop"]
    elif latent_all_signed and all_post_latent_checkpoints_signed:
        candidates = ["feedback_isolation_candidate"]
    elif latent_all_signed and not all_post_latent_checkpoints_signed:
        candidates = [
            "feedback_isolation_candidate",
            "decoder_carrier_mismatch_candidate",
            "multiple_candidates",
        ]
    elif latent_all_failed and not all_post_latent_checkpoints_signed:
        candidates = ["stop_current_additive_random_carrier"]
    else:
        candidates = ["indeterminate_stop"]
    return {
        "candidate_classifications": candidates,
        "classification_status": candidates[-1],
        "feedback_isolation_candidate": (
            "feedback_isolation_candidate" in candidates
        ),
        "decoder_carrier_mismatch_candidate": (
            "decoder_carrier_mismatch_candidate" in candidates
        ),
        "stop_current_additive_random_carrier": (
            "stop_current_additive_random_carrier" in candidates
        ),
        "multiple_candidates": "multiple_candidates" in candidates,
        "unique_root_cause_claim_allowed": False,
        "formal_result": False,
        "stage_progression_allowed": False,
    }


def governed_spatiotemporal_record(
    record: SpatiotemporalSignedResponseRecord,
) -> dict[str, Any]:
    value = {
        "record_version": RECORD_VERSION,
        "profile_id": PROFILE_ID,
        "spatiotemporal_pair_id": record.pair_id,
        "spatiotemporal_representation_id": record.representation_id,
        "spatiotemporal_segment_id": record.segment_id,
        "spatiotemporal_sample_start": record.sample_start,
        "spatiotemporal_sample_stop": record.sample_stop,
        "spatiotemporal_source_frame_start": record.source_frame_start,
        "spatiotemporal_source_frame_stop": record.source_frame_stop,
        "spatiotemporal_positive_centered_norm": (
            record.positive_centered_norm
        ),
        "spatiotemporal_negative_centered_norm": (
            record.negative_centered_norm
        ),
        "spatiotemporal_odd_norm": record.odd_norm,
        "spatiotemporal_common_norm": record.common_norm,
        "spatiotemporal_common_odd_ratio": record.common_odd_ratio,
        "spatiotemporal_antisymmetry_cosine": (
            record.antisymmetry_cosine
        ),
        "spatiotemporal_antisymmetry_residual": (
            record.antisymmetry_residual
        ),
        "spatiotemporal_statistics_finite": record.finite,
        "spatiotemporal_signed_response_gate_passed": (
            record.signed_response_gate_passed
        ),
        "formal_result": False,
        "stage_progression_allowed": False,
        "claim_support_status": CLAIM_SUPPORT_STATUS,
    }
    value["spatiotemporal_signed_response_record_id"] = (
        canonical_json_digest(value)
    )
    return value


def records_as_dicts(
    records: Sequence[SpatiotemporalSignedResponseRecord],
) -> list[dict[str, Any]]:
    return [governed_spatiotemporal_record(record) for record in records]
