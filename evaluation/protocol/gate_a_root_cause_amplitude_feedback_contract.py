"""Contract primitives for the post-Gate-A amplitude/feedback diagnostic.

This protocol never re-evaluates or overrides Gate A.  It validates the
immutable 0.12 Gate A failure, freezes a separate six-video 0.06 plan, and
computes paired odd/common response diagnostics.  All classifications are
candidate explanations, may overlap, and are not method evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from evaluation.protocol.impulse_observability_contract import (
    CONSTRUCTION_LATENT_LAYOUT_SHAPE,
    ActualImpulseExposureTrace,
    ImpulseProbePlanRecord,
    _validate_trace_schedule_and_budget,
    _validate_trace_shape,
    assemble_actual_design_matrix,
    build_impulse_triage_plan,
    canonical_json_digest,
    construction_feature_row_binding_digest,
    evaluate_gate_a_statistics,
    load_impulse_observability_config,
    validate_construction_output_features,
)


DEFAULT_DIAGNOSTIC_CONFIG_PATH = (
    "configs/protocol/"
    "sstw_gate_a_root_cause_amplitude_feedback_diagnostic.json"
)
FROZEN_DIAGNOSTIC_CONFIG_DIGEST = (
    "e1a89a4ebc3d30d9007c49ad1b6532da0ee55d53959bc41de24add1c6ca57a43"
)
DIAGNOSTIC_PROFILE_ID = (
    "sstw_gate_a_root_cause_amplitude_feedback_diagnostic"
)
DIAGNOSTIC_RECORD_VERSION = (
    "gate_a_root_cause_amplitude_feedback_diagnostic_v1"
)
DIAGNOSTIC_PLAN_IDS = (
    "clean_start",
    "positive_early_flow_channel_0",
    "negative_early_flow_channel_0",
    "positive_late_flow_channel_0",
    "negative_late_flow_channel_0",
    "clean_end",
)
PAIR_IDS = ("early_flow_channel_0", "late_flow_channel_0")
COMPARABLE_CHECKPOINT_IDS = (
    "T_latent_six_basis",
    "T_decoded_48d",
    "T_saved_video_48d",
    "T_reencoded_256d",
    "T_output_feature_256d",
    "T_saved_video_full_rgb24",
)
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
HISTORICAL_PROJECTION_MAX_BACKOFF_COUNT = 16
HISTORICAL_PROJECTION_REFINEMENT_COUNT = 12


@dataclass(frozen=True)
class HistoricalGateAFailureSource:
    """Validated immutable 0.12 Gate A failure inputs."""

    root: Path
    source_snapshot_digest: str
    decision: Mapping[str, Any]
    manifest: Mapping[str, Any]
    plan: tuple[ImpulseProbePlanRecord, ...]
    generation_records: tuple[Mapping[str, Any], ...]
    step_records: tuple[Mapping[str, Any], ...]
    exposure_traces: tuple[ActualImpulseExposureTrace, ...]
    checkpoint_records: tuple[Mapping[str, Any], ...]
    feature_records: tuple[Mapping[str, Any], ...]
    checkpoint_values: Mapping[tuple[str, str], np.ndarray]
    video_paths: Mapping[str, Path]
    actual_exposure_by_pair: Mapping[str, float]


@dataclass(frozen=True)
class PairedResponseStatistics:
    """Odd/common decomposition for one pair and one checkpoint."""

    pair_id: str
    checkpoint_id: str
    clean_distance: float
    positive_centered_norm: float
    negative_centered_norm: float
    odd_norm: float
    common_norm: float
    common_odd_ratio: float | None
    antisymmetry_cosine: float | None
    antisymmetry_residual: float | None
    finite: bool


@dataclass(frozen=True)
class ScalingComparison:
    """Apply-only comparison between the new 0.06 and historical 0.12 run."""

    pair_id: str
    checkpoint_id: str
    actual_amplitude_ratio: float | None
    actual_amplitude_ratio_ready: bool
    odd_ratio_to_full: float | None
    common_ratio_to_full: float | None
    normalized_odd_scaling: float | None
    normalized_common_scaling: float | None
    antisymmetry_cosine_improvement: float | None
    antisymmetry_residual_improvement: float | None
    local_linear_scaling_ready: bool
    comparison_status: str


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} 顶层必须是 JSON 对象")
    return dict(value)


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    rows = tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    )
    if any(not isinstance(row, dict) for row in rows):
        raise TypeError(f"{path.name} 每行必须是 JSON 对象")
    return tuple(dict(row) for row in rows)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_unique(root: Path, suffix: str) -> Path:
    candidates = tuple(
        path
        for path in root.rglob(Path(suffix).name)
        if path.as_posix().endswith(suffix)
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"historical Gate A source 必须唯一包含 {suffix}: "
            f"observed={len(candidates)}"
        )
    if candidates[0].is_symlink() or not candidates[0].is_file():
        raise RuntimeError(f"historical Gate A source 文件非法: {suffix}")
    return candidates[0]


def _reject_unknown_fields(
    value: Mapping[str, Any],
    expected: set[str],
    label: str,
) -> None:
    unknown = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if unknown or missing:
        raise ValueError(
            f"{label} 字段集合不一致: missing={missing}, unknown={unknown}"
        )


def load_gate_a_root_cause_diagnostic_config(
    path: str | Path = DEFAULT_DIAGNOSTIC_CONFIG_PATH,
) -> dict[str, Any]:
    """Load and strictly validate the independent diagnostic contract."""

    config = _read_json(Path(path))
    _reject_unknown_fields(
        config,
        {
            "profile_id",
            "method_id",
            "diagnostic_role",
            "execution_authorized",
            "historical_gate_a_failure_required",
            "historical_gate_a_source_binding",
            "base_construction_contract",
            "execution_identity",
            "diagnostic_plan",
            "checkpoint_contract",
            "interpretation_contract",
            "authorization_boundary",
            "formal_result",
            "stage_progression_allowed",
            "claim_support_status",
        },
        "root-cause diagnostic config",
    )
    if (
        config["profile_id"] != DIAGNOSTIC_PROFILE_ID
        or config["execution_authorized"] is not True
        or config["historical_gate_a_failure_required"] is not True
        or config["formal_result"] is not False
        or config["stage_progression_allowed"] is not False
    ):
        raise ValueError("root-cause diagnostic 顶层授权边界不一致")
    _reject_unknown_fields(
        config["historical_gate_a_source_binding"],
        {
            "repository_commit",
            "profile_id",
            "record_version",
            "manifest_kind",
            "decision",
            "config_digest",
            "construction_feature_schema_digest",
            "impulse_waveform_schema_digest",
            "impulse_runtime_adapter_schema_digest",
            "generation_record_count",
            "trajectory_step_record_count",
            "actual_exposure_trace_count",
            "checkpoint_record_count",
            "feature_record_count",
            "historical_lambda_max",
            "gate_a_pass_required",
            "formal_result_required",
            "stage_progression_allowed_required",
        },
        "historical Gate A source binding",
    )
    _reject_unknown_fields(
        config["base_construction_contract"],
        {
            "config_path",
            "config_digest",
            "reuse_actual_fp32_projection_and_backoff",
            "reuse_basis_schedule_and_feature_schema",
        },
        "base construction contract",
    )
    _reject_unknown_fields(
        config["execution_identity"],
        {
            "prompt_suite_id",
            "prompt_id",
            "prompt_text",
            "prompt_text_sha256",
            "negative_prompt_text",
            "negative_prompt_text_sha256",
            "seed_id",
            "seed_value",
            "generation_model_id",
            "generation_model_revision",
            "scheduler_signature",
            "num_inference_steps",
            "num_frames",
            "width",
            "height",
            "guidance_scale",
            "fps",
            "same_initial_generator_state_required",
        },
        "diagnostic execution identity",
    )
    _reject_unknown_fields(
        config["diagnostic_plan"],
        {
            "video_count",
            "plan_order",
            "signed_probe_count",
            "diagnostic_lambda_max",
            "historical_lambda_max",
            "historical_lambda_rerun_allowed",
            "strength_sweep_allowed",
            "identity_sweep_allowed",
            "channel_sweep_allowed",
            "clean_repeat_role",
            "clean_repeat_formal_noise_distribution",
        },
        "diagnostic plan",
    )
    _reject_unknown_fields(
        config["checkpoint_contract"],
        {
            "signed_comparison_checkpoint_ids",
            "diagnostic_only_checkpoint_ids",
            "full_final_latent_storage",
            "full_final_latent_shape",
            "latent_signed_metric_excludes_l2_norm",
            "pre_save_decoded_representation",
            "saved_video_full_rgb24_analysis",
            "clean_intercept",
            "odd_formula",
            "common_formula",
            "antisymmetry_cosine_formula",
            "antisymmetry_residual_formula",
            "common_odd_ratio_formula",
            "norm_denominator_epsilon",
        },
        "diagnostic checkpoint contract",
    )
    _reject_unknown_fields(
        config["interpretation_contract"],
        {
            "threshold_role",
            "ideal_half_odd_ratio",
            "minimum_actual_amplitude_ratio_for_comparison",
            "maximum_actual_amplitude_ratio_for_comparison",
            "minimum_half_odd_ratio_for_local_linear_candidate",
            "maximum_half_odd_ratio_for_local_linear_candidate",
            "ideal_half_common_ratio",
            "minimum_half_common_ratio_for_local_linear_candidate",
            "maximum_half_common_ratio_for_local_linear_candidate",
            "minimum_antisymmetry_cosine_improvement",
            "minimum_antisymmetry_residual_improvement",
            "maximum_half_residual_fraction_for_local_linear_candidate",
            "minimum_comparable_norm",
            "maximum_half_odd_ratio_for_floor_collapse_candidate",
            "minimum_half_odd_ratio_for_plateau_candidate",
            "minimum_early_late_antisymmetry_cosine_gap_for_feedback_candidate",
            "minimum_early_late_common_odd_ratio_factor_for_feedback_candidate",
            "maximum_early_late_exposure_normalized_odd_transfer_ratio_for_feedback_candidate",
            "minimum_late_latent_antisymmetry_cosine_for_signed_response",
            "maximum_late_latent_antisymmetry_residual_for_signed_response",
            "maximum_late_latent_common_odd_ratio_for_signed_response",
            "minimum_post_latent_common_odd_ratio_for_mismatch_candidate",
            "maximum_post_latent_antisymmetry_cosine_for_mismatch_candidate",
            "candidate_classifications",
            "unique_root_cause_claim_allowed",
            "historical_gate_a_failure_override_allowed",
        },
        "diagnostic interpretation contract",
    )
    _reject_unknown_fields(
        config["authorization_boundary"],
        {
            "gate_a_retry",
            "gate_a_pass",
            "cross_identity_confirmation_allowed",
            "gate_b_execution_allowed",
            "gate_c_execution_allowed",
            "wrong_key_execution_allowed",
            "observer_execution_allowed",
            "state_dynamics_design_allowed",
            "attack_execution_allowed",
            "pilot_execution_allowed",
            "fixed_fpr_execution_allowed",
            "external_baseline_execution_allowed",
            "paper_claim_allowed",
            "frozen_feedback_diagnostic_design_may_be_allowed",
            "carrier_feature_redesign_may_be_allowed",
            "automatic_followup_execution_allowed",
        },
        "diagnostic authorization boundary",
    )
    plan = config["diagnostic_plan"]
    binding = config["historical_gate_a_source_binding"]
    base_contract = config["base_construction_contract"]
    if (
        binding["historical_lambda_max"] != 0.12
        or binding["gate_a_pass_required"] is not False
        or binding["formal_result_required"] is not False
        or binding["stage_progression_allowed_required"] is not False
        or base_contract["reuse_actual_fp32_projection_and_backoff"]
        is not True
        or base_contract["reuse_basis_schedule_and_feature_schema"]
        is not True
    ):
        raise ValueError("historical/base construction binding 边界不一致")
    if (
        int(plan["video_count"]) != 6
        or tuple(plan["plan_order"]) != DIAGNOSTIC_PLAN_IDS
        or int(plan["signed_probe_count"]) != 4
        or float(plan["diagnostic_lambda_max"]) != 0.06
        or float(plan["historical_lambda_max"]) != 0.12
        or plan["historical_lambda_rerun_allowed"] is not False
        or plan["strength_sweep_allowed"] is not False
        or plan["identity_sweep_allowed"] is not False
        or plan["channel_sweep_allowed"] is not False
        or plan["clean_repeat_formal_noise_distribution"] is not False
    ):
        raise ValueError("root-cause diagnostic 6-video/half-amplitude 计划不一致")
    identity = config["execution_identity"]
    if identity["same_initial_generator_state_required"] is not True:
        raise ValueError("diagnostic 必须为每视频重建同一初始 generator state")
    for text_key, digest_key in (
        ("prompt_text", "prompt_text_sha256"),
        ("negative_prompt_text", "negative_prompt_text_sha256"),
    ):
        observed = sha256(str(identity[text_key]).encode("utf-8")).hexdigest()
        if observed != identity[digest_key]:
            raise ValueError(f"{text_key} 与冻结 SHA-256 不一致")
    checkpoints = config["checkpoint_contract"]
    if (
        tuple(checkpoints["signed_comparison_checkpoint_ids"])
        != COMPARABLE_CHECKPOINT_IDS[:-1]
        or tuple(checkpoints["full_final_latent_shape"])
        != CONSTRUCTION_LATENT_LAYOUT_SHAPE
        or checkpoints["latent_signed_metric_excludes_l2_norm"] is not True
        or checkpoints["clean_intercept"]
        != "arithmetic_mean_clean_start_clean_end"
        or float(checkpoints["norm_denominator_epsilon"]) <= 0.0
    ):
        raise ValueError("root-cause diagnostic checkpoint 数值合同不一致")
    interpretation = config["interpretation_contract"]
    expected_classifications = {
        "local_linear_support_candidate",
        "quantization_or_observation_floor_candidate",
        "feedback_nonlinearity_candidate",
        "carrier_decoder_feature_mismatch_candidate",
        "generation_order_state_pollution_candidate",
        "multiple_factors_candidate",
        "indeterminate",
    }
    if set(interpretation["candidate_classifications"]) != (
        expected_classifications
    ) or len(interpretation["candidate_classifications"]) != len(
        expected_classifications
    ):
        raise ValueError("root-cause candidate classification 集合不一致")
    ordered_thresholds = (
        (
            interpretation[
                "minimum_actual_amplitude_ratio_for_comparison"
            ],
            interpretation[
                "maximum_actual_amplitude_ratio_for_comparison"
            ],
        ),
        (
            interpretation[
                "minimum_half_odd_ratio_for_local_linear_candidate"
            ],
            interpretation[
                "maximum_half_odd_ratio_for_local_linear_candidate"
            ],
        ),
        (
            interpretation[
                "minimum_half_common_ratio_for_local_linear_candidate"
            ],
            interpretation[
                "maximum_half_common_ratio_for_local_linear_candidate"
            ],
        ),
    )
    if any(
        not 0.0 <= float(lower) < float(upper)
        for lower, upper in ordered_thresholds
    ):
        raise ValueError("root-cause diagnostic interpretation 阈值非法")
    if interpretation["unique_root_cause_claim_allowed"] is not False:
        raise ValueError("root-cause diagnostic 不允许唯一根因 claim")
    authorization = config["authorization_boundary"]
    forbidden_true = {
        "gate_a_retry",
        "gate_a_pass",
        "cross_identity_confirmation_allowed",
        "gate_b_execution_allowed",
        "gate_c_execution_allowed",
        "wrong_key_execution_allowed",
        "observer_execution_allowed",
        "state_dynamics_design_allowed",
        "attack_execution_allowed",
        "pilot_execution_allowed",
        "fixed_fpr_execution_allowed",
        "external_baseline_execution_allowed",
        "paper_claim_allowed",
        "automatic_followup_execution_allowed",
    }
    if any(authorization[field] is not False for field in forbidden_true):
        raise ValueError("root-cause diagnostic 越权打开后续阶段")

    base = load_impulse_observability_config(
        config["base_construction_contract"]["config_path"]
    )
    base_digest = canonical_json_digest(base)
    if (
        base_digest
        != config["base_construction_contract"]["config_digest"]
        or base_digest
        != config["historical_gate_a_source_binding"]["config_digest"]
    ):
        raise ValueError("base Gate A construction config digest 漂移")
    base_identity = base["execution_identity"]
    exact_identity = {
        "prompt_suite_id": "prompt_suite_id",
        "prompt_id": "prompt_id",
        "prompt_text_sha256": "positive_prompt_text_sha256",
        "negative_prompt_text_sha256": "negative_prompt_text_sha256",
        "seed_id": "seed_id",
        "seed_value": "seed_value",
        "generation_model_id": "generation_model_id",
        "generation_model_revision": "generation_model_revision",
        "scheduler_signature": "scheduler_signature",
        "num_inference_steps": "num_inference_steps",
        "num_frames": "num_frames",
        "width": "width",
        "height": "height",
        "guidance_scale": "guidance_scale",
        "fps": "fps",
    }
    mismatches = {
        diagnostic_key: {
            "expected": base_identity[base_key],
            "observed": identity[diagnostic_key],
        }
        for diagnostic_key, base_key in exact_identity.items()
        if identity[diagnostic_key] != base_identity[base_key]
    }
    if mismatches:
        raise ValueError(
            "root-cause diagnostic execution identity 漂移: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )
    observed_config_digest = canonical_json_digest(config)
    if observed_config_digest != FROZEN_DIAGNOSTIC_CONFIG_DIGEST:
        raise ValueError(
            "root-cause diagnostic config 未匹配预冻结 exact contract: "
            f"expected={FROZEN_DIAGNOSTIC_CONFIG_DIGEST} "
            f"observed={observed_config_digest}"
        )
    return config


def build_gate_a_root_cause_diagnostic_plan(
    config: Mapping[str, Any],
) -> tuple[ImpulseProbePlanRecord, ...]:
    """Build the exact six-video half-amplitude plan."""

    if tuple(config["diagnostic_plan"]["plan_order"]) != DIAGNOSTIC_PLAN_IDS:
        raise ValueError("diagnostic plan order 未冻结")
    amplitude = float(config["diagnostic_plan"]["diagnostic_lambda_max"])
    records = (
        ImpulseProbePlanRecord(
            probe_id="clean_start",
            probe_role="clean_order_drift_reference",
            stage_index=None,
            stage_name=None,
            channel_index=None,
            polarity=0,
            nominal_signed_amplitude=0.0,
        ),
        ImpulseProbePlanRecord(
            probe_id="positive_early_flow_channel_0",
            probe_role="signed_root_cause_impulse",
            stage_index=0,
            stage_name="early_flow",
            channel_index=0,
            polarity=1,
            nominal_signed_amplitude=amplitude,
        ),
        ImpulseProbePlanRecord(
            probe_id="negative_early_flow_channel_0",
            probe_role="signed_root_cause_impulse",
            stage_index=0,
            stage_name="early_flow",
            channel_index=0,
            polarity=-1,
            nominal_signed_amplitude=-amplitude,
        ),
        ImpulseProbePlanRecord(
            probe_id="positive_late_flow_channel_0",
            probe_role="signed_root_cause_impulse",
            stage_index=2,
            stage_name="late_flow",
            channel_index=0,
            polarity=1,
            nominal_signed_amplitude=amplitude,
        ),
        ImpulseProbePlanRecord(
            probe_id="negative_late_flow_channel_0",
            probe_role="signed_root_cause_impulse",
            stage_index=2,
            stage_name="late_flow",
            channel_index=0,
            polarity=-1,
            nominal_signed_amplitude=-amplitude,
        ),
        ImpulseProbePlanRecord(
            probe_id="clean_end",
            probe_role="clean_order_drift_reference",
            stage_index=None,
            stage_name=None,
            channel_index=None,
            polarity=0,
            nominal_signed_amplitude=0.0,
        ),
    )
    if tuple(record.probe_id for record in records) != DIAGNOSTIC_PLAN_IDS:
        raise AssertionError("6-video root-cause diagnostic plan 组装失败")
    return records


def _trace_from_record(record: Mapping[str, Any]) -> ActualImpulseExposureTrace:
    tuple_fields = {
        "step_indices",
        "flow_phase_by_step",
        "delta_sigma_by_step",
        "macro_interval_index_by_step",
        "intended_velocity_waveform_by_step",
        "reference_base_velocity_norm_by_step",
        "remaining_control_energy_before_step_by_step",
        "reference_energy_increment_by_step",
        "reference_cumulative_energy_by_step",
        "intended_delta_norm_by_step",
        "actual_velocity_basis_coordinate_by_step",
        "intended_signed_exposure_by_step",
        "actual_signed_exposure_by_step",
        "actual_exposure_vector",
        "delta_norm_by_step",
        "projection_scale_by_step",
        "cumulative_energy_by_step",
        "direction_cosine_by_step",
        "norm_guard_passed_by_step",
        "energy_guard_passed_by_step",
    }
    nested_tuple_fields = {
        "actual_channel_velocity_coordinate_by_step",
        "actual_channel_exposure_by_step",
    }
    payload: dict[str, Any] = {}
    for key, value in record.items():
        if key in tuple_fields:
            payload[key] = tuple(value)
        elif key in nested_tuple_fields:
            payload[key] = tuple(tuple(item) for item in value)
        else:
            payload[key] = value
    return ActualImpulseExposureTrace(**payload)


_HISTORICAL_STEP_RECORD_FIELDS = frozenset(
    {
        "record_version",
        "impulse_probe_id",
        "impulse_flow_step_index",
        "impulse_flow_phase",
        "impulse_delta_sigma",
        "impulse_macro_interval_index",
        "impulse_stage_index",
        "impulse_state_channel_index",
        "impulse_polarity",
        "impulse_intended_velocity_waveform",
        "impulse_reference_base_velocity_norm",
        "impulse_remaining_control_energy_before_step",
        "impulse_reference_energy_increment",
        "impulse_reference_cumulative_energy",
        "impulse_intended_delta_norm",
        "impulse_actual_velocity_basis_coordinate",
        "impulse_actual_channel_velocity_coordinate",
        "impulse_intended_signed_exposure",
        "impulse_actual_signed_exposure",
        "impulse_actual_channel_exposure",
        "impulse_actual_delta_norm",
        "impulse_actual_projection_scale",
        "impulse_cumulative_control_energy",
        "impulse_actual_direction_cosine",
        "impulse_norm_guard_passed",
        "impulse_energy_guard_passed",
        "impulse_direction_guard_passed",
        "impulse_inactive_exact_noop",
        "impulse_finite_precision_projection_status",
        "impulse_finite_precision_projection_scale",
        "impulse_finite_precision_projection_attempt_count",
        "impulse_finite_precision_backoff_count",
        "formal_result",
        "stage_progression_allowed",
    }
)


def _validate_historical_step_records(
    plan: Sequence[ImpulseProbePlanRecord],
    records: Sequence[Mapping[str, Any]],
    traces: Sequence[ActualImpulseExposureTrace],
) -> None:
    """Bind every historical step row to its validated exposure trace."""

    trace_by_probe = {trace.probe_id: trace for trace in traces}
    if len(trace_by_probe) != len(traces):
        raise ValueError("historical Gate A exposure trace probe 重复")
    rows_by_probe: dict[str, list[Mapping[str, Any]]] = {
        record.probe_id: [] for record in plan
    }
    for row in records:
        if set(row) != _HISTORICAL_STEP_RECORD_FIELDS:
            raise ValueError("historical Gate A step record fields 漂移")
        probe_id = str(row["impulse_probe_id"])
        if probe_id not in rows_by_probe:
            raise ValueError("historical Gate A step probe identity 未冻结")
        rows_by_probe[probe_id].append(row)

    if not traces:
        raise ValueError("historical Gate A exposure traces 缺失")
    schedule_reference = traces[0]
    clean_reference: tuple[Mapping[str, Any], ...] | None = None
    for probe in plan:
        rows = tuple(rows_by_probe[probe.probe_id])
        if len(rows) != 8:
            raise ValueError("historical Gate A 每个 probe 必须精确8 step")
        if tuple(int(row["impulse_flow_step_index"]) for row in rows) != tuple(
            range(8)
        ):
            raise ValueError("historical Gate A step index/order 不一致")
        for row in rows:
            if (
                row["record_version"]
                != "output_feature_impulse_observability_construction_v1"
                or row["formal_result"] is not False
                or row["stage_progression_allowed"] is not False
                or row["impulse_norm_guard_passed"] is not True
                or row["impulse_energy_guard_passed"] is not True
                or row["impulse_direction_guard_passed"] is not True
            ):
                raise ValueError("historical Gate A step provenance/guard 未冻结")
            cosine = float(row["impulse_actual_direction_cosine"])
            if not math.isfinite(cosine) or not -1.0 <= cosine <= 1.0:
                raise ValueError("historical Gate A step direction cosine 非法")

        if probe.polarity == 0:
            for step_index, row in enumerate(rows):
                zero_scalars = (
                    "impulse_intended_velocity_waveform",
                    "impulse_intended_delta_norm",
                    "impulse_actual_velocity_basis_coordinate",
                    "impulse_intended_signed_exposure",
                    "impulse_actual_signed_exposure",
                    "impulse_actual_delta_norm",
                    "impulse_actual_projection_scale",
                    "impulse_cumulative_control_energy",
                    "impulse_finite_precision_projection_scale",
                    "impulse_finite_precision_projection_attempt_count",
                    "impulse_finite_precision_backoff_count",
                )
                if (
                    row["impulse_flow_phase"]
                    != schedule_reference.flow_phase_by_step[step_index]
                    or row["impulse_delta_sigma"]
                    != schedule_reference.delta_sigma_by_step[step_index]
                    or row["impulse_macro_interval_index"]
                    != schedule_reference.macro_interval_index_by_step[
                        step_index
                    ]
                    or row["impulse_stage_index"] is not None
                    or row["impulse_state_channel_index"] is not None
                    or int(row["impulse_polarity"]) != 0
                    or any(float(row[key]) != 0.0 for key in zero_scalars)
                    or any(
                        float(value) != 0.0
                        for value in row[
                            "impulse_actual_channel_velocity_coordinate"
                        ]
                    )
                    or any(
                        float(value) != 0.0
                        for value in row["impulse_actual_channel_exposure"]
                    )
                    or row["impulse_inactive_exact_noop"] is not True
                    or row["impulse_finite_precision_projection_status"]
                    != "inactive_exact_noop"
                ):
                    raise ValueError(
                        "historical Gate A clean step 必须是 exact no-op"
                    )
            normalized = tuple(
                {
                    key: value
                    for key, value in row.items()
                    if key != "impulse_probe_id"
                }
                for row in rows
            )
            if clean_reference is None:
                clean_reference = normalized
            elif normalized != clean_reference:
                raise ValueError(
                    "historical Gate A clean A/B step provenance 不一致"
                )
            continue

        trace = trace_by_probe.get(probe.probe_id)
        if trace is None:
            raise ValueError("historical Gate A signed step 缺少 exposure trace")
        scalar_trace_fields = {
            "impulse_flow_step_index": trace.step_indices,
            "impulse_flow_phase": trace.flow_phase_by_step,
            "impulse_delta_sigma": trace.delta_sigma_by_step,
            "impulse_macro_interval_index": trace.macro_interval_index_by_step,
            "impulse_intended_velocity_waveform": (
                trace.intended_velocity_waveform_by_step
            ),
            "impulse_reference_base_velocity_norm": (
                trace.reference_base_velocity_norm_by_step
            ),
            "impulse_remaining_control_energy_before_step": (
                trace.remaining_control_energy_before_step_by_step
            ),
            "impulse_reference_energy_increment": (
                trace.reference_energy_increment_by_step
            ),
            "impulse_reference_cumulative_energy": (
                trace.reference_cumulative_energy_by_step
            ),
            "impulse_intended_delta_norm": trace.intended_delta_norm_by_step,
            "impulse_actual_velocity_basis_coordinate": (
                trace.actual_velocity_basis_coordinate_by_step
            ),
            "impulse_intended_signed_exposure": (
                trace.intended_signed_exposure_by_step
            ),
            "impulse_actual_signed_exposure": (
                trace.actual_signed_exposure_by_step
            ),
            "impulse_actual_delta_norm": trace.delta_norm_by_step,
            "impulse_actual_projection_scale": trace.projection_scale_by_step,
            "impulse_cumulative_control_energy": (
                trace.cumulative_energy_by_step
            ),
            "impulse_actual_direction_cosine": (
                trace.direction_cosine_by_step
            ),
            "impulse_norm_guard_passed": trace.norm_guard_passed_by_step,
            "impulse_energy_guard_passed": trace.energy_guard_passed_by_step,
        }
        nested_trace_fields = {
            "impulse_actual_channel_velocity_coordinate": (
                trace.actual_channel_velocity_coordinate_by_step
            ),
            "impulse_actual_channel_exposure": (
                trace.actual_channel_exposure_by_step
            ),
        }
        for index, row in enumerate(rows):
            if (
                int(row["impulse_stage_index"]) != trace.stage_index
                or int(row["impulse_state_channel_index"])
                != trace.channel_index
                or int(row["impulse_polarity"]) != trace.polarity
            ):
                raise ValueError(
                    "historical Gate A signed step stage/channel/polarity 漂移"
                )
            for field, expected_values in scalar_trace_fields.items():
                if row[field] != expected_values[index]:
                    raise ValueError(
                        f"historical Gate A step/trace 不一致: {field}"
                    )
            for field, expected_values in nested_trace_fields.items():
                if tuple(row[field]) != tuple(expected_values[index]):
                    raise ValueError(
                        f"historical Gate A step/trace 不一致: {field}"
                    )
            active = float(trace.intended_velocity_waveform_by_step[index]) != 0.0
            if active:
                projection_status = str(
                    row["impulse_finite_precision_projection_status"]
                )
                finite_projection_scale = float(
                    row["impulse_finite_precision_projection_scale"]
                )
                attempt_raw = row[
                    "impulse_finite_precision_projection_attempt_count"
                ]
                backoff_raw = row["impulse_finite_precision_backoff_count"]
                integer_counts_ready = bool(
                    not isinstance(attempt_raw, bool)
                    and not isinstance(backoff_raw, bool)
                    and isinstance(attempt_raw, (int, float))
                    and isinstance(backoff_raw, (int, float))
                    and math.isfinite(float(attempt_raw))
                    and math.isfinite(float(backoff_raw))
                    and float(attempt_raw).is_integer()
                    and float(backoff_raw).is_integer()
                )
                projection_attempt_count = (
                    int(attempt_raw) if integer_counts_ready else -1
                )
                backoff_count = (
                    int(backoff_raw) if integer_counts_ready else -1
                )
                direct_projection = bool(
                    projection_status == "direct_actual_delta_pass"
                    and math.isfinite(finite_projection_scale)
                    and finite_projection_scale == 1.0
                    and projection_attempt_count == 1
                    and backoff_count == 0
                )
                bounded_projection = bool(
                    projection_status
                    == "bounded_actual_delta_backoff_pass"
                    and math.isfinite(finite_projection_scale)
                    and 0.0 < finite_projection_scale < 1.0
                    and 1
                    <= backoff_count
                    <= HISTORICAL_PROJECTION_MAX_BACKOFF_COUNT
                    and 1 + backoff_count
                    <= projection_attempt_count
                    <= (
                        1
                        + backoff_count
                        + HISTORICAL_PROJECTION_REFINEMENT_COUNT
                    )
                )
                if (
                    float(row["impulse_intended_delta_norm"]) <= 0.0
                    or float(row["impulse_actual_delta_norm"]) <= 0.0
                    or float(row["impulse_actual_projection_scale"]) <= 0.0
                    or row["impulse_inactive_exact_noop"] is not False
                    or not (direct_projection or bounded_projection)
                ):
                    raise ValueError(
                        "historical Gate A active step FP32 projection provenance 非法"
                    )
            elif (
                float(row["impulse_actual_delta_norm"]) != 0.0
                or float(row["impulse_actual_signed_exposure"]) != 0.0
                or any(
                    float(value) != 0.0
                    for value in row[
                        "impulse_actual_channel_velocity_coordinate"
                    ]
                )
                or any(
                    float(value) != 0.0
                    for value in row["impulse_actual_channel_exposure"]
                )
                or row["impulse_inactive_exact_noop"] is not True
                or row["impulse_finite_precision_projection_status"]
                != "inactive_exact_noop"
            ):
                raise ValueError(
                    "historical Gate A inactive signed step 不是 exact no-op"
                )


def _checkpoint_map(
    records: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], np.ndarray]:
    result: dict[tuple[str, str], np.ndarray] = {}
    for record in records:
        key = (
            str(record["impulse_probe_id"]),
            str(record["impulse_transfer_checkpoint_id"]),
        )
        if key in result:
            raise ValueError(f"historical checkpoint 重复: {key}")
        result[key] = np.asarray(
            record["impulse_transfer_checkpoint_values"],
            dtype=np.float64,
        )
    return result


def _validate_historical_checkpoint_and_feature_records(
    base_config: Mapping[str, Any],
    plan: Sequence[ImpulseProbePlanRecord],
    checkpoint_records: Sequence[Mapping[str, Any]],
    feature_records: Sequence[Mapping[str, Any]],
) -> tuple[Any, dict[str, bool]]:
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
        for record in checkpoint_records
    )
    if observed_sequence != expected_sequence:
        raise ValueError("historical five-checkpoint identity/order 不一致")
    contracts = base_config["transfer_checkpoint_contract"]
    checkpoint_values: dict[tuple[str, str], np.ndarray] = {}
    for plan_index, probe_id in enumerate(expected_ids):
        for checkpoint_offset, checkpoint_id in enumerate(
            PRIMARY_CHECKPOINT_IDS
        ):
            record = checkpoint_records[
                plan_index * len(PRIMARY_CHECKPOINT_IDS)
                + checkpoint_offset
            ]
            values = np.asarray(
                record.get("impulse_transfer_checkpoint_values"),
                dtype=np.float64,
            )
            expected_dimension = int(contracts[checkpoint_id]["dimension"])
            expected_record_id = canonical_json_digest(
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
                    f"historical {probe_id}/{checkpoint_id} checkpoint 不完整"
                )
            checkpoint_values[(probe_id, checkpoint_id)] = values
    if len(feature_records) != len(expected_ids):
        raise ValueError("historical feature records 必须精确为14条")
    feature_ids = tuple(
        str(record.get("impulse_probe_id") or "")
        for record in feature_records
    )
    if feature_ids != expected_ids:
        raise ValueError("historical feature identity/order 不一致")
    schema_digest = str(
        base_config["construction_feature_schema"]["feature_schema_digest"]
    )
    feature_values: list[Sequence[float]] = []
    row_bindings: list[str] = []
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
        expected_record_id = canonical_json_digest(
            {
                key: value
                for key, value in record.items()
                if key != "construction_feature_record_id"
            }
        )
        if (
            record.get("impulse_probe_plan_index") != plan_index
            or record.get("construction_feature_schema_digest")
            != schema_digest
            or record.get("construction_feature_row_binding_digest")
            != expected_binding
            or record.get(
                "construction_feature_row_identity_binding_status"
            )
            != "ready"
            or record.get("construction_feature_record_id")
            != expected_record_id
            or record.get("formal_result") is not False
            or record.get("stage_progression_allowed") is not False
        ):
            raise ValueError(
                f"historical {expected_ids[plan_index]} feature binding 不完整"
            )
        if not np.array_equal(
            values,
            checkpoint_values[
                (expected_ids[plan_index], "T_output_feature")
            ],
        ):
            raise ValueError("historical feature 与 output checkpoint 不一致")
        feature_values.append(values)
        row_bindings.append(expected_binding)
    validated = validate_construction_output_features(
        base_config,
        feature_values,
        feature_schema_digest=schema_digest,
        probe_ids=feature_ids,
        row_binding_digests=row_bindings,
    )
    return validated, {
        checkpoint_id: all(
            (probe_id, checkpoint_id) in checkpoint_values
            for probe_id in expected_ids
        )
        for checkpoint_id in PRIMARY_CHECKPOINT_IDS
    }


def _source_snapshot_digest(root: Path, files: Sequence[Path]) -> str:
    rows = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": _sha256_file(path),
            "size": path.stat().st_size,
        }
        for path in sorted(files)
    ]
    return canonical_json_digest({"files": rows})


def validate_historical_gate_a_failure_source(
    source_root: str | Path,
    config: Mapping[str, Any],
) -> HistoricalGateAFailureSource:
    """Validate the complete immutable 0.12 FAIL package, never recovery."""

    root = Path(source_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if any(root.rglob("*recovery_manifest*.json")) or any(
        path.name.startswith("partial_") for path in root.rglob("*")
    ):
        raise ValueError("historical Gate A source 不得是 recovery/partial 包")
    paths = {
        "decision": _require_unique(
            root,
            "artifacts/output_feature_impulse_observability_decision.json",
        ),
        "manifest": _require_unique(
            root,
            "artifacts/output_feature_impulse_observability_manifest.json",
        ),
        "design": _require_unique(
            root,
            "artifacts/impulse_actual_design_matrix.json",
        ),
        "plan": _require_unique(
            root,
            "records/impulse_generation_plan.jsonl",
        ),
        "generation": _require_unique(
            root,
            "records/impulse_generation_records.jsonl",
        ),
        "steps": _require_unique(
            root,
            "records/impulse_trajectory_step_records.jsonl",
        ),
        "traces": _require_unique(
            root,
            "records/impulse_actual_exposure_traces.jsonl",
        ),
        "checkpoints": _require_unique(
            root,
            "records/impulse_checkpoint_records.jsonl",
        ),
        "features": _require_unique(
            root,
            "records/impulse_feature_records.jsonl",
        ),
    }
    decision = _read_json(paths["decision"])
    manifest = _read_json(paths["manifest"])
    design_record = _read_json(paths["design"])
    binding = config["historical_gate_a_source_binding"]
    exact_decision = {
        "record_version": binding["record_version"],
        "profile_id": binding["profile_id"],
        "impulse_observability_construction_decision": binding["decision"],
        "impulse_sample_internal_observability_gate_ready": False,
        "formal_result": binding["formal_result_required"],
        "stage_progression_allowed": binding[
            "stage_progression_allowed_required"
        ],
        "repository_commit": binding["repository_commit"],
        "config_digest": binding["config_digest"],
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
        "prompt_id": config["execution_identity"]["prompt_id"],
        "positive_prompt_text_sha256": config["execution_identity"][
            "prompt_text_sha256"
        ],
        "seed_id": config["execution_identity"]["seed_id"],
        "generation_seed_random": config["execution_identity"]["seed_value"],
        "diffusers_version": "0.35.2",
    }
    mismatches = {
        key: {"expected": expected, "observed": decision.get(key)}
        for key, expected in exact_decision.items()
        if decision.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            "historical Gate A FAIL decision 未冻结: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )
    exact_manifest = {
        "manifest_kind": binding["manifest_kind"],
        "profile_id": binding["profile_id"],
        "repository_commit": binding["repository_commit"],
        "config_digest": binding["config_digest"],
        "construction_feature_schema_digest": binding[
            "construction_feature_schema_digest"
        ],
        "impulse_waveform_schema_digest": binding[
            "impulse_waveform_schema_digest"
        ],
        "impulse_runtime_adapter_schema_digest": binding[
            "impulse_runtime_adapter_schema_digest"
        ],
        "formal_result": False,
        "stage_progression_allowed": False,
        "same_initial_generator_state_verified": True,
        "prompt_id": config["execution_identity"]["prompt_id"],
        "positive_prompt_text_sha256": config["execution_identity"][
            "prompt_text_sha256"
        ],
        "seed_id": config["execution_identity"]["seed_id"],
        "generation_seed_random": config["execution_identity"]["seed_value"],
        "diffusers_version": "0.35.2",
    }
    manifest_mismatches = {
        key: {"expected": expected, "observed": manifest.get(key)}
        for key, expected in exact_manifest.items()
        if manifest.get(key) != expected
    }
    if manifest_mismatches:
        raise ValueError(
            "historical Gate A manifest 未冻结: "
            + json.dumps(
                manifest_mismatches,
                ensure_ascii=False,
                sort_keys=True,
            )
        )

    base_config = load_impulse_observability_config(
        config["base_construction_contract"]["config_path"]
    )
    expected_plan = build_impulse_triage_plan(base_config)
    plan_rows = _read_jsonl(paths["plan"])
    if tuple(plan_rows) != tuple(asdict(record) for record in expected_plan):
        raise ValueError("historical Gate A 14-video plan identity/order 漂移")
    generations = _read_jsonl(paths["generation"])
    steps = _read_jsonl(paths["steps"])
    trace_rows = _read_jsonl(paths["traces"])
    checkpoint_records = _read_jsonl(paths["checkpoints"])
    feature_records = _read_jsonl(paths["features"])
    if (
        len(generations) != int(binding["generation_record_count"])
        or len(steps) != int(binding["trajectory_step_record_count"])
        or len(trace_rows) != int(binding["actual_exposure_trace_count"])
        or len(checkpoint_records) != int(binding["checkpoint_record_count"])
        or len(feature_records) != int(binding["feature_record_count"])
    ):
        raise ValueError("historical Gate A record counts 不一致")
    expected_ids = tuple(record.probe_id for record in expected_plan)
    observed_ids = tuple(
        str(record.get("impulse_probe_id") or "")
        for record in generations
    )
    identity = config["execution_identity"]
    generator_digests = {
        str(record.get("generation_generator_state_digest_random") or "")
        for record in generations
    }
    if (
        observed_ids != expected_ids
        or any(
            record.get("generation_status") != "success"
            or record.get("prompt_id") != identity["prompt_id"]
            or record.get("positive_prompt_text_sha256")
            != identity["prompt_text_sha256"]
            or record.get("negative_prompt_text_sha256")
            != identity["negative_prompt_text_sha256"]
            or record.get("seed_id") != identity["seed_id"]
            or record.get("generation_seed_random") != identity["seed_value"]
            or record.get("generation_model_id")
            != identity["generation_model_id"]
            or record.get("generation_model_revision")
            != identity["generation_model_revision"]
            or record.get("scheduler_signature")
            != identity["scheduler_signature"]
            or record.get("trajectory_step_count")
            != identity["num_inference_steps"]
            or record.get("endpoint_control_enabled") is not False
            or record.get("formal_result") is not False
            or record.get("stage_progression_allowed") is not False
            or record.get("impulse_generation_record_id")
            != canonical_json_digest(
                {
                    key: value
                    for key, value in record.items()
                    if key != "impulse_generation_record_id"
                }
            )
            for record in generations
        )
        or generator_digests == {""}
        or len(generator_digests) != 1
        or next(iter(generator_digests))
        != manifest["generation_generator_state_digest_random"]
    ):
        raise ValueError("historical Gate A generation identity/RNG 未冻结")
    expected_step_sequence = tuple(
        (probe_id, step_index)
        for probe_id in expected_ids
        for step_index in range(identity["num_inference_steps"])
    )
    observed_step_sequence = tuple(
        (
            str(record.get("impulse_probe_id") or ""),
            int(record.get("impulse_flow_step_index", -1)),
        )
        for record in steps
    )
    if observed_step_sequence != expected_step_sequence:
        raise ValueError("historical Gate A step identity/order 不一致")

    traces = tuple(_trace_from_record(row) for row in trace_rows)
    expected_trace_ids = tuple(
        record.probe_id
        for record in expected_plan
        if record.polarity != 0
    )
    if tuple(trace.probe_id for trace in traces) != expected_trace_ids:
        raise ValueError("historical Gate A exposure trace identity/order 不一致")
    for trace in traces:
        _validate_trace_shape(trace)
        _validate_trace_schedule_and_budget(base_config, trace)
    _validate_historical_step_records(expected_plan, steps, traces)
    design = assemble_actual_design_matrix(base_config, traces)
    if (
        design_record.get("impulse_actual_design_probe_ids")
        != list(design.probe_ids)
        or not np.array_equal(
            np.asarray(
                design_record.get("impulse_actual_design_values"),
                dtype=np.float64,
            ),
            design.values,
        )
        or design_record.get("impulse_actual_design_rank") != design.rank
        or not math.isclose(
            float(
                design_record[
                    "impulse_actual_design_condition_number"
                ]
            ),
            design.condition_number,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or design_record.get("impulse_actual_design_compression_allowed")
        is not True
        or design_record.get("construction_basis_digest")
        != decision["construction_basis_digest"]
    ):
        raise ValueError("historical Gate A A_actual artifact 未由 traces 重建")
    validated_features, checkpoint_ready = (
        _validate_historical_checkpoint_and_feature_records(
            base_config,
            expected_plan,
            checkpoint_records,
            feature_records,
        )
    )
    recomputed_stats = evaluate_gate_a_statistics(
        base_config,
        validated_features,
        design,
        primary_checkpoint_ready=checkpoint_ready,
    )
    expected_gate_statistics = {
        **asdict(recomputed_stats),
        "impulse_sample_internal_observability_gate_ready": (
            recomputed_stats.gate_a_ready
        ),
    }
    if (
        recomputed_stats.gate_a_ready is not False
        or decision.get("gate_a_statistics") != expected_gate_statistics
        or decision.get("gate_a_statistics", {}).get("gate_a_ready")
        is not False
        or decision.get("cross_identity_confirmation_design_allowed")
        is not False
    ):
        raise ValueError("historical Gate A FAIL 不得被重解释为 PASS")

    video_paths: dict[str, Path] = {}
    packaged_videos = tuple(
        path
        for path in root.rglob("*.mp4")
        if path.is_file() and not path.is_symlink()
    )
    if len(packaged_videos) != len(expected_plan):
        raise ValueError("historical Gate A package 必须精确包含14个 MP4")
    for record in generations:
        probe_id = str(record["impulse_probe_id"])
        basename = Path(str(record["video_path"])).name
        video_path = _require_unique(root, f"videos/{basename}")
        if _sha256_file(video_path) != record["video_sha256"]:
            raise ValueError(f"historical video SHA-256 不一致: {probe_id}")
        video_paths[probe_id] = video_path
    expected_manifest_videos = [
        {
            "impulse_probe_id": record["impulse_probe_id"],
            "video_path": record["video_path"],
            "video_sha256": record["video_sha256"],
        }
        for record in generations
    ]
    if manifest.get("video_records") != expected_manifest_videos:
        raise ValueError("historical manifest video records 未绑定 generation")
    checkpoint_values = _checkpoint_map(checkpoint_records)
    exposure_by_pair: dict[str, float] = {}
    trace_map = {trace.probe_id: trace for trace in traces}
    for pair_id in PAIR_IDS:
        positive = trace_map[f"positive_{pair_id}"]
        negative = trace_map[f"negative_{pair_id}"]
        coordinate = positive.stage_index * 2 + positive.channel_index
        positive_value = float(positive.actual_exposure_vector[coordinate])
        negative_value = float(negative.actual_exposure_vector[coordinate])
        if positive_value <= 0.0 or negative_value >= 0.0:
            raise ValueError(f"historical {pair_id} exposure polarity 不一致")
        exposure_by_pair[pair_id] = 0.5 * (
            abs(positive_value) + abs(negative_value)
        )
    snapshot_files = tuple(paths.values()) + tuple(video_paths.values())
    return HistoricalGateAFailureSource(
        root=root,
        source_snapshot_digest=_source_snapshot_digest(root, snapshot_files),
        decision=decision,
        manifest=manifest,
        plan=expected_plan,
        generation_records=generations,
        step_records=steps,
        exposure_traces=traces,
        checkpoint_records=checkpoint_records,
        feature_records=feature_records,
        checkpoint_values=checkpoint_values,
        video_paths=video_paths,
        actual_exposure_by_pair=exposure_by_pair,
    )


def compute_paired_response_statistics(
    *,
    pair_id: str,
    checkpoint_id: str,
    clean_start: Sequence[float] | np.ndarray,
    clean_end: Sequence[float] | np.ndarray,
    positive: Sequence[float] | np.ndarray,
    negative: Sequence[float] | np.ndarray,
    denominator_epsilon: float,
) -> PairedResponseStatistics:
    """Compute the frozen clean-centered odd/common decomposition."""

    arrays = tuple(
        np.asarray(value, dtype=np.float64).reshape(-1)
        for value in (clean_start, clean_end, positive, negative)
    )
    if len({array.shape for array in arrays}) != 1:
        raise ValueError(f"{pair_id}/{checkpoint_id} response shapes 不一致")
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError(f"{pair_id}/{checkpoint_id} response 含非有限值")
    clean_a, clean_b, plus, minus = arrays
    intercept = 0.5 * (clean_a + clean_b)
    delta_plus = plus - intercept
    delta_minus = minus - intercept
    odd = 0.5 * (delta_plus - delta_minus)
    common = 0.5 * (delta_plus + delta_minus)
    plus_norm = float(np.linalg.norm(delta_plus))
    minus_norm = float(np.linalg.norm(delta_minus))
    odd_norm = float(np.linalg.norm(odd))
    common_norm = float(np.linalg.norm(common))
    clean_distance = float(np.linalg.norm(clean_a - clean_b))
    common_ratio = (
        None
        if odd_norm <= denominator_epsilon
        else common_norm / odd_norm
    )
    cosine_denominator = plus_norm * minus_norm
    cosine = (
        None
        if cosine_denominator <= denominator_epsilon
        else float(delta_plus @ (-delta_minus) / cosine_denominator)
    )
    residual_denominator = plus_norm + minus_norm
    residual = (
        None
        if residual_denominator <= denominator_epsilon
        else float(np.linalg.norm(delta_plus + delta_minus))
        / residual_denominator
    )
    finite_values = (
        clean_distance,
        plus_norm,
        minus_norm,
        odd_norm,
        common_norm,
    ) + tuple(
        value
        for value in (common_ratio, cosine, residual)
        if value is not None
    )
    return PairedResponseStatistics(
        pair_id=pair_id,
        checkpoint_id=checkpoint_id,
        clean_distance=clean_distance,
        positive_centered_norm=plus_norm,
        negative_centered_norm=minus_norm,
        odd_norm=odd_norm,
        common_norm=common_norm,
        common_odd_ratio=common_ratio,
        antisymmetry_cosine=cosine,
        antisymmetry_residual=residual,
        finite=all(math.isfinite(value) for value in finite_values),
    )


def compute_paired_response_statistics_from_gram(
    *,
    pair_id: str,
    checkpoint_id: str,
    gram_matrix: Sequence[Sequence[float]] | np.ndarray,
    row_ids: Sequence[str],
    denominator_epsilon: float,
) -> PairedResponseStatistics:
    """Compute exact quadratic odd/common statistics from a frozen Gram matrix."""

    gram = np.asarray(gram_matrix, dtype=np.float64)
    ids = tuple(str(value) for value in row_ids)
    if (
        ids != DIAGNOSTIC_PLAN_IDS
        or gram.shape != (len(ids), len(ids))
        or not np.all(np.isfinite(gram))
        or not np.allclose(gram, gram.T, rtol=1e-10, atol=1e-8)
    ):
        raise ValueError(f"{checkpoint_id} full-representation Gram 未冻结")
    index = {probe_id: offset for offset, probe_id in enumerate(ids)}
    coefficients: dict[str, np.ndarray] = {}
    clean_difference = np.zeros(len(ids), dtype=np.float64)
    clean_difference[index["clean_start"]] = 1.0
    clean_difference[index["clean_end"]] = -1.0
    positive_id = f"positive_{pair_id}"
    negative_id = f"negative_{pair_id}"
    delta_plus = np.zeros(len(ids), dtype=np.float64)
    delta_minus = np.zeros(len(ids), dtype=np.float64)
    for value in (delta_plus, delta_minus):
        value[index["clean_start"]] = -0.5
        value[index["clean_end"]] = -0.5
    delta_plus[index[positive_id]] = 1.0
    delta_minus[index[negative_id]] = 1.0
    coefficients["clean"] = clean_difference
    coefficients["plus"] = delta_plus
    coefficients["minus"] = delta_minus
    coefficients["odd"] = 0.5 * (delta_plus - delta_minus)
    coefficients["common"] = 0.5 * (delta_plus + delta_minus)

    def squared_norm(coefficient: np.ndarray) -> float:
        value = float(coefficient @ gram @ coefficient)
        if value < 0.0 and abs(value) <= 1e-7:
            value = 0.0
        if value < 0.0:
            raise ValueError(f"{checkpoint_id} Gram 二次型为负")
        return value

    clean_distance = math.sqrt(squared_norm(coefficients["clean"]))
    plus_norm = math.sqrt(squared_norm(coefficients["plus"]))
    minus_norm = math.sqrt(squared_norm(coefficients["minus"]))
    odd_norm = math.sqrt(squared_norm(coefficients["odd"]))
    common_norm = math.sqrt(squared_norm(coefficients["common"]))
    common_ratio = (
        None
        if odd_norm <= denominator_epsilon
        else common_norm / odd_norm
    )
    cosine_denominator = plus_norm * minus_norm
    cosine = (
        None
        if cosine_denominator <= denominator_epsilon
        else float(
            coefficients["plus"]
            @ gram
            @ (-coefficients["minus"])
            / cosine_denominator
        )
    )
    residual_denominator = plus_norm + minus_norm
    residual = (
        None
        if residual_denominator <= denominator_epsilon
        else 2.0 * common_norm / residual_denominator
    )
    finite_values = (
        clean_distance,
        plus_norm,
        minus_norm,
        odd_norm,
        common_norm,
    ) + tuple(
        value
        for value in (common_ratio, cosine, residual)
        if value is not None
    )
    return PairedResponseStatistics(
        pair_id=pair_id,
        checkpoint_id=checkpoint_id,
        clean_distance=clean_distance,
        positive_centered_norm=plus_norm,
        negative_centered_norm=minus_norm,
        odd_norm=odd_norm,
        common_norm=common_norm,
        common_odd_ratio=common_ratio,
        antisymmetry_cosine=cosine,
        antisymmetry_residual=residual,
        finite=all(math.isfinite(value) for value in finite_values),
    )


def compare_half_to_historical_full(
    config: Mapping[str, Any],
    *,
    half: PairedResponseStatistics,
    full: PairedResponseStatistics,
    half_actual_amplitude: float,
    full_actual_amplitude: float,
) -> ScalingComparison:
    """Apply predeclared amplitude-scaling rules without selecting a result."""

    if half.pair_id != full.pair_id or half.checkpoint_id != full.checkpoint_id:
        raise ValueError("half/full pair/checkpoint identity 不一致")
    thresholds = config["interpretation_contract"]
    minimum_norm = float(thresholds["minimum_comparable_norm"])
    amplitude_ratio = (
        None
        if full_actual_amplitude <= minimum_norm
        else half_actual_amplitude / full_actual_amplitude
    )
    amplitude_ready = bool(
        amplitude_ratio is not None
        and math.isfinite(amplitude_ratio)
        and float(
            thresholds["minimum_actual_amplitude_ratio_for_comparison"]
        )
        <= amplitude_ratio
        <= float(
            thresholds["maximum_actual_amplitude_ratio_for_comparison"]
        )
    )
    odd_ratio = (
        None
        if full.odd_norm <= minimum_norm
        else half.odd_norm / full.odd_norm
    )
    common_ratio = (
        None
        if full.common_norm <= minimum_norm
        else half.common_norm / full.common_norm
    )
    normalized_odd = (
        None
        if odd_ratio is None
        or amplitude_ratio is None
        or amplitude_ratio <= minimum_norm
        else odd_ratio / amplitude_ratio
    )
    normalized_common = (
        None
        if common_ratio is None
        or amplitude_ratio is None
        or amplitude_ratio <= minimum_norm
        else common_ratio / amplitude_ratio**2
    )
    cosine_improvement = (
        None
        if half.antisymmetry_cosine is None
        or full.antisymmetry_cosine is None
        else half.antisymmetry_cosine - full.antisymmetry_cosine
    )
    residual_improvement = (
        None
        if half.antisymmetry_residual is None
        or full.antisymmetry_residual is None
        else full.antisymmetry_residual - half.antisymmetry_residual
    )
    residual_fraction_ready = bool(
        half.antisymmetry_residual is not None
        and full.antisymmetry_residual is not None
        and (
            (
                full.antisymmetry_residual > minimum_norm
                and half.antisymmetry_residual
                <= float(
                    thresholds[
                        "maximum_half_residual_fraction_for_local_linear_candidate"
                    ]
                )
                * full.antisymmetry_residual
            )
            or (
                full.antisymmetry_residual <= minimum_norm
                and half.antisymmetry_residual <= minimum_norm
            )
        )
    )
    local_linear_ready = bool(
        amplitude_ready
        and odd_ratio is not None
        and common_ratio is not None
        and float(
            thresholds[
                "minimum_half_odd_ratio_for_local_linear_candidate"
            ]
        )
        <= odd_ratio
        <= float(
            thresholds[
                "maximum_half_odd_ratio_for_local_linear_candidate"
            ]
        )
        and float(
            thresholds[
                "minimum_half_common_ratio_for_local_linear_candidate"
            ]
        )
        <= common_ratio
        <= float(
            thresholds[
                "maximum_half_common_ratio_for_local_linear_candidate"
            ]
        )
        and cosine_improvement is not None
        and cosine_improvement
        >= float(thresholds["minimum_antisymmetry_cosine_improvement"])
        and residual_improvement is not None
        and residual_improvement
        >= float(thresholds["minimum_antisymmetry_residual_improvement"])
        and residual_fraction_ready
    )
    status = (
        "ready"
        if amplitude_ready
        and odd_ratio is not None
        and common_ratio is not None
        else "indeterminate_unresolved_amplitude_or_response_floor"
    )
    return ScalingComparison(
        pair_id=half.pair_id,
        checkpoint_id=half.checkpoint_id,
        actual_amplitude_ratio=amplitude_ratio,
        actual_amplitude_ratio_ready=amplitude_ready,
        odd_ratio_to_full=odd_ratio,
        common_ratio_to_full=common_ratio,
        normalized_odd_scaling=normalized_odd,
        normalized_common_scaling=normalized_common,
        antisymmetry_cosine_improvement=cosine_improvement,
        antisymmetry_residual_improvement=residual_improvement,
        local_linear_scaling_ready=local_linear_ready,
        comparison_status=status,
    )


def classify_root_cause_candidates(
    config: Mapping[str, Any],
    *,
    half_statistics: Mapping[tuple[str, str], PairedResponseStatistics],
    scaling: Mapping[tuple[str, str], ScalingComparison],
    half_actual_amplitude_by_pair: Mapping[str, float],
) -> dict[str, Any]:
    """Return overlapping candidate explanations, never a unique cause."""

    thresholds = config["interpretation_contract"]
    output_keys = tuple(
        (pair_id, "T_output_feature_256d") for pair_id in PAIR_IDS
    )
    local_linear = all(
        scaling.get(key) is not None
        and scaling[key].local_linear_scaling_ready
        for key in output_keys
    )
    floor_or_plateau = any(
        scaling.get(key) is not None
        and scaling[key].actual_amplitude_ratio_ready
        and scaling[key].odd_ratio_to_full is not None
        and (
            scaling[key].odd_ratio_to_full
            <= float(
                thresholds[
                    "maximum_half_odd_ratio_for_floor_collapse_candidate"
                ]
            )
            or scaling[key].odd_ratio_to_full
            >= float(
                thresholds[
                    "minimum_half_odd_ratio_for_plateau_candidate"
                ]
            )
        )
        for key in output_keys
    )
    early = half_statistics.get(
        ("early_flow_channel_0", "T_latent_six_basis")
    )
    late = half_statistics.get(
        ("late_flow_channel_0", "T_latent_six_basis")
    )
    feedback = False
    mismatch = False
    if early is not None and late is not None:
        cosine_gap = (
            None
            if early.antisymmetry_cosine is None
            or late.antisymmetry_cosine is None
            else late.antisymmetry_cosine - early.antisymmetry_cosine
        )
        common_ratio_factor = (
            None
            if early.common_odd_ratio is None
            or late.common_odd_ratio is None
            or late.common_odd_ratio
            <= float(thresholds["minimum_comparable_norm"])
            else early.common_odd_ratio / late.common_odd_ratio
        )
        early_amplitude = float(
            half_actual_amplitude_by_pair.get(
                "early_flow_channel_0",
                0.0,
            )
        )
        late_amplitude = float(
            half_actual_amplitude_by_pair.get(
                "late_flow_channel_0",
                0.0,
            )
        )
        exposure_normalized_odd_ratio = (
            None
            if early_amplitude <= 0.0
            or late_amplitude <= 0.0
            or late.odd_norm <= 0.0
            else (early.odd_norm / early_amplitude)
            / (late.odd_norm / late_amplitude)
        )
        feedback = bool(
            (
                cosine_gap is not None
                and cosine_gap
                >= float(
                    thresholds[
                        "minimum_early_late_antisymmetry_cosine_gap_for_feedback_candidate"
                    ]
                )
            )
            or (
                common_ratio_factor is not None
                and common_ratio_factor
                >= float(
                    thresholds[
                        "minimum_early_late_common_odd_ratio_factor_for_feedback_candidate"
                    ]
                )
            )
            or (
                exposure_normalized_odd_ratio is not None
                and exposure_normalized_odd_ratio
                <= float(
                    thresholds[
                        "maximum_early_late_exposure_normalized_odd_transfer_ratio_for_feedback_candidate"
                    ]
                )
            )
        )
        late_signed = bool(
            late.antisymmetry_cosine is not None
            and late.antisymmetry_cosine
            >= float(
                thresholds[
                    "minimum_late_latent_antisymmetry_cosine_for_signed_response"
                ]
            )
            and late.antisymmetry_residual is not None
            and late.antisymmetry_residual
            <= float(
                thresholds[
                    "maximum_late_latent_antisymmetry_residual_for_signed_response"
                ]
            )
            and late.common_odd_ratio is not None
            and late.common_odd_ratio
            <= float(
                thresholds[
                    "maximum_late_latent_common_odd_ratio_for_signed_response"
                ]
            )
        )
        post_latent = tuple(
            half_statistics.get(("late_flow_channel_0", checkpoint_id))
            for checkpoint_id in (
                "T_decoded_48d",
                "T_saved_video_full_rgb24",
                "T_output_feature_256d",
            )
        )
        mismatch = bool(
            late_signed
            and any(
                item is not None
                and (
                    item.common_odd_ratio is None
                    or item.common_odd_ratio
                    >= float(
                        thresholds[
                            "minimum_post_latent_common_odd_ratio_for_mismatch_candidate"
                        ]
                    )
                    or item.antisymmetry_cosine is None
                    or item.antisymmetry_cosine
                    < float(
                        thresholds[
                            "maximum_post_latent_antisymmetry_cosine_for_mismatch_candidate"
                        ]
                    )
                )
                for item in post_latent
            )
        )
    candidates = [
        name
        for name, ready in (
            ("local_linear_support_candidate", local_linear),
            (
                "quantization_or_observation_floor_candidate",
                floor_or_plateau,
            ),
            ("feedback_nonlinearity_candidate", feedback),
            ("carrier_decoder_feature_mismatch_candidate", mismatch),
        )
        if ready
    ]
    if len(candidates) > 1:
        candidates.append("multiple_factors_candidate")
    if not candidates:
        candidates.append("indeterminate")
    return {
        "candidate_classifications": candidates,
        "local_linear_support_candidate": local_linear,
        "quantization_or_observation_floor_candidate": floor_or_plateau,
        "feedback_nonlinearity_candidate": feedback,
        "carrier_decoder_feature_mismatch_candidate": mismatch,
        "half_actual_amplitude_by_pair": {
            key: float(value)
            for key, value in half_actual_amplitude_by_pair.items()
        },
        "unique_root_cause_claim_allowed": False,
        "classification_status": (
            "multiple_candidates"
            if len(candidates) > 1
            and candidates[-1] == "multiple_factors_candidate"
            else candidates[0]
        ),
    }
