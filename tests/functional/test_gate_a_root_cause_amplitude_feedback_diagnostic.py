from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest

import evaluation.protocol.gate_a_root_cause_amplitude_feedback_contract as contract_module
import experiments.generative_video_model_probe.output_feature_impulse_observability_construction as construction_module
from evaluation.protocol.gate_a_root_cause_amplitude_feedback_contract import (
    COMPARABLE_CHECKPOINT_IDS,
    DEFAULT_DIAGNOSTIC_CONFIG_PATH,
    DIAGNOSTIC_PLAN_IDS,
    FROZEN_DIAGNOSTIC_CONFIG_DIGEST,
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
    ActualImpulseExposureTrace,
    ConstructionStageBasis,
    canonical_json_digest,
    compute_intended_impulse_control,
)
from experiments.generative_video_model_probe.gate_a_root_cause_amplitude_feedback_diagnostic import (
    DiagnosticArrayCapture,
    PAIRED_RESPONSE_RECORD_FIELDS,
    SCALING_COMPARISON_RECORD_FIELDS,
    _apply_clean_pollution_fail_closed,
    _checkpoint_view,
    _diagnostic_runtime_config,
    _gram_matrix_from_array_paths,
    _validate_diagnostic_generation_batch,
    _load_latent_artifact,
    _load_npy_memmap,
    _pair_statistics_from_views,
    _write_root_cause_metric_records,
    run_gate_a_root_cause_amplitude_feedback_diagnostic,
)
from experiments.generative_video_model_probe.output_feature_impulse_observability_construction import (
    ConstructionGenerationBatch,
    apply_numpy_float32_impulse,
    run_output_feature_impulse_observability_construction,
)
from workflows.colab_test_request import (
    GATE_A_ROOT_CAUSE_AMPLITUDE_FEEDBACK_DIAGNOSTIC_TEST_ID,
    REQUEST_SCHEMA_VERSION,
    build_colab_test_dry_run_plan,
    load_colab_test_request,
    package_colab_test_recovery_bundle,
    run_colab_test_request,
)


def _config() -> dict:
    return load_gate_a_root_cause_diagnostic_config(
        DEFAULT_DIAGNOSTIC_CONFIG_PATH
    )


def _write_config(path: Path, config: dict) -> None:
    path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


@pytest.mark.quick
def test_config_and_plan_freeze_six_videos_and_half_amplitude() -> None:
    config = _config()
    plan = build_gate_a_root_cause_diagnostic_plan(config)
    assert tuple(record.probe_id for record in plan) == DIAGNOSTIC_PLAN_IDS
    assert [record.nominal_signed_amplitude for record in plan] == [
        0.0,
        0.06,
        -0.06,
        0.06,
        -0.06,
        0.0,
    ]
    assert config["authorization_boundary"]["gate_a_pass"] is False
    assert config["authorization_boundary"]["observer_execution_allowed"] is False
    assert config["formal_result"] is False
    assert config["stage_progression_allowed"] is False


@pytest.mark.quick
def test_half_amplitude_uses_same_strict_fp32_projection_guards() -> None:
    config = _config()
    runtime = _diagnostic_runtime_config(config)
    assert runtime["actual_exposure_contract"]["lambda_max"] == 0.06
    assert (
        load_gate_a_root_cause_diagnostic_config()[
            "base_construction_contract"
        ]["config_digest"]
        == config["historical_gate_a_source_binding"]["config_digest"]
    )
    rng = np.random.default_rng(19)
    base = rng.standard_normal(100_000).astype(np.float32)
    direction = rng.standard_normal(100_000).astype(np.float32)
    direction /= np.linalg.norm(direction.astype(np.float64))
    budget = (
        float(np.linalg.norm(base.astype(np.float64)))
        * float(runtime["actual_exposure_contract"]["velocity_norm_ratio_budget"])
        * float(runtime["actual_exposure_contract"]["lambda_max"])
    )
    delta_sigma = float(
        runtime["flow_schedule_contract"]["delta_sigma_by_step"][3]
    )
    result = apply_numpy_float32_impulse(
        base,
        direction,
        signed_delta_norm=budget,
        delta_sigma=delta_sigma,
        norm_budget=budget,
        remaining_energy=delta_sigma**2 * budget**2,
        minimum_direction_cosine=float(
            runtime["actual_exposure_contract"]["minimum_direction_cosine"]
        ),
    )
    assert result.selection is not None
    selected = result.selection.evaluation
    assert 0.0 < selected.actual_delta_norm <= budget
    assert selected.energy_increment <= delta_sigma**2 * budget**2
    assert selected.direction_cosine is not None
    assert selected.direction_cosine >= 0.999


def _half_trace(
    runtime: dict,
    probe,
    *,
    scale: float = 0.999,
) -> ActualImpulseExposureTrace:
    schedule = runtime["flow_schedule_contract"]
    exposure = runtime["actual_exposure_contract"]
    target = int(probe.stage_index) * 2 + int(probe.channel_index)
    waveform = tuple(
        schedule["temporal_waveform_by_macro_interval"][
            int(probe.stage_index)
        ]
    )
    previous_reference = 0.0
    previous_control = 0.0
    base_norms = tuple(100.0 + index for index in range(8))
    remaining = []
    reference_increment = []
    reference_cumulative = []
    intended_norms = []
    intended_exposures = []
    target_coordinates = []
    channel_coordinates = []
    actual_exposures = []
    actual_signed = []
    delta_norms = []
    projection_scales = []
    cumulative = []
    direction_cosines = []
    norm_guards = []
    energy_guards = []
    for step_index, delta_sigma in enumerate(
        schedule["delta_sigma_by_step"]
    ):
        control = compute_intended_impulse_control(
            probe_state_update_polarity=probe.polarity,
            temporal_waveform=waveform[step_index],
            delta_sigma=float(delta_sigma),
            base_velocity_norm=base_norms[step_index],
            cumulative_control_energy=previous_control,
            cumulative_reference_energy=previous_reference,
            remaining_step_count=8 - step_index,
            lambda_max=float(exposure["lambda_max"]),
            velocity_norm_ratio_budget=float(
                exposure["velocity_norm_ratio_budget"]
            ),
            flow_energy_budget_ratio=float(
                exposure["flow_energy_budget_ratio"]
            ),
        )
        coordinates = [0.0] * 6
        actual_norm = control.intended_delta_norm * scale
        coordinates[target] = control.signed_velocity_coordinate * scale
        state_exposure = [
            float(delta_sigma) * value for value in coordinates
        ]
        increment = float(delta_sigma) ** 2 * actual_norm**2
        previous_reference += control.reference_energy_increment
        previous_control += increment
        remaining.append(control.remaining_control_energy)
        reference_increment.append(control.reference_energy_increment)
        reference_cumulative.append(previous_reference)
        intended_norms.append(control.intended_delta_norm)
        intended_exposures.append(control.signed_state_update_exposure)
        target_coordinates.append(coordinates[target])
        channel_coordinates.append(tuple(coordinates))
        actual_exposures.append(tuple(state_exposure))
        actual_signed.append(state_exposure[target])
        delta_norms.append(actual_norm)
        projection_scales.append(
            scale if control.intended_delta_norm > 0.0 else 0.0
        )
        cumulative.append(previous_control)
        direction_cosines.append(1.0)
        norm_guards.append(True)
        energy_guards.append(True)
    return ActualImpulseExposureTrace(
        probe_id=probe.probe_id,
        stage_index=int(probe.stage_index),
        channel_index=int(probe.channel_index),
        polarity=probe.polarity,
        step_indices=tuple(schedule["step_indices"]),
        flow_phase_by_step=tuple(schedule["flow_phase_by_step"]),
        delta_sigma_by_step=tuple(schedule["delta_sigma_by_step"]),
        macro_interval_index_by_step=tuple(
            schedule["macro_interval_index_by_step"]
        ),
        intended_velocity_waveform_by_step=waveform,
        reference_base_velocity_norm_by_step=base_norms,
        remaining_control_energy_before_step_by_step=tuple(remaining),
        reference_energy_increment_by_step=tuple(reference_increment),
        reference_cumulative_energy_by_step=tuple(reference_cumulative),
        intended_delta_norm_by_step=tuple(intended_norms),
        actual_velocity_basis_coordinate_by_step=tuple(
            target_coordinates
        ),
        actual_channel_velocity_coordinate_by_step=tuple(
            channel_coordinates
        ),
        intended_signed_exposure_by_step=tuple(intended_exposures),
        actual_signed_exposure_by_step=tuple(actual_signed),
        actual_channel_exposure_by_step=tuple(actual_exposures),
        actual_exposure_vector=tuple(
            np.asarray(actual_exposures, dtype=np.float64).sum(axis=0)
        ),
        delta_norm_by_step=tuple(delta_norms),
        projection_scale_by_step=tuple(projection_scales),
        cumulative_energy_by_step=tuple(cumulative),
        direction_cosine_by_step=tuple(direction_cosines),
        norm_guard_passed_by_step=tuple(norm_guards),
        energy_guard_passed_by_step=tuple(energy_guards),
        waveform_schema_digest=schedule["waveform_schema_digest"],
        runtime_adapter_schema_digest=runtime[
            "runtime_adapter_contract"
        ]["adapter_schema_digest"],
        basis_digest="a" * 64,
    )


def _diagnostic_generation_fixture(
    tmp_path: Path,
) -> tuple[
    dict,
    tuple,
    ConstructionGenerationBatch,
    HistoricalGateAFailureSource,
    ConstructionStageBasis,
]:
    config = _config()
    runtime = _diagnostic_runtime_config(config)
    plan = build_gate_a_root_cause_diagnostic_plan(config)
    generation_records = []
    step_records = []
    traces = []
    for plan_index, probe in enumerate(plan):
        video = tmp_path / "videos" / f"{plan_index:02d}_{probe.probe_id}.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(f"video-{plan_index}".encode())
        identity = config["execution_identity"]
        generation_records.append(
            {
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
                "generation_generator_state_digest_random": "b" * 64,
                "generation_model_id": identity["generation_model_id"],
                "generation_model_revision": identity[
                    "generation_model_revision"
                ],
                "scheduler_signature": identity["scheduler_signature"],
                "prompt_id": identity["prompt_id"],
                "positive_prompt_text_sha256": identity[
                    "prompt_text_sha256"
                ],
                "negative_prompt_text_sha256": identity[
                    "negative_prompt_text_sha256"
                ],
                "seed_id": identity["seed_id"],
                "generation_seed_random": identity["seed_value"],
                "trajectory_step_count": 8,
                "endpoint_control_enabled": False,
                "video_path": str(video),
                "video_sha256": __import__("hashlib").sha256(
                    video.read_bytes()
                ).hexdigest(),
            }
        )
        if probe.polarity:
            trace = _half_trace(runtime, probe)
            traces.append(trace)
        for step_index in range(8):
            step_records.append(
                {
                    "impulse_probe_id": probe.probe_id,
                    "impulse_flow_step_index": step_index,
                    "impulse_inactive_exact_noop": probe.polarity == 0,
                    "impulse_actual_delta_norm": (
                        0.0
                        if probe.polarity == 0
                        else traces[-1].delta_norm_by_step[step_index]
                    ),
                    "impulse_actual_channel_exposure": (
                        [0.0] * 6
                        if probe.polarity == 0
                        else list(
                            traces[-1].actual_channel_exposure_by_step[
                                step_index
                            ]
                        )
                    ),
                }
            )
    amplitudes = {}
    for pair_id in PAIR_IDS:
        positive = next(
            trace for trace in traces if trace.probe_id == f"positive_{pair_id}"
        )
        coordinate = positive.stage_index * 2 + positive.channel_index
        amplitudes[pair_id] = abs(
            positive.actual_exposure_vector[coordinate]
        )
    historical = HistoricalGateAFailureSource(
        root=tmp_path,
        source_snapshot_digest="c" * 64,
        decision={"construction_basis_digest": "a" * 64},
        manifest={},
        plan=(),
        generation_records=(),
        step_records=(),
        exposure_traces=(),
        checkpoint_records=(),
        feature_records=(),
        checkpoint_values={},
        video_paths={},
        actual_exposure_by_pair={
            key: 2.0 * value for key, value in amplitudes.items()
        },
    )
    batch = ConstructionGenerationBatch(
        generation_records=tuple(generation_records),
        trajectory_step_records=tuple(step_records),
        exposure_traces=tuple(traces),
        checkpoint_records=tuple({} for _ in range(18)),
        diagnostic_capture_records=tuple({} for _ in range(6)),
    )
    basis = ConstructionStageBasis(
        values=np.zeros((1, 6), dtype=np.float32),
        basis_digest="a" * 64,
        latent_layout_shape=(1, 16, 9, 40, 64),
        wrong_key_candidate_index=None,
    )
    return config, plan, batch, historical, basis


@pytest.mark.quick
def test_six_video_batch_validates_full_eight_step_half_exposure(
    tmp_path: Path,
) -> None:
    config, plan, batch, historical, basis = (
        _diagnostic_generation_fixture(tmp_path)
    )
    amplitudes = _validate_diagnostic_generation_batch(
        _diagnostic_runtime_config(config),
        config,
        plan,
        batch,
        historical=historical,
        basis=basis,
    )
    assert amplitudes["early_flow_channel_0"] > 0.0
    assert amplitudes["late_flow_channel_0"] > 0.0
    assert all(
        amplitudes[key] / historical.actual_exposure_by_pair[key]
        == pytest.approx(0.5)
        for key in PAIR_IDS
    )


@pytest.mark.quick
def test_active_half_waveform_cannot_be_silently_recorded_as_noop(
    tmp_path: Path,
) -> None:
    config, plan, batch, historical, basis = (
        _diagnostic_generation_fixture(tmp_path)
    )
    trace = batch.exposure_traces[0]
    mutated = replace(
        trace,
        delta_norm_by_step=tuple(0.0 for _ in range(8)),
    )
    batch = replace(
        batch,
        exposure_traces=(mutated,) + batch.exposure_traces[1:],
    )
    with pytest.raises(ValueError, match="positive_early_flow_channel_0"):
        _validate_diagnostic_generation_batch(
            _diagnostic_runtime_config(config),
            config,
            plan,
            batch,
            historical=historical,
            basis=basis,
        )


@pytest.mark.quick
@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda value: value["diagnostic_plan"].__setitem__(
                "diagnostic_lambda_max",
                0.061,
            ),
            "6-video/half-amplitude",
        ),
        (
            lambda value: value["diagnostic_plan"]["plan_order"].reverse(),
            "6-video/half-amplitude",
        ),
        (
            lambda value: value["authorization_boundary"].__setitem__(
                "gate_a_pass",
                True,
            ),
            "越权",
        ),
        (
            lambda value: value["execution_identity"].__setitem__(
                "prompt_text",
                value["execution_identity"]["prompt_text"] + " altered",
            ),
            "SHA-256",
        ),
    ],
)
def test_config_mutations_fail_closed(
    tmp_path: Path,
    mutation,
    match: str,
) -> None:
    config = _config()
    mutation(config)
    path = tmp_path / "config.json"
    _write_config(path, config)
    with pytest.raises(ValueError, match=match):
        load_gate_a_root_cause_diagnostic_config(path)


def _set_nested(value: dict, path: tuple[str, ...], replacement) -> None:
    cursor = value
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = replacement


@pytest.mark.quick
@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (
            (
                "interpretation_contract",
                "minimum_antisymmetry_cosine_improvement",
            ),
            -999.0,
        ),
        (
            (
                "interpretation_contract",
                "minimum_early_late_common_odd_ratio_factor_for_feedback_candidate",
            ),
            0.0,
        ),
        (
            (
                "interpretation_contract",
                "minimum_post_latent_common_odd_ratio_for_mismatch_candidate",
            ),
            0.0,
        ),
        (
            ("interpretation_contract", "ideal_half_common_ratio"),
            0.3,
        ),
        (
            ("interpretation_contract", "unique_root_cause_claim_allowed"),
            True,
        ),
        (
            ("checkpoint_contract", "odd_formula"),
            "caller_selected_formula",
        ),
        (
            ("checkpoint_contract", "diagnostic_only_checkpoint_ids"),
            ["T_final_latent_full"],
        ),
        (
            ("checkpoint_contract", "full_final_latent_storage"),
            "caller_selected_storage",
        ),
        (
            ("checkpoint_contract", "full_final_latent_shape"),
            [1, 16, 8, 40, 64],
        ),
        (
            ("checkpoint_contract", "norm_denominator_epsilon"),
            2e-15,
        ),
        (
            ("diagnostic_role",),
            "gate_a_retry",
        ),
        (
            ("claim_support_status",),
            "method_evidence",
        ),
        (
            ("diagnostic_plan", "clean_repeat_role"),
            "formal_noise_distribution",
        ),
        (
            (
                "historical_gate_a_source_binding",
                "repository_commit",
            ),
            "0" * 40,
        ),
        (
            ("base_construction_contract", "config_path"),
            "configs/protocol/caller_selected.json",
        ),
        (
            (
                "authorization_boundary",
                "frozen_feedback_diagnostic_design_may_be_allowed",
            ),
            False,
        ),
        (
            (
                "authorization_boundary",
                "carrier_feature_redesign_may_be_allowed",
            ),
            False,
        ),
    ],
)
def test_every_interpretation_and_semantic_boundary_is_exactly_frozen(
    tmp_path: Path,
    path: tuple[str, ...],
    replacement,
) -> None:
    config = _config()
    assert canonical_json_digest(config) == FROZEN_DIAGNOSTIC_CONFIG_DIGEST
    _set_nested(config, path, replacement)
    mutated_path = tmp_path / "mutated.json"
    _write_config(mutated_path, config)
    with pytest.raises((FileNotFoundError, ValueError)):
        load_gate_a_root_cause_diagnostic_config(mutated_path)


def _mutated_json_value(value):
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 1.0
    if isinstance(value, str):
        return value + "_mutated"
    if isinstance(value, list):
        return list(reversed(value)) if len(value) > 1 else value + value
    raise AssertionError(f"unsupported frozen config value: {value!r}")


@pytest.mark.quick
def test_all_nested_config_fields_are_covered_by_exact_digest_freeze(
    tmp_path: Path,
) -> None:
    baseline = _config()
    nested_sections = (
        "historical_gate_a_source_binding",
        "base_construction_contract",
        "execution_identity",
        "diagnostic_plan",
        "checkpoint_contract",
        "interpretation_contract",
        "authorization_boundary",
    )
    paths = [
        (section, key)
        for section in nested_sections
        for key in baseline[section]
    ] + [
        (key,)
        for key in (
            "profile_id",
            "method_id",
            "diagnostic_role",
            "execution_authorized",
            "historical_gate_a_failure_required",
            "formal_result",
            "stage_progression_allowed",
            "claim_support_status",
        )
    ]
    for index, path in enumerate(paths):
        config = _config()
        cursor = config
        for key in path[:-1]:
            cursor = cursor[key]
        cursor[path[-1]] = _mutated_json_value(cursor[path[-1]])
        mutated_path = tmp_path / f"mutated_{index:03d}.json"
        _write_config(mutated_path, config)
        with pytest.raises((FileNotFoundError, ValueError)):
            load_gate_a_root_cause_diagnostic_config(mutated_path)


def _statistics(
    pair_id: str,
    checkpoint_id: str,
    *,
    odd: float,
    common: float,
    cosine: float,
    residual: float,
) -> PairedResponseStatistics:
    return PairedResponseStatistics(
        pair_id=pair_id,
        checkpoint_id=checkpoint_id,
        clean_distance=0.0,
        positive_centered_norm=odd + common,
        negative_centered_norm=odd + common,
        odd_norm=odd,
        common_norm=common,
        common_odd_ratio=common / odd if odd else None,
        antisymmetry_cosine=cosine,
        antisymmetry_residual=residual,
        finite=True,
    )


@pytest.mark.quick
def test_odd_common_formula_matches_synthetic_vectors() -> None:
    clean_a = np.array([2.0, -1.0])
    clean_b = clean_a.copy()
    linear = np.array([0.8, -0.4])
    common = np.array([0.1, 0.2])
    stats = compute_paired_response_statistics(
        pair_id="early_flow_channel_0",
        checkpoint_id="T_output_feature_256d",
        clean_start=clean_a,
        clean_end=clean_b,
        positive=clean_a + linear + common,
        negative=clean_a - linear + common,
        denominator_epsilon=1e-15,
    )
    assert stats.odd_norm == pytest.approx(np.linalg.norm(linear))
    assert stats.common_norm == pytest.approx(np.linalg.norm(common))
    assert stats.common_odd_ratio == pytest.approx(
        np.linalg.norm(common) / np.linalg.norm(linear)
    )


@pytest.mark.quick
def test_half_scaling_contract_accepts_first_and_second_order_response() -> None:
    config = _config()
    full = _statistics(
        "early_flow_channel_0",
        "T_output_feature_256d",
        odd=2.0,
        common=1.0,
        cosine=0.1,
        residual=0.8,
    )
    half = _statistics(
        "early_flow_channel_0",
        "T_output_feature_256d",
        odd=1.0,
        common=0.25,
        cosine=0.4,
        residual=0.5,
    )
    result = compare_half_to_historical_full(
        config,
        half=half,
        full=full,
        half_actual_amplitude=0.5,
        full_actual_amplitude=1.0,
    )
    assert result.actual_amplitude_ratio_ready is True
    assert result.odd_ratio_to_full == pytest.approx(0.5)
    assert result.common_ratio_to_full == pytest.approx(0.25)
    assert result.normalized_odd_scaling == pytest.approx(1.0)
    assert result.normalized_common_scaling == pytest.approx(1.0)
    assert result.local_linear_scaling_ready is True


@pytest.mark.quick
def test_scaling_comparison_is_indeterminate_when_actual_exposure_is_not_half() -> None:
    config = _config()
    full = _statistics(
        "early_flow_channel_0",
        "T_output_feature_256d",
        odd=2.0,
        common=1.0,
        cosine=0.0,
        residual=0.8,
    )
    half = _statistics(
        "early_flow_channel_0",
        "T_output_feature_256d",
        odd=1.0,
        common=0.25,
        cosine=0.5,
        residual=0.4,
    )
    result = compare_half_to_historical_full(
        config,
        half=half,
        full=full,
        half_actual_amplitude=0.7,
        full_actual_amplitude=1.0,
    )
    assert result.actual_amplitude_ratio_ready is False
    assert result.local_linear_scaling_ready is False
    assert result.comparison_status.startswith("indeterminate")


@pytest.mark.quick
def test_governed_root_cause_jsonl_fields_match_registry_exactly(
    tmp_path: Path,
) -> None:
    full = _statistics(
        "early_flow_channel_0",
        "T_output_feature_256d",
        odd=2.0,
        common=1.0,
        cosine=0.1,
        residual=0.8,
    )
    half = _statistics(
        "early_flow_channel_0",
        "T_output_feature_256d",
        odd=1.0,
        common=0.25,
        cosine=0.4,
        residual=0.5,
    )
    comparison = compare_half_to_historical_full(
        _config(),
        half=half,
        full=full,
        half_actual_amplitude=0.5,
        full_actual_amplitude=1.0,
    )
    key = (half.pair_id, half.checkpoint_id)
    _write_root_cause_metric_records(
        tmp_path,
        half_statistics={key: half},
        historical_statistics={key: full},
        scaling={key: comparison},
    )
    statistics_path = (
        tmp_path
        / "records"
        / "root_cause_paired_response_statistics.jsonl"
    )
    comparison_path = (
        tmp_path / "records" / "root_cause_scaling_comparisons.jsonl"
    )
    statistics_record = json.loads(
        statistics_path.read_text(encoding="utf-8").strip()
    )
    comparison_record = json.loads(
        comparison_path.read_text(encoding="utf-8").strip()
    )
    assert set(statistics_record) == PAIRED_RESPONSE_RECORD_FIELDS
    assert set(comparison_record) == SCALING_COMPARISON_RECORD_FIELDS
    assert "pair_id" not in statistics_record
    assert "odd_norm" not in statistics_record
    assert "actual_amplitude_ratio_ready" not in comparison_record
    assert statistics_record["formal_result"] is False
    assert statistics_record["stage_progression_allowed"] is False
    assert comparison_record["formal_result"] is False
    assert comparison_record["stage_progression_allowed"] is False

    registered_fields = {
        line.split("|")[1].strip()
        for line in Path("docs/field_registry.md").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.startswith("|") and line.count("|") >= 2
    }
    assert PAIRED_RESPONSE_RECORD_FIELDS <= registered_fields
    assert SCALING_COMPARISON_RECORD_FIELDS <= registered_fields
    assert {
        "candidate_classification",
        "clean_order_drift_diagnostic",
    } <= registered_fields


@pytest.mark.quick
def test_full_representation_gram_matches_dense_statistics(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    arrays = {
        probe_id: rng.standard_normal((3, 4)).astype(np.float32)
        for probe_id in DIAGNOSTIC_PLAN_IDS
    }
    paths = {}
    for probe_id, array in arrays.items():
        path = tmp_path / f"{probe_id}.npy"
        np.save(path, array, allow_pickle=False)
        paths[probe_id] = path
    gram = _gram_matrix_from_array_paths(paths, loader=_load_npy_memmap)
    from_gram = compute_paired_response_statistics_from_gram(
        pair_id="early_flow_channel_0",
        checkpoint_id="T_saved_video_full_rgb24",
        gram_matrix=gram,
        row_ids=DIAGNOSTIC_PLAN_IDS,
        denominator_epsilon=1e-15,
    )
    dense = compute_paired_response_statistics(
        pair_id="early_flow_channel_0",
        checkpoint_id="T_saved_video_full_rgb24",
        clean_start=arrays["clean_start"],
        clean_end=arrays["clean_end"],
        positive=arrays["positive_early_flow_channel_0"],
        negative=arrays["negative_early_flow_channel_0"],
        denominator_epsilon=1e-15,
    )
    assert from_gram.odd_norm == pytest.approx(dense.odd_norm, rel=1e-10)
    assert from_gram.common_norm == pytest.approx(
        dense.common_norm,
        rel=1e-10,
    )
    assert from_gram.antisymmetry_cosine == pytest.approx(
        dense.antisymmetry_cosine,
        rel=1e-10,
    )


@pytest.mark.quick
def test_latent_signed_view_drops_sign_even_l2_dimension() -> None:
    values = {}
    for probe_id in DIAGNOSTIC_PLAN_IDS:
        latent = np.arange(7, dtype=np.float64)
        latent[-1] = 10_000.0 + len(probe_id)
        for checkpoint_id, dimension in (
            ("T_latent", 7),
            ("T_decoded", 48),
            ("T_saved_video", 48),
            ("T_reencoded", 256),
            ("T_output_feature", 256),
        ):
            values[(probe_id, checkpoint_id)] = (
                latent.copy()
                if checkpoint_id == "T_latent"
                else np.zeros(dimension)
            )
    view = _checkpoint_view(values, historical=False)
    assert all(
        row.shape == (6,)
        for row in view["T_latent_six_basis"].values()
    )
    assert all(
        10_000.0 not in row
        for row in view["T_latent_six_basis"].values()
    )
    historical_values = {}
    historical_ids = (
        "clean_a",
        "clean_b",
        "positive_early_flow_channel_0",
        "negative_early_flow_channel_0",
        "positive_late_flow_channel_0",
        "negative_late_flow_channel_0",
    )
    for probe_id in historical_ids:
        for checkpoint_id, dimension in (
            ("T_latent", 7),
            ("T_decoded", 48),
            ("T_saved_video", 48),
            ("T_reencoded", 256),
            ("T_output_feature", 256),
        ):
            historical_values[(probe_id, checkpoint_id)] = np.zeros(
                dimension
            )
    historical_view = _checkpoint_view(
        historical_values,
        historical=True,
    )
    assert tuple(historical_view["T_latent_six_basis"]) == DIAGNOSTIC_PLAN_IDS


@pytest.mark.quick
def test_classifier_allows_multiple_candidates_and_never_unique_claim() -> None:
    config = _config()
    half = {}
    scaling = {}
    for pair_id in PAIR_IDS:
        for checkpoint_id in COMPARABLE_CHECKPOINT_IDS:
            half[(pair_id, checkpoint_id)] = _statistics(
                pair_id,
                checkpoint_id,
                odd=1.0 if pair_id.startswith("late") else 0.2,
                common=2.0,
                cosine=0.95 if checkpoint_id == "T_latent_six_basis" else 0.0,
                residual=0.1 if checkpoint_id == "T_latent_six_basis" else 0.9,
            )
            scaling[(pair_id, checkpoint_id)] = ScalingComparison(
                pair_id=pair_id,
                checkpoint_id=checkpoint_id,
                actual_amplitude_ratio=0.5,
                actual_amplitude_ratio_ready=True,
                odd_ratio_to_full=0.8,
                common_ratio_to_full=0.9,
                normalized_odd_scaling=1.6,
                normalized_common_scaling=3.6,
                antisymmetry_cosine_improvement=0.0,
                antisymmetry_residual_improvement=0.0,
                local_linear_scaling_ready=False,
                comparison_status="ready",
            )
    result = classify_root_cause_candidates(
        config,
        half_statistics=half,
        scaling=scaling,
        half_actual_amplitude_by_pair={
            "early_flow_channel_0": 0.4,
            "late_flow_channel_0": 0.4,
        },
    )
    assert result["unique_root_cause_claim_allowed"] is False
    assert "quantization_or_observation_floor_candidate" in result[
        "candidate_classifications"
    ]
    assert "feedback_nonlinearity_candidate" in result[
        "candidate_classifications"
    ]
    assert "multiple_factors_candidate" in result[
        "candidate_classifications"
    ]


def _gate_a_test_helpers() -> dict:
    return runpy.run_path(
        "tests/functional/"
        "test_output_feature_impulse_observability_construction.py"
    )


def _build_historical_fail_source(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    helpers = _gate_a_test_helpers()
    source = root / "prompt_source"
    helpers["_write_source_root"](source)
    monkeypatch.setenv(
        "SSTW_TRAJECTORY_AUTHENTICATION_KEY",
        "owner-key-material-for-construction-tests",
    )
    monkeypatch.setattr(
        construction_module,
        "_package_version",
        lambda _name: "0.35.2",
    )
    monkeypatch.setattr(
        construction_module,
        "_repository_commit",
        lambda: _config()["historical_gate_a_source_binding"][
            "repository_commit"
        ],
    )

    def generation_executor(config: dict, **kwargs) -> ConstructionGenerationBatch:
        batch = helpers["_generation_batch"](
            config,
            Path(kwargs["output_root"]),
        )
        trace_by_probe = {
            trace.probe_id: trace for trace in batch.exposure_traces
        }
        schedule_reference = batch.exposure_traces[0]
        complete_step_records = []
        for minimal in batch.trajectory_step_records:
            probe_id = str(minimal["impulse_probe_id"])
            step_index = int(minimal["impulse_flow_step_index"])
            trace = trace_by_probe.get(probe_id)
            active = bool(
                trace is not None
                and trace.intended_velocity_waveform_by_step[step_index]
                != 0.0
            )

            def scalar(name: str, default: float = 0.0):
                return (
                    getattr(trace, name)[step_index]
                    if trace is not None
                    else default
                )

            complete_step_records.append(
                {
                    "record_version": (
                        "output_feature_impulse_observability_"
                        "construction_v1"
                    ),
                    "impulse_probe_id": probe_id,
                    "impulse_flow_step_index": step_index,
                    "impulse_flow_phase": (
                        schedule_reference.flow_phase_by_step[step_index]
                    ),
                    "impulse_delta_sigma": (
                        schedule_reference.delta_sigma_by_step[step_index]
                    ),
                    "impulse_macro_interval_index": (
                        schedule_reference.macro_interval_index_by_step[
                            step_index
                        ]
                    ),
                    "impulse_stage_index": (
                        None if trace is None else trace.stage_index
                    ),
                    "impulse_state_channel_index": (
                        None if trace is None else trace.channel_index
                    ),
                    "impulse_polarity": (
                        0 if trace is None else trace.polarity
                    ),
                    "impulse_intended_velocity_waveform": scalar(
                        "intended_velocity_waveform_by_step"
                    ),
                    "impulse_reference_base_velocity_norm": scalar(
                        "reference_base_velocity_norm_by_step",
                        1.0,
                    ),
                    "impulse_remaining_control_energy_before_step": scalar(
                        "remaining_control_energy_before_step_by_step"
                    ),
                    "impulse_reference_energy_increment": scalar(
                        "reference_energy_increment_by_step",
                        1.0,
                    ),
                    "impulse_reference_cumulative_energy": scalar(
                        "reference_cumulative_energy_by_step",
                        float(step_index + 1),
                    ),
                    "impulse_intended_delta_norm": scalar(
                        "intended_delta_norm_by_step"
                    ),
                    "impulse_actual_velocity_basis_coordinate": scalar(
                        "actual_velocity_basis_coordinate_by_step"
                    ),
                    "impulse_actual_channel_velocity_coordinate": (
                        [0.0] * 6
                        if trace is None
                        else list(
                            trace.actual_channel_velocity_coordinate_by_step[
                                step_index
                            ]
                        )
                    ),
                    "impulse_intended_signed_exposure": scalar(
                        "intended_signed_exposure_by_step"
                    ),
                    "impulse_actual_signed_exposure": scalar(
                        "actual_signed_exposure_by_step"
                    ),
                    "impulse_actual_channel_exposure": (
                        [0.0] * 6
                        if trace is None
                        else list(
                            trace.actual_channel_exposure_by_step[step_index]
                        )
                    ),
                    "impulse_actual_delta_norm": scalar(
                        "delta_norm_by_step"
                    ),
                    "impulse_actual_projection_scale": scalar(
                        "projection_scale_by_step"
                    ),
                    "impulse_cumulative_control_energy": scalar(
                        "cumulative_energy_by_step"
                    ),
                    "impulse_actual_direction_cosine": scalar(
                        "direction_cosine_by_step",
                        1.0,
                    ),
                    "impulse_norm_guard_passed": bool(
                        scalar("norm_guard_passed_by_step", True)
                    ),
                    "impulse_energy_guard_passed": bool(
                        scalar("energy_guard_passed_by_step", True)
                    ),
                    "impulse_direction_guard_passed": True,
                    "impulse_inactive_exact_noop": not active,
                    "impulse_finite_precision_projection_status": (
                        "direct_actual_delta_pass"
                        if active
                        else "inactive_exact_noop"
                    ),
                    "impulse_finite_precision_projection_scale": (
                        1.0 if active else 0.0
                    ),
                    "impulse_finite_precision_projection_attempt_count": (
                        1 if active else 0
                    ),
                    "impulse_finite_precision_backoff_count": 0,
                    "formal_result": False,
                    "stage_progression_allowed": False,
                }
            )
        batch = replace(
            batch,
            trajectory_step_records=tuple(complete_step_records),
        )
        records = []
        for record in batch.generation_records:
            value = dict(record)
            value["record_version"] = (
                "output_feature_impulse_observability_construction_v1"
            )
            value["formal_result"] = False
            value["stage_progression_allowed"] = False
            value["claim_support_status"] = (
                "output_feature_impulse_observability_construction_only_"
                "not_method_evidence"
            )
            value["impulse_generation_record_id"] = canonical_json_digest(
                value
            )
            records.append(value)
        return replace(batch, generation_records=tuple(records))

    def tiny_features(config: dict, **_kwargs):
        batch = helpers["_feature_batch"](config, response=1e-8)
        records = []
        for record in batch.feature_records:
            value = dict(record)
            value["record_version"] = (
                "output_feature_impulse_observability_construction_v1"
            )
            value["formal_result"] = False
            value["stage_progression_allowed"] = False
            value["claim_support_status"] = (
                "output_feature_impulse_observability_construction_only_"
                "not_method_evidence"
            )
            value.pop("construction_feature_record_id", None)
            value["construction_feature_record_id"] = canonical_json_digest(
                value
            )
            records.append(value)
        return replace(batch, feature_records=tuple(records))

    output = root / "historical"
    decision = run_output_feature_impulse_observability_construction(
        source,
        output,
        generation_executor=generation_executor,
        feature_executor=tiny_features,
        basis_builder=helpers["_fake_basis"],
    )
    assert decision["impulse_sample_internal_observability_gate_ready"] is False
    return output


@pytest.mark.quick
def test_historical_gate_a_fail_source_validates_complete_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _build_historical_fail_source(tmp_path, monkeypatch)
    historical = validate_historical_gate_a_failure_source(
        source,
        _config(),
    )
    assert historical.decision[
        "impulse_observability_construction_decision"
    ].endswith("_failed_stop")
    assert len(historical.generation_records) == 14
    assert len(historical.exposure_traces) == 12
    assert len(historical.video_paths) == 14
    assert historical.actual_exposure_by_pair["early_flow_channel_0"] > 0.0


@pytest.mark.quick
@pytest.mark.parametrize(
    "mutation",
    [
        "pass_decision",
        "recovery_marker",
        "video_bytes",
        "generation_order",
        "step_delta",
        "step_guard",
        "step_trace_mismatch",
        "clean_nonzero",
        "fp32_projection_scale",
        "fp32_projection_status",
        "fp32_backoff_count",
        "fp32_attempt_count",
    ],
)
def test_historical_gate_a_source_mutations_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    source = _build_historical_fail_source(tmp_path, monkeypatch)
    if mutation == "pass_decision":
        path = (
            source
            / "artifacts"
            / "output_feature_impulse_observability_decision.json"
        )
        value = json.loads(path.read_text(encoding="utf-8"))
        value["impulse_sample_internal_observability_gate_ready"] = True
        value["impulse_observability_construction_decision"] = (
            "sample_internal_causal_observability_gate_pass"
        )
        _write_config(path, value)
    elif mutation == "recovery_marker":
        (source / "colab_test_recovery_manifest.json").write_text(
            "{}\n",
            encoding="utf-8",
        )
    elif mutation == "video_bytes":
        next((source / "videos").glob("*.mp4")).write_bytes(b"mutated")
    elif mutation == "generation_order":
        path = source / "records" / "impulse_generation_records.jsonl"
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
        ]
        rows[0], rows[1] = rows[1], rows[0]
        path.write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )
    else:
        path = source / "records" / "impulse_trajectory_step_records.jsonl"
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
        ]
        if mutation == "step_delta":
            rows[8]["impulse_actual_delta_norm"] = 12345.0
        elif mutation == "step_guard":
            rows[8]["impulse_norm_guard_passed"] = False
        elif mutation == "step_trace_mismatch":
            rows[8]["impulse_actual_signed_exposure"] += 1e-6
        elif mutation == "clean_nonzero":
            rows[0]["impulse_actual_delta_norm"] = 1e-6
        elif mutation == "fp32_projection_scale":
            rows[8]["impulse_finite_precision_projection_scale"] = 999.0
        elif mutation == "fp32_projection_status":
            rows[8]["impulse_finite_precision_projection_status"] = (
                "forged_status"
            )
        elif mutation == "fp32_backoff_count":
            rows[8]["impulse_finite_precision_backoff_count"] = -999
        elif mutation == "fp32_attempt_count":
            rows[8][
                "impulse_finite_precision_projection_attempt_count"
            ] = 999
        else:
            raise AssertionError(mutation)
        path.write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )
    with pytest.raises(ValueError):
        validate_historical_gate_a_failure_source(source, _config())


@pytest.mark.quick
def test_runner_invalid_source_writes_recovery_only_decision(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    source = tmp_path / "empty_source"
    source.mkdir()
    with pytest.raises(RuntimeError, match="必须唯一包含"):
        run_gate_a_root_cause_amplitude_feedback_diagnostic(
            source,
            output,
        )
    decision = json.loads(
        (
            output
            / "artifacts"
            / "gate_a_root_cause_amplitude_feedback_decision.json"
        ).read_text(encoding="utf-8")
    )
    assert decision["formal_result"] is False
    assert decision["stage_progression_allowed"] is False
    assert decision["gate_a_pass"] is False
    assert decision["claim_support_status"] == (
        "failure_recovery_only_not_claim_evidence"
    )
    for field in (
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
        "frozen_feedback_diagnostic_design_allowed",
        "carrier_feature_redesign_allowed",
        "automatic_followup_execution_allowed",
        "formal_result",
        "stage_progression_allowed",
    ):
        assert decision[field] is False


@pytest.mark.quick
def test_clean_order_pollution_invalidates_other_root_cause_candidates() -> None:
    result = _apply_clean_pollution_fail_closed(
        {
            "candidate_classifications": [
                "feedback_nonlinearity_candidate",
                "carrier_decoder_feature_mismatch_candidate",
            ],
            "local_linear_support_candidate": True,
            "quantization_or_observation_floor_candidate": True,
            "feedback_nonlinearity_candidate": True,
            "carrier_decoder_feature_mismatch_candidate": True,
            "classification_status": "multiple_candidates",
        },
        contaminated=True,
    )
    assert result["candidate_classifications"] == [
        "generation_order_state_pollution_candidate",
        "indeterminate",
    ]
    assert result["classification_status"] == "contaminated_indeterminate"
    assert result["clean_intercept_and_pair_classification_valid"] is False
    assert all(
        result[field] is False
        for field in (
            "local_linear_support_candidate",
            "quantization_or_observation_floor_candidate",
            "feedback_nonlinearity_candidate",
            "carrier_decoder_feature_mismatch_candidate",
        )
    )
    assert not (
        result["feedback_nonlinearity_candidate"]
        or result["classification_status"] == "indeterminate"
    )
    assert result["carrier_decoder_feature_mismatch_candidate"] is False


class _ArrayLikeLatent:
    def __init__(self, value: np.ndarray) -> None:
        self.value = value


@pytest.mark.quick
def test_array_capture_keeps_full_latent_and_temporary_decoded_local(
    tmp_path: Path,
) -> None:
    capture = DiagnosticArrayCapture(tmp_path)
    plan = build_gate_a_root_cause_diagnostic_plan(_config())
    latent = np.zeros((1, 16, 9, 40, 64), dtype=np.float32)
    decoded = np.zeros((33, 320, 512, 3), dtype=np.float32)
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    basis = _gate_a_test_helpers()["_fake_basis"]("key")
    records = capture(
        probe=plan[0],
        plan_index=0,
        final_latent=latent,
        decoded_frames=decoded,
        video_path=video,
        video_sha256="a" * 64,
        basis=basis,
    )
    assert len(records) == 1
    artifact = Path(records[0]["final_latent_artifact_path"])
    assert artifact.is_file()
    assert np.array_equal(_load_latent_artifact(artifact), latent)
    assert capture.decoded_paths["clean_start"].is_file()
    capture.remove_temporary_decoded_arrays()
    assert not capture.decoded_paths["clean_start"].exists()


def _write_historical_source_zip(source: Path, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for item in sorted(source.rglob("*")):
            if item.is_file():
                archive.write(item, item.relative_to(source).as_posix())


def _request_payload(source_zip: Path) -> dict:
    return {
        "request_schema_version": REQUEST_SCHEMA_VERSION,
        "test_id": GATE_A_ROOT_CAUSE_AMPLITUDE_FEEDBACK_DIAGNOSTIC_TEST_ID,
        "repository": {
            "url": "https://github.com/RICHAAARC/SSTW.git",
            "ref": "47485be2b6f734014b74f73e797174b911d2aeb5",
        },
        "parameters": {
            "phase": "root_cause_diagnostic",
            "run_series_id": "gate_a_root_cause_amplitude_feedback",
            "source_package_path": str(source_zip),
            "resume_package_path": "",
        },
    }


@pytest.mark.quick
def test_colab_request_allowlist_routes_single_zip_without_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    historical = _build_historical_fail_source(tmp_path, monkeypatch)
    project = tmp_path / "drive" / "SSTW"
    source_zip = project / "inputs" / "gate_a_fail.zip"
    _write_historical_source_zip(historical, source_zip)
    request = project / "requests" / "colab_test_request.json"
    request.parent.mkdir(parents=True)
    request.write_text(
        json.dumps(_request_payload(source_zip)) + "\n",
        encoding="utf-8",
    )
    resolved = load_colab_test_request(request, project_root=project)
    assert resolved["phase"] == "root_cause_diagnostic"
    assert build_colab_test_dry_run_plan(
        request,
        project_root=project,
    )["test_id"] == GATE_A_ROOT_CAUSE_AMPLITUDE_FEEDBACK_DIAGNOSTIC_TEST_ID

    def fake_runner(source_root: Path, output_root: Path) -> dict:
        assert list(
            source_root.rglob(
                "output_feature_impulse_observability_decision.json"
            )
        )
        artifact = output_root / "artifacts" / "decision.json"
        artifact.parent.mkdir(parents=True)
        artifact.write_text('{"formal_result":false}\n', encoding="utf-8")
        return {
            "gate_a_pass": False,
            "formal_result": False,
            "stage_progression_allowed": False,
        }

    result = run_colab_test_request(
        request,
        project_root=project,
        repo_root=Path.cwd(),
        local_workspace_root=tmp_path / "content" / "workspace",
        local_package_cache_root=tmp_path / "content" / "packages",
        gate_a_root_cause_diagnostic_runner=fake_runner,
    )
    assert Path(result["drive_result_zip"]).is_file()
    assert Path(result["drive_result_manifest"]).is_file()
    assert result["diagnostic_decision"]["gate_a_pass"] is False
    manifest = json.loads(
        Path(result["drive_result_manifest"]).read_text(encoding="utf-8")
    )
    assert manifest["generation_model_ids"] == [
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    ]

    payload = _request_payload(source_zip)
    payload["parameters"]["resume_package_path"] = str(source_zip)
    request.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="完整历史 FAIL"):
        load_colab_test_request(request, project_root=project)


@pytest.mark.quick
def test_server_cli_dry_run_routes_root_cause_diagnostic_without_gpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    historical = _build_historical_fail_source(tmp_path, monkeypatch)
    project = tmp_path / "drive" / "SSTW"
    source_zip = project / "inputs" / "gate_a_fail.zip"
    _write_historical_source_zip(historical, source_zip)
    request = project / "requests" / "colab_test_request.json"
    request.parent.mkdir(parents=True)
    request.write_text(
        json.dumps(_request_payload(source_zip)) + "\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_generative_video_server_workflow.py",
            "--project-root",
            str(project),
            "--workflow-profile",
            "colab_test",
            "--pipeline",
            "colab_test",
            "--colab-test-request-path",
            str(request),
            "--dry-run",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    decision = json.loads(completed.stdout)
    assert decision["server_workflow_decision"] == "DRY_RUN"
    assert decision["pipeline_results"][0]["test_id"] == (
        GATE_A_ROOT_CAUSE_AMPLITUDE_FEEDBACK_DIAGNOSTIC_TEST_ID
    )
    assert decision["pipeline_results"][0]["phase"] == (
        "root_cause_diagnostic"
    )


@pytest.mark.quick
def test_failure_recovery_remains_nonformal_for_new_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    historical = _build_historical_fail_source(tmp_path, monkeypatch)
    runtime_root = tmp_path / "content"
    project = tmp_path / "drive" / "SSTW"
    source_zip = project / "inputs" / "gate_a_fail.zip"
    _write_historical_source_zip(historical, source_zip)
    request = project / "requests" / "colab_test_request.json"
    request.parent.mkdir(parents=True)
    request.write_text(
        json.dumps(_request_payload(source_zip)) + "\n",
        encoding="utf-8",
    )
    workspace = runtime_root / "workspace"
    cache = runtime_root / "packages"
    partial = (
        workspace
        / "runs"
        / GATE_A_ROOT_CAUSE_AMPLITUDE_FEEDBACK_DIAGNOSTIC_TEST_ID
        / "gate_a_root_cause_amplitude_feedback"
        / "records"
        / "impulse_generation_records.jsonl"
    )
    partial.parent.mkdir(parents=True)
    partial.write_text(
        '{"generation_status":"failed","formal_result":false}\n',
        encoding="utf-8",
    )
    result = package_colab_test_recovery_bundle(
        request,
        project_root=project,
        repo_root=Path.cwd(),
        local_runtime_root=runtime_root,
        local_workspace_root=workspace,
        local_package_cache_root=cache,
    )
    assert result["formal_result"] is False
    assert result["stage_progression_allowed"] is False
    assert result["claim_support_status"] == (
        "failure_recovery_only_not_claim_evidence"
    )


@pytest.mark.quick
def test_fixed_notebook_has_no_root_cause_diagnostic_logic() -> None:
    notebook = Path(
        "paper_workflow/colab_notebooks/colab_test_runner.ipynb"
    )
    source = notebook.read_text(encoding="utf-8")
    assert "scripts/run_generative_video_server_workflow.py" in source
    assert "gate_a_root_cause_amplitude_feedback_diagnostic.py" not in source
