"""Lightweight contract tests for output-feature impulse observability."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import inspect
import json
import math
from pathlib import Path

import numpy as np
import pytest

from evaluation.protocol.impulse_observability_contract import (
    CONSTRUCTION_FEATURE_ENCODER_ID,
    CONSTRUCTION_FEATURE_ENCODER_REVISION,
    CONSTRUCTION_FEATURE_OUTPUT_DIMENSION,
    CONSTRUCTION_FLOW_STEP_COUNT,
    CONSTRUCTION_LATENT_LAYOUT_SHAPE,
    FLOW_MACRO_INTERVAL_COUNT,
    IMPULSE_CLEAN_REPEAT_COUNT,
    IMPULSE_OBSERVABILITY_PROFILE_ID,
    IMPULSE_PROBE_COUNT,
    IMPULSE_TRIAGE_VIDEO_COUNT,
    OBSERVER_SYNCHRONIZED_METHOD_ID,
    STAGE_BASIS_RANK,
    WATERMARK_STATE_DIMENSION,
    ActualImpulseExposureTrace,
    assemble_actual_design_matrix,
    build_construction_stage_basis,
    build_impulse_triage_plan,
    build_stage_selector_blocks,
    construction_feature_schema_digest,
    construction_feature_row_binding_digest,
    compute_intended_impulse_control,
    derive_construction_basis_subkey,
    estimate_construction_transfer,
    evaluate_gate_a_statistics,
    extract_construction_output_feature_from_normalized_latent,
    extract_construction_reencoded_summary_from_normalized_latent,
    flow_schedule_waveform_schema_digest,
    load_impulse_observability_config,
    runtime_adapter_schema_digest,
    validate_construction_output_features,
    validate_impulse_observability_config,
)


pytestmark = pytest.mark.quick
CONFIG_PATH = Path(
    "configs/protocol/"
    "sstw_output_feature_impulse_observability_construction.json"
)


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _valid_traces(
    config: dict,
    *,
    actual_scale_by_probe: dict[str, float] | None = None,
    actual_step_scale_by_probe: dict[str, tuple[float, ...]] | None = None,
) -> tuple[ActualImpulseExposureTrace, ...]:
    schedule = config["flow_schedule_contract"]
    exposure = config["actual_exposure_contract"]
    traces: list[ActualImpulseExposureTrace] = []
    for record in build_impulse_triage_plan(config):
        if record.probe_role != "signed_interval_impulse":
            continue
        assert record.stage_index is not None
        assert record.channel_index is not None
        coordinate = (
            record.stage_index * WATERMARK_STATE_DIMENSION
            + record.channel_index
        )
        temporal_waveform = tuple(
            schedule["temporal_waveform_by_macro_interval"][
                record.stage_index
            ]
        )
        intended_exposure: list[float] = []
        actual_exposure: list[float] = []
        actual_velocity: list[float] = []
        channel_velocity: list[tuple[float, ...]] = []
        channel_exposure: list[tuple[float, ...]] = []
        delta_norms: list[float] = []
        projection_scales: list[float] = []
        cumulative_energy: list[float] = []
        reference_increments: list[float] = []
        reference_cumulative: list[float] = []
        remaining_energy: list[float] = []
        intended_norms: list[float] = []
        direction_cosines: list[float] = []
        control_energy_before = 0.0
        reference_energy_before = 0.0
        for step_index, (waveform_value, delta_sigma) in enumerate(
            zip(
                temporal_waveform,
                schedule["delta_sigma_by_step"],
                strict=True,
            )
        ):
            control = compute_intended_impulse_control(
                probe_state_update_polarity=record.polarity,
                temporal_waveform=waveform_value,
                delta_sigma=delta_sigma,
                base_velocity_norm=100.0,
                cumulative_control_energy=control_energy_before,
                cumulative_reference_energy=reference_energy_before,
                remaining_step_count=CONSTRUCTION_FLOW_STEP_COUNT - step_index,
                lambda_max=exposure["lambda_max"],
                velocity_norm_ratio_budget=(
                    exposure["velocity_norm_ratio_budget"]
                ),
                flow_energy_budget_ratio=(
                    exposure["flow_energy_budget_ratio"]
                ),
            )
            active = control.intended_delta_norm > 0.0
            scale = (
                (actual_scale_by_probe or {}).get(record.probe_id, 0.8)
            )
            if record.probe_id in (actual_step_scale_by_probe or {}):
                scale = (actual_step_scale_by_probe or {})[
                    record.probe_id
                ][step_index]
            actual_norm = (
                scale * control.intended_delta_norm if active else 0.0
            )
            desired_velocity_sign = (
                record.polarity * (1 if delta_sigma > 0.0 else -1)
            )
            target_velocity = desired_velocity_sign * actual_norm
            velocity_vector = [0.0] * STAGE_BASIS_RANK
            velocity_vector[coordinate] = target_velocity
            state_vector = [
                delta_sigma * value for value in velocity_vector
            ]
            control_increment = delta_sigma**2 * actual_norm**2
            control_energy_before += control_increment
            reference_energy_before += control.reference_energy_increment
            intended_exposure.append(
                control.signed_state_update_exposure
            )
            actual_exposure.append(state_vector[coordinate])
            actual_velocity.append(target_velocity)
            channel_velocity.append(tuple(velocity_vector))
            channel_exposure.append(tuple(state_vector))
            delta_norms.append(actual_norm)
            projection_scales.append(scale if active else 0.0)
            cumulative_energy.append(control_energy_before)
            reference_increments.append(control.reference_energy_increment)
            reference_cumulative.append(reference_energy_before)
            remaining_energy.append(control.remaining_control_energy)
            intended_norms.append(control.intended_delta_norm)
            direction_cosines.append(1.0)
        actual_vector = tuple(
            np.asarray(channel_exposure, dtype=np.float64).sum(axis=0)
        )
        traces.append(
            ActualImpulseExposureTrace(
                probe_id=record.probe_id,
                stage_index=record.stage_index,
                channel_index=record.channel_index,
                polarity=record.polarity,
                step_indices=tuple(schedule["step_indices"]),
                flow_phase_by_step=tuple(schedule["flow_phase_by_step"]),
                delta_sigma_by_step=tuple(schedule["delta_sigma_by_step"]),
                macro_interval_index_by_step=tuple(
                    schedule["macro_interval_index_by_step"]
                ),
                intended_velocity_waveform_by_step=temporal_waveform,
                reference_base_velocity_norm_by_step=(100.0,) * 8,
                remaining_control_energy_before_step_by_step=tuple(
                    remaining_energy
                ),
                reference_energy_increment_by_step=tuple(
                    reference_increments
                ),
                reference_cumulative_energy_by_step=tuple(
                    reference_cumulative
                ),
                intended_delta_norm_by_step=tuple(intended_norms),
                actual_velocity_basis_coordinate_by_step=tuple(
                    actual_velocity
                ),
                actual_channel_velocity_coordinate_by_step=tuple(
                    channel_velocity
                ),
                intended_signed_exposure_by_step=tuple(intended_exposure),
                actual_signed_exposure_by_step=tuple(actual_exposure),
                actual_channel_exposure_by_step=tuple(channel_exposure),
                actual_exposure_vector=actual_vector,
                delta_norm_by_step=tuple(delta_norms),
                projection_scale_by_step=tuple(projection_scales),
                cumulative_energy_by_step=tuple(cumulative_energy),
                direction_cosine_by_step=tuple(direction_cosines),
                norm_guard_passed_by_step=(True,) * 8,
                energy_guard_passed_by_step=(True,) * 8,
                waveform_schema_digest=schedule[
                    "waveform_schema_digest"
                ],
                runtime_adapter_schema_digest=config[
                    "runtime_adapter_contract"
                ]["adapter_schema_digest"],
                basis_digest="a" * 64,
            )
        )
    return tuple(traces)


def _normalized_probe_features(
    *,
    response_amplitudes: tuple[float, ...] = (0.01,) * 6,
    clean_separation: float = 0.0,
) -> np.ndarray:
    features = np.zeros(
        (IMPULSE_TRIAGE_VIDEO_COUNT, CONSTRUCTION_FEATURE_OUTPUT_DIMENSION),
        dtype=np.float64,
    )
    clean_offset = float(clean_separation) / 2.0
    features[0, 0] = math.sqrt(max(0.0, 1.0 - clean_offset**2))
    features[0, 7] = clean_offset
    features[1, 0] = math.sqrt(max(0.0, 1.0 - clean_offset**2))
    features[1, 7] = -clean_offset
    for coordinate, response in enumerate(response_amplitudes):
        common = math.sqrt(max(0.0, 1.0 - response**2))
        features[2 + coordinate, 0] = common
        features[2 + coordinate, 1 + coordinate] = response
        features[8 + coordinate, 0] = common
        features[8 + coordinate, 1 + coordinate] = -response
    return features


def _ready_primary_checkpoints() -> dict[str, bool]:
    return {
        "T_latent": True,
        "T_decoded": True,
        "T_saved_video": True,
        "T_reencoded": True,
        "T_output_feature": True,
    }


def _probe_ids(config: dict) -> tuple[str, ...]:
    return tuple(
        record.probe_id for record in build_impulse_triage_plan(config)
    )


def _row_binding_digests(
    config: dict,
    features: np.ndarray,
    *,
    probe_ids: tuple[str, ...] | None = None,
    feature_schema_digest: str | None = None,
) -> tuple[str, ...]:
    identities = probe_ids or _probe_ids(config)
    schema_digest = feature_schema_digest or config[
        "construction_feature_schema"
    ]["feature_schema_digest"]
    return tuple(
        construction_feature_row_binding_digest(
            probe_id=probe_id,
            feature_schema_digest=schema_digest,
            feature_values=features[row_index],
        )
        for row_index, probe_id in enumerate(identities)
    )


def test_canonical_impulse_observability_config_is_frozen() -> None:
    config = load_impulse_observability_config(CONFIG_PATH)
    assert config["profile_id"] == IMPULSE_OBSERVABILITY_PROFILE_ID
    assert config["method_id"] == OBSERVER_SYNCHRONIZED_METHOD_ID
    assert config["flow_macro_interval_count"] == FLOW_MACRO_INTERVAL_COUNT
    assert config["watermark_state_dimension"] == WATERMARK_STATE_DIMENSION
    assert config["stage_basis_rank"] == STAGE_BASIS_RANK
    assert config["impulse_triage_video_count"] == IMPULSE_TRIAGE_VIDEO_COUNT
    assert config["impulse_probe_count"] == IMPULSE_PROBE_COUNT
    assert config["clean_repeat_count"] == IMPULSE_CLEAN_REPEAT_COUNT
    assert config["formal_result"] is False
    assert config["stage_progression_allowed"] is False
    assert config["observer_execution_allowed"] is False
    assert config["state_dynamics_construction_allowed"] is False
    assert config["notebook_change_allowed"] is False
    assert config["training_or_finetuning_allowed"] is False
    assert config["inference_time_sampler_control_only"] is True
    assert config["execution_identity"]["generation_model_revision"] == (
        "0fad780a534b6463e45facd96134c9f345acfa5b"
    )
    assert config["execution_identity"]["num_inference_steps"] == 8
    assert config["execution_identity"]["seed_value"] == 2201
    assert config["execution_identity"]["positive_prompt_text_sha256"] == (
        "c4f3a636c9c4393ebf98448f2c30c6648f7e9141a2886bac0cd950001ec03980"
    )


def test_construction_feature_schema_is_key_independent_and_digest_bound() -> None:
    feature = _config()["construction_feature_schema"]
    assert feature["encoder_id"] == CONSTRUCTION_FEATURE_ENCODER_ID
    assert feature["encoder_revision"] == (
        CONSTRUCTION_FEATURE_ENCODER_REVISION
    )
    assert feature["candidate_key_independent"] is True
    assert feature["freeze_before_impulse_generation"] is True
    assert feature["impulse_sample_training_allowed"] is False
    assert feature["owner_label_access_allowed"] is False
    assert feature["postrun_feature_selection_allowed"] is False
    assert feature["output_dimension"] == CONSTRUCTION_FEATURE_OUTPUT_DIMENSION
    assert feature["feature_schema_digest"] == (
        construction_feature_schema_digest(feature)
    )
    assert feature["diffusers_version"] == "0.35.2"
    assert feature["encode_execution_dtype"] == "bfloat16"
    assert feature["input_frame_count"] == 33
    assert feature["input_height"] == 320
    assert feature["input_width"] == 512
    assert feature["input_resize_allowed"] is False
    assert feature["output_row_identity_binding"] == (
        "exact_frozen_probe_ids_from_per_video_feature_records"
    )
    assert feature["output_row_binding_digest_algorithm"] == (
        "sha256_probe_id_feature_schema_digest_and_"
        "little_endian_float64_feature_bytes"
    )
    assert feature["output_row_identity_posthoc_matrix_labeling_allowed"] is (
        False
    )
    assert feature["streaming_memory_config"] == {
        "maximum_incremental_cuda_peak_gib": 16.0,
        "minimum_cuda_free_gib": 12.0,
        "temporal_chunk_frame_count": 4,
        "tile_sample_height": 128,
        "tile_sample_stride_height": 96,
        "tile_sample_stride_width": 96,
        "tile_sample_width": 128,
    }


def test_phi_construction_numpy_reference_freezes_pooling_and_l2() -> None:
    latent = np.zeros(CONSTRUCTION_LATENT_LAYOUT_SHAPE, dtype=np.float32)
    expected = []
    for channel in range(16):
        for row in range(4):
            for column in range(4):
                value = float(1 + channel * 16 + row * 4 + column)
                latent[
                    0,
                    channel,
                    :,
                    row * 10 : (row + 1) * 10,
                    column * 16 : (column + 1) * 16,
                ] = value
                expected.append(value)
    feature = extract_construction_output_feature_from_normalized_latent(
        latent
    )
    reencoded = (
        extract_construction_reencoded_summary_from_normalized_latent(latent)
    )
    assert reencoded.shape == (256,)
    assert np.allclose(reencoded, np.asarray(expected, dtype=np.float64))
    expected_array = np.asarray(expected, dtype=np.float64)
    expected_array /= np.linalg.norm(expected_array)
    assert feature.dtype == np.float64
    assert feature.shape == (256,)
    assert np.array_equal(feature, expected_array)
    assert np.linalg.norm(feature) == pytest.approx(1.0)


def test_phi_construction_rejects_wrong_shape_dtype_zero_and_nonfinite() -> None:
    with pytest.raises(ValueError, match=r"\[1,16,9,40,64\]"):
        extract_construction_output_feature_from_normalized_latent(
            np.ones((1, 16, 8, 40, 64), dtype=np.float32)
        )
    with pytest.raises(ValueError, match="float32"):
        extract_construction_output_feature_from_normalized_latent(
            np.ones(CONSTRUCTION_LATENT_LAYOUT_SHAPE, dtype=np.float64)
        )
    with pytest.raises(ValueError, match="zero rejection"):
        extract_construction_output_feature_from_normalized_latent(
            np.zeros(CONSTRUCTION_LATENT_LAYOUT_SHAPE, dtype=np.float32)
        )
    nonfinite = np.ones(CONSTRUCTION_LATENT_LAYOUT_SHAPE, dtype=np.float32)
    nonfinite[0, 0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="非有限"):
        extract_construction_output_feature_from_normalized_latent(
            nonfinite
        )


def test_schedule_and_runtime_adapter_digests_bind_complete_schemas() -> None:
    config = _config()
    schedule = config["flow_schedule_contract"]
    adapter = config["runtime_adapter_contract"]
    assert schedule["waveform_schema_digest"] == (
        flow_schedule_waveform_schema_digest(schedule)
    )
    assert adapter["adapter_schema_digest"] == (
        runtime_adapter_schema_digest(adapter)
    )
    assert schedule["step_indices"] == list(range(8))
    assert len(schedule["sigma_grid"]) == 9
    assert len(schedule["flow_phase_by_step"]) == 8
    assert len(schedule["delta_sigma_by_step"]) == 8
    assert schedule["macro_interval_index_by_step"] == [
        0,
        0,
        0,
        0,
        1,
        1,
        2,
        2,
    ]


def test_construction_basis_kdf_and_cpu_basis_are_reproducible() -> None:
    assert set(
        inspect.signature(derive_construction_basis_subkey).parameters
    ) == {"master_key_text", "wrong_key_candidate_index"}
    key = "owner-construction-master-key"
    owner_subkey = derive_construction_basis_subkey(key)
    wrong_subkey = derive_construction_basis_subkey(
        key,
        wrong_key_candidate_index=0,
    )
    assert owner_subkey != wrong_subkey
    owner = build_construction_stage_basis(key)
    repeated = build_construction_stage_basis(key)
    wrong = build_construction_stage_basis(
        key,
        wrong_key_candidate_index=0,
    )
    assert owner.latent_layout_shape == CONSTRUCTION_LATENT_LAYOUT_SHAPE
    assert owner.values.shape == (math.prod(CONSTRUCTION_LATENT_LAYOUT_SHAPE), 6)
    assert owner.values.dtype == np.dtype("<f4")
    assert owner.basis_digest == repeated.basis_digest
    assert np.array_equal(owner.values, repeated.values)
    assert owner.basis_digest != wrong.basis_digest
    assert np.max(
        np.abs(
            owner.values.astype(np.float64).T
            @ owner.values.astype(np.float64)
            - np.eye(6)
        )
    ) <= 1e-5
    wrong_coherence = np.max(
        np.abs(
            owner.values.astype(np.float64).T
            @ wrong.values.astype(np.float64)
        )
    )
    assert wrong_coherence <= 0.25
    selectors = build_stage_selector_blocks()
    assert all(
        np.array_equal(
            owner.values @ selector,
            owner.values[:, stage_index * 2 : stage_index * 2 + 2],
        )
        for stage_index, selector in enumerate(selectors)
    )


def test_intended_control_uses_state_update_polarity_and_both_budgets() -> None:
    positive = compute_intended_impulse_control(
        probe_state_update_polarity=1,
        temporal_waveform=1.0,
        delta_sigma=-0.1,
        base_velocity_norm=100.0,
        cumulative_control_energy=0.0,
        cumulative_reference_energy=0.0,
        remaining_step_count=8,
    )
    negative = compute_intended_impulse_control(
        probe_state_update_polarity=-1,
        temporal_waveform=1.0,
        delta_sigma=-0.1,
        base_velocity_norm=100.0,
        cumulative_control_energy=0.0,
        cumulative_reference_energy=0.0,
        remaining_step_count=8,
    )
    assert positive.norm_limited_delta_norm == pytest.approx(0.24)
    assert positive.intended_delta_norm <= positive.energy_limited_delta_norm
    assert positive.signed_velocity_coordinate < 0.0
    assert positive.signed_state_update_exposure > 0.0
    assert negative.signed_velocity_coordinate > 0.0
    assert negative.signed_state_update_exposure < 0.0


@pytest.mark.parametrize(
    ("path", "mutated_value"),
    [
        (("formal_result",), True),
        (("stage_progression_allowed",), True),
        (("observer_execution_allowed",), True),
        (("training_or_finetuning_allowed",), True),
        (("execution_identity", "num_inference_steps"), 20),
        (
            ("execution_identity", "positive_prompt_text_sha256"),
            "0" * 64,
        ),
        (("time_axes", "video_frame_time_analysis_allowed"), True),
        (("stage_basis", "construction"), "B_K_j_equals_U_K_R_j"),
        (
            (
                "construction_feature_schema",
                "candidate_key_independent",
            ),
            False,
        ),
        (
            (
                "construction_feature_schema",
                "postrun_feature_selection_allowed",
            ),
            True,
        ),
        (
            (
                "construction_feature_schema",
                "input_resize_allowed",
            ),
            True,
        ),
        (
            (
                "construction_feature_schema",
                "output_row_identity_posthoc_matrix_labeling_allowed",
            ),
            True,
        ),
        (
            (
                "construction_feature_schema",
                "output_row_binding_digest_algorithm",
            ),
            "caller_chosen_digest",
        ),
        (
            ("construction_basis", "subkey_domain"),
            "caller_chosen_domain",
        ),
        (
            ("flow_schedule_contract", "step_count"),
            1,
        ),
        (
            (
                "actual_exposure_contract",
                "signed_exposure_required",
            ),
            False,
        ),
        (
            (
                "actual_exposure_contract",
                "maximum_actual_design_condition_number",
            ),
            100.0,
        ),
        (
            (
                "gate_b_cross_identity_identifiability",
                "test_identity_procrustes_allowed",
            ),
            True,
        ),
        (
            (
                "authorization_state_machine",
                "impulse_triage_execution_allowed",
            ),
            False,
        ),
        (
            (
                "authorization_state_machine",
                "current_state",
            ),
            "sample_internal_causal_observability_gate",
        ),
        (
            (
                "gate_a_sample_internal_causal_observability",
                "replay_diagnostic_required",
            ),
            True,
        ),
        (
            (
                "gate_a_sample_internal_causal_observability",
                "zero_clean_distance_auto_pass_allowed",
            ),
            True,
        ),
        (
            (
                "key_selectivity_construction",
                "adaptive_wrong_key_selection_allowed",
            ),
            True,
        ),
        (
            (
                "gate_c_composite_trajectory_order_identifiability",
                "dynamics_contribution_claim_allowed",
            ),
            True,
        ),
        (
            (
                "future_observer_boundary",
                "formal_llr_name_allowed",
            ),
            True,
        ),
    ],
)
def test_contract_mutations_are_rejected_fail_closed(
    path: tuple[str, ...],
    mutated_value: object,
) -> None:
    config = deepcopy(_config())
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = mutated_value
    with pytest.raises(ValueError):
        validate_impulse_observability_config(config)


def test_feature_schema_mutation_without_digest_update_is_rejected() -> None:
    config = deepcopy(_config())
    config["construction_feature_schema"]["output_dimension"] = 96
    with pytest.raises(ValueError, match="feature extractor"):
        validate_impulse_observability_config(config)


def test_unregistered_top_level_authorization_is_rejected() -> None:
    config = deepcopy(_config())
    config["observer_override_allowed"] = True
    with pytest.raises(ValueError, match="顶层字段集合"):
        validate_impulse_observability_config(config)


def test_stage_selectors_define_three_disjoint_two_dimensional_blocks() -> None:
    selectors = build_stage_selector_blocks()
    assert len(selectors) == 3
    for selector in selectors:
        assert selector.shape == (6, 2)
        assert np.allclose(selector.T @ selector, np.eye(2))
    for left_index, left in enumerate(selectors):
        for right_index, right in enumerate(selectors):
            gram = left.T @ right
            if left_index == right_index:
                assert np.allclose(gram, np.eye(2))
            else:
                assert np.allclose(gram, np.zeros((2, 2)))
    assert not np.array_equal(selectors[0], selectors[1])
    assert not np.array_equal(selectors[1], selectors[2])


def test_impulse_plan_is_exactly_clean_plus_six_positive_six_negative() -> None:
    plan = build_impulse_triage_plan(_config())
    assert len(plan) == 14
    assert [record.probe_id for record in plan[:2]] == [
        "clean_a",
        "clean_b",
    ]
    assert all(record.polarity == 1 for record in plan[2:8])
    assert all(record.polarity == -1 for record in plan[8:14])
    assert {
        (record.stage_index, record.channel_index)
        for record in plan[2:8]
    } == {
        (stage, channel)
        for stage in range(3)
        for channel in range(2)
    }
    assert {
        (record.stage_index, record.channel_index)
        for record in plan[8:14]
    } == {
        (stage, channel)
        for stage in range(3)
        for channel in range(2)
    }
    assert all(record.formal_result is False for record in plan)
    assert all(record.stage_progression_allowed is False for record in plan)


def test_actual_design_uses_signed_exposure_and_is_well_conditioned() -> None:
    config = _config()
    design = assemble_actual_design_matrix(config, _valid_traces(config))
    assert design.values.shape == (6, 14)
    assert np.array_equal(design.values[:, :2], np.zeros((6, 2)))
    positive = design.values[:, 2:8]
    negative = design.values[:, 8:14]
    assert np.all(np.diag(positive) > 0.0)
    assert np.allclose(positive, np.diag(np.diag(positive)))
    assert np.allclose(negative, -positive)
    assert design.rank == 6
    assert design.condition_number < 2.0
    assert design.compression_allowed is True
    assert all(
        value == pytest.approx(1.0)
        for value in design.waveform_cosine_by_probe.values()
    )
    assert all(
        value == pytest.approx(0.0)
        for value in design.cross_channel_leakage_by_probe.values()
    )


def _slice_trace_to_one_step(
    trace: ActualImpulseExposureTrace,
) -> ActualImpulseExposureTrace:
    values = dict(trace.__dict__)
    for field_name, field_value in tuple(values.items()):
        if (
            isinstance(field_value, tuple)
            and field_name not in {"actual_exposure_vector"}
            and len(field_value) == CONSTRUCTION_FLOW_STEP_COUNT
        ):
            values[field_name] = field_value[:1]
    return ActualImpulseExposureTrace(**values)


def test_one_step_caller_chosen_waveform_forgery_is_rejected() -> None:
    config = _config()
    traces = list(_valid_traces(config))
    forged = _slice_trace_to_one_step(traces[0])
    forged = replace(
        forged,
        waveform_schema_digest="b" * 64,
    )
    traces[0] = forged
    with pytest.raises(ValueError, match="精确覆盖8步"):
        assemble_actual_design_matrix(config, traces)


@pytest.mark.parametrize(
    ("field_name", "mutated"),
    [
        ("step_indices", (1, 0, 2, 3, 4, 5, 6, 7)),
        (
            "flow_phase_by_step",
            (0.03, 0.08483484387397766, 0.15818744897842407,
             0.2526847720146179, 0.3790806531906128,
             0.5569911301136017, 0.8265496320091188,
             0.9955357140861452),
        ),
        (
            "delta_sigma_by_step",
            (-0.05, -0.06475478410720825, -0.08195042610168457,
             -0.10704421997070312, -0.14574754238128662,
             -0.21007341146469116, -0.32904359698295593,
             -0.008928571827709675),
        ),
        ("waveform_schema_digest", "b" * 64),
    ],
)
def test_trace_schedule_mutations_fail_closed(
    field_name: str,
    mutated: object,
) -> None:
    config = _config()
    traces = list(_valid_traces(config))
    traces[0] = replace(traces[0], **{field_name: mutated})
    with pytest.raises(ValueError):
        assemble_actual_design_matrix(config, traces)


def test_trace_rejects_nonfinite_or_out_of_range_direction_cosine() -> None:
    config = _config()
    traces = list(_valid_traces(config))
    traces[0] = replace(
        traces[0],
        direction_cosine_by_step=(
            float("nan"),
            *traces[0].direction_cosine_by_step[1:],
        ),
    )
    with pytest.raises(ValueError, match="非有限"):
        assemble_actual_design_matrix(config, traces)
    traces = list(_valid_traces(config))
    traces[0] = replace(
        traces[0],
        direction_cosine_by_step=(
            1.1,
            *traces[0].direction_cosine_by_step[1:],
        ),
    )
    with pytest.raises(ValueError, match=r"\[-1,1\]"):
        assemble_actual_design_matrix(config, traces)


def test_trace_rejects_active_waveform_with_zero_feasible_control() -> None:
    config = _config()
    traces = list(_valid_traces(config))
    trace = traces[0]

    def replace_first(values: tuple, replacement: object) -> tuple:
        return (replacement, *values[1:])

    zero_channels = (0.0,) * STAGE_BASIS_RANK
    traces[0] = replace(
        trace,
        reference_base_velocity_norm_by_step=replace_first(
            trace.reference_base_velocity_norm_by_step,
            0.0,
        ),
        remaining_control_energy_before_step_by_step=replace_first(
            trace.remaining_control_energy_before_step_by_step,
            0.0,
        ),
        reference_energy_increment_by_step=replace_first(
            trace.reference_energy_increment_by_step,
            0.0,
        ),
        reference_cumulative_energy_by_step=replace_first(
            trace.reference_cumulative_energy_by_step,
            0.0,
        ),
        intended_delta_norm_by_step=replace_first(
            trace.intended_delta_norm_by_step,
            0.0,
        ),
        actual_velocity_basis_coordinate_by_step=replace_first(
            trace.actual_velocity_basis_coordinate_by_step,
            0.0,
        ),
        actual_channel_velocity_coordinate_by_step=replace_first(
            trace.actual_channel_velocity_coordinate_by_step,
            zero_channels,
        ),
        intended_signed_exposure_by_step=replace_first(
            trace.intended_signed_exposure_by_step,
            0.0,
        ),
        actual_signed_exposure_by_step=replace_first(
            trace.actual_signed_exposure_by_step,
            0.0,
        ),
        actual_channel_exposure_by_step=replace_first(
            trace.actual_channel_exposure_by_step,
            zero_channels,
        ),
        actual_exposure_vector=tuple(
            np.asarray(trace.actual_exposure_vector)
            - np.asarray(trace.actual_channel_exposure_by_step[0])
        ),
        delta_norm_by_step=replace_first(
            trace.delta_norm_by_step,
            0.0,
        ),
        projection_scale_by_step=replace_first(
            trace.projection_scale_by_step,
            0.0,
        ),
        cumulative_energy_by_step=replace_first(
            trace.cumulative_energy_by_step,
            0.0,
        ),
    )
    with pytest.raises(ValueError, match="可行非零 intended control"):
        assemble_actual_design_matrix(config, traces)


def test_actual_design_rejects_cross_channel_leakage() -> None:
    config = _config()
    traces = list(_valid_traces(config))
    target = traces[0]
    velocity = [
        list(values)
        for values in target.actual_channel_velocity_coordinate_by_step
    ]
    active_index = 0
    original = velocity[active_index][0]
    velocity[active_index][0] = original * math.sqrt(1.0 - 0.01)
    velocity[active_index][1] = abs(original) * 0.1
    channel_exposure = [
        tuple(
            target.delta_sigma_by_step[index] * value
            for value in values
        )
        for index, values in enumerate(velocity)
    ]
    signed = tuple(values[0] for values in channel_exposure)
    vector = tuple(np.asarray(channel_exposure).sum(axis=0))
    traces[0] = ActualImpulseExposureTrace(
        **{
            **target.__dict__,
            "actual_velocity_basis_coordinate_by_step": tuple(
                values[0] for values in velocity
            ),
            "actual_channel_velocity_coordinate_by_step": tuple(
                tuple(values) for values in velocity
            ),
            "actual_signed_exposure_by_step": signed,
            "actual_channel_exposure_by_step": tuple(channel_exposure),
            "actual_exposure_vector": vector,
            "direction_cosine_by_step": (
                math.sqrt(1.0 - 0.01),
                *target.direction_cosine_by_step[1:],
            ),
        }
    )
    with pytest.raises(ValueError, match="direction guard|cross-channel leakage"):
        assemble_actual_design_matrix(config, traces)


def test_actual_design_rejects_step_direction_guard_failure() -> None:
    config = _config()
    traces = list(_valid_traces(config))
    target = traces[0]
    traces[0] = ActualImpulseExposureTrace(
        **{
            **target.__dict__,
            "direction_cosine_by_step": (
                0.998,
                *target.direction_cosine_by_step[1:],
            ),
        }
    )
    with pytest.raises(ValueError, match="direction cosine 未重算"):
        assemble_actual_design_matrix(config, traces)


def test_actual_design_rejects_unsigned_or_reversed_target_exposure() -> None:
    config = _config()
    traces = list(_valid_traces(config))
    target = traces[0]
    traces[0] = ActualImpulseExposureTrace(
        **{
            **target.__dict__,
            "intended_signed_exposure_by_step": tuple(
                -value
                for value in target.intended_signed_exposure_by_step
            ),
        }
    )
    with pytest.raises(ValueError, match="intended exposure"):
        assemble_actual_design_matrix(config, traces)


def test_actual_design_rejects_waveform_collapse_and_preserves_stepwise_input() -> None:
    config = _config()
    probe_id = "positive_early_flow_channel_0"
    traces = _valid_traces(
        config,
        actual_step_scale_by_probe={
            probe_id: (0.01, 0.01, 0.01, 1.0, 0.0, 0.0, 0.0, 0.0)
        },
    )
    with pytest.raises(ValueError, match="waveform cosine"):
        assemble_actual_design_matrix(config, traces)
    assert len(traces[0].actual_signed_exposure_by_step) == 8


def test_actual_design_rejects_positive_negative_asymmetry() -> None:
    config = _config()
    traces = _valid_traces(
        config,
        actual_scale_by_probe={
            "negative_early_flow_channel_0": 0.6,
        },
    )
    with pytest.raises(ValueError, match="amplitude symmetry"):
        assemble_actual_design_matrix(config, traces)


def test_transfer_estimate_uses_clean_intercept_and_actual_design() -> None:
    config = _config()
    design = assemble_actual_design_matrix(config, _valid_traces(config))
    features = _normalized_probe_features()
    validated = validate_construction_output_features(
        config,
        features,
        feature_schema_digest=config["construction_feature_schema"][
            "feature_schema_digest"
        ],
        probe_ids=_probe_ids(config),
        row_binding_digests=_row_binding_digests(config, features),
    )
    estimate = estimate_construction_transfer(config, validated, design)
    assert np.array_equal(estimate.clean_intercept, features[0])
    for coordinate in range(6):
        amplitude_difference = (
            design.values[coordinate, 2 + coordinate]
            - design.values[coordinate, 8 + coordinate]
        )
        assert estimate.signed_center_difference_transfer_matrix[
            1 + coordinate,
            coordinate,
        ] == pytest.approx(0.02 / amplitude_difference)


def test_transfer_estimate_rejects_reordered_actual_design_identity() -> None:
    config = _config()
    design = assemble_actual_design_matrix(config, _valid_traces(config))
    reordered = replace(
        design,
        probe_ids=(
            design.probe_ids[1],
            design.probe_ids[0],
            *design.probe_ids[2:],
        ),
    )
    features = validate_construction_output_features(
        config,
        _normalized_probe_features(),
        feature_schema_digest=config["construction_feature_schema"][
            "feature_schema_digest"
        ],
        probe_ids=_probe_ids(config),
        row_binding_digests=_row_binding_digests(
            config,
            _normalized_probe_features(),
        ),
    )
    with pytest.raises(ValueError, match="identity"):
        estimate_construction_transfer(
            config,
            features,
            reordered,
        )


def test_feature_validation_rejects_zero_wrong_l2_and_wrong_schema() -> None:
    config = _config()
    digest = config["construction_feature_schema"]["feature_schema_digest"]
    features = _normalized_probe_features()
    zero = features.copy()
    zero[0] = 0.0
    with pytest.raises(ValueError, match="zero rejection"):
        validate_construction_output_features(
            config,
            zero,
            feature_schema_digest=digest,
            probe_ids=_probe_ids(config),
            row_binding_digests=_row_binding_digests(config, zero),
        )
    wrong_norm = features.copy()
    wrong_norm[0] *= 0.5
    with pytest.raises(ValueError, match="归一化"):
        validate_construction_output_features(
            config,
            wrong_norm,
            feature_schema_digest=digest,
            probe_ids=_probe_ids(config),
            row_binding_digests=_row_binding_digests(
                config,
                wrong_norm,
            ),
        )
    with pytest.raises(ValueError, match="schema digest"):
        validate_construction_output_features(
            config,
            features,
            feature_schema_digest="f" * 64,
            probe_ids=_probe_ids(config),
            row_binding_digests=_row_binding_digests(
                config,
                features,
                feature_schema_digest="f" * 64,
            ),
        )


def test_feature_rows_reordered_with_ids_fail_closed() -> None:
    config = _config()
    features = _normalized_probe_features()
    probe_ids = _probe_ids(config)
    permutation = (0, 1, 3, 2, 4, 5, 6, 7, 9, 8, 10, 11, 12, 13)
    reordered_values = features[list(permutation)]
    reordered_ids = tuple(probe_ids[index] for index in permutation)
    reordered_bindings = _row_binding_digests(
        config,
        reordered_values,
        probe_ids=reordered_ids,
    )
    with pytest.raises(ValueError, match="精确匹配冻结14-video顺序"):
        validate_construction_output_features(
            config,
            reordered_values,
            feature_schema_digest=config["construction_feature_schema"][
                "feature_schema_digest"
            ],
            probe_ids=reordered_ids,
            row_binding_digests=reordered_bindings,
        )


def test_feature_value_only_reorder_breaks_row_binding_digest() -> None:
    config = _config()
    features = _normalized_probe_features()
    probe_ids = _probe_ids(config)
    original_bindings = _row_binding_digests(config, features)
    reordered_values = features.copy()
    reordered_values[[2, 3]] = reordered_values[[3, 2]]
    reordered_values[[8, 9]] = reordered_values[[9, 8]]
    with pytest.raises(ValueError, match="row binding digest"):
        validate_construction_output_features(
            config,
            reordered_values,
            feature_schema_digest=config["construction_feature_schema"][
                "feature_schema_digest"
            ],
            probe_ids=probe_ids,
            row_binding_digests=original_bindings,
        )


@pytest.mark.parametrize("identity_mutation", ["duplicate", "missing", "unknown"])
def test_feature_probe_identity_set_mutations_are_rejected(
    identity_mutation: str,
) -> None:
    config = _config()
    features = _normalized_probe_features()
    probe_ids = list(_probe_ids(config))
    if identity_mutation == "duplicate":
        probe_ids[3] = probe_ids[2]
    elif identity_mutation == "missing":
        probe_ids.pop()
    else:
        probe_ids[3] = "unknown_impulse_probe"
    with pytest.raises(ValueError, match="probe_ids"):
        validate_construction_output_features(
            config,
            features,
            feature_schema_digest=config["construction_feature_schema"][
                "feature_schema_digest"
            ],
            probe_ids=tuple(probe_ids),
            row_binding_digests=("a" * 64,) * len(probe_ids),
        )


def test_estimator_rejects_forged_feature_identity_even_after_validation() -> None:
    config = _config()
    design = assemble_actual_design_matrix(config, _valid_traces(config))
    features = _normalized_probe_features()
    validated = validate_construction_output_features(
        config,
        features,
        feature_schema_digest=config["construction_feature_schema"][
            "feature_schema_digest"
        ],
        probe_ids=_probe_ids(config),
        row_binding_digests=_row_binding_digests(config, features),
    )
    swapped_ids = list(validated.probe_ids)
    swapped_ids[2], swapped_ids[3] = swapped_ids[3], swapped_ids[2]
    forged = replace(validated, probe_ids=tuple(swapped_ids))
    with pytest.raises(ValueError, match="identity"):
        estimate_construction_transfer(config, forged, design)
    with pytest.raises(ValueError, match="identity"):
        evaluate_gate_a_statistics(
            config,
            forged,
            design,
            primary_checkpoint_ready=_ready_primary_checkpoints(),
        )


def test_center_difference_divides_asymmetric_actual_amplitudes() -> None:
    config = _config()
    traces = _valid_traces(
        config,
        actual_scale_by_probe={
            "negative_early_flow_channel_0": 0.77,
        },
    )
    design = assemble_actual_design_matrix(config, traces)
    transfer_gain = 0.1
    features = np.zeros(
        (14, CONSTRUCTION_FEATURE_OUTPUT_DIMENSION),
        dtype=np.float64,
    )
    features[:2, 0] = 1.0
    for coordinate in range(6):
        for row_index in (2 + coordinate, 8 + coordinate):
            response = transfer_gain * design.values[
                coordinate,
                row_index,
            ]
            features[row_index, 0] = math.sqrt(1.0 - response**2)
            features[row_index, 1 + coordinate] = response
    validated = validate_construction_output_features(
        config,
        features,
        feature_schema_digest=config["construction_feature_schema"][
            "feature_schema_digest"
        ],
        probe_ids=_probe_ids(config),
        row_binding_digests=_row_binding_digests(config, features),
    )
    estimate = estimate_construction_transfer(
        config,
        validated,
        design,
    )
    assert estimate.signed_center_difference_transfer_matrix[
        1,
        0,
    ] == pytest.approx(transfer_gain)


def test_estimator_recomputes_clean_zero_rank_and_condition() -> None:
    config = _config()
    design = assemble_actual_design_matrix(config, _valid_traces(config))
    features = validate_construction_output_features(
        config,
        _normalized_probe_features(),
        feature_schema_digest=config["construction_feature_schema"][
            "feature_schema_digest"
        ],
        probe_ids=_probe_ids(config),
        row_binding_digests=_row_binding_digests(
            config,
            _normalized_probe_features(),
        ),
    )
    nonzero_clean_values = design.values.copy()
    nonzero_clean_values[0, 0] = 1e-6
    with pytest.raises(ValueError, match="clean"):
        estimate_construction_transfer(
            config,
            features,
            replace(design, values=nonzero_clean_values),
        )
    ill_conditioned = design.values.copy()
    ill_conditioned[5] *= 1e-9
    with pytest.raises(ValueError, match="rank/condition"):
        estimate_construction_transfer(
            config,
            features,
            replace(design, values=ill_conditioned),
        )


def test_zero_clean_distance_uses_finite_floor_and_cannot_auto_pass() -> None:
    config = _config()
    design = assemble_actual_design_matrix(config, _valid_traces(config))
    tiny = validate_construction_output_features(
        config,
        _normalized_probe_features(response_amplitudes=(1e-8,) * 6),
        feature_schema_digest=config["construction_feature_schema"][
            "feature_schema_digest"
        ],
        probe_ids=_probe_ids(config),
        row_binding_digests=_row_binding_digests(
            config,
            _normalized_probe_features(
                response_amplitudes=(1e-8,) * 6
            ),
        ),
    )
    statistics = evaluate_gate_a_statistics(
        config,
        tiny,
        design,
        primary_checkpoint_ready=_ready_primary_checkpoints(),
    )
    assert statistics.clean_repeat_distance == 0.0
    assert statistics.finite_noise_floor == pytest.approx(1e-6)
    assert math.isfinite(statistics.minimum_output_feature_snr)
    assert math.isfinite(
        statistics.noise_normalized_minimum_singular_value
    )
    assert statistics.gate_a_ready is False
    assert statistics.formal_result is False
    assert statistics.stage_progression_allowed is False


def test_gate_a_formulas_are_finite_and_executable_for_visible_response() -> None:
    config = _config()
    design = assemble_actual_design_matrix(config, _valid_traces(config))
    response_amplitudes = tuple(
        0.1 * float(design.values[index, 2 + index])
        for index in range(6)
    )
    visible = validate_construction_output_features(
        config,
        _normalized_probe_features(
            response_amplitudes=response_amplitudes,
            clean_separation=2e-6,
        ),
        feature_schema_digest=config["construction_feature_schema"][
            "feature_schema_digest"
        ],
        probe_ids=_probe_ids(config),
        row_binding_digests=_row_binding_digests(
            config,
            _normalized_probe_features(
                response_amplitudes=response_amplitudes,
                clean_separation=2e-6,
            ),
        ),
    )
    statistics = evaluate_gate_a_statistics(
        config,
        visible,
        design,
        primary_checkpoint_ready=_ready_primary_checkpoints(),
    )
    assert statistics.effective_rank == 6
    assert statistics.output_transfer_condition_number == pytest.approx(1.0)
    assert statistics.minimum_positive_negative_antisymmetry_cosine > 0.99
    assert statistics.maximum_positive_negative_antisymmetry_residual_ratio < 0.01
    assert statistics.gate_a_ready is True


def test_gate_a_requires_five_primary_checkpoints_but_not_replay() -> None:
    config = _config()
    design = assemble_actual_design_matrix(config, _valid_traces(config))
    response_amplitudes = tuple(
        0.1 * float(design.values[index, 2 + index])
        for index in range(6)
    )
    visible = validate_construction_output_features(
        config,
        _normalized_probe_features(
            response_amplitudes=response_amplitudes,
            clean_separation=2e-6,
        ),
        feature_schema_digest=config["construction_feature_schema"][
            "feature_schema_digest"
        ],
        probe_ids=_probe_ids(config),
        row_binding_digests=_row_binding_digests(
            config,
            _normalized_probe_features(
                response_amplitudes=response_amplitudes,
                clean_separation=2e-6,
            ),
        ),
    )
    statuses = _ready_primary_checkpoints()
    statuses["T_saved_video"] = False
    statistics = evaluate_gate_a_statistics(
        config,
        visible,
        design,
        primary_checkpoint_ready=statuses,
    )
    assert statistics.primary_checkpoint_chain_ready is False
    assert statistics.gate_a_ready is False
    missing = _ready_primary_checkpoints()
    missing.pop("T_decoded")
    with pytest.raises(ValueError, match="checkpoint readiness"):
        evaluate_gate_a_statistics(
            config,
            visible,
            design,
            primary_checkpoint_ready=missing,
        )


def test_replay_checkpoint_is_optional_and_never_primary_support() -> None:
    config = _config()
    gate = config["gate_a_sample_internal_causal_observability"]
    replay = config["transfer_checkpoint_contract"][
        "T_replay_diagnostic"
    ]
    assert gate["replay_diagnostic_required"] is False
    assert "T_replay_diagnostic" not in (
        gate["primary_required_transfer_checkpoints"]
    )
    assert replay["required"] is False
    assert replay["primary_gate_support_allowed"] is False
    assert set(gate["primary_required_transfer_checkpoints"]) == {
        "T_latent",
        "T_decoded",
        "T_saved_video",
        "T_reencoded",
        "T_output_feature",
    }


def test_primary_gate_and_future_authorizations_remain_closed() -> None:
    config = _config()
    assert config["primary_gate_checkpoint"] == "T_output_feature"
    assert config["prompt_conditioned_replay_role"] == (
        "construction_diagnostic_only"
    )
    assert config["authorization_state_machine"]["current_state"] == (
        "impulse_triage_execution_authorized_pending_user_colab_run"
    )
    assert config["authorization_state_machine"][
        "impulse_triage_execution_allowed"
    ] is True
    assert config["authorization_state_machine"][
        "batch_observer_design_allowed"
    ] is False
    assert config["future_observer_boundary"]["formal_llr_name_allowed"] is False


def test_contract_documents_freeze_output_only_and_gate_boundaries() -> None:
    method_contract = Path(
        "docs/builds/"
        "observer_synchronized_state_space_trajectory_method_contract.md"
    ).read_text(encoding="utf-8")
    probe_contract = Path(
        "docs/builds/output_feature_impulse_observability_probe.md"
    ).read_text(encoding="utf-8")
    required_method_terms = (
        "B_{K,j}=U_KE_j",
        "phi_{\\mathrm{construction}}",
        "T_{\\mathrm{output\\_feature}}",
        "A_{\\mathrm{actual}}",
        "prediction-error score",
        "formal_result=false",
        "stage_progression_allowed=false",
        "0.9904",
    )
    assert all(term in method_contract for term in required_method_terms)
    required_probe_terms = (
        "clean-A",
        "clean-B",
        "Gate A",
        "Gate B",
        "Gate C",
        "组合轨迹与阶段顺序可辨识",
        "test identity Procrustes",
        "same-energy permuted composite",
    )
    assert all(term in probe_contract for term in required_probe_terms)
