"""Lightweight Gate A runtime and Colab handler regressions."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

import numpy as np
import pytest

from evaluation.protocol.impulse_observability_contract import (
    CONSTRUCTION_DELTA_SIGMA,
    CONSTRUCTION_FEATURE_OUTPUT_DIMENSION,
    CONSTRUCTION_FLOW_PHASES,
    CONSTRUCTION_FLOW_STEP_COUNT,
    CONSTRUCTION_LATENT_LAYOUT_SHAPE,
    CONSTRUCTION_MACRO_INTERVAL_INDICES,
    CONSTRUCTION_TEMPORAL_WAVEFORMS,
    ActualImpulseExposureTrace,
    ConstructionStageBasis,
    build_construction_stage_basis,
    build_impulse_triage_plan,
    compute_intended_impulse_control,
    construction_feature_row_binding_digest,
    effective_construction_basis_directions,
    load_impulse_observability_config,
)
from experiments.generative_video_model_probe.output_feature_impulse_observability_construction import (
    DEFAULT_CONFIG_PATH,
    PRIMARY_CHECKPOINT_IDS,
    ConstructionFeatureBatch,
    ConstructionGenerationBatch,
    _checkpoint_record,
    _require_active_control_feasible,
    _validate_generation_batch,
    _validate_prompt_seed_source,
    apply_numpy_float32_impulse,
    build_output_feature_records,
    read_saved_video_rgb24_summary,
    measure_numpy_actual_delta_coordinates,
    run_output_feature_impulse_observability_construction,
    validate_checkpoint_and_feature_records,
    validate_output_vae_metadata,
)
from workflows.colab_test_request import (
    OUTPUT_FEATURE_IMPULSE_OBSERVABILITY_CONSTRUCTION_TEST_ID,
    REQUEST_SCHEMA_VERSION,
    build_colab_test_dry_run_plan,
    load_colab_test_request,
    package_colab_test_recovery_bundle,
    run_colab_test_request,
)


def _config() -> dict:
    return load_impulse_observability_config(DEFAULT_CONFIG_PATH)


def _prompt_suite() -> dict:
    return {
        "prompt_suite_id": "sstw_nested_paper_prompt_seed_universe_v1",
        "prompts": [
            {
                "prompt_id": "probe_paper_paper_master_prompt_003",
                "prompt_text": (
                    "Three brightly colored boxes travel on a conveyor belt "
                    "from left to right across the entire frame, fixed camera, "
                    "each box remains large and easy to track."
                ),
                "prompt_negative_text": (
                    "static image, frozen conveyor, subtle motion, blurry, "
                    "jittery, distorted"
                ),
                "prompt_category": "structured_motion",
                "prompt_suite_role": "probe_paper",
                "motion_pattern_id": "conveyor_translation_registered_00",
                "split": "probe_paper",
            }
        ],
        "seeds": [
            {
                "seed_id": "probe_paper_paper_master_test_seed_01",
                "seed_value": 2201,
                "prompt_suite_role": "probe_paper",
                "split": "test",
            }
        ],
    }


def _write_source_root(path: Path) -> None:
    suite = path / "datasets" / "prompt_seed_suite.json"
    suite.parent.mkdir(parents=True, exist_ok=True)
    suite.write_text(
        json.dumps(_prompt_suite(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_source_zip(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w") as archive:
        archive.writestr(
            "source/datasets/prompt_seed_suite.json",
            json.dumps(_prompt_suite(), ensure_ascii=False) + "\n",
        )


def _actual_traces(
    config: dict,
    *,
    scale: float = 0.9,
) -> tuple[ActualImpulseExposureTrace, ...]:
    traces = []
    for probe in build_impulse_triage_plan(config)[2:]:
        previous_reference = 0.0
        previous_control = 0.0
        base_norms = tuple(100.0 + index for index in range(8))
        remaining_values = []
        reference_increments = []
        reference_cumulative = []
        intended_norms = []
        intended_exposures = []
        actual_coordinates = []
        actual_target = []
        actual_exposures = []
        actual_signed = []
        delta_norms = []
        projection_scales = []
        cumulative_control = []
        directions = []
        norm_guards = []
        energy_guards = []
        waveform = CONSTRUCTION_TEMPORAL_WAVEFORMS[
            int(probe.stage_index)
        ]
        target = int(probe.stage_index) * 2 + int(probe.channel_index)
        for step_index in range(CONSTRUCTION_FLOW_STEP_COUNT):
            interval = CONSTRUCTION_DELTA_SIGMA[step_index]
            control = compute_intended_impulse_control(
                probe_state_update_polarity=probe.polarity,
                temporal_waveform=waveform[step_index],
                delta_sigma=interval,
                base_velocity_norm=base_norms[step_index],
                cumulative_control_energy=previous_control,
                cumulative_reference_energy=previous_reference,
                remaining_step_count=CONSTRUCTION_FLOW_STEP_COUNT
                - step_index,
            )
            coordinate = [0.0] * 6
            actual_norm = control.intended_delta_norm * scale
            coordinate[target] = control.signed_velocity_coordinate * scale
            exposure = [interval * value for value in coordinate]
            increment = interval**2 * actual_norm**2
            previous_reference += control.reference_energy_increment
            previous_control += increment
            remaining_values.append(control.remaining_control_energy)
            reference_increments.append(control.reference_energy_increment)
            reference_cumulative.append(previous_reference)
            intended_norms.append(control.intended_delta_norm)
            intended_exposures.append(control.signed_state_update_exposure)
            actual_coordinates.append(tuple(coordinate))
            actual_target.append(coordinate[target])
            actual_exposures.append(tuple(exposure))
            actual_signed.append(exposure[target])
            delta_norms.append(actual_norm)
            projection_scales.append(
                scale if control.intended_delta_norm > 0.0 else 0.0
            )
            cumulative_control.append(previous_control)
            directions.append(1.0)
            norm_guards.append(True)
            energy_guards.append(True)
        traces.append(
            ActualImpulseExposureTrace(
                probe_id=probe.probe_id,
                stage_index=int(probe.stage_index),
                channel_index=int(probe.channel_index),
                polarity=probe.polarity,
                step_indices=tuple(range(8)),
                flow_phase_by_step=CONSTRUCTION_FLOW_PHASES,
                delta_sigma_by_step=CONSTRUCTION_DELTA_SIGMA,
                macro_interval_index_by_step=(
                    CONSTRUCTION_MACRO_INTERVAL_INDICES
                ),
                intended_velocity_waveform_by_step=waveform,
                reference_base_velocity_norm_by_step=base_norms,
                remaining_control_energy_before_step_by_step=tuple(
                    remaining_values
                ),
                reference_energy_increment_by_step=tuple(
                    reference_increments
                ),
                reference_cumulative_energy_by_step=tuple(
                    reference_cumulative
                ),
                intended_delta_norm_by_step=tuple(intended_norms),
                actual_velocity_basis_coordinate_by_step=tuple(
                    actual_target
                ),
                actual_channel_velocity_coordinate_by_step=tuple(
                    actual_coordinates
                ),
                intended_signed_exposure_by_step=tuple(
                    intended_exposures
                ),
                actual_signed_exposure_by_step=tuple(actual_signed),
                actual_channel_exposure_by_step=tuple(actual_exposures),
                actual_exposure_vector=tuple(
                    np.asarray(actual_exposures, dtype=np.float64).sum(
                        axis=0
                    )
                ),
                delta_norm_by_step=tuple(delta_norms),
                projection_scale_by_step=tuple(projection_scales),
                cumulative_energy_by_step=tuple(cumulative_control),
                direction_cosine_by_step=tuple(directions),
                norm_guard_passed_by_step=tuple(norm_guards),
                energy_guard_passed_by_step=tuple(energy_guards),
                waveform_schema_digest=config["flow_schedule_contract"][
                    "waveform_schema_digest"
                ],
                runtime_adapter_schema_digest=config[
                    "runtime_adapter_contract"
                ]["adapter_schema_digest"],
                basis_digest="a" * 64,
            )
        )
    return tuple(traces)


def _feature_matrix(response: float = 0.02) -> np.ndarray:
    features = np.zeros(
        (14, CONSTRUCTION_FEATURE_OUTPUT_DIMENSION),
        dtype=np.float64,
    )
    features[:, 0] = 1.0
    for coordinate in range(6):
        features[2 + coordinate, 1 + coordinate] = response
        features[8 + coordinate, 1 + coordinate] = -response
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    return features


def _generation_batch(
    config: dict,
    output_root: Path,
) -> ConstructionGenerationBatch:
    plan = build_impulse_triage_plan(config)
    generation_records = []
    checkpoints = []
    steps = []
    for index, probe in enumerate(plan):
        video = output_root / "videos" / f"{index:02d}_{probe.probe_id}.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(f"video-{index}".encode("utf-8"))
        generation_records.append(
            {
                "impulse_probe_id": probe.probe_id,
                "impulse_probe_plan_index": index,
                "impulse_probe_role": probe.probe_role,
                "impulse_stage_index": probe.stage_index,
                "impulse_state_channel_index": probe.channel_index,
                "impulse_polarity": probe.polarity,
                "generation_status": "success",
                "generation_generator_state_digest_random": "b" * 64,
                "generation_model_id": (
                    config["execution_identity"]["generation_model_id"]
                ),
                "generation_model_revision": (
                    config["execution_identity"][
                        "generation_model_revision"
                    ]
                ),
                "scheduler_signature": config["execution_identity"][
                    "scheduler_signature"
                ],
                "prompt_id": config["execution_identity"]["prompt_id"],
                "positive_prompt_text_sha256": config[
                    "execution_identity"
                ]["positive_prompt_text_sha256"],
                "negative_prompt_text_sha256": config[
                    "execution_identity"
                ]["negative_prompt_text_sha256"],
                "seed_id": config["execution_identity"]["seed_id"],
                "generation_seed_random": config["execution_identity"][
                    "seed_value"
                ],
                "video_path": str(video),
                "video_sha256": sha256(video.read_bytes()).hexdigest(),
                "trajectory_step_count": 8,
                "endpoint_control_enabled": False,
            }
        )
        for step_index in range(8):
            steps.append(
                {
                    "impulse_probe_id": probe.probe_id,
                    "impulse_flow_step_index": step_index,
                    "impulse_actual_delta_norm": 0.0,
                    "impulse_inactive_exact_noop": (
                        probe.probe_role == "clean_runtime_repeat"
                    ),
                }
            )
        checkpoints.extend(
            [
                _checkpoint_record(
                    probe_id=probe.probe_id,
                    plan_index=index,
                    checkpoint_id="T_latent",
                    values=np.arange(7, dtype=np.float64),
                    source_path="final_latent",
                    source_status="ready_actual_final_latent",
                ),
                _checkpoint_record(
                    probe_id=probe.probe_id,
                    plan_index=index,
                    checkpoint_id="T_decoded",
                    values=np.zeros(48, dtype=np.float64),
                    source_path=str(video),
                    source_status="ready_pre_save_rgb_float32",
                ),
                _checkpoint_record(
                    probe_id=probe.probe_id,
                    plan_index=index,
                    checkpoint_id="T_saved_video",
                    values=np.zeros(48, dtype=np.float64),
                    source_path=str(video),
                    source_status="ready_rgb24_exact_shape_readback",
                ),
            ]
        )
    return ConstructionGenerationBatch(
        generation_records=tuple(generation_records),
        trajectory_step_records=tuple(steps),
        exposure_traces=_actual_traces(config),
        checkpoint_records=tuple(checkpoints),
    )


def _feature_batch(
    config: dict,
    *,
    response: float = 0.02,
) -> ConstructionFeatureBatch:
    plan = build_impulse_triage_plan(config)
    values = _feature_matrix(response)
    checkpoints = []
    records = []
    schema = config["construction_feature_schema"]["feature_schema_digest"]
    for index, probe in enumerate(plan):
        reencoded = np.zeros(256, dtype=np.float64)
        checkpoints.extend(
            [
                _checkpoint_record(
                    probe_id=probe.probe_id,
                    plan_index=index,
                    checkpoint_id="T_reencoded",
                    values=reencoded,
                    source_path=f"{probe.probe_id}.mp4",
                    source_status="ready_streaming_vae_reencode",
                ),
                _checkpoint_record(
                    probe_id=probe.probe_id,
                    plan_index=index,
                    checkpoint_id="T_output_feature",
                    values=values[index],
                    source_path=f"{probe.probe_id}.mp4",
                    source_status=(
                        "ready_governed_per_video_feature_record"
                    ),
                ),
            ]
        )
        record = {
            "impulse_probe_id": probe.probe_id,
            "impulse_probe_plan_index": index,
            "construction_feature_values": values[index].tolist(),
            "construction_feature_schema_digest": schema,
            "construction_feature_row_binding_digest": (
                construction_feature_row_binding_digest(
                    probe_id=probe.probe_id,
                    feature_schema_digest=schema,
                    feature_values=values[index],
                )
            ),
            "construction_feature_row_identity_binding_status": "ready",
        }
        record["construction_feature_record_id"] = sha256(
            json.dumps(
                record,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        records.append(record)
    return ConstructionFeatureBatch(
        checkpoint_records=tuple(checkpoints),
        feature_records=tuple(records),
    )


def _fake_basis(_master_key: str) -> ConstructionStageBasis:
    return ConstructionStageBasis(
        values=np.zeros((1, 6), dtype=np.float32),
        basis_digest="a" * 64,
        latent_layout_shape=CONSTRUCTION_LATENT_LAYOUT_SHAPE,
        wrong_key_candidate_index=None,
    )


@pytest.mark.quick
def test_numpy_fp32_projection_backoff_and_inactive_noop() -> None:
    rng = np.random.default_rng(3)
    base = rng.standard_normal(100_000).astype(np.float32)
    direction = rng.standard_normal(100_000).astype(np.float32)
    direction /= np.linalg.norm(direction.astype(np.float64))
    budget = float(np.linalg.norm(base.astype(np.float64))) * 0.02 * 0.12
    delta_sigma = CONSTRUCTION_DELTA_SIGMA[3]
    result = apply_numpy_float32_impulse(
        base,
        direction,
        signed_delta_norm=budget,
        delta_sigma=delta_sigma,
        norm_budget=budget,
        remaining_energy=delta_sigma**2 * budget**2,
        minimum_direction_cosine=0.999,
    )
    assert result.selection is not None
    assert result.selection.status == "bounded_actual_delta_backoff_pass"
    selected = result.selection.evaluation
    assert 0.0 < selected.actual_delta_norm <= budget
    assert selected.energy_increment <= delta_sigma**2 * budget**2
    assert selected.direction_cosine >= 0.999
    assert selected.scale < 1.0

    inactive = apply_numpy_float32_impulse(
        base,
        direction,
        signed_delta_norm=0.0,
        delta_sigma=delta_sigma,
        norm_budget=0.0,
        remaining_energy=0.0,
        minimum_direction_cosine=0.999,
    )
    assert inactive.inactive_noop is True
    assert inactive.constrained is base


@pytest.mark.quick
def test_numpy_projection_does_not_backoff_exact_representable_control() -> None:
    base = np.ones(32, dtype=np.float32)
    direction = np.zeros(32, dtype=np.float32)
    direction[0] = 1.0
    result = apply_numpy_float32_impulse(
        base,
        direction,
        signed_delta_norm=0.5,
        delta_sigma=-0.1,
        norm_budget=0.5,
        remaining_energy=0.01,
        minimum_direction_cosine=0.999,
    )
    assert result.selection is not None
    assert result.selection.status == "direct_actual_delta_pass"
    assert result.selection.evaluation.scale == 1.0


@pytest.mark.quick
def test_real_canonical_float32_basis_uses_one_stable_direction_definition() -> None:
    basis = build_construction_stage_basis(
        "canonical-runtime-regression-key-0"
    )
    raw_column = basis.values[:, 1]
    float32_reduction_norm = float(
        np.sqrt(
            np.sum(raw_column * raw_column, dtype=np.float32),
            dtype=np.float32,
        )
    )
    assert float32_reduction_norm != 1.0

    effective = effective_construction_basis_directions(basis)
    actual = np.asarray(effective[:, 1] * np.float32(0.125), dtype=np.float32)
    norm, coordinates, cosine = measure_numpy_actual_delta_coordinates(
        actual,
        basis,
        target_coordinate=1,
        desired_velocity_sign=1,
    )
    assert 0.0 < cosine <= 1.0
    assert cosine == pytest.approx(1.0, abs=1e-12)
    assert coordinates[1] / norm == pytest.approx(cosine, abs=1e-12)
    assert max(abs(value) for index, value in enumerate(coordinates) if index != 1) < 1e-6


@pytest.mark.quick
def test_active_waveform_zero_control_is_never_reclassified_as_inactive() -> None:
    assert _require_active_control_feasible(
        waveform=0.0,
        intended_delta_norm=0.0,
        remaining_energy=0.0,
        base_velocity_norm=0.0,
        step_index=4,
    ) is False
    with pytest.raises(RuntimeError, match="active impulse waveform"):
        _require_active_control_feasible(
            waveform=1.0,
            intended_delta_norm=0.0,
            remaining_energy=1.0,
            base_velocity_norm=0.0,
            step_index=0,
        )
    with pytest.raises(RuntimeError, match="remaining_energy=0.0"):
        _require_active_control_feasible(
            waveform=1.0,
            intended_delta_norm=0.0,
            remaining_energy=0.0,
            base_velocity_norm=100.0,
            step_index=1,
        )


@pytest.mark.quick
def test_actual_traces_cover_schedule_budgets_direction_and_symmetry() -> None:
    config = _config()
    traces = _actual_traces(config)
    assert len(traces) == 12
    for trace in traces:
        assert trace.step_indices == tuple(range(8))
        assert trace.flow_phase_by_step == CONSTRUCTION_FLOW_PHASES
        assert trace.delta_sigma_by_step == CONSTRUCTION_DELTA_SIGMA
        assert all(trace.norm_guard_passed_by_step)
        assert all(trace.energy_guard_passed_by_step)
        assert min(trace.direction_cosine_by_step) >= 0.999
    for coordinate in range(6):
        positive = traces[coordinate]
        negative = traces[6 + coordinate]
        assert np.allclose(
            positive.actual_signed_exposure_by_step,
            -np.asarray(negative.actual_signed_exposure_by_step),
        )


@pytest.mark.quick
def test_numpy_eight_step_actual_delta_preserves_state_update_polarity() -> None:
    rng = np.random.default_rng(17)
    direction = rng.standard_normal(4096).astype(np.float32)
    direction /= np.linalg.norm(direction.astype(np.float64))
    positive_exposures = []
    negative_exposures = []
    for polarity, destination in (
        (1, positive_exposures),
        (-1, negative_exposures),
    ):
        cumulative_reference = 0.0
        cumulative_control = 0.0
        for step_index, interval in enumerate(CONSTRUCTION_DELTA_SIGMA):
            base = rng.standard_normal(4096).astype(np.float32)
            base_norm = float(np.linalg.norm(base.astype(np.float64)))
            control = compute_intended_impulse_control(
                probe_state_update_polarity=polarity,
                temporal_waveform=CONSTRUCTION_TEMPORAL_WAVEFORMS[0][
                    step_index
                ],
                delta_sigma=interval,
                base_velocity_norm=base_norm,
                cumulative_control_energy=cumulative_control,
                cumulative_reference_energy=cumulative_reference,
                remaining_step_count=8 - step_index,
            )
            applied = apply_numpy_float32_impulse(
                base,
                direction,
                signed_delta_norm=control.signed_velocity_coordinate,
                delta_sigma=interval,
                norm_budget=control.intended_delta_norm,
                remaining_energy=control.remaining_control_energy,
                minimum_direction_cosine=0.999,
            )
            if control.intended_delta_norm == 0.0:
                assert applied.inactive_noop is True
                destination.append(0.0)
            else:
                selected = applied.selection.evaluation
                actual = (
                    applied.constrained.astype(np.float32)
                    - base.astype(np.float32)
                )
                coordinate = float(
                    actual.astype(np.float64)
                    @ direction.astype(np.float64)
                )
                destination.append(interval * coordinate)
                assert selected.actual_delta_norm <= (
                    control.intended_delta_norm
                )
                assert selected.energy_increment <= (
                    control.remaining_control_energy
                )
                assert selected.direction_cosine >= 0.999
                cumulative_control += selected.energy_increment
            cumulative_reference += control.reference_energy_increment
    assert all(value > 0.0 for value in positive_exposures[:4])
    assert all(value < 0.0 for value in negative_exposures[:4])
    assert positive_exposures[4:] == [0.0] * 4
    assert negative_exposures[4:] == [0.0] * 4


@pytest.mark.quick
def test_saved_video_summary_requires_real_rgb24_boundary(
    tmp_path: Path,
) -> None:
    path = tmp_path / "video.mp4"
    path.write_bytes(b"not-decoded-by-test-double")
    frame = np.zeros((320, 512, 3), dtype=np.uint8)
    summary = read_saved_video_rgb24_summary(
        path,
        frame_iterator=lambda _path: (frame for _ in range(33)),
    )
    assert summary.shape == (48,)
    with pytest.raises(ValueError, match="RGB24"):
        read_saved_video_rgb24_summary(
            path,
            frame_iterator=lambda _path: (
                frame.astype(np.float32) for _ in range(33)
            ),
        )


@pytest.mark.quick
def test_output_feature_adapter_builds_per_video_row_binding() -> None:
    config = _config()
    latent = np.zeros(
        CONSTRUCTION_LATENT_LAYOUT_SHAPE,
        dtype=np.float32,
    )
    latent[:, 0] = 1.0
    reencoded, output, feature = build_output_feature_records(
        config,
        probe_id="clean_a",
        plan_index=0,
        video_path="clean_a.mp4",
        video_sha256="c" * 64,
        normalized_latent=latent,
        encoder_metadata={"endpoint_vae_encode_status": "ready"},
    )
    assert reencoded["impulse_transfer_checkpoint_dimension"] == 256
    assert output["impulse_transfer_checkpoint_dimension"] == 256
    assert len(feature["construction_feature_values"]) == 256
    assert (
        feature["construction_feature_row_identity_binding_status"]
        == "ready"
    )
    schema = config["construction_feature_schema"]
    memory = schema["streaming_memory_config"]
    metadata = {
        "endpoint_vae_encode_strategy": (
            "cpu_resident_spatiotemporal_streaming"
        ),
        "endpoint_vae_temporal_chunk_frame_count": 4,
        "endpoint_vae_tile_sample_height": 128,
        "endpoint_vae_tile_sample_width": 128,
        "endpoint_vae_tile_sample_stride_height": 96,
        "endpoint_vae_tile_sample_stride_width": 96,
        "endpoint_vae_maximum_incremental_cuda_peak_gib": memory[
            "maximum_incremental_cuda_peak_gib"
        ],
        "endpoint_vae_minimum_cuda_free_gib": memory[
            "minimum_cuda_free_gib"
        ],
        "endpoint_video_frame_count": 33,
        "endpoint_vae_model_class": "AutoencoderKLWan",
        "endpoint_vae_encode_status": "ready",
        "endpoint_latent_shape": [1, 16, 9, 40, 64],
    }
    validate_output_vae_metadata(config, metadata)
    with pytest.raises(RuntimeError, match="metadata"):
        validate_output_vae_metadata(
            config,
            {**metadata, "endpoint_video_frame_count": 32},
        )


@pytest.mark.quick
def test_checkpoint_validator_rejects_missing_record_even_with_ready_bool() -> None:
    config = _config()
    plan = build_impulse_triage_plan(config)
    generation = _generation_batch(config, Path("/tmp/fake"))
    feature = _feature_batch(config)
    checkpoint_map = {
        (
            row["impulse_probe_id"],
            row["impulse_transfer_checkpoint_id"],
        ): row
        for row in (
            *generation.checkpoint_records,
            *feature.checkpoint_records,
        )
    }
    ordered = [
        checkpoint_map[(probe.probe_id, checkpoint_id)]
        for probe in plan
        for checkpoint_id in PRIMARY_CHECKPOINT_IDS
    ]
    validated, ready = validate_checkpoint_and_feature_records(
        config,
        plan,
        ordered,
        feature.feature_records,
    )
    assert validated.values.shape == (14, 256)
    assert all(ready.values())
    forged = list(ordered[:-1])
    forged[0] = {**forged[0], "all_checkpoints_ready": True}
    with pytest.raises(ValueError, match="精确覆盖"):
        validate_checkpoint_and_feature_records(
            config,
            plan,
            forged,
            feature.feature_records,
        )


@pytest.mark.quick
def test_gate_a_runner_pass_and_fail_remain_nonformal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    _write_source_root(source)
    monkeypatch.setenv(
        "SSTW_TRAJECTORY_AUTHENTICATION_KEY",
        "owner-key-material-for-construction-tests",
    )

    def generation_executor(config: dict, **kwargs: object) -> ConstructionGenerationBatch:
        return _generation_batch(config, Path(kwargs["output_root"]))

    def passing_features(config: dict, **_kwargs: object) -> ConstructionFeatureBatch:
        return _feature_batch(config, response=0.02)

    passed = run_output_feature_impulse_observability_construction(
        source,
        tmp_path / "pass",
        generation_executor=generation_executor,
        feature_executor=passing_features,
        basis_builder=_fake_basis,
    )
    assert passed["impulse_sample_internal_observability_gate_ready"] is True
    assert passed["cross_identity_confirmation_design_allowed"] is True
    assert passed["formal_result"] is False
    assert passed["stage_progression_allowed"] is False
    assert passed["gate_b_execution_allowed"] is False

    def tiny_features(config: dict, **_kwargs: object) -> ConstructionFeatureBatch:
        return _feature_batch(config, response=1e-8)

    failed = run_output_feature_impulse_observability_construction(
        source,
        tmp_path / "fail",
        generation_executor=generation_executor,
        feature_executor=tiny_features,
        basis_builder=_fake_basis,
    )
    assert failed["impulse_sample_internal_observability_gate_ready"] is False
    assert failed["cross_identity_confirmation_design_allowed"] is False
    assert failed["impulse_observability_construction_decision"].endswith(
        "_failed_stop"
    )


@pytest.mark.quick
def test_runner_runtime_failure_writes_recovery_only_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    _write_source_root(source)
    monkeypatch.setenv(
        "SSTW_TRAJECTORY_AUTHENTICATION_KEY",
        "owner-key-material-for-construction-tests",
    )

    def failed_generation(
        _config: dict,
        **_kwargs: object,
    ) -> ConstructionGenerationBatch:
        raise RuntimeError("planned generation failure")

    output = tmp_path / "failure"
    with pytest.raises(RuntimeError, match="planned generation failure"):
        run_output_feature_impulse_observability_construction(
            source,
            output,
            generation_executor=failed_generation,
            basis_builder=_fake_basis,
        )
    decision = json.loads(
        (
            output
            / "artifacts"
            / "output_feature_impulse_observability_decision.json"
        ).read_text(encoding="utf-8")
    )
    assert decision["formal_result"] is False
    assert decision["stage_progression_allowed"] is False
    assert decision["claim_support_status"] == (
        "failure_recovery_only_not_claim_evidence"
    )


@pytest.mark.quick
def test_prompt_source_rejects_same_id_with_modified_positive_text(
    tmp_path: Path,
) -> None:
    config = _config()
    source = tmp_path / "source"
    _write_source_root(source)
    _validate_prompt_seed_source(source, config)
    path = source / "datasets" / "prompt_seed_suite.json"
    suite = json.loads(path.read_text(encoding="utf-8"))
    suite["prompts"][0]["prompt_text"] += " altered"
    path.write_text(
        json.dumps(suite, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="execution identity"):
        _validate_prompt_seed_source(source, config)


@pytest.mark.quick
def test_generation_validation_rejects_rng_or_filename_identity_drift(
    tmp_path: Path,
) -> None:
    config = _config()
    plan = build_impulse_triage_plan(config)
    batch = _generation_batch(config, tmp_path)
    _validate_generation_batch(config, plan, batch)
    records = [dict(row) for row in batch.generation_records]
    records[1]["generation_generator_state_digest_random"] = "d" * 64
    with pytest.raises(RuntimeError, match="RNG"):
        _validate_generation_batch(
            config,
            plan,
            ConstructionGenerationBatch(
                generation_records=tuple(records),
                trajectory_step_records=batch.trajectory_step_records,
                exposure_traces=batch.exposure_traces,
                checkpoint_records=batch.checkpoint_records,
            ),
        )
    for field, wrong_value in (
        ("prompt_id", "wrong_prompt_id"),
        ("positive_prompt_text_sha256", "0" * 64),
        ("seed_id", "wrong_seed_id"),
        ("generation_seed_random", 2202),
    ):
        mutated = [dict(row) for row in batch.generation_records]
        mutated[0][field] = wrong_value
        with pytest.raises(RuntimeError, match="identity"):
            _validate_generation_batch(
                config,
                plan,
                ConstructionGenerationBatch(
                    generation_records=tuple(mutated),
                    trajectory_step_records=batch.trajectory_step_records,
                    exposure_traces=batch.exposure_traces,
                    checkpoint_records=batch.checkpoint_records,
                ),
            )
    duplicate_paths = [dict(row) for row in batch.generation_records]
    duplicate_paths[1]["video_path"] = duplicate_paths[0]["video_path"]
    duplicate_paths[1]["video_sha256"] = duplicate_paths[0]["video_sha256"]
    with pytest.raises(RuntimeError, match="coverage"):
        _validate_generation_batch(
            config,
            plan,
            ConstructionGenerationBatch(
                generation_records=tuple(duplicate_paths),
                trajectory_step_records=batch.trajectory_step_records,
                exposure_traces=batch.exposure_traces,
                checkpoint_records=batch.checkpoint_records,
            ),
        )


def _request_payload(source_zip: Path) -> dict:
    return {
        "request_schema_version": REQUEST_SCHEMA_VERSION,
        "test_id": (
            OUTPUT_FEATURE_IMPULSE_OBSERVABILITY_CONSTRUCTION_TEST_ID
        ),
        "repository": {
            "url": "https://github.com/RICHAAARC/SSTW.git",
            "ref": "d567af1bd3e1b822c55ea90e255846eb7b6b0b35",
        },
        "parameters": {
            "phase": "gate_a",
            "run_series_id": "output_feature_impulse_gate_a",
            "source_package_path": str(source_zip),
            "resume_package_path": "",
        },
    }


@pytest.mark.quick
def test_colab_request_allowlist_dry_run_and_single_zip_packaging(
    tmp_path: Path,
) -> None:
    project = tmp_path / "SSTW"
    source_zip = (
        project
        / "inputs"
        / "output_feature_impulse_observability"
        / "prompt_seed_source.zip"
    )
    request = project / "requests" / "colab_test_request.json"
    _write_source_zip(source_zip)
    request.parent.mkdir(parents=True)
    request.write_text(
        json.dumps(_request_payload(source_zip), indent=2) + "\n",
        encoding="utf-8",
    )
    resolved = load_colab_test_request(request, project_root=project)
    assert resolved["phase"] == "gate_a"
    dry_run = build_colab_test_dry_run_plan(
        request,
        project_root=project,
    )
    assert dry_run["test_id"] == (
        OUTPUT_FEATURE_IMPULSE_OBSERVABILITY_CONSTRUCTION_TEST_ID
    )

    def fake_runner(source_root: Path, output_root: Path) -> dict:
        assert list(source_root.rglob("prompt_seed_suite.json"))
        artifact = output_root / "artifacts" / "decision.json"
        artifact.parent.mkdir(parents=True)
        artifact.write_text('{"formal_result":false}\n', encoding="utf-8")
        return {
            "impulse_sample_internal_observability_gate_ready": False,
            "formal_result": False,
            "stage_progression_allowed": False,
        }

    result = run_colab_test_request(
        request,
        project_root=project,
        repo_root=Path.cwd(),
        local_workspace_root=tmp_path / "content" / "workspace",
        local_package_cache_root=tmp_path / "content" / "packages",
        output_feature_impulse_observability_runner=fake_runner,
    )
    drive_root = Path(result["drive_result_zip"]).parent
    assert Path(result["drive_result_zip"]).is_file()
    assert Path(result["drive_result_manifest"]).is_file()
    assert len(list(drive_root.glob("*.zip"))) == 1
    manifest = json.loads(
        Path(result["drive_result_manifest"]).read_text(encoding="utf-8")
    )
    assert manifest["generation_model_ids"] == [
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    ]
    assert manifest["diagnostic_decision"]["formal_result"] is False


@pytest.mark.quick
def test_server_cli_dry_run_routes_gate_a_without_gpu(
    tmp_path: Path,
) -> None:
    project = tmp_path / "SSTW"
    source_zip = project / "inputs" / "source.zip"
    request = project / "requests" / "colab_test_request.json"
    _write_source_zip(source_zip)
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
    assert decision["pipeline_results"][0]["phase"] == "gate_a"
    assert decision["pipeline_results"][0]["test_id"] == (
        OUTPUT_FEATURE_IMPULSE_OBSERVABILITY_CONSTRUCTION_TEST_ID
    )


@pytest.mark.quick
def test_colab_output_feature_request_rejects_resume(
    tmp_path: Path,
) -> None:
    project = tmp_path / "SSTW"
    source_zip = project / "inputs" / "source.zip"
    request = project / "requests" / "colab_test_request.json"
    _write_source_zip(source_zip)
    payload = _request_payload(source_zip)
    payload["parameters"]["resume_package_path"] = str(source_zip)
    request.parent.mkdir(parents=True)
    request.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="不接受 resume package"):
        load_colab_test_request(request, project_root=project)


@pytest.mark.quick
def test_colab_output_feature_partial_failure_uses_nonformal_recovery(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "content"
    project = tmp_path / "SSTW"
    source_zip = project / "inputs" / "source.zip"
    request = project / "requests" / "colab_test_request.json"
    _write_source_zip(source_zip)
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
        / OUTPUT_FEATURE_IMPULSE_OBSERVABILITY_CONSTRUCTION_TEST_ID
        / "output_feature_impulse_gate_a"
        / "records"
        / "impulse_generation_records.jsonl"
    )
    partial.parent.mkdir(parents=True)
    partial.write_text(
        '{"generation_status":"failed","formal_result":false}\n',
        encoding="utf-8",
    )
    recovered = package_colab_test_recovery_bundle(
        request,
        project_root=project,
        repo_root=Path.cwd(),
        local_runtime_root=runtime_root,
        local_workspace_root=workspace,
        local_package_cache_root=cache,
    )
    assert recovered["formal_result"] is False
    assert recovered["stage_progression_allowed"] is False
    assert recovered["claim_support_status"] == (
        "failure_recovery_only_not_claim_evidence"
    )


@pytest.mark.quick
def test_fixed_notebook_remains_unmodified_server_cli_entrypoint() -> None:
    notebook = Path(
        "paper_workflow/colab_notebooks/colab_test_runner.ipynb"
    )
    assert notebook.is_file()
    source = notebook.read_text(encoding="utf-8")
    assert "scripts/run_generative_video_server_workflow.py" in source
    assert "output_feature_impulse_observability_construction.py" not in source
