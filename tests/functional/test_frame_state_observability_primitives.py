from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from evaluation.protocol.frame_state_signed_observability_contract import (
    DEFAULT_CONFIG_PATH,
)
from main.methods.state_space_watermark.frame_state_observability import (
    FEATURE_DIMENSION,
    FROZEN_SIGMA_GRID,
    FROZEN_WAVEFORM,
    LATENT_SHAPE,
    CheckpointResponses,
    SignedResponseStatistics,
    accumulate_actual_signed_exposure,
    apply_frame_state_control_numpy,
    build_flow_schedule,
    build_public_frame_state_atom,
    build_public_rademacher_initialization,
    clamp_machine_roundoff_cosine,
    compute_signed_response_statistics,
    estimate_construction_t0,
    evaluate_gate_zero,
    extract_local_temporal_feature,
    final_latent_carrier_projection,
    predict_apply_only_odd,
    read_public_frame_state_atom,
    validate_public_frame_state_atom,
    write_public_frame_state_atom,
)


pytestmark = pytest.mark.quick


def _flow_schedule():
    config = json.loads(Path(DEFAULT_CONFIG_PATH).read_text(encoding="utf-8"))
    frozen = config["protocol_contract"]["flow_injection_contract"][
        "flow_schedule_contract"
    ]
    return build_flow_schedule(
        [float(value) for value in frozen["sigma_grid_decimal"]],
        [
            value / 1_000_000
            for value in frozen["waveform_by_step_millionths"]
        ],
    )


def _atom():
    return build_public_frame_state_atom(lambda value: value)


def test_public_rademacher_and_power_iteration_are_deterministic() -> None:
    first = build_public_rademacher_initialization()
    second = build_public_rademacher_initialization()
    calls = []

    def operator(value: np.ndarray) -> np.ndarray:
        calls.append(value.copy())
        return value

    atom = build_public_frame_state_atom(operator)

    assert np.array_equal(first, second)
    assert len(calls) == 8
    assert atom.values.shape == LATENT_SHAPE
    assert atom.values.dtype == np.dtype("<f4")
    assert np.all(atom.values[:, :, (0, 1, 2, 6, 7, 8), :, :] == 0.0)
    assert np.linalg.norm(atom.values.astype(np.float64)) == pytest.approx(
        1.0, abs=20e-6
    )
    assert float(np.linalg.norm(atom.values)) == pytest.approx(
        1.0, abs=20e-6
    )
    assert all(value.dtype == np.dtype("float32") for value in calls)
    assert atom.array_digest == (
        "643961b437301334c3016b6a3fad67b08ddadacae26ca569cb53c6cb99c153e8"
    )


def test_public_atom_npz_round_trip_and_extra_array_rejection(
    tmp_path: Path,
) -> None:
    atom = _atom()
    path = tmp_path / "public_atom.npz"
    write_public_frame_state_atom(path, atom)
    restored = read_public_frame_state_atom(path)

    assert restored.array_digest == atom.array_digest
    assert np.array_equal(restored.values, atom.values)
    with pytest.raises(FileExistsError):
        write_public_frame_state_atom(path, atom)

    forged = tmp_path / "forged.npz"
    np.savez(
        forged,
        frame_state_public_atom=atom.values,
        unexpected=np.zeros(1, dtype=np.float32),
    )
    with pytest.raises(ValueError, match="唯一"):
        read_public_frame_state_atom(forged)


def test_public_atom_rejects_support_sign_and_dtype_mutations() -> None:
    atom = _atom().values
    outside = atom.copy()
    outside[0, 0, 0, 0, 0] = 1.0
    with pytest.raises(ValueError, match="support"):
        validate_public_frame_state_atom(outside)
    with pytest.raises(ValueError, match="dtype"):
        validate_public_frame_state_atom(atom.astype(np.float64))
    with pytest.raises(ValueError, match="sign"):
        validate_public_frame_state_atom(-atom)


def test_power_iteration_callback_rejects_dtype_shape_and_finite_drift() -> None:
    with pytest.raises(ValueError, match="dtype"):
        build_public_frame_state_atom(
            lambda value: value.astype(np.float64)
        )
    with pytest.raises(ValueError, match="shape"):
        build_public_frame_state_atom(lambda value: value.reshape(-1))

    def nonfinite(value: np.ndarray) -> np.ndarray:
        result = value.copy()
        result.reshape(-1)[0] = np.nan
        return result

    with pytest.raises(ValueError, match="非有限"):
        build_public_frame_state_atom(nonfinite)


def test_frozen_eight_step_schedule_has_explicit_nonhistorical_waveform() -> None:
    schedule = _flow_schedule()

    assert len(schedule) == 8
    assert [step.step_index for step in schedule] == list(range(8))
    assert [step.active for step in schedule] == [
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        False,
    ]
    assert [step.waveform for step in schedule] == [
        0.0,
        0.0,
        0.0,
        0.0,
        0.25,
        1.0,
        0.5,
        0.0,
    ]
    assert [step.delta_sigma for step in schedule] == pytest.approx(
        [
            -0.052457451820373535,
            -0.06475478410720825,
            -0.08195042610168457,
            -0.10704421997070312,
            -0.14574754238128662,
            -0.21007341146469116,
            -0.32904359698295593,
            -0.008928571827709675,
        ],
        abs=0.0,
    )


def test_schedule_rejects_implicit_all_one_and_wrong_active_window() -> None:
    sigma = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3, 0.1, 0.0]
    with pytest.raises(ValueError, match="冻结8步"):
        build_flow_schedule(sigma, [1.0] * 8)
    with pytest.raises(ValueError, match="冻结8步"):
        build_flow_schedule(sigma, [0, 0, 0, 1, 1, 1, 0, 0])
    frozen = _flow_schedule()
    mutated_sigma = list(FROZEN_SIGMA_GRID)
    mutated_sigma[5] = 0.5
    with pytest.raises(ValueError, match="冻结8步"):
        build_flow_schedule(mutated_sigma, FROZEN_WAVEFORM)
    mutated_waveform = list(FROZEN_WAVEFORM)
    mutated_waveform[5] = 0.99
    with pytest.raises(ValueError, match="冻结8步"):
        build_flow_schedule(FROZEN_SIGMA_GRID, mutated_waveform)
    assert frozen == build_flow_schedule()


def test_numpy_control_preserves_state_update_polarity_and_strict_budgets() -> None:
    atom = _atom().values
    base = np.ones(LATENT_SHAPE, dtype=np.float32)
    step = _flow_schedule()[5]
    positive = apply_frame_state_control_numpy(
        base,
        atom,
        signed_state_coefficient=1,
        step=step,
        cumulative_control_energy=0.0,
        cumulative_reference_energy=0.0,
        remaining_step_count=3,
    )
    negative = apply_frame_state_control_numpy(
        base,
        atom,
        signed_state_coefficient=-1,
        step=step,
        cumulative_control_energy=0.0,
        cumulative_reference_energy=0.0,
        remaining_step_count=3,
    )

    assert positive.signed_state_update_exposure > 0.0
    assert negative.signed_state_update_exposure < 0.0
    assert positive.actual_delta_norm <= positive.joint_norm_budget
    assert positive.energy_increment <= positive.remaining_energy
    assert positive.direction_cosine is not None
    assert 0.999 <= positive.direction_cosine <= 1.0
    assert not positive.inactive_exact_noop


def test_cosine_clamp_only_accepts_machine_roundoff_boundary() -> None:
    assert clamp_machine_roundoff_cosine(1.0 + 5e-13) == 1.0
    assert clamp_machine_roundoff_cosine(-1.0 - 5e-13) == -1.0
    with pytest.raises(ValueError, match="Cauchy"):
        clamp_machine_roundoff_cosine(1.0 + 2e-12)
    with pytest.raises(ValueError, match="Cauchy"):
        clamp_machine_roundoff_cosine(float("nan"))


def test_ordered_eight_step_actual_exposure_is_signed_and_complete() -> None:
    atom = _atom().values
    base = np.ones(LATENT_SHAPE, dtype=np.float32)
    results = []
    control_energy = 0.0
    reference_energy = 0.0
    base_norm = float(np.linalg.norm(base.astype(np.float64)))
    schedule = _flow_schedule()
    for step in schedule:
        result = apply_frame_state_control_numpy(
            base,
            atom,
            signed_state_coefficient=1,
            step=step,
            cumulative_control_energy=control_energy,
            cumulative_reference_energy=reference_energy,
            remaining_step_count=8 - step.step_index,
        )
        results.append(result)
        control_energy += result.energy_increment
        reference_energy += step.delta_sigma**2 * base_norm**2

    assert accumulate_actual_signed_exposure(results, atom) > 0.0
    assert sum(not result.inactive_exact_noop for result in results) == 3
    with pytest.raises(ValueError, match="有序8步"):
        accumulate_actual_signed_exposure(tuple(reversed(results)), atom)

    with pytest.raises(ValueError, match="自报值"):
        accumulate_actual_signed_exposure(
            [
                replace(result, signed_state_update_exposure=123.0)
                for result in results
            ],
            atom,
        )
    with pytest.raises(ValueError, match="schedule binding"):
        accumulate_actual_signed_exposure(
            [
                replace(results[0], delta_sigma=-0.5),
                *results[1:],
            ],
            atom,
        )
    with pytest.raises(ValueError, match="schedule binding"):
        accumulate_actual_signed_exposure(
            [
                replace(results[0], waveform=0.5),
                *results[1:],
            ],
            atom,
        )
    forged_delta = results[0].actual_delta_velocity.copy()
    forged_delta.reshape(-1)[0] = np.float32(1.0)
    with pytest.raises(ValueError, match="inactive step"):
        accumulate_actual_signed_exposure(
            [
                replace(results[0], actual_delta_velocity=forged_delta),
                *results[1:],
            ],
            atom,
        )
    with pytest.raises(ValueError, match="actual delta velocity"):
        accumulate_actual_signed_exposure(
            [
                replace(
                    results[0],
                    actual_delta_velocity=results[
                        0
                    ].actual_delta_velocity.astype(np.float64),
                ),
                *results[1:],
            ],
            atom,
        )
    nonfinite_delta = results[0].actual_delta_velocity.copy()
    nonfinite_delta.reshape(-1)[0] = np.nan
    with pytest.raises(ValueError, match="actual delta velocity"):
        accumulate_actual_signed_exposure(
            [
                replace(
                    results[0],
                    actual_delta_velocity=nonfinite_delta,
                ),
                *results[1:],
            ],
            atom,
        )


def test_clean_and_inactive_use_exact_noop_but_active_cannot_collapse() -> None:
    atom = _atom().values
    base = np.ones(LATENT_SHAPE, dtype=np.float32)
    active = _flow_schedule()[4]
    inactive = _flow_schedule()[0]
    clean = apply_frame_state_control_numpy(
        base,
        atom,
        signed_state_coefficient=0,
        step=active,
        cumulative_control_energy=0.0,
        cumulative_reference_energy=0.0,
        remaining_step_count=4,
    )
    outside = apply_frame_state_control_numpy(
        base,
        atom,
        signed_state_coefficient=1,
        step=inactive,
        cumulative_control_energy=0.0,
        cumulative_reference_energy=0.0,
        remaining_step_count=8,
    )

    assert clean.constrained_velocity is base
    assert outside.constrained_velocity is base
    assert clean.inactive_exact_noop and outside.inactive_exact_noop
    with pytest.raises(RuntimeError, match="norm/energy"):
        apply_frame_state_control_numpy(
            np.zeros(LATENT_SHAPE, dtype=np.float32),
            atom,
            signed_state_coefficient=1,
            step=active,
            cumulative_control_energy=0.0,
            cumulative_reference_energy=0.0,
            remaining_step_count=4,
        )


@pytest.mark.parametrize(
    "bad_base",
    [
        np.ones(LATENT_SHAPE, dtype=np.float64),
        np.full(LATENT_SHAPE, np.nan, dtype=np.float32),
    ],
)
@pytest.mark.parametrize(
    ("coefficient", "step_index", "remaining"),
    [(0, 0, 8), (1, 5, 3)],
)
def test_control_rejects_base_velocity_dtype_or_finite_drift(
    bad_base: np.ndarray,
    coefficient: int,
    step_index: int,
    remaining: int,
) -> None:
    with pytest.raises(ValueError, match="base velocity"):
        apply_frame_state_control_numpy(
            bad_base,
            _atom().values,
            signed_state_coefficient=coefficient,
            step=_flow_schedule()[step_index],
            cumulative_control_energy=0.0,
            cumulative_reference_energy=0.0,
            remaining_step_count=remaining,
        )


def test_step_level_control_rejects_forged_schedule_and_remaining_count() -> None:
    atom = _atom().values
    base = np.ones(LATENT_SHAPE, dtype=np.float32)
    step = _flow_schedule()[5]
    forged = replace(step, waveform=0.99)
    with pytest.raises(ValueError, match="冻结8步"):
        apply_frame_state_control_numpy(
            base,
            atom,
            signed_state_coefficient=1,
            step=forged,
            cumulative_control_energy=0.0,
            cumulative_reference_energy=0.0,
            remaining_step_count=3,
        )
    with pytest.raises(ValueError, match="remaining count"):
        apply_frame_state_control_numpy(
            base,
            atom,
            signed_state_coefficient=1,
            step=step,
            cumulative_control_energy=0.0,
            cumulative_reference_energy=0.0,
            remaining_step_count=8,
        )


def test_local_features_and_latent_projection_have_exact_representations() -> None:
    rgb = np.zeros((33, 320, 512, 3), dtype=np.uint8)
    rgb[11:22, :, :, 0] = 255
    saved = extract_local_temporal_feature(rgb, rgb24=True)
    decoded = extract_local_temporal_feature(
        rgb.astype(np.float32) / 255.0,
        rgb24=False,
    )
    atom = _atom().values

    assert saved.shape == (FEATURE_DIMENSION,)
    assert np.array_equal(saved, decoded)
    assert saved[:3].tolist() == [1.0, 0.0, 0.0]
    projection = final_latent_carrier_projection(atom, atom)
    assert projection.shape == (1,)
    assert projection[0] == pytest.approx(1.0, abs=20e-6)
    with pytest.raises(ValueError, match="33x320x512"):
        extract_local_temporal_feature(
            np.zeros((33, 8, 8, 3), dtype=np.uint8),
            rgb24=True,
        )
    with pytest.raises(ValueError, match="final latent"):
        final_latent_carrier_projection(atom.astype(np.float64), atom)
    nonfinite_latent = atom.copy()
    nonfinite_latent.reshape(-1)[0] = np.nan
    with pytest.raises(ValueError, match="final latent"):
        final_latent_carrier_projection(nonfinite_latent, atom)


def _responses(
    odd: np.ndarray,
    *,
    common: np.ndarray | None = None,
) -> CheckpointResponses:
    common_value = np.zeros_like(odd) if common is None else common
    clean_a = np.zeros_like(odd)
    clean_b = np.zeros_like(odd)
    return CheckpointResponses(
        clean_a=clean_a,
        clean_b=clean_b,
        positive=odd + common_value,
        negative=-odd + common_value,
    )


def test_c0_t0_and_identity_a_apply_only_prediction() -> None:
    odd_c0 = np.linspace(1.0, 2.0, FEATURE_DIMENSION)
    c0 = _responses(odd_c0)
    t0 = estimate_construction_t0(
        c0,
        positive_actual_exposure=2.0,
        negative_actual_exposure=-2.0,
    )
    prediction = predict_apply_only_odd(
        t0,
        positive_actual_exposure=1.0,
        negative_actual_exposure=-1.0,
    )

    assert np.allclose(t0, odd_c0 / 2.0)
    assert np.allclose(prediction, odd_c0 / 2.0)
    with pytest.raises(ValueError, match="exposure difference"):
        estimate_construction_t0(
            c0,
            positive_actual_exposure=1.0,
            negative_actual_exposure=1.0,
        )


def test_gate_zero_requires_all_checkpoint_and_primary_transfer_gates() -> None:
    primary_odd = np.linspace(1.0, 2.0, FEATURE_DIMENSION)
    checkpoints = {
        "final_latent_carrier_projection": _responses(np.ones(1)),
        "decoded_local_temporal_feature": _responses(
            np.ones(FEATURE_DIMENSION)
        ),
        "saved_video_local_temporal_feature": _responses(primary_odd),
    }
    passed = evaluate_gate_zero(
        checkpoints,
        predicted_primary_odd=primary_odd,
    )

    assert passed.gate_zero_ready
    assert passed.formal_result is False
    assert passed.stage_progression_allowed is False
    failed_checkpoints = dict(checkpoints)
    failed_checkpoints["decoded_local_temporal_feature"] = _responses(
        np.ones(FEATURE_DIMENSION),
        common=np.ones(FEATURE_DIMENSION),
    )
    with pytest.raises(ValueError, match="派生统计"):
        evaluate_gate_zero(
            failed_checkpoints,
            predicted_primary_odd=primary_odd,
        )
    assert not evaluate_gate_zero(
        checkpoints,
        predicted_primary_odd=-primary_odd,
    ).gate_zero_ready


def test_gate_zero_rejects_missing_checkpoint_and_wrong_prediction_shape() -> None:
    checkpoints = {
        "final_latent_carrier_projection": _responses(np.ones(1)),
        "decoded_local_temporal_feature": _responses(
            np.ones(FEATURE_DIMENSION)
        ),
    }
    with pytest.raises(ValueError, match="coverage"):
        evaluate_gate_zero(
            checkpoints,
            predicted_primary_odd=np.ones(FEATURE_DIMENSION),
        )
    checkpoints["saved_video_local_temporal_feature"] = _responses(
        np.ones(FEATURE_DIMENSION)
    )
    with pytest.raises(ValueError, match="shape"):
        evaluate_gate_zero(
            checkpoints,
            predicted_primary_odd=np.ones(1),
        )


def test_gate_zero_recomputes_raw_statistics_and_rejects_forged_scalars() -> None:
    forged = SignedResponseStatistics(
        clean_intercept=np.zeros(FEATURE_DIMENSION),
        observed_odd=np.ones(FEATURE_DIMENSION),
        observed_common=np.full(FEATURE_DIMENSION, 999.0),
        clean_noise_norm=float("nan"),
        odd_norm=float("nan"),
        common_norm=float("nan"),
        antisymmetry_cosine=1.0,
        antisymmetry_residual=0.0,
        common_odd_ratio=0.0,
        odd_clean_noise_ratio=999.0,
    )
    forged_checkpoints = {
        "final_latent_carrier_projection": forged,
        "decoded_local_temporal_feature": forged,
        "saved_video_local_temporal_feature": forged,
    }
    with pytest.raises(TypeError, match="raw CheckpointResponses"):
        evaluate_gate_zero(
            forged_checkpoints,
            predicted_primary_odd=np.ones(FEATURE_DIMENSION),
        )

    raw = {
        "final_latent_carrier_projection": _responses(np.ones(1)),
        "decoded_local_temporal_feature": _responses(
            np.ones(FEATURE_DIMENSION)
        ),
        "saved_video_local_temporal_feature": CheckpointResponses(
            clean_a=np.zeros(FEATURE_DIMENSION),
            clean_b=np.zeros(FEATURE_DIMENSION),
            positive=np.full(FEATURE_DIMENSION, np.nan),
            negative=-np.ones(FEATURE_DIMENSION),
        ),
    }
    with pytest.raises(ValueError, match="有限"):
        evaluate_gate_zero(
            raw,
            predicted_primary_odd=np.ones(FEATURE_DIMENSION),
        )


def test_signed_statistics_clean_floor_is_not_caller_configurable() -> None:
    responses = _responses(np.full(FEATURE_DIMENSION, 2e-6))
    statistics = compute_signed_response_statistics(responses)
    assert statistics.odd_clean_noise_ratio == pytest.approx(
        np.sqrt(FEATURE_DIMENSION) * 2.0
    )
    with pytest.raises(TypeError):
        compute_signed_response_statistics(
            responses,
            clean_noise_floor=1e-30,
        )
