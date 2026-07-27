from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from evaluation.protocol.patch_relation_gate0_contract import (
    DEFAULT_CONFIG_PATH,
    EXPECTED_ACTUAL_SIGNED_EXPOSURE_INPUT,
    EXPECTED_STATISTICS_AND_TRANSFER_FORMULA_CONTRACT,
    FROZEN_PROTOCOL_DIGEST,
    build_patch_relation_gate0_plan,
    load_patch_relation_gate0_config,
    protocol_digest,
)
from main.methods.state_space_watermark.patch_relation_carrier import (
    FEATURE_SCHEMA_ID,
    FEATURE_SHAPE,
    PATCH_A_TOKEN_COORDINATE,
    PATCH_B_TOKEN_COORDINATE,
    PATCH_SIZE_PIXELS,
    PHASE_BUDGET_RADIANS,
    ROPE_TUPLE_SHAPE,
    TOKEN_GRID_SHAPE,
    VIDEO_SHAPE_RGB24,
    apply_wan_rotary_phase_numpy,
    build_public_patch_relation_descriptor,
    build_relation_phase_delta,
    construct_c0_relation_transfer,
    compute_signed_relation_statistics,
    derive_signed_relation_coefficient,
    evaluate_gate0_apply_only,
    extract_saved_rgb24_patch_relation_feature,
    fit_c0_whitening,
    frozen_method_boundary,
    validate_public_patch_relation_descriptor,
)


pytestmark = pytest.mark.quick


def _feature_pattern(scale: float = 1.0) -> np.ndarray:
    values = np.arange(np.prod(FEATURE_SHAPE), dtype=np.float64).reshape(
        FEATURE_SHAPE
    )
    return np.ascontiguousarray((1.0 + values / 100.0) * scale, dtype="<f8")


def _write_mutation(tmp_path: Path, mutate) -> Path:
    config = json.loads(Path(DEFAULT_CONFIG_PATH).read_text(encoding="utf-8"))
    mutate(config)
    path = tmp_path / "mutation.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _c0_construction():
    descriptor = build_public_patch_relation_descriptor()
    clean = np.zeros(FEATURE_SHAPE, dtype="<f8")
    odd = _feature_pattern(1e-4)
    construction = construct_c0_relation_transfer(
        descriptor=descriptor,
        clean_a=clean,
        clean_b=clean,
        positive=odd,
        negative=-odd,
        positive_exposure=PHASE_BUDGET_RADIANS,
        negative_exposure=-PHASE_BUDGET_RADIANS,
    )
    return descriptor, construction, clean, odd


def test_frozen_config_digest_plan_and_authorization() -> None:
    config = load_patch_relation_gate0_config()
    assert protocol_digest(config["protocol_contract"]) == FROZEN_PROTOCOL_DIGEST
    plan = build_patch_relation_gate0_plan(config)
    assert len(plan) == 8
    assert [row.plan_index for row in plan] == list(range(8))
    assert [row.probe_role for row in plan] == [
        "clean_a",
        "clean_b",
        "positive",
        "negative",
    ] * 2
    assert [row.signed_state_coefficient for row in plan] == [0, 0, 1, -1] * 2
    assert plan[0].identity_placeholder != plan[4].identity_placeholder
    assert all(not row.execution_authorized for row in plan)
    assert all(value is False for value in frozen_method_boundary().values())
    mutated = json.loads(json.dumps(config))
    mutated["protocol_contract"]["rope_phase_contract"][
        "temporal_rope_pair_index"
    ] = 1
    with pytest.raises(ValueError, match="冻结 config"):
        build_patch_relation_gate0_plan(mutated)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["authorization_boundary"].__setitem__(
            "gpu_execution_allowed", True
        ),
        lambda value: value["protocol_contract"]["rope_phase_contract"].__setitem__(
            "phase_budget_radians_decimal", "0.03125"
        ),
        lambda value: value["protocol_contract"]["token_relation_contract"].__setitem__(
            "patch_b_token_coordinate_row_column", [9, 17]
        ),
        lambda value: value["protocol_contract"]["output_feature_contract"].__setitem__(
            "per_video_l2_normalization_allowed", True
        ),
        lambda value: value["protocol_contract"][
            "construction_and_gate_contract"
        ]["signed_gate_thresholds"].__setitem__(
            "minimum_antisymmetry_cosine_decimal", "0.0"
        ),
    ],
)
def test_config_mutations_fail_closed(tmp_path: Path, mutation) -> None:
    with pytest.raises(ValueError):
        load_patch_relation_gate0_config(_write_mutation(tmp_path, mutation))


def test_rehashed_mutation_still_rejected(tmp_path: Path) -> None:
    def mutate(config: dict) -> None:
        config["protocol_contract"]["rope_phase_contract"][
            "temporal_rope_pair_index"
        ] = 1
        config["protocol_digest"] = protocol_digest(config["protocol_contract"])

    with pytest.raises(ValueError, match="RoPE phase contract"):
        load_patch_relation_gate0_config(_write_mutation(tmp_path, mutate))


@pytest.mark.parametrize(
    "formula_id",
    tuple(EXPECTED_STATISTICS_AND_TRANSFER_FORMULA_CONTRACT),
)
def test_each_formula_mutation_is_rejected_after_self_rehash(
    tmp_path: Path,
    formula_id: str,
) -> None:
    def mutate(config: dict) -> None:
        formulas = config["protocol_contract"]["construction_and_gate_contract"][
            "statistics_and_transfer_formula_contract"
        ]
        formulas[formula_id] = f"{formulas[formula_id]}_mutated"
        config["protocol_digest"] = protocol_digest(config["protocol_contract"])

    with pytest.raises(ValueError, match="exact formula contract"):
        load_patch_relation_gate0_config(_write_mutation(tmp_path, mutate))


def test_actual_exposure_evidence_boundary_mutation_is_rejected(
    tmp_path: Path,
) -> None:
    def mutate(config: dict) -> None:
        gate = config["protocol_contract"]["construction_and_gate_contract"]
        assert (
            gate["actual_signed_exposure_input"]
            == EXPECTED_ACTUAL_SIGNED_EXPOSURE_INPUT
        )
        gate["actual_signed_exposure_input"] = (
            "caller_scalar_is_runtime_execution_evidence"
        )
        config["protocol_digest"] = protocol_digest(config["protocol_contract"])

    with pytest.raises(ValueError, match="exposure"):
        load_patch_relation_gate0_config(_write_mutation(tmp_path, mutate))


def test_descriptor_is_exact_zero_sum_c_order_and_key_independent() -> None:
    first = build_public_patch_relation_descriptor()
    second = build_public_patch_relation_descriptor()
    assert first.descriptor_digest == second.descriptor_digest
    assert np.array_equal(first.coefficients, second.coefficients)
    assert first.coefficients.shape == TOKEN_GRID_SHAPE
    assert first.coefficients.dtype == np.dtype("<f4")
    assert first.coefficients.flags.c_contiguous
    assert np.count_nonzero(first.coefficients) == 6
    for time_index in (3, 4, 5):
        assert float(np.sum(first.coefficients[time_index])) == 0.0
        assert first.coefficients[time_index, *PATCH_A_TOKEN_COORDINATE] == 1.0
        assert first.coefficients[time_index, *PATCH_B_TOKEN_COORDINATE] == -1.0


def test_descriptor_mutations_fail_closed() -> None:
    descriptor = build_public_patch_relation_descriptor()
    outside = descriptor.coefficients.copy()
    outside[0, 0, 0] = 1.0
    with pytest.raises(ValueError, match="冻结"):
        validate_public_patch_relation_descriptor(
            replace(descriptor, coefficients=outside)
        )
    with pytest.raises(ValueError, match="dtype"):
        validate_public_patch_relation_descriptor(
            replace(
                descriptor,
                coefficients=descriptor.coefficients.astype(np.float64),
            )
        )
    with pytest.raises(ValueError, match="digest"):
        validate_public_patch_relation_descriptor(
            replace(descriptor, descriptor_digest="0" * 64)
        )


def test_key_context_derivation_is_domain_separated() -> None:
    descriptor = build_public_patch_relation_descriptor()
    owner = derive_signed_relation_coefficient(
        master_key=b"owner-key-material",
        context_digest="1" * 64,
        descriptor=descriptor,
    )
    wrong = derive_signed_relation_coefficient(
        master_key=b"wrong-key-material",
        context_digest="1" * 64,
        descriptor=descriptor,
    )
    changed_context = derive_signed_relation_coefficient(
        master_key=b"owner-key-material",
        context_digest="2" * 64,
        descriptor=descriptor,
    )
    assert owner.signed_coefficient in (-1, 1)
    assert wrong.signed_coefficient in (-1, 1)
    assert owner.derivation_digest != wrong.derivation_digest
    assert owner.derivation_digest != changed_context.derivation_digest
    assert descriptor.descriptor_digest == build_public_patch_relation_descriptor().descriptor_digest


def test_phase_delta_clean_noop_and_positive_negative_antisymmetry() -> None:
    descriptor = build_public_patch_relation_descriptor()
    clean = build_relation_phase_delta(descriptor, signed_coefficient=0)
    positive = build_relation_phase_delta(descriptor, signed_coefficient=1)
    negative = build_relation_phase_delta(descriptor, signed_coefficient=-1)
    assert clean.shape == ROPE_TUPLE_SHAPE
    assert clean.dtype == np.dtype("<f8")
    assert np.count_nonzero(clean) == 0
    assert np.array_equal(positive, -negative)
    assert float(np.max(np.abs(positive))) == PHASE_BUDGET_RADIANS
    assert np.count_nonzero(positive) == 12
    assert np.all(positive[..., 2:] == 0.0)
    with pytest.raises(ValueError, match="signed_coefficient"):
        build_relation_phase_delta(descriptor, signed_coefficient=2)


def test_wan_rope_rotation_and_input_rejection() -> None:
    descriptor = build_public_patch_relation_descriptor()
    cosine = np.ones(ROPE_TUPLE_SHAPE, dtype="<f8")
    sine = np.zeros(ROPE_TUPLE_SHAPE, dtype="<f8")
    clean_cos, clean_sin = apply_wan_rotary_phase_numpy(
        cosine,
        sine,
        descriptor=descriptor,
        signed_coefficient=0,
    )
    assert np.array_equal(clean_cos, cosine)
    assert np.array_equal(clean_sin, sine)
    positive_cos, positive_sin = apply_wan_rotary_phase_numpy(
        cosine,
        sine,
        descriptor=descriptor,
        signed_coefficient=1,
    )
    negative_cos, negative_sin = apply_wan_rotary_phase_numpy(
        cosine,
        sine,
        descriptor=descriptor,
        signed_coefficient=-1,
    )
    assert np.array_equal(positive_cos, negative_cos)
    assert np.allclose(positive_sin, -negative_sin, rtol=0.0, atol=1e-18)
    with pytest.raises(ValueError, match="dtype"):
        apply_wan_rotary_phase_numpy(
            cosine.astype(np.float32),
            sine,
            descriptor=descriptor,
            signed_coefficient=1,
        )
    nonfinite = cosine.copy()
    nonfinite.reshape(-1)[0] = np.nan
    with pytest.raises(ValueError, match="有限"):
        apply_wan_rotary_phase_numpy(
            nonfinite,
            sine,
            descriptor=descriptor,
            signed_coefficient=1,
        )


def test_rgb24_patch_relation_feature_preserves_time_and_sign() -> None:
    video = np.zeros(VIDEO_SHAPE_RGB24, dtype=np.uint8)
    row_a, column_a = PATCH_A_TOKEN_COORDINATE
    y0 = row_a * PATCH_SIZE_PIXELS
    x0 = column_a * PATCH_SIZE_PIXELS
    ramp = np.arange(PATCH_SIZE_PIXELS, dtype=np.uint8)[None, :, None]
    for multiplier, frame_index in enumerate(range(11, 22), start=1):
        video[
            frame_index,
            y0 : y0 + PATCH_SIZE_PIXELS,
            x0 : x0 + PATCH_SIZE_PIXELS,
            :,
        ] = ramp * multiplier
    feature = extract_saved_rgb24_patch_relation_feature(video)
    assert feature.shape == FEATURE_SHAPE
    assert feature.dtype == np.dtype("<f8")
    assert not np.array_equal(feature[0], feature[-1])
    assert np.linalg.norm(feature[0]) > 0.0
    assert np.allclose(feature[:, 3:], 0.0, atol=1e-12)

    swapped = np.zeros_like(video)
    row_b, column_b = PATCH_B_TOKEN_COORDINATE
    swapped[
        :,
        row_b * PATCH_SIZE_PIXELS : (row_b + 1) * PATCH_SIZE_PIXELS,
        column_b * PATCH_SIZE_PIXELS : (column_b + 1) * PATCH_SIZE_PIXELS,
        :,
    ] = video[
        :,
        row_a * PATCH_SIZE_PIXELS : (row_a + 1) * PATCH_SIZE_PIXELS,
        column_a * PATCH_SIZE_PIXELS : (column_a + 1) * PATCH_SIZE_PIXELS,
        :,
    ]
    assert np.allclose(
        extract_saved_rgb24_patch_relation_feature(swapped),
        -feature,
        rtol=0.0,
        atol=1e-12,
    )


def test_feature_rejects_resize_dtype_and_noncontiguous() -> None:
    video = np.zeros(VIDEO_SHAPE_RGB24, dtype=np.uint8)
    with pytest.raises(ValueError, match="shape"):
        extract_saved_rgb24_patch_relation_feature(video[:, :, :-1])
    with pytest.raises(ValueError, match="dtype"):
        extract_saved_rgb24_patch_relation_feature(video.astype(np.float32))
    with pytest.raises(ValueError, match="C-contiguous"):
        extract_saved_rgb24_patch_relation_feature(video[:, :, ::-1, :])


def test_c0_transfer_and_identity_a_apply_only_pass() -> None:
    descriptor, construction, clean, odd = _c0_construction()
    assert construction.construction_ready
    assert construction.feature_schema_id == FEATURE_SCHEMA_ID
    assert construction.statistics.antisymmetry_cosine == pytest.approx(1.0)
    result = evaluate_gate0_apply_only(
        descriptor=descriptor,
        construction=construction,
        clean_a=clean,
        clean_b=clean,
        positive=odd,
        negative=-odd,
        positive_exposure=PHASE_BUDGET_RADIANS,
        negative_exposure=-PHASE_BUDGET_RADIANS,
    )
    assert result.gate_zero_ready
    assert result.transfer_direction_cosine == pytest.approx(1.0)
    assert result.transfer_relative_error == pytest.approx(0.0)
    assert not result.formal_result
    assert not result.stage_progression_allowed
    assert not result.observer_implementation_allowed


def test_asymmetric_raw_vectors_lock_signed_statistic_formulas() -> None:
    clean = np.zeros(FEATURE_SHAPE, dtype="<f8")
    positive_whitened = np.zeros(FEATURE_SHAPE, dtype="<f8")
    negative_whitened = np.zeros(FEATURE_SHAPE, dtype="<f8")
    positive_whitened.reshape(-1)[:2] = [3.0, 1.0]
    negative_whitened.reshape(-1)[:2] = [-1.0, -2.0]
    whitening = fit_c0_whitening(clean, clean)
    statistics = compute_signed_relation_statistics(
        clean_a=clean,
        clean_b=clean,
        positive=np.ascontiguousarray(positive_whitened * 1e-6, dtype="<f8"),
        negative=np.ascontiguousarray(negative_whitened * 1e-6, dtype="<f8"),
        whitening=whitening,
    )

    expected_odd = np.array([2.0, 1.5])
    expected_common = np.array([1.0, -0.5])
    assert np.array_equal(statistics.observed_odd.reshape(-1)[:2], expected_odd)
    assert np.array_equal(
        statistics.observed_common.reshape(-1)[:2],
        expected_common,
    )
    assert statistics.clean_noise_norm == pytest.approx(1e-6, abs=0.0)
    assert statistics.odd_norm == pytest.approx(2.5)
    assert statistics.common_norm == pytest.approx(np.sqrt(1.25))
    assert statistics.antisymmetry_cosine == pytest.approx(
        5.0 / (np.sqrt(10.0) * np.sqrt(5.0))
    )
    assert statistics.antisymmetry_residual == pytest.approx(
        np.sqrt(5.0) / (np.sqrt(10.0) + np.sqrt(5.0))
    )
    assert statistics.common_odd_ratio == pytest.approx(np.sqrt(1.25) / 2.5)
    assert statistics.odd_clean_noise_ratio == pytest.approx(2.5e6)


def test_asymmetric_identity_a_locks_prediction_and_relative_error() -> None:
    descriptor, construction, clean, odd = _c0_construction()
    result = evaluate_gate0_apply_only(
        descriptor=descriptor,
        construction=construction,
        clean_a=clean,
        clean_b=clean,
        positive=np.ascontiguousarray(1.6 * odd, dtype="<f8"),
        negative=np.ascontiguousarray(-1.4 * odd, dtype="<f8"),
        positive_exposure=PHASE_BUDGET_RADIANS,
        negative_exposure=-PHASE_BUDGET_RADIANS,
    )
    assert result.gate_zero_ready
    assert result.transfer_direction_cosine == pytest.approx(1.0)
    assert result.transfer_relative_error == pytest.approx(1.0 / 3.0)


def test_bad_exposure_common_response_and_forged_transfer_fail_closed() -> None:
    descriptor, construction, clean, odd = _c0_construction()
    with pytest.raises(ValueError, match="exposures"):
        construct_c0_relation_transfer(
            descriptor=descriptor,
            clean_a=clean,
            clean_b=clean,
            positive=odd,
            negative=-odd,
            positive_exposure=0.0,
            negative_exposure=-PHASE_BUDGET_RADIANS,
        )
    common_dominated = evaluate_gate0_apply_only(
        descriptor=descriptor,
        construction=construction,
        clean_a=clean,
        clean_b=clean,
        positive=2.0 * odd,
        negative=np.zeros_like(odd),
        positive_exposure=PHASE_BUDGET_RADIANS,
        negative_exposure=-PHASE_BUDGET_RADIANS,
    )
    assert not common_dominated.gate_zero_ready
    forged = replace(
        construction,
        transfer_values=np.ascontiguousarray(
            construction.transfer_values * 2.0,
            dtype="<f8",
        ),
    )
    with pytest.raises(ValueError, match="digest"):
        evaluate_gate0_apply_only(
            descriptor=descriptor,
            construction=forged,
            clean_a=clean,
            clean_b=clean,
            positive=odd,
            negative=-odd,
            positive_exposure=PHASE_BUDGET_RADIANS,
            negative_exposure=-PHASE_BUDGET_RADIANS,
        )
    forged_statistics = replace(
        construction.statistics,
        antisymmetry_cosine=0.0,
    )
    with pytest.raises(ValueError):
        evaluate_gate0_apply_only(
            descriptor=descriptor,
            construction=replace(
                construction,
                statistics=forged_statistics,
            ),
            clean_a=clean,
            clean_b=clean,
            positive=odd,
            negative=-odd,
            positive_exposure=PHASE_BUDGET_RADIANS,
            negative_exposure=-PHASE_BUDGET_RADIANS,
        )
