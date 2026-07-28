from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
import math
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
from main.methods.state_space_watermark.patch_relation_wan_runtime import (
    FLOW_ENERGY_BUDGET_RATIO,
    LAMBDA_MAX,
    VELOCITY_NORM_RATIO_BUDGET,
    CfgRopeApplicationPair,
    PhaseProjectionSignEvaluation,
    SymmetricPhaseProjectionSelection,
    WanRopeBranchApplicationRecord,
    evaluate_phase_projection_sign_numpy,
    measure_cfg_state_update_numpy,
    select_symmetric_phase_projection,
    validate_cfg_rope_application_pair,
)
from main.methods.state_space_watermark import (
    patch_relation_wan_runtime as runtime_module,
)
from experiments.generative_video_model_probe.patch_relation_gate0_construction import (
    GovernedPatchRelationStep,
    PatchRelationProbeMeasurement,
    PatchRelationRuntimeBatch,
    _output_binding_digest,
    _feature_record,
    _generation_record,
    _governed_step_from_measurement,
    _require_native_bfloat16_runtime,
    _step_record,
    _validate_c0_artifact,
    _validate_runtime_batch,
    _write_c0_artifact,
    run_patch_relation_gate0_construction,
)
from experiments.generative_video_model_probe import (
    patch_relation_gate0_construction as runner_module,
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
    assert plan[0].identity_id != plan[4].identity_id
    assert all(row.execution_authorized for row in plan)
    assert all(value is False for value in frozen_method_boundary().values())
    mutated = json.loads(json.dumps(config))
    mutated["protocol_contract"]["rope_phase_contract"][
        "temporal_rope_pair_index"
    ] = 1
    with pytest.raises(ValueError, match="冻结 config"):
        build_patch_relation_gate0_plan(mutated)


class _FakeCudaBfloat16:
    def __init__(
        self,
        *,
        native_supported: bool = True,
        capability: tuple[int, int] = (8, 0),
    ) -> None:
        self.native_supported = native_supported
        self.capability = capability
        self.including_emulation_values: list[bool] = []

    def is_bf16_supported(self, *, including_emulation: bool = True) -> bool:
        self.including_emulation_values.append(including_emulation)
        return True if including_emulation else self.native_supported

    def get_device_capability(self) -> tuple[int, int]:
        return self.capability


class _FakeTorchBfloat16:
    def __init__(self, cuda: object) -> None:
        self.cuda = cuda
        self.bfloat16 = object()
        self.float16 = object()


def test_native_bfloat16_preflight_disables_emulation_and_requires_exact_dtype() -> None:
    cuda = _FakeCudaBfloat16()
    torch = _FakeTorchBfloat16(cuda)
    _require_native_bfloat16_runtime(
        torch,
        selected_dtype=torch.bfloat16,
    )
    assert cuda.including_emulation_values == [False]

    emulated_only = _FakeCudaBfloat16(native_supported=False)
    torch_emulated = _FakeTorchBfloat16(emulated_only)
    with pytest.raises(RuntimeError, match="原生bfloat16"):
        _require_native_bfloat16_runtime(
            torch_emulated,
            selected_dtype=torch_emulated.bfloat16,
        )
    with pytest.raises(RuntimeError, match="selected dtype"):
        _require_native_bfloat16_runtime(
            torch,
            selected_dtype=torch.float16,
        )
    old_gpu = _FakeTorchBfloat16(
        _FakeCudaBfloat16(capability=(7, 5))
    )
    with pytest.raises(RuntimeError, match="compute capability"):
        _require_native_bfloat16_runtime(
            old_gpu,
            selected_dtype=old_gpu.bfloat16,
        )


def test_native_bfloat16_preflight_rejects_legacy_torch_signature() -> None:
    class LegacyCuda:
        @staticmethod
        def is_bf16_supported() -> bool:
            return True

        @staticmethod
        def get_device_capability() -> tuple[int, int]:
            return (8, 0)

    torch = _FakeTorchBfloat16(LegacyCuda())
    with pytest.raises(RuntimeError, match="including_emulation=False"):
        _require_native_bfloat16_runtime(
            torch,
            selected_dtype=torch.bfloat16,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["authorization_boundary"].__setitem__(
            "gpu_execution_allowed", False
        ),
        lambda value: value["protocol_contract"]["rope_phase_contract"].__setitem__(
            "maximum_phase_budget_radians_decimal", "0.03125"
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
    "mutation",
    [
        lambda contract: contract["execution_identity_contract"][
            "construction_identity_c0"
        ].__setitem__("seed_value", 1276),
        lambda contract: contract["execution_identity_contract"][
            "gate0_identity_a"
        ].__setitem__(
            "prompt_text",
            contract["execution_identity_contract"]["construction_identity_c0"][
                "prompt_text"
            ],
        ),
        lambda contract: contract["gate0_runtime_execution_contract"].__setitem__(
            "transformer_cache_enabled", True
        ),
        lambda contract: contract["gate0_runtime_execution_contract"].__setitem__(
            "scheduler_consumes", "base_cfg_velocity"
        ),
        lambda contract: contract["gate0_runtime_execution_contract"][
            "sigma_grid_decimal"
        ].__setitem__(1, "0.9"),
        lambda contract: contract["gate0_runtime_execution_contract"][
            "timestep_by_step_decimal"
        ].__setitem__(1, "947.5"),
        lambda contract: contract["gate0_runtime_execution_contract"].__setitem__(
            "transformer_bfloat16_output_cast_to_float32_before_pipeline_cfg",
            False,
        ),
        lambda contract: contract["gate0_runtime_execution_contract"].__setitem__(
            "scheduler_returned_controlled_next_state_and_counterfactual_base_next_state_revalidated",
            False,
        ),
        lambda contract: contract["gate0_runtime_execution_contract"].__setitem__(
            "scheduler_internal_step_index_before_after_progression_required",
            False,
        ),
        lambda contract: contract["gate0_runtime_execution_contract"].__setitem__(
            "cuda_native_bfloat16_check_including_emulation_false_required",
            False,
        ),
        lambda contract: contract["gate0_runtime_execution_contract"].__setitem__(
            "minimum_cuda_compute_capability_major",
            7,
        ),
        lambda contract: contract["gate0_runtime_execution_contract"].__setitem__(
            "selected_pipeline_dtype_exact_torch_bfloat16_required",
            False,
        ),
        lambda contract: contract["wan_runtime_adapter_contract"][
            "phase_domain_bounded_projection_contract"
        ].__setitem__("maximum_candidate_attempt_count", 12),
        lambda contract: contract["wan_runtime_adapter_contract"][
            "phase_domain_bounded_projection_contract"
        ].__setitem__("backoff_safety_factor_decimal", "1.0"),
        lambda contract: contract["wan_runtime_adapter_contract"][
            "phase_domain_bounded_projection_contract"
        ].__setitem__("candidate_velocity_linear_rescaling_allowed", True),
        lambda contract: contract["wan_runtime_adapter_contract"][
            "phase_domain_bounded_projection_contract"
        ].__setitem__("common_scale_selected_from_worst_positive_and_negative_budget_usage", False),
        lambda contract: contract["gate0_runtime_execution_contract"][
            "probe_order"
        ].reverse(),
    ],
)
def test_rehashed_execution_identity_and_schedule_mutations_fail_closed(
    tmp_path: Path,
    mutation,
) -> None:
    def mutate(config: dict) -> None:
        mutation(config["protocol_contract"])
        config["protocol_digest"] = protocol_digest(config["protocol_contract"])

    with pytest.raises(ValueError):
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
    half_positive = build_relation_phase_delta(
        descriptor,
        signed_coefficient=1,
        phase_projection_scale=0.5,
    )
    half_negative = build_relation_phase_delta(
        descriptor,
        signed_coefficient=-1,
        phase_projection_scale=0.5,
    )
    assert np.array_equal(half_positive, -half_negative)
    assert np.array_equal(half_positive, positive * 0.5)
    with pytest.raises(ValueError, match="signed_coefficient"):
        build_relation_phase_delta(descriptor, signed_coefficient=2)
    for bad_scale in (0.0, -0.1, 1.1, float("nan")):
        with pytest.raises(ValueError, match="phase_projection_scale"):
            build_relation_phase_delta(
                descriptor,
                signed_coefficient=1,
                phase_projection_scale=bad_scale,
            )


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


def test_c0_artifact_readback_rejects_nonprimary_member_mutation(
    tmp_path: Path,
) -> None:
    _descriptor, construction, _clean, _odd = _c0_construction()
    path = tmp_path / "c0.npz"
    _write_c0_artifact(path, construction)
    _validate_c0_artifact(path, construction)
    with np.load(path, allow_pickle=False) as payload:
        members = {name: payload[name].copy() for name in payload.files}
    members["observed_common"].reshape(-1)[0] = 1.0
    np.savez(path, **members)
    with pytest.raises(ValueError, match="readback"):
        _validate_c0_artifact(path, construction)


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


def _fake_rope_pair(
    *,
    probe_id: str,
    step_index: int,
    control_role: str,
    coefficient: int,
) -> CfgRopeApplicationPair:
    descriptor_digest = build_public_patch_relation_descriptor().descriptor_digest
    rows = []
    for branch_index, branch_role in enumerate(
        ("conditional", "unconditional")
    ):
        rows.append(
            WanRopeBranchApplicationRecord(
                probe_id=probe_id,
                step_index=step_index,
                control_role=control_role,
                cfg_branch_role=branch_role,
                cfg_branch_order_index=branch_index,
                signed_coefficient=coefficient,
                maximum_phase_budget_radians=PHASE_BUDGET_RADIANS,
                phase_projection_scale=1.0,
                realized_phase_magnitude_radians=(
                    0.0 if coefficient == 0 else PHASE_BUDGET_RADIANS
                ),
                input_binding_digest="1" * 64,
                descriptor_digest=descriptor_digest,
                rope_call_attempt_count=1,
                successful_rope_call_count=1,
                expected_rope_call_count=1,
                scope_completed_successfully=True,
                clean_exact_noop=coefficient == 0,
            )
        )
    return validate_cfg_rope_application_pair(rows[0], rows[1])


def _fake_phase_projection(
    measurement,
    *,
    base_pair: CfgRopeApplicationPair,
    controlled_pair: CfgRopeApplicationPair,
    base_conditional_velocity: np.ndarray,
    base_unconditional_velocity: np.ndarray,
    controlled_conditional_velocity: np.ndarray,
    controlled_unconditional_velocity: np.ndarray,
) -> SymmetricPhaseProjectionSelection:
    canonical_conditional_delta = np.multiply(
        np.subtract(
            controlled_conditional_velocity,
            base_conditional_velocity,
            dtype=np.float32,
        ),
        np.float32(measurement.signed_coefficient),
        dtype=np.float32,
    )
    canonical_unconditional_delta = np.multiply(
        np.subtract(
            controlled_unconditional_velocity,
            base_unconditional_velocity,
            dtype=np.float32,
        ),
        np.float32(measurement.signed_coefficient),
        dtype=np.float32,
    )

    def one(sign: int) -> PhaseProjectionSignEvaluation:
        pair = controlled_pair
        if sign != measurement.signed_coefficient:
            pair = validate_cfg_rope_application_pair(
                replace(controlled_pair.conditional, signed_coefficient=sign),
                replace(controlled_pair.unconditional, signed_coefficient=sign),
            )
        controlled_conditional = np.ascontiguousarray(
            base_conditional_velocity
            + np.float32(sign) * canonical_conditional_delta,
            dtype="<f4",
        )
        controlled_unconditional = np.ascontiguousarray(
            base_unconditional_velocity
            + np.float32(sign) * canonical_unconditional_delta,
            dtype="<f4",
        )
        return evaluate_phase_projection_sign_numpy(
            base_pair=base_pair,
            controlled_pair=pair,
            phase_projection_scale=1.0,
            base_conditional_velocity=base_conditional_velocity,
            base_unconditional_velocity=base_unconditional_velocity,
            controlled_conditional_velocity=controlled_conditional,
            controlled_unconditional_velocity=controlled_unconditional,
            scheduler_sample=measurement.scheduler_sample,
            delta_sigma=measurement.delta_sigma,
            cumulative_reference_energy_before_step=(
                measurement.cumulative_reference_energy_before_step
            ),
            cumulative_control_energy_before_step=(
                measurement.cumulative_control_energy_before_step
            ),
            remaining_step_count=measurement.remaining_step_count,
        )

    return select_symmetric_phase_projection(
        lambda scale: (one(1), one(-1))
    )


def _fake_governed_steps(
    probe_id: str,
    coefficient: int,
    deltas: tuple[float, ...],
) -> tuple[GovernedPatchRelationStep, ...]:
    rows = []
    cumulative_reference = 0.0
    cumulative_control = 0.0
    shape = runtime_module.SCHEDULER_VELOCITY_SHAPE
    for step_index, delta in enumerate(deltas):
        base_conditional = np.full(shape, 1.0, dtype="<f4")
        base_unconditional = np.full(shape, 0.5, dtype="<f4")
        controlled_conditional = base_conditional.copy()
        controlled_unconditional = base_unconditional.copy()
        if coefficient != 0:
            offset = np.float32(coefficient * 1e-4)
            controlled_conditional = np.ascontiguousarray(
                controlled_conditional + offset,
                dtype="<f4",
            )
            controlled_unconditional = np.ascontiguousarray(
                controlled_unconditional + offset,
                dtype="<f4",
            )
        base_cfg = np.ascontiguousarray(
            base_unconditional
            + np.float32(5.0)
            * np.subtract(
                base_conditional,
                base_unconditional,
                dtype=np.float32,
            ),
            dtype="<f4",
        )
        controlled_cfg = np.ascontiguousarray(
            controlled_unconditional
            + np.float32(5.0)
            * np.subtract(
                controlled_conditional,
                controlled_unconditional,
                dtype=np.float32,
            ),
            dtype="<f4",
        )
        sample = np.ones(shape, dtype="<f4")
        base_next = np.ascontiguousarray(
            sample + np.float32(delta) * base_cfg,
            dtype="<f4",
        )
        controlled_next = np.ascontiguousarray(
            sample + np.float32(delta) * controlled_cfg,
            dtype="<f4",
        )
        base_pair = _fake_rope_pair(
            probe_id=probe_id,
            step_index=step_index,
            control_role="base",
            coefficient=0,
        )
        controlled_pair = _fake_rope_pair(
            probe_id=probe_id,
            step_index=step_index,
            control_role="controlled",
            coefficient=coefficient,
        )
        measurement = measure_cfg_state_update_numpy(
            base_pair=base_pair,
            controlled_pair=controlled_pair,
            base_conditional_velocity=base_conditional,
            base_unconditional_velocity=base_unconditional,
            controlled_conditional_velocity=controlled_conditional,
            controlled_unconditional_velocity=controlled_unconditional,
            scheduler_consumed_velocity=controlled_cfg,
            scheduler_sample=sample,
            scheduler_base_next_state=base_next,
            scheduler_controlled_next_state=controlled_next,
            delta_sigma=delta,
            cumulative_reference_energy_before_step=cumulative_reference,
            cumulative_control_energy_before_step=cumulative_control,
            remaining_step_count=8 - step_index,
        )
        rows.append(
            _governed_step_from_measurement(
                measurement,
                base_pair=base_pair,
                controlled_pair=controlled_pair,
                base_conditional_velocity=base_conditional,
                base_unconditional_velocity=base_unconditional,
                controlled_conditional_velocity=controlled_conditional,
                controlled_unconditional_velocity=controlled_unconditional,
                conditional_encoder_digest="2" * 64,
                unconditional_encoder_digest="3" * 64,
                phase_projection=(
                    None
                    if coefficient == 0
                    else _fake_phase_projection(
                        measurement,
                        base_pair=base_pair,
                        controlled_pair=controlled_pair,
                        base_conditional_velocity=base_conditional,
                        base_unconditional_velocity=base_unconditional,
                        controlled_conditional_velocity=controlled_conditional,
                        controlled_unconditional_velocity=controlled_unconditional,
                    )
                ),
            )
        )
        cumulative_reference += measurement.reference_energy_increment
        cumulative_control += measurement.energy_increment
    return tuple(rows)


def _fake_saved_rgb24(
    coefficient: int,
    *,
    invert_negative: bool = False,
) -> np.ndarray:
    video = np.zeros((33, 320, 512, 3), dtype=np.uint8)
    if coefficient == 0:
        return video
    gradient = np.broadcast_to(
        np.arange(PATCH_SIZE_PIXELS, dtype=np.uint8)[None, :, None] * 8,
        (PATCH_SIZE_PIXELS, PATCH_SIZE_PIXELS, 3),
    )
    row, column = (
        PATCH_A_TOKEN_COORDINATE
        if coefficient > 0 or invert_negative
        else PATCH_B_TOKEN_COORDINATE
    )
    for frame_index in range(11, 22):
        video[
            frame_index,
            row * PATCH_SIZE_PIXELS : (row + 1) * PATCH_SIZE_PIXELS,
            column * PATCH_SIZE_PIXELS : (column + 1) * PATCH_SIZE_PIXELS,
            :,
        ] = gradient
    return np.ascontiguousarray(video)


def _install_fake_velocity_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    small_shape = (1, 1, 1, 1, 2)
    monkeypatch.setattr(
        runtime_module,
        "SCHEDULER_VELOCITY_SHAPE",
        small_shape,
    )
    monkeypatch.setattr(
        runner_module,
        "SCHEDULER_VELOCITY_SHAPE",
        small_shape,
    )


def _install_fake_video_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_velocity_shape(monkeypatch)
    monkeypatch.setattr(
        runner_module,
        "_read_saved_rgb24_file",
        lambda path: np.load(Path(path).with_suffix(".npy")),
    )


def _fake_patch_relation_runtime(
    config,
    plan,
    output_root: Path,
    *,
    fail_gate: bool = False,
) -> PatchRelationRuntimeBatch:
    runtime = config["protocol_contract"]["gate0_runtime_execution_contract"]
    deltas = tuple(
        float(value) for value in runtime["delta_sigma_by_step_decimal"]
    )
    measurements = []
    generation_records = []
    step_records = []
    feature_records = []
    for probe in plan:
        coefficient = probe.signed_state_coefficient
        saved = _fake_saved_rgb24(
            coefficient,
            invert_negative=bool(
                fail_gate
                and probe.identity_role == "gate0_identity_a"
                and coefficient == -1
            ),
        )
        feature = extract_saved_rgb24_patch_relation_feature(saved)
        steps = _fake_governed_steps(probe.probe_id, coefficient, deltas)
        exposure = math.fsum(
            step.signed_state_update_exposure for step in steps
        )
        video = (
            output_root
            / "videos"
            / f"{probe.plan_index:02d}_{probe.probe_id}.mp4"
        )
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(probe.probe_id.encode("utf-8"))
        np.save(video.with_suffix(".npy"), saved)
        video_sha = sha256(video.read_bytes()).hexdigest()
        saved_digest = sha256(saved.tobytes(order="C")).hexdigest()
        feature_digest = sha256(feature.tobytes(order="C")).hexdigest()
        output_binding = _output_binding_digest(
            probe,
            video_sha256=video_sha,
            saved_rgb24_digest=saved_digest,
            feature_digest=feature_digest,
        )
        measurement = PatchRelationProbeMeasurement(
            plan_index=probe.plan_index,
            identity_role=probe.identity_role,
            probe_id=probe.probe_id,
            signed_coefficient=coefficient,
            generator_state_digest_random=(
                "a" * 64
                if probe.identity_role == "construction_c0"
                else "b" * 64
            ),
            initial_hidden_state_digest_random=(
                "c" * 64
                if probe.identity_role == "construction_c0"
                else "d" * 64
            ),
            video_path=str(video),
            video_sha256=video_sha,
            saved_rgb24_digest=saved_digest,
            output_binding_digest=output_binding,
            feature=feature,
            steps=steps,
            actual_signed_exposure=exposure,
        )
        measurements.append(measurement)
        step_records.extend(_step_record(probe, step) for step in steps)
        feature_records.append(
            _feature_record(
                probe,
                feature,
                video_sha256=video_sha,
                saved_rgb24_digest=saved_digest,
                output_binding_digest=output_binding,
            )
        )
        generation_records.append(
            _generation_record(
                config,
                probe,
                measurement,
                generation_runtime_sec=0.0,
            )
        )
    return PatchRelationRuntimeBatch(
        measurements=tuple(measurements),
        generation_records=tuple(generation_records),
        step_records=tuple(step_records),
        feature_records=tuple(feature_records),
    )


@pytest.mark.quick
def test_fake_patch_relation_runner_writes_8_64_8_nonformal_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_video_reader(monkeypatch)
    output = tmp_path / "run"
    decision = run_patch_relation_gate0_construction(
        output,
        runtime_executor=lambda config, plan, root: (
            _fake_patch_relation_runtime(config, plan, root)
        ),
    )
    assert decision["patch_relation_gate0_ready"] is True
    assert decision["generation_record_count"] == 8
    assert decision["trajectory_step_record_count"] == 64
    assert decision["feature_record_count"] == 8
    assert decision["patch_relation_maximum_phase_budget_radians"] == 0.015625
    assert len(decision["patch_relation_phase_projection_selected_scales"]) == 64
    assert (
        len(decision["patch_relation_phase_projection_step_records_digest"])
        == 64
    )
    assert decision["next_double_window_gate_a_design_allowed"] is True
    assert decision["next_double_window_gate_a_execution_allowed"] is False
    assert decision["formal_result"] is False
    assert decision["stage_progression_allowed"] is False
    assert (output / "artifacts/patch_relation_c0_construction.npz").is_file()


def test_runtime_batch_rejects_forged_exposure_and_step_schedule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_video_reader(monkeypatch)
    config = load_patch_relation_gate0_config()
    plan = build_patch_relation_gate0_plan(config)
    batch = _fake_patch_relation_runtime(config, plan, tmp_path / "source")
    first = batch.measurements[0]
    forged_exposure = replace(
        first,
        actual_signed_exposure=first.actual_signed_exposure + 1.0,
    )
    with pytest.raises(ValueError, match="exposure aggregation"):
        _validate_runtime_batch(
            config,
            plan,
            replace(
                batch,
                measurements=(forged_exposure, *batch.measurements[1:]),
            ),
            output_root=tmp_path / "source",
        )
    forged_step = replace(first.steps[0], remaining_step_count=7)
    forged_measurement = replace(
        first,
        steps=(forged_step, *first.steps[1:]),
    )
    with pytest.raises(ValueError, match="governed step"):
        _validate_runtime_batch(
            config,
            plan,
            replace(
                batch,
                measurements=(forged_measurement, *batch.measurements[1:]),
            ),
            output_root=tmp_path / "source",
        )


def test_runtime_batch_rejects_phase_projection_record_forgery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_video_reader(monkeypatch)
    config = load_patch_relation_gate0_config()
    plan = build_patch_relation_gate0_plan(config)
    output = tmp_path / "source"
    batch = _fake_patch_relation_runtime(config, plan, output)
    active_index = 2
    original = batch.measurements[active_index]
    forged_step = replace(
        original.steps[0],
        selected_phase_projection_scale=0.5,
        realized_phase_magnitude_radians=PHASE_BUDGET_RADIANS * 0.5,
    )
    forged_measurement = replace(
        original,
        steps=(forged_step, *original.steps[1:]),
    )
    measurements = list(batch.measurements)
    measurements[active_index] = forged_measurement
    with pytest.raises(ValueError, match="seal"):
        _validate_runtime_batch(
            config,
            plan,
            replace(batch, measurements=tuple(measurements)),
            output_root=output,
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda step: replace(
                step,
                actual_delta_norm=0.5,
                norm_budget=1.0,
                norm_guard_passed=True,
            ),
            "governed step",
        ),
        (
            lambda step: replace(
                step,
                state_update_delta_norm=0.5,
                energy_increment=0.25,
                remaining_flow_energy=1.0,
                energy_guard_passed=True,
            ),
            "governed step",
        ),
        (
            lambda step: replace(
                step,
                state_update_direction_dot=(
                    -step.direction_actual_norm
                    * step.direction_intended_norm
                ),
                direction_cosine=1.0,
                direction_guard_passed=True,
            ),
            "governed step",
        ),
    ],
)
def test_runtime_batch_recomputes_budget_energy_and_direction_guards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation,
    match: str,
) -> None:
    _install_fake_video_reader(monkeypatch)
    config = load_patch_relation_gate0_config()
    plan = build_patch_relation_gate0_plan(config)
    batch = _fake_patch_relation_runtime(config, plan, tmp_path / "source")
    signed_index = 2
    signed = batch.measurements[signed_index]
    forged_step = mutation(signed.steps[0])
    forged = replace(
        signed,
        steps=(forged_step, *signed.steps[1:]),
    )
    with pytest.raises(ValueError, match=match):
        _validate_runtime_batch(
            config,
            plan,
            replace(
                batch,
                measurements=(
                    *batch.measurements[:signed_index],
                    forged,
                    *batch.measurements[signed_index + 1 :],
                ),
            ),
            output_root=tmp_path / "source",
        )


def test_runtime_batch_rejects_coordinated_transition_statistic_forgery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raw-transition seal rejects the exact scalar/budget coordination attack."""

    _install_fake_video_reader(monkeypatch)
    config = load_patch_relation_gate0_config()
    plan = build_patch_relation_gate0_plan(config)
    output = tmp_path / "source"
    batch = _fake_patch_relation_runtime(config, plan, output)
    measurement_index = 2
    step_offset = measurement_index * 8
    signed = batch.measurements[measurement_index]
    original = signed.steps[0]
    seal = original._validated_transition_seal
    assert seal is not None
    assert not hasattr(seal, "payload")
    assert not hasattr(seal, "__dict__")
    with pytest.raises(AttributeError):
        seal.payload = ("forged",)
    assert not hasattr(runner_module, "_seal_validated_transition")
    forged_base_state_norm = 0.052457
    forged_state_norm = 5.2457e-8
    forged_reference_increment = forged_base_state_norm**2
    forged_projected_reference = (
        original.cumulative_reference_energy_before_step
        + forged_reference_increment * original.remaining_step_count
    )
    forged_total_budget = (
        FLOW_ENERGY_BUDGET_RATIO * forged_projected_reference
    )
    forged_remaining = max(
        0.0,
        forged_total_budget
        - original.cumulative_control_energy_before_step,
    )
    forged = replace(
        original,
        base_velocity_norm=1000.0,
        intended_delta_norm=0.5,
        actual_delta_norm=0.5,
        base_state_update_norm=forged_base_state_norm,
        intended_state_update_norm=forged_state_norm,
        state_update_delta_norm=forged_state_norm,
        state_update_direction_dot=forged_state_norm**2,
        direction_actual_norm=forged_state_norm,
        direction_intended_norm=forged_state_norm,
        norm_budget=(
            1000.0 * VELOCITY_NORM_RATIO_BUDGET * LAMBDA_MAX
        ),
        reference_energy_increment=forged_reference_increment,
        projected_reference_energy=forged_projected_reference,
        total_flow_energy_budget=forged_total_budget,
        remaining_flow_energy=forged_remaining,
        energy_increment=forged_state_norm**2,
        direction_cosine=1.0,
        signed_state_update_exposure=forged_state_norm,
        norm_guard_passed=True,
        energy_guard_passed=True,
        direction_guard_passed=True,
    )
    forged_measurement = replace(
        signed,
        steps=(forged, *signed.steps[1:]),
        actual_signed_exposure=(
            signed.actual_signed_exposure
            - original.signed_state_update_exposure
            + forged.signed_state_update_exposure
        ),
    )
    forged_step_records = list(batch.step_records)
    forged_step_records[step_offset] = _step_record(
        plan[measurement_index],
        forged,
    )
    with pytest.raises(ValueError, match="transition数组验证seal"):
        _validate_runtime_batch(
            config,
            plan,
            replace(
                batch,
                measurements=(
                    *batch.measurements[:measurement_index],
                    forged_measurement,
                    *batch.measurements[measurement_index + 1 :],
                ),
                step_records=tuple(forged_step_records),
            ),
            output_root=output,
        )


def test_runtime_batch_requires_exact_output_video_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_video_reader(monkeypatch)
    config = load_patch_relation_gate0_config()
    plan = build_patch_relation_gate0_plan(config)
    output = tmp_path / "source"
    batch = _fake_patch_relation_runtime(config, plan, output)
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(Path(batch.measurements[0].video_path).read_bytes())
    forged = replace(batch.measurements[0], video_path=str(outside))
    with pytest.raises(ValueError, match="video path/SHA"):
        _validate_runtime_batch(
            config,
            plan,
            replace(
                batch,
                measurements=(forged, *batch.measurements[1:]),
            ),
            output_root=output,
        )


def test_runtime_batch_rejects_invalid_mp4_even_with_self_reported_feature(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_velocity_shape(monkeypatch)
    config = load_patch_relation_gate0_config()
    plan = build_patch_relation_gate0_plan(config)
    output = tmp_path / "source"
    batch = _fake_patch_relation_runtime(config, plan, output)
    with pytest.raises(ValueError, match="RGB24 MP4 回读失败"):
        _validate_runtime_batch(
            config,
            plan,
            batch,
            output_root=output,
        )


def test_runtime_batch_rejects_changed_video_with_coordinated_feature_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_video_reader(monkeypatch)
    config = load_patch_relation_gate0_config()
    plan = build_patch_relation_gate0_plan(config)
    output = tmp_path / "source"
    batch = _fake_patch_relation_runtime(config, plan, output)
    first = batch.measurements[0]
    video_path = Path(first.video_path)
    video_path.write_bytes(b"changed-video-content")
    changed_saved = _fake_saved_rgb24(1)
    np.save(video_path.with_suffix(".npy"), changed_saved)
    changed_feature = extract_saved_rgb24_patch_relation_feature(changed_saved)
    changed_rgb_digest = sha256(
        changed_saved.tobytes(order="C")
    ).hexdigest()
    changed_feature_record = _feature_record(
        plan[0],
        changed_feature,
        video_sha256=first.video_sha256,
        saved_rgb24_digest=changed_rgb_digest,
        output_binding_digest=first.output_binding_digest,
    )
    forged = replace(
        first,
        feature=changed_feature,
        saved_rgb24_digest=changed_rgb_digest,
    )
    with pytest.raises(ValueError, match="video path/SHA"):
        _validate_runtime_batch(
            config,
            plan,
            replace(
                batch,
                measurements=(forged, *batch.measurements[1:]),
                feature_records=(
                    changed_feature_record,
                    *batch.feature_records[1:],
                ),
            ),
            output_root=output,
        )


@pytest.mark.quick
def test_patch_relation_method_gate_failure_is_normal_nonformal_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_video_reader(monkeypatch)
    output = tmp_path / "run"
    decision = run_patch_relation_gate0_construction(
        output,
        runtime_executor=lambda config, plan, root: (
            _fake_patch_relation_runtime(
                config,
                plan,
                root,
                fail_gate=True,
            )
        ),
    )
    assert decision["patch_relation_gate0_ready"] is False
    assert decision["patch_relation_gate0_decision"] == (
        "gate0_fail_stop_current_patch_relation_carrier_or_feature"
    )
    assert decision["formal_result"] is False
    assert decision["stage_progression_allowed"] is False
