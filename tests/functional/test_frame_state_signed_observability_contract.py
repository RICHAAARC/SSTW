from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from evaluation.protocol.frame_state_signed_observability_contract import (
    CONTRACT_STATE,
    DEFAULT_CONFIG_PATH,
    EXPECTED_AUTHORIZATION_BOUNDARY,
    EXPECTED_CHECKPOINT_SPECS,
    FROZEN_PROTOCOL_DIGEST,
    METHOD_ID,
    PROFILE_ID,
    build_frame_state_probe_plan,
    build_public_context_record,
    canonical_json_bytes,
    canonical_json_digest,
    frame_state_probe_plan_digest,
    load_frame_state_signed_observability_config,
    parse_canonical_public_context_bytes,
    protocol_contract_canonical_bytes,
    validate_frame_state_probe_plan,
    validate_frame_state_signed_observability_config,
    validate_public_context_record,
)


CONFIG_PATH = Path(DEFAULT_CONFIG_PATH)
NONCE = "0123456789abcdef0123456789abcdef"
EXPECTED_PUBLIC_CONTEXT_DIGEST = (
    "b69c41a26b9ed98176d69db1845a58bf1830dab43da4b013541dd77bc463a936"
)
pytestmark = pytest.mark.quick


def _load_raw_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_loads_frozen_design_only_contract() -> None:
    config = load_frame_state_signed_observability_config()

    assert config["profile_id"] == PROFILE_ID
    assert config["method_id"] == METHOD_ID
    assert config["contract_state"] == CONTRACT_STATE
    assert config["protocol_digest"] == FROZEN_PROTOCOL_DIGEST
    assert config["authorization_boundary"] == EXPECTED_AUTHORIZATION_BOUNDARY
    assert all(
        value is False
        for key, value in config["authorization_boundary"].items()
        if key.endswith("_allowed") or key.endswith("_authorized")
    )
    assert config["authorization_boundary"]["formal_result"] is False
    assert (
        config["authorization_boundary"]["stage_progression_allowed"] is False
    )


def test_protocol_digest_binds_only_no_float_protocol_contract() -> None:
    config = load_frame_state_signed_observability_config()

    assert canonical_json_digest(config["protocol_contract"]) == (
        FROZEN_PROTOCOL_DIGEST
    )
    assert protocol_contract_canonical_bytes(config) == canonical_json_bytes(
        config["protocol_contract"]
    )
    assert "public_context_contract" in config["protocol_contract"]
    assert "execution_identity_contract" not in config["protocol_contract"]


def test_public_dictionary_is_a_separate_frozen_artifact() -> None:
    dictionary = load_frame_state_signed_observability_config()[
        "protocol_contract"
    ]["carrier_dictionary_contract"]

    assert dictionary["construction_target"] == (
        "decoded_local_temporal_feature_surrogate_for_saved_video_gate"
    )
    assert dictionary["public_dictionary_artifact_required"] is True
    assert dictionary["per_video_dictionary_duplication_allowed"] is False
    assert dictionary["dictionary_artifact_format"] == (
        "public_npz_with_exact_shape_dtype_and_array_digest"
    )
    assert dictionary["dictionary_artifact_array_count"] == 1
    assert dictionary["dictionary_artifact_array_key"] == (
        "frame_state_public_atom"
    )
    assert dictionary["dictionary_artifact_array_shape"] == [
        1,
        16,
        9,
        40,
        64,
    ]
    assert dictionary["dictionary_artifact_array_shape"] == (
        dictionary["latent_layout"]
    )
    assert dictionary["dictionary_artifact_byte_order"] == "little_endian"
    assert dictionary["atom_sign_tie_break_rule"] == (
        "lowest_c_order_flat_index"
    )
    assert dictionary[
        "dictionary_zero_outside_restricted_latent_time_indices"
    ] is True


def test_public_dictionary_construction_is_unique_and_not_keyed() -> None:
    protocol = load_frame_state_signed_observability_config()[
        "protocol_contract"
    ]
    dictionary = protocol["carrier_dictionary_contract"]
    feature = protocol["local_output_feature_contract"]

    assert dictionary["power_iteration_initialization_algorithm"] == (
        "sha256_domain_nul_u64be_counter_bits_msb_first_rademacher"
    )
    assert dictionary["power_iteration_initialization_keyed"] is False
    assert dictionary["power_iteration_initialization_bit_mapping"] == (
        "zero_to_negative_one_one_to_positive_one"
    )
    assert dictionary["power_iteration_initialization_counter_start"] == 0
    assert dictionary["power_iteration_initialization_domain_encoding"] == (
        "utf8"
    )
    assert dictionary["power_iteration_initialization_support"] == (
        "restricted_latent_time_indices_only"
    )
    assert dictionary["power_iteration_initialization_coordinate_order"] == (
        "compact_restricted_support_c_order_batch_channel_restricted_time_"
        "height_width_equivalent_full_c_order_skip_non_support"
    )
    assert dictionary["power_iteration_initialization_digest_stream"] == (
        "concatenate_sha256_digest_bits_without_padding_until_support_filled"
    )
    assert dictionary["power_iteration_operator"] == (
        "restricted_support_projection_times_"
        "decoder_local_surrogate_jacobian_transpose_times_jacobian_"
        "times_restricted_support_projection"
    )
    assert dictionary["power_iteration_count"] == 8
    assert (
        dictionary["power_iteration_postrun_iteration_selection_allowed"]
        is False
    )
    assert dictionary["construction_method"] == (
        "public_wan_vae_decoder_local_jacobian_fixed_eight_iteration_"
        "temporally_weighted_direction"
    )
    assert dictionary["construction_direction_claim"] == (
        "fixed_eight_iteration_temporally_weighted_decoder_jacobian_"
        "aligned_not_exact_leading_singular"
    )
    assert dictionary["temporal_weight_application"].startswith(
        "after_eight_iterations_"
    )
    assert dictionary["construction_surrogate_frame_indices"] == (
        feature["video_window_frame_indices"]
    )
    assert dictionary["construction_surrogate_feature_dimension"] == (
        feature["feature_dimension"]
    )
    assert dictionary["construction_surrogate_geometry_source"] == (
        "local_output_feature_contract_except_float32_pre_save"
    )
    assert feature["feature_flatten_order"] == (
        "frame_then_cell_row_then_cell_column_then_channel"
    )
    assert dictionary["construction_surrogate_decode_boundary"] == (
        "wan_vae_decode_sample_then_exact_"
        "video_processor_denormalization_mirror"
    )
    assert dictionary["construction_surrogate_postprocess_formula"] == (
        "clamp_raw_vae_decode_sample_div_two_plus_"
        "one_half_to_closed_zero_one"
    )
    assert dictionary["construction_surrogate_clamp_in_jacobian"] is True
    assert dictionary["construction_surrogate_value_range"] == (
        "closed_zero_one"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (
            "construction_direction_claim",
            "exact_leading_right_singular_direction",
        ),
        (
            "power_iteration_initialization_coordinate_order",
            "caller_selected_order",
        ),
        (
            "construction_surrogate_decode_boundary",
            "raw_vae_decode_sample",
        ),
        ("construction_surrogate_clamp_in_jacobian", False),
    ],
)
def test_dictionary_algorithm_mutations_fail_semantically_before_frozen_digest(
    field: str,
    value: object,
) -> None:
    config = _load_raw_config()
    config["protocol_contract"]["carrier_dictionary_contract"][field] = value
    config["protocol_digest"] = canonical_json_digest(
        config["protocol_contract"]
    )

    with pytest.raises(ValueError, match="public dictionary"):
        validate_frame_state_signed_observability_config(config)


def test_protocol_contains_no_hidden_true_execution_authorization() -> None:
    protocol = load_frame_state_signed_observability_config()[
        "protocol_contract"
    ]

    def visit(value: object) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key.endswith("_allowed") or key.endswith("_authorized"):
                    assert item is False
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(protocol)
    construction = protocol["construction_identity_contract"]
    assert construction["dictionary_construction_scope"] == (
        "construction_identity_design_responsibility_only"
    )
    assert construction["transfer_estimation_scope"] == (
        "construction_identity_design_responsibility_only"
    )


def test_hidden_protocol_execution_authorization_is_rejected_before_digest() -> None:
    config = _load_raw_config()
    config["protocol_contract"]["runtime_override_allowed"] = True

    with pytest.raises(ValueError, match="design-only protocol"):
        validate_frame_state_signed_observability_config(config)


def test_runtime_placeholders_are_excluded_but_exactly_frozen() -> None:
    config = _load_raw_config()
    original_digest = canonical_json_digest(config["protocol_contract"])
    config["execution_identity_contract"][
        "construction_prompt_id_placeholder"
    ] = "caller_value"

    assert canonical_json_digest(config["protocol_contract"]) == original_digest
    with pytest.raises(ValueError, match="execution_identity_contract"):
        validate_frame_state_signed_observability_config(config)


def test_protocol_mutation_with_recomputed_self_digest_still_fails() -> None:
    config = _load_raw_config()
    config["protocol_contract"]["signed_observability_gate_contract"][
        "minimum_antisymmetry_cosine_millionths"
    ] = 1
    config["protocol_digest"] = canonical_json_digest(
        config["protocol_contract"]
    )

    with pytest.raises(ValueError, match="代码冻结合同摘要"):
        validate_frame_state_signed_observability_config(config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gpu_execution_allowed", True),
        ("colab_execution_allowed", True),
        ("construction_execution_allowed", True),
        ("runner_implementation_allowed", True),
        ("observer_implementation_allowed", True),
        ("formal_result", True),
        ("stage_progression_allowed", True),
        ("paper_claim_allowed", True),
    ],
)
def test_authorization_mutations_fail_closed(field: str, value: bool) -> None:
    config = _load_raw_config()
    config["authorization_boundary"][field] = value

    with pytest.raises(ValueError, match="authorization_boundary"):
        validate_frame_state_signed_observability_config(config)


def test_unknown_top_level_field_is_rejected() -> None:
    config = _load_raw_config()
    config["runtime_override"] = {}

    with pytest.raises(ValueError, match="顶层字段集合"):
        validate_frame_state_signed_observability_config(config)


def test_protocol_contract_float_is_rejected_before_digest() -> None:
    config = _load_raw_config()
    config["protocol_contract"]["runtime_float"] = 0.5

    with pytest.raises(TypeError, match="只允许"):
        validate_frame_state_signed_observability_config(config)


def test_duplicate_config_key_is_rejected(tmp_path: Path) -> None:
    raw = CONFIG_PATH.read_text(encoding="utf-8")
    duplicate = '{"profile_id":"forged",' + raw[1:]
    path = tmp_path / "duplicate.json"
    path.write_text(duplicate, encoding="utf-8")

    with pytest.raises(ValueError, match="重复 key"):
        load_frame_state_signed_observability_config(path)


def test_public_context_exact_record_and_fixed_canonical_digest() -> None:
    config = load_frame_state_signed_observability_config()
    record = build_public_context_record(
        config,
        public_nonce_random=NONCE,
    )
    raw = canonical_json_bytes(record)

    assert set(record) == {
        "context_schema_id",
        "method_id",
        "protocol_digest",
        "public_nonce_random",
        "state_clock_rate_denominator",
        "state_clock_rate_numerator",
        "state_window_count",
    }
    assert record["protocol_digest"] == FROZEN_PROTOCOL_DIGEST
    assert record["state_clock_rate_numerator"] == 8
    assert record["state_clock_rate_denominator"] == 1
    assert record["state_window_count"] == 1
    assert canonical_json_digest(record) == EXPECTED_PUBLIC_CONTEXT_DIGEST
    assert canonical_json_digest(record) == validate_public_context_record(
        record, config
    )
    parsed, digest = parse_canonical_public_context_bytes(raw, config)
    assert parsed == record
    assert digest == canonical_json_digest(record)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("public_nonce_random", "A" * 32, "nonce"),
        ("public_nonce_random", "0" * 31, "nonce"),
        ("protocol_digest", "0" * 64, "protocol_digest"),
        ("state_clock_rate_numerator", True, "正整数"),
        ("state_clock_rate_numerator", 7, "冻结 protocol"),
        ("state_clock_rate_denominator", 0, "正整数"),
        ("state_window_count", 2, "冻结 protocol"),
    ],
)
def test_public_context_mutations_fail_closed(
    field: str,
    value: object,
    error: str,
) -> None:
    config = load_frame_state_signed_observability_config()
    record = build_public_context_record(
        config,
        public_nonce_random=NONCE,
    )
    record[field] = value

    with pytest.raises((TypeError, ValueError), match=error):
        validate_public_context_record(record, config)


def test_public_context_extra_field_is_rejected() -> None:
    config = load_frame_state_signed_observability_config()
    record = build_public_context_record(
        config,
        public_nonce_random=NONCE,
    )
    record["prompt_id"] = "forbidden"

    with pytest.raises(ValueError, match="字段集合"):
        validate_public_context_record(record, config)


@pytest.mark.parametrize(
    "transform",
    [
        lambda raw: b" " + raw,
        lambda raw: raw + b"\n",
        lambda raw: raw.replace(b",", b", ", 1),
        lambda raw: b"\xef\xbb\xbf" + raw,
    ],
)
def test_noncanonical_public_context_bytes_are_rejected(transform) -> None:
    config = load_frame_state_signed_observability_config()
    record = build_public_context_record(
        config,
        public_nonce_random=NONCE,
    )

    with pytest.raises(ValueError):
        parse_canonical_public_context_bytes(
            transform(canonical_json_bytes(record)),
            config,
        )


def test_duplicate_public_context_key_is_rejected() -> None:
    config = load_frame_state_signed_observability_config()
    record = build_public_context_record(
        config,
        public_nonce_random=NONCE,
    )
    raw = canonical_json_bytes(record)
    duplicate = (
        b'{"method_id":"forged",' + raw[1:]
    )

    with pytest.raises(ValueError, match="重复 key"):
        parse_canonical_public_context_bytes(duplicate, config)


def test_unicode_is_not_normalized_by_canonicalizer() -> None:
    assert canonical_json_digest({"value": "\u00e9"}) != (
        canonical_json_digest({"value": "e\u0301"})
    )


def test_builds_exact_eight_item_identity_isolated_plan() -> None:
    config = load_frame_state_signed_observability_config()
    plan = build_frame_state_probe_plan(config)

    assert len(plan) == 8
    assert [row.plan_index for row in plan] == list(range(8))
    assert [row.probe_id for row in plan] == [
        "construction_clean_a",
        "construction_clean_b",
        "construction_positive",
        "construction_negative",
        "signed_observability_clean_a",
        "signed_observability_clean_b",
        "signed_observability_positive",
        "signed_observability_negative",
    ]
    assert [row.signed_state_coefficient for row in plan] == [
        0,
        0,
        1,
        -1,
        0,
        0,
        1,
        -1,
    ]
    assert {row.state_window_id for row in plan} == {
        "central_video_frame_window"
    }
    assert {row.state_dimension_index for row in plan} == {0}
    assert all(row.execution_authorized is False for row in plan)
    assert all(row.formal_result is False for row in plan)
    assert all(row.stage_progression_allowed is False for row in plan)
    assert frame_state_probe_plan_digest(plan, config) == (
        "bee85afe0969e4c95eec180acea9ca7f056d77925045744aaa3817b5568e0665"
    )


def test_reordered_or_mutated_probe_plan_is_rejected() -> None:
    config = load_frame_state_signed_observability_config()
    plan = list(build_frame_state_probe_plan(config))
    reordered = deepcopy(plan)
    reordered[2], reordered[3] = reordered[3], reordered[2]
    mutated = deepcopy(plan)
    mutated[6] = replace(mutated[6], execution_authorized=True)

    with pytest.raises(ValueError, match="精确一致"):
        validate_frame_state_probe_plan(reordered, config)
    with pytest.raises(ValueError, match="精确一致"):
        validate_frame_state_probe_plan(mutated, config)


def test_transfer_contract_does_not_authorize_independent_h0() -> None:
    config = load_frame_state_signed_observability_config()
    transfer = config["protocol_contract"]["transfer_estimation_contract"]

    assert transfer["estimated_object"] == (
        "restricted_transfer_t0_equals_h0_d0"
    )
    assert transfer["conceptual_operator_independently_identifiable"] is False
    assert transfer["independent_h0_artifact_allowed"] is False
    assert transfer["actual_signed_exposure_required"] is True
    assert (
        transfer["actual_signed_exposure_zero_denominator_fail_closed"]
        is True
    )
    assert transfer["signed_observability_prediction_formula"] == (
        "t0_times_one_half_positive_minus_negative_actual_signed_exposure"
    )


def test_actual_exposure_is_scheduler_state_update_signed() -> None:
    flow = load_frame_state_signed_observability_config()[
        "protocol_contract"
    ]["flow_injection_contract"]

    assert flow["probe_polarity_coordinate"] == (
        "scheduler_state_update_coordinate"
    )
    assert flow["scheduler_delta_sigma_formula"] == (
        "next_sigma_minus_current_sigma"
    )
    assert flow["intended_velocity_sign_formula"] == (
        "signed_state_coefficient_times_sign_delta_sigma"
    )
    assert flow["actual_signed_exposure_formula"] == (
        "sum_flow_steps_delta_sigma_times_inner_product_"
        "actual_delta_velocity_and_public_dictionary_atom"
    )
    assert flow["actual_signed_exposure_accumulation_dtype"] == "float64"


def test_gate_noise_and_t0_prediction_formulas_are_frozen() -> None:
    gate = load_frame_state_signed_observability_config()[
        "protocol_contract"
    ]["signed_observability_gate_contract"]

    assert gate["clean_noise_norm_formula"] == (
        "one_half_l2_clean_a_minus_clean_b"
    )
    assert gate["clean_noise_denominator_formula"] == (
        "max_clean_noise_norm_and_frozen_floor"
    )
    assert gate["odd_clean_noise_ratio_formula"] == (
        "l2_odd_div_max_clean_noise_norm_and_frozen_floor"
    )
    assert gate["transfer_direction_cosine_formula"] == (
        "cosine_observed_odd_vs_"
        "t0_times_half_actual_exposure_difference"
    )
    assert gate["transfer_relative_error_formula"] == (
        "l2_observed_odd_minus_prediction_"
        "div_max_l2_observed_odd_and_frozen_floor"
    )


def test_checkpoint_representation_and_gate_applicability_matrix_is_exact() -> None:
    protocol = load_frame_state_signed_observability_config()[
        "protocol_contract"
    ]
    checkpoint = protocol["checkpoint_contract"]
    transfer = protocol["transfer_estimation_contract"]

    assert checkpoint["checkpoint_specs"] == EXPECTED_CHECKPOINT_SPECS
    assert [item["feature_dimension"] for item in EXPECTED_CHECKPOINT_SPECS] == [
        1,
        528,
        528,
    ]
    assert all(
        item["signed_gate_ids"] == [
            "antisymmetry_cosine",
            "antisymmetry_residual",
            "common_odd_ratio",
            "odd_clean_noise_ratio",
        ]
        for item in EXPECTED_CHECKPOINT_SPECS
    )
    assert [item["transfer_gate_ids"] for item in EXPECTED_CHECKPOINT_SPECS] == [
        [],
        [],
        [
            "t0_transfer_direction_cosine",
            "t0_transfer_relative_error",
        ],
    ]
    assert checkpoint["primary_checkpoint_id"] == (
        "saved_video_local_temporal_feature"
    )
    assert checkpoint["common_signed_gate_identity_roles"] == [
        "construction_identity",
        "signed_observability_identity",
    ]
    assert checkpoint["construction_identity_signed_gate_role"] == (
        "construction_readiness_only_not_gate0_pass"
    )
    assert checkpoint["gate0_decision_identity_role"] == (
        "signed_observability_identity"
    )
    assert checkpoint["primary_t0_estimation_identity_role"] == (
        "construction_identity"
    )
    assert checkpoint["primary_t0_apply_only_gate_identity_role"] == (
        "signed_observability_identity"
    )
    assert transfer["checkpoint_id"] == checkpoint["primary_checkpoint_id"]
    assert transfer["transfer_feature_dimension"] == 528


def test_checkpoint_features_are_raw_then_identity_centered() -> None:
    feature = load_frame_state_signed_observability_config()[
        "protocol_contract"
    ]["local_output_feature_contract"]

    assert feature["feature_output_centering"] == "none_raw_representation"
    assert feature["construction_transfer_clean_intercept_source"] == (
        "construction_clean_a_clean_b_arithmetic_mean"
    )
    assert feature["signed_gate_clean_intercept_source"] == (
        "current_identity_clean_a_clean_b_arithmetic_mean"
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "latent_dimension",
        "decoded_transfer_gate",
        "primary_transfer_gate_set",
        "transfer_checkpoint",
        "gate0_identity_role",
    ],
)
def test_checkpoint_applicability_mutations_fail_closed(
    mutation: str,
) -> None:
    config = _load_raw_config()
    protocol = config["protocol_contract"]
    if mutation == "latent_dimension":
        protocol["checkpoint_contract"]["checkpoint_specs"][0][
            "feature_dimension"
        ] = 528
    elif mutation == "decoded_transfer_gate":
        protocol["checkpoint_contract"]["checkpoint_specs"][1][
            "transfer_gate_ids"
        ] = ["t0_transfer_direction_cosine"]
    elif mutation == "primary_transfer_gate_set":
        protocol["checkpoint_contract"]["primary_transfer_gate_ids"] = []
    elif mutation == "transfer_checkpoint":
        protocol["transfer_estimation_contract"]["checkpoint_id"] = (
            "decoded_local_temporal_feature"
        )
    else:
        protocol["checkpoint_contract"]["gate0_decision_identity_role"] = (
            "construction_identity"
        )
    config["protocol_digest"] = canonical_json_digest(protocol)

    with pytest.raises(ValueError, match="checkpoint|T0|transfer gate|Gate 0"):
        validate_frame_state_signed_observability_config(config)


def test_c0_and_identity_a_probe_ids_are_disjoint() -> None:
    protocol = load_frame_state_signed_observability_config()[
        "protocol_contract"
    ]
    construction = protocol["construction_identity_contract"]
    signed = protocol["signed_observability_identity_contract"]

    assert set(construction["probe_ids"]).isdisjoint(signed["probe_ids"])
    assert construction["probe_polarities"] == [0, 0, 1, -1]
    assert signed["probe_polarities"] == [0, 0, 1, -1]


def test_identity_a_is_apply_only_and_cannot_refit() -> None:
    config = load_frame_state_signed_observability_config()
    identity = config["protocol_contract"][
        "signed_observability_identity_contract"
    ]

    assert identity["dictionary_apply_only"] is True
    assert identity["feature_apply_only"] is True
    assert identity["transfer_apply_only"] is True
    assert identity["whitening_apply_only"] is True
    assert identity["threshold_apply_only"] is True
