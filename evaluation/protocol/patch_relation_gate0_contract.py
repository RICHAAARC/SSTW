"""Fail-closed local contract for the first Patch-relation Gate 0 primitive.

This module validates one exact design/local-primitives/runtime-adapter config
and constructs the design-only C0/A probe plan.  It implements and validates
only the local adapter contract; it does not authorize or execute a real model
runtime, GPU, Colab, a runner, an observer, or stage progression.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Mapping

from main.methods.state_space_watermark.patch_relation_carrier import (
    FEATURE_SCHEMA_ID,
    PHASE_BUDGET_RADIANS,
    RELATION_DESCRIPTOR_ID,
    build_public_patch_relation_descriptor,
    frozen_method_boundary,
)
from main.methods.state_space_watermark.patch_relation_wan_runtime import (
    CFG_GUIDANCE_SCALE,
    DIFFUSERS_VERSION,
    FLOW_ENERGY_BUDGET_RATIO,
    LAMBDA_MAX,
    MINIMUM_DIRECTION_COSINE,
    RUNTIME_ADAPTER_PROTOCOL_DIGEST,
    VELOCITY_NORM_RATIO_BUDGET,
)


DEFAULT_CONFIG_PATH = "configs/protocol/sstw_patch_relation_gate0_construction.json"
PROFILE_ID = "sstw_patch_relation_gate0_construction"
METHOD_ID = "frame_state_synchronized_generative_flow_video_watermark"
CONTRACT_STATE = "local_patch_relation_gate0_runtime_adapter_only"
FROZEN_PROTOCOL_DIGEST = (
    "be8574c59b2b2fa3479f841a685d61aa4a028a860e255188a9a1f8e4fe380cd6"
)
_LOWER_HEX_64 = re.compile(r"[0-9a-f]{64}\Z")

EXPECTED_TOP_LEVEL_KEYS = {
    "profile_id",
    "method_id",
    "contract_state",
    "authorization_boundary",
    "protocol_contract",
    "protocol_digest",
}
EXPECTED_AUTHORIZATION_BOUNDARY = {
    **frozen_method_boundary(),
    "construction_execution_allowed": False,
    "claim_support_status": (
        "local_contract_numpy_primitives_and_runtime_adapter_only_"
        "not_method_evidence"
    ),
}
EXPECTED_PROTOCOL_SECTIONS = {
    "authority",
    "historical_failure_boundary",
    "wan_runtime_adapter_contract",
    "token_relation_contract",
    "rope_phase_contract",
    "coefficient_derivation_contract",
    "output_feature_contract",
    "construction_and_gate_contract",
    "method_scope",
}
EXPECTED_PROBE_ROLES = ("clean_a", "clean_b", "positive", "negative")
EXPECTED_ACTUAL_SIGNED_EXPOSURE_INPUT = (
    "local_runtime_adapter_realized_cfg_velocity_state_update_measurement_"
    "requires_future_governed_schedule_binding_not_execution_evidence"
)
EXPECTED_WAN_RUNTIME_ADAPTER_CONTRACT = {
    "diffusers_version": DIFFUSERS_VERSION,
    "model_class": (
        "diffusers.models.transformers.transformer_wan.WanTransformer3DModel"
    ),
    "rotary_class": (
        "diffusers.models.transformers.transformer_wan.WanRotaryPosEmbed"
    ),
    "attention_processor_class": (
        "diffusers.models.transformers.transformer_wan.WanAttnProcessor"
    ),
    "write_primitive": "scoped_transformer_rope_output_temporal_phase_pair",
    "write_boundary": (
        "after_self_rope_hidden_states_before_patch_embedding_and_all_"
        "self_attention_blocks"
    ),
    "official_rope_output": "tuple_freqs_cos_freqs_sin",
    "official_rope_output_shape": [1, 5760, 1, 128],
    "official_rope_output_dtype_non_mps": "float64",
    "runtime_adapter_module": (
        "main.methods.state_space_watermark.patch_relation_wan_runtime"
    ),
    "runtime_adapter_implemented": True,
    "runtime_adapter_execution_allowed": False,
    "scoped_rope_output_contract": {
        "scope": "one_transformer_forward_branch_only",
        "expected_rope_call_attempts": 1,
        "expected_successful_rope_calls": 1,
        "nested_scope_allowed": False,
        "hidden_states_shape": [1, 16, 9, 40, 64],
        "hidden_states_dtype": "bfloat16",
        "hidden_states_layout": "torch_strided_contiguous",
        "rope_tuple_layout": "torch_strided_contiguous_same_device",
        "gradient_enabled_allowed": False,
        "clean_zero_returns_original_tuple": True,
        "active_returns_new_tuple_without_mutating_original": True,
        "exception_or_exit_restores_original_forward_and_scope_state": True,
        "record_requires_clean_body_and_completed_cleanup": True,
        "body_or_cleanup_exception_permanently_rejects_record": True,
    },
    "cfg_branch_contract": {
        "official_pipeline_order": ["conditional", "unconditional"],
        "guidance_scale_decimal": format(CFG_GUIDANCE_SCALE, ".1f"),
        "combination_formula": (
            "cfg_velocity=unconditional+guidance_scale*"
            "(conditional-unconditional)_all_float32"
        ),
        "same_relation_control_on_conditional_and_unconditional": True,
        "base_control_coefficient": 0,
        "input_state_timestep_and_probe_binding_must_match": True,
        "single_branch_control_allowed": False,
    },
    "scheduler_state_update_measurement_contract": {
        "scheduler_velocity_shape": [1, 16, 9, 40, 64],
        "branch_velocity_dtype_after_transformer_cast": (
            "little_endian_float32"
        ),
        "base_cfg_velocity": (
            "float32_cfg_combine_base_conditional_and_unconditional"
        ),
        "controlled_cfg_velocity": (
            "float32_cfg_combine_controlled_conditional_and_unconditional"
        ),
        "intended_delta_velocity": (
            "controlled_cfg_velocity_minus_base_cfg_velocity_float32"
        ),
        "constrained_velocity": (
            "base_cfg_velocity_plus_intended_delta_velocity_float32"
        ),
        "actual_delta_velocity": (
            "constrained_velocity_minus_base_cfg_velocity_float32"
        ),
        "delta_sigma_requirement": (
            "future_governed_flow_scheduler_input_canonicalized_to_float32_"
            "must_be_finite_and_negative"
        ),
        "actual_state_update_delta": (
            "float32_delta_sigma_times_actual_delta_velocity"
        ),
        "lambda_max_decimal": format(LAMBDA_MAX, ".2f"),
        "velocity_norm_ratio_budget_decimal": format(
            VELOCITY_NORM_RATIO_BUDGET,
            ".2f",
        ),
        "flow_energy_budget_ratio_decimal": format(
            FLOW_ENERGY_BUDGET_RATIO,
            ".6f",
        ),
        "norm_budget": (
            "float32_l2_base_cfg_velocity_times_velocity_norm_ratio_budget_"
            "times_lambda_max"
        ),
        "schedule_energy_context": (
            "future_governed_inputs_cumulative_reference_energy_cumulative_"
            "control_energy_and_positive_remaining_step_count"
        ),
        "reference_energy_increment": (
            "delta_sigma_squared_times_base_cfg_velocity_l2_squared"
        ),
        "projected_reference_energy": (
            "cumulative_reference_energy_plus_reference_energy_increment_"
            "times_remaining_step_count"
        ),
        "total_flow_energy_budget": (
            "flow_energy_budget_ratio_times_projected_reference_energy"
        ),
        "remaining_flow_energy": (
            "maximum_zero_total_flow_energy_budget_minus_cumulative_control_"
            "energy"
        ),
        "energy_increment": (
            "delta_sigma_squared_times_actual_delta_velocity_l2_squared"
        ),
        "direction_cosine": (
            "safe_cosine_actual_state_update_delta_vs_delta_sigma_times_"
            "intended_delta_velocity"
        ),
        "minimum_direction_cosine_decimal": format(
            MINIMUM_DIRECTION_COSINE,
            ".3f",
        ),
        "signed_state_update_exposure": (
            "signed_coefficient_times_absolute_delta_sigma_times_"
            "actual_delta_velocity_l2"
        ),
        "active_zero_actual_delta_allowed": False,
        "clean_exact_noop_required": True,
        "local_measurement_is_execution_evidence": False,
    },
    "attention_bias_fallback_allowed": False,
    "direct_additive_velocity_or_latent_carrier_allowed": False,
}
EXPECTED_STATISTICS_AND_TRANSFER_FORMULA_CONTRACT = {
    "numeric_layout": (
        "arrays=float64_le_c_contiguous_shape_11x6;"
        "vectors=reshape_order_C_before_norm_and_dot"
    ),
    "whitened_feature": (
        "(raw_feature-frozen_c0_center)/frozen_c0_scale_elementwise"
    ),
    "identity_intercept": "0.5*(whitened_clean_a+whitened_clean_b)",
    "positive_centered_delta": "whitened_positive-clean_intercept",
    "negative_centered_delta": "whitened_negative-clean_intercept",
    "observed_odd": "0.5*(positive_delta-negative_delta)",
    "observed_common": "0.5*(positive_delta+negative_delta)",
    "clean_noise_norm": (
        "max(l2(flatten_C(whitened_clean_a-whitened_clean_b)),1e-6)"
    ),
    "odd_norm": "l2(flatten_C(observed_odd))",
    "common_norm": "l2(flatten_C(observed_common))",
    "antisymmetry_cosine": (
        "safe_cosine(flatten_C(positive_delta),flatten_C(-negative_delta));"
        "zero_norm=0;reject_if_abs_raw_gt_1_plus_1e-12;"
        "clamp_to_closed_minus1_plus1"
    ),
    "antisymmetry_residual": (
        "l2(flatten_C(positive_delta+negative_delta))/"
        "max(l2(flatten_C(positive_delta))+"
        "l2(flatten_C(negative_delta)),1e-12)"
    ),
    "common_odd_ratio": "common_norm/max(odd_norm,1e-12)",
    "odd_clean_noise_ratio": "odd_norm/clean_noise_norm",
    "c0_transfer": (
        "(whitened_positive-whitened_negative)/"
        "(positive_exposure-negative_exposure)_elementwise"
    ),
    "gate0_predicted_odd": (
        "T_rel*0.5*(positive_exposure-negative_exposure)"
    ),
    "transfer_direction_cosine": (
        "safe_cosine(flatten_C(predicted_odd),flatten_C(observed_odd));"
        "zero_norm=0;reject_if_abs_raw_gt_1_plus_1e-12;"
        "clamp_to_closed_minus1_plus1"
    ),
    "transfer_relative_error": (
        "l2(flatten_C(observed_odd-predicted_odd))/"
        "max(l2(flatten_C(observed_odd)),1e-12)"
    ),
}


@dataclass(frozen=True)
class PatchRelationProbePlanRecord:
    plan_index: int
    identity_role: str
    identity_placeholder: str
    probe_id: str
    probe_role: str
    signed_state_coefficient: int
    relation_id: str
    feature_schema_id: str
    execution_authorized: bool = False
    formal_result: bool = False
    stage_progression_allowed: bool = False


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"非有限 JSON 常量被禁止: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON object 存在重复 key: {key}")
        result[key] = value
    return result


def _strict_json_loads(raw: str) -> Any:
    return json.loads(
        raw,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_json_constant,
    )


def _require_canonical_value(value: Any, label: str) -> None:
    if isinstance(value, str) or type(value) is bool or type(value) is int:
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _require_canonical_value(item, f"{label}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{label} key 必须是字符串")
            _require_canonical_value(item, f"{label}.{key}")
        return
    raise TypeError(
        f"{label} 只允许 string/integer/boolean/array/object，"
        f"observed={type(value).__name__}"
    )


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    _require_canonical_value(value, "protocol_contract")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def protocol_digest(protocol_contract: Mapping[str, Any]) -> str:
    return sha256(canonical_json_bytes(protocol_contract)).hexdigest()


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    label: str,
) -> None:
    observed = set(value)
    if observed != expected:
        raise ValueError(
            f"{label} keys 不匹配: missing={sorted(expected-observed)}, "
            f"extra={sorted(observed-expected)}"
        )


def load_patch_relation_gate0_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    config_path = Path(path)
    config = _strict_json_loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("config 顶层必须为 object")
    _require_exact_keys(config, EXPECTED_TOP_LEVEL_KEYS, "config")
    if config["profile_id"] != PROFILE_ID:
        raise ValueError("profile_id 不匹配")
    if config["method_id"] != METHOD_ID:
        raise ValueError("method_id 不匹配")
    if config["contract_state"] != CONTRACT_STATE:
        raise ValueError("contract_state 不匹配")
    if config["authorization_boundary"] != EXPECTED_AUTHORIZATION_BOUNDARY:
        raise ValueError("authorization boundary 必须保持执行与证据授权关闭")
    contract = config["protocol_contract"]
    if not isinstance(contract, dict):
        raise ValueError("protocol_contract 必须为 object")
    _require_exact_keys(contract, EXPECTED_PROTOCOL_SECTIONS, "protocol_contract")
    observed_digest = protocol_digest(contract)
    if config["protocol_digest"] != observed_digest:
        raise ValueError("protocol digest 与当前 protocol_contract 不匹配")
    _validate_contract_semantics(contract)
    if observed_digest != FROZEN_PROTOCOL_DIGEST:
        raise ValueError("protocol digest 与冻结合同常量不匹配")
    if observed_digest != RUNTIME_ADAPTER_PROTOCOL_DIGEST:
        raise ValueError("protocol digest 与 runtime adapter 常量不匹配")
    return config


def _validate_contract_semantics(contract: Mapping[str, Any]) -> None:
    historical = contract["historical_failure_boundary"]
    if historical != {
        "source_commit": "346d6c97bbfdef5d3ff0e61ebb6f69b1e7b6cea3",
        "decision": "gate0_fail_stop_current_carrier_or_feature",
        "failed_combination": (
            "decoder_jacobian_additive_atom_plus_local_rgb_mean_feature_"
            "plus_heldout_transfer"
        ),
        "patch_relation_embedding_evaluated": False,
        "observer_or_state_space_synchronization_evaluated": False,
    }:
        raise ValueError("historical FAIL boundary 不匹配")

    runtime = contract["wan_runtime_adapter_contract"]
    if runtime != EXPECTED_WAN_RUNTIME_ADAPTER_CONTRACT:
        raise ValueError("Wan RoPE adapter boundary 不匹配")

    relation = contract["token_relation_contract"]
    if (
        relation["token_grid_shape"] != [9, 20, 32]
        or relation["video_window_frame_indices"] != list(range(11, 22))
        or relation["latent_token_time_indices"] != [3, 4, 5]
        or relation["relation_id"] != RELATION_DESCRIPTOR_ID
        or relation["public_relation_descriptor_digest"]
        != build_public_patch_relation_descriptor().descriptor_digest
        or relation["patch_a_token_coordinate_row_column"] != [9, 13]
        or relation["patch_b_token_coordinate_row_column"] != [9, 18]
        or relation["pair_coefficients"] != [1, -1]
        or relation["zero_sum_per_active_token_time"] is not True
        or relation["candidate_key_independent_public_dictionary"] is not True
    ):
        raise ValueError("Patch relation layout 不匹配")

    phase = contract["rope_phase_contract"]
    if (
        phase["attention_head_dim"] != 128
        or phase["temporal_rope_dimension"] != 44
        or phase["temporal_rope_pair_index"] != 0
        or phase["phase_budget_radians_decimal"]
        != format(PHASE_BUDGET_RADIANS, ".6f")
        or phase["phase_pair_entries"] != [0, 1]
        or phase["runtime_strength_sweep_allowed"] is not False
        or phase["runtime_relation_selection_allowed"] is not False
    ):
        raise ValueError("RoPE phase contract 不匹配")

    feature = contract["output_feature_contract"]
    if (
        feature["feature_schema_id"] != FEATURE_SCHEMA_ID
        or feature["source_shape"] != [33, 320, 512, 3]
        or feature["output_shape"] != [11, 6]
        or feature["feature_component_order"]
        != "horizontal_dct_0_1_rgb_then_vertical_dct_1_0_rgb"
        or feature["time_axis_preserved"] is not True
        or feature["whole_video_time_mean_allowed"] is not False
        or feature["per_video_l2_normalization_allowed"] is not False
        or feature["four_by_four_rgb_mean_primary_allowed"] is not False
        or feature["candidate_key_independent"] is not True
    ):
        raise ValueError("output Patch-relation feature contract 不匹配")

    derivation = contract["coefficient_derivation_contract"]
    if (
        derivation["domain"] != "sstw.patch_relation_gate0.coefficient.v1"
        or derivation["algorithm"] != "HMAC_SHA256"
        or derivation["public_context_schema_id"]
        != "sstw_frame_state_public_context_v1"
        or derivation["context_digest_rule"]
        != "sha256_exact_canonical_public_context_record"
        or derivation["wrong_key_gate_authorized"] is not False
    ):
        raise ValueError("key/context coefficient derivation contract 不匹配")

    gate = contract["construction_and_gate_contract"]
    c0_placeholder = gate["construction_identity_c0_placeholder"]
    a_placeholder = gate["gate0_identity_a_placeholder"]
    if (
        not c0_placeholder.endswith("execution_forbidden")
        or not a_placeholder.endswith("execution_forbidden")
        or c0_placeholder == a_placeholder
    ):
        raise ValueError("C0/A placeholder identity 必须区分")
    if gate["probe_roles_per_identity"] != list(EXPECTED_PROBE_ROLES):
        raise ValueError("probe roles 不匹配")
    if (
        gate["identity_backflow_allowed"] is not False
        or gate["c0_gate0_evidence_allowed"] is not False
        or gate["gate0_refit_or_reselection_allowed"] is not False
        or gate["automatic_next_execution_allowed"] is not False
    ):
        raise ValueError("C0/A apply-only boundary 被放宽")
    if gate["actual_signed_exposure_input"] != EXPECTED_ACTUAL_SIGNED_EXPOSURE_INPUT:
        raise ValueError("actual signed exposure 证据边界不匹配")
    if (
        gate["statistics_and_transfer_formula_contract"]
        != EXPECTED_STATISTICS_AND_TRANSFER_FORMULA_CONTRACT
    ):
        raise ValueError("statistics/transfer exact formula contract 不匹配")
    if gate["signed_gate_thresholds"] != {
        "minimum_antisymmetry_cosine_decimal": "0.9",
        "maximum_antisymmetry_residual_decimal": "0.25",
        "maximum_common_odd_ratio_decimal": "0.5",
        "minimum_odd_clean_noise_ratio_decimal": "3.0",
    }:
        raise ValueError("signed gate thresholds 不匹配")
    if gate["transfer_gate_thresholds"] != {
        "minimum_direction_cosine_decimal": "0.9",
        "maximum_relative_error_decimal": "0.5",
    }:
        raise ValueError("transfer gate thresholds 不匹配")

    method_scope = contract["method_scope"]
    if method_scope != {
        "payload_prc_state_dynamics_implemented": False,
        "local_wan_rope_runtime_adapter_implemented": True,
        "local_cfg_state_update_measurement_adapter_implemented": True,
        "flow_velocity_deflection_runtime_implemented": False,
        "output_encoder_runtime_adapter_implemented": False,
        "clock_path_implemented": False,
        "state_observer_implemented": False,
        "wrong_key_detection_implemented": False,
        "attack_or_fixed_fpr_implemented": False,
        "checked_in_outputs_allowed": False,
    }:
        raise ValueError("method scope 必须只开放本地 adapter 实现状态")
    build_public_patch_relation_descriptor()


def build_patch_relation_gate0_plan(
    config: Mapping[str, Any],
) -> list[PatchRelationProbePlanRecord]:
    _require_exact_keys(config, EXPECTED_TOP_LEVEL_KEYS, "plan config")
    if (
        config.get("profile_id") != PROFILE_ID
        or config.get("method_id") != METHOD_ID
        or config.get("contract_state") != CONTRACT_STATE
        or config.get("authorization_boundary") != EXPECTED_AUTHORIZATION_BOUNDARY
    ):
        raise ValueError("plan 必须消费已验证冻结 config identity/boundary")
    contract = config.get("protocol_contract")
    if (
        not isinstance(contract, Mapping)
        or protocol_digest(contract) != FROZEN_PROTOCOL_DIGEST
        or config.get("protocol_digest") != FROZEN_PROTOCOL_DIGEST
    ):
        raise ValueError("plan 必须消费已验证冻结 config")
    _require_exact_keys(contract, EXPECTED_PROTOCOL_SECTIONS, "plan protocol_contract")
    _validate_contract_semantics(contract)
    gate = config["protocol_contract"]["construction_and_gate_contract"]
    identities = (
        ("construction_c0", gate["construction_identity_c0_placeholder"]),
        ("gate0_identity_a", gate["gate0_identity_a_placeholder"]),
    )
    coefficients = {
        "clean_a": 0,
        "clean_b": 0,
        "positive": 1,
        "negative": -1,
    }
    records: list[PatchRelationProbePlanRecord] = []
    for identity_role, placeholder in identities:
        for probe_role in EXPECTED_PROBE_ROLES:
            records.append(
                PatchRelationProbePlanRecord(
                    plan_index=len(records),
                    identity_role=identity_role,
                    identity_placeholder=placeholder,
                    probe_id=f"{identity_role}_{probe_role}",
                    probe_role=probe_role,
                    signed_state_coefficient=coefficients[probe_role],
                    relation_id=RELATION_DESCRIPTOR_ID,
                    feature_schema_id=FEATURE_SCHEMA_ID,
                )
            )
    return records
