"""Fail-closed contract for the first Patch-relation Gate 0 execution.

This module validates one exact design/runtime config and constructs the
predeclared C0/A probe plan.  The runner is implemented and may execute only
after an explicit user Colab run; results remain non-formal and cannot advance
the stage or authorize an observer, attack, fixed-FPR evaluation, or paper
claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np

from main.methods.state_space_watermark.patch_relation_carrier import (
    FEATURE_SCHEMA_ID,
    PHASE_BUDGET_RADIANS,
    RELATION_DESCRIPTOR_ID,
    build_public_patch_relation_descriptor,
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
CONTRACT_STATE = (
    "patch_relation_gate0_runner_implemented_pending_user_colab_run"
)
FROZEN_PROTOCOL_DIGEST = (
    "454e380c2900b9bd989ff8f95c3c0563545037650331f941b33eee650c0a0ddc"
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
    "runtime_implementation_authorized": True,
    "construction_execution_allowed": True,
    "gpu_execution_allowed": True,
    "colab_execution_allowed": True,
    "runner_implementation_allowed": True,
    "notebook_handler_implementation_allowed": False,
    "drive_update_allowed": False,
    "observer_implementation_allowed": False,
    "attack_execution_allowed": False,
    "fixed_fpr_execution_allowed": False,
    "baseline_execution_allowed": False,
    "paper_claim_allowed": False,
    "formal_result": False,
    "stage_progression_allowed": False,
    "claim_support_status": (
        "gate0_execution_pending_user_colab_run_not_method_evidence"
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
    "execution_identity_contract",
    "gate0_runtime_execution_contract",
    "method_scope",
}
EXPECTED_PROBE_ROLES = ("clean_a", "clean_b", "positive", "negative")
EXPECTED_ACTUAL_SIGNED_EXPOSURE_INPUT = (
    "governed_runner_realized_scheduler_next_state_difference_with_exact_"
    "frozen_schedule_timestep_and_internal_index_binding"
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
    "runtime_adapter_execution_allowed": True,
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
        "transformer_branch_output_cast_before_pipeline_cfg": (
            "base_and_controlled_branch_velocity_explicit_float32_before_"
            "official_pipeline_cfg"
        ),
        "same_relation_control_on_conditional_and_unconditional": True,
        "base_control_coefficient": 0,
        "input_state_timestep_and_probe_binding_must_match": True,
        "single_branch_control_allowed": False,
    },
    "scheduler_state_update_measurement_contract": {
        "scheduler_velocity_shape": [1, 16, 9, 40, 64],
        "branch_velocity_dtype_after_transformer_cast": (
            "real_transformer_bfloat16_output_explicitly_cast_to_float32_"
            "before_pipeline_cfg"
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
            "controlled_cfg_velocity_exact_scheduler_model_output_float32"
        ),
        "actual_delta_velocity": (
            "constrained_velocity_minus_base_cfg_velocity_float32"
        ),
        "delta_sigma_requirement": (
            "governed_runner_exact_frozen_flow_scheduler_input_"
            "canonicalized_to_float32_must_be_finite_and_negative"
        ),
        "scheduler_sample": (
            "actual_scheduler_input_sample_cast_to_float32_by_official_"
            "FlowMatchEuler_step"
        ),
        "base_next_state": (
            "counterfactual_float32_sample_plus_delta_sigma_times_base_cfg_"
            "velocity_same_frozen_euler_semantics"
        ),
        "controlled_next_state": (
            "actual_scheduler_returned_prev_sample_float32_bound_to_"
            "controlled_cfg_velocity"
        ),
        "actual_state_update_delta": (
            "controlled_next_state_minus_base_next_state_float32_after_"
            "scheduler_output_cast"
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
            "governed_runner_recomputed_cumulative_reference_energy_"
            "cumulative_control_energy_and_exact_remaining_step_count"
        ),
        "reference_energy_increment": (
            "actual_base_next_state_minus_scheduler_sample_l2_squared"
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
            "actual_state_update_delta_l2_squared"
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
            "signed_coefficient_times_actual_state_update_delta_l2"
        ),
        "active_zero_actual_delta_allowed": False,
        "clean_exact_noop_required": True,
        "local_measurement_is_execution_evidence": False,
        "governed_step_record_requires_runner_schedule_and_scheduler_"
        "consumption_revalidation": True,
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
EXPECTED_EXECUTION_IDENTITY_CONTRACT = {
    "execution_common": {
        "model_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "model_revision": "0fad780a534b6463e45facd96134c9f345acfa5b",
        "diffusers_version": "0.35.2",
        "scheduler_id": "wan21_flow_match_euler_discrete_scheduler_shift_3",
        "scheduler_signature": (
            "FlowMatchEulerDiscreteScheduler:"
            "a63b40d76d729371591d03526e14d24359c732866c07f51e4cc5f918f4941d2b"
        ),
        "flow_shift_decimal": "3.0",
        "num_inference_steps": 8,
        "num_frames": 33,
        "height": 320,
        "width": 512,
        "guidance_scale_decimal": "5.0",
        "fps": 8,
        "video_encoder_backend": "imageio_ffmpeg",
        "video_exporter": "diffusers.utils.export_to_video",
    },
    "construction_identity_c0": {
        "identity_id": "patch_relation_construction_identity_c0",
        "prompt_id": "probe_paper_paper_master_prompt_005",
        "prompt_text": (
            "A large blue flag sweeps from the lower left to the upper right "
            "across most of the image, fixed camera, wide amplitude motion, "
            "high contrast indoor background."
        ),
        "prompt_text_sha256": (
            "3829af00523428c3013ee2393830d2311b664c4eb7c2e3b932982430b99d23bd"
        ),
        "negative_prompt_text": (
            "static image, frozen cloth, tiny flag, weak motion, blurry, "
            "jittery, distorted"
        ),
        "negative_prompt_text_sha256": (
            "32d7fd6487d8da472af022a61ed114ec11278a6100f04a997df2129024257cc0"
        ),
        "seed_id": "probe_paper_paper_master_calibration_seed_03",
        "seed_value": 1275,
        "initial_noise_rule": (
            "new_torch_generator_manual_seed_per_probe_same_seed_within_"
            "identity"
        ),
    },
    "gate0_identity_a": {
        "identity_id": "patch_relation_gate0_identity_a",
        "prompt_id": "probe_paper_paper_master_prompt_006",
        "prompt_text": (
            "A red toy train moves along a circular track around a yellow "
            "tower, fixed camera, the train occupies a large foreground area "
            "and visibly changes position every frame."
        ),
        "prompt_text_sha256": (
            "2323d9045f96945cec4a935aea41bbf950bcb2257f44f1540386089ceeeb99c1"
        ),
        "negative_prompt_text": (
            "static image, frozen train, subtle motion, tiny object, blurry, "
            "jittery, distorted"
        ),
        "negative_prompt_text_sha256": (
            "528c33a7ed958ca70add888f83232410f820e532d0053a80151202195b6940ab"
        ),
        "seed_id": "probe_paper_paper_master_test_seed_03",
        "seed_value": 2275,
        "initial_noise_rule": (
            "new_torch_generator_manual_seed_per_probe_same_seed_within_"
            "identity"
        ),
    },
    "identities_must_be_distinct": True,
    "same_initial_noise_within_identity": True,
    "identity_a_record_backflow_allowed": False,
    "historical_decoder_jacobian_result_selection_backflow_allowed": False,
}
EXPECTED_GATE0_RUNTIME_EXECUTION_CONTRACT = {
    "test_id": "patch_relation_gate0_construction",
    "phase": "gate0",
    "probe_order": [
        "construction_c0_clean_a",
        "construction_c0_clean_b",
        "construction_c0_positive",
        "construction_c0_negative",
        "gate0_identity_a_clean_a",
        "gate0_identity_a_clean_b",
        "gate0_identity_a_positive",
        "gate0_identity_a_negative",
    ],
    "signed_coefficients": [0, 0, 1, -1, 0, 0, 1, -1],
    "sigma_grid_decimal": [
        "1.0",
        "0.9475425481796265",
        "0.8827877640724182",
        "0.8008373379707336",
        "0.6937931180000305",
        "0.5480455756187439",
        "0.33797216415405273",
        "0.008928571827709675",
        "0.0",
    ],
    "delta_sigma_by_step_decimal": [
        "-0.052457451820373535",
        "-0.06475478410720825",
        "-0.08195042610168457",
        "-0.10704421997070312",
        "-0.14574754238128662",
        "-0.21007341146469116",
        "-0.32904359698295593",
        "-0.008928571827709675",
    ],
    "timestep_by_step_decimal": [
        "1000.0",
        "947.5425415039062",
        "882.7877807617188",
        "800.8373413085938",
        "693.7930908203125",
        "548.0455932617188",
        "337.97216796875",
        "8.928571701049805",
    ],
    "transformer_external_branch_order_per_step": [
        "conditional",
        "unconditional",
    ],
    "base_then_controlled_forward_per_branch": True,
    "transformer_forward_count_per_step": 4,
    "transformer_cache_must_be_disabled": True,
    "base_and_controlled_share_exact_hidden_timestep_encoder_input_per_"
    "branch": True,
    "scheduler_consumes_controlled_cfg_velocity_only": True,
    "transformer_bfloat16_output_cast_to_float32_before_pipeline_cfg": True,
    "scheduler_model_output_revalidated_against_measured_controlled_cfg": True,
    "scheduler_returned_controlled_next_state_and_counterfactual_base_next_"
    "state_revalidated": True,
    "scheduler_timestep_exact_frozen_row_required": True,
    "scheduler_internal_step_index_before_after_progression_required": True,
    "cuda_bfloat16_capability_checked_before_model_load": True,
    "cuda_native_bfloat16_check_including_emulation_false_required": True,
    "minimum_cuda_compute_capability_major": 8,
    "selected_pipeline_dtype_exact_torch_bfloat16_required": True,
    "remaining_step_count_formula": "8_minus_step_index",
    "cumulative_reference_and_control_energy_recomputed_in_step_order": True,
    "step_input_binding_digest": (
        "sha256_canonical_probe_identity_step_hidden_timestep_and_branch_"
        "encoder_tensor_signatures"
    ),
    "clean_probe_runs_same_base_and_controlled_forward_path": True,
    "clean_exact_noop_required": True,
    "output_video_shape_rgb24": [33, 320, 512, 3],
    "output_feature_shape": [11, 6],
    "expected_generation_record_count": 8,
    "expected_step_record_count": 64,
    "expected_feature_record_count": 8,
    "c0_fits_whitening_and_T_rel_only": True,
    "identity_a_apply_only": True,
    "method_gate_failure_is_normal_nonformal_stop": True,
    "runtime_or_contract_failure_is_recovery_only": True,
    "successful_run_published_after_local_completion_as_single_zip_and_"
    "manifest": True,
}


@dataclass(frozen=True)
class PatchRelationProbePlanRecord:
    plan_index: int
    identity_role: str
    identity_id: str
    probe_id: str
    probe_role: str
    signed_state_coefficient: int
    relation_id: str
    feature_schema_id: str
    execution_authorized: bool = True
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
    if (
        gate["construction_identity_c0"]
        != "patch_relation_construction_identity_c0"
        or gate["gate0_identity_a"] != "patch_relation_gate0_identity_a"
        or gate["construction_identity_c0"] == gate["gate0_identity_a"]
    ):
        raise ValueError("C0/A execution identity 必须冻结且区分")
    if gate["probe_roles_per_identity"] != list(EXPECTED_PROBE_ROLES):
        raise ValueError("probe roles 不匹配")
    if (
        gate["identity_backflow_allowed"] is not False
        or gate["c0_gate0_evidence_allowed"] is not True
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

    identities = contract["execution_identity_contract"]
    if identities != EXPECTED_EXECUTION_IDENTITY_CONTRACT:
        raise ValueError("Patch-relation execution identity contract 不匹配")
    for role in ("construction_identity_c0", "gate0_identity_a"):
        identity = identities[role]
        if sha256(identity["prompt_text"].encode("utf-8")).hexdigest() != (
            identity["prompt_text_sha256"]
        ):
            raise ValueError(f"{role} prompt text digest 不匹配")
        if sha256(
            identity["negative_prompt_text"].encode("utf-8")
        ).hexdigest() != identity["negative_prompt_text_sha256"]:
            raise ValueError(f"{role} negative prompt digest 不匹配")
    if (
        identities["construction_identity_c0"]["prompt_id"]
        == identities["gate0_identity_a"]["prompt_id"]
        or identities["construction_identity_c0"]["seed_value"]
        == identities["gate0_identity_a"]["seed_value"]
    ):
        raise ValueError("C0/A prompt 与 seed 必须同时隔离")

    runtime_execution = contract["gate0_runtime_execution_contract"]
    if runtime_execution != EXPECTED_GATE0_RUNTIME_EXECUTION_CONTRACT:
        raise ValueError("Patch-relation governed runner contract 不匹配")
    sigma = tuple(
        float(value) for value in runtime_execution["sigma_grid_decimal"]
    )
    deltas = tuple(
        float(value)
        for value in runtime_execution["delta_sigma_by_step_decimal"]
    )
    timesteps = tuple(
        float(value)
        for value in runtime_execution["timestep_by_step_decimal"]
    )
    if len(sigma) != 9 or len(deltas) != 8 or len(timesteps) != 8:
        raise ValueError("Gate0 frozen Flow schedule 长度不匹配")
    if any(
        float(np.float32(sigma[index + 1] - sigma[index]))
        != float(np.float32(deltas[index]))
        for index in range(8)
    ):
        raise ValueError("Gate0 frozen delta_sigma 与 sigma grid 不一致")
    if any(
        float(np.float32(sigma[index] * 1000.0))
        != float(np.float32(timesteps[index]))
        for index in range(8)
    ):
        raise ValueError("Gate0 frozen timestep 与 sigma row 不一致")

    method_scope = contract["method_scope"]
    if method_scope != {
        "payload_prc_state_dynamics_implemented": False,
        "local_wan_rope_runtime_adapter_implemented": True,
        "local_cfg_state_update_measurement_adapter_implemented": True,
        "flow_velocity_deflection_runtime_implemented": True,
        "output_encoder_runtime_adapter_implemented": True,
        "gate0_runner_implemented": True,
        "clock_path_implemented": False,
        "state_observer_implemented": False,
        "wrong_key_detection_implemented": False,
        "attack_or_fixed_fpr_implemented": False,
        "checked_in_outputs_allowed": False,
    }:
        raise ValueError("method scope 必须精确开放 Gate0 runner 实现状态")
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
    execution_identities = config["protocol_contract"][
        "execution_identity_contract"
    ]
    identities = (
        (
            "construction_c0",
            execution_identities["construction_identity_c0"]["identity_id"],
        ),
        (
            "gate0_identity_a",
            execution_identities["gate0_identity_a"]["identity_id"],
        ),
    )
    coefficients = {
        "clean_a": 0,
        "clean_b": 0,
        "positive": 1,
        "negative": -1,
    }
    records: list[PatchRelationProbePlanRecord] = []
    for identity_role, identity_id in identities:
        for probe_role in EXPECTED_PROBE_ROLES:
            records.append(
                PatchRelationProbePlanRecord(
                    plan_index=len(records),
                    identity_role=identity_role,
                    identity_id=identity_id,
                    probe_id=f"{identity_role}_{probe_role}",
                    probe_role=probe_role,
                    signed_state_coefficient=coefficients[probe_role],
                    relation_id=RELATION_DESCRIPTOR_ID,
                    feature_schema_id=FEATURE_SCHEMA_ID,
                )
            )
    return records
