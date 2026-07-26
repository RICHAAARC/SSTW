"""Static contract primitives for frame-state signed observability.

This module validates the frozen local-primitives execution contract.  It does
not implement a model, GPU runner, Colab handler, observer, or stage transition.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


DEFAULT_CONFIG_PATH = (
    "configs/protocol/"
    "sstw_frame_state_signed_observability_construction.json"
)
PROFILE_ID = "sstw_frame_state_signed_observability_construction"
METHOD_ID = "frame_state_synchronized_generative_flow_video_watermark"
CONTRACT_STATE = (
    "gate0_runner_implemented_execution_pending_explicit_user_colab_run"
)
PUBLIC_CONTEXT_SCHEMA_ID = "sstw_frame_state_public_context_v1"
FROZEN_PROTOCOL_DIGEST = (
    "31a3332a05ffeea853ce43495ca100c8fa065f96d7fd53aaf6cb0aeac6c00af4"
)

_LOWER_HEX_32 = re.compile(r"[0-9a-f]{32}\Z")
_LOWER_HEX_64 = re.compile(r"[0-9a-f]{64}\Z")

EXPECTED_AUTHORIZATION_BOUNDARY = {
    "attack_execution_allowed": False,
    "baseline_execution_allowed": False,
    "claim_support_status": (
        "gate0_execution_pending_user_colab_run_not_method_evidence"
    ),
    "colab_execution_allowed": True,
    "construction_execution_allowed": True,
    "drive_update_allowed": False,
    "fixed_fpr_execution_allowed": False,
    "formal_result": False,
    "gpu_execution_allowed": True,
    "notebook_handler_implementation_allowed": False,
    "observer_implementation_allowed": False,
    "paper_claim_allowed": False,
    "runner_implementation_allowed": True,
    "runtime_implementation_authorized": True,
    "stage_progression_allowed": False,
}

EXPECTED_EXECUTION_IDENTITY_CONTRACT = {
    "construction_identity": {
        "guidance_scale_denominator": 1,
        "guidance_scale_numerator": 5,
        "initial_noise_rule":
            "new_torch_generator_manual_seed_per_probe_same_seed_within_identity",
        "negative_prompt_text":
            "static image, frozen conveyor, subtle motion, blurry, jittery, distorted",
        "negative_prompt_text_sha256":
            "798a65de2dd61dffee2b6d5229d1167e4c0aa7053948562ae53dff3d1c0d0d11",
        "prompt_id": "probe_paper_paper_master_prompt_003",
        "prompt_text": (
            "Three brightly colored boxes travel on a conveyor belt from left "
            "to right across the entire frame, fixed camera, each box remains "
            "large and easy to track."
        ),
        "prompt_text_sha256":
            "c4f3a636c9c4393ebf98448f2c30c6648f7e9141a2886bac0cd950001ec03980",
        "seed_id": "frame_state_construction_seed_2201",
        "seed_value": 2201,
    },
    "execution_common": {
        "diffusers_version": "0.35.2",
        "model_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "model_revision": "0fad780a534b6463e45facd96134c9f345acfa5b",
        "scheduler_id": "wan21_flow_match_euler_discrete_scheduler_shift_3",
        "scheduler_signature": (
            "FlowMatchEulerDiscreteScheduler:"
            "a63b40d76d729371591d03526e14d24359c732866c07f51e4cc5f918f4941d2b"
        ),
        "video_encoder_backend": "imageio_ffmpeg",
        "video_exporter": "diffusers.utils.export_to_video",
    },
    "identity_record_backflow_allowed": False,
    "signed_observability_identity": {
        "guidance_scale_denominator": 1,
        "guidance_scale_numerator": 5,
        "initial_noise_rule":
            "new_torch_generator_manual_seed_per_probe_same_seed_within_identity",
        "negative_prompt_text":
            "static image, frozen conveyor, subtle motion, blurry, jittery, distorted",
        "negative_prompt_text_sha256":
            "798a65de2dd61dffee2b6d5229d1167e4c0aa7053948562ae53dff3d1c0d0d11",
        "prompt_id": "sstw_frame_state_gate0_prompt_001",
        "prompt_text": (
            "Four red toy cars travel steadily from right to left across a "
            "bright tabletop, fixed camera, each car remains large and easy "
            "to track."
        ),
        "prompt_text_sha256":
            "84c550acb01548c57f64bd0c9bc14e2ddec701cf9ed4c8cbea1603e66c691703",
        "seed_id": "frame_state_signed_observability_seed_2202",
        "seed_value": 2202,
    },
}

EXPECTED_TOP_LEVEL_KEYS = {
    "authorization_boundary",
    "contract_state",
    "execution_identity_contract",
    "method_id",
    "profile_id",
    "protocol_contract",
    "protocol_digest",
}

PUBLIC_CONTEXT_KEYS = {
    "context_schema_id",
    "method_id",
    "protocol_digest",
    "public_nonce_random",
    "state_clock_rate_denominator",
    "state_clock_rate_numerator",
    "state_window_count",
}

EXPECTED_ACTUAL_SIGNED_EXPOSURE_FORMULA = (
    "sum_flow_steps_delta_sigma_times_inner_product_"
    "actual_delta_velocity_and_public_dictionary_atom"
)
EXPECTED_TRANSFER_ESTIMATOR = (
    "positive_negative_center_difference_"
    "div_actual_signed_exposure_difference"
)
COMMON_SIGNED_GATE_IDS = [
    "antisymmetry_cosine",
    "antisymmetry_residual",
    "common_odd_ratio",
    "odd_clean_noise_ratio",
]
PRIMARY_TRANSFER_GATE_IDS = [
    "t0_transfer_direction_cosine",
    "t0_transfer_relative_error",
]
EXPECTED_CHECKPOINT_SPECS = [
    {
        "centering_formula":
            "identity_clean_a_clean_b_arithmetic_mean_in_checkpoint_coordinates",
        "checkpoint_id": "final_latent_carrier_projection",
        "feature_dimension": 1,
        "flatten_order": "scalar",
        "raw_feature_formula": (
            "float64_dot_c_order_final_latent_float32_"
            "with_public_atom_float32"
        ),
        "source_boundary": "final_latent_float32_before_vae_decode",
        "signed_gate_ids": COMMON_SIGNED_GATE_IDS,
        "transfer_gate_ids": [],
        "value_dtype": "float64",
    },
    {
        "centering_formula":
            "identity_clean_a_clean_b_arithmetic_mean_in_checkpoint_coordinates",
        "checkpoint_id": "decoded_local_temporal_feature",
        "feature_dimension": 528,
        "flatten_order":
            "frame_then_cell_row_then_cell_column_then_channel",
        "raw_feature_formula": (
            "float64_four_by_four_cell_means_of_"
            "postprocessed_float32_zero_one_frames"
        ),
        "source_boundary": (
            "vae_decode_then_exact_video_processor_"
            "denormalization_before_video_export"
        ),
        "signed_gate_ids": COMMON_SIGNED_GATE_IDS,
        "transfer_gate_ids": [],
        "value_dtype": "float64",
    },
    {
        "centering_formula":
            "identity_clean_a_clean_b_arithmetic_mean_in_checkpoint_coordinates",
        "checkpoint_id": "saved_video_local_temporal_feature",
        "feature_dimension": 528,
        "flatten_order":
            "frame_then_cell_row_then_cell_column_then_channel",
        "raw_feature_formula": (
            "float64_four_by_four_cell_means_of_saved_rgb24_div_255"
        ),
        "source_boundary": "saved_video_rgb24_readback",
        "signed_gate_ids": COMMON_SIGNED_GATE_IDS,
        "transfer_gate_ids": PRIMARY_TRANSFER_GATE_IDS,
        "value_dtype": "float64",
    },
]


@dataclass(frozen=True)
class FrameStateProbePlanRecord:
    """One immutable design-only probe row."""

    plan_index: int
    identity_role: str
    probe_id: str
    probe_role: str
    signed_state_coefficient: int
    state_window_id: str
    state_dimension_index: int
    same_initial_noise_within_identity: bool
    public_context_binding: str
    execution_authorized: bool
    formal_result: bool
    stage_progression_allowed: bool


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


def _require_canonical_json_value(value: Any, label: str) -> None:
    if isinstance(value, str) or type(value) is bool:
        return
    if type(value) is int:
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _require_canonical_json_value(item, f"{label}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{label} 的 JSON key 必须是字符串")
            _require_canonical_json_value(item, f"{label}.{key}")
        return
    raise TypeError(
        f"{label} 只允许 string/integer/boolean/array/object，"
        f"observed={type(value).__name__}"
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize the frozen no-float canonical JSON subset."""

    _require_canonical_json_value(value, "canonical_json")
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_json_digest(value: Any) -> str:
    return sha256(canonical_json_bytes(value)).hexdigest()


def _read_config(path: Path) -> dict[str, Any]:
    value = _strict_json_loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("frame-state signed-observability config 顶层必须是 object")
    return dict(value)


def _require_exact_mapping(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
    label: str,
) -> None:
    if dict(observed) != dict(expected):
        raise ValueError(f"{label} 必须与冻结合同精确一致")


def _reject_nested_true_execution_authorization(
    value: Any,
    label: str,
) -> None:
    """Reject hidden execution authorization inside the protocol contract."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            child_label = f"{label}.{key}"
            if (
                key.endswith("_allowed") or key.endswith("_authorized")
            ) and item is True:
                raise ValueError(
                    f"{child_label} 不得在 design-only protocol 中为 true"
                )
            _reject_nested_true_execution_authorization(item, child_label)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nested_true_execution_authorization(
                item,
                f"{label}[{index}]",
            )


def validate_frame_state_signed_observability_config(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the exact design-only protocol and non-recursive digest."""

    if set(config) != EXPECTED_TOP_LEVEL_KEYS:
        raise ValueError(
            "frame-state config 顶层字段集合不一致: "
            f"observed={sorted(config)}"
        )
    if config.get("profile_id") != PROFILE_ID:
        raise ValueError("frame-state config profile_id 不匹配")
    if config.get("method_id") != METHOD_ID:
        raise ValueError("frame-state config method_id 不匹配")
    if config.get("contract_state") != CONTRACT_STATE:
        raise ValueError("frame-state config contract_state 不匹配")

    authorization = config.get("authorization_boundary")
    if not isinstance(authorization, Mapping):
        raise TypeError("authorization_boundary 必须是 object")
    _require_exact_mapping(
        authorization,
        EXPECTED_AUTHORIZATION_BOUNDARY,
        "authorization_boundary",
    )

    execution_identity = config.get("execution_identity_contract")
    if not isinstance(execution_identity, Mapping):
        raise TypeError("execution_identity_contract 必须是 object")
    _require_exact_mapping(
        execution_identity,
        EXPECTED_EXECUTION_IDENTITY_CONTRACT,
        "execution_identity_contract",
    )
    construction_execution = execution_identity["construction_identity"]
    signed_execution = execution_identity["signed_observability_identity"]
    if (
        construction_execution["prompt_id"] == signed_execution["prompt_id"]
        or construction_execution["seed_id"] == signed_execution["seed_id"]
        or construction_execution["seed_value"] == signed_execution["seed_value"]
    ):
        raise ValueError("C0/A prompt 与 seed 必须严格不同")
    for identity in (construction_execution, signed_execution):
        if sha256(identity["prompt_text"].encode("utf-8")).hexdigest() != (
            identity["prompt_text_sha256"]
        ):
            raise ValueError("execution identity positive prompt digest 不匹配")
        if sha256(identity["negative_prompt_text"].encode("utf-8")).hexdigest() != (
            identity["negative_prompt_text_sha256"]
        ):
            raise ValueError("execution identity negative prompt digest 不匹配")

    protocol_contract = config.get("protocol_contract")
    if not isinstance(protocol_contract, Mapping):
        raise TypeError("protocol_contract 必须是 object")
    _require_canonical_json_value(protocol_contract, "protocol_contract")
    _reject_nested_true_execution_authorization(
        protocol_contract,
        "protocol_contract",
    )

    protocol_digest = config.get("protocol_digest")
    if not isinstance(protocol_digest, str) or not _LOWER_HEX_64.fullmatch(
        protocol_digest
    ):
        raise ValueError("protocol_digest 必须是64位小写 SHA-256 hex")
    recomputed_digest = canonical_json_digest(dict(protocol_contract))
    if protocol_digest != recomputed_digest:
        raise ValueError("protocol_digest 与 protocol_contract 重算值不一致")
    digest_contract = protocol_contract.get("protocol_digest_contract")
    if not isinstance(digest_contract, Mapping):
        raise TypeError("protocol_digest_contract 必须是 object")
    if digest_contract.get("canonical_projection") != "protocol_contract_only":
        raise ValueError("protocol digest 必须只绑定 protocol_contract")
    excluded = digest_contract.get("excluded_runtime_fields")
    required_exclusions = {
        "protocol_digest",
        "public_context_record",
        "public_nonce_random",
        "context_digest",
        "execution_identity_contract",
        "authorization_boundary",
        "runtime_records",
        "runtime_paths",
        "timestamps",
    }
    if not isinstance(excluded, list) or set(excluded) != required_exclusions:
        raise ValueError("protocol digest runtime exclusion 集合不一致")

    context_contract = protocol_contract.get("public_context_contract")
    if not isinstance(context_contract, Mapping):
        raise TypeError("public_context_contract 必须是 object")
    if set(context_contract.get("exact_key_order_independent_set", ())) != (
        PUBLIC_CONTEXT_KEYS
    ):
        raise ValueError("public context exact key set 不一致")
    if context_contract.get("state_window_count") != 1:
        raise ValueError("Gate 0 state_window_count 必须为1")

    dictionary = protocol_contract.get("carrier_dictionary_contract")
    feature = protocol_contract.get("local_output_feature_contract")
    if not isinstance(dictionary, Mapping) or not isinstance(feature, Mapping):
        raise TypeError("dictionary/feature contract 必须是 object")
    if dictionary.get("construction_method") != (
        "public_wan_vae_decoder_local_jacobian_fixed_eight_iteration_"
        "temporally_weighted_direction"
    ):
        raise ValueError("public dictionary construction method 不匹配")
    if dictionary.get("construction_direction_claim") != (
        "fixed_eight_iteration_temporally_weighted_decoder_jacobian_"
        "aligned_not_exact_leading_singular"
    ):
        raise ValueError("public dictionary direction claim 不准确")
    if dictionary.get("dictionary_artifact_array_key") != (
        "frame_state_public_atom"
    ):
        raise ValueError("public dictionary NPZ array key 不匹配")
    if dictionary.get("dictionary_artifact_array_count") != 1:
        raise ValueError("public dictionary NPZ 必须只含一个 array")
    if dictionary.get("atom_sign_tie_break_rule") != (
        "lowest_c_order_flat_index"
    ):
        raise ValueError("public dictionary atom sign tie-break 不匹配")
    if dictionary.get("dictionary_artifact_array_shape") != (
        dictionary.get("latent_layout")
    ):
        raise ValueError("public dictionary shape 必须等于冻结 latent layout")
    if dictionary.get(
        "dictionary_zero_outside_restricted_latent_time_indices"
    ) is not True:
        raise ValueError("public dictionary 必须在受限 latent-time 外严格为零")
    if dictionary.get(
        "unit_norm_float32_absolute_tolerance_millionths"
    ) != 20:
        raise ValueError("public dictionary float32 unit-norm tolerance 不匹配")
    if dictionary.get("unit_norm_required") is not True:
        raise ValueError("public dictionary unit norm 必须强制")
    if dictionary.get("power_iteration_initialization_keyed") is not False:
        raise ValueError("public dictionary 初始化不得依赖 key")
    if dictionary.get("power_iteration_initialization_algorithm") != (
        "sha256_domain_nul_u64be_counter_bits_msb_first_rademacher"
    ):
        raise ValueError("public dictionary 初始化算法不匹配")
    if dictionary.get("power_iteration_initialization_bit_mapping") != (
        "zero_to_negative_one_one_to_positive_one"
    ):
        raise ValueError("public dictionary Rademacher bit mapping 不匹配")
    if dictionary.get("power_iteration_initialization_counter_start") != 0:
        raise ValueError("public dictionary initialization counter 必须从0开始")
    if dictionary.get("power_iteration_initialization_domain_encoding") != (
        "utf8"
    ):
        raise ValueError("public dictionary initialization domain 必须按 UTF-8")
    if dictionary.get("power_iteration_initialization_support") != (
        "restricted_latent_time_indices_only"
    ):
        raise ValueError("public dictionary initialization support 不匹配")
    if dictionary.get("power_iteration_initialization_coordinate_order") != (
        "compact_restricted_support_c_order_batch_channel_restricted_time_"
        "height_width_equivalent_full_c_order_skip_non_support"
    ):
        raise ValueError("public dictionary initialization coordinate order 不匹配")
    if dictionary.get("power_iteration_initialization_digest_stream") != (
        "concatenate_sha256_digest_bits_without_padding_until_support_filled"
    ):
        raise ValueError("public dictionary initialization digest stream 不匹配")
    if dictionary.get("power_iteration_operator") != (
        "restricted_support_projection_times_"
        "decoder_local_surrogate_jacobian_transpose_times_jacobian_"
        "times_restricted_support_projection"
    ):
        raise ValueError("public dictionary power operator 不匹配")
    if dictionary.get("power_iteration_postrun_iteration_selection_allowed"):
        raise ValueError("public dictionary 不得按运行结果选择 iteration")
    if dictionary.get("construction_surrogate_frame_indices") != (
        feature.get("video_window_frame_indices")
    ):
        raise ValueError("dictionary surrogate 与 saved-video window 不一致")
    if dictionary.get("construction_surrogate_feature_dimension") != (
        feature.get("feature_dimension")
    ):
        raise ValueError("dictionary surrogate 与 primary feature 维数不一致")
    if dictionary.get("construction_surrogate_geometry_source") != (
        "local_output_feature_contract_except_float32_pre_save"
    ):
        raise ValueError("dictionary surrogate geometry source 不匹配")
    expected_surrogate = {
        "construction_surrogate_clamp_in_jacobian": True,
        "construction_surrogate_decode_boundary": (
            "wan_vae_decode_sample_then_exact_"
            "video_processor_denormalization_mirror"
        ),
        "construction_surrogate_feature_id": (
            "pre_save_postprocessed_float32_"
            "framewise_four_by_four_cell_mean"
        ),
        "construction_surrogate_postprocess_formula": (
            "clamp_raw_vae_decode_sample_div_two_plus_"
            "one_half_to_closed_zero_one"
        ),
        "construction_surrogate_tensor_layout":
            "batch_frame_height_width_channel",
        "construction_surrogate_value_range": "closed_zero_one",
        "decoder_class": "AutoencoderKLWan",
        "decoder_latent_denormalization_formula": (
            "latent_times_vae_config_latents_std_"
            "plus_vae_config_latents_mean"
        ),
        "decoder_library_version": "diffusers_0.35.2",
        "decoder_model_revision_source": "video_contract_model_revision",
        "decoder_spatial_tiling_enabled_in_jacobian": False,
        "temporal_weight_application": (
            "after_eight_iterations_multiply_restricted_slices_then_"
            "global_float32_l2_normalize_before_sign_canonicalization"
        ),
    }
    for field, expected in expected_surrogate.items():
        if dictionary.get(field) != expected:
            raise ValueError(f"public dictionary {field} 不匹配")
    if feature.get("feature_flatten_order") != (
        "frame_then_cell_row_then_cell_column_then_channel"
    ):
        raise ValueError("primary local feature flatten order 不匹配")
    expected_feature_centering = {
        "construction_transfer_clean_intercept_source":
            "construction_clean_a_clean_b_arithmetic_mean",
        "feature_output_centering": "none_raw_representation",
        "signed_gate_clean_intercept_source":
            "current_identity_clean_a_clean_b_arithmetic_mean",
    }
    for field, expected in expected_feature_centering.items():
        if feature.get(field) != expected:
            raise ValueError(f"primary local feature {field} 不匹配")

    checkpoint = protocol_contract.get("checkpoint_contract")
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint_contract 必须是 object")
    if checkpoint.get("checkpoint_specs") != EXPECTED_CHECKPOINT_SPECS:
        raise ValueError("checkpoint representation/applicability matrix 不匹配")
    if checkpoint.get("common_signed_gate_identity_roles") != [
        "construction_identity",
        "signed_observability_identity",
    ]:
        raise ValueError("公共 signed gate identity applicability 不匹配")
    if checkpoint.get("construction_identity_signed_gate_role") != (
        "construction_readiness_only_not_gate0_pass"
    ):
        raise ValueError("C0 signed gate role 不匹配")
    if checkpoint.get("gate0_decision_identity_role") != (
        "signed_observability_identity"
    ):
        raise ValueError("Gate 0 decision identity role 不匹配")
    if checkpoint.get("primary_t0_estimation_identity_role") != (
        "construction_identity"
    ):
        raise ValueError("T0 estimation identity role 不匹配")
    if checkpoint.get("primary_t0_apply_only_gate_identity_role") != (
        "signed_observability_identity"
    ):
        raise ValueError("T0 apply-only gate identity role 不匹配")
    if checkpoint.get("checkpoint_ids") != [
        item["checkpoint_id"] for item in EXPECTED_CHECKPOINT_SPECS
    ]:
        raise ValueError("checkpoint ID/order 与 representation matrix 不一致")
    if checkpoint.get("primary_checkpoint_id") != (
        "saved_video_local_temporal_feature"
    ):
        raise ValueError("primary checkpoint 不匹配")
    if checkpoint.get("primary_transfer_gate_ids") != (
        PRIMARY_TRANSFER_GATE_IDS
    ):
        raise ValueError("primary transfer gate applicability 不匹配")
    if checkpoint.get("require_all_checkpoint_signed_gates") is not True:
        raise ValueError("三个 checkpoint 必须全部通过公共 signed gates")
    if checkpoint.get(
        "require_primary_checkpoint_transfer_gates"
    ) is not True:
        raise ValueError("primary checkpoint 必须通过 T0 transfer gates")

    construction = protocol_contract.get("construction_identity_contract")
    signed_gate = protocol_contract.get(
        "signed_observability_identity_contract"
    )
    if not isinstance(construction, Mapping) or not isinstance(
        signed_gate, Mapping
    ):
        raise TypeError("C0/A identity contract 必须是 object")
    if construction.get("identity_role") != "construction_identity":
        raise ValueError("C0 identity role 不匹配")
    if construction.get("dictionary_construction_scope") != (
        "construction_identity_design_responsibility_only"
    ):
        raise ValueError("C0 dictionary construction scope 不匹配")
    if construction.get("transfer_estimation_scope") != (
        "construction_identity_design_responsibility_only"
    ):
        raise ValueError("C0 transfer estimation scope 不匹配")
    if signed_gate.get("identity_role") != "signed_observability_identity":
        raise ValueError("A identity role 不匹配")
    if signed_gate.get("dictionary_apply_only") is not True:
        raise ValueError("identity A 必须只 apply dictionary")
    if signed_gate.get("feature_apply_only") is not True:
        raise ValueError("identity A 必须只 apply feature")
    if signed_gate.get("transfer_apply_only") is not True:
        raise ValueError("identity A 必须只 apply transfer")
    construction_probe_ids = construction.get("probe_ids")
    signed_probe_ids = signed_gate.get("probe_ids")
    if not isinstance(construction_probe_ids, list) or not isinstance(
        signed_probe_ids, list
    ):
        raise TypeError("C0/A probe_ids 必须是 array")
    if len(construction_probe_ids) != 4 or len(signed_probe_ids) != 4:
        raise ValueError("C0/A 必须各自精确包含4个 probe")
    if set(construction_probe_ids) & set(signed_probe_ids):
        raise ValueError("C0/A probe identity 必须严格隔离")
    if construction.get("probe_polarities") != [0, 0, 1, -1]:
        raise ValueError("C0 probe polarity 不匹配")
    if signed_gate.get("probe_polarities") != [0, 0, 1, -1]:
        raise ValueError("identity A probe polarity 不匹配")

    flow = protocol_contract.get("flow_injection_contract")
    if not isinstance(flow, Mapping):
        raise TypeError("flow_injection_contract 必须是 object")
    if flow.get("probe_polarity_coordinate") != (
        "scheduler_state_update_coordinate"
    ):
        raise ValueError("probe polarity 必须定义在 scheduler state update 坐标")
    if flow.get("scheduler_delta_sigma_formula") != (
        "next_sigma_minus_current_sigma"
    ):
        raise ValueError("delta_sigma 公式不匹配")
    if flow.get("intended_velocity_sign_formula") != (
        "signed_state_coefficient_times_sign_delta_sigma"
    ):
        raise ValueError("velocity sign 与 state-update polarity 映射不匹配")
    if flow.get("actual_signed_exposure_formula") != (
        EXPECTED_ACTUAL_SIGNED_EXPOSURE_FORMULA
    ):
        raise ValueError("actual signed exposure 公式不匹配")
    if flow.get("actual_signed_exposure_accumulation_dtype") != "float64":
        raise ValueError("actual signed exposure 必须以 float64 累加")
    schedule = flow.get("flow_schedule_contract")
    if not isinstance(schedule, Mapping):
        raise TypeError("flow_schedule_contract 必须是 object")
    expected_schedule = {
        "active_step_indices": [4, 5, 6],
        "active_window_id": "late_tapered_preterminal_frame_state_write",
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
        "flow_phase_by_step_decimal": [
            "0.026228725910186768",
            "0.08483484387397766",
            "0.15818744897842407",
            "0.2526847720146179",
            "0.3790806531906128",
            "0.5569911301136017",
            "0.8265496320091188",
            "0.9955357140861452",
        ],
        "historical_flow_stage_waveform_reuse": False,
        "implicit_all_one_waveform": False,
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
        "step_count": 8,
        "step_indices": list(range(8)),
        "waveform_by_step_millionths": [
            0, 0, 0, 0, 250000, 1000000, 500000, 0
        ],
        "waveform_definition": (
            "fixed_late_tapered_preterminal_three_step_write_"
            "not_derived_from_run_results"
        ),
    }
    _require_exact_mapping(
        schedule,
        expected_schedule,
        "flow_schedule_contract",
    )

    transfer = protocol_contract.get("transfer_estimation_contract")
    if not isinstance(transfer, Mapping):
        raise TypeError("transfer_estimation_contract 必须是 object")
    if transfer.get("estimated_object") != (
        "restricted_transfer_t0_equals_h0_d0"
    ):
        raise ValueError("实际估计对象必须是 T0=H0D0")
    if transfer.get("conceptual_operator_independently_identifiable") is not (
        False
    ):
        raise ValueError("H0 不得被标记为可独立识别")
    if transfer.get("independent_h0_artifact_allowed") is not False:
        raise ValueError("不得允许独立 H0 artifact")
    if transfer.get("estimator") != EXPECTED_TRANSFER_ESTIMATOR:
        raise ValueError("T0 estimator 公式不匹配")
    if transfer.get(
        "actual_signed_exposure_zero_denominator_fail_closed"
    ) is not True:
        raise ValueError("T0 exposure denominator 为零时必须 fail-closed")
    if transfer.get("signed_observability_prediction_formula") != (
        "t0_times_one_half_positive_minus_negative_actual_signed_exposure"
    ):
        raise ValueError("identity A T0 prediction 公式不匹配")
    if transfer.get("checkpoint_id") != (
        "saved_video_local_temporal_feature"
    ):
        raise ValueError("唯一 T0 必须只绑定 saved-video primary checkpoint")
    if transfer.get("transfer_feature_dimension") != 528:
        raise ValueError("唯一 T0 feature dimension 必须为528")

    gate = protocol_contract.get("signed_observability_gate_contract")
    if not isinstance(gate, Mapping):
        raise TypeError("signed_observability_gate_contract 必须是 object")
    expected_gate_formulas = {
        "clean_noise_norm_formula":
            "one_half_l2_clean_a_minus_clean_b",
        "clean_noise_denominator_formula":
            "max_clean_noise_norm_and_frozen_floor",
        "odd_clean_noise_ratio_formula":
            "l2_odd_div_max_clean_noise_norm_and_frozen_floor",
        "transfer_direction_cosine_formula": (
            "cosine_observed_odd_vs_"
            "t0_times_half_actual_exposure_difference"
        ),
        "transfer_relative_error_formula": (
            "l2_observed_odd_minus_prediction_"
            "div_max_l2_observed_odd_and_frozen_floor"
        ),
    }
    for field, expected in expected_gate_formulas.items():
        if gate.get(field) != expected:
            raise ValueError(f"Gate 0 {field} 公式不匹配")

    if protocol_digest != FROZEN_PROTOCOL_DIGEST:
        raise ValueError("protocol_digest 不等于代码冻结合同摘要")

    return dict(config)


def load_frame_state_signed_observability_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    return validate_frame_state_signed_observability_config(
        _read_config(Path(path))
    )


def protocol_contract_canonical_bytes(
    config: Mapping[str, Any],
) -> bytes:
    validated = validate_frame_state_signed_observability_config(config)
    return canonical_json_bytes(validated["protocol_contract"])


def build_public_context_record(
    config: Mapping[str, Any],
    *,
    public_nonce_random: str,
) -> dict[str, Any]:
    """Build, but never generate, the public context runtime value."""

    validated = validate_frame_state_signed_observability_config(config)
    if not isinstance(public_nonce_random, str) or not _LOWER_HEX_32.fullmatch(
        public_nonce_random
    ):
        raise ValueError(
            "public_nonce_random 必须是嵌入前生成的32位小写hex"
        )
    context_contract = validated["protocol_contract"][
        "public_context_contract"
    ]
    record = {
        "context_schema_id": context_contract["context_schema_id"],
        "method_id": METHOD_ID,
        "protocol_digest": validated["protocol_digest"],
        "public_nonce_random": public_nonce_random,
        "state_clock_rate_denominator": context_contract[
            "state_clock_rate_denominator"
        ],
        "state_clock_rate_numerator": context_contract[
            "state_clock_rate_numerator"
        ],
        "state_window_count": context_contract["state_window_count"],
    }
    validate_public_context_record(record, validated)
    return record


def validate_public_context_record(
    record: Mapping[str, Any],
    config: Mapping[str, Any],
) -> str:
    """Validate an exact public context and return its context digest."""

    validated = validate_frame_state_signed_observability_config(config)
    if set(record) != PUBLIC_CONTEXT_KEYS:
        raise ValueError(
            "public_context_record 字段集合不一致: "
            f"observed={sorted(record)}"
        )
    if record.get("context_schema_id") != PUBLIC_CONTEXT_SCHEMA_ID:
        raise ValueError("public context schema 不匹配")
    if record.get("method_id") != METHOD_ID:
        raise ValueError("public context method_id 不匹配")
    if record.get("protocol_digest") != validated["protocol_digest"]:
        raise ValueError("public context protocol_digest 不匹配")
    nonce = record.get("public_nonce_random")
    if not isinstance(nonce, str) or not _LOWER_HEX_32.fullmatch(nonce):
        raise ValueError("public context nonce 必须是32位小写hex")

    context_contract = validated["protocol_contract"][
        "public_context_contract"
    ]
    for field in (
        "state_clock_rate_numerator",
        "state_clock_rate_denominator",
        "state_window_count",
    ):
        value = record.get(field)
        if type(value) is not int or value <= 0:
            raise TypeError(f"{field} 必须是正整数")
        if value != context_contract[field]:
            raise ValueError(f"{field} 与冻结 protocol 不一致")
    return canonical_json_digest(dict(record))


def parse_canonical_public_context_bytes(
    raw: bytes,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """Require byte-for-byte canonical public-context serialization."""

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("public context 必须是 UTF-8") from exc
    value = _strict_json_loads(text)
    if not isinstance(value, dict):
        raise TypeError("public context 顶层必须是 object")
    canonical = canonical_json_bytes(value)
    if raw != canonical:
        raise ValueError("public context bytes 不是冻结 canonical JSON")
    digest = validate_public_context_record(value, config)
    return dict(value), digest


def build_frame_state_probe_plan(
    config: Mapping[str, Any],
) -> tuple[FrameStateProbePlanRecord, ...]:
    """Build the frozen 4-item C0 plus 4-item identity-A design plan."""

    validated = validate_frame_state_signed_observability_config(config)
    protocol = validated["protocol_contract"]
    window_id = protocol["video_contract"]["video_window_id"]
    rows: list[FrameStateProbePlanRecord] = []
    role_specs = (
        (
            "construction_identity",
            protocol["construction_identity_contract"]["probe_ids"],
            protocol["construction_identity_contract"]["probe_polarities"],
        ),
        (
            "signed_observability_identity",
            protocol["signed_observability_identity_contract"]["probe_ids"],
            protocol["signed_observability_identity_contract"][
                "probe_polarities"
            ],
        ),
    )
    for identity_role, probe_ids, polarities in role_specs:
        for probe_id, polarity in zip(probe_ids, polarities, strict=True):
            rows.append(
                FrameStateProbePlanRecord(
                    plan_index=len(rows),
                    identity_role=identity_role,
                    probe_id=str(probe_id),
                    probe_role=(
                        "clean"
                        if polarity == 0
                        else "positive"
                        if polarity == 1
                        else "negative"
                    ),
                    signed_state_coefficient=int(polarity),
                    state_window_id=str(window_id),
                    state_dimension_index=0,
                    same_initial_noise_within_identity=True,
                    public_context_binding=(
                        "one_pre_embedding_nonce_shared_within_identity"
                    ),
                    execution_authorized=True,
                    formal_result=False,
                    stage_progression_allowed=False,
                )
            )
    return tuple(rows)


def validate_frame_state_probe_plan(
    records: Sequence[FrameStateProbePlanRecord],
    config: Mapping[str, Any],
) -> tuple[FrameStateProbePlanRecord, ...]:
    expected = build_frame_state_probe_plan(config)
    observed = tuple(records)
    if observed != expected:
        raise ValueError("frame-state probe plan 必须与冻结8项计划精确一致")
    return observed


def frame_state_probe_plan_digest(
    records: Sequence[FrameStateProbePlanRecord],
    config: Mapping[str, Any],
) -> str:
    validated = validate_frame_state_probe_plan(records, config)
    return canonical_json_digest([asdict(record) for record in validated])
