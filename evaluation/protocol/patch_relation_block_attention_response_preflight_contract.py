"""Exact contract for the block-local Patch-relation attention preflight.

This module validates the single-step real Wan block-attention preflight
contract only. It does not authorize Gate 0, observer work, stage progression,
or a method claim.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping


DEFAULT_CONFIG_PATH = (
    "configs/protocol/"
    "sstw_patch_relation_block_attention_response_preflight.json"
)
PROFILE_ID = "sstw_patch_relation_block_attention_response_preflight"
METHOD_ID = "frame_state_synchronized_generative_flow_video_watermark"
CONTRACT_STATE = (
    "block_local_attention_response_runtime_adapter_contract_implemented_"
    "pending_real_wan_gpu_preflight"
)
FROZEN_PROTOCOL_DIGEST = (
    "4cc1959aa9a73223e0327f50ae75757a199825e0299595fb7f0572669d7cee72"
)
EXPECTED_AUTHORIZATION_BOUNDARY = {
    "runtime_implementation_authorized": True,
    "colab_execution_allowed": True,
    "gpu_execution_allowed": True,
    "gate0_execution_allowed": False,
    "observer_implementation_allowed": False,
    "fixed_fpr_execution_allowed": False,
    "paper_claim_allowed": False,
    "formal_result": False,
    "stage_progression_allowed": False,
    "claim_support_status": (
        "block_attention_single_step_runtime_preflight_only_not_gate_or_method_"
        "evidence"
    ),
}
EXPECTED_BLOCK_LOCAL_ATTENTION_CONTROL_CONTRACT = {
    "carrier_family": "block_local_patch_relation_attention_control",
    "local_runtime_primitives_implemented": True,
    "real_wan_runtime_adapter_implemented": True,
    "colab_runner_implemented": True,
    "target_model": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "diffusers_version": "0.35.2",
    "target_module_boundary": (
        "one_predeclared_transformer_block_self_attention_pre_softmax_"
        "qk_logits"
    ),
    "target_block_index": 18,
    "target_num_attention_heads": 12,
    "token_grid_shape": [9, 20, 32],
    "video_frame_window": [11, 22],
    "active_token_time_indices": [3, 4, 5],
    "patch_token_a": {"row": 9, "column": 13},
    "patch_token_b": {"row": 9, "column": 18},
    "head_group_indices": [0, 1, 2, 3],
    "sparse_entry_order": [
        "time_index_ascending",
        "head_index_ascending",
        "directed_pair_order",
    ],
    "directed_pair_order": [
        {"query": "patch_a", "key": "patch_b", "sign": 1},
        {"query": "patch_b", "key": "patch_a", "sign": -1},
    ],
    "zero_sum_per_active_time_and_head": True,
    "candidate_key_independent_descriptor": True,
    "master_key_may_only_select_signed_coefficient": True,
    "global_shared_rope_phase_forbidden": True,
    "direct_velocity_additive_projection_forbidden": True,
    "scoped_runtime_adapter_contract_implemented": True,
    "sparse_qk_bias_runtime_method": (
        "official_no_mask_attention_plus_exact_selected_query_head_row_recompute"
    ),
    "runtime_adapter_forbids_dense_attention_mask_allocation": True,
    "scope_record_requires_complete_clean_exit": True,
    "scope_record_requires_exact_callback_coverage": True,
    "scope_failure_must_restore_original_binding": True,
}
EXPECTED_LOCAL_SINGLE_STEP_RESPONSE_CONTRACT = {
    "authorized_now": "real_wan_single_step_preflight_only",
    "requires_separate_real_wan_adapter_review": False,
    "real_wan_single_step_runner_implemented": True,
    "transformer_forward_count": 16,
    "zero_control_repeat_count": 2,
    "candidate_bias_magnitudes": [
        "0.015625",
        "0.00390625",
        "0.0009765625",
    ],
    "candidate_signs": [1, -1],
    "scheduler_step_call_count": 0,
    "decode_allowed": False,
    "video_export_allowed": False,
    "gate0_allowed": False,
    "observer_allowed": False,
    "strength_sweep_allowed": False,
    "repeatability_floor_ratio_threshold": "0.1",
    "near_antisymmetry_cosine_threshold": "-0.9",
    "nonzero_response_floor": "1e-12",
    "step0_norm_budget": "2.937237890625",
    "record_validator_recomputes_candidate_guards": True,
    "record_validator_recomputes_attention_local_and_propagated_vectors": True,
    "runtime_rejects_unfrozen_candidate_magnitude": True,
    "attention_local_readout_contract": (
        "selected_query_head_row_pre_bias_post_bias_probability_delta_from_actual_"
        "processor_observations"
    ),
    "propagated_cfg_readout_contract": (
        "token_grid_9x20x32_to_latent_grid_9x40x64_patch_aligned_2x2_patch_a_"
        "minus_patch_b_mean_relation"
    ),
    "candidate_feasible_requires": [
        "attention_local_nonzero_response",
        "attention_local_repeatability_floor_ratio_at_most_0.1",
        "attention_local_positive_negative_pair_cosine_at_most_minus_0.9",
        "propagated_cfg_patch_relation_nonzero_response",
        "propagated_cfg_patch_relation_repeatability_floor_ratio_at_most_0.1",
        "propagated_cfg_patch_relation_response_l2_norm_at_most_step0_norm_budget",
        "propagated_cfg_patch_relation_positive_negative_pair_cosine_at_most_minus_0.9",
    ],
    "pass_condition": (
        "attention_local_and_propagated_cfg_patch_relation_both_nonzero_"
        "repeatable_budgeted_near_antisymmetric_only_allows_c0_design"
    ),
    "fail_condition": (
        "no_budgeted_nonzero_repeatable_response_stops_this_carrier_or_"
        "block_location"
    ),
}
EXPECTED_SOURCE_FAILURE_BOUNDARY = {
    "failed_carrier": "global_shared_wan_rope_temporal_phase_relation",
    "failure_classification": "bf16_or_attention_piecewise_plateau_candidate",
    "source_run_id": "20260729_034340_596839_e8ab1070",
    "source_repository_commit": "e8ab107089d73d37fc0af3160e50a34b6785a92d",
    "source_zip_sha256": (
        "4c50e53ea33bc0ba6d25a3d12496d4b2a23171a3da48cff789f8c91efa27fa7d"
    ),
    "source_manifest_sha256": (
        "8eb264aa3c4a311f661766fe2f73e5f45b0f1587e5f4ac004afd07ef430d5dd0"
    ),
    "minimum_candidate_actual_delta_norm": "17.629608154296875",
    "norm_budget": "2.937237890625",
    "candidate_to_historical_plateau_ratio": "0.880489058130237",
    "decision": "stop_global_shared_rope_carrier_no_scale_tuning_no_c0_no_gate0",
    "evidence_boundary": (
        "single_step_non_gate_preflight_not_method_or_paper_evidence"
    ),
}


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def protocol_digest(value: Mapping[str, Any]) -> str:
    return sha256(canonical_json_bytes(dict(value))).hexdigest()


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    label: str,
) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} keys 与冻结合同不一致")


def load_patch_relation_block_attention_response_preflight_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require_exact_keys(
        payload,
        {
            "profile_id",
            "method_id",
            "contract_state",
            "authorization_boundary",
            "protocol_contract",
            "protocol_digest",
        },
        "block-attention response preflight config",
    )
    if (
        payload["profile_id"] != PROFILE_ID
        or payload["method_id"] != METHOD_ID
        or payload["contract_state"] != CONTRACT_STATE
    ):
        raise ValueError("block-attention response preflight identity 漂移")
    authorization = payload["authorization_boundary"]
    if authorization != EXPECTED_AUTHORIZATION_BOUNDARY:
        raise ValueError(
            "block-attention response preflight authorization boundary 漂移"
        )
    contract = payload["protocol_contract"]
    observed_digest = protocol_digest(contract)
    if payload["protocol_digest"] != observed_digest:
        raise ValueError(
            "block-attention response preflight self digest 不匹配"
        )
    if observed_digest != FROZEN_PROTOCOL_DIGEST:
        raise ValueError(
            "block-attention response preflight frozen digest 不匹配"
        )
    _require_exact_keys(
        contract,
        {
            "source_failure_boundary",
            "block_local_attention_control_contract",
            "local_single_step_response_contract",
            "runtime_execution_binding",
            "result_boundary",
        },
        "block-attention protocol contract",
    )
    if (
        contract["source_failure_boundary"]
        != EXPECTED_SOURCE_FAILURE_BOUNDARY
    ):
        raise ValueError("global RoPE failure boundary 漂移")
    if (
        contract["block_local_attention_control_contract"]
        != EXPECTED_BLOCK_LOCAL_ATTENTION_CONTROL_CONTRACT
    ):
        raise ValueError("block-local attention control contract 漂移")
    if (
        contract["local_single_step_response_contract"]
        != EXPECTED_LOCAL_SINGLE_STEP_RESPONSE_CONTRACT
    ):
        raise ValueError("local single-step response contract 漂移")
    if contract["runtime_execution_binding"] != {
        "probe_id": "block_attention_c0_clean_a_step0_runtime_preflight"
    }:
        raise ValueError("block-attention runtime execution binding 漂移")
    if contract["result_boundary"] != EXPECTED_AUTHORIZATION_BOUNDARY:
        raise ValueError("block-attention result boundary 漂移")
    return payload
