"""Exact local contract for the block-local Patch-relation attention preflight.

This module validates the next carrier-contract batch only. It does not
authorize a real Wan adapter, Colab execution, Gate 0, observer work, or a
method claim.
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
    "block_local_attention_response_contract_implemented_pending_"
    "runtime_adapter"
)
FROZEN_PROTOCOL_DIGEST = (
    "feda6c5e9356abbb12a4c7ab017faf6bcd13b245ba0082f419f4cf642f257e1e"
)
EXPECTED_AUTHORIZATION_BOUNDARY = {
    "runtime_implementation_authorized": False,
    "colab_execution_allowed": False,
    "gpu_execution_allowed": False,
    "gate0_execution_allowed": False,
    "observer_implementation_allowed": False,
    "fixed_fpr_execution_allowed": False,
    "paper_claim_allowed": False,
    "formal_result": False,
    "stage_progression_allowed": False,
    "claim_support_status": (
        "block_attention_local_contract_only_not_runtime_gate_or_method_"
        "evidence"
    ),
}
EXPECTED_BLOCK_LOCAL_ATTENTION_CONTROL_CONTRACT = {
    "carrier_family": "block_local_patch_relation_attention_control",
    "runtime_adapter_implemented": False,
    "colab_runner_implemented": False,
    "target_model": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "diffusers_version": "0.35.2",
    "target_module_boundary": (
        "one_predeclared_transformer_block_self_attention_pre_softmax_"
        "qk_logits"
    ),
    "target_block_index": 18,
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
}
EXPECTED_FUTURE_SINGLE_STEP_RESPONSE_PLAN = {
    "authorized_now": False,
    "requires_separate_runtime_adapter_review": True,
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
    "pass_condition": (
        "budget_with_nonzero_repeatable_near_antisymmetric_cfg_response_"
        "only_allows_c0_design"
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
            "future_single_step_response_plan",
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
        contract["future_single_step_response_plan"]
        != EXPECTED_FUTURE_SINGLE_STEP_RESPONSE_PLAN
    ):
        raise ValueError("future single-step response plan 漂移")
    if contract["result_boundary"] != EXPECTED_AUTHORIZATION_BOUNDARY:
        raise ValueError("block-attention result boundary 漂移")
    return payload
