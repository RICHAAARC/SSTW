"""Exact contract for the non-Gate Patch-relation phase-response preflight."""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping


DEFAULT_CONFIG_PATH = (
    "configs/protocol/sstw_patch_relation_phase_response_preflight.json"
)
PROFILE_ID = "sstw_patch_relation_phase_response_preflight"
METHOD_ID = "frame_state_synchronized_generative_flow_video_watermark"
CONTRACT_STATE = (
    "single_step_phase_response_preflight_implemented_pending_user_colab_run"
)
FROZEN_PROTOCOL_DIGEST = (
    "09a8643b6a7a763bef57dcf834dc4a1fe0fb98a50490c982e94f3aa1046c6d83"
)
NEXT_DETERMINISTIC_SCALE = 0.0002802486139811453
NEXT_REALIZED_PHASE_RADIANS = NEXT_DETERMINISTIC_SCALE * 0.015625
PLATEAU_RATIO_THRESHOLD = 0.8
REPEATABILITY_FLOOR_RATIO_THRESHOLD = 0.1
TUPLE_DELTA_RELATIVE_TOLERANCE = 0.00001
EXPECTED_AUTHORIZATION_BOUNDARY = {
    "runtime_implementation_authorized": True,
    "diagnostic_execution_allowed": True,
    "gpu_execution_allowed": True,
    "colab_execution_allowed": True,
    "runner_implementation_allowed": True,
    "notebook_handler_implementation_allowed": False,
    "drive_update_allowed": False,
    "gate0_execution_allowed": False,
    "observer_implementation_allowed": False,
    "attack_execution_allowed": False,
    "fixed_fpr_execution_allowed": False,
    "baseline_execution_allowed": False,
    "paper_claim_allowed": False,
    "formal_result": False,
    "stage_progression_allowed": False,
    "claim_support_status": (
        "single_step_runtime_preflight_only_not_gate_or_method_evidence"
    ),
}
EXPECTED_CLASSIFICATION_CONTRACT = {
    "classification_order": [
        "scale_application_failure",
        "forward_repeatability_floor_candidate",
        "feasible_nonzero_phase_region_candidate",
        "quantization_dead_zone_candidate",
        "bf16_or_attention_piecewise_plateau_candidate",
        "indeterminate",
        "multiple_candidates",
    ],
    "scale_application_failure": (
        "any_zero_phase_tuple_delta_nonzero_or_any_active_tuple_delta_"
        "nonfinite_or_zero_or_actual_expected_elementwise_binding_failure_"
        "or_positive_negative_tuple_delta_relative_difference_above_0.00001"
    ),
    "forward_repeatability_floor_candidate": (
        "maximum_base_or_per_sign_control_repeatability_floor_ratio_"
        "strictly_above_0.1"
    ),
    "feasible_nonzero_phase_region_candidate": (
        "all_four_candidate_evaluations_feasible_and_repeatability_floor_"
        "not_detected"
    ),
    "quantization_dead_zone_candidate": (
        "all_control_cfg_arrays_byte_equal_to_first_zero_phase_base_cfg_"
        "while_active_rope_tuple_delta_is_nonzero_and_repeatability_floor_"
        "not_detected"
    ),
    "bf16_or_attention_piecewise_plateau_candidate": (
        "repeatability_floor_not_detected_and_not_dead_zone_and_minimum_"
        "candidate_actual_delta_norm_divided_by_historical_last_worst_"
        "actual_delta_norm_at_least_0.8"
    ),
    "repeatability_floor_ratio_threshold": "0.1",
    "plateau_ratio_threshold": "0.8",
    "tuple_delta_relative_tolerance": "0.00001",
    "unique_root_cause_claim_allowed": False,
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


def load_patch_relation_phase_response_preflight_config(
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
        "phase-response preflight config",
    )
    if (
        payload["profile_id"] != PROFILE_ID
        or payload["method_id"] != METHOD_ID
        or payload["contract_state"] != CONTRACT_STATE
    ):
        raise ValueError("phase-response preflight identity 漂移")
    contract = payload["protocol_contract"]
    observed_digest = protocol_digest(contract)
    if payload["protocol_digest"] != observed_digest:
        raise ValueError("phase-response preflight self digest 不匹配")
    if observed_digest != FROZEN_PROTOCOL_DIGEST:
        raise ValueError("phase-response preflight frozen digest 不匹配")
    authorization = payload["authorization_boundary"]
    if authorization != EXPECTED_AUTHORIZATION_BOUNDARY:
        raise ValueError("phase-response preflight authorization boundary 漂移")
    source = contract["source_failure_boundary"]
    final_scale = float(source["attempted_scales"][-1])
    final_norm = float(source["worst_actual_delta_norms"][-1])
    final_energy = float(source["worst_energy_increments"][-1])
    norm_budget = float(source["norm_budget"])
    remaining = float(source["remaining_flow_energy"])
    safety = float(source["backoff_safety_factor"])
    rebuilt_scale = final_scale * min(
        norm_budget / final_norm,
        math.sqrt(remaining / final_energy),
    ) * safety
    if rebuilt_scale != NEXT_DETERMINISTIC_SCALE:
        raise ValueError("next deterministic phase scale 重算不一致")
    if float(source["next_deterministic_scale"]) != NEXT_DETERMINISTIC_SCALE:
        raise ValueError("next deterministic phase scale 未冻结")
    if (
        float(source["next_realized_phase_radians"])
        != NEXT_REALIZED_PHASE_RADIANS
    ):
        raise ValueError("next realized phase 未冻结")
    plan = contract["forward_plan"]
    if plan != {
        "zero_phase_base_repeat_count": 2,
        "candidate_signs": [1, -1],
        "candidate_scale": format(NEXT_DETERMINISTIC_SCALE, ".17g"),
        "candidate_repeat_count_per_sign": 2,
        "additional_scale_points": [],
        "cfg_branch_order": ["conditional", "unconditional"],
        "expected_transformer_forward_count": 12,
        "real_scheduler_step_call_count": 0,
        "decode_allowed": False,
        "video_export_allowed": False,
        "phase_selection_allowed": False,
        "strength_sweep_allowed": False,
    }:
        raise ValueError("phase-response preflight forward plan 漂移")
    classification = contract["classification_contract"]
    if classification != EXPECTED_CLASSIFICATION_CONTRACT:
        raise ValueError("phase-response preflight classification 漂移")
    result = contract["result_boundary"]
    if any(
        result.get(name) is not False
        for name in {
            "gate0_pass",
            "gate0_execution_allowed",
            "full_eight_video_rerun_allowed",
            "formal_result",
            "stage_progression_allowed",
            "observer_implementation_allowed",
            "paper_claim_allowed",
        }
    ):
        raise ValueError("phase-response preflight result boundary 漂移")
    return payload
