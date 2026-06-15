"""B5 机制 gate 审计。"""

from __future__ import annotations


def audit_mechanism(runtime_status: dict, generalization_status: dict) -> dict:
    """根据 B5 gate 条件给出严格审计结果。"""
    runnable = runtime_status.get("generation_model_runnable_status") == "runnable"
    generation_model_main_table_ready = False
    trajectory_observation_gain_confirmed = False
    fixed_low_fpr_audit_pass = False
    quality_motion_semantic_consistency_pass = False
    cross_prompt_seed_generalization_pass = bool(generalization_status.get("cross_prompt_seed_generalization_pass"))
    mechanism_pass = all([
        runnable,
        generation_model_main_table_ready,
        trajectory_observation_gain_confirmed,
        fixed_low_fpr_audit_pass,
        quality_motion_semantic_consistency_pass,
        cross_prompt_seed_generalization_pass,
    ])
    return {
        "gpu_validation_status": runtime_status.get("gpu_validation_status"),
        "gpu_validation_reason": runtime_status.get("gpu_validation_reason"),
        "generation_model_runnable_status": runtime_status.get("generation_model_runnable_status"),
        "generation_model_not_run_reason": runtime_status.get("generation_model_not_run_reason"),
        "generation_model_main_table_ready": generation_model_main_table_ready,
        "trajectory_observation_gain_confirmed": trajectory_observation_gain_confirmed,
        "fixed_low_fpr_audit_pass": fixed_low_fpr_audit_pass,
        "quality_motion_semantic_consistency_pass": quality_motion_semantic_consistency_pass,
        "cross_prompt_seed_generalization_pass": cross_prompt_seed_generalization_pass,
        "negative_state_over_threshold_count": None,
        "attacked_negative_fpr": None,
        "formal_claim_status": "blocked_until_gpu_generation_run" if not mechanism_pass else "supported_by_generation_records",
        "top_conference_b5_gate": "FAIL" if not mechanism_pass else "PASS",
        "mechanism_pass": mechanism_pass,
    }
