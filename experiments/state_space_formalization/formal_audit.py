"""审计 state_space_inference_formalization state-space formalization gate。"""

from __future__ import annotations

from statistics import mean

from main.analysis.admissibility_audit import admissibility_negative_tail_status
from main.analysis.generalization_audit import generalization_status
from main.analysis.key_condition_ablation import key_condition_ablation_gain
from main.analysis.state_variable_ablation import state_variable_ablation_all_nontrivial

COMPLEX_ATTACKS = {"irregular_frame_dropping", "frame_duplication", "frame_rate_resampling", "segment_jump", "local_clip"}


def _mean_score(records: list[dict], method_variant: str, attack_name: str, sample_role: str) -> float:
    values = [float(record["S_final"]) for record in records if record["split"] == "test" and record["method_variant"] == method_variant and record["attack_name"] == attack_name and record["sample_role"] == sample_role]
    if not values:
        raise ValueError(f"missing records for {method_variant} {attack_name} {sample_role}")
    return mean(values)


def audit_formalization(records: list[dict], ablation_records: list[dict], generalization_records: list[dict], target_fpr: float) -> dict:
    """返回 state_space_inference_formalization formalization 机制审计结果。"""
    aggregators = ["conv1d_temporal_aggregator", "gru_temporal_aggregator", "transformer_temporal_aggregator"]
    beats_temporal_aggregators = all(any(_mean_score(records, "key_conditioned_state_space_inference", attack, "attacked_positive") > _mean_score(records, method, attack, "attacked_positive") for attack in COMPLEX_ATTACKS) for method in aggregators)
    beats_generic_ssm = all(_mean_score(records, "key_conditioned_state_space_inference", attack, "attacked_positive") > _mean_score(records, "generic_state_space_model", attack, "attacked_positive") for attack in COMPLEX_ATTACKS)
    beats_key_agnostic = all(_mean_score(records, "key_conditioned_state_space_inference", attack, "attacked_positive") > _mean_score(records, "key_agnostic_state_space_model", attack, "attacked_positive") for attack in COMPLEX_ATTACKS)
    attacked_negative = [record for record in records if record["split"] == "test" and record["sample_role"] == "attacked_negative"]
    attacked_negative_fpr = mean(1.0 if record["decision"] == "positive" else 0.0 for record in attacked_negative)
    negative_state_over_threshold_count = sum(1 for record in records if record["sample_role"].endswith("negative") and record["method_variant"] == "key_conditioned_state_space_inference" and record["S_final"] >= record["threshold_value"])
    key_gain = key_condition_ablation_gain(ablation_records)
    admissibility_status = admissibility_negative_tail_status(records)
    state_variable_status = "PASS" if state_variable_ablation_all_nontrivial(ablation_records) else "FAIL"
    unseen_key_status = generalization_status(generalization_records, "unseen_key")
    unseen_attack_status = generalization_status(generalization_records, "unseen_attack_type")
    mechanism_pass = all([beats_temporal_aggregators, beats_generic_ssm, beats_key_agnostic, key_gain > 0.0, admissibility_status == "PASS", state_variable_status == "PASS", unseen_key_status == "PASS", unseen_attack_status == "PASS", attacked_negative_fpr <= max(2 * target_fpr, target_fpr + 1 / max(len(attacked_negative), 1)), negative_state_over_threshold_count == 0])
    return {
        "state_space_inference_formal_decision": "PASS" if mechanism_pass else "FAIL",
        "mechanism_pass": mechanism_pass,
        "beats_temporal_aggregators": beats_temporal_aggregators,
        "beats_generic_state_space_model": beats_generic_ssm,
        "beats_key_agnostic_state_space_model": beats_key_agnostic,
        "key_condition_ablation_gain": round(key_gain, 6),
        "admissibility_negative_tail_status": admissibility_status,
        "state_variable_ablation_all_nontrivial": state_variable_status,
        "unseen_key_generalization_status": unseen_key_status,
        "unseen_attack_generalization_status": unseen_attack_status,
        "attacked_negative_fpr": round(attacked_negative_fpr, 6),
        "negative_state_over_threshold_count": negative_state_over_threshold_count,
        "trajectory_status": "EXPLICIT",
    }
