"""分析 state_space_inference_formalization 状态变量消融结果。"""

from __future__ import annotations

STATE_VARIABLE_VARIANTS = {
    "key_conditioned_state_space_without_phase_state",
    "key_conditioned_state_space_without_evidence_state",
    "key_conditioned_state_space_without_confidence_state",
    "key_conditioned_state_space_without_disturbance_state",
    "key_conditioned_state_space_without_bidirectional_smoothing",
    "key_conditioned_state_space_without_entropy_gate",
}


def state_variable_ablation_all_nontrivial(ablation_records: list[dict]) -> bool:
    """判断状态变量消融是否至少 3 个组件产生非平凡影响。"""
    nontrivial = [record for record in ablation_records if record.get("ablation_name") in STATE_VARIABLE_VARIANTS and abs(float(record.get("ablation_observed_delta_tpr", 0.0))) > 0.0]
    return len(nontrivial) >= 3
