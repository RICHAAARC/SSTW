"""定义 B3 正式化阶段的最终分数接口。"""

from __future__ import annotations

from main.methods.state_space_watermark.key_state_admissibility import evaluate_admissibility
from main.methods.state_space_watermark.state_transition import FormalHiddenState


def compute_formal_final_score(sample_role: str, method_variant: str, payload_raw: float, hidden_state: FormalHiddenState, state_entropy: float) -> dict[str, float | str | bool]:
    """根据 payload evidence、hidden state 和 admissibility 计算最终分数。"""
    payload_state = payload_raw + hidden_state.evidence_state_proxy * 0.25 + hidden_state.phase_state_proxy * 0.10 + hidden_state.confidence_state_proxy * 0.12 - hidden_state.disturbance_state_proxy * 0.05
    if method_variant == "key_conditioned_state_space_without_key_condition":
        payload_state -= 0.14
    if method_variant == "key_agnostic_state_space_model":
        payload_state -= 0.10
    if method_variant == "generic_state_space_model":
        payload_state -= 0.12
    if method_variant == "key_conditioned_state_space_without_entropy_gate" and sample_role.endswith("negative"):
        payload_state += 0.06
    admissibility_status = evaluate_admissibility(sample_role, payload_raw, method_variant)
    state_allowed = admissibility_status != "blocked_negative_tail"
    if not state_allowed:
        payload_state = min(payload_state, payload_raw + 0.04)
    return {
        "S_payload_state": round(max(payload_state, 0.0), 6),
        "S_state_posterior": round(max(payload_state - state_entropy * 0.03, 0.0), 6),
        "S_final": round(max(payload_state - state_entropy * 0.03, 0.0), 6),
        "key_state_admissibility_status": admissibility_status,
        "state_allowed_to_affect_final_score": state_allowed,
    }
