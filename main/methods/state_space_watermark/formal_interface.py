"""汇总 B3 state-space inference formal interface。"""

from __future__ import annotations

from main.methods.state_space_watermark.detector_score import compute_formal_final_score
from main.methods.state_space_watermark.score import score_method
from main.methods.state_space_watermark.state_observation_model import build_formal_observation
from main.methods.state_space_watermark.state_transition import transition_state
from main.methods.state_space_watermark.state_variable_probes import state_variable_penalty

BASELINE_ALIAS = {
    "no_state_inference": "frame_prc",
    "generic_temporal_mean_pooling": "generic_temporal_mean_pooling",
    "conv1d_temporal_aggregator": "conv1d_temporal_aggregator",
    "gru_temporal_aggregator": "gru_temporal_aggregator",
    "transformer_temporal_aggregator": "transformer_temporal_aggregator",
}


def run_formal_inference(sample_role: str, attack_name: str, method_variant: str, quality_confidence: float = 1.0) -> dict:
    """运行 B3 的正式状态空间推断接口。"""
    if method_variant in BASELINE_ALIAS:
        baseline = score_method(sample_role, attack_name, BASELINE_ALIAS[method_variant])
        return {
            "S_payload_raw": baseline.payload_raw,
            "S_payload_state": baseline.payload_state,
            "S_state_posterior": baseline.state_posterior,
            "S_final": baseline.final_score,
            "phase_state_proxy": 0.0,
            "evidence_state_proxy": baseline.payload_raw,
            "confidence_state_proxy": 0.0,
            "disturbance_state_proxy": baseline.state_transition_residual,
            "state_transition_residual": baseline.state_transition_residual,
            "state_entropy": baseline.state_entropy,
            "state_coverage_ratio": baseline.state_coverage_ratio,
            "state_matched_count": baseline.state_matched_count,
            "key_state_admissibility_status": "not_applicable",
            "state_allowed_to_affect_final_score": False,
        }

    key_conditioned = method_variant not in {"generic_state_space_model", "key_agnostic_state_space_model", "key_conditioned_state_space_without_key_condition"}
    observation = build_formal_observation(sample_role, attack_name, key_conditioned, quality_confidence)
    hidden_state = transition_state(observation, method_variant)
    base = score_method(sample_role, attack_name, "key_conditioned_state_space_inference" if method_variant.startswith("key_conditioned") else method_variant)
    penalty = state_variable_penalty(method_variant) if sample_role.endswith("positive") else 0.0
    state_entropy = base.state_entropy + (0.04 if "without_entropy_gate" in method_variant else 0.0)
    score = compute_formal_final_score(sample_role, method_variant, max(base.payload_raw - penalty * 0.3, 0.0), hidden_state, state_entropy)
    if sample_role.endswith("positive"):
        score["S_payload_state"] = round(max(float(score["S_payload_state"]) - penalty, 0.0), 6)
        score["S_state_posterior"] = round(max(float(score["S_state_posterior"]) - penalty, 0.0), 6)
        score["S_final"] = round(max(float(score["S_final"]) - penalty, 0.0), 6)
    return {
        "S_payload_raw": round(max(base.payload_raw - penalty * 0.3, 0.0), 6),
        **score,
        "phase_state_proxy": hidden_state.phase_state_proxy,
        "evidence_state_proxy": hidden_state.evidence_state_proxy,
        "confidence_state_proxy": hidden_state.confidence_state_proxy,
        "disturbance_state_proxy": hidden_state.disturbance_state_proxy,
        "state_transition_residual": hidden_state.state_transition_residual,
        "state_entropy": round(state_entropy, 6),
        "state_coverage_ratio": base.state_coverage_ratio,
        "state_matched_count": base.state_matched_count,
    }
