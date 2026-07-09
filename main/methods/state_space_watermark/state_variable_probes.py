"""提供 state_space_inference_formalization 状态变量消融的影响建模。"""

from __future__ import annotations

STATE_VARIABLE_PENALTY = {
    "key_conditioned_state_space_without_phase_state": 0.11,
    "key_conditioned_state_space_without_evidence_state": 0.14,
    "key_conditioned_state_space_without_confidence_state": 0.10,
    "key_conditioned_state_space_without_disturbance_state": 0.08,
    "key_conditioned_state_space_without_bidirectional_smoothing": 0.07,
    "key_conditioned_state_space_without_entropy_gate": -0.02,
}


def state_variable_penalty(method_variant: str) -> float:
    """返回移除某个状态变量后对正样本分数的惩罚。"""
    return STATE_VARIABLE_PENALTY.get(method_variant, 0.0)


def removed_state_component(method_variant: str) -> str:
    """返回方法变体移除的状态组件。"""
    return method_variant.replace("key_conditioned_state_space_without_", "") if "without" in method_variant else "none"
