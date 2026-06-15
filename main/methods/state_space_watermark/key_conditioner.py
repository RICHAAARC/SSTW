"""提供 key condition 的轻量注入逻辑。"""

from __future__ import annotations


def key_condition_gain(method_variant: str) -> float:
    """返回方法变体对应的 key condition 增益。"""
    if method_variant in {"key_conditioned_state_space_inference", "key_conditioned_state_space_without_admissibility"}:
        return 0.16
    if method_variant in {"key_conditioned_state_space_without_key_condition", "key_agnostic_state_space_model"}:
        return 0.0
    return 0.04 if "tubelet" in method_variant else 0.0
