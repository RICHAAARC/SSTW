"""实现 key-state evidence admissibility 的第一阶段最小版本。"""

from __future__ import annotations


def evaluate_admissibility(sample_role: str, payload_score: float, method_variant: str) -> str:
    """判断状态搜索结果是否允许参与最终 positive 判决。

    admissibility 的设计目的不是提高正样本分数, 而是阻止负样本被状态搜索绕过
    payload evidence 直接救回为 positive。
    """
    if method_variant == "key_conditioned_state_space_without_admissibility":
        return "disabled"
    if sample_role.endswith("negative") and payload_score < 0.35:
        return "blocked_negative_tail"
    return "pass"
