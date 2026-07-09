"""第一阶段状态过滤接口。"""

from __future__ import annotations


def filter_state_score(score: float) -> float:
    """返回过滤后的状态分数。当前实现是 synthetic_state_inference_sanity 接口冻结用的恒等映射。"""
    return score
