"""第一阶段状态平滑接口。"""

from __future__ import annotations


def smooth_state_score(score: float) -> float:
    """返回平滑后的状态分数。当前实现是 synthetic_state_inference_sanity 接口冻结用的恒等映射。"""
    return score
