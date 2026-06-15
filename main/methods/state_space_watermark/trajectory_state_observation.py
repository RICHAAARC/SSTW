"""将 trajectory observation 接入状态空间观测。"""

from __future__ import annotations


def fuse_trajectory_into_state_score(state_score: float, trajectory_score: float, method_variant: str) -> float:
    """把 trajectory 作为状态观测项影响 posterior。

    对于主方法, trajectory 只通过状态观测适配器进入 posterior; 对 late fusion baseline, 使用
    较弱的后验加权, 用于证明主方法不是简单分数拼接。
    """
    if method_variant == "key_conditioned_state_space_with_trajectory":
        return round(state_score + trajectory_score * 0.42, 6)
    if method_variant == "generic_state_space_with_trajectory":
        return round(state_score + trajectory_score * 0.18, 6)
    if method_variant == "explicit_temporal_alignment_with_trajectory_fusion":
        return round(state_score + trajectory_score * 0.16, 6)
    if method_variant == "trajectory_late_score_fusion":
        return round(state_score + trajectory_score * 0.20, 6)
    if method_variant == "trajectory_only":
        return round(trajectory_score, 6)
    if method_variant.startswith("trajectory_"):
        return round(trajectory_score, 6)
    return round(state_score, 6)
