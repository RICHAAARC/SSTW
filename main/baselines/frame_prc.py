"""提供 `frame_prc` baseline 的轻量调用入口。"""

from __future__ import annotations

from main.methods.state_space_watermark.score import score_method


def score(sample_role: str, attack_name: str):
    """计算 `frame_prc` 在指定样本和攻击下的分数。"""
    return score_method(sample_role, attack_name, "frame_prc")
