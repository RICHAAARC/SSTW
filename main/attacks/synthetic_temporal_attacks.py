"""定义第一阶段 synthetic temporal attack 矩阵。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SyntheticAttack:
    """记录攻击名称、强度和是否属于复杂时间扰动。"""

    attack_name: str
    attack_strength: float
    complex_temporal: bool


def default_synthetic_attacks() -> tuple[SyntheticAttack, ...]:
    """返回第一阶段必须覆盖的攻击矩阵。"""
    return (
        SyntheticAttack("no_attack", 0.0, False),
        SyntheticAttack("temporal_crop", 0.25, True),
        SyntheticAttack("local_clip", 0.25, True),
        SyntheticAttack("regular_frame_dropping", 0.25, False),
        SyntheticAttack("irregular_frame_dropping", 0.30, True),
        SyntheticAttack("frame_duplication", 0.25, True),
        SyntheticAttack("speed_change", 1.25, True),
        SyntheticAttack("frame_rate_resampling", 0.50, True),
        SyntheticAttack("segment_jump", 0.35, True),
        SyntheticAttack("latent_gaussian_noise", 0.10, False),
    )
