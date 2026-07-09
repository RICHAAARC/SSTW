"""定义 real_video_latent_transfer_check 真实视频时间攻击代理。"""

from __future__ import annotations

TEMPORAL_ATTACKS = {"temporal_crop", "local_clip", "regular_frame_dropping", "irregular_frame_dropping", "frame_duplication", "speed_change", "frame_rate_resampling"}


def is_temporal_attack(attack_name: str) -> bool:
    """判断攻击是否属于时间扰动。"""
    return attack_name in TEMPORAL_ATTACKS
