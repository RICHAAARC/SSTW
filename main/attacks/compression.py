"""定义 real_video_latent_transfer_check 压缩攻击代理。"""

from __future__ import annotations

COMPRESSION_ATTACKS = {"h264_compression", "h265_compression"}


def is_compression_attack(attack_name: str) -> bool:
    """判断攻击是否属于压缩攻击。"""
    return attack_name in COMPRESSION_ATTACKS
