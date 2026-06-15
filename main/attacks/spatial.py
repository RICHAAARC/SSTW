"""定义 B2 空间攻击代理。"""

from __future__ import annotations

SPATIAL_ATTACKS = {"spatial_resize", "crop_resize"}


def is_spatial_attack(attack_name: str) -> bool:
    """判断攻击是否属于空间攻击。"""
    return attack_name in SPATIAL_ATTACKS
