"""定义 B2 像素扰动攻击代理。"""

from __future__ import annotations

CORRUPTION_ATTACKS = {"gaussian_noise", "blur"}


def is_corruption_attack(attack_name: str) -> bool:
    """判断攻击是否属于像素扰动攻击。"""
    return attack_name in CORRUPTION_ATTACKS
