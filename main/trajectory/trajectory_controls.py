"""实现 trajectory_observation_core_probe trajectory controls。"""

from __future__ import annotations

CONTROL_TO_VARIANT = {
    "random_key": "trajectory_random_key_control",
    "time_shuffle": "trajectory_time_shuffled_control",
    "direction_shuffle": "trajectory_direction_shuffled_control",
}


def control_variant(control_type: str) -> str:
    """返回 control 类型对应的方法变体。"""
    return CONTROL_TO_VARIANT[control_type]


def control_status(main_score: float, control_score: float) -> str:
    """判断 control 是否被主 trajectory 观测抑制。"""
    return "suppressed" if control_score < main_score else "not_suppressed"
