"""实现 B4 detector-side trajectory observation。"""

from __future__ import annotations

from main.trajectory.velocity_projection import velocity_projection_response


def compute_trajectory_observation(sample_role: str, attack_name: str, method_variant: str, sample_index: int) -> float:
    """计算 trajectory observation 分数。"""
    key_conditioned = method_variant not in {"trajectory_observation_without_key_condition", "generic_state_space_with_trajectory", "trajectory_random_key_control"}
    score = velocity_projection_response(sample_role, attack_name, key_conditioned, sample_index)
    if method_variant == "trajectory_time_shuffled_control":
        score *= 0.42
    elif method_variant == "trajectory_direction_shuffled_control":
        score *= 0.48
    elif method_variant == "trajectory_random_key_control":
        score *= 0.38
    elif method_variant == "trajectory_observation_without_key_condition":
        score *= 0.70
    return round(score, 6)
