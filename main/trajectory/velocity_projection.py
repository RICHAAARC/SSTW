"""实现 trajectory_observation_core_probe velocity projection statistic 的轻量代理。"""

from __future__ import annotations

ATTACK_TRAJECTORY_BONUS = {
    "no_attack": 0.030,
    "temporal_crop": 0.026,
    "local_clip": 0.034,
    "regular_frame_dropping": 0.028,
    "irregular_frame_dropping": 0.032,
    "frame_duplication": 0.024,
    "speed_change": 0.036,
    "frame_rate_resampling": 0.030,
    "segment_jump": 0.027,
    "latent_gaussian_noise": 0.033,
}


def velocity_projection_response(sample_role: str, attack_name: str, key_conditioned: bool, sample_index: int) -> float:
    """计算 key-conditioned velocity projection 响应。

    该响应保留 H0/H1 和 key condition 分离, 但不随静态 payload 难度单调变化, 以避免
    trajectory observation 被人为构造成 payload evidence 的线性副本或反向副本。
    """
    positive = sample_role.endswith("positive")
    base = 0.18 if positive else 0.08
    key_gain = 0.11 if positive and key_conditioned else 0.0
    attack_bonus = ATTACK_TRAJECTORY_BONUS.get(attack_name, 0.030)
    orthogonal_jitter = ((sample_index % 3) - 1) * 0.018
    return round(max(base + key_gain + attack_bonus + orthogonal_jitter, 0.0), 6)
