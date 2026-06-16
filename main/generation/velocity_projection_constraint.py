"""提供 sampling-time weak constraint 的投影算子。"""

from __future__ import annotations

from math import sqrt


def _l2_norm(values: list[float]) -> float:
    """计算向量 L2 范数。"""
    return sqrt(sum(value * value for value in values))


def normalize_direction(direction: list[float]) -> list[float]:
    """归一化投影方向, 零向量会触发显式错误。"""
    norm = _l2_norm(direction)
    if norm == 0.0:
        raise ValueError("constraint direction must be non-zero")
    return [value / norm for value in direction]


def project_velocity_toward_direction(
    velocity: list[float],
    direction: list[float],
    lambda_value: float,
    norm_budget: float,
) -> list[float]:
    """把当前速度弱投影到 key-conditioned tubelet 方向。

    该函数属于项目特定写法。B6 的约束不是强制覆盖采样速度, 而是在范数预算内施加弱偏置, 以避免破坏视觉质量、运动连续性和语义一致性。
    """
    if len(velocity) != len(direction):
        raise ValueError("velocity and direction must have the same dimension")
    unit_direction = normalize_direction(direction)
    requested_step = [lambda_value * value for value in unit_direction]
    requested_norm = _l2_norm(requested_step)
    if requested_norm > norm_budget > 0.0:
        scale = norm_budget / requested_norm
        requested_step = [value * scale for value in requested_step]
    return [round(v + delta, 6) for v, delta in zip(velocity, requested_step)]


def directional_alignment(velocity: list[float], direction: list[float]) -> float:
    """计算速度与约束方向的余弦对齐分数。"""
    unit_direction = normalize_direction(direction)
    velocity_norm = _l2_norm(velocity)
    if velocity_norm == 0.0:
        return 0.0
    score = sum(v * d for v, d in zip(velocity, unit_direction)) / velocity_norm
    return round(float(score), 6)
