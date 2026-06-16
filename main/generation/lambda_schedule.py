"""提供 sampling-time weak constraint 的 lambda schedule。"""

from __future__ import annotations


def build_lambda_schedule(
    schedule_id: str,
    num_steps: int,
    lambda_max: float,
    time_window: tuple[float, float] = (0.25, 0.75),
) -> list[float]:
    """构造有界 lambda schedule。

    该函数属于通用工程写法。输入只包含 schedule 名称、步数、最大强度和时间窗口, 因此可复用于不同视频生成后端。
    项目特定设计在于: 默认推荐中期采样约束, 因为 B6 假设中期 trajectory 更能反映生成语义与运动结构。
    """
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    lower, upper = time_window
    if not 0.0 <= lower <= upper <= 1.0:
        raise ValueError("time_window must satisfy 0 <= lower <= upper <= 1")

    values: list[float] = []
    for step_index in range(num_steps):
        progress = step_index / max(num_steps - 1, 1)
        in_window = lower <= progress <= upper
        if schedule_id == "mid_window_weak_constraint":
            center = (lower + upper) / 2.0
            width = max((upper - lower) / 2.0, 1e-6)
            shape = max(0.0, 1.0 - abs(progress - center) / width)
            value = lambda_max * shape if in_window else 0.0
        elif schedule_id == "early_only_constraint":
            value = lambda_max if progress <= upper else 0.0
        elif schedule_id == "late_only_constraint":
            value = lambda_max if progress >= lower else 0.0
        elif schedule_id == "constant_weak_constraint":
            value = lambda_max if in_window else 0.0
        elif schedule_id == "strong_lambda_constraint":
            value = lambda_max * 1.8 if in_window else 0.0
        else:
            raise ValueError(f"unknown lambda schedule: {schedule_id}")
        values.append(round(float(value), 6))
    return values


def active_step_count(lambda_values: list[float]) -> int:
    """统计实际启用约束的采样步数。"""
    return sum(1 for value in lambda_values if value > 0.0)
