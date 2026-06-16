"""提供 sampling-time weak constraint 控制器。"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from main.generation.lambda_schedule import active_step_count, build_lambda_schedule
from main.generation.velocity_projection_constraint import directional_alignment, project_velocity_toward_direction


@dataclass(frozen=True)
class SamplingConstraintConfig:
    """描述一次 sampling-time weak constraint 的可复用配置。"""

    sampling_constraint_config_id: str
    lambda_schedule_id: str
    lambda_max: float
    lambda_time_window: tuple[float, float]
    constraint_norm_budget: float
    constraint_direction: tuple[float, ...]


def apply_sampling_constraint(
    velocities: list[list[float]],
    config: SamplingConstraintConfig,
) -> dict:
    """对一段采样速度轨迹应用弱约束, 并返回可写入 records 的统计量。

    该函数属于项目特定写法。它不直接依赖某个生成模型的 callback API, 因而可先在 preflight 中验证机制, 后续再接入 LTX / DiT 采样过程。
    """
    lambda_values = build_lambda_schedule(
        config.lambda_schedule_id,
        len(velocities),
        config.lambda_max,
        config.lambda_time_window,
    )
    direction = list(config.constraint_direction)
    before_scores = [directional_alignment(velocity, direction) for velocity in velocities]
    constrained_velocities = [
        project_velocity_toward_direction(
            velocity,
            direction,
            lambda_values[index],
            config.constraint_norm_budget,
        )
        for index, velocity in enumerate(velocities)
    ]
    after_scores = [directional_alignment(velocity, direction) for velocity in constrained_velocities]
    before_mean = mean(before_scores) if before_scores else 0.0
    after_mean = mean(after_scores) if after_scores else 0.0
    return {
        "sampling_constraint_config_id": config.sampling_constraint_config_id,
        "lambda_schedule_id": config.lambda_schedule_id,
        "lambda_max": config.lambda_max,
        "lambda_time_window": f"{config.lambda_time_window[0]}:{config.lambda_time_window[1]}",
        "constraint_apply_steps": active_step_count(lambda_values),
        "constraint_norm_budget": config.constraint_norm_budget,
        "S_trajectory_observation_before_constraint": round(before_mean, 6),
        "S_trajectory_observation_after_constraint": round(after_mean, 6),
        "trajectory_constraint_gain": round(after_mean - before_mean, 6),
        "lambda_values": lambda_values,
    }
