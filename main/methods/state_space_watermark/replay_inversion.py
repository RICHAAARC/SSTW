"""使用生成模型真实 velocity 函数执行 Flow inversion、replay 与不确定性估计。"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from statistics import mean
from typing import Any, Callable, Iterable, Sequence


VelocityFunction = Callable[[Any, Any, int], Any]


@dataclass(frozen=True)
class FlowSchedulePoint:
    """绑定模型 timestep 与 Flow scheduler sigma。"""

    timestep: Any
    sigma: float


@dataclass(frozen=True)
class ReplayTrajectory:
    """保存一次真实模型 replay 的状态序列和循环误差。"""

    reverse_states: tuple[Any, ...]
    forward_states: tuple[Any, ...]
    cycle_relative_error: float
    reverse_step_count: int
    forward_step_count: int


@dataclass(frozen=True)
class ReplayUncertainty:
    """保存多次 replay 的误差、方差与可靠性权重。"""

    cycle_error_mean: float
    cycle_error_maximum: float
    endpoint_ensemble_variance: float
    replay_reliability: float
    replay_count: int

    def as_dict(self) -> dict[str, float | int | str]:
        """转换为正式 replay records 字段。"""

        return {
            "replay_inversion_status": "ready",
            "replay_cycle_error_mean": round(self.cycle_error_mean, 8),
            "replay_cycle_error_maximum": round(self.cycle_error_maximum, 8),
            "replay_endpoint_ensemble_variance": round(self.endpoint_ensemble_variance, 8),
            "replay_uncertainty_mean": round(1.0 - self.replay_reliability, 8),
            "replay_reliability_weight": round(self.replay_reliability, 8),
            "replay_ensemble_count": self.replay_count,
            "replay_trajectory_source": "attacked_video_endpoint_model_velocity_inversion",
        }


def _relative_error(left: Any, right: Any) -> float:
    difference = (left.detach().float() - right.detach().float()).norm()
    denominator = right.detach().float().norm().clamp_min(1e-8)
    return float((difference / denominator).item())


def reverse_flow_trajectory(
    endpoint_latent: Any,
    schedule: Sequence[FlowSchedulePoint],
    velocity_function: VelocityFunction,
) -> list[Any]:
    """从 sigma 最低的 endpoint 沿真实模型 velocity 反向积分到初始噪声。"""

    if len(schedule) < 2:
        raise ValueError("Flow inversion 至少需要两个 schedule point")
    current = endpoint_latent
    reverse_states = [current.detach().clone()]
    for reverse_index in range(len(schedule) - 2, -1, -1):
        target = schedule[reverse_index]
        source = schedule[reverse_index + 1]
        delta_sigma = float(target.sigma) - float(source.sigma)
        # 反向 Euler 的当前状态位于 source sigma, 因而模型时间条件必须使用
        # source timestep。若使用 target timestep, 会把相邻网格端点错配并放大循环误差。
        velocity = velocity_function(current, source.timestep, reverse_index)
        current = current + delta_sigma * velocity
        reverse_states.append(current.detach().clone())
    reverse_states.reverse()
    return reverse_states


def forward_flow_replay(
    initial_latent: Any,
    schedule: Sequence[FlowSchedulePoint],
    velocity_function: VelocityFunction,
) -> list[Any]:
    """从恢复的初始状态沿同一真实 velocity 函数前向重放到 endpoint。"""

    if len(schedule) < 2:
        raise ValueError("Flow replay 至少需要两个 schedule point")
    current = initial_latent
    states = [current.detach().clone()]
    for step_index in range(len(schedule) - 1):
        source = schedule[step_index]
        target = schedule[step_index + 1]
        delta_sigma = float(target.sigma) - float(source.sigma)
        velocity = velocity_function(current, source.timestep, step_index)
        current = current + delta_sigma * velocity
        states.append(current.detach().clone())
    return states


def run_flow_inversion_and_replay(
    endpoint_latent: Any,
    schedule: Sequence[FlowSchedulePoint],
    velocity_function: VelocityFunction,
) -> ReplayTrajectory:
    """执行完整 reverse inversion 与 forward replay 循环。"""

    reverse_states = reverse_flow_trajectory(endpoint_latent, schedule, velocity_function)
    forward_states = forward_flow_replay(reverse_states[0], schedule, velocity_function)
    return ReplayTrajectory(
        reverse_states=tuple(reverse_states),
        forward_states=tuple(forward_states),
        cycle_relative_error=_relative_error(forward_states[-1], endpoint_latent),
        reverse_step_count=len(reverse_states) - 1,
        forward_step_count=len(forward_states) - 1,
    )


def estimate_replay_uncertainty(replays: Iterable[ReplayTrajectory]) -> ReplayUncertainty:
    """根据真实循环误差和多配置 endpoint 方差估计 replay 可靠性。"""

    import torch

    rows = list(replays)
    if not rows:
        raise ValueError("replay uncertainty 至少需要一次 replay")
    errors = [float(row.cycle_relative_error) for row in rows]
    endpoints = torch.stack([row.forward_states[-1].detach().float() for row in rows], dim=0)
    variance = float(endpoints.var(dim=0, unbiased=False).mean().item()) if len(rows) > 1 else 0.0
    uncertainty = mean(errors) + variance**0.5
    reliability = exp(-max(0.0, uncertainty))
    return ReplayUncertainty(
        cycle_error_mean=mean(errors),
        cycle_error_maximum=max(errors),
        endpoint_ensemble_variance=variance,
        replay_reliability=reliability,
        replay_count=len(rows),
    )
