"""执行 key 无关 Flow inversion、候选假设 replay 与不确定性估计。"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from statistics import mean, pstdev
from typing import Any, Callable, Iterable, Sequence


VelocityFunction = Callable[[Any, Any, int], Any]


@dataclass(frozen=True)
class FlowSchedulePoint:
    """绑定模型 timestep 与 Flow scheduler sigma。"""

    timestep: Any
    sigma: float


@dataclass(frozen=True)
class ReplayTrajectory:
    """保存固定反演路径、null replay 和候选 key replay。"""

    reverse_states: tuple[Any, ...]
    forward_states: tuple[Any, ...]
    null_forward_states: tuple[Any, ...]
    candidate_cycle_relative_error: float
    null_cycle_relative_error: float
    replay_log_likelihood_ratio: float
    reverse_step_count: int
    forward_step_count: int

    @property
    def cycle_relative_error(self) -> float:
        """保留旧调用方使用的候选假设循环误差只读别名。"""

        return self.candidate_cycle_relative_error


@dataclass(frozen=True)
class ReplayUncertainty:
    """保存多网格假设误差、似然比离散度与可靠性权重。"""

    cycle_error_mean: float
    cycle_error_maximum: float
    null_cycle_error_mean: float
    log_likelihood_ratio_mean: float
    log_likelihood_ratio_standard_deviation: float
    endpoint_ensemble_variance: float
    replay_reliability: float
    replay_count: int

    def as_dict(self) -> dict[str, float | int | str]:
        """转换为正式 replay records 字段。"""

        return {
            "replay_inversion_status": "ready",
            "replay_cycle_error_mean": round(self.cycle_error_mean, 8),
            "replay_cycle_error_maximum": round(self.cycle_error_maximum, 8),
            "replay_null_cycle_error_mean": round(self.null_cycle_error_mean, 8),
            "replay_log_likelihood_ratio_mean": round(self.log_likelihood_ratio_mean, 8),
            "replay_log_likelihood_ratio_standard_deviation": round(
                self.log_likelihood_ratio_standard_deviation,
                8,
            ),
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


def run_key_independent_inversion_hypothesis(
    endpoint_latent: Any,
    schedule: Sequence[FlowSchedulePoint],
    inversion_velocity_function: VelocityFunction,
    candidate_velocity_function: VelocityFunction,
) -> ReplayTrajectory:
    """在同一固定初态上比较 null 与候选 key 的 forward hypothesis。

    reverse inversion 只使用不含水印约束的基础模型 velocity, 因而候选 key
    无法改变观测路径或初始状态。候选 key 只参与 forward prediction, 最终证据
    来自候选 endpoint 对真实 attacked-video endpoint 的解释能力是否优于 null。
    """

    reverse_states = reverse_flow_trajectory(
        endpoint_latent,
        schedule,
        inversion_velocity_function,
    )
    null_forward_states = forward_flow_replay(
        reverse_states[0],
        schedule,
        inversion_velocity_function,
    )
    candidate_forward_states = forward_flow_replay(
        reverse_states[0],
        schedule,
        candidate_velocity_function,
    )
    null_error = _relative_error(null_forward_states[-1], endpoint_latent)
    candidate_error = _relative_error(candidate_forward_states[-1], endpoint_latent)
    epsilon = 1e-8
    log_likelihood_ratio = log((null_error + epsilon) / (candidate_error + epsilon))
    return ReplayTrajectory(
        reverse_states=tuple(reverse_states),
        forward_states=tuple(candidate_forward_states),
        null_forward_states=tuple(null_forward_states),
        candidate_cycle_relative_error=candidate_error,
        null_cycle_relative_error=null_error,
        replay_log_likelihood_ratio=log_likelihood_ratio,
        reverse_step_count=len(reverse_states) - 1,
        forward_step_count=len(candidate_forward_states) - 1,
    )


def evaluate_candidate_on_fixed_inversion(
    endpoint_latent: Any,
    schedule: Sequence[FlowSchedulePoint],
    fixed_trajectory: ReplayTrajectory,
    candidate_velocity_function: VelocityFunction,
) -> ReplayTrajectory:
    """在已有 key 无关反演路径上只执行新的候选 forward hypothesis。

    该函数用于同一 clean video 的多 key calibration。reverse states 与 null replay
    完全复用第一次模型反演, 每个 trial key 只能改变候选 forward prediction,
    从而同时避免循环证据和把重复 key trial 误当独立视频。
    """

    if len(fixed_trajectory.reverse_states) != len(schedule):
        raise ValueError("固定反演路径与 Flow schedule 长度不一致")
    candidate_forward_states = forward_flow_replay(
        fixed_trajectory.reverse_states[0],
        schedule,
        candidate_velocity_function,
    )
    candidate_error = _relative_error(candidate_forward_states[-1], endpoint_latent)
    null_error = float(fixed_trajectory.null_cycle_relative_error)
    epsilon = 1e-8
    return ReplayTrajectory(
        reverse_states=fixed_trajectory.reverse_states,
        forward_states=tuple(candidate_forward_states),
        null_forward_states=fixed_trajectory.null_forward_states,
        candidate_cycle_relative_error=candidate_error,
        null_cycle_relative_error=null_error,
        replay_log_likelihood_ratio=log((null_error + epsilon) / (candidate_error + epsilon)),
        reverse_step_count=fixed_trajectory.reverse_step_count,
        forward_step_count=len(candidate_forward_states) - 1,
    )


def run_flow_inversion_and_replay(
    endpoint_latent: Any,
    schedule: Sequence[FlowSchedulePoint],
    velocity_function: VelocityFunction,
) -> ReplayTrajectory:
    """兼容单 velocity round-trip 测试, 正式检测应使用独立候选假设接口。"""

    return run_key_independent_inversion_hypothesis(
        endpoint_latent,
        schedule,
        velocity_function,
        velocity_function,
    )


def estimate_replay_uncertainty(replays: Iterable[ReplayTrajectory]) -> ReplayUncertainty:
    """根据真实循环误差和多配置 endpoint 方差估计 replay 可靠性。"""

    import torch

    rows = list(replays)
    if not rows:
        raise ValueError("replay uncertainty 至少需要一次 replay")
    errors = [float(row.candidate_cycle_relative_error) for row in rows]
    null_errors = [float(row.null_cycle_relative_error) for row in rows]
    likelihood_ratios = [float(row.replay_log_likelihood_ratio) for row in rows]
    endpoints = torch.stack([row.forward_states[-1].detach().float() for row in rows], dim=0)
    variance = float(endpoints.var(dim=0, unbiased=False).mean().item()) if len(rows) > 1 else 0.0
    likelihood_dispersion = pstdev(likelihood_ratios) if len(likelihood_ratios) > 1 else 0.0
    uncertainty = mean(errors) + variance**0.5 + likelihood_dispersion
    reliability = exp(-max(0.0, uncertainty))
    return ReplayUncertainty(
        cycle_error_mean=mean(errors),
        cycle_error_maximum=max(errors),
        null_cycle_error_mean=mean(null_errors),
        log_likelihood_ratio_mean=mean(likelihood_ratios),
        log_likelihood_ratio_standard_deviation=likelihood_dispersion,
        endpoint_ensemble_variance=variance,
        replay_reliability=reliability,
        replay_count=len(rows),
    )
