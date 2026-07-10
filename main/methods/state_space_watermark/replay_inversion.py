"""执行 key 无关 Flow inversion、候选假设 replay 与不确定性估计。"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, pi
from statistics import mean, pstdev
from typing import Any, Callable, Iterable, Sequence


VelocityFunction = Callable[[Any, Any, int], Any]

REPLAY_GAUSSIAN_LIKELIHOOD_MODEL_ID = (
    "endpoint_energy_scaled_isotropic_gaussian_per_latent_dimension"
)


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
    candidate_residual_mean_squared_error: float
    null_residual_mean_squared_error: float
    observation_noise_variance: float
    candidate_log_likelihood_per_dimension: float
    null_log_likelihood_per_dimension: float
    replay_likelihood_model_id: str
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
    observation_noise_variance_mean: float
    candidate_log_likelihood_per_dimension_mean: float
    null_log_likelihood_per_dimension_mean: float
    replay_likelihood_model_id: str

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
            "replay_observation_noise_variance_mean": round(
                self.observation_noise_variance_mean,
                10,
            ),
            "replay_candidate_log_likelihood_per_dimension_mean": round(
                self.candidate_log_likelihood_per_dimension_mean,
                8,
            ),
            "replay_null_log_likelihood_per_dimension_mean": round(
                self.null_log_likelihood_per_dimension_mean,
                8,
            ),
            "replay_likelihood_model_id": self.replay_likelihood_model_id,
        }


@dataclass(frozen=True)
class ReplayGaussianLikelihoodConfig:
    """定义 endpoint replay 残差的各向同性高斯观测模型。"""

    relative_observation_noise_standard_deviation: float = 0.05
    minimum_observation_noise_variance: float = 1e-8
    likelihood_model_id: str = REPLAY_GAUSSIAN_LIKELIHOOD_MODEL_ID


@dataclass(frozen=True)
class ReplayGaussianLikelihood:
    """保存候选与 null replay 在同一噪声模型下的对数似然。"""

    candidate_residual_mean_squared_error: float
    null_residual_mean_squared_error: float
    observation_noise_variance: float
    candidate_log_likelihood_per_dimension: float
    null_log_likelihood_per_dimension: float
    log_likelihood_ratio_per_dimension: float
    likelihood_model_id: str


def gaussian_replay_residual_likelihood(
    candidate_endpoint: Any,
    null_endpoint: Any,
    observed_endpoint: Any,
    *,
    config: ReplayGaussianLikelihoodConfig | None = None,
) -> ReplayGaussianLikelihood:
    """按预注册高斯残差模型计算候选相对 null 的逐维真实 LLR。

    方差只由 attacked-video endpoint 能量与预注册相对噪声比例确定, 不使用
    positive/test 标签，也不按候选 key 单独调节。候选与 null 共享同一方差，
    因而 LLR 是两个明确概率模型的对数似然差，而不是误差比改名。
    """

    config = config or ReplayGaussianLikelihoodConfig()
    relative_std = float(config.relative_observation_noise_standard_deviation)
    if relative_std <= 0.0:
        raise ValueError("replay 相对观测噪声标准差必须为正数")
    observed = observed_endpoint.detach().float()
    candidate_residual = candidate_endpoint.detach().float() - observed
    null_residual = null_endpoint.detach().float() - observed
    candidate_mse = float(candidate_residual.pow(2).mean().item())
    null_mse = float(null_residual.pow(2).mean().item())
    endpoint_energy = float(observed.pow(2).mean().item())
    variance = max(
        float(config.minimum_observation_noise_variance),
        endpoint_energy * relative_std**2,
    )
    normalizer = log(2.0 * pi * variance)
    candidate_log_likelihood = -0.5 * (candidate_mse / variance + normalizer)
    null_log_likelihood = -0.5 * (null_mse / variance + normalizer)
    return ReplayGaussianLikelihood(
        candidate_residual_mean_squared_error=candidate_mse,
        null_residual_mean_squared_error=null_mse,
        observation_noise_variance=variance,
        candidate_log_likelihood_per_dimension=candidate_log_likelihood,
        null_log_likelihood_per_dimension=null_log_likelihood,
        log_likelihood_ratio_per_dimension=candidate_log_likelihood - null_log_likelihood,
        likelihood_model_id=config.likelihood_model_id,
    )


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
    likelihood = gaussian_replay_residual_likelihood(
        candidate_forward_states[-1],
        null_forward_states[-1],
        endpoint_latent,
    )
    return ReplayTrajectory(
        reverse_states=tuple(reverse_states),
        forward_states=tuple(candidate_forward_states),
        null_forward_states=tuple(null_forward_states),
        candidate_cycle_relative_error=candidate_error,
        null_cycle_relative_error=null_error,
        replay_log_likelihood_ratio=likelihood.log_likelihood_ratio_per_dimension,
        candidate_residual_mean_squared_error=(
            likelihood.candidate_residual_mean_squared_error
        ),
        null_residual_mean_squared_error=likelihood.null_residual_mean_squared_error,
        observation_noise_variance=likelihood.observation_noise_variance,
        candidate_log_likelihood_per_dimension=(
            likelihood.candidate_log_likelihood_per_dimension
        ),
        null_log_likelihood_per_dimension=likelihood.null_log_likelihood_per_dimension,
        replay_likelihood_model_id=likelihood.likelihood_model_id,
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
    likelihood = gaussian_replay_residual_likelihood(
        candidate_forward_states[-1],
        fixed_trajectory.null_forward_states[-1],
        endpoint_latent,
    )
    return ReplayTrajectory(
        reverse_states=fixed_trajectory.reverse_states,
        forward_states=tuple(candidate_forward_states),
        null_forward_states=fixed_trajectory.null_forward_states,
        candidate_cycle_relative_error=candidate_error,
        null_cycle_relative_error=null_error,
        replay_log_likelihood_ratio=likelihood.log_likelihood_ratio_per_dimension,
        candidate_residual_mean_squared_error=(
            likelihood.candidate_residual_mean_squared_error
        ),
        null_residual_mean_squared_error=likelihood.null_residual_mean_squared_error,
        observation_noise_variance=likelihood.observation_noise_variance,
        candidate_log_likelihood_per_dimension=(
            likelihood.candidate_log_likelihood_per_dimension
        ),
        null_log_likelihood_per_dimension=likelihood.null_log_likelihood_per_dimension,
        replay_likelihood_model_id=likelihood.likelihood_model_id,
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
    observation_variances = [float(row.observation_noise_variance) for row in rows]
    candidate_log_likelihoods = [
        float(row.candidate_log_likelihood_per_dimension) for row in rows
    ]
    null_log_likelihoods = [float(row.null_log_likelihood_per_dimension) for row in rows]
    normalized_candidate_residuals = [
        float(row.candidate_residual_mean_squared_error)
        / max(float(row.observation_noise_variance), 1e-12)
        for row in rows
    ]
    endpoint_energy = float(
        torch.stack([row.forward_states[-1].detach().float().pow(2).mean() for row in rows])
        .mean()
        .item()
    )
    numerical_instability = variance**0.5 / max(endpoint_energy**0.5, 1e-8)
    reliability = exp(-0.5 * mean(normalized_candidate_residuals)) * exp(
        -max(0.0, numerical_instability + likelihood_dispersion)
    )
    return ReplayUncertainty(
        cycle_error_mean=mean(errors),
        cycle_error_maximum=max(errors),
        null_cycle_error_mean=mean(null_errors),
        log_likelihood_ratio_mean=mean(likelihood_ratios),
        log_likelihood_ratio_standard_deviation=likelihood_dispersion,
        endpoint_ensemble_variance=variance,
        replay_reliability=reliability,
        replay_count=len(rows),
        observation_noise_variance_mean=mean(observation_variances),
        candidate_log_likelihood_per_dimension_mean=mean(candidate_log_likelihoods),
        null_log_likelihood_per_dimension_mean=mean(null_log_likelihoods),
        replay_likelihood_model_id=rows[0].replay_likelihood_model_id,
    )
