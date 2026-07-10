"""从真实 endpoint、velocity、path 与 replay 观测推断 Flow 水印状态后验。"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log, pi
from statistics import mean
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class FlowEvidenceObservation:
    """描述一个时间位置上的多源水印观测。"""

    endpoint_score: float
    velocity_score: float
    path_score: float
    path_endpoint_consistency: float
    coverage_ratio: float
    replay_reliability: float
    time_grid_reliability: float
    flow_phase: float


@dataclass(frozen=True)
class FlowEvidenceCalibration:
    """保存只允许从 calibration negative 冻结的归一化与门禁参数。"""

    means: Mapping[str, float]
    standard_deviations: Mapping[str, float]
    admissibility_thresholds: Mapping[str, float]
    process_variance: float = 0.05
    observation_variance: float = 0.1


@dataclass(frozen=True)
class FlowStatePosterior:
    """保存过滤后的可解释状态和保守检测分数。"""

    phase_state: float
    endpoint_state: float
    posterior_confidence: float
    temporal_disturbance_state: float
    path_consistency_state: float
    velocity_consistency_state: float
    replay_reliability_state: float
    time_grid_reliability_state: float
    posterior_entropy: float
    admissible: bool
    admissibility_failures: tuple[str, ...]
    unconstrained_score: float
    conservative_score: float

    def as_dict(self) -> dict[str, Any]:
        """转换为正式 detector record 字段。"""

        return {
            "flow_phase_state": round(self.phase_state, 8),
            "flow_endpoint_state": round(self.endpoint_state, 8),
            "flow_posterior_confidence": round(self.posterior_confidence, 8),
            "flow_temporal_disturbance_state": round(self.temporal_disturbance_state, 8),
            "flow_path_consistency_state": round(self.path_consistency_state, 8),
            "flow_velocity_consistency_state": round(self.velocity_consistency_state, 8),
            "flow_replay_reliability_state": round(self.replay_reliability_state, 8),
            "flow_time_grid_reliability_state": round(self.time_grid_reliability_state, 8),
            "flow_state_posterior_entropy": round(self.posterior_entropy, 8),
            "flow_state_admissibility_status": "pass" if self.admissible else "blocked",
            "flow_state_admissibility_failures": list(self.admissibility_failures),
            "S_final_unconstrained": round(self.unconstrained_score, 8),
            "S_final_conservative": round(self.conservative_score, 8),
        }


def _sigmoid(value: float) -> float:
    value = max(-60.0, min(60.0, value))
    return 1.0 / (1.0 + exp(-value))


def _normalized_score(name: str, value: float, calibration: FlowEvidenceCalibration) -> float:
    center = float(calibration.means.get(name, 0.0))
    scale = max(float(calibration.standard_deviations.get(name, 1.0)), 1e-8)
    return _sigmoid((float(value) - center) / scale)


def _threshold(calibration: FlowEvidenceCalibration, name: str, default: float) -> float:
    return float(calibration.admissibility_thresholds.get(name, default))


def infer_flow_state_posterior(
    observations: Iterable[FlowEvidenceObservation],
    calibration: FlowEvidenceCalibration,
) -> FlowStatePosterior:
    """执行一维 Kalman-like 证据过滤并输出八个可解释状态。

    该推断只消费真实观测和冻结 calibration 参数, 不读取 sample role、test label
    或攻击强度标签, 因而不会像早期 deterministic sanity 实现一样由标签直接生成分数。
    """

    rows = list(observations)
    if not rows:
        raise ValueError("Flow state posterior 至少需要一个观测")
    posterior_mean = 0.5
    posterior_variance = 1.0
    endpoint_values: list[float] = []
    velocity_values: list[float] = []
    path_values: list[float] = []
    consistency_values: list[float] = []
    replay_values: list[float] = []
    grid_values: list[float] = []
    phase_values: list[float] = []
    coverage_values: list[float] = []

    for observation in rows:
        endpoint = _normalized_score("endpoint", observation.endpoint_score, calibration)
        velocity = _normalized_score("velocity", observation.velocity_score, calibration)
        path = _normalized_score("path", observation.path_score, calibration)
        consistency = max(0.0, min(1.0, observation.path_endpoint_consistency))
        replay = max(0.0, min(1.0, observation.replay_reliability))
        grid = max(0.0, min(1.0, observation.time_grid_reliability))
        coverage = max(0.0, min(1.0, observation.coverage_ratio))
        reliability = max(1e-4, replay * grid * coverage)
        measurement = min(endpoint, velocity, path, consistency)
        predicted_variance = posterior_variance + max(calibration.process_variance, 1e-8)
        effective_observation_variance = max(calibration.observation_variance, 1e-8) / reliability
        kalman_gain = predicted_variance / (predicted_variance + effective_observation_variance)
        posterior_mean = posterior_mean + kalman_gain * (measurement - posterior_mean)
        posterior_variance = max(1e-8, (1.0 - kalman_gain) * predicted_variance)

        endpoint_values.append(endpoint)
        velocity_values.append(velocity)
        path_values.append(path)
        consistency_values.append(consistency)
        replay_values.append(replay)
        grid_values.append(grid)
        phase_values.append(max(0.0, min(1.0, observation.flow_phase)))
        coverage_values.append(coverage)

    endpoint_state = mean(endpoint_values)
    velocity_state = mean(velocity_values)
    path_state = mean(path_values)
    consistency_state = mean(consistency_values)
    replay_state = mean(replay_values)
    grid_state = mean(grid_values)
    coverage_state = mean(coverage_values)
    posterior_confidence = 1.0 / (1.0 + posterior_variance)
    posterior_entropy = 0.5 * log(2.0 * pi * exp(1.0) * posterior_variance)
    disturbance_state = 1.0 - mean([consistency_state, replay_state, grid_state])

    checks = {
        "endpoint": endpoint_state >= _threshold(calibration, "endpoint", 0.5),
        "path": path_state >= _threshold(calibration, "path", 0.5),
        "path_endpoint_consistency": consistency_state
        >= _threshold(calibration, "path_endpoint_consistency", 0.5),
        "posterior_confidence": posterior_confidence
        >= _threshold(calibration, "posterior_confidence", 0.5),
        "coverage": coverage_state >= _threshold(calibration, "coverage", 0.5),
        "replay_reliability": replay_state >= _threshold(calibration, "replay_reliability", 0.5),
        "time_grid_reliability": grid_state >= _threshold(calibration, "time_grid_reliability", 0.5),
    }
    failures = tuple(name for name, passed in checks.items() if not passed)
    admissible = not failures
    unconstrained = min(
        endpoint_state,
        velocity_state,
        path_state,
        consistency_state,
        posterior_mean,
        replay_state,
        grid_state,
    )
    conservative = unconstrained
    if not admissible:
        conservative = 0.0
    return FlowStatePosterior(
        phase_state=mean(phase_values),
        endpoint_state=endpoint_state,
        posterior_confidence=posterior_confidence,
        temporal_disturbance_state=disturbance_state,
        path_consistency_state=min(path_state, consistency_state),
        velocity_consistency_state=velocity_state,
        replay_reliability_state=replay_state,
        time_grid_reliability_state=grid_state,
        posterior_entropy=posterior_entropy,
        admissible=admissible,
        admissibility_failures=failures,
        unconstrained_score=unconstrained,
        conservative_score=conservative,
    )
