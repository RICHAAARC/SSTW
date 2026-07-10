"""使用 calibration 拟合模型输出可靠的 SSTW 水印概率后验。"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log
from statistics import mean
from typing import Any, Iterable, Mapping


POSTERIOR_FEATURE_NAMES = (
    "endpoint_score",
    "velocity_score",
    "path_score",
    "path_endpoint_consistency",
    "replay_log_likelihood_ratio",
    "replay_reliability",
    "time_grid_reliability",
    "coverage_ratio",
)


@dataclass(frozen=True)
class FlowEvidenceObservation:
    """描述 attacked-video 固定观测与 replay hypothesis 的检测特征。"""

    endpoint_score: float
    velocity_score: float
    path_score: float
    path_endpoint_consistency: float
    replay_log_likelihood_ratio: float
    coverage_ratio: float
    replay_reliability: float
    time_grid_reliability: float
    flow_phase: float


@dataclass(frozen=True)
class CalibratedFlowPosteriorModel:
    """保存由 calibration split 拟合并交叉验证的概率后验参数。"""

    feature_names: tuple[str, ...]
    feature_means: tuple[float, ...]
    feature_scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    platt_slope: float
    platt_intercept: float
    admissibility_thresholds: Mapping[str, float]
    calibration_brier_score: float
    calibration_log_loss: float
    calibration_expected_calibration_error: float
    calibration_group_count: int
    calibration_record_count: int

    def as_dict(self) -> dict[str, Any]:
        """转换为可重建概率模型的 threshold artifact 字段。"""

        return {
            "posterior_model_type": "group_cross_fitted_logistic_with_platt_calibration",
            "posterior_reference_prior": 0.5,
            "posterior_probability_semantics": "class_balanced_calibration_reference_probability",
            "posterior_feature_names": list(self.feature_names),
            "posterior_feature_means": list(self.feature_means),
            "posterior_feature_scales": list(self.feature_scales),
            "posterior_coefficients": list(self.coefficients),
            "posterior_intercept": self.intercept,
            "posterior_platt_slope": self.platt_slope,
            "posterior_platt_intercept": self.platt_intercept,
            "posterior_admissibility_thresholds": dict(self.admissibility_thresholds),
            "posterior_calibration_brier_score": self.calibration_brier_score,
            "posterior_calibration_log_loss": self.calibration_log_loss,
            "posterior_calibration_expected_calibration_error": (
                self.calibration_expected_calibration_error
            ),
            "posterior_calibration_group_count": self.calibration_group_count,
            "posterior_calibration_record_count": self.calibration_record_count,
        }


@dataclass(frozen=True)
class FlowStatePosterior:
    """保存概率后验、数据质量状态与可解释特征。"""

    watermark_probability: float
    posterior_log_odds: float
    phase_state: float
    endpoint_state: float
    temporal_disturbance_state: float
    path_consistency_state: float
    velocity_consistency_state: float
    replay_reliability_state: float
    time_grid_reliability_state: float
    posterior_entropy: float
    admissible: bool
    admissibility_failures: tuple[str, ...]
    conservative_score: float

    def as_dict(self) -> dict[str, Any]:
        """转换为正式 detector record 字段。"""

        return {
            "flow_watermark_posterior_probability": round(self.watermark_probability, 8),
            "flow_watermark_posterior_log_odds": round(self.posterior_log_odds, 8),
            "flow_phase_state": round(self.phase_state, 8),
            "flow_endpoint_state": round(self.endpoint_state, 8),
            "flow_posterior_confidence": round(
                max(self.watermark_probability, 1.0 - self.watermark_probability),
                8,
            ),
            "flow_temporal_disturbance_state": round(self.temporal_disturbance_state, 8),
            "flow_path_consistency_state": round(self.path_consistency_state, 8),
            "flow_velocity_consistency_state": round(self.velocity_consistency_state, 8),
            "flow_replay_reliability_state": round(self.replay_reliability_state, 8),
            "flow_time_grid_reliability_state": round(self.time_grid_reliability_state, 8),
            "flow_state_posterior_entropy": round(self.posterior_entropy, 8),
            "flow_state_admissibility_status": "pass" if self.admissible else "blocked",
            "flow_state_admissibility_failures": list(self.admissibility_failures),
            "S_final_unconstrained": round(self.watermark_probability, 8),
            "S_final_conservative": round(self.conservative_score, 8),
        }


def _sigmoid(value: float) -> float:
    value = max(-60.0, min(60.0, float(value)))
    return 1.0 / (1.0 + exp(-value))


def posterior_feature_vector(observation: FlowEvidenceObservation) -> tuple[float, ...]:
    """按冻结顺序提取概率模型特征。"""

    return (
        float(observation.endpoint_score),
        float(observation.velocity_score),
        float(observation.path_score),
        float(observation.path_endpoint_consistency),
        float(observation.replay_log_likelihood_ratio),
        float(observation.replay_reliability),
        float(observation.time_grid_reliability),
        float(observation.coverage_ratio),
    )


def infer_flow_state_posterior(
    observations: Iterable[FlowEvidenceObservation],
    model: CalibratedFlowPosteriorModel,
) -> FlowStatePosterior:
    """使用冻结 logistic + Platt 模型计算 attacked-video 水印概率。

    所有参数都来自 calibration split。函数不读取 sample role、test label 或攻击
    强度标签, 也不包含人工指定的过程方差和观测方差。
    """

    rows = list(observations)
    if not rows:
        raise ValueError("Flow state posterior 至少需要一个观测")
    if tuple(model.feature_names) != POSTERIOR_FEATURE_NAMES:
        raise ValueError("概率后验特征顺序与冻结模型不一致")

    vectors = [posterior_feature_vector(row) for row in rows]
    feature_values = tuple(mean(vector[index] for vector in vectors) for index in range(len(POSTERIOR_FEATURE_NAMES)))
    standardized = [
        (value - model.feature_means[index]) / max(model.feature_scales[index], 1e-8)
        for index, value in enumerate(feature_values)
    ]
    raw_logit = model.intercept + sum(
        coefficient * value
        for coefficient, value in zip(model.coefficients, standardized)
    )
    calibrated_logit = model.platt_intercept + model.platt_slope * raw_logit
    probability = _sigmoid(calibrated_logit)
    clipped = max(1e-12, min(1.0 - 1e-12, probability))
    entropy = -(clipped * log(clipped) + (1.0 - clipped) * log(1.0 - clipped))

    coverage = mean(max(0.0, min(1.0, row.coverage_ratio)) for row in rows)
    replay = mean(max(0.0, min(1.0, row.replay_reliability)) for row in rows)
    grid = mean(max(0.0, min(1.0, row.time_grid_reliability)) for row in rows)
    checks = {
        "coverage": coverage >= float(model.admissibility_thresholds.get("coverage", 0.0)),
        "replay_reliability": replay
        >= float(model.admissibility_thresholds.get("replay_reliability", 0.0)),
        "time_grid_reliability": grid
        >= float(model.admissibility_thresholds.get("time_grid_reliability", 0.0)),
    }
    failures = tuple(name for name, passed in checks.items() if not passed)
    admissible = not failures
    endpoint = mean(row.endpoint_score for row in rows)
    velocity = mean(row.velocity_score for row in rows)
    path = mean(row.path_score for row in rows)
    consistency = mean(row.path_endpoint_consistency for row in rows)
    return FlowStatePosterior(
        watermark_probability=probability,
        posterior_log_odds=calibrated_logit,
        phase_state=mean(max(0.0, min(1.0, row.flow_phase)) for row in rows),
        endpoint_state=endpoint,
        temporal_disturbance_state=1.0 - mean([consistency, replay, grid]),
        path_consistency_state=min(path, consistency),
        velocity_consistency_state=velocity,
        replay_reliability_state=replay,
        time_grid_reliability_state=grid,
        posterior_entropy=entropy,
        admissible=admissible,
        admissibility_failures=failures,
        conservative_score=probability if admissible else 0.0,
    )
