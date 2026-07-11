"""使用双假设线性高斯状态空间模型推断 SSTW 水印概率后验。"""

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
    "flow_phase",
)

FLOW_STATE_ADMISSIBILITY_THRESHOLD_NAMES = (
    "endpoint_score",
    "path_score",
    "path_endpoint_consistency",
    "posterior_confidence",
    "coverage",
    "replay_reliability",
    "time_grid_reliability",
    "posterior_entropy_maximum",
)

FLOW_STATE_POSTERIOR_MODEL_TYPE = (
    "dual_hypothesis_linear_gaussian_state_space_filter_rts_smoother_"
    "with_group_cross_fitted_platt_calibration"
)
FLOW_STATE_POSTERIOR_CONTRACT_VERSION = (
    "flow_phase_conditioned_reliability_heteroscedastic_v2"
)


@dataclass(frozen=True)
class FlowEvidenceObservation:
    """描述一个 Flow phase 上的 key-conditioned 状态空间观测。"""

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
class LinearGaussianFlowStateModel:
    """保存一个水印假设下可重建的线性高斯状态空间参数。

    状态转移为 ``h_t = A h_(t-1) + b + q_t``，观测模型为
    ``y_t = h_t + r_t``。矩阵由 calibration split 拟合并执行协方差收缩，
    held-out 推理只运行 Kalman filter 与 RTS smoother，不更新参数。
    """

    transition_matrix: tuple[tuple[float, ...], ...]
    transition_bias: tuple[float, ...]
    process_covariance: tuple[tuple[float, ...], ...]
    observation_covariance: tuple[tuple[float, ...], ...]
    initial_mean: tuple[float, ...]
    initial_covariance: tuple[tuple[float, ...], ...]
    training_sequence_count: int
    training_group_count: int
    training_transition_count: int
    training_transition_group_count: int
    phase_transition_matrix: tuple[tuple[float, ...], ...] = ()
    phase_transition_reference: float = 0.5
    reliability_observation_variance_scale: float = 1.0

    def as_dict(self) -> dict[str, Any]:
        """转换为可直接重建 Kalman filtering/smoothing 的参数。"""

        return {
            "state_space_dynamics_contract": (
                "phase_conditioned_transition_with_reliability_heteroscedastic_observation"
            ),
            "transition_matrix": [list(row) for row in self.transition_matrix],
            "transition_bias": list(self.transition_bias),
            "process_covariance": [list(row) for row in self.process_covariance],
            "observation_covariance": [list(row) for row in self.observation_covariance],
            "initial_mean": list(self.initial_mean),
            "initial_covariance": [list(row) for row in self.initial_covariance],
            "training_sequence_count": self.training_sequence_count,
            "training_group_count": self.training_group_count,
            "training_transition_count": self.training_transition_count,
            "training_transition_group_count": self.training_transition_group_count,
            "phase_transition_matrix": [
                list(row) for row in self.phase_transition_matrix
            ],
            "phase_transition_reference": self.phase_transition_reference,
            "reliability_observation_variance_scale": (
                self.reliability_observation_variance_scale
            ),
            "phase_conditioned_transition_configured": bool(
                self.phase_transition_matrix
            ),
            "reliability_heteroscedastic_observation_configured": (
                self.reliability_observation_variance_scale > 0.0
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LinearGaussianFlowStateModel":
        """从 governed threshold artifact 重建状态空间参数。"""

        return cls(
            transition_matrix=tuple(
                tuple(float(value) for value in row)
                for row in payload["transition_matrix"]
            ),
            transition_bias=tuple(float(value) for value in payload["transition_bias"]),
            process_covariance=tuple(
                tuple(float(value) for value in row)
                for row in payload["process_covariance"]
            ),
            observation_covariance=tuple(
                tuple(float(value) for value in row)
                for row in payload["observation_covariance"]
            ),
            initial_mean=tuple(float(value) for value in payload["initial_mean"]),
            initial_covariance=tuple(
                tuple(float(value) for value in row)
                for row in payload["initial_covariance"]
            ),
            training_sequence_count=int(payload["training_sequence_count"]),
            training_group_count=int(payload["training_group_count"]),
            training_transition_count=int(payload["training_transition_count"]),
            training_transition_group_count=int(
                payload["training_transition_group_count"]
            ),
            phase_transition_matrix=tuple(
                tuple(float(value) for value in row)
                for row in payload.get("phase_transition_matrix", ())
            ),
            phase_transition_reference=float(
                payload.get("phase_transition_reference", 0.5)
            ),
            reliability_observation_variance_scale=float(
                payload.get("reliability_observation_variance_scale", 0.0)
            ),
        )


@dataclass(frozen=True)
class CalibratedFlowPosteriorModel:
    """保存双假设状态空间模型与组外概率校准参数。"""

    feature_names: tuple[str, ...]
    feature_means: tuple[float, ...]
    feature_scales: tuple[float, ...]
    negative_state_space_model: LinearGaussianFlowStateModel
    positive_state_space_model: LinearGaussianFlowStateModel
    platt_slope: float
    platt_intercept: float
    admissibility_thresholds: Mapping[str, float]
    calibration_brier_score: float
    calibration_log_loss: float
    calibration_expected_calibration_error: float
    calibration_group_count: int
    calibration_record_count: int

    def as_dict(self) -> dict[str, Any]:
        """转换为可重建概率状态空间模型的 threshold artifact 字段。"""

        return {
            "posterior_model_type": FLOW_STATE_POSTERIOR_MODEL_TYPE,
            "posterior_model_contract_version": FLOW_STATE_POSTERIOR_CONTRACT_VERSION,
            "posterior_reference_prior": 0.5,
            "posterior_probability_semantics": (
                "class_balanced_calibration_probability_from_state_space_marginal_likelihood_ratio"
            ),
            "posterior_feature_names": list(self.feature_names),
            "posterior_feature_means": list(self.feature_means),
            "posterior_feature_scales": list(self.feature_scales),
            "posterior_negative_state_space_model": self.negative_state_space_model.as_dict(),
            "posterior_positive_state_space_model": self.positive_state_space_model.as_dict(),
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
            "posterior_flow_phase_feature_included": (
                "flow_phase" in self.feature_names
            ),
            "posterior_phase_conditioned_transition_configured": bool(
                self.negative_state_space_model.phase_transition_matrix
                and self.positive_state_space_model.phase_transition_matrix
            ),
            "posterior_reliability_heteroscedastic_observation_configured": (
                self.negative_state_space_model.reliability_observation_variance_scale
                > 0.0
                and self.positive_state_space_model.reliability_observation_variance_scale
                > 0.0
            ),
            "posterior_admissibility_context_complete": all(
                name in self.admissibility_thresholds
                for name in FLOW_STATE_ADMISSIBILITY_THRESHOLD_NAMES
            ),
        }


@dataclass(frozen=True)
class FlowStatePosterior:
    """保存概率后验、平滑状态、边际似然与证据准入状态。"""

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
    positive_log_likelihood_per_step: float
    negative_log_likelihood_per_step: float
    state_space_log_likelihood_ratio: float
    filter_step_count: int
    admissible: bool
    admissibility_failures: tuple[str, ...]
    admissibility_context_complete: bool
    endpoint_admissibility_measurement: float
    path_admissibility_measurement: float
    path_endpoint_consistency_admissibility_measurement: float
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
            "flow_state_positive_log_likelihood_per_step": round(
                self.positive_log_likelihood_per_step,
                8,
            ),
            "flow_state_negative_log_likelihood_per_step": round(
                self.negative_log_likelihood_per_step,
                8,
            ),
            "flow_state_log_likelihood_ratio": round(
                self.state_space_log_likelihood_ratio,
                8,
            ),
            "flow_state_filter_step_count": self.filter_step_count,
            "flow_state_filtering_status": "kalman_filter_ready",
            "flow_state_smoothing_status": "rauch_tung_striebel_smoother_ready",
            "flow_state_admissibility_status": (
                "pass"
                if self.admissible
                else "blocked_incomplete_context"
                if not self.admissibility_context_complete
                else "blocked"
            ),
            "flow_state_admissibility_failures": list(self.admissibility_failures),
            "flow_state_admissibility_context_complete": (
                self.admissibility_context_complete
            ),
            "flow_endpoint_admissibility_measurement": round(
                self.endpoint_admissibility_measurement,
                8,
            ),
            "flow_path_admissibility_measurement": round(
                self.path_admissibility_measurement,
                8,
            ),
            "flow_path_endpoint_consistency_admissibility_measurement": round(
                self.path_endpoint_consistency_admissibility_measurement,
                8,
            ),
            "S_final_unconstrained": round(self.watermark_probability, 8),
            "S_final_conservative": round(self.conservative_score, 8),
        }


def _sigmoid(value: float) -> float:
    value = max(-60.0, min(60.0, float(value)))
    return 1.0 / (1.0 + exp(-value))


def posterior_feature_vector(observation: FlowEvidenceObservation) -> tuple[float, ...]:
    """按冻结顺序提取一个 Flow phase 的状态空间观测向量。"""

    return (
        float(observation.endpoint_score),
        float(observation.velocity_score),
        float(observation.path_score),
        float(observation.path_endpoint_consistency),
        float(observation.replay_log_likelihood_ratio),
        float(observation.replay_reliability),
        float(observation.time_grid_reliability),
        float(observation.coverage_ratio),
        float(observation.flow_phase),
    )


def _state_space_arrays(model: LinearGaussianFlowStateModel) -> tuple[Any, ...]:
    import numpy as np

    transition = np.asarray(model.transition_matrix, dtype=np.float64)
    phase_transition = (
        np.asarray(model.phase_transition_matrix, dtype=np.float64)
        if model.phase_transition_matrix
        else np.zeros_like(transition)
    )
    return (
        transition,
        phase_transition,
        np.asarray(model.transition_bias, dtype=np.float64),
        np.asarray(model.process_covariance, dtype=np.float64),
        np.asarray(model.observation_covariance, dtype=np.float64),
        np.asarray(model.initial_mean, dtype=np.float64),
        np.asarray(model.initial_covariance, dtype=np.float64),
    )


def _kalman_filter_and_rts_smooth(
    observations: Any,
    model: LinearGaussianFlowStateModel,
    *,
    flow_phases: Any,
    replay_reliabilities: Any,
) -> tuple[float, Any]:
    """计算 phase-conditioned、可靠性异方差模型的边际似然与 RTS 状态。"""

    import numpy as np

    y = np.asarray(observations, dtype=np.float64)
    if y.ndim != 2 or len(y) == 0:
        raise ValueError("Kalman filter 至少需要一个二维观测序列")
    phases = np.asarray(flow_phases, dtype=np.float64)
    reliabilities = np.asarray(replay_reliabilities, dtype=np.float64)
    if phases.shape != (len(y),) or reliabilities.shape != (len(y),):
        raise ValueError("Flow phase、replay reliability 与观测序列长度不一致")
    if not bool(np.isfinite(phases).all()) or not bool(np.isfinite(reliabilities).all()):
        raise ValueError("Flow phase 与 replay reliability 必须为有限数值")
    if bool(np.any(phases < 0.0)) or bool(np.any(phases > 1.0)):
        raise ValueError("Flow phase 必须位于 [0, 1]")
    reliabilities = np.clip(reliabilities, 1e-4, 1.0)
    (
        transition,
        phase_transition,
        bias,
        process_cov,
        observation_cov,
        initial_mean,
        initial_cov,
    ) = (
        _state_space_arrays(model)
    )
    dimension = y.shape[1]
    if transition.shape != (dimension, dimension):
        raise ValueError("状态转移矩阵与观测维度不一致")
    transition_diagonal = np.diag(transition)
    phase_transition_diagonal = np.diag(phase_transition)
    process_variance = np.diag(process_cov)
    observation_variance = np.diag(observation_cov)
    initial_variance = np.diag(initial_cov)
    if not (
        np.allclose(transition, np.diag(transition_diagonal))
        and np.allclose(
            phase_transition,
            np.diag(phase_transition_diagonal),
        )
        and np.allclose(process_cov, np.diag(process_variance))
        and np.allclose(observation_cov, np.diag(observation_variance))
        and np.allclose(initial_cov, np.diag(initial_variance))
    ):
        raise ValueError("当前 Kalman 实现要求对角状态转移、phase 调制与协方差矩阵")
    reliability_variance_scale = float(
        model.reliability_observation_variance_scale
    )
    if reliability_variance_scale < 0.0:
        raise ValueError("reliability observation variance scale 不能为负数")
    filtered_means: list[Any] = []
    filtered_variances: list[Any] = []
    predicted_means: list[Any] = []
    predicted_variances: list[Any] = []
    predicted_transition_diagonals: list[Any] = []
    log_likelihood = 0.0
    mean_state = initial_mean
    state_variance = initial_variance
    for step_index, observation in enumerate(y):
        centered_phase = phases[step_index] - float(
            model.phase_transition_reference
        )
        active_transition_diagonal = np.clip(
            transition_diagonal
            + centered_phase * phase_transition_diagonal,
            -0.995,
            0.995,
        )
        if step_index == 0:
            predicted_mean = mean_state
            predicted_variance = state_variance
        else:
            predicted_mean = active_transition_diagonal * mean_state + bias
            predicted_variance = (
                active_transition_diagonal**2 * state_variance + process_variance
            )
        innovation = observation - predicted_mean
        reliability_penalty = (
            (1.0 - reliabilities[step_index])
            / max(reliabilities[step_index], 1e-4)
        )
        active_observation_variance = observation_variance * (
            1.0 + reliability_variance_scale * reliability_penalty
        )
        innovation_variance = predicted_variance + active_observation_variance
        if bool(np.any(innovation_variance <= 0.0)):
            raise RuntimeError("状态空间 innovation variance 必须为正数")
        log_likelihood += -0.5 * (
            dimension * log(2.0 * 3.141592653589793)
            + float(np.log(innovation_variance).sum())
            + float((innovation**2 / innovation_variance).sum())
        )
        kalman_gain = predicted_variance / innovation_variance
        mean_state = predicted_mean + kalman_gain * innovation
        state_variance = (
            (1.0 - kalman_gain) ** 2 * predicted_variance
            + kalman_gain**2 * active_observation_variance
        )
        predicted_means.append(predicted_mean)
        predicted_variances.append(predicted_variance)
        predicted_transition_diagonals.append(active_transition_diagonal)
        filtered_means.append(mean_state)
        filtered_variances.append(state_variance)

    smoothed_means = [value.copy() for value in filtered_means]
    smoothed_variances = [value.copy() for value in filtered_variances]
    for step_index in range(len(y) - 2, -1, -1):
        next_prediction_variance = predicted_variances[step_index + 1]
        smoother_gain = (
            filtered_variances[step_index]
            * predicted_transition_diagonals[step_index + 1]
            / np.maximum(next_prediction_variance, 1e-12)
        )
        smoothed_means[step_index] = filtered_means[step_index] + smoother_gain * (
            smoothed_means[step_index + 1] - predicted_means[step_index + 1]
        )
        smoothed_variances[step_index] = filtered_variances[step_index] + smoother_gain**2 * (
            smoothed_variances[step_index + 1] - next_prediction_variance
        )
    return float(log_likelihood), np.asarray(smoothed_means, dtype=np.float64)


def state_space_marginal_log_likelihood_ratio(
    observations: Iterable[FlowEvidenceObservation],
    model: CalibratedFlowPosteriorModel,
) -> tuple[float, float, float, Any, Any]:
    """返回 H1/H0 每步边际似然、LLR 与两个假设下的平滑状态。"""

    import numpy as np

    rows = list(observations)
    if not rows:
        raise ValueError("Flow state posterior 至少需要一个观测")
    matrix = np.asarray([posterior_feature_vector(row) for row in rows], dtype=np.float64)
    means = np.asarray(model.feature_means, dtype=np.float64)
    scales = np.asarray(model.feature_scales, dtype=np.float64)
    standardized = (matrix - means) / np.maximum(scales, 1e-8)
    flow_phases = np.asarray([float(row.flow_phase) for row in rows], dtype=np.float64)
    replay_reliabilities = np.asarray(
        [float(row.replay_reliability) for row in rows],
        dtype=np.float64,
    )
    positive_likelihood, positive_smoothed = _kalman_filter_and_rts_smooth(
        standardized,
        model.positive_state_space_model,
        flow_phases=flow_phases,
        replay_reliabilities=replay_reliabilities,
    )
    negative_likelihood, negative_smoothed = _kalman_filter_and_rts_smooth(
        standardized,
        model.negative_state_space_model,
        flow_phases=flow_phases,
        replay_reliabilities=replay_reliabilities,
    )
    step_count = len(rows)
    positive_per_step = positive_likelihood / step_count
    negative_per_step = negative_likelihood / step_count
    return (
        positive_per_step,
        negative_per_step,
        positive_per_step - negative_per_step,
        positive_smoothed,
        negative_smoothed,
    )


def infer_flow_state_posterior(
    observations: Iterable[FlowEvidenceObservation],
    model: CalibratedFlowPosteriorModel,
) -> FlowStatePosterior:
    """运行双假设 Kalman filter、RTS smoother 与冻结 Platt 概率校准。"""

    import numpy as np

    rows = list(observations)
    if not rows:
        raise ValueError("Flow state posterior 至少需要一个观测")
    if tuple(model.feature_names) != POSTERIOR_FEATURE_NAMES:
        raise ValueError("概率后验特征顺序与冻结模型不一致")
    (
        positive_likelihood,
        negative_likelihood,
        state_space_llr,
        positive_smoothed,
        negative_smoothed,
    ) = state_space_marginal_log_likelihood_ratio(rows, model)
    calibrated_logit = model.platt_intercept + model.platt_slope * state_space_llr
    probability = _sigmoid(calibrated_logit)
    clipped = max(1e-12, min(1.0 - 1e-12, probability))
    entropy = -(clipped * log(clipped) + (1.0 - clipped) * log(1.0 - clipped))

    mixture_smoothed = probability * positive_smoothed + (1.0 - probability) * negative_smoothed
    means = np.asarray(model.feature_means, dtype=np.float64)
    scales = np.asarray(model.feature_scales, dtype=np.float64)
    restored_smoothed = mixture_smoothed * scales + means
    state_means = restored_smoothed.mean(axis=0)
    coverage = mean(max(0.0, min(1.0, row.coverage_ratio)) for row in rows)
    replay = mean(max(0.0, min(1.0, row.replay_reliability)) for row in rows)
    grid = mean(max(0.0, min(1.0, row.time_grid_reliability)) for row in rows)
    endpoint = float(state_means[POSTERIOR_FEATURE_NAMES.index("endpoint_score")])
    velocity = float(state_means[POSTERIOR_FEATURE_NAMES.index("velocity_score")])
    path = float(state_means[POSTERIOR_FEATURE_NAMES.index("path_score")])
    consistency = float(
        state_means[POSTERIOR_FEATURE_NAMES.index("path_endpoint_consistency")]
    )
    endpoint = max(0.0, min(1.0, endpoint))
    velocity = max(-1.0, min(1.0, velocity))
    path = max(-1.0, min(1.0, path))
    consistency = max(0.0, min(1.0, consistency))
    # P6 的冻结阈值由 OOF 原始观测的独立视频簇分位数拟合。因此推理时必须使用
    # 完全同口径的观测均值执行准入，而不能改用 Kalman 平滑隐状态。平滑状态仍
    # 作为后验解释字段输出，但不允许改变 fixed-FPR calibration 的门禁定义。
    endpoint_admissibility_measurement = mean(
        max(0.0, min(1.0, row.endpoint_score)) for row in rows
    )
    path_admissibility_measurement = mean(
        max(-1.0, min(1.0, row.path_score)) for row in rows
    )
    consistency_admissibility_measurement = mean(
        max(0.0, min(1.0, row.path_endpoint_consistency)) for row in rows
    )
    posterior_confidence = max(probability, 1.0 - probability)
    admissibility_context_complete = all(
        name in model.admissibility_thresholds
        for name in FLOW_STATE_ADMISSIBILITY_THRESHOLD_NAMES
    )
    checks = {
        "endpoint_score": endpoint_admissibility_measurement
        >= float(model.admissibility_thresholds.get("endpoint_score", 0.0)),
        "path_score": path_admissibility_measurement
        >= float(model.admissibility_thresholds.get("path_score", -1.0)),
        "path_endpoint_consistency": consistency_admissibility_measurement
        >= float(
            model.admissibility_thresholds.get(
                "path_endpoint_consistency",
                0.0,
            )
        ),
        "posterior_confidence": posterior_confidence
        >= float(
            model.admissibility_thresholds.get("posterior_confidence", 0.5)
        ),
        "coverage": coverage
        >= float(model.admissibility_thresholds.get("coverage", 0.0)),
        "replay_reliability": replay
        >= float(model.admissibility_thresholds.get("replay_reliability", 0.0)),
        "time_grid_reliability": grid
        >= float(model.admissibility_thresholds.get("time_grid_reliability", 0.0)),
        "posterior_entropy": entropy
        <= float(model.admissibility_thresholds.get("posterior_entropy_maximum", log(2.0))),
    }
    failures = tuple(
        [name for name, passed in checks.items() if not passed]
        + ([] if admissibility_context_complete else ["admissibility_context"])
    )
    admissible = not failures
    return FlowStatePosterior(
        watermark_probability=probability,
        posterior_log_odds=calibrated_logit,
        phase_state=mean(max(0.0, min(1.0, row.flow_phase)) for row in rows),
        endpoint_state=endpoint,
        temporal_disturbance_state=1.0 - mean([consistency, replay, grid]),
        path_consistency_state=min(max(0.0, path), consistency),
        velocity_consistency_state=velocity,
        replay_reliability_state=replay,
        time_grid_reliability_state=grid,
        posterior_entropy=entropy,
        positive_log_likelihood_per_step=positive_likelihood,
        negative_log_likelihood_per_step=negative_likelihood,
        state_space_log_likelihood_ratio=state_space_llr,
        filter_step_count=len(rows),
        admissible=admissible,
        admissibility_failures=failures,
        admissibility_context_complete=admissibility_context_complete,
        endpoint_admissibility_measurement=endpoint_admissibility_measurement,
        path_admissibility_measurement=path_admissibility_measurement,
        path_endpoint_consistency_admissibility_measurement=(
            consistency_admissibility_measurement
        ),
        conservative_score=probability if admissible else 0.0,
    )
