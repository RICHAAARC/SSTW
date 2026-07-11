"""拟合分组交叉验证概率后验并执行固定 FPR SSTW 检测。"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, inf, isfinite, log, nextafter
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

from main.methods.state_space_watermark.flow_state_posterior import (
    FLOW_STATE_POSTERIOR_MODEL_TYPE,
    POSTERIOR_FEATURE_NAMES,
    CalibratedFlowPosteriorModel,
    FlowEvidenceObservation,
    LinearGaussianFlowStateModel,
    infer_flow_state_posterior,
    posterior_feature_vector,
    state_space_marginal_log_likelihood_ratio,
)


FLOW_STATE_POSTERIOR_SCORE_SOURCE = (
    "dual_hypothesis_state_space_marginal_likelihood_calibrated_probability_posterior"
)

REQUIRED_FLOW_PHASE_OBSERVATION_FIELDS = (
    "flow_phase",
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
class FlowDetectorMechanismConfig:
    """定义核心检测器启用的参数化机制。

    该对象只描述 SSTW 检测原语是否执行可接受域约束，不包含任何外层实验
    组合名称。外层可以通过显式参数构造对照，但核心包不需要了解其研究语义。
    """

    enforce_state_admissibility: bool = True

    def as_dict(self) -> dict[str, bool]:
        """返回可随冻结阈值保存的核心机制参数。"""

        return {
            "flow_state_admissibility_enforced": self.enforce_state_admissibility,
        }


@dataclass(frozen=True)
class FrozenFlowDetectorCalibration:
    """保存核心概率模型、机制参数与 fixed-FPR 阈值。"""

    target_fpr: float
    posterior_model: CalibratedFlowPosteriorModel
    mechanism_config: FlowDetectorMechanismConfig
    final_score_threshold: float
    calibration_negative_count: int
    calibration_positive_count: int
    calibration_negative_cluster_count: int
    calibration_positive_cluster_count: int
    posterior_probability_calibration_protocol: str
    posterior_probability_calibration_outer_fold_count: int
    posterior_probability_calibration_inner_fold_minimum: int
    fixed_fpr_threshold_score_source: str
    detector_configuration_id: str = "sstw_core"

    def as_dict(self) -> dict[str, Any]:
        """转换为可重建 detection 的 threshold artifact。"""

        return {
            "detector_configuration_id": self.detector_configuration_id,
            "target_fpr": self.target_fpr,
            "calibration_negative_count": self.calibration_negative_count,
            "calibration_positive_count": self.calibration_positive_count,
            "calibration_negative_cluster_count": self.calibration_negative_cluster_count,
            "calibration_positive_cluster_count": self.calibration_positive_cluster_count,
            "posterior_probability_calibration_protocol": (
                self.posterior_probability_calibration_protocol
            ),
            "posterior_probability_calibration_outer_fold_count": (
                self.posterior_probability_calibration_outer_fold_count
            ),
            "posterior_probability_calibration_inner_fold_minimum": (
                self.posterior_probability_calibration_inner_fold_minimum
            ),
            "fixed_fpr_threshold_score_source": self.fixed_fpr_threshold_score_source,
            **self.posterior_model.as_dict(),
            "frozen_final_score_threshold": self.final_score_threshold,
            **self.mechanism_config.as_dict(),
        }


def frozen_flow_detector_calibration_from_dict(
    payload: Mapping[str, Any],
) -> FrozenFlowDetectorCalibration:
    """从 governed threshold artifact 重建只读检测器。

    该函数属于通用工程写法。任何离线评分调用方都必须消费已经冻结的
    calibration artifact, 不能在查询候选视频时重新拟合模型或阈值。
    """

    if payload.get("posterior_model_type") != FLOW_STATE_POSTERIOR_MODEL_TYPE:
        raise ValueError("threshold artifact 不是受支持的双假设状态空间后验模型")
    feature_names = tuple(str(value) for value in payload["posterior_feature_names"])
    if feature_names != POSTERIOR_FEATURE_NAMES:
        raise ValueError("threshold artifact 的概率后验特征顺序不兼容")
    model = CalibratedFlowPosteriorModel(
        feature_names=feature_names,
        feature_means=tuple(float(value) for value in payload["posterior_feature_means"]),
        feature_scales=tuple(float(value) for value in payload["posterior_feature_scales"]),
        negative_state_space_model=LinearGaussianFlowStateModel.from_dict(
            payload["posterior_negative_state_space_model"]
        ),
        positive_state_space_model=LinearGaussianFlowStateModel.from_dict(
            payload["posterior_positive_state_space_model"]
        ),
        platt_slope=float(payload["posterior_platt_slope"]),
        platt_intercept=float(payload["posterior_platt_intercept"]),
        admissibility_thresholds={
            str(key): float(value)
            for key, value in dict(payload["posterior_admissibility_thresholds"]).items()
        },
        calibration_brier_score=float(payload["posterior_calibration_brier_score"]),
        calibration_log_loss=float(payload["posterior_calibration_log_loss"]),
        calibration_expected_calibration_error=float(
            payload["posterior_calibration_expected_calibration_error"]
        ),
        calibration_group_count=int(payload["posterior_calibration_group_count"]),
        calibration_record_count=int(payload["posterior_calibration_record_count"]),
    )
    return FrozenFlowDetectorCalibration(
        target_fpr=float(payload["target_fpr"]),
        posterior_model=model,
        mechanism_config=FlowDetectorMechanismConfig(
            enforce_state_admissibility=bool(
                payload.get("flow_state_admissibility_enforced", True)
            )
        ),
        final_score_threshold=float(payload["frozen_final_score_threshold"]),
        calibration_negative_count=int(payload["calibration_negative_count"]),
        calibration_positive_count=int(payload["calibration_positive_count"]),
        calibration_negative_cluster_count=int(
            payload["calibration_negative_cluster_count"]
        ),
        calibration_positive_cluster_count=int(
            payload["calibration_positive_cluster_count"]
        ),
        posterior_probability_calibration_protocol=str(
            payload.get("posterior_probability_calibration_protocol")
            or "legacy_unspecified_probability_calibration"
        ),
        posterior_probability_calibration_outer_fold_count=int(
            payload.get("posterior_probability_calibration_outer_fold_count") or 0
        ),
        posterior_probability_calibration_inner_fold_minimum=int(
            payload.get("posterior_probability_calibration_inner_fold_minimum") or 0
        ),
        fixed_fpr_threshold_score_source=str(
            payload.get("fixed_fpr_threshold_score_source")
            or "legacy_unspecified_threshold_score_source"
        ),
        detector_configuration_id=str(
            payload.get("detector_configuration_id", "sstw_core")
        ),
    )


def _quantile(values: Iterable[float], quantile: float) -> float:
    rows = sorted(float(value) for value in values)
    if not rows:
        raise ValueError("quantile 需要至少一个数值")
    position = max(0, min(len(rows) - 1, round(float(quantile) * (len(rows) - 1))))
    return rows[position]


def _observation_from_mapping(
    values: Mapping[str, Any],
) -> FlowEvidenceObservation:
    """把一个完整规范化 phase 映射为状态空间输入。

    正式后验的每个维度都具有独立机制含义, 因而缺失值不能用0、记录级
    聚合量或其他字段静默替代。显式的数值0仍是合法实测值。
    """

    missing_fields = [
        field_name
        for field_name in REQUIRED_FLOW_PHASE_OBSERVATION_FIELDS
        if field_name not in values or values.get(field_name) is None
    ]
    if missing_fields:
        raise KeyError(
            "Flow phase 观测缺少必要字段: " + ", ".join(missing_fields)
        )

    def numeric_value(field_name: str) -> float:
        """读取一个已确认存在的规范字段, 并拒绝非有限数值。"""

        result = float(values[field_name])
        if not isfinite(result):
            raise ValueError(f"Flow phase 观测字段必须为有限数值: {field_name}")
        return result

    return FlowEvidenceObservation(
        endpoint_score=numeric_value("endpoint_score"),
        velocity_score=numeric_value("velocity_score"),
        path_score=numeric_value("path_score"),
        path_endpoint_consistency=numeric_value("path_endpoint_consistency"),
        replay_log_likelihood_ratio=numeric_value("replay_log_likelihood_ratio"),
        coverage_ratio=numeric_value("coverage_ratio"),
        replay_reliability=numeric_value("replay_reliability"),
        time_grid_reliability=numeric_value("time_grid_reliability"),
        flow_phase=numeric_value("flow_phase"),
    )


def flow_evidence_observation_sequence_from_mappings(
    phase_mappings: Sequence[Mapping[str, Any]],
) -> list[FlowEvidenceObservation]:
    """把规范化逐 phase 映射转换为核心观测序列。

    该函数只处理 SSTW 观测本身, 不读取外层数据分区、样本角色或统计簇字段。
    外层 runner 应先从 governed record 提取 phase 列表, 再调用此转换函数。
    """

    if len(phase_mappings) < 2:
        raise ValueError("核心检测器必须接收至少2个真实 phase 观测")
    if not all(isinstance(row, Mapping) for row in phase_mappings):
        raise TypeError("每个 phase 观测必须为映射对象")
    return [_observation_from_mapping(row) for row in phase_mappings]


def _balanced_weights(labels: Any, group_ids: Sequence[str] | None = None) -> Any:
    """先平衡类别, 再让每个独立视频簇获得相同类内总权重。"""

    import numpy as np

    labels = np.asarray(labels, dtype=np.float64)
    groups = np.asarray(
        list(group_ids) if group_ids is not None else [str(index) for index in range(len(labels))]
    )
    if len(groups) != len(labels):
        raise ValueError("group_ids 与 labels 长度不一致")
    weights = np.zeros_like(labels)
    for label in (0.0, 1.0):
        mask = labels == label
        class_groups = sorted(set(groups[mask].tolist()))
        if not class_groups:
            raise ValueError("概率后验 calibration 必须同时包含 positive 和 negative")
        for group in class_groups:
            group_mask = mask & (groups == group)
            weights[group_mask] = 0.5 / len(class_groups) / int(group_mask.sum())
    return weights


def _fit_logistic(
    matrix: Any,
    labels: Any,
    *,
    sample_weights: Any | None = None,
    l2_penalty: float = 1.0,
    maximum_iterations: int = 2000,
) -> tuple[Any, float]:
    """使用确定性全批量梯度下降拟合带 L2 正则的 logistic 模型。

    特征先在外层标准化且样本权重总和为1, 因而固定学习率可以稳定收敛。
    此实现避免不同 BLAS/LAPACK 后端在近奇异 Hessian 上直接终止进程。
    """

    import numpy as np

    x = np.asarray(matrix, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if x.ndim != 2 or len(y) != len(x):
        raise ValueError("logistic calibration matrix 形状不正确")
    weights = (
        np.ones(len(y), dtype=np.float64) / max(1, len(y))
        if sample_weights is None
        else np.asarray(sample_weights, dtype=np.float64)
    )
    design = np.concatenate([np.ones((len(x), 1)), x], axis=1)
    parameters = np.zeros(design.shape[1], dtype=np.float64)
    penalty = np.ones(design.shape[1], dtype=np.float64) * float(l2_penalty)
    penalty[0] = 0.0
    learning_rate = 0.2
    for iteration in range(maximum_iterations):
        logits = np.clip(design @ parameters, -40.0, 40.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        gradient = design.T @ (weights * (probabilities - y)) + penalty * parameters
        step = (learning_rate / (1.0 + iteration * 0.001)) * gradient
        parameters -= step
        if float(np.max(np.abs(step))) < 1e-9:
            break
    return parameters[1:], float(parameters[0])


def _equal_group_feature_moments(
    sequences: Sequence[Any],
    group_ids: Sequence[str],
) -> tuple[Any, Any]:
    """按 source-video group 等权估计观测均值与尺度。"""

    import numpy as np

    if not sequences:
        raise ValueError("状态空间模型至少需要一个观测序列")
    if len(sequences) != len(group_ids):
        raise ValueError("状态空间序列与 source-video group 数量不一致")
    dimension = int(np.asarray(sequences[0]).shape[1])
    mean_value = np.zeros(dimension, dtype=np.float64)
    second_moment = np.zeros(dimension, dtype=np.float64)
    unique_groups = sorted(set(group_ids))
    for group_id in unique_groups:
        group_rows: list[Any] = []
        for sequence, current_group_id in zip(sequences, group_ids):
            if current_group_id != group_id:
                continue
            matrix = np.asarray(sequence, dtype=np.float64)
            if matrix.ndim != 2 or matrix.shape[1] != dimension or len(matrix) == 0:
                raise ValueError("状态空间 calibration 序列形状不一致")
            group_rows.append(matrix)
        group_matrix = np.concatenate(group_rows, axis=0)
        mean_value += group_matrix.mean(axis=0) / len(unique_groups)
        second_moment += (group_matrix**2).mean(axis=0) / len(unique_groups)
    variance = np.maximum(second_moment - mean_value**2, 1e-8)
    return mean_value, np.sqrt(variance)


def _shrunk_covariance(matrix: Any, *, minimum_variance: float = 1e-3) -> Any:
    """估计严格对角正定协方差, 兼顾小样本稳定性与跨平台复现。"""

    import numpy as np

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("协方差估计需要非空二维矩阵")
    centered = values - values.mean(axis=0)
    variances = np.maximum((centered**2).mean(axis=0), minimum_variance)
    return np.diag(variances + minimum_variance)


def _fit_linear_gaussian_state_model(
    sequences: Sequence[Any],
    *,
    group_ids: Sequence[str],
    feature_means: Any,
    feature_scales: Any,
) -> LinearGaussianFlowStateModel:
    """按 source-video group 等权拟合线性转移和高斯噪声。"""

    import numpy as np

    standardized = [
        (np.asarray(sequence, dtype=np.float64) - feature_means)
        / np.maximum(feature_scales, 1e-8)
        for sequence in sequences
    ]
    if len(standardized) != len(group_ids):
        raise ValueError("状态空间序列与 source-video group 数量不一致")
    dimension = standardized[0].shape[1]
    unique_groups = sorted(set(group_ids))
    grouped_sequences = {
        group_id: [
            sequence
            for sequence, current_group_id in zip(standardized, group_ids)
            if current_group_id == group_id
        ]
        for group_id in unique_groups
    }
    initial_states = np.stack(
        [
            np.stack([sequence[0] for sequence in grouped_sequences[group_id]], axis=0)
            .mean(axis=0)
            for group_id in unique_groups
        ],
        axis=0,
    )
    transition_rows_by_group: dict[str, tuple[Any, Any]] = {}
    for group_id, group_sequences in grouped_sequences.items():
        previous = [sequence[:-1] for sequence in group_sequences if len(sequence) > 1]
        following = [sequence[1:] for sequence in group_sequences if len(sequence) > 1]
        if previous:
            transition_rows_by_group[group_id] = (
                np.concatenate(previous, axis=0),
                np.concatenate(following, axis=0),
            )
    transition_count = sum(
        len(previous) for previous, _following in transition_rows_by_group.values()
    )
    transition_group_count = len(transition_rows_by_group)
    if transition_count:
        previous_mean = np.stack(
            [rows[0].mean(axis=0) for rows in transition_rows_by_group.values()],
            axis=0,
        ).mean(axis=0)
        following_mean = np.stack(
            [rows[1].mean(axis=0) for rows in transition_rows_by_group.values()],
            axis=0,
        ).mean(axis=0)
        numerator = np.stack(
            [
                ((previous - previous_mean) * (following - following_mean)).mean(axis=0)
                for previous, following in transition_rows_by_group.values()
            ],
            axis=0,
        ).mean(axis=0)
        denominator = np.stack(
            [
                ((previous - previous_mean) ** 2).mean(axis=0)
                for previous, _following in transition_rows_by_group.values()
            ],
            axis=0,
        ).mean(axis=0) + 0.05
        diagonal_transition = numerator / denominator
        diagonal_transition = np.clip(diagonal_transition, -0.98, 0.98)
        transition = np.diag(diagonal_transition)
        bias = following_mean - diagonal_transition * previous_mean
        process_variance = np.stack(
            [
                np.square(
                    following - (previous * diagonal_transition + bias)
                ).mean(axis=0)
                for previous, following in transition_rows_by_group.values()
            ],
            axis=0,
        ).mean(axis=0)
    else:
        transition = np.eye(dimension, dtype=np.float64) * 0.5
        bias = initial_states.mean(axis=0) * 0.5
        process_variance = ((initial_states - initial_states.mean(axis=0)) ** 2).mean(
            axis=0
        )
    process_covariance = np.diag(np.maximum(process_variance, 0.02))
    observation_second_moment = np.stack(
        [
            (np.concatenate(grouped_sequences[group_id], axis=0) ** 2).mean(axis=0)
            for group_id in unique_groups
        ],
        axis=0,
    ).mean(axis=0)
    observation_variance = np.maximum(observation_second_moment * 0.1, 0.02)
    observation_covariance = np.diag(observation_variance)
    initial_covariance = _shrunk_covariance(initial_states, minimum_variance=0.05)
    return LinearGaussianFlowStateModel(
        transition_matrix=tuple(tuple(float(value) for value in row) for row in transition),
        transition_bias=tuple(float(value) for value in bias),
        process_covariance=tuple(
            tuple(float(value) for value in row) for row in process_covariance
        ),
        observation_covariance=tuple(
            tuple(float(value) for value in row) for row in observation_covariance
        ),
        initial_mean=tuple(float(value) for value in initial_states.mean(axis=0)),
        initial_covariance=tuple(
            tuple(float(value) for value in row) for row in initial_covariance
        ),
        training_sequence_count=len(standardized),
        training_group_count=len(unique_groups),
        training_transition_count=transition_count,
        training_transition_group_count=transition_group_count,
    )


def _build_uncalibrated_state_posterior_model(
    sequences: Sequence[Sequence[FlowEvidenceObservation]],
    labels: Sequence[int],
    group_ids: Sequence[str],
) -> CalibratedFlowPosteriorModel:
    """拟合 H0/H1 状态空间模型, 暂不拟合概率校准映射。"""

    import numpy as np

    matrices = [
        np.asarray([posterior_feature_vector(row) for row in sequence], dtype=np.float64)
        for sequence in sequences
    ]
    if not (len(matrices) == len(labels) == len(group_ids)):
        raise ValueError("状态空间序列、标签与 source-video group 数量不一致")
    means, scales = _equal_group_feature_moments(matrices, group_ids)
    negative_sequences = [
        matrix for matrix, label in zip(matrices, labels) if label == 0
    ]
    negative_group_ids = [
        group_id for group_id, label in zip(group_ids, labels) if label == 0
    ]
    positive_sequences = [
        matrix for matrix, label in zip(matrices, labels) if label == 1
    ]
    positive_group_ids = [
        group_id for group_id, label in zip(group_ids, labels) if label == 1
    ]
    if not negative_sequences or not positive_sequences:
        raise ValueError("状态空间 calibration 必须同时包含 H0 与 H1 序列")
    negative_model = _fit_linear_gaussian_state_model(
        negative_sequences,
        group_ids=negative_group_ids,
        feature_means=means,
        feature_scales=scales,
    )
    positive_model = _fit_linear_gaussian_state_model(
        positive_sequences,
        group_ids=positive_group_ids,
        feature_means=means,
        feature_scales=scales,
    )
    return CalibratedFlowPosteriorModel(
        feature_names=POSTERIOR_FEATURE_NAMES,
        feature_means=tuple(float(value) for value in means),
        feature_scales=tuple(float(value) for value in scales),
        negative_state_space_model=negative_model,
        positive_state_space_model=positive_model,
        platt_slope=1.0,
        platt_intercept=0.0,
        admissibility_thresholds={},
        calibration_brier_score=0.0,
        calibration_log_loss=0.0,
        calibration_expected_calibration_error=0.0,
        calibration_group_count=0,
        calibration_record_count=0,
    )


def _calibration_metrics(
    probabilities: Any,
    labels: Any,
    group_ids: Sequence[str],
) -> tuple[float, float, float]:
    """计算 class-balanced Brier、log loss 与 ECE。"""

    import numpy as np

    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-8, 1.0 - 1e-8)
    y = np.asarray(labels, dtype=np.float64)
    weights = _balanced_weights(y, group_ids)
    weights = weights / weights.sum()
    brier = float(np.sum(weights * (p - y) ** 2))
    log_loss = float(-np.sum(weights * (y * np.log(p) + (1.0 - y) * np.log(1.0 - p))))
    ece = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        mask = (p >= lower) & (p < upper if upper < 1.0 else p <= upper)
        if not bool(mask.any()):
            continue
        bin_weight = float(weights[mask].sum())
        confidence = float(np.average(p[mask], weights=weights[mask]))
        accuracy = float(np.average(y[mask], weights=weights[mask]))
        ece += bin_weight * abs(confidence - accuracy)
    return brier, log_loss, float(ece)


def _fit_admissibility_thresholds(
    observation_sequences: Sequence[Sequence[FlowEvidenceObservation]],
    probabilities: Sequence[float],
    labels: Sequence[int],
    group_ids: Sequence[str],
) -> dict[str, float]:
    """按独立视频簇等权拟合质量与正类后验熵准入阈值。"""

    if not (
        len(observation_sequences)
        == len(probabilities)
        == len(labels)
        == len(group_ids)
    ):
        raise ValueError("admissibility calibration 输入长度不一致")
    unique_groups = sorted(set(group_ids))
    if not unique_groups:
        raise ValueError("admissibility calibration 缺少独立视频簇")
    group_indices = {
        group_id: [
            index
            for index, current_group_id in enumerate(group_ids)
            if current_group_id == group_id
        ]
        for group_id in unique_groups
    }
    quality_extractors = {
        "coverage": lambda row: row.coverage_ratio,
        "replay_reliability": lambda row: row.replay_reliability,
        "time_grid_reliability": lambda row: row.time_grid_reliability,
    }
    quality = {
        name: [
            mean(
                mean(extractor(row) for row in observation_sequences[index])
                for index in group_indices[group_id]
            )
            for group_id in unique_groups
        ]
        for name, extractor in quality_extractors.items()
    }
    entropy_by_record = [
        -(
            float(probability) * log(max(float(probability), 1e-8))
            + (1.0 - float(probability))
            * log(max(1.0 - float(probability), 1e-8))
        )
        for probability in probabilities
    ]
    positive_entropies: list[float] = []
    for group_id in unique_groups:
        positive_indices = [
            index
            for index in group_indices[group_id]
            if int(labels[index]) == 1
        ]
        if positive_indices:
            positive_entropies.append(
                mean(entropy_by_record[index] for index in positive_indices)
            )
    if not positive_entropies:
        raise ValueError("admissibility calibration 缺少正类视频簇")
    return {
        **{
            name: _quantile(values, 0.05)
            for name, values in quality.items()
        },
        "posterior_entropy_maximum": _quantile(positive_entropies, 0.95),
    }


def _stratified_group_fold_ids(
    labels: Sequence[int],
    group_ids: Sequence[str],
    *,
    maximum_fold_count: int = 5,
) -> tuple[Any, int]:
    """按标签组合分配独立视频簇，确保同一簇不会跨 fold 泄漏。"""

    import numpy as np

    if len(labels) != len(group_ids):
        raise ValueError("labels 与 group_ids 长度不一致")
    labels_by_group: dict[str, set[int]] = {}
    for group_id, label in zip(group_ids, labels):
        labels_by_group.setdefault(str(group_id), set()).add(int(label))
    groups_by_label = {
        label: sorted(
            group_id
            for group_id, group_labels in labels_by_group.items()
            if label in group_labels
        )
        for label in (0, 1)
    }
    if min(len(groups_by_label[0]), len(groups_by_label[1])) < 2:
        raise ValueError("严格 group cross-fitting 要求 H0 与 H1 各至少2个独立视频簇")
    fold_count = min(
        int(maximum_fold_count),
        len(groups_by_label[0]),
        len(groups_by_label[1]),
    )
    fold_by_group: dict[str, int] = {}
    groups_by_label_signature: dict[tuple[int, ...], list[str]] = {}
    for group_id, group_labels in labels_by_group.items():
        groups_by_label_signature.setdefault(
            tuple(sorted(group_labels)), []
        ).append(group_id)
    for signature in sorted(groups_by_label_signature):
        for index, group_id in enumerate(
            sorted(groups_by_label_signature[signature])
        ):
            fold_by_group[group_id] = index % fold_count
    return np.asarray([fold_by_group[str(group_id)] for group_id in group_ids]), fold_count


@dataclass(frozen=True)
class _NestedPosteriorCalibrationFit:
    """保存最终模型及未观察各自视频簇时得到的校准概率。"""

    model: CalibratedFlowPosteriorModel
    nested_cross_fitted_probabilities: tuple[float, ...]
    nested_admissibility_thresholds: tuple[Mapping[str, float], ...]
    outer_fold_count: int
    inner_fold_minimum: int


def _fit_calibrated_posterior_model(
    observation_sequences: Sequence[Sequence[FlowEvidenceObservation]],
    labels: Sequence[int],
    group_ids: Sequence[str],
) -> _NestedPosteriorCalibrationFit:
    """执行嵌套视频簇交叉拟合，并用外层未见样本评价概率校准。"""

    import numpy as np

    y = np.asarray(labels, dtype=np.float64)
    unique_groups = sorted(set(group_ids))
    if len(unique_groups) < 2:
        raise ValueError("概率后验 calibration 至少需要2个独立视频簇")
    fold_ids, fold_count = _stratified_group_fold_ids(labels, group_ids)
    out_of_fold_llrs = np.full(len(y), np.nan, dtype=np.float64)
    nested_cross_fitted_probabilities = np.full(len(y), np.nan, dtype=np.float64)
    nested_admissibility_thresholds: list[dict[str, float] | None] = [
        None for _ in range(len(y))
    ]
    inner_fold_counts: list[int] = []
    for fold in range(fold_count):
        validation_mask = fold_ids == fold
        training_mask = ~validation_mask
        if not bool(validation_mask.any()) or len(set(y[training_mask].tolist())) < 2:
            raise RuntimeError("分层 source-video group fold 无法形成双假设训练集")
        fold_sequences = [
            sequence
            for index, sequence in enumerate(observation_sequences)
            if training_mask[index]
        ]
        fold_group_ids = [
            group_id
            for index, group_id in enumerate(group_ids)
            if training_mask[index]
        ]
        fold_model = _build_uncalibrated_state_posterior_model(
            fold_sequences,
            y[training_mask].astype(int).tolist(),
            fold_group_ids,
        )
        for index, sequence in enumerate(observation_sequences):
            if validation_mask[index]:
                out_of_fold_llrs[index] = state_space_marginal_log_likelihood_ratio(
                    sequence,
                    fold_model,
                )[2]

        # 外层 validation 概率只能由外层 training 内部再次交叉拟合的 Platt
        # 映射产生。这样 calibration 指标不会评价刚刚拟合自身的概率。
        outer_training_indices = np.flatnonzero(training_mask)
        outer_training_labels = y[training_mask].astype(int).tolist()
        outer_training_groups = [
            group_ids[index] for index in outer_training_indices
        ]
        inner_fold_ids, inner_fold_count = _stratified_group_fold_ids(
            outer_training_labels,
            outer_training_groups,
        )
        inner_fold_counts.append(inner_fold_count)
        inner_oof_llrs = np.full(len(outer_training_indices), np.nan, dtype=np.float64)
        for inner_fold in range(inner_fold_count):
            inner_validation_mask = inner_fold_ids == inner_fold
            inner_training_mask = ~inner_validation_mask
            inner_sequences = [
                observation_sequences[outer_training_indices[index]]
                for index in range(len(outer_training_indices))
                if inner_training_mask[index]
            ]
            inner_labels = [
                outer_training_labels[index]
                for index in range(len(outer_training_indices))
                if inner_training_mask[index]
            ]
            inner_groups = [
                outer_training_groups[index]
                for index in range(len(outer_training_indices))
                if inner_training_mask[index]
            ]
            if len(set(inner_labels)) < 2:
                raise RuntimeError("嵌套 group cross-fitting 的内层训练集缺少双假设")
            inner_model = _build_uncalibrated_state_posterior_model(
                inner_sequences,
                inner_labels,
                inner_groups,
            )
            for inner_index, source_index in enumerate(outer_training_indices):
                if inner_validation_mask[inner_index]:
                    inner_oof_llrs[inner_index] = (
                        state_space_marginal_log_likelihood_ratio(
                            observation_sequences[source_index],
                            inner_model,
                        )[2]
                    )
        if bool(np.isnan(inner_oof_llrs).any()):
            raise RuntimeError("嵌套 group cross-fitting 存在未生成的内层 OOF 分数")
        outer_coefficients, outer_intercept = _fit_logistic(
            inner_oof_llrs.reshape(-1, 1),
            np.asarray(outer_training_labels, dtype=np.float64),
            sample_weights=_balanced_weights(
                outer_training_labels,
                outer_training_groups,
            ),
            l2_penalty=0.01,
        )
        outer_slope = float(outer_coefficients[0])
        outer_training_logits = (
            outer_intercept + outer_slope * inner_oof_llrs
        )
        outer_training_probabilities = 1.0 / (
            1.0 + np.exp(-np.clip(outer_training_logits, -40.0, 40.0))
        )
        outer_admissibility_thresholds = _fit_admissibility_thresholds(
            fold_sequences,
            outer_training_probabilities.tolist(),
            outer_training_labels,
            outer_training_groups,
        )
        for index in np.flatnonzero(validation_mask):
            logit = outer_intercept + outer_slope * out_of_fold_llrs[index]
            nested_cross_fitted_probabilities[index] = 1.0 / (
                1.0 + np.exp(-np.clip(logit, -40.0, 40.0))
            )
            nested_admissibility_thresholds[index] = dict(
                outer_admissibility_thresholds
            )
    full_model = _build_uncalibrated_state_posterior_model(
        observation_sequences,
        [int(value) for value in labels],
        group_ids,
    )
    if bool(np.isnan(out_of_fold_llrs).any()):
        raise RuntimeError("存在未由严格 group cross-fitting 生成的状态空间 OOF 分数")

    platt_coefficients, platt_intercept = _fit_logistic(
        out_of_fold_llrs.reshape(-1, 1),
        y,
        sample_weights=_balanced_weights(y, group_ids),
        l2_penalty=0.01,
    )
    platt_slope = float(platt_coefficients[0])
    if bool(np.isnan(nested_cross_fitted_probabilities).any()):
        raise RuntimeError("存在未由嵌套 group cross-fitting 生成的校准概率")
    brier, log_loss_value, ece = _calibration_metrics(
        nested_cross_fitted_probabilities,
        y,
        group_ids,
    )
    if any(value is None for value in nested_admissibility_thresholds):
        raise RuntimeError("存在未由外层 training fold 拟合的准入阈值")
    final_admissibility_thresholds = _fit_admissibility_thresholds(
        observation_sequences,
        nested_cross_fitted_probabilities.tolist(),
        [int(value) for value in labels],
        group_ids,
    )
    model = CalibratedFlowPosteriorModel(
        feature_names=POSTERIOR_FEATURE_NAMES,
        feature_means=full_model.feature_means,
        feature_scales=full_model.feature_scales,
        negative_state_space_model=full_model.negative_state_space_model,
        positive_state_space_model=full_model.positive_state_space_model,
        platt_slope=platt_slope,
        platt_intercept=float(platt_intercept),
        admissibility_thresholds=final_admissibility_thresholds,
        calibration_brier_score=brier,
        calibration_log_loss=log_loss_value,
        calibration_expected_calibration_error=ece,
        calibration_group_count=len(unique_groups),
        calibration_record_count=len(observation_sequences),
    )
    return _NestedPosteriorCalibrationFit(
        model=model,
        nested_cross_fitted_probabilities=tuple(
            float(value) for value in nested_cross_fitted_probabilities
        ),
        nested_admissibility_thresholds=tuple(
            dict(value)
            for value in nested_admissibility_thresholds
            if value is not None
        ),
        outer_fold_count=fold_count,
        inner_fold_minimum=min(inner_fold_counts),
    )


def _cross_fitted_conservative_score(
    observations: Sequence[FlowEvidenceObservation],
    probability: float,
    admissibility_thresholds: Mapping[str, float],
    mechanism_config: FlowDetectorMechanismConfig,
) -> float:
    """把外层未见概率与冻结准入阈值组合为阈值校准分数。"""

    if not mechanism_config.enforce_state_admissibility:
        return float(probability)
    quality = {
        "coverage": mean(max(0.0, min(1.0, row.coverage_ratio)) for row in observations),
        "replay_reliability": mean(
            max(0.0, min(1.0, row.replay_reliability)) for row in observations
        ),
        "time_grid_reliability": mean(
            max(0.0, min(1.0, row.time_grid_reliability)) for row in observations
        ),
    }
    entropy = -(
        float(probability) * log(max(float(probability), 1e-8))
        + (1.0 - float(probability))
        * log(max(1.0 - float(probability), 1e-8))
    )
    quality_ready = all(
        value >= float(admissibility_thresholds.get(name, 0.0))
        for name, value in quality.items()
    )
    entropy_ready = entropy <= float(
        admissibility_thresholds.get("posterior_entropy_maximum", log(2.0))
    )
    return float(probability) if quality_ready and entropy_ready else 0.0


def _fixed_fpr_threshold(values: Iterable[float], target_fpr: float) -> float:
    """选择使 calibration empirical FPR 不超过目标值的冻结阈值。"""

    rows = sorted(float(value) for value in values)
    if not rows:
        raise ValueError("fixed-FPR threshold 需要至少一个数值")
    allowed_false_positives = floor(float(target_fpr) * len(rows) + 1e-12)
    if allowed_false_positives <= 0:
        return nextafter(rows[-1], inf)
    for candidate in sorted(set(rows)):
        accepted = sum(value >= candidate for value in rows)
        if accepted <= allowed_false_positives:
            return candidate
    return nextafter(rows[-1], inf)


def score_flow_observation_sequence(
    observation_sequence: Sequence[FlowEvidenceObservation],
    posterior_model: CalibratedFlowPosteriorModel,
    *,
    mechanism_config: FlowDetectorMechanismConfig | None = None,
) -> dict[str, Any]:
    """使用冻结概率模型计算一个核心观测序列。

    输入只包含 SSTW 的逐 phase 状态观测。样本属于哪个数据分区、承担什么
    论文角色以及如何写入正式记录, 均由外层实验协议负责。
    """

    mechanism_config = mechanism_config or FlowDetectorMechanismConfig()
    if len(observation_sequence) < 2:
        raise ValueError("核心检测器必须接收至少2个真实 phase 观测")
    if any(
        not isinstance(observation, FlowEvidenceObservation)
        for observation in observation_sequence
    ):
        raise TypeError("核心评分只接受 FlowEvidenceObservation 序列")
    posterior = infer_flow_state_posterior(observation_sequence, posterior_model)
    payload = posterior.as_dict()
    # 判定分数必须保留计算精度。若沿用展示字段的8位小数，靠近冻结阈值的
    # 样本可能因格式化舍入改变判定，从而破坏 fixed-FPR 的比较语义。
    payload["S_final_unconstrained"] = float(posterior.watermark_probability)
    payload["S_final_conservative"] = float(posterior.conservative_score)
    if not mechanism_config.enforce_state_admissibility:
        payload["flow_state_admissibility_status"] = "disabled_by_mechanism_config"
        payload["flow_state_admissibility_failures"] = []
        payload["S_final_conservative"] = payload["S_final_unconstrained"]
    payload["flow_detector_score_source"] = (
        FLOW_STATE_POSTERIOR_SCORE_SOURCE
    )
    return payload


def fit_flow_detector_calibration(
    observation_sequences: Iterable[Sequence[FlowEvidenceObservation]],
    binary_labels: Iterable[int | bool],
    cluster_ids: Iterable[str],
    *,
    target_fpr: float,
    mechanism_config: FlowDetectorMechanismConfig | None = None,
    detector_configuration_id: str = "sstw_core",
) -> FrozenFlowDetectorCalibration:
    """拟合核心概率后验, 并使用负类视频簇冻结 fixed-FPR 阈值。

    `binary_labels` 中1表示水印假设, 0表示非水印假设。`cluster_ids`
    表示统计独立单元, 同一视频的重复 key trial 必须使用相同簇标识。该
    API 不解释外层样本角色或数据分区, 因而可以在不同运行环境中复用。
    """

    mechanism_config = mechanism_config or FlowDetectorMechanismConfig()
    if not 0.0 < float(target_fpr) < 1.0:
        raise ValueError("target_fpr 必须位于 (0, 1)")
    sequences = [list(sequence) for sequence in observation_sequences]
    labels_raw = list(binary_labels)
    groups_raw = list(cluster_ids)
    if not sequences:
        raise ValueError("核心 calibration 至少需要一个观测序列")
    if len(sequences) != len(labels_raw) or len(sequences) != len(groups_raw):
        raise ValueError("observation_sequences、binary_labels 与 cluster_ids 长度必须一致")
    invalid_label_values = sorted({
        repr(label)
        for label in labels_raw
        if label not in (0, 1, False, True)
    })
    if invalid_label_values:
        raise ValueError(f"binary_labels 只能包含0或1, 收到: {invalid_label_values}")
    labels = [int(label) for label in labels_raw]
    if any(
        not isinstance(group_id, str) or not group_id.strip()
        for group_id in groups_raw
    ):
        raise ValueError("cluster_ids 必须是非空字符串")
    groups = [str(group_id) for group_id in groups_raw]
    if any(len(sequence) < 2 for sequence in sequences):
        raise ValueError("每个 calibration 输入必须包含至少2个真实 phase 观测")
    if any(
        not isinstance(observation, FlowEvidenceObservation)
        for sequence in sequences
        for observation in sequence
    ):
        raise TypeError("核心 calibration 只接受 FlowEvidenceObservation 序列")
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    if positive_count < 2 or negative_count < 2:
        raise ValueError("概率后验 calibration 至少需要2条 positive 和2条 negative")
    fitted = _fit_calibrated_posterior_model(
        sequences,
        labels,
        groups,
    )
    model = fitted.model
    negative_probabilities = [
        probability
        for probability, label in zip(
            fitted.nested_cross_fitted_probabilities,
            labels,
        )
        if label == 0
    ]
    negative_observations = [
        observations
        for observations, label in zip(sequences, labels)
        if label == 0
    ]
    negative_admissibility_thresholds = [
        thresholds
        for thresholds, label in zip(
            fitted.nested_admissibility_thresholds,
            labels,
        )
        if label == 0
    ]
    negative_scores_by_cluster: dict[str, list[float]] = {}
    negative_groups = [group for group, label in zip(groups, labels) if label == 0]
    for group, observations, probability, admissibility_thresholds in zip(
        negative_groups,
        negative_observations,
        negative_probabilities,
        negative_admissibility_thresholds,
    ):
        negative_scores_by_cluster.setdefault(group, []).append(
            _cross_fitted_conservative_score(
                observations,
                probability,
                admissibility_thresholds,
                mechanism_config,
            )
        )
    # 同一 clean video 的多个 key trial 是簇内重复测量。阈值按每视频最大分数
    # 冻结, 使目标 FPR 对应独立视频而不是人为扩增的 trial 数量。
    negative_cluster_scores = [max(values) for values in negative_scores_by_cluster.values()]
    return FrozenFlowDetectorCalibration(
        target_fpr=float(target_fpr),
        posterior_model=model,
        mechanism_config=mechanism_config,
        final_score_threshold=_fixed_fpr_threshold(negative_cluster_scores, target_fpr),
        calibration_negative_count=negative_count,
        calibration_positive_count=positive_count,
        calibration_negative_cluster_count=len(negative_scores_by_cluster),
        calibration_positive_cluster_count=len({
            group for group, label in zip(groups, labels) if label == 1
        }),
        posterior_probability_calibration_protocol=(
            "nested_source_video_group_cross_fitted_state_space_llr_and_platt"
        ),
        posterior_probability_calibration_outer_fold_count=fitted.outer_fold_count,
        posterior_probability_calibration_inner_fold_minimum=fitted.inner_fold_minimum,
        fixed_fpr_threshold_score_source=(
            "outer_group_heldout_nested_cross_fitted_conservative_scores"
        ),
        detector_configuration_id=str(detector_configuration_id),
    )


def apply_frozen_flow_detector(
    observation_sequence: Sequence[FlowEvidenceObservation],
    calibration: FrozenFlowDetectorCalibration,
) -> dict[str, Any]:
    """对一个核心观测序列应用冻结后验与阈值, 不更新任何参数。"""

    score_payload = score_flow_observation_sequence(
        observation_sequence,
        calibration.posterior_model,
        mechanism_config=calibration.mechanism_config,
    )
    score = float(score_payload["S_final_conservative"])
    return {
        **score_payload,
        "frozen_final_score_threshold": calibration.final_score_threshold,
        "target_fpr": calibration.target_fpr,
        "decision": score >= calibration.final_score_threshold,
    }
