"""拟合分组交叉验证概率后验并执行固定 FPR SSTW 检测。"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, inf, log, nextafter
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


FORMAL_METHOD_VARIANTS = (
    "sstw_full_method",
    "endpoint_only_control",
    "trajectory_only_score",
    "without_velocity_constraint",
    "without_endpoint_aware_control",
    "without_replay_uncertainty_weighting",
    "without_flow_state_admissibility",
    "generic_ssm_baseline",
)

FLOW_STATE_POSTERIOR_SCORE_SOURCE = (
    "dual_hypothesis_state_space_marginal_likelihood_calibrated_probability_posterior"
)


@dataclass(frozen=True)
class FrozenFlowDetectorCalibration:
    """保存一个 method variant 的概率模型与 fixed-FPR 阈值。"""

    method_variant: str
    target_fpr: float
    posterior_model: CalibratedFlowPosteriorModel
    final_score_threshold: float
    calibration_negative_count: int
    calibration_positive_count: int
    calibration_negative_cluster_count: int
    calibration_positive_cluster_count: int

    def as_dict(self) -> dict[str, Any]:
        """转换为可重建 detection 的 threshold artifact。"""

        return {
            "method_variant": self.method_variant,
            "target_fpr": self.target_fpr,
            "calibration_negative_count": self.calibration_negative_count,
            "calibration_positive_count": self.calibration_positive_count,
            "calibration_negative_cluster_count": self.calibration_negative_cluster_count,
            "calibration_positive_cluster_count": self.calibration_positive_cluster_count,
            **self.posterior_model.as_dict(),
            "frozen_final_score_threshold": self.final_score_threshold,
            "threshold_source_split": "calibration",
            "test_time_threshold_update_blocked": True,
        }


def frozen_flow_detector_calibration_from_dict(
    payload: Mapping[str, Any],
) -> FrozenFlowDetectorCalibration:
    """从 governed threshold artifact 重建只读检测器。

    该函数属于通用工程写法。adaptive attack 与服务器离线评分必须消费已经
    冻结的 calibration artifact, 不能在查询候选视频时重新拟合模型或阈值。
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
        method_variant=str(payload["method_variant"]),
        target_fpr=float(payload["target_fpr"]),
        posterior_model=model,
        final_score_threshold=float(payload["frozen_final_score_threshold"]),
        calibration_negative_count=int(payload["calibration_negative_count"]),
        calibration_positive_count=int(payload["calibration_positive_count"]),
        calibration_negative_cluster_count=int(
            payload["calibration_negative_cluster_count"]
        ),
        calibration_positive_cluster_count=int(
            payload["calibration_positive_cluster_count"]
        ),
    )


def _quantile(values: Iterable[float], quantile: float) -> float:
    rows = sorted(float(value) for value in values)
    if not rows:
        raise ValueError("quantile 需要至少一个数值")
    position = max(0, min(len(rows) - 1, round(float(quantile) * (len(rows) - 1))))
    return rows[position]


def _record_value(record: Mapping[str, Any], field_name: str) -> float:
    value = record.get(field_name)
    if value is None:
        raise KeyError(f"正式 Flow evidence 缺少字段: {field_name}")
    return float(value)


def _cluster_id(record: Mapping[str, Any]) -> str:
    """读取统计独立单元, 禁止把同视频 key trial 当成独立视频。"""

    value = (
        record.get("statistical_cluster_id")
        or record.get("source_video_cluster_id")
        or record.get("trajectory_trace_id")
    )
    if not value:
        raise KeyError("正式 calibration record 缺少 statistical_cluster_id")
    return str(value)


def observation_from_flow_evidence_record(
    record: Mapping[str, Any],
    *,
    method_variant: str,
) -> FlowEvidenceObservation:
    """返回记录的第一个状态空间观测, 供兼容调用方使用。"""

    return observation_sequence_from_flow_evidence_record(
        record,
        method_variant=method_variant,
    )[0]


def _observation_from_mapping(
    values: Mapping[str, Any],
    *,
    method_variant: str,
) -> FlowEvidenceObservation:
    """把一个 phase 观测映射为指定消融变体的状态空间输入。"""

    endpoint = float(values.get("endpoint_score") or 0.0)
    velocity = float(values.get("velocity_score") or values.get("S_velocity") or 0.0)
    path = float(values.get("path_score") or values.get("S_path_inv") or 0.0)
    consistency = float(values.get("path_endpoint_consistency") or 0.0)
    coverage = float(values.get("coverage_ratio") or values.get("endpoint_coverage_ratio") or 0.0)
    replay_reliability = float(values.get("replay_reliability") or values.get("replay_reliability_weight") or 0.0)
    grid_reliability = float(values.get("time_grid_reliability") or replay_reliability)
    replay_log_likelihood_ratio = float(
        values.get("replay_log_likelihood_ratio")
        or values.get("replay_log_likelihood_ratio_mean")
        or 0.0
    )
    if method_variant == "endpoint_only_control":
        velocity = 0.0
        path = 0.0
        consistency = 0.0
        replay_log_likelihood_ratio = 0.0
        replay_reliability = 1.0
        grid_reliability = 1.0
    elif method_variant == "trajectory_only_score":
        endpoint = 0.0
    elif method_variant == "without_replay_uncertainty_weighting":
        replay_reliability = 1.0
        grid_reliability = 1.0
    elif method_variant == "generic_ssm_baseline":
        # generic SSM 只能消费不含候选 key 投影的轨迹能量和一致性特征。
        # 此处显式清零 key-conditioned endpoint/path/velocity/LLR, 防止把完整
        # SSTW 观测复制一份后改名为 baseline。
        endpoint = float(values.get("key_agnostic_endpoint_energy") or 0.0)
        velocity = float(values.get("key_agnostic_velocity_energy") or 0.0)
        path = float(values.get("key_agnostic_path_energy") or 0.0)
        consistency = float(values.get("path_velocity_consistency") or consistency)
        replay_log_likelihood_ratio = 0.0
    return FlowEvidenceObservation(
        endpoint_score=endpoint,
        velocity_score=velocity,
        path_score=path,
        path_endpoint_consistency=consistency,
        replay_log_likelihood_ratio=replay_log_likelihood_ratio,
        coverage_ratio=coverage,
        replay_reliability=replay_reliability,
        time_grid_reliability=grid_reliability,
        flow_phase=float(values.get("flow_phase") or 0.5),
    )


def observation_sequence_from_flow_evidence_record(
    record: Mapping[str, Any],
    *,
    method_variant: str,
) -> list[FlowEvidenceObservation]:
    """读取真实逐 phase 观测序列; 历史聚合记录只形成单步兼容观测。"""

    raw_sequence = record.get("flow_state_observation_sequence")
    if raw_sequence is not None:
        if not isinstance(raw_sequence, list) or not raw_sequence:
            raise ValueError("flow_state_observation_sequence 必须是非空对象列表")
        if not all(isinstance(row, Mapping) for row in raw_sequence):
            raise TypeError("flow_state_observation_sequence 每一项必须为对象")
        return [
            _observation_from_mapping(row, method_variant=method_variant)
            for row in raw_sequence
        ]
    # 该分支只保留轻量单元测试和旧诊断记录兼容。正式 paper records 的公共门禁
    # 会要求 measured observation sequence, 不允许单步聚合观测支持状态空间主张。
    return [_observation_from_mapping(record, method_variant=method_variant)]


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
    """估计严格对角正定协方差, 适配 probe_paper 小样本和稳定复现。"""

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


def _fit_calibrated_posterior_model(
    observation_sequences: Sequence[Sequence[FlowEvidenceObservation]],
    labels: Sequence[int],
    group_ids: Sequence[str],
) -> CalibratedFlowPosteriorModel:
    """按视频簇交叉拟合双假设 SSM, 再对 OOF LLR 执行 Platt calibration。"""

    import numpy as np

    y = np.asarray(labels, dtype=np.float64)
    unique_groups = sorted(set(group_ids))
    if len(unique_groups) < 2:
        raise ValueError("概率后验 calibration 至少需要2个独立视频簇")
    labels_by_group: dict[str, set[int]] = {}
    for group_id, label in zip(group_ids, labels):
        labels_by_group.setdefault(group_id, set()).add(int(label))
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
    fold_count = min(5, len(groups_by_label[0]), len(groups_by_label[1]))
    fold_by_group: dict[str, int] = {}
    groups_by_label_signature: dict[tuple[int, ...], list[str]] = {}
    for group_id, group_labels in labels_by_group.items():
        groups_by_label_signature.setdefault(tuple(sorted(group_labels)), []).append(
            group_id
        )
    for signature in sorted(groups_by_label_signature):
        for index, group_id in enumerate(sorted(groups_by_label_signature[signature])):
            fold_by_group[group_id] = index % fold_count
    fold_ids = np.asarray([fold_by_group[group_id] for group_id in group_ids])
    out_of_fold_llrs = np.full(len(y), np.nan, dtype=np.float64)
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
    calibrated_logits = platt_intercept + platt_slope * out_of_fold_llrs
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(calibrated_logits, -40.0, 40.0)))
    brier, log_loss_value, ece = _calibration_metrics(probabilities, y, group_ids)
    group_indices = {
        group_id: [
            index for index, current_group_id in enumerate(group_ids)
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
        -(probability * log(max(probability, 1e-8))
          + (1.0 - probability) * log(max(1.0 - probability, 1e-8)))
        for probability in probabilities
    ]
    positive_entropies = [
        mean(entropy_by_record[index] for index in group_indices[group_id])
        for group_id in groups_by_label[1]
    ]
    return CalibratedFlowPosteriorModel(
        feature_names=POSTERIOR_FEATURE_NAMES,
        feature_means=full_model.feature_means,
        feature_scales=full_model.feature_scales,
        negative_state_space_model=full_model.negative_state_space_model,
        positive_state_space_model=full_model.positive_state_space_model,
        platt_slope=platt_slope,
        platt_intercept=float(platt_intercept),
        admissibility_thresholds={
            **{
                name: _quantile(values, 0.05)
                for name, values in quality.items()
            },
            "posterior_entropy_maximum": _quantile(positive_entropies, 0.95),
        },
        calibration_brier_score=brier,
        calibration_log_loss=log_loss_value,
        calibration_expected_calibration_error=ece,
        calibration_group_count=len(unique_groups),
        calibration_record_count=len(observation_sequences),
    )


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


def score_flow_evidence_record(
    record: Mapping[str, Any],
    posterior_model: CalibratedFlowPosteriorModel,
    *,
    method_variant: str,
) -> dict[str, Any]:
    """使用冻结概率模型计算一个 comparison unit。"""

    observations = observation_sequence_from_flow_evidence_record(
        record,
        method_variant=method_variant,
    )
    posterior = infer_flow_state_posterior(observations, posterior_model)
    payload = posterior.as_dict()
    if method_variant == "without_flow_state_admissibility":
        payload["flow_state_admissibility_status"] = "disabled_by_ablation"
        payload["flow_state_admissibility_failures"] = []
        payload["S_final_conservative"] = payload["S_final_unconstrained"]
    payload["flow_detector_score_source"] = (
        FLOW_STATE_POSTERIOR_SCORE_SOURCE
    )
    return payload


def fit_flow_evidence_calibration(
    calibration_records: Iterable[Mapping[str, Any]],
    *,
    method_variant: str,
    target_fpr: float,
) -> FrozenFlowDetectorCalibration:
    """使用 calibration positive/negative 拟合后验, 再由 negative 冻结 FPR 阈值。"""

    if method_variant not in FORMAL_METHOD_VARIANTS:
        raise ValueError(f"未注册的正式 method variant: {method_variant}")
    if not 0.0 < float(target_fpr) < 1.0:
        raise ValueError("target_fpr 必须位于 (0, 1)")
    rows = [dict(row) for row in calibration_records]
    labels = [1 if row.get("sample_role") == "attacked_positive" else 0 for row in rows]
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    if positive_count < 2 or negative_count < 2:
        raise ValueError("概率后验 calibration 至少需要2条 positive 和2条 negative")
    observation_sequences = [
        observation_sequence_from_flow_evidence_record(row, method_variant=method_variant)
        for row in rows
    ]
    model = _fit_calibrated_posterior_model(
        observation_sequences,
        labels,
        [_cluster_id(row) for row in rows],
    )
    negative_rows = [row for row, label in zip(rows, labels) if label == 0]
    negative_scores_by_cluster: dict[str, list[float]] = {}
    for row in negative_rows:
        negative_scores_by_cluster.setdefault(_cluster_id(row), []).append(float(
            score_flow_evidence_record(
            row,
            model,
            method_variant=method_variant,
        )["S_final_conservative"]
        ))
    # 同一 clean video 的多个 key trial 是簇内重复测量。阈值按每视频最大分数
    # 冻结, 使目标 FPR 对应独立视频而不是人为扩增的 trial 数量。
    negative_cluster_scores = [max(values) for values in negative_scores_by_cluster.values()]
    return FrozenFlowDetectorCalibration(
        method_variant=method_variant,
        target_fpr=float(target_fpr),
        posterior_model=model,
        final_score_threshold=_fixed_fpr_threshold(negative_cluster_scores, target_fpr),
        calibration_negative_count=negative_count,
        calibration_positive_count=positive_count,
        calibration_negative_cluster_count=len(negative_scores_by_cluster),
        calibration_positive_cluster_count=len({
            _cluster_id(row) for row, label in zip(rows, labels) if label == 1
        }),
    )


def apply_frozen_flow_detector(
    record: Mapping[str, Any],
    calibration: FrozenFlowDetectorCalibration,
) -> dict[str, Any]:
    """在 held-out record 上应用冻结概率后验, 不更新任何参数。"""

    score_payload = score_flow_evidence_record(
        record,
        calibration.posterior_model,
        method_variant=calibration.method_variant,
    )
    score = float(score_payload["S_final_conservative"])
    return {
        **score_payload,
        "frozen_final_score_threshold": calibration.final_score_threshold,
        "target_fpr": calibration.target_fpr,
        "decision": score >= calibration.final_score_threshold,
        "threshold_source_split": "calibration",
        "test_time_threshold_update_blocked": True,
        "metric_status": "measured_formal",
    }
