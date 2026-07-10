"""拟合分组交叉验证概率后验并执行固定 FPR SSTW 检测。"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import floor, inf, log, nextafter
from typing import Any, Iterable, Mapping, Sequence

from main.methods.state_space_watermark.flow_state_posterior import (
    POSTERIOR_FEATURE_NAMES,
    CalibratedFlowPosteriorModel,
    FlowEvidenceObservation,
    infer_flow_state_posterior,
    posterior_feature_vector,
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


@dataclass(frozen=True)
class FrozenFlowDetectorCalibration:
    """保存一个 method variant 的概率模型与 fixed-FPR 阈值。"""

    method_variant: str
    target_fpr: float
    posterior_model: CalibratedFlowPosteriorModel
    final_score_threshold: float
    calibration_negative_count: int
    calibration_positive_count: int

    def as_dict(self) -> dict[str, Any]:
        """转换为可重建 detection 的 threshold artifact。"""

        return {
            "method_variant": self.method_variant,
            "target_fpr": self.target_fpr,
            "calibration_negative_count": self.calibration_negative_count,
            "calibration_positive_count": self.calibration_positive_count,
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

    feature_names = tuple(str(value) for value in payload["posterior_feature_names"])
    if feature_names != POSTERIOR_FEATURE_NAMES:
        raise ValueError("threshold artifact 的概率后验特征顺序不兼容")
    model = CalibratedFlowPosteriorModel(
        feature_names=feature_names,
        feature_means=tuple(float(value) for value in payload["posterior_feature_means"]),
        feature_scales=tuple(float(value) for value in payload["posterior_feature_scales"]),
        coefficients=tuple(float(value) for value in payload["posterior_coefficients"]),
        intercept=float(payload["posterior_intercept"]),
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
    """把固定路径、endpoint 与 replay hypothesis 转换为概率模型观测。"""

    endpoint = _record_value(record, "endpoint_score")
    velocity = _record_value(record, "S_velocity")
    path = _record_value(record, "S_path_inv")
    consistency = float(record.get("path_endpoint_consistency") or 0.0)
    coverage = float(record.get("endpoint_coverage_ratio") or 0.0)
    replay_reliability = float(record.get("replay_reliability_weight") or 0.0)
    grid_reliability = float(record.get("time_grid_reliability") or replay_reliability)
    replay_log_likelihood_ratio = float(
        record.get("replay_log_likelihood_ratio_mean") or 0.0
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
    return FlowEvidenceObservation(
        endpoint_score=endpoint,
        velocity_score=velocity,
        path_score=path,
        path_endpoint_consistency=consistency,
        replay_log_likelihood_ratio=replay_log_likelihood_ratio,
        coverage_ratio=coverage,
        replay_reliability=replay_reliability,
        time_grid_reliability=grid_reliability,
        flow_phase=float(record.get("flow_phase") or 0.5),
    )


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


def _fit_standardized_logistic(
    matrix: Any,
    labels: Any,
    group_ids: Sequence[str],
) -> tuple[Any, Any, Any, float]:
    """冻结特征标准化并拟合 class-balanced logistic。"""

    import numpy as np

    x = np.asarray(matrix, dtype=np.float64)
    means = x.mean(axis=0)
    scales = x.std(axis=0)
    scales = np.where(scales < 1e-8, 1.0, scales)
    standardized = (x - means) / scales
    coefficients, intercept = _fit_logistic(
        standardized,
        labels,
        sample_weights=_balanced_weights(labels, group_ids),
        l2_penalty=0.1,
    )
    return means, scales, coefficients, intercept


def _raw_logits(matrix: Any, model: tuple[Any, Any, Any, float]) -> Any:
    import numpy as np

    means, scales, coefficients, intercept = model
    x = np.asarray(matrix, dtype=np.float64)
    return intercept + ((x - means) / scales) @ coefficients


def _stable_fold(group_id: str, fold_count: int) -> int:
    digest = sha256(group_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % fold_count


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
    observations: Sequence[FlowEvidenceObservation],
    labels: Sequence[int],
    group_ids: Sequence[str],
) -> CalibratedFlowPosteriorModel:
    """按视频簇交叉拟合 logistic, 再用 OOF logits 执行 Platt calibration。"""

    import numpy as np

    matrix = np.asarray([posterior_feature_vector(row) for row in observations], dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    unique_groups = sorted(set(group_ids))
    if len(unique_groups) < 2:
        raise ValueError("概率后验 calibration 至少需要2个独立视频簇")
    fold_count = min(5, len(unique_groups))
    group_array = np.asarray(list(group_ids))
    fold_ids = np.asarray([_stable_fold(group_id, fold_count) for group_id in group_ids])
    out_of_fold_logits = np.full(len(y), np.nan, dtype=np.float64)
    for fold in range(fold_count):
        validation_mask = fold_ids == fold
        training_mask = ~validation_mask
        if not bool(validation_mask.any()) or len(set(y[training_mask].tolist())) < 2:
            continue
        fold_model = _fit_standardized_logistic(
            matrix[training_mask],
            y[training_mask],
            group_array[training_mask].tolist(),
        )
        out_of_fold_logits[validation_mask] = _raw_logits(matrix[validation_mask], fold_model)
    full_model = _fit_standardized_logistic(matrix, y, group_ids)
    missing_mask = np.isnan(out_of_fold_logits)
    if bool(missing_mask.any()):
        out_of_fold_logits[missing_mask] = _raw_logits(matrix[missing_mask], full_model)

    platt_coefficients, platt_intercept = _fit_logistic(
        out_of_fold_logits.reshape(-1, 1),
        y,
        sample_weights=_balanced_weights(y, group_ids),
        l2_penalty=0.01,
    )
    platt_slope = float(platt_coefficients[0])
    calibrated_logits = platt_intercept + platt_slope * out_of_fold_logits
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(calibrated_logits, -40.0, 40.0)))
    brier, log_loss_value, ece = _calibration_metrics(probabilities, y, group_ids)
    means, scales, coefficients, intercept = full_model
    quality = {
        "coverage": [row.coverage_ratio for row in observations],
        "replay_reliability": [row.replay_reliability for row in observations],
        "time_grid_reliability": [row.time_grid_reliability for row in observations],
    }
    return CalibratedFlowPosteriorModel(
        feature_names=POSTERIOR_FEATURE_NAMES,
        feature_means=tuple(float(value) for value in means),
        feature_scales=tuple(float(value) for value in scales),
        coefficients=tuple(float(value) for value in coefficients),
        intercept=float(intercept),
        platt_slope=platt_slope,
        platt_intercept=float(platt_intercept),
        admissibility_thresholds={
            name: _quantile(values, 0.05)
            for name, values in quality.items()
        },
        calibration_brier_score=brier,
        calibration_log_loss=log_loss_value,
        calibration_expected_calibration_error=ece,
        calibration_group_count=len(unique_groups),
        calibration_record_count=len(observations),
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

    observation = observation_from_flow_evidence_record(record, method_variant=method_variant)
    posterior = infer_flow_state_posterior([observation], posterior_model)
    payload = posterior.as_dict()
    if method_variant == "without_flow_state_admissibility":
        payload["flow_state_admissibility_status"] = "disabled_by_ablation"
        payload["flow_state_admissibility_failures"] = []
        payload["S_final_conservative"] = payload["S_final_unconstrained"]
    payload["flow_detector_score_source"] = (
        "group_cross_fitted_calibrated_probability_posterior"
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
    observations = [
        observation_from_flow_evidence_record(row, method_variant=method_variant)
        for row in rows
    ]
    model = _fit_calibrated_posterior_model(
        observations,
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
