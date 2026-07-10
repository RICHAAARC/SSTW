"""冻结 calibration negative 参数并计算 SSTW 多证据正式检测分数。"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, exp, floor, inf, nextafter
from statistics import mean, pstdev
from typing import Any, Iterable, Mapping

from main.methods.state_space_watermark.flow_state_posterior import (
    FlowEvidenceCalibration,
    FlowEvidenceObservation,
    infer_flow_state_posterior,
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
    """保存一个 method variant 的冻结标准化参数与 fixed-FPR 阈值。"""

    method_variant: str
    target_fpr: float
    evidence_calibration: FlowEvidenceCalibration
    final_score_threshold: float
    calibration_negative_count: int

    def as_dict(self) -> dict[str, Any]:
        """转换为可重建 detection 的 threshold artifact。"""

        return {
            "method_variant": self.method_variant,
            "target_fpr": self.target_fpr,
            "calibration_negative_count": self.calibration_negative_count,
            "evidence_means": dict(self.evidence_calibration.means),
            "evidence_standard_deviations": dict(self.evidence_calibration.standard_deviations),
            "admissibility_thresholds": dict(self.evidence_calibration.admissibility_thresholds),
            "flow_process_variance": self.evidence_calibration.process_variance,
            "flow_observation_variance": self.evidence_calibration.observation_variance,
            "frozen_final_score_threshold": self.final_score_threshold,
            "threshold_source_split": "calibration_negative",
            "test_time_threshold_update_blocked": True,
        }


def _quantile(values: Iterable[float], quantile: float) -> float:
    rows = sorted(float(value) for value in values)
    if not rows:
        raise ValueError("quantile 需要至少一个数值")
    position = max(0, min(len(rows) - 1, ceil(float(quantile) * len(rows)) - 1))
    return rows[position]


def _record_value(record: Mapping[str, Any], field_name: str) -> float:
    value = record.get(field_name)
    if value is None:
        raise KeyError(f"正式 Flow evidence 缺少字段: {field_name}")
    return float(value)


def _sigmoid(value: float) -> float:
    """把 calibration negative 的原始分数阈值映射到统一状态空间。"""

    value = max(-60.0, min(60.0, float(value)))
    return 1.0 / (1.0 + exp(-value))


def _normalized_threshold(values: list[float], target_fpr: float) -> float:
    """计算与 `infer_flow_state_posterior` 相同标准化口径的门禁阈值。"""

    center = mean(values)
    scale = max(pstdev(values), 1e-6)
    raw_threshold = _quantile(values, 1.0 - target_fpr)
    return _sigmoid((raw_threshold - center) / scale)


def _fixed_fpr_threshold(values: Iterable[float], target_fpr: float) -> float:
    """选择使 calibration empirical FPR 不超过目标值的冻结阈值。

    判定规则使用 `score >= threshold`, 因而普通分位数在边界相等时可能多接收
    一个或多个 negative。此实现显式处理并列值, 保证 calibration 侧的经验 FPR
    不会因为比较符号而超过契约。
    """

    rows = sorted(float(value) for value in values)
    if not rows:
        raise ValueError("fixed-FPR threshold 需要至少一个数值")
    allowed_false_positives = floor(float(target_fpr) * len(rows) + 1e-12)
    if allowed_false_positives <= 0:
        return nextafter(rows[-1], inf)
    # 从低到高寻找第一个满足预算的阈值, 以在不超 FPR 的前提下保留最大检出功效。
    for candidate in sorted(set(rows)):
        accepted = sum(value >= candidate for value in rows)
        if accepted <= allowed_false_positives:
            return candidate
    return nextafter(rows[-1], inf)


def observation_from_flow_evidence_record(
    record: Mapping[str, Any],
    *,
    method_variant: str,
) -> FlowEvidenceObservation:
    """把 endpoint、path 与 replay records 转换为统一状态观测。"""

    endpoint = _record_value(record, "endpoint_score")
    velocity = _record_value(record, "S_velocity")
    path = _record_value(record, "S_path_inv")
    consistency = float(record.get("path_endpoint_consistency") or 0.0)
    coverage = float(record.get("endpoint_coverage_ratio") or 0.0)
    replay_reliability = float(record.get("replay_reliability_weight") or 0.0)
    grid_reliability = float(record.get("time_grid_reliability") or replay_reliability)
    if method_variant == "endpoint_only_control":
        velocity = endpoint
        path = endpoint
        consistency = 1.0
    elif method_variant == "trajectory_only_score":
        endpoint = path
    elif method_variant == "without_velocity_constraint":
        velocity = 0.0
    elif method_variant == "without_replay_uncertainty_weighting":
        replay_reliability = 1.0
    return FlowEvidenceObservation(
        endpoint_score=endpoint,
        velocity_score=velocity,
        path_score=path,
        path_endpoint_consistency=consistency,
        coverage_ratio=coverage,
        replay_reliability=replay_reliability,
        time_grid_reliability=grid_reliability,
        flow_phase=float(record.get("flow_phase") or 0.5),
    )


def score_flow_evidence_record(
    record: Mapping[str, Any],
    calibration: FlowEvidenceCalibration,
    *,
    method_variant: str,
) -> dict[str, Any]:
    """使用冻结参数计算一个 comparison unit 的正式多证据分数。"""

    observation = observation_from_flow_evidence_record(record, method_variant=method_variant)
    if method_variant == "generic_ssm_baseline":
        score = mean([
            observation.endpoint_score,
            observation.velocity_score,
            observation.path_score,
        ])
        return {
            "flow_state_admissibility_status": "not_applicable_generic_ssm",
            "flow_state_admissibility_failures": [],
            "S_final_conservative": round(max(0.0, min(1.0, score)), 8),
            "flow_detector_score_source": "generic_unconditioned_temporal_mean_baseline",
        }
    posterior = infer_flow_state_posterior([observation], calibration)
    payload = posterior.as_dict()
    if method_variant == "without_flow_state_admissibility":
        payload["flow_state_admissibility_status"] = "disabled_by_ablation"
        payload["flow_state_admissibility_failures"] = []
        payload["S_final_conservative"] = payload["S_final_unconstrained"]
    payload["flow_detector_score_source"] = "endpoint_path_replay_state_posterior"
    return payload


def fit_flow_evidence_calibration(
    calibration_negative_records: Iterable[Mapping[str, Any]],
    *,
    method_variant: str,
    target_fpr: float,
) -> FrozenFlowDetectorCalibration:
    """只使用 calibration negative 冻结标准化参数、admissibility 和最终阈值。"""

    if method_variant not in FORMAL_METHOD_VARIANTS:
        raise ValueError(f"未注册的正式 method variant: {method_variant}")
    if not 0.0 < float(target_fpr) < 1.0:
        raise ValueError("target_fpr 必须位于 (0, 1)")
    rows = list(calibration_negative_records)
    if len(rows) < 2:
        raise ValueError("冻结 Flow detector 至少需要2条 calibration negative")
    observations = [observation_from_flow_evidence_record(record, method_variant=method_variant) for record in rows]
    values = {
        "endpoint": [observation.endpoint_score for observation in observations],
        "velocity": [observation.velocity_score for observation in observations],
        "path": [observation.path_score for observation in observations],
    }
    evidence_calibration = FlowEvidenceCalibration(
        means={name: mean(items) for name, items in values.items()},
        standard_deviations={name: max(pstdev(items), 1e-6) for name, items in values.items()},
        admissibility_thresholds={
            # endpoint 与 path 在 posterior 中会先经过 calibration negative 标准化,
            # 因而这里必须保存同一 [0, 1] 状态空间中的阈值, 不能混用原始分数。
            "endpoint": _normalized_threshold(values["endpoint"], target_fpr),
            "path": _normalized_threshold(values["path"], target_fpr),
            "path_endpoint_consistency": _quantile(
                [observation.path_endpoint_consistency for observation in observations],
                1.0 - target_fpr,
            ),
            "posterior_confidence": 0.5,
            "coverage": _quantile(
                [observation.coverage_ratio for observation in observations],
                min(0.5, 1.0 - target_fpr),
            ),
            "replay_reliability": _quantile(
                [observation.replay_reliability for observation in observations],
                min(0.5, 1.0 - target_fpr),
            ),
            "time_grid_reliability": _quantile(
                [observation.time_grid_reliability for observation in observations],
                min(0.5, 1.0 - target_fpr),
            ),
        },
    )
    scored = [
        float(score_flow_evidence_record(record, evidence_calibration, method_variant=method_variant)["S_final_conservative"])
        for record in rows
    ]
    threshold = _fixed_fpr_threshold(scored, target_fpr)
    return FrozenFlowDetectorCalibration(
        method_variant=method_variant,
        target_fpr=float(target_fpr),
        evidence_calibration=evidence_calibration,
        final_score_threshold=threshold,
        calibration_negative_count=len(rows),
    )


def apply_frozen_flow_detector(
    record: Mapping[str, Any],
    calibration: FrozenFlowDetectorCalibration,
) -> dict[str, Any]:
    """在 held-out record 上应用冻结 detector, 不允许更新任何参数。"""

    score_payload = score_flow_evidence_record(
        record,
        calibration.evidence_calibration,
        method_variant=calibration.method_variant,
    )
    score = float(score_payload["S_final_conservative"])
    return {
        **score_payload,
        "frozen_final_score_threshold": calibration.final_score_threshold,
        "target_fpr": calibration.target_fpr,
        "decision": score >= calibration.final_score_threshold,
        "threshold_source_split": "calibration_negative",
        "test_time_threshold_update_blocked": True,
        "metric_status": "measured_formal",
    }
