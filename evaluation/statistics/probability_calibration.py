"""提供按独立视频簇等权的概率可靠性统计与 bootstrap 区间。"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ProbabilityCalibrationEstimate:
    """保存 held-out 概率指标及其视频簇 bootstrap 区间。"""

    brier_score: float
    log_loss: float
    expected_calibration_error: float
    brier_score_ci_lower: float
    brier_score_ci_upper: float
    log_loss_ci_lower: float
    log_loss_ci_upper: float
    expected_calibration_error_ci_lower: float
    expected_calibration_error_ci_upper: float
    positive_cluster_count: int
    negative_cluster_count: int
    record_count: int
    bootstrap_resample_count: int

    def as_dict(self, *, prefix: str = "heldout_posterior_") -> dict[str, float | int]:
        """使用 governed 字段前缀序列化，便于实验层直接写入 records。"""

        return {
            f"{prefix}brier_score": self.brier_score,
            f"{prefix}brier_score_ci_lower": self.brier_score_ci_lower,
            f"{prefix}brier_score_ci_upper": self.brier_score_ci_upper,
            f"{prefix}log_loss": self.log_loss,
            f"{prefix}log_loss_ci_lower": self.log_loss_ci_lower,
            f"{prefix}log_loss_ci_upper": self.log_loss_ci_upper,
            f"{prefix}expected_calibration_error": (
                self.expected_calibration_error
            ),
            f"{prefix}expected_calibration_error_ci_lower": (
                self.expected_calibration_error_ci_lower
            ),
            f"{prefix}expected_calibration_error_ci_upper": (
                self.expected_calibration_error_ci_upper
            ),
            f"{prefix}positive_cluster_count": self.positive_cluster_count,
            f"{prefix}negative_cluster_count": self.negative_cluster_count,
            f"{prefix}record_count": self.record_count,
            f"{prefix}bootstrap_resample_count": self.bootstrap_resample_count,
        }


def _validated_inputs(
    probabilities: Iterable[float],
    labels: Iterable[int | bool],
    cluster_ids: Iterable[str],
) -> tuple[list[float], list[int], list[str]]:
    """验证概率、标签和统计簇，拒绝以事件数冒充独立样本数。"""

    probability_values = [float(value) for value in probabilities]
    label_values = [int(value) for value in labels]
    cluster_values = [str(value) for value in cluster_ids]
    if not probability_values:
        raise ValueError("概率可靠性评测至少需要一条记录")
    if not (
        len(probability_values) == len(label_values) == len(cluster_values)
    ):
        raise ValueError("probabilities、labels 与 cluster_ids 长度必须一致")
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probability_values):
        raise ValueError("概率必须是 [0, 1] 内的有限数")
    if any(value not in {0, 1} for value in label_values):
        raise ValueError("二元标签只能为0或1")
    if any(not value.strip() for value in cluster_values):
        raise ValueError("cluster_ids 必须是非空字符串")
    for label in (0, 1):
        if not any(value == label for value in label_values):
            raise ValueError("概率可靠性评测必须同时包含 positive 与 negative")
    return probability_values, label_values, cluster_values


def _balanced_record_weights(
    labels: Sequence[int],
    cluster_ids: Sequence[str],
) -> list[float]:
    """先平衡类别，再让每个独立视频簇获得相同的类内权重。"""

    weights = [0.0 for _ in labels]
    for label in (0, 1):
        class_groups = sorted({
            cluster_id
            for current_label, cluster_id in zip(labels, cluster_ids)
            if current_label == label
        })
        if not class_groups:
            raise ValueError("概率可靠性评测必须同时包含 positive 与 negative")
        for group_id in class_groups:
            indices = [
                index
                for index, (current_label, cluster_id) in enumerate(
                    zip(labels, cluster_ids)
                )
                if current_label == label and cluster_id == group_id
            ]
            group_weight = 0.5 / len(class_groups)
            for index in indices:
                weights[index] = group_weight / len(indices)
    return weights


def cluster_balanced_probability_metrics(
    probabilities: Iterable[float],
    labels: Iterable[int | bool],
    cluster_ids: Iterable[str],
    *,
    calibration_bin_count: int = 10,
) -> tuple[float, float, float]:
    """计算类平衡且按独立视频簇等权的 Brier、log loss 与 ECE。"""

    probability_values, label_values, cluster_values = _validated_inputs(
        probabilities,
        labels,
        cluster_ids,
    )
    if calibration_bin_count < 2:
        raise ValueError("calibration_bin_count 必须至少为2")
    weights = _balanced_record_weights(label_values, cluster_values)
    clipped = [min(max(value, 1e-8), 1.0 - 1e-8) for value in probability_values]
    brier_score = sum(
        weight * (probability - label) ** 2
        for probability, label, weight in zip(clipped, label_values, weights)
    )
    log_loss = -sum(
        weight
        * (
            label * math.log(probability)
            + (1 - label) * math.log(1.0 - probability)
        )
        for probability, label, weight in zip(clipped, label_values, weights)
    )
    expected_calibration_error = 0.0
    for bin_index in range(calibration_bin_count):
        lower = bin_index / calibration_bin_count
        upper = (bin_index + 1) / calibration_bin_count
        indices = [
            index
            for index, probability in enumerate(clipped)
            if lower <= probability < upper
            or (bin_index == calibration_bin_count - 1 and probability == upper)
        ]
        if not indices:
            continue
        bin_weight = sum(weights[index] for index in indices)
        confidence = sum(
            weights[index] * clipped[index] for index in indices
        ) / bin_weight
        accuracy = sum(
            weights[index] * label_values[index] for index in indices
        ) / bin_weight
        expected_calibration_error += bin_weight * abs(confidence - accuracy)
    return (
        float(brier_score),
        float(log_loss),
        float(expected_calibration_error),
    )


def _percentile(values: Sequence[float], probability: float) -> float:
    """在线性插值下计算确定性分位点。"""

    if not values:
        raise ValueError("分位点输入不能为空")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return ordered[lower_index]
    fraction = position - lower_index
    return ordered[lower_index] * (1.0 - fraction) + ordered[upper_index] * fraction


def clustered_probability_calibration_interval(
    probabilities: Iterable[float],
    labels: Iterable[int | bool],
    cluster_ids: Iterable[str],
    *,
    bootstrap_resample_count: int = 5000,
    confidence_level: float = 0.95,
    purpose: str = "heldout_posterior_calibration",
) -> ProbabilityCalibrationEstimate:
    """按类别分别重采样独立视频簇，估计 held-out 概率指标区间。

    同一视频簇中的多攻击或多 key trial 会作为整体被重采样，避免把簇内重复
    测量误当成更多独立样本。类别分别重采样用于保持每次 replicate 都包含
    H0 与 H1，并与点估计的 class-balanced 解释一致。
    """

    probability_values, label_values, cluster_values = _validated_inputs(
        probabilities,
        labels,
        cluster_ids,
    )
    if bootstrap_resample_count < 20:
        raise ValueError("bootstrap_resample_count 必须至少为20")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level 必须位于 (0, 1)")
    groups_by_label = {
        label: sorted({
            cluster_id
            for current_label, cluster_id in zip(label_values, cluster_values)
            if current_label == label
        })
        for label in (0, 1)
    }
    rows_by_label_group = {
        (label, group_id): [
            index
            for index, (current_label, cluster_id) in enumerate(
                zip(label_values, cluster_values)
            )
            if current_label == label and cluster_id == group_id
        ]
        for label in (0, 1)
        for group_id in groups_by_label[label]
    }
    seed_payload = {
        "probabilities": probability_values,
        "labels": label_values,
        "cluster_ids": cluster_values,
        "purpose": str(purpose),
    }
    seed = int.from_bytes(
        sha256(
            json.dumps(seed_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).digest()[:8],
        "big",
    )
    # 先把每个视频簇压缩成 Brier、log loss 和 ECE 分箱充分统计量，再使用
    # multinomial counts 表示“有放回抽取同样多视频簇”。与逐条复制记录相比，
    # 该实现保持完全相同的簇 bootstrap 语义，同时能处理 full_paper 的大样本。
    import numpy as np

    clipped = np.clip(
        np.asarray(probability_values, dtype=np.float64),
        1e-8,
        1.0 - 1e-8,
    )
    label_array = np.asarray(label_values, dtype=np.int64)
    group_summaries: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    calibration_bin_count = 10
    for label in (0, 1):
        group_brier: list[float] = []
        group_log_loss: list[float] = []
        group_bin_probability_mass: list[list[float]] = []
        group_bin_label_mass: list[list[float]] = []
        for group_id in groups_by_label[label]:
            indices = np.asarray(rows_by_label_group[(label, group_id)], dtype=np.int64)
            group_probabilities = clipped[indices]
            group_labels = label_array[indices]
            group_brier.append(float(np.mean((group_probabilities - group_labels) ** 2)))
            group_log_loss.append(float(-np.mean(
                group_labels * np.log(group_probabilities)
                + (1 - group_labels) * np.log(1.0 - group_probabilities)
            )))
            probability_mass = [0.0] * calibration_bin_count
            label_mass = [0.0] * calibration_bin_count
            for probability, current_label in zip(group_probabilities, group_labels):
                bin_index = min(
                    int(float(probability) * calibration_bin_count),
                    calibration_bin_count - 1,
                )
                probability_mass[bin_index] += float(probability) / len(indices)
                label_mass[bin_index] += float(current_label) / len(indices)
            group_bin_probability_mass.append(probability_mass)
            group_bin_label_mass.append(label_mass)
        group_summaries[label] = (
            np.asarray(group_brier, dtype=np.float64),
            np.asarray(group_log_loss, dtype=np.float64),
            np.asarray(group_bin_probability_mass, dtype=np.float64),
            np.asarray(group_bin_label_mass, dtype=np.float64),
        )

    random_generator = np.random.default_rng(seed)
    brier_values: list[float] = []
    log_loss_values: list[float] = []
    ece_values: list[float] = []
    completed = 0
    while completed < bootstrap_resample_count:
        # Windows 上部分 BLAS/OpenMP 组合在 pytest 长进程中会让小型整数-浮点
        # 矩阵乘法触发原生进程中止。这里使用不委托 BLAS 的 einsum 收缩；统计
        # 结果完全相同，同时避免全量测试与服务器后处理争用原生线程池。
        chunk_size = min(64, bootstrap_resample_count - completed)
        chunk_brier = np.zeros(chunk_size, dtype=np.float64)
        chunk_log_loss = np.zeros(chunk_size, dtype=np.float64)
        chunk_probability_mass = np.zeros(
            (chunk_size, calibration_bin_count),
            dtype=np.float64,
        )
        chunk_label_mass = np.zeros_like(chunk_probability_mass)
        for label in (0, 1):
            (
                group_brier,
                group_log_loss,
                group_probability_mass,
                group_label_mass,
            ) = group_summaries[label]
            group_count = len(group_brier)
            bootstrap_counts = random_generator.multinomial(
                group_count,
                np.full(group_count, 1.0 / group_count, dtype=np.float64),
                size=chunk_size,
            )
            class_scale = 0.5 / group_count
            chunk_brier += class_scale * np.einsum(
                "cg,g->c",
                bootstrap_counts,
                group_brier,
                optimize=False,
            )
            chunk_log_loss += class_scale * np.einsum(
                "cg,g->c",
                bootstrap_counts,
                group_log_loss,
                optimize=False,
            )
            chunk_probability_mass += class_scale * (
                np.einsum(
                    "cg,gb->cb",
                    bootstrap_counts,
                    group_probability_mass,
                    optimize=False,
                )
            )
            chunk_label_mass += class_scale * (
                np.einsum(
                    "cg,gb->cb",
                    bootstrap_counts,
                    group_label_mass,
                    optimize=False,
                )
            )
        chunk_ece = np.sum(
            np.abs(chunk_probability_mass - chunk_label_mass),
            axis=1,
        )
        brier_values.extend(float(value) for value in chunk_brier)
        log_loss_values.extend(float(value) for value in chunk_log_loss)
        ece_values.extend(float(value) for value in chunk_ece)
        completed += chunk_size
    point_brier, point_log_loss, point_ece = cluster_balanced_probability_metrics(
        probability_values,
        label_values,
        cluster_values,
    )
    alpha = (1.0 - confidence_level) / 2.0
    return ProbabilityCalibrationEstimate(
        brier_score=point_brier,
        log_loss=point_log_loss,
        expected_calibration_error=point_ece,
        brier_score_ci_lower=_percentile(brier_values, alpha),
        brier_score_ci_upper=_percentile(brier_values, 1.0 - alpha),
        log_loss_ci_lower=_percentile(log_loss_values, alpha),
        log_loss_ci_upper=_percentile(log_loss_values, 1.0 - alpha),
        expected_calibration_error_ci_lower=_percentile(ece_values, alpha),
        expected_calibration_error_ci_upper=_percentile(ece_values, 1.0 - alpha),
        positive_cluster_count=len(groups_by_label[1]),
        negative_cluster_count=len(groups_by_label[0]),
        record_count=len(probability_values),
        bootstrap_resample_count=bootstrap_resample_count,
    )
