"""按 source-video cluster 计算 rate 与配对效应区间。"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import exp, lgamma, log
import random
from statistics import mean
from typing import Iterable, Mapping, Sequence


def _binomial_cdf(success_count: int, trial_count: int, probability: float) -> float:
    """使用 log-sum-exp 稳定计算二项分布累计概率。"""

    if success_count >= trial_count:
        return 1.0
    if probability <= 0.0:
        return 1.0
    if probability >= 1.0:
        return 0.0
    logs = [
        lgamma(trial_count + 1)
        - lgamma(index + 1)
        - lgamma(trial_count - index + 1)
        + index * log(probability)
        + (trial_count - index) * log(1.0 - probability)
        for index in range(success_count + 1)
    ]
    maximum = max(logs)
    return min(1.0, exp(maximum) * sum(exp(value - maximum) for value in logs))


def one_sided_binomial_upper_bound(
    success_count: int,
    trial_count: int,
    *,
    confidence_level: float = 0.95,
) -> float:
    """计算 exact Clopper-Pearson 单侧上界。

    该函数用于 held-out FPR。输入 trial 必须已经按 source video 聚合, 禁止把
    同一视频的 key trial 数量作为二项分布样本量。
    """

    successes = int(success_count)
    trials = int(trial_count)
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("二项区间需要 0 <= success_count <= trial_count")
    if successes == trials:
        return 1.0
    alpha = 1.0 - float(confidence_level)
    lower, upper = 0.0, 1.0
    for _ in range(80):
        midpoint = (lower + upper) / 2.0
        if _binomial_cdf(successes, trials, midpoint) > alpha:
            lower = midpoint
        else:
            upper = midpoint
    return upper


@dataclass(frozen=True)
class ClusteredEstimate:
    """保存 cluster-equal-weight 点估计和 bootstrap 区间。"""

    estimate: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    cluster_count: int
    observation_count: int
    bootstrap_resample_count: int

    def as_dict(self, prefix: str) -> dict[str, float | int]:
        """使用稳定字段前缀导出统计结果。"""

        return {
            f"{prefix}_estimate": round(self.estimate, 8),
            f"{prefix}_ci_95_lower": round(self.confidence_interval_lower, 8),
            f"{prefix}_ci_95_upper": round(self.confidence_interval_upper, 8),
            f"{prefix}_cluster_count": self.cluster_count,
            f"{prefix}_observation_count": self.observation_count,
            f"{prefix}_bootstrap_resample_count": self.bootstrap_resample_count,
        }


def _percentile(values: Sequence[float], probability: float) -> float:
    rows = sorted(float(value) for value in values)
    if not rows:
        raise ValueError("bootstrap percentile 不能为空")
    position = max(0, min(len(rows) - 1, round(probability * (len(rows) - 1))))
    return rows[position]


def _bootstrap_seed(cluster_ids: Iterable[str], purpose: str) -> int:
    payload = purpose + "::" + "::".join(sorted(str(value) for value in cluster_ids))
    return int(sha256(payload.encode("utf-8")).hexdigest()[:16], 16)


def clustered_mean_interval(
    values_by_cluster: Mapping[str, Sequence[float]],
    *,
    bootstrap_resamples: int = 5000,
    confidence_level: float = 0.95,
    purpose: str = "clustered_mean",
) -> ClusteredEstimate:
    """先在簇内求均值, 再以簇为独立单位 bootstrap。

    该实现不会把同一视频上的不同 key trial、攻击强度或重复检测当成独立视频。
    每个 source-video cluster 在总体估计中具有相同权重。
    """

    rows = {
        str(cluster_id): [float(value) for value in values]
        for cluster_id, values in values_by_cluster.items()
        if values
    }
    if len(rows) < 2:
        raise ValueError("cluster-aware inference 至少需要2个独立簇")
    if bootstrap_resamples < 200:
        raise ValueError("cluster bootstrap 至少需要200次重采样")
    cluster_ids = sorted(rows)
    cluster_means = [mean(rows[cluster_id]) for cluster_id in cluster_ids]
    estimate = mean(cluster_means)
    generator = random.Random(_bootstrap_seed(cluster_ids, purpose))
    bootstrap_values = [
        mean(generator.choice(cluster_means) for _ in cluster_means)
        for _ in range(int(bootstrap_resamples))
    ]
    alpha = 1.0 - float(confidence_level)
    return ClusteredEstimate(
        estimate=estimate,
        confidence_interval_lower=_percentile(bootstrap_values, alpha / 2.0),
        confidence_interval_upper=_percentile(bootstrap_values, 1.0 - alpha / 2.0),
        cluster_count=len(cluster_ids),
        observation_count=sum(len(values) for values in rows.values()),
        bootstrap_resample_count=int(bootstrap_resamples),
    )


def clustered_binary_rate_interval(
    records: Iterable[Mapping[str, object]],
    *,
    outcome_field: str,
    cluster_field: str = "statistical_cluster_id",
    bootstrap_resamples: int = 5000,
    purpose: str = "clustered_binary_rate",
) -> ClusteredEstimate:
    """计算二元事件率, 独立统计单位固定为 source-video cluster。"""

    grouped: dict[str, list[float]] = {}
    for record in records:
        cluster_id = str(record.get(cluster_field) or "")
        if not cluster_id:
            raise KeyError(f"统计记录缺少 {cluster_field}")
        grouped.setdefault(cluster_id, []).append(float(bool(record.get(outcome_field))))
    return clustered_mean_interval(
        grouped,
        bootstrap_resamples=bootstrap_resamples,
        purpose=purpose,
    )


def paired_cluster_difference_interval(
    paired_rows: Iterable[Mapping[str, object]],
    *,
    difference_field: str,
    cluster_field: str = "statistical_cluster_id",
    bootstrap_resamples: int = 5000,
    purpose: str = "paired_cluster_difference",
) -> ClusteredEstimate:
    """对同 prompt/seed/attack 配对差值执行 cluster bootstrap。"""

    grouped: dict[str, list[float]] = {}
    for row in paired_rows:
        cluster_id = str(row.get(cluster_field) or "")
        if not cluster_id:
            raise KeyError(f"配对记录缺少 {cluster_field}")
        value = row.get(difference_field)
        if value is None:
            raise KeyError(f"配对记录缺少 {difference_field}")
        grouped.setdefault(cluster_id, []).append(float(value))
    return clustered_mean_interval(
        grouped,
        bootstrap_resamples=bootstrap_resamples,
        purpose=purpose,
    )
