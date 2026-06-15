"""提供显式帧匹配时间配准 external baseline 的轻量实现。"""

from __future__ import annotations

from typing import Sequence


Vector = Sequence[float]
SequenceEmbedding = Sequence[Vector]


def _squared_distance(left: Vector, right: Vector) -> float:
    """计算两个 embedding 向量的平方距离, 用于最近邻帧匹配。"""
    if len(left) != len(right):
        raise ValueError("帧匹配输入向量维度必须一致")
    return float(sum((a - b) ** 2 for a, b in zip(left, right)))


def match_frames(reference_sequence: SequenceEmbedding, observed_sequence: SequenceEmbedding, search_radius: int | None = None) -> list[dict]:
    """为 observed 序列中的每个元素寻找最相近的 reference 帧。

    该函数属于通用显式配准写法。`search_radius` 用于模拟只在局部时间窗口内搜索的工程约束;
    当其为 None 时, baseline 使用全局最近邻匹配。
    """
    if not reference_sequence or not observed_sequence:
        raise ValueError("帧匹配输入序列不能为空")

    matches: list[dict] = []
    for observed_index, observed_vector in enumerate(observed_sequence):
        if search_radius is None:
            candidate_indices = range(len(reference_sequence))
        else:
            start = max(0, observed_index - search_radius)
            stop = min(len(reference_sequence), observed_index + search_radius + 1)
            candidate_indices = range(start, stop)

        best_index = min(candidate_indices, key=lambda index: _squared_distance(reference_sequence[index], observed_vector))
        matches.append({
            "observed_index": observed_index,
            "reference_index": best_index,
            "matching_distance": _squared_distance(reference_sequence[best_index], observed_vector),
        })
    return matches


def compute_registration_cost(reference_sequence: SequenceEmbedding, observed_sequence: SequenceEmbedding, search_radius: int | None = None) -> float:
    """计算显式帧匹配配准的平均代价。"""
    matches = match_frames(reference_sequence, observed_sequence, search_radius)
    return float(sum(item["matching_distance"] for item in matches) / len(matches))


def adapter_status() -> dict:
    """返回本地帧匹配 baseline 的可运行状态, 但不自动支撑正向 claim。"""
    return {
        "external_baseline_runnable_status": "runnable",
        "external_baseline_not_run_reason": "none",
        "external_baseline_result_used_for_claim": False,
    }
