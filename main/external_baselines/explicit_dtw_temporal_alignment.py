"""提供显式 DTW 时间对齐 external baseline 的轻量实现。"""

from __future__ import annotations

from math import inf
from typing import Sequence


Vector = Sequence[float]
SequenceEmbedding = Sequence[Vector]


def _squared_distance(left: Vector, right: Vector) -> float:
    """计算两个等长向量的平方距离, 作为 DTW 局部匹配代价。"""
    if len(left) != len(right):
        raise ValueError("DTW 输入向量维度必须一致")
    return float(sum((a - b) ** 2 for a, b in zip(left, right)))


def compute_dtw_alignment_cost(reference_sequence: SequenceEmbedding, observed_sequence: SequenceEmbedding) -> float:
    """计算 reference 与 observed 序列之间的显式 DTW 对齐代价。

    该函数属于通用工程写法, 可复用于任意短序列 embedding 的时间对齐对照。
    在 SSTW 中, 它表示“先恢复或近似恢复时间路径, 再进行检测”的显式同步路线。
    """
    if not reference_sequence or not observed_sequence:
        raise ValueError("DTW 输入序列不能为空")

    row_count = len(reference_sequence)
    col_count = len(observed_sequence)
    dp = [[inf] * (col_count + 1) for _ in range(row_count + 1)]
    dp[0][0] = 0.0

    for row_index in range(1, row_count + 1):
        for col_index in range(1, col_count + 1):
            local_cost = _squared_distance(reference_sequence[row_index - 1], observed_sequence[col_index - 1])
            dp[row_index][col_index] = local_cost + min(
                dp[row_index - 1][col_index],
                dp[row_index][col_index - 1],
                dp[row_index - 1][col_index - 1],
            )

    return dp[row_count][col_count] / float(row_count + col_count)


def adapter_status() -> dict:
    """返回本地显式同步 baseline 的可运行状态, 但不自动支撑正向 claim。"""
    return {
        "external_baseline_runnable_status": "runnable",
        "external_baseline_not_run_reason": "none",
        "external_baseline_result_used_for_claim": False,
    }
