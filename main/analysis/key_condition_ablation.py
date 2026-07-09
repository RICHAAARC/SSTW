"""分析 state_space_inference_formalization key condition 消融。"""

from __future__ import annotations


def key_condition_ablation_gain(ablation_records: list[dict]) -> float:
    """返回 key-conditioned 相对 without-key 的 TPR 增益。"""
    gains = [float(record["ablation_observed_delta_tpr"]) for record in ablation_records if record.get("ablation_family") == "key_condition"]
    return max(gains) if gains else 0.0
