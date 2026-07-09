"""分析 state_space_inference_formalization 泛化记录。"""

from __future__ import annotations


def generalization_status(records: list[dict], axis: str) -> str:
    """判断指定泛化轴是否通过。"""
    axis_records = [record for record in records if record.get("generalization_axis") == axis]
    if not axis_records:
        return "FAIL"
    return "PASS" if all(float(record.get("generalization_delta_fpr", 1.0)) <= 0.0 for record in axis_records) else "FAIL"
