"""实现 fixed low-FPR threshold 校准。"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable


def calibrate_thresholds(records: Iterable[dict], target_fpr: float) -> dict[str, dict]:
    """仅使用 calibration negative records 计算每个方法的阈值。"""
    scores_by_method: dict[str, list[float]] = defaultdict(list)
    for record in records:
        if record["split"] == "calibration" and record["sample_role"].endswith("negative"):
            scores_by_method[record["method_variant"]].append(float(record["S_final"]))
    thresholds: dict[str, dict] = {}
    for method_variant, scores in scores_by_method.items():
        if not scores:
            raise ValueError(f"missing calibration negative scores for {method_variant}")
        thresholds[method_variant] = {"threshold_id": f"threshold_{method_variant}_calibration_negative", "method_variant": method_variant, "target_fpr": target_fpr, "threshold_source_split": "calibration", "threshold_value": round(max(scores) + 1e-6, 6), "calibration_negative_count": len(scores)}
    return thresholds


def apply_thresholds(records: Iterable[dict], thresholds: dict[str, dict]) -> list[dict]:
    """把固定阈值写入 records 并生成 positive / negative 判决。"""
    decided_records: list[dict] = []
    for record in records:
        threshold = thresholds[record["method_variant"]]
        enriched = dict(record)
        enriched["threshold_id"] = threshold["threshold_id"]
        enriched["threshold_source_split"] = threshold["threshold_source_split"]
        enriched["threshold_value"] = threshold["threshold_value"]
        enriched["decision"] = "positive" if enriched["S_final"] >= threshold["threshold_value"] else "negative"
        enriched["decision_reason"] = "fixed_calibration_threshold"
        enriched["test_time_threshold_update_blocked"] = True
        decided_records.append(enriched)
    return decided_records
