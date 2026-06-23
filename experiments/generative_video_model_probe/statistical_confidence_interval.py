"""validation-scale 统计置信区间报告。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from main.protocol.flow_evidence_fields import with_flow_evidence_protocol_defaults
from main.protocol.record_writer import write_json, write_jsonl
from main.protocol.table_builder import write_csv


def _read_jsonl(path: Path) -> list[dict]:
    """读取 JSONL records, 文件不存在时返回空列表。"""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _wilson_interval(success_count: int, total_count: int, z_value: float = 1.96) -> tuple[float | None, float | None]:
    """计算二项比例 Wilson 区间。

    该函数属于通用统计写法。它只基于当前 records 的计数, 不访问阈值或最终论文 claim。
    """
    if total_count <= 0:
        return None, None
    phat = success_count / total_count
    denominator = 1.0 + z_value * z_value / total_count
    center = (phat + z_value * z_value / (2.0 * total_count)) / denominator
    spread = z_value * math.sqrt((phat * (1.0 - phat) + z_value * z_value / (4.0 * total_count)) / total_count) / denominator
    return round(max(0.0, center - spread), 6), round(min(1.0, center + spread), 6)


def build_statistical_confidence_interval_records(run_root: str | Path) -> list[dict]:
    """从 runtime detection records 构建 validation-scale 置信区间 records。"""
    run_root = Path(run_root)
    detection_records = [
        record for record in _read_jsonl(run_root / "records" / "runtime_detection_records.jsonl")
        if record.get("runtime_detection_status") == "ready"
    ]
    total_count = len(detection_records)
    detectable_count = sum(1 for record in detection_records if record.get("attacked_video_detectable") is True)
    lower, upper = _wilson_interval(detectable_count, total_count)
    rate = round(detectable_count / total_count, 6) if total_count else None
    record = with_flow_evidence_protocol_defaults({
        "record_version": "validation_statistical_confidence_interval_v1",
        "statistical_confidence_interval_family": "runtime_detection_detectable_rate",
        "ci_success_count": detectable_count,
        "ci_total_count": total_count,
        "ci_point_estimate": rate,
        "ci_wilson_lower": lower,
        "ci_wilson_upper": upper,
        "ci_confidence_level": 0.95,
        "ci_evidence_level": "validation_runtime_detection_proxy",
        "cluster_by_video_interval_status": "not_available_until_full_paper_unique_video_bootstrap",
        "paper_low_fpr_ci_status": "not_available_until_full_paper_negative_split",
        "claim_support_status": "validation_ci_proxy_only",
    }, trajectory_source_level="validation_runtime_detection_ci", claim_support_status="validation_ci_proxy_only")
    return [record]


def audit_statistical_confidence_interval_records(records: list[dict]) -> dict[str, Any]:
    """审计 validation-scale CI records 是否可用于后续 gate。"""
    record = records[0] if records else {}
    total_count = int(record.get("ci_total_count") or 0)
    lower = record.get("ci_wilson_lower")
    upper = record.get("ci_wilson_upper")
    decision = "PASS" if total_count > 0 and lower is not None and upper is not None else "FAIL"
    return {
        "stage_id": "statistical_confidence_interval_reporter",
        "statistical_confidence_interval_decision": decision,
        "claim_support_status": "validation_ci_proxy_only" if decision == "PASS" else "validation_ci_blocked",
        "ci_record_count": len(records),
        "ci_total_count": total_count,
        "ci_success_count": record.get("ci_success_count"),
        "ci_point_estimate": record.get("ci_point_estimate"),
        "ci_wilson_lower": lower,
        "ci_wilson_upper": upper,
        "paper_low_fpr_ci_status": record.get("paper_low_fpr_ci_status", "missing"),
        "cluster_by_video_interval_status": record.get("cluster_by_video_interval_status", "missing"),
    }


def run_statistical_confidence_interval_reporter(run_root: str | Path) -> dict[str, Any]:
    """写出 validation-scale CI records、table、decision 和 report。"""
    run_root = Path(run_root)
    records = build_statistical_confidence_interval_records(run_root)
    audit = audit_statistical_confidence_interval_records(records)
    write_jsonl(run_root / "records" / "statistical_confidence_interval_records.jsonl", records)
    write_csv(run_root / "tables" / "statistical_confidence_interval_table.csv", records)
    write_json(run_root / "artifacts" / "statistical_confidence_interval_decision.json", audit)
    report = (
        "# Statistical Confidence Interval Report\n\n"
        "该报告只为 validation-scale runtime detection proxy 计算轻量 Wilson 区间。"
        "它不是 full-paper FPR=0.001 统计报告。\n\n"
        f"- statistical_confidence_interval_decision: {audit['statistical_confidence_interval_decision']}\n"
        f"- ci_total_count: {audit['ci_total_count']}\n"
        f"- ci_point_estimate: {audit['ci_point_estimate']}\n"
        f"- ci_wilson_lower: {audit['ci_wilson_lower']}\n"
        f"- ci_wilson_upper: {audit['ci_wilson_upper']}\n"
        f"- claim_support_status: {audit['claim_support_status']}\n"
    )
    report_path = run_root / "reports" / "statistical_confidence_interval_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 validation-scale 统计置信区间报告。")
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    payload = run_statistical_confidence_interval_reporter(args.run_root)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
