"""paper profile 统计置信区间报告。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from main.protocol.flow_evidence_fields import with_flow_evidence_protocol_defaults
from main.protocol.record_writer import write_json, write_jsonl
from main.protocol.table_builder import write_csv


DEFAULT_PROTOCOL_CONFIG = "configs/protocol/probe_paper_generative_probe.json"


def _read_json(path: Path) -> dict[str, Any]:
    """读取 JSON config 或 artifact, 并兼容 UTF-8 BOM。"""
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON 顶层必须是对象: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict]:
    """读取 JSONL records, 文件不存在时返回空列表。"""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_profile_context(config_path: str | Path) -> dict[str, Any]:
    """从 protocol config 读取当前 CI 报告所属的 profile 语义。"""
    config = _read_json(Path(config_path))
    if "target_fpr" not in config:
        raise KeyError(f"protocol config 缺少 target_fpr: {config_path}")
    return {
        "target_fpr": float(config["target_fpr"]),
        "paper_result_level": config.get("paper_result_level", "probe_paper"),
        "target_fpr_source_config_path": str(config_path),
    }


def _format_fpr(value: float | None) -> str:
    """把 FPR 数值格式化为报告中的稳定短文本。"""
    if value is None:
        return "未配置"
    return f"{float(value):g}"


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


def build_statistical_confidence_interval_records(
    run_root: str | Path,
    config_path: str | Path = DEFAULT_PROTOCOL_CONFIG,
) -> list[dict]:
    """从公平校准 records 构建 paper profile 置信区间 records。"""
    run_root = Path(run_root)
    profile_context = _load_profile_context(config_path)
    fair_records = [
        record for record in _read_jsonl(run_root / "records" / "fair_detection_calibration_records.jsonl")
        if record.get("fair_comparison_status") == "ready"
        and record.get("metric_status") == "measured_formal"
    ]
    rows: list[dict] = []
    for source in fair_records:
        total_count = int(source.get("attacked_positive_score_count") or 0)
        success_count = int(source.get("detected_positive_count_at_target_fpr") or 0)
        lower, upper = _wilson_interval(success_count, total_count)
        rate = round(success_count / total_count, 6) if total_count else None
        claim_status = "formal_ci_from_fair_detection_calibration" if total_count and lower is not None and upper is not None else "formal_ci_blocked"
        rows.append(with_flow_evidence_protocol_defaults({
            "record_version": "formal_statistical_confidence_interval_v1",
            "statistical_confidence_interval_family": "tpr_at_target_fpr",
            "method_id": source.get("method_id"),
            "method_role": source.get("method_role"),
            **profile_context,
            "ci_success_count": success_count,
            "ci_total_count": total_count,
            "ci_point_estimate": rate,
            "ci_wilson_lower": lower,
            "ci_wilson_upper": upper,
            "ci_confidence_level": 0.95,
            "ci_evidence_level": "fair_detection_calibration_measured_formal",
            "cluster_by_video_interval_status": "available_from_prompt_seed_attack_anchor_bootstrap",
            "paper_low_fpr_ci_status": "formal_target_fpr_ci_ready",
            "claim_support_status": claim_status,
        }, trajectory_source_level="fair_detection_calibration_records", claim_support_status=claim_status))
    return rows


def audit_statistical_confidence_interval_records(records: list[dict]) -> dict[str, Any]:
    """审计 paper profile CI records 是否可用于后续 gate。"""
    ready_records = [
        record
        for record in records
        if int(record.get("ci_total_count") or 0) > 0
        and record.get("ci_wilson_lower") is not None
        and record.get("ci_wilson_upper") is not None
        and record.get("ci_evidence_level") == "fair_detection_calibration_measured_formal"
    ]
    record = ready_records[0] if ready_records else (records[0] if records else {})
    total_count = sum(int(row.get("ci_total_count") or 0) for row in ready_records)
    success_count = sum(int(row.get("ci_success_count") or 0) for row in ready_records)
    decision = "PASS" if ready_records and len(ready_records) == len(records) else "FAIL"
    return {
        "stage_id": "statistical_confidence_interval_reporter",
        "statistical_confidence_interval_decision": decision,
        "claim_support_status": "formal_ci_from_fair_detection_calibration" if decision == "PASS" else "formal_ci_blocked",
        "ci_record_count": len(records),
        "paper_result_level": record.get("paper_result_level"),
        "target_fpr": record.get("target_fpr"),
        "target_fpr_source_config_path": record.get("target_fpr_source_config_path"),
        "ci_total_count": total_count,
        "ci_success_count": success_count,
        "ci_point_estimate": round(success_count / total_count, 6) if total_count else None,
        "ci_wilson_lower": record.get("ci_wilson_lower"),
        "ci_wilson_upper": record.get("ci_wilson_upper"),
        "paper_low_fpr_ci_status": record.get("paper_low_fpr_ci_status", "missing"),
        "cluster_by_video_interval_status": record.get("cluster_by_video_interval_status", "missing"),
    }


def run_statistical_confidence_interval_reporter(
    run_root: str | Path,
    config_path: str | Path = DEFAULT_PROTOCOL_CONFIG,
) -> dict[str, Any]:
    """写出 paper profile CI records、table、decision 和 report。"""
    run_root = Path(run_root)
    records = build_statistical_confidence_interval_records(run_root, config_path)
    audit = audit_statistical_confidence_interval_records(records)
    write_jsonl(run_root / "records" / "statistical_confidence_interval_records.jsonl", records)
    write_csv(run_root / "tables" / "statistical_confidence_interval_table.csv", records)
    write_json(run_root / "artifacts" / "statistical_confidence_interval_decision.json", audit)
    target_fpr_text = _format_fpr(audit.get("target_fpr"))
    report = (
        "# Statistical Confidence Interval Report\n\n"
        "该报告基于 fair_detection_calibration_records 中的 measured_formal TPR@target FPR "
        "计算 Wilson 区间。"
        f"当前 profile 的 target_fpr={target_fpr_text}, 该数值来自 protocol config。"
        "本报告不会自动替代更低 FPR profile 的正式大规模统计报告。\n\n"
        f"- statistical_confidence_interval_decision: {audit['statistical_confidence_interval_decision']}\n"
        f"- paper_result_level: {audit['paper_result_level']}\n"
        f"- target_fpr: {target_fpr_text}\n"
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
    parser = argparse.ArgumentParser(description="生成 paper profile 统计置信区间报告。")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--config-path", default=DEFAULT_PROTOCOL_CONFIG)
    args = parser.parse_args()
    payload = run_statistical_confidence_interval_reporter(args.run_root, args.config_path)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
