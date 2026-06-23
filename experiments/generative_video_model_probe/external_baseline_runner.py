"""构建 B5 外部 baseline 状态记录与 comparison 产物。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

from external_baseline.registry import get_adapter
from external_baseline.runtime_trace_io import comparable_detection_records, read_jsonl, safe_float
from main.core.digest import build_stable_digest
from main.external_baselines.baseline_registry import audit_external_baseline_records, build_external_baseline_records
from main.protocol.flow_evidence_fields import with_flow_evidence_protocol_defaults, with_flow_evidence_protocol_defaults_many
from main.protocol.record_writer import write_json, write_jsonl
from main.protocol.table_builder import write_csv


DEFAULT_EXTERNAL_BASELINE_CONFIG = "configs/external_baselines/external_baselines.json"
EXTERNAL_BASELINE_SCORE_RECORDS = "records/external_baseline_score_records.jsonl"
EXTERNAL_BASELINE_COMPARISON_TABLE = "tables/external_baseline_comparison_table.csv"
EXTERNAL_BASELINE_COMPARISON_DECISION = "artifacts/external_baseline_comparison_decision.json"
EXTERNAL_BASELINE_COMPARISON_REPORT = "reports/external_baseline_comparison_report.md"


def run_external_baseline_status(config_path: str = DEFAULT_EXTERNAL_BASELINE_CONFIG) -> list[dict[str, Any]]:
    """返回外部 baseline governed 状态 records。

    该函数只记录外部 baseline 是否具备可运行 adapter、协议兼容边界和 claim 边界。它不会把 unavailable
    modern baseline 伪装成正式比较结果。真正的 baseline comparison 由
    `write_external_baseline_comparison_outputs` 读取 runtime detection records 后生成。
    """
    return with_flow_evidence_protocol_defaults_many(
        build_external_baseline_records(config_path),
        trajectory_source_level="not_applicable",
        claim_support_status="external_baseline_status_record_only",
    )


def build_external_baseline_status_audit(config_path: str = DEFAULT_EXTERNAL_BASELINE_CONFIG) -> dict[str, Any]:
    """构建外部 baseline 状态审计摘要。"""
    return audit_external_baseline_records(run_external_baseline_status(config_path))


def write_external_baseline_status_outputs(
    run_root: str | Path,
    config_path: str = DEFAULT_EXTERNAL_BASELINE_CONFIG,
) -> dict[str, Any]:
    """写出外部 baseline 状态 records、table、decision 和 report。"""
    run_root = Path(run_root)
    records = run_external_baseline_status(config_path)
    audit = audit_external_baseline_records(records)
    write_jsonl(run_root / "records" / "external_baseline_records.jsonl", records)
    write_csv(run_root / "tables" / "external_baseline_status_table.csv", records)
    write_json(run_root / "artifacts" / "external_baseline_status_decision.json", audit)
    report = (
        "# External Baseline Status Report\n\n"
        "该报告由 external baseline 配置和 adapter 状态自动生成。状态记录用于治理外部 baseline 接入边界, "
        "不能把 non-run modern baseline 写成 SSTW 已经优于该 baseline。\n\n"
        f"- external_baseline_status_decision: {audit['external_baseline_status_decision']}\n"
        f"- external_baseline_record_count: {audit['external_baseline_record_count']}\n"
        f"- modern_external_baseline_record_count: {audit['modern_external_baseline_record_count']}\n"
        f"- modern_external_baseline_main_comparison_ready_count: {audit['modern_external_baseline_main_comparison_ready_count']}\n"
        f"- external_baseline_claim_support_status: {audit['external_baseline_claim_support_status']}\n"
    )
    report_path = run_root / "reports" / "external_baseline_status_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return audit


def _non_run_score_record(baseline_record: Mapping[str, Any], comparable_count: int) -> dict[str, Any]:
    """为尚未接入 adapter 的外部 baseline 写出 comparison unsupported row。

    该记录属于治理性结果, 作用是让 comparison table 显式显示缺口。它不等价于 baseline 实验失败,
    也不能支持 SSTW 优于该 baseline 的 claim。
    """
    payload = {
        "external_baseline_name": baseline_record.get("external_baseline_name"),
        "external_baseline_family": baseline_record.get("external_baseline_family"),
        "external_baseline_layer": baseline_record.get("external_baseline_layer"),
        "external_baseline_runnable_status": baseline_record.get("external_baseline_runnable_status"),
        "external_baseline_not_run_reason": baseline_record.get("external_baseline_not_run_reason"),
        "comparison_unit_count": comparable_count,
    }
    digest = build_stable_digest(payload)
    return with_flow_evidence_protocol_defaults({
        "record_version": "external_baseline_score_v1",
        "external_baseline_score_record_id": f"external_baseline_score_{digest[:16]}",
        "external_baseline_adapter_path": baseline_record.get("external_baseline_adapter_path", "not_integrated"),
        "metric_status": "unsupported",
        "external_baseline_score_status": "adapter_not_integrated",
        "external_baseline_score_source": "governed_non_run_record",
        "external_baseline_score_failure_reason": baseline_record.get("external_baseline_not_run_reason") or "adapter_not_integrated",
        "external_baseline_reference_sequence_length": 0,
        "external_baseline_observed_sequence_length": 0,
        "external_baseline_distance": None,
        "external_baseline_score": None,
        "baseline_score_margin": None,
        "external_baseline_result_used_for_claim": False,
        "claim_support_status": "external_baseline_governed_non_run_comparison_row",
        **payload,
    }, trajectory_source_level="not_applicable", claim_support_status="external_baseline_governed_non_run_comparison_row")


def build_external_baseline_comparison_records(
    run_root: str | Path,
    config_path: str = DEFAULT_EXTERNAL_BASELINE_CONFIG,
) -> list[dict[str, Any]]:
    """调用 `external_baseline/` 下的 adapter 生成 baseline comparison score records。

    该函数是项目特定调度层。它不在 `main/` 中执行第三方方法, 而是读取已落盘的 runtime detection 与
    trajectory records, 再调用 `external_baseline/primary/.../adapter` 下的适配器。这样可以保证 Notebook、
    Colab 冷启动和本地后处理都通过同一条受治理链路产出 baseline comparison 结果。
    """
    run_root = Path(run_root)
    baseline_records = run_external_baseline_status(config_path)
    comparable_records = comparable_detection_records(run_root)
    records: list[dict[str, Any]] = []
    for baseline_record in baseline_records:
        baseline_name = str(baseline_record.get("external_baseline_name") or "")
        adapter = get_adapter(baseline_name)
        if adapter is None or baseline_record.get("external_baseline_runnable_status") != "runnable":
            records.append(_non_run_score_record(baseline_record, len(comparable_records)))
            continue
        adapter_records = adapter.build_score_records(run_root, baseline_record)
        if adapter_records:
            records.extend(adapter_records)
        else:
            records.append(_non_run_score_record({**baseline_record, "external_baseline_not_run_reason": "no_comparable_runtime_detection_records"}, len(comparable_records)))
    return records


def _mean_numeric(rows: list[Mapping[str, Any]], field: str) -> float | None:
    """聚合数值字段均值, 不可用时返回 None。"""
    values = [safe_float(row.get(field)) for row in rows if row.get(field) not in {None, "", "unsupported"}]
    if not values:
        return None
    return round(mean(values), 6)


def _proposed_method_row(runtime_detection_records: list[dict[str, Any]]) -> dict[str, Any]:
    """构造 SSTW 当前方法在同一 runtime detection 协议下的比较表行。"""
    ready_records = [record for record in runtime_detection_records if record.get("runtime_detection_status") == "ready"]
    attack_names = {str(record.get("attack_name")) for record in ready_records if record.get("attack_name")}
    return {
        "method_id": "sstw_key_conditioned_flow_trajectory",
        "method_role": "proposed_method_runtime_proxy",
        "comparison_scope": "runtime_detection_common_protocol",
        "metric_status": "measured_proxy" if ready_records else "unsupported",
        "external_baseline_name": "not_applicable",
        "external_baseline_family": "not_applicable",
        "external_baseline_layer": "proposed_method",
        "external_baseline_score_status": "not_applicable",
        "external_baseline_result_used_for_claim": False,
        "comparison_record_count": len(ready_records),
        "comparison_attack_count": len(attack_names),
        "proposed_method_score_mean": _mean_numeric(ready_records, "S_runtime_attack_detection"),
        "external_baseline_score_mean": None,
        "external_baseline_distance_mean": None,
        "baseline_score_margin_mean": None,
        "claim_support_status": "runtime_proxy_comparison_not_claim_supporting",
    }


def build_external_baseline_comparison_table_rows(
    run_root: str | Path,
    comparison_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """由 comparison records 重建外部 baseline 对比表。"""
    runtime_detection_records = read_jsonl(Path(run_root) / "records" / "runtime_detection_records.jsonl")
    rows = [_proposed_method_row(runtime_detection_records)]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in comparison_records:
        grouped.setdefault(str(record.get("external_baseline_name") or "unknown_external_baseline"), []).append(record)
    for baseline_name, group in sorted(grouped.items()):
        first = group[0]
        measured_rows = [record for record in group if record.get("metric_status") == "measured_proxy"]
        attack_names = {str(record.get("attack_name")) for record in measured_rows if record.get("attack_name")}
        rows.append({
            "method_id": baseline_name,
            "method_role": f"external_baseline_{first.get('external_baseline_layer')}",
            "comparison_scope": "external_baseline_adapter_proxy" if measured_rows else "external_baseline_result_missing",
            "metric_status": "measured_proxy" if measured_rows else "unsupported",
            "external_baseline_name": baseline_name,
            "external_baseline_family": first.get("external_baseline_family"),
            "external_baseline_layer": first.get("external_baseline_layer"),
            "external_baseline_score_status": "measured_proxy" if measured_rows else first.get("external_baseline_score_status"),
            "external_baseline_result_used_for_claim": False,
            "comparison_record_count": len(group),
            "comparison_attack_count": len(attack_names),
            "proposed_method_score_mean": None,
            "external_baseline_score_mean": _mean_numeric(measured_rows, "external_baseline_score"),
            "external_baseline_distance_mean": _mean_numeric(measured_rows, "external_baseline_distance"),
            "baseline_score_margin_mean": _mean_numeric(measured_rows, "baseline_score_margin"),
            "claim_support_status": first.get("claim_support_status"),
        })
    return rows


def audit_external_baseline_comparison_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """审计 external baseline comparison 是否已经由 adapter 产出。"""
    measured_records = [record for record in records if record.get("metric_status") == "measured_proxy"]
    measured_adapter_names = {str(record.get("external_baseline_name")) for record in measured_records if record.get("external_baseline_name")}
    unsupported_records = [record for record in records if record.get("metric_status") == "unsupported"]
    decision = "PASS" if records and measured_adapter_names else "FAIL"
    return {
        "stage_id": "external_baseline_comparison_audit",
        "external_baseline_comparison_decision": decision,
        "external_baseline_comparison_record_count": len(records),
        "external_baseline_comparison_ready_count": len(measured_records),
        "external_baseline_measured_adapter_count": len(measured_adapter_names),
        "external_baseline_measured_adapter_names": sorted(measured_adapter_names),
        "external_baseline_unsupported_adapter_count": len(unsupported_records),
        "external_baseline_comparison_status": "adapter_proxy_records_written" if decision == "PASS" else "comparison_records_missing",
        "external_baseline_claim_support_status": "external_baseline_proxy_comparison_not_claim_supporting" if decision == "PASS" else "external_baseline_comparison_blocked",
    }


def write_external_baseline_comparison_outputs(
    run_root: str | Path,
    config_path: str = DEFAULT_EXTERNAL_BASELINE_CONFIG,
) -> dict[str, Any]:
    """写出外部 baseline comparison records、table、decision 和 report。"""
    run_root = Path(run_root)
    records = build_external_baseline_comparison_records(run_root, config_path)
    audit = audit_external_baseline_comparison_records(records)
    table_rows = build_external_baseline_comparison_table_rows(run_root, records)
    audit["external_baseline_comparison_table_status"] = "ready" if table_rows else "missing"
    write_jsonl(run_root / EXTERNAL_BASELINE_SCORE_RECORDS, records)
    write_csv(run_root / EXTERNAL_BASELINE_COMPARISON_TABLE, table_rows)
    write_json(run_root / EXTERNAL_BASELINE_COMPARISON_DECISION, audit)
    report = (
        "# External Baseline Comparison Report\n\n"
        "该报告由 `external_baseline/` adapter 产出, 用于证明本项目已经具备 baseline 对比结果落盘链路。"
        "当前显式同步 baseline 仍属于 proxy control, modern video watermark baseline 在 adapter 未接入前保持 unsupported, "
        "因此本报告不支持正向论文主 claim。\n\n"
        f"- external_baseline_comparison_decision: {audit['external_baseline_comparison_decision']}\n"
        f"- external_baseline_comparison_record_count: {audit['external_baseline_comparison_record_count']}\n"
        f"- external_baseline_comparison_ready_count: {audit['external_baseline_comparison_ready_count']}\n"
        f"- external_baseline_measured_adapter_count: {audit['external_baseline_measured_adapter_count']}\n"
        f"- external_baseline_measured_adapter_names: {', '.join(audit['external_baseline_measured_adapter_names']) or 'none'}\n"
        f"- external_baseline_claim_support_status: {audit['external_baseline_claim_support_status']}\n"
    )
    report_path = run_root / EXTERNAL_BASELINE_COMPARISON_REPORT
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="写出外部 baseline governed 状态和 comparison 结果。")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--config-path", default=DEFAULT_EXTERNAL_BASELINE_CONFIG)
    parser.add_argument("--mode", choices=("status", "comparison", "all"), default="all")
    args = parser.parse_args()
    payload: dict[str, Any] = {}
    if args.mode in {"status", "all"}:
        payload["status"] = write_external_baseline_status_outputs(args.run_root, args.config_path)
    if args.mode in {"comparison", "all"}:
        payload["comparison"] = write_external_baseline_comparison_outputs(args.run_root, args.config_path)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
