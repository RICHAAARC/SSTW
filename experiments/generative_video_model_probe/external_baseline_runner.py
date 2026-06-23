"""构建 B5 外部 baseline 状态记录与审计产物。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from main.external_baselines.baseline_registry import audit_external_baseline_records, build_external_baseline_records
from main.protocol.flow_evidence_fields import with_flow_evidence_protocol_defaults_many
from main.protocol.record_writer import write_json, write_jsonl
from main.protocol.table_builder import write_csv


DEFAULT_EXTERNAL_BASELINE_CONFIG = "configs/external_baselines/external_baselines.json"


def run_external_baseline_status(config_path: str = DEFAULT_EXTERNAL_BASELINE_CONFIG) -> list[dict]:
    """返回外部 baseline governed 状态 records。

    该函数只记录外部 baseline 是否具备可运行和协议兼容条件, 不把 unavailable baseline
    伪装成正式比较结果。现代 baseline 即使尚未接入 adapter, 也必须写出 non-run record,
    便于后续 full-paper gate 明确阻断原因。
    """
    return with_flow_evidence_protocol_defaults_many(
        build_external_baseline_records(config_path),
        trajectory_source_level="not_applicable",
        claim_support_status="external_baseline_status_record_only",
    )


def build_external_baseline_status_audit(config_path: str = DEFAULT_EXTERNAL_BASELINE_CONFIG) -> dict:
    """构建外部 baseline 状态审计摘要。"""
    return audit_external_baseline_records(run_external_baseline_status(config_path))


def write_external_baseline_status_outputs(
    run_root: str | Path,
    config_path: str = DEFAULT_EXTERNAL_BASELINE_CONFIG,
) -> dict:
    """写出外部 baseline 状态 records、table、decision 和 report。

    该函数属于阶段性工程推进能力。它不会运行现代 baseline 的检测器, 只负责把当前
    可运行状态、non-run reason 和 protocol gap 落盘, 防止后续 full-paper 主表静默缺少
    现代外部 baseline。
    """
    run_root = Path(run_root)
    records = run_external_baseline_status(config_path)
    audit = audit_external_baseline_records(records)
    write_jsonl(run_root / "records" / "external_baseline_records.jsonl", records)
    write_csv(run_root / "tables" / "external_baseline_status_table.csv", records)
    write_json(run_root / "artifacts" / "external_baseline_status_decision.json", audit)
    report = (
        "# External Baseline Status Report\n\n"
        "该报告由 external baseline 配置和 adapter 状态自动生成。当前阶段只记录 governed 状态, "
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


def main() -> None:
    parser = argparse.ArgumentParser(description="写出外部 baseline governed 状态记录。")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--config-path", default=DEFAULT_EXTERNAL_BASELINE_CONFIG)
    args = parser.parse_args()
    payload = write_external_baseline_status_outputs(args.run_root, args.config_path)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
