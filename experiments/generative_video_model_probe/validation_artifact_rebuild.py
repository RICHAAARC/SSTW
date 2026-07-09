"""paper profile artifact rebuild dry-run 检查。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from main.protocol.flow_evidence_fields import with_flow_evidence_protocol_defaults
from main.protocol.record_writer import write_json, write_jsonl
from main.protocol.table_builder import write_csv
from experiments.generative_video_model_probe.paper_result_artifact_builders import (
    PAPER_RESULT_ARTIFACT_RELPATHS,
)


REQUIRED_REBUILD_INPUTS = (
    "records/generation_records.jsonl",
    "records/trajectory_trace.jsonl",
    "records/runtime_attack_records.jsonl",
    "records/runtime_detection_records.jsonl",
    "records/motion_consistency_exclusion_records.jsonl",
    "records/sstw_measured_formal_records.jsonl",
    "records/external_baseline_records.jsonl",
    "records/external_baseline_score_records.jsonl",
    "records/fair_detection_calibration_records.jsonl",
    "records/formal_method_baseline_comparison_records.jsonl",
    "records/formal_baseline_difference_interval_records.jsonl",
    "records/formal_internal_ablation_summary_records.jsonl",
    "records/validation_internal_ablation_records.jsonl",
    "records/adaptive_attack_records.jsonl",
    "records/trajectory_sketch_verification_records.jsonl",
    "records/replay_uncertainty_records.jsonl",
    "records/wrong_sampler_replay_records.jsonl",
    "records/wrong_prompt_replay_records.jsonl",
    "records/claim3_downgrade_records.jsonl",
    "records/statistical_confidence_interval_records.jsonl",
    "records/low_fpr_formal_statistics_records.jsonl",
    "artifacts/generative_video_colab_runtime_decision.json",
    "artifacts/runtime_attack_decision.json",
    "artifacts/runtime_detection_decision.json",
    "artifacts/motion_consistency_exclusion_decision.json",
    "artifacts/sstw_measured_formal_decision.json",
    "artifacts/external_baseline_status_decision.json",
    "artifacts/external_baseline_comparison_decision.json",
    "artifacts/fair_detection_calibration_decision.json",
    "artifacts/formal_method_baseline_comparison_decision.json",
    "artifacts/formal_baseline_difference_interval_decision.json",
    "artifacts/formal_internal_ablation_summary_decision.json",
    "artifacts/validation_internal_ablation_decision.json",
    "artifacts/adaptive_attack_decision.json",
    "artifacts/replay_and_sketch_gate_decision.json",
    "artifacts/claim3_downgrade_decision.json",
    "artifacts/statistical_confidence_interval_decision.json",
    "artifacts/low_fpr_formal_statistics_decision.json",
)

REQUIRED_REBUILD_OUTPUTS = (
    "tables/generation_runtime_table.csv",
    "tables/external_baseline_status_table.csv",
    "tables/external_baseline_comparison_table.csv",
    "tables/runtime_attack_table.csv",
    "tables/runtime_detection_table.csv",
    "tables/motion_consistency_exclusion_table.csv",
    "tables/sstw_measured_formal_table.csv",
    "tables/fair_detection_calibration_table.csv",
    "tables/validation_internal_ablation_table.csv",
    "tables/formal_method_baseline_comparison_table.csv",
    "tables/formal_baseline_difference_interval_table.csv",
    "tables/formal_internal_ablation_summary_table.csv",
    "tables/adaptive_attack_table.csv",
    "tables/replay_verification_table.csv",
    "tables/claim3_downgrade_table.csv",
    "tables/statistical_confidence_interval_table.csv",
    "tables/low_fpr_formal_statistics_table.csv",
    "reports/external_baseline_comparison_report.md",
    "reports/motion_consistency_exclusion_report.md",
    "reports/sstw_measured_formal_report.md",
    "reports/fair_detection_calibration_report.md",
    "reports/formal_method_baseline_comparison_report.md",
    "reports/formal_baseline_difference_interval_report.md",
    "reports/formal_internal_ablation_summary_report.md",
    "reports/validation_internal_ablation_report.md",
    "reports/adaptive_attack_report.md",
    "reports/replay_and_sketch_gate_report.md",
    "reports/claim3_downgrade_report.md",
    "reports/statistical_confidence_interval_report.md",
    "reports/low_fpr_formal_statistics_report.md",
    *PAPER_RESULT_ARTIFACT_RELPATHS,
)


def _file_status(run_root: Path, relative_paths: tuple[str, ...]) -> list[dict]:
    """检查一组相对路径是否存在且非空。"""
    rows: list[dict] = []
    for relative_path in relative_paths:
        path = run_root / relative_path
        rows.append({
            "artifact_relative_path": relative_path,
            "artifact_exists": path.exists(),
            "artifact_size_bytes": path.stat().st_size if path.exists() else 0,
            "artifact_status": "ready" if path.exists() and path.stat().st_size > 0 else "missing_or_empty",
        })
    return rows


def build_validation_artifact_rebuild_dry_run_records(run_root: str | Path) -> list[dict]:
    """构建 paper profile artifact rebuild dry-run 检查 records。"""
    run_root = Path(run_root)
    records: list[dict] = []
    for artifact_role, rows in (
        ("required_input", _file_status(run_root, REQUIRED_REBUILD_INPUTS)),
        ("required_output", _file_status(run_root, REQUIRED_REBUILD_OUTPUTS)),
    ):
        for row in rows:
            records.append(with_flow_evidence_protocol_defaults({
                "record_version": "validation_artifact_rebuild_dry_run_v1",
                "artifact_role": artifact_role,
                "artifact_rebuild_check_scope": "paper_profile_generative_probe",
                "claim_support_status": "validation_artifact_rebuild_dry_run_only",
                **row,
            }, trajectory_source_level="not_applicable", claim_support_status="validation_artifact_rebuild_dry_run_only"))
    return records


def audit_validation_artifact_rebuild_dry_run(records: list[dict]) -> dict[str, Any]:
    """审计 artifact rebuild dry-run 是否通过。"""
    missing = [record["artifact_relative_path"] for record in records if record.get("artifact_status") != "ready"]
    decision = "PASS" if records and not missing else "FAIL"
    return {
        "stage_id": "validation_artifact_rebuild_dry_run",
        "validation_artifact_rebuild_dry_run_decision": decision,
        "claim_support_status": "validation_artifact_rebuild_dry_run_only" if decision == "PASS" else "validation_artifact_rebuild_blocked",
        "artifact_rebuild_check_record_count": len(records),
        "artifact_rebuild_missing_count": len(missing),
        "artifact_rebuild_missing_paths": missing,
        "artifact_rebuild_scope": "paper_profile_generative_probe",
    }


def run_validation_artifact_rebuild_dry_run(run_root: str | Path) -> dict[str, Any]:
    """写出 paper profile artifact rebuild dry-run records、table、decision 和 report。"""
    run_root = Path(run_root)
    records = build_validation_artifact_rebuild_dry_run_records(run_root)
    audit = audit_validation_artifact_rebuild_dry_run(records)
    write_jsonl(run_root / "records" / "validation_artifact_rebuild_dry_run_records.jsonl", records)
    write_csv(run_root / "tables" / "validation_artifact_rebuild_dry_run_table.csv", records)
    write_json(run_root / "artifacts" / "validation_artifact_rebuild_dry_run_decision.json", audit)
    report = (
        "# Validation Artifact Rebuild Dry-run Report\n\n"
        "该报告只检查 paper profile 产物是否具备 records -> tables / reports 的重建入口, "
        "不生成 full-paper 主表。\n\n"
        f"- validation_artifact_rebuild_dry_run_decision: {audit['validation_artifact_rebuild_dry_run_decision']}\n"
        f"- artifact_rebuild_check_record_count: {audit['artifact_rebuild_check_record_count']}\n"
        f"- artifact_rebuild_missing_count: {audit['artifact_rebuild_missing_count']}\n"
        f"- claim_support_status: {audit['claim_support_status']}\n"
    )
    report_path = run_root / "reports" / "validation_artifact_rebuild_dry_run_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="执行 paper profile artifact rebuild dry-run 检查。")
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    payload = run_validation_artifact_rebuild_dry_run(args.run_root)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
