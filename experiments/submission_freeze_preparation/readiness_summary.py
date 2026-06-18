"""根据 claim audit records 生成 submission readiness summary。"""

from __future__ import annotations

import json
from pathlib import Path

from main.protocol.record_writer import write_json
from main.protocol.table_builder import write_csv


def _read_json(path: Path) -> dict:
    """读取 JSON 文件, 文件不存在时返回空对象。"""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    """读取 JSONL records, 文件不存在时返回空列表。"""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _latest_package_manifest(package_dir: Path) -> Path:
    """返回最新 submission freeze package manifest。

    该函数属于通用工程写法。package 文件名包含 UTC 时间和短 commit 后,
    readiness summary 不能再依赖固定文件名, 必须按 manifest 修改时间选择最新批次。
    """
    candidates = sorted(
        package_dir.glob("submission_freeze_preparation_package_*_manifest.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    return package_dir / "submission_freeze_preparation_package_manifest.json"


def _claim_rows(claim_records: list[dict]) -> list[dict]:
    """把 claim audit records 转换为面向审稿准备的扁平 summary rows。"""
    rows: list[dict] = []
    for record in claim_records:
        claim_status = str(record.get("claim_status"))
        claim_scope = str(record.get("claim_scope"))
        if claim_scope == "main" and claim_status == "supported":
            readiness_bucket = "main_text_ready"
        elif claim_status == "needs_downgrade":
            readiness_bucket = "downgraded_to_exploratory"
        elif claim_scope == "exploratory" and claim_status == "supported":
            readiness_bucket = "exploratory_ready"
        else:
            readiness_bucket = "blocked"
        rows.append({
            "claim_id": record.get("claim_id"),
            "claim_scope": claim_scope,
            "claim_status": claim_status,
            "readiness_bucket": readiness_bucket,
            "downgrade_reason": record.get("downgrade_reason", "none"),
            "supporting_stage_ids": record.get("supporting_stage_ids", []),
            "supporting_artifact_paths": record.get("supporting_artifact_paths", []),
            "supported_by_governed_artifacts": record.get("supported_by_governed_artifacts") is True,
        })
    return rows


def build_submission_readiness_summary(run_root: str | Path) -> dict:
    """生成 submission readiness summary artifacts。

    该函数属于通用工程写法。它不新增 claim, 只从已有 claim audit records 归纳主文可用性、降级边界和仍需补齐的产物。
    """
    run_root = Path(run_root)
    claim_records = _read_jsonl(run_root / "records" / "claim_audit_records.jsonl")
    decision = _read_json(run_root / "artifacts" / "submission_freeze_preparation_decision.json")
    package_manifest = _read_json(_latest_package_manifest(run_root / "packages"))
    rows = _claim_rows(claim_records)

    main_text_ready_claim_count = sum(1 for row in rows if row["readiness_bucket"] == "main_text_ready")
    exploratory_ready_claim_count = sum(1 for row in rows if row["readiness_bucket"] == "exploratory_ready")
    downgraded_claim_count = sum(1 for row in rows if row["readiness_bucket"] == "downgraded_to_exploratory")
    blocked_claim_count = sum(1 for row in rows if row["readiness_bucket"] == "blocked")
    sstw_t_ready = decision.get("details", {}).get("sstw_t_submission_preparation_status") == "PASS"
    sstw_tc_status = decision.get("details", {}).get("sstw_tc_submission_freeze_status")
    package_ready = decision.get("details", {}).get("release_package_rebuildable") == "PASS" and bool(package_manifest.get("package_digest"))

    readiness_summary = {
        "stage_id": "submission_readiness_summary",
        "submission_readiness_decision": "PASS" if sstw_t_ready and package_ready and blocked_claim_count == 0 else "FAIL",
        "main_submission_variant": "SSTW-T" if sstw_t_ready else "blocked",
        "exploratory_variants": ["SSTW-TC"] if sstw_tc_status == "DOWNGRADED_TO_EXPLORATORY" else [],
        "main_text_ready_claim_count": main_text_ready_claim_count,
        "exploratory_ready_claim_count": exploratory_ready_claim_count,
        "downgraded_claim_count": downgraded_claim_count,
        "blocked_claim_count": blocked_claim_count,
        "package_ready": package_ready,
        "package_digest": package_manifest.get("package_digest"),
        "remaining_submission_tasks": [
            "build_camera_ready_figures_from_governed_records",
            "build_main_tables_from_governed_records",
            "write_failure_case_taxonomy_from_records",
            "prepare_appendix_for_sstw_tc_exploratory_probe",
        ],
        "claim_boundary_statement": "SSTW-T is the current main submission variant; SSTW-TC remains exploratory until final submission-freeze evidence is available.",
    }

    write_csv(run_root / "tables" / "submission_readiness_claim_table.csv", rows)
    write_json(run_root / "artifacts" / "submission_readiness_summary.json", readiness_summary)
    report_lines = [
        "# Submission Readiness Summary",
        "",
        "该报告由 `claim_audit_records.jsonl` 和 submission preparation decision 重建, 不手工新增 claim。",
        "",
        f"- submission_readiness_decision: {readiness_summary['submission_readiness_decision']}",
        f"- main_submission_variant: {readiness_summary['main_submission_variant']}",
        f"- exploratory_variants: {', '.join(readiness_summary['exploratory_variants']) if readiness_summary['exploratory_variants'] else 'none'}",
        f"- main_text_ready_claim_count: {main_text_ready_claim_count}",
        f"- exploratory_ready_claim_count: {exploratory_ready_claim_count}",
        f"- downgraded_claim_count: {downgraded_claim_count}",
        f"- blocked_claim_count: {blocked_claim_count}",
        f"- package_ready: {package_ready}",
        "",
        "## Claim boundary",
        "",
        readiness_summary["claim_boundary_statement"],
        "",
        "## Remaining submission tasks",
        "",
    ]
    report_lines.extend(f"- {item}" for item in readiness_summary["remaining_submission_tasks"])
    (run_root / "reports").mkdir(parents=True, exist_ok=True)
    (run_root / "reports" / "submission_readiness_summary_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return readiness_summary
