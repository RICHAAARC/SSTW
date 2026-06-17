"""从 governed records 重建 submission preparation 主表候选。"""

from __future__ import annotations

import json
from pathlib import Path

from main.protocol.record_writer import write_json
from main.protocol.table_builder import write_csv


STAGE_LABELS = {
    "synthetic_state_protocol": "B1 synthetic latent state inference",
    "state_space_inference_formalization": "B2 formal state-space mechanism",
    "real_video_latent_transfer": "B3 real-video VAE latent transfer",
    "trajectory_observation_core_probe": "B4 trajectory observation core",
    "sampling_time_constraint_preflight": "B6 sampling-time preflight",
    "generative_video_model_probe_colab": "B5 generative video model probe",
    "sampling_time_constraint_colab_probe": "B6 real sampling callback probe",
}


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


def _as_table_value(value: object) -> str:
    """将复杂对象转换为表格中的稳定字符串。"""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if value is None:
        return ""
    return str(value)


def _primary_metrics(details: dict) -> dict:
    """从阶段 evidence details 中抽取适合论文主表的关键指标。"""
    fields = [
        "attacked_negative_fpr",
        "negative_state_over_threshold_count",
        "trajectory_gain_over_state_space",
        "generation_record_count",
        "formal_metric_record_count",
        "constraint_record_count",
        "trajectory_constraint_gain_over_unconstrained",
        "keyed_constraint_alignment_gain_mean",
        "quality_motion_semantic_constraint_gate",
        "claim_boundary",
    ]
    return {field: details.get(field) for field in fields if field in details}


def _build_stage_evidence_rows(evidence_records: list[dict]) -> list[dict]:
    """构建阶段证据主表候选。"""
    rows: list[dict] = []
    for record in evidence_records:
        evidence_stage_id = str(record.get("evidence_stage_id"))
        details = record.get("evidence_details", {})
        rows.append({
            "table_id": "submission_stage_evidence_main_table",
            "evidence_stage_id": evidence_stage_id,
            "stage_label": STAGE_LABELS.get(evidence_stage_id, evidence_stage_id),
            "evidence_decision": record.get("evidence_decision"),
            "primary_metrics": _as_table_value(_primary_metrics(details if isinstance(details, dict) else {})),
            "supporting_artifact_paths": _as_table_value(record.get("supporting_artifact_paths", [])),
        })
    return rows


def _build_main_claim_rows(claim_records: list[dict]) -> list[dict]:
    """构建主文 claim 表候选。"""
    rows: list[dict] = []
    for record in claim_records:
        if record.get("claim_scope") != "main" or record.get("claim_status") != "supported":
            continue
        rows.append({
            "table_id": "submission_main_claim_table",
            "claim_id": record.get("claim_id"),
            "claim_status": record.get("claim_status"),
            "claim_scope": record.get("claim_scope"),
            "supporting_stage_ids": _as_table_value(record.get("supporting_stage_ids", [])),
            "supporting_artifact_paths": _as_table_value(record.get("supporting_artifact_paths", [])),
            "supported_by_governed_artifacts": record.get("supported_by_governed_artifacts") is True,
        })
    return rows


def _build_exploratory_boundary_rows(claim_records: list[dict]) -> list[dict]:
    """构建 exploratory / downgrade 边界表候选。"""
    rows: list[dict] = []
    for record in claim_records:
        if record.get("claim_scope") != "exploratory":
            continue
        rows.append({
            "table_id": "submission_exploratory_boundary_table",
            "claim_id": record.get("claim_id"),
            "claim_status": record.get("claim_status"),
            "downgrade_reason": record.get("downgrade_reason", "none"),
            "supporting_stage_ids": _as_table_value(record.get("supporting_stage_ids", [])),
            "supporting_artifact_paths": _as_table_value(record.get("supporting_artifact_paths", [])),
            "allowed_paper_location": "appendix_or_exploratory_section",
        })
    return rows


def build_submission_main_tables(run_root: str | Path) -> dict:
    """从 submission preparation records 重建主表候选。

    该函数属于通用工程写法。它不手工拼接论文结论, 只把 governed records 映射为可审阅的表格候选。
    """
    run_root = Path(run_root)
    evidence_records = _read_jsonl(run_root / "records" / "submission_stage_evidence_records.jsonl")
    claim_records = _read_jsonl(run_root / "records" / "claim_audit_records.jsonl")
    readiness_summary = _read_json(run_root / "artifacts" / "submission_readiness_summary.json")

    stage_rows = _build_stage_evidence_rows(evidence_records)
    main_claim_rows = _build_main_claim_rows(claim_records)
    exploratory_rows = _build_exploratory_boundary_rows(claim_records)

    write_csv(run_root / "tables" / "submission_stage_evidence_main_table.csv", stage_rows)
    write_csv(run_root / "tables" / "submission_main_claim_table.csv", main_claim_rows)
    write_csv(run_root / "tables" / "submission_exploratory_boundary_table.csv", exploratory_rows)

    table_manifest = {
        "artifact_id": "submission_main_tables_manifest",
        "artifact_type": "table_manifest",
        "stage_id": "submission_freeze_preparation",
        "input_records": [
            str(run_root / "records" / "submission_stage_evidence_records.jsonl"),
            str(run_root / "records" / "claim_audit_records.jsonl"),
            str(run_root / "artifacts" / "submission_readiness_summary.json"),
        ],
        "output_tables": [
            str(run_root / "tables" / "submission_stage_evidence_main_table.csv"),
            str(run_root / "tables" / "submission_main_claim_table.csv"),
            str(run_root / "tables" / "submission_exploratory_boundary_table.csv"),
        ],
        "main_submission_variant": readiness_summary.get("main_submission_variant", "SSTW-T"),
        "exploratory_variants": readiness_summary.get("exploratory_variants", ["SSTW-TC"]),
        "stage_evidence_row_count": len(stage_rows),
        "main_claim_row_count": len(main_claim_rows),
        "exploratory_boundary_row_count": len(exploratory_rows),
        "table_rebuild_status": "PASS" if stage_rows and main_claim_rows and exploratory_rows else "FAIL",
    }
    write_json(run_root / "artifacts" / "submission_main_tables_manifest.json", table_manifest)
    report = (
        "# Submission Main Tables Report\n\n"
        "该报告说明 submission preparation 主表候选已经由 governed records 重建。"
        "表格仅作为论文主表候选, 不新增人工 claim。\n\n"
        f"- stage_evidence_row_count: {len(stage_rows)}\n"
        f"- main_claim_row_count: {len(main_claim_rows)}\n"
        f"- exploratory_boundary_row_count: {len(exploratory_rows)}\n"
        f"- table_rebuild_status: {table_manifest['table_rebuild_status']}\n"
    )
    (run_root / "reports").mkdir(parents=True, exist_ok=True)
    (run_root / "reports" / "submission_main_tables_report.md").write_text(report, encoding="utf-8")
    return table_manifest
