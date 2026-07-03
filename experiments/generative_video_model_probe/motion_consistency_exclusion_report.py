"""motion consistency 阻断样本处理报告。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.generative_video_model_probe.formal_motion_claim_filter import (
    formal_record_supports_motion_claim,
    record_identity_key,
    select_motion_claim_generation_records,
)
from main.protocol.flow_evidence_fields import with_flow_evidence_protocol_defaults
from main.protocol.record_writer import write_json, write_jsonl
from main.protocol.table_builder import write_csv


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL records, 文件不存在时返回空列表。"""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _motion_claim_role(record: dict[str, Any]) -> str:
    """从 generation 或 formal metric record 中读取 motion claim 角色。"""
    explicit = record.get("motion_claim_role") or record.get("motion_calibration_role")
    if explicit:
        return str(explicit)
    prompt_suite_role = str(record.get("prompt_suite_role") or "")
    if "negative_static" in prompt_suite_role:
        return "negative_static"
    if "ambiguous_low_motion" in prompt_suite_role:
        return "ambiguous_low_motion"
    return "positive_motion"


def _exclusion_reason(generation_record: dict[str, Any], formal_record: dict[str, Any] | None) -> str:
    """给出某个生成样本是否被 motion consistency 排除的原因。"""
    if formal_record is None:
        return "formal_metric_record_missing"
    if _motion_claim_role(formal_record or generation_record) != "positive_motion":
        return "boundary_motion_role_not_used_for_positive_motion_claim"
    if formal_record.get("formal_metric_result_used_for_claim") is False:
        return "formal_metric_result_not_used_for_claim"
    if formal_record.get("formal_visual_quality_ready") is not True:
        return "formal_visual_quality_blocked"
    if formal_record.get("formal_motion_consistency_ready") is not True:
        return "formal_motion_consistency_blocked"
    if formal_record.get("formal_semantic_consistency_ready") is not True:
        return "formal_semantic_consistency_blocked"
    if formal_record_supports_motion_claim(formal_record):
        return "included_for_motion_claim"
    return "formal_motion_claim_filter_blocked"


def build_motion_consistency_exclusion_records(run_root: str | Path) -> list[dict[str, Any]]:
    """构建 motion consistency 阻断样本处理 records。"""
    run_root = Path(run_root)
    generation_records = [
        record
        for record in _read_jsonl(run_root / "records" / "generation_records.jsonl")
        if record.get("generation_status") == "success"
    ]
    formal_records = _read_jsonl(run_root / "records" / "formal_quality_motion_semantic_records.jsonl")
    formal_by_key = {record_identity_key(record): record for record in formal_records}
    selection = select_motion_claim_generation_records(generation_records, formal_records)
    records: list[dict[str, Any]] = []
    for generation_record in generation_records:
        key = record_identity_key(generation_record)
        formal_record = formal_by_key.get(key)
        reason = _exclusion_reason(generation_record, formal_record)
        included = key in selection.eligible_generation_keys
        records.append(with_flow_evidence_protocol_defaults({
            "record_version": "motion_consistency_exclusion_v1",
            "generation_model_id": generation_record.get("generation_model_id"),
            "prompt_id": generation_record.get("prompt_id"),
            "seed_id": generation_record.get("seed_id"),
            "trajectory_trace_id": generation_record.get("trajectory_trace_id"),
            "motion_claim_role": _motion_claim_role(formal_record or generation_record),
            "formal_motion_consistency_ready": formal_record.get("formal_motion_consistency_ready") if formal_record else None,
            "formal_visual_quality_ready": formal_record.get("formal_visual_quality_ready") if formal_record else None,
            "formal_semantic_consistency_ready": formal_record.get("formal_semantic_consistency_ready") if formal_record else None,
            "motion_consistency_exclusion_reason": reason,
            "excluded_from_motion_claim": not included,
            "included_in_motion_claim": included,
            "excluded_from_effect_size_claim": not included,
            "retained_for_audit": True,
            "claim_support_status": "motion_consistency_exclusion_audit_record",
        }, trajectory_source_level="motion_consistency_exclusion_audit", claim_support_status="motion_consistency_exclusion_audit_record"))
    return records


def audit_motion_consistency_exclusion_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """审计 motion consistency 阻断处理是否已明确落盘。"""
    included_count = sum(1 for record in records if record.get("included_in_motion_claim") is True)
    excluded_count = sum(1 for record in records if record.get("excluded_from_motion_claim") is True)
    reasons = sorted({str(record.get("motion_consistency_exclusion_reason")) for record in records if record.get("motion_consistency_exclusion_reason")})
    decision = "PASS" if records and included_count + excluded_count == len(records) else "FAIL"
    return {
        "stage_id": "motion_consistency_exclusion_report",
        "motion_consistency_exclusion_decision": decision,
        "claim_support_status": "motion_consistency_exclusion_audit_record" if decision == "PASS" else "motion_consistency_exclusion_blocked",
        "motion_consistency_exclusion_record_count": len(records),
        "motion_consistency_included_count": included_count,
        "motion_consistency_excluded_count": excluded_count,
        "motion_consistency_exclusion_reasons": reasons,
        "motion_consistency_claim_filter_applied": True,
    }


def run_motion_consistency_exclusion_report(run_root: str | Path) -> dict[str, Any]:
    """写出 motion consistency 阻断样本 records、table、decision 和 report。"""
    run_root = Path(run_root)
    records = build_motion_consistency_exclusion_records(run_root)
    audit = audit_motion_consistency_exclusion_records(records)
    write_jsonl(run_root / "records" / "motion_consistency_exclusion_records.jsonl", records)
    write_csv(run_root / "tables" / "motion_consistency_exclusion_table.csv", records)
    write_json(run_root / "artifacts" / "motion_consistency_exclusion_decision.json", audit)
    report = (
        "# Motion Consistency Exclusion Report\n\n"
        "该报告明确说明 formal motion consistency 阻断样本如何处理: 被阻断样本保留为审计记录, "
        "但不进入 motion / trajectory effect-size claim 统计。\n\n"
        f"- motion_consistency_exclusion_decision: {audit['motion_consistency_exclusion_decision']}\n"
        f"- motion_consistency_exclusion_record_count: {audit['motion_consistency_exclusion_record_count']}\n"
        f"- motion_consistency_included_count: {audit['motion_consistency_included_count']}\n"
        f"- motion_consistency_excluded_count: {audit['motion_consistency_excluded_count']}\n"
        f"- motion_consistency_exclusion_reasons: {', '.join(audit['motion_consistency_exclusion_reasons']) if audit['motion_consistency_exclusion_reasons'] else 'none'}\n"
        f"- claim_support_status: {audit['claim_support_status']}\n"
    )
    report_path = run_root / "reports" / "motion_consistency_exclusion_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 motion consistency 阻断样本处理报告。")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--config-path", default="", help="保留 profile config provenance, 当前阶段不读取该配置。")
    args = parser.parse_args()
    payload = run_motion_consistency_exclusion_report(args.run_root)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
