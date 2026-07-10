"""检查 protocol split 隔离和阈值来源。

该检查器用于在进入 paper 级结果前确认 calibration、held-out test、stress 和
ablation 记录没有共享同一视频身份。它只读取 run_root 中已经落盘的 records 和
decision artifacts, 不运行模型, 不重算阈值。
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from evaluation.protocol.flow_evidence_fields import with_flow_evidence_protocol_defaults
from evaluation.protocol.record_writer import write_json, write_jsonl
from evaluation.protocol.table_builder import write_csv


RECORD_SOURCES = (
    ("generation_records", "records/generation_records.jsonl"),
    ("formal_quality_motion_semantic_records", "records/formal_quality_motion_semantic_records.jsonl"),
    ("runtime_detection_records", "records/runtime_detection_records.jsonl"),
    ("sstw_clean_negative_score_records", "records/sstw_clean_negative_score_records.jsonl"),
    ("sstw_measured_formal_records", "records/sstw_measured_formal_records.jsonl"),
    ("external_baseline_score_records", "records/external_baseline_score_records.jsonl"),
    ("fair_detection_calibration_records", "records/fair_detection_calibration_records.jsonl"),
    ("formal_internal_ablation_variant_records", "records/formal_internal_ablation_variant_records.jsonl"),
    ("validation_internal_ablation_records", "records/validation_internal_ablation_records.jsonl"),
)
SPLIT_FIELDS = ("split", "protocol_split", "data_split", "sample_split")
CALIBRATION_SPLITS = {"calibration", "calibration_negative", "calibration_split"}
HELDOUT_TEST_SPLITS = {"test", "heldout", "heldout_test", "test_split"}
STRESS_SPLITS = {"stress", "stress_test", "adaptive_attack"}
ABLATION_SPLITS = {"ablation", "internal_ablation"}


def _read_json(path: Path) -> dict[str, Any]:
    """读取 JSON artifact, 文件不存在时返回空对象。"""
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON 顶层必须是对象: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL records, 文件不存在时返回空列表。"""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _raw_split(record: Mapping[str, Any]) -> str:
    """从常见 split 字段中提取原始 split 名称。"""
    for field in SPLIT_FIELDS:
        value = str(record.get(field) or "").strip()
        if value:
            return value
    return ""


def _canonical_split(value: str) -> str:
    """把不同写法归一为少量审计 split。"""
    normalized = value.strip().lower()
    if normalized in CALIBRATION_SPLITS:
        return "calibration"
    if normalized in HELDOUT_TEST_SPLITS:
        return "heldout_test"
    if normalized in STRESS_SPLITS:
        return "stress"
    if normalized in ABLATION_SPLITS:
        return "ablation"
    return normalized or "unspecified"


def _identity_key(record: Mapping[str, Any]) -> str:
    """构造用于泄漏检查的视频身份键。

    优先使用视频哈希或 trajectory trace。若二者都不存在, 则退化为
    generation_model_id / prompt_id / seed_id 的组合。该退化键只用于轻量治理,
    不能替代正式 video identity manifest。
    """
    for field in ("video_sha256", "source_video_sha256", "attacked_video_sha256", "trajectory_trace_id"):
        value = str(record.get(field) or "").strip()
        if value:
            return f"{field}:{value}"
    generation_model_id = str(record.get("generation_model_id") or "unknown_model")
    prompt_id = str(record.get("prompt_id") or "unknown_prompt")
    seed_id = str(record.get("seed_id") or "unknown_seed")
    return f"prompt_seed:{generation_model_id}:{prompt_id}:{seed_id}"


def _collect_split_rows(run_root: Path) -> list[dict[str, Any]]:
    """从 run_root 收集带 split 的轻量审计行。"""
    rows: list[dict[str, Any]] = []
    for source_name, relative_path in RECORD_SOURCES:
        records = _read_jsonl(run_root / relative_path)
        for index, record in enumerate(records):
            raw_split = _raw_split(record)
            canonical = _canonical_split(raw_split)
            if canonical == "unspecified":
                continue
            rows.append({
                "record_source": source_name,
                "record_index": index,
                "raw_split": raw_split,
                "canonical_split": canonical,
                "identity_key": _identity_key(record),
                "prompt_id": record.get("prompt_id"),
                "seed_id": record.get("seed_id"),
                "trajectory_trace_id": record.get("trajectory_trace_id"),
                "attack_name": record.get("attack_name"),
            })
    return rows


def _find_leakage_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """查找同一 identity_key 是否跨 calibration 与 heldout_test 出现。"""
    split_by_key: dict[str, set[str]] = defaultdict(set)
    source_by_key: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        key = str(row.get("identity_key") or "")
        split = str(row.get("canonical_split") or "")
        if key:
            split_by_key[key].add(split)
            source_by_key[key].add(str(row.get("record_source") or ""))
    leakage_rows = []
    for key, splits in sorted(split_by_key.items()):
        if "calibration" in splits and "heldout_test" in splits:
            leakage_rows.append({
                "identity_key": key,
                "leaked_splits": sorted(splits),
                "record_sources": sorted(source_by_key[key]),
                "leakage_type": "calibration_heldout_test_identity_overlap",
            })
    return leakage_rows


def _threshold_source_rows(run_root: Path) -> list[dict[str, Any]]:
    """读取阈值来源 artifact, 检查是否只来自 calibration split。"""
    candidates = [
        (
            "motion_threshold_calibration_decision",
            run_root / "artifacts" / "motion_threshold_calibration_decision.json",
            "motion_threshold_source_split",
        ),
        (
            "pilot_paper_frozen_threshold",
            run_root / "thresholds" / "pilot_paper_frozen_threshold.json",
            "threshold_source_split",
        ),
        (
            "pilot_paper_gate_decision",
            run_root / "artifacts" / "pilot_paper_gate_decision.json",
            "threshold_source_split",
        ),
    ]
    rows = []
    for artifact_name, path, field in candidates:
        payload = _read_json(path)
        if not payload:
            continue
        source_split = str(payload.get(field) or "")
        rows.append({
            "artifact_name": artifact_name,
            "artifact_path": str(path),
            "threshold_source_field": field,
            "threshold_source_split": source_split,
            "threshold_source_split_valid": _canonical_split(source_split) == "calibration",
        })
    return rows


def build_data_split_and_leakage_guard(
    run_root: str | Path,
) -> dict[str, Any]:
    """构建数据切分与泄漏检查结果。"""
    run_root = Path(run_root)
    split_rows = _collect_split_rows(run_root)
    leakage_rows = _find_leakage_rows(split_rows)
    threshold_rows = _threshold_source_rows(run_root)
    split_counts: dict[str, int] = defaultdict(int)
    for row in split_rows:
        split_counts[str(row["canonical_split"])] += 1
    formal_split_detected = bool({"calibration", "heldout_test"} & set(split_counts))
    invalid_threshold_rows = [
        row for row in threshold_rows
        if not row["threshold_source_split_valid"]
    ]
    missing_requirements: list[str] = []
    if leakage_rows:
        missing_requirements.append("calibration_heldout_test_identity_disjoint")
    if invalid_threshold_rows:
        missing_requirements.append("threshold_source_split_is_calibration")
    decision = "PASS" if not missing_requirements else "FAIL"
    return {
        "stage_id": "data_split_and_leakage_guard",
        "run_root": str(run_root),
        "data_split_and_leakage_guard_decision": decision,
        "claim_support_status": "data_split_and_leakage_guard_passed"
        if decision == "PASS"
        else "data_split_and_leakage_guard_blocked",
        "formal_split_detected": formal_split_detected,
        "split_record_count": len(split_rows),
        "split_counts": dict(sorted(split_counts.items())),
        "leakage_count": len(leakage_rows),
        "leakage_rows": leakage_rows,
        "threshold_source_rows": threshold_rows,
        "invalid_threshold_source_count": len(invalid_threshold_rows),
        "missing_data_split_requirements": missing_requirements,
        "data_split_missing_requirement_count": len(missing_requirements),
    }


def write_data_split_and_leakage_guard(
    run_root: str | Path,
) -> dict[str, Any]:
    """写出数据切分与泄漏检查 decision、records、table 和报告。"""
    run_root = Path(run_root)
    audit = build_data_split_and_leakage_guard(run_root)
    record = with_flow_evidence_protocol_defaults(
        {"record_version": "data_split_and_leakage_guard_v1", **audit},
        trajectory_source_level="data_split_and_leakage_governance",
        flow_state_admissibility_status="not_applicable",
        claim_support_status=audit["claim_support_status"],
    )
    write_jsonl(run_root / "records" / "data_split_and_leakage_guard_records.jsonl", [record])
    table_rows = audit["leakage_rows"] or [
        {
            "identity_key": "none",
            "leaked_splits": "none",
            "record_sources": "none",
            "leakage_type": "no_calibration_heldout_test_identity_overlap",
        }
    ]
    write_csv(run_root / "tables" / "data_split_and_leakage_guard_table.csv", table_rows)
    write_json(run_root / "artifacts" / "data_split_and_leakage_guard_decision.json", audit)
    report = (
        "# Data Split And Leakage Guard Report\n\n"
        "该报告检查 calibration / held-out test 等 split 的身份隔离和阈值来源。"
        "它不重算阈值, 只审计已落盘 artifact 的来源字段。\n\n"
        f"- data_split_and_leakage_guard_decision: {audit['data_split_and_leakage_guard_decision']}\n"
        f"- formal_split_detected: {str(audit['formal_split_detected']).lower()}\n"
        f"- split_counts: {audit['split_counts']}\n"
        f"- leakage_count: {audit['leakage_count']}\n"
        f"- invalid_threshold_source_count: {audit['invalid_threshold_source_count']}\n"
        f"- missing_data_split_requirements: "
        f"{', '.join(audit['missing_data_split_requirements']) if audit['missing_data_split_requirements'] else 'none'}\n"
    )
    report_path = run_root / "reports" / "data_split_and_leakage_guard_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="检查数据切分隔离和阈值来源。")
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    payload = write_data_split_and_leakage_guard(args.run_root)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
