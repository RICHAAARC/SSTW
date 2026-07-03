"""检查 external baseline 是否满足项目内自包含产出要求。

该检查器只消费已经落盘的 source intake、clone、comparison records 和 execution
manifest。它不会调用第三方仓库, 也不会把 non-run record、手工 JSON 或 proxy 分数
升级为正式 `measured_formal` baseline 结果。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from main.protocol.flow_evidence_fields import with_flow_evidence_protocol_defaults
from main.protocol.record_writer import write_json, write_jsonl
from main.protocol.table_builder import write_csv


DEFAULT_PROTOCOL_CONFIG = "configs/protocol/validation_scale_generative_probe.json"
DEFAULT_REQUIRED_MODERN_BASELINES = (
    "videoshield",
    "sigmark",
    "videomark",
    "vidsig",
    "videoseal",
)
EVIDENCE_PATH_FIELDS = (
    "external_baseline_official_output_path",
    "external_baseline_official_stdout_path",
    "external_baseline_official_stderr_path",
    "external_baseline_official_command_manifest_path",
)


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
    return [json.loads(line) for line in path.read_text(encoding="utf-8") .splitlines() if line.strip()]


def _load_required_modern_baselines(config_path: str | Path) -> list[str]:
    """从 protocol config 中读取必须正式测量的现代 baseline 清单。"""
    config = _read_json(Path(config_path))
    names = [
        str(name)
        for name in config.get("required_modern_external_baseline_adapter_names", [])
        if str(name)
    ]
    return names or list(DEFAULT_REQUIRED_MODERN_BASELINES)


def _path_exists(path_text: str | None, run_root: Path) -> bool:
    """检查 evidence path 是否存在, 同时兼容绝对路径和相对 run_root 路径。"""
    if not path_text:
        return False
    path = Path(path_text)
    candidates = [path]
    if not path.is_absolute():
        candidates.append(run_root / path)
    return any(candidate.exists() for candidate in candidates)


def _manifest_return_code_ok(path_text: str | None, run_root: Path) -> bool:
    """读取官方命令 manifest, 确认命令成功返回。"""
    if not path_text:
        return False
    path = Path(path_text)
    if not path.is_absolute():
        path = run_root / path
    if not path.exists():
        return False
    payload = _read_json(path)
    return payload.get("command_return_code") in {0, "0", None}


def _index_rows(rows: Iterable[Mapping[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    """按指定字段索引列表对象。"""
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row.get(key) or "")
        if value:
            indexed[value] = dict(row)
    return indexed


def _rows_from_manifest(manifest: Mapping[str, Any], field_name: str) -> list[dict[str, Any]]:
    """从 manifest 中读取列表行, 字段缺失时返回空列表。"""
    rows = manifest.get(field_name, [])
    if not isinstance(rows, list):
        return []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def _build_baseline_rows(
    run_root: Path,
    required_names: list[str],
    score_records: list[dict[str, Any]],
    intake_manifest: Mapping[str, Any],
    inspection_manifest: Mapping[str, Any],
    clone_manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """构造每个现代 baseline 的 clone/build/run/adapt/record 自包含检查行。"""
    intake_by_name = _index_rows(_rows_from_manifest(intake_manifest, "baseline_sources"), "baseline_id")
    inspection_by_name = _index_rows(_rows_from_manifest(inspection_manifest, "source_inspections"), "baseline_id")
    clone_by_name = _index_rows(_rows_from_manifest(clone_manifest, "clone_results"), "baseline_id")
    rows: list[dict[str, Any]] = []
    for baseline_name in required_names:
        records = [
            record for record in score_records
            if record.get("external_baseline_name") == baseline_name
        ]
        measured_records = [
            record for record in records
            if record.get("metric_status") == "measured_formal"
            and record.get("external_baseline_score_status") in {None, "measured_formal"}
        ]
        intake = intake_by_name.get(baseline_name, {})
        inspection = inspection_by_name.get(baseline_name, {})
        clone = clone_by_name.get(baseline_name, {})
        evidence_paths = [
            str(record.get(field) or "")
            for record in measured_records
            for field in EVIDENCE_PATH_FIELDS
            if record.get(field)
        ]
        materialized_evidence_paths = [
            path for path in evidence_paths
            if _path_exists(path, run_root)
        ]
        command_manifest_paths = [
            str(record.get("external_baseline_official_command_manifest_path") or "")
            for record in measured_records
            if record.get("external_baseline_official_command_manifest_path")
        ]
        command_manifest_ok_count = sum(
            1 for path in command_manifest_paths
            if _manifest_return_code_ok(path, run_root)
        )
        source_dir_exists = bool(intake.get("source_dir_exists") or inspection.get("source_dir_exists") or clone.get("source_dir_exists"))
        clone_operation_status = str(clone.get("clone_operation_status") or "missing")
        clone_ready = source_dir_exists or clone_operation_status in {"cloned", "updated"}
        build_ready = bool(command_manifest_paths) and command_manifest_ok_count == len(command_manifest_paths)
        run_ready = bool(measured_records)
        adapt_ready = all(
            record.get("external_baseline_adapter_path")
            and record.get("external_baseline_score_source") not in {"sstw_proxy_score", "paper_number_only", "manual_result_json"}
            for record in measured_records
        ) if measured_records else False
        record_ready = bool(measured_records) and len(materialized_evidence_paths) >= len(command_manifest_paths)
        rows.append({
            "baseline_name": baseline_name,
            "source_intake_status": intake.get("source_intake_status", "missing"),
            "source_dir_exists": source_dir_exists,
            "clone_operation_status": clone_operation_status,
            "clone_ready": clone_ready,
            "build_ready": build_ready,
            "run_ready": run_ready,
            "adapt_ready": adapt_ready,
            "record_ready": record_ready,
            "measured_formal_record_count": len(measured_records),
            "unsupported_record_count": sum(1 for record in records if record.get("metric_status") == "unsupported"),
            "official_command_manifest_count": len(command_manifest_paths),
            "official_command_manifest_ok_count": command_manifest_ok_count,
            "materialized_evidence_path_count": len(materialized_evidence_paths),
            "external_baseline_self_contained": all([clone_ready, build_ready, run_ready, adapt_ready, record_ready]),
        })
    return rows


def build_external_baseline_self_containment_decision(
    run_root: str | Path,
    config_path: str | Path = DEFAULT_PROTOCOL_CONFIG,
) -> dict[str, Any]:
    """构建 external baseline 自包含产出判定。"""
    run_root = Path(run_root)
    required_names = _load_required_modern_baselines(config_path)
    score_records = _read_jsonl(run_root / "records" / "external_baseline_score_records.jsonl")
    comparison_decision = _read_json(run_root / "artifacts" / "external_baseline_comparison_decision.json")
    execution_manifest = _read_json(run_root / "artifacts" / "external_baseline_execution_manifest.json")
    intake_manifest = _read_json(run_root / "artifacts" / "external_baseline_intake_manifest.json")
    inspection_manifest = _read_json(run_root / "artifacts" / "external_baseline_source_inspection.json")
    clone_manifest = _read_json(run_root / "artifacts" / "external_baseline_clone_results.json")

    rows = _build_baseline_rows(
        run_root,
        required_names,
        score_records,
        intake_manifest,
        inspection_manifest,
        clone_manifest,
    )
    missing_names = [
        row["baseline_name"]
        for row in rows
        if not row["external_baseline_self_contained"]
    ]
    comparison_passed = comparison_decision.get("external_baseline_comparison_decision") == "PASS"
    execution_manifest_bound = execution_manifest.get("formal_evidence_status") == "evidence_paths_bound"
    formal_measured_names = set(execution_manifest.get("modern_external_baseline_formal_measured_adapter_names") or [])
    missing_formal_names = sorted(set(required_names) - formal_measured_names)
    missing_requirements: list[str] = []
    if not comparison_passed:
        missing_requirements.append("external_baseline_comparison_decision_passed")
    if not execution_manifest_bound:
        missing_requirements.append("external_baseline_execution_manifest_evidence_paths_bound")
    if missing_formal_names:
        missing_requirements.append("all_required_modern_baselines_measured_formal")
    if missing_names:
        missing_requirements.append("all_required_modern_baselines_clone_build_run_adapt_record")

    decision = "PASS" if not missing_requirements else "FAIL"
    return {
        "stage_id": "external_baseline_self_containment_decision",
        "run_root": str(run_root),
        "external_baseline_self_containment_decision": decision,
        "claim_support_status": "external_baseline_self_contained_measured_formal_ready"
        if decision == "PASS"
        else "external_baseline_self_containment_blocked",
        "required_modern_external_baseline_adapter_names": required_names,
        "required_modern_external_baseline_adapter_count": len(required_names),
        "self_contained_modern_external_baseline_count": sum(1 for row in rows if row["external_baseline_self_contained"]),
        "missing_self_contained_modern_external_baseline_names": missing_names,
        "missing_formal_modern_external_baseline_names": missing_formal_names,
        "missing_self_containment_requirements": missing_requirements,
        "self_containment_missing_requirement_count": len(missing_requirements),
        "external_baseline_comparison_decision": comparison_decision.get("external_baseline_comparison_decision"),
        "formal_evidence_status": execution_manifest.get("formal_evidence_status"),
        "evidence_path_count": execution_manifest.get("evidence_path_count", 0),
        "baseline_self_containment_rows": rows,
    }


def write_external_baseline_self_containment_decision(
    run_root: str | Path,
    config_path: str | Path = DEFAULT_PROTOCOL_CONFIG,
) -> dict[str, Any]:
    """写出 external baseline 自包含产出 decision、records、table 和 report。"""
    run_root = Path(run_root)
    audit = build_external_baseline_self_containment_decision(run_root, config_path)
    record = with_flow_evidence_protocol_defaults(
        {"record_version": "external_baseline_self_containment_decision_v1", **audit},
        trajectory_source_level="external_baseline_self_containment_governance",
        flow_state_admissibility_status="not_applicable",
        claim_support_status=audit["claim_support_status"],
    )
    write_jsonl(run_root / "records" / "external_baseline_self_containment_records.jsonl", [record])
    write_csv(run_root / "tables" / "external_baseline_self_containment_table.csv", audit["baseline_self_containment_rows"])
    write_json(run_root / "artifacts" / "external_baseline_self_containment_decision.json", audit)
    report = (
        "# External Baseline Self-containment Report\n\n"
        "该报告检查现代 external baseline 是否由项目内 clone / build / run / adapt / record "
        "闭环产出。non-run record 只能作为阻断记录, 不能替代 measured_formal。\n\n"
        f"- external_baseline_self_containment_decision: {audit['external_baseline_self_containment_decision']}\n"
        f"- self_contained_modern_external_baseline_count: {audit['self_contained_modern_external_baseline_count']}\n"
        f"- required_modern_external_baseline_adapter_count: {audit['required_modern_external_baseline_adapter_count']}\n"
        f"- missing_self_contained_modern_external_baseline_names: "
        f"{', '.join(audit['missing_self_contained_modern_external_baseline_names']) if audit['missing_self_contained_modern_external_baseline_names'] else 'none'}\n"
        f"- missing_self_containment_requirements: "
        f"{', '.join(audit['missing_self_containment_requirements']) if audit['missing_self_containment_requirements'] else 'none'}\n"
    )
    report_path = run_root / "reports" / "external_baseline_self_containment_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 external baseline 自包含产出闭环。")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--config-path", default=DEFAULT_PROTOCOL_CONFIG)
    args = parser.parse_args()
    payload = write_external_baseline_self_containment_decision(args.run_root, args.config_path)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
