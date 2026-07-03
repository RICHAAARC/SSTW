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
OFFICIAL_BUNDLE_PATH_FIELDS = (
    "external_baseline_official_result_bundle_path",
    "external_baseline_official_execution_manifest_path",
)
REPOSITORY_GENERATED_OFFICIAL_PROVENANCE = "repository_generated_from_third_party_official_code"


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
    return _resolve_existing_path(path_text, run_root) is not None


def _resolve_existing_path(path_text: str | None, run_root: Path) -> Path | None:
    """解析已经落盘的证据路径。

    该函数属于通用工程写法。paper gate 在 Colab 中使用绝对路径, 本地测试
    常使用相对路径, 因此这里统一尝试原路径和相对 `run_root` 的路径。
    """
    if not path_text:
        return None
    path = Path(path_text)
    candidates = [path]
    if not path.is_absolute():
        candidates.append(run_root / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _manifest_return_code_ok(path_text: str | None, run_root: Path) -> bool:
    """读取官方命令 manifest, 确认命令成功返回。"""
    path = _resolve_existing_path(path_text, run_root)
    if path is None:
        return False
    payload = _read_json(path)
    return payload.get("command_return_code") in {0, "0", None}


def _official_execution_manifest_ok(
    path_text: str | None,
    run_root: Path,
    baseline_name: str,
) -> bool:
    """校验项目内 official reference 执行 manifest。

    baseline 专用 Notebook 会把 clone / build / run / adapt 的重型证据写入
    official bundle 下的 execution manifest。paper gate 后处理通常不会保留
    第三方源码目录本身, 因此 self-containment 不能只看当前 checkout 中的
    `source_dir_exists`, 还必须接受该 execution manifest 作为项目内运行闭环证据。
    """

    path = _resolve_existing_path(path_text, run_root)
    if path is None:
        return False
    payload = _read_json(path)
    manifest_baseline = str(payload.get("baseline_id") or "")
    if manifest_baseline and manifest_baseline != baseline_name:
        return False
    positive_evidence_found = False
    failed_count = payload.get("failed_bundle_record_count")
    try:
        if failed_count is not None and int(failed_count) != 0:
            return False
    except (TypeError, ValueError):
        return False
    execution_status = str(payload.get("execution_status") or "").strip()
    if execution_status and execution_status not in {"executed", "completed", "generated", "ready"}:
        return False
    if execution_status:
        positive_evidence_found = True
    command_results = payload.get("command_results")
    if isinstance(command_results, list) and command_results:
        for item in command_results:
            if not isinstance(item, Mapping):
                return False
            if item.get("return_code") not in {0, "0", None}:
                return False
        positive_evidence_found = True
    generated_count = payload.get("generated_bundle_record_count")
    if generated_count is not None:
        try:
            if int(generated_count) <= 0:
                return False
            positive_evidence_found = True
        except (TypeError, ValueError):
            return False
    return positive_evidence_found


def _official_bundle_payload_ok(path_text: str | None, run_root: Path, baseline_name: str) -> bool:
    """校验单条 official bundle record 来自项目内生成链路。

    该函数是项目特定门禁。只有带有
    `repository_generated_from_third_party_official_code` provenance 且绑定
    official execution manifest 的 bundle JSON, 才能替代当前 paper gate
    checkout 中的源码目录作为自包含运行证据。
    """

    path = _resolve_existing_path(path_text, run_root)
    if path is None:
        return False
    payload = _read_json(path)
    if str(payload.get("official_result_provenance") or "") != REPOSITORY_GENERATED_OFFICIAL_PROVENANCE:
        return False
    payload_baseline = str(payload.get("official_adapter_baseline_id") or payload.get("baseline_id") or "")
    if payload_baseline and payload_baseline != baseline_name:
        return False
    manifest_path = str(payload.get("official_execution_manifest_path") or "")
    if not manifest_path:
        return False
    if not _record_clean_negative_ready(payload):
        return False
    return _official_execution_manifest_ok(manifest_path, run_root, baseline_name)


def _record_clean_negative_ready(record: Mapping[str, Any]) -> bool:
    """检查 measured_formal baseline record 是否携带公平校准所需 clean negative 分数。

    validation_scale 的 self-containment 不应只证明 positive score 来自项目内官方流程,
    还必须证明同一 official bundle 已经提供 baseline 自身 clean negative 分布的
    分数来源。否则该 baseline 后续无法参与 `TPR@target FPR` 公平比较。
    """

    score = record.get("external_baseline_clean_negative_score", record.get("clean_negative_score"))
    if score is None or score == "" or score == "unsupported":
        return False
    try:
        float(score)
    except (TypeError, ValueError):
        return False
    return bool(
        record.get("external_baseline_clean_negative_video_path")
        or record.get("official_clean_negative_source_video_path")
        or record.get("official_clean_negative_bit_accuracy_npz_path")
        or record.get("official_clean_negative_results_json_path")
        or record.get("official_clean_negative_frame_array_path")
    )


def _record_anchor_ready(record: Mapping[str, Any]) -> bool:
    """检查 formal baseline record 是否保留公平比较所需的 prompt / seed / attack anchor。"""

    return all(record.get(field_name) not in {None, ""} for field_name in ("prompt_id", "seed_id", "attack_name"))


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
        formal_candidate_records = [
            record for record in records
            if record.get("metric_status") == "measured_formal"
            and record.get("external_baseline_score_status") in {None, "measured_formal"}
        ]
        measured_records = [
            record for record in formal_candidate_records
            if _record_anchor_ready(record)
        ]
        formal_anchor_missing_count = len(formal_candidate_records) - len(measured_records)
        intake = intake_by_name.get(baseline_name, {})
        inspection = inspection_by_name.get(baseline_name, {})
        clone = clone_by_name.get(baseline_name, {})
        evidence_paths = [
            str(record.get(field) or "")
            for record in measured_records
            for field in EVIDENCE_PATH_FIELDS
            if record.get(field)
        ]
        official_bundle_paths = [
            str(record.get(OFFICIAL_BUNDLE_PATH_FIELDS[0]) or "")
            for record in measured_records
            if record.get(OFFICIAL_BUNDLE_PATH_FIELDS[0])
        ]
        official_execution_manifest_paths = [
            str(record.get(OFFICIAL_BUNDLE_PATH_FIELDS[1]) or "")
            for record in measured_records
            if record.get(OFFICIAL_BUNDLE_PATH_FIELDS[1])
        ]
        materialized_evidence_paths = [
            path for path in evidence_paths
            if _path_exists(path, run_root)
        ]
        materialized_official_bundle_paths = [
            path for path in official_bundle_paths
            if _path_exists(path, run_root)
        ]
        materialized_official_execution_manifest_paths = [
            path for path in official_execution_manifest_paths
            if _path_exists(path, run_root)
        ]
        official_bundle_record_ok_count = sum(
            1 for path in official_bundle_paths
            if _official_bundle_payload_ok(path, run_root, baseline_name)
        )
        official_execution_manifest_ok_count = sum(
            1 for path in official_execution_manifest_paths
            if _official_execution_manifest_ok(path, run_root, baseline_name)
        )
        clean_negative_ready_count = sum(1 for record in measured_records if _record_clean_negative_ready(record))
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
        source_clone_ready = source_dir_exists or clone_operation_status in {"cloned", "updated"}
        repository_generated_official_bundle_ready = (
            bool(measured_records)
            and len(official_bundle_paths) == len(measured_records)
            and len(official_execution_manifest_paths) == len(measured_records)
            and official_bundle_record_ok_count == len(measured_records)
            and official_execution_manifest_ok_count == len(measured_records)
        )
        clone_ready = source_clone_ready or repository_generated_official_bundle_ready
        build_ready = bool(command_manifest_paths) and command_manifest_ok_count == len(command_manifest_paths)
        run_ready = bool(measured_records) and repository_generated_official_bundle_ready
        anchor_ready = bool(formal_candidate_records) and formal_anchor_missing_count == 0
        adapt_ready = all(
            record.get("external_baseline_adapter_path")
            and record.get("external_baseline_score_source") not in {"sstw_proxy_score", "paper_number_only", "manual_result_json"}
            for record in measured_records
        ) if measured_records else False
        record_ready = bool(measured_records) and len(materialized_evidence_paths) >= len(command_manifest_paths)
        clean_negative_ready = bool(measured_records) and clean_negative_ready_count == len(measured_records)
        rows.append({
            "baseline_name": baseline_name,
            "source_intake_status": intake.get("source_intake_status", "missing"),
            "source_dir_exists": source_dir_exists,
            "clone_operation_status": clone_operation_status,
            "source_clone_ready": source_clone_ready,
            "repository_generated_official_bundle_ready": repository_generated_official_bundle_ready,
            "clone_ready": clone_ready,
            "build_ready": build_ready,
            "run_ready": run_ready,
            "anchor_ready": anchor_ready,
            "adapt_ready": adapt_ready,
            "record_ready": record_ready,
            "clean_negative_ready": clean_negative_ready,
            "measured_formal_record_count": len(measured_records),
            "formal_candidate_record_count": len(formal_candidate_records),
            "formal_anchor_missing_count": formal_anchor_missing_count,
            "clean_negative_ready_count": clean_negative_ready_count,
            "unsupported_record_count": sum(1 for record in records if record.get("metric_status") == "unsupported"),
            "official_command_manifest_count": len(command_manifest_paths),
            "official_command_manifest_ok_count": command_manifest_ok_count,
            "official_bundle_record_count": len(official_bundle_paths),
            "official_bundle_record_ok_count": official_bundle_record_ok_count,
            "official_execution_manifest_count": len(official_execution_manifest_paths),
            "official_execution_manifest_ok_count": official_execution_manifest_ok_count,
            "materialized_evidence_path_count": len(materialized_evidence_paths),
            "materialized_official_bundle_path_count": len(materialized_official_bundle_paths),
            "materialized_official_execution_manifest_path_count": len(materialized_official_execution_manifest_paths),
            "external_baseline_self_contained": all([
                clone_ready,
                build_ready,
                run_ready,
                repository_generated_official_bundle_ready,
                anchor_ready,
                adapt_ready,
                record_ready,
                clean_negative_ready,
            ]),
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
    missing_clean_negative_names = [
        row["baseline_name"]
        for row in rows
        if not row.get("clean_negative_ready")
    ]
    missing_anchor_names = [
        row["baseline_name"]
        for row in rows
        if not row.get("anchor_ready")
    ]
    missing_repository_generated_official_bundle_names = [
        row["baseline_name"]
        for row in rows
        if not row.get("repository_generated_official_bundle_ready")
    ]
    comparison_passed = comparison_decision.get("external_baseline_comparison_decision") == "PASS"
    execution_manifest_bound = execution_manifest.get("formal_evidence_status") == "evidence_paths_bound"
    formal_measured_names = {
        str(row["baseline_name"])
        for row in rows
        if int(row.get("measured_formal_record_count") or 0) > 0
    }
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
    if missing_repository_generated_official_bundle_names:
        missing_requirements.append("all_required_modern_baselines_repository_generated_official_bundles")
    if missing_clean_negative_names:
        missing_requirements.append("all_required_modern_baselines_clean_negative_scores")
    if missing_anchor_names:
        missing_requirements.append("all_required_modern_baselines_prompt_seed_attack_anchors")

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
        "missing_clean_negative_modern_external_baseline_names": missing_clean_negative_names,
        "missing_anchor_modern_external_baseline_names": missing_anchor_names,
        "missing_repository_generated_official_bundle_modern_external_baseline_names": missing_repository_generated_official_bundle_names,
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
