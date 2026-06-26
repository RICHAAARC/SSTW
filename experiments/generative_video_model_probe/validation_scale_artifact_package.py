"""重建 validation_scale 诊断图与 package manifest。

该模块只从 run_root 中已经存在的 governed records、tables、reports 和 decision
artifacts 生成派生产物。它不写入正式效果结论, 也不手工补造缺失的实验结果。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from main.protocol.record_writer import write_json


VALIDATION_SCALE_REQUIRED_PACKAGE_RELPATHS = (
    "records/validation_scale_gate_records.jsonl",
    "tables/validation_scale_gate_table.csv",
    "artifacts/validation_scale_gate_decision.json",
    "reports/validation_scale_gate_report.md",
    "artifacts/external_baseline_self_containment_decision.json",
    "artifacts/data_split_and_leakage_guard_decision.json",
    "artifacts/validation_scale_to_pilot_paper_transition_decision.json",
    "figures/validation_scale_gate_figure.json",
)


def _read_json(path: Path) -> dict[str, Any]:
    """读取 JSON artifact, 文件不存在时返回空对象。"""
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON 顶层必须是对象: {path}")
    return payload


def _sha256(path: Path) -> str | None:
    """计算文件 sha256 摘要, 文件不存在时返回 None。"""
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _requirement_rows(decision: Mapping[str, Any]) -> list[dict[str, Any]]:
    """把 validation_scale gate 的 requirement 列表转成图数据行。"""
    missing = {str(item) for item in decision.get("missing_validation_requirements", [])}
    known = [
        "small_scale_claim_pilot_gate_passed",
        "validation_generation_records_ready",
        "validation_motion_threshold_calibration_ready",
        "validation_formal_motion_claim_ready",
        "validation_attack_records_ready",
        "validation_detection_records_ready",
        "validation_external_baseline_status_records_ready",
        "validation_external_baseline_comparison_records_ready",
        "validation_external_baseline_self_containment_ready",
        "validation_data_split_and_leakage_guard_ready",
        "validation_internal_ablation_records_ready",
        "validation_adaptive_attack_records_ready",
        "validation_replay_or_sketch_records_ready",
        "validation_confidence_interval_report_ready",
        "validation_artifact_rebuild_dry_run_ready",
    ]
    observed = list(dict.fromkeys([*known, *sorted(missing)]))
    return [
        {
            "requirement_name": name,
            "requirement_status": "FAIL" if name in missing else "PASS",
            "requirement_ready_value": 0 if name in missing else 1,
        }
        for name in observed
    ]


def build_validation_scale_gate_figure(run_root: str | Path) -> dict[str, Any]:
    """构造 validation_scale gate 诊断图 manifest。"""
    run_root = Path(run_root)
    decision_path = run_root / "artifacts" / "validation_scale_gate_decision.json"
    decision = _read_json(decision_path)
    rows = _requirement_rows(decision)
    return {
        "artifact_name": "validation_scale_gate_figure.json",
        "artifact_type": "figure_manifest",
        "figure_id": "validation_scale_gate_figure",
        "figure_title": "validation_scale gate requirement readiness",
        "run_root": str(run_root),
        "source_artifact_paths": [str(decision_path)],
        "validation_scale_gate_decision": decision.get("validation_scale_gate_decision", "missing"),
        "paper_result_level": decision.get("paper_result_level"),
        "target_fpr": decision.get("target_fpr"),
        "claim_support_status": "validation_scale_diagnostic_figure_not_effect_size_claim",
        "encoding": {
            "x": "requirement_name",
            "y": "requirement_ready_value",
            "color": "requirement_status",
        },
        "figure_rows": rows,
    }


def write_validation_scale_gate_figure(run_root: str | Path) -> dict[str, Any]:
    """写出 validation_scale gate 诊断图 manifest。"""
    run_root = Path(run_root)
    figure = build_validation_scale_gate_figure(run_root)
    write_json(run_root / "figures" / "validation_scale_gate_figure.json", figure)
    return figure


def _inventory_rows(run_root: Path, relpaths: Iterable[str]) -> list[dict[str, Any]]:
    """为 package manifest 构造 artifact inventory。"""
    rows: list[dict[str, Any]] = []
    for relpath in relpaths:
        path = run_root / relpath
        rows.append({
            "artifact_relpath": relpath,
            "artifact_exists": path.exists(),
            "artifact_size_bytes": path.stat().st_size if path.exists() and path.is_file() else 0,
            "artifact_sha256": _sha256(path),
        })
    return rows


def build_validation_scale_package_manifest(run_root: str | Path) -> dict[str, Any]:
    """构建 validation_scale package manifest。"""
    run_root = Path(run_root)
    inventory = _inventory_rows(run_root, VALIDATION_SCALE_REQUIRED_PACKAGE_RELPATHS)
    missing = [row["artifact_relpath"] for row in inventory if not row["artifact_exists"]]
    validation_gate = _read_json(run_root / "artifacts" / "validation_scale_gate_decision.json")
    self_containment = _read_json(run_root / "artifacts" / "external_baseline_self_containment_decision.json")
    data_guard = _read_json(run_root / "artifacts" / "data_split_and_leakage_guard_decision.json")
    transition = _read_json(run_root / "artifacts" / "validation_scale_to_pilot_paper_transition_decision.json")
    decision_ready = (
        validation_gate.get("validation_scale_gate_decision") == "PASS"
        and self_containment.get("external_baseline_self_containment_decision") == "PASS"
        and data_guard.get("data_split_and_leakage_guard_decision") == "PASS"
        and transition.get("validation_scale_to_pilot_paper_transition_decision") == "PASS"
        and not missing
    )
    return {
        "artifact_name": "validation_scale_package_manifest.json",
        "artifact_type": "package_manifest",
        "manifest_kind": "validation_scale_package_manifest",
        "run_root": str(run_root),
        "validation_scale_package_manifest_decision": "PASS" if decision_ready else "FAIL",
        "claim_support_status": "validation_scale_package_ready_not_effect_size_claim"
        if decision_ready
        else "validation_scale_package_blocked",
        "validation_scale_gate_decision": validation_gate.get("validation_scale_gate_decision"),
        "paper_result_level": validation_gate.get("paper_result_level"),
        "target_fpr": validation_gate.get("target_fpr"),
        "external_baseline_self_containment_decision": self_containment.get("external_baseline_self_containment_decision"),
        "data_split_and_leakage_guard_decision": data_guard.get("data_split_and_leakage_guard_decision"),
        "validation_scale_to_pilot_paper_transition_decision": transition.get("validation_scale_to_pilot_paper_transition_decision"),
        "required_artifact_count": len(inventory),
        "present_artifact_count": sum(1 for row in inventory if row["artifact_exists"]),
        "missing_artifact_count": len(missing),
        "missing_artifact_relpaths": missing,
        "artifact_inventory": inventory,
    }


def write_validation_scale_package_manifest(run_root: str | Path) -> dict[str, Any]:
    """写出 validation_scale package manifest 和简短报告。"""
    run_root = Path(run_root)
    manifest = build_validation_scale_package_manifest(run_root)
    write_json(run_root / "manifests" / "validation_scale_package_manifest.json", manifest)
    report = (
        "# Validation-scale Package Manifest Report\n\n"
        "该报告由 validation_scale package manifest 自动派生, 用于确认门禁产物是否齐全。"
        "它不包含人工填写的效果结论。\n\n"
        f"- validation_scale_package_manifest_decision: {manifest['validation_scale_package_manifest_decision']}\n"
        f"- validation_scale_gate_decision: {manifest['validation_scale_gate_decision']}\n"
        f"- paper_result_level: {manifest['paper_result_level']}\n"
        f"- target_fpr: {manifest['target_fpr']}\n"
        f"- external_baseline_self_containment_decision: {manifest['external_baseline_self_containment_decision']}\n"
        f"- data_split_and_leakage_guard_decision: {manifest['data_split_and_leakage_guard_decision']}\n"
        f"- validation_scale_to_pilot_paper_transition_decision: {manifest['validation_scale_to_pilot_paper_transition_decision']}\n"
        f"- missing_artifact_relpaths: {', '.join(manifest['missing_artifact_relpaths']) if manifest['missing_artifact_relpaths'] else 'none'}\n"
    )
    report_path = run_root / "reports" / "validation_scale_package_manifest_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return manifest


def write_validation_scale_artifact_package(run_root: str | Path) -> dict[str, Any]:
    """同时写出 validation_scale 诊断图和 package manifest。"""
    figure = write_validation_scale_gate_figure(run_root)
    manifest = write_validation_scale_package_manifest(run_root)
    return {
        "validation_scale_gate_figure": figure,
        "validation_scale_package_manifest": manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="重建 validation_scale 诊断图与 package manifest。")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--mode", choices=("figure", "manifest", "all"), default="all")
    args = parser.parse_args()
    if args.mode == "figure":
        payload = write_validation_scale_gate_figure(args.run_root)
    elif args.mode == "manifest":
        payload = write_validation_scale_package_manifest(args.run_root)
    else:
        payload = write_validation_scale_artifact_package(args.run_root)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
