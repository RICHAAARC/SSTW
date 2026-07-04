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

from experiments.generative_video_model_probe.validation_artifact_rebuild import (
    REQUIRED_REBUILD_INPUTS,
    REQUIRED_REBUILD_OUTPUTS,
)
from main.protocol.record_writer import write_json


VALIDATION_SCALE_GATE_PACKAGE_RELPATHS = (
    "records/validation_scale_gate_records.jsonl",
    "tables/validation_scale_gate_table.csv",
    "artifacts/validation_scale_gate_decision.json",
    "reports/validation_scale_gate_report.md",
    "artifacts/external_baseline_self_containment_decision.json",
    "artifacts/validation_artifact_rebuild_dry_run_decision.json",
    "artifacts/data_split_and_leakage_guard_decision.json",
    "artifacts/validation_scale_to_pilot_paper_transition_decision.json",
    "figures/validation_scale_gate_figure.json",
)
VALIDATION_SCALE_REQUIRED_PACKAGE_RELPATHS = tuple(dict.fromkeys((
    *REQUIRED_REBUILD_INPUTS,
    *REQUIRED_REBUILD_OUTPUTS,
    *VALIDATION_SCALE_GATE_PACKAGE_RELPATHS,
)))


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
        "validation_generation_records_ready",
        "validation_motion_threshold_calibration_ready",
        "validation_formal_motion_claim_ready",
        "validation_motion_consistency_exclusion_report_ready",
        "validation_attack_records_ready",
        "validation_detection_records_ready",
        "validation_external_baseline_status_records_ready",
        "validation_external_baseline_comparison_records_ready",
        "validation_external_baseline_self_containment_ready",
        "validation_sstw_measured_formal_records_ready",
        "validation_fair_detection_calibration_ready",
        "validation_formal_method_baseline_comparison_ready",
        "validation_formal_baseline_difference_interval_ready",
        "validation_scale_formal_internal_ablation_ready",
        "validation_low_fpr_formal_statistics_blocking_record_ready",
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
    motion_exclusion = _read_json(run_root / "artifacts" / "motion_consistency_exclusion_decision.json")
    self_containment = _read_json(run_root / "artifacts" / "external_baseline_self_containment_decision.json")
    sstw_formal = _read_json(run_root / "artifacts" / "sstw_measured_formal_decision.json")
    fair_calibration = _read_json(run_root / "artifacts" / "fair_detection_calibration_decision.json")
    formal_comparison = _read_json(run_root / "artifacts" / "formal_method_baseline_comparison_decision.json")
    difference_interval = _read_json(run_root / "artifacts" / "formal_baseline_difference_interval_decision.json")
    formal_ablation = _read_json(run_root / "artifacts" / "validation_scale_formal_internal_ablation_decision.json")
    low_fpr = _read_json(run_root / "artifacts" / "low_fpr_formal_statistics_decision.json")
    data_guard = _read_json(run_root / "artifacts" / "data_split_and_leakage_guard_decision.json")
    transition = _read_json(run_root / "artifacts" / "validation_scale_to_pilot_paper_transition_decision.json")
    decision_ready = (
        validation_gate.get("validation_scale_gate_decision") == "PASS"
        and motion_exclusion.get("motion_consistency_exclusion_decision") == "PASS"
        and self_containment.get("external_baseline_self_containment_decision") == "PASS"
        and sstw_formal.get("sstw_measured_formal_decision") == "PASS"
        and fair_calibration.get("fair_detection_calibration_decision") == "PASS"
        and formal_comparison.get("formal_method_baseline_comparison_decision") == "PASS"
        and difference_interval.get("formal_baseline_difference_interval_decision") == "PASS"
        and formal_ablation.get("validation_scale_formal_internal_ablation_decision") == "PASS"
        and low_fpr.get("low_fpr_formal_statistics_decision") == "PASS"
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
        "motion_consistency_exclusion_decision": motion_exclusion.get("motion_consistency_exclusion_decision"),
        "external_baseline_self_containment_decision": self_containment.get("external_baseline_self_containment_decision"),
        "sstw_measured_formal_decision": sstw_formal.get("sstw_measured_formal_decision"),
        "fair_detection_calibration_decision": fair_calibration.get("fair_detection_calibration_decision"),
        "formal_method_baseline_comparison_decision": formal_comparison.get("formal_method_baseline_comparison_decision"),
        "formal_baseline_difference_interval_decision": difference_interval.get("formal_baseline_difference_interval_decision"),
        "validation_scale_formal_internal_ablation_decision": formal_ablation.get("validation_scale_formal_internal_ablation_decision"),
        "low_fpr_formal_statistics_decision": low_fpr.get("low_fpr_formal_statistics_decision"),
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
        f"- motion_consistency_exclusion_decision: {manifest['motion_consistency_exclusion_decision']}\n"
        f"- external_baseline_self_containment_decision: {manifest['external_baseline_self_containment_decision']}\n"
        f"- sstw_measured_formal_decision: {manifest['sstw_measured_formal_decision']}\n"
        f"- fair_detection_calibration_decision: {manifest['fair_detection_calibration_decision']}\n"
        f"- formal_method_baseline_comparison_decision: {manifest['formal_method_baseline_comparison_decision']}\n"
        f"- formal_baseline_difference_interval_decision: {manifest['formal_baseline_difference_interval_decision']}\n"
        f"- validation_scale_formal_internal_ablation_decision: {manifest['validation_scale_formal_internal_ablation_decision']}\n"
        f"- low_fpr_formal_statistics_decision: {manifest['low_fpr_formal_statistics_decision']}\n"
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
