"""重建 paper_profile 诊断图与 package manifest。

该模块只从 run_root 中已经存在的 governed records、tables、reports 和 decision
artifacts 生成派生产物。它只记录已由门禁审计过的 target_fpr=0.1 结论状态,
不手工补造缺失的实验结果。
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
from experiments.generative_video_model_probe.paper_result_artifact_builders import (
    PAPER_RESULT_ARTIFACT_RELPATHS,
)
from evaluation.protocol.record_writer import write_json


PAPER_PROFILE_SHARED_PACKAGE_RELPATHS = (
    "artifacts/external_baseline_self_containment_decision.json",
    "artifacts/validation_artifact_rebuild_dry_run_decision.json",
    "artifacts/data_split_and_leakage_guard_decision.json",
    "artifacts/paper_result_artifact_skeleton_decision.json",
    "records/reviewer_evidence_index_records.jsonl",
    "tables/reviewer_evidence_index_table.csv",
    "artifacts/reviewer_evidence_index_decision.json",
    "reports/reviewer_evidence_index.json",
    "reports/reviewer_evidence_index.md",
)


def _paper_profile_from_decision(decision: Mapping[str, Any]) -> str:
    """从 gate decision 中解析当前 paper profile。"""

    level = str(decision.get("paper_result_level") or "").strip()
    return level if level else "probe_paper"


def _gate_decision_for_current_profile(run_root: Path) -> tuple[str, Path, dict[str, Any]]:
    """读取当前主干 paper profile 的 gate decision。

    主干移除 paper_profile 后, probe_paper 是默认入口。为了兼容旧结果包,
    若不存在 profile 专属 gate, 才回退读取 paper_profile 旧文件。
    """

    for candidate in (
        run_root / "artifacts" / "probe_paper_gate_decision.json",
        run_root / "artifacts" / "pilot_paper_gate_decision.json",
        run_root / "artifacts" / "full_paper_result_checker_decision.json",
        run_root / "artifacts" / "paper_profile_gate_decision.json",
    ):
        payload = _read_json(candidate)
        if payload:
            return _paper_profile_from_decision(payload), candidate, payload
    return "probe_paper", run_root / "artifacts" / "probe_paper_gate_decision.json", {}


def _paper_profile_gate_package_relpaths(profile: str) -> tuple[str, ...]:
    """构造当前 paper profile 的 package manifest 必需文件列表。"""

    transition_relpaths = {
        "probe_paper": ("artifacts/probe_paper_to_pilot_paper_transition_decision.json",),
        "pilot_paper": ("artifacts/pilot_paper_to_full_paper_transition_decision.json",),
        "full_paper": ("artifacts/full_paper_to_submission_freeze_transition_decision.json",),
    }.get(profile, ())
    gate_relpaths = (
        f"records/{profile}_gate_records.jsonl",
        f"tables/{profile}_gate_table.csv",
        f"artifacts/{profile}_gate_decision.json",
        f"reports/{profile}_gate_report.md",
        f"figures/{profile}_gate_figure.json",
    )
    return tuple(dict.fromkeys((
        *REQUIRED_REBUILD_INPUTS,
        *REQUIRED_REBUILD_OUTPUTS,
        *PAPER_RESULT_ARTIFACT_RELPATHS,
        *PAPER_PROFILE_SHARED_PACKAGE_RELPATHS,
        *gate_relpaths,
        *transition_relpaths,
    )))


# 兼容旧测试与旧导入名。主干语义已经切换为 probe_paper。
PAPER_PROFILE_REQUIRED_PACKAGE_RELPATHS = _paper_profile_gate_package_relpaths("probe_paper")


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
    """把 paper_profile gate 的 requirement 列表转成图数据行。"""
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
        "paper_profile_sstw_advantage_claim_ready",
        "formal_internal_ablation_summary_ready",
        "validation_low_fpr_formal_statistics_blocking_record_ready",
        "validation_paper_result_artifact_skeleton_ready",
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
            "requirement_status": (
                "NOT_REQUIRED"
                if name == "paper_profile_sstw_advantage_claim_ready"
                and decision.get("paper_result_level") == "paper_profile"
                else ("FAIL" if name in missing else "PASS")
            ),
            "requirement_ready_value": (
                0
                if name in missing
                else (0 if name == "paper_profile_sstw_advantage_claim_ready" and decision.get("paper_result_level") == "paper_profile" else 1)
            ),
        }
        for name in observed
    ]


def build_paper_profile_gate_figure(run_root: str | Path) -> dict[str, Any]:
    """构造当前 paper profile gate 诊断图 manifest。"""
    run_root = Path(run_root)
    profile, decision_path, decision = _gate_decision_for_current_profile(run_root)
    rows = _requirement_rows(decision)
    return {
        "artifact_name": f"{profile}_gate_figure.json",
        "artifact_type": "figure_manifest",
        "figure_id": f"{profile}_gate_figure",
        "figure_title": f"{profile} gate requirement readiness",
        "run_root": str(run_root),
        "source_artifact_paths": [str(decision_path)],
        "paper_profile_gate_decision": decision.get("paper_profile_gate_decision", "missing"),
        f"{profile}_gate_decision": decision.get(f"{profile}_gate_decision", decision.get("paper_profile_gate_decision", "missing")),
        "paper_result_level": decision.get("paper_result_level"),
        "target_fpr": decision.get("target_fpr"),
        "claim_support_status": decision.get("claim_support_status", f"{profile}_diagnostic_figure_blocked"),
        "encoding": {
            "x": "requirement_name",
            "y": "requirement_ready_value",
            "color": "requirement_status",
        },
        "figure_rows": rows,
    }


def write_paper_profile_gate_figure(run_root: str | Path) -> dict[str, Any]:
    """写出当前 paper profile gate 诊断图 manifest。"""
    run_root = Path(run_root)
    figure = build_paper_profile_gate_figure(run_root)
    write_json(run_root / "figures" / str(figure["artifact_name"]), figure)
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


def build_paper_profile_package_manifest(run_root: str | Path) -> dict[str, Any]:
    """构建当前 paper profile package manifest。"""
    run_root = Path(run_root)
    profile, gate_path, profile_gate = _gate_decision_for_current_profile(run_root)
    inventory = _inventory_rows(run_root, _paper_profile_gate_package_relpaths(profile))
    missing = [row["artifact_relpath"] for row in inventory if not row["artifact_exists"]]
    motion_exclusion = _read_json(run_root / "artifacts" / "motion_consistency_exclusion_decision.json")
    self_containment = _read_json(run_root / "artifacts" / "external_baseline_self_containment_decision.json")
    sstw_formal = _read_json(run_root / "artifacts" / "sstw_measured_formal_decision.json")
    fair_calibration = _read_json(run_root / "artifacts" / "fair_detection_calibration_decision.json")
    formal_comparison = _read_json(run_root / "artifacts" / "formal_method_baseline_comparison_decision.json")
    difference_interval = _read_json(run_root / "artifacts" / "formal_baseline_difference_interval_decision.json")
    formal_ablation = _read_json(run_root / "artifacts" / "formal_internal_ablation_summary_decision.json")
    low_fpr = _read_json(run_root / "artifacts" / "low_fpr_formal_statistics_decision.json")
    data_guard = _read_json(run_root / "artifacts" / "data_split_and_leakage_guard_decision.json")
    paper_skeleton = _read_json(run_root / "artifacts" / "paper_result_artifact_skeleton_decision.json")
    transition_field = {
        "probe_paper": "probe_paper_to_pilot_paper_transition_decision",
        "pilot_paper": "pilot_paper_to_full_paper_transition_decision",
        "full_paper": "full_paper_to_submission_freeze_transition_decision",
    }.get(profile, "")
    transition = _read_json(run_root / "artifacts" / f"{transition_field}.json") if transition_field else {}
    gate_field = f"{profile}_gate_decision"
    manifest_field = f"{profile}_package_manifest_decision"
    decision_ready = (
        profile_gate.get(gate_field, profile_gate.get("paper_profile_gate_decision")) == "PASS"
        and motion_exclusion.get("motion_consistency_exclusion_decision") == "PASS"
        and self_containment.get("external_baseline_self_containment_decision") == "PASS"
        and sstw_formal.get("sstw_measured_formal_decision") == "PASS"
        and fair_calibration.get("fair_detection_calibration_decision") == "PASS"
        and formal_comparison.get("formal_method_baseline_comparison_decision") == "PASS"
        and difference_interval.get("formal_baseline_difference_interval_decision") == "PASS"
        and formal_ablation.get("formal_internal_ablation_summary_decision") == "PASS"
        and low_fpr.get("low_fpr_formal_statistics_decision") == "PASS"
        and paper_skeleton.get("paper_result_artifact_skeleton_decision") == "PASS"
        and data_guard.get("data_split_and_leakage_guard_decision") == "PASS"
        and (not transition_field or transition.get(transition_field) == "PASS")
        and not missing
    )
    return {
        "artifact_name": f"{profile}_package_manifest.json",
        "artifact_type": "package_manifest",
        "manifest_kind": f"{profile}_package_manifest",
        "run_root": str(run_root),
        manifest_field: "PASS" if decision_ready else "FAIL",
        "claim_support_status": profile_gate.get("claim_support_status", f"{profile}_package_blocked")
        if decision_ready
        else f"{profile}_package_blocked",
        "paper_claim_id": profile_gate.get("paper_claim_id"),
        "paper_claim_level": profile_gate.get("paper_claim_level"),
        "paper_claim_support_status": profile_gate.get("paper_claim_support_status"),
        "paper_result_formality_guard_decision": profile_gate.get("paper_result_formality_guard_decision"),
        "paper_result_formality_guard_violation_count": profile_gate.get("paper_result_formality_guard_violation_count"),
        "paper_profile": profile,
        "paper_profile_gate_decision_path": str(gate_path),
        "paper_profile_gate_decision": profile_gate.get("paper_profile_gate_decision"),
        gate_field: profile_gate.get(gate_field, profile_gate.get("paper_profile_gate_decision")),
        "paper_result_level": profile_gate.get("paper_result_level"),
        "target_fpr": profile_gate.get("target_fpr"),
        "motion_consistency_exclusion_decision": motion_exclusion.get("motion_consistency_exclusion_decision"),
        "external_baseline_self_containment_decision": self_containment.get("external_baseline_self_containment_decision"),
        "sstw_measured_formal_decision": sstw_formal.get("sstw_measured_formal_decision"),
        "fair_detection_calibration_decision": fair_calibration.get("fair_detection_calibration_decision"),
        "formal_method_baseline_comparison_decision": formal_comparison.get("formal_method_baseline_comparison_decision"),
        "formal_baseline_difference_interval_decision": difference_interval.get("formal_baseline_difference_interval_decision"),
        "formal_internal_ablation_summary_decision": formal_ablation.get("formal_internal_ablation_summary_decision"),
        "low_fpr_formal_statistics_decision": low_fpr.get("low_fpr_formal_statistics_decision"),
        "paper_result_artifact_skeleton_decision": paper_skeleton.get("paper_result_artifact_skeleton_decision"),
        "data_split_and_leakage_guard_decision": data_guard.get("data_split_and_leakage_guard_decision"),
        transition_field: transition.get(transition_field) if transition_field else None,
        "required_artifact_count": len(inventory),
        "present_artifact_count": sum(1 for row in inventory if row["artifact_exists"]),
        "missing_artifact_count": len(missing),
        "missing_artifact_relpaths": missing,
        "artifact_inventory": inventory,
    }


def write_paper_profile_package_manifest(run_root: str | Path) -> dict[str, Any]:
    """写出当前 paper profile package manifest 和简短报告。"""
    run_root = Path(run_root)
    manifest = build_paper_profile_package_manifest(run_root)
    profile = str(manifest.get("paper_profile") or "probe_paper")
    manifest_field = f"{profile}_package_manifest_decision"
    transition_field = {
        "probe_paper": "probe_paper_to_pilot_paper_transition_decision",
        "pilot_paper": "pilot_paper_to_full_paper_transition_decision",
        "full_paper": "full_paper_to_submission_freeze_transition_decision",
    }.get(profile, "")
    write_json(run_root / "manifests" / str(manifest["artifact_name"]), manifest)
    report = (
        f"# {profile} Package Manifest Report\n\n"
        "该报告由 paper profile package manifest 自动派生, 用于确认门禁产物是否齐全。"
        "其中 claim_support_status 只能来自已通过的 governed gate, 不能人工填写效果结论。\n\n"
        f"- {manifest_field}: {manifest[manifest_field]}\n"
        f"- {profile}_gate_decision: {manifest.get(f'{profile}_gate_decision')}\n"
        f"- paper_result_level: {manifest['paper_result_level']}\n"
        f"- target_fpr: {manifest['target_fpr']}\n"
        f"- motion_consistency_exclusion_decision: {manifest['motion_consistency_exclusion_decision']}\n"
        f"- external_baseline_self_containment_decision: {manifest['external_baseline_self_containment_decision']}\n"
        f"- sstw_measured_formal_decision: {manifest['sstw_measured_formal_decision']}\n"
        f"- fair_detection_calibration_decision: {manifest['fair_detection_calibration_decision']}\n"
        f"- formal_method_baseline_comparison_decision: {manifest['formal_method_baseline_comparison_decision']}\n"
        f"- formal_baseline_difference_interval_decision: {manifest['formal_baseline_difference_interval_decision']}\n"
        f"- formal_internal_ablation_summary_decision: {manifest['formal_internal_ablation_summary_decision']}\n"
        f"- low_fpr_formal_statistics_decision: {manifest['low_fpr_formal_statistics_decision']}\n"
        f"- paper_result_artifact_skeleton_decision: {manifest['paper_result_artifact_skeleton_decision']}\n"
        f"- data_split_and_leakage_guard_decision: {manifest['data_split_and_leakage_guard_decision']}\n"
        + (f"- {transition_field}: {manifest.get(transition_field)}\n" if transition_field else "")
        + f"- missing_artifact_relpaths: {', '.join(manifest['missing_artifact_relpaths']) if manifest['missing_artifact_relpaths'] else 'none'}\n"
    )
    report_path = run_root / "reports" / f"{profile}_package_manifest_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return manifest


def write_paper_profile_artifact_package(run_root: str | Path) -> dict[str, Any]:
    """同时写出当前 paper profile 诊断图和 package manifest。"""
    figure = write_paper_profile_gate_figure(run_root)
    manifest = write_paper_profile_package_manifest(run_root)
    return {
        "paper_profile_gate_figure": figure,
        "paper_profile_package_manifest": manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="重建当前 paper profile 诊断图与 package manifest。")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--mode", choices=("figure", "manifest", "all"), default="all")
    args = parser.parse_args()
    if args.mode == "figure":
        payload = write_paper_profile_gate_figure(args.run_root)
    elif args.mode == "manifest":
        payload = write_paper_profile_package_manifest(args.run_root)
    else:
        payload = write_paper_profile_artifact_package(args.run_root)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
