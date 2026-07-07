"""full_paper 结果充分性检查器。

该脚本是 full_paper profile 的 source gate。它只读取已经落盘的 governed
records、tables、figures、reports 和 manifests, 不运行 GPU, 不人工补造结果。
通过该 gate 后, 才允许生成 `full_paper_to_submission_freeze_transition_decision`。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from main.attacks.video_runtime_attack_protocol import (
    load_protocol_config_with_shared_attack_protocol,
    required_runtime_attack_names_from_config,
)
from main.protocol.flow_evidence_fields import with_flow_evidence_protocol_defaults
from main.protocol.record_writer import write_json, write_jsonl
from main.protocol.table_builder import write_csv


DEFAULT_FULL_PAPER_CONFIG = "configs/protocol/full_paper_generative_probe.json"


def _read_json(path: Path) -> dict[str, Any]:
    """读取 JSON 对象, 文件不存在时返回空对象。"""

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
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _safe_int(value: Any) -> int:
    """把任意字段安全转换为非负整数。"""

    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> float | None:
    """把任意字段安全转换为 float。"""

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _decision_pass(payload: Mapping[str, Any], *fields: str) -> bool:
    """检查至少一个 decision 字段是否为 PASS。"""

    return any(payload.get(field) == "PASS" for field in fields)


def _target_fpr_matches(payload: Mapping[str, Any], target_fpr: float) -> bool:
    """检查 artifact 是否来自当前 full_paper target_fpr。"""

    value = _safe_float(payload.get("target_fpr"))
    return value is not None and abs(value - float(target_fpr)) <= 1e-12


def _resolve_pilot_transition(run_root: Path) -> dict[str, Any]:
    """读取 pilot_paper -> full_paper 跳转判定, 支持隔离 profile 目录。"""

    local = run_root / "artifacts" / "pilot_paper_to_full_paper_transition_decision.json"
    if local.exists():
        return _read_json(local)
    sibling = run_root.parent / "pilot_paper" / "artifacts" / "pilot_paper_to_full_paper_transition_decision.json"
    return _read_json(sibling)


def _full_generation_records(run_root: Path) -> list[dict[str, Any]]:
    """筛选 full_paper profile 的成功生成记录。"""

    return [
        record
        for record in _read_jsonl(run_root / "records" / "generation_records.jsonl")
        if record.get("generation_status") == "success"
        and record.get("colab_runtime_profile") == "full_paper"
    ]


def build_full_paper_result_checker_audit(
    run_root: str | Path,
    config_path: str | Path = DEFAULT_FULL_PAPER_CONFIG,
) -> dict[str, Any]:
    """构建 full_paper 结果充分性审计。"""

    run_root = Path(run_root)
    config = load_protocol_config_with_shared_attack_protocol(config_path)
    if "target_fpr" not in config:
        raise KeyError(f"full_paper protocol config 缺少 target_fpr: {config_path}")
    target_fpr = float(config["target_fpr"])
    generation_records = _full_generation_records(run_root)
    fair = _read_json(run_root / "artifacts" / "fair_detection_calibration_decision.json")
    formal = _read_json(run_root / "artifacts" / "formal_method_baseline_comparison_decision.json")
    interval = _read_json(run_root / "artifacts" / "formal_baseline_difference_interval_decision.json")
    self_containment = _read_json(run_root / "artifacts" / "external_baseline_self_containment_decision.json")
    data_guard = _read_json(run_root / "artifacts" / "data_split_and_leakage_guard_decision.json")
    runtime_attack = _read_json(run_root / "artifacts" / "runtime_attack_decision.json")
    runtime_detection = _read_json(run_root / "artifacts" / "runtime_detection_decision.json")
    low_fpr = _read_json(run_root / "artifacts" / "low_fpr_formal_statistics_decision.json")
    skeleton = _read_json(run_root / "artifacts" / "paper_result_artifact_skeleton_decision.json")
    pilot_transition = _resolve_pilot_transition(run_root)
    required_attack_names = list(required_runtime_attack_names_from_config(config))
    runtime_missing = runtime_attack.get("runtime_attack_missing_required_names")
    if not isinstance(runtime_missing, list):
        runtime_missing = []
    detection_missing = runtime_detection.get("runtime_detection_missing_required_names")
    if not isinstance(detection_missing, list):
        detection_missing = []

    checks = {
        "paper_result_level_is_full_paper": config.get("paper_result_level") == "full_paper",
        "pilot_paper_to_full_paper_transition_passed": pilot_transition.get("pilot_paper_to_full_paper_transition_decision") == "PASS",
        "full_paper_generation_sample_count_ready": len(generation_records) >= _safe_int(config.get("minimum_unique_video_count")),
        "runtime_attack_decision_passed": _decision_pass(runtime_attack, "runtime_attack_decision"),
        "runtime_detection_decision_passed": _decision_pass(runtime_detection, "runtime_detection_decision"),
        "runtime_attack_required_names_ready": not runtime_missing and _safe_int(runtime_attack.get("runtime_attack_ready_count")) >= len(required_attack_names),
        "runtime_detection_required_names_ready": not detection_missing and _safe_int(runtime_detection.get("runtime_detection_ready_count")) >= len(required_attack_names),
        "external_baseline_self_containment_passed": self_containment.get("external_baseline_self_containment_decision") == "PASS",
        "fair_detection_calibration_passed": fair.get("fair_detection_calibration_decision") == "PASS" and _target_fpr_matches(fair, target_fpr),
        "formal_method_baseline_comparison_passed": formal.get("formal_method_baseline_comparison_decision") == "PASS" and _target_fpr_matches(formal, target_fpr),
        "formal_baseline_difference_interval_passed": interval.get("formal_baseline_difference_interval_decision") == "PASS" and _target_fpr_matches(interval, target_fpr),
        "low_fpr_current_profile_statistics_ready": low_fpr.get("low_fpr_formal_statistics_decision") == "PASS" and low_fpr.get("current_profile_low_fpr_claim_allowed") is True,
        "paper_result_artifact_skeleton_passed": skeleton.get("paper_result_artifact_skeleton_decision") == "PASS" and _target_fpr_matches(skeleton, target_fpr),
        "data_split_and_leakage_guard_passed": data_guard.get("data_split_and_leakage_guard_decision") == "PASS",
    }
    missing = [name for name, passed in checks.items() if not passed]
    decision = "PASS" if not missing else "FAIL"
    return {
        "stage_id": "full_paper_result_checker",
        "run_root": str(run_root),
        "full_paper_result_checker_decision": decision,
        "full_paper_result_decision": decision,
        "paper_result_level": config.get("paper_result_level"),
        "target_fpr": target_fpr,
        "full_paper_claim_allowed": decision == "PASS",
        "submission_freeze_allowed": False,
        "missing_full_paper_requirements": missing,
        "full_paper_missing_requirement_count": len(missing),
        "full_paper_generation_record_count": len(generation_records),
        "minimum_unique_video_count": _safe_int(config.get("minimum_unique_video_count")),
        "required_runtime_attack_count": len(required_attack_names),
        "runtime_attack_missing_required_names": runtime_missing,
        "runtime_detection_missing_required_names": detection_missing,
        "pilot_paper_to_full_paper_transition_decision": pilot_transition.get("pilot_paper_to_full_paper_transition_decision"),
        "fair_detection_calibration_decision": fair.get("fair_detection_calibration_decision"),
        "formal_method_baseline_comparison_decision": formal.get("formal_method_baseline_comparison_decision"),
        "formal_baseline_difference_interval_decision": interval.get("formal_baseline_difference_interval_decision"),
        "external_baseline_self_containment_decision": self_containment.get("external_baseline_self_containment_decision"),
        "low_fpr_formal_statistics_decision": low_fpr.get("low_fpr_formal_statistics_decision"),
        "paper_result_artifact_skeleton_decision": skeleton.get("paper_result_artifact_skeleton_decision"),
        "data_split_and_leakage_guard_decision": data_guard.get("data_split_and_leakage_guard_decision"),
        "claim_support_status": "full_paper_claim_ready" if decision == "PASS" else "full_paper_result_blocked",
    }


def write_full_paper_result_checker_audit(
    run_root: str | Path,
    config_path: str | Path = DEFAULT_FULL_PAPER_CONFIG,
) -> dict[str, Any]:
    """写出 full_paper result checker 的 records、table、decision 和 report。"""

    run_root = Path(run_root)
    audit = build_full_paper_result_checker_audit(run_root, config_path)
    record = with_flow_evidence_protocol_defaults(
        {"record_version": "full_paper_result_checker_v1", **audit},
        trajectory_source_level="full_paper_result_checker_aggregated_records",
        flow_state_admissibility_status="full_paper_ready" if audit["full_paper_result_checker_decision"] == "PASS" else "full_paper_blocked",
        claim_support_status=audit["claim_support_status"],
    )
    write_jsonl(run_root / "records" / "full_paper_result_checker_records.jsonl", [record])
    write_csv(run_root / "tables" / "full_paper_result_checker_table.csv", [record])
    write_json(run_root / "artifacts" / "full_paper_result_checker_decision.json", audit)
    write_json(run_root / "artifacts" / "full_paper_result_decision.json", audit)
    report = (
        "# Full-paper Result Checker Report\n\n"
        "该报告只读取 full_paper profile 已有 governed artifacts, 不运行实验, 不补造分数。"
        "通过后才允许进入 full_paper -> submission_freeze 跳转判定。\n\n"
        f"- full_paper_result_checker_decision: {audit['full_paper_result_checker_decision']}\n"
        f"- target_fpr: {audit['target_fpr']}\n"
        f"- full_paper_claim_allowed: {str(audit['full_paper_claim_allowed']).lower()}\n"
        f"- missing_full_paper_requirements: {', '.join(audit['missing_full_paper_requirements']) if audit['missing_full_paper_requirements'] else 'none'}\n"
        f"- paper_result_artifact_skeleton_decision: {audit['paper_result_artifact_skeleton_decision']}\n"
    )
    report_path = run_root / "reports" / "full_paper_result_checker_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 full_paper 结果包是否满足投稿级主张前置。")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--config-path", default=DEFAULT_FULL_PAPER_CONFIG)
    parser.add_argument("--write-outputs", action="store_true")
    args = parser.parse_args()
    payload = (
        write_full_paper_result_checker_audit(args.run_root, args.config_path)
        if args.write_outputs
        else build_full_paper_result_checker_audit(args.run_root, args.config_path)
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
