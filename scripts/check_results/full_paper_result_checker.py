"""full_paper 结果充分性检查器。

该脚本是 full_paper profile 的 source gate。它只读取已经落盘的 governed
records、tables、figures、reports 和 manifests, 不运行 GPU, 不人工补造结果。
通过该 gate 后, 才允许生成 `full_paper_to_submission_freeze_transition_decision`。
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
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


def _identity_key(record: Mapping[str, Any]) -> str:
    """构造跨 generation / attack / detection records 的视频身份键。

    该函数属于通用工程写法, 用于防止 full_paper checker 只按总数通过,
    却没有确认 runtime attack 和 detection 记录确实来自本次 full_paper 生成单元。
    """

    return "::".join(
        str(record.get(field) or "")
        for field in ("generation_model_id", "prompt_id", "seed_id", "trajectory_trace_id")
    )


def _identity_keys(records: list[dict[str, Any]]) -> set[str]:
    """提取非空视频身份键集合。"""

    return {_identity_key(record) for record in records if record.get("prompt_id") and record.get("seed_id")}


def _records_in_keys(records: list[dict[str, Any]], keys: set[str]) -> list[dict[str, Any]]:
    """筛选属于指定视频身份集合的 records。"""

    return [record for record in records if _identity_key(record) in keys]


def _records_by_split(records: list[dict[str, Any]], split_name: str) -> list[dict[str, Any]]:
    """按 split 字段筛选 records。"""

    return [record for record in records if record.get("split") == split_name]


def _unique_nonempty(records: list[dict[str, Any]], field_name: str) -> set[str]:
    """提取指定字段的非空唯一值集合。"""

    return {str(record[field_name]) for record in records if record.get(field_name) not in {None, ""}}


def _seed_per_prompt_min(records: list[dict[str, Any]]) -> int:
    """计算每个 prompt 覆盖 seed 数量的最小值。"""

    by_prompt: dict[str, set[str]] = defaultdict(set)
    for record in records:
        prompt_id = record.get("prompt_id")
        seed_id = record.get("seed_id")
        if prompt_id and seed_id:
            by_prompt[str(prompt_id)].add(str(seed_id))
    return min((len(seed_ids) for seed_ids in by_prompt.values()), default=0)


def _ready_attack_counts(records: list[dict[str, Any]], status_field: str, ready_value: str) -> dict[str, int]:
    """统计每个 attack 的 ready 事件数。"""

    counts: dict[str, int] = defaultdict(int)
    for record in records:
        attack_name = record.get("attack_name")
        if attack_name and record.get(status_field) == ready_value:
            counts[str(attack_name)] += 1
    return dict(counts)


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
    full_generation_keys = _identity_keys(generation_records)
    calibration_generation_records = _records_by_split(generation_records, "calibration")
    test_generation_records = _records_by_split(generation_records, "test")
    runtime_attack_records = _records_in_keys(
        _read_jsonl(run_root / "records" / "runtime_attack_records.jsonl"),
        full_generation_keys,
    )
    runtime_detection_records = _records_in_keys(
        _read_jsonl(run_root / "records" / "runtime_detection_records.jsonl"),
        full_generation_keys,
    )
    fair = _read_json(run_root / "artifacts" / "fair_detection_calibration_decision.json")
    formal = _read_json(run_root / "artifacts" / "formal_method_baseline_comparison_decision.json")
    interval = _read_json(run_root / "artifacts" / "formal_baseline_difference_interval_decision.json")
    self_containment = _read_json(run_root / "artifacts" / "external_baseline_self_containment_decision.json")
    data_guard = _read_json(run_root / "artifacts" / "data_split_and_leakage_guard_decision.json")
    runtime_attack = _read_json(run_root / "artifacts" / "runtime_attack_decision.json")
    runtime_detection = _read_json(run_root / "artifacts" / "runtime_detection_decision.json")
    low_fpr = _read_json(run_root / "artifacts" / "low_fpr_formal_statistics_decision.json")
    skeleton = _read_json(run_root / "artifacts" / "paper_result_artifact_skeleton_decision.json")
    statistical_ci = _read_json(run_root / "artifacts" / "statistical_confidence_interval_decision.json")
    artifact_rebuild = _read_json(run_root / "artifacts" / "validation_artifact_rebuild_dry_run_decision.json")
    pilot_transition = _resolve_pilot_transition(run_root)
    required_attack_names = list(required_runtime_attack_names_from_config(config))
    runtime_missing = runtime_attack.get("runtime_attack_missing_required_names")
    if not isinstance(runtime_missing, list):
        runtime_missing = []
    detection_missing = runtime_detection.get("runtime_detection_missing_required_names")
    if not isinstance(detection_missing, list):
        detection_missing = []
    prompt_count = len(_unique_nonempty(generation_records, "prompt_id"))
    seed_per_prompt_min = _seed_per_prompt_min(generation_records)
    calibration_seed_per_prompt_min = _seed_per_prompt_min(calibration_generation_records)
    test_seed_per_prompt_min = _seed_per_prompt_min(test_generation_records)
    unique_video_count = len(full_generation_keys)
    calibration_unique_video_count = len(_identity_keys(calibration_generation_records))
    test_unique_video_count = len(_identity_keys(test_generation_records))
    runtime_attack_event_counts = _ready_attack_counts(
        runtime_attack_records,
        status_field="attack_runtime_status",
        ready_value="ready",
    )
    runtime_detection_event_counts = _ready_attack_counts(
        runtime_detection_records,
        status_field="runtime_detection_status",
        ready_value="ready",
    )
    required_attack_set = {str(name) for name in required_attack_names if str(name)}
    runtime_attack_event_count_per_attack_min = min(
        (runtime_attack_event_counts.get(name, 0) for name in required_attack_set),
        default=0,
    )
    runtime_detection_event_count_per_attack_min = min(
        (runtime_detection_event_counts.get(name, 0) for name in required_attack_set),
        default=0,
    )

    checks = {
        "paper_result_level_is_full_paper": config.get("paper_result_level") == "full_paper",
        "pilot_paper_to_full_paper_transition_passed": pilot_transition.get("pilot_paper_to_full_paper_transition_decision") == "PASS",
        "full_paper_generation_sample_count_ready": len(generation_records) >= _safe_int(config.get("minimum_unique_video_count")),
        "full_paper_generation_prompt_seed_grid_ready": prompt_count >= _safe_int(config.get("minimum_prompt_count"))
        and seed_per_prompt_min >= _safe_int(config.get("minimum_seed_per_prompt"))
        and unique_video_count >= _safe_int(config.get("minimum_unique_video_count")),
        "full_paper_calibration_split_ready": calibration_seed_per_prompt_min >= _safe_int(config.get("minimum_calibration_seed_per_prompt"))
        and calibration_unique_video_count >= _safe_int(config.get("minimum_calibration_unique_video_count")),
        "full_paper_heldout_test_split_ready": test_seed_per_prompt_min >= _safe_int(config.get("minimum_test_seed_per_prompt"))
        and test_unique_video_count >= _safe_int(config.get("minimum_test_unique_video_count")),
        "runtime_attack_decision_passed": _decision_pass(runtime_attack, "runtime_attack_decision"),
        "runtime_detection_decision_passed": _decision_pass(runtime_detection, "runtime_detection_decision"),
        "runtime_attack_required_names_ready": not runtime_missing and _safe_int(runtime_attack.get("runtime_attack_ready_count")) >= len(required_attack_names),
        "runtime_detection_required_names_ready": not detection_missing and _safe_int(runtime_detection.get("runtime_detection_ready_count")) >= len(required_attack_names),
        "full_paper_runtime_attack_event_coverage_ready": set(runtime_attack_event_counts) >= required_attack_set
        and runtime_attack_event_count_per_attack_min >= _safe_int(config.get("minimum_attack_event_count_per_attack")),
        "full_paper_runtime_detection_event_coverage_ready": set(runtime_detection_event_counts) >= required_attack_set
        and runtime_detection_event_count_per_attack_min >= _safe_int(config.get("minimum_attack_event_count_per_attack")),
        "external_baseline_self_containment_passed": self_containment.get("external_baseline_self_containment_decision") == "PASS",
        "fair_detection_calibration_passed": fair.get("fair_detection_calibration_decision") == "PASS" and _target_fpr_matches(fair, target_fpr),
        "formal_method_baseline_comparison_passed": formal.get("formal_method_baseline_comparison_decision") == "PASS" and _target_fpr_matches(formal, target_fpr),
        "formal_baseline_difference_interval_passed": interval.get("formal_baseline_difference_interval_decision") == "PASS" and _target_fpr_matches(interval, target_fpr),
        "low_fpr_current_profile_statistics_ready": low_fpr.get("low_fpr_formal_statistics_decision") == "PASS" and low_fpr.get("current_profile_low_fpr_claim_allowed") is True,
        "paper_result_artifact_skeleton_passed": skeleton.get("paper_result_artifact_skeleton_decision") == "PASS" and _target_fpr_matches(skeleton, target_fpr),
        "statistical_confidence_interval_passed": (not bool(config.get("require_statistical_confidence_interval_decision", True)))
        or _decision_pass(statistical_ci, "statistical_confidence_interval_decision"),
        "artifact_rebuild_dry_run_passed": (not bool(config.get("require_artifact_rebuild_dry_run", True)))
        or _decision_pass(artifact_rebuild, "validation_artifact_rebuild_dry_run_decision", "artifact_rebuild_dry_run_decision"),
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
        "full_paper_prompt_count": prompt_count,
        "full_paper_seed_per_prompt_min": seed_per_prompt_min,
        "full_paper_calibration_seed_per_prompt_min": calibration_seed_per_prompt_min,
        "full_paper_test_seed_per_prompt_min": test_seed_per_prompt_min,
        "full_paper_unique_video_count": unique_video_count,
        "full_paper_calibration_unique_video_count": calibration_unique_video_count,
        "full_paper_test_unique_video_count": test_unique_video_count,
        "full_paper_runtime_attack_event_count_per_attack_min": runtime_attack_event_count_per_attack_min,
        "full_paper_runtime_detection_event_count_per_attack_min": runtime_detection_event_count_per_attack_min,
        "full_paper_runtime_attack_event_counts": runtime_attack_event_counts,
        "full_paper_runtime_detection_event_counts": runtime_detection_event_counts,
        "minimum_unique_video_count": _safe_int(config.get("minimum_unique_video_count")),
        "minimum_calibration_seed_per_prompt": _safe_int(config.get("minimum_calibration_seed_per_prompt")),
        "minimum_test_seed_per_prompt": _safe_int(config.get("minimum_test_seed_per_prompt")),
        "minimum_calibration_unique_video_count": _safe_int(config.get("minimum_calibration_unique_video_count")),
        "minimum_test_unique_video_count": _safe_int(config.get("minimum_test_unique_video_count")),
        "minimum_attack_event_count_per_attack": _safe_int(config.get("minimum_attack_event_count_per_attack")),
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
        "statistical_confidence_interval_decision": statistical_ci.get("statistical_confidence_interval_decision"),
        "validation_artifact_rebuild_dry_run_decision": artifact_rebuild.get("validation_artifact_rebuild_dry_run_decision"),
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
        f"- full_paper_prompt_count: {audit['full_paper_prompt_count']}\n"
        f"- full_paper_unique_video_count: {audit['full_paper_unique_video_count']}\n"
        f"- full_paper_calibration_unique_video_count: {audit['full_paper_calibration_unique_video_count']}\n"
        f"- full_paper_test_unique_video_count: {audit['full_paper_test_unique_video_count']}\n"
        f"- full_paper_runtime_attack_event_count_per_attack_min: {audit['full_paper_runtime_attack_event_count_per_attack_min']}\n"
        f"- full_paper_runtime_detection_event_count_per_attack_min: {audit['full_paper_runtime_detection_event_count_per_attack_min']}\n"
        f"- statistical_confidence_interval_decision: {audit['statistical_confidence_interval_decision']}\n"
        f"- validation_artifact_rebuild_dry_run_decision: {audit['validation_artifact_rebuild_dry_run_decision']}\n"
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
