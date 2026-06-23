"""small-scale claim pilot gate 的自动审计。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from experiments.generative_video_model_probe.formal_motion_claim_filter import (
    FORMAL_MOTION_CLAIM_READY_STATUSES,
    filter_records_to_motion_claim_eligible,
    select_motion_claim_generation_records,
)
from main.protocol.flow_evidence_fields import with_flow_evidence_protocol_defaults
from main.protocol.record_writer import write_json, write_jsonl
from main.protocol.table_builder import write_csv


MIN_PROMPT_COUNT = 8
MIN_SEED_PER_PROMPT = 2
MIN_ATTACK_COUNT = 3
MIN_NEGATIVE_FAMILY_COUNT = 4
MIN_METHOD_VARIANT_COUNT = 6
HEURISTIC_MOTION_THRESHOLD_ID = "motion_delta_heuristic_v1"
HEURISTIC_MOTION_THRESHOLD_SOURCE_SPLIT = "heuristic_precalibration"


def _read_json(path: Path) -> dict:
    """读取 JSON 文件, 文件不存在时返回空对象。

    使用 `utf-8-sig` 是为了兼容 Windows PowerShell 写出的 UTF-8 BOM 文件。
    该兼容只影响文件解码, 不改变任何 governed record 的语义。
    """
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_jsonl(path: Path) -> list[dict]:
    """读取 JSONL records, 文件不存在时返回空列表。"""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _unique_nonempty(records: list[dict], field: str) -> set[str]:
    """从 records 中提取某个字段的非空唯一值集合。"""
    return {str(record.get(field)) for record in records if record.get(field) not in {None, ""}}


def _numeric_values(records: list[dict], field: str) -> list[float]:
    """提取可转换为 float 的数值字段。"""
    values: list[float] = []
    for record in records:
        value = record.get(field)
        if value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def _seed_per_prompt_min(generation_records: list[dict]) -> int:
    """统计每个 prompt 下成功 seed 的最小数量。"""
    grouped: dict[str, set[str]] = {}
    for record in generation_records:
        if record.get("generation_status") != "success":
            continue
        prompt_id = str(record.get("prompt_id") or "")
        seed_id = str(record.get("seed_id") or "")
        if prompt_id and seed_id:
            grouped.setdefault(prompt_id, set()).add(seed_id)
    return min((len(seeds) for seeds in grouped.values()), default=0)


def _attack_names(*record_groups: list[dict]) -> set[str]:
    """从多个 record 集合中提取攻击名称, 并排除后处理占位类 no-op 名称。"""
    names: set[str] = set()
    for records in record_groups:
        names.update(_unique_nonempty(records, "attack_name"))
    return {name for name in names if name not in {"postprocess_no_attack", "no_attack"}}


def _negative_families(*record_groups: list[dict]) -> set[str]:
    """从多个 record 集合中提取 negative family。

    如果旧 records 只有 control_name 而没有 negative_family, 这里不会把 control_name 自动冒充为 negative_family,
    以避免把受控 proxy 误写为完整 negative family 覆盖。
    """
    families: set[str] = set()
    for records in record_groups:
        families.update(_unique_nonempty(records, "negative_family"))
    return {
        family for family in families
        if family not in {"not_applicable", "not_evaluated", "none"}
    }


def _has_wrong_sampler_replay(records: list[dict]) -> bool:
    """判断 records 是否包含 wrong_sampler_replay 控制证据。"""
    for record in records:
        joined = " ".join(str(record.get(field) or "") for field in ("negative_family", "control_name", "attack_name", "sample_role"))
        if "wrong_sampler_replay" in joined:
            return True
    return False


def _wrong_sampler_replay_not_equivalent(records: list[dict]) -> bool:
    """判断 wrong_sampler_replay 是否被记录为不能伪造正确轨迹。"""
    for record in records:
        joined = " ".join(str(record.get(field) or "") for field in ("negative_family", "control_name", "attack_name", "sample_role"))
        if "wrong_sampler_replay" not in joined:
            continue
        if record.get("wrong_sampler_replay_control_not_equivalent") is True:
            return True
        if record.get("decision") in {"not_equivalent", "controlled_negative_below_threshold", "replay_rejected"}:
            return True
    return False


def _boolean_any(records: list[dict], field: str) -> bool:
    """检查任意 record 是否显式给出布尔真值。"""
    return any(record.get(field) is True for record in records)


def _formal_motion_claim_status(formal_metric_records: list[dict], formal_decision: dict, postprocess_decision: dict) -> str | None:
    """从 formal records 推断当前 formal motion claim 状态。

    该函数优先使用 records 而不是旧 decision artifact, 因为旧批次可能尚未重跑修正后的 formal metric runner。
    """
    if formal_metric_records:
        visual_blocked = any(record.get("formal_visual_quality_ready") is not True for record in formal_metric_records)
        motion_blocked = any(record.get("formal_motion_consistency_ready") is not True for record in formal_metric_records)
        semantic_blocked = any(record.get("formal_semantic_consistency_ready") is not True for record in formal_metric_records)
        if visual_blocked:
            return "blocked_by_formal_visual_quality"
        if motion_blocked:
            return "blocked_by_formal_motion_consistency"
        if semantic_blocked:
            return "blocked_until_semantic_metric_ready"
        return "ready"
    return formal_decision.get("formal_metric_claim_status") or postprocess_decision.get("details", {}).get("formal_claim_status")


def build_small_scale_claim_pilot_audit(run_root: str | Path) -> dict:
    """构建 small-scale claim pilot 的自动审计结果。

    该函数属于项目特定写法。它只汇总已有 governed records 中的证据, 不补造 attack、negative family 或
    replay 结果。当前证据不足时, 输出应是明确的 missing_pilot_requirements, 而不是人工解释。
    """
    run_root = Path(run_root)
    generation_records = _read_jsonl(run_root / "records" / "generation_records.jsonl")
    mechanism_records = _read_jsonl(run_root / "records" / "mechanism_score_records.jsonl")
    controlled_negative_records = _read_jsonl(run_root / "records" / "controlled_negative_records.jsonl")
    pilot_matrix_records = _read_jsonl(run_root / "records" / "small_scale_claim_pilot_matrix_records.jsonl")
    runtime_attack_records = _read_jsonl(run_root / "records" / "runtime_attack_records.jsonl")
    runtime_detection_records = _read_jsonl(run_root / "records" / "runtime_detection_records.jsonl")
    quality_proxy_records = _read_jsonl(run_root / "records" / "quality_motion_semantic_proxy_records.jsonl")
    formal_metric_records = _read_jsonl(run_root / "records" / "formal_quality_motion_semantic_records.jsonl")
    postprocess_decision = _read_json(run_root / "artifacts" / "generative_video_mechanism_postprocess_decision.json")
    formal_decision = _read_json(run_root / "artifacts" / "formal_quality_motion_semantic_decision.json")
    motion_calibration_decision = _read_json(run_root / "artifacts" / "motion_threshold_calibration_decision.json")

    successful_generation_records = [record for record in generation_records if record.get("generation_status") == "success"]
    motion_claim_selection = select_motion_claim_generation_records(successful_generation_records, formal_metric_records)
    motion_claim_generation_records = motion_claim_selection.eligible_generation_records
    prompt_ids = _unique_nonempty(motion_claim_generation_records, "prompt_id")
    seed_per_prompt_min = _seed_per_prompt_min(motion_claim_generation_records)
    eligible_mechanism_records = filter_records_to_motion_claim_eligible(mechanism_records, motion_claim_selection)
    eligible_controlled_negative_records = filter_records_to_motion_claim_eligible(controlled_negative_records, motion_claim_selection)
    eligible_pilot_matrix_records = filter_records_to_motion_claim_eligible(pilot_matrix_records, motion_claim_selection)
    eligible_runtime_attack_records = filter_records_to_motion_claim_eligible(runtime_attack_records, motion_claim_selection)
    eligible_runtime_detection_records = filter_records_to_motion_claim_eligible(runtime_detection_records, motion_claim_selection)
    matrix_source_records = eligible_mechanism_records + eligible_controlled_negative_records + eligible_pilot_matrix_records
    ready_runtime_attack_records = [
        record for record in runtime_attack_records
        if record.get("attack_runtime_status") == "ready"
    ]
    eligible_ready_runtime_attack_records = [
        record for record in eligible_runtime_attack_records
        if record.get("attack_runtime_status") == "ready"
    ]
    ready_runtime_detection_records = [
        record for record in runtime_detection_records
        if record.get("runtime_detection_status") == "ready"
    ]
    eligible_ready_runtime_detection_records = [
        record for record in eligible_runtime_detection_records
        if record.get("runtime_detection_status") == "ready"
    ]
    attack_names = _attack_names(eligible_mechanism_records, eligible_controlled_negative_records, eligible_pilot_matrix_records, eligible_ready_runtime_attack_records)
    negative_families = _negative_families(eligible_mechanism_records, eligible_controlled_negative_records, eligible_pilot_matrix_records)
    method_variants = _unique_nonempty(eligible_mechanism_records + eligible_pilot_matrix_records, "method_variant")

    path_gain_values = _numeric_values(matrix_source_records, "path_marginal_gain_at_fixed_fpr")
    path_gain = round(mean(path_gain_values), 6) if path_gain_values else None
    path_gain_pass = path_gain is not None and path_gain > 0

    replay_uncertainty_values = _numeric_values(matrix_source_records, "replay_uncertainty_mean")
    replay_uncertainty_mean = round(mean(replay_uncertainty_values), 6) if replay_uncertainty_values else None
    replay_uncertainty_recorded = replay_uncertainty_mean is not None

    negative_tail_statuses = _unique_nonempty(matrix_source_records, "negative_tail_status")
    negative_tail_not_inflated = bool(negative_tail_statuses & {"not_inflated", "pass", "negative_tail_not_inflated"})
    threshold_details = _read_json(run_root / "thresholds" / "mechanism_proxy_thresholds.json")
    fixed_low_fpr_proxy_pass = postprocess_decision.get("details", {}).get("fixed_low_fpr_proxy_pass") is True
    controlled_negative_fpr = threshold_details.get("controlled_negative_fpr")

    wrong_key_score_separation_passed = _boolean_any(matrix_source_records, "wrong_key_score_separation_passed")
    wrong_sampler_present = _has_wrong_sampler_replay(matrix_source_records)
    wrong_sampler_not_equivalent = _wrong_sampler_replay_not_equivalent(matrix_source_records)

    formal_motion_claim_status = motion_claim_selection.formal_motion_claim_status or _formal_motion_claim_status(formal_metric_records, formal_decision, postprocess_decision)
    motion_threshold_calibration_ready = motion_calibration_decision.get("motion_threshold_calibration_ready") is True
    motion_threshold_id = motion_calibration_decision.get("motion_threshold_id") or HEURISTIC_MOTION_THRESHOLD_ID
    motion_threshold_source_split = motion_calibration_decision.get("motion_threshold_source_split") or HEURISTIC_MOTION_THRESHOLD_SOURCE_SPLIT
    formal_motion_uses_heuristic_threshold = not motion_threshold_calibration_ready

    requirement_checks: dict[str, bool] = {
        "prompt_coverage_ready": len(prompt_ids) >= MIN_PROMPT_COUNT,
        "seed_coverage_ready": seed_per_prompt_min >= MIN_SEED_PER_PROMPT,
        "attack_matrix_ready": len(attack_names) >= MIN_ATTACK_COUNT,
        "negative_family_ready": len(negative_families) >= MIN_NEGATIVE_FAMILY_COUNT,
        "method_variant_ready": len(method_variants) >= MIN_METHOD_VARIANT_COUNT,
        "path_marginal_gain_ready": path_gain_pass,
        "negative_tail_ready": negative_tail_not_inflated,
        "wrong_key_separation_ready": wrong_key_score_separation_passed,
        "wrong_sampler_replay_ready": wrong_sampler_present and wrong_sampler_not_equivalent,
        "replay_uncertainty_ready": replay_uncertainty_recorded,
        "quality_proxy_ready": bool(quality_proxy_records) and postprocess_decision.get("details", {}).get("quality_motion_semantic_proxy_pass") is True,
        "formal_motion_claim_ready": formal_motion_claim_status in FORMAL_MOTION_CLAIM_READY_STATUSES,
        "runtime_detection_ready": (not eligible_ready_runtime_attack_records) or len(eligible_ready_runtime_detection_records) >= len(eligible_ready_runtime_attack_records),
    }
    missing = [name for name, passed in requirement_checks.items() if not passed]

    if not successful_generation_records:
        claim_support_status = "blocked_until_generation_records"
    elif missing:
        claim_support_status = "workflow_progression_only"
    elif formal_motion_uses_heuristic_threshold:
        claim_support_status = "blocked_until_motion_threshold_calibration"
    else:
        claim_support_status = "supported_by_small_scale_claim_pilot_records"

    pilot_gate_decision = "PASS" if claim_support_status == "supported_by_small_scale_claim_pilot_records" else "FAIL"
    return {
        "stage_id": "small_scale_claim_pilot_gate",
        "run_root": str(run_root),
        "pilot_gate_decision": pilot_gate_decision,
        "claim_support_status": claim_support_status,
        "missing_pilot_requirements": missing,
        "pilot_missing_requirement_count": len(missing),
        "generation_record_count": len(generation_records),
        "successful_generation_count": len(successful_generation_records),
        **motion_claim_selection.audit_fields(),
        "prompt_count": len(prompt_ids),
        "seed_per_prompt_min": seed_per_prompt_min,
        "attack_count": len(attack_names),
        "negative_family_count": len(negative_families),
        "method_variant_count": len(method_variants),
        "pilot_matrix_record_count": len(eligible_pilot_matrix_records),
        "runtime_attack_record_count": len(runtime_attack_records),
        "runtime_attack_ready_count": len(eligible_ready_runtime_attack_records),
        "runtime_detection_record_count": len(runtime_detection_records),
        "runtime_detection_ready_count": len(eligible_ready_runtime_detection_records),
        "path_marginal_gain_at_fixed_fpr": path_gain,
        "negative_tail_status": "not_inflated" if negative_tail_not_inflated else "missing_or_not_ready",
        "wrong_key_score_separation_passed": wrong_key_score_separation_passed,
        "wrong_sampler_replay_control_not_equivalent": wrong_sampler_present and wrong_sampler_not_equivalent,
        "replay_uncertainty_mean": replay_uncertainty_mean,
        "quality_motion_semantic_proxy_pass": requirement_checks["quality_proxy_ready"],
        "fixed_low_fpr_proxy_pass": fixed_low_fpr_proxy_pass,
        "controlled_negative_fpr": controlled_negative_fpr,
        "formal_motion_claim_status": formal_motion_claim_status,
        "motion_threshold_id": motion_threshold_id,
        "motion_threshold_source_split": motion_threshold_source_split,
        "motion_threshold_calibration_decision": motion_calibration_decision.get("motion_threshold_calibration_decision"),
        "motion_threshold_calibration_required": not motion_threshold_calibration_ready,
        "test_time_threshold_update_blocked": True,
    }


def write_small_scale_claim_pilot_audit(run_root: str | Path) -> dict:
    """写出 small-scale claim pilot gate 的 records、table、decision 和 report。"""
    run_root = Path(run_root)
    audit = build_small_scale_claim_pilot_audit(run_root)
    record = with_flow_evidence_protocol_defaults(
        {"record_version": "small_scale_claim_pilot_gate_v1", **audit},
        trajectory_source_level="pilot_gate_aggregated_records",
        flow_state_admissibility_status="pilot_gate_ready"
        if audit["pilot_gate_decision"] == "PASS"
        else "pilot_gate_blocked",
        claim_support_status=audit["claim_support_status"],
    )
    write_jsonl(run_root / "records" / "small_scale_claim_pilot_gate_records.jsonl", [record])
    write_csv(run_root / "tables" / "small_scale_claim_pilot_gate_table.csv", [record])
    write_json(run_root / "artifacts" / "small_scale_claim_pilot_gate_decision.json", audit)
    report = (
        "# Small-scale Claim Pilot Gate Report\n\n"
        "该报告由 governed records 自动生成, 用于说明当前 pilot 是否满足进入 full experiment 的 claim gate。"
        "报告不会把缺失的 attack、negative family 或 replay 证据补造为通过。\n\n"
        f"- pilot_gate_decision: {audit['pilot_gate_decision']}\n"
        f"- claim_support_status: {audit['claim_support_status']}\n"
        f"- missing_pilot_requirements: {', '.join(audit['missing_pilot_requirements']) if audit['missing_pilot_requirements'] else 'none'}\n"
        f"- motion_claim_eligible_generation_count: {audit['motion_claim_eligible_generation_count']}\n"
        f"- motion_claim_excluded_generation_count: {audit['motion_claim_excluded_generation_count']}\n"
        f"- formal_motion_claim_status: {audit['formal_motion_claim_status']}\n"
        f"- motion_threshold_source_split: {audit['motion_threshold_source_split']}\n"
        f"- motion_threshold_calibration_required: {audit['motion_threshold_calibration_required']}\n"
    )
    report_path = run_root / "reports" / "small_scale_claim_pilot_gate_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="审计 small-scale claim pilot gate。")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--write-outputs", action="store_true")
    args = parser.parse_args()
    payload = write_small_scale_claim_pilot_audit(args.run_root) if args.write_outputs else build_small_scale_claim_pilot_audit(args.run_root)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
