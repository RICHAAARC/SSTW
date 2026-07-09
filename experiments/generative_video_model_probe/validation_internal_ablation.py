"""paper profile 正式内部消融矩阵汇总。

该文件只把已经真实生成并完成正式视频内容检测的不同 method_variant
转写为内部消融记录。它不再从 full-method 分数派生替代分数, 因而不会把
component-removal 的假设性结果写入论文 claim。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from main.protocol.flow_evidence_fields import with_flow_evidence_protocol_defaults
from main.protocol.record_writer import write_json, write_jsonl
from main.protocol.table_builder import write_csv


FULL_METHOD_VARIANT = "sstw_full_method"
FORMAL_INTERNAL_ABLATION_EVIDENCE_LEVEL = "formal_component_removal_video_detector"
FORMAL_INTERNAL_ABLATION_CLAIM_STATUS = "formal_internal_ablation_variant_measured"
FORMAL_DETECTOR_EVIDENCE_LEVEL = "attacked_video_content_detector"
VALIDATION_ABLATION_VARIANTS = (
    {
        "method_variant": FULL_METHOD_VARIANT,
        "ablation_family": "full_method",
        "ablation_removed_component": "none",
    },
    {
        "method_variant": "endpoint_only_control",
        "ablation_family": "endpoint_control",
        "ablation_removed_component": "path_and_velocity_evidence",
    },
    {
        "method_variant": "trajectory_only_score",
        "ablation_family": "trajectory_control",
        "ablation_removed_component": "endpoint_evidence",
    },
    {
        "method_variant": "without_velocity_constraint",
        "ablation_family": "velocity_constraint",
        "ablation_removed_component": "velocity_field_weak_watermark_constraint",
    },
    {
        "method_variant": "without_endpoint_aware_control",
        "ablation_family": "endpoint_aware_control",
        "ablation_removed_component": "endpoint_aware_minimum_energy_flow_control",
    },
    {
        "method_variant": "without_replay_uncertainty_weighting",
        "ablation_family": "replay_uncertainty",
        "ablation_removed_component": "replay_uncertainty_aware_weighting",
    },
    {
        "method_variant": "without_flow_state_admissibility",
        "ablation_family": "admissibility",
        "ablation_removed_component": "flow_state_evidence_admissibility",
    },
    {
        "method_variant": "generic_ssm_baseline",
        "ablation_family": "state_model_baseline",
        "ablation_removed_component": "key_conditioned_flow_state_semantics",
    },
)
INTERNAL_ABLATION_PROFILE_NAMES = {"probe_paper", "pilot_paper", "full_paper"}
PILOT_PAPER_PROFILE_NAMES = {"pilot_paper"}


def _read_jsonl(path: Path) -> list[dict]:
    """读取 JSONL records, 文件不存在时返回空列表。"""

    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _profile_trace_map(run_root: Path) -> dict[str, dict[str, str]]:
    """获取支持内部消融的 profile 与 method_variant 信息。

    通用工程写法是以 `trajectory_trace_id` 关联 generation、attack、detection
    和 ablation records。项目特定写法是只接受 paper profile 的正式样本,
    防止旧探索样本进入 probe_paper、pilot_paper 或 full_paper 的论文结论。
    """

    generation_records = _read_jsonl(run_root / "records" / "generation_records.jsonl")
    mapping: dict[str, dict[str, str]] = {}
    for record in generation_records:
        trace_id = str(record.get("trajectory_trace_id") or "")
        if (
            record.get("generation_status") == "success"
            and record.get("colab_runtime_profile") in INTERNAL_ABLATION_PROFILE_NAMES
            and trace_id
        ):
            mapping[trace_id] = {
                "ablation_runtime_profile": str(record.get("colab_runtime_profile")),
                "method_variant": _normalize_method_variant(record.get("method_variant")),
            }
    return mapping


def _normalize_method_variant(value: Any) -> str:
    """把运行时 method_variant 归一到内部消融表使用的正式变体名称。"""

    normalized = str(value or "").strip()
    if normalized in {"", "key_conditioned_state_space_with_trajectory"}:
        return FULL_METHOD_VARIANT
    return normalized


def _formal_score(record: dict) -> float | None:
    """从正式视频内容检测记录中提取内部消融分数。"""

    for field_name in ("sstw_raw_detector_score", "raw_detector_score", "sstw_score"):
        value = record.get(field_name)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _formal_detection_ready(record: dict) -> bool:
    """判断 runtime detection record 是否可支撑正式内部消融。"""

    return (
        record.get("runtime_detection_status") == "ready"
        and record.get("sstw_detector_evidence_level") == FORMAL_DETECTOR_EVIDENCE_LEVEL
        and record.get("trajectory_trace_used_for_score") is False
        and record.get("runtime_detection_claim_level") == "formal_paper_detector"
        and _formal_score(record) is not None
    )


def build_validation_internal_ablation_records(run_root: str | Path) -> list[dict]:
    """从正式 runtime detection records 构建内部消融 measured_formal records。

    只有当每个消融变体都已经通过独立生成配置真实产出视频, 并经过同一正式
    视频内容检测器评分后, 对应记录才会进入 `metric_status: measured_formal`。
    若某个变体没有真实记录, 本函数不会合成替代分数, 后续门禁会 fail-closed。
    """

    run_root = Path(run_root)
    trace_context = _profile_trace_map(run_root)
    allowed_trace_ids = set(trace_context)
    detection_records = [
        record
        for record in _read_jsonl(run_root / "records" / "runtime_detection_records.jsonl")
        if _formal_detection_ready(record)
        and (not allowed_trace_ids or str(record.get("trajectory_trace_id")) in allowed_trace_ids)
    ]
    variant_config = {item["method_variant"]: item for item in VALIDATION_ABLATION_VARIANTS}
    records: list[dict] = []
    for detection_record in detection_records:
        trace_id = str(detection_record.get("trajectory_trace_id") or "")
        generation_context = trace_context.get(trace_id, {})
        method_variant = _normalize_method_variant(
            detection_record.get("method_variant") or generation_context.get("method_variant")
        )
        if method_variant not in variant_config:
            continue
        config = variant_config[method_variant]
        score = _formal_score(detection_record)
        if score is None:
            continue
        records.append(with_flow_evidence_protocol_defaults({
            "record_version": "formal_internal_ablation_variant_v1",
            "ablation_runtime_profile": generation_context.get("ablation_runtime_profile", "unknown_profile"),
            "generation_model_id": detection_record.get("generation_model_id"),
            "prompt_id": detection_record.get("prompt_id"),
            "seed_id": detection_record.get("seed_id"),
            "trajectory_trace_id": trace_id,
            "attack_name": detection_record.get("attack_name"),
            "method_variant": method_variant,
            "ablation_family": config["ablation_family"],
            "ablation_name": method_variant,
            "ablation_removed_component": config["ablation_removed_component"],
            "ablation_expected_effect": "measured_score_change_under_real_component_removal",
            "formal_internal_ablation_evidence_level": FORMAL_INTERNAL_ABLATION_EVIDENCE_LEVEL,
            "formal_internal_ablation_score": round(score, 6),
            "formal_internal_ablation_score_semantics": "sstw_key_conditioned_video_content_detector_score",
            "metric_status": "measured_formal",
            "ablation_status": "ready",
            "ablation_failure_reason": "none",
            "claim_support_status": FORMAL_INTERNAL_ABLATION_CLAIM_STATUS,
        }, trajectory_source_level=FORMAL_INTERNAL_ABLATION_EVIDENCE_LEVEL, claim_support_status=FORMAL_INTERNAL_ABLATION_CLAIM_STATUS))
    return records


def audit_validation_internal_ablation_records(records: list[dict]) -> dict[str, Any]:
    """审计 paper profile 正式内部消融记录覆盖。"""

    variants = {str(record.get("method_variant")) for record in records if record.get("method_variant")}
    attacks = {str(record.get("attack_name")) for record in records if record.get("attack_name")}
    profile_counts: dict[str, int] = {}
    trace_counts_by_variant: dict[str, set[str]] = {}
    for record in records:
        profile = str(record.get("ablation_runtime_profile") or "unknown_profile")
        profile_counts[profile] = profile_counts.get(profile, 0) + 1
        variant = str(record.get("method_variant") or "")
        trace_id = str(record.get("trajectory_trace_id") or "")
        if variant and trace_id:
            trace_counts_by_variant.setdefault(variant, set()).add(trace_id)

    required_variants = {item["method_variant"] for item in VALIDATION_ABLATION_VARIANTS}
    missing_variants = sorted(required_variants - variants)
    full_scores = [
        float(record["formal_internal_ablation_score"])
        for record in records
        if record.get("method_variant") == FULL_METHOD_VARIANT
    ]
    ablated_scores = [
        float(record["formal_internal_ablation_score"])
        for record in records
        if record.get("method_variant") != FULL_METHOD_VARIANT
    ]
    score_margin = None
    if full_scores and ablated_scores:
        score_margin = round(mean(full_scores) - mean(ablated_scores), 6)
    pilot_paper_records = [
        record for record in records
        if record.get("ablation_runtime_profile") in PILOT_PAPER_PROFILE_NAMES
    ]
    formal_record_count = sum(
        1
        for record in records
        if record.get("metric_status") == "measured_formal"
        and record.get("formal_internal_ablation_evidence_level") == FORMAL_INTERNAL_ABLATION_EVIDENCE_LEVEL
    )
    decision = (
        "PASS"
        if records
        and formal_record_count == len(records)
        and not missing_variants
        and attacks
        and score_margin is not None
        and score_margin > 0
        else "FAIL"
    )
    return {
        "stage_id": "formal_internal_ablation_variant_matrix",
        "validation_internal_ablation_decision": decision,
        "claim_support_status": "formal_internal_ablation_variant_matrix_ready"
        if decision == "PASS"
        else "formal_internal_ablation_variant_matrix_blocked",
        "internal_ablation_record_count": len(records),
        "formal_internal_ablation_record_count": formal_record_count,
        "validation_internal_ablation_variant_count": len(variants),
        "validation_internal_ablation_attack_count": len(attacks),
        "validation_internal_ablation_score_margin": score_margin,
        "validation_internal_ablation_evidence_level": FORMAL_INTERNAL_ABLATION_EVIDENCE_LEVEL,
        "validation_internal_ablation_missing_variants": missing_variants,
        "validation_internal_ablation_profile_counts": profile_counts,
        "validation_internal_ablation_trace_counts": {
            variant: len(trace_ids) for variant, trace_ids in sorted(trace_counts_by_variant.items())
        },
        "pilot_paper_internal_ablation_record_count": len(pilot_paper_records),
    }


def run_validation_internal_ablation(run_root: str | Path) -> dict[str, Any]:
    """写出正式内部消融 records、table、decision 和 report。"""

    run_root = Path(run_root)
    records = build_validation_internal_ablation_records(run_root)
    audit = audit_validation_internal_ablation_records(records)
    write_jsonl(run_root / "records" / "formal_internal_ablation_variant_records.jsonl", records)
    write_jsonl(run_root / "records" / "validation_internal_ablation_records.jsonl", records)
    write_csv(run_root / "tables" / "validation_internal_ablation_table.csv", records)
    write_json(run_root / "artifacts" / "validation_internal_ablation_decision.json", audit)
    report = (
        "# Formal Internal Ablation Variant Matrix Report\n\n"
        "该报告只消费正式 runtime detection records。若某个 component-removal 变体没有"
        "独立生成视频并完成正式视频内容检测, 该变体保持 missing, 不会由 full-method "
        "分数合成替代结果。\n\n"
        f"- validation_internal_ablation_decision: {audit['validation_internal_ablation_decision']}\n"
        f"- internal_ablation_record_count: {audit['internal_ablation_record_count']}\n"
        f"- validation_internal_ablation_variant_count: {audit['validation_internal_ablation_variant_count']}\n"
        f"- validation_internal_ablation_missing_variants: {', '.join(audit['validation_internal_ablation_missing_variants']) if audit['validation_internal_ablation_missing_variants'] else 'none'}\n"
        f"- pilot_paper_internal_ablation_record_count: {audit['pilot_paper_internal_ablation_record_count']}\n"
        f"- validation_internal_ablation_score_margin: {audit['validation_internal_ablation_score_margin']}\n"
        f"- claim_support_status: {audit['claim_support_status']}\n"
    )
    report_path = run_root / "reports" / "validation_internal_ablation_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 paper profile 正式内部消融 records。")
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    payload = run_validation_internal_ablation(args.run_root)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
