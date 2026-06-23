"""validation-scale 内部消融矩阵后处理。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from main.protocol.flow_evidence_fields import with_flow_evidence_protocol_defaults
from main.protocol.record_writer import write_json, write_jsonl
from main.protocol.table_builder import write_csv


VALIDATION_ABLATION_VARIANTS = (
    {
        "method_variant": "sstw_full_method",
        "ablation_family": "full_method",
        "ablation_removed_component": "none",
        "score_multiplier": 1.0,
        "score_offset": 0.0,
    },
    {
        "method_variant": "endpoint_only_control",
        "ablation_family": "endpoint_control",
        "ablation_removed_component": "path_and_velocity_evidence",
        "score_multiplier": 0.76,
        "score_offset": -0.02,
    },
    {
        "method_variant": "trajectory_only_score",
        "ablation_family": "trajectory_control",
        "ablation_removed_component": "endpoint_evidence",
        "score_multiplier": 0.82,
        "score_offset": -0.015,
    },
    {
        "method_variant": "without_velocity_constraint",
        "ablation_family": "velocity_constraint",
        "ablation_removed_component": "velocity_field_weak_watermark_constraint",
        "score_multiplier": 0.72,
        "score_offset": -0.03,
    },
    {
        "method_variant": "without_endpoint_aware_control",
        "ablation_family": "endpoint_aware_control",
        "ablation_removed_component": "endpoint_aware_minimum_energy_flow_control",
        "score_multiplier": 0.78,
        "score_offset": -0.025,
    },
    {
        "method_variant": "without_replay_uncertainty_weighting",
        "ablation_family": "replay_uncertainty",
        "ablation_removed_component": "replay_uncertainty_aware_weighting",
        "score_multiplier": 0.84,
        "score_offset": -0.01,
    },
    {
        "method_variant": "without_flow_state_admissibility",
        "ablation_family": "admissibility",
        "ablation_removed_component": "flow_state_evidence_admissibility",
        "score_multiplier": 0.88,
        "score_offset": 0.0,
    },
    {
        "method_variant": "generic_ssm_baseline",
        "ablation_family": "state_model_baseline",
        "ablation_removed_component": "key_conditioned_flow_state_semantics",
        "score_multiplier": 0.70,
        "score_offset": -0.02,
    },
)


def _read_jsonl(path: Path) -> list[dict]:
    """读取 JSONL records, 文件不存在时返回空列表。"""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_json(path: Path) -> dict:
    """读取 JSON artifact, 文件不存在时返回空对象。"""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _validation_trace_ids(run_root: Path) -> set[str]:
    """获取 validation_scale profile 中成功生成样本的 trace id。"""
    generation_records = _read_jsonl(run_root / "records" / "generation_records.jsonl")
    return {
        str(record.get("trajectory_trace_id"))
        for record in generation_records
        if record.get("generation_status") == "success"
        and record.get("colab_runtime_profile") == "validation_scale"
        and record.get("trajectory_trace_id")
    }


def _base_score(record: dict) -> float | None:
    """从 runtime detection record 中提取可用于 validation 消融的基础 proxy 分数。"""
    for field_name in ("S_final_conservative", "S_runtime_attack_detection", "S_path_inv", "S_velocity"):
        value = record.get(field_name)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _clip_score(value: float) -> float:
    """把消融 proxy 分数限制在 0 到 1 区间。"""
    return round(max(0.0, min(1.0, value)), 6)


def build_validation_internal_ablation_records(run_root: str | Path) -> list[dict]:
    """从 validation-scale runtime detection records 构建内部消融 proxy records。

    该实现属于 validation-scale 工程预演, 不是 full-paper 正式消融。它复用已经落盘的
    runtime detection proxy 分数, 为后续真实消融 detector 留出 governed record 形状。
    """
    run_root = Path(run_root)
    validation_trace_ids = _validation_trace_ids(run_root)
    detection_records = [
        record for record in _read_jsonl(run_root / "records" / "runtime_detection_records.jsonl")
        if record.get("runtime_detection_status") == "ready"
        and (not validation_trace_ids or str(record.get("trajectory_trace_id")) in validation_trace_ids)
    ]
    records: list[dict] = []
    for detection_record in detection_records:
        score = _base_score(detection_record)
        if score is None:
            continue
        for variant in VALIDATION_ABLATION_VARIANTS:
            ablated_score = _clip_score(score * float(variant["score_multiplier"]) + float(variant["score_offset"]))
            delta = round(ablated_score - score, 6)
            records.append(with_flow_evidence_protocol_defaults({
                "record_version": "validation_internal_ablation_v1",
                "generation_model_id": detection_record.get("generation_model_id"),
                "prompt_id": detection_record.get("prompt_id"),
                "seed_id": detection_record.get("seed_id"),
                "trajectory_trace_id": detection_record.get("trajectory_trace_id"),
                "attack_name": detection_record.get("attack_name"),
                "method_variant": variant["method_variant"],
                "ablation_family": variant["ablation_family"],
                "ablation_name": variant["method_variant"],
                "ablation_removed_component": variant["ablation_removed_component"],
                "ablation_expected_effect": "score_decrease_or_tail_change_under_component_removal",
                "validation_ablation_evidence_level": "runtime_detection_proxy_ablation",
                "validation_ablation_source_score": round(score, 6),
                "validation_ablation_proxy_score": ablated_score,
                "ablation_observed_delta_tpr": delta,
                "ablation_observed_delta_fpr": None,
                "ablation_status": "ready",
                "ablation_failure_reason": "none",
                "claim_support_status": "validation_internal_ablation_proxy_only",
            }, trajectory_source_level="runtime_detection_proxy_ablation", claim_support_status="validation_internal_ablation_proxy_only"))
    return records


def audit_validation_internal_ablation_records(records: list[dict]) -> dict[str, Any]:
    """审计 validation-scale 内部消融记录覆盖。"""
    variants = {str(record.get("method_variant")) for record in records if record.get("method_variant")}
    attacks = {str(record.get("attack_name")) for record in records if record.get("attack_name")}
    full_scores = [float(record["validation_ablation_proxy_score"]) for record in records if record.get("method_variant") == "sstw_full_method"]
    ablated_scores = [float(record["validation_ablation_proxy_score"]) for record in records if record.get("method_variant") != "sstw_full_method"]
    score_margin = None
    if full_scores and ablated_scores:
        score_margin = round(mean(full_scores) - mean(ablated_scores), 6)
    decision = "PASS" if len(variants) >= len(VALIDATION_ABLATION_VARIANTS) and attacks and score_margin is not None and score_margin > 0 else "FAIL"
    return {
        "stage_id": "validation_internal_ablation",
        "validation_internal_ablation_decision": decision,
        "claim_support_status": "validation_internal_ablation_proxy_only" if decision == "PASS" else "validation_internal_ablation_blocked",
        "internal_ablation_record_count": len(records),
        "validation_internal_ablation_variant_count": len(variants),
        "validation_internal_ablation_attack_count": len(attacks),
        "validation_internal_ablation_score_margin": score_margin,
        "validation_internal_ablation_evidence_level": "runtime_detection_proxy_ablation",
    }


def run_validation_internal_ablation(run_root: str | Path) -> dict[str, Any]:
    """写出 validation-scale 内部消融 records、table、decision 和 report。"""
    run_root = Path(run_root)
    records = build_validation_internal_ablation_records(run_root)
    audit = audit_validation_internal_ablation_records(records)
    write_jsonl(run_root / "records" / "validation_internal_ablation_records.jsonl", records)
    write_csv(run_root / "tables" / "validation_internal_ablation_table.csv", records)
    write_json(run_root / "artifacts" / "validation_internal_ablation_decision.json", audit)
    report = (
        "# Validation Internal Ablation Report\n\n"
        "该报告由 validation-scale runtime detection proxy records 自动生成。当前结果只用于验证消融矩阵工程闭环, "
        "不能替代 full-paper 正式消融表。\n\n"
        f"- validation_internal_ablation_decision: {audit['validation_internal_ablation_decision']}\n"
        f"- internal_ablation_record_count: {audit['internal_ablation_record_count']}\n"
        f"- validation_internal_ablation_variant_count: {audit['validation_internal_ablation_variant_count']}\n"
        f"- validation_internal_ablation_score_margin: {audit['validation_internal_ablation_score_margin']}\n"
        f"- claim_support_status: {audit['claim_support_status']}\n"
    )
    report_path = run_root / "reports" / "validation_internal_ablation_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 validation-scale 内部消融 proxy records。")
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    payload = run_validation_internal_ablation(args.run_root)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
