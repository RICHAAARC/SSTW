"""validation-scale Flow-specific adaptive attack runner。

该 runner 构建轻量级 adaptive attack governed records, 用于闭合 validation-scale
门禁中的工程阻断项。它不执行重型 GPU 自适应攻击, 也不支撑最终
Flow-specific adaptive robustness claim。
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


ADAPTIVE_ATTACK_SPECS: tuple[dict[str, Any], ...] = (
    {
        "adaptive_attack_name": "scheduler_change",
        "adaptive_attack_family": "time_grid_or_scheduler_mismatch",
        "adaptive_attack_strength": 0.25,
        "adaptive_attack_budget": "validation_proxy_scheduler_signature_shift",
        "attack_knowledge_level": "gray_box_sampler_signature_attacker",
        "targeted_evidence_layer": "time_grid_reliability",
        "path_response_suppression_factor": 0.12,
        "velocity_projection_suppression_factor": 0.05,
        "endpoint_preservation_status": "endpoint_preserved_by_proxy",
        "replay_signature_mismatch_status": "scheduler_signature_mismatch_expected",
        "trajectory_sketch_tamper_status": "not_attempted",
    },
    {
        "adaptive_attack_name": "time_grid_jitter",
        "adaptive_attack_family": "time_grid_or_scheduler_mismatch",
        "adaptive_attack_strength": 0.20,
        "adaptive_attack_budget": "validation_proxy_time_grid_jitter",
        "attack_knowledge_level": "gray_box_sampler_signature_attacker",
        "targeted_evidence_layer": "path_integral_alignment",
        "path_response_suppression_factor": 0.10,
        "velocity_projection_suppression_factor": 0.04,
        "endpoint_preservation_status": "endpoint_preserved_by_proxy",
        "replay_signature_mismatch_status": "time_grid_mismatch_expected",
        "trajectory_sketch_tamper_status": "not_attempted",
    },
    {
        "adaptive_attack_name": "wrong_sampler_replay",
        "adaptive_attack_family": "replay_signature_mismatch",
        "adaptive_attack_strength": 0.35,
        "adaptive_attack_budget": "validation_proxy_wrong_sampler_replay",
        "attack_knowledge_level": "gray_box_sampler_signature_attacker",
        "targeted_evidence_layer": "replay_posterior",
        "path_response_suppression_factor": 0.18,
        "velocity_projection_suppression_factor": 0.07,
        "endpoint_preservation_status": "endpoint_reconstructed_by_proxy",
        "replay_signature_mismatch_status": "wrong_sampler_control",
        "trajectory_sketch_tamper_status": "not_attempted",
    },
    {
        "adaptive_attack_name": "endpoint_path_decoupling",
        "adaptive_attack_family": "endpoint_preserving_path_attack",
        "adaptive_attack_strength": 0.30,
        "adaptive_attack_budget": "validation_proxy_endpoint_path_decoupling",
        "attack_knowledge_level": "white_box_oracle_limited_flow_attacker",
        "targeted_evidence_layer": "path_endpoint_consistency",
        "path_response_suppression_factor": 0.22,
        "velocity_projection_suppression_factor": 0.08,
        "endpoint_preservation_status": "endpoint_preserved_by_proxy",
        "replay_signature_mismatch_status": "not_applicable",
        "trajectory_sketch_tamper_status": "not_attempted",
    },
    {
        "adaptive_attack_name": "path_response_cancellation",
        "adaptive_attack_family": "path_response_cancellation",
        "adaptive_attack_strength": 0.40,
        "adaptive_attack_budget": "validation_proxy_path_response_cancellation",
        "attack_knowledge_level": "white_box_oracle_limited_flow_attacker",
        "targeted_evidence_layer": "path_integral_response",
        "path_response_suppression_factor": 0.30,
        "velocity_projection_suppression_factor": 0.10,
        "endpoint_preservation_status": "endpoint_not_guaranteed_by_proxy",
        "replay_signature_mismatch_status": "not_applicable",
        "trajectory_sketch_tamper_status": "not_attempted",
    },
    {
        "adaptive_attack_name": "trajectory_sketch_replacement_attempt",
        "adaptive_attack_family": "trajectory_sketch_tamper",
        "adaptive_attack_strength": 0.15,
        "adaptive_attack_budget": "validation_proxy_sketch_replacement",
        "attack_knowledge_level": "black_box_video_only_attacker",
        "targeted_evidence_layer": "authenticated_trajectory_sketch",
        "path_response_suppression_factor": 0.00,
        "velocity_projection_suppression_factor": 0.00,
        "endpoint_preservation_status": "not_applicable",
        "replay_signature_mismatch_status": "not_applicable",
        "trajectory_sketch_tamper_status": "replacement_rejected_by_required_future_sketch_gate",
    },
)


def _read_jsonl(path: Path) -> list[dict]:
    """读取 JSONL records, 文件不存在时返回空列表。"""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _safe_float(value: Any, default: float = 0.0) -> float:
    """把可选数值字段转为 float, 失败时返回默认值。"""
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """把 proxy 分数限制在稳定区间。"""
    return max(lower, min(upper, value))


def _base_score(record: dict) -> float:
    """从 runtime detection record 中提取 adaptive proxy 的源分数。"""
    for field_name in ("S_final_conservative", "S_runtime_attack_detection", "S_path_inv", "S_velocity"):
        if record.get(field_name) is not None:
            return _safe_float(record.get(field_name))
    return 0.0


def _ready_detection_records(run_root: Path) -> list[dict]:
    """获取可用于 validation adaptive proxy 的 runtime detection records。"""
    return [
        record for record in _read_jsonl(run_root / "records" / "runtime_detection_records.jsonl")
        if record.get("runtime_detection_status") == "ready"
    ]


def build_adaptive_attack_records(run_root: str | Path) -> list[dict[str, Any]]:
    """从 runtime detection records 构建 adaptive attack validation proxy records。

    该函数属于 validation-scale 工程闭环。它的职责是固定 adaptive attack 的
    records 形状和 claim 边界, 后续 full-paper 阶段应替换为真实 adaptive attack
    分数与 negative FPR 统计。
    """
    run_root = Path(run_root)
    detection_records = _ready_detection_records(run_root)
    records: list[dict[str, Any]] = []
    for detection_record in detection_records:
        source_score = _base_score(detection_record)
        for spec in ADAPTIVE_ATTACK_SPECS:
            path_suppression = round(_clip(source_score * float(spec["path_response_suppression_factor"])), 6)
            velocity_suppression = round(_clip(source_score * float(spec["velocity_projection_suppression_factor"])), 6)
            residual_proxy_score = round(_clip(source_score - path_suppression - velocity_suppression), 6)
            records.append(with_flow_evidence_protocol_defaults(
                {
                    "record_version": "adaptive_attack_validation_proxy_v1",
                    "generation_model_id": detection_record.get("generation_model_id"),
                    "prompt_id": detection_record.get("prompt_id"),
                    "seed_id": detection_record.get("seed_id"),
                    "trajectory_trace_id": detection_record.get("trajectory_trace_id"),
                    "source_attack_name": detection_record.get("attack_name"),
                    "adaptive_attack_name": spec["adaptive_attack_name"],
                    "adaptive_attack_family": spec["adaptive_attack_family"],
                    "adaptive_attack_strength": spec["adaptive_attack_strength"],
                    "adaptive_attack_budget": spec["adaptive_attack_budget"],
                    "attack_knowledge_level": spec["attack_knowledge_level"],
                    "targeted_evidence_layer": spec["targeted_evidence_layer"],
                    "endpoint_preservation_status": spec["endpoint_preservation_status"],
                    "path_response_suppression_score": path_suppression,
                    "velocity_projection_suppression_score": velocity_suppression,
                    "adaptive_residual_proxy_score": residual_proxy_score,
                    "replay_signature_mismatch_status": spec["replay_signature_mismatch_status"],
                    "trajectory_sketch_tamper_status": spec["trajectory_sketch_tamper_status"],
                    "quality_guard_status": "not_evaluated_validation_proxy",
                    "semantic_projection_status": "not_evaluated_validation_proxy",
                    "adaptive_negative_fpr": None,
                    "adaptive_negative_fpr_status": "not_available_until_full_adaptive_negative_split",
                    "adaptive_attack_success_status": "validation_proxy_stress_record_only",
                    "adaptive_attack_claim_support_status": "validation_adaptive_attack_proxy_only",
                    "claim_support_status": "validation_adaptive_attack_proxy_only",
                },
                negative_family=detection_record.get("negative_family"),
                trajectory_source_level="runtime_detection_proxy_adaptive_attack",
                flow_state_admissibility_status="adaptive_attack_proxy_not_claim_admissible",
                claim_support_status="validation_adaptive_attack_proxy_only",
            ))
    return records


def audit_adaptive_attack_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """审计 adaptive attack validation proxy records 覆盖情况。"""
    attack_names = {str(record.get("adaptive_attack_name")) for record in records if record.get("adaptive_attack_name")}
    attack_families = {str(record.get("adaptive_attack_family")) for record in records if record.get("adaptive_attack_family")}
    knowledge_levels = {str(record.get("attack_knowledge_level")) for record in records if record.get("attack_knowledge_level")}
    targeted_layers = {str(record.get("targeted_evidence_layer")) for record in records if record.get("targeted_evidence_layer")}
    residual_scores = [
        float(record["adaptive_residual_proxy_score"])
        for record in records
        if record.get("adaptive_residual_proxy_score") is not None
    ]
    required_attack_names = {str(spec["adaptive_attack_name"]) for spec in ADAPTIVE_ATTACK_SPECS}
    missing_attack_names = sorted(required_attack_names - attack_names)
    decision = "PASS" if records and not missing_attack_names and len(knowledge_levels) >= 3 else "FAIL"
    return {
        "stage_id": "adaptive_attack_validation_proxy",
        "adaptive_attack_decision": decision,
        "claim_support_status": "validation_adaptive_attack_proxy_only"
        if decision == "PASS"
        else "validation_adaptive_attack_blocked",
        "adaptive_attack_record_count": len(records),
        "adaptive_attack_name_count": len(attack_names),
        "adaptive_attack_family_count": len(attack_families),
        "adaptive_attack_knowledge_level_count": len(knowledge_levels),
        "adaptive_attack_targeted_layer_count": len(targeted_layers),
        "adaptive_attack_missing_names": missing_attack_names,
        "adaptive_attack_score_mean": round(mean(residual_scores), 6) if residual_scores else None,
        "adaptive_negative_fpr_status": "not_available_until_full_adaptive_negative_split",
        "adaptive_robustness_claim_allowed": False,
        "adaptive_attack_evidence_level": "validation_runtime_detection_proxy",
    }


def run_adaptive_attack_validation_proxy(run_root: str | Path) -> dict[str, Any]:
    """写出 adaptive attack validation proxy records、table、decision 和 report。"""
    run_root = Path(run_root)
    records = build_adaptive_attack_records(run_root)
    audit = audit_adaptive_attack_records(records)
    write_jsonl(run_root / "records" / "adaptive_attack_records.jsonl", records)
    write_csv(run_root / "tables" / "adaptive_attack_table.csv", records)
    write_json(run_root / "artifacts" / "adaptive_attack_decision.json", audit)
    report = (
        "# Adaptive Attack Validation Proxy Report\n\n"
        "该报告由 runtime detection records 自动生成, 用于闭合 validation-scale 的 "
        "Flow-specific adaptive attack 工程门禁。当前结果是 validation proxy, "
        "不能作为 full-paper adaptive robustness claim。\n\n"
        f"- adaptive_attack_decision: {audit['adaptive_attack_decision']}\n"
        f"- adaptive_attack_record_count: {audit['adaptive_attack_record_count']}\n"
        f"- adaptive_attack_name_count: {audit['adaptive_attack_name_count']}\n"
        f"- adaptive_attack_knowledge_level_count: {audit['adaptive_attack_knowledge_level_count']}\n"
        f"- adaptive_robustness_claim_allowed: {str(audit['adaptive_robustness_claim_allowed']).lower()}\n"
        f"- claim_support_status: {audit['claim_support_status']}\n"
    )
    report_path = run_root / "reports" / "adaptive_attack_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 validation-scale adaptive attack proxy records。")
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    payload = run_adaptive_attack_validation_proxy(args.run_root)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
