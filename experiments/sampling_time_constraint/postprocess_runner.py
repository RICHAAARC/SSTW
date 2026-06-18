"""后处理 B6 sampling-time constraint Colab probe 输出。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from main.protocol.flow_evidence_fields import flow_evidence_protocol_defaults
from main.protocol.record_writer import write_json, write_jsonl
from main.protocol.table_builder import write_csv

MIN_KEY_SEPARATION_GAIN = 5e-4
MIN_KEY_SEPARATION_FLOW_VELOCITY_GAIN = 5e-4


def _read_json(path: Path) -> dict:
    """读取 JSON 文件, 文件不存在时返回空对象。"""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    """读取 JSONL records, 文件不存在时返回空列表。"""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _mean(records: list[dict], field: str) -> float:
    """计算字段均值, 空列表返回 0。"""
    values = [float(record[field]) for record in records if record.get(field) is not None]
    return mean(values) if values else 0.0


def _variant_rows(constraint_records: list[dict], formal_records: list[dict]) -> list[dict]:
    """按 method_variant 聚合 constraint 与 formal metric records。"""
    variants = sorted({record.get("method_variant") for record in constraint_records if record.get("method_variant")})
    rows: list[dict] = []
    for variant in variants:
        constraint_group = [record for record in constraint_records if record.get("method_variant") == variant]
        formal_group = [record for record in formal_records if record.get("method_variant") == variant]
        rows.append({
            "method_variant": variant,
            **flow_evidence_protocol_defaults(
                negative_family="aggregated_by_method_variant",
                trajectory_source_level="callback_latent_trace",
                sampler_signature_placeholder=next((record.get("sampler_signature_placeholder") for record in constraint_group if record.get("sampler_signature_placeholder")), None),
                flow_state_admissibility_status="all_enabled" if all(record.get("flow_state_admissibility_status") == "enabled" for record in constraint_group) else "mixed_or_disabled",
                claim_support_status="pending_postprocess_gate",
            ),
            "constraint_record_count": len(constraint_group),
            "formal_metric_record_count": len(formal_group),
            "latent_alignment_gain_mean": round(_mean(constraint_group, "latent_alignment_gain"), 6),
            "flow_velocity_alignment_gain_mean": round(_mean(constraint_group, "flow_velocity_alignment_gain"), 6),
            "flow_velocity_proxy_record_count": sum(1 for record in constraint_group if record.get("flow_velocity_proxy_available") is True),
            "constraint_applied_step_count": sum(1 for record in constraint_group if record.get("constraint_apply_status") == "applied"),
            "formal_visual_motion_ready": bool(formal_group) and all(record.get("formal_visual_quality_ready") is True and record.get("formal_motion_consistency_ready") is True for record in formal_group),
            "formal_semantic_ready": bool(formal_group) and all(record.get("formal_semantic_consistency_ready") is True for record in formal_group),
        })
    return rows


def postprocess_sampling_constraint_colab_run(run_root: str | Path) -> dict:
    """读取 B6 Colab records 并写出 postprocess artifacts。"""
    run_root = Path(run_root)
    runtime_decision = _read_json(run_root / "artifacts" / "sampling_time_constraint_colab_runtime_decision.json")
    constraint_records = _read_jsonl(run_root / "records" / "constraint_records.jsonl")
    formal_records = _read_jsonl(run_root / "records" / "formal_quality_motion_semantic_records.jsonl")
    rows = _variant_rows(constraint_records, formal_records)
    keyed_row = next((row for row in rows if row["method_variant"] == "keyed_state_trajectory_constraint"), {})
    baseline_row = next((row for row in rows if row["method_variant"] == "key_conditioned_state_space_with_trajectory"), {})
    keyed_gain = float(keyed_row.get("latent_alignment_gain_mean", 0.0))
    baseline_gain = float(baseline_row.get("latent_alignment_gain_mean", 0.0))
    keyed_flow_velocity_gain = float(keyed_row.get("flow_velocity_alignment_gain_mean", 0.0))
    baseline_flow_velocity_gain = float(baseline_row.get("flow_velocity_alignment_gain_mean", 0.0))
    without_key_row = next((row for row in rows if row["method_variant"] == "trajectory_constraint_without_key_condition"), {})
    wrong_key_row = next((row for row in rows if row["method_variant"] == "trajectory_constraint_wrong_key_control"), {})
    without_key_gain = float(without_key_row.get("latent_alignment_gain_mean", 0.0))
    wrong_key_gain = float(wrong_key_row.get("latent_alignment_gain_mean", 0.0))
    without_key_flow_velocity_gain = float(without_key_row.get("flow_velocity_alignment_gain_mean", 0.0))
    wrong_key_flow_velocity_gain = float(wrong_key_row.get("flow_velocity_alignment_gain_mean", 0.0))
    control_gain_ceiling = max(without_key_gain, wrong_key_gain)
    control_flow_velocity_gain_ceiling = max(without_key_flow_velocity_gain, wrong_key_flow_velocity_gain)
    key_separation_gain_over_control = round(keyed_gain - control_gain_ceiling, 6)
    key_separation_flow_velocity_gain_over_control = round(keyed_flow_velocity_gain - control_flow_velocity_gain_ceiling, 6)
    flow_velocity_gain_over_baseline = round(keyed_flow_velocity_gain - baseline_flow_velocity_gain, 6)
    flow_velocity_proxy_ready = bool(keyed_row) and int(keyed_row.get("flow_velocity_proxy_record_count", 0)) > 0
    gain_over_baseline = round(keyed_gain - baseline_gain, 6)
    formal_ready = bool(keyed_row) and keyed_row.get("formal_visual_motion_ready") is True and keyed_row.get("formal_semantic_ready") is True
    implementation_pass = runtime_decision.get("implementation_decision") == "PASS" and bool(constraint_records)
    probe_pass = all([
        implementation_pass,
        keyed_row.get("constraint_applied_step_count", 0) > 0,
        gain_over_baseline > 0.0,
        flow_velocity_proxy_ready,
        flow_velocity_gain_over_baseline > 0.0,
        key_separation_gain_over_control >= MIN_KEY_SEPARATION_GAIN,
        key_separation_flow_velocity_gain_over_control >= MIN_KEY_SEPARATION_FLOW_VELOCITY_GAIN,
        formal_ready,
    ])
    for row in rows:
        row["path_marginal_gain_at_fixed_fpr"] = gain_over_baseline if row["method_variant"] == "keyed_state_trajectory_constraint" else 0.0
        row["replay_uncertainty_mean"] = None
        row["S_path_inv"] = row["latent_alignment_gain_mean"]
        row["S_velocity"] = row["flow_velocity_alignment_gain_mean"]
        row["S_final_conservative"] = round(min(float(row["S_path_inv"]), float(row["S_velocity"])), 6)
        row["key_separation_gain_over_control"] = key_separation_gain_over_control if row["method_variant"] == "keyed_state_trajectory_constraint" else 0.0
        row["key_separation_flow_velocity_gain_over_control"] = key_separation_flow_velocity_gain_over_control if row["method_variant"] == "keyed_state_trajectory_constraint" else 0.0
        row["claim_support_status"] = "supported_by_probe_records_not_submission_freeze" if probe_pass and row["method_variant"] == "keyed_state_trajectory_constraint" else "not_supported_or_control_variant"
    decision = {
        "stage_id": "sampling_time_constraint_colab_postprocess",
        "mechanism_postprocess_decision": "PASS" if probe_pass else "FAIL",
        "mechanism_decision": "PASS" if probe_pass else "FAIL",
        "details": {
            "formal_claim_status": "real_sampling_probe_supported_by_governed_records_not_submission_freeze" if probe_pass else "blocked_until_sampling_constraint_colab_probe_ready",
            "constraint_record_count": len(constraint_records),
            "formal_metric_record_count": len(formal_records),
            "keyed_constraint_alignment_gain_mean": keyed_gain,
            "baseline_alignment_gain_mean": baseline_gain,
            "trajectory_constraint_gain_over_unconstrained": gain_over_baseline,
            "keyed_flow_velocity_alignment_gain_mean": keyed_flow_velocity_gain,
            "baseline_flow_velocity_alignment_gain_mean": baseline_flow_velocity_gain,
            "without_key_alignment_gain_mean": without_key_gain,
            "wrong_key_alignment_gain_mean": wrong_key_gain,
            "without_key_flow_velocity_alignment_gain_mean": without_key_flow_velocity_gain,
            "wrong_key_flow_velocity_alignment_gain_mean": wrong_key_flow_velocity_gain,
            "key_separation_gain_over_control": key_separation_gain_over_control,
            "key_separation_flow_velocity_gain_over_control": key_separation_flow_velocity_gain_over_control,
            "minimum_key_separation_gain": MIN_KEY_SEPARATION_GAIN,
            "minimum_key_separation_flow_velocity_gain": MIN_KEY_SEPARATION_FLOW_VELOCITY_GAIN,
            "flow_velocity_gain_over_unconstrained": flow_velocity_gain_over_baseline,
            "flow_velocity_proxy_ready": flow_velocity_proxy_ready,
            "formal_quality_semantic_ready": formal_ready,
            "constraint_main_claim_status": "real_sampling_probe_not_final_b6_submission_claim",
        },
    }
    write_jsonl(run_root / "records" / "constraint_variant_summary_records.jsonl", rows)
    write_csv(run_root / "tables" / "sampling_constraint_colab_summary_table.csv", rows)
    write_json(run_root / "artifacts" / "sampling_time_constraint_colab_postprocess_decision.json", decision)
    report_path = run_root / "reports" / "sampling_time_constraint_colab_postprocess_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "# Sampling-Time Constraint Colab Postprocess Report\n\n"
        "该报告基于真实 Colab sampling callback records 与 formal quality/motion/semantic records 生成。"
        "它支持 B6 real sampling probe, 但不等同于最终 submission freeze claim。\n\n"
        + json.dumps(decision, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "run_root": str(run_root),
        "constraint_record_count": len(constraint_records),
        "formal_metric_record_count": len(formal_records),
        "mechanism_postprocess_decision": decision["mechanism_postprocess_decision"],
        "mechanism_decision": decision["mechanism_decision"],
        "formal_claim_status": decision["details"]["formal_claim_status"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="后处理 B6 sampling-time constraint Colab probe。")
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    print(json.dumps(postprocess_sampling_constraint_colab_run(args.run_root), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
