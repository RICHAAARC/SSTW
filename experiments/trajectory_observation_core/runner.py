"""运行 B4 trajectory observation core probe。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from experiments.trajectory_observation_core.mechanism_audit import audit_mechanism
from main.attacks.synthetic_temporal_attacks import default_synthetic_attacks
from main.backends.synthetic_video_latent import build_synthetic_samples
from main.methods.state_space_watermark.detector_score_with_trajectory import score_with_trajectory
from main.methods.state_space_watermark.trajectory_state_adapter import trajectory_state_adapter_status
from main.protocol.calibrator import apply_thresholds, calibrate_thresholds
from main.protocol.decision import build_stage_decision
from main.protocol.manifest import build_run_manifest
from main.protocol.record_writer import write_json, write_jsonl
from main.protocol.table_builder import build_method_attack_table, write_csv
from main.trajectory.trajectory_controls import control_status
from main.trajectory.trajectory_reconstruction import reconstruct_trajectory_status
from main.trajectory.trajectory_runtime import runtime_overhead_status
from main.trajectory.trajectory_trace import build_trajectory_trace

SPLITS = ("dev", "calibration", "test")
SAMPLE_ROLES = ("clean_negative", "attacked_negative", "watermarked_positive", "attacked_positive")


def _load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _build_config() -> dict:
    return {
        "protocol": _load_json("configs/protocol/trajectory_observation_core.json"),
        "fixed_low_fpr": _load_json("configs/protocol/fixed_low_fpr.json"),
        "synthetic": _load_json("configs/protocol/synthetic_state_inference.json"),
        "trajectory": _load_json("configs/trajectory/trajectory_observation.json"),
        "controls": _load_json("configs/trajectory/trajectory_controls.json"),
        "time_grid": _load_json("configs/trajectory/trajectory_time_grid.json"),
        "methods": _load_json("configs/methods/method_variants_trajectory_observation.json"),
    }


def _sample_index(sample_id: str) -> int:
    """从 sample_id 提取确定性样本序号。"""
    try:
        return int(sample_id.rsplit("_", 1)[-1])
    except ValueError:
        return 0


def build_event_and_trace_records(config: dict) -> tuple[list[dict], list[dict]]:
    """构建 B4 event records 和 trajectory trace records。"""
    samples = build_synthetic_samples(SPLITS, SAMPLE_ROLES, int(config["synthetic"]["sample_count_per_cell"]), tuple(config["synthetic"]["latent_shape"]))
    attacks = default_synthetic_attacks()
    method_variants = config["methods"]["method_variants"]
    event_records: list[dict] = []
    trajectory_records: list[dict] = []
    for sample in samples:
        for attack in attacks:
            if sample.sample_role in {"clean_negative", "watermarked_positive"} and attack.attack_name != "no_attack":
                continue
            if sample.sample_role in {"attacked_negative", "attacked_positive"} and attack.attack_name == "no_attack":
                continue
            sample_index = _sample_index(sample.sample_id)
            trace = build_trajectory_trace(f"{sample.sample_id}_{attack.attack_name}", config["trajectory"])
            reconstruction_status = reconstruct_trajectory_status(trace.trajectory_source)
            trajectory_records.append({
                "sample_id": sample.sample_id,
                "attack_name": attack.attack_name,
                "trajectory_enabled": True,
                "trajectory_source": trace.trajectory_source,
                "trajectory_source_status": trace.trajectory_source_status,
                "trajectory_status_reason": "trajectory trace provided by B4 latent replay proxy",
                "trajectory_trace_id": trace.trajectory_trace_id,
                "trajectory_time_grid_id": trace.trajectory_time_grid_id,
                "trajectory_num_steps": trace.trajectory_num_steps,
                "trajectory_time_points": list(trace.trajectory_time_points),
                "trajectory_scheduler_id_placeholder": None,
                "velocity_estimator_id": trace.velocity_estimator_id,
                "velocity_projection_operator_id": trace.velocity_projection_operator_id,
                "trajectory_runtime_sec": trace.trajectory_runtime_sec,
                "trajectory_reconstruction_status": reconstruction_status,
            })
            for method_variant in method_variants:
                scores = score_with_trajectory(sample.sample_role, attack.attack_name, method_variant, sample_index)
                trajectory_enabled = method_variant != "key_conditioned_state_space_inference"
                s_traj = scores.get("S_trajectory_observation")
                event_records.append({
                    "record_version": "trajectory_observation_core_v1",
                    "sample_id": sample.sample_id,
                    "split": sample.split,
                    "sample_role": sample.sample_role,
                    "method_variant": method_variant,
                    "attack_name": attack.attack_name,
                    "attack_strength": attack.attack_strength,
                    "key_id": sample.key_id,
                    "content_id": sample.content_id,
                    "prompt_id_placeholder": None,
                    "generation_model_id_placeholder": "synthetic_gaussian_v1",
                    "semantic_consistency_placeholder": None,
                    "sampling_constraint_placeholder": None,
                    "trajectory_enabled": trajectory_enabled,
                    "trajectory_source": trace.trajectory_source,
                    "trajectory_source_status": trace.trajectory_source_status,
                    "trajectory_status_reason": "trajectory disabled for core baseline" if not trajectory_enabled else "trajectory trace provided by B4 latent replay proxy",
                    "trajectory_trace_id": trace.trajectory_trace_id,
                    "trajectory_time_grid_id": trace.trajectory_time_grid_id,
                    "trajectory_num_steps": trace.trajectory_num_steps,
                    "trajectory_time_points": list(trace.trajectory_time_points),
                    "trajectory_scheduler_id_placeholder": None,
                    "velocity_estimator_id": trace.velocity_estimator_id,
                    "velocity_projection_operator_id": trace.velocity_projection_operator_id,
                    "trajectory_runtime_sec": trace.trajectory_runtime_sec,
                    "trajectory_runtime_status": runtime_overhead_status(trace.trajectory_runtime_sec, float(config["protocol"]["runtime_overhead_blocking_sec"])),
                    "trajectory_state_adapter_status": trajectory_state_adapter_status(trajectory_enabled, trace.trajectory_source_status),
                    "target_fpr": config["fixed_low_fpr"]["target_fpr"],
                    "negative_state_over_threshold_count": 0,
                    "trajectory_state_gain": None,
                    "trajectory_gain_over_state_space": None,
                    "trajectory_negative_leakage_delta": 0.0,
                    "trajectory_payload_correlation": None,
                    "trajectory_state_correlation": None,
                    "trajectory_control_suppression_status": None,
                    "trajectory_control_failure_reason": "none",
                    **scores,
                })
    return event_records, trajectory_records


def _mean_score(records: list[dict], method_variant: str, sample_role: str = "attacked_positive") -> float:
    values = [float(record["S_final"]) for record in records if record["split"] == "test" and record["sample_role"] == sample_role and record["method_variant"] == method_variant]
    return mean(values) if values else 0.0


def build_control_records(records: list[dict]) -> list[dict]:
    """根据主 trajectory 和 control 方法构建 control records。"""
    controls: list[dict] = []
    control_map = {
        "random_key": "trajectory_random_key_control",
        "time_shuffle": "trajectory_time_shuffled_control",
        "direction_shuffle": "trajectory_direction_shuffled_control",
    }
    main_score = _mean_score(records, "key_conditioned_state_space_with_trajectory")
    for control_type, variant in control_map.items():
        observed_score = _mean_score(records, variant)
        controls.append({
            "control_type": control_type,
            "control_expected_effect": "suppressed",
            "control_observed_score": round(observed_score, 6),
            "control_delta_vs_main": round(main_score - observed_score, 6),
            "control_status": control_status(main_score, observed_score),
            "control_not_run_reason": "none",
        })
    return controls


def enrich_records_with_audit(records: list[dict], audit: dict) -> list[dict]:
    """把 trajectory 审计摘要写回 records, 便于表格重建。"""
    enriched: list[dict] = []
    for record in records:
        item = dict(record)
        if record["method_variant"] == "key_conditioned_state_space_with_trajectory":
            item["trajectory_gain_over_state_space"] = audit["trajectory_gain_over_state_space"]
            item["trajectory_negative_leakage_delta"] = audit["trajectory_negative_leakage_delta"]
            item["trajectory_payload_correlation"] = audit["trajectory_payload_correlation"]
            item["trajectory_state_correlation"] = audit["trajectory_state_correlation"]
            item["trajectory_control_suppression_status"] = audit["control_suppression_status"]
        if record.get("S_traj_state") is not None and record["method_variant"] != "key_conditioned_state_space_inference":
            item["trajectory_state_gain"] = round(float(record["S_traj_state"]) - float(record["S_state_posterior"]), 6)
        enriched.append(item)
    return enriched


def run(output_root: str | Path) -> dict:
    """运行 B4 并写出 records、tables、reports 和 decision。"""
    output_root = Path(output_root)
    config = _build_config()
    raw_records, trajectory_records = build_event_and_trace_records(config)
    thresholds = calibrate_thresholds(raw_records, target_fpr=config["fixed_low_fpr"]["target_fpr"])
    decided_records = apply_thresholds(raw_records, thresholds)
    control_records = build_control_records(decided_records)
    audit = audit_mechanism(decided_records, control_records, config["protocol"])
    records = enrich_records_with_audit(decided_records, audit)

    event_path = output_root / "records" / "event_scores.jsonl"
    state_path = output_root / "records" / "state_trace.jsonl"
    trajectory_path = output_root / "records" / "trajectory_trace.jsonl"
    control_path = output_root / "records" / "trajectory_control_records.jsonl"
    threshold_path = output_root / "thresholds" / "thresholds.json"
    table_path = output_root / "tables" / "trajectory_main_table.csv"
    control_table_path = output_root / "tables" / "trajectory_control_table.csv"
    correlation_table_path = output_root / "tables" / "score_correlation_table.csv"
    runtime_table_path = output_root / "tables" / "trajectory_runtime_table.csv"
    report_path = output_root / "reports" / "trajectory_observation_report.md"
    control_report_path = output_root / "reports" / "trajectory_control_report.md"
    audit_report_path = output_root / "reports" / "trajectory_mechanism_audit_report.md"
    decision_path = output_root / "artifacts" / "trajectory_observation_decision.json"
    manifest_path = output_root / "artifacts" / "trajectory_manifest.json"

    write_jsonl(event_path, records)
    write_jsonl(state_path, [{"sample_id": r["sample_id"], "method_variant": r["method_variant"], "attack_name": r["attack_name"], "S_state_posterior": r["S_state_posterior"], "S_traj_state": r.get("S_traj_state"), "state_entropy": r["state_entropy"]} for r in records])
    write_jsonl(trajectory_path, trajectory_records)
    write_jsonl(control_path, control_records)
    write_json(threshold_path, thresholds)
    write_csv(table_path, build_method_attack_table(records))
    write_csv(control_table_path, control_records)
    write_csv(correlation_table_path, [{"correlation_name": "trajectory_payload", "correlation_value": audit["trajectory_payload_correlation"], "correlation_status": audit["correlation_status"]}, {"correlation_name": "trajectory_state", "correlation_value": audit["trajectory_state_correlation"], "correlation_status": audit["correlation_status"]}])
    write_csv(runtime_table_path, [{"trajectory_runtime_sec": r["trajectory_runtime_sec"], "trajectory_runtime_status": r["trajectory_runtime_status"]} for r in records if r["method_variant"] == "key_conditioned_state_space_with_trajectory"][:10])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("# Trajectory Observation Report\n\nB4 lightweight trajectory observation core probe completed.\n", encoding="utf-8")
    control_report_path.write_text("# Trajectory Control Report\n\n" + json.dumps(control_records, ensure_ascii=False, indent=2), encoding="utf-8")
    audit_report_path.write_text("# Trajectory Mechanism Audit\n\n" + json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    implementation_pass = all([event_path.exists(), trajectory_path.exists(), control_path.exists(), table_path.exists(), all(r["trajectory_source_status"] for r in records), all(r["trajectory_state_adapter_status"] for r in records)])
    decision = build_stage_decision(implementation_pass, bool(audit["mechanism_pass"]), audit)
    decision["stage_id"] = "trajectory_observation_core_probe"
    write_json(decision_path, decision)
    write_json(manifest_path, build_run_manifest("trajectory_observation_core_probe", config, [str(event_path), str(trajectory_path), str(table_path), str(decision_path)]))
    return {"output_root": str(output_root), "event_record_count": len(records), "trajectory_trace_count": len(trajectory_records), "control_record_count": len(control_records), "implementation_decision": decision["implementation_decision"], "mechanism_decision": decision["mechanism_decision"], "audit": audit}


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 B4 trajectory observation core probe。")
    parser.add_argument("--output-root", default="outputs/runs/trajectory_observation_core_probe")
    args = parser.parse_args()
    print(json.dumps(run(args.output_root), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
