"""运行第一阶段 synthetic state protocol。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.synthetic_state_inference.mechanism_audit import audit_mechanism
from main.attacks.synthetic_temporal_attacks import default_synthetic_attacks
from main.backends.synthetic_video_latent import build_synthetic_samples
from main.methods.state_space_watermark.method_factory import list_method_variants
from main.methods.state_space_watermark.score import score_method
from main.methods.state_space_watermark.tubelet_code import build_tubelet_code_config
from main.protocol.calibrator import apply_thresholds, calibrate_thresholds
from main.protocol.decision import build_stage_decision
from main.protocol.flow_evidence_fields import conservative_flow_score, flow_evidence_protocol_defaults
from main.protocol.manifest import build_run_manifest
from main.protocol.record_writer import write_json, write_jsonl
from main.protocol.table_builder import build_method_attack_table, write_csv

SPLITS = ("dev", "calibration", "test")
SAMPLE_ROLES = ("clean_negative", "attacked_negative", "watermarked_positive", "attacked_positive")


def _load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_event_records(config: dict) -> list[dict]:
    """构建未决策的 event records。"""
    samples = build_synthetic_samples(SPLITS, SAMPLE_ROLES, int(config["synthetic"]["sample_count_per_cell"]), tuple(config["synthetic"]["latent_shape"]))
    attacks = default_synthetic_attacks()
    tubelet_config = build_tubelet_code_config()
    records: list[dict] = []
    for sample in samples:
        for attack in attacks:
            if sample.sample_role in {"clean_negative", "watermarked_positive"} and attack.attack_name != "no_attack":
                continue
            if sample.sample_role in {"attacked_negative", "attacked_positive"} and attack.attack_name == "no_attack":
                continue
            for method_variant in list_method_variants():
                result = score_method(sample.sample_role, attack.attack_name, method_variant)
                record = {
                    "record_version": "synthetic_state_protocol_v1", "sample_id": sample.sample_id, "split": sample.split, "sample_role": sample.sample_role, "method_variant": method_variant, "attack_name": attack.attack_name, "attack_strength": attack.attack_strength, "key_id": sample.key_id, "content_id": sample.content_id, "prompt_id_placeholder": None, "seed_id": sample.seed_id, "generation_model_id_placeholder": config["synthetic"]["generation_model_id_placeholder"], "backend_id": config["synthetic"]["backend_id"],
                    "tubelet_length": tubelet_config.tubelet_length, "tubelet_spatial_patch": tubelet_config.tubelet_spatial_patch, "tubelet_stride_t": tubelet_config.tubelet_stride_t, "tubelet_stride_xy": tubelet_config.tubelet_stride_xy, "watermark_alpha": tubelet_config.watermark_alpha, "payload_code_id": tubelet_config.payload_code_id, "sync_code_id": tubelet_config.sync_code_id, "joint_code_mode": tubelet_config.joint_code_mode, "embedding_mode": tubelet_config.embedding_mode,
                    "state_model_id": "synthetic_state_proxy", "state_dim": 5, "key_condition_mode": "method_variant_controlled", "filter_mode": "forward_proxy", "smoother_enabled": False, "phase_state_proxy": result.payload_state, "evidence_state_proxy": result.payload_raw, "confidence_state_proxy": result.state_posterior, "disturbance_state_proxy": result.state_transition_residual, "state_entropy": result.state_entropy, "state_coverage_ratio": result.state_coverage_ratio, "state_matched_count": result.state_matched_count, "state_transition_residual": result.state_transition_residual,
                    "S_payload_raw": result.payload_raw, "S_payload_state": result.payload_state, "S_state_posterior": result.state_posterior, "S_trajectory_observation_placeholder": None, "S_final": result.final_score, "payload_state_gain": round(result.payload_state - result.payload_raw, 6), "key_state_admissibility_status": result.admissibility_status, "negative_state_over_threshold_count": 0, "target_fpr": config["fixed_low_fpr"]["target_fpr"],
                    "trajectory_trace_placeholder": None, "real_video_quality_metrics_placeholder": None, "semantic_consistency_placeholder": None, "placeholder_reason": "fields are reserved for later non_synthetic stages", "replacement_stage": "trajectory_observation_core_probe_or_generative_video_model_probe", "replacement_field_name": "stage_specific_real_field",
                    **flow_evidence_protocol_defaults(
                        negative_family="synthetic_clean_or_attacked_negative" if "negative" in sample.sample_role else "not_applicable_positive",
                        trajectory_source_level="synthetic_state_proxy",
                        flow_state_admissibility_status=result.admissibility_status,
                        claim_support_status="not_supported_synthetic_sanity_only",
                    ),
                }
                record["S_final_conservative"] = conservative_flow_score(record)
                records.append(record)
    return records


def run(output_root: str | Path) -> dict:
    """运行第一阶段并把所有产物写入指定输出目录。"""
    output_root = Path(output_root)
    config = {"protocol": _load_json("configs/protocol/sstw_protocol.json"), "fixed_low_fpr": _load_json("configs/protocol/fixed_low_fpr.json"), "synthetic": _load_json("configs/protocol/synthetic_state_inference.json"), "methods": _load_json("configs/methods/method_variants_synthetic_state.json"), "attacks": _load_json("configs/attacks/synthetic_temporal_attacks.json")}
    raw_records = build_event_records(config)
    thresholds = calibrate_thresholds(raw_records, target_fpr=config["fixed_low_fpr"]["target_fpr"])
    records = apply_thresholds(raw_records, thresholds)
    audit = audit_mechanism(records, target_fpr=config["fixed_low_fpr"]["target_fpr"])
    event_path = output_root / "records" / "event_scores.jsonl"
    state_path = output_root / "records" / "state_trace.jsonl"
    threshold_path = output_root / "thresholds" / "thresholds.json"
    table_path = output_root / "tables" / "synthetic_state_main_table.csv"
    audit_report_path = output_root / "reports" / "synthetic_state_mechanism_audit_report.md"
    decision_path = output_root / "artifacts" / "synthetic_state_inference_decision.json"
    manifest_path = output_root / "artifacts" / "run_manifest.json"
    runtime_config_path = output_root / "artifacts" / "runtime_config.json"
    write_jsonl(event_path, records)
    write_jsonl(state_path, [{"sample_id": record["sample_id"], "method_variant": record["method_variant"], "attack_name": record["attack_name"], "state_entropy": record["state_entropy"], "state_coverage_ratio": record["state_coverage_ratio"], "state_transition_residual": record["state_transition_residual"], "trajectory_source_level": record["trajectory_source_level"], "flow_state_admissibility_status": record["flow_state_admissibility_status"]} for record in records])
    write_json(threshold_path, thresholds)
    write_csv(table_path, build_method_attack_table(records))
    audit_report_path.parent.mkdir(parents=True, exist_ok=True)
    audit_report_path.write_text("# Synthetic State Mechanism Audit\n\n" + json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    implementation_pass = all([event_path.exists(), threshold_path.exists(), table_path.exists(), set(config["methods"]["method_variants"]) == {record["method_variant"] for record in records}, {attack["attack_name"] for attack in config["attacks"]["attacks"]} == {record["attack_name"] for record in records}, all(record["test_time_threshold_update_blocked"] for record in records)])
    decision = build_stage_decision(implementation_pass, bool(audit["mechanism_pass"]), audit)
    write_json(decision_path, decision)
    write_json(runtime_config_path, config)
    write_json(manifest_path, build_run_manifest("synthetic_state_protocol", config, [str(event_path), str(threshold_path), str(table_path), str(decision_path)]))
    return {"output_root": str(output_root), "event_record_count": len(records), "implementation_decision": decision["implementation_decision"], "mechanism_decision": decision["mechanism_decision"], "audit": audit}


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 synthetic state protocol 第一阶段。")
    parser.add_argument("--output-root", default="outputs/runs/synthetic_state_protocol")
    args = parser.parse_args()
    print(json.dumps(run(args.output_root), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
