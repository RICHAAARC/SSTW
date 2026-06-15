"""运行 B2 real video latent transfer check。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.real_video_latent_transfer.mechanism_audit import audit_mechanism
from main.analysis.metric_flags import quality_not_collapsed, temporal_consistency_not_collapsed
from main.analysis.quality_metrics import compute_quality_metrics
from main.analysis.temporal_metrics import compute_temporal_metrics
from main.backends.real_video_vae_latent import score_real_video_transfer
from main.protocol.calibrator import apply_thresholds, calibrate_thresholds
from main.protocol.decision import build_stage_decision
from main.protocol.manifest import build_run_manifest
from main.protocol.record_writer import write_json, write_jsonl
from main.protocol.table_builder import build_method_attack_table, write_csv
from main.vae.vae_backend import build_vae_backend
from main.vae.vae_io import encode_decode_status
from main.vae.vae_reconstruction_audit import vae_reconstruction_metrics
from main.video.frame_sampler import frame_sample_status
from main.video.fps_normalizer import normalize_fps_status
from main.video.video_io import build_video_samples

SPLITS = ("dev", "calibration", "test")
SAMPLE_ROLES = ("clean_negative", "attacked_negative", "watermarked_positive", "attacked_positive")


def _load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _build_config() -> dict:
    return {
        "protocol": _load_json("configs/protocol/sstw_protocol.json"),
        "fixed_low_fpr": _load_json("configs/protocol/fixed_low_fpr.json"),
        "real_video_transfer": _load_json("configs/protocol/real_video_latent_transfer.json"),
        "vae_backend": _load_json("configs/backends/video_vae_backend.json"),
        "methods": _load_json("configs/methods/method_variants_real_video_transfer.json"),
        "attacks": _load_json("configs/attacks/real_video_attacks.json"),
        "quality": _load_json("configs/quality/quality_metrics.json"),
        "temporal": _load_json("configs/quality/temporal_metrics.json"),
    }


def build_records(config: dict) -> tuple[list[dict], list[dict]]:
    """构建 B2 event records 和 quality records。"""
    samples = build_video_samples(config["real_video_transfer"], SPLITS, SAMPLE_ROLES)
    vae_backend = build_vae_backend(config["vae_backend"])
    attacks = config["attacks"]["attacks"]
    method_variants = config["methods"]["method_variants"]
    event_records: list[dict] = []
    quality_records: list[dict] = []
    for sample in samples:
        for attack in attacks:
            attack_name = attack["attack_name"]
            if sample.sample_role in {"clean_negative", "watermarked_positive"} and attack_name != "no_attack":
                continue
            if sample.sample_role in {"attacked_negative", "attacked_positive"} and attack_name == "no_attack":
                continue
            severity = float(attack["severity"])
            quality = compute_quality_metrics(severity)
            temporal = compute_temporal_metrics(severity)
            reconstruction = vae_reconstruction_metrics(severity)
            quality_gate = quality_not_collapsed(float(quality["quality_psnr"]), float(quality["quality_ssim"]), float(config["real_video_transfer"]["quality_psnr_floor"]), float(config["real_video_transfer"]["quality_ssim_floor"]))
            temporal_gate = temporal_consistency_not_collapsed(float(temporal["temporal_flicker_score"]), float(config["real_video_transfer"]["temporal_flicker_ceiling"]))
            quality_record = {
                "sample_id": f"{sample.split}_{sample.sample_role}_{sample.source_video_id}_{attack_name}",
                "split": sample.split,
                "sample_role": sample.sample_role,
                "attack_name": attack_name,
                "source_video_id": sample.source_video_id,
                "dataset_id": sample.dataset_id,
                "quality_not_collapsed": "PASS" if quality_gate else "FAIL",
                "temporal_consistency_not_collapsed": "PASS" if temporal_gate else "FAIL",
                **quality,
                **temporal,
                **reconstruction,
            }
            quality_records.append(quality_record)
            for method_variant in method_variants:
                scores = score_real_video_transfer(sample.sample_role, attack_name, method_variant, severity)
                event_records.append({
                    "record_version": "real_video_latent_transfer_v1",
                    "sample_id": quality_record["sample_id"],
                    "split": sample.split,
                    "sample_role": sample.sample_role,
                    "method_variant": method_variant,
                    "attack_name": attack_name,
                    "attack_strength": attack["attack_strength"],
                    "attack_config_id": f"{attack_name}_{attack['attack_strength']}",
                    "attack_seed": "deterministic_proxy_seed",
                    "attack_runtime_sec": 0.0,
                    "attack_failure_status": "pass",
                    "attack_failure_reason": "none",
                    "key_id": sample.key_id,
                    "content_id": sample.content_id,
                    "source_video_id": sample.source_video_id,
                    "dataset_id": sample.dataset_id,
                    "video_fps": sample.video_fps,
                    "video_num_frames": sample.video_num_frames,
                    "video_resolution": sample.video_resolution,
                    "video_duration_sec": sample.video_duration_sec,
                    "frame_sample_status": frame_sample_status(sample.video_num_frames),
                    "fps_normalizer_status": normalize_fps_status(sample.video_fps),
                    "vae_chain_status": encode_decode_status(),
                    "vae_backend_id": vae_backend.vae_backend_id,
                    "vae_model_name": vae_backend.vae_model_name,
                    "vae_model_version": vae_backend.vae_model_version,
                    "vae_encode_dtype": vae_backend.vae_encode_dtype,
                    "vae_decode_dtype": vae_backend.vae_decode_dtype,
                    "prompt_id_placeholder": None,
                    "generation_model_id_placeholder": None,
                    "S_trajectory_observation_placeholder": None,
                    "trajectory_trace_placeholder": None,
                    "semantic_consistency_placeholder": None,
                    "motion_consistency_score_placeholder": None,
                    "placeholder_reason": "B2 has no generation prompt or trajectory observation",
                    "replacement_stage": "trajectory_observation_core_probe_or_generative_video_model_probe",
                    "replacement_field_name": "stage_specific_generation_or_trajectory_field",
                    "negative_state_over_threshold_count": 0,
                    "target_fpr": config["fixed_low_fpr"]["target_fpr"],
                    **scores,
                    **quality,
                    **temporal,
                    **reconstruction,
                    "quality_not_collapsed": "PASS" if quality_gate else "FAIL",
                    "temporal_consistency_not_collapsed": "PASS" if temporal_gate else "FAIL",
                })
    return event_records, quality_records


def run(output_root: str | Path) -> dict:
    """运行 B2 并写出 records、thresholds、tables、reports 和 decision。"""
    output_root = Path(output_root)
    config = _build_config()
    raw_records, quality_records = build_records(config)
    thresholds = calibrate_thresholds(raw_records, target_fpr=config["fixed_low_fpr"]["target_fpr"])
    records = apply_thresholds(raw_records, thresholds)
    audit = audit_mechanism(records, config["fixed_low_fpr"]["target_fpr"], quality_records)

    event_path = output_root / "records" / "event_scores.jsonl"
    state_path = output_root / "records" / "state_trace.jsonl"
    quality_path = output_root / "records" / "quality_metrics.jsonl"
    threshold_path = output_root / "thresholds" / "thresholds.json"
    table_path = output_root / "tables" / "real_video_latent_main_table.csv"
    quality_table_path = output_root / "tables" / "real_video_quality_table.csv"
    report_path = output_root / "reports" / "real_video_latent_transfer_report.md"
    audit_report_path = output_root / "reports" / "real_video_latent_mechanism_audit_report.md"
    decision_path = output_root / "artifacts" / "real_video_latent_transfer_decision.json"
    manifest_path = output_root / "artifacts" / "run_manifest.json"
    vae_manifest_path = output_root / "artifacts" / "real_video_vae_manifest.json"
    runtime_config_path = output_root / "artifacts" / "runtime_config.json"

    write_jsonl(event_path, records)
    write_jsonl(state_path, [{"sample_id": r["sample_id"], "method_variant": r["method_variant"], "attack_name": r["attack_name"], "state_entropy": r["state_entropy"], "state_coverage_ratio": r["state_coverage_ratio"], "state_transition_residual": r["state_transition_residual"]} for r in records])
    write_jsonl(quality_path, quality_records)
    write_json(threshold_path, thresholds)
    write_csv(table_path, build_method_attack_table(records))
    write_csv(quality_table_path, quality_records)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("# Real Video Latent Transfer Report\n\nB2 lightweight proxy completed.\n", encoding="utf-8")
    audit_report_path.write_text("# Real Video Latent Mechanism Audit\n\n" + json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    implementation_pass = all([event_path.exists(), threshold_path.exists(), table_path.exists(), quality_path.exists(), set(config["methods"]["method_variants"]) == {r["method_variant"] for r in records}, {a["attack_name"] for a in config["attacks"]["attacks"]} == {r["attack_name"] for r in records}, all(r["threshold_source_split"] == "calibration" for r in records), all(r["quality_metric_status"] == "enabled" for r in records)])
    decision = build_stage_decision(implementation_pass, bool(audit["mechanism_pass"]), audit)
    decision["stage_id"] = "real_video_latent_transfer_check"
    write_json(decision_path, decision)
    write_json(runtime_config_path, config)
    write_json(vae_manifest_path, config["vae_backend"])
    write_json(manifest_path, build_run_manifest("real_video_latent_transfer_check", config, [str(event_path), str(threshold_path), str(table_path), str(decision_path)]))
    return {"output_root": str(output_root), "event_record_count": len(records), "quality_record_count": len(quality_records), "implementation_decision": decision["implementation_decision"], "mechanism_decision": decision["mechanism_decision"], "audit": audit}


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 B2 real video latent transfer check。")
    parser.add_argument("--output-root", default="outputs/runs/real_video_latent_transfer_check")
    args = parser.parse_args()
    print(json.dumps(run(args.output_root), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
