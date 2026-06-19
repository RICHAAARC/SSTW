"""验证 formal motion threshold calibration。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.generative_video_model_probe.motion_threshold_calibration import run_motion_threshold_calibration
from experiments.generative_video_model_probe.pilot_claim_gate import build_small_scale_claim_pilot_audit
from main.protocol.record_writer import write_json, write_jsonl


def _write_formal_motion_records(run_root: Path, negative_count: int, positive_count: int, ambiguous_count: int = 0, source_split: str = "calibration") -> None:
    """写出轻量 formal motion records, 用于验证 calibration 阶段而不依赖真实视频解码。"""
    generation_rows = []
    formal_rows = []
    for index in range(negative_count):
        generation_rows.append({
            "generation_model_id": "model",
            "prompt_id": f"static_{index}",
            "seed_id": "seed",
            "generation_status": "success",
            "prompt_suite_role": "calibration_negative_static" if source_split == "calibration" else "main",
            "motion_pattern_id": "negative_static",
            "split": source_split,
            "trajectory_trace_id": f"trace_static_{index}",
        })
        formal_rows.append({
            "generation_model_id": "model",
            "prompt_id": f"static_{index}",
            "seed_id": "seed",
            "trajectory_trace_id": f"trace_static_{index}",
            "video_decode_status": "ready",
            "motion_delta_score": 0.00005 + index * 0.000001,
            "temporal_flicker_score": 0.01,
            "split": source_split,
        })
    for index in range(positive_count):
        generation_rows.append({
            "generation_model_id": "model",
            "prompt_id": f"motion_{index}",
            "seed_id": "seed",
            "generation_status": "success",
            "prompt_suite_role": "calibration_positive_motion" if source_split == "calibration" else "main",
            "motion_pattern_id": "moving_object",
            "motion_calibration_role": "positive_motion",
            "split": source_split,
            "trajectory_trace_id": f"trace_motion_{index}",
        })
        formal_rows.append({
            "generation_model_id": "model",
            "prompt_id": f"motion_{index}",
            "seed_id": "seed",
            "trajectory_trace_id": f"trace_motion_{index}",
            "video_decode_status": "ready",
            "motion_delta_score": 0.001 + index * 0.00001,
            "temporal_flicker_score": 0.02,
            "split": source_split,
        })
    for index in range(ambiguous_count):
        generation_rows.append({
            "generation_model_id": "model",
            "prompt_id": f"ambiguous_{index}",
            "seed_id": "seed",
            "generation_status": "success",
            "prompt_suite_role": "calibration_ambiguous_low_motion" if source_split == "calibration" else "main",
            "motion_pattern_id": "ambiguous_low_motion",
            "motion_calibration_role": "ambiguous_low_motion",
            "split": source_split,
            "trajectory_trace_id": f"trace_ambiguous_{index}",
        })
        formal_rows.append({
            "generation_model_id": "model",
            "prompt_id": f"ambiguous_{index}",
            "seed_id": "seed",
            "trajectory_trace_id": f"trace_ambiguous_{index}",
            "video_decode_status": "ready",
            "motion_delta_score": 0.0002 + index * 0.000002,
            "temporal_flicker_score": 0.015,
            "split": source_split,
        })
    write_jsonl(run_root / "records" / "generation_records.jsonl", generation_rows)
    write_jsonl(run_root / "records" / "formal_quality_motion_semantic_records.jsonl", formal_rows)


@pytest.mark.quick
def test_motion_threshold_calibration_reports_insufficient_without_calibration_split(tmp_path: Path) -> None:
    """没有独立 calibration split 时, calibration runner 必须继续阻塞 claim。"""
    run_root = tmp_path / "run"
    _write_formal_motion_records(run_root, negative_count=2, positive_count=4, ambiguous_count=1, source_split="pilot_main")

    audit = run_motion_threshold_calibration(run_root)
    records = [json.loads(line) for line in (run_root / "records" / "motion_threshold_calibration_records.jsonl").read_text(encoding="utf-8").splitlines()]

    assert audit["motion_threshold_calibration_decision"] == "INSUFFICIENT_SAMPLE"
    assert audit["motion_threshold_calibration_ready"] is False
    assert audit["motion_threshold_id"] == "motion_delta_heuristic_v1"
    assert audit["motion_threshold_source_split"] == "heuristic_precalibration"
    assert audit["motion_threshold_calibration_required"] is True
    assert "negative_static_calibration_count_below_min" in audit["motion_threshold_calibration_missing_reasons"]
    assert "ambiguous_low_motion_calibration_count_below_min" in audit["motion_threshold_calibration_missing_reasons"]
    assert records
    assert (run_root / "tables" / "motion_threshold_calibration_table.csv").exists()
    assert (run_root / "thresholds" / "motion_threshold_calibration_threshold.json").exists()
    assert (run_root / "artifacts" / "motion_threshold_calibration_decision.json").exists()
    assert (run_root / "reports" / "motion_threshold_calibration_report.md").exists()


@pytest.mark.quick
def test_motion_threshold_calibration_passes_with_governed_calibration_split(tmp_path: Path) -> None:
    """样本角色和 calibration split 足够时, runner 才能冻结 calibrated threshold。"""
    run_root = tmp_path / "run"
    _write_formal_motion_records(run_root, negative_count=128, positive_count=64, ambiguous_count=32, source_split="calibration")

    audit = run_motion_threshold_calibration(run_root)

    assert audit["motion_threshold_calibration_decision"] == "PASS"
    assert audit["motion_threshold_calibration_ready"] is True
    assert audit["motion_threshold_id"] == "motion_delta_calibrated_v1"
    assert audit["motion_threshold_source_split"] == "calibration"
    assert audit["negative_static_calibration_count"] == 128
    assert audit["positive_motion_calibration_count"] == 64
    assert audit["ambiguous_low_motion_calibration_count"] == 32
    assert audit["motion_threshold_calibration_required"] is False
    assert audit["test_time_threshold_update_blocked"] is True
    assert audit["claim_support_status"] == "motion_threshold_calibrated"


@pytest.mark.quick
def test_pilot_gate_unblocks_only_when_motion_calibration_passes(tmp_path: Path) -> None:
    """pilot gate 只有在 motion calibration artifact PASS 后才允许 claim_support_status 进入 supported。"""
    run_root = tmp_path / "run"
    generation_rows = []
    for prompt_index in range(8):
        for seed_index in range(2):
            generation_rows.append({
                "generation_model_id": "model",
                "prompt_id": f"prompt_{prompt_index}",
                "seed_id": f"seed_{seed_index}",
                "generation_status": "success",
                "trajectory_trace_id": f"trace_{prompt_index}_{seed_index}",
            })
    write_jsonl(run_root / "records" / "generation_records.jsonl", generation_rows)
    write_jsonl(run_root / "records" / "quality_motion_semantic_proxy_records.jsonl", [{
        "visual_quality_proxy_status": "ready",
        "motion_consistency_proxy_status": "ready",
    }])
    write_json(run_root / "artifacts" / "generative_video_mechanism_postprocess_decision.json", {
        "details": {"fixed_low_fpr_proxy_pass": True, "quality_motion_semantic_proxy_pass": True},
    })
    write_json(run_root / "thresholds" / "mechanism_proxy_thresholds.json", {"controlled_negative_fpr": 0.0})
    write_jsonl(run_root / "records" / "small_scale_claim_pilot_matrix_records.jsonl", [{
        "attack_name": attack,
        "negative_family": family,
        "method_variant": method,
        "path_marginal_gain_at_fixed_fpr": 0.1,
        "negative_tail_status": "not_inflated",
        "wrong_key_score_separation_passed": True,
        "wrong_sampler_replay_control_not_equivalent": True,
        "replay_uncertainty_mean": 0.02,
        "sample_role": "wrong_sampler_replay",
        "decision": "not_equivalent",
    } for attack in ("attack_a", "attack_b", "attack_c") for family in ("negative_static", "wrong_key", "wrong_sampler_replay", "cross_prompt") for method in (
        "m0", "m1", "m2", "m3", "m4", "m5"
    )])
    write_jsonl(run_root / "records" / "runtime_attack_records.jsonl", [{"attack_name": "attack_b", "attack_runtime_status": "ready"}])
    write_jsonl(run_root / "records" / "runtime_detection_records.jsonl", [{"runtime_detection_status": "ready"}])
    write_json(run_root / "artifacts" / "motion_threshold_calibration_decision.json", {
        "motion_threshold_calibration_decision": "PASS",
        "motion_threshold_calibration_ready": True,
        "motion_threshold_id": "motion_delta_calibrated_v1",
        "motion_threshold_source_split": "calibration",
    })

    summary = build_small_scale_claim_pilot_audit(run_root)

    assert summary["missing_pilot_requirements"] == []
    assert summary["claim_support_status"] == "supported_by_small_scale_claim_pilot_records"
    assert summary["pilot_gate_decision"] == "PASS"
    assert summary["motion_threshold_calibration_required"] is False
    assert summary["motion_threshold_id"] == "motion_delta_calibrated_v1"
