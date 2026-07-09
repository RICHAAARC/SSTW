"""验证 formal motion threshold calibration。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.generative_video_model_probe.motion_threshold_calibration import run_motion_threshold_calibration
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
    assert (run_root / "artifacts" / "prompt_contamination_audit.json").exists()
    assert (run_root / "artifacts" / "threshold_stability_audit.json").exists()


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
    assert audit["claim_support_status"] == "motion_threshold_engineering_calibrated"
    assert audit["negative_static_contamination_status"] == "none_detected"
    assert audit["motion_threshold_selection_strategy"] == "prompt_aware_robust_quantile_p95"


@pytest.mark.quick
def test_motion_threshold_calibration_isolates_contaminated_negative_tail(tmp_path: Path) -> None:
    """异常高运动的 negative_static 只进入污染审计, 不直接抬高主阈值。"""
    run_root = tmp_path / "run"
    generation_rows = []
    formal_rows = []

    for index in range(128):
        score = 0.001 + index * 0.00001
        if index >= 120:
            score = 0.02 + (index - 120) * 0.001
        generation_rows.append({
            "generation_model_id": "model",
            "prompt_id": f"static_{index}",
            "seed_id": "seed",
            "generation_status": "success",
            "prompt_suite_role": "motion_calibration_negative_static",
            "motion_calibration_role": "negative_static",
            "split": "calibration",
            "trajectory_trace_id": f"trace_static_{index}",
        })
        formal_rows.append({
            "generation_model_id": "model",
            "prompt_id": f"static_{index}",
            "seed_id": "seed",
            "trajectory_trace_id": f"trace_static_{index}",
            "video_decode_status": "ready",
            "motion_delta_score": score,
            "temporal_flicker_score": 0.01,
            "split": "calibration",
        })

    for index in range(64):
        generation_rows.append({
            "generation_model_id": "model",
            "prompt_id": f"motion_{index}",
            "seed_id": "seed",
            "generation_status": "success",
            "prompt_suite_role": "motion_calibration_positive_motion",
            "motion_calibration_role": "positive_motion",
            "split": "calibration",
            "trajectory_trace_id": f"trace_motion_{index}",
        })
        formal_rows.append({
            "generation_model_id": "model",
            "prompt_id": f"motion_{index}",
            "seed_id": "seed",
            "trajectory_trace_id": f"trace_motion_{index}",
            "video_decode_status": "ready",
            "motion_delta_score": 0.004 + index * 0.00001,
            "temporal_flicker_score": 0.01,
            "split": "calibration",
        })

    for index in range(32):
        generation_rows.append({
            "generation_model_id": "model",
            "prompt_id": f"ambiguous_{index}",
            "seed_id": "seed",
            "generation_status": "success",
            "prompt_suite_role": "motion_calibration_ambiguous_low_motion",
            "motion_calibration_role": "ambiguous_low_motion",
            "split": "calibration",
            "trajectory_trace_id": f"trace_ambiguous_{index}",
        })
        formal_rows.append({
            "generation_model_id": "model",
            "prompt_id": f"ambiguous_{index}",
            "seed_id": "seed",
            "trajectory_trace_id": f"trace_ambiguous_{index}",
            "video_decode_status": "ready",
            "motion_delta_score": 0.0006 + index * 0.000005,
            "temporal_flicker_score": 0.01,
            "split": "calibration",
        })

    write_jsonl(run_root / "records" / "generation_records.jsonl", generation_rows)
    write_jsonl(run_root / "records" / "formal_quality_motion_semantic_records.jsonl", formal_rows)

    audit = run_motion_threshold_calibration(run_root)

    assert audit["motion_threshold_calibration_decision"] == "PASS"
    assert audit["negative_static_contamination_status"] == "suspected"
    assert audit["negative_static_contamination_count"] == 8
    assert audit["negative_static_clean_calibration_count"] == 120
    assert audit["motion_threshold_selection_strategy"] == "prompt_aware_robust_quantile_p95"
    assert audit["motion_delta_threshold"] < audit["conservative_motion_delta_threshold"]
    assert audit["positive_motion_pass_rate_at_threshold"] == 1.0


@pytest.mark.quick
def test_motion_threshold_calibration_fails_when_positive_motion_not_separable(tmp_path: Path) -> None:
    """当 positive_motion 与 negative_static 分数重叠时, calibration 不能被误判为 PASS。"""
    run_root = tmp_path / "run"
    generation_rows = []
    formal_rows = []

    for index in range(128):
        generation_rows.append({
            "generation_model_id": "model",
            "prompt_id": f"static_{index}",
            "seed_id": "seed",
            "generation_status": "success",
            "prompt_suite_role": "motion_calibration_negative_static",
            "motion_calibration_role": "negative_static",
            "split": "calibration",
            "trajectory_trace_id": f"trace_static_{index}",
        })
        formal_rows.append({
            "generation_model_id": "model",
            "prompt_id": f"static_{index}",
            "seed_id": "seed",
            "trajectory_trace_id": f"trace_static_{index}",
            "video_decode_status": "ready",
            "motion_delta_score": 0.001 + index * 0.00007,
            "temporal_flicker_score": 0.01,
            "split": "calibration",
        })

    for index in range(64):
        generation_rows.append({
            "generation_model_id": "model",
            "prompt_id": f"motion_{index}",
            "seed_id": "seed",
            "generation_status": "success",
            "prompt_suite_role": "motion_calibration_positive_motion",
            "motion_calibration_role": "positive_motion",
            "split": "calibration",
            "trajectory_trace_id": f"trace_motion_{index}",
        })
        score = 0.0015 + index * 0.00003
        formal_rows.append({
            "generation_model_id": "model",
            "prompt_id": f"motion_{index}",
            "seed_id": "seed",
            "trajectory_trace_id": f"trace_motion_{index}",
            "video_decode_status": "ready",
            "motion_delta_score": score,
            "temporal_flicker_score": 0.01,
            "split": "calibration",
        })

    for index in range(32):
        generation_rows.append({
            "generation_model_id": "model",
            "prompt_id": f"ambiguous_{index}",
            "seed_id": "seed",
            "generation_status": "success",
            "prompt_suite_role": "motion_calibration_ambiguous_low_motion",
            "motion_calibration_role": "ambiguous_low_motion",
            "split": "calibration",
            "trajectory_trace_id": f"trace_ambiguous_{index}",
        })
        formal_rows.append({
            "generation_model_id": "model",
            "prompt_id": f"ambiguous_{index}",
            "seed_id": "seed",
            "trajectory_trace_id": f"trace_ambiguous_{index}",
            "video_decode_status": "ready",
            "motion_delta_score": 0.0004 + index * 0.000002,
            "temporal_flicker_score": 0.01,
            "split": "calibration",
        })

    write_jsonl(run_root / "records" / "generation_records.jsonl", generation_rows)
    write_jsonl(run_root / "records" / "formal_quality_motion_semantic_records.jsonl", formal_rows)

    audit = run_motion_threshold_calibration(run_root)

    assert audit["motion_threshold_calibration_decision"] == "FAIL_NOT_SEPARABLE"
    assert audit["motion_threshold_calibration_ready"] is False
    assert audit["motion_threshold_calibration_required"] is True
    assert audit["claim_support_status"] == "blocked_until_motion_threshold_calibration"
    assert "positive_motion_pass_rate_below_min" in audit["motion_threshold_calibration_missing_reasons"]
    assert "positive_negative_motion_score_overlap" in audit["motion_threshold_calibration_missing_reasons"]
    assert audit["positive_motion_pass_rate_at_threshold"] < audit["minimum_positive_motion_pass_rate_at_threshold"]


@pytest.mark.quick
def test_motion_threshold_calibration_prefers_focus_score_when_available(tmp_path: Path) -> None:
    """当 formal records 写出 focus score 时, calibration 应优先使用局部运动分数。"""
    run_root = tmp_path / "run"
    generation_rows = []
    formal_rows = []

    for index in range(128):
        generation_rows.append({
            "generation_model_id": "model",
            "prompt_id": f"static_{index}",
            "seed_id": "seed",
            "generation_status": "success",
            "prompt_suite_role": "motion_calibration_negative_static",
            "motion_calibration_role": "negative_static",
            "split": "calibration",
            "trajectory_trace_id": f"trace_static_{index}",
        })
        formal_rows.append({
            "generation_model_id": "model",
            "prompt_id": f"static_{index}",
            "seed_id": "seed",
            "trajectory_trace_id": f"trace_static_{index}",
            "video_decode_status": "ready",
            "motion_delta_score": 0.01,
            "motion_delta_focus_score": 0.0001 + index * 0.000001,
            "temporal_flicker_score": 0.01,
            "split": "calibration",
        })

    for index in range(64):
        generation_rows.append({
            "generation_model_id": "model",
            "prompt_id": f"motion_{index}",
            "seed_id": "seed",
            "generation_status": "success",
            "prompt_suite_role": "motion_calibration_positive_motion",
            "motion_calibration_role": "positive_motion",
            "split": "calibration",
            "trajectory_trace_id": f"trace_motion_{index}",
        })
        formal_rows.append({
            "generation_model_id": "model",
            "prompt_id": f"motion_{index}",
            "seed_id": "seed",
            "trajectory_trace_id": f"trace_motion_{index}",
            "video_decode_status": "ready",
            "motion_delta_score": 0.0002,
            "motion_delta_focus_score": 0.004 + index * 0.00001,
            "temporal_flicker_score": 0.01,
            "split": "calibration",
        })

    for index in range(32):
        generation_rows.append({
            "generation_model_id": "model",
            "prompt_id": f"ambiguous_{index}",
            "seed_id": "seed",
            "generation_status": "success",
            "prompt_suite_role": "motion_calibration_ambiguous_low_motion",
            "motion_calibration_role": "ambiguous_low_motion",
            "split": "calibration",
            "trajectory_trace_id": f"trace_ambiguous_{index}",
        })
        formal_rows.append({
            "generation_model_id": "model",
            "prompt_id": f"ambiguous_{index}",
            "seed_id": "seed",
            "trajectory_trace_id": f"trace_ambiguous_{index}",
            "video_decode_status": "ready",
            "motion_delta_score": 0.0002,
            "motion_delta_focus_score": 0.0003 + index * 0.000001,
            "temporal_flicker_score": 0.01,
            "split": "calibration",
        })

    write_jsonl(run_root / "records" / "generation_records.jsonl", generation_rows)
    write_jsonl(run_root / "records" / "formal_quality_motion_semantic_records.jsonl", formal_rows)

    audit = run_motion_threshold_calibration(run_root)
    records = [json.loads(line) for line in (run_root / "records" / "motion_threshold_calibration_records.jsonl").read_text(encoding="utf-8").splitlines()]

    assert audit["motion_threshold_calibration_decision"] == "PASS"
    assert audit["motion_calibration_score_name"] == "motion_delta_focus_score_preferred"
    assert audit["positive_motion_pass_rate_at_threshold"] == 1.0
    assert records[0]["motion_calibration_score_name"] == "motion_delta_focus_score"
    assert records[0]["motion_calibration_score"] == records[0]["motion_delta_focus_score"]
