"""验证 small-scale claim pilot gate 自动审计。"""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.generative_video_model_probe.pilot_claim_gate import build_small_scale_claim_pilot_audit, write_small_scale_claim_pilot_audit
from experiments.generative_video_model_probe.pilot_matrix_postprocess import write_pilot_matrix_postprocess
from main.protocol.record_writer import write_json, write_jsonl
from scripts.check_results.small_scale_claim_pilot_result_checker import check_small_scale_claim_pilot_results


def _write_generation_records(run_root: Path, prompt_count: int = 8, seed_count: int = 2) -> None:
    """写出满足 prompt / seed 覆盖的轻量 generation records。"""
    rows = []
    for prompt_index in range(prompt_count):
        for seed_index in range(seed_count):
            rows.append({
                "generation_model_id": "model",
                "prompt_id": f"prompt_{prompt_index}",
                "seed_id": f"seed_{seed_index}",
                "generation_status": "success",
                "video_path": f"video_{prompt_index}_{seed_index}.mp4",
                "video_sha256": "digest",
                "trajectory_trace_id": f"trace_{prompt_index}_{seed_index}",
            })
    write_jsonl(run_root / "records" / "generation_records.jsonl", rows)


def _write_proxy_postprocess(run_root: Path) -> None:
    """写出当前 Colab postprocess 已能提供的 proxy artifacts。"""
    write_json(run_root / "artifacts" / "generative_video_mechanism_postprocess_decision.json", {
        "stage_id": "generative_video_mechanism_postprocess",
        "mechanism_postprocess_decision": "PASS",
        "mechanism_decision": "FAIL",
        "details": {
            "trajectory_gain_confirmed_by_proxy": True,
            "fixed_low_fpr_proxy_pass": True,
            "quality_motion_semantic_proxy_pass": True,
            "formal_claim_status": "blocked_by_formal_motion_consistency",
        },
    })
    write_json(run_root / "thresholds" / "mechanism_proxy_thresholds.json", {
        "controlled_negative_fpr": 0.0,
        "target_fpr": 0.01,
    })
    write_json(run_root / "artifacts" / "formal_quality_motion_semantic_decision.json", {
        "formal_metric_claim_status": "blocked_by_formal_motion_consistency",
    })
    write_jsonl(run_root / "records" / "quality_motion_semantic_proxy_records.jsonl", [{
        "visual_quality_proxy_status": "ready",
        "motion_consistency_proxy_status": "ready",
    }])


def _write_trajectory_records(run_root: Path, prompt_count: int = 8, seed_count: int = 2) -> None:
    """写出可支撑 pilot matrix proxy 的轻量 trajectory records。"""
    rows = []
    for prompt_index in range(prompt_count):
        for seed_index in range(seed_count):
            trace_id = f"trace_{prompt_index}_{seed_index}"
            for step_index in range(4):
                rows.append({
                    "trajectory_trace_id": trace_id,
                    "trajectory_step_index": step_index,
                    "latent_norm": 100.0 - 8.0 * step_index,
                    "latent_mean": 0.01 * step_index,
                    "latent_std": 0.9 - 0.05 * step_index,
                })
    write_jsonl(run_root / "records" / "trajectory_trace.jsonl", rows)


@pytest.mark.quick
def test_small_scale_claim_pilot_checker_reports_missing_matrix(tmp_path: Path) -> None:
    """当前只有生成和 proxy 证据时, pilot gate 必须明确报告矩阵缺口。"""
    run_root = tmp_path / "generative_video_model_probe_colab"
    _write_generation_records(run_root)
    _write_proxy_postprocess(run_root)
    write_jsonl(run_root / "records" / "mechanism_score_records.jsonl", [
        {
            "method_variant": method,
            "attack_name": "postprocess_no_attack",
            "S_final": 0.8,
            "S_trajectory_observation": 0.8,
        }
        for method in (
            "key_conditioned_state_space_with_trajectory",
            "generic_state_space_with_trajectory",
            "explicit_dtw_temporal_alignment",
            "explicit_frame_matching_temporal_registration",
        )
    ])
    write_jsonl(run_root / "records" / "controlled_negative_records.jsonl", [{
        "control_name": "trajectory_key_agnostic_control",
        "S_final": 0.2,
    }])

    summary = check_small_scale_claim_pilot_results(run_root)

    assert summary["pilot_gate_decision"] == "FAIL"
    assert summary["claim_support_status"] == "workflow_progression_only"
    assert summary["prompt_count"] == 8
    assert summary["seed_per_prompt_min"] == 2
    assert "attack_matrix_ready" in summary["missing_pilot_requirements"]
    assert "negative_family_ready" in summary["missing_pilot_requirements"]
    assert "wrong_sampler_replay_ready" in summary["missing_pilot_requirements"]
    assert summary["motion_threshold_id"] == "motion_delta_heuristic_v1"
    assert summary["motion_threshold_calibration_required"] is True
    assert summary["test_time_threshold_update_blocked"] is True


@pytest.mark.quick
def test_small_scale_claim_pilot_gate_writes_governed_artifacts(tmp_path: Path) -> None:
    """pilot gate 写出结果必须可由 records、table、decision 和 report 复核。"""
    run_root = tmp_path / "generative_video_model_probe_colab"
    _write_generation_records(run_root)
    _write_proxy_postprocess(run_root)
    write_jsonl(run_root / "records" / "mechanism_score_records.jsonl", [])
    write_jsonl(run_root / "records" / "controlled_negative_records.jsonl", [])

    summary = write_small_scale_claim_pilot_audit(run_root)
    dry_run_summary = build_small_scale_claim_pilot_audit(run_root)

    assert summary["pilot_gate_decision"] == dry_run_summary["pilot_gate_decision"]
    assert (run_root / "records" / "small_scale_claim_pilot_gate_records.jsonl").exists()
    assert (run_root / "tables" / "small_scale_claim_pilot_gate_table.csv").exists()
    assert (run_root / "artifacts" / "small_scale_claim_pilot_gate_decision.json").exists()
    assert (run_root / "reports" / "small_scale_claim_pilot_gate_report.md").exists()


@pytest.mark.quick
def test_pilot_matrix_postprocess_fills_proxy_matrix_but_keeps_calibration_block(tmp_path: Path) -> None:
    """pilot matrix postprocess 可补齐 proxy 矩阵, 但不能越过 motion threshold calibration gate。"""
    run_root = tmp_path / "generative_video_model_probe_colab"
    _write_generation_records(run_root)
    _write_trajectory_records(run_root)
    _write_proxy_postprocess(run_root)
    write_jsonl(run_root / "records" / "mechanism_score_records.jsonl", [])
    write_jsonl(run_root / "records" / "controlled_negative_records.jsonl", [])

    matrix_audit = write_pilot_matrix_postprocess(run_root)
    summary = check_small_scale_claim_pilot_results(run_root)

    assert matrix_audit["pilot_matrix_postprocess_decision"] == "PASS"
    assert matrix_audit["pilot_matrix_attack_count"] == 3
    assert matrix_audit["pilot_matrix_method_variant_count"] == 6
    assert matrix_audit["pilot_matrix_negative_family_count"] == 4
    assert summary["missing_pilot_requirements"] == []
    assert summary["pilot_gate_decision"] == "FAIL"
    assert summary["claim_support_status"] == "blocked_until_motion_threshold_calibration"
    assert summary["path_marginal_gain_at_fixed_fpr"] is not None
    assert summary["replay_uncertainty_mean"] is not None
