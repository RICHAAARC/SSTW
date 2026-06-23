"""验证 small-scale claim pilot gate 自动审计。"""

from __future__ import annotations

import json
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
            "formal_claim_status": "ready",
        },
    })
    write_json(run_root / "thresholds" / "mechanism_proxy_thresholds.json", {
        "controlled_negative_fpr": 0.0,
        "target_fpr": 0.01,
    })
    write_json(run_root / "artifacts" / "formal_quality_motion_semantic_decision.json", {
        "formal_metric_claim_status": "ready",
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


def _write_formal_metric_records(
    run_root: Path,
    prompt_count: int = 8,
    seed_count: int = 2,
    failed_prompt_index: int | None = None,
    failed_seed_index: int | None = None,
) -> None:
    """写出 formal metric records, 可指定 1 个低运动失败样本。"""
    rows = []
    for prompt_index in range(prompt_count):
        for seed_index in range(seed_count):
            motion_ready = not (
                failed_prompt_index == prompt_index
                and failed_seed_index == seed_index
            )
            rows.append({
                "record_version": "generative_video_formal_quality_motion_semantic_v1",
                "generation_model_id": "model",
                "prompt_id": f"prompt_{prompt_index}",
                "seed_id": f"seed_{seed_index}",
                "trajectory_trace_id": f"trace_{prompt_index}_{seed_index}",
                "motion_claim_role": "positive_motion",
                "formal_visual_quality_ready": True,
                "formal_motion_consistency_ready": motion_ready,
                "formal_semantic_consistency_ready": True,
                "formal_motion_gate_policy": "positive_motion_requires_min_delta",
                "formal_motion_gate_failure_reason": "none" if motion_ready else "motion_delta_below_min",
                "formal_metric_blocking_reason": "none" if motion_ready else "formal_motion_consistency_not_ready",
                "formal_metric_result_used_for_claim": motion_ready,
            })
    write_jsonl(run_root / "records" / "formal_quality_motion_semantic_records.jsonl", rows)
    write_json(run_root / "artifacts" / "formal_quality_motion_semantic_decision.json", {
        "formal_metric_record_count": prompt_count * seed_count,
        "formal_motion_consistency_ready_count": sum(
            1 for row in rows if row["formal_motion_consistency_ready"] is True
        ),
        "formal_motion_consistency_blocked_count": sum(
            1 for row in rows if row["formal_motion_consistency_ready"] is not True
        ),
        "formal_metric_claim_status": "ready" if all(row["formal_metric_result_used_for_claim"] for row in rows) else "blocked_by_formal_motion_consistency",
    })


def _write_motion_calibration_ready(run_root: Path) -> None:
    """写出已冻结的工程 motion threshold calibration artifact。"""
    write_json(run_root / "artifacts" / "motion_threshold_calibration_decision.json", {
        "motion_threshold_calibration_decision": "PASS",
        "motion_threshold_calibration_ready": True,
        "motion_threshold_id": "motion_delta_calibrated_v1",
        "motion_threshold_source_split": "calibration",
    })


def _write_motion_calibration_ready_with_bom(run_root: Path) -> None:
    """写出带 UTF-8 BOM 的 calibration artifact, 模拟 Windows PowerShell 重写后的 Drive 文件。"""
    path = run_root / "artifacts" / "motion_threshold_calibration_decision.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "motion_threshold_calibration_decision": "PASS",
            "motion_threshold_calibration_ready": True,
            "motion_threshold_id": "motion_delta_calibrated_v1",
            "motion_threshold_source_split": "calibration",
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8-sig",
    )


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
    records = [
        json.loads(line)
        for line in (run_root / "records" / "small_scale_claim_pilot_matrix_records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    required_protocol_fields = {
        "negative_family",
        "sampler_signature_placeholder",
        "trajectory_source_level",
        "S_path_inv",
        "S_velocity",
        "S_final_conservative",
        "path_marginal_gain_at_fixed_fpr",
        "replay_uncertainty_mean",
        "flow_state_admissibility_status",
        "claim_support_status",
    }
    assert records
    assert all(required_protocol_fields <= set(record) for record in records)
    assert all(record["S_final_conservative"] is not None for record in records)
    assert any(record["wrong_sampler_replay_status"] == "replay_rejected" for record in records)


@pytest.mark.quick
def test_formal_motion_failed_sample_is_excluded_from_motion_claim_gate(tmp_path: Path) -> None:
    """formal motion 失败样本必须保留记录, 但不能计入 motion claim 证据。"""
    run_root = tmp_path / "generative_video_model_probe_colab"
    _write_generation_records(run_root)
    _write_trajectory_records(run_root)
    _write_proxy_postprocess(run_root)
    _write_motion_calibration_ready(run_root)
    _write_formal_metric_records(run_root, failed_prompt_index=0, failed_seed_index=1)
    write_jsonl(run_root / "records" / "mechanism_score_records.jsonl", [])
    write_jsonl(run_root / "records" / "controlled_negative_records.jsonl", [])

    matrix_audit = write_pilot_matrix_postprocess(run_root)
    summary = build_small_scale_claim_pilot_audit(run_root)

    assert matrix_audit["pilot_matrix_postprocess_decision"] == "PASS"
    assert matrix_audit["motion_claim_eligible_generation_count"] == 15
    assert matrix_audit["motion_claim_excluded_generation_count"] == 1
    assert matrix_audit["pilot_matrix_record_count"] == 15 * 3 * (6 + 4)
    assert summary["pilot_gate_decision"] == "FAIL"
    assert summary["claim_support_status"] == "workflow_progression_only"
    assert summary["formal_motion_claim_status"] == "ready_with_formal_motion_exclusions"
    assert summary["motion_claim_eligible_generation_count"] == 15
    assert summary["motion_claim_excluded_generation_count"] == 1
    assert summary["seed_per_prompt_min"] == 1
    assert "formal_motion_claim_ready" not in summary["missing_pilot_requirements"]
    assert "seed_coverage_ready" in summary["missing_pilot_requirements"]


@pytest.mark.quick
def test_validation_like_pilot_allows_single_formal_motion_exclusion_when_coverage_remains(tmp_path: Path) -> None:
    """当剔除 1 个低运动样本后仍满足覆盖率时, pilot gate 不应被旧失败样本整体阻断。"""
    run_root = tmp_path / "generative_video_model_probe_colab"
    _write_generation_records(run_root, seed_count=3)
    _write_trajectory_records(run_root, seed_count=3)
    _write_proxy_postprocess(run_root)
    _write_motion_calibration_ready(run_root)
    _write_formal_metric_records(run_root, seed_count=3, failed_prompt_index=0, failed_seed_index=2)
    write_jsonl(run_root / "records" / "mechanism_score_records.jsonl", [])
    write_jsonl(run_root / "records" / "controlled_negative_records.jsonl", [])

    matrix_audit = write_pilot_matrix_postprocess(run_root)
    summary = build_small_scale_claim_pilot_audit(run_root)

    assert matrix_audit["pilot_matrix_postprocess_decision"] == "PASS"
    assert matrix_audit["motion_claim_eligible_generation_count"] == 23
    assert matrix_audit["motion_claim_excluded_generation_count"] == 1
    assert summary["pilot_gate_decision"] == "PASS"
    assert summary["claim_support_status"] == "supported_by_small_scale_claim_pilot_records"
    assert summary["formal_motion_claim_status"] == "ready_with_formal_motion_exclusions"
    assert summary["motion_claim_eligible_generation_count"] == 23
    assert summary["motion_claim_excluded_generation_count"] == 1
    assert summary["seed_per_prompt_min"] == 2
    assert summary["missing_pilot_requirements"] == []


@pytest.mark.quick
def test_pilot_gate_reads_bom_encoded_motion_calibration_artifact(tmp_path: Path) -> None:
    """pilot gate 必须能读取带 BOM 的 calibration artifact, 避免 Drive 本地映射编码差异阻断 Colab。"""
    run_root = tmp_path / "generative_video_model_probe_colab"
    _write_generation_records(run_root)
    _write_trajectory_records(run_root)
    _write_proxy_postprocess(run_root)
    _write_formal_metric_records(run_root)
    _write_motion_calibration_ready_with_bom(run_root)
    write_jsonl(run_root / "records" / "mechanism_score_records.jsonl", [])
    write_jsonl(run_root / "records" / "controlled_negative_records.jsonl", [])
    write_pilot_matrix_postprocess(run_root)

    summary = build_small_scale_claim_pilot_audit(run_root)

    assert summary["motion_threshold_calibration_required"] is False
    assert summary["motion_threshold_id"] == "motion_delta_calibrated_v1"
    assert summary["motion_threshold_source_split"] == "calibration"
    assert "formal_motion_claim_ready" not in summary["missing_pilot_requirements"]
