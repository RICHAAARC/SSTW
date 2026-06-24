from pathlib import Path

import pytest

from experiments.generative_video_model_probe.pilot_paper_gate import (
    build_pilot_paper_gate_audit,
    write_pilot_paper_gate_audit,
)
from main.protocol.record_writer import write_json, write_jsonl


ATTACKS = ("video_compression_runtime", "temporal_crop_runtime", "frame_rate_resampling_runtime")
NEGATIVE_FAMILIES = ("wrong_key_control", "without_key_control", "wrong_sampler_replay", "trajectory_time_shuffle_control")
CALIBRATION_SEEDS = tuple(f"seed_calibration_{index:02d}" for index in range(4))
TEST_SEEDS = tuple(f"seed_test_{index:02d}" for index in range(4))
EXTERNAL_BASELINE_NAMES = (
    "explicit_dtw_temporal_alignment",
    "explicit_frame_matching_temporal_registration",
    "videoshield",
    "sigmark",
    "spdmark",
    "videomark",
    "vidsig",
    "videoseal",
)
MODERN_EXTERNAL_BASELINE_NAMES = {"videoshield", "sigmark", "spdmark", "videomark", "vidsig", "videoseal"}
INTERNAL_ABLATION_VARIANTS = (
    "sstw_full_method",
    "endpoint_only_control",
    "trajectory_only_score",
    "without_velocity_constraint",
    "without_endpoint_aware_control",
    "without_replay_uncertainty_weighting",
    "without_flow_state_admissibility",
    "generic_ssm_baseline",
)


def _seed_pilot_paper_run(
    run_root: Path,
    *,
    profile: str = "pilot_paper",
    prompt_count: int = 21,
    calibration_seed_count: int = 4,
    test_seed_count: int = 4,
    validation_scale_gate_decision: str | None = "PASS",
    write_external_baseline: bool = True,
    write_internal_ablation: bool = True,
) -> None:
    """构造轻量 pilot_paper fixture, 不写入任何真实视频文件。

    该 fixture 显式模拟论文同构流程: calibration split 只用于冻结阈值,
    test split 只用于 held-out FPR / TPR 报告。
    """
    generation_records = []
    formal_records = []
    runtime_detection_records = []
    pilot_matrix_records = []
    external_baseline_records = []
    internal_ablation_records = []
    split_seed_pairs = [
        ("calibration", seed_id) for seed_id in CALIBRATION_SEEDS[:calibration_seed_count]
    ] + [
        ("test", seed_id) for seed_id in TEST_SEEDS[:test_seed_count]
    ]
    for prompt_index in range(prompt_count):
        for split_name, seed_id in split_seed_pairs:
            trace_id = f"trace_{prompt_index:02d}_{seed_id}"
            base = {
                "generation_model_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                "prompt_id": f"prompt_{prompt_index:02d}",
                "seed_id": seed_id,
                "split": split_name,
                "trajectory_trace_id": trace_id,
            }
            generation_records.append({
                **base,
                "generation_status": "success",
                "colab_runtime_profile": profile,
                "motion_claim_role": "positive_motion",
            })
            formal_records.append({
                **base,
                "formal_visual_quality_ready": True,
                "formal_motion_consistency_ready": True,
                "formal_semantic_consistency_ready": True,
                "formal_metric_result_used_for_claim": True,
                "motion_claim_role": "positive_motion",
            })
            for attack_name in ATTACKS:
                runtime_detection_records.append({
                    **base,
                    "attack_name": attack_name,
                    "runtime_detection_status": "ready",
                    "S_runtime_attack_detection": 0.82,
                    "S_final_conservative": 0.82,
                    "attacked_video_detectable": True,
                })
                for method_variant in (
                    "sstw_full_method",
                    "endpoint_only_control",
                    "trajectory_only_score",
                    "without_velocity_constraint",
                    "without_endpoint_aware_control",
                    "without_replay_uncertainty_weighting",
                ):
                    pilot_matrix_records.append({
                        **base,
                        "attack_name": attack_name,
                        "method_variant": method_variant,
                        "sample_role": "generated_positive",
                        "S_final_conservative": 0.80,
                        "path_marginal_gain_at_fixed_fpr": 0.07,
                        "replay_uncertainty_mean": 0.05,
                    })
                for negative_family in NEGATIVE_FAMILIES:
                    pilot_matrix_records.append({
                        **base,
                        "attack_name": attack_name,
                        "method_variant": "sstw_full_method",
                        "sample_role": "controlled_negative",
                        "negative_family": negative_family,
                        "control_name": negative_family,
                        "S_final_conservative": 0.20,
                        "S_final": 0.20,
                        "path_marginal_gain_at_fixed_fpr": 0.07,
                        "replay_uncertainty_mean": 0.05,
                        "negative_tail_status": "not_inflated",
                        "wrong_sampler_replay_control_not_equivalent": negative_family == "wrong_sampler_replay",
                        "wrong_sampler_replay_status": "replay_rejected" if negative_family == "wrong_sampler_replay" else "not_applicable",
                        "decision": "replay_rejected" if negative_family == "wrong_sampler_replay" else "controlled_negative_below_threshold",
                    })
                if split_name == "test":
                    for baseline_name in EXTERNAL_BASELINE_NAMES:
                        external_baseline_records.append({
                            **base,
                            "attack_name": attack_name,
                            "external_baseline_name": baseline_name,
                            "metric_status": "measured_formal" if baseline_name in MODERN_EXTERNAL_BASELINE_NAMES else "measured_proxy",
                            "external_baseline_score": 0.35,
                            "external_baseline_distance": 1.85,
                            "baseline_score_margin": 0.45,
                            "claim_support_status": "modern_external_baseline_formal_measured" if baseline_name in MODERN_EXTERNAL_BASELINE_NAMES else "external_baseline_proxy_comparison_not_claim_supporting",
                        })
                    for method_variant in INTERNAL_ABLATION_VARIANTS:
                        internal_ablation_records.append({
                            **base,
                            "attack_name": attack_name,
                            "method_variant": method_variant,
                            "ablation_runtime_profile": profile,
                            "validation_ablation_proxy_score": 0.80 if method_variant == "sstw_full_method" else 0.62,
                            "claim_support_status": "validation_internal_ablation_proxy_only",
                        })
    write_jsonl(run_root / "records" / "generation_records.jsonl", generation_records)
    write_jsonl(run_root / "records" / "formal_quality_motion_semantic_records.jsonl", formal_records)
    write_jsonl(run_root / "records" / "runtime_detection_records.jsonl", runtime_detection_records)
    write_jsonl(run_root / "records" / "small_scale_claim_pilot_matrix_records.jsonl", pilot_matrix_records)
    if write_external_baseline:
        write_jsonl(run_root / "records" / "external_baseline_score_records.jsonl", external_baseline_records)
        write_json(run_root / "artifacts" / "external_baseline_comparison_decision.json", {
            "external_baseline_comparison_decision": "PASS",
            "external_baseline_comparison_table_status": "ready",
            "external_baseline_measured_adapter_count": len(EXTERNAL_BASELINE_NAMES),
            "external_baseline_measured_adapter_names": list(EXTERNAL_BASELINE_NAMES),
            "modern_external_baseline_formal_measured_adapter_count": len(MODERN_EXTERNAL_BASELINE_NAMES),
            "modern_external_baseline_formal_measured_adapter_names": sorted(MODERN_EXTERNAL_BASELINE_NAMES),
            "external_baseline_claim_support_status": "external_baseline_formal_and_proxy_records_written",
        })
    if write_internal_ablation:
        write_jsonl(run_root / "records" / "validation_internal_ablation_records.jsonl", internal_ablation_records)
        write_json(run_root / "artifacts" / "validation_internal_ablation_decision.json", {
            "validation_internal_ablation_decision": "PASS",
            "internal_ablation_record_count": len(internal_ablation_records),
            "validation_internal_ablation_variant_count": len(INTERNAL_ABLATION_VARIANTS),
            "validation_internal_ablation_score_margin": 0.18,
            "claim_support_status": "validation_internal_ablation_proxy_only",
        })
    write_json(run_root / "artifacts" / "small_scale_claim_pilot_gate_decision.json", {"pilot_gate_decision": "PASS"})
    write_json(run_root / "artifacts" / "motion_threshold_calibration_decision.json", {
        "motion_threshold_calibration_decision": "PASS",
        "motion_threshold_calibration_ready": True,
        "motion_threshold_id": "motion_delta_calibrated_v1",
        "motion_threshold_source_split": "calibration",
    })
    if validation_scale_gate_decision is not None:
        write_json(run_root / "artifacts" / "validation_scale_gate_decision.json", {
            "validation_scale_gate_decision": validation_scale_gate_decision,
            "claim_support_status": "validation_scale_ready_for_pilot_paper"
            if validation_scale_gate_decision == "PASS"
            else "validation_scale_blocked",
        })


@pytest.mark.quick
def test_pilot_paper_gate_blocks_empty_run(tmp_path: Path) -> None:
    """空 run_root 不能被解释为 pilot_paper fixed-FPR paper 证据。"""
    audit = build_pilot_paper_gate_audit(tmp_path / "empty")

    assert audit["pilot_paper_gate_decision"] == "FAIL"
    assert audit["claim_support_status"] == "blocked_until_pilot_paper_generation_records"
    assert audit["paper_result_level"] == "pilot_paper"
    assert audit["tpr_at_fpr_01_pilot_claim_allowed"] is False
    assert audit["pilot_paper_claim_allowed"] is False
    assert audit["tpr_at_fpr_001_claim_allowed"] is False
    assert "pilot_paper_profile_generation_records_ready" in audit["missing_pilot_paper_requirements"]


@pytest.mark.quick
def test_pilot_paper_gate_rejects_validation_scale_profile(tmp_path: Path) -> None:
    """validation_scale profile 不能冒充 pilot_paper profile。"""
    run_root = tmp_path / "run"
    _seed_pilot_paper_run(run_root, profile="validation_scale")

    audit = build_pilot_paper_gate_audit(run_root)

    assert audit["pilot_paper_gate_decision"] == "FAIL"
    assert audit["pilot_paper_generation_record_count"] == 0
    assert "pilot_paper_profile_generation_records_ready" in audit["missing_pilot_paper_requirements"]


@pytest.mark.quick
def test_pilot_paper_gate_passes_calibrated_heldout_fixture(tmp_path: Path) -> None:
    """满足 calibration/test split 与 1000+ held-out negative events 时允许 pilot_paper 级 TPR@FPR=0.01。"""
    run_root = tmp_path / "run"
    _seed_pilot_paper_run(run_root)

    audit = write_pilot_paper_gate_audit(run_root)

    assert audit["pilot_paper_gate_decision"] == "PASS"
    assert audit["claim_support_status"] == "pilot_paper_calibrated_heldout_claim_ready"
    assert audit["paper_result_level"] == "pilot_paper"
    assert audit["paper_protocol_level"] == "paper_grade_protocol"
    assert audit["paper_protocol_difference_from_full_paper"] == "sample_scale_only"
    assert audit["pilot_paper_protocol_matches_full_paper"] is True
    assert audit["validation_scale_gate_decision"] == "PASS"
    assert audit["external_baseline_comparison_decision"] == "PASS"
    assert audit["external_baseline_measured_adapter_count"] == 8
    assert audit["modern_external_baseline_formal_measured_adapter_count"] == 6
    assert audit["missing_modern_external_baseline_formal_adapter_names"] == []
    assert audit["pilot_paper_external_baseline_trace_count"] == 84
    assert audit["pilot_paper_external_baseline_trace_count_min"] == 84
    assert audit["validation_internal_ablation_decision"] == "PASS"
    assert audit["validation_internal_ablation_variant_count"] >= 8
    assert audit["pilot_paper_internal_ablation_trace_count_min"] == 84
    assert audit["threshold_protocol"] == "calibration_split_to_frozen_threshold_to_heldout_test_split"
    assert audit["threshold_source_split"] == "calibration"
    assert audit["test_time_threshold_update_blocked"] is True
    assert audit["pilot_paper_generation_record_count"] == 168
    assert audit["pilot_paper_unique_video_count"] == 168
    assert audit["pilot_paper_calibration_unique_video_count"] == 84
    assert audit["pilot_paper_test_unique_video_count"] == 84
    assert audit["pilot_paper_calibration_seed_per_prompt_min"] == 4
    assert audit["pilot_paper_test_seed_per_prompt_min"] == 4
    assert audit["calibration_negative_event_count"] == 1008
    assert audit["heldout_test_negative_event_count"] == 1008
    assert audit["heldout_attacked_positive_event_count"] == 252
    assert audit["calibration_negative_family_count"] == 4
    assert audit["heldout_negative_family_count"] == 4
    assert audit["calibration_negative_event_count_per_family_min"] == 252
    assert audit["heldout_negative_event_count_per_family_min"] == 252
    assert audit["attack_event_count_per_attack_min"] == 84
    assert audit["calibration_negative_fpr_at_threshold"] <= 0.01
    assert audit["heldout_negative_fpr_at_threshold"] <= 0.01
    assert audit["observed_negative_fpr_at_threshold"] == audit["heldout_negative_fpr_at_threshold"]
    assert audit["tpr_at_fpr_01"] == 1.0
    assert audit["tpr_at_fpr_01_pilot_claim_allowed"] is True
    assert audit["pilot_paper_claim_allowed"] is True
    assert audit["tpr_at_fpr_001_claim_allowed"] is False
    assert audit["full_paper_allowed"] is False
    assert (run_root / "records" / "pilot_paper_gate_records.jsonl").exists()
    assert (run_root / "tables" / "pilot_paper_gate_table.csv").exists()
    assert (run_root / "thresholds" / "pilot_paper_frozen_threshold.json").exists()
    assert (run_root / "artifacts" / "pilot_paper_gate_decision.json").exists()
    assert (run_root / "reports" / "pilot_paper_gate_report.md").exists()


@pytest.mark.quick
def test_pilot_paper_gate_requires_validation_scale_gate(tmp_path: Path) -> None:
    """pilot_paper 是 full_paper 协议的小规模预演, 因此必须先通过 validation-scale。"""
    run_root = tmp_path / "run"
    _seed_pilot_paper_run(run_root, validation_scale_gate_decision=None)

    audit = build_pilot_paper_gate_audit(run_root)

    assert audit["pilot_paper_gate_decision"] == "FAIL"
    assert "validation_scale_gate_passed" in audit["missing_pilot_paper_requirements"]
    assert audit["pilot_paper_claim_allowed"] is False


@pytest.mark.quick
def test_pilot_paper_gate_requires_external_baseline_and_ablation(tmp_path: Path) -> None:
    """pilot_paper 是完整协议预演, 因此必须同时具备 baseline comparison 与内部消融矩阵。"""
    run_root = tmp_path / "run"
    _seed_pilot_paper_run(run_root, write_external_baseline=False, write_internal_ablation=False)

    audit = build_pilot_paper_gate_audit(run_root)

    assert audit["pilot_paper_gate_decision"] == "FAIL"
    assert "pilot_paper_external_baseline_comparison_ready" in audit["missing_pilot_paper_requirements"]
    assert "pilot_paper_internal_ablation_matrix_ready" in audit["missing_pilot_paper_requirements"]
    assert audit["pilot_paper_claim_allowed"] is False


@pytest.mark.quick
def test_pilot_paper_gate_blocks_insufficient_negative_events(tmp_path: Path) -> None:
    """negative event 数量不足时不能报告 pilot 级低 FPR 结论。"""
    run_root = tmp_path / "run"
    _seed_pilot_paper_run(run_root, prompt_count=4, calibration_seed_count=2, test_seed_count=2)

    audit = build_pilot_paper_gate_audit(run_root)

    assert audit["pilot_paper_gate_decision"] == "FAIL"
    assert audit["calibration_negative_event_count"] < audit["minimum_calibration_negative_event_count"]
    assert audit["heldout_test_negative_event_count"] < audit["minimum_heldout_test_negative_event_count"]
    assert "calibration_negative_event_count_ready" in audit["missing_pilot_paper_requirements"]
    assert "heldout_test_negative_event_count_ready" in audit["missing_pilot_paper_requirements"]
    assert audit["tpr_at_fpr_01_pilot_claim_allowed"] is False
