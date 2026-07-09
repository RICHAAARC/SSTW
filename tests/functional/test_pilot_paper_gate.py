import json
from pathlib import Path

import pytest

from experiments.generative_video_model_probe.pilot_paper_gate import (
    build_pilot_paper_gate_audit,
    write_pilot_paper_gate_audit,
)
from main.attacks.video_runtime_attack_protocol import PILOT_PAPER_RUNTIME_ATTACKS
from main.protocol.record_writer import write_json, write_jsonl


ATTACKS = PILOT_PAPER_RUNTIME_ATTACKS
NEGATIVE_FAMILIES = ("wrong_key_control", "without_key_control", "wrong_sampler_replay", "trajectory_time_shuffle_control")
CALIBRATION_SEEDS = tuple(f"seed_calibration_{index:02d}" for index in range(5))
TEST_SEEDS = tuple(f"seed_test_{index:02d}" for index in range(5))
SSTW_METHOD_ID = "sstw_key_conditioned_flow_trajectory"
EXTERNAL_BASELINE_NAMES = (
    "explicit_dtw_temporal_alignment",
    "explicit_frame_matching_temporal_registration",
    "videoshield",
    "vidsig",
    "videoseal",
    "revmark",
    "wam_frame",
)
MODERN_EXTERNAL_BASELINE_NAMES = {"videoshield", "vidsig", "videoseal", "revmark", "wam_frame"}
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


def _validation_scale_gate_pass_payload() -> dict:
    """构造 pilot_paper gate 可接受的完整 validation_scale PASS 摘要。"""

    return {
        "validation_scale_gate_decision": "PASS",
        "claim_support_status": "validation_scale_full_protocol_handoff_ready",
        "paper_result_level": "validation_scale",
        "target_fpr": 0.1,
        "missing_validation_requirements": [],
        "validation_missing_requirement_count": 0,
        "required_modern_external_baseline_adapter_names": sorted(MODERN_EXTERNAL_BASELINE_NAMES),
        "missing_modern_external_baseline_formal_adapter_names": [],
        "modern_external_baseline_formal_measured_adapter_count": len(MODERN_EXTERNAL_BASELINE_NAMES),
        "external_baseline_self_containment_decision": "PASS",
        "data_split_and_leakage_guard_decision": "PASS",
        "sstw_measured_formal_record_count": 24,
        "sstw_measured_formal_status": "sstw_measured_formal_paper_profile_claim_candidate",
        "fair_detection_calibration_ready_count": len(MODERN_EXTERNAL_BASELINE_NAMES) + 1,
        "fair_detection_calibration_status": "fair_detection_calibration_validation_scale_ready",
        "formal_method_baseline_comparison_ready_count": len(MODERN_EXTERNAL_BASELINE_NAMES) + 1,
        "formal_method_baseline_comparison_status": "formal_method_baseline_comparison_paper_profile_claim_candidate",
        "formal_baseline_difference_interval_ready_count": len(MODERN_EXTERNAL_BASELINE_NAMES),
        "formal_baseline_difference_interval_status": "formal_baseline_difference_interval_paper_profile_claim_candidate",
        "validation_scale_sstw_advantage_claim_ready": True,
        "validation_scale_sstw_advantage_claim_status": "validation_scale_target_fpr_0_1_sstw_advantage_claim_supported",
        "full_paper_allowed": False,
    }


def _validation_scale_to_probe_transition_pass_payload() -> dict:
    """构造由 stage_transition_decision 写出的 validation_scale -> probe_paper PASS 摘要。"""

    return {
        "stage_id": "stage_transition_decision",
        "transition_id": "validation_scale_to_probe_paper",
        "validation_scale_to_probe_paper_transition_decision": "PASS",
        "source_stage": "validation_scale",
        "target_stage": "probe_paper",
        "source_gate_passed": True,
        "source_gate_decisions": {"validation_scale_gate_decision": "PASS"},
        "missing_transition_requirements": [],
        "transition_missing_requirement_count": 0,
        "allowed_next_result_profiles": ["probe_paper"],
        "blocked_next_result_profiles": ["pilot_paper", "full_paper", "submission_freeze"],
        "full_paper_allowed": False,
        "claim_support_status": "validation_scale_ready_to_enter_probe_paper",
    }


def _probe_paper_gate_pass_payload() -> dict:
    """构造 pilot_paper gate 可接受的完整 probe_paper PASS 摘要。"""

    payload = _validation_scale_gate_pass_payload()
    payload.update({
        "stage_id": "probe_paper_generative_probe_gate",
        "probe_paper_gate_decision": "PASS",
        "paper_result_level": "probe_paper",
        "claim_support_status": "probe_paper_target_fpr_0_1_paper_claim_supported",
        "sstw_measured_formal_record_count": 10,
        "validation_generation_record_count": 10,
        "validation_prompt_count": 5,
        "validation_seed_per_prompt_min": 2,
        "validation_scale_sstw_advantage_claim_status": "probe_paper_target_fpr_0_1_sstw_advantage_claim_supported",
    })
    return payload


def _probe_paper_to_pilot_transition_pass_payload() -> dict:
    """构造由 stage_transition_decision 写出的 probe_paper -> pilot_paper PASS 摘要。"""

    return {
        "stage_id": "stage_transition_decision",
        "transition_id": "probe_paper_to_pilot_paper",
        "probe_paper_to_pilot_paper_transition_decision": "PASS",
        "source_stage": "probe_paper",
        "target_stage": "pilot_paper",
        "source_gate_passed": True,
        "source_gate_decisions": {"probe_paper_gate_decision": "PASS"},
        "missing_transition_requirements": [],
        "transition_missing_requirement_count": 0,
        "allowed_next_result_profiles": ["pilot_paper"],
        "blocked_next_result_profiles": ["full_paper", "submission_freeze"],
        "full_paper_allowed": False,
        "claim_support_status": "probe_paper_ready_to_enter_pilot_paper",
    }


def _write_pilot_paper_fair_comparison_fixture(
    run_root: Path,
    anchor_units: list[dict[str, str]],
    *,
    blocked_modern_baseline_names: set[str],
    target_fpr: float = 0.01,
    decision_target_fpr: float | None = None,
) -> None:
    """写入 pilot_paper gate 消费的轻量公平比较产物。

    该 fixture 不模拟 detector 细节, 只表达 gate 必须检查的产物契约:
    每个方法已经在自身 clean negative 分布上校准到同一 target FPR, 并且
    attacked positive 使用完全一致的 prompt / seed / attack anchor 集合。
    """

    method_ids = [SSTW_METHOD_ID, *sorted(MODERN_EXTERNAL_BASELINE_NAMES)]
    decision_target_fpr = target_fpr if decision_target_fpr is None else decision_target_fpr
    anchor_keys = [unit["comparison_anchor_key"] for unit in anchor_units]
    fair_records = []
    comparison_records = []
    difference_records = []
    for method_id in method_ids:
        is_blocked = method_id in blocked_modern_baseline_names
        method_role = "proposed_method" if method_id == SSTW_METHOD_ID else "modern_external_baseline"
        positive_detection_units = [
            {
                **unit,
                "score": 0.82,
                "detected_at_target_fpr": True,
            }
            for unit in anchor_units
        ] if not is_blocked else []
        fair_records.append({
            "record_version": "fair_detection_calibration_v1",
            "method_id": method_id,
            "method_role": method_role,
            "metric_status": "measured_formal" if not is_blocked else "missing",
            "fair_comparison_status": "ready" if not is_blocked else "blocked",
            "target_fpr": target_fpr,
            "paper_result_level": "pilot_paper",
            "clean_negative_score_count": 5000 if not is_blocked else 0,
            "attacked_positive_score_count": len(anchor_units) if not is_blocked else 0,
            "positive_anchor_count": len(anchor_units) if not is_blocked else 0,
            "positive_anchor_keys": anchor_keys if not is_blocked else [],
            "positive_anchor_missing_count": 0,
            "positive_formal_evidence_missing_count": 0 if not is_blocked else 1,
            "positive_detection_units_at_target_fpr": positive_detection_units,
            "tpr_at_target_fpr": 1.0 if not is_blocked else None,
            "claim_support_status": "fair_detection_calibration_pilot_paper_ready"
            if not is_blocked
            else "fair_detection_calibration_blocked",
        })
        comparison_alignment = (
            "reference_method_anchor_set_ready"
            if method_id == SSTW_METHOD_ID and not is_blocked
            else "aligned_with_sstw_reference_anchors"
            if not is_blocked
            else "anchor_set_mismatch_with_sstw"
        )
        comparison_records.append({
            "record_version": "formal_method_baseline_comparison_v1",
            "method_id": method_id,
            "method_role": method_role,
            "metric_status": "measured_formal" if not is_blocked else "missing",
            "target_fpr": target_fpr,
            "comparison_primary_metric_name": "tpr_at_target_fpr",
            "comparison_primary_metric_value": 1.0 if not is_blocked else None,
            "comparison_anchor_count": len(anchor_units) if not is_blocked else 0,
            "reference_anchor_count": len(anchor_units),
            "missing_reference_anchor_count": 0 if not is_blocked else len(anchor_units),
            "extra_anchor_count": 0,
            "comparison_anchor_alignment_status": comparison_alignment,
            "claim_support_status": "formal_method_baseline_comparison_pilot_paper_ready"
            if not is_blocked
            else "formal_method_baseline_comparison_missing_measured_formal",
        })
    for baseline_id in sorted(MODERN_EXTERNAL_BASELINE_NAMES):
        is_blocked = baseline_id in blocked_modern_baseline_names
        difference_records.append({
            "record_version": "formal_baseline_difference_interval_v1",
            "reference_method_id": SSTW_METHOD_ID,
            "baseline_method_id": baseline_id,
            "metric_status": "measured_formal" if not is_blocked else "missing",
            "target_fpr": target_fpr,
            "reference_tpr_at_target_fpr": 1.0,
            "baseline_tpr_at_target_fpr": 1.0 if not is_blocked else None,
            "tpr_at_target_fpr_difference": 0.0 if not is_blocked else None,
            "paired_comparison_unit_count": len(anchor_units) if not is_blocked else 0,
            "unpaired_reference_anchor_count": 0 if not is_blocked else len(anchor_units),
            "unpaired_baseline_anchor_count": 0,
            "comparison_anchor_alignment_status": "aligned_with_sstw_reference_anchors"
            if not is_blocked
            else "anchor_set_mismatch_with_sstw",
            "difference_interval_status": "ready" if not is_blocked else "missing_or_unaligned_paired_anchors",
            "difference_ci_confidence_level": 0.95,
            "difference_ci_lower": 0.0 if not is_blocked else None,
            "difference_ci_upper": 0.0 if not is_blocked else None,
            "claim_support_status": "formal_baseline_difference_interval_pilot_paper_ready"
            if not is_blocked
            else "formal_baseline_difference_interval_blocked",
        })
    ready_fair_records = [record for record in fair_records if record["fair_comparison_status"] == "ready"]
    ready_comparison_records = [record for record in comparison_records if record["metric_status"] == "measured_formal"]
    ready_difference_records = [record for record in difference_records if record["difference_interval_status"] == "ready"]
    write_jsonl(run_root / "records" / "fair_detection_calibration_records.jsonl", fair_records)
    write_json(run_root / "artifacts" / "fair_detection_calibration_decision.json", {
        "fair_detection_calibration_decision": "PASS" if len(ready_fair_records) == len(method_ids) else "FAIL",
        "target_fpr": decision_target_fpr,
        "fair_detection_calibration_method_count": len(method_ids),
        "fair_detection_calibration_ready_count": len(ready_fair_records),
        "fair_detection_calibration_missing_method_ids": [
            record["method_id"] for record in fair_records if record["fair_comparison_status"] != "ready"
        ],
        "claim_support_status": "fair_detection_calibration_pilot_paper_ready"
        if len(ready_fair_records) == len(method_ids)
        else "fair_detection_calibration_blocked",
    })
    write_jsonl(run_root / "records" / "formal_method_baseline_comparison_records.jsonl", comparison_records)
    write_json(run_root / "artifacts" / "formal_method_baseline_comparison_decision.json", {
        "formal_method_baseline_comparison_decision": "PASS" if len(ready_comparison_records) == len(method_ids) else "FAIL",
        "target_fpr": decision_target_fpr,
        "formal_comparison_required_method_count": len(method_ids),
        "formal_comparison_ready_method_count": len(ready_comparison_records),
        "formal_comparison_missing_method_ids": [
            record["method_id"] for record in comparison_records if record["metric_status"] != "measured_formal"
        ],
        "claim_support_status": "formal_method_baseline_comparison_pilot_paper_ready"
        if len(ready_comparison_records) == len(method_ids)
        else "formal_method_baseline_comparison_blocked",
    })
    write_jsonl(run_root / "records" / "formal_baseline_difference_interval_records.jsonl", difference_records)
    write_json(run_root / "artifacts" / "formal_baseline_difference_interval_decision.json", {
        "formal_baseline_difference_interval_decision": "PASS"
        if len(ready_difference_records) == len(MODERN_EXTERNAL_BASELINE_NAMES)
        else "FAIL",
        "target_fpr": decision_target_fpr,
        "difference_interval_record_count": len(difference_records),
        "difference_interval_ready_count": len(ready_difference_records),
        "difference_interval_missing_baseline_ids": [
            record["baseline_method_id"]
            for record in difference_records
            if record["difference_interval_status"] != "ready"
        ],
        "claim_support_status": "formal_baseline_difference_interval_pilot_paper_ready"
        if len(ready_difference_records) == len(MODERN_EXTERNAL_BASELINE_NAMES)
        else "formal_baseline_difference_interval_blocked",
    })


def _seed_pilot_paper_run(
    run_root: Path,
    *,
    profile: str = "pilot_paper",
    prompt_count: int = 25,
    calibration_seed_count: int = 2,
    test_seed_count: int = 2,
    validation_scale_gate_decision: str | None = "PASS",
    write_external_baseline: bool = True,
    write_internal_ablation: bool = True,
    incomplete_modern_external_baseline_names: set[str] | None = None,
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
    fair_anchor_units = []
    incomplete_modern_external_baseline_names = incomplete_modern_external_baseline_names or set()
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
                if split_name == "test":
                    fair_anchor_units.append({
                        "comparison_anchor_key": f"{base['prompt_id']}::{seed_id}::{attack_name}",
                        "prompt_id": base["prompt_id"],
                        "seed_id": seed_id,
                        "attack_name": attack_name,
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
                        metric_status = "measured_formal" if baseline_name in MODERN_EXTERNAL_BASELINE_NAMES else "measured_proxy"
                        external_baseline_record = {
                            **base,
                            "attack_name": attack_name,
                            "external_baseline_name": baseline_name,
                            "metric_status": metric_status,
                            "external_baseline_score_status": metric_status,
                            "external_baseline_score": 0.35,
                            "external_baseline_distance": 1.85,
                            "baseline_score_margin": 0.45,
                            "claim_support_status": "modern_external_baseline_formal_measured" if baseline_name in MODERN_EXTERNAL_BASELINE_NAMES else "external_baseline_proxy_comparison_not_claim_supporting",
                        }
                        if (
                            baseline_name in MODERN_EXTERNAL_BASELINE_NAMES
                            and baseline_name not in incomplete_modern_external_baseline_names
                        ):
                            evidence_id = f"{trace_id}_{attack_name}_{baseline_name}"
                            external_baseline_record.update({
                                "external_baseline_raw_detector_score": 0.35,
                                "external_baseline_score_semantics": "watermark_presence_detector_score",
                                "external_baseline_score_orientation": "higher_is_more_watermarked",
                                "external_baseline_clean_negative_score": 0.20,
                                "external_baseline_clean_negative_video_path": str(run_root / "artifacts" / "external_baseline_evidence" / baseline_name / f"{evidence_id}_clean.mp4"),
                                "external_baseline_official_output_path": str(run_root / "artifacts" / "external_baseline_evidence" / baseline_name / f"{evidence_id}_official_output.json"),
                                "external_baseline_official_command_manifest_path": str(run_root / "artifacts" / "external_baseline_evidence" / baseline_name / f"{evidence_id}_official_command_manifest.json"),
                                "external_baseline_official_result_provenance": "repository_generated_from_third_party_official_code",
                                "external_baseline_official_result_bundle_path": str(run_root / "artifacts" / "external_baseline_evidence" / baseline_name / f"{evidence_id}_official_result_bundle.json"),
                                "external_baseline_official_execution_manifest_path": str(run_root / "artifacts" / "external_baseline_evidence" / baseline_name / f"{evidence_id}_official_execution_manifest.json"),
                                "external_baseline_official_score_extraction_policy": "test_official_detector_confidence",
                                "external_baseline_official_reference_protocol_anchor": "same_prompt_seed_attack_runtime_comparison_unit",
                            })
                        external_baseline_records.append(external_baseline_record)
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
        missing_self_contained_names = sorted(incomplete_modern_external_baseline_names)
        write_json(run_root / "artifacts" / "external_baseline_self_containment_decision.json", {
            "external_baseline_self_containment_decision": "PASS" if not missing_self_contained_names else "FAIL",
            "self_contained_modern_external_baseline_count": len(MODERN_EXTERNAL_BASELINE_NAMES) - len(missing_self_contained_names),
            "missing_self_contained_modern_external_baseline_names": missing_self_contained_names,
            "missing_self_containment_requirements": []
            if not missing_self_contained_names
            else ["all_required_modern_baselines_repository_generated_official_bundles"],
            "claim_support_status": "external_baseline_self_containment_ready"
            if not missing_self_contained_names
            else "external_baseline_self_containment_blocked",
        })
        _write_pilot_paper_fair_comparison_fixture(
            run_root,
            fair_anchor_units,
            blocked_modern_baseline_names=incomplete_modern_external_baseline_names,
        )
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
        validation_payload = _validation_scale_gate_pass_payload() if validation_scale_gate_decision == "PASS" else {
            "validation_scale_gate_decision": validation_scale_gate_decision,
            "claim_support_status": "validation_scale_blocked",
        }
        write_json(run_root / "artifacts" / "validation_scale_gate_decision.json", validation_payload)
        if validation_scale_gate_decision == "PASS":
            write_json(
                run_root / "artifacts" / "validation_scale_to_probe_paper_transition_decision.json",
                _validation_scale_to_probe_transition_pass_payload(),
            )
            write_json(
                run_root / "artifacts" / "probe_paper_gate_decision.json",
                _probe_paper_gate_pass_payload(),
            )
            write_json(
                run_root / "artifacts" / "probe_paper_to_pilot_paper_transition_decision.json",
                _probe_paper_to_pilot_transition_pass_payload(),
            )


@pytest.mark.quick
def test_pilot_paper_gate_blocks_empty_run(tmp_path: Path) -> None:
    """空 run_root 不能被解释为 pilot_paper fixed-FPR paper 证据。"""
    audit = build_pilot_paper_gate_audit(tmp_path / "empty")
    protocol = json.loads(Path("configs/protocol/pilot_paper_generative_probe.json").read_text(encoding="utf-8"))

    assert audit["pilot_paper_gate_decision"] == "FAIL"
    assert audit["claim_support_status"] == "blocked_until_pilot_paper_generation_records"
    assert audit["paper_result_level"] == "pilot_paper"
    assert audit["target_fpr"] == protocol["target_fpr"]
    assert audit["tpr_at_target_fpr"] is None
    assert audit["target_fpr_claim_allowed"] is False
    assert audit["tpr_at_fpr_01_pilot_claim_allowed"] is False
    assert audit["pilot_paper_claim_allowed"] is False
    assert audit["tpr_at_fpr_001_claim_allowed"] is False
    assert "pilot_paper_profile_generation_records_ready" in audit["missing_pilot_paper_requirements"]


@pytest.mark.quick
def test_pilot_paper_gate_cannot_disable_validation_scale_fairness_prerequisites(tmp_path: Path) -> None:
    """pilot_paper 不能通过配置关闭 validation_scale 与公平比较硬前置。"""
    config_path = tmp_path / "pilot_paper_config.json"
    config_path.write_text(json.dumps({
        "target_fpr": 0.01,
        "blocked_target_fpr": 0.001,
        "paper_result_level": "pilot_paper",
        "paper_protocol_level": "paper_grade_protocol",
        "paper_protocol_difference_from_full_paper": "sample_scale_and_target_fpr_only",
        "minimum_prompt_count": 0,
        "minimum_seed_per_prompt": 0,
        "minimum_calibration_seed_per_prompt": 0,
        "minimum_test_seed_per_prompt": 0,
        "minimum_unique_video_count": 0,
        "minimum_calibration_unique_video_count": 0,
        "minimum_test_unique_video_count": 0,
        "minimum_calibration_negative_event_count": 0,
        "minimum_heldout_test_negative_event_count": 0,
        "minimum_heldout_attacked_positive_event_count": 0,
        "minimum_clean_negative_count": 0,
        "minimum_negative_family_count": 0,
        "minimum_calibration_negative_event_count_per_family": 0,
        "minimum_heldout_negative_event_count_per_family": 0,
        "minimum_attack_event_count_per_attack": 0,
        "minimum_external_baseline_measured_adapter_count": 0,
        "minimum_modern_external_baseline_formal_adapter_count": 0,
        "minimum_pilot_paper_external_baseline_trace_count": 0,
        "minimum_pilot_paper_internal_ablation_trace_count": 0,
        "minimum_internal_ablation_variant_count": 0,
        "required_external_baseline_adapter_names": [],
        "required_modern_external_baseline_adapter_names": [],
        "required_internal_ablation_variants": [],
        "require_probe_paper_gate_passed": False,
        "require_probe_paper_to_pilot_paper_transition_decision": False,
        "require_validation_scale_gate_passed": False,
        "require_external_baseline_comparison_ready": False,
        "require_external_baseline_self_contained_outputs": False,
        "require_modern_external_baseline_formal_results": False,
        "require_fair_detection_calibration": False,
        "require_formal_method_baseline_comparison": False,
        "require_formal_baseline_difference_interval": False,
        "require_internal_ablation_matrix_ready": False,
        "require_motion_threshold_calibration_ready": False,
        "require_formal_motion_claim_ready": False,
    }), encoding="utf-8")

    audit = build_pilot_paper_gate_audit(tmp_path / "run", config_path)

    assert audit["pilot_paper_gate_decision"] == "FAIL"
    assert audit["pilot_paper_claim_allowed"] is False
    assert audit["pilot_paper_hard_required_config_missing_count"] == 8
    assert "require_probe_paper_gate_passed_must_be_true" in audit["missing_pilot_paper_requirements"]
    assert "require_probe_paper_to_pilot_paper_transition_decision_must_be_true" in audit["missing_pilot_paper_requirements"]
    assert "require_fair_detection_calibration_must_be_true" in audit["missing_pilot_paper_requirements"]
    assert "require_formal_method_baseline_comparison_must_be_true" in audit["missing_pilot_paper_requirements"]
    assert "require_formal_baseline_difference_interval_must_be_true" in audit["missing_pilot_paper_requirements"]


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
    """满足 calibration/test split 与 5000+ held-out negative events 时允许 pilot_paper 级 TPR@target_fpr。"""
    run_root = tmp_path / "run"
    _seed_pilot_paper_run(run_root)

    audit = write_pilot_paper_gate_audit(run_root)

    assert audit["pilot_paper_gate_decision"] == "PASS"
    assert audit["claim_support_status"] == "pilot_paper_calibrated_heldout_claim_ready"
    assert audit["paper_result_level"] == "pilot_paper"
    assert audit["paper_protocol_level"] == "paper_grade_protocol"
    assert audit["paper_protocol_difference_from_full_paper"] == "sample_scale_and_target_fpr_only"
    assert audit["pilot_paper_protocol_matches_full_paper"] is True
    assert audit["pilot_paper_hard_required_config_missing_count"] == 0
    assert audit["probe_paper_gate_decision"] == "PASS"
    assert audit["probe_paper_to_pilot_paper_transition_decision"] == "PASS"
    assert audit["external_baseline_comparison_decision"] == "PASS"
    assert audit["external_baseline_self_containment_decision"] == "PASS"
    assert audit["external_baseline_measured_adapter_count"] == len(EXTERNAL_BASELINE_NAMES)
    assert audit["modern_external_baseline_formal_measured_adapter_count"] == len(MODERN_EXTERNAL_BASELINE_NAMES)
    assert audit["missing_modern_external_baseline_formal_adapter_names"] == []
    assert audit["fair_detection_calibration_decision"] == "PASS"
    assert audit["fair_detection_calibration_ready_count"] == len(MODERN_EXTERNAL_BASELINE_NAMES) + 1
    assert audit["fair_detection_calibration_missing_method_ids"] == []
    assert audit["formal_method_baseline_comparison_decision"] == "PASS"
    assert audit["formal_method_baseline_comparison_ready_count"] == len(MODERN_EXTERNAL_BASELINE_NAMES) + 1
    assert audit["formal_method_baseline_comparison_missing_method_ids"] == []
    assert audit["formal_baseline_difference_interval_decision"] == "PASS"
    assert audit["formal_baseline_difference_interval_ready_count"] == len(MODERN_EXTERNAL_BASELINE_NAMES)
    assert audit["formal_baseline_difference_interval_missing_baseline_ids"] == []
    assert audit["pilot_paper_external_baseline_trace_count"] == 50
    assert audit["pilot_paper_external_baseline_trace_count_min"] == 50
    assert audit["validation_internal_ablation_decision"] == "PASS"
    assert audit["validation_internal_ablation_variant_count"] >= 8
    assert audit["pilot_paper_internal_ablation_trace_count_min"] == 50
    assert audit["threshold_protocol"] == "calibration_split_to_frozen_threshold_to_heldout_test_split"
    assert audit["threshold_source_split"] == "calibration"
    assert audit["test_time_threshold_update_blocked"] is True
    assert audit["pilot_paper_generation_record_count"] == 100
    assert audit["pilot_paper_unique_video_count"] == 100
    assert audit["pilot_paper_calibration_unique_video_count"] == 50
    assert audit["pilot_paper_test_unique_video_count"] == 50
    assert audit["pilot_paper_calibration_seed_per_prompt_min"] == 2
    assert audit["pilot_paper_test_seed_per_prompt_min"] == 2
    expected_attacked_positive_count = 50 * len(ATTACKS)
    assert audit["calibration_negative_event_count"] == expected_attacked_positive_count * len(NEGATIVE_FAMILIES)
    assert audit["heldout_test_negative_event_count"] == expected_attacked_positive_count * len(NEGATIVE_FAMILIES)
    assert audit["heldout_attacked_positive_event_count"] == expected_attacked_positive_count
    assert audit["calibration_negative_family_count"] == 4
    assert audit["heldout_negative_family_count"] == 4
    assert audit["calibration_negative_event_count_per_family_min"] == expected_attacked_positive_count
    assert audit["heldout_negative_event_count_per_family_min"] == expected_attacked_positive_count
    assert audit["attack_event_count_per_attack_min"] == 50
    assert audit["calibration_negative_fpr_at_threshold"] <= audit["target_fpr"]
    assert audit["heldout_negative_fpr_at_threshold"] <= audit["target_fpr"]
    assert audit["observed_negative_fpr_at_threshold"] == audit["heldout_negative_fpr_at_threshold"]
    assert audit["tpr_at_target_fpr"] == 1.0
    assert audit["target_fpr_claim_allowed"] is True
    assert audit["tpr_at_fpr_01"] == 1.0
    assert audit["tpr_at_fpr_01_pilot_claim_allowed"] is True
    assert audit["pilot_paper_claim_allowed"] is True
    assert audit["blocked_target_fpr_claim_allowed"] is False
    assert audit["tpr_at_fpr_001_claim_allowed"] is False
    assert audit["full_paper_allowed"] is False
    assert (run_root / "records" / "pilot_paper_gate_records.jsonl").exists()
    assert (run_root / "tables" / "pilot_paper_gate_table.csv").exists()
    assert (run_root / "thresholds" / "pilot_paper_frozen_threshold.json").exists()
    assert (run_root / "artifacts" / "pilot_paper_gate_decision.json").exists()
    assert (run_root / "reports" / "pilot_paper_gate_report.md").exists()


@pytest.mark.quick
def test_pilot_paper_gate_rejects_incomplete_formal_external_baseline(tmp_path: Path) -> None:
    """pilot_paper 不能把缺少 clean negative 或 official evidence 的 measured_formal 当作公平比较证据。"""
    run_root = tmp_path / "run"
    _seed_pilot_paper_run(run_root, incomplete_modern_external_baseline_names={"videoseal"})

    audit = build_pilot_paper_gate_audit(run_root)

    assert audit["pilot_paper_gate_decision"] == "FAIL"
    assert "pilot_paper_external_baseline_comparison_ready" in audit["missing_pilot_paper_requirements"]
    assert "pilot_paper_external_baseline_self_containment_ready" in audit["missing_pilot_paper_requirements"]
    assert "pilot_paper_fair_detection_calibration_ready" in audit["missing_pilot_paper_requirements"]
    assert "pilot_paper_formal_method_baseline_comparison_ready" in audit["missing_pilot_paper_requirements"]
    assert "pilot_paper_formal_baseline_difference_interval_ready" in audit["missing_pilot_paper_requirements"]
    assert audit["external_baseline_formal_incomplete_record_count"] > 0
    assert audit["modern_external_baseline_formal_measured_adapter_count"] == len(MODERN_EXTERNAL_BASELINE_NAMES) - 1
    assert audit["missing_modern_external_baseline_formal_adapter_names"] == ["videoseal"]
    assert audit["fair_detection_calibration_missing_method_ids"] == ["videoseal"]
    assert audit["formal_method_baseline_comparison_missing_method_ids"] == ["videoseal"]
    assert audit["formal_baseline_difference_interval_missing_baseline_ids"] == ["videoseal"]
    assert audit["pilot_paper_claim_allowed"] is False


@pytest.mark.quick
def test_pilot_paper_gate_requires_probe_paper_gate(tmp_path: Path) -> None:
    """pilot_paper 是 full_paper 协议的小规模预演, 因此必须先通过 probe_paper。"""
    run_root = tmp_path / "run"
    _seed_pilot_paper_run(run_root, validation_scale_gate_decision=None)

    audit = build_pilot_paper_gate_audit(run_root)

    assert audit["pilot_paper_gate_decision"] == "FAIL"
    assert "probe_paper_gate_passed" in audit["missing_pilot_paper_requirements"]
    assert audit["pilot_paper_claim_allowed"] is False


@pytest.mark.quick
def test_pilot_paper_gate_rejects_legacy_probe_paper_pass_without_fair_summary(
    tmp_path: Path,
) -> None:
    """旧版 probe_paper PASS 缺少公平比较摘要时, pilot_paper 不能继续放行。"""

    run_root = tmp_path / "run"
    _seed_pilot_paper_run(run_root)
    write_json(run_root / "artifacts" / "probe_paper_gate_decision.json", {
        "probe_paper_gate_decision": "PASS",
        "claim_support_status": "probe_paper_ready_for_pilot_paper",
    })

    audit = build_pilot_paper_gate_audit(run_root)

    assert audit["pilot_paper_gate_decision"] == "FAIL"
    assert "probe_paper_gate_passed" in audit["missing_pilot_paper_requirements"]
    assert "probe_paper_claim_support_status_ready" in audit["probe_paper_gate_fairness_missing_requirements"]
    assert "probe_paper_result_level_current" in audit["probe_paper_gate_fairness_missing_requirements"]
    assert "probe_paper_missing_requirements_empty" in audit["probe_paper_gate_fairness_missing_requirements"]
    assert audit["pilot_paper_claim_allowed"] is False


@pytest.mark.quick
def test_pilot_paper_gate_rejects_bare_probe_transition_pass(tmp_path: Path) -> None:
    """只有裸 PASS 跳转字段时不能替代完整 stage_transition_decision 产物。"""

    run_root = tmp_path / "run"
    _seed_pilot_paper_run(run_root)
    write_json(run_root / "artifacts" / "probe_paper_to_pilot_paper_transition_decision.json", {
        "probe_paper_to_pilot_paper_transition_decision": "PASS",
        "claim_support_status": "probe_paper_ready_to_enter_pilot_paper",
    })

    audit = build_pilot_paper_gate_audit(run_root)

    assert audit["pilot_paper_gate_decision"] == "FAIL"
    assert "probe_paper_to_pilot_paper_transition_decision_passed" in audit["missing_pilot_paper_requirements"]
    assert "probe_paper_transition_source_gate_passed" in audit["probe_paper_transition_fairness_missing_requirements"]
    assert "probe_paper_transition_missing_requirements_empty" in audit["probe_paper_transition_fairness_missing_requirements"]
    assert audit["pilot_paper_claim_allowed"] is False


@pytest.mark.quick
def test_pilot_paper_gate_requires_external_baseline_and_ablation(tmp_path: Path) -> None:
    """pilot_paper 是完整协议预演, 因此必须同时具备 baseline comparison 与内部消融矩阵。"""
    run_root = tmp_path / "run"
    _seed_pilot_paper_run(run_root, write_external_baseline=False, write_internal_ablation=False)

    audit = build_pilot_paper_gate_audit(run_root)

    assert audit["pilot_paper_gate_decision"] == "FAIL"
    assert "pilot_paper_external_baseline_comparison_ready" in audit["missing_pilot_paper_requirements"]
    assert "pilot_paper_external_baseline_self_containment_ready" in audit["missing_pilot_paper_requirements"]
    assert "pilot_paper_fair_detection_calibration_ready" in audit["missing_pilot_paper_requirements"]
    assert "pilot_paper_formal_method_baseline_comparison_ready" in audit["missing_pilot_paper_requirements"]
    assert "pilot_paper_formal_baseline_difference_interval_ready" in audit["missing_pilot_paper_requirements"]
    assert "pilot_paper_internal_ablation_matrix_ready" in audit["missing_pilot_paper_requirements"]
    assert audit["pilot_paper_claim_allowed"] is False


@pytest.mark.quick
def test_pilot_paper_gate_requires_fair_detection_calibration_decision(tmp_path: Path) -> None:
    """pilot_paper 不能在缺少同 target FPR 公平校准 decision 时通过。"""
    run_root = tmp_path / "run"
    _seed_pilot_paper_run(run_root)
    (run_root / "artifacts" / "fair_detection_calibration_decision.json").unlink()

    audit = build_pilot_paper_gate_audit(run_root)

    assert audit["pilot_paper_gate_decision"] == "FAIL"
    assert "pilot_paper_fair_detection_calibration_ready" in audit["missing_pilot_paper_requirements"]
    assert audit["fair_detection_calibration_status"] == "missing_fair_detection_calibration_decision"
    assert audit["pilot_paper_claim_allowed"] is False


@pytest.mark.quick
def test_pilot_paper_gate_rejects_fair_comparison_with_negative_evidence_gap(tmp_path: Path) -> None:
    """pilot_paper 不能消费 clean negative official evidence 缺失的 fair calibration 记录。"""
    run_root = tmp_path / "run"
    _seed_pilot_paper_run(run_root)
    fair_path = run_root / "records" / "fair_detection_calibration_records.jsonl"
    records = [json.loads(line) for line in fair_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for record in records:
        if record["method_id"] == "videoseal":
            record["negative_formal_evidence_missing_count"] = 1
    write_jsonl(fair_path, records)

    audit = build_pilot_paper_gate_audit(run_root)

    assert audit["pilot_paper_gate_decision"] == "FAIL"
    assert "pilot_paper_fair_detection_calibration_ready" in audit["missing_pilot_paper_requirements"]
    assert audit["fair_detection_calibration_missing_method_ids"] == ["videoseal"]
    assert audit["pilot_paper_claim_allowed"] is False


@pytest.mark.quick
def test_pilot_paper_gate_rejects_fair_calibration_target_fpr_mismatch(tmp_path: Path) -> None:
    """pilot_paper 公平比较产物必须与当前 protocol config 的 target_fpr 完全一致。"""
    run_root = tmp_path / "run"
    _seed_pilot_paper_run(run_root)
    decision_path = run_root / "artifacts" / "fair_detection_calibration_decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["target_fpr"] = 0.1
    write_json(decision_path, decision)

    audit = build_pilot_paper_gate_audit(run_root)

    assert audit["pilot_paper_gate_decision"] == "FAIL"
    assert audit["fair_detection_calibration_target_fpr"] == 0.1
    assert "pilot_paper_fair_detection_calibration_ready" in audit["missing_pilot_paper_requirements"]
    assert audit["pilot_paper_claim_allowed"] is False


@pytest.mark.quick
def test_pilot_paper_gate_blocks_insufficient_negative_events(tmp_path: Path) -> None:
    """negative event 数量不足时不能报告 pilot 级低 FPR 结论。"""
    run_root = tmp_path / "run"
    _seed_pilot_paper_run(run_root, prompt_count=1, calibration_seed_count=1, test_seed_count=1)

    audit = build_pilot_paper_gate_audit(run_root)

    assert audit["pilot_paper_gate_decision"] == "FAIL"
    assert audit["calibration_negative_event_count"] < audit["minimum_calibration_negative_event_count"]
    assert audit["heldout_test_negative_event_count"] < audit["minimum_heldout_test_negative_event_count"]
    assert "calibration_negative_event_count_ready" in audit["missing_pilot_paper_requirements"]
    assert "heldout_test_negative_event_count_ready" in audit["missing_pilot_paper_requirements"]
    assert audit["tpr_at_fpr_01_pilot_claim_allowed"] is False
