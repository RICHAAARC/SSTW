"""验证三个正式 profile 共享同一套论文证据闭合条件。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.protocol.paper_profile_evidence_closure import (
    build_paper_profile_evidence_closure_audit,
)
from evaluation.protocol.record_writer import write_json, write_jsonl


MODERN_BASELINES = ("videoshield", "vidsig", "videoseal", "videomark", "wam_frame")


def _write_decision(
    run_root: Path,
    relative_name: str,
    decision_field: str,
    target_fpr: float | None = None,
    **extra: object,
) -> None:
    """写入公共门禁测试所需的最小 governed decision。"""

    payload: dict[str, object] = {decision_field: "PASS", **extra}
    if target_fpr is not None:
        payload["target_fpr"] = target_fpr
    write_json(run_root / "artifacts" / relative_name, payload)


def _write_closed_evidence(run_root: Path, target_fpr: float) -> None:
    """为三层主张写入同构的完整证据 fixture。"""

    profile = run_root.name
    claim_fields = {
        "claim_1_velocity_constraint_detectable_watermark_decision": "PASS",
        "claim_2_path_evidence_independent_gain_decision": "PASS",
        "claim_3_attacked_video_replay_posterior_decision": "PASS",
    }
    _write_decision(
        run_root,
        "complete_paper_mechanism_claim_decision.json",
        "complete_paper_mechanism_claim_decision",
        **claim_fields,
    )
    _write_decision(
        run_root,
        "three_layer_mechanism_evidence_decision.json",
        "three_layer_mechanism_pre_replay_decision",
        **claim_fields,
    )
    _write_decision(
        run_root,
        "formal_flow_evidence_decision.json",
        "formal_flow_evidence_decision",
        posterior_probability_calibration_decision="PASS",
        state_space_posterior_mechanism_decision="PASS",
        formal_negative_hypothesis_family_decision="PASS",
        **claim_fields,
    )
    _write_decision(
        run_root,
        "replay_and_sketch_gate_decision.json",
        "replay_and_sketch_gate_decision",
        claim3_full_support_allowed=True,
    )
    _write_decision(
        run_root,
        "formal_adaptive_attack_execution_decision.json",
        "formal_adaptive_attack_execution_decision",
        per_video_adaptive_attack_optimization=True,
        adaptive_attack_query_provenance_decision="PASS",
        adaptive_detector_feedback_search_decision="PASS",
        adaptive_model_vae_regeneration_decision="PASS",
        adaptive_public_negative_probe_decision="PASS",
        adaptive_watermark_retention_decision="PASS",
        adaptive_spoof_rejection_decision="PASS",
        adaptive_replay_control_rejection_decision="PASS",
        adaptive_robustness_claim_allowed=True,
    )
    _write_decision(
        run_root,
        "statistical_confidence_interval_decision.json",
        "statistical_confidence_interval_decision",
        target_fpr,
        heldout_fpr_confidence_upper_within_target=True,
        claim_support_status="formal_cluster_bootstrap_ci_from_fair_detection_calibration",
        cluster_by_video_interval_status="source_video_cluster_bootstrap_complete",
        ci_statistical_cluster_count=30,
        heldout_fpr_ci_record_count=6,
    )
    write_jsonl(
        run_root / "records" / "statistical_confidence_interval_records.jsonl",
        [
            record
            for method_id in (
                "sstw_key_conditioned_flow_trajectory",
                *MODERN_BASELINES,
            )
            for record in (
                {
                    "method_id": method_id,
                    "target_fpr": target_fpr,
                    "statistical_confidence_interval_family": "tpr_at_target_fpr",
                    "ci_evidence_level": "fair_detection_calibration_measured_formal",
                    "cluster_by_video_interval_status": "source_video_cluster_bootstrap_complete",
                    "ci_statistical_cluster_count": 30,
                    "ci_bootstrap_resample_count": 500,
                    "ci_cluster_bootstrap_lower": 0.7,
                    "ci_cluster_bootstrap_upper": 0.9,
                },
                {
                    "method_id": method_id,
                    "target_fpr": target_fpr,
                    "statistical_confidence_interval_family": "heldout_fpr_at_frozen_threshold",
                    "ci_evidence_level": "fair_detection_calibration_measured_formal",
                    "cluster_by_video_interval_status": "source_video_exact_binomial_complete",
                    "ci_statistical_cluster_count": 30,
                    "ci_one_sided_exact_upper": target_fpr,
                },
            )
        ],
    )
    _write_decision(
        run_root,
        "low_fpr_formal_statistics_decision.json",
        "low_fpr_formal_statistics_decision",
        current_profile_low_fpr_claim_allowed=True,
    )
    for file_name, field_name in (
        ("low_fpr_curve_decision.json", "low_fpr_curve_decision"),
        ("video_quality_metric_decision.json", "video_quality_metric_decision"),
        ("efficiency_metric_decision.json", "efficiency_metric_decision"),
        ("paper_result_artifact_skeleton_decision.json", "paper_result_artifact_skeleton_decision"),
        ("fair_detection_calibration_decision.json", "fair_detection_calibration_decision"),
        ("formal_method_baseline_comparison_decision.json", "formal_method_baseline_comparison_decision"),
        ("formal_baseline_difference_interval_decision.json", "formal_baseline_difference_interval_decision"),
    ):
        _write_decision(run_root, file_name, field_name, target_fpr)
    for file_name, field_name in (
        ("formal_internal_ablation_summary_decision.json", "formal_internal_ablation_summary_decision"),
        ("validation_internal_ablation_decision.json", "validation_internal_ablation_decision"),
        ("real_adaptive_attack_decision.json", "real_adaptive_attack_decision"),
        ("real_world_attack_decision.json", "real_world_attack_decision"),
        ("validation_artifact_rebuild_dry_run_decision.json", "validation_artifact_rebuild_dry_run_decision"),
        ("data_split_and_leakage_guard_decision.json", "data_split_and_leakage_guard_decision"),
        ("external_baseline_self_containment_decision.json", "external_baseline_self_containment_decision"),
        ("external_baseline_comparison_decision.json", "external_baseline_comparison_decision"),
        ("sstw_measured_formal_decision.json", "sstw_measured_formal_decision"),
        ("adaptive_attack_decision.json", "adaptive_attack_decision"),
    ):
        _write_decision(run_root, file_name, field_name)
    _write_decision(
        run_root,
        "validation_internal_ablation_decision.json",
        "validation_internal_ablation_decision",
        validation_internal_ablation_evidence_level=(
            "formal_component_removal_video_detector"
        ),
        detector_only_video_reuse_decision="PASS",
        generation_variant_independent_video_decision="PASS",
    )
    generation_variants = (
        "sstw_full_method",
        "endpoint_only_control",
        "without_velocity_constraint",
        "without_endpoint_aware_control",
    )
    detector_only_variants = (
        "trajectory_only_score",
        "without_replay_uncertainty_weighting",
        "without_flow_state_admissibility",
        "generic_ssm_baseline",
    )
    full_traces = ("full_trace_0", "full_trace_1")
    write_jsonl(
        run_root / "records" / "formal_internal_ablation_variant_records.jsonl",
        [
            {
                "method_variant": variant,
                "trajectory_trace_id": trace_id,
                "split": "test",
                "ablation_runtime_profile": profile,
                "metric_status": "measured_formal",
                "formal_internal_ablation_evidence_level": (
                    "formal_component_removal_video_detector"
                ),
                "ablation_video_execution_mode": (
                    "independent_generation_variant_video"
                ),
                "ablation_source_method_variant": variant,
                "ablation_independent_video_generation_required": True,
            }
            for variant in generation_variants
            for trace_id in (
                full_traces
                if variant == "sstw_full_method"
                else (f"{variant}_trace_0", f"{variant}_trace_1")
            )
        ]
        + [
            {
                "method_variant": variant,
                "trajectory_trace_id": trace_id,
                "split": "test",
                "ablation_runtime_profile": profile,
                "metric_status": "measured_formal",
                "formal_internal_ablation_evidence_level": (
                    "formal_component_removal_video_detector"
                ),
                "ablation_video_execution_mode": (
                    "detector_only_reuse_full_method_video"
                ),
                "ablation_source_method_variant": "sstw_full_method",
                "ablation_independent_video_generation_required": False,
            }
            for variant in detector_only_variants
            for trace_id in full_traces
        ],
    )
    _write_decision(
        run_root,
        "motion_threshold_calibration_decision.json",
        "motion_threshold_calibration_decision",
        motion_threshold_calibration_ready=True,
    )
    _write_decision(
        run_root,
        "cross_model_generalization_decision.json",
        "cross_model_generalization_decision",
        cross_model_generalization_claim_scope="supportive_not_primary_fixed_fpr_closure",
        cross_model_generalization_model_ids=["Lightricks/LTX-Video"],
    )
    _write_decision(
        run_root,
        "motion_consistency_exclusion_decision.json",
        "motion_consistency_exclusion_decision",
        motion_consistency_claim_filter_applied=True,
        motion_consistency_included_count=30,
    )
    write_jsonl(
        run_root / "records" / "motion_consistency_exclusion_records.jsonl",
        [{"included_in_motion_claim": True}],
    )
    (run_root / "reports" / "complete_paper_mechanism_claim_report.md").parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    (run_root / "reports" / "complete_paper_mechanism_claim_report.md").write_text(
        "# 完整方法主张审计\n",
        encoding="utf-8",
    )
    write_jsonl(
        run_root / "records" / "formal_baseline_difference_interval_records.jsonl",
        [
            {
                "baseline_method_id": baseline_id,
                "target_fpr": target_fpr,
                "difference_interval_status": "ready",
                "metric_status": "measured_formal",
                "tpr_at_target_fpr_difference": 0.1,
                "difference_ci_lower": 0.01,
            }
            for baseline_id in MODERN_BASELINES
        ],
    )


def _write_config(path: Path, profile: str, target_fpr: float) -> None:
    """写入仅改变 profile、目标 FPR 和样本语义的公共门禁配置。"""

    required_flags = {
        name: True
        for name in (
            "require_complete_paper_mechanism_contract",
            "require_claim_audit_report",
            "require_cluster_aware_statistics",
            "require_claim1_full_support",
            "require_claim1_velocity_causal_gain",
            "require_claim2_full_support",
            "require_claim3_full_support",
            "require_replay_and_sketch_full_support",
            "require_calibrated_probability_posterior",
            "require_per_video_adaptive_attack_optimization",
            "require_real_adaptive_attack_records",
            "require_statistical_confidence_interval_decision",
            "require_heldout_fpr_confidence_upper_bound",
            "require_low_fpr_formal_statistics_blocking_record",
            "require_low_fpr_curve_records",
            "require_formal_internal_ablation_summary",
            "require_internal_ablation_matrix_ready",
            "require_internal_ablation_video_reuse_policy",
            "require_video_quality_metric_records",
            "require_efficiency_metric_records",
            "require_real_world_attack_records",
            "require_paper_result_artifact_skeleton",
            "require_artifact_rebuild_dry_run",
            "require_data_split_and_leakage_guard",
            "require_fair_detection_calibration",
            "require_formal_method_baseline_comparison",
            "require_formal_baseline_difference_interval",
            "require_formal_motion_claim_ready",
            "require_sstw_advantage_claim_ready",
            "require_external_baseline_self_containment_decision",
            "require_external_baseline_comparison_ready",
            "require_sstw_measured_formal_records",
            "require_motion_threshold_calibration_ready",
            "require_adaptive_attack_records",
            "require_cross_model_generalization",
        )
    }
    path.write_text(
        json.dumps(
            {
                "paper_result_level": profile,
                "target_fpr": target_fpr,
                "required_modern_external_baseline_adapter_names": list(MODERN_BASELINES),
                "minimum_sstw_advantage_baseline_count": len(MODERN_BASELINES),
                "minimum_sstw_tpr_at_target_fpr_difference": 0.0,
                "require_sstw_advantage_ci_lower_above_zero": True,
                "minimum_internal_ablation_trace_count": 2,
                "required_internal_ablation_variants": [
                    "sstw_full_method",
                    "endpoint_only_control",
                    "trajectory_only_score",
                    "without_velocity_constraint",
                    "without_endpoint_aware_control",
                    "without_replay_uncertainty_weighting",
                    "without_flow_state_admissibility",
                    "generic_ssm_baseline",
                ],
                "generation_internal_ablation_variants": [
                    "sstw_full_method",
                    "endpoint_only_control",
                    "without_velocity_constraint",
                    "without_endpoint_aware_control",
                ],
                "detector_only_internal_ablation_variants": [
                    "trajectory_only_score",
                    "without_replay_uncertainty_weighting",
                    "without_flow_state_admissibility",
                    "generic_ssm_baseline",
                ],
                **required_flags,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


@pytest.mark.quick
@pytest.mark.parametrize(
    ("profile", "target_fpr"),
    (("probe_paper", 0.1), ("pilot_paper", 0.01), ("full_paper", 0.001)),
)
def test_formal_profiles_use_identical_evidence_closure(
    tmp_path: Path,
    profile: str,
    target_fpr: float,
) -> None:
    """三个正式层级只能改变 FPR 与统计规模, 不能减少三层证据。"""

    run_root = tmp_path / profile
    config_path = tmp_path / f"{profile}.json"
    _write_config(config_path, profile, target_fpr)
    _write_closed_evidence(run_root, target_fpr)

    audit = build_paper_profile_evidence_closure_audit(run_root, config_path)

    assert audit["paper_profile_evidence_closure_decision"] == "PASS"
    assert audit["paper_profile_evidence_closure_missing_requirements"] == []
    assert audit["paper_profile_evidence_closure_required_check_count"] > 20


@pytest.mark.quick
def test_claim3_cannot_be_downgraded_in_any_formal_profile(tmp_path: Path) -> None:
    """replay 后验未获 full support 时, 正式层级不得以降级语义放行。"""

    run_root = tmp_path / "probe_paper"
    config_path = tmp_path / "probe_paper.json"
    _write_config(config_path, "probe_paper", 0.1)
    _write_closed_evidence(run_root, 0.1)
    _write_decision(
        run_root,
        "replay_and_sketch_gate_decision.json",
        "replay_and_sketch_gate_decision",
        claim3_full_support_allowed=False,
    )

    audit = build_paper_profile_evidence_closure_audit(run_root, config_path)

    assert audit["paper_profile_evidence_closure_decision"] == "FAIL"
    assert "claim_3_replay_posterior_full_support_passed" in audit[
        "paper_profile_evidence_closure_missing_requirements"
    ]


@pytest.mark.quick
def test_calibrated_probability_requires_real_state_space_mechanism(tmp_path: Path) -> None:
    """只有校准指标而没有动态状态空间证据时, 正式 profile 必须失败。"""

    run_root = tmp_path / "probe_paper"
    config_path = tmp_path / "probe_paper.json"
    _write_config(config_path, "probe_paper", 0.1)
    _write_closed_evidence(run_root, 0.1)
    _write_decision(
        run_root,
        "formal_flow_evidence_decision.json",
        "formal_flow_evidence_decision",
        posterior_probability_calibration_decision="PASS",
        state_space_posterior_mechanism_decision="FAIL",
    )

    audit = build_paper_profile_evidence_closure_audit(run_root, config_path)

    assert audit["paper_profile_evidence_closure_decision"] == "FAIL"
    assert "calibrated_probability_posterior_passed" in audit[
        "paper_profile_evidence_closure_missing_requirements"
    ]


@pytest.mark.quick
def test_every_profile_requires_positive_sstw_advantage_ci_for_all_baselines(
    tmp_path: Path,
) -> None:
    """任一 baseline 的差值 CI 下界不为正时, 公共闭合器必须阻断。"""

    run_root = tmp_path / "pilot_paper"
    config_path = tmp_path / "pilot_paper.json"
    _write_config(config_path, "pilot_paper", 0.01)
    _write_closed_evidence(run_root, 0.01)
    records_path = run_root / "records" / "formal_baseline_difference_interval_records.jsonl"
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    records[0]["difference_ci_lower"] = 0.0
    write_jsonl(records_path, records)

    audit = build_paper_profile_evidence_closure_audit(run_root, config_path)

    assert audit["paper_profile_evidence_closure_decision"] == "FAIL"
    assert "sstw_advantage_claim_passed" in audit[
        "paper_profile_evidence_closure_missing_requirements"
    ]


@pytest.mark.quick
def test_detector_only_ablation_must_reuse_full_method_videos(tmp_path: Path) -> None:
    """detector-only 消融使用独立视频时, 三档公共闭合器必须阻断。"""

    run_root = tmp_path / "probe_paper"
    config_path = tmp_path / "probe_paper.json"
    _write_config(config_path, "probe_paper", 0.1)
    _write_closed_evidence(run_root, 0.1)
    records_path = run_root / "records" / "formal_internal_ablation_variant_records.jsonl"
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    for record in records:
        if record["method_variant"] == "trajectory_only_score":
            record["trajectory_trace_id"] = "independently_generated_detector_video"
            record["ablation_video_execution_mode"] = "independent_generation_variant_video"
            record["ablation_independent_video_generation_required"] = True
    write_jsonl(records_path, records)

    audit = build_paper_profile_evidence_closure_audit(run_root, config_path)

    assert audit["paper_profile_evidence_closure_decision"] == "FAIL"
    assert "internal_ablation_video_reuse_policy_passed" in audit[
        "paper_profile_evidence_closure_missing_requirements"
    ]
