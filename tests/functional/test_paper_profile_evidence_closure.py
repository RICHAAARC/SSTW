"""验证三个正式 profile 共享同一套论文证据闭合条件。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.protocol.paper_profile_evidence_closure import (
    build_paper_profile_evidence_closure_audit,
)
from evaluation.protocol.record_writer import write_json


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
        adaptive_robustness_claim_allowed=True,
    )
    _write_decision(
        run_root,
        "statistical_confidence_interval_decision.json",
        "statistical_confidence_interval_decision",
        target_fpr,
        heldout_fpr_confidence_upper_within_target=True,
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
        "motion_threshold_calibration_decision.json",
        "motion_threshold_calibration_decision",
        motion_threshold_calibration_ready=True,
    )


def _write_config(path: Path, profile: str, target_fpr: float) -> None:
    """写入仅改变 profile、目标 FPR 和样本语义的公共门禁配置。"""

    required_flags = {
        name: True
        for name in (
            "require_complete_paper_mechanism_contract",
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
            "require_video_quality_metric_records",
            "require_efficiency_metric_records",
            "require_real_world_attack_records",
            "require_paper_result_artifact_skeleton",
            "require_artifact_rebuild_dry_run",
            "require_data_split_and_leakage_guard",
            "require_fair_detection_calibration",
            "require_formal_method_baseline_comparison",
            "require_formal_baseline_difference_interval",
            "require_external_baseline_self_containment_decision",
            "require_external_baseline_comparison_ready",
            "require_sstw_measured_formal_records",
            "require_motion_threshold_calibration_ready",
            "require_adaptive_attack_records",
        )
    }
    path.write_text(
        json.dumps(
            {
                "paper_result_level": profile,
                "target_fpr": target_fpr,
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
