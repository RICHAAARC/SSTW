"""统一审计三个正式 paper profile 的论文证据闭合条件。

该模块属于通用工程治理写法。probe_paper、pilot_paper 和 full_paper
分别使用不同的样本规模与目标 FPR, 但它们必须读取同一组机制证据和正式结果
artifact。把公共条件集中在这里, 可以避免三个入口各自维护一套门禁而发生静默漂移。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from evaluation.attacks.video_runtime_attack_protocol import (
    load_protocol_config_with_shared_attack_protocol,
)


def _read_json(path: Path) -> dict[str, Any]:
    """读取 governed JSON artifact, 文件不存在时返回空对象。"""

    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON 顶层必须是对象: {path}")
    return payload


def _decision_passed(payload: Mapping[str, Any], *field_names: str) -> bool:
    """判断 artifact 的任一规范 decision 字段是否为 PASS。"""

    return any(payload.get(field_name) == "PASS" for field_name in field_names)


def _target_fpr_matches(payload: Mapping[str, Any], target_fpr: float) -> bool:
    """验证 artifact 明确绑定到当前 profile 的目标 FPR。"""

    value = payload.get("target_fpr")
    if value is None:
        return False
    try:
        return math.isclose(float(value), target_fpr, rel_tol=0.0, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False


def _required(config: Mapping[str, Any], *flag_names: str) -> bool:
    """只在公共契约显式启用能力时激活对应门禁。"""

    return any(config.get(flag_name) is True for flag_name in flag_names)


def build_paper_profile_evidence_closure_audit(
    run_root: str | Path,
    config_path: str | Path,
) -> dict[str, Any]:
    """构建三个正式 profile 共用的论文证据闭合审计。

    此处不重新计算任何实验分数。该函数只核对由内层 runner 生成的 governed
    artifacts, 因而可以被 probe、pilot 和 full 的外层门禁共同复用。审稿证据
    索引依赖 profile 门禁本身, 所以它在门禁通过后由 package 阶段校验, 不在此处
    形成循环依赖。
    """

    root = Path(run_root)
    config = load_protocol_config_with_shared_attack_protocol(config_path)
    target_fpr = float(config["target_fpr"])
    artifacts = root / "artifacts"

    complete_claim = _read_json(artifacts / "complete_paper_mechanism_claim_decision.json")
    three_layer = _read_json(artifacts / "three_layer_mechanism_evidence_decision.json")
    flow_evidence = _read_json(artifacts / "formal_flow_evidence_decision.json")
    replay = _read_json(artifacts / "replay_and_sketch_gate_decision.json")
    adaptive_execution = _read_json(artifacts / "formal_adaptive_attack_execution_decision.json")
    statistical_ci = _read_json(artifacts / "statistical_confidence_interval_decision.json")
    low_fpr = _read_json(artifacts / "low_fpr_formal_statistics_decision.json")
    low_fpr_curve = _read_json(artifacts / "low_fpr_curve_decision.json")
    formal_ablation = _read_json(artifacts / "formal_internal_ablation_summary_decision.json")
    validation_ablation = _read_json(artifacts / "validation_internal_ablation_decision.json")
    quality = _read_json(artifacts / "video_quality_metric_decision.json")
    efficiency = _read_json(artifacts / "efficiency_metric_decision.json")
    real_adaptive = _read_json(artifacts / "real_adaptive_attack_decision.json")
    real_world = _read_json(artifacts / "real_world_attack_decision.json")
    skeleton = _read_json(artifacts / "paper_result_artifact_skeleton_decision.json")
    rebuild = _read_json(artifacts / "validation_artifact_rebuild_dry_run_decision.json")
    data_guard = _read_json(artifacts / "data_split_and_leakage_guard_decision.json")
    fair = _read_json(artifacts / "fair_detection_calibration_decision.json")
    comparison = _read_json(artifacts / "formal_method_baseline_comparison_decision.json")
    difference = _read_json(artifacts / "formal_baseline_difference_interval_decision.json")
    baseline_self_containment = _read_json(
        artifacts / "external_baseline_self_containment_decision.json"
    )
    baseline_comparison = _read_json(artifacts / "external_baseline_comparison_decision.json")
    sstw_measured = _read_json(artifacts / "sstw_measured_formal_decision.json")
    motion_threshold = _read_json(artifacts / "motion_threshold_calibration_decision.json")
    adaptive = _read_json(artifacts / "adaptive_attack_decision.json")
    cross_model = _read_json(artifacts / "cross_model_generalization_decision.json")

    checks: dict[str, bool] = {}

    def add(check_name: str, enabled: bool, passed: bool) -> None:
        """登记已启用的公共证据条件, 未启用条件不影响临时测试配置。"""

        if enabled:
            checks[check_name] = bool(passed)

    common_contract_declared = bool(config.get("paper_profile_common_contract_path"))
    add(
        "common_profile_contract_matched",
        common_contract_declared,
        config.get("paper_profile_common_contract_status") == "matched",
    )
    add(
        "complete_paper_mechanism_claim_passed",
        _required(config, "require_complete_paper_mechanism_contract"),
        _decision_passed(complete_claim, "complete_paper_mechanism_claim_decision"),
    )
    add(
        "claim_1_velocity_constraint_full_support_passed",
        _required(config, "require_claim1_full_support", "require_claim1_velocity_causal_gain"),
        _decision_passed(
            complete_claim,
            "claim_1_velocity_constraint_detectable_watermark_decision",
        )
        and _decision_passed(
            three_layer,
            "claim_1_velocity_constraint_detectable_watermark_decision",
        )
        and _decision_passed(
            flow_evidence,
            "claim_1_velocity_constraint_detectable_watermark_decision",
        ),
    )
    add(
        "claim_2_path_evidence_full_support_passed",
        _required(config, "require_claim2_full_support"),
        _decision_passed(
            complete_claim,
            "claim_2_path_evidence_independent_gain_decision",
        )
        and _decision_passed(
            three_layer,
            "claim_2_path_evidence_independent_gain_decision",
        )
        and _decision_passed(
            flow_evidence,
            "claim_2_path_evidence_independent_gain_decision",
        ),
    )
    add(
        "claim_3_replay_posterior_full_support_passed",
        _required(config, "require_claim3_full_support", "require_replay_and_sketch_full_support"),
        _decision_passed(
            complete_claim,
            "claim_3_attacked_video_replay_posterior_decision",
        )
        and _decision_passed(replay, "replay_and_sketch_gate_decision")
        and replay.get("claim3_full_support_allowed") is True,
    )
    add(
        "calibrated_probability_posterior_passed",
        _required(config, "require_calibrated_probability_posterior"),
        _decision_passed(flow_evidence, "formal_flow_evidence_decision")
        and _decision_passed(flow_evidence, "posterior_probability_calibration_decision")
        and _decision_passed(flow_evidence, "state_space_posterior_mechanism_decision")
        and _decision_passed(flow_evidence, "formal_negative_hypothesis_family_decision"),
    )
    add(
        "per_video_adaptive_attack_optimization_passed",
        _required(
            config,
            "require_per_video_adaptive_attack_optimization",
            "require_real_adaptive_attack_records",
        ),
        _decision_passed(
            adaptive_execution,
            "formal_adaptive_attack_execution_decision",
        )
        and adaptive_execution.get("per_video_adaptive_attack_optimization") is True
        and _decision_passed(
            adaptive_execution,
            "adaptive_attack_query_provenance_decision",
        )
        and _decision_passed(
            adaptive_execution,
            "adaptive_watermark_retention_decision",
        )
        and _decision_passed(
            adaptive_execution,
            "adaptive_spoof_rejection_decision",
        )
        and _decision_passed(
            adaptive_execution,
            "adaptive_replay_control_rejection_decision",
        )
        and adaptive_execution.get("adaptive_robustness_claim_allowed") is True,
    )
    add(
        "statistical_confidence_interval_passed",
        _required(config, "require_statistical_confidence_interval_decision", "require_confidence_interval_report"),
        _decision_passed(statistical_ci, "statistical_confidence_interval_decision")
        and _target_fpr_matches(statistical_ci, target_fpr),
    )
    add(
        "heldout_fpr_confidence_upper_within_target",
        _required(config, "require_heldout_fpr_confidence_upper_bound"),
        statistical_ci.get("heldout_fpr_confidence_upper_within_target") is True,
    )
    add(
        "low_fpr_formal_statistics_passed",
        _required(config, "require_low_fpr_formal_statistics_blocking_record"),
        _decision_passed(low_fpr, "low_fpr_formal_statistics_decision")
        and low_fpr.get("current_profile_low_fpr_claim_allowed") is True,
    )
    add(
        "low_fpr_curve_passed",
        _required(config, "require_low_fpr_curve_records"),
        _decision_passed(low_fpr_curve, "low_fpr_curve_decision")
        and _target_fpr_matches(low_fpr_curve, target_fpr),
    )
    add(
        "formal_internal_ablation_summary_passed",
        _required(config, "require_formal_internal_ablation_summary"),
        _decision_passed(formal_ablation, "formal_internal_ablation_summary_decision"),
    )
    add(
        "internal_ablation_matrix_passed",
        _required(config, "require_internal_ablation_matrix_ready"),
        _decision_passed(
            validation_ablation,
            "validation_internal_ablation_decision",
            "internal_ablation_decision",
        ),
    )
    add(
        "video_quality_metrics_passed",
        _required(config, "require_video_quality_metric_records"),
        _decision_passed(quality, "video_quality_metric_decision")
        and _target_fpr_matches(quality, target_fpr),
    )
    add(
        "efficiency_metrics_passed",
        _required(config, "require_efficiency_metric_records"),
        _decision_passed(efficiency, "efficiency_metric_decision")
        and _target_fpr_matches(efficiency, target_fpr),
    )
    add(
        "real_adaptive_attack_records_passed",
        _required(config, "require_real_adaptive_attack_records"),
        _decision_passed(real_adaptive, "real_adaptive_attack_decision"),
    )
    add(
        "real_world_attack_records_passed",
        _required(config, "require_real_world_attack_records"),
        _decision_passed(real_world, "real_world_attack_decision"),
    )
    add(
        "paper_result_artifact_skeleton_passed",
        _required(
            config,
            "require_paper_result_artifact_skeleton",
            "require_complete_result_artifact_skeleton",
        ),
        _decision_passed(skeleton, "paper_result_artifact_skeleton_decision")
        and _target_fpr_matches(skeleton, target_fpr),
    )
    add(
        "artifact_rebuild_dry_run_passed",
        _required(config, "require_artifact_rebuild_dry_run", "require_artifact_rebuild_report"),
        _decision_passed(
            rebuild,
            "validation_artifact_rebuild_dry_run_decision",
            "artifact_rebuild_dry_run_decision",
        ),
    )
    add(
        "data_split_and_leakage_guard_passed",
        _required(config, "require_data_split_and_leakage_guard"),
        _decision_passed(data_guard, "data_split_and_leakage_guard_decision"),
    )
    add(
        "fair_detection_calibration_passed",
        _required(config, "require_fair_detection_calibration"),
        _decision_passed(fair, "fair_detection_calibration_decision")
        and _target_fpr_matches(fair, target_fpr),
    )
    add(
        "formal_method_baseline_comparison_passed",
        _required(config, "require_formal_method_baseline_comparison"),
        _decision_passed(comparison, "formal_method_baseline_comparison_decision")
        and _target_fpr_matches(comparison, target_fpr),
    )
    add(
        "formal_baseline_difference_interval_passed",
        _required(config, "require_formal_baseline_difference_interval"),
        _decision_passed(difference, "formal_baseline_difference_interval_decision")
        and _target_fpr_matches(difference, target_fpr),
    )
    add(
        "external_baseline_self_containment_passed",
        _required(
            config,
            "require_external_baseline_self_containment_decision",
            "require_external_baseline_self_contained_outputs",
        ),
        _decision_passed(
            baseline_self_containment,
            "external_baseline_self_containment_decision",
        ),
    )
    add(
        "external_baseline_comparison_passed",
        _required(
            config,
            "require_external_baseline_comparison_ready",
            "require_modern_external_baseline_formal_results",
        ),
        _decision_passed(
            baseline_comparison,
            "external_baseline_comparison_decision",
        ),
    )
    add(
        "sstw_measured_formal_records_passed",
        _required(config, "require_sstw_measured_formal_records"),
        _decision_passed(sstw_measured, "sstw_measured_formal_decision"),
    )
    add(
        "motion_threshold_calibration_passed",
        _required(config, "require_motion_threshold_calibration_ready"),
        motion_threshold.get("motion_threshold_calibration_ready") is True,
    )
    add(
        "adaptive_attack_protocol_passed",
        _required(config, "require_adaptive_attack_records"),
        _decision_passed(adaptive, "adaptive_attack_decision"),
    )
    add(
        "cross_model_generalization_passed",
        _required(config, "require_cross_model_generalization"),
        _decision_passed(cross_model, "cross_model_generalization_decision")
        and cross_model.get("cross_model_generalization_claim_scope")
        == "supportive_not_primary_fixed_fpr_closure"
        and bool(cross_model.get("cross_model_generalization_model_ids")),
    )

    missing = [name for name, passed in checks.items() if not passed]
    decision = "PASS" if not missing else "FAIL"
    return {
        "paper_profile_evidence_closure_decision": decision,
        "paper_result_level": config.get("paper_result_level"),
        "target_fpr": target_fpr,
        "paper_profile_evidence_closure_checks": checks,
        "paper_profile_evidence_closure_required_check_count": len(checks),
        "paper_profile_evidence_closure_missing_requirements": missing,
        "paper_profile_evidence_closure_missing_requirement_count": len(missing),
        "post_gate_requirements": ["reviewer_evidence_index_required"],
    }
