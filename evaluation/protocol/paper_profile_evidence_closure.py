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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 governed JSONL records, 文件不存在时返回空列表。"""

    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise TypeError(f"JSONL 每一行必须是对象: {path}")
        rows.append(payload)
    return rows


def _nonempty_file(path: Path) -> bool:
    """检查报告或 artifact 文件确实存在且包含内容。"""

    return path.is_file() and path.stat().st_size > 0


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


def _safe_float(value: object) -> float | None:
    """把证据字段安全转换为 float, 不可解析时返回 None。"""

    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _safe_int(value: object) -> int | None:
    """把证据字段安全转换为 int, 不可解析时返回 None。"""

    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _sstw_advantage_ready(
    records: list[dict[str, Any]],
    config: Mapping[str, Any],
    target_fpr: float,
) -> tuple[bool, list[str]]:
    """统一审计 SSTW 相对全部正式 baseline 的正差值与 CI 下界。

    该函数只消费公平比较链已经生成的配对差值 records, 不重新计算论文指标。
    probe、pilot 与 full 共用相同判定, 差别只来自各自 target FPR 和样本量。
    """

    required_baselines = {
        str(value)
        for value in config.get("required_modern_external_baseline_adapter_names", [])
        if str(value)
    }
    minimum_count = int(
        config.get("minimum_sstw_advantage_baseline_count", len(required_baselines))
    )
    minimum_difference = float(
        config.get("minimum_sstw_tpr_at_target_fpr_difference", 0.0)
    )
    require_positive_ci = bool(
        config.get("require_sstw_advantage_ci_lower_above_zero", True)
    )
    ready_baselines: set[str] = set()
    failures: list[str] = []
    for record in records:
        baseline_id = str(record.get("baseline_method_id") or "")
        if baseline_id not in required_baselines:
            continue
        if not _target_fpr_matches(record, target_fpr):
            failures.append(f"{baseline_id}:target_fpr_mismatch")
            continue
        if (
            record.get("difference_interval_status") != "ready"
            or record.get("metric_status") != "measured_formal"
        ):
            failures.append(f"{baseline_id}:difference_interval_not_ready")
            continue
        difference = _safe_float(record.get("tpr_at_target_fpr_difference"))
        if difference is None or difference <= minimum_difference:
            failures.append(f"{baseline_id}:difference_not_above_minimum")
            continue
        ci_lower = _safe_float(record.get("difference_ci_lower"))
        if require_positive_ci and (ci_lower is None or ci_lower <= 0.0):
            failures.append(f"{baseline_id}:difference_ci_lower_not_above_zero")
            continue
        ready_baselines.add(baseline_id)
    missing = sorted(required_baselines - ready_baselines)
    failures.extend(f"{baseline_id}:missing" for baseline_id in missing)
    return (
        bool(required_baselines)
        and len(ready_baselines) >= minimum_count
        and ready_baselines == required_baselines
        and not failures,
        failures,
    )


def _cluster_aware_statistics_ready(
    records: list[dict[str, Any]],
    config: Mapping[str, Any],
    target_fpr: float,
) -> bool:
    """验证每个正式方法同时具备 source-video TPR 与 FPR 统计记录。"""

    required_methods = {
        "sstw_key_conditioned_flow_trajectory",
        *{
            str(value)
            for value in config.get(
                "required_modern_external_baseline_adapter_names",
                [],
            )
            if str(value)
        },
    }
    tpr_methods: set[str] = set()
    fpr_methods: set[str] = set()
    for record in records:
        method_id = str(record.get("method_id") or "")
        if method_id not in required_methods or not _target_fpr_matches(
            record,
            target_fpr,
        ):
            continue
        if record.get("ci_evidence_level") != (
            "fair_detection_calibration_measured_formal"
        ):
            continue
        family = record.get("statistical_confidence_interval_family")
        cluster_count = _safe_int(record.get("ci_statistical_cluster_count")) or 0
        if (
            family == "tpr_at_target_fpr"
            and record.get("cluster_by_video_interval_status")
            == "source_video_cluster_bootstrap_complete"
            and cluster_count >= 2
            and (_safe_int(record.get("ci_bootstrap_resample_count")) or 0) >= 200
            and _safe_float(record.get("ci_cluster_bootstrap_lower")) is not None
            and _safe_float(record.get("ci_cluster_bootstrap_upper")) is not None
        ):
            tpr_methods.add(method_id)
        if (
            family == "heldout_fpr_at_frozen_threshold"
            and record.get("cluster_by_video_interval_status")
            == "source_video_exact_binomial_complete"
            and cluster_count > 0
            and _safe_float(record.get("ci_one_sided_exact_upper")) is not None
        ):
            fpr_methods.add(method_id)
    return bool(required_methods) and tpr_methods == required_methods == fpr_methods


def _internal_ablation_profile_evidence_ready(
    records: list[dict[str, Any]],
    decision: Mapping[str, Any],
    config: Mapping[str, Any],
) -> bool:
    """统一审计三档内部消融的视频生成与 detector-only 复用语义。"""

    profile = str(config.get("paper_result_level") or "")
    minimum_trace_count = _safe_int(
        config.get("minimum_internal_ablation_trace_count")
    )
    required_variants = {
        str(value)
        for value in config.get("required_internal_ablation_variants", [])
        if str(value)
    }
    generation_variants = {
        str(value)
        for value in config.get("generation_internal_ablation_variants", [])
        if str(value)
    }
    detector_only_variants = {
        str(value)
        for value in config.get("detector_only_internal_ablation_variants", [])
        if str(value)
    }
    if (
        minimum_trace_count is None
        or minimum_trace_count <= 0
        or not required_variants
        or generation_variants & detector_only_variants
        or generation_variants | detector_only_variants != required_variants
    ):
        return False
    rows = [
        record
        for record in records
        if record.get("metric_status") == "measured_formal"
        and record.get("formal_internal_ablation_evidence_level")
        == "formal_component_removal_video_detector"
        and record.get("ablation_runtime_profile") == profile
        and str(record.get("split") or "").lower()
        in {"test", "heldout", "heldout_test"}
    ]
    traces_by_variant = {
        variant: {
            str(record.get("trajectory_trace_id"))
            for record in rows
            if record.get("method_variant") == variant
            and record.get("trajectory_trace_id")
        }
        for variant in required_variants
    }
    full_traces = traces_by_variant.get("sstw_full_method", set())
    detector_only_ready = bool(full_traces) and all(
        traces_by_variant[variant] == full_traces
        and all(
            record.get("ablation_video_execution_mode")
            == "detector_only_reuse_full_method_video"
            and record.get("ablation_source_method_variant") == "sstw_full_method"
            and record.get("ablation_independent_video_generation_required") is False
            for record in rows
            if record.get("method_variant") == variant
        )
        for variant in detector_only_variants
    )
    generation_trace_sets = [
        traces_by_variant[variant] for variant in sorted(generation_variants)
    ]
    generation_ready = all(
        len(traces_by_variant[variant]) >= minimum_trace_count
        and all(
            record.get("ablation_video_execution_mode")
            == "independent_generation_variant_video"
            and record.get("ablation_source_method_variant") == variant
            and record.get("ablation_independent_video_generation_required") is True
            for record in rows
            if record.get("method_variant") == variant
        )
        for variant in generation_variants
    ) and all(
        not left & right
        for index, left in enumerate(generation_trace_sets)
        for right in generation_trace_sets[index + 1 :]
    )
    trace_coverage_ready = all(
        len(trace_ids) >= minimum_trace_count
        for trace_ids in traces_by_variant.values()
    )
    return (
        decision.get("validation_internal_ablation_decision") == "PASS"
        and decision.get("detector_only_video_reuse_decision") == "PASS"
        and decision.get("generation_variant_independent_video_decision") == "PASS"
        and detector_only_ready
        and generation_ready
        and trace_coverage_ready
    )


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
    motion_exclusion = _read_json(
        artifacts / "motion_consistency_exclusion_decision.json"
    )
    difference_records = _read_jsonl(
        root / "records" / "formal_baseline_difference_interval_records.jsonl"
    )
    statistical_records = _read_jsonl(
        root / "records" / "statistical_confidence_interval_records.jsonl"
    )
    internal_ablation_records = _read_jsonl(
        root / "records" / "formal_internal_ablation_variant_records.jsonl"
    )
    if not internal_ablation_records:
        internal_ablation_records = _read_jsonl(
            root / "records" / "validation_internal_ablation_records.jsonl"
        )

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
        "claim_audit_report_passed",
        _required(config, "require_claim_audit_report"),
        _decision_passed(complete_claim, "complete_paper_mechanism_claim_decision")
        and _nonempty_file(root / "reports" / "complete_paper_mechanism_claim_report.md"),
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
            "adaptive_detector_feedback_search_decision",
        )
        and _decision_passed(
            adaptive_execution,
            "adaptive_model_vae_regeneration_decision",
        )
        and _decision_passed(
            adaptive_execution,
            "adaptive_public_negative_probe_decision",
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
        "cluster_aware_statistics_passed",
        _required(config, "require_cluster_aware_statistics"),
        _decision_passed(statistical_ci, "statistical_confidence_interval_decision")
        and statistical_ci.get("claim_support_status")
        == "formal_cluster_bootstrap_ci_from_fair_detection_calibration"
        and statistical_ci.get("cluster_by_video_interval_status")
        == "source_video_cluster_bootstrap_complete"
        and (_safe_int(statistical_ci.get("ci_statistical_cluster_count")) or 0) > 0
        and (_safe_int(statistical_ci.get("heldout_fpr_ci_record_count")) or 0) > 0
        and _cluster_aware_statistics_ready(
            statistical_records,
            config,
            target_fpr,
        ),
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
        "internal_ablation_video_reuse_policy_passed",
        _required(config, "require_internal_ablation_video_reuse_policy"),
        _internal_ablation_profile_evidence_ready(
            internal_ablation_records,
            validation_ablation,
            config,
        ),
    )
    add(
        "video_quality_metrics_passed",
        _required(config, "require_video_quality_metric_records"),
        _decision_passed(quality, "video_quality_metric_decision")
        and _target_fpr_matches(quality, target_fpr),
    )
    add(
        "baseline_matched_video_quality_passed",
        _required(config, "require_baseline_matched_video_quality_metrics"),
        quality.get("baseline_matched_video_quality_ready") is True
        and quality.get("sstw_paired_video_quality_ready") is True
        and quality.get("video_quality_comparison_protocol")
        == config.get("video_quality_comparison_protocol")
        and not quality.get("video_quality_missing_method_ids")
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
    sstw_advantage_ready, sstw_advantage_failures = _sstw_advantage_ready(
        difference_records,
        config,
        target_fpr,
    )
    add(
        "sstw_advantage_claim_passed",
        _required(config, "require_sstw_advantage_claim_ready"),
        _decision_passed(difference, "formal_baseline_difference_interval_decision")
        and _target_fpr_matches(difference, target_fpr)
        and sstw_advantage_ready,
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
        "formal_motion_claim_passed",
        _required(config, "require_formal_motion_claim_ready"),
        _decision_passed(
            motion_exclusion,
            "motion_consistency_exclusion_decision",
        )
        and motion_exclusion.get("motion_consistency_claim_filter_applied") is True
        and (_safe_int(motion_exclusion.get("motion_consistency_included_count")) or 0) > 0
        and _nonempty_file(
            root / "records" / "motion_consistency_exclusion_records.jsonl"
        ),
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
        "sstw_advantage_claim_failures": sstw_advantage_failures,
        "post_gate_requirements": ["reviewer_evidence_index_required"],
    }
