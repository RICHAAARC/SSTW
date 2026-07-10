import json
from pathlib import Path

import pytest

from experiments.generative_video_model_probe.external_baseline_runner import run_external_baseline_status
from experiments.generative_video_model_probe.paper_profile_gate import (
    build_paper_profile_gate_audit,
    write_paper_profile_gate_audit,
)
from main.attacks.video_runtime_attack_protocol import (
    FULL_PAPER_NON_RUNTIME_ATTACK_PROTOCOLS,
    FULL_PAPER_RUNTIME_ATTACKS,
)
from main.protocol.record_writer import write_json, write_jsonl


EXTERNAL_BASELINE_NAMES = (
    "explicit_dtw_temporal_alignment",
    "explicit_frame_matching_temporal_registration",
    "videoshield",
    "vidsig",
    "videoseal",
    "revmark",
    "wam_frame",
)
MODERN_EXTERNAL_BASELINE_NAMES = {
    "videoshield",
    "vidsig",
    "videoseal",
    "revmark",
    "wam_frame",
}
REQUIRED_RUNTIME_ATTACK_NAMES = FULL_PAPER_RUNTIME_ATTACKS
REQUIRED_NON_RUNTIME_ATTACK_PROTOCOLS = FULL_PAPER_NON_RUNTIME_ATTACK_PROTOCOLS
REQUIRED_ANCHOR_KEYS = tuple(f"prompt_0::seed_0::{attack_name}" for attack_name in REQUIRED_RUNTIME_ATTACK_NAMES)


def _formal_runtime_attack_record(attack_name: str) -> dict:
    """构造 paper gate 可接受的正式 runtime attack 记录。

    该 fixture 明确表达测试中的 attack 不是 proxy, 避免测试继续依赖旧的弱证据层级。
    """

    return {
        "attack_name": attack_name,
        "attack_runtime_status": "ready",
        "runtime_attack_implementation_level": "formal_runtime_video_transform",
        "runtime_attack_formal_evidence_level": "formal_runtime_video_transform",
        "runtime_attack_claim_level": "paper_runtime_attack_protocol",
        "runtime_attack_proxy_free": True,
    }


def _formal_runtime_detection_record(attack_name: str) -> dict:
    """构造 paper gate 可接受的正式 SSTW 视频内容检测记录。"""

    return {
        "attack_name": attack_name,
        "runtime_detection_status": "ready",
        "sstw_detector_evidence_level": "attacked_video_content_detector",
        "sstw_detector_input_contract": "video_file_plus_project_watermark_key",
        "sstw_raw_detector_score": 0.82,
        "raw_detector_score": 0.82,
        "trajectory_trace_used_for_score": False,
        "runtime_detection_claim_level": "formal_paper_detector",
    }


def _formal_sstw_measured_records() -> list[dict]:
    """构造包含 positive 与 clean negative 的正式 SSTW measured_formal fixture。"""

    records = [
        {
            "metric_status": "measured_formal",
            "sample_role": "attacked_positive",
            "sstw_score": 0.82,
            "sstw_raw_detector_score": 0.82,
            "raw_detector_score": 0.82,
            "prompt_id": "prompt_0",
            "seed_id": "seed_0",
            "attack_name": attack_name,
            "sstw_detector_evidence_level": "attacked_video_content_detector",
            "sstw_detector_input_contract": "video_file_plus_project_watermark_key",
            "trajectory_trace_used_for_score": False,
            "runtime_detection_claim_level": "formal_paper_detector",
            "claim_support_status": "sstw_measured_formal_paper_profile_claim_candidate",
        }
        for attack_name in REQUIRED_RUNTIME_ATTACK_NAMES
    ]
    records.append({
        "metric_status": "measured_formal",
        "sample_role": "clean_negative",
        "sstw_score": 0.05,
        "sstw_clean_negative_score": 0.05,
        "prompt_id": "negative_prompt_0",
        "seed_id": "seed_0",
        "clean_negative_evidence_level": "project_owned_clean_video_content_detector",
        "trajectory_trace_used_for_score": False,
    })
    return records


def _formal_internal_ablation_summary_records() -> list[dict]:
    """构造全变体 measured_formal 的内部消融汇总记录。"""

    return [
        {
            "method_variant": variant,
            "metric_status": "measured_formal",
            "formal_internal_ablation_evidence_level": "formal_component_removal_video_detector",
        }
        for variant in (
            "sstw_full_method",
            "endpoint_only_control",
            "trajectory_only_score",
            "without_velocity_constraint",
            "without_endpoint_aware_control",
            "without_replay_uncertainty_weighting",
            "without_flow_state_admissibility",
            "generic_ssm_baseline",
        )
    ]


def _formal_adaptive_attack_record(protocol_name: str) -> dict:
    """构造非 runtime / adaptive 协议的正式执行证据记录。"""

    return {
        "adaptive_attack_name": protocol_name,
        "non_runtime_attack_protocol": protocol_name,
        "adaptive_attack_status": "ready",
        "metric_status": "measured_formal",
        "adaptive_attack_evidence_level": "formal_adaptive_attack_execution",
        "adaptive_robustness_claim_allowed": True,
    }


def _external_baseline_self_containment_pass_payload() -> dict:
    """构造 probe-paper gate 接受的完整 external baseline 自包含 PASS 摘要。"""

    return {
        "external_baseline_self_containment_decision": "PASS",
        "claim_support_status": "external_baseline_self_contained_measured_formal_ready",
        "required_modern_external_baseline_adapter_names": sorted(MODERN_EXTERNAL_BASELINE_NAMES),
        "required_modern_external_baseline_adapter_count": len(MODERN_EXTERNAL_BASELINE_NAMES),
        "self_contained_modern_external_baseline_count": len(MODERN_EXTERNAL_BASELINE_NAMES),
        "missing_self_contained_modern_external_baseline_names": [],
        "missing_repository_generated_official_bundle_modern_external_baseline_names": [],
        "missing_clean_negative_modern_external_baseline_names": [],
        "missing_score_extraction_modern_external_baseline_names": [],
        "missing_official_identity_modern_external_baseline_names": [],
        "missing_anchor_modern_external_baseline_names": [],
        "missing_formal_modern_external_baseline_names": [],
        "missing_self_containment_requirements": [],
        "self_containment_missing_requirement_count": 0,
        "baseline_self_containment_rows": [
            {
                "baseline_name": name,
                "external_baseline_self_contained": True,
                "repository_generated_official_bundle_ready": True,
                "clean_negative_ready": True,
                "score_extraction_ready": True,
                "official_baseline_identity_ready": True,
                "anchor_ready": True,
                "measured_formal_record_count": 1,
            }
            for name in sorted(MODERN_EXTERNAL_BASELINE_NAMES)
        ],
    }


def _formal_external_baseline_records() -> list[dict]:
    """构造 probe-paper 通过所需的完整 external baseline records fixture。"""
    records: list[dict] = []
    for name in sorted(MODERN_EXTERNAL_BASELINE_NAMES):
        record = {
            "external_baseline_name": name,
            "external_baseline_layer": "modern_external_baseline",
            "metric_status": "measured_formal",
            "claim_support_status": "modern_external_baseline_formal_measured",
        }
        record.update({
            "prompt_id": "prompt_0",
            "seed_id": "seed_0",
            "attack_name": "video_compression_runtime",
            "external_baseline_score": 0.6,
            "external_baseline_raw_detector_score": 0.6,
            "external_baseline_score_semantics": "watermark_presence_detector_score",
            "external_baseline_score_orientation": "higher_is_more_watermarked",
            "external_baseline_clean_negative_score": 0.08,
            "external_baseline_clean_negative_video_path": f"official/{name}/clean_negative.mp4",
            "external_baseline_official_output_path": f"official/{name}/official_output.json",
            "external_baseline_official_command_manifest_path": f"official/{name}/official_command_manifest.json",
            "external_baseline_official_result_provenance": "repository_generated_from_third_party_official_code",
            "external_baseline_official_result_bundle_path": f"official/{name}/official_result_bundle.json",
            "external_baseline_official_execution_manifest_path": f"official/{name}/official_execution_manifest.json",
            "external_baseline_official_score_extraction_policy": "test_official_detector_confidence",
            "external_baseline_official_reference_protocol_anchor": "same_prompt_seed_attack_runtime_comparison_unit",
        })
        records.append(record)
    return records


@pytest.mark.quick
def test_paper_profile_gate_blocks_empty_run(tmp_path: Path) -> None:
    """空 run_root 必须被 probe-paper gate 阻断, 不能进入 pilot_paper。"""
    audit = build_paper_profile_gate_audit(tmp_path / "empty_run")

    assert audit["paper_profile_gate_decision"] == "FAIL"
    assert audit["claim_support_status"] == "probe_paper_blocked"
    assert audit["full_paper_allowed"] is False
    assert "small_scale_claim_pilot_gate_passed" not in audit["missing_validation_requirements"]
    assert "validation_generation_records_ready" in audit["missing_validation_requirements"]
    assert "validation_internal_ablation_records_ready" in audit["missing_validation_requirements"]
    assert "validation_sstw_measured_formal_records_ready" in audit["missing_validation_requirements"]
    assert "validation_fair_detection_calibration_ready" in audit["missing_validation_requirements"]
    assert "validation_formal_method_baseline_comparison_ready" in audit["missing_validation_requirements"]
    assert "validation_formal_baseline_difference_interval_ready" in audit["missing_validation_requirements"]
    assert "paper_profile_formal_internal_ablation_ready" in audit["missing_validation_requirements"]
    assert "validation_low_fpr_formal_statistics_blocking_record_ready" in audit["missing_validation_requirements"]
    assert "validation_motion_consistency_exclusion_report_ready" in audit["missing_validation_requirements"]


@pytest.mark.quick
def test_paper_profile_gate_rejects_pilot_profile_as_validation(tmp_path: Path) -> None:
    """pilot profile 不能冒充 probe-paper profile。"""
    run_root = tmp_path / "run"
    generation_records = []
    for prompt_index in range(8):
        for seed_index in range(3):
            generation_records.append({
                "generation_status": "success",
                "colab_runtime_profile": "pilot",
                "prompt_id": f"prompt_{prompt_index}",
                "seed_id": f"seed_{seed_index}",
            })
    write_jsonl(run_root / "records" / "generation_records.jsonl", generation_records)
    audit = build_paper_profile_gate_audit(run_root)

    assert audit["paper_profile_gate_decision"] == "FAIL"
    assert audit["validation_generation_record_count"] == 0
    assert "validation_generation_records_ready" in audit["missing_validation_requirements"]


@pytest.mark.quick
def test_paper_profile_gate_cannot_disable_fair_comparison_hard_requirements(tmp_path: Path) -> None:
    """probe_paper 不能通过配置关闭公平比较硬前置后进入 pilot_paper。"""
    config_path = tmp_path / "probe_paper_config.json"
    config_path.write_text(json.dumps({
        "target_fpr": 0.1,
        "paper_result_level": "probe_paper",
        "minimum_prompt_count": 0,
        "minimum_seed_per_prompt": 0,
        "minimum_attack_count": 0,
        "minimum_external_baseline_measured_adapter_count": 0,
        "minimum_modern_external_baseline_formal_adapter_count": 0,
        "required_modern_external_baseline_adapter_names": [],
        "require_external_baseline_status_records": False,
        "require_external_baseline_comparison_records": False,
        "require_external_baseline_self_containment_decision": False,
        "require_sstw_measured_formal_records": False,
        "require_fair_detection_calibration": False,
        "require_formal_method_baseline_comparison": False,
        "require_formal_baseline_difference_interval": False,
        "require_data_split_and_leakage_guard": False,
        "require_sstw_advantage_claim_ready": False,
        "require_motion_threshold_calibration_ready": False,
        "require_formal_motion_claim_ready": False,
        "require_motion_consistency_exclusion_report": False,
        "require_internal_ablation_records": False,
        "require_formal_internal_ablation_summary": False,
        "require_adaptive_attack_records": False,
        "require_replay_or_sketch_records_or_claim3_downgrade": False,
        "require_confidence_interval_report": False,
        "require_low_fpr_formal_statistics_blocking_record": False,
        "require_artifact_rebuild_dry_run": False,
    }), encoding="utf-8")

    audit = build_paper_profile_gate_audit(tmp_path / "run", config_path)

    assert audit["paper_profile_gate_decision"] == "FAIL"
    assert audit["claim_support_status"] == "probe_paper_blocked"
    assert audit["paper_profile_hard_required_config_missing_count"] == 8
    assert "require_fair_detection_calibration_must_be_true" in audit["missing_validation_requirements"]
    assert "require_formal_method_baseline_comparison_must_be_true" in audit["missing_validation_requirements"]
    assert "require_formal_baseline_difference_interval_must_be_true" in audit["missing_validation_requirements"]
    assert "require_sstw_measured_formal_records_must_be_true" in audit["missing_validation_requirements"]


@pytest.mark.quick
def test_probe_paper_gate_cannot_disable_advantage_claim_requirement(tmp_path: Path) -> None:
    """probe_paper 是 fpr=0.1 论文闭合层, 不能关闭 SSTW 优势 claim 硬前置。"""
    config_path = tmp_path / "probe_paper_config.json"
    config_path.write_text(json.dumps({
        "target_fpr": 0.1,
        "paper_result_level": "probe_paper",
        "minimum_prompt_count": 0,
        "minimum_seed_per_prompt": 0,
        "minimum_attack_count": 0,
        "minimum_external_baseline_measured_adapter_count": 0,
        "minimum_modern_external_baseline_formal_adapter_count": 0,
        "required_modern_external_baseline_adapter_names": [],
        "require_external_baseline_status_records": False,
        "require_external_baseline_comparison_records": False,
        "require_external_baseline_self_containment_decision": False,
        "require_sstw_measured_formal_records": False,
        "require_fair_detection_calibration": False,
        "require_formal_method_baseline_comparison": False,
        "require_formal_baseline_difference_interval": False,
        "require_data_split_and_leakage_guard": False,
        "require_sstw_advantage_claim_ready": False,
        "require_motion_threshold_calibration_ready": False,
        "require_formal_motion_claim_ready": False,
        "require_motion_consistency_exclusion_report": False,
        "require_internal_ablation_records": False,
        "require_formal_internal_ablation_summary": False,
        "require_adaptive_attack_records": False,
        "require_replay_or_sketch_records_or_claim3_downgrade": False,
        "require_confidence_interval_report": False,
        "require_low_fpr_formal_statistics_blocking_record": False,
        "require_artifact_rebuild_dry_run": False,
    }), encoding="utf-8")

    audit = build_paper_profile_gate_audit(tmp_path / "run", config_path)

    assert audit["paper_profile_gate_decision"] == "FAIL"
    assert audit["claim_support_status"] == "probe_paper_blocked"
    assert audit["paper_profile_hard_required_config_missing_count"] == 8
    assert "require_sstw_advantage_claim_ready_must_be_true" in audit["missing_validation_requirements"]


@pytest.mark.quick
def test_paper_profile_gate_rejects_stale_self_containment_pass_without_rows(tmp_path: Path) -> None:
    """旧版 self-containment 仅写 PASS 不能满足 probe_paper 公平比较门禁。"""
    run_root = tmp_path / "run"
    config_path = tmp_path / "probe_paper_config.json"
    config_path.write_text(json.dumps({
        "target_fpr": 0.1,
        "paper_result_level": "probe_paper",
        "minimum_prompt_count": 0,
        "minimum_seed_per_prompt": 0,
        "minimum_attack_count": 0,
        "minimum_external_baseline_measured_adapter_count": 0,
        "minimum_modern_external_baseline_formal_adapter_count": 1,
        "required_modern_external_baseline_adapter_names": ["videoseal"],
        "require_external_baseline_status_records": False,
        "require_external_baseline_comparison_records": True,
        "require_external_baseline_self_containment_decision": True,
        "require_sstw_measured_formal_records": True,
        "require_fair_detection_calibration": True,
        "require_formal_method_baseline_comparison": True,
        "require_formal_baseline_difference_interval": True,
        "require_data_split_and_leakage_guard": True,
        "require_motion_threshold_calibration_ready": False,
        "require_formal_motion_claim_ready": False,
        "require_motion_consistency_exclusion_report": False,
        "require_internal_ablation_records": False,
        "require_formal_internal_ablation_summary": False,
        "require_adaptive_attack_records": False,
        "require_replay_or_sketch_records_or_claim3_downgrade": False,
        "require_confidence_interval_report": False,
        "require_low_fpr_formal_statistics_blocking_record": False,
        "require_artifact_rebuild_dry_run": False,
    }), encoding="utf-8")
    write_json(run_root / "artifacts" / "external_baseline_self_containment_decision.json", {
        "external_baseline_self_containment_decision": "PASS",
        "claim_support_status": "legacy_pass_without_required_rows",
    })

    audit = build_paper_profile_gate_audit(run_root, config_path)

    assert audit["paper_profile_gate_decision"] == "FAIL"
    assert "validation_external_baseline_self_containment_ready" in audit["missing_validation_requirements"]
    assert "external_baseline_self_containment_required_rows_present" in audit["external_baseline_self_containment_gate_missing_requirements"]
    assert "videoseal" in audit["missing_self_contained_modern_external_baseline_names"]


@pytest.mark.quick
def test_paper_profile_gate_passes_when_all_governed_inputs_exist(tmp_path: Path) -> None:
    """当 probe-paper 所需 records 和 decision artifacts 齐全时, gate 应允许进入 pilot_paper。"""
    run_root = tmp_path / "run"
    generation_records = []
    for prompt_index in range(8):
        for seed_index in range(3):
            generation_records.append({
                "generation_status": "success",
                "colab_runtime_profile": "probe_paper",
                "prompt_id": f"prompt_{prompt_index}",
                "seed_id": f"seed_{seed_index}",
            })
    write_jsonl(run_root / "records" / "generation_records.jsonl", generation_records)
    write_jsonl(run_root / "records" / "formal_quality_motion_semantic_records.jsonl", [
        {
            "prompt_id": record["prompt_id"],
            "seed_id": record["seed_id"],
            "formal_visual_quality_ready": True,
            "formal_motion_consistency_ready": True,
            "formal_semantic_consistency_ready": True,
            "formal_metric_result_used_for_claim": True,
            "motion_claim_role": "positive_motion",
        }
        for record in generation_records
    ])
    write_jsonl(run_root / "records" / "runtime_attack_records.jsonl", [
        _formal_runtime_attack_record(attack_name)
        for attack_name in REQUIRED_RUNTIME_ATTACK_NAMES
    ])
    write_jsonl(run_root / "records" / "runtime_detection_records.jsonl", [
        _formal_runtime_detection_record(attack_name)
        for attack_name in REQUIRED_RUNTIME_ATTACK_NAMES
    ])
    write_jsonl(run_root / "records" / "external_baseline_records.jsonl", run_external_baseline_status())
    write_jsonl(run_root / "records" / "external_baseline_score_records.jsonl", _formal_external_baseline_records())
    write_json(run_root / "artifacts" / "external_baseline_comparison_decision.json", {
        "external_baseline_comparison_decision": "PASS",
        "external_baseline_comparison_record_count": len(MODERN_EXTERNAL_BASELINE_NAMES),
        "external_baseline_comparison_ready_count": len(MODERN_EXTERNAL_BASELINE_NAMES),
        "external_baseline_measured_adapter_count": len(MODERN_EXTERNAL_BASELINE_NAMES),
        "modern_external_baseline_formal_measured_adapter_count": len(MODERN_EXTERNAL_BASELINE_NAMES),
        "modern_external_baseline_formal_measured_adapter_names": sorted(MODERN_EXTERNAL_BASELINE_NAMES),
        "external_baseline_claim_support_status": "external_baseline_formal_records_written",
    })
    write_json(
        run_root / "artifacts" / "external_baseline_self_containment_decision.json",
        _external_baseline_self_containment_pass_payload(),
    )
    write_json(run_root / "artifacts" / "data_split_and_leakage_guard_decision.json", {
        "data_split_and_leakage_guard_decision": "PASS",
        "claim_support_status": "data_split_and_leakage_guard_passed",
    })
    write_jsonl(run_root / "records" / "motion_consistency_exclusion_records.jsonl", [
        {"prompt_id": "prompt_0", "included_in_motion_claim": True, "excluded_from_motion_claim": False},
    ])
    write_json(run_root / "artifacts" / "motion_consistency_exclusion_decision.json", {
        "motion_consistency_exclusion_decision": "PASS",
        "motion_consistency_excluded_count": 0,
        "claim_support_status": "motion_consistency_exclusion_audit_record",
    })
    write_jsonl(run_root / "records" / "sstw_measured_formal_records.jsonl", _formal_sstw_measured_records())
    write_json(run_root / "artifacts" / "sstw_measured_formal_decision.json", {
        "sstw_measured_formal_decision": "PASS",
        "sstw_measured_formal_record_count": len(REQUIRED_RUNTIME_ATTACK_NAMES),
        "claim_support_status": "sstw_measured_formal_paper_profile_claim_candidate",
    })
    write_jsonl(run_root / "records" / "fair_detection_calibration_records.jsonl", [
        {
            "method_id": "sstw_key_conditioned_flow_trajectory",
            "fair_comparison_status": "ready",
            "metric_status": "measured_formal",
            "target_fpr": 0.1,
            "clean_negative_score_count": 500,
            "positive_anchor_count": len(REQUIRED_ANCHOR_KEYS),
            "positive_anchor_keys": list(REQUIRED_ANCHOR_KEYS),
            "positive_attack_names": list(REQUIRED_RUNTIME_ATTACK_NAMES),
            "positive_anchor_missing_count": 0,
            "positive_formal_evidence_missing_count": 0,
            "negative_formal_evidence_missing_count": 0,
        },
        *[
            {
                "method_id": baseline_id,
                "fair_comparison_status": "ready",
                "metric_status": "measured_formal",
                "target_fpr": 0.1,
                "clean_negative_score_count": 500,
                "positive_anchor_count": len(REQUIRED_ANCHOR_KEYS),
                "positive_anchor_keys": list(REQUIRED_ANCHOR_KEYS),
                "positive_attack_names": list(REQUIRED_RUNTIME_ATTACK_NAMES),
                "positive_anchor_missing_count": 0,
                "positive_formal_evidence_missing_count": 0,
                "negative_formal_evidence_missing_count": 0,
            }
            for baseline_id in sorted(MODERN_EXTERNAL_BASELINE_NAMES)
        ],
    ])
    write_json(run_root / "artifacts" / "fair_detection_calibration_decision.json", {
        "fair_detection_calibration_decision": "PASS",
        "fair_detection_calibration_ready_count": len(MODERN_EXTERNAL_BASELINE_NAMES) + 1,
        "target_fpr": 0.1,
        "claim_support_status": "fair_detection_calibration_paper_profile_ready",
    })
    write_jsonl(run_root / "records" / "formal_method_baseline_comparison_records.jsonl", [
        {
            "method_id": "sstw_key_conditioned_flow_trajectory",
            "method_role": "proposed_method",
            "metric_status": "measured_formal",
            "target_fpr": 0.1,
            "comparison_anchor_count": len(REQUIRED_ANCHOR_KEYS),
            "comparison_anchor_keys": list(REQUIRED_ANCHOR_KEYS),
            "comparison_attack_names": list(REQUIRED_RUNTIME_ATTACK_NAMES),
            "reference_anchor_count": len(REQUIRED_ANCHOR_KEYS),
            "missing_reference_anchor_count": 0,
            "extra_anchor_count": 0,
            "comparison_anchor_alignment_status": "reference_method_anchor_set_ready",
        },
        *[
            {
                "method_id": baseline_id,
                "method_role": "modern_external_baseline",
                "metric_status": "measured_formal",
                "target_fpr": 0.1,
                "comparison_anchor_count": len(REQUIRED_ANCHOR_KEYS),
                "comparison_anchor_keys": list(REQUIRED_ANCHOR_KEYS),
                "comparison_attack_names": list(REQUIRED_RUNTIME_ATTACK_NAMES),
                "reference_anchor_count": len(REQUIRED_ANCHOR_KEYS),
                "missing_reference_anchor_count": 0,
                "extra_anchor_count": 0,
                "comparison_anchor_alignment_status": "aligned_with_sstw_reference_anchors",
            }
            for baseline_id in sorted(MODERN_EXTERNAL_BASELINE_NAMES)
        ],
    ])
    write_json(run_root / "artifacts" / "formal_method_baseline_comparison_decision.json", {
        "formal_method_baseline_comparison_decision": "PASS",
        "formal_comparison_ready_method_count": len(MODERN_EXTERNAL_BASELINE_NAMES) + 1,
        "target_fpr": 0.1,
        "claim_support_status": "formal_method_baseline_comparison_paper_profile_claim_candidate",
    })
    write_jsonl(run_root / "records" / "formal_baseline_difference_interval_records.jsonl", [
        {
            "baseline_method_id": baseline_id,
            "difference_interval_status": "ready",
            "metric_status": "measured_formal",
            "target_fpr": 0.1,
            "paired_comparison_unit_count": len(REQUIRED_ANCHOR_KEYS),
            "paired_comparison_anchor_keys": list(REQUIRED_ANCHOR_KEYS),
            "paired_attack_names": list(REQUIRED_RUNTIME_ATTACK_NAMES),
            "unpaired_reference_anchor_count": 0,
            "unpaired_baseline_anchor_count": 0,
            "comparison_anchor_alignment_status": "aligned_with_sstw_reference_anchors",
            "tpr_at_target_fpr_difference": 0.18,
            "difference_ci_lower": 0.02,
            "difference_ci_upper": 0.34,
        }
        for baseline_id in sorted(MODERN_EXTERNAL_BASELINE_NAMES)
    ])
    write_json(run_root / "artifacts" / "formal_baseline_difference_interval_decision.json", {
        "formal_baseline_difference_interval_decision": "PASS",
        "difference_interval_ready_count": len(MODERN_EXTERNAL_BASELINE_NAMES),
        "target_fpr": 0.1,
        "claim_support_status": "formal_baseline_difference_interval_paper_profile_claim_candidate",
    })
    write_jsonl(
        run_root / "records" / "formal_internal_ablation_summary_records.jsonl",
        _formal_internal_ablation_summary_records(),
    )
    write_json(run_root / "artifacts" / "formal_internal_ablation_summary_decision.json", {
        "formal_internal_ablation_summary_decision": "PASS",
        "formal_internal_ablation_variant_count": 8,
        "claim_support_status": "formal_internal_ablation_summary_ready_for_target_fpr_0_1_claim_context",
    })
    write_jsonl(
        run_root / "records" / "formal_internal_ablation_variant_records.jsonl",
        _formal_internal_ablation_summary_records(),
    )
    write_jsonl(
        run_root / "records" / "validation_internal_ablation_records.jsonl",
        _formal_internal_ablation_summary_records(),
    )
    write_jsonl(run_root / "records" / "adaptive_attack_records.jsonl", [
        _formal_adaptive_attack_record(protocol_name)
        for protocol_name in REQUIRED_NON_RUNTIME_ATTACK_PROTOCOLS
    ])
    write_json(run_root / "artifacts" / "motion_threshold_calibration_decision.json", {
        "motion_threshold_calibration_decision": "PASS",
        "motion_threshold_calibration_ready": True,
        "motion_threshold_id": "motion_delta_calibrated_v1",
        "motion_threshold_source_split": "calibration",
    })
    write_json(run_root / "artifacts" / "runtime_attack_decision.json", {
        "runtime_attack_decision": "PASS",
        "runtime_attack_ready_count": len(REQUIRED_RUNTIME_ATTACK_NAMES),
        "runtime_attack_count": len(REQUIRED_RUNTIME_ATTACK_NAMES),
    })
    write_json(run_root / "artifacts" / "runtime_detection_decision.json", {
        "runtime_detection_decision": "PASS",
        "runtime_detection_ready_count": len(REQUIRED_RUNTIME_ATTACK_NAMES),
    })
    write_json(run_root / "artifacts" / "validation_internal_ablation_decision.json", {
        "validation_internal_ablation_decision": "PASS",
        "claim_support_status": "formal_internal_ablation_variant_matrix_ready",
        "validation_internal_ablation_evidence_level": "formal_component_removal_video_detector",
    })
    write_json(run_root / "artifacts" / "adaptive_attack_decision.json", {
        "adaptive_attack_decision": "PASS",
        "claim_support_status": "validation_adaptive_attack_ready",
    })
    write_json(run_root / "artifacts" / "replay_and_sketch_gate_decision.json", {
        "replay_and_sketch_gate_decision": "PASS",
        "claim3_full_support_allowed": True,
    })
    write_json(run_root / "artifacts" / "complete_paper_mechanism_claim_decision.json", {
        "complete_paper_mechanism_claim_decision": "PASS",
    })
    write_json(run_root / "artifacts" / "statistical_confidence_interval_decision.json", {
        "statistical_confidence_interval_decision": "PASS",
        "claim_support_status": "validation_ci_ready",
    })
    write_jsonl(run_root / "records" / "low_fpr_formal_statistics_records.jsonl", [
        {"blocked_result_profile": "pilot_paper", "formal_low_fpr_claim_allowed": False},
        {"blocked_result_profile": "full_paper", "formal_low_fpr_claim_allowed": False},
    ])
    write_json(run_root / "artifacts" / "low_fpr_formal_statistics_decision.json", {
        "low_fpr_formal_statistics_decision": "PASS",
        "low_fpr_formal_statistics_record_count": 2,
        "claim_support_status": "low_fpr_formal_statistics_blocking_record",
    })
    write_json(run_root / "artifacts" / "paper_result_artifact_skeleton_decision.json", {
        "paper_result_artifact_skeleton_decision": "PASS",
        "target_fpr": 0.1,
        "claim_support_status": "paper_result_artifact_skeleton_ready",
    })
    write_json(run_root / "artifacts" / "validation_artifact_rebuild_dry_run_decision.json", {
        "validation_artifact_rebuild_dry_run_decision": "PASS",
        "claim_support_status": "validation_artifact_rebuild_ready",
    })

    audit = write_paper_profile_gate_audit(run_root)
    protocol = json.loads(Path("configs/protocol/probe_paper_generative_probe.json").read_text(encoding="utf-8"))

    assert audit["paper_profile_gate_decision"] == "PASS"
    assert audit["claim_support_status"] == "probe_paper_target_fpr_0_1_paper_claim_supported"
    assert audit["paper_claim_id"] == "probe_claim"
    assert audit["paper_claim_support_status"] == "probe_claim_supported"
    assert audit["paper_result_formality_guard_decision"] == "PASS"
    assert audit["paper_result_formality_guard_violation_count"] == 0
    assert audit["paper_result_level"] == "probe_paper"
    assert audit["target_fpr"] == protocol["target_fpr"]
    assert audit["validation_generation_record_count"] == 24
    assert audit["validation_prompt_count"] == 8
    assert audit["validation_seed_per_prompt_min"] == 3
    assert audit["required_runtime_attack_names"] == sorted(REQUIRED_RUNTIME_ATTACK_NAMES)
    assert audit["runtime_attack_missing_required_names"] == []
    assert audit["runtime_detection_missing_required_names"] == []
    assert audit["motion_threshold_calibration_ready"] is True
    assert audit["formal_motion_claim_status"] == "ready"
    assert audit["full_paper_allowed"] is False
    assert audit["full_paper_next_gate"] == "pilot_paper_generative_probe_gate"
    assert audit["external_baseline_measured_adapter_count"] == len(MODERN_EXTERNAL_BASELINE_NAMES)
    assert audit["modern_external_baseline_formal_measured_adapter_count"] == len(MODERN_EXTERNAL_BASELINE_NAMES)
    assert audit["external_baseline_self_containment_decision"] == "PASS"
    assert audit["external_baseline_self_containment_ready_count"] == len(MODERN_EXTERNAL_BASELINE_NAMES)
    assert audit["external_baseline_self_containment_gate_missing_requirements"] == []
    assert audit["motion_consistency_exclusion_excluded_count"] == 0
    assert audit["motion_consistency_exclusion_status"] == "motion_consistency_exclusion_audit_record"
    assert audit["sstw_measured_formal_record_count"] == len(REQUIRED_RUNTIME_ATTACK_NAMES) + 1
    assert audit["sstw_measured_formal_status"] == "sstw_measured_formal_paper_profile_claim_candidate"
    assert audit["fair_detection_calibration_ready_count"] == len(MODERN_EXTERNAL_BASELINE_NAMES) + 1
    assert audit["fair_detection_calibration_status"] == "fair_detection_calibration_paper_profile_ready"
    assert audit["formal_method_baseline_comparison_ready_count"] == len(MODERN_EXTERNAL_BASELINE_NAMES) + 1
    assert audit["formal_method_baseline_comparison_status"] == "formal_method_baseline_comparison_paper_profile_claim_candidate"
    assert audit["formal_baseline_difference_interval_ready_count"] == len(MODERN_EXTERNAL_BASELINE_NAMES)
    assert audit["formal_baseline_difference_interval_status"] == "formal_baseline_difference_interval_paper_profile_claim_candidate"
    assert audit["paper_profile_sstw_advantage_claim_ready"] is True
    assert audit["paper_profile_sstw_advantage_ready_baseline_count"] == len(MODERN_EXTERNAL_BASELINE_NAMES)
    assert audit["adaptive_attack_missing_non_runtime_protocols"] == []
    assert audit["non_runtime_attack_protocol_count"] == len(REQUIRED_NON_RUNTIME_ATTACK_PROTOCOLS)
    assert audit["formal_internal_ablation_summary_variant_count"] == 8
    assert audit["formal_internal_ablation_summary_status"] == "formal_internal_ablation_summary_ready_for_target_fpr_0_1_claim_context"
    assert audit["low_fpr_formal_statistics_record_count"] == 2
    assert audit["low_fpr_formal_statistics_status"] == "low_fpr_formal_statistics_blocking_record"
    assert audit["data_split_and_leakage_guard_decision"] == "PASS"
    assert audit["missing_modern_external_baseline_formal_adapter_names"] == []
    assert (run_root / "records" / "paper_profile_gate_records.jsonl").exists()
    assert (run_root / "tables" / "paper_profile_gate_table.csv").exists()
    assert (run_root / "artifacts" / "paper_profile_gate_decision.json").exists()
    assert (run_root / "reports" / "paper_profile_gate_report.md").exists()


@pytest.mark.quick
def test_paper_profile_gate_rejects_stale_fair_comparison_decision_without_required_methods(tmp_path: Path) -> None:
    """probe-paper 不能只凭过期 PASS decision 放行缺 baseline 的公平比较中间态。"""

    run_root = tmp_path / "run"
    config_path = tmp_path / "probe_paper_config.json"
    config_path.write_text(
        json.dumps(
            {
                "target_fpr": 0.1,
                "paper_result_level": "probe_paper",
                "minimum_prompt_count": 0,
                "minimum_seed_per_prompt": 0,
                "minimum_attack_count": 0,
                "required_modern_external_baseline_adapter_names": ["videoseal"],
                "require_external_baseline_status_records": False,
                "require_external_baseline_comparison_records": False,
                "require_external_baseline_self_containment_decision": False,
                "require_sstw_measured_formal_records": False,
                "require_fair_detection_calibration": True,
                "require_formal_method_baseline_comparison": True,
                "require_formal_baseline_difference_interval": True,
                "require_motion_threshold_calibration_ready": False,
                "require_formal_motion_claim_ready": False,
                "require_motion_consistency_exclusion_report": False,
                "require_internal_ablation_records": False,
                "require_formal_internal_ablation_summary": False,
                "require_adaptive_attack_records": False,
                "require_replay_or_sketch_records_or_claim3_downgrade": False,
                "require_confidence_interval_report": False,
                "require_low_fpr_formal_statistics_blocking_record": False,
                "require_artifact_rebuild_dry_run": False,
                "require_data_split_and_leakage_guard": False,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    write_jsonl(run_root / "records" / "fair_detection_calibration_records.jsonl", [
        {
            "method_id": "sstw_key_conditioned_flow_trajectory",
            "fair_comparison_status": "ready",
            "metric_status": "measured_formal",
            "target_fpr": 0.1,
        },
    ])
    write_json(run_root / "artifacts" / "fair_detection_calibration_decision.json", {
        "fair_detection_calibration_decision": "PASS",
        "fair_detection_calibration_ready_count": 2,
        "target_fpr": 0.1,
        "claim_support_status": "fair_detection_calibration_paper_profile_ready",
    })
    write_jsonl(run_root / "records" / "formal_method_baseline_comparison_records.jsonl", [
        {"method_id": "sstw_key_conditioned_flow_trajectory", "metric_status": "measured_formal", "target_fpr": 0.1},
    ])
    write_json(run_root / "artifacts" / "formal_method_baseline_comparison_decision.json", {
        "formal_method_baseline_comparison_decision": "PASS",
        "formal_comparison_ready_method_count": 2,
        "target_fpr": 0.1,
        "claim_support_status": "formal_method_baseline_comparison_probe_paper_only",
    })
    write_jsonl(run_root / "records" / "formal_baseline_difference_interval_records.jsonl", [
        {"baseline_method_id": "other_baseline", "difference_interval_status": "ready", "metric_status": "measured_formal", "target_fpr": 0.1},
    ])
    write_json(run_root / "artifacts" / "formal_baseline_difference_interval_decision.json", {
        "formal_baseline_difference_interval_decision": "PASS",
        "difference_interval_ready_count": 1,
        "target_fpr": 0.1,
        "claim_support_status": "formal_baseline_difference_interval_probe_paper_only",
    })

    audit = build_paper_profile_gate_audit(run_root, config_path)

    assert audit["paper_profile_gate_decision"] == "FAIL"
    assert "validation_fair_detection_calibration_ready" in audit["missing_validation_requirements"]
    assert "validation_formal_method_baseline_comparison_ready" in audit["missing_validation_requirements"]
    assert "validation_formal_baseline_difference_interval_ready" in audit["missing_validation_requirements"]


@pytest.mark.quick
def test_paper_profile_gate_recomputes_external_baseline_records_before_pass(tmp_path: Path) -> None:
    """probe-paper 不能只凭旧 external baseline decision 放行缺 evidence 的 formal 记录。"""

    run_root = tmp_path / "run"
    config_path = tmp_path / "probe_paper_config.json"
    config_path.write_text(
        json.dumps(
            {
                "target_fpr": 0.1,
                "paper_result_level": "probe_paper",
                "minimum_prompt_count": 0,
                "minimum_seed_per_prompt": 0,
                "minimum_attack_count": 0,
                "minimum_external_baseline_measured_adapter_count": 1,
                "minimum_modern_external_baseline_formal_adapter_count": 1,
                "required_modern_external_baseline_adapter_names": ["videoseal"],
                "require_external_baseline_status_records": False,
                "require_external_baseline_comparison_records": True,
                "require_external_baseline_self_containment_decision": False,
                "require_sstw_measured_formal_records": False,
                "require_fair_detection_calibration": False,
                "require_formal_method_baseline_comparison": False,
                "require_formal_baseline_difference_interval": False,
                "require_motion_threshold_calibration_ready": False,
                "require_formal_motion_claim_ready": False,
                "require_motion_consistency_exclusion_report": False,
                "require_internal_ablation_records": False,
                "require_formal_internal_ablation_summary": False,
                "require_adaptive_attack_records": False,
                "require_replay_or_sketch_records_or_claim3_downgrade": False,
                "require_confidence_interval_report": False,
                "require_low_fpr_formal_statistics_blocking_record": False,
                "require_artifact_rebuild_dry_run": False,
                "require_data_split_and_leakage_guard": False,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    write_jsonl(run_root / "records" / "external_baseline_score_records.jsonl", [
        {
            "external_baseline_name": "videoseal",
            "external_baseline_layer": "modern_external_baseline",
            "metric_status": "measured_formal",
            "prompt_id": "prompt_0",
            "seed_id": "seed_0",
            "attack_name": "video_compression_runtime",
        },
    ])
    write_json(run_root / "artifacts" / "external_baseline_comparison_decision.json", {
        "external_baseline_comparison_decision": "PASS",
        "external_baseline_measured_adapter_count": 1,
        "modern_external_baseline_formal_measured_adapter_count": 1,
        "modern_external_baseline_formal_measured_adapter_names": ["videoseal"],
        "external_baseline_claim_support_status": "external_baseline_formal_records_written",
    })

    audit = build_paper_profile_gate_audit(run_root, config_path)

    assert audit["paper_profile_gate_decision"] == "FAIL"
    assert "validation_external_baseline_comparison_records_ready" in audit["missing_validation_requirements"]
    assert audit["modern_external_baseline_formal_measured_adapter_count"] == 0
    assert audit["missing_modern_external_baseline_formal_adapter_names"] == ["videoseal"]


@pytest.mark.quick
def test_paper_profile_gate_rejects_fair_comparison_without_anchor_alignment(tmp_path: Path) -> None:
    """probe-paper 不能只凭 measured_formal 字段放行缺少 anchor 对齐证据的公平比较。"""

    run_root = tmp_path / "run"
    config_path = tmp_path / "probe_paper_config.json"
    config_path.write_text(
        json.dumps(
            {
                "target_fpr": 0.1,
                "paper_result_level": "probe_paper",
                "minimum_prompt_count": 0,
                "minimum_seed_per_prompt": 0,
                "minimum_attack_count": 0,
                "required_modern_external_baseline_adapter_names": ["videoseal"],
                "require_external_baseline_status_records": False,
                "require_external_baseline_comparison_records": False,
                "require_external_baseline_self_containment_decision": False,
                "require_sstw_measured_formal_records": False,
                "require_fair_detection_calibration": True,
                "require_formal_method_baseline_comparison": True,
                "require_formal_baseline_difference_interval": True,
                "require_motion_threshold_calibration_ready": False,
                "require_formal_motion_claim_ready": False,
                "require_motion_consistency_exclusion_report": False,
                "require_internal_ablation_records": False,
                "require_formal_internal_ablation_summary": False,
                "require_adaptive_attack_records": False,
                "require_replay_or_sketch_records_or_claim3_downgrade": False,
                "require_confidence_interval_report": False,
                "require_low_fpr_formal_statistics_blocking_record": False,
                "require_artifact_rebuild_dry_run": False,
                "require_data_split_and_leakage_guard": False,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    write_jsonl(run_root / "records" / "fair_detection_calibration_records.jsonl", [
        {
            "method_id": "sstw_key_conditioned_flow_trajectory",
            "fair_comparison_status": "ready",
            "metric_status": "measured_formal",
            "target_fpr": 0.1,
        },
        {
            "method_id": "videoseal",
            "fair_comparison_status": "ready",
            "metric_status": "measured_formal",
            "target_fpr": 0.1,
        },
    ])
    write_json(run_root / "artifacts" / "fair_detection_calibration_decision.json", {
        "fair_detection_calibration_decision": "PASS",
        "fair_detection_calibration_ready_count": 2,
        "target_fpr": 0.1,
        "claim_support_status": "fair_detection_calibration_paper_profile_ready",
    })
    write_jsonl(run_root / "records" / "formal_method_baseline_comparison_records.jsonl", [
        {
            "method_id": "sstw_key_conditioned_flow_trajectory",
            "metric_status": "measured_formal",
            "target_fpr": 0.1,
        },
        {
            "method_id": "videoseal",
            "metric_status": "measured_formal",
            "target_fpr": 0.1,
        },
    ])
    write_json(run_root / "artifacts" / "formal_method_baseline_comparison_decision.json", {
        "formal_method_baseline_comparison_decision": "PASS",
        "formal_comparison_ready_method_count": 2,
        "target_fpr": 0.1,
        "claim_support_status": "formal_method_baseline_comparison_probe_paper_only",
    })
    write_jsonl(run_root / "records" / "formal_baseline_difference_interval_records.jsonl", [
        {
            "baseline_method_id": "videoseal",
            "difference_interval_status": "ready",
            "metric_status": "measured_formal",
            "target_fpr": 0.1,
        },
    ])
    write_json(run_root / "artifacts" / "formal_baseline_difference_interval_decision.json", {
        "formal_baseline_difference_interval_decision": "PASS",
        "difference_interval_ready_count": 1,
        "target_fpr": 0.1,
        "claim_support_status": "formal_baseline_difference_interval_probe_paper_only",
    })

    audit = build_paper_profile_gate_audit(run_root, config_path)

    assert audit["paper_profile_gate_decision"] == "FAIL"
    assert "validation_fair_detection_calibration_ready" in audit["missing_validation_requirements"]
    assert "validation_formal_method_baseline_comparison_ready" in audit["missing_validation_requirements"]
    assert "validation_formal_baseline_difference_interval_ready" in audit["missing_validation_requirements"]


@pytest.mark.quick
def test_paper_profile_gate_rejects_fair_comparison_with_negative_evidence_gap(tmp_path: Path) -> None:
    """probe-paper 不能放行 clean negative official evidence 仍有缺口的公平比较。"""

    run_root = tmp_path / "run"
    config_path = tmp_path / "probe_paper_config.json"
    config_path.write_text(
        json.dumps(
            {
                "target_fpr": 0.1,
                "paper_result_level": "probe_paper",
                "minimum_prompt_count": 0,
                "minimum_seed_per_prompt": 0,
                "minimum_attack_count": 0,
                "minimum_clean_negative_count": 2,
                "required_runtime_attack_names": list(REQUIRED_RUNTIME_ATTACK_NAMES),
                "required_modern_external_baseline_adapter_names": ["videoseal"],
                "require_external_baseline_status_records": False,
                "require_external_baseline_comparison_records": False,
                "require_external_baseline_self_containment_decision": False,
                "require_sstw_measured_formal_records": False,
                "require_fair_detection_calibration": True,
                "require_formal_method_baseline_comparison": False,
                "require_formal_baseline_difference_interval": False,
                "require_motion_threshold_calibration_ready": False,
                "require_formal_motion_claim_ready": False,
                "require_motion_consistency_exclusion_report": False,
                "require_internal_ablation_records": False,
                "require_formal_internal_ablation_summary": False,
                "require_adaptive_attack_records": False,
                "require_replay_or_sketch_records_or_claim3_downgrade": False,
                "require_confidence_interval_report": False,
                "require_low_fpr_formal_statistics_blocking_record": False,
                "require_artifact_rebuild_dry_run": False,
                "require_data_split_and_leakage_guard": False,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    write_jsonl(run_root / "records" / "fair_detection_calibration_records.jsonl", [
        {
            "method_id": "sstw_key_conditioned_flow_trajectory",
            "fair_comparison_status": "ready",
            "metric_status": "measured_formal",
            "target_fpr": 0.1,
                "clean_negative_score_count": 2,
                "positive_anchor_count": 3,
                "positive_anchor_keys": list(REQUIRED_ANCHOR_KEYS),
                "positive_attack_names": list(REQUIRED_RUNTIME_ATTACK_NAMES),
                "positive_anchor_missing_count": 0,
            "positive_formal_evidence_missing_count": 0,
            "negative_formal_evidence_missing_count": 0,
        },
        {
            "method_id": "videoseal",
            "fair_comparison_status": "ready",
            "metric_status": "measured_formal",
            "target_fpr": 0.1,
                "clean_negative_score_count": 2,
                "positive_anchor_count": 3,
                "positive_anchor_keys": list(REQUIRED_ANCHOR_KEYS),
                "positive_attack_names": list(REQUIRED_RUNTIME_ATTACK_NAMES),
                "positive_anchor_missing_count": 0,
            "positive_formal_evidence_missing_count": 0,
            "negative_formal_evidence_missing_count": 1,
        },
    ])
    write_json(run_root / "artifacts" / "fair_detection_calibration_decision.json", {
        "fair_detection_calibration_decision": "PASS",
        "fair_detection_calibration_ready_count": 2,
        "target_fpr": 0.1,
        "claim_support_status": "fair_detection_calibration_paper_profile_ready",
    })

    audit = build_paper_profile_gate_audit(run_root, config_path)

    assert audit["paper_profile_gate_decision"] == "FAIL"
    assert "validation_fair_detection_calibration_ready" in audit["missing_validation_requirements"]
    assert audit["fair_detection_calibration_ready_count"] == 1


@pytest.mark.quick
def test_paper_profile_gate_rejects_fair_comparison_with_wrong_target_fpr(tmp_path: Path) -> None:
    """公平比较产物的 target_fpr 必须与当前 probe-paper protocol config 一致。"""

    run_root = tmp_path / "run"
    config_path = tmp_path / "probe_paper_config.json"
    config_path.write_text(
        json.dumps(
            {
                "target_fpr": 0.1,
                "paper_result_level": "probe_paper",
                "minimum_prompt_count": 0,
                "minimum_seed_per_prompt": 0,
                "minimum_attack_count": 0,
                "required_modern_external_baseline_adapter_names": ["videoseal"],
                "require_external_baseline_status_records": False,
                "require_external_baseline_comparison_records": False,
                "require_external_baseline_self_containment_decision": False,
                "require_sstw_measured_formal_records": False,
                "require_fair_detection_calibration": True,
                "require_formal_method_baseline_comparison": False,
                "require_formal_baseline_difference_interval": False,
                "require_motion_threshold_calibration_ready": False,
                "require_formal_motion_claim_ready": False,
                "require_motion_consistency_exclusion_report": False,
                "require_internal_ablation_records": False,
                "require_formal_internal_ablation_summary": False,
                "require_adaptive_attack_records": False,
                "require_replay_or_sketch_records_or_claim3_downgrade": False,
                "require_confidence_interval_report": False,
                "require_low_fpr_formal_statistics_blocking_record": False,
                "require_artifact_rebuild_dry_run": False,
                "require_data_split_and_leakage_guard": False,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    write_jsonl(run_root / "records" / "fair_detection_calibration_records.jsonl", [
        {
            "method_id": "sstw_key_conditioned_flow_trajectory",
            "fair_comparison_status": "ready",
            "metric_status": "measured_formal",
            "target_fpr": 0.01,
            "positive_anchor_count": 3,
            "positive_anchor_missing_count": 0,
        },
        {
            "method_id": "videoseal",
            "fair_comparison_status": "ready",
            "metric_status": "measured_formal",
            "target_fpr": 0.01,
            "positive_anchor_count": 3,
            "positive_anchor_missing_count": 0,
        },
    ])
    write_json(run_root / "artifacts" / "fair_detection_calibration_decision.json", {
        "fair_detection_calibration_decision": "PASS",
        "fair_detection_calibration_ready_count": 2,
        "target_fpr": 0.01,
        "claim_support_status": "fair_detection_calibration_paper_profile_ready",
    })

    audit = build_paper_profile_gate_audit(run_root, config_path)

    assert audit["paper_profile_gate_decision"] == "FAIL"
    assert "validation_fair_detection_calibration_ready" in audit["missing_validation_requirements"]


@pytest.mark.quick
def test_paper_profile_gate_requires_reused_motion_threshold_and_formal_motion_records(tmp_path: Path) -> None:
    """probe-paper 正式门禁必须确认 motion threshold 复用和 formal motion claim 均已闭合。"""
    run_root = tmp_path / "run"
    generation_records = []
    for prompt_index in range(8):
        for seed_index in range(3):
            generation_records.append({
                "generation_status": "success",
                "colab_runtime_profile": "probe_paper",
                "prompt_id": f"prompt_{prompt_index}",
                "seed_id": f"seed_{seed_index}",
            })
    write_jsonl(run_root / "records" / "generation_records.jsonl", generation_records)
    audit = build_paper_profile_gate_audit(run_root)

    assert audit["paper_profile_gate_decision"] == "FAIL"
    assert "validation_motion_threshold_calibration_ready" in audit["missing_validation_requirements"]
    assert "validation_formal_motion_claim_ready" in audit["missing_validation_requirements"]


