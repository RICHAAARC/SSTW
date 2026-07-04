from pathlib import Path

import pytest

from experiments.generative_video_model_probe.claim3_downgrade import (
    build_claim3_downgrade_audit,
    write_claim3_downgrade_outputs,
)
from experiments.generative_video_model_probe.external_baseline_runner import run_external_baseline_status
from experiments.generative_video_model_probe.validation_scale_gate import build_validation_scale_gate_audit
from main.attacks.video_runtime_attack_protocol import (
    FULL_PAPER_NON_RUNTIME_ATTACK_PROTOCOLS,
    FULL_PAPER_RUNTIME_ATTACKS,
)
from main.protocol.record_writer import read_jsonl, write_json, write_jsonl


EXTERNAL_BASELINE_NAMES = (
    "explicit_dtw_temporal_alignment",
    "explicit_frame_matching_temporal_registration",
    "videoshield",
    "sigmark",
    "videomark",
    "vidsig",
    "videoseal",
)
MODERN_EXTERNAL_BASELINE_NAMES = {
    "videoshield",
    "sigmark",
    "videomark",
    "vidsig",
    "videoseal",
}
REQUIRED_RUNTIME_ATTACK_NAMES = FULL_PAPER_RUNTIME_ATTACKS
REQUIRED_NON_RUNTIME_ATTACK_PROTOCOLS = FULL_PAPER_NON_RUNTIME_ATTACK_PROTOCOLS
REQUIRED_ANCHOR_KEYS = tuple(f"prompt_0::seed_0::{attack_name}" for attack_name in REQUIRED_RUNTIME_ATTACK_NAMES)


def _external_baseline_self_containment_pass_payload() -> dict:
    """构造 validation-scale gate 所需的完整 self-containment PASS fixture。"""

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
    """构造完整 baseline fixture, 使该测试只聚焦 Claim-3 降级路径。"""
    records: list[dict] = []
    for name in EXTERNAL_BASELINE_NAMES:
        record = {
            "external_baseline_name": name,
            "external_baseline_layer": "modern_external_baseline" if name in MODERN_EXTERNAL_BASELINE_NAMES else "explicit_synchronization_control",
            "metric_status": "measured_formal" if name in MODERN_EXTERNAL_BASELINE_NAMES else "measured_proxy",
        }
        if name in MODERN_EXTERNAL_BASELINE_NAMES:
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
def test_claim3_downgrade_gate_writes_explicit_downgrade_records(tmp_path: Path) -> None:
    """Claim-3 降级门禁必须写出显式降级记录, 不能伪装 replay/sketch 已通过。"""
    run_root = tmp_path / "run"

    audit = write_claim3_downgrade_outputs(run_root)
    records = read_jsonl(run_root / "records" / "claim3_downgrade_records.jsonl")

    assert audit["claim3_downgrade_decision"] == "PASS"
    assert audit["claim3_downgraded"] is True
    assert audit["claim3_full_support_allowed"] is False
    assert audit["replay_or_sketch_status"] == "claim3_explicitly_downgraded"
    assert records[0]["claim_support_status"] == "claim3_downgraded_validation_scale_only"
    assert records[0]["trajectory_source_level"] == "claim3_downgrade_governance_record"
    assert (run_root / "tables" / "claim3_downgrade_table.csv").exists()
    assert (run_root / "artifacts" / "claim3_downgrade_decision.json").exists()
    assert (run_root / "reports" / "claim3_downgrade_report.md").exists()


@pytest.mark.quick
def test_claim3_downgrade_gate_preserves_full_support_when_replay_gate_passed(tmp_path: Path) -> None:
    """若 replay/sketch gate 已通过, Claim-3 降级 runner 不应再把 Claim-3 降级。"""
    run_root = tmp_path / "run"
    write_json(run_root / "artifacts" / "replay_and_sketch_gate_decision.json", {
        "replay_and_sketch_gate_decision": "PASS",
        "claim3_full_support_allowed": True,
    })

    audit = build_claim3_downgrade_audit(run_root)

    assert audit["claim3_downgraded"] is False
    assert audit["claim3_full_support_allowed"] is True
    assert audit["replay_or_sketch_status"] == "replay_and_sketch_gate_passed"


@pytest.mark.quick
def test_claim3_downgrade_keeps_downgrade_for_validation_proxy_replay_gate(tmp_path: Path) -> None:
    """validation proxy 级 replay/sketch gate 通过后, Claim-3 仍不能升级为 full-paper 强支持。"""
    run_root = tmp_path / "run"
    write_json(run_root / "artifacts" / "replay_and_sketch_gate_decision.json", {
        "replay_and_sketch_gate_decision": "PASS",
        "claim3_full_support_allowed": False,
        "replay_or_sketch_status": "replay_and_sketch_gate_passed_validation_proxy",
        "replay_and_sketch_evidence_level": "validation_runtime_trace_proxy",
    })

    audit = build_claim3_downgrade_audit(run_root)

    assert audit["claim3_downgraded"] is True
    assert audit["claim3_full_support_allowed"] is False
    assert audit["claim3_allowed_scope"] == "owner_side_audit_or_exploratory_replay_analysis"
    assert audit["replay_or_sketch_status"] == "replay_and_sketch_gate_passed_validation_proxy"
    assert audit["claim3_downgrade_reason"] == "replay_and_sketch_gate_validation_proxy_only"


@pytest.mark.quick
def test_validation_scale_gate_accepts_claim3_downgrade_path(tmp_path: Path) -> None:
    """validation-scale gate 应接受 Claim-3 显式降级路径, 但不把它当作强 replay claim。"""
    run_root = tmp_path / "run"
    generation_records = []
    for prompt_index in range(8):
        for seed_index in range(3):
            generation_records.append({
                "generation_status": "success",
                "colab_runtime_profile": "validation_scale",
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
        {"attack_name": attack_name, "attack_runtime_status": "ready"}
        for attack_name in REQUIRED_RUNTIME_ATTACK_NAMES
    ])
    write_jsonl(run_root / "records" / "runtime_detection_records.jsonl", [
        {"attack_name": attack_name, "runtime_detection_status": "ready"}
        for attack_name in REQUIRED_RUNTIME_ATTACK_NAMES
    ])
    write_jsonl(run_root / "records" / "external_baseline_records.jsonl", run_external_baseline_status())
    write_jsonl(run_root / "records" / "external_baseline_score_records.jsonl", _formal_external_baseline_records())
    write_jsonl(run_root / "records" / "validation_internal_ablation_records.jsonl", [
        {"method_variant": "without_velocity_constraint", "ablation_status": "ready"},
    ])
    write_jsonl(run_root / "records" / "adaptive_attack_records.jsonl", [
        {
            "adaptive_attack_name": protocol_name,
            "non_runtime_attack_protocol": protocol_name,
            "adaptive_attack_status": "ready",
        }
        for protocol_name in REQUIRED_NON_RUNTIME_ATTACK_PROTOCOLS
    ])
    write_json(run_root / "artifacts" / "small_scale_claim_pilot_gate_decision.json", {"pilot_gate_decision": "PASS"})
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
    write_json(run_root / "artifacts" / "external_baseline_comparison_decision.json", {
        "external_baseline_comparison_decision": "PASS",
        "external_baseline_measured_adapter_count": 7,
        "modern_external_baseline_formal_measured_adapter_count": 5,
        "modern_external_baseline_formal_measured_adapter_names": sorted(MODERN_EXTERNAL_BASELINE_NAMES),
        "external_baseline_claim_support_status": "external_baseline_formal_and_proxy_records_written",
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
    write_jsonl(run_root / "records" / "sstw_measured_formal_records.jsonl", [
        {
            "metric_status": "measured_formal",
            "sstw_score": 0.82,
            "prompt_id": "prompt_0",
            "seed_id": "seed_0",
            "attack_name": attack_name,
            "claim_support_status": "sstw_measured_formal_paper_profile_claim_candidate",
        }
        for attack_name in REQUIRED_RUNTIME_ATTACK_NAMES
    ])
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
            "tpr_at_target_fpr": 1.0,
            "clean_negative_score_count": 10,
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
                "tpr_at_target_fpr": 1.0,
                "clean_negative_score_count": 10,
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
        "fair_detection_calibration_ready_count": 6,
        "target_fpr": 0.1,
        "claim_support_status": "fair_detection_calibration_validation_scale_ready",
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
        "formal_comparison_ready_method_count": 6,
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
            "claim_support_status": "formal_baseline_difference_interval_paper_profile_claim_candidate",
        }
        for baseline_id in sorted(MODERN_EXTERNAL_BASELINE_NAMES)
    ])
    write_json(run_root / "artifacts" / "formal_baseline_difference_interval_decision.json", {
        "formal_baseline_difference_interval_decision": "PASS",
        "difference_interval_ready_count": 5,
        "target_fpr": 0.1,
        "claim_support_status": "formal_baseline_difference_interval_paper_profile_claim_candidate",
    })
    write_jsonl(run_root / "records" / "validation_scale_formal_internal_ablation_records.jsonl", [
        {"method_variant": "sstw_full_method", "metric_status": "measured_formal"},
        {"method_variant": "without_velocity_constraint", "metric_status": "measured_proxy"},
    ])
    write_json(run_root / "artifacts" / "validation_scale_formal_internal_ablation_decision.json", {
        "validation_scale_formal_internal_ablation_decision": "PASS",
        "formal_internal_ablation_variant_count": 8,
        "claim_support_status": "validation_scale_formal_internal_ablation_ready_for_target_fpr_0_1_claim_context",
    })
    write_json(run_root / "artifacts" / "validation_internal_ablation_decision.json", {
        "validation_internal_ablation_decision": "PASS",
        "claim_support_status": "validation_internal_ablation_ready",
    })
    write_json(run_root / "artifacts" / "adaptive_attack_decision.json", {
        "adaptive_attack_decision": "PASS",
        "claim_support_status": "validation_adaptive_attack_ready",
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
    write_json(run_root / "artifacts" / "validation_artifact_rebuild_dry_run_decision.json", {
        "validation_artifact_rebuild_dry_run_decision": "PASS",
        "claim_support_status": "validation_artifact_rebuild_ready",
    })
    write_claim3_downgrade_outputs(run_root)

    audit = build_validation_scale_gate_audit(run_root)

    assert audit["validation_scale_gate_decision"] == "PASS"
    assert audit["replay_or_sketch_status"] == "claim3_explicitly_downgraded"
    assert "validation_replay_or_sketch_records_ready" not in audit["missing_validation_requirements"]
