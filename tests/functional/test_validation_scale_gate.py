import json
from pathlib import Path

import pytest

from experiments.generative_video_model_probe.external_baseline_runner import run_external_baseline_status
from experiments.generative_video_model_probe.validation_scale_gate import (
    build_validation_scale_gate_audit,
    write_validation_scale_gate_audit,
)
from main.protocol.record_writer import write_json, write_jsonl


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


def _formal_external_baseline_records() -> list[dict]:
    """构造 validation-scale 通过所需的完整 external baseline records fixture。"""
    records: list[dict] = []
    for name in EXTERNAL_BASELINE_NAMES:
        record = {
            "external_baseline_name": name,
            "external_baseline_layer": "modern_external_baseline" if name in MODERN_EXTERNAL_BASELINE_NAMES else "explicit_synchronization_control",
            "metric_status": "measured_formal" if name in MODERN_EXTERNAL_BASELINE_NAMES else "measured_proxy",
            "claim_support_status": "modern_external_baseline_formal_measured"
            if name in MODERN_EXTERNAL_BASELINE_NAMES
            else "external_baseline_proxy_comparison_not_claim_supporting",
        }
        if name in MODERN_EXTERNAL_BASELINE_NAMES:
            record.update({
                "prompt_id": "prompt_0",
                "seed_id": "seed_0",
                "attack_name": "video_compression_runtime",
                "external_baseline_clean_negative_score": 0.08,
                "external_baseline_clean_negative_video_path": f"official/{name}/clean_negative.mp4",
                "external_baseline_official_output_path": f"official/{name}/official_output.json",
                "external_baseline_official_command_manifest_path": f"official/{name}/official_command_manifest.json",
            })
        records.append(record)
    return records


@pytest.mark.quick
def test_validation_scale_gate_blocks_empty_run(tmp_path: Path) -> None:
    """空 run_root 必须被 validation-scale gate 阻断, 不能进入 pilot_paper。"""
    audit = build_validation_scale_gate_audit(tmp_path / "empty_run")

    assert audit["validation_scale_gate_decision"] == "FAIL"
    assert audit["claim_support_status"] == "validation_scale_blocked"
    assert audit["full_paper_allowed"] is False
    assert "small_scale_claim_pilot_gate_passed" not in audit["missing_validation_requirements"]
    assert "validation_generation_records_ready" in audit["missing_validation_requirements"]
    assert "validation_internal_ablation_records_ready" in audit["missing_validation_requirements"]
    assert "validation_sstw_measured_formal_records_ready" in audit["missing_validation_requirements"]
    assert "validation_fair_detection_calibration_ready" in audit["missing_validation_requirements"]
    assert "validation_formal_method_baseline_comparison_ready" in audit["missing_validation_requirements"]
    assert "validation_formal_baseline_difference_interval_ready" in audit["missing_validation_requirements"]
    assert "validation_scale_formal_internal_ablation_ready" in audit["missing_validation_requirements"]
    assert "validation_low_fpr_formal_statistics_blocking_record_ready" in audit["missing_validation_requirements"]
    assert "validation_motion_consistency_exclusion_report_ready" in audit["missing_validation_requirements"]


@pytest.mark.quick
def test_validation_scale_gate_rejects_pilot_profile_as_validation(tmp_path: Path) -> None:
    """pilot profile 不能冒充 validation-scale profile。"""
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
    audit = build_validation_scale_gate_audit(run_root)

    assert audit["validation_scale_gate_decision"] == "FAIL"
    assert audit["validation_generation_record_count"] == 0
    assert "validation_generation_records_ready" in audit["missing_validation_requirements"]


@pytest.mark.quick
def test_validation_scale_gate_passes_when_all_governed_inputs_exist(tmp_path: Path) -> None:
    """当 validation-scale 所需 records 和 decision artifacts 齐全时, gate 应允许进入 pilot_paper。"""
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
        {"attack_name": "video_compression_runtime", "attack_runtime_status": "ready"},
        {"attack_name": "temporal_crop_runtime", "attack_runtime_status": "ready"},
        {"attack_name": "frame_rate_resampling_runtime", "attack_runtime_status": "ready"},
    ])
    write_jsonl(run_root / "records" / "runtime_detection_records.jsonl", [
        {"attack_name": "video_compression_runtime", "runtime_detection_status": "ready"},
        {"attack_name": "temporal_crop_runtime", "runtime_detection_status": "ready"},
        {"attack_name": "frame_rate_resampling_runtime", "runtime_detection_status": "ready"},
    ])
    write_jsonl(run_root / "records" / "external_baseline_records.jsonl", run_external_baseline_status())
    write_jsonl(run_root / "records" / "external_baseline_score_records.jsonl", _formal_external_baseline_records())
    write_json(run_root / "artifacts" / "external_baseline_comparison_decision.json", {
        "external_baseline_comparison_decision": "PASS",
        "external_baseline_comparison_record_count": 7,
        "external_baseline_comparison_ready_count": 7,
        "external_baseline_measured_adapter_count": 7,
        "modern_external_baseline_formal_measured_adapter_count": 5,
        "modern_external_baseline_formal_measured_adapter_names": sorted(MODERN_EXTERNAL_BASELINE_NAMES),
        "external_baseline_claim_support_status": "external_baseline_formal_and_proxy_records_written",
    })
    write_json(run_root / "artifacts" / "external_baseline_self_containment_decision.json", {
        "external_baseline_self_containment_decision": "PASS",
        "claim_support_status": "external_baseline_self_contained_measured_formal_ready",
    })
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
        {"metric_status": "measured_formal", "sstw_score": 0.82, "claim_support_status": "sstw_measured_formal_validation_scale_only"},
    ])
    write_json(run_root / "artifacts" / "sstw_measured_formal_decision.json", {
        "sstw_measured_formal_decision": "PASS",
        "sstw_measured_formal_record_count": 1,
        "claim_support_status": "sstw_measured_formal_validation_scale_only",
    })
    write_jsonl(run_root / "records" / "fair_detection_calibration_records.jsonl", [
        {
            "method_id": "sstw_key_conditioned_flow_trajectory",
            "fair_comparison_status": "ready",
            "metric_status": "measured_formal",
            "target_fpr": 0.1,
            "positive_anchor_count": 3,
            "positive_anchor_missing_count": 0,
        },
        *[
            {
                "method_id": baseline_id,
                "fair_comparison_status": "ready",
                "metric_status": "measured_formal",
                "target_fpr": 0.1,
                "positive_anchor_count": 3,
                "positive_anchor_missing_count": 0,
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
            "comparison_anchor_count": 3,
            "reference_anchor_count": 3,
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
                "comparison_anchor_count": 3,
                "reference_anchor_count": 3,
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
        "claim_support_status": "formal_method_baseline_comparison_validation_scale_only",
    })
    write_jsonl(run_root / "records" / "formal_baseline_difference_interval_records.jsonl", [
        {
            "baseline_method_id": baseline_id,
            "difference_interval_status": "ready",
            "metric_status": "measured_formal",
            "target_fpr": 0.1,
            "paired_comparison_unit_count": 3,
            "unpaired_reference_anchor_count": 0,
            "unpaired_baseline_anchor_count": 0,
            "comparison_anchor_alignment_status": "aligned_with_sstw_reference_anchors",
        }
        for baseline_id in sorted(MODERN_EXTERNAL_BASELINE_NAMES)
    ])
    write_json(run_root / "artifacts" / "formal_baseline_difference_interval_decision.json", {
        "formal_baseline_difference_interval_decision": "PASS",
        "difference_interval_ready_count": 5,
        "target_fpr": 0.1,
        "claim_support_status": "formal_baseline_difference_interval_validation_scale_only",
    })
    write_jsonl(run_root / "records" / "validation_scale_formal_internal_ablation_records.jsonl", [
        {"method_variant": "sstw_full_method", "metric_status": "measured_formal"},
        *[
            {"method_variant": variant, "metric_status": "measured_proxy"}
            for variant in (
                "endpoint_only_control",
                "trajectory_only_score",
                "without_velocity_constraint",
                "without_endpoint_aware_control",
                "without_replay_uncertainty_weighting",
                "without_flow_state_admissibility",
                "generic_ssm_baseline",
            )
        ],
    ])
    write_json(run_root / "artifacts" / "validation_scale_formal_internal_ablation_decision.json", {
        "validation_scale_formal_internal_ablation_decision": "PASS",
        "formal_internal_ablation_variant_count": 8,
        "claim_support_status": "validation_scale_formal_internal_ablation_ready_not_effect_size_claim",
    })
    write_jsonl(run_root / "records" / "validation_internal_ablation_records.jsonl", [
        {"method_variant": "without_velocity_constraint", "ablation_status": "ready"},
    ])
    write_jsonl(run_root / "records" / "adaptive_attack_records.jsonl", [
        {"adaptive_attack_name": "time_grid_jitter", "adaptive_attack_status": "ready"},
    ])
    write_json(run_root / "artifacts" / "motion_threshold_calibration_decision.json", {
        "motion_threshold_calibration_decision": "PASS",
        "motion_threshold_calibration_ready": True,
        "motion_threshold_id": "motion_delta_calibrated_v1",
        "motion_threshold_source_split": "calibration",
    })
    write_json(run_root / "artifacts" / "runtime_attack_decision.json", {
        "runtime_attack_decision": "PASS",
        "runtime_attack_ready_count": 3,
        "runtime_attack_count": 3,
    })
    write_json(run_root / "artifacts" / "runtime_detection_decision.json", {
        "runtime_detection_decision": "PASS",
        "runtime_detection_ready_count": 3,
    })
    write_json(run_root / "artifacts" / "validation_internal_ablation_decision.json", {
        "validation_internal_ablation_decision": "PASS",
        "claim_support_status": "validation_internal_ablation_ready",
    })
    write_json(run_root / "artifacts" / "adaptive_attack_decision.json", {
        "adaptive_attack_decision": "PASS",
        "claim_support_status": "validation_adaptive_attack_ready",
    })
    write_json(run_root / "artifacts" / "replay_and_sketch_gate_decision.json", {
        "replay_and_sketch_gate_decision": "PASS",
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

    audit = write_validation_scale_gate_audit(run_root)
    protocol = json.loads(Path("configs/protocol/validation_scale_generative_probe.json").read_text(encoding="utf-8"))

    assert audit["validation_scale_gate_decision"] == "PASS"
    assert audit["claim_support_status"] == "validation_scale_ready_for_pilot_paper"
    assert audit["paper_result_level"] == "validation_scale"
    assert audit["target_fpr"] == protocol["target_fpr"]
    assert audit["validation_generation_record_count"] == 24
    assert audit["validation_prompt_count"] == 8
    assert audit["validation_seed_per_prompt_min"] == 3
    assert audit["motion_threshold_calibration_ready"] is True
    assert audit["formal_motion_claim_status"] == "ready"
    assert audit["full_paper_allowed"] is False
    assert audit["full_paper_next_gate"] == "pilot_paper_generative_probe_gate"
    assert audit["external_baseline_measured_adapter_count"] == 7
    assert audit["modern_external_baseline_formal_measured_adapter_count"] == 5
    assert audit["external_baseline_self_containment_decision"] == "PASS"
    assert audit["motion_consistency_exclusion_excluded_count"] == 0
    assert audit["motion_consistency_exclusion_status"] == "motion_consistency_exclusion_audit_record"
    assert audit["sstw_measured_formal_record_count"] == 1
    assert audit["sstw_measured_formal_status"] == "sstw_measured_formal_validation_scale_only"
    assert audit["fair_detection_calibration_ready_count"] == 6
    assert audit["fair_detection_calibration_status"] == "fair_detection_calibration_validation_scale_ready"
    assert audit["formal_method_baseline_comparison_ready_count"] == 6
    assert audit["formal_method_baseline_comparison_status"] == "formal_method_baseline_comparison_validation_scale_only"
    assert audit["formal_baseline_difference_interval_ready_count"] == 5
    assert audit["formal_baseline_difference_interval_status"] == "formal_baseline_difference_interval_validation_scale_only"
    assert audit["validation_scale_formal_internal_ablation_variant_count"] == 8
    assert audit["validation_scale_formal_internal_ablation_status"] == "validation_scale_formal_internal_ablation_ready_not_effect_size_claim"
    assert audit["low_fpr_formal_statistics_record_count"] == 2
    assert audit["low_fpr_formal_statistics_status"] == "low_fpr_formal_statistics_blocking_record"
    assert audit["data_split_and_leakage_guard_decision"] == "PASS"
    assert audit["missing_modern_external_baseline_formal_adapter_names"] == []
    assert (run_root / "records" / "validation_scale_gate_records.jsonl").exists()
    assert (run_root / "tables" / "validation_scale_gate_table.csv").exists()
    assert (run_root / "artifacts" / "validation_scale_gate_decision.json").exists()
    assert (run_root / "reports" / "validation_scale_gate_report.md").exists()


@pytest.mark.quick
def test_validation_scale_gate_rejects_stale_fair_comparison_decision_without_required_methods(tmp_path: Path) -> None:
    """validation-scale 不能只凭过期 PASS decision 放行缺 baseline 的公平比较中间态。"""

    run_root = tmp_path / "run"
    config_path = tmp_path / "validation_scale_config.json"
    config_path.write_text(
        json.dumps(
            {
                "target_fpr": 0.1,
                "paper_result_level": "validation_scale",
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
                "require_validation_scale_formal_internal_ablation": False,
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
        "claim_support_status": "fair_detection_calibration_validation_scale_ready",
    })
    write_jsonl(run_root / "records" / "formal_method_baseline_comparison_records.jsonl", [
        {"method_id": "sstw_key_conditioned_flow_trajectory", "metric_status": "measured_formal", "target_fpr": 0.1},
    ])
    write_json(run_root / "artifacts" / "formal_method_baseline_comparison_decision.json", {
        "formal_method_baseline_comparison_decision": "PASS",
        "formal_comparison_ready_method_count": 2,
        "target_fpr": 0.1,
        "claim_support_status": "formal_method_baseline_comparison_validation_scale_only",
    })
    write_jsonl(run_root / "records" / "formal_baseline_difference_interval_records.jsonl", [
        {"baseline_method_id": "other_baseline", "difference_interval_status": "ready", "metric_status": "measured_formal", "target_fpr": 0.1},
    ])
    write_json(run_root / "artifacts" / "formal_baseline_difference_interval_decision.json", {
        "formal_baseline_difference_interval_decision": "PASS",
        "difference_interval_ready_count": 1,
        "target_fpr": 0.1,
        "claim_support_status": "formal_baseline_difference_interval_validation_scale_only",
    })

    audit = build_validation_scale_gate_audit(run_root, config_path)

    assert audit["validation_scale_gate_decision"] == "FAIL"
    assert "validation_fair_detection_calibration_ready" in audit["missing_validation_requirements"]
    assert "validation_formal_method_baseline_comparison_ready" in audit["missing_validation_requirements"]
    assert "validation_formal_baseline_difference_interval_ready" in audit["missing_validation_requirements"]


@pytest.mark.quick
def test_validation_scale_gate_recomputes_external_baseline_records_before_pass(tmp_path: Path) -> None:
    """validation-scale 不能只凭旧 external baseline decision 放行缺 evidence 的 formal 记录。"""

    run_root = tmp_path / "run"
    config_path = tmp_path / "validation_scale_config.json"
    config_path.write_text(
        json.dumps(
            {
                "target_fpr": 0.1,
                "paper_result_level": "validation_scale",
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
                "require_validation_scale_formal_internal_ablation": False,
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
        "external_baseline_claim_support_status": "external_baseline_formal_and_proxy_records_written",
    })

    audit = build_validation_scale_gate_audit(run_root, config_path)

    assert audit["validation_scale_gate_decision"] == "FAIL"
    assert "validation_external_baseline_comparison_records_ready" in audit["missing_validation_requirements"]
    assert audit["modern_external_baseline_formal_measured_adapter_count"] == 0
    assert audit["missing_modern_external_baseline_formal_adapter_names"] == ["videoseal"]


@pytest.mark.quick
def test_validation_scale_gate_rejects_fair_comparison_without_anchor_alignment(tmp_path: Path) -> None:
    """validation-scale 不能只凭 measured_formal 字段放行缺少 anchor 对齐证据的公平比较。"""

    run_root = tmp_path / "run"
    config_path = tmp_path / "validation_scale_config.json"
    config_path.write_text(
        json.dumps(
            {
                "target_fpr": 0.1,
                "paper_result_level": "validation_scale",
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
                "require_validation_scale_formal_internal_ablation": False,
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
        "claim_support_status": "fair_detection_calibration_validation_scale_ready",
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
        "claim_support_status": "formal_method_baseline_comparison_validation_scale_only",
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
        "claim_support_status": "formal_baseline_difference_interval_validation_scale_only",
    })

    audit = build_validation_scale_gate_audit(run_root, config_path)

    assert audit["validation_scale_gate_decision"] == "FAIL"
    assert "validation_fair_detection_calibration_ready" in audit["missing_validation_requirements"]
    assert "validation_formal_method_baseline_comparison_ready" in audit["missing_validation_requirements"]
    assert "validation_formal_baseline_difference_interval_ready" in audit["missing_validation_requirements"]


@pytest.mark.quick
def test_validation_scale_gate_rejects_fair_comparison_with_wrong_target_fpr(tmp_path: Path) -> None:
    """公平比较产物的 target_fpr 必须与当前 validation-scale protocol config 一致。"""

    run_root = tmp_path / "run"
    config_path = tmp_path / "validation_scale_config.json"
    config_path.write_text(
        json.dumps(
            {
                "target_fpr": 0.1,
                "paper_result_level": "validation_scale",
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
                "require_validation_scale_formal_internal_ablation": False,
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
        "claim_support_status": "fair_detection_calibration_validation_scale_ready",
    })

    audit = build_validation_scale_gate_audit(run_root, config_path)

    assert audit["validation_scale_gate_decision"] == "FAIL"
    assert "validation_fair_detection_calibration_ready" in audit["missing_validation_requirements"]


@pytest.mark.quick
def test_validation_scale_gate_requires_reused_motion_threshold_and_formal_motion_records(tmp_path: Path) -> None:
    """validation-scale 正式门禁必须确认 motion threshold 复用和 formal motion claim 均已闭合。"""
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
    audit = build_validation_scale_gate_audit(run_root)

    assert audit["validation_scale_gate_decision"] == "FAIL"
    assert "validation_motion_threshold_calibration_ready" in audit["missing_validation_requirements"]
    assert "validation_formal_motion_claim_ready" in audit["missing_validation_requirements"]
