from pathlib import Path

import pytest

from experiments.generative_video_model_probe.claim3_downgrade import (
    build_claim3_downgrade_audit,
    write_claim3_downgrade_outputs,
)
from experiments.generative_video_model_probe.external_baseline_runner import run_external_baseline_status
from experiments.generative_video_model_probe.validation_scale_gate import build_validation_scale_gate_audit
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


def _formal_external_baseline_records() -> list[dict]:
    """构造完整 baseline fixture, 使该测试只聚焦 Claim-3 降级路径。"""
    return [
        {
            "external_baseline_name": name,
            "external_baseline_layer": "modern_external_baseline" if name in MODERN_EXTERNAL_BASELINE_NAMES else "explicit_synchronization_control",
            "metric_status": "measured_formal" if name in MODERN_EXTERNAL_BASELINE_NAMES else "measured_proxy",
        }
        for name in EXTERNAL_BASELINE_NAMES
    ]


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
    write_jsonl(run_root / "records" / "validation_internal_ablation_records.jsonl", [
        {"method_variant": "without_velocity_constraint", "ablation_status": "ready"},
    ])
    write_jsonl(run_root / "records" / "adaptive_attack_records.jsonl", [
        {"adaptive_attack_name": "time_grid_jitter", "adaptive_attack_status": "ready"},
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
        "runtime_attack_ready_count": 3,
        "runtime_attack_count": 3,
    })
    write_json(run_root / "artifacts" / "runtime_detection_decision.json", {
        "runtime_detection_decision": "PASS",
        "runtime_detection_ready_count": 3,
    })
    write_json(run_root / "artifacts" / "external_baseline_comparison_decision.json", {
        "external_baseline_comparison_decision": "PASS",
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
    write_json(run_root / "artifacts" / "validation_artifact_rebuild_dry_run_decision.json", {
        "validation_artifact_rebuild_dry_run_decision": "PASS",
        "claim_support_status": "validation_artifact_rebuild_ready",
    })
    write_claim3_downgrade_outputs(run_root)

    audit = build_validation_scale_gate_audit(run_root)

    assert audit["validation_scale_gate_decision"] == "PASS"
    assert audit["replay_or_sketch_status"] == "claim3_explicitly_downgraded"
    assert "validation_replay_or_sketch_records_ready" not in audit["missing_validation_requirements"]
