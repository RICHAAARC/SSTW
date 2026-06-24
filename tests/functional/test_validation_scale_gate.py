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
    "spdmark",
    "videomark",
    "vidsig",
    "videoseal",
)
MODERN_EXTERNAL_BASELINE_NAMES = {
    "videoshield",
    "sigmark",
    "spdmark",
    "videomark",
    "vidsig",
    "videoseal",
}


def _formal_external_baseline_records() -> list[dict]:
    """构造 validation-scale 通过所需的完整 external baseline records fixture。"""
    return [
        {
            "external_baseline_name": name,
            "external_baseline_layer": "modern_external_baseline" if name in MODERN_EXTERNAL_BASELINE_NAMES else "explicit_synchronization_control",
            "metric_status": "measured_formal" if name in MODERN_EXTERNAL_BASELINE_NAMES else "measured_proxy",
            "claim_support_status": "modern_external_baseline_formal_measured"
            if name in MODERN_EXTERNAL_BASELINE_NAMES
            else "external_baseline_proxy_comparison_not_claim_supporting",
        }
        for name in EXTERNAL_BASELINE_NAMES
    ]


@pytest.mark.quick
def test_validation_scale_gate_blocks_empty_run(tmp_path: Path) -> None:
    """空 run_root 必须被 validation-scale gate 阻断, 不能进入 pilot_paper。"""
    audit = build_validation_scale_gate_audit(tmp_path / "empty_run")

    assert audit["validation_scale_gate_decision"] == "FAIL"
    assert audit["claim_support_status"] == "validation_scale_blocked"
    assert audit["full_paper_allowed"] is False
    assert "small_scale_claim_pilot_gate_passed" in audit["missing_validation_requirements"]
    assert "validation_generation_records_ready" in audit["missing_validation_requirements"]
    assert "validation_internal_ablation_records_ready" in audit["missing_validation_requirements"]


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
    write_json(run_root / "artifacts" / "small_scale_claim_pilot_gate_decision.json", {"pilot_gate_decision": "PASS"})

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
        "external_baseline_comparison_record_count": 8,
        "external_baseline_comparison_ready_count": 8,
        "external_baseline_measured_adapter_count": 8,
        "modern_external_baseline_formal_measured_adapter_count": 6,
        "modern_external_baseline_formal_measured_adapter_names": sorted(MODERN_EXTERNAL_BASELINE_NAMES),
        "external_baseline_claim_support_status": "external_baseline_formal_and_proxy_records_written",
    })
    write_jsonl(run_root / "records" / "validation_internal_ablation_records.jsonl", [
        {"method_variant": "without_velocity_constraint", "ablation_status": "ready"},
    ])
    write_jsonl(run_root / "records" / "adaptive_attack_records.jsonl", [
        {"adaptive_attack_name": "time_grid_jitter", "adaptive_attack_status": "ready"},
    ])
    write_json(run_root / "artifacts" / "small_scale_claim_pilot_gate_decision.json", {"pilot_gate_decision": "PASS"})
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
    write_json(run_root / "artifacts" / "validation_artifact_rebuild_dry_run_decision.json", {
        "validation_artifact_rebuild_dry_run_decision": "PASS",
        "claim_support_status": "validation_artifact_rebuild_ready",
    })

    audit = write_validation_scale_gate_audit(run_root)

    assert audit["validation_scale_gate_decision"] == "PASS"
    assert audit["claim_support_status"] == "validation_scale_ready_for_pilot_paper"
    assert audit["validation_generation_record_count"] == 24
    assert audit["validation_prompt_count"] == 8
    assert audit["validation_seed_per_prompt_min"] == 3
    assert audit["full_paper_allowed"] is False
    assert audit["full_paper_next_gate"] == "pilot_paper_generative_probe_gate"
    assert audit["external_baseline_measured_adapter_count"] == 8
    assert audit["modern_external_baseline_formal_measured_adapter_count"] == 6
    assert audit["missing_modern_external_baseline_formal_adapter_names"] == []
    assert (run_root / "records" / "validation_scale_gate_records.jsonl").exists()
    assert (run_root / "tables" / "validation_scale_gate_table.csv").exists()
    assert (run_root / "artifacts" / "validation_scale_gate_decision.json").exists()
    assert (run_root / "reports" / "validation_scale_gate_report.md").exists()
