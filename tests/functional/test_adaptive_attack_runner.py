from pathlib import Path

import pytest

from experiments.generative_video_model_probe.adaptive_attack_runner import (
    ADAPTIVE_ATTACK_SPECS,
    run_adaptive_attack_formal_protocol,
)
from evaluation.protocol.record_writer import read_jsonl, write_jsonl


@pytest.mark.quick
def test_adaptive_attack_runner_writes_formal_records(tmp_path: Path) -> None:
    """adaptive attack runner 只接受正式 adaptive attack 执行记录。"""
    run_root = tmp_path / "run"
    search_protocols = {
        "generative_recompression_or_regeneration_attack",
        "endpoint_preserving_path_perturbation_attack",
        "detector_probing_with_public_negatives",
        "watermark_removal_optimization_attack",
        "adversarial_detector_evasion_attack",
    }
    cross_protocols = {"watermark_spoofing_or_copy_attack", "collusion_multi_sample_attack"}

    def formal_record(spec: dict) -> dict:
        protocol = spec["non_runtime_attack_protocol"]
        record = {
            "adaptive_attack_status": "ready",
            "non_runtime_attack_protocol": protocol,
            "adaptive_attack_name": spec["adaptive_attack_name"],
            "adaptive_attack_family": spec["adaptive_attack_family"],
            "metric_status": "measured_formal",
            "adaptive_attack_evidence_level": "per_video_frozen_flow_detector_adaptive_execution",
            "adaptive_robustness_claim_allowed": True,
            "adaptive_attack_score": 0.72,
            "adaptive_attack_query_count": 2 if protocol in search_protocols else 1,
            "statistical_cluster_id": "video-cluster-a",
            "test_time_threshold_update_blocked": True,
            "adaptive_attack_execution_backend": "per_video_precomputed_key_independent_replay_control",
        }
        if protocol in search_protocols:
            record.update({
                "adaptive_attack_output_video_sha256": "candidate-digest",
                "adaptive_attack_candidate_records": [{"candidate_index": 0}, {"candidate_index": 1}],
                "adaptive_attack_execution_backend": "actual_video_candidate_generation_and_frozen_flow_queries",
            })
        elif protocol in cross_protocols:
            record["adaptive_attack_output_video_sha256"] = "blend-digest"
            record["adaptive_attack_execution_backend"] = "actual_cross_video_frame_blend_then_frozen_flow_query"
        return record

    write_jsonl(
        run_root / "records" / "formal_adaptive_attack_execution_records.jsonl",
        [formal_record(spec) for spec in ADAPTIVE_ATTACK_SPECS],
    )

    audit = run_adaptive_attack_formal_protocol(run_root)
    records = read_jsonl(run_root / "records" / "adaptive_attack_records.jsonl")

    assert audit["adaptive_attack_decision"] == "PASS"
    assert audit["adaptive_attack_record_count"] == len(ADAPTIVE_ATTACK_SPECS)
    assert audit["formal_adaptive_attack_record_count"] == len(ADAPTIVE_ATTACK_SPECS)
    assert audit["adaptive_robustness_claim_allowed"] is True
    assert all(record["claim_support_status"] == "formal_adaptive_attack_measured_ready" for record in records)
    assert all(record["metric_status"] == "measured_formal" for record in records)
    assert all(record["adaptive_attack_evidence_level"] == "per_video_frozen_flow_detector_adaptive_execution" for record in records)
    assert any(record["adaptive_attack_name"] == "generative_recompression_or_regeneration_attack" for record in records)
    assert any(record["adaptive_attack_name"] == "detector_probing_with_public_negatives" for record in records)
    assert (run_root / "tables" / "adaptive_attack_table.csv").exists()
    assert (run_root / "artifacts" / "adaptive_attack_decision.json").exists()
    assert (run_root / "reports" / "adaptive_attack_report.md").exists()


@pytest.mark.quick
def test_adaptive_attack_runner_blocks_when_runtime_detection_missing(tmp_path: Path) -> None:
    """缺少正式 adaptive attack 执行记录时, runner 只能报告阻断。"""
    run_root = tmp_path / "run"

    audit = run_adaptive_attack_formal_protocol(run_root)
    records = read_jsonl(run_root / "records" / "adaptive_attack_records.jsonl")

    assert audit["adaptive_attack_decision"] == "FAIL"
    assert audit["claim_support_status"] == "formal_adaptive_attack_execution_blocked"
    assert len(records) == len(ADAPTIVE_ATTACK_SPECS)
    assert all(record["metric_status"] == "missing" for record in records)
