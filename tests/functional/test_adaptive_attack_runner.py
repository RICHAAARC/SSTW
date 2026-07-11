from pathlib import Path

import pytest

from experiments.generative_video_model_probe.adaptive_attack_runner import (
    ADAPTIVE_ATTACK_SPECS,
    run_adaptive_attack_formal_protocol,
)
from evaluation.protocol.record_writer import read_jsonl, write_jsonl


@pytest.mark.quick
def test_adaptive_attack_runner_writes_formal_records(tmp_path: Path) -> None:
    """runner 只接受具备真实查询前缀和独立簇覆盖的正式执行记录。"""

    run_root = tmp_path / "run"
    search_protocols = {
        "generative_recompression_or_regeneration_attack",
        "endpoint_preserving_path_perturbation_attack",
        "detector_probing_with_public_negatives",
        "watermark_removal_optimization_attack",
        "adversarial_detector_evasion_attack",
    }
    cross_protocols = {
        "watermark_spoofing_or_copy_attack",
        "collusion_multi_sample_attack",
    }
    clusters = ("video-cluster-a", "video-cluster-b")

    def formal_record(spec: dict, cluster: str) -> dict:
        protocol = spec["non_runtime_attack_protocol"]
        record = {
            "adaptive_attack_status": "ready",
            "non_runtime_attack_protocol": protocol,
            "adaptive_attack_name": spec["adaptive_attack_name"],
            "adaptive_attack_family": spec["adaptive_attack_family"],
            "metric_status": "measured_formal",
            "adaptive_attack_evidence_level": (
                "formal_adaptive_attack_execution"
            ),
            "adaptive_attack_execution_granularity": (
                "per_video_frozen_flow_detector_adaptive_execution"
            ),
            "adaptive_robustness_claim_allowed": True,
            "adaptive_attack_score": 0.72,
            "adaptive_attack_query_count": 3 if protocol in search_protocols else 1,
            "adaptive_attack_query_budget": 3,
            "adaptive_attack_query_budget_checkpoints": [1, 3],
            "adaptive_attack_query_budget_checkpoint_protocol": (
                "nested_actual_query_prefix_best_admissible_candidate_v1"
            ),
            "adaptive_attack_source_cluster_selection_protocol": (
                "stable_sha256_rank_over_heldout_source_cluster_before_attack_scoring_v1"
            ),
            "minimum_adaptive_attack_source_video_cluster_count_per_protocol": 2,
            "statistical_cluster_id": cluster,
            "adaptive_attack_source_statistical_cluster_id": cluster,
            "statistical_independent_unit": "source_video_prompt_seed",
            "test_time_threshold_update_blocked": True,
            "adaptive_attack_execution_backend": (
                "per_video_precomputed_key_independent_replay_control"
            ),
        }
        if protocol in search_protocols:
            candidates = [
                {
                    "candidate_index": index,
                    "video_sha256": f"{cluster}-candidate-{index}",
                }
                for index in range(3)
            ]
            record.update({
                "adaptive_attack_output_video_sha256": f"{cluster}-candidate-2",
                "adaptive_attack_total_detector_query_count": 3,
                "adaptive_attack_query_accounting_protocol": (
                    "all_target_and_public_negative_frozen_detector_calls"
                ),
                "adaptive_attack_candidate_records": candidates,
                "adaptive_attack_query_budget_checkpoint_records": [
                    {
                        "adaptive_attack_query_budget_checkpoint": checkpoint,
                        "adaptive_attack_checkpoint_observed_query_count": checkpoint,
                        "adaptive_attack_checkpoint_has_admissible_candidate": True,
                        "adaptive_attack_checkpoint_output_video_sha256": (
                            f"{cluster}-candidate-{checkpoint - 1}"
                        ),
                    }
                    for checkpoint in (1, 3)
                ],
                "adaptive_attack_execution_backend": (
                    "actual_video_candidate_generation_and_frozen_flow_queries"
                ),
            })
        elif protocol in cross_protocols:
            record["adaptive_attack_output_video_sha256"] = "blend-digest"
            record["adaptive_attack_execution_backend"] = (
                "actual_cross_video_frame_blend_then_frozen_flow_query"
            )
        return record

    formal_records = []
    for spec in ADAPTIVE_ATTACK_SPECS:
        protocol = spec["non_runtime_attack_protocol"]
        if protocol == "collusion_multi_sample_attack":
            record = formal_record(spec, "collusion-pair-a-b")
            record.update({
                "adaptive_attack_source_statistical_cluster_id": clusters[0],
                "adaptive_attack_member_statistical_cluster_ids": list(clusters),
                "statistical_independent_unit": "disjoint_source_video_pair",
            })
            formal_records.append(record)
        else:
            formal_records.extend(formal_record(spec, cluster) for cluster in clusters)

    write_jsonl(
        run_root / "records" / "formal_adaptive_attack_execution_records.jsonl",
        formal_records,
    )
    decision_path = (
        run_root / "artifacts" / "formal_adaptive_attack_execution_decision.json"
    )
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    decision_path.write_text(
        __import__("json").dumps({
            "formal_adaptive_attack_execution_decision": "PASS",
            "adaptive_attack_source_cluster_coverage_decision": "PASS",
            "adaptive_attack_independent_unit_uniqueness_decision": "PASS",
            "adaptive_attack_query_budget_checkpoint_coverage_decision": "PASS",
            "minimum_adaptive_attack_source_video_cluster_count_per_protocol": 2,
            "minimum_adaptive_spoof_source_video_cluster_count": 2,
            "adaptive_watermark_retention_decision": "PASS",
            "adaptive_spoof_rejection_decision": "PASS",
            "adaptive_replay_control_rejection_decision": "PASS",
            "adaptive_robustness_claim_allowed": True,
        }),
        encoding="utf-8",
    )

    audit = run_adaptive_attack_formal_protocol(run_root)
    records = read_jsonl(run_root / "records" / "adaptive_attack_records.jsonl")

    assert audit["adaptive_attack_decision"] == "PASS"
    assert audit["adaptive_attack_record_count"] == len(formal_records)
    assert audit["formal_adaptive_attack_record_count"] == len(formal_records)
    assert audit["adaptive_attack_independent_video_count"] == 2
    assert audit["adaptive_attack_pseudoreplication_decision"] == "PASS"
    assert audit["adaptive_attack_query_budget_checkpoint_decision"] == "PASS"
    assert audit["adaptive_robustness_claim_allowed"] is True
    assert all(
        record["claim_support_status"] == "formal_adaptive_attack_measured_ready"
        for record in records
    )
    assert all(record["metric_status"] == "measured_formal" for record in records)
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
