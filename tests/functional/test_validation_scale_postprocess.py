from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.generative_video_model_probe.formal_method_baseline_comparison import run_formal_method_baseline_comparison
from experiments.generative_video_model_probe.formal_baseline_difference_interval import run_formal_baseline_difference_interval
from experiments.generative_video_model_probe.low_fpr_formal_statistics import run_low_fpr_formal_statistics
from experiments.generative_video_model_probe.motion_consistency_exclusion_report import run_motion_consistency_exclusion_report
from experiments.generative_video_model_probe.sstw_formal_result import run_sstw_measured_formal_result
from experiments.generative_video_model_probe.statistical_confidence_interval import run_statistical_confidence_interval_reporter
from experiments.generative_video_model_probe.validation_artifact_rebuild import run_validation_artifact_rebuild_dry_run
from experiments.generative_video_model_probe.validation_internal_ablation import run_validation_internal_ablation
from experiments.generative_video_model_probe.validation_scale_formal_internal_ablation import run_validation_scale_formal_internal_ablation
from main.protocol.record_writer import read_jsonl, write_json, write_jsonl


@pytest.mark.quick
def test_validation_internal_ablation_writes_proxy_records(tmp_path: Path) -> None:
    """validation 内部消融 runner 必须从 runtime detection records 写出 proxy 消融矩阵。"""
    run_root = tmp_path / "run"
    write_jsonl(run_root / "records" / "generation_records.jsonl", [
        {
            "generation_status": "success",
            "colab_runtime_profile": "validation_scale",
            "trajectory_trace_id": "trace_a",
            "prompt_id": "prompt_a",
            "seed_id": "seed_a",
        }
    ])
    write_jsonl(run_root / "records" / "runtime_detection_records.jsonl", [
        {
            "runtime_detection_status": "ready",
            "trajectory_trace_id": "trace_a",
            "generation_model_id": "model",
            "prompt_id": "prompt_a",
            "seed_id": "seed_a",
            "attack_name": "video_compression_runtime",
            "S_runtime_attack_detection": 0.8,
            "S_final_conservative": 0.78,
        }
    ])

    audit = run_validation_internal_ablation(run_root)
    records = read_jsonl(run_root / "records" / "validation_internal_ablation_records.jsonl")

    assert audit["validation_internal_ablation_decision"] == "PASS"
    assert audit["validation_internal_ablation_variant_count"] >= 8
    assert audit["internal_ablation_record_count"] == len(records)
    assert any(record["method_variant"] == "without_velocity_constraint" for record in records)
    assert all(record["claim_support_status"] == "validation_internal_ablation_proxy_only" for record in records)
    assert all(record["ablation_runtime_profile"] == "validation_scale" for record in records)
    assert (run_root / "tables" / "validation_internal_ablation_table.csv").exists()
    assert (run_root / "reports" / "validation_internal_ablation_report.md").exists()


@pytest.mark.quick
def test_pilot_paper_internal_ablation_writes_same_profile_records(tmp_path: Path) -> None:
    """pilot_paper 运行时内部消融必须覆盖 pilot_paper trace, 不能只复用 validation proxy。"""
    run_root = tmp_path / "run"
    write_jsonl(run_root / "records" / "generation_records.jsonl", [
        {
            "generation_status": "success",
            "colab_runtime_profile": "pilot_paper",
            "trajectory_trace_id": "trace_pilot_paper",
            "prompt_id": "prompt_pilot_paper",
            "seed_id": "seed_pilot_paper",
        }
    ])
    write_jsonl(run_root / "records" / "runtime_detection_records.jsonl", [
        {
            "runtime_detection_status": "ready",
            "trajectory_trace_id": "trace_pilot_paper",
            "generation_model_id": "model",
            "prompt_id": "prompt_pilot_paper",
            "seed_id": "seed_pilot_paper",
            "attack_name": "video_compression_runtime",
            "S_runtime_attack_detection": 0.82,
            "S_final_conservative": 0.8,
        }
    ])

    audit = run_validation_internal_ablation(run_root)
    records = read_jsonl(run_root / "records" / "validation_internal_ablation_records.jsonl")

    assert audit["validation_internal_ablation_decision"] == "PASS"
    assert audit["pilot_paper_internal_ablation_record_count"] == len(records)
    assert all(record["ablation_runtime_profile"] == "pilot_paper" for record in records)


@pytest.mark.quick
def test_statistical_confidence_interval_reporter_writes_wilson_interval(tmp_path: Path) -> None:
    """统计 CI reporter 必须基于 runtime detection records 写出 Wilson 区间。"""
    run_root = tmp_path / "run"
    write_jsonl(run_root / "records" / "runtime_detection_records.jsonl", [
        {"runtime_detection_status": "ready", "attacked_video_detectable": True},
        {"runtime_detection_status": "ready", "attacked_video_detectable": True},
        {"runtime_detection_status": "ready", "attacked_video_detectable": False},
    ])

    audit = run_statistical_confidence_interval_reporter(run_root)
    records = read_jsonl(run_root / "records" / "statistical_confidence_interval_records.jsonl")
    protocol = json.loads(Path("configs/protocol/validation_scale_generative_probe.json").read_text(encoding="utf-8"))

    assert audit["statistical_confidence_interval_decision"] == "PASS"
    assert audit["paper_result_level"] == "validation_scale"
    assert audit["target_fpr"] == protocol["target_fpr"]
    assert records[0]["target_fpr"] == protocol["target_fpr"]
    assert audit["ci_total_count"] == 3
    assert audit["ci_success_count"] == 2
    assert 0 <= audit["ci_wilson_lower"] <= audit["ci_wilson_upper"] <= 1
    assert records[0]["paper_low_fpr_ci_status"] == "not_available_until_full_paper_negative_split"
    assert (run_root / "tables" / "statistical_confidence_interval_table.csv").exists()
    assert (run_root / "reports" / "statistical_confidence_interval_report.md").exists()


@pytest.mark.quick
def test_sstw_measured_formal_result_writes_project_method_records(tmp_path: Path) -> None:
    """SSTW 本方法必须转写为与 external baseline 对齐的 measured_formal records。"""
    run_root = tmp_path / "run"
    write_jsonl(run_root / "records" / "runtime_detection_records.jsonl", [
        {
            "runtime_detection_status": "ready",
            "runtime_detection_evidence_level": "runtime_attacked_video_file",
            "generation_model_id": "wan21",
            "prompt_id": "prompt_a",
            "seed_id": "seed_a",
            "trajectory_trace_id": "trace_a",
            "attack_name": "video_compression_runtime",
            "source_video_path": "videos/source.mp4",
            "source_video_sha256": "source_digest",
            "attacked_video_path": "videos/attacked.mp4",
            "attacked_video_sha256": "attacked_digest",
            "S_runtime_attack_detection": 0.82,
            "S_final_conservative": 0.8,
            "attacked_video_detectable": True,
        },
        {
            "runtime_detection_status": "failed",
            "prompt_id": "prompt_b",
            "S_runtime_attack_detection": 0.1,
        },
    ])

    audit = run_sstw_measured_formal_result(run_root)
    records = read_jsonl(run_root / "records" / "sstw_measured_formal_records.jsonl")
    protocol = json.loads(Path("configs/protocol/validation_scale_generative_probe.json").read_text(encoding="utf-8"))

    assert audit["sstw_measured_formal_decision"] == "PASS"
    assert audit["sstw_measured_formal_record_count"] == 1
    assert audit["sstw_measured_formal_score_mean"] == 0.8
    assert audit["target_fpr"] == protocol["target_fpr"]
    assert records[0]["metric_status"] == "measured_formal"
    assert records[0]["method_id"] == "sstw_key_conditioned_flow_trajectory"
    assert records[0]["method_role"] == "proposed_method"
    assert records[0]["comparison_scope"] == "paper_protocol_formal_adapter"
    assert records[0]["claim_support_status"] == "sstw_measured_formal_validation_scale_only"
    assert records[0]["sstw_detection_score_field"] == "S_final_conservative"
    assert (run_root / "tables" / "sstw_measured_formal_table.csv").exists()
    assert (run_root / "artifacts" / "sstw_measured_formal_decision.json").exists()
    assert (run_root / "reports" / "sstw_measured_formal_report.md").exists()


@pytest.mark.quick
def test_formal_method_baseline_comparison_requires_sstw_and_five_baselines(tmp_path: Path) -> None:
    """同协议统计表必须同时包含 SSTW 和 5 个 measured_formal modern baseline。"""
    run_root = tmp_path / "run"
    write_jsonl(run_root / "records" / "sstw_measured_formal_records.jsonl", [
        {
            "method_id": "sstw_key_conditioned_flow_trajectory",
            "method_role": "proposed_method",
            "metric_status": "measured_formal",
            "prompt_id": "prompt_a",
            "attack_name": "video_compression_runtime",
            "sstw_score": 0.84,
            "sstw_detected": True,
        }
    ])
    baseline_records = []
    for baseline_id, score in {
        "videoshield": 0.62,
        "sigmark": 0.67,
        "videomark": 0.58,
        "vidsig": 0.55,
        "videoseal": 0.71,
    }.items():
        baseline_records.append({
            "external_baseline_name": baseline_id,
            "external_baseline_layer": "modern_external_baseline",
            "metric_status": "measured_formal",
            "prompt_id": "prompt_a",
            "attack_name": "video_compression_runtime",
            "external_baseline_score": score,
            "external_baseline_detected": score > 0.6,
        })
    write_jsonl(run_root / "records" / "external_baseline_score_records.jsonl", baseline_records)

    audit = run_formal_method_baseline_comparison(run_root)
    records = read_jsonl(run_root / "records" / "formal_method_baseline_comparison_records.jsonl")

    assert audit["formal_method_baseline_comparison_decision"] == "PASS"
    assert audit["formal_comparison_required_method_count"] == 6
    assert audit["formal_comparison_ready_method_count"] == 6
    assert audit["formal_comparison_modern_baseline_ready_count"] == 5
    assert audit["formal_comparison_missing_method_ids"] == []
    assert {record["method_id"] for record in records} == {
        "sstw_key_conditioned_flow_trajectory",
        "videoshield",
        "sigmark",
        "videomark",
        "vidsig",
        "videoseal",
    }
    assert all(record["metric_status"] == "measured_formal" for record in records)
    assert all(record["comparison_scope"] == "paper_protocol_formal_adapter" for record in records)
    assert (run_root / "tables" / "formal_method_baseline_comparison_table.csv").exists()
    assert (run_root / "artifacts" / "formal_method_baseline_comparison_decision.json").exists()
    assert (run_root / "reports" / "formal_method_baseline_comparison_report.md").exists()


@pytest.mark.quick
def test_formal_baseline_difference_interval_writes_sstw_vs_each_baseline_ci(tmp_path: Path) -> None:
    """差值 CI 报告必须覆盖 SSTW 相对 5 个 modern baseline 的 measured_formal 分数差。"""
    run_root = tmp_path / "run"
    write_jsonl(run_root / "records" / "sstw_measured_formal_records.jsonl", [
        {
            "metric_status": "measured_formal",
            "prompt_id": f"prompt_{index}",
            "seed_id": "seed_a",
            "attack_name": "video_compression_runtime",
            "sstw_score": 0.8 + index * 0.01,
        }
        for index in range(3)
    ])
    baseline_records = []
    for baseline_id in ("videoshield", "sigmark", "videomark", "vidsig", "videoseal"):
        for index in range(3):
            baseline_records.append({
                "external_baseline_name": baseline_id,
                "metric_status": "measured_formal",
                "prompt_id": f"prompt_{index}",
                "seed_id": "seed_a",
                "attack_name": "video_compression_runtime",
                "external_baseline_score": 0.6 + index * 0.01,
            })
    write_jsonl(run_root / "records" / "external_baseline_score_records.jsonl", baseline_records)

    audit = run_formal_baseline_difference_interval(run_root)
    records = read_jsonl(run_root / "records" / "formal_baseline_difference_interval_records.jsonl")

    assert audit["formal_baseline_difference_interval_decision"] == "PASS"
    assert audit["difference_interval_record_count"] == 5
    assert audit["difference_interval_ready_count"] == 5
    assert audit["difference_interval_missing_baseline_ids"] == []
    assert all(record["score_mean_difference"] == 0.2 for record in records)
    assert all(record["difference_interval_status"] == "ready" for record in records)
    assert all(record["significance_claim_status"] == "validation_scale_interval_not_significance_claim" for record in records)
    assert all(record["paired_comparison_unit_count"] == 3 for record in records)
    assert (run_root / "tables" / "formal_baseline_difference_interval_table.csv").exists()
    assert (run_root / "artifacts" / "formal_baseline_difference_interval_decision.json").exists()
    assert (run_root / "reports" / "formal_baseline_difference_interval_report.md").exists()


@pytest.mark.quick
def test_validation_scale_formal_internal_ablation_binds_full_method_formal_result(tmp_path: Path) -> None:
    """validation_scale 内部消融汇总必须把 full-method 行绑定到 SSTW measured_formal。"""
    run_root = tmp_path / "run"
    write_jsonl(run_root / "records" / "sstw_measured_formal_records.jsonl", [
        {"metric_status": "measured_formal", "sstw_score": 0.8},
        {"metric_status": "measured_formal", "sstw_score": 0.82},
    ])
    proxy_records = []
    for variant_name in (
        "endpoint_only_control",
        "trajectory_only_score",
        "without_velocity_constraint",
        "without_endpoint_aware_control",
        "without_replay_uncertainty_weighting",
        "without_flow_state_admissibility",
        "generic_ssm_baseline",
    ):
        proxy_records.append({
            "method_variant": variant_name,
            "ablation_status": "ready",
            "validation_ablation_proxy_score": 0.6,
        })
    write_jsonl(run_root / "records" / "validation_internal_ablation_records.jsonl", proxy_records)

    audit = run_validation_scale_formal_internal_ablation(run_root)
    records = read_jsonl(run_root / "records" / "validation_scale_formal_internal_ablation_records.jsonl")

    assert audit["validation_scale_formal_internal_ablation_decision"] == "PASS"
    assert audit["formal_internal_ablation_variant_count"] == 8
    full_row = next(record for record in records if record["method_variant"] == "sstw_full_method")
    assert full_row["metric_status"] == "measured_formal"
    assert full_row["formal_internal_ablation_score_mean"] == 0.81
    assert full_row["formal_internal_ablation_evidence_level"] == "sstw_measured_formal_full_method"
    assert any(record["metric_status"] == "measured_proxy" for record in records if record["method_variant"] != "sstw_full_method")
    assert (run_root / "tables" / "validation_scale_formal_internal_ablation_table.csv").exists()
    assert (run_root / "artifacts" / "validation_scale_formal_internal_ablation_decision.json").exists()
    assert (run_root / "reports" / "validation_scale_formal_internal_ablation_report.md").exists()


@pytest.mark.quick
def test_low_fpr_formal_statistics_writes_blocking_record(tmp_path: Path) -> None:
    """validation_scale 必须显式写出低 FPR 正式统计阻断记录。"""
    run_root = tmp_path / "run"
    write_jsonl(run_root / "records" / "generation_records.jsonl", [
        {"prompt_id": "negative_prompt", "negative_family": "clean_negative"},
        {"prompt_id": "positive_prompt", "negative_family": "not_applicable"},
    ])

    audit = run_low_fpr_formal_statistics(run_root)
    records = read_jsonl(run_root / "records" / "low_fpr_formal_statistics_records.jsonl")

    assert audit["low_fpr_formal_statistics_decision"] == "PASS"
    assert audit["formal_low_fpr_claim_allowed"] is False
    assert audit["low_fpr_formal_statistics_record_count"] >= 2
    assert {record["blocked_result_profile"] for record in records} >= {"pilot_paper", "full_paper"}
    assert all(record["formal_low_fpr_claim_allowed"] is False for record in records)
    assert all(record["claim_support_status"] == "low_fpr_formal_statistics_blocking_record" for record in records)
    assert (run_root / "tables" / "low_fpr_formal_statistics_table.csv").exists()
    assert (run_root / "artifacts" / "low_fpr_formal_statistics_decision.json").exists()
    assert (run_root / "reports" / "low_fpr_formal_statistics_report.md").exists()


@pytest.mark.quick
def test_motion_consistency_exclusion_report_classifies_blocked_samples(tmp_path: Path) -> None:
    """motion consistency 阻断样本必须保留审计记录且排除出效果 claim。"""
    run_root = tmp_path / "run"
    generation_records = [
        {
            "generation_status": "success",
            "generation_model_id": "model",
            "prompt_id": "prompt_ready",
            "seed_id": "seed_a",
            "trajectory_trace_id": "trace_ready",
            "motion_claim_role": "positive_motion",
        },
        {
            "generation_status": "success",
            "generation_model_id": "model",
            "prompt_id": "prompt_blocked",
            "seed_id": "seed_a",
            "trajectory_trace_id": "trace_blocked",
            "motion_claim_role": "positive_motion",
        },
    ]
    formal_records = [
        {
            **generation_records[0],
            "formal_visual_quality_ready": True,
            "formal_motion_consistency_ready": True,
            "formal_semantic_consistency_ready": True,
            "formal_metric_result_used_for_claim": True,
        },
        {
            **generation_records[1],
            "formal_visual_quality_ready": True,
            "formal_motion_consistency_ready": False,
            "formal_semantic_consistency_ready": True,
            "formal_metric_result_used_for_claim": True,
        },
    ]
    write_jsonl(run_root / "records" / "generation_records.jsonl", generation_records)
    write_jsonl(run_root / "records" / "formal_quality_motion_semantic_records.jsonl", formal_records)

    audit = run_motion_consistency_exclusion_report(run_root)
    records = read_jsonl(run_root / "records" / "motion_consistency_exclusion_records.jsonl")

    assert audit["motion_consistency_exclusion_decision"] == "PASS"
    assert audit["motion_consistency_included_count"] == 1
    assert audit["motion_consistency_excluded_count"] == 1
    blocked = next(record for record in records if record["prompt_id"] == "prompt_blocked")
    assert blocked["motion_consistency_exclusion_reason"] == "formal_motion_consistency_blocked"
    assert blocked["excluded_from_effect_size_claim"] is True
    assert blocked["retained_for_audit"] is True
    assert (run_root / "tables" / "motion_consistency_exclusion_table.csv").exists()
    assert (run_root / "artifacts" / "motion_consistency_exclusion_decision.json").exists()
    assert (run_root / "reports" / "motion_consistency_exclusion_report.md").exists()


@pytest.mark.quick
def test_validation_artifact_rebuild_dry_run_reports_missing_and_pass_states(tmp_path: Path) -> None:
    """artifact rebuild dry-run 必须能报告缺失状态, 并在必要产物齐全时通过。"""
    run_root = tmp_path / "run"
    failed = run_validation_artifact_rebuild_dry_run(run_root)
    assert failed["validation_artifact_rebuild_dry_run_decision"] == "FAIL"
    assert failed["artifact_rebuild_missing_count"] > 0

    required_files = [
        "records/generation_records.jsonl",
        "records/trajectory_trace.jsonl",
        "records/runtime_attack_records.jsonl",
        "records/runtime_detection_records.jsonl",
        "records/motion_consistency_exclusion_records.jsonl",
        "records/sstw_measured_formal_records.jsonl",
        "records/external_baseline_records.jsonl",
        "records/external_baseline_score_records.jsonl",
        "records/formal_method_baseline_comparison_records.jsonl",
        "records/formal_baseline_difference_interval_records.jsonl",
        "records/validation_scale_formal_internal_ablation_records.jsonl",
        "records/validation_internal_ablation_records.jsonl",
        "records/adaptive_attack_records.jsonl",
        "records/trajectory_sketch_verification_records.jsonl",
        "records/replay_uncertainty_records.jsonl",
        "records/wrong_sampler_replay_records.jsonl",
        "records/wrong_prompt_replay_records.jsonl",
        "records/claim3_downgrade_records.jsonl",
        "records/statistical_confidence_interval_records.jsonl",
        "records/low_fpr_formal_statistics_records.jsonl",
        "artifacts/generative_video_colab_runtime_decision.json",
        "artifacts/runtime_attack_decision.json",
        "artifacts/runtime_detection_decision.json",
        "artifacts/motion_consistency_exclusion_decision.json",
        "artifacts/sstw_measured_formal_decision.json",
        "artifacts/external_baseline_status_decision.json",
        "artifacts/external_baseline_comparison_decision.json",
        "artifacts/formal_method_baseline_comparison_decision.json",
        "artifacts/formal_baseline_difference_interval_decision.json",
        "artifacts/validation_scale_formal_internal_ablation_decision.json",
        "artifacts/validation_internal_ablation_decision.json",
        "artifacts/adaptive_attack_decision.json",
        "artifacts/replay_and_sketch_gate_decision.json",
        "artifacts/claim3_downgrade_decision.json",
        "artifacts/statistical_confidence_interval_decision.json",
        "artifacts/low_fpr_formal_statistics_decision.json",
        "tables/generation_runtime_table.csv",
        "tables/external_baseline_status_table.csv",
        "tables/external_baseline_comparison_table.csv",
        "tables/runtime_attack_table.csv",
        "tables/runtime_detection_table.csv",
        "tables/motion_consistency_exclusion_table.csv",
        "tables/sstw_measured_formal_table.csv",
        "tables/formal_method_baseline_comparison_table.csv",
        "tables/formal_baseline_difference_interval_table.csv",
        "tables/validation_scale_formal_internal_ablation_table.csv",
        "tables/validation_internal_ablation_table.csv",
        "tables/adaptive_attack_table.csv",
        "tables/replay_verification_table.csv",
        "tables/claim3_downgrade_table.csv",
        "tables/statistical_confidence_interval_table.csv",
        "tables/low_fpr_formal_statistics_table.csv",
        "reports/external_baseline_comparison_report.md",
        "reports/motion_consistency_exclusion_report.md",
        "reports/sstw_measured_formal_report.md",
        "reports/formal_method_baseline_comparison_report.md",
        "reports/formal_baseline_difference_interval_report.md",
        "reports/validation_scale_formal_internal_ablation_report.md",
        "reports/validation_internal_ablation_report.md",
        "reports/adaptive_attack_report.md",
        "reports/replay_and_sketch_gate_report.md",
        "reports/claim3_downgrade_report.md",
        "reports/statistical_confidence_interval_report.md",
        "reports/low_fpr_formal_statistics_report.md",
    ]
    for relative_path in required_files:
        path = run_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative_path.endswith(".json"):
            write_json(path, {"status": "ready"})
        else:
            path.write_text("ready\n", encoding="utf-8")

    passed = run_validation_artifact_rebuild_dry_run(run_root)

    assert passed["validation_artifact_rebuild_dry_run_decision"] == "PASS"
    assert passed["artifact_rebuild_missing_count"] == 0
    assert (run_root / "records" / "validation_artifact_rebuild_dry_run_records.jsonl").exists()
    assert (run_root / "tables" / "validation_artifact_rebuild_dry_run_table.csv").exists()
    assert (run_root / "reports" / "validation_artifact_rebuild_dry_run_report.md").exists()
