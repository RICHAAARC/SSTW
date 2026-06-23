from __future__ import annotations

from pathlib import Path

import pytest

from experiments.generative_video_model_probe.statistical_confidence_interval import run_statistical_confidence_interval_reporter
from experiments.generative_video_model_probe.validation_artifact_rebuild import run_validation_artifact_rebuild_dry_run
from experiments.generative_video_model_probe.validation_internal_ablation import run_validation_internal_ablation
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
    assert (run_root / "tables" / "validation_internal_ablation_table.csv").exists()
    assert (run_root / "reports" / "validation_internal_ablation_report.md").exists()


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

    assert audit["statistical_confidence_interval_decision"] == "PASS"
    assert audit["ci_total_count"] == 3
    assert audit["ci_success_count"] == 2
    assert 0 <= audit["ci_wilson_lower"] <= audit["ci_wilson_upper"] <= 1
    assert records[0]["paper_low_fpr_ci_status"] == "not_available_until_full_paper_negative_split"
    assert (run_root / "tables" / "statistical_confidence_interval_table.csv").exists()
    assert (run_root / "reports" / "statistical_confidence_interval_report.md").exists()


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
        "records/external_baseline_records.jsonl",
        "records/external_baseline_score_records.jsonl",
        "records/validation_internal_ablation_records.jsonl",
        "records/adaptive_attack_records.jsonl",
        "records/claim3_downgrade_records.jsonl",
        "records/statistical_confidence_interval_records.jsonl",
        "artifacts/generative_video_colab_runtime_decision.json",
        "artifacts/runtime_attack_decision.json",
        "artifacts/runtime_detection_decision.json",
        "artifacts/external_baseline_status_decision.json",
        "artifacts/external_baseline_comparison_decision.json",
        "artifacts/validation_internal_ablation_decision.json",
        "artifacts/adaptive_attack_decision.json",
        "artifacts/claim3_downgrade_decision.json",
        "artifacts/statistical_confidence_interval_decision.json",
        "tables/generation_runtime_table.csv",
        "tables/external_baseline_status_table.csv",
        "tables/external_baseline_comparison_table.csv",
        "tables/runtime_attack_table.csv",
        "tables/runtime_detection_table.csv",
        "tables/validation_internal_ablation_table.csv",
        "tables/adaptive_attack_table.csv",
        "tables/claim3_downgrade_table.csv",
        "tables/statistical_confidence_interval_table.csv",
        "reports/external_baseline_comparison_report.md",
        "reports/validation_internal_ablation_report.md",
        "reports/adaptive_attack_report.md",
        "reports/claim3_downgrade_report.md",
        "reports/statistical_confidence_interval_report.md",
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
