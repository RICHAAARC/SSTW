from __future__ import annotations

from pathlib import Path

import pytest

from experiments.generative_video_model_probe.validation_scale_artifact_package import (
    write_validation_scale_gate_figure,
    write_validation_scale_package_manifest,
)
from main.protocol.record_writer import write_json, write_jsonl
from scripts.check_results.data_split_and_leakage_guard import write_data_split_and_leakage_guard
from scripts.check_results.external_baseline_self_containment_decision import (
    write_external_baseline_self_containment_decision,
)
from scripts.check_results.stage_transition_decision import write_stage_transition_decision


MODERN_BASELINES = (
    "videoshield",
    "sigmark",
    "videomark",
    "vidsig",
    "videoseal",
)


@pytest.mark.quick
def test_validation_scale_to_pilot_transition_writes_governed_records(tmp_path: Path) -> None:
    """validation_scale PASS 后只能生成进入 pilot_paper 的跳转判定, 不能直接放行 full_paper。"""
    run_root = tmp_path / "run"
    write_json(run_root / "artifacts" / "validation_scale_gate_decision.json", {
        "validation_scale_gate_decision": "PASS",
        "full_paper_allowed": False,
        "claim_support_status": "validation_scale_ready_for_pilot_paper",
    })

    audit = write_stage_transition_decision(run_root, "validation_scale_to_pilot_paper")

    assert audit["validation_scale_to_pilot_paper_transition_decision"] == "PASS"
    assert audit["allowed_next_result_profiles"] == ["pilot_paper"]
    assert "full_paper" in audit["blocked_next_result_profiles"]
    assert audit["full_paper_allowed"] is False
    assert (run_root / "artifacts" / "validation_scale_to_pilot_paper_transition_decision.json").exists()
    assert (run_root / "records" / "validation_scale_to_pilot_paper_transition_decision_records.jsonl").exists()


@pytest.mark.quick
def test_pilot_to_full_transition_requires_previous_transition(tmp_path: Path) -> None:
    """pilot_paper -> full_paper 不能绕过 validation_scale -> pilot_paper 跳转记录。"""
    run_root = tmp_path / "run"
    write_json(run_root / "artifacts" / "pilot_paper_gate_decision.json", {
        "pilot_paper_gate_decision": "PASS",
        "pilot_paper_claim_allowed": True,
    })

    blocked = write_stage_transition_decision(run_root, "pilot_paper_to_full_paper")
    assert blocked["pilot_paper_to_full_paper_transition_decision"] == "FAIL"
    assert "validation_scale_to_pilot_paper_transition_decision_passed" in blocked["missing_transition_requirements"]

    write_json(run_root / "artifacts" / "validation_scale_to_pilot_paper_transition_decision.json", {
        "validation_scale_to_pilot_paper_transition_decision": "PASS",
    })
    passed = write_stage_transition_decision(run_root, "pilot_paper_to_full_paper")
    assert passed["pilot_paper_to_full_paper_transition_decision"] == "PASS"
    assert passed["full_paper_allowed"] is True


@pytest.mark.quick
def test_pilot_to_full_transition_can_consume_sibling_validation_scale_run_root(tmp_path: Path) -> None:
    """profile 隔离时, pilot_paper 可以消费同级 validation_scale run_root 中的跳转判定。"""
    project_run_root = tmp_path / "runs" / "generative_video_model_probe"
    validation_run_root = project_run_root / "validation_scale"
    pilot_run_root = project_run_root / "pilot_paper"
    write_json(pilot_run_root / "artifacts" / "pilot_paper_gate_decision.json", {
        "pilot_paper_gate_decision": "PASS",
    })
    write_json(validation_run_root / "artifacts" / "validation_scale_to_pilot_paper_transition_decision.json", {
        "validation_scale_to_pilot_paper_transition_decision": "PASS",
    })

    audit = write_stage_transition_decision(pilot_run_root, "pilot_paper_to_full_paper")

    assert audit["pilot_paper_to_full_paper_transition_decision"] == "PASS"


def _write_self_contained_external_baseline_fixture(run_root: Path) -> None:
    """构造 5 个主实验现代 baseline 的自包含 measured_formal fixture。"""
    score_records = []
    intake_rows = []
    inspection_rows = []
    clone_rows = []
    evidence_paths = []
    for baseline_name in MODERN_BASELINES:
        evidence_root = run_root / "artifacts" / "external_baseline_evidence" / baseline_name / "unit_000"
        output_path = evidence_root / "official_output.json"
        stdout_path = evidence_root / "official_stdout.txt"
        stderr_path = evidence_root / "official_stderr.txt"
        manifest_path = evidence_root / "official_command_manifest.json"
        write_json(output_path, {"score": 0.7})
        stdout_path.write_text("ok", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        write_json(manifest_path, {
            "command_return_code": 0,
            "claim_support_status": "external_baseline_official_command_evidence",
        })
        evidence_paths.extend([str(output_path), str(stdout_path), str(stderr_path), str(manifest_path)])
        score_records.append({
            "external_baseline_name": baseline_name,
            "external_baseline_layer": "modern_external_baseline",
            "external_baseline_adapter_path": f"external_baseline/primary/{baseline_name}/adapter/run_sstw_eval.py",
            "external_baseline_score_source": "official_command_adapter",
            "metric_status": "measured_formal",
            "external_baseline_score_status": "measured_formal",
            "external_baseline_official_output_path": str(output_path),
            "external_baseline_official_stdout_path": str(stdout_path),
            "external_baseline_official_stderr_path": str(stderr_path),
            "external_baseline_official_command_manifest_path": str(manifest_path),
        })
        intake_rows.append({
            "baseline_id": baseline_name,
            "source_intake_status": "source_snapshot_available",
            "source_dir_exists": True,
        })
        inspection_rows.append({
            "baseline_id": baseline_name,
            "source_dir_exists": True,
        })
        clone_rows.append({
            "baseline_id": baseline_name,
            "source_dir_exists": True,
            "clone_operation_status": "updated",
        })
    write_jsonl(run_root / "records" / "external_baseline_score_records.jsonl", score_records)
    write_json(run_root / "artifacts" / "external_baseline_comparison_decision.json", {
        "external_baseline_comparison_decision": "PASS",
    })
    write_json(run_root / "artifacts" / "external_baseline_execution_manifest.json", {
        "formal_evidence_status": "evidence_paths_bound",
        "evidence_path_count": len(evidence_paths),
        "modern_external_baseline_formal_measured_adapter_names": list(MODERN_BASELINES),
    })
    write_json(run_root / "artifacts" / "external_baseline_intake_manifest.json", {
        "baseline_sources": intake_rows,
    })
    write_json(run_root / "artifacts" / "external_baseline_source_inspection.json", {
        "source_inspections": inspection_rows,
    })
    write_json(run_root / "artifacts" / "external_baseline_clone_results.json", {
        "clone_results": clone_rows,
    })


@pytest.mark.quick
def test_external_baseline_self_containment_requires_measured_formal_evidence(tmp_path: Path) -> None:
    """正式 external baseline 必须同时具备 clone/build/run/adapt/record 证据。"""
    run_root = tmp_path / "run"
    _write_self_contained_external_baseline_fixture(run_root)

    audit = write_external_baseline_self_containment_decision(run_root)

    assert audit["external_baseline_self_containment_decision"] == "PASS"
    assert audit["self_contained_modern_external_baseline_count"] == 5
    assert audit["missing_self_contained_modern_external_baseline_names"] == []
    assert (run_root / "artifacts" / "external_baseline_self_containment_decision.json").exists()


@pytest.mark.quick
def test_external_baseline_self_containment_accepts_repository_generated_official_bundles(tmp_path: Path) -> None:
    """paper gate 可用项目内 official bundle execution manifest 证明 clone / build / run 闭环。"""
    run_root = tmp_path / "run"
    score_records = []
    evidence_paths = []
    intake_rows = []
    inspection_rows = []
    clone_rows = []
    for baseline_name in MODERN_BASELINES:
        evidence_root = run_root / "artifacts" / "external_baseline_evidence" / baseline_name / "unit_000"
        output_path = evidence_root / "official_output.json"
        stdout_path = evidence_root / "official_stdout.txt"
        stderr_path = evidence_root / "official_stderr.txt"
        command_manifest_path = evidence_root / "official_command_manifest.json"
        bundle_root = run_root / "external_baseline_official_result_bundles" / "validation_scale" / baseline_name
        execution_manifest_path = bundle_root / "official_reference_execution_manifest.json"
        bundle_record_path = bundle_root / "records" / "prompt_0__seed_0__video_compression_runtime.json"
        write_json(output_path, {"score": 0.7})
        stdout_path.write_text("ok", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        write_json(command_manifest_path, {
            "command_return_code": 0,
            "claim_support_status": "external_baseline_official_command_evidence",
        })
        write_json(execution_manifest_path, {
            "baseline_id": baseline_name,
            "execution_status": "executed",
            "failed_bundle_record_count": 0,
            "generated_bundle_record_count": 1,
            "command_results": [{"return_code": 0}],
            "claim_support_status": "official_reference_execution_evidence_not_measured_formal_record",
        })
        write_json(bundle_record_path, {
            "external_baseline_score": 0.7,
            "official_result_provenance": "repository_generated_from_third_party_official_code",
            "official_execution_manifest_path": str(execution_manifest_path),
        })
        evidence_paths.extend([str(output_path), str(stdout_path), str(stderr_path), str(command_manifest_path)])
        score_records.append({
            "external_baseline_name": baseline_name,
            "external_baseline_layer": "modern_external_baseline",
            "external_baseline_adapter_path": f"external_baseline/primary/{baseline_name}/adapter/run_sstw_eval.py",
            "external_baseline_score_source": "official_command_adapter",
            "metric_status": "measured_formal",
            "external_baseline_score_status": "measured_formal",
            "external_baseline_official_output_path": str(output_path),
            "external_baseline_official_stdout_path": str(stdout_path),
            "external_baseline_official_stderr_path": str(stderr_path),
            "external_baseline_official_command_manifest_path": str(command_manifest_path),
            "external_baseline_official_result_bundle_path": str(bundle_record_path),
            "external_baseline_official_execution_manifest_path": str(execution_manifest_path),
        })
        intake_rows.append({
            "baseline_id": baseline_name,
            "source_intake_status": "official_command_configured",
            "source_dir_exists": False,
        })
        inspection_rows.append({
            "baseline_id": baseline_name,
            "source_dir_exists": False,
        })
        clone_rows.append({
            "baseline_id": baseline_name,
            "source_dir_exists": False,
            "clone_operation_status": "planned_not_executed",
        })
    write_jsonl(run_root / "records" / "external_baseline_score_records.jsonl", score_records)
    write_json(run_root / "artifacts" / "external_baseline_comparison_decision.json", {
        "external_baseline_comparison_decision": "PASS",
    })
    write_json(run_root / "artifacts" / "external_baseline_execution_manifest.json", {
        "formal_evidence_status": "evidence_paths_bound",
        "evidence_path_count": len(evidence_paths),
        "modern_external_baseline_formal_measured_adapter_names": list(MODERN_BASELINES),
    })
    write_json(run_root / "artifacts" / "external_baseline_intake_manifest.json", {
        "baseline_sources": intake_rows,
    })
    write_json(run_root / "artifacts" / "external_baseline_source_inspection.json", {
        "source_inspections": inspection_rows,
    })
    write_json(run_root / "artifacts" / "external_baseline_clone_results.json", {
        "clone_results": clone_rows,
    })

    audit = write_external_baseline_self_containment_decision(run_root)

    assert audit["external_baseline_self_containment_decision"] == "PASS"
    assert audit["self_contained_modern_external_baseline_count"] == 5
    for row in audit["baseline_self_containment_rows"]:
        assert row["source_clone_ready"] is False
        assert row["repository_generated_official_bundle_ready"] is True
        assert row["clone_ready"] is True
        assert row["official_bundle_record_ok_count"] == 1
        assert row["official_execution_manifest_ok_count"] == 1


@pytest.mark.quick
def test_data_split_guard_detects_calibration_heldout_identity_leakage(tmp_path: Path) -> None:
    """同一视频身份不能同时出现在 calibration 与 held-out test split。"""
    run_root = tmp_path / "run"
    write_jsonl(run_root / "records" / "generation_records.jsonl", [
        {
            "generation_model_id": "model",
            "prompt_id": "prompt_0",
            "seed_id": "seed_0",
            "split": "calibration",
            "trajectory_trace_id": "trace_shared",
        },
        {
            "generation_model_id": "model",
            "prompt_id": "prompt_0",
            "seed_id": "seed_0",
            "split": "test",
            "trajectory_trace_id": "trace_shared",
        },
    ])
    write_json(run_root / "artifacts" / "motion_threshold_calibration_decision.json", {
        "motion_threshold_source_split": "calibration",
    })

    audit = write_data_split_and_leakage_guard(run_root)

    assert audit["data_split_and_leakage_guard_decision"] == "FAIL"
    assert audit["leakage_count"] == 1
    assert "calibration_heldout_test_identity_disjoint" in audit["missing_data_split_requirements"]


@pytest.mark.quick
def test_validation_scale_figure_and_package_manifest_are_rebuilt_from_artifacts(tmp_path: Path) -> None:
    """validation_scale 诊断图和 package manifest 必须由已落盘 artifact 派生。"""
    run_root = tmp_path / "run"
    write_jsonl(run_root / "records" / "validation_scale_gate_records.jsonl", [{"stage_id": "validation_scale"}])
    (run_root / "tables").mkdir(parents=True, exist_ok=True)
    (run_root / "tables" / "validation_scale_gate_table.csv").write_text("stage_id\nvalidation_scale\n", encoding="utf-8")
    (run_root / "reports").mkdir(parents=True, exist_ok=True)
    (run_root / "reports" / "validation_scale_gate_report.md").write_text("# report\n", encoding="utf-8")
    write_json(run_root / "artifacts" / "validation_scale_gate_decision.json", {
        "validation_scale_gate_decision": "PASS",
        "missing_validation_requirements": [],
    })
    write_json(run_root / "artifacts" / "external_baseline_self_containment_decision.json", {
        "external_baseline_self_containment_decision": "PASS",
    })
    write_json(run_root / "artifacts" / "sstw_measured_formal_decision.json", {
        "sstw_measured_formal_decision": "PASS",
    })
    write_json(run_root / "artifacts" / "formal_method_baseline_comparison_decision.json", {
        "formal_method_baseline_comparison_decision": "PASS",
    })
    write_json(run_root / "artifacts" / "formal_baseline_difference_interval_decision.json", {
        "formal_baseline_difference_interval_decision": "PASS",
    })
    write_json(run_root / "artifacts" / "data_split_and_leakage_guard_decision.json", {
        "data_split_and_leakage_guard_decision": "PASS",
    })
    write_json(run_root / "artifacts" / "validation_scale_to_pilot_paper_transition_decision.json", {
        "validation_scale_to_pilot_paper_transition_decision": "PASS",
    })

    figure = write_validation_scale_gate_figure(run_root)
    manifest = write_validation_scale_package_manifest(run_root)

    assert figure["validation_scale_gate_decision"] == "PASS"
    assert manifest["validation_scale_package_manifest_decision"] == "PASS"
    assert manifest["sstw_measured_formal_decision"] == "PASS"
    assert manifest["formal_method_baseline_comparison_decision"] == "PASS"
    assert manifest["formal_baseline_difference_interval_decision"] == "PASS"
    assert manifest["missing_artifact_relpaths"] == []
    assert (run_root / "figures" / "validation_scale_gate_figure.json").exists()
    assert (run_root / "manifests" / "validation_scale_package_manifest.json").exists()
