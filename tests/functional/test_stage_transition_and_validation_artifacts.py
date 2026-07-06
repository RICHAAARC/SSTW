from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.generative_video_model_probe.validation_scale_artifact_package import (
    VALIDATION_SCALE_REQUIRED_PACKAGE_RELPATHS,
    write_validation_scale_gate_figure,
    write_validation_scale_package_manifest,
)
from main.attacks.video_runtime_attack_protocol import VALIDATION_SCALE_RUNTIME_ATTACKS
from main.protocol.record_writer import write_json, write_jsonl
from scripts.check_results.data_split_and_leakage_guard import write_data_split_and_leakage_guard
from scripts.check_results.external_baseline_self_containment_decision import (
    write_external_baseline_self_containment_decision,
)
from scripts.check_results.stage_transition_decision import write_stage_transition_decision


MODERN_BASELINES = (
    "videoshield",
    "vidsig",
    "videoseal",
    "revmark",
    "wam_frame",
)


def _write_minimal_artifact_file(path: Path) -> None:
    """按文件类型写入最小可审计内容, 用于测试 package manifest 完整性。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".jsonl":
        path.write_text('{"status":"ready"}\n', encoding="utf-8")
    elif path.suffix == ".json":
        path.write_text('{"status":"ready"}\n', encoding="utf-8")
    elif path.suffix == ".csv":
        path.write_text("stage_id,status\nvalidation_scale,ready\n", encoding="utf-8")
    elif path.suffix == ".md":
        path.write_text("# Report\n\nready\n", encoding="utf-8")
    else:
        path.write_text("ready\n", encoding="utf-8")


def _write_minimal_validation_scale_package_artifacts(run_root: Path) -> None:
    """补齐 validation_scale package manifest 所需的通用最小 fixture。"""

    for relpath in VALIDATION_SCALE_REQUIRED_PACKAGE_RELPATHS:
        if relpath == "figures/validation_scale_gate_figure.json":
            continue
        target = run_root / relpath
        if not target.exists():
            _write_minimal_artifact_file(target)


def _official_score_extraction_payload() -> dict:
    """构造 official bundle 分数抽取口径 fixture。"""

    return {
        "score_semantics": "watermark_presence_confidence",
        "score_orientation": "higher_is_more_watermarked",
        "official_score_extraction_policy": "test_official_detector_confidence",
        "official_reference_protocol_anchor": "same_prompt_seed_attack_runtime_comparison_unit",
    }


def _validation_scale_gate_pass_payload() -> dict:
    """构造当前 validation_scale -> pilot_paper 所需的完整 PASS gate fixture。"""

    return {
        "validation_scale_gate_decision": "PASS",
        "claim_support_status": "validation_scale_target_fpr_0_1_paper_claim_supported",
        "paper_result_level": "validation_scale",
        "target_fpr": 0.1,
        "missing_validation_requirements": [],
        "validation_missing_requirement_count": 0,
        "required_modern_external_baseline_adapter_names": list(MODERN_BASELINES),
        "missing_modern_external_baseline_formal_adapter_names": [],
        "modern_external_baseline_formal_measured_adapter_count": len(MODERN_BASELINES),
        "external_baseline_self_containment_decision": "PASS",
        "data_split_and_leakage_guard_decision": "PASS",
        "runtime_attack_protocol_decision": "PASS",
        "required_runtime_attack_names": list(VALIDATION_SCALE_RUNTIME_ATTACKS),
        "runtime_attack_missing_required_names": [],
        "runtime_detection_missing_required_names": [],
        "runtime_detection_ready_count": len(VALIDATION_SCALE_RUNTIME_ATTACKS),
        "sstw_measured_formal_record_count": 24,
        "sstw_measured_formal_status": "sstw_measured_formal_paper_profile_claim_candidate",
        "fair_detection_calibration_ready_count": len(MODERN_BASELINES) + 1,
        "fair_detection_calibration_status": "fair_detection_calibration_validation_scale_ready",
        "formal_method_baseline_comparison_ready_count": len(MODERN_BASELINES) + 1,
        "formal_method_baseline_comparison_status": "formal_method_baseline_comparison_paper_profile_claim_candidate",
        "formal_baseline_difference_interval_ready_count": len(MODERN_BASELINES),
        "formal_baseline_difference_interval_status": "formal_baseline_difference_interval_paper_profile_claim_candidate",
        "validation_scale_sstw_advantage_claim_ready": True,
        "validation_scale_sstw_advantage_claim_status": "validation_scale_target_fpr_0_1_sstw_advantage_claim_supported",
        "full_paper_allowed": False,
    }


@pytest.mark.quick
def test_validation_scale_to_pilot_transition_writes_governed_records(tmp_path: Path) -> None:
    """validation_scale PASS 后只能生成进入 pilot_paper 的跳转判定, 不能直接放行 full_paper。"""
    run_root = tmp_path / "run"
    write_json(run_root / "artifacts" / "validation_scale_gate_decision.json", _validation_scale_gate_pass_payload())

    audit = write_stage_transition_decision(run_root, "validation_scale_to_pilot_paper")

    assert audit["validation_scale_to_pilot_paper_transition_decision"] == "PASS"
    assert audit["allowed_next_result_profiles"] == ["pilot_paper"]
    assert "full_paper" in audit["blocked_next_result_profiles"]
    assert audit["full_paper_allowed"] is False
    assert (run_root / "artifacts" / "validation_scale_to_pilot_paper_transition_decision.json").exists()
    assert (run_root / "records" / "validation_scale_to_pilot_paper_transition_decision_records.jsonl").exists()


@pytest.mark.quick
def test_validation_scale_to_pilot_transition_rejects_legacy_pass_without_fair_comparison(
    tmp_path: Path,
) -> None:
    """旧版 validation_scale PASS 若缺公平比较字段, 不能作为进入 pilot_paper 的依据。"""

    run_root = tmp_path / "run"
    write_json(run_root / "artifacts" / "validation_scale_gate_decision.json", {
        "validation_scale_gate_decision": "PASS",
        "full_paper_allowed": False,
        "claim_support_status": "validation_scale_ready_for_pilot_paper",
    })

    audit = write_stage_transition_decision(run_root, "validation_scale_to_pilot_paper")

    assert audit["validation_scale_to_pilot_paper_transition_decision"] == "FAIL"
    assert "validation_scale_fair_detection_calibration_ready" in audit["missing_transition_requirements"]
    assert "validation_scale_formal_method_baseline_comparison_ready" in audit["missing_transition_requirements"]
    assert "validation_scale_formal_baseline_difference_interval_ready" in audit["missing_transition_requirements"]


@pytest.mark.quick
def test_validation_scale_to_pilot_transition_requires_runtime_attack_coverage(tmp_path: Path) -> None:
    """validation_scale -> pilot_paper 跳转必须显式证明完整 runtime attack 已参与检测。"""

    run_root = tmp_path / "run"
    payload = _validation_scale_gate_pass_payload()
    payload["runtime_attack_protocol_decision"] = "FAIL"
    payload["required_runtime_attack_names"] = ["video_compression_runtime"]
    payload["runtime_attack_missing_required_names"] = ["temporal_crop_runtime"]
    payload["runtime_detection_missing_required_names"] = ["frame_rate_resampling_runtime"]
    payload["runtime_detection_ready_count"] = 1
    write_json(run_root / "artifacts" / "validation_scale_gate_decision.json", payload)

    audit = write_stage_transition_decision(run_root, "validation_scale_to_pilot_paper")

    assert audit["validation_scale_to_pilot_paper_transition_decision"] == "FAIL"
    assert "validation_scale_runtime_attack_protocol_passed" in audit["missing_transition_requirements"]
    assert "validation_scale_required_runtime_attacks_registered" in audit["missing_transition_requirements"]
    assert "validation_scale_runtime_attack_missing_required_names_empty" in audit["missing_transition_requirements"]
    assert "validation_scale_runtime_detection_missing_required_names_empty" in audit["missing_transition_requirements"]
    assert "validation_scale_runtime_detection_ready_count_covers_required_attacks" in audit["missing_transition_requirements"]


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
        bundle_root = run_root / "external_baseline_official_result_bundles" / "validation_scale" / baseline_name
        execution_manifest_path = bundle_root / "official_reference_execution_manifest.json"
        write_json(execution_manifest_path, {
            "baseline_id": baseline_name,
            "execution_status": "executed",
            "failed_bundle_record_count": 0,
            "generated_bundle_record_count": len(VALIDATION_SCALE_RUNTIME_ATTACKS),
            "command_results": [{"return_code": 0}],
            "claim_support_status": "official_reference_execution_evidence_not_measured_formal_record",
        })
        for attack_index, attack_name in enumerate(VALIDATION_SCALE_RUNTIME_ATTACKS):
            evidence_root = (
                run_root
                / "artifacts"
                / "external_baseline_evidence"
                / baseline_name
                / f"unit_{attack_index:03d}"
            )
            output_path = evidence_root / "official_output.json"
            stdout_path = evidence_root / "official_stdout.txt"
            stderr_path = evidence_root / "official_stderr.txt"
            manifest_path = evidence_root / "official_command_manifest.json"
            bundle_record_path = bundle_root / "records" / f"prompt_0__seed_0__{attack_name}.json"
            write_json(output_path, {"score": 0.7})
            stdout_path.write_text("ok", encoding="utf-8")
            stderr_path.write_text("", encoding="utf-8")
            write_json(manifest_path, {
                "command_return_code": 0,
                "claim_support_status": "external_baseline_official_command_evidence",
            })
            write_json(bundle_record_path, {
                "external_baseline_score": 0.7,
                "external_baseline_clean_negative_score": 0.2,
                "external_baseline_clean_negative_video_path": str(bundle_record_path),
                "official_result_provenance": "repository_generated_from_third_party_official_code",
                "official_adapter_baseline_id": baseline_name,
                "official_baseline_id": baseline_name,
                "official_execution_manifest_path": str(execution_manifest_path),
                "prompt_id": "prompt_0",
                "seed_id": "seed_0",
                "attack_name": attack_name,
                **_official_score_extraction_payload(),
            })
            evidence_paths.extend([str(output_path), str(stdout_path), str(stderr_path), str(manifest_path)])
            score_records.append({
                "external_baseline_name": baseline_name,
                "external_baseline_layer": "modern_external_baseline",
                "external_baseline_adapter_path": f"external_baseline/primary/{baseline_name}/adapter/run_sstw_eval.py",
                "external_baseline_score_source": "official_command_adapter",
                "metric_status": "measured_formal",
                "external_baseline_score_status": "measured_formal",
                "prompt_id": "prompt_0",
                "seed_id": "seed_0",
                "attack_name": attack_name,
                "external_baseline_clean_negative_score": 0.2,
                "external_baseline_clean_negative_video_path": str(output_path),
                "external_baseline_official_output_path": str(output_path),
                "external_baseline_official_stdout_path": str(stdout_path),
                "external_baseline_official_stderr_path": str(stderr_path),
                "external_baseline_official_command_manifest_path": str(manifest_path),
                "external_baseline_official_result_provenance": "repository_generated_from_third_party_official_code",
                "external_baseline_official_result_bundle_path": str(bundle_record_path),
                "external_baseline_official_execution_manifest_path": str(execution_manifest_path),
                "external_baseline_official_adapter_baseline_id": baseline_name,
                "external_baseline_official_baseline_id": baseline_name,
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
    assert audit["self_contained_modern_external_baseline_count"] == len(MODERN_BASELINES)
    assert audit["missing_self_contained_modern_external_baseline_names"] == []
    assert (run_root / "artifacts" / "external_baseline_self_containment_decision.json").exists()


@pytest.mark.quick
def test_external_baseline_self_containment_requires_each_runtime_attack_per_baseline(tmp_path: Path) -> None:
    """每个现代 baseline 都必须覆盖 validation_scale 要求的 runtime attack 集合。"""

    run_root = tmp_path / "run"
    _write_self_contained_external_baseline_fixture(run_root)
    score_path = run_root / "records" / "external_baseline_score_records.jsonl"
    score_records = [
        json.loads(line)
        for line in score_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    score_records = [
        record
        for record in score_records
        if not (
            record.get("external_baseline_name") == "videoseal"
            and record.get("attack_name") == "frame_rate_resampling_runtime"
        )
    ]
    write_jsonl(score_path, score_records)

    audit = write_external_baseline_self_containment_decision(run_root)
    row = next(item for item in audit["baseline_self_containment_rows"] if item["baseline_name"] == "videoseal")

    assert audit["external_baseline_self_containment_decision"] == "FAIL"
    assert row["runtime_attack_coverage_ready"] is False
    assert row["missing_runtime_attack_names"] == ["frame_rate_resampling_runtime"]
    assert audit["missing_runtime_attack_coverage_modern_external_baseline_names"] == ["videoseal"]
    assert "all_required_modern_baselines_required_runtime_attack_coverage" in audit["missing_self_containment_requirements"]


@pytest.mark.quick
def test_external_baseline_self_containment_rejects_bundle_anchor_mismatch(tmp_path: Path) -> None:
    """official bundle payload 的 prompt / seed / attack anchor 必须与 measured_formal record 一致。"""

    run_root = tmp_path / "run"
    _write_self_contained_external_baseline_fixture(run_root)
    baseline_name = "videoseal"
    bundle_record_path = (
        run_root
        / "external_baseline_official_result_bundles"
        / "validation_scale"
        / baseline_name
        / "records"
        / "prompt_0__seed_0__temporal_crop_runtime.json"
    )
    payload = json.loads(bundle_record_path.read_text(encoding="utf-8"))
    payload["attack_name"] = "video_compression_runtime"
    write_json(bundle_record_path, payload)

    audit = write_external_baseline_self_containment_decision(run_root)
    row = next(item for item in audit["baseline_self_containment_rows"] if item["baseline_name"] == baseline_name)

    assert audit["external_baseline_self_containment_decision"] == "FAIL"
    assert row["official_bundle_anchor_ready"] is False
    assert row["official_bundle_anchor_ready_count"] == len(VALIDATION_SCALE_RUNTIME_ATTACKS) - 1
    assert audit["missing_official_bundle_anchor_modern_external_baseline_names"] == [baseline_name]
    assert "all_required_modern_baselines_official_bundle_prompt_seed_attack_anchors" in audit["missing_self_containment_requirements"]


@pytest.mark.quick
def test_external_baseline_self_containment_rejects_command_only_formal_evidence(tmp_path: Path) -> None:
    """只有 detector command manifest 时不能替代项目内 official bundle 执行闭环。"""

    run_root = tmp_path / "run"
    baseline_name = "videoseal"
    evidence_root = run_root / "artifacts" / "external_baseline_evidence" / baseline_name / "unit_000"
    output_path = evidence_root / "official_output.json"
    stdout_path = evidence_root / "official_stdout.txt"
    stderr_path = evidence_root / "official_stderr.txt"
    command_manifest_path = evidence_root / "official_command_manifest.json"
    write_json(output_path, {"score": 0.7})
    stdout_path.write_text("ok", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    write_json(command_manifest_path, {
        "command_return_code": 0,
        "claim_support_status": "external_baseline_official_command_evidence",
    })
    write_jsonl(run_root / "records" / "external_baseline_score_records.jsonl", [{
        "external_baseline_name": baseline_name,
        "external_baseline_layer": "modern_external_baseline",
        "external_baseline_adapter_path": f"external_baseline/primary/{baseline_name}/adapter/run_sstw_eval.py",
        "external_baseline_score_source": "official_command_adapter",
        "metric_status": "measured_formal",
        "external_baseline_score_status": "measured_formal",
        "prompt_id": "prompt_0",
        "seed_id": "seed_0",
        "attack_name": "video_compression_runtime",
        "external_baseline_clean_negative_score": 0.2,
        "external_baseline_clean_negative_video_path": str(output_path),
        "external_baseline_official_output_path": str(output_path),
        "external_baseline_official_stdout_path": str(stdout_path),
        "external_baseline_official_stderr_path": str(stderr_path),
        "external_baseline_official_command_manifest_path": str(command_manifest_path),
    }])
    write_json(run_root / "artifacts" / "external_baseline_comparison_decision.json", {
        "external_baseline_comparison_decision": "PASS",
    })
    write_json(run_root / "artifacts" / "external_baseline_execution_manifest.json", {
        "formal_evidence_status": "evidence_paths_bound",
        "evidence_path_count": 4,
        "modern_external_baseline_formal_measured_adapter_names": [baseline_name],
    })
    write_json(run_root / "artifacts" / "external_baseline_intake_manifest.json", {
        "baseline_sources": [{"baseline_id": baseline_name, "source_intake_status": "source_snapshot_available", "source_dir_exists": True}],
    })
    write_json(run_root / "artifacts" / "external_baseline_source_inspection.json", {
        "source_inspections": [{"baseline_id": baseline_name, "source_dir_exists": True}],
    })
    write_json(run_root / "artifacts" / "external_baseline_clone_results.json", {
        "clone_results": [{"baseline_id": baseline_name, "source_dir_exists": True, "clone_operation_status": "updated"}],
    })
    config_path = run_root / "validation_scale_protocol.json"
    write_json(config_path, {"required_modern_external_baseline_adapter_names": [baseline_name]})

    audit = write_external_baseline_self_containment_decision(run_root, config_path)
    row = audit["baseline_self_containment_rows"][0]

    assert audit["external_baseline_self_containment_decision"] == "FAIL"
    assert row["repository_generated_official_bundle_ready"] is False
    assert row["run_ready"] is False
    assert "all_required_modern_baselines_repository_generated_official_bundles" in audit["missing_self_containment_requirements"]


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
        bundle_root = run_root / "external_baseline_official_result_bundles" / "validation_scale" / baseline_name
        execution_manifest_path = bundle_root / "official_reference_execution_manifest.json"
        write_json(execution_manifest_path, {
            "baseline_id": baseline_name,
            "execution_status": "executed",
            "failed_bundle_record_count": 0,
            "generated_bundle_record_count": len(VALIDATION_SCALE_RUNTIME_ATTACKS),
            "command_results": [{"return_code": 0}],
            "claim_support_status": "official_reference_execution_evidence_not_measured_formal_record",
        })
        for attack_index, attack_name in enumerate(VALIDATION_SCALE_RUNTIME_ATTACKS):
            evidence_root = (
                run_root
                / "artifacts"
                / "external_baseline_evidence"
                / baseline_name
                / f"unit_{attack_index:03d}"
            )
            output_path = evidence_root / "official_output.json"
            stdout_path = evidence_root / "official_stdout.txt"
            stderr_path = evidence_root / "official_stderr.txt"
            command_manifest_path = evidence_root / "official_command_manifest.json"
            bundle_record_path = bundle_root / "records" / f"prompt_0__seed_0__{attack_name}.json"
            write_json(output_path, {"score": 0.7})
            stdout_path.write_text("ok", encoding="utf-8")
            stderr_path.write_text("", encoding="utf-8")
            write_json(command_manifest_path, {
                "command_return_code": 0,
                "claim_support_status": "external_baseline_official_command_evidence",
            })
            write_json(bundle_record_path, {
                "external_baseline_score": 0.7,
                "external_baseline_clean_negative_score": 0.2,
                "external_baseline_clean_negative_video_path": str(bundle_record_path),
                "official_result_provenance": "repository_generated_from_third_party_official_code",
                "official_adapter_baseline_id": baseline_name,
                "official_baseline_id": baseline_name,
                "official_execution_manifest_path": str(execution_manifest_path),
                "prompt_id": "prompt_0",
                "seed_id": "seed_0",
                "attack_name": attack_name,
                **_official_score_extraction_payload(),
            })
            evidence_paths.extend([str(output_path), str(stdout_path), str(stderr_path), str(command_manifest_path)])
            score_records.append({
                "external_baseline_name": baseline_name,
                "external_baseline_layer": "modern_external_baseline",
                "external_baseline_adapter_path": f"external_baseline/primary/{baseline_name}/adapter/run_sstw_eval.py",
                "external_baseline_score_source": "official_command_adapter",
                "metric_status": "measured_formal",
                "external_baseline_score_status": "measured_formal",
                "prompt_id": "prompt_0",
                "seed_id": "seed_0",
                "attack_name": attack_name,
                "external_baseline_clean_negative_score": 0.2,
                "external_baseline_clean_negative_video_path": str(bundle_record_path),
                "external_baseline_official_output_path": str(output_path),
                "external_baseline_official_stdout_path": str(stdout_path),
                "external_baseline_official_stderr_path": str(stderr_path),
                "external_baseline_official_command_manifest_path": str(command_manifest_path),
                "external_baseline_official_result_provenance": "repository_generated_from_third_party_official_code",
                "external_baseline_official_result_bundle_path": str(bundle_record_path),
                "external_baseline_official_execution_manifest_path": str(execution_manifest_path),
                "external_baseline_official_adapter_baseline_id": baseline_name,
                "external_baseline_official_baseline_id": baseline_name,
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
    assert audit["self_contained_modern_external_baseline_count"] == len(MODERN_BASELINES)
    for row in audit["baseline_self_containment_rows"]:
        assert row["source_clone_ready"] is False
        assert row["repository_generated_official_bundle_ready"] is True
        assert row["clone_ready"] is True
        assert row["official_bundle_record_ok_count"] == len(VALIDATION_SCALE_RUNTIME_ATTACKS)
        assert row["official_execution_manifest_ok_count"] == len(VALIDATION_SCALE_RUNTIME_ATTACKS)
        assert row["runtime_attack_coverage_ready"] is True
        assert row["clean_negative_ready"] is True


@pytest.mark.quick
@pytest.mark.quick
def test_external_baseline_self_containment_requires_complete_official_baseline_identity(tmp_path: Path) -> None:
    """official bundle 必须同时声明 adapter baseline 和 official baseline 身份。"""

    run_root = tmp_path / "run"
    _write_self_contained_external_baseline_fixture(run_root)
    baseline_name = "videoseal"
    bundle_record_path = (
        run_root
        / "external_baseline_official_result_bundles"
        / "validation_scale"
        / baseline_name
        / "records"
        / "prompt_0__seed_0__video_compression_runtime.json"
    )
    payload = json.loads(bundle_record_path.read_text(encoding="utf-8"))
    payload.pop("official_adapter_baseline_id", None)
    write_json(bundle_record_path, payload)

    audit = write_external_baseline_self_containment_decision(run_root)
    row = next(item for item in audit["baseline_self_containment_rows"] if item["baseline_name"] == baseline_name)

    assert audit["external_baseline_self_containment_decision"] == "FAIL"
    assert row["repository_generated_official_bundle_ready"] is False
    assert row["official_baseline_identity_ready"] is True
    assert baseline_name in audit["missing_repository_generated_official_bundle_modern_external_baseline_names"]
    assert "all_required_modern_baselines_repository_generated_official_bundles" in audit["missing_self_containment_requirements"]


@pytest.mark.quick
def test_external_baseline_self_containment_requires_record_official_baseline_identity(tmp_path: Path) -> None:
    """measured_formal record 必须保留 official bundle 转写后的完整 baseline 身份。"""

    run_root = tmp_path / "run"
    _write_self_contained_external_baseline_fixture(run_root)
    baseline_name = "videoseal"
    score_path = run_root / "records" / "external_baseline_score_records.jsonl"
    score_records = [
        json.loads(line)
        for line in score_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for record in score_records:
        if record.get("external_baseline_name") == baseline_name:
            record.pop("external_baseline_official_adapter_baseline_id", None)
    write_jsonl(score_path, score_records)

    audit = write_external_baseline_self_containment_decision(run_root)
    row = next(item for item in audit["baseline_self_containment_rows"] if item["baseline_name"] == baseline_name)

    assert audit["external_baseline_self_containment_decision"] == "FAIL"
    assert row["repository_generated_official_bundle_ready"] is True
    assert row["official_baseline_identity_ready"] is False
    assert baseline_name in audit["missing_official_identity_modern_external_baseline_names"]
    assert "all_required_modern_baselines_official_baseline_identity" in audit["missing_self_containment_requirements"]


@pytest.mark.quick
def test_external_baseline_self_containment_rejects_bundle_without_clean_negative(tmp_path: Path) -> None:
    """旧 official bundle 若缺少 clean negative 分数, 不能作为公平比较自包含证据。"""

    run_root = tmp_path / "run"
    baseline_name = "videoseal"
    bundle_root = run_root / "external_baseline_official_result_bundles" / "validation_scale" / baseline_name
    execution_manifest_path = bundle_root / "official_reference_execution_manifest.json"
    bundle_record_path = bundle_root / "records" / "prompt_0__seed_0__video_compression_runtime.json"
    output_path = run_root / "artifacts" / "external_baseline_evidence" / baseline_name / "unit_000" / "official_output.json"
    stdout_path = output_path.with_name("official_stdout.txt")
    stderr_path = output_path.with_name("official_stderr.txt")
    command_manifest_path = output_path.with_name("official_command_manifest.json")
    write_json(output_path, {"score": 0.7})
    stdout_path.write_text("ok", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    write_json(command_manifest_path, {"command_return_code": 0})
    write_json(execution_manifest_path, {
        "baseline_id": baseline_name,
        "execution_status": "executed",
        "failed_bundle_record_count": 0,
        "generated_bundle_record_count": 1,
        "command_results": [{"return_code": 0}],
    })
    write_json(bundle_record_path, {
        "external_baseline_score": 0.7,
        "official_result_provenance": "repository_generated_from_third_party_official_code",
        "official_adapter_baseline_id": baseline_name,
        "official_baseline_id": baseline_name,
        "official_execution_manifest_path": str(execution_manifest_path),
        "prompt_id": "prompt_0",
        "seed_id": "seed_0",
        "attack_name": "video_compression_runtime",
        **_official_score_extraction_payload(),
    })
    write_jsonl(run_root / "records" / "external_baseline_score_records.jsonl", [{
        "external_baseline_name": baseline_name,
        "external_baseline_layer": "modern_external_baseline",
        "external_baseline_adapter_path": f"external_baseline/primary/{baseline_name}/adapter/run_sstw_eval.py",
        "external_baseline_score_source": "official_command_adapter",
        "metric_status": "measured_formal",
        "external_baseline_score_status": "measured_formal",
        "prompt_id": "prompt_0",
        "seed_id": "seed_0",
        "attack_name": "video_compression_runtime",
        "external_baseline_official_output_path": str(output_path),
        "external_baseline_official_stdout_path": str(stdout_path),
        "external_baseline_official_stderr_path": str(stderr_path),
        "external_baseline_official_command_manifest_path": str(command_manifest_path),
        "external_baseline_official_result_provenance": "repository_generated_from_third_party_official_code",
        "external_baseline_official_result_bundle_path": str(bundle_record_path),
        "external_baseline_official_execution_manifest_path": str(execution_manifest_path),
        "external_baseline_official_adapter_baseline_id": baseline_name,
        "external_baseline_official_baseline_id": baseline_name,
    }])
    write_json(run_root / "artifacts" / "external_baseline_comparison_decision.json", {
        "external_baseline_comparison_decision": "PASS",
    })
    write_json(run_root / "artifacts" / "external_baseline_execution_manifest.json", {
        "formal_evidence_status": "evidence_paths_bound",
        "evidence_path_count": 4,
        "modern_external_baseline_formal_measured_adapter_names": [baseline_name],
    })
    write_json(run_root / "artifacts" / "external_baseline_intake_manifest.json", {
        "baseline_sources": [{"baseline_id": baseline_name, "source_intake_status": "official_command_configured"}],
    })
    write_json(run_root / "artifacts" / "external_baseline_source_inspection.json", {
        "source_inspections": [{"baseline_id": baseline_name, "source_dir_exists": False}],
    })
    write_json(run_root / "artifacts" / "external_baseline_clone_results.json", {
        "clone_results": [{"baseline_id": baseline_name, "source_dir_exists": False, "clone_operation_status": "planned_not_executed"}],
    })
    config_path = run_root / "validation_scale_protocol.json"
    write_json(config_path, {"required_modern_external_baseline_adapter_names": [baseline_name]})

    audit = write_external_baseline_self_containment_decision(run_root, config_path)

    assert audit["external_baseline_self_containment_decision"] == "FAIL"
    assert audit["missing_clean_negative_modern_external_baseline_names"] == [baseline_name]
    assert "all_required_modern_baselines_clean_negative_scores" in audit["missing_self_containment_requirements"]


@pytest.mark.quick
def test_external_baseline_self_containment_rejects_bundle_without_score_extraction_policy(tmp_path: Path) -> None:
    """official bundle 若缺少分数口径, 不能作为公平比较自包含证据。"""

    run_root = tmp_path / "run"
    baseline_name = "videoseal"
    bundle_root = run_root / "external_baseline_official_result_bundles" / "validation_scale" / baseline_name
    execution_manifest_path = bundle_root / "official_reference_execution_manifest.json"
    bundle_record_path = bundle_root / "records" / "prompt_0__seed_0__video_compression_runtime.json"
    output_path = run_root / "artifacts" / "external_baseline_evidence" / baseline_name / "unit_000" / "official_output.json"
    stdout_path = output_path.with_name("official_stdout.txt")
    stderr_path = output_path.with_name("official_stderr.txt")
    command_manifest_path = output_path.with_name("official_command_manifest.json")
    write_json(output_path, {"score": 0.7})
    stdout_path.write_text("ok", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    write_json(command_manifest_path, {"command_return_code": 0})
    write_json(execution_manifest_path, {
        "baseline_id": baseline_name,
        "execution_status": "executed",
        "failed_bundle_record_count": 0,
        "generated_bundle_record_count": 1,
        "command_results": [{"return_code": 0}],
    })
    write_json(bundle_record_path, {
        "external_baseline_score": 0.7,
        "external_baseline_clean_negative_score": 0.2,
        "external_baseline_clean_negative_video_path": str(bundle_record_path),
        "official_result_provenance": "repository_generated_from_third_party_official_code",
        "official_adapter_baseline_id": baseline_name,
        "official_baseline_id": baseline_name,
        "official_execution_manifest_path": str(execution_manifest_path),
        "prompt_id": "prompt_0",
        "seed_id": "seed_0",
        "attack_name": "video_compression_runtime",
    })
    write_jsonl(run_root / "records" / "external_baseline_score_records.jsonl", [{
        "external_baseline_name": baseline_name,
        "external_baseline_layer": "modern_external_baseline",
        "external_baseline_adapter_path": f"external_baseline/primary/{baseline_name}/adapter/run_sstw_eval.py",
        "external_baseline_score_source": "official_command_adapter",
        "metric_status": "measured_formal",
        "external_baseline_score_status": "measured_formal",
        "prompt_id": "prompt_0",
        "seed_id": "seed_0",
        "attack_name": "video_compression_runtime",
        "external_baseline_clean_negative_score": 0.2,
        "external_baseline_clean_negative_video_path": str(bundle_record_path),
        "external_baseline_official_output_path": str(output_path),
        "external_baseline_official_stdout_path": str(stdout_path),
        "external_baseline_official_stderr_path": str(stderr_path),
        "external_baseline_official_command_manifest_path": str(command_manifest_path),
        "external_baseline_official_result_provenance": "repository_generated_from_third_party_official_code",
        "external_baseline_official_result_bundle_path": str(bundle_record_path),
        "external_baseline_official_execution_manifest_path": str(execution_manifest_path),
        "external_baseline_official_adapter_baseline_id": baseline_name,
        "external_baseline_official_baseline_id": baseline_name,
    }])
    write_json(run_root / "artifacts" / "external_baseline_comparison_decision.json", {
        "external_baseline_comparison_decision": "PASS",
    })
    write_json(run_root / "artifacts" / "external_baseline_execution_manifest.json", {
        "formal_evidence_status": "evidence_paths_bound",
        "evidence_path_count": 4,
        "modern_external_baseline_formal_measured_adapter_names": [baseline_name],
    })
    write_json(run_root / "artifacts" / "external_baseline_intake_manifest.json", {
        "baseline_sources": [{"baseline_id": baseline_name, "source_intake_status": "official_command_configured"}],
    })
    write_json(run_root / "artifacts" / "external_baseline_source_inspection.json", {
        "source_inspections": [{"baseline_id": baseline_name, "source_dir_exists": False}],
    })
    write_json(run_root / "artifacts" / "external_baseline_clone_results.json", {
        "clone_results": [{"baseline_id": baseline_name, "source_dir_exists": False, "clone_operation_status": "planned_not_executed"}],
    })
    config_path = run_root / "validation_scale_protocol.json"
    write_json(config_path, {"required_modern_external_baseline_adapter_names": [baseline_name]})

    audit = write_external_baseline_self_containment_decision(run_root, config_path)
    row = audit["baseline_self_containment_rows"][0]

    assert audit["external_baseline_self_containment_decision"] == "FAIL"
    assert row["score_extraction_ready"] is False
    assert audit["missing_score_extraction_modern_external_baseline_names"] == [baseline_name]
    assert "all_required_modern_baselines_official_score_extraction" in audit["missing_self_containment_requirements"]


@pytest.mark.quick
def test_external_baseline_self_containment_rejects_records_without_complete_anchor(tmp_path: Path) -> None:
    """formal baseline record 缺少 prompt / seed / attack anchor 时不能进入公平比较。"""

    run_root = tmp_path / "run"
    baseline_name = "videoseal"
    evidence_root = run_root / "artifacts" / "external_baseline_evidence" / baseline_name / "unit_000"
    output_path = evidence_root / "official_output.json"
    stdout_path = evidence_root / "official_stdout.txt"
    stderr_path = evidence_root / "official_stderr.txt"
    command_manifest_path = evidence_root / "official_command_manifest.json"
    write_json(output_path, {"score": 0.7})
    stdout_path.write_text("ok", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    write_json(command_manifest_path, {"command_return_code": 0})
    write_jsonl(run_root / "records" / "external_baseline_score_records.jsonl", [{
        "external_baseline_name": baseline_name,
        "external_baseline_layer": "modern_external_baseline",
        "external_baseline_adapter_path": f"external_baseline/primary/{baseline_name}/adapter/run_sstw_eval.py",
        "external_baseline_score_source": "official_command_adapter",
        "metric_status": "measured_formal",
        "external_baseline_score_status": "measured_formal",
        "prompt_id": "prompt_0",
        "attack_name": "video_compression_runtime",
        "external_baseline_clean_negative_score": 0.2,
        "external_baseline_clean_negative_video_path": str(output_path),
        "external_baseline_official_output_path": str(output_path),
        "external_baseline_official_stdout_path": str(stdout_path),
        "external_baseline_official_stderr_path": str(stderr_path),
        "external_baseline_official_command_manifest_path": str(command_manifest_path),
    }])
    write_json(run_root / "artifacts" / "external_baseline_comparison_decision.json", {
        "external_baseline_comparison_decision": "PASS",
    })
    write_json(run_root / "artifacts" / "external_baseline_execution_manifest.json", {
        "formal_evidence_status": "evidence_paths_bound",
        "evidence_path_count": 4,
        "modern_external_baseline_formal_measured_adapter_names": [baseline_name],
    })
    write_json(run_root / "artifacts" / "external_baseline_intake_manifest.json", {
        "baseline_sources": [{"baseline_id": baseline_name, "source_intake_status": "source_snapshot_available", "source_dir_exists": True}],
    })
    write_json(run_root / "artifacts" / "external_baseline_source_inspection.json", {
        "source_inspections": [{"baseline_id": baseline_name, "source_dir_exists": True}],
    })
    write_json(run_root / "artifacts" / "external_baseline_clone_results.json", {
        "clone_results": [{"baseline_id": baseline_name, "source_dir_exists": True, "clone_operation_status": "updated"}],
    })
    config_path = run_root / "validation_scale_protocol.json"
    write_json(config_path, {"required_modern_external_baseline_adapter_names": [baseline_name]})

    audit = write_external_baseline_self_containment_decision(run_root, config_path)
    row = audit["baseline_self_containment_rows"][0]

    assert audit["external_baseline_self_containment_decision"] == "FAIL"
    assert row["formal_candidate_record_count"] == 1
    assert row["formal_anchor_missing_count"] == 1
    assert row["anchor_ready"] is False
    assert audit["missing_anchor_modern_external_baseline_names"] == [baseline_name]
    assert "all_required_modern_baselines_prompt_seed_attack_anchors" in audit["missing_self_containment_requirements"]


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
    _write_minimal_validation_scale_package_artifacts(run_root)
    write_jsonl(run_root / "records" / "validation_scale_gate_records.jsonl", [{"stage_id": "validation_scale"}])
    (run_root / "tables").mkdir(parents=True, exist_ok=True)
    (run_root / "tables" / "validation_scale_gate_table.csv").write_text("stage_id\nvalidation_scale\n", encoding="utf-8")
    (run_root / "reports").mkdir(parents=True, exist_ok=True)
    (run_root / "reports" / "validation_scale_gate_report.md").write_text("# report\n", encoding="utf-8")
    write_json(run_root / "artifacts" / "validation_scale_gate_decision.json", {
        "validation_scale_gate_decision": "PASS",
        "missing_validation_requirements": [],
    })
    write_json(run_root / "artifacts" / "motion_consistency_exclusion_decision.json", {
        "motion_consistency_exclusion_decision": "PASS",
    })
    write_json(run_root / "artifacts" / "external_baseline_self_containment_decision.json", {
        "external_baseline_self_containment_decision": "PASS",
    })
    write_json(run_root / "artifacts" / "sstw_measured_formal_decision.json", {
        "sstw_measured_formal_decision": "PASS",
    })
    write_json(run_root / "artifacts" / "fair_detection_calibration_decision.json", {
        "fair_detection_calibration_decision": "PASS",
    })
    write_json(run_root / "artifacts" / "formal_method_baseline_comparison_decision.json", {
        "formal_method_baseline_comparison_decision": "PASS",
    })
    write_json(run_root / "artifacts" / "formal_baseline_difference_interval_decision.json", {
        "formal_baseline_difference_interval_decision": "PASS",
    })
    write_json(run_root / "artifacts" / "validation_scale_formal_internal_ablation_decision.json", {
        "validation_scale_formal_internal_ablation_decision": "PASS",
    })
    write_json(run_root / "artifacts" / "low_fpr_formal_statistics_decision.json", {
        "low_fpr_formal_statistics_decision": "PASS",
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
    assert manifest["motion_consistency_exclusion_decision"] == "PASS"
    assert manifest["sstw_measured_formal_decision"] == "PASS"
    assert manifest["fair_detection_calibration_decision"] == "PASS"
    assert manifest["formal_method_baseline_comparison_decision"] == "PASS"
    assert manifest["formal_baseline_difference_interval_decision"] == "PASS"
    assert manifest["validation_scale_formal_internal_ablation_decision"] == "PASS"
    assert manifest["low_fpr_formal_statistics_decision"] == "PASS"
    assert manifest["missing_artifact_relpaths"] == []
    inventory_relpaths = {row["artifact_relpath"] for row in manifest["artifact_inventory"]}
    assert "records/fair_detection_calibration_records.jsonl" in inventory_relpaths
    assert "tables/fair_detection_calibration_table.csv" in inventory_relpaths
    assert "reports/fair_detection_calibration_report.md" in inventory_relpaths
    assert (run_root / "figures" / "validation_scale_gate_figure.json").exists()
    assert (run_root / "manifests" / "validation_scale_package_manifest.json").exists()
