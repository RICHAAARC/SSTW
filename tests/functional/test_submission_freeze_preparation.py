"""验证 submission freeze preparation claim audit 闭环。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.submission_freeze_preparation.runner import run_submission_freeze_preparation


def _write_json(path: Path, payload: dict) -> None:
    """写入 JSON 测试文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    """写入 JSONL 测试文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n", encoding="utf-8")


@pytest.mark.quick
def test_submission_freeze_preparation_downgrades_sstw_tc_claim(tmp_path: Path) -> None:
    """submission preparation 必须支持 SSTW-T, 同时降级 SSTW-TC 最终 claim。"""
    stage_paths: dict[str, Path] = {}
    for stage_id in (
        "synthetic_state_protocol",
        "state_space_inference_formalization",
        "real_video_latent_transfer",
        "trajectory_observation_core_probe",
        "sampling_time_constraint_preflight",
    ):
        path = tmp_path / "stage_decisions" / stage_id / "decision.json"
        _write_json(path, {
            "stage_id": stage_id,
            "implementation_decision": "PASS",
            "mechanism_decision": "PASS",
            "details": {"mechanism_pass": True},
        })
        stage_paths[stage_id] = path

    b5_run = tmp_path / "b5_run"
    b5_video = b5_run / "videos" / "sample.mp4"
    b5_video.parent.mkdir(parents=True, exist_ok=True)
    b5_video.write_bytes(b"b5-video")
    import hashlib

    b5_hash = hashlib.sha256(b5_video.read_bytes()).hexdigest()
    b5_generation_records = []
    b5_trajectory_records = []
    for index in range(4):
        trace_id = f"b5_trace_{index}"
        b5_generation_records.append({
            "generation_status": "success",
            "trajectory_trace_id": trace_id,
            "generation_model_id": "test_model",
            "prompt_id": f"prompt_{index:03d}",
            "seed_id": "seed_001",
            "video_path": str(b5_video),
            "video_sha256": b5_hash,
        })
        b5_trajectory_records.append({"trajectory_trace_id": trace_id})
    _write_jsonl(b5_run / "records" / "generation_records.jsonl", b5_generation_records)
    _write_jsonl(b5_run / "records" / "trajectory_trace.jsonl", b5_trajectory_records)
    _write_jsonl(b5_run / "records" / "external_baseline_records.jsonl", [{"external_baseline_runnable_status": "runnable"}])
    _write_jsonl(b5_run / "records" / "formal_quality_motion_semantic_records.jsonl", [
        {
            "formal_visual_quality_ready": True,
            "formal_motion_consistency_ready": True,
            "formal_semantic_consistency_ready": True,
        }
        for _ in range(4)
    ])
    _write_json(b5_run / "artifacts" / "generation_manifest.json", {"artifact_id": "b5_manifest"})
    _write_json(b5_run / "artifacts" / "generative_video_colab_runtime_decision.json", {
        "stage_id": "generative_video_model_probe_colab",
        "implementation_decision": "PASS",
        "mechanism_decision": "FAIL",
        "details": {"fixed_low_fpr_audit_pass": True, "trajectory_observation_gain_confirmed": True},
    })
    _write_json(b5_run / "artifacts" / "generative_video_mechanism_postprocess_decision.json", {
        "stage_id": "generative_video_mechanism_postprocess",
        "mechanism_postprocess_decision": "PASS",
        "mechanism_decision": "PASS",
        "details": {"formal_quality_semantic_ready": True},
    })

    b6_run = tmp_path / "b6_run"
    b6_video = b6_run / "videos" / "sample.mp4"
    b6_video.parent.mkdir(parents=True, exist_ok=True)
    b6_video.write_bytes(b"b6-video")
    b6_hash = hashlib.sha256(b6_video.read_bytes()).hexdigest()
    variants = [
        "key_conditioned_state_space_with_trajectory",
        "keyed_state_trajectory_constraint",
        "trajectory_constraint_without_admissibility",
        "trajectory_constraint_without_key_condition",
        "trajectory_constraint_wrong_key_control",
    ]
    generation_records = []
    trajectory_records = []
    constraint_records = []
    formal_records = []
    summary_records = []
    for index, variant in enumerate(variants):
        trace_id = f"b6_trace_{index}"
        constraint_trace_id = f"b6_constraint_{index}"
        generation_records.append({
            "colab_runtime_profile": "recommended",
            "generation_status": "success",
            "generation_model_id": "test_model",
            "generation_model_family": "diffusers_wan21_flow_matching_dit",
            "flow_matching_backbone_claim_status": "wan21_primary_flow_matching_claim",
            "method_variant": variant,
            "prompt_id": "prompt_001",
            "seed_id": "seed_001",
            "trajectory_trace_id": trace_id,
            "constraint_trace_id": constraint_trace_id,
            "video_path": str(b6_video),
            "video_sha256": b6_hash,
        })
        trajectory_records.append({"trajectory_trace_id": trace_id})
        applied = variant == "keyed_state_trajectory_constraint"
        constraint_records.append({
            "constraint_trace_id": constraint_trace_id,
            "trajectory_trace_id": trace_id,
            "method_variant": variant,
            "constraint_apply_status": "applied" if applied else "not_applied",
            "latent_alignment_gain": 0.1 if applied else 0.0,
            "flow_velocity_proxy_available": True,
            "flow_velocity_alignment_gain": 0.1 if applied else 0.0,
        })
        formal_records.append({
            "method_variant": variant,
            "formal_visual_quality_ready": True,
            "formal_motion_consistency_ready": True,
            "formal_semantic_consistency_ready": True,
        })
        summary_records.append({"method_variant": variant})
    _write_jsonl(b6_run / "records" / "generation_records.jsonl", generation_records)
    _write_jsonl(b6_run / "records" / "trajectory_trace.jsonl", trajectory_records)
    _write_jsonl(b6_run / "records" / "constraint_records.jsonl", constraint_records)
    _write_jsonl(b6_run / "records" / "formal_quality_motion_semantic_records.jsonl", formal_records)
    _write_jsonl(b6_run / "records" / "constraint_variant_summary_records.jsonl", summary_records)
    _write_json(b6_run / "artifacts" / "generation_manifest.json", {"artifact_id": "b6_manifest"})
    _write_json(b6_run / "artifacts" / "sampling_time_constraint_colab_runtime_decision.json", {
        "stage_id": "sampling_time_constraint_colab_probe",
        "implementation_decision": "PASS",
        "mechanism_decision": "FAIL",
    })
    _write_json(b6_run / "artifacts" / "sampling_time_constraint_colab_postprocess_decision.json", {
        "stage_id": "sampling_time_constraint_colab_postprocess",
        "mechanism_postprocess_decision": "PASS",
        "mechanism_decision": "PASS",
        "details": {
            "keyed_constraint_alignment_gain_mean": 0.1,
            "baseline_alignment_gain_mean": 0.0,
                "keyed_flow_velocity_alignment_gain_mean": 0.1,
                "baseline_flow_velocity_alignment_gain_mean": 0.0,
                "without_key_alignment_gain_mean": 0.0,
                "wrong_key_alignment_gain_mean": 0.0,
                "without_key_flow_velocity_alignment_gain_mean": 0.0,
                "wrong_key_flow_velocity_alignment_gain_mean": 0.0,
                "key_separation_gain_over_control": 0.1,
                "key_separation_flow_velocity_gain_over_control": 0.1,
                "flow_velocity_proxy_ready": True,
                "formal_quality_semantic_ready": True,
            },
    })
    _write_json(b6_run / "artifacts" / "formal_quality_motion_semantic_decision.json", {
        "formal_quality_motion_semantic_ready": True,
        "formal_metric_claim_status": "ready",
    })

    payload = run_submission_freeze_preparation(
        tmp_path / "submission_freeze_preparation",
        b5_run_root=b5_run,
        b6_run_root=b6_run,
        stage_decision_paths=stage_paths,
    )

    decision = payload["decision"]
    assert decision["mechanism_decision"] == "PASS"
    assert decision["details"]["sstw_t_submission_preparation_status"] == "PASS"
    assert decision["details"]["sstw_tc_submission_freeze_status"] == "DOWNGRADED_TO_EXPLORATORY"
    assert decision["details"]["release_package_rebuildable"] == "PASS"
    assert decision["details"]["package_digest"]
    assert Path(decision["details"]["archive_path"]).exists()
    assert Path(decision["details"]["package_manifest_path"]).exists()
    assert payload["submission_readiness_decision"] == "PASS"

    claim_records = [
        json.loads(line)
        for line in (tmp_path / "submission_freeze_preparation" / "records" / "claim_audit_records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sstw_tc = next(record for record in claim_records if record["claim_id"] == "claim_sstw_tc_submission_freeze")
    assert sstw_tc["claim_status"] == "needs_downgrade"
    assert sstw_tc["claim_scope"] == "exploratory"

    readiness_summary = json.loads(
        (tmp_path / "submission_freeze_preparation" / "artifacts" / "submission_readiness_summary.json").read_text(encoding="utf-8")
    )
    assert readiness_summary["submission_readiness_decision"] == "PASS"
    assert readiness_summary["main_submission_variant"] == "SSTW-T"
    assert readiness_summary["exploratory_variants"] == ["SSTW-TC"]
    assert "submission-freeze evidence" in readiness_summary["claim_boundary_statement"]

    main_tables_manifest = json.loads(
        (tmp_path / "submission_freeze_preparation" / "artifacts" / "submission_main_tables_manifest.json").read_text(encoding="utf-8")
    )
    assert main_tables_manifest["table_rebuild_status"] == "PASS"
    assert main_tables_manifest["main_submission_variant"] == "SSTW-T"
    assert main_tables_manifest["main_claim_row_count"] >= 1
    assert (tmp_path / "submission_freeze_preparation" / "tables" / "submission_stage_evidence_main_table.csv").exists()
    assert (tmp_path / "submission_freeze_preparation" / "tables" / "submission_main_claim_table.csv").exists()
    assert (tmp_path / "submission_freeze_preparation" / "tables" / "submission_exploratory_boundary_table.csv").exists()
