"""验证 B5 Colab 机制后处理流程。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.generative_video_model_probe.postprocess_runner import postprocess_colab_run
from main.protocol.record_writer import write_json, write_jsonl
from scripts.check_results.generative_video_colab_result_checker import check_generative_video_colab_results


@pytest.mark.quick
def test_generative_video_postprocess_builds_proxy_records(tmp_path: Path) -> None:
    """后处理必须从已有 Colab records 重建 proxy records, 且不得直接声明正式机制 claim。"""
    run_root = tmp_path / "generative_video_runtime"
    video_path = run_root / "videos" / "sample.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"fake video")

    generation_records = []
    trajectory_records = []
    for index in range(4):
        trace_id = f"trace_{index:04d}"
        generation_records.append({
            "generation_model_id": "model",
            "prompt_id": f"prompt_{index // 2}",
            "seed_id": f"seed_{index % 2}",
            "generation_status": "success",
            "video_path": str(video_path),
            "video_sha256": "digest",
            "trajectory_trace_id": trace_id,
        })
        for step_index in range(4):
            trajectory_records.append({
                "trajectory_trace_id": trace_id,
                "trajectory_step_index": step_index,
                "latent_norm": 100.0 - 8.0 * step_index,
                "latent_mean": 0.01 * step_index,
                "latent_std": 0.9 - 0.05 * step_index,
            })

    write_jsonl(run_root / "records" / "generation_records.jsonl", generation_records)
    write_jsonl(run_root / "records" / "trajectory_trace.jsonl", trajectory_records)
    write_jsonl(run_root / "records" / "external_baseline_records.jsonl", [{"external_baseline_runnable_status": "runnable"}])
    write_json(run_root / "artifacts" / "generation_manifest.json", {"artifact_id": "manifest"})
    write_json(run_root / "artifacts" / "generative_video_colab_runtime_decision.json", {
        "stage_id": "generative_video_generation",
        "implementation_decision": "PASS",
        "mechanism_decision": "FAIL",
        "details": {
            "fixed_low_fpr_audit_pass": False,
            "trajectory_observation_gain_confirmed": False,
            "quality_motion_semantic_consistency_pass": False,
        },
    })

    summary = postprocess_colab_run(run_root)
    validation_protocol = json.loads(Path("configs/protocol/validation_scale_generative_probe.json").read_text(encoding="utf-8"))

    assert summary["mechanism_score_record_count"] == 16
    assert summary["controlled_negative_record_count"] == 12
    assert summary["quality_proxy_record_count"] == 4
    assert summary["target_fpr"] == validation_protocol["target_fpr"]
    assert summary["mechanism_postprocess_decision"] == "PASS"
    assert summary["mechanism_decision"] == "FAIL"
    assert summary["formal_claim_status"] == "blocked_until_formal_quality_motion_semantic_metrics"

    decision = json.loads((run_root / "artifacts" / "generative_video_mechanism_postprocess_decision.json").read_text(encoding="utf-8"))
    assert decision["details"]["fixed_low_fpr_proxy_pass"] is True
    assert decision["details"]["trajectory_gain_confirmed_by_proxy"] is True
    assert (run_root / "tables" / "mechanism_proxy_comparison_table.csv").exists()


@pytest.mark.quick
def test_colab_checker_reports_postprocess_progress_without_formal_claim(tmp_path: Path) -> None:
    """结果检查器应识别后处理进展, 但正式机制证据仍需 formal quality motion semantic metrics。"""
    run_root = tmp_path / "generative_video_runtime"
    video_path = run_root / "videos" / "sample.mp4"
    video_path.parent.mkdir(parents=True)
    payload = b"fake video"
    video_path.write_bytes(payload)
    import hashlib

    digest = hashlib.sha256(payload).hexdigest()
    write_jsonl(run_root / "records" / "generation_records.jsonl", [{
        "generation_model_id": "model",
        "prompt_id": "prompt",
        "seed_id": "seed",
        "generation_status": "success",
        "video_path": str(video_path),
        "video_sha256": digest,
        "trajectory_trace_id": "trace_0000",
    } for _ in range(4)])
    write_jsonl(run_root / "records" / "trajectory_trace.jsonl", [{
        "trajectory_trace_id": "trace_0000",
        "trajectory_step_index": index,
        "latent_norm": 100.0 - 8.0 * index,
        "latent_mean": 0.01 * index,
        "latent_std": 0.9 - 0.05 * index,
    } for index in range(4)])
    write_jsonl(run_root / "records" / "external_baseline_records.jsonl", [{"external_baseline_runnable_status": "runnable"}])
    write_json(run_root / "artifacts" / "generation_manifest.json", {"artifact_id": "manifest"})
    write_json(run_root / "artifacts" / "generative_video_colab_runtime_decision.json", {
        "stage_id": "generative_video_generation",
        "implementation_decision": "PASS",
        "mechanism_decision": "FAIL",
        "details": {},
    })
    postprocess_colab_run(run_root)

    summary = check_generative_video_colab_results(run_root)
    assert summary["mechanism_postprocess_status"] == "PASS"
    assert summary["mechanism_evidence_status"] == "FAIL"
    assert "formal_quality_motion_semantic_metrics_missing" in summary["missing_mechanism_requirements"]
