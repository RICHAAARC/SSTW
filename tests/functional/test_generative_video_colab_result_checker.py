"""验证 B5 Colab 结果检查器。"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from main.protocol.record_writer import write_json, write_jsonl
from scripts.check_results.generative_video_colab_result_checker import check_generative_video_colab_results


@pytest.mark.quick
def test_generative_video_colab_result_checker_distinguishes_evidence_levels(tmp_path: Path) -> None:
    """检查器必须把生成链路成功与机制证据不足区分开。"""
    run_root = tmp_path / "generative_video_model_probe_colab"
    video_path = run_root / "videos" / "sample.mp4"
    video_path.parent.mkdir(parents=True)
    video_path.write_bytes(b"fake mp4 payload")
    digest = hashlib.sha256(video_path.read_bytes()).hexdigest()

    write_jsonl(run_root / "records" / "generation_records.jsonl", [{
        "generation_model_id": "model",
        "prompt_id": "prompt",
        "seed_id": "seed",
        "generation_status": "success",
        "video_path": str(video_path),
        "video_sha256": digest,
        "trajectory_trace_id": "trace_0000",
    }])
    write_jsonl(run_root / "records" / "trajectory_trace.jsonl", [{
        "trajectory_trace_id": "trace_0000",
        "trajectory_step_index": 0,
        "latent_norm": 1.0,
    }])
    write_jsonl(run_root / "records" / "quality_motion_semantic_records.jsonl", [{
        "quality_metric_status": "not_run",
    }])
    write_jsonl(run_root / "records" / "external_baseline_records.jsonl", [{
        "external_baseline_runnable_status": "not_runnable",
    }])
    write_json(run_root / "artifacts" / "generation_manifest.json", {"artifact_id": "manifest"})
    write_json(run_root / "artifacts" / "generative_video_colab_runtime_decision.json", {
        "stage_id": "generative_video_model_probe_colab_runtime",
        "implementation_decision": "PASS",
        "mechanism_decision": "FAIL",
        "details": {
            "fixed_low_fpr_audit_pass": False,
            "trajectory_observation_gain_confirmed": False,
            "quality_motion_semantic_consistency_pass": False,
        },
    })

    summary = check_generative_video_colab_results(run_root)
    assert summary["implementation_evidence_status"] == "PASS"
    assert summary["mechanism_evidence_status"] == "FAIL"
    assert "fixed_low_fpr_audit_not_passed" in summary["missing_mechanism_requirements"]
    assert "external_baseline_not_runnable" in summary["missing_mechanism_requirements"]
    assert summary["video_checks"][0]["video_sha256_match"] is True
