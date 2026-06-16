"""验证 B5 生成视频正式质量与运动 metric runner。"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from experiments.generative_video_model_probe.formal_metric_runner import run_formal_metric_audit
from experiments.generative_video_model_probe.postprocess_runner import postprocess_colab_run
from main.protocol.record_writer import read_jsonl, write_json, write_jsonl
from scripts.check_results.generative_video_colab_result_checker import check_generative_video_colab_results


def _write_tiny_video(path: Path) -> None:
    """写出一个极小 mp4, 用于 quick 测试真实视频解码链路。"""
    import imageio.v3 as iio
    import numpy as np

    frames = []
    for index in range(6):
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        frame[:, :, 0] = 40 + index * 10
        frame[8:24, 8 + index:16 + index, 1] = 180
        frames.append(frame)
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, frames, fps=4)


@pytest.mark.quick
def test_formal_metric_runner_builds_video_file_metrics(tmp_path: Path) -> None:
    """formal metric runner 必须读取真实 mp4 并记录语义 metric 未配置。"""
    run_root = tmp_path / "generative_video_model_probe_colab"
    video_path = run_root / "videos" / "tiny.mp4"
    _write_tiny_video(video_path)
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

    audit = run_formal_metric_audit(run_root)
    records = read_jsonl(run_root / "records" / "formal_quality_motion_semantic_records.jsonl")

    assert audit["formal_metric_record_count"] == 1
    assert audit["formal_visual_motion_ready"] is True
    assert audit["formal_semantic_ready"] is False
    assert audit["formal_metric_claim_status"] == "blocked_until_semantic_metric_configured"
    assert records[0]["video_decode_status"] == "ready"
    assert records[0]["formal_visual_quality_ready"] is True
    assert records[0]["formal_motion_consistency_ready"] is True
    assert records[0]["formal_semantic_consistency_ready"] is False


@pytest.mark.quick
def test_checker_reports_semantic_only_block_after_formal_visual_motion_metrics(tmp_path: Path) -> None:
    """补齐正式质量/运动 metric 后, checker 应只保留正式语义 metric 阻断。"""
    run_root = tmp_path / "generative_video_model_probe_colab"
    video_path = run_root / "videos" / "tiny.mp4"
    _write_tiny_video(video_path)
    digest = hashlib.sha256(video_path.read_bytes()).hexdigest()
    write_jsonl(run_root / "records" / "generation_records.jsonl", [{
        "generation_model_id": "model",
        "prompt_id": f"prompt_{index // 2}",
        "seed_id": f"seed_{index % 2}",
        "generation_status": "success",
        "video_path": str(video_path),
        "video_sha256": digest,
        "trajectory_trace_id": f"trace_{index:04d}",
    } for index in range(4)])
    write_jsonl(run_root / "records" / "trajectory_trace.jsonl", [{
        "trajectory_trace_id": f"trace_{trace_index:04d}",
        "trajectory_step_index": step_index,
        "latent_norm": 100.0 - 8.0 * step_index,
        "latent_mean": 0.01 * step_index,
        "latent_std": 0.9 - 0.05 * step_index,
    } for trace_index in range(4) for step_index in range(4)])
    write_jsonl(run_root / "records" / "external_baseline_records.jsonl", [{"external_baseline_runnable_status": "runnable"}])
    write_json(run_root / "artifacts" / "generation_manifest.json", {"artifact_id": "manifest"})
    write_json(run_root / "artifacts" / "generative_video_colab_runtime_decision.json", {
        "stage_id": "generative_video_model_probe_colab_runtime",
        "implementation_decision": "PASS",
        "mechanism_decision": "FAIL",
        "details": {},
    })

    run_formal_metric_audit(run_root)
    postprocess_colab_run(run_root)
    summary = check_generative_video_colab_results(run_root)

    assert summary["mechanism_postprocess_status"] == "PASS"
    assert summary["formal_visual_motion_ready_count"] == 4
    assert summary["formal_semantic_ready_count"] == 0
    assert summary["missing_mechanism_requirements"] == ["formal_semantic_metric_missing"]
