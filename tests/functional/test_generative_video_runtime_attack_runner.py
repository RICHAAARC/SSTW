"""验证生成视频 runtime attack runner。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.generative_video_model_probe.attack_runner import run_runtime_attacks
from experiments.generative_video_model_probe.detection_runner import run_runtime_detection
from main.protocol.record_writer import write_jsonl


def _write_tiny_video(path: Path) -> None:
    """写出一个极小 mp4, 用于验证 runtime attack 文件级链路。"""
    import imageio.v3 as iio
    import numpy as np

    frames = []
    for index in range(6):
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        frame[:, :, 0] = 40 + index * 8
        frame[8:24, 8 + index:16 + index, 1] = 180
        frames.append(frame)
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, frames, fps=4)


@pytest.mark.quick
def test_runtime_attack_runner_writes_attacked_videos_and_records(tmp_path: Path) -> None:
    """runtime attack runner 必须对真实 mp4 生成 attacked video 与 governed records。"""
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

    summary = run_runtime_attacks(run_root)
    records = [json.loads(line) for line in (run_root / "records" / "runtime_attack_records.jsonl").read_text(encoding="utf-8").splitlines()]

    assert summary["runtime_attack_decision"] == "PASS"
    assert summary["runtime_attack_record_count"] == 3
    assert summary["runtime_attack_ready_count"] == 3
    assert summary["runtime_attack_count"] == 3
    assert len(records) == 3
    assert all(record["attack_runtime_status"] == "ready" for record in records)
    assert all(record["attack_matrix_evidence_level"] == "runtime_video_file" for record in records)
    assert all("sampler_signature_placeholder" in record for record in records)
    assert all(record["trajectory_source_level"] == "runtime_video_file_attack" for record in records)
    assert all(record["flow_state_admissibility_status"] == "not_evaluated" for record in records)
    assert all(Path(record["attacked_video_path"]).exists() for record in records)
    assert all(record["attacked_video_sha256"] for record in records)
    assert (run_root / "tables" / "runtime_attack_table.csv").exists()
    assert (run_root / "artifacts" / "runtime_attack_decision.json").exists()
    assert (run_root / "reports" / "runtime_attack_report.md").exists()


@pytest.mark.quick
def test_runtime_detection_runner_scores_attacked_videos(tmp_path: Path) -> None:
    """runtime detection runner 必须把 attacked videos 接入 governed detection records。"""
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
    write_jsonl(run_root / "records" / "trajectory_trace.jsonl", [
        {"trajectory_trace_id": "trace_0000", "trajectory_step_index": 0, "latent_norm": 100.0, "latent_std": 0.8},
        {"trajectory_trace_id": "trace_0000", "trajectory_step_index": 1, "latent_norm": 85.0, "latent_std": 0.7},
        {"trajectory_trace_id": "trace_0000", "trajectory_step_index": 2, "latent_norm": 70.0, "latent_std": 0.6},
    ])

    attack_summary = run_runtime_attacks(run_root)
    detection_summary = run_runtime_detection(run_root)
    records = [json.loads(line) for line in (run_root / "records" / "runtime_detection_records.jsonl").read_text(encoding="utf-8").splitlines()]

    assert attack_summary["runtime_attack_decision"] == "PASS"
    assert detection_summary["runtime_detection_decision"] == "PASS"
    assert detection_summary["runtime_detection_record_count"] == 3
    assert detection_summary["runtime_detection_ready_count"] == 3
    assert detection_summary["runtime_detection_detectable_count"] == 3
    assert len(records) == 3
    assert all(record["runtime_detection_status"] == "ready" for record in records)
    assert all(record["runtime_detection_evidence_level"] == "runtime_attacked_video_file" for record in records)
    assert all(record["S_runtime_attack_detection"] is not None for record in records)
    assert all(record["S_path_inv"] is not None for record in records)
    assert all(record["S_velocity"] is not None for record in records)
    assert all(record["S_final_conservative"] is not None for record in records)
    assert all(record["flow_state_admissibility_status"] == "proxy_admissible" for record in records)
    assert (run_root / "tables" / "runtime_detection_table.csv").exists()
    assert (run_root / "artifacts" / "runtime_detection_decision.json").exists()
    assert (run_root / "reports" / "runtime_detection_report.md").exists()
