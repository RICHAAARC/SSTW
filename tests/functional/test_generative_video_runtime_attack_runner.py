"""验证生成视频 runtime attack runner。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.generative_video_model_probe.attack_runner import run_runtime_attacks
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
    assert all(Path(record["attacked_video_path"]).exists() for record in records)
    assert all(record["attacked_video_sha256"] for record in records)
    assert (run_root / "tables" / "runtime_attack_table.csv").exists()
    assert (run_root / "artifacts" / "runtime_attack_decision.json").exists()
    assert (run_root / "reports" / "runtime_attack_report.md").exists()
