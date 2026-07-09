"""验证 real_video_latent_transfer_check real video latent transfer check 的轻量闭环。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.real_video_latent_transfer.runner import run
from main.protocol.record_writer import read_jsonl


@pytest.mark.quick
def test_real_video_latent_transfer_builds_outputs(tmp_path: Path) -> None:
    """real_video_latent_transfer_check runner 必须生成 records、quality records、thresholds、tables 和 decision。"""
    output_root = tmp_path / "real_video_latent_transfer_check"
    summary = run(output_root)

    assert summary["implementation_decision"] == "PASS"
    assert summary["mechanism_decision"] == "PASS"

    event_path = output_root / "records" / "event_scores.jsonl"
    quality_path = output_root / "records" / "quality_metrics.jsonl"
    decision_path = output_root / "artifacts" / "real_video_latent_transfer_decision.json"
    assert event_path.exists()
    assert quality_path.exists()
    assert decision_path.exists()

    records = read_jsonl(event_path)
    assert records
    assert all(record["threshold_source_split"] == "calibration" for record in records)
    assert all("source_video_id" in record for record in records)
    assert all("vae_backend_id" in record for record in records)

    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert decision["details"]["quality_not_collapsed"] == "PASS"
    assert decision["details"]["temporal_consistency_not_collapsed"] == "PASS"


@pytest.mark.quick
def test_real_video_latent_transfer_preserves_low_fpr(tmp_path: Path) -> None:
    """real_video_latent_transfer_check 机制审计必须保持负样本安全。"""
    output_root = tmp_path / "real_video_latent_transfer_check"
    summary = run(output_root)
    assert summary["audit"]["attacked_negative_fpr"] == 0.0
    assert summary["audit"]["negative_state_over_threshold_count"] == 0
    assert summary["audit"]["state_beats_explicit_non_uniform_attack_count"] >= 1
