"""验证 generative_video_model_probe generative video model probe readiness 框架。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.generative_video_model_probe.runner import run
from main.protocol.record_writer import read_jsonl


@pytest.mark.quick
def test_generative_video_model_probe_builds_blocked_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """generative_video_model_probe runner 在显式模拟无 GPU 环境时必须生成可审计 blocked decision。"""
    monkeypatch.setenv("PATH", "")
    output_root = tmp_path / "generative_video_model_probe"
    summary = run(output_root)

    assert summary["generation_record_count"] > 0
    assert summary["event_record_count"] > 0
    assert summary["implementation_decision"] == "FAIL"
    assert summary["mechanism_decision"] == "FAIL"

    generation_path = output_root / "records" / "generation_records.jsonl"
    event_path = output_root / "records" / "event_scores.jsonl"
    quality_path = output_root / "records" / "quality_motion_semantic_records.jsonl"
    external_path = output_root / "records" / "external_baseline_records.jsonl"
    decision_path = output_root / "artifacts" / "generative_video_model_decision.json"
    manifest_path = output_root / "artifacts" / "generation_manifest.json"
    assert generation_path.exists()
    assert event_path.exists()
    assert quality_path.exists()
    assert external_path.exists()
    assert decision_path.exists()
    assert manifest_path.exists()

    records = read_jsonl(generation_path)
    assert all(record["generation_model_id"] for record in records)
    assert all(record["prompt_id"] for record in records)
    assert all(record["generation_model_runnable_status"] in {"not_runnable", "runnable"} for record in records)

    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert decision["details"]["formal_claim_status"] == "blocked_until_gpu_generation_run"
    assert decision["details"]["top_conference_generative_video_model_probe_gate"] == "FAIL"


@pytest.mark.quick
def test_generative_video_model_probe_does_not_support_positive_claim_without_runtime(tmp_path: Path) -> None:
    """无真实生成运行时, generative_video_model_probe 不得声明 fixed-FPR、trajectory gain 或质量一致性通过。"""
    output_root = tmp_path / "generative_video_model_probe"
    summary = run(output_root)
    audit = summary["audit"]

    assert audit["generation_model_main_table_ready"] is False
    assert audit["trajectory_observation_gain_confirmed"] is False
    assert audit["fixed_low_fpr_audit_pass"] is False
    assert audit["quality_motion_semantic_consistency_pass"] is False
    assert audit["cross_prompt_seed_generalization_pass"] is False
