"""验证 trajectory_observation_core_probe trajectory observation core probe 的轻量闭环。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.trajectory_observation_core.runner import run
from main.protocol.record_writer import read_jsonl


@pytest.mark.quick
def test_trajectory_observation_core_builds_outputs(tmp_path: Path) -> None:
    """trajectory_observation_core_probe runner 必须生成 trajectory records、control records 和 decision。"""
    output_root = tmp_path / "trajectory_observation_core_probe"
    summary = run(output_root)

    assert summary["implementation_decision"] == "PASS"
    assert summary["mechanism_decision"] == "PASS"

    event_path = output_root / "records" / "event_scores.jsonl"
    trajectory_path = output_root / "records" / "trajectory_trace.jsonl"
    control_path = output_root / "records" / "trajectory_control_records.jsonl"
    decision_path = output_root / "artifacts" / "trajectory_observation_decision.json"
    assert event_path.exists()
    assert trajectory_path.exists()
    assert control_path.exists()
    assert decision_path.exists()

    records = read_jsonl(event_path)
    assert records
    assert "key_conditioned_state_space_with_trajectory" in {record["method_variant"] for record in records}
    assert all(record["trajectory_source_status"] for record in records)

    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert decision["details"]["trajectory_observation_mechanism_decision"] == "PASS"
    assert decision["details"]["top_conference_trajectory_gate"] == "PASS"


@pytest.mark.quick
def test_trajectory_observation_core_mechanism_gates(tmp_path: Path) -> None:
    """trajectory_observation_core_probe 机制审计必须证明 trajectory gain、非冗余和 control 抑制。"""
    output_root = tmp_path / "trajectory_observation_core_probe"
    summary = run(output_root)
    audit = summary["audit"]

    assert audit["trajectory_gain_over_state_space"] > 0
    assert audit["trajectory_negative_leakage_delta"] <= 0
    assert audit["correlation_status"] == "PASS"
    assert audit["control_suppression_status"] == "PASS"
    assert audit["runtime_overhead_status"] == "PASS"
