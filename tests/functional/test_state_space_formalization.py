"""验证 state_space_inference_formalization state-space inference formalization 的轻量闭环。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.state_space_formalization.runner import run
from main.protocol.record_writer import read_jsonl


@pytest.mark.quick
def test_state_space_formalization_builds_outputs(tmp_path: Path) -> None:
    """state_space_inference_formalization runner 必须生成 event、state、ablation、generalization 和 decision。"""
    output_root = tmp_path / "state_space_inference_formalization"
    summary = run(output_root)

    assert summary["implementation_decision"] == "PASS"
    assert summary["mechanism_decision"] == "PASS"

    event_path = output_root / "records" / "event_scores.jsonl"
    ablation_path = output_root / "records" / "ablation_records.jsonl"
    generalization_path = output_root / "records" / "generalization_records.jsonl"
    decision_path = output_root / "artifacts" / "state_space_formal_decision.json"
    assert event_path.exists()
    assert ablation_path.exists()
    assert generalization_path.exists()
    assert decision_path.exists()

    records = read_jsonl(event_path)
    assert records
    assert all(record["trajectory_enabled"] is False for record in records)
    assert all(record["trajectory_status"] == "EXPLICIT" for record in records)
    assert all("formal_state_schema_version" in record for record in records)

    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert decision["details"]["state_space_inference_formal_decision"] == "PASS"
    assert decision["details"]["state_variable_ablation_all_nontrivial"] == "PASS"


@pytest.mark.quick
def test_state_space_formalization_core_mechanism_gates(tmp_path: Path) -> None:
    """state_space_inference_formalization 机制审计必须证明 key condition、admissibility 和泛化 gate。"""
    output_root = tmp_path / "state_space_inference_formalization"
    summary = run(output_root)
    audit = summary["audit"]

    assert audit["key_condition_ablation_gain"] > 0
    assert audit["admissibility_negative_tail_status"] == "PASS"
    assert audit["unseen_key_generalization_status"] == "PASS"
    assert audit["unseen_attack_generalization_status"] == "PASS"
    assert audit["negative_state_over_threshold_count"] == 0
