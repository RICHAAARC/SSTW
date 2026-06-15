"""验证第一阶段 synthetic state protocol 的轻量闭环。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.synthetic_state_inference.runner import run
from main.protocol.record_writer import read_jsonl


@pytest.mark.quick
def test_synthetic_state_protocol_builds_rebuildable_artifacts(tmp_path: Path) -> None:
    """第一阶段 runner 必须在临时目录生成 records、thresholds、tables 和 decision。"""
    output_root = tmp_path / "synthetic_state_protocol"
    summary = run(output_root)

    assert summary["implementation_decision"] == "PASS"
    assert summary["mechanism_decision"] == "PASS"
    event_path = output_root / "records" / "event_scores.jsonl"
    threshold_path = output_root / "thresholds" / "thresholds.json"
    table_path = output_root / "tables" / "synthetic_state_main_table.csv"
    decision_path = output_root / "artifacts" / "synthetic_state_inference_decision.json"

    assert event_path.exists()
    assert threshold_path.exists()
    assert table_path.exists()
    assert decision_path.exists()

    records = read_jsonl(event_path)
    assert records
    assert {record["threshold_source_split"] for record in records} == {"calibration"}
    assert all(record["test_time_threshold_update_blocked"] for record in records)
    assert all("S_trajectory_observation_placeholder" in record for record in records)

    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert decision["details"]["negative_state_over_threshold_count"] == 0


@pytest.mark.quick
def test_synthetic_state_protocol_covers_required_methods_and_attacks(tmp_path: Path) -> None:
    """第一阶段 records 必须覆盖构建文档要求的方法变体和攻击矩阵。"""
    output_root = tmp_path / "synthetic_state_protocol"
    run(output_root)
    records = read_jsonl(output_root / "records" / "event_scores.jsonl")

    method_variants = {record["method_variant"] for record in records}
    assert "frame_prc" in method_variants
    assert "tubelet_only" in method_variants
    assert "key_conditioned_state_space_inference" in method_variants
    assert "key_agnostic_state_space_model" in method_variants

    attack_names = {record["attack_name"] for record in records}
    assert "no_attack" in attack_names
    assert "temporal_crop" in attack_names
    assert "irregular_frame_dropping" in attack_names
    assert "segment_jump" in attack_names
