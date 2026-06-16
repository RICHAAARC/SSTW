"""验证 sampling-time weak constraint preflight 的轻量闭环。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.sampling_time_constraint.runner import run
from main.generation.constraint_controller import SamplingConstraintConfig, apply_sampling_constraint
from main.generation.lambda_schedule import active_step_count, build_lambda_schedule
from main.generation.velocity_projection_constraint import directional_alignment, project_velocity_toward_direction
from main.protocol.record_writer import read_jsonl


@pytest.mark.quick
def test_lambda_schedule_prefers_mid_window() -> None:
    """推荐 schedule 必须在中期窗口启用, 且不能在全部采样步强制施加约束。"""
    values = build_lambda_schedule("mid_window_weak_constraint", num_steps=16, lambda_max=0.12, time_window=(0.25, 0.75))

    assert len(values) == 16
    assert active_step_count(values) > 0
    assert active_step_count(values) < 16
    assert values[0] == 0.0
    assert values[-1] == 0.0
    assert 0.10 <= max(values) <= 0.12


@pytest.mark.quick
def test_velocity_projection_improves_directional_alignment() -> None:
    """弱投影算子应提升速度与 key-conditioned 方向的对齐度。"""
    velocity = [-0.02, 0.08, -0.05]
    direction = [1.0, 0.5, 0.25]
    before = directional_alignment(velocity, direction)
    after_velocity = project_velocity_toward_direction(velocity, direction, lambda_value=0.12, norm_budget=0.06)
    after = directional_alignment(after_velocity, direction)

    assert after > before


@pytest.mark.quick
def test_constraint_controller_builds_governed_summary() -> None:
    """约束控制器应返回可写入 governed records 的统计字段。"""
    velocities = [[-0.02, 0.08, -0.05] for _ in range(8)]
    config = SamplingConstraintConfig(
        sampling_constraint_config_id="test_constraint",
        lambda_schedule_id="mid_window_weak_constraint",
        lambda_max=0.12,
        lambda_time_window=(0.25, 0.75),
        constraint_norm_budget=0.06,
        constraint_direction=(1.0, 0.5, 0.25),
    )
    summary = apply_sampling_constraint(velocities, config)

    assert summary["trajectory_constraint_gain"] > 0
    assert summary["constraint_apply_steps"] > 0
    assert summary["S_trajectory_observation_after_constraint"] > summary["S_trajectory_observation_before_constraint"]


@pytest.mark.quick
def test_sampling_time_constraint_preflight_builds_outputs(tmp_path: Path) -> None:
    """B6 preflight runner 必须生成 records、tables、reports 和 decision。"""
    output_root = tmp_path / "sampling_time_constraint_preflight"
    summary = run(output_root)

    assert summary["implementation_decision"] == "PASS"
    assert summary["mechanism_decision"] == "PASS"
    assert summary["audit"]["constraint_main_claim_status"] == "preflight_only_not_final_b6_claim"

    records_path = output_root / "records" / "constraint_records.jsonl"
    decision_path = output_root / "artifacts" / "sampling_time_constraint_preflight_decision.json"
    table_path = output_root / "tables" / "sampling_constraint_ablation_table.csv"
    report_path = output_root / "reports" / "sampling_time_constraint_preflight_report.md"

    assert records_path.exists()
    assert decision_path.exists()
    assert table_path.exists()
    assert report_path.exists()

    records = read_jsonl(records_path)
    assert len(records) == 32
    assert "keyed_state_trajectory_constraint" in {record["method_variant"] for record in records}
    assert all(record["constraint_main_claim_status"] == "preflight_only_not_final_b6_claim" for record in records)

    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert decision["stage_id"] == "sampling_time_constraint_preflight"
    assert decision["details"]["sampling_time_constraint_preflight_decision"] == "PASS"
