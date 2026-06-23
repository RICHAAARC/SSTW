from pathlib import Path

import pytest

from experiments.generative_video_model_probe.adaptive_attack_runner import (
    ADAPTIVE_ATTACK_SPECS,
    run_adaptive_attack_validation_proxy,
)
from main.protocol.record_writer import read_jsonl, write_jsonl


@pytest.mark.quick
def test_adaptive_attack_runner_writes_validation_proxy_records(tmp_path: Path) -> None:
    """adaptive attack runner 必须从 runtime detection records 写出 validation proxy 记录。"""
    run_root = tmp_path / "run"
    write_jsonl(run_root / "records" / "runtime_detection_records.jsonl", [
        {
            "runtime_detection_status": "ready",
            "generation_model_id": "model",
            "prompt_id": "prompt_a",
            "seed_id": "seed_a",
            "trajectory_trace_id": "trace_a",
            "attack_name": "video_compression_runtime",
            "S_final_conservative": 0.72,
        }
    ])

    audit = run_adaptive_attack_validation_proxy(run_root)
    records = read_jsonl(run_root / "records" / "adaptive_attack_records.jsonl")

    assert audit["adaptive_attack_decision"] == "PASS"
    assert audit["adaptive_attack_record_count"] == len(ADAPTIVE_ATTACK_SPECS)
    assert audit["adaptive_robustness_claim_allowed"] is False
    assert all(record["claim_support_status"] == "validation_adaptive_attack_proxy_only" for record in records)
    assert any(record["adaptive_attack_name"] == "path_response_cancellation" for record in records)
    assert any(record["attack_knowledge_level"] == "white_box_oracle_limited_flow_attacker" for record in records)
    assert all(record["adaptive_negative_fpr"] is None for record in records)
    assert (run_root / "tables" / "adaptive_attack_table.csv").exists()
    assert (run_root / "artifacts" / "adaptive_attack_decision.json").exists()
    assert (run_root / "reports" / "adaptive_attack_report.md").exists()


@pytest.mark.quick
def test_adaptive_attack_runner_blocks_when_runtime_detection_missing(tmp_path: Path) -> None:
    """缺少 runtime detection records 时, adaptive attack runner 只能报告阻断。"""
    run_root = tmp_path / "run"

    audit = run_adaptive_attack_validation_proxy(run_root)
    records = read_jsonl(run_root / "records" / "adaptive_attack_records.jsonl")

    assert audit["adaptive_attack_decision"] == "FAIL"
    assert audit["claim_support_status"] == "validation_adaptive_attack_blocked"
    assert records == []
