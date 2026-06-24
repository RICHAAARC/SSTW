from pathlib import Path

import pytest

from experiments.generative_video_model_probe.claim3_downgrade import write_claim3_downgrade_outputs
from experiments.generative_video_model_probe.replay_and_sketch_gate import run_replay_and_sketch_gate
from main.protocol.record_writer import read_jsonl, write_jsonl


@pytest.mark.quick
def test_replay_and_sketch_gate_writes_validation_proxy_records(tmp_path: Path) -> None:
    """replay/sketch gate 必须从轨迹 records 写出四类 governed records, 但不伪装成 full-paper 强 Claim-3。"""
    run_root = tmp_path / "run"
    generation_records = []
    trajectory_records = []
    for prompt_index in range(2):
        for seed_index in range(2):
            trace_id = f"trace_{prompt_index}_{seed_index}"
            generation_records.append({
                "generation_status": "success",
                "colab_runtime_profile": "validation_scale",
                "generation_model_id": "model",
                "prompt_id": f"prompt_{prompt_index}",
                "seed_id": f"seed_{seed_index}",
                "trajectory_trace_id": trace_id,
                "sampler_signature_placeholder": "sampler_placeholder",
            })
            for step_index in range(3):
                trajectory_records.append({
                    "trajectory_trace_id": trace_id,
                    "trajectory_step_index": step_index,
                    "latent_norm": 10.0 - step_index,
                    "latent_mean": 0.01 * step_index,
                    "latent_std": 0.9 - 0.1 * step_index,
                })
    write_jsonl(run_root / "records" / "generation_records.jsonl", generation_records)
    write_jsonl(run_root / "records" / "trajectory_trace.jsonl", trajectory_records)

    audit = run_replay_and_sketch_gate(run_root)
    sketch_records = read_jsonl(run_root / "records" / "trajectory_sketch_verification_records.jsonl")
    uncertainty_records = read_jsonl(run_root / "records" / "replay_uncertainty_records.jsonl")
    wrong_sampler_records = read_jsonl(run_root / "records" / "wrong_sampler_replay_records.jsonl")
    wrong_prompt_records = read_jsonl(run_root / "records" / "wrong_prompt_replay_records.jsonl")

    assert audit["replay_and_sketch_gate_decision"] == "PASS"
    assert audit["replay_and_sketch_evidence_level"] == "validation_runtime_trace_proxy"
    assert audit["claim3_full_support_allowed"] is False
    assert audit["trajectory_sketch_verified_count"] == 4
    assert audit["replay_uncertainty_ready_count"] == 4
    assert audit["wrong_sampler_replay_rejected_count"] == 4
    assert audit["wrong_prompt_replay_rejected_count"] == 4
    assert len(sketch_records) == 4
    assert len(uncertainty_records) == 4
    assert len(wrong_sampler_records) == 4
    assert len(wrong_prompt_records) == 4
    assert sketch_records[0]["trajectory_sketch_digest_random"]
    assert sketch_records[0]["trajectory_sketch_verification_status"] == "verified"
    assert uncertainty_records[0]["replay_uncertainty_weight"] is not None
    assert wrong_sampler_records[0]["replay_control_status"] == "replay_rejected"
    assert wrong_prompt_records[0]["replay_control_status"] == "replay_rejected"
    assert (run_root / "tables" / "replay_verification_table.csv").exists()
    assert (run_root / "artifacts" / "replay_and_sketch_gate_decision.json").exists()
    assert (run_root / "reports" / "replay_and_sketch_gate_report.md").exists()

    claim3_audit = write_claim3_downgrade_outputs(run_root)
    assert claim3_audit["claim3_downgraded"] is True
    assert claim3_audit["claim3_full_support_allowed"] is False
    assert claim3_audit["replay_or_sketch_status"] == "replay_and_sketch_gate_passed_validation_proxy"


@pytest.mark.quick
def test_replay_and_sketch_gate_blocks_missing_trace_records(tmp_path: Path) -> None:
    """缺少 trajectory trace 时 replay/sketch gate 必须失败, 不能凭 generation records 伪造 sketch。"""
    run_root = tmp_path / "run"
    write_jsonl(run_root / "records" / "generation_records.jsonl", [{
        "generation_status": "success",
        "colab_runtime_profile": "validation_scale",
        "generation_model_id": "model",
        "prompt_id": "prompt_a",
        "seed_id": "seed_a",
        "trajectory_trace_id": "trace_a",
    }])

    audit = run_replay_and_sketch_gate(run_root)

    assert audit["replay_and_sketch_gate_decision"] == "FAIL"
    assert "authenticated_trajectory_sketch_records_ready" in audit["replay_and_sketch_missing_requirements"]
    assert audit["claim3_full_support_allowed"] is False
