"""验证 B5 external baseline 推荐与 claim 约束。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from main.external_baselines.baseline_registry import audit_external_baseline_records, build_external_baseline_records
from main.external_baselines.explicit_dtw_temporal_alignment import compute_dtw_alignment_cost
from main.external_baselines.frame_matching_temporal_registration import compute_registration_cost, match_frames
from experiments.generative_video_model_probe.external_baseline_runner import write_external_baseline_comparison_outputs, write_external_baseline_status_outputs
from main.protocol.record_writer import read_jsonl, write_jsonl


@pytest.mark.quick
def test_external_baseline_selection_keeps_modern_non_run_records() -> None:
    """外部 baseline 必须同时保留显式同步 control 和现代视频水印 non-run 记录。"""
    config_path = Path("configs/external_baselines/external_baselines.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    records = build_external_baseline_records(config_path)

    names = [record["external_baseline_name"] for record in records]
    assert "explicit_dtw_temporal_alignment" in names
    assert "explicit_frame_matching_temporal_registration" in names
    assert {"videoshield", "sigmark", "spdmark", "videomark_or_vidsig", "videoseal"} <= set(names)
    excluded_related_work_names = {"riva" + "gan", "vid" + "stamp"}
    assert excluded_related_work_names.isdisjoint(names)
    assert config["selection_policy"]["claim_rule"]
    assert "key_conditioned_state_space_inference" in config["internal_mechanism_baselines"]
    assert all(record["external_baseline_result_used_for_claim"] is False for record in records)

    explicit_records = [record for record in records if record["external_baseline_layer"] == "explicit_synchronization_control"]
    modern_records = [record for record in records if record["external_baseline_layer"] == "modern_external_baseline"]
    assert len(explicit_records) == 2
    assert len(modern_records) >= 5
    assert all(record["external_baseline_runnable_status"] == "runnable" for record in explicit_records)
    assert all(record["external_baseline_runnable_status"] == "not_runnable" for record in modern_records)
    assert all(record["external_baseline_claim_support_status"] == "governed_non_run_record_only" for record in modern_records)


@pytest.mark.quick
def test_external_baseline_status_audit_reports_modern_gap() -> None:
    """现代 baseline 已有 governed 状态记录, 但尚未达到主表比较 ready。"""
    records = build_external_baseline_records("configs/external_baselines/external_baselines.json")
    audit = audit_external_baseline_records(records)

    assert audit["external_baseline_status_decision"] == "PASS"
    assert audit["modern_external_baseline_status_records_ready"] is True
    assert audit["modern_external_baseline_record_count"] >= 5
    assert audit["modern_external_baseline_main_comparison_ready_count"] == 0
    assert audit["external_baseline_claim_support_status"] == "governed_status_records_only"


@pytest.mark.quick
def test_external_baseline_runner_writes_governed_status_outputs(tmp_path: Path) -> None:
    """外部 baseline runner 必须写出 records、table、decision 和 report。"""
    run_root = tmp_path / "generative_video_model_probe_colab"
    audit = write_external_baseline_status_outputs(run_root)
    records = read_jsonl(run_root / "records" / "external_baseline_records.jsonl")

    assert audit["external_baseline_status_decision"] == "PASS"
    assert len(records) == audit["external_baseline_record_count"]
    assert all("external_baseline_adapter_status" in record for record in records)
    assert all("claim_support_status" in record for record in records)
    assert (run_root / "tables" / "external_baseline_status_table.csv").exists()
    assert (run_root / "artifacts" / "external_baseline_status_decision.json").exists()
    assert (run_root / "reports" / "external_baseline_status_report.md").exists()


@pytest.mark.quick
def test_explicit_synchronization_adapters_run_on_small_sequences() -> None:
    """两个 external synchronization control adapter 必须能在轻量 embedding 序列上运行。"""
    reference = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
    observed = [[0.0, 0.0], [1.1, 0.0], [2.0, 0.0]]

    assert compute_dtw_alignment_cost(reference, observed) >= 0.0
    assert compute_registration_cost(reference, observed) >= 0.0

    matches = match_frames(reference, observed)
    assert [item["reference_index"] for item in matches] == [0, 1, 2]



def _write_external_baseline_runtime_fixture(run_root: Path) -> None:
    """写出 external_baseline adapter 可消费的最小 runtime detection 与 trajectory fixture。"""
    trajectory_records = []
    for step_index in range(4):
        trajectory_records.append({
            "trajectory_trace_id": "trace_0",
            "trajectory_step_index": step_index,
            "latent_norm": 4.0 - step_index * 0.4,
            "latent_mean": 0.1 * step_index,
            "latent_std": 0.2 + step_index * 0.05,
        })
    write_jsonl(run_root / "records" / "trajectory_trace.jsonl", trajectory_records)
    write_jsonl(run_root / "records" / "runtime_detection_records.jsonl", [
        {
            "runtime_detection_status": "ready",
            "generation_model_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            "prompt_id": "prompt_0",
            "seed_id": "seed_0",
            "trajectory_trace_id": "trace_0",
            "attack_name": "video_compression_runtime",
            "source_frame_count": 4,
            "attacked_frame_count": 4,
            "attacked_video_decoded_frame_count": 4,
            "S_runtime_attack_detection": 0.82,
        },
        {
            "runtime_detection_status": "ready",
            "generation_model_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            "prompt_id": "prompt_0",
            "seed_id": "seed_0",
            "trajectory_trace_id": "trace_0",
            "attack_name": "frame_rate_resampling_runtime",
            "source_frame_count": 4,
            "attacked_frame_count": 2,
            "attacked_video_decoded_frame_count": 2,
            "S_runtime_attack_detection": 0.71,
        },
    ])


@pytest.mark.quick
def test_external_baseline_comparison_runner_uses_external_baseline_adapters(tmp_path: Path) -> None:
    """baseline comparison 必须通过 external_baseline/ adapter 产出 records、table、decision 和 report。"""
    run_root = tmp_path / "generative_video_model_probe_colab"
    _write_external_baseline_runtime_fixture(run_root)

    audit = write_external_baseline_comparison_outputs(run_root)
    records = read_jsonl(run_root / "records" / "external_baseline_score_records.jsonl")

    assert audit["external_baseline_comparison_decision"] == "PASS"
    assert audit["external_baseline_measured_adapter_count"] == 2
    assert "explicit_dtw_temporal_alignment" in audit["external_baseline_measured_adapter_names"]
    assert "explicit_frame_matching_temporal_registration" in audit["external_baseline_measured_adapter_names"]
    assert any(record["external_baseline_adapter_path"].startswith("external_baseline/") for record in records)
    assert all(record["external_baseline_result_used_for_claim"] is False for record in records)
    assert all(record.get("S_final") is None for record in records)
    assert (run_root / "tables" / "external_baseline_comparison_table.csv").exists()
    assert (run_root / "artifacts" / "external_baseline_comparison_decision.json").exists()
    assert (run_root / "reports" / "external_baseline_comparison_report.md").exists()
