"""验证 Stage 0-D 独立 profile、immutable preflight 与停止规则。"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from experiments.generative_video_model_probe.trajectory_signal_localization_diagnostic import (
    build_diagnostic_decision,
    build_immutable_input_snapshot,
    validate_signal_localization_config,
)


def _config() -> dict[str, object]:
    return {
        "claim_support_status": "trajectory_signal_localization_diagnostic_only_not_paper_evidence",
        "conditional_attacked_phase_allowed": True,
        "cross_project_integration_allowed": False,
        "external_baseline_execution_allowed": False,
        "fixed_fpr_evaluation_allowed": False,
        "frozen_likelihood_calibration_step_count": 20,
        "generation_aligned_replay_step_count": 8,
        "large_scale_generation_allowed": False,
        "minimum_full_correct_over_wrong_fraction": 0.75,
        "minimum_full_path_margin_over_clean_fraction": 0.5,
        "minimum_full_path_margin_over_endpoint_fraction": 0.5,
        "minimum_replay_reliability": 0.05,
        "no_attack_video_condition_id": "no_attack_original_video",
        "primary_replay_step_count": 20,
        "profile_id": "sstw_trajectory_signal_localization_diagnostic",
        "replay_grid_step_counts": [8, 20, 40],
        "required_attacked_video_condition_ids": [
            "h264_crf28_runtime",
            "temporal_crop_runtime",
        ],
        "required_source_method_variants": [
            "sstw_full_method",
            "endpoint_only_control",
            "sstw_clean_unwatermarked_reference",
        ],
        "required_source_video_count": 4,
        "stage_progression_allowed": False,
        "test_split_claims_allowed": False,
        "time_grid_selection_allowed": False,
        "trajectory_signal_fine_replay_step_count": 40,
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


@pytest.mark.quick
def test_stage0d_config_rejects_stage_progression() -> None:
    config = _config()
    validate_signal_localization_config(config)
    config["stage_progression_allowed"] = True
    with pytest.raises(ValueError, match="禁止项"):
        validate_signal_localization_config(config)


@pytest.mark.quick
def test_immutable_preflight_hashes_all_existing_inputs(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    variants = _config()["required_source_method_variants"]
    generations: list[dict[str, object]] = []
    attacks: list[dict[str, object]] = []
    for source_index in range(4):
        for variant in variants:
            video = source / "videos" / f"{source_index}_{variant}.mp4"
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(f"original-{source_index}-{variant}".encode())
            generations.append({
                "generation_status": "success",
                "method_variant": variant,
                "trajectory_trace_id": f"trace-{source_index}-{variant}",
                "video_path": str(video),
                "video_sha256": sha256(video.read_bytes()).hexdigest(),
            })
            for attack in _config()["required_attacked_video_condition_ids"]:
                attacked = source / "attacked_videos" / f"{source_index}_{variant}_{attack}.mp4"
                attacked.parent.mkdir(parents=True, exist_ok=True)
                attacked.write_bytes(f"attack-{source_index}-{variant}-{attack}".encode())
                attacks.append({
                    "attack_runtime_status": "ready",
                    "attack_name": attack,
                    "method_variant": variant,
                    "trajectory_trace_id": f"trace-{source_index}-{variant}",
                    "attacked_video_path": str(attacked),
                    "attacked_video_sha256": sha256(attacked.read_bytes()).hexdigest(),
                })
    _write_jsonl(source / "records" / "generation_records.jsonl", generations)
    _write_jsonl(
        source / "records" / "trajectory_replay_smoke_attack_records.jsonl",
        attacks,
    )
    _write_jsonl(
        source / "records" / "trajectory_replay_smoke_likelihood_calibrations.jsonl",
        [{
            "replay_likelihood_calibration_record_id": "frozen-calibration",
            "replay_likelihood_calibration_step_counts": [20],
            "replay_relative_observation_noise_standard_deviation": 0.4,
        }],
    )
    _write_json(source / "artifacts" / "trajectory_replay_smoke_decision.json", {"go_no_go": "NO_GO"})
    _write_json(source / "artifacts" / "trajectory_replay_smoke_manifest.json", {"artifact_id": "source"})

    snapshot = build_immutable_input_snapshot(source, output, _config())

    assert snapshot["immutable_input_preflight_status"] == "ready"
    assert snapshot["generation_record_count"] == 12
    assert snapshot["attack_record_count"] == 24
    assert len(snapshot["video_inputs"]) == 36


def _decision_rows(*, primary_ready: bool = True, fine_ready: bool = True):
    config = _config()
    summaries: list[dict[str, object]] = []
    pairs: list[dict[str, object]] = []
    condition = "no_attack_original_video"
    for grid, ready in ((8, False), (20, primary_ready), (40, fine_ready)):
        for source_index in range(4):
            for variant in config["required_source_method_variants"]:
                summaries.append({
                    "video_condition_id": condition,
                    "replay_grid_step_count": grid,
                    "method_variant": variant,
                })
                pairs.append({
                    "video_condition_id": condition,
                    "replay_grid_step_count": grid,
                    "method_variant": variant,
                    "trajectory_signal_comparison_kind": "correct_owner_key_over_wrong_owner_key",
                    "correct_over_wrong_path_margin": 1.0 if ready else -1.0,
                    "correct_over_wrong_likelihood_margin": 1.0 if ready else -1.0,
                    "minimum_pair_reliability": 0.5,
                })
            for control in (
                "endpoint_only_control",
                "sstw_clean_unwatermarked_reference",
            ):
                pairs.append({
                    "video_condition_id": condition,
                    "replay_grid_step_count": grid,
                    "control_method_variant": control,
                    "trajectory_signal_comparison_kind": "full_over_control_path_margin_gain",
                    "full_over_control_path_margin_gain": 1.0 if ready else -1.0,
                })
    return summaries, pairs, config


@pytest.mark.quick
def test_no_attack_primary_and_fine_must_both_pass_before_attacked_phase() -> None:
    summaries, pairs, config = _decision_rows()
    decision = build_diagnostic_decision(summaries, pairs, [], config)
    assert decision["no_attack_signal_separation_ready"] is True
    assert decision["attacked_phase_execution_allowed"] is True
    assert decision["stage_progression_allowed"] is False


@pytest.mark.quick
def test_primary_fine_disagreement_stops_as_grid_sensitive() -> None:
    summaries, pairs, config = _decision_rows(primary_ready=True, fine_ready=False)
    decision = build_diagnostic_decision(summaries, pairs, [], config)
    assert decision["trajectory_signal_diagnostic_decision"] == (
        "replay_grid_sensitive_stop"
    )
    assert decision["attacked_phase_execution_allowed"] is False
