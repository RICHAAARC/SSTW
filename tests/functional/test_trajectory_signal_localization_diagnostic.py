"""验证 Stage 0-D 独立 profile、immutable preflight 与停止规则。"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from experiments.generative_video_model_probe.trajectory_signal_localization_diagnostic import (
    build_diagnostic_decision,
    build_immutable_input_snapshot,
    build_owner_key_direction_preflight,
    run_stage0d,
    validate_signal_localization_config,
)


def _config() -> dict[str, object]:
    return {
        "claim_support_status": "trajectory_signal_localization_diagnostic_only_not_paper_evidence",
        "conditional_attacked_phase_allowed": True,
        "cross_project_integration_allowed": False,
        "external_baseline_execution_allowed": False,
        "expected_wan_endpoint_latent_shape": [1, 16, 9, 40, 64],
        "fixed_fpr_evaluation_allowed": False,
        "frozen_likelihood_calibration_step_count": 20,
        "generation_aligned_replay_step_count": 8,
        "large_scale_generation_allowed": False,
        "minimum_full_correct_over_wrong_fraction": 0.75,
        "minimum_full_path_margin_over_clean_fraction": 0.5,
        "minimum_full_path_margin_over_endpoint_fraction": 0.5,
        "minimum_replay_reliability": 0.05,
        "no_attack_video_condition_id": "no_attack_original_video",
        "owner_key_direction_preflight_required": True,
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
                "generation_model_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                "generation_model_family": "diffusers_wan21_flow_matching_dit",
                "generation_model_commit_or_hash": "a" * 40,
                "generation_model_revision_resolution_status": "resolved_and_frozen",
                "generation_model_revision_source": "configured_immutable_commit_offline",
                "prompt_id": f"prompt-{source_index}",
                "seed_id": f"seed-{source_index}",
                "num_inference_steps": 8,
                "watermark_key_id": "sstw-paper-20260710-v1",
                "endpoint_key_direction_digest": f"direction-{source_index}",
                "endpoint_key_context_digest": f"context-{source_index}",
                "endpoint_integrated_phase_count": 8,
                "endpoint_integrated_weight_sum": 0.5,
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
    _write_json(
        source / "datasets" / "prompt_seed_suite.json",
        {
            "prompts": [
                {
                    "prompt_id": f"prompt-{source_index}",
                    "prompt_text": f"prompt text {source_index}",
                }
                for source_index in range(4)
            ]
        },
    )

    snapshot = build_immutable_input_snapshot(source, output, _config())

    assert snapshot["immutable_input_preflight_status"] == "ready"
    assert snapshot["generation_record_count"] == 12
    assert snapshot["attack_record_count"] == 24
    assert len(snapshot["video_inputs"]) == 36

    config_path = tmp_path / "stage0d.json"
    _write_json(config_path, _config())
    decision = run_stage0d(
        source,
        output,
        config_path,
        phase="decision",
    )
    assert decision["summary_record_count"] == 0
    assert decision["trajectory_signal_diagnostic_decision"] == (
        "no_attack_replay_pending"
    )
    assert decision["controlled_embedding_profile_construction_allowed"] is False
    assert (
        output / "artifacts" / "trajectory_signal_diagnostic_manifest.json"
    ).is_file()
    assert (
        output / "records" / "trajectory_signal_summary_records.jsonl"
    ).is_file()


def _fake_scheduler_loader(**_kwargs: object) -> object:
    return object()


def _matching_direction_metadata_builder(
    *, source_record: dict[str, object], **_kwargs: object
) -> dict[str, object]:
    return {
        "endpoint_key_direction_digest": source_record[
            "endpoint_key_direction_digest"
        ],
        "endpoint_key_context_digest": source_record["endpoint_key_context_digest"],
        "endpoint_integrated_phase_count": source_record[
            "endpoint_integrated_phase_count"
        ],
        "endpoint_integrated_weight_sum": source_record[
            "endpoint_integrated_weight_sum"
        ],
    }


def _mismatching_direction_metadata_builder(
    *, source_record: dict[str, object], **_kwargs: object
) -> dict[str, object]:
    result = _matching_direction_metadata_builder(source_record=source_record)
    result["endpoint_key_direction_digest"] = "different-owner-direction"
    return result


def _ready_owner_key_preflight() -> dict[str, object]:
    return {
        "owner_key_direction_preflight_status": "ready",
        "owner_key_direction_all_match": True,
        "owner_key_context_all_match": True,
        "owner_key_phase_grid_all_match": True,
    }


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
    decision = build_diagnostic_decision(
        summaries,
        pairs,
        [],
        config,
        owner_key_preflight=_ready_owner_key_preflight(),
    )
    assert decision["no_attack_signal_separation_ready"] is True
    assert decision["attacked_phase_execution_allowed"] is True
    assert decision["stage_progression_allowed"] is False


@pytest.mark.quick
def test_primary_fine_disagreement_stops_as_grid_sensitive() -> None:
    summaries, pairs, config = _decision_rows(primary_ready=True, fine_ready=False)
    decision = build_diagnostic_decision(
        summaries,
        pairs,
        [],
        config,
        owner_key_preflight=_ready_owner_key_preflight(),
    )
    assert decision["trajectory_signal_diagnostic_decision"] == (
        "replay_grid_sensitive_stop"
    )
    assert decision["attacked_phase_execution_allowed"] is False


@pytest.mark.quick
def test_owner_key_preflight_mismatch_is_redacted_and_blocks_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    secret = "owner-secret-material-that-must-not-leak"
    monkeypatch.setenv("SSTW_TRAJECTORY_AUTHENTICATION_KEY", secret)
    monkeypatch.setenv(
        "SSTW_TRAJECTORY_AUTHENTICATION_KEY_ID",
        "sstw-paper-20260710-v1",
    )
    rows = []
    for source_index in range(4):
        rows.append({
            "generation_status": "success",
            "method_variant": "sstw_full_method",
            "trajectory_trace_id": f"full-trace-{source_index}",
            "generation_model_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            "generation_model_family": "diffusers_wan21_flow_matching_dit",
            "generation_model_commit_or_hash": "a" * 40,
            "generation_model_revision_resolution_status": "resolved_and_frozen",
            "generation_model_revision_source": "configured_immutable_commit_offline",
            "prompt_id": f"prompt-{source_index}",
            "seed_id": f"seed-{source_index}",
            "num_inference_steps": 8,
            "watermark_key_id": "sstw-paper-20260710-v1",
            "endpoint_key_direction_digest": f"direction-{source_index}",
            "endpoint_key_context_digest": f"context-{source_index}",
            "endpoint_integrated_phase_count": 8,
            "endpoint_integrated_weight_sum": 0.5,
        })
    _write_jsonl(source / "records" / "generation_records.jsonl", rows)
    _write_json(
        source / "datasets" / "prompt_seed_suite.json",
        {
            "prompts": [
                {
                    "prompt_id": f"prompt-{source_index}",
                    "prompt_text": f"prompt text {source_index}",
                }
                for source_index in range(4)
            ]
        },
    )

    observed_key_texts: list[str] = []

    def capturing_direction_metadata_builder(
        *, key_text: str, source_record: dict[str, object], **_kwargs: object
    ) -> dict[str, object]:
        observed_key_texts.append(key_text)
        return _matching_direction_metadata_builder(source_record=source_record)

    ready_preflight = build_owner_key_direction_preflight(
        source,
        _config(),
        scheduler_loader=_fake_scheduler_loader,
        direction_metadata_builder=capturing_direction_metadata_builder,
    )
    assert ready_preflight["owner_key_direction_preflight_status"] == "ready"
    assert ready_preflight["owner_key_direction_match_count"] == 4
    assert ready_preflight["owner_key_direction_all_match"] is True
    assert ready_preflight["owner_key_context_all_match"] is True
    assert ready_preflight["owner_key_phase_grid_all_match"] is True
    assert len(observed_key_texts) == 4
    assert len(set(observed_key_texts)) == 4

    preflight = build_owner_key_direction_preflight(
        source,
        _config(),
        scheduler_loader=_fake_scheduler_loader,
        direction_metadata_builder=_mismatching_direction_metadata_builder,
    )

    assert preflight["owner_key_direction_preflight_status"] == "mismatch"
    assert preflight["owner_key_direction_match_count"] == 0
    assert len(preflight["owner_key_direction_mismatch_trace_ids"]) == 4
    assert secret not in json.dumps(preflight)
    decision = build_diagnostic_decision(
        [], [], [], _config(), owner_key_preflight=preflight
    )
    assert decision["trajectory_signal_diagnostic_decision"] == (
        "owner_key_direction_mismatch_stop"
    )
    assert decision["controlled_embedding_profile_construction_allowed"] is False

    config_path = tmp_path / "stage0d.json"
    _write_json(config_path, _config())
    monkeypatch.setattr(
        "experiments.generative_video_model_probe.trajectory_signal_localization_diagnostic.build_immutable_input_snapshot",
        lambda *_args, **_kwargs: {
            "record_version": "trajectory_signal_localization_diagnostic_v1",
            "profile_id": _config()["profile_id"],
            "immutable_input_preflight_status": "ready",
            "immutable_input_snapshot_digest": "fixture-snapshot",
        },
    )
    pipeline_load_count = 0

    def forbidden_pipeline_loader(**_kwargs: object) -> object:
        nonlocal pipeline_load_count
        pipeline_load_count += 1
        raise AssertionError("owner-key mismatch 后不得加载 replay pipeline")

    run_decision = run_stage0d(
        source,
        output,
        config_path,
        phase="no_attack",
        pipeline_loader=forbidden_pipeline_loader,
        scheduler_loader=_fake_scheduler_loader,
        direction_metadata_builder=_mismatching_direction_metadata_builder,
    )
    assert pipeline_load_count == 0
    assert run_decision["trajectory_signal_diagnostic_decision"] == (
        "owner_key_direction_mismatch_stop"
    )
    preflight_text = (
        output / "artifacts" / "trajectory_signal_owner_key_preflight.json"
    ).read_text(encoding="utf-8")
    assert secret not in preflight_text


@pytest.mark.quick
def test_direct_attacked_phase_requires_ready_no_attack_gate() -> None:
    decision = build_diagnostic_decision(
        [],
        [],
        [],
        _config(),
        owner_key_preflight=_ready_owner_key_preflight(),
        attacked_phase_requested=True,
    )
    assert decision["trajectory_signal_diagnostic_decision"] == (
        "attacked_phase_precondition_not_ready_stop"
    )
    assert decision["attacked_phase_execution_allowed"] is False
