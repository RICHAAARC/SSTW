"""验证 controlled embedding no-attack 执行输入、配对与停止语义。"""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import shutil

import pytest

from evaluation.protocol.record_writer import write_jsonl
from experiments.generative_video_model_probe.controlled_embedding_strength_diagnostic import (
    build_controlled_embedding_strength_diagnostic_decision,
    build_controlled_embedding_strength_pair_records,
    run_controlled_embedding_strength_diagnostic,
    validate_controlled_embedding_execution_input,
    validate_controlled_embedding_diagnostic_config,
)
from experiments.generative_video_model_probe.controlled_embedding_strength_profile import (
    construct_controlled_embedding_strength_profile,
)


CONSTRUCTION_CONFIG_PATH = Path(
    "configs/protocol/sstw_controlled_embedding_strength_profile.json"
)
DIAGNOSTIC_CONFIG_PATH = Path(
    "configs/protocol/sstw_controlled_embedding_strength_diagnostic.json"
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _stable_digest(value: dict[str, object]) -> str:
    return sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _prompt_suite() -> dict[str, object]:
    return {
        "prompt_suite_id": "controlled-embedding-test-suite",
        "prompts": [
            {
                "prompt_id": f"prompt-{index}",
                "prompt_text": f"prompt text {index}",
                "prompt_negative_text": "",
                "prompt_category": "object_motion",
                "motion_pattern_id": f"motion-{index}",
                "motion_claim_role": "positive_motion",
                "motion_calibration_role": None,
                "prompt_suite_role": "probe_paper",
            }
            for index in range(2)
        ],
        "seeds": [
            {
                "seed_id": f"seed-{index}",
                "seed_value": index + 10,
                "split": "calibration",
                "prompt_suite_role": "probe_paper",
            }
            for index in range(2)
        ],
    }


def _source_decision() -> dict[str, object]:
    return {
        "profile_id": "sstw_trajectory_signal_localization_diagnostic",
        "trajectory_signal_diagnostic_decision": (
            "embedding_or_replay_signal_not_separated_stop"
        ),
        "controlled_embedding_profile_construction_allowed": True,
        "owner_key_direction_preflight_status": "ready",
        "owner_key_direction_all_match": True,
        "owner_key_context_all_match": True,
        "owner_key_phase_grid_all_match": True,
        "no_attack_signal_separation_ready": False,
        "attacked_phase_executed": False,
        "stage_progression_allowed": False,
        "failure_record_count": 0,
        "summary_record_count": 72,
        "pair_record_count": 60,
    }


def _portable_input_bundle(tmp_path: Path) -> Path:
    source_root = tmp_path / "source_files"
    source_decision_path = (
        source_root / "artifacts" / "trajectory_signal_diagnostic_decision.json"
    )
    source_snapshot_path = (
        source_root
        / "artifacts"
        / "trajectory_signal_immutable_input_snapshot.json"
    )
    source_manifest_path = (
        source_root / "artifacts" / "trajectory_signal_diagnostic_manifest.json"
    )
    prompt_suite_path = source_root / "datasets" / "prompt_seed_suite.json"
    likelihood_path = (
        source_root
        / "records"
        / "trajectory_replay_smoke_likelihood_calibrations.jsonl"
    )
    _write_json(source_decision_path, _source_decision())
    _write_json(prompt_suite_path, _prompt_suite())
    write_jsonl(
        likelihood_path,
        [
            {
                "replay_relative_observation_noise_standard_deviation": 0.01,
                "replay_minimum_observation_noise_variance": 1e-8,
                "replay_likelihood_model_id": "test",
                "replay_likelihood_calibration_protocol": "test",
                "replay_likelihood_calibration_cluster_count": 4,
            }
        ],
    )
    snapshot: dict[str, object] = {
        "profile_id": "sstw_trajectory_signal_localization_diagnostic",
        "immutable_input_preflight_status": "ready",
        "immutable_input_scope": "full_replay_diagnostic_inputs",
        "generation_record_count": 12,
        "attack_record_count": 24,
        "likelihood_calibration_input_status": "ready",
        "governed_input_sha256": {
            "/content/workspace/source/datasets/prompt_seed_suite.json": sha256(
                prompt_suite_path.read_bytes()
            ).hexdigest(),
            (
                "/content/workspace/source/records/"
                "trajectory_replay_smoke_likelihood_calibrations.jsonl"
            ): sha256(likelihood_path.read_bytes()).hexdigest(),
        },
    }
    snapshot["immutable_input_snapshot_digest"] = _stable_digest(snapshot)
    _write_json(source_snapshot_path, snapshot)
    _write_json(
        source_manifest_path,
        {
            "profile_id": "sstw_trajectory_signal_localization_diagnostic",
            "immutable_input_snapshot_digest": snapshot[
                "immutable_input_snapshot_digest"
            ],
            "output_sha256": {
                (
                    "/content/workspace/result/artifacts/"
                    "trajectory_signal_diagnostic_decision.json"
                ): sha256(source_decision_path.read_bytes()).hexdigest(),
                (
                    "/content/workspace/result/artifacts/"
                    "trajectory_signal_immutable_input_snapshot.json"
                ): sha256(source_snapshot_path.read_bytes()).hexdigest(),
            },
        },
    )
    construction_root = tmp_path / "construction_output"
    construct_controlled_embedding_strength_profile(
        source_decision_path,
        source_snapshot_path,
        source_manifest_path,
        prompt_suite_path,
        construction_root,
        CONSTRUCTION_CONFIG_PATH,
    )

    package_root = tmp_path / "portable_input"
    copy_pairs = {
        (
            package_root
            / "construction"
            / "artifacts"
            / "controlled_embedding_profile_construction_decision.json"
        ): (
            construction_root
            / "artifacts"
            / "controlled_embedding_profile_construction_decision.json"
        ),
        (
            package_root
            / "construction"
            / "artifacts"
            / "controlled_embedding_profile_construction_manifest.json"
        ): (
            construction_root
            / "artifacts"
            / "controlled_embedding_profile_construction_manifest.json"
        ),
        (
            package_root
            / "construction"
            / "records"
            / "controlled_embedding_generation_plan.jsonl"
        ): (
            construction_root
            / "records"
            / "controlled_embedding_generation_plan.jsonl"
        ),
        (
            package_root
            / "source"
            / "artifacts"
            / "trajectory_signal_diagnostic_decision.json"
        ): source_decision_path,
        (
            package_root
            / "source"
            / "artifacts"
            / "trajectory_signal_immutable_input_snapshot.json"
        ): source_snapshot_path,
        (
            package_root
            / "source"
            / "artifacts"
            / "trajectory_signal_diagnostic_manifest.json"
        ): source_manifest_path,
        (
            package_root / "source" / "datasets" / "prompt_seed_suite.json"
        ): prompt_suite_path,
        (
            package_root
            / "source"
            / "records"
            / "trajectory_replay_smoke_likelihood_calibrations.jsonl"
        ): likelihood_path,
    }
    for target, source in copy_pairs.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return package_root


def _synthetic_summaries() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    levels = {
        "reference_default": (0.12, 0.2, 0.3),
        "moderate_increase": (0.24, 0.8, 0.2),
        "high_increase": (0.48, 0.9, 0.1),
        "clean_unwatermarked_control": (0.0, 0.4, 0.4),
    }
    for prompt_index in range(2):
        for seed_index in range(2):
            for level_id, (lambda_max, correct_value, wrong_value) in levels.items():
                for grid in (8, 20, 40):
                    for key_role, value in (
                        ("correct_owner_key", correct_value),
                        ("wrong_owner_key", wrong_value),
                    ):
                        rows.append(
                            {
                                "generation_model_id": (
                                    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
                                ),
                                "prompt_id": f"prompt-{prompt_index}",
                                "seed_id": f"seed-{seed_index}",
                                "embedding_strength_level_id": level_id,
                                "lambda_max": lambda_max,
                                "replay_grid_step_count": grid,
                                "candidate_key_role": key_role,
                                "trajectory_velocity_projection": value,
                                "trajectory_path_projection": value,
                                "replay_log_likelihood_ratio": value,
                                "endpoint_score": value,
                                "trajectory_global_reliability": 0.5,
                            }
                        )
    return rows


@pytest.mark.quick
def test_execution_input_rebuilds_portable_construction_bundle(
    tmp_path: Path,
) -> None:
    validated = validate_controlled_embedding_execution_input(
        _portable_input_bundle(tmp_path)
    )

    assert len(validated["construction_plan"]) == 16
    assert validated["construction_decision"][
        "generation_execution_allowed"
    ] is False
    assert {
        row["embedding_strength_level_id"]
        for row in validated["construction_plan"]
    } == {
        "reference_default",
        "moderate_increase",
        "high_increase",
        "clean_unwatermarked_control",
    }


@pytest.mark.quick
def test_execution_input_rejects_tampered_plan(tmp_path: Path) -> None:
    input_root = _portable_input_bundle(tmp_path)
    plan_path = (
        input_root
        / "construction"
        / "records"
        / "controlled_embedding_generation_plan.jsonl"
    )
    rows = [
        json.loads(line)
        for line in plan_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["lambda_max"] = 9.0
    write_jsonl(plan_path, rows)

    with pytest.raises(ValueError, match="construction output digest"):
        validate_controlled_embedding_execution_input(input_root)


@pytest.mark.quick
def test_pair_and_decision_report_predeclared_lambda_repair() -> None:
    config = json.loads(DIAGNOSTIC_CONFIG_PATH.read_text(encoding="utf-8"))
    summaries = _synthetic_summaries()
    pairs = build_controlled_embedding_strength_pair_records(
        summaries,
        config,
    )
    generation_records = [
        {"generation_status": "success"} for _ in range(16)
    ]
    decision = build_controlled_embedding_strength_diagnostic_decision(
        generation_records,
        summaries,
        pairs,
        [],
        config,
    )

    assert len(summaries) == 96
    assert len(pairs) == 84
    assert decision["controlled_embedding_strength_diagnostic_decision"] == (
        "lambda_increase_repaired_path_signal"
    )
    assert decision["lambda_increase_path_signal_repair_observed"] is True
    assert decision["path_signal_separated_strength_level_ids"] == [
        "moderate_increase",
        "high_increase",
    ]
    assert decision["attacked_phase_executed"] is False
    assert decision["stage_progression_allowed"] is False


@pytest.mark.quick
@pytest.mark.parametrize(
    ("field", "mutated_value"),
    [
        ("minimum_correct_over_wrong_fraction", 0.0),
        ("minimum_path_margin_gain_over_clean_fraction", 0.0),
        ("minimum_replay_reliability", 0.0),
        ("generation_aligned_replay_step_count", 20),
        ("primary_replay_step_count", 8),
        ("trajectory_signal_fine_replay_step_count", 8),
        ("no_attack_video_condition_id", "observed_result_condition"),
        ("claim_support_status", "formal_paper_evidence"),
        ("paper_result_level", "full_paper"),
        ("required_construction_profile_id", "unreviewed_profile"),
        ("required_construction_decision", "generation_authorized"),
    ],
)
def test_diagnostic_config_rejects_gate_or_claim_mutation(
    field: str,
    mutated_value: object,
) -> None:
    config = json.loads(DIAGNOSTIC_CONFIG_PATH.read_text(encoding="utf-8"))
    mutated = deepcopy(config)
    mutated[field] = mutated_value

    with pytest.raises(ValueError):
        validate_controlled_embedding_diagnostic_config(mutated)


@pytest.mark.quick
def test_runner_uses_separate_execution_decision_and_no_attack_only(
    tmp_path: Path,
) -> None:
    input_root = _portable_input_bundle(tmp_path)
    output_root = tmp_path / "run_output"

    def fake_generation(
        validated: dict[str, object],
        output: Path,
    ) -> dict[str, object]:
        del validated
        write_jsonl(
            output / "records" / "generation_records.jsonl",
            [
                {
                    "generation_status": "success",
                    "generation_model_id": (
                        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
                    ),
                }
                for _ in range(16)
            ],
        )
        return {"generation_record_count": 16}

    observed_conditions: list[str] = []

    def fake_replay(
        source: Path,
        output: Path,
        config: dict[str, object],
        *,
        condition: str,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
        del source, output, config
        observed_conditions.append(condition)
        return _synthetic_summaries(), [], []

    decision = run_controlled_embedding_strength_diagnostic(
        input_root,
        output_root,
        generation_runner=fake_generation,
        replay_runner=fake_replay,
    )
    execution_decision = json.loads(
        (
            output_root
            / "artifacts"
            / "controlled_embedding_execution_decision.json"
        ).read_text(encoding="utf-8")
    )
    source_construction = json.loads(
        (
            output_root
            / "inputs"
            / "construction"
            / "controlled_embedding_profile_construction_decision.json"
        ).read_text(encoding="utf-8")
    )

    assert observed_conditions == ["no_attack"]
    assert execution_decision["controlled_embedding_execution_allowed"] is True
    assert execution_decision["attacked_phase_execution_allowed"] is False
    assert source_construction["generation_execution_allowed"] is False
    assert decision["formal_result"] is False
    assert decision["stage_progression_allowed"] is False
