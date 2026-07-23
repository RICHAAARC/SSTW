"""验证 Stage 0-D 失败后的 controlled embedding construction-only profile。"""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from experiments.generative_video_model_probe.controlled_embedding_strength_profile import (
    build_controlled_embedding_generation_plan,
    build_velocity_constraint_config_for_strength_level,
    construct_controlled_embedding_strength_profile,
    validate_controlled_embedding_strength_profile,
    validate_source_trajectory_signal_decision,
)
from main.methods.state_space_watermark.velocity_field_constraint import (
    VelocityFieldConstraintConfig,
)


CONFIG_PATH = Path(
    "configs/protocol/sstw_controlled_embedding_strength_profile.json"
)


def _config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


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


def _prompt_suite() -> dict[str, object]:
    return {
        "prompts": [
            {
                "prompt_id": f"prompt-{index}",
                "prompt_text": f"prompt text {index}",
                "prompt_suite_role": "probe_paper",
            }
            for index in range(3)
        ],
        "seeds": [
            {
                "seed_id": f"calibration-seed-{index}",
                "seed_value": index + 10,
                "split": "calibration",
                "prompt_suite_role": "probe_paper",
            }
            for index in range(3)
        ]
        + [
            {
                "seed_id": "test-seed",
                "seed_value": 99,
                "split": "test",
                "prompt_suite_role": "probe_paper",
            }
        ],
    }


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


def _write_source_bundle(
    root: Path,
    *,
    unexpected_secret: str | None = None,
) -> tuple[Path, Path, Path, Path]:
    source = root / "source_decision.json"
    snapshot_path = root / "source_snapshot.json"
    source_manifest = root / "source_manifest.json"
    prompt_suite = root / "prompt_suite.json"
    decision = _source_decision()
    if unexpected_secret is not None:
        decision["unexpected_secret"] = unexpected_secret
    _write_json(source, decision)
    _write_json(prompt_suite, _prompt_suite())
    snapshot: dict[str, object] = {
        "profile_id": "sstw_trajectory_signal_localization_diagnostic",
        "immutable_input_preflight_status": "ready",
        "immutable_input_scope": "full_replay_diagnostic_inputs",
        "generation_record_count": 12,
        "attack_record_count": 24,
        "likelihood_calibration_input_status": "ready",
        "governed_input_sha256": {
            str(prompt_suite.resolve()): sha256(prompt_suite.read_bytes()).hexdigest()
        },
    }
    snapshot["immutable_input_snapshot_digest"] = _stable_digest(snapshot)
    _write_json(snapshot_path, snapshot)
    _write_json(
        source_manifest,
        {
            "profile_id": "sstw_trajectory_signal_localization_diagnostic",
            "immutable_input_snapshot_digest": snapshot[
                "immutable_input_snapshot_digest"
            ],
            "output_sha256": {
                str(source.resolve()): sha256(source.read_bytes()).hexdigest(),
                str(snapshot_path.resolve()): sha256(
                    snapshot_path.read_bytes()
                ).hexdigest(),
            },
        },
    )
    return source, snapshot_path, source_manifest, prompt_suite


@pytest.mark.quick
def test_reference_level_matches_current_runtime_and_ladder_is_single_factor() -> None:
    config = _config()
    validate_controlled_embedding_strength_profile(config)
    default = VelocityFieldConstraintConfig()
    reference = build_velocity_constraint_config_for_strength_level(
        config, "reference_default"
    )
    moderate = build_velocity_constraint_config_for_strength_level(
        config, "moderate_increase"
    )
    high = build_velocity_constraint_config_for_strength_level(
        config, "high_increase"
    )

    assert reference == default
    assert [reference.lambda_max, moderate.lambda_max, high.lambda_max] == [
        0.12,
        0.24,
        0.48,
    ]
    assert {
        reference.velocity_norm_ratio_budget,
        moderate.velocity_norm_ratio_budget,
        high.velocity_norm_ratio_budget,
    } == {0.02}
    assert {
        reference.flow_energy_budget_ratio,
        moderate.flow_energy_budget_ratio,
        high.flow_energy_budget_ratio,
    } == {0.000015}


@pytest.mark.quick
def test_generation_plan_is_four_source_three_strength_plus_clean_and_not_authorized() -> None:
    plan = build_controlled_embedding_generation_plan(_prompt_suite(), _config())

    assert len(plan) == 16
    assert sum(row["method_variant"] == "sstw_full_method" for row in plan) == 12
    assert sum(
        row["method_variant"] == "sstw_clean_unwatermarked_reference"
        for row in plan
    ) == 4
    assert {row["split"] for row in plan} == {"calibration"}
    assert all(row["generation_execution_allowed"] is False for row in plan)
    assert all(row["stage_progression_allowed"] is False for row in plan)
    assert {row["generation_seed_random"] for row in plan} == {10, 11}
    assert all(len(row["prompt_text_hash"]) == 64 for row in plan)
    assert len({row["controlled_embedding_plan_record_id"] for row in plan}) == 16
    for prompt_id in ("prompt-0", "prompt-1"):
        for seed_id in ("calibration-seed-0", "calibration-seed-1"):
            paired = [
                row
                for row in plan
                if row["prompt_id"] == prompt_id and row["seed_id"] == seed_id
            ]
            assert {row["embedding_strength_level_id"] for row in paired} == {
                "reference_default",
                "moderate_increase",
                "high_increase",
                "clean_unwatermarked_control",
            }


@pytest.mark.quick
def test_construction_materializes_rebuildable_plan_without_copying_source_secret(
    tmp_path: Path,
) -> None:
    secret = "source-decision-secret-that-must-not-propagate"
    output = tmp_path / "constructed_profile"
    source, snapshot, source_manifest, prompt_suite = _write_source_bundle(
        tmp_path,
        unexpected_secret=secret,
    )

    decision = construct_controlled_embedding_strength_profile(
        source,
        snapshot,
        source_manifest,
        prompt_suite,
        output,
        CONFIG_PATH,
    )

    assert decision["controlled_embedding_profile_construction_status"] == "ready"
    assert decision["controlled_embedding_profile_construction_decision"] == (
        "profile_constructed_generation_not_authorized"
    )
    assert decision["generation_execution_allowed"] is False
    assert decision["controlled_embedding_strength_level_count"] == 3
    assert decision["controlled_embedding_plan_record_count"] == 16
    manifest_path = (
        output
        / "artifacts"
        / "controlled_embedding_profile_construction_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["config_digest"] == sha256(CONFIG_PATH.read_bytes()).hexdigest()
    builder_path = Path(manifest["controlled_embedding_profile_builder_source_path"])
    assert sha256(builder_path.read_bytes()).hexdigest() == manifest[
        "controlled_embedding_profile_builder_source_sha256"
    ]
    for raw_path, expected in manifest["output_sha256"].items():
        assert sha256(Path(raw_path).read_bytes()).hexdigest() == expected
    for path in output.rglob("*"):
        if path.is_file():
            assert secret.encode("utf-8") not in path.read_bytes()


@pytest.mark.quick
def test_construction_rejects_source_without_valid_owner_key_evidence() -> None:
    config = _config()
    source = _source_decision()
    source["owner_key_direction_all_match"] = False

    with pytest.raises(ValueError, match="owner_key_direction_all_match"):
        validate_source_trajectory_signal_decision(source, config)


@pytest.mark.quick
def test_construction_profile_cannot_self_authorize_generation_or_multi_factor_tuning() -> None:
    generation_enabled = deepcopy(_config())
    generation_enabled["generation_execution_allowed"] = True
    with pytest.raises(ValueError, match="禁止项未冻结"):
        validate_controlled_embedding_strength_profile(generation_enabled)

    multi_factor = deepcopy(_config())
    multi_factor["strength_levels"][1]["flow_energy_budget_ratio"] = 0.00003
    with pytest.raises(ValueError, match="flow energy budget"):
        validate_controlled_embedding_strength_profile(multi_factor)


@pytest.mark.quick
def test_construction_requires_new_output_root(tmp_path: Path) -> None:
    output = tmp_path / "constructed_profile"
    source, snapshot, source_manifest, prompt_suite = _write_source_bundle(tmp_path)
    construct_controlled_embedding_strength_profile(
        source, snapshot, source_manifest, prompt_suite, output, CONFIG_PATH
    )

    with pytest.raises(FileExistsError, match="新的空 output root"):
        construct_controlled_embedding_strength_profile(
            source, snapshot, source_manifest, prompt_suite, output, CONFIG_PATH
        )


@pytest.mark.quick
def test_construction_rejects_prompt_suite_outside_immutable_snapshot(
    tmp_path: Path,
) -> None:
    source, snapshot, source_manifest, prompt_suite = _write_source_bundle(tmp_path)
    prompt_value = json.loads(prompt_suite.read_text(encoding="utf-8"))
    prompt_value["prompts"][0]["prompt_text"] = "tampered prompt"
    _write_json(prompt_suite, prompt_value)

    with pytest.raises(ValueError, match="immutable snapshot 绑定"):
        construct_controlled_embedding_strength_profile(
            source,
            snapshot,
            source_manifest,
            prompt_suite,
            tmp_path / "output",
            CONFIG_PATH,
        )
