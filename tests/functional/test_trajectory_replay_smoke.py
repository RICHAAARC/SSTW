"""验证 minimal trajectory replay smoke 的边界、包恢复与 go/no-go。"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from evaluation.attacks.video_runtime_attack_protocol import (
    load_protocol_config_with_shared_attack_protocol,
)
from experiments.generative_video_model_probe.trajectory_replay_smoke import (
    build_smoke_decision,
    materialize_source_package,
    validate_minimal_trajectory_profile,
)


pytestmark = pytest.mark.quick

CONFIG_PATH = Path("configs/protocol/sstw_minimal_trajectory_paper.json")
VARIANTS = (
    "sstw_full_method",
    "endpoint_only_control",
    "sstw_clean_unwatermarked_reference",
)
ATTACKS = ("h264_crf28_runtime", "temporal_crop_runtime")


def _config() -> dict:
    return load_protocol_config_with_shared_attack_protocol(CONFIG_PATH)


def _write_source_package(path: Path) -> None:
    records: list[dict] = []
    with zipfile.ZipFile(path, mode="w") as archive:
        for source_index in range(4):
            for variant in VARIANTS:
                video_name = f"source_{source_index}_{variant}.mp4"
                records.append({
                    "generation_status": "success",
                    "generation_model_id": "model-a",
                    "prompt_id": f"prompt-{source_index}",
                    "seed_id": f"seed-{source_index}",
                    "trajectory_trace_id": f"trace-{source_index}-{variant}",
                    "split": "calibration",
                    "colab_runtime_profile": "method_mechanism_validation",
                    "method_variant": variant,
                    "sample_role": (
                        "clean_negative"
                        if variant == "sstw_clean_unwatermarked_reference"
                        else "attacked_positive_source"
                    ),
                    "generation_sample_role": (
                        "clean_negative"
                        if variant == "sstw_clean_unwatermarked_reference"
                        else "attacked_positive_source"
                    ),
                    "video_path": f"/remote/videos/{video_name}",
                })
                archive.writestr(f"run/videos/{video_name}", b"video")
        archive.writestr(
            "run/records/generation_records.jsonl",
            "".join(json.dumps(record) + "\n" for record in records),
        )
        archive.writestr(
            "datasets/generative_video_prompt_suite/prompt_seed_suite.json",
            json.dumps({
                "prompts": [
                    {"prompt_id": f"prompt-{index}", "prompt_text": f"prompt {index}"}
                    for index in range(4)
                ]
            }),
        )


def test_minimal_profile_is_independent_and_fail_closed() -> None:
    config = _config()

    validate_minimal_trajectory_profile(config)

    assert config["paper_result_level"] == "trajectory_paper_smoke"
    assert config["fixed_fpr_evaluation_allowed"] is False
    assert config["large_scale_generation_allowed"] is False
    assert config["external_baseline_execution_allowed"] is False
    assert config["cross_project_integration_allowed"] is False
    assert config["required_runtime_attack_names"] == list(ATTACKS)
    assert "paper_profile_common_contract_path" not in config
    assert "shared_attack_protocol_config_path" not in config


def test_source_package_materialization_requires_four_complete_variant_blocks(
    tmp_path: Path,
) -> None:
    package_path = tmp_path / "source.zip"
    run_root = tmp_path / "smoke"
    _write_source_package(package_path)

    manifest = materialize_source_package(package_path, run_root, _config())

    assert manifest["source_video_count"] == 4
    assert manifest["source_generation_record_count"] == 12
    assert manifest["source_video_file_count"] == 12
    assert manifest["source_split"] == "calibration"
    assert len(list((run_root / "videos").glob("*.mp4"))) == 12


def _attack_records() -> list[dict]:
    return [
        {
            "generation_model_id": "model-a",
            "prompt_id": f"prompt-{source_index}",
            "seed_id": f"seed-{source_index}",
            "method_variant": variant,
            "attack_name": attack,
            "attack_runtime_status": "ready",
        }
        for source_index in range(4)
        for variant in VARIANTS
        for attack in ATTACKS
    ]


def _replay_records() -> list[dict]:
    margin_by_variant = {
        "sstw_full_method": 2.0,
        "endpoint_only_control": 1.0,
        "sstw_clean_unwatermarked_reference": 0.0,
    }
    return [
        {
            "generation_model_id": "model-a",
            "prompt_id": f"prompt-{source_index}",
            "seed_id": f"seed-{source_index}",
            "method_variant": variant,
            "attack_name": attack,
            "correct_key_path_margin": margin_by_variant[variant],
            "correct_key_replay_likelihood_margin": (
                1.0 if variant == "sstw_full_method" else 0.0
            ),
            "replay_reliability": 0.8,
            "replay_numeric_finite": True,
            "wrong_key_fixed_reverse_path_reused": True,
        }
        for source_index in range(4)
        for variant in VARIANTS
        for attack in ATTACKS
    ]


def test_go_decision_only_allows_next_protocol_construction() -> None:
    config = _config()
    decision = build_smoke_decision(
        config,
        {"source_video_count": 4},
        _attack_records(),
        _replay_records(),
        [],
        {
            "replay_runtime_preflight_status": "ready",
            "replay_runtime_blockers": [],
        },
    )

    assert decision["go_no_go_decision"] == "GO"
    assert decision["stage_progression"] == (
        "minimal_trajectory_paper_protocol_construction_only"
    )
    assert decision["paper_claim_allowed"] is False
    assert decision["fixed_fpr_evaluated"] is False
    assert decision["large_scale_generation_allowed"] is False


def test_missing_gpu_runtime_is_explicit_no_go_not_mechanism_failure() -> None:
    decision = build_smoke_decision(
        _config(),
        {"source_video_count": 4},
        _attack_records(),
        [],
        [],
        {
            "replay_runtime_preflight_status": "blocked",
            "replay_runtime_blockers": ["missing_python_package:torch"],
        },
    )

    assert decision["go_no_go_decision"] == "NO_GO"
    assert decision["go_no_go_reason_category"] == "runtime_environment_blocked"
    assert decision["stage_progression"] == (
        "stop_large_scale_generation_and_preserve_diagnostics"
    )
