from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

import experiments.generative_video_model_probe.temporal_code_isolation_replay_smoke as temporal_module
import workflows.colab_test_request as colab_request_module
from experiments.generative_video_model_probe.temporal_code_isolation_replay_smoke import (
    OWNER_TEMPORAL_ROLE,
    WRONG_TEMPORAL_ROLE,
    build_temporal_isolation_decision,
    build_temporal_isolation_pair_and_identity_records,
    select_temporal_wrong_candidates,
    validate_predictive_replay_source_result,
    validate_temporal_code_isolation_config,
)
from main.methods.state_space_watermark.flow_tubelet_key_code import (
    FlowTubeletKeyCodeConfig,
    FlowTubeletKeyContext,
)
from main.methods.state_space_watermark.replay_inversion import (
    FlowSchedulePoint,
)
from workflows.colab_test_request import (
    TEMPORAL_CODE_ISOLATION_REPLAY_SMOKE_TEST_ID,
    load_colab_test_request,
    run_colab_test_request,
)


pytestmark = pytest.mark.quick
CONFIG_PATH = Path(
    "configs/protocol/sstw_temporal_code_isolation_replay_smoke.json"
)


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_temporal_code_isolation_config_is_frozen():
    validate_temporal_code_isolation_config(_config())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("profile_id", "other"),
        ("replay_step_count", 40),
        ("replay_generation_record_count", 8),
        ("temporal_wrong_candidate_count", 4),
        ("temporal_wrong_candidate_pool_count", 64),
        ("summary_record_count", 35),
        ("pair_record_count", 31),
        ("minimum_temporal_owner_over_wrong_pair_fraction", 0.5),
        ("minimum_temporal_owner_top1_identity_fraction", 0.5),
        ("maximum_prompt_temporal_pair_fraction_gap", 0.5),
        ("stage_progression_allowed", True),
    ],
)
def test_temporal_code_isolation_config_rejects_mutations(field, value):
    config = deepcopy(_config())
    config[field] = value
    with pytest.raises(ValueError):
        validate_temporal_code_isolation_config(config)


def test_temporal_code_isolation_config_rejects_unknown_field():
    config = deepcopy(_config())
    config["adaptive_candidate_selection"] = True
    with pytest.raises(ValueError, match="未声明字段"):
        validate_temporal_code_isolation_config(config)


def _flow_schedule() -> tuple[FlowSchedulePoint, ...]:
    sigmas = [
        1.0,
        0.9818745851516724,
        0.9623856544494629,
        0.9413732290267944,
        0.9186515212059021,
        0.8940030336380005,
        0.8671719431877136,
        0.8378547430038452,
        0.80568927526474,
        0.7702391743659973,
        0.7309743165969849,
        0.6872438788414001,
        0.6382400989532471,
        0.5829479694366455,
        0.5200741291046143,
        0.4479442834854126,
        0.36435219645500183,
        0.26632970571517944,
        0.14978723227977753,
        0.008928571827709675,
        0.0,
    ]
    return tuple(
        FlowSchedulePoint(timestep=index, sigma=sigma)
        for index, sigma in enumerate(sigmas)
    )


def test_temporal_candidate_pool_selects_eight_unique_low_correlations(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        temporal_module,
        "_wrong_owner_generation_key",
        lambda _source, *, extra_context: (
            f"wrong-{extra_context['temporal_code_isolation_candidate_index']}"
        ),
    )
    owner, selected = select_temporal_wrong_candidates(
        {},
        _flow_schedule(),
        correct_key="owner",
        key_context=FlowTubeletKeyContext(
            prompt_digest="a" * 64,
            sampler_signature="scheduler:test",
        ),
        config=_config(),
        tubelet_config=FlowTubeletKeyCodeConfig(),
    )
    assert len(selected) == 8
    assert len(
        {row["temporal_code_signature_digest"] for row in selected}
    ) == 8
    assert all(
        abs(float(row["weighted_code_correlation"])) <= 0.75
        for row in selected
    )
    assert _stable_schedule_signature(owner) not in {
        row["temporal_code_signature_digest"] for row in selected
    }


def _stable_schedule_signature(schedule) -> str:
    return temporal_module._candidate_code_signature(schedule)


def _summary(
    prompt_id: str,
    seed_id: str,
    role: str,
    score: float,
    *,
    candidate_index: int | None,
) -> dict:
    suffix = "owner" if candidate_index is None else f"wrong-{candidate_index}"
    return {
        "record_version": "temporal_code_isolation_replay_smoke",
        "profile_id": "sstw_temporal_code_isolation_replay_smoke",
        "prompt_id": prompt_id,
        "seed_id": seed_id,
        "temporal_code_candidate_role": role,
        "temporal_code_candidate_index": candidate_index,
        "temporal_code_signature_digest": (
            f"{prompt_id}-{seed_id}-{suffix}"
        ),
        "temporal_code_weighted_correlation_to_owner": (
            1.0 if candidate_index is None else 0.1
        ),
        "predictive_replay_path_score": score,
        "predictive_replay_path_quadrature_context_complete": True,
        "predictive_replay_joint_schedule_context_complete": True,
        "predictive_replay_path_weighted_aggregation_applied": True,
        "predictive_replay_path_reliability_mode": (
            "null_forward_key_independent"
        ),
        "trajectory_global_reliability": 0.8,
        "claim_support_status": (
            "temporal_code_isolation_replay_smoke_only_not_paper_evidence"
        ),
    }


def _passing_summaries() -> list[dict]:
    rows = []
    for prompt_id in _config()["heldout_prompt_ids"]:
        for seed_id in _config()["heldout_seed_ids"]:
            rows.append(
                _summary(
                    prompt_id,
                    seed_id,
                    OWNER_TEMPORAL_ROLE,
                    1.0,
                    candidate_index=None,
                )
            )
            rows.extend(
                _summary(
                    prompt_id,
                    seed_id,
                    WRONG_TEMPORAL_ROLE,
                    0.1 - index * 0.01,
                    candidate_index=index,
                )
                for index in range(8)
            )
    return rows


def test_temporal_pair_identity_and_decision_pass_only_reduced_design():
    config = _config()
    summaries = _passing_summaries()
    pairs, identities = build_temporal_isolation_pair_and_identity_records(
        summaries,
        config,
    )
    decision = build_temporal_isolation_decision(
        summaries,
        pairs,
        identities,
        [],
        config,
    )
    assert len(pairs) == 32
    assert len(identities) == 4
    assert all(row["temporal_owner_rank"] == 1 for row in identities)
    assert decision["temporal_code_isolation_gate_ready"] is True
    assert decision["temporal_code_isolation_smoke_decision"] == (
        "temporal_code_isolation_passed_reduced_generation_design_allowed"
    )
    assert decision["generation_execution_allowed"] is False
    assert decision["stage_progression_allowed"] is False
    assert decision["formal_result"] is False


def test_temporal_decision_detects_prompt_dependent_failure():
    config = _config()
    summaries = _passing_summaries()
    for row in summaries:
        if (
            row["prompt_id"] == config["heldout_prompt_ids"][0]
            and row["temporal_code_candidate_role"] == WRONG_TEMPORAL_ROLE
        ):
            row["predictive_replay_path_score"] = 2.0
    pairs, identities = build_temporal_isolation_pair_and_identity_records(
        summaries,
        config,
    )
    decision = build_temporal_isolation_decision(
        summaries,
        pairs,
        identities,
        [],
        config,
    )
    assert decision["temporal_code_isolation_gate_ready"] is False
    assert (
        decision["temporal_owner_over_wrong_pair_fraction_prompt_gap"]
        == 1.0
    )
    assert decision["temporal_code_isolation_smoke_decision"] == (
        "temporal_code_isolation_failed_stop_method"
    )


def _write_predictive_source(root: Path, *, primary="S_path_inv") -> None:
    artifacts = root / "artifacts"
    records = root / "records"
    artifacts.mkdir(parents=True)
    records.mkdir()
    artifacts.joinpath(
        "predictive_trajectory_smoke_decision.json"
    ).write_text(
        json.dumps(
            {
                "profile_id": "sstw_predictive_trajectory_synchronization_smoke",
                "predictive_trajectory_smoke_decision": (
                    "predictive_trajectory_gate_failed_stop_method"
                ),
                "coverage_ready": True,
                "predictive_path_evidence_ready": True,
                "predictive_code_separation_ready": True,
                "summary_record_count": 24,
                "pair_record_count": 20,
                "failure_record_count": 0,
                "primary_predictive_path_statistic": primary,
                "predictive_path_reliability_mode": (
                    "null_forward_key_independent"
                ),
                "predictive_endpoint_llr_role": "diagnostic_only_not_gate",
                "stage_progression_allowed": False,
                "formal_result": False,
            }
        ),
        encoding="utf-8",
    )
    artifacts.joinpath(
        "predictive_trajectory_smoke_manifest.json"
    ).write_text(
        json.dumps(
            {
                "profile_id": "sstw_predictive_trajectory_synchronization_smoke",
                "generation_record_count": 8,
                "summary_record_count": 24,
                "pair_record_count": 20,
                "failure_record_count": 0,
                "primary_predictive_path_statistic": "S_path_inv",
                "predictive_path_reliability_mode": (
                    "null_forward_key_independent"
                ),
                "stage_progression_allowed": False,
                "formal_result": False,
                "generation_result": {
                    "generation_reused_for_replay_only": True
                },
            }
        ),
        encoding="utf-8",
    )
    records.joinpath(
        "predictive_trajectory_summary_records.jsonl"
    ).write_text("{}\n" * 24, encoding="utf-8")
    records.joinpath(
        "predictive_trajectory_pair_records.jsonl"
    ).write_text("{}\n" * 20, encoding="utf-8")
    records.joinpath(
        "predictive_trajectory_failure_records.jsonl"
    ).write_text("", encoding="utf-8")


def test_predictive_replay_source_validation_accepts_repaired_failure(
    tmp_path: Path,
):
    _write_predictive_source(tmp_path)
    result = validate_predictive_replay_source_result(tmp_path, _config())
    assert result["decision"]["primary_predictive_path_statistic"] == (
        "S_path_inv"
    )


def test_predictive_replay_source_validation_rejects_old_llr_source(
    tmp_path: Path,
):
    _write_predictive_source(tmp_path, primary="endpoint_llr")
    with pytest.raises(ValueError, match="decision"):
        validate_predictive_replay_source_result(tmp_path, _config())


def _zip_tree(source: Path, destination: Path) -> None:
    with ZipFile(destination, "w") as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source))


def test_colab_temporal_isolation_requires_resume_package(tmp_path: Path):
    project = tmp_path / "drive" / "SSTW"
    project.mkdir(parents=True)
    source = project / "source.zip"
    source.write_bytes(b"zip")
    request = project / "request.json"
    request.write_text(
        json.dumps(
            {
                "request_schema_version": "sstw_colab_test_request_v1",
                "test_id": TEMPORAL_CODE_ISOLATION_REPLAY_SMOKE_TEST_ID,
                "repository": {
                    "url": "https://github.com/RICHAAARC/SSTW.git",
                    "ref": "main",
                },
                "parameters": {
                    "phase": "no_attack",
                    "run_series_id": "temporal_isolation",
                    "source_package_path": str(source),
                    "resume_package_path": "",
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="resume_package_path"):
        load_colab_test_request(request, project_root=project)


def test_colab_temporal_isolation_dispatches_replay_only_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    project = tmp_path / "drive" / "SSTW"
    project.mkdir(parents=True)
    source_tree = tmp_path / "source"
    source_tree.mkdir()
    source_tree.joinpath("source.txt").write_text("source", encoding="utf-8")
    source_zip = project / "source.zip"
    _zip_tree(source_tree, source_zip)
    replay_tree = tmp_path / "replay"
    _write_predictive_source(replay_tree)
    records = replay_tree / "records"
    videos = replay_tree / "videos"
    videos.mkdir()
    records.joinpath(
        "predictive_trajectory_generation_plan.jsonl"
    ).write_text("{}\n", encoding="utf-8")
    records.joinpath("generation_records.jsonl").write_text(
        "{}\n",
        encoding="utf-8",
    )
    records.joinpath("trajectory_trace.jsonl").write_text(
        "{}\n",
        encoding="utf-8",
    )
    videos.joinpath("video.mp4").write_bytes(b"video")
    replay_zip = project / "replay.zip"
    _zip_tree(replay_tree, replay_zip)
    request = project / "request.json"
    request.write_text(
        json.dumps(
            {
                "request_schema_version": "sstw_colab_test_request_v1",
                "test_id": TEMPORAL_CODE_ISOLATION_REPLAY_SMOKE_TEST_ID,
                "repository": {
                    "url": "https://github.com/RICHAAARC/SSTW.git",
                    "ref": "main",
                },
                "parameters": {
                    "phase": "no_attack",
                    "run_series_id": "temporal_isolation",
                    "source_package_path": str(source_zip),
                    "resume_package_path": str(replay_zip),
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        colab_request_module,
        "_minimal_signed_trajectory_generation_model_ids",
        lambda _root: ["model"],
    )
    observed = {}

    def runner(source_root, output_root, replay_source_root):
        observed["source_root"] = source_root
        observed["replay_source_root"] = replay_source_root
        output_root.joinpath("result.json").write_text(
            "{}",
            encoding="utf-8",
        )
        return {
            "temporal_code_isolation_smoke_decision": "diagnostic",
            "formal_result": False,
            "stage_progression_allowed": False,
        }

    result = run_colab_test_request(
        request,
        project_root=project,
        repo_root=Path.cwd(),
        local_workspace_root=tmp_path / "workspace",
        local_package_cache_root=tmp_path / "cache",
        temporal_code_isolation_runner=runner,
    )
    assert observed["source_root"].is_dir()
    assert observed["replay_source_root"].is_dir()
    assert result["diagnostic_decision"]["formal_result"] is False
    assert Path(result["drive_result_zip"]).is_file()
