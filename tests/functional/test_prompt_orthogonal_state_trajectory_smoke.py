"""纯 CPU 验证 prompt-orthogonal state-trajectory smoke 的冻结边界。"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

import workflows.colab_test_request as colab_request_module
from experiments.generative_video_model_probe.prompt_orthogonal_state_trajectory_smoke import (
    CLEAN_VARIANT,
    OWNER_ROLE,
    PROFILE_ID,
    WATERMARKED_VARIANT,
    WRONG_ROLE,
    build_prompt_orthogonal_decision,
    build_prompt_orthogonal_generation_plan,
    build_prompt_orthogonal_identity_records,
    build_prompt_orthogonal_pair_records,
    validate_prompt_orthogonal_smoke_config,
    validate_prompt_orthogonal_generation_execution,
    validate_temporal_failure_source,
)
from workflows.colab_test_request import (
    PROMPT_ORTHOGONAL_STATE_TRAJECTORY_SMOKE_TEST_ID,
    load_colab_test_request,
    run_colab_test_request,
)


pytestmark = pytest.mark.quick
CONFIG_PATH = Path(
    "configs/protocol/sstw_prompt_orthogonal_state_trajectory_smoke.json"
)


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _zip_tree(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(target, "w") as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source))


def test_prompt_orthogonal_config_is_frozen() -> None:
    validate_prompt_orthogonal_smoke_config(_config())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("profile_id", "other"),
        ("generation_step_count", 20),
        ("replay_step_count", 40),
        ("smoke_generation_record_count", 12),
        ("summary_record_count", 71),
        ("pair_record_count", 31),
        ("wrong_owner_candidate_count", 4),
        ("prompt_orthogonal_replay_reliability_mode", "candidate_specific"),
        ("lambda_max", 0.24),
        ("minimum_owner_over_wrong_pair_fraction", 0.5),
        ("minimum_owner_top1_identity_fraction", 0.5),
        ("maximum_prompt_pair_fraction_gap", 0.5),
        ("minimum_replay_reliability", 0.0),
        ("minimum_watermarked_over_clean_owner_fraction", 0.5),
        ("maximum_clean_owner_top1_fraction", 1.0),
        ("stage_progression_allowed", True),
        ("fixed_fpr_evaluation_allowed", True),
    ],
)
def test_prompt_orthogonal_config_rejects_mutations(
    field: str,
    value: object,
) -> None:
    config = deepcopy(_config())
    config[field] = value
    with pytest.raises(ValueError):
        validate_prompt_orthogonal_smoke_config(config)


def test_prompt_orthogonal_config_rejects_unknown_field() -> None:
    config = deepcopy(_config())
    config["adaptive_strength_selection"] = True
    with pytest.raises(ValueError, match="未声明字段"):
        validate_prompt_orthogonal_smoke_config(config)


def _validated_source() -> dict:
    prompts = [
        {
            "prompt_id": f"probe_paper_paper_master_prompt_{index:03d}",
            "prompt_text": f"prompt {index}",
            "prompt_suite_role": (
                "calibration" if index < 3 else "test"
            ),
        }
        for index in range(1, 5)
    ]
    seeds = [
        {
            "seed_id": f"probe_paper_paper_master_test_seed_{index:02d}",
            "seed_value": 100 + index,
            "generation_seed_random": 100 + index,
            "prompt_suite_role": "test",
            "seed_suite_role": "test",
        }
        for index in range(1, 3)
    ]
    return {
        "prompt_suite": {"prompts": prompts, "seeds": seeds},
        "generation_rows": [
            {
                "prompt_id": "probe_paper_paper_master_prompt_001",
                "seed_id": "probe_paper_paper_master_calibration_seed_01",
            },
            {
                "prompt_id": "probe_paper_paper_master_prompt_002",
                "seed_id": "probe_paper_paper_master_calibration_seed_02",
            },
        ],
    }


def test_prompt_orthogonal_plan_is_exact_2x2x2_and_held_out() -> None:
    plan = build_prompt_orthogonal_generation_plan(
        _validated_source(),
        _config(),
    )
    assert len(plan) == 8
    assert {
        row["trajectory_carrier_variant_id"] for row in plan
    } == {WATERMARKED_VARIANT, CLEAN_VARIANT}
    assert {
        (row["prompt_id"], row["seed_id"]) for row in plan
    } == {
        (prompt_id, seed_id)
        for prompt_id in _config()["heldout_prompt_ids"]
        for seed_id in _config()["heldout_seed_ids"]
    }
    assert all(row["lambda_max"] == 0.12 for row in plan)
    assert all(
        row["attacked_phase_execution_allowed"] is False
        and row["stage_progression_allowed"] is False
        for row in plan
    )


def test_prompt_orthogonal_plan_rejects_source_identity_overlap() -> None:
    validated = _validated_source()
    validated["generation_rows"][0]["prompt_id"] = (
        _config()["heldout_prompt_ids"][0]
    )
    with pytest.raises(ValueError, match="held-out identity"):
        build_prompt_orthogonal_generation_plan(validated, _config())


def _generation_execution_fixture(root: Path) -> dict:
    rows: list[dict] = []
    traces: list[dict] = []
    for index in range(8):
        watermarked = index % 2 == 0
        trace_id = f"trace-{index}"
        rows.append(
            {
                "generation_status": "success",
                "trajectory_trace_id": trace_id,
                "trajectory_carrier_variant_id": (
                    WATERMARKED_VARIANT if watermarked else CLEAN_VARIANT
                ),
            }
        )
        traces.append(
            {
                "trajectory_trace_id": trace_id,
                "prompt_orthogonal_scheduler_control_dtype": "float32",
                "flow_runtime_step_formal_context_complete": watermarked,
                "prompt_orthogonal_inactive_phase_noop": (
                    False if watermarked else None
                ),
                "prompt_orthogonal_norm_guard_passed": (
                    True if watermarked else None
                ),
                "prompt_orthogonal_energy_guard_passed": (
                    True if watermarked else None
                ),
                "prompt_orthogonal_direction_guard_passed": (
                    True if watermarked else None
                ),
                "prompt_orthogonal_finite_precision_projection_status": (
                    "direct_actual_delta_pass" if watermarked else None
                ),
                "prompt_orthogonal_finite_precision_projection_scale": (
                    1.0 if watermarked else None
                ),
                "prompt_orthogonal_finite_precision_projection_attempt_count": (
                    1 if watermarked else None
                ),
                "prompt_orthogonal_candidate_delta_norm": (
                    0.1 if watermarked else None
                ),
                "prompt_orthogonal_intended_delta_norm_before_projection": (
                    0.1 if watermarked else None
                ),
                "velocity_constraint_delta_norm": (
                    0.1 if watermarked else 0.0
                ),
                "prompt_orthogonal_finite_precision_backoff_count": (
                    0 if watermarked else None
                ),
                "endpoint_control_enabled": False,
            }
        )
    _write_jsonl(root / "records" / "generation_records.jsonl", rows)
    _write_jsonl(root / "records" / "trajectory_trace.jsonl", traces)
    return {
        "decision": {
            "implementation_decision": "PASS",
            "mechanism_decision": (
                "GENERATION_READY_NO_ATTACK_REPLAY_PENDING"
            ),
        }
    }


def test_generation_execution_requires_shared_float32_scheduler_control(
    tmp_path: Path,
) -> None:
    result = _generation_execution_fixture(tmp_path)
    validate_prompt_orthogonal_generation_execution(tmp_path, result)
    trace_path = tmp_path / "records" / "trajectory_trace.jsonl"
    traces = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    traces[1]["prompt_orthogonal_scheduler_control_dtype"] = "bfloat16"
    _write_jsonl(trace_path, traces)
    with pytest.raises(RuntimeError, match="scheduler dtype"):
        validate_prompt_orthogonal_generation_execution(tmp_path, result)


def _temporal_source(root: Path) -> Path:
    config = _config()
    artifacts = root / "artifacts"
    records = root / "records"
    _write_json(
        artifacts / "temporal_code_isolation_smoke_decision.json",
        {
            "profile_id": config["required_temporal_source_profile_id"],
            "temporal_code_isolation_smoke_decision": config[
                "required_temporal_source_decision"
            ],
            "temporal_code_isolation_gate_ready": False,
            "generation_execution_allowed": False,
            "summary_record_count": 36,
            "pair_record_count": 32,
            "identity_record_count": 4,
            "stage_progression_allowed": False,
            "formal_result": False,
        },
    )
    _write_json(
        artifacts / "temporal_code_isolation_smoke_manifest.json",
        {
            "profile_id": config["required_temporal_source_profile_id"],
            "summary_record_count": 36,
            "pair_record_count": 32,
            "identity_record_count": 4,
            "failure_record_count": 0,
            "generation_execution_allowed": False,
            "stage_progression_allowed": False,
            "formal_result": False,
        },
    )
    profile = config["required_temporal_source_profile_id"]
    _write_jsonl(
        records / "temporal_code_isolation_summary_records.jsonl",
        [{"profile_id": profile, "record": index} for index in range(36)],
    )
    _write_jsonl(
        records / "temporal_code_isolation_pair_records.jsonl",
        [{"profile_id": profile, "record": index} for index in range(32)],
    )
    _write_jsonl(
        records / "temporal_code_isolation_identity_records.jsonl",
        [{"profile_id": profile, "record": index} for index in range(4)],
    )
    _write_jsonl(
        records / "temporal_code_isolation_failure_records.jsonl",
        [],
    )
    return root


def test_temporal_failure_source_is_content_bound_and_fail_closed(
    tmp_path: Path,
) -> None:
    root = _temporal_source(tmp_path / "temporal")
    validated = validate_temporal_failure_source(root, _config())
    assert validated["decision"]["formal_result"] is False
    assert validated["manifest"]["failure_record_count"] == 0

    pair_path = root / "records" / "temporal_code_isolation_pair_records.jsonl"
    rows = pair_path.read_text(encoding="utf-8").splitlines()
    pair_path.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="record"):
        validate_temporal_failure_source(root, _config())


def _summary(
    prompt_id: str,
    seed_id: str,
    variant: str,
    role: str,
    score: float,
    *,
    candidate_index: int | None,
) -> dict:
    return {
        "profile_id": PROFILE_ID,
        "prompt_id": prompt_id,
        "seed_id": seed_id,
        "trajectory_carrier_variant_id": variant,
        "prompt_orthogonal_candidate_role": role,
        "prompt_orthogonal_wrong_candidate_index": candidate_index,
        "prompt_orthogonal_matched_cosine_score": score,
        "prompt_orthogonal_vector_context_complete": True,
        "prompt_orthogonal_key_independent_trace_complete": True,
        "prompt_orthogonal_fixed_trace_key_independent": True,
        "prompt_orthogonal_base_velocity_call_count": 20,
        "prompt_orthogonal_trace_construction_base_velocity_call_count": 20,
        "prompt_orthogonal_total_base_velocity_call_count": 40,
        "prompt_orthogonal_candidate_count": 9,
        "prompt_orthogonal_replay_reliability_mode": (
            "base_predicted_transition_residual_key_independent"
        ),
        "prompt_orthogonal_plane_construction_device": "cpu",
        "minimum_replay_reliability": 0.8,
    }


def _passing_summaries() -> list[dict]:
    rows: list[dict] = []
    for prompt_id in _config()["heldout_prompt_ids"]:
        for seed_id in _config()["heldout_seed_ids"]:
            rows.append(
                _summary(
                    prompt_id,
                    seed_id,
                    WATERMARKED_VARIANT,
                    OWNER_ROLE,
                    0.9,
                    candidate_index=None,
                )
            )
            rows.extend(
                _summary(
                    prompt_id,
                    seed_id,
                    WATERMARKED_VARIANT,
                    WRONG_ROLE,
                    0.1 - index * 0.01,
                    candidate_index=index,
                )
                for index in range(8)
            )
            rows.append(
                _summary(
                    prompt_id,
                    seed_id,
                    CLEAN_VARIANT,
                    OWNER_ROLE,
                    0.05,
                    candidate_index=None,
                )
            )
            rows.extend(
                _summary(
                    prompt_id,
                    seed_id,
                    CLEAN_VARIANT,
                    WRONG_ROLE,
                    0.2 + index * 0.01,
                    candidate_index=index,
                )
                for index in range(8)
            )
    return rows


def _records_for_decision(
    summaries: list[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    return (
        build_prompt_orthogonal_pair_records(summaries),
        build_prompt_orthogonal_identity_records(
            summaries,
            variant=WATERMARKED_VARIANT,
        ),
        build_prompt_orthogonal_identity_records(
            summaries,
            variant=CLEAN_VARIANT,
        ),
    )


def test_prompt_orthogonal_decision_passes_only_mechanism_smoke() -> None:
    summaries = _passing_summaries()
    pairs, identities, clean = _records_for_decision(summaries)
    decision = build_prompt_orthogonal_decision(
        summaries,
        pairs,
        identities,
        clean,
        [],
        _config(),
    )
    assert len(summaries) == 72
    assert len(pairs) == 32
    assert len(identities) == 4
    assert len(clean) == 4
    assert decision["prompt_orthogonal_gate_ready"] is True
    assert decision["prompt_orthogonal_smoke_decision"] == (
        "prompt_orthogonal_mechanism_smoke_passed_"
        "independent_calibration_design_allowed"
    )
    assert decision["formal_result"] is False
    assert decision["stage_progression_allowed"] is False


def test_prompt_orthogonal_decision_rejects_prompt_dependent_failure() -> None:
    summaries = _passing_summaries()
    failed_prompt = _config()["heldout_prompt_ids"][0]
    for row in summaries:
        if (
            row["prompt_id"] == failed_prompt
            and row["trajectory_carrier_variant_id"] == WATERMARKED_VARIANT
            and row["prompt_orthogonal_candidate_role"] == WRONG_ROLE
        ):
            row["prompt_orthogonal_matched_cosine_score"] = 1.0
    pairs, identities, clean = _records_for_decision(summaries)
    decision = build_prompt_orthogonal_decision(
        summaries,
        pairs,
        identities,
        clean,
        [],
        _config(),
    )
    assert decision["prompt_orthogonal_gate_ready"] is False
    assert decision["prompt_orthogonal_prompt_pair_fraction_gap"] == 1.0
    assert decision["prompt_orthogonal_smoke_decision"] == (
        "prompt_orthogonal_mechanism_smoke_failed_stop_instance"
    )


def test_prompt_orthogonal_decision_requires_complete_shared_trace() -> None:
    summaries = _passing_summaries()
    summaries[0]["prompt_orthogonal_key_independent_trace_complete"] = False
    pairs, identities, clean = _records_for_decision(summaries)
    decision = build_prompt_orthogonal_decision(
        summaries,
        pairs,
        identities,
        clean,
        [],
        _config(),
    )
    assert decision["coverage_ready"] is False
    assert decision["prompt_orthogonal_gate_ready"] is False


def test_colab_prompt_orthogonal_handler_requires_two_sources_and_packages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "drive" / "SSTW"
    source_tree = tmp_path / "controlled"
    source_tree.mkdir(parents=True)
    _write_json(source_tree / "placeholder.json", {})
    source_zip = project / "inputs" / "controlled.zip"
    _zip_tree(source_tree, source_zip)
    temporal_tree = _temporal_source(tmp_path / "temporal")
    temporal_zip = project / "inputs" / "temporal.zip"
    _zip_tree(temporal_tree, temporal_zip)
    request_path = project / "requests" / "request.json"
    payload = {
        "request_schema_version": "sstw_colab_test_request_v1",
        "test_id": PROMPT_ORTHOGONAL_STATE_TRAJECTORY_SMOKE_TEST_ID,
        "repository": {
            "url": "https://github.com/RICHAAARC/SSTW.git",
            "ref": "main",
        },
        "parameters": {
            "phase": "no_attack",
            "run_series_id": "prompt_orthogonal_smoke",
            "source_package_path": str(source_zip),
            "resume_package_path": str(temporal_zip),
        },
    }
    _write_json(request_path, payload)
    monkeypatch.setattr(
        colab_request_module,
        "_minimal_signed_trajectory_generation_model_ids",
        lambda _root: ["Wan-AI/Wan2.1-T2V-1.3B-Diffusers"],
    )
    observed: dict[str, Path] = {}

    def runner(source_root, output_root, temporal_source_root):
        observed["source_root"] = source_root
        observed["temporal_source_root"] = temporal_source_root
        _write_json(output_root / "result.json", {"formal_result": False})
        return {
            "prompt_orthogonal_smoke_decision": "diagnostic",
            "formal_result": False,
            "stage_progression_allowed": False,
        }

    result = run_colab_test_request(
        request_path,
        project_root=project,
        repo_root=Path.cwd(),
        local_workspace_root=tmp_path / "workspace",
        local_package_cache_root=tmp_path / "cache",
        prompt_orthogonal_state_trajectory_runner=runner,
    )
    assert observed["source_root"].is_dir()
    assert observed["temporal_source_root"].is_dir()
    assert Path(result["drive_result_zip"]).is_file()
    assert Path(result["drive_result_manifest"]).is_file()
    assert result["diagnostic_decision"]["formal_result"] is False

    payload["parameters"]["resume_package_path"] = ""
    _write_json(request_path, payload)
    with pytest.raises(ValueError, match="resume_package_path"):
        load_colab_test_request(request_path, project_root=project)
