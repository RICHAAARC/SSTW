from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

import experiments.generative_video_model_probe.formal_flow_evidence_runner as formal_flow_runner
import experiments.generative_video_model_probe.predictive_trajectory_synchronization_smoke as predictive_smoke_module
import main.methods.state_space_watermark.predictive_trajectory_carrier as predictive_carrier_module
import main.methods.state_space_watermark.wan_flow_replay_backend as wan_replay_module
from experiments.generative_video_model_probe.predictive_trajectory_synchronization_smoke import (
    NONNEGATIVE_VARIANT,
    PREDICTIVE_VARIANT,
    build_predictive_decision,
    build_predictive_pair_records,
    build_predictive_trajectory_generation_plan,
    validate_predictive_generation_execution,
    validate_predictive_trajectory_config,
)
from main.methods.state_space_watermark.flow_tubelet_key_code import (
    FlowTubeletKeyContext,
    FlowTubeletKeyCodeConfig,
    flow_phase_weight,
)
from main.methods.state_space_watermark.predictive_trajectory_carrier import (
    PredictiveTrajectoryCarrierConfig,
    apply_predictive_trajectory_constraint,
    build_predictive_trajectory_schedule,
    predictive_trajectory_weighted_code_correlation,
)
from main.methods.state_space_watermark.velocity_field_constraint import (
    VelocityControlContext,
    VelocityFieldConstraintConfig,
)
from main.methods.state_space_watermark.watermark_key_derivation import (
    derive_wrong_key_control_text,
)
from main.methods.state_space_watermark.wan_flow_replay_backend import (
    WanFlowReplayResult,
)
from workflows.colab_test_request import (
    PREDICTIVE_TRAJECTORY_SYNCHRONIZATION_SMOKE_TEST_ID,
    load_colab_test_request,
    run_colab_test_request,
)


pytestmark = pytest.mark.quick
CONFIG_PATH = Path(
    "configs/protocol/sstw_predictive_trajectory_synchronization_smoke.json"
)


class _NumpyTensor:
    __array_priority__ = 1000

    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float64)

    @property
    def shape(self):
        return self.values.shape

    @property
    def dtype(self):
        return self.values.dtype

    def detach(self):
        return self

    def float(self):
        return self

    def norm(self):
        return _NumpyTensor(np.linalg.norm(self.values))

    def square(self):
        return _NumpyTensor(np.square(self.values))

    def sum(self):
        return _NumpyTensor(np.sum(self.values))

    def mean(self):
        return _NumpyTensor(np.mean(self.values))

    def sqrt(self):
        return _NumpyTensor(np.sqrt(self.values))

    def abs(self):
        return _NumpyTensor(np.abs(self.values))

    def reshape(self, *shape):
        return _NumpyTensor(self.values.reshape(*shape))

    def clamp_min(self, minimum):
        return _NumpyTensor(np.maximum(self.values, float(minimum)))

    def item(self):
        return self.values.item()

    def to(self, *args, **kwargs):
        return self

    def __add__(self, other):
        return _NumpyTensor(self.values + self._values(other))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return _NumpyTensor(self.values - self._values(other))

    def __rsub__(self, other):
        return _NumpyTensor(self._values(other) - self.values)

    def __mul__(self, other):
        return _NumpyTensor(self.values * self._values(other))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return _NumpyTensor(self.values / self._values(other))

    def __matmul__(self, other):
        return _NumpyTensor(self.values @ self._values(other))

    @staticmethod
    def _values(value):
        return value.values if isinstance(value, _NumpyTensor) else value


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _schedule(key: str):
    phases = [(index + 0.5) / 20 for index in range(20)]
    tubelet = FlowTubeletKeyCodeConfig()
    weights = [
        flow_phase_weight(phase, tubelet) / 20 for phase in phases
    ]
    return build_predictive_trajectory_schedule(
        key_text=key,
        key_context_digest="a" * 64,
        flow_phases=phases,
        active_weights=weights,
    )


def test_predictive_schedule_has_multiple_signed_segments_and_zero_mean():
    schedule = _schedule("owner-key-a")
    assert schedule.active_phase_count == 10
    assert len(schedule.phase_codebook_signs) == 8
    assert schedule.phase_codebook_signs.count(1) == 4
    assert schedule.phase_codebook_signs.count(-1) == 4
    assert any(value > 0.0 for value in schedule.codes)
    assert any(value < 0.0 for value in schedule.codes)
    assert abs(schedule.weighted_residual) <= 1e-10
    assert schedule.minimum_active_code_magnitude >= 0.25
    assert schedule.weighted_code_energy >= 0.20


def test_predictive_owner_and_wrong_code_are_distinct_on_same_grid():
    owner = _schedule("key5")
    wrong = _schedule("wrong5")
    correlation = predictive_trajectory_weighted_code_correlation(
        owner,
        wrong,
    )
    assert owner.phase_function_digest != wrong.phase_function_digest
    assert owner.raw_signs != wrong.raw_signs
    assert abs(correlation) <= 0.75


def test_predictive_schedule_is_noncollapsed_for_frozen_wan_20_step_grid():
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
    phases = [
        (
            (sigmas[index] + sigmas[index + 1]) / 2.0
            - sigmas[0]
        )
        / (sigmas[-1] - sigmas[0])
        for index in range(20)
    ]
    tubelet = FlowTubeletKeyCodeConfig()
    weights = [
        abs(sigmas[index + 1] - sigmas[index])
        * flow_phase_weight(phases[index], tubelet)
        for index in range(20)
    ]
    for index in range(128):
        schedule = build_predictive_trajectory_schedule(
            key_text=f"key-{index}",
            key_context_digest="b" * 64,
            flow_phases=phases,
            active_weights=weights,
        )
        assert schedule.active_phase_count == 7
        assert schedule.minimum_active_code_magnitude >= 0.25
        assert schedule.weighted_code_energy >= 0.20
        assert abs(schedule.weighted_residual) <= 1e-10


def test_predictive_wrong_owner_control_has_frozen_code_separation_search():
    owner = _schedule("owner-key")
    candidates = [
        derive_wrong_key_control_text(
            b"x" * 32,
            key_id="owner",
            generation_model_id="model",
            prompt_id="prompt",
            seed_id="seed",
            extra_context={
                "predictive_wrong_owner_key_control_candidate_index": index
            },
        )
        for index in range(32)
    ]
    correlations = [
        predictive_trajectory_weighted_code_correlation(
            owner,
            _schedule(candidate),
        )
        for candidate in candidates
    ]
    selected = next(
        index
        for index, correlation in enumerate(correlations)
        if abs(correlation) <= 0.75
    )
    assert selected == min(
        index
        for index, correlation in enumerate(correlations)
        if abs(correlation) <= 0.75
    )
    assert len(set(candidates)) == 32


def test_predictive_replay_entry_uses_the_formal_runner_symbol():
    assert (
        predictive_smoke_module._run_attacked_video_replay_for_model
        is formal_flow_runner._run_attacked_video_replay_for_model
    )
    with pytest.raises(TypeError):
        predictive_smoke_module._run_attacked_video_replay_for_model()


def test_predictive_replay_dispatch_preserves_endpoint_disabled_control(
    monkeypatch: pytest.MonkeyPatch,
):
    observed = []

    def fake_run(*args, **kwargs):
        observed.append(("run", kwargs["endpoint_control_enabled"]))
        return "run-result"

    def fake_evaluate(*args, **kwargs):
        observed.append(("evaluate", kwargs["endpoint_control_enabled"]))
        return "evaluate-result"

    monkeypatch.setattr(
        formal_flow_runner,
        "run_wan_attacked_video_replay",
        fake_run,
    )
    monkeypatch.setattr(
        formal_flow_runner,
        "evaluate_fixed_wan_replay_hypothesis_for_key",
        fake_evaluate,
    )
    context = FlowTubeletKeyContext(
        prompt_digest="a" * 64,
        sampler_signature="scheduler:test",
    )
    run_result = formal_flow_runner._run_attacked_video_replay_for_model(
        object(),
        "video.mp4",
        prompt="prompt",
        key_text="key",
        key_context=context,
        likelihood_config=object(),
        endpoint_control_enabled=False,
    )
    evaluate_result = formal_flow_runner._evaluate_fixed_replay_hypothesis_for_key(
        object(),
        object(),
        prompt="prompt",
        key_text="key",
        key_context=context,
        endpoint_control_enabled=False,
    )
    assert run_result == "run-result"
    assert evaluate_result == "evaluate-result"
    assert observed == [("run", False), ("evaluate", False)]


def test_wan_fixed_replay_inherits_endpoint_disabled_result(
    monkeypatch: pytest.MonkeyPatch,
):
    observed = {}

    monkeypatch.setattr(
        wan_replay_module,
        "WanPromptConditionedVelocity",
        lambda *args, **kwargs: object(),
    )

    class FakeKeyedVelocity:
        def __init__(self, *args, **kwargs):
            observed["endpoint_control_enabled"] = kwargs[
                "endpoint_control_enabled"
            ]

    monkeypatch.setattr(
        wan_replay_module,
        "WanKeyConditionedVelocity",
        FakeKeyedVelocity,
    )
    monkeypatch.setattr(
        wan_replay_module,
        "evaluate_candidate_on_fixed_inversion",
        lambda *args, **kwargs: "hypothesis",
    )
    monkeypatch.setattr(
        wan_replay_module,
        "score_replay_trajectory_for_key",
        lambda *args, **kwargs: {},
    )
    replay = WanFlowReplayResult(
        endpoint_evidence=object(),
        path_evidence={},
        replay_uncertainty=object(),
        replay_trajectories=(object(),),
        endpoint_metadata={},
        replay_step_counts=(20,),
        endpoint_latent=object(),
        primary_schedule=(),
        primary_replay_index=0,
        replay_likelihood_config=object(),
        endpoint_control_enabled=False,
    )
    hypothesis, _path = (
        wan_replay_module.evaluate_fixed_wan_replay_hypothesis_for_key(
            object(),
            replay,
            prompt="prompt",
            key_text="key",
        )
    )
    assert hypothesis == "hypothesis"
    assert observed["endpoint_control_enabled"] is False


def test_predictive_constraint_has_no_independent_endpoint_channel():
    generator = np.random.default_rng(seed=3)
    model_output = _NumpyTensor(generator.normal(size=(1, 2, 2, 2, 2)))
    sample = _NumpyTensor(generator.normal(size=(1, 2, 2, 2, 2)))
    direction = _NumpyTensor(generator.normal(size=(1, 2, 2, 2, 2)))
    direction = direction / direction.norm()
    constrained, record = apply_predictive_trajectory_constraint(
        model_output,
        sample,
        direction,
        ac_code=-0.8,
        flow_phase=0.5,
        config=VelocityFieldConstraintConfig(),
        tubelet_config=FlowTubeletKeyCodeConfig(),
        carrier_config=PredictiveTrajectoryCarrierConfig(),
        control_context=VelocityControlContext(
            delta_sigma=-0.05,
            cumulative_control_energy=0.0,
            cumulative_reference_energy=10.0,
            remaining_step_count=10,
        ),
    )
    assert constrained.shape == model_output.shape
    assert record["endpoint_control_enabled"] is False
    assert record["predictive_trajectory_norm_guard_passed"] is True
    assert record["predictive_trajectory_energy_guard_passed"] is True
    assert (
        record["predictive_trajectory_observability_mode"]
        == "bounded_terminal_residual_from_phase_conditioned_carrier"
    )


def test_predictive_budget_guard_accepts_only_float_reduction_roundoff():
    budget = 1.0
    next_float32 = float(
        np.nextafter(
            np.float32(budget),
            np.float32(np.inf),
        )
    )
    assert next_float32 > budget + 1e-10
    assert predictive_carrier_module._predictive_budget_guard_passed(
        next_float32,
        budget,
    )
    assert not predictive_carrier_module._predictive_budget_guard_passed(
        budget * 1.00002,
        budget,
    )
    assert not predictive_carrier_module._predictive_budget_guard_passed(
        float("nan"),
        budget,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("generation_step_count", 8),
        ("replay_step_count", 40),
        ("lambda_max", 0.24),
        ("phase_segment_count", 4),
        ("minimum_active_phase_count", 2),
        ("maximum_absolute_code_correlation", 1.0),
        ("wrong_owner_key_control_candidate_count", 16),
        ("minimum_predictive_correct_over_wrong_fraction", 0.5),
        ("minimum_predictive_over_nonnegative_margin_fraction", 0.5),
        ("minimum_replay_reliability", 0.0),
        ("endpoint_gate_execution_allowed", True),
        ("stage_progression_allowed", True),
        ("claim_support_status", "formal_paper_evidence"),
    ],
)
def test_predictive_config_mutations_fail_closed(field: str, value: object):
    config = _config()
    config[field] = value
    with pytest.raises(ValueError):
        validate_predictive_trajectory_config(config)


def test_predictive_plan_is_four_heldout_identities_by_two_variants():
    prompts = [
        {
            "prompt_id": prompt_id,
            "prompt_text": f"text for {prompt_id}",
            "prompt_category": "motion",
            "prompt_suite_role": "probe_paper",
            "motion_pattern_id": prompt_id,
            "motion_claim_role": "positive_motion",
            "motion_calibration_role": None,
            "prompt_negative_text": "",
        }
        for prompt_id in _config()["heldout_prompt_ids"]
    ]
    seeds = [
        {
            "seed_id": seed_id,
            "prompt_suite_role": "probe_paper",
            "seed_value": 2000 + index,
        }
        for index, seed_id in enumerate(_config()["heldout_seed_ids"])
    ]
    plan = build_predictive_trajectory_generation_plan(
        {
            "prompt_suite": {"prompts": prompts, "seeds": seeds},
            "generation_rows": [
                {
                    "prompt_id": "prior_prompt",
                    "seed_id": "prior_seed",
                }
            ],
        },
        _config(),
    )
    assert len(plan) == 8
    assert {
        row["trajectory_carrier_variant_id"] for row in plan
    } == {PREDICTIVE_VARIANT, NONNEGATIVE_VARIANT}
    assert len(
        {(row["prompt_id"], row["seed_id"]) for row in plan}
    ) == 4
    assert {row["lambda_max"] for row in plan} == {0.12}
    assert all(row["stage_progression_allowed"] is False for row in plan)


def test_predictive_plan_rejects_heldout_overlap_with_source_generation():
    config = _config()
    prompts = [
        {
            "prompt_id": prompt_id,
            "prompt_text": prompt_id,
            "prompt_category": "motion",
            "prompt_suite_role": "probe_paper",
            "motion_pattern_id": prompt_id,
            "motion_claim_role": "positive_motion",
            "prompt_negative_text": "",
        }
        for prompt_id in config["heldout_prompt_ids"]
    ]
    seeds = [
        {
            "seed_id": seed_id,
            "prompt_suite_role": "probe_paper",
            "seed_value": 2201 + index,
        }
        for index, seed_id in enumerate(config["heldout_seed_ids"])
    ]
    with pytest.raises(ValueError, match="held-out 身份已出现在"):
        build_predictive_trajectory_generation_plan(
            {
                "prompt_suite": {"prompts": prompts, "seeds": seeds},
                "generation_rows": [
                    {
                        "prompt_id": config["heldout_prompt_ids"][0],
                        "seed_id": "prior_seed",
                    }
                ],
            },
            config,
        )


def test_predictive_generation_failure_stops_before_replay(tmp_path: Path):
    with pytest.raises(RuntimeError, match="runtime decision 未就绪"):
        validate_predictive_generation_execution(
            tmp_path,
            [],
            {
                "generation_record_count": 1,
                "trajectory_record_count": 0,
                "decision": {
                    "implementation_decision": "FAIL",
                    "mechanism_decision": "FAIL",
                },
            },
        )


def test_predictive_generation_validation_accepts_complete_20_step_records(
    tmp_path: Path,
):
    plan = [
        {
            "predictive_trajectory_plan_record_id": f"plan-{index}",
            "trajectory_carrier_variant_id": (
                PREDICTIVE_VARIANT if index < 4 else NONNEGATIVE_VARIANT
            ),
        }
        for index in range(8)
    ]
    generation_rows = [
        {
            **row,
            "generation_status": "success",
            "colab_runtime_profile": (
                "predictive_trajectory_synchronization_smoke"
            ),
        }
        for row in plan
    ]
    trajectory_rows = []
    for row in plan:
        for _step_index in range(20):
            step = {
                "predictive_trajectory_plan_record_id": row[
                    "predictive_trajectory_plan_record_id"
                ],
                "endpoint_control_enabled": False,
            }
            if row["trajectory_carrier_variant_id"] == PREDICTIVE_VARIANT:
                step.update(
                    {
                        "predictive_trajectory_noncollapse_verified": True,
                        "predictive_trajectory_inactive_phase_noop": True,
                    }
                )
            trajectory_rows.append(step)
    records = tmp_path / "records"
    records.mkdir()
    (records / "generation_records.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in generation_rows),
        encoding="utf-8",
    )
    (records / "trajectory_trace.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in trajectory_rows),
        encoding="utf-8",
    )
    validate_predictive_generation_execution(
        tmp_path,
        plan,
        {
            "generation_record_count": 8,
            "trajectory_record_count": 160,
            "decision": {
                "implementation_decision": "PASS",
                "mechanism_decision": (
                    "GENERATION_READY_NO_ATTACK_REPLAY_PENDING"
                ),
            },
        },
    )


def _summary(
    prompt_id: str,
    seed_id: str,
    variant: str,
    key_role: str,
    llr: float,
) -> dict:
    return {
        "prompt_id": prompt_id,
        "seed_id": seed_id,
        "trajectory_carrier_variant_id": variant,
        "candidate_key_role": key_role,
        "predictive_replay_log_likelihood_ratio": llr,
        "trajectory_global_reliability": 0.8,
        "predictive_wrong_owner_key_control_candidate_index": 2,
        "predictive_owner_wrong_weighted_code_correlation": (
            0.2 if variant == PREDICTIVE_VARIANT else None
        ),
    }


def test_predictive_gate_uses_only_forward_llr_control_and_reliability():
    summaries = []
    for index in range(4):
        prompt_id = f"prompt_{index // 2}"
        seed_id = f"seed_{index % 2}"
        signed_margin = 1.0 if index < 3 else -0.3
        control_margin = 0.1 if index < 3 else -0.2
        for variant, margin in (
            (PREDICTIVE_VARIANT, signed_margin),
            (NONNEGATIVE_VARIANT, control_margin),
        ):
            summaries.extend(
                [
                    _summary(
                        prompt_id,
                        seed_id,
                        variant,
                        "correct_owner_key",
                        margin,
                    ),
                    _summary(
                        prompt_id,
                        seed_id,
                        variant,
                        "wrong_owner_key",
                        0.0,
                    ),
                ]
            )
    pairs = build_predictive_pair_records(summaries)
    decision = build_predictive_decision(
        summaries,
        pairs,
        [],
        _config(),
    )
    assert len(pairs) == 12
    assert decision["predictive_correct_over_wrong_fraction"] == 0.75
    assert (
        decision["predictive_over_nonnegative_margin_fraction"] == 0.75
    )
    assert decision["predictive_trajectory_gate_ready"] is True
    assert decision["endpoint_gate_executed"] is False
    assert decision["state_space_posterior_executed"] is False
    assert decision["stage_progression_allowed"] is False


def test_predictive_gate_fails_closed_when_code_correlation_is_missing():
    summaries = []
    for index in range(4):
        prompt_id = f"prompt_{index // 2}"
        seed_id = f"seed_{index % 2}"
        for variant in (PREDICTIVE_VARIANT, NONNEGATIVE_VARIANT):
            summaries.extend(
                [
                    _summary(
                        prompt_id,
                        seed_id,
                        variant,
                        "correct_owner_key",
                        1.0,
                    ),
                    _summary(
                        prompt_id,
                        seed_id,
                        variant,
                        "wrong_owner_key",
                        0.0,
                    ),
                ]
            )
    summaries[0]["predictive_owner_wrong_weighted_code_correlation"] = None
    decision = build_predictive_decision(
        summaries,
        build_predictive_pair_records(summaries),
        [],
        _config(),
    )
    assert decision["coverage_ready"] is True
    assert decision["predictive_code_separation_ready"] is False
    assert decision["predictive_trajectory_gate_ready"] is False


def test_predictive_control_gain_rejects_mismatched_wrong_key_identity():
    summaries = []
    for variant in (PREDICTIVE_VARIANT, NONNEGATIVE_VARIANT):
        summaries.extend(
            [
                _summary("prompt", "seed", variant, "correct_owner_key", 1.0),
                _summary("prompt", "seed", variant, "wrong_owner_key", 0.0),
            ]
        )
    summaries[-1][
        "predictive_wrong_owner_key_control_candidate_index"
    ] = 3
    pairs = build_predictive_pair_records(summaries)
    assert all(
        row["comparison_kind"]
        != "predictive_signed_over_nonnegative_llr_margin"
        for row in pairs
    )


def test_predictive_request_dispatches_without_notebook_change(
    tmp_path: Path,
):
    drive_root = tmp_path / "drive" / "SSTW"
    source_zip = drive_root / "inputs" / "controlled_embedding_result.zip"
    source_zip.parent.mkdir(parents=True)
    with ZipFile(source_zip, "w") as archive:
        archive.writestr(
            "bundle/records/generation_records.jsonl",
            json.dumps(
                {
                    "generation_status": "success",
                    "generation_model_id": (
                        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
                    ),
                }
            )
            + "\n",
        )
    request_path = drive_root / "requests" / "colab_test_request.json"
    request_path.parent.mkdir(parents=True)
    request_path.write_text(
        json.dumps(
            {
                "request_schema_version": "sstw_colab_test_request_v1",
                "test_id": (
                    PREDICTIVE_TRAJECTORY_SYNCHRONIZATION_SMOKE_TEST_ID
                ),
                "repository": {
                    "url": "https://github.com/RICHAAARC/SSTW.git",
                    "ref": "main",
                },
                "parameters": {
                    "phase": "no_attack",
                    "run_series_id": "predictive_trajectory_sync_smoke",
                    "source_package_path": str(source_zip),
                    "resume_package_path": "",
                },
            }
        ),
        encoding="utf-8",
    )
    loaded = load_colab_test_request(request_path, project_root=drive_root)
    assert loaded["test_id"] == (
        PREDICTIVE_TRAJECTORY_SYNCHRONIZATION_SMOKE_TEST_ID
    )

    def fake_runner(_source: Path, output: Path) -> dict:
        output.joinpath("artifacts").mkdir(parents=True)
        output.joinpath("artifacts", "decision.json").write_text(
            "{}",
            encoding="utf-8",
        )
        return {
            "predictive_trajectory_smoke_decision": (
                "predictive_trajectory_gate_failed_stop_method"
            ),
            "stage_progression_allowed": False,
            "formal_result": False,
        }

    result = run_colab_test_request(
        request_path,
        project_root=drive_root,
        repo_root=Path.cwd(),
        local_workspace_root=tmp_path / "content" / "workspace",
        local_package_cache_root=tmp_path / "content" / "cache",
        predictive_trajectory_runner=fake_runner,
    )
    assert result["test_id"] == (
        PREDICTIVE_TRAJECTORY_SYNCHRONIZATION_SMOKE_TEST_ID
    )
    assert Path(result["drive_result_zip"]).is_file()


def test_predictive_request_rejects_resume(tmp_path: Path):
    drive_root = tmp_path / "drive" / "SSTW"
    source = drive_root / "inputs" / "source.zip"
    resume = drive_root / "inputs" / "resume.zip"
    source.parent.mkdir(parents=True)
    for path in (source, resume):
        with ZipFile(path, "w") as archive:
            archive.writestr("placeholder.txt", "x")
    request_path = drive_root / "requests" / "colab_test_request.json"
    request_path.parent.mkdir(parents=True)
    request_path.write_text(
        json.dumps(
            {
                "request_schema_version": "sstw_colab_test_request_v1",
                "test_id": (
                    PREDICTIVE_TRAJECTORY_SYNCHRONIZATION_SMOKE_TEST_ID
                ),
                "repository": {
                    "url": "https://github.com/RICHAAARC/SSTW.git",
                    "ref": "main",
                },
                "parameters": {
                    "phase": "no_attack",
                    "run_series_id": "predictive_trajectory_sync_smoke",
                    "source_package_path": str(source),
                    "resume_package_path": str(resume),
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="不接受 resume"):
        load_colab_test_request(request_path, project_root=drive_root)
