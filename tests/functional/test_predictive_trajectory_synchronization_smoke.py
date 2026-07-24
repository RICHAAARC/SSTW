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
from main.methods.state_space_watermark.replay_inversion import (
    FlowSchedulePoint,
    ReplayGaussianLikelihoodConfig,
    replay_step_null_reliability_weight,
    replay_step_reliability_weight,
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

    def pow(self, exponent):
        return _NumpyTensor(np.power(self.values, exponent))

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


def test_predictive_replay_writes_joint_spatial_and_temporal_path_controls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    records = tmp_path / "records"
    datasets = tmp_path / "datasets"
    videos = tmp_path / "videos"
    for path in (records, datasets, videos):
        path.mkdir()
    generation_rows = []
    for variant in (PREDICTIVE_VARIANT, NONNEGATIVE_VARIANT):
        video = videos / f"{variant}.mp4"
        video.write_bytes(b"video")
        generation_rows.append(
            {
                "generation_status": "success",
                "generation_model_id": "model",
                "prompt_id": "prompt",
                "seed_id": "seed",
                "trajectory_trace_id": f"trace-{variant}",
                "predictive_trajectory_plan_record_id": f"plan-{variant}",
                "trajectory_carrier_variant_id": variant,
                "method_variant": f"method-{variant}",
                "video_path": str(video),
            }
        )
    records.joinpath("generation_records.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in generation_rows),
        encoding="utf-8",
    )
    datasets.joinpath("prompt_seed_suite.json").write_text(
        json.dumps(
            {
                "prompts": [
                    {"prompt_id": "prompt", "prompt_text": "prompt text"}
                ]
            }
        ),
        encoding="utf-8",
    )
    context = FlowTubeletKeyContext(
        prompt_digest="a" * 64,
        sampler_signature="scheduler:test",
    )

    class FakeTrajectory:
        replay_log_likelihood_ratio = 0.1
        candidate_cycle_relative_error = 0.2
        null_cycle_relative_error = 0.3

    class FakeUncertainty:
        replay_reliability = 0.8

    class FakeReplay:
        replay_trajectories = (FakeTrajectory(),)
        primary_schedule = (object(),)
        replay_uncertainty = FakeUncertainty()

    class FakePipeline:
        scheduler = object()

    monkeypatch.setattr(
        predictive_smoke_module,
        "validate_generation_model_provenance",
        lambda _row: "revision",
    )
    monkeypatch.setattr(
        predictive_smoke_module,
        "_validated_flow_key_context",
        lambda *_args, **_kwargs: context,
    )
    monkeypatch.setattr(
        predictive_smoke_module,
        "_generation_key",
        lambda _row: "owner-key",
    )
    monkeypatch.setattr(
        predictive_smoke_module,
        "_wrong_owner_generation_key",
        lambda *_args, **_kwargs: "wrong-key",
    )
    monkeypatch.setattr(
        predictive_smoke_module,
        "_run_attacked_video_replay_for_model",
        lambda *_args, **_kwargs: FakeReplay(),
    )
    monkeypatch.setattr(
        predictive_smoke_module,
        "predictive_schedule_for_replay",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        predictive_smoke_module,
        "predictive_trajectory_weighted_code_correlation",
        lambda *_args, **_kwargs: 0.2,
    )

    def fake_score(
        _trajectory,
        _schedule,
        *,
        key_text,
        predictive_trajectory_carrier_config,
        trajectory_carrier_key_text=None,
        **_kwargs,
    ):
        if predictive_trajectory_carrier_config is None:
            score = 1.0 if key_text == "owner-key" else 0.2
        elif key_text == "owner-key" and trajectory_carrier_key_text is None:
            score = 2.0
        elif key_text == "wrong-key" and trajectory_carrier_key_text is None:
            score = 0.5
        elif (
            key_text == "wrong-key"
            and trajectory_carrier_key_text == "owner-key"
        ):
            score = 1.0
        else:
            score = 1.2
        return {
            "S_path_inv": score,
            "S_velocity": score / 2.0,
            "path_observation_step_count": 20,
            "path_quadrature_context_complete": True,
            "replay_joint_schedule_context_complete": True,
            "path_replay_weighted_aggregation_applied": True,
            "path_replay_reliability_mode": (
                "null_forward_key_independent"
            ),
            "path_spatial_temporal_key_decoupled": (
                trajectory_carrier_key_text is not None
            ),
        }

    monkeypatch.setattr(
        predictive_smoke_module,
        "score_replay_trajectory_for_key",
        fake_score,
    )
    summaries, failures = predictive_smoke_module._execute_replay(
        tmp_path,
        _config(),
        likelihood=object(),
        pipeline_loader=lambda *_args, **_kwargs: FakePipeline(),
    )
    assert failures == []
    assert len(summaries) == 6
    predictive_rows = [
        row
        for row in summaries
        if row["trajectory_carrier_variant_id"] == PREDICTIVE_VARIANT
    ]
    assert {
        row["candidate_key_role"] for row in predictive_rows
    } == {
        "correct_owner_key",
        "wrong_owner_key",
        "wrong_owner_spatial_key_only",
        "wrong_owner_temporal_code_only",
    }
    assert {
        row["candidate_key_role"]: row["predictive_replay_path_score"]
        for row in predictive_rows
    } == {
        "correct_owner_key": 2.0,
        "wrong_owner_key": 0.5,
        "wrong_owner_spatial_key_only": 1.0,
        "wrong_owner_temporal_code_only": 1.2,
    }


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


def test_predictive_path_score_decouples_spatial_and_temporal_keys(
    monkeypatch: pytest.MonkeyPatch,
):
    observed = {}

    class FakeSchedule:
        codes = (1.0,)

        @staticmethod
        def metadata_for_step(_step_index):
            return {
                "predictive_trajectory_phase_function_digest": "digest"
            }

    class FakeObservation:
        @staticmethod
        def as_dict():
            return {
                "path_projection_normalized": 0.5,
                "path_quadrature_context_complete": True,
            }

    def fake_predictive_schedule(_schedule, *, key_text, **_kwargs):
        observed["carrier_key"] = key_text
        return FakeSchedule()

    monkeypatch.setattr(
        wan_replay_module,
        "predictive_schedule_for_replay",
        fake_predictive_schedule,
    )

    def fake_direction(_state, *, key_text, **_kwargs):
        observed["spatial_key"] = key_text
        return _NumpyTensor([1.0]), {
            "flow_tubelet_formal_context_complete": True
        }

    monkeypatch.setattr(
        wan_replay_module,
        "build_flow_tubelet_key_direction_like",
        fake_direction,
    )
    monkeypatch.setattr(
        wan_replay_module,
        "compute_path_step_observation",
        lambda *_args, **_kwargs: FakeObservation(),
    )
    monkeypatch.setattr(
        wan_replay_module,
        "aggregate_path_observations",
        lambda _records: {
            "S_path_inv": 0.25,
            "S_velocity": 0.5,
            "path_observation_step_count": 1,
            "path_quadrature_context_complete": True,
            "path_replay_weighted_aggregation_applied": True,
        },
    )
    monkeypatch.setattr(
        wan_replay_module,
        "replay_step_null_reliability_weight",
        lambda *_args, **_kwargs: observed.setdefault(
            "null_weight_used",
            0.75,
        ),
    )
    monkeypatch.setattr(
        wan_replay_module,
        "replay_step_reliability_weight",
        lambda *_args, **_kwargs: pytest.fail(
            "candidate-forward reliability must not be used"
        ),
    )

    class FakeTrajectory:
        reverse_states = (
            _NumpyTensor([0.0]),
            _NumpyTensor([1.0]),
        )

    evidence = wan_replay_module.score_replay_trajectory_for_key(
        FakeTrajectory(),
        (
            FlowSchedulePoint(timestep=1.0, sigma=1.0),
            FlowSchedulePoint(timestep=0.0, sigma=0.0),
        ),
        key_text="spatial-key",
        trajectory_carrier_key_text="temporal-key",
        key_context=FlowTubeletKeyContext(
            prompt_digest="a" * 64,
            sampler_signature="scheduler:test",
        ),
        likelihood_config=object(),
        predictive_trajectory_carrier_config=(
            PredictiveTrajectoryCarrierConfig()
        ),
        path_reliability_mode="null_forward_key_independent",
    )
    assert observed == {
        "carrier_key": "temporal-key",
        "spatial_key": "spatial-key",
        "null_weight_used": 0.75,
    }
    assert evidence["path_replay_reliability_mode"] == (
        "null_forward_key_independent"
    )
    assert evidence["path_spatial_temporal_key_decoupled"] is True
    assert evidence["replay_joint_schedule_context_complete"] is True


def test_null_replay_path_weight_is_candidate_key_independent():
    config = ReplayGaussianLikelihoodConfig(
        relative_observation_noise_standard_deviation=0.5
    )

    class FakeTrajectory:
        reverse_states = (
            _NumpyTensor([0.0]),
            _NumpyTensor([1.0]),
        )
        null_forward_states = (
            _NumpyTensor([0.0]),
            _NumpyTensor([0.9]),
        )

        def __init__(self, candidate_value):
            self.forward_states = (
                _NumpyTensor([0.0]),
                _NumpyTensor([candidate_value]),
            )

    first = FakeTrajectory(1.0)
    second = FakeTrajectory(-4.0)
    assert replay_step_null_reliability_weight(
        first,
        1,
        config=config,
    ) == pytest.approx(
        replay_step_null_reliability_weight(
            second,
            1,
            config=config,
        )
    )
    assert replay_step_reliability_weight(
        first,
        1,
        config=config,
    ) != pytest.approx(
        replay_step_reliability_weight(
            second,
            1,
            config=config,
        )
    )


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
        ("minimum_predictive_correct_over_wrong_path_fraction", 0.5),
        (
            "minimum_predictive_over_nonnegative_path_margin_fraction",
            0.5,
        ),
        (
            "minimum_predictive_correct_over_spatial_only_wrong_fraction",
            0.5,
        ),
        (
            "minimum_predictive_correct_over_temporal_only_wrong_fraction",
            0.5,
        ),
        ("primary_predictive_path_statistic", "S_velocity"),
        ("predictive_path_reliability_mode", "candidate_forward"),
        ("smoke_summary_record_count", 16),
        ("smoke_pair_record_count", 12),
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


def test_predictive_replay_only_reuses_generation_without_old_detector_records(
    tmp_path: Path,
):
    source = tmp_path / "prior"
    output = tmp_path / "output"
    output.mkdir()
    plan = [
        {
            "predictive_trajectory_plan_record_id": f"plan-{index}",
            "trajectory_carrier_variant_id": (
                PREDICTIVE_VARIANT if index < 4 else NONNEGATIVE_VARIANT
            ),
        }
        for index in range(8)
    ]
    source.joinpath("artifacts").mkdir(parents=True)
    source.joinpath("records").mkdir(parents=True)
    source.joinpath("videos").mkdir(parents=True)
    source.joinpath(
        "artifacts",
        "predictive_trajectory_smoke_manifest.json",
    ).write_text(
        json.dumps(
            {
                "profile_id": (
                    "sstw_predictive_trajectory_synchronization_smoke"
                ),
                "generation_record_count": 8,
                "formal_result": False,
                "stage_progression_allowed": False,
                "generation_result": {
                    "decision": {
                        "implementation_decision": "PASS",
                        "mechanism_decision": (
                            "GENERATION_READY_NO_ATTACK_REPLAY_PENDING"
                        ),
                    },
                    "generation_record_count": 8,
                    "trajectory_record_count": 160,
                },
            }
        ),
        encoding="utf-8",
    )
    source.joinpath(
        "artifacts",
        "predictive_trajectory_smoke_decision.json",
    ).write_text(
        json.dumps(
            {
                "profile_id": (
                    "sstw_predictive_trajectory_synchronization_smoke"
                ),
                "formal_result": False,
                "stage_progression_allowed": False,
                "failure_record_count": 0,
            }
        ),
        encoding="utf-8",
    )
    source.joinpath(
        "records",
        "predictive_trajectory_generation_plan.jsonl",
    ).write_text(
        "".join(json.dumps(row) + "\n" for row in plan),
        encoding="utf-8",
    )
    generation_rows = []
    for index, row in enumerate(plan):
        video = source / "videos" / f"video-{index}.mp4"
        video.write_bytes(f"video-{index}".encode())
        generation_rows.append(
            {
                **row,
                "generation_status": "success",
                "video_path": f"/content/old/videos/{video.name}",
            }
        )
    source.joinpath("records", "generation_records.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in generation_rows),
        encoding="utf-8",
    )
    source.joinpath("records", "trajectory_trace.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "predictive_trajectory_plan_record_id": (
                        f"plan-{index // 20}"
                    )
                }
            )
            + "\n"
            for index in range(160)
        ),
        encoding="utf-8",
    )
    source.joinpath(
        "records",
        "predictive_trajectory_summary_records.jsonl",
    ).write_text('{"old_detector_record": true}\n', encoding="utf-8")

    result = (
        predictive_smoke_module._reuse_predictive_generation_for_replay_only(
            source,
            output,
            plan,
        )
    )
    rebound = [
        json.loads(line)
        for line in output.joinpath(
            "records",
            "generation_records.jsonl",
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert result["generation_reused_for_replay_only"] is True
    assert len(rebound) == 8
    assert all(
        row["predictive_generation_reused_for_replay_only"] is True
        and Path(row["video_path"]).is_relative_to(output)
        and Path(row["video_path"]).is_file()
        for row in rebound
    )
    assert not output.joinpath(
        "records",
        "predictive_trajectory_summary_records.jsonl",
    ).exists()


def _summary(
    prompt_id: str,
    seed_id: str,
    variant: str,
    key_role: str,
    path_score: float,
    *,
    llr: float = 0.0,
) -> dict:
    return {
        "prompt_id": prompt_id,
        "seed_id": seed_id,
        "trajectory_carrier_variant_id": variant,
        "candidate_key_role": key_role,
        "predictive_replay_log_likelihood_ratio": llr,
        "predictive_replay_path_score": path_score,
        "predictive_replay_velocity_score": path_score,
        "predictive_replay_path_observation_step_count": 20,
        "predictive_replay_path_quadrature_context_complete": True,
        "predictive_replay_joint_schedule_context_complete": True,
        "predictive_replay_path_weighted_aggregation_applied": True,
        "predictive_replay_path_reliability_mode": (
            "null_forward_key_independent"
        ),
        "trajectory_global_reliability": 0.8,
        "predictive_wrong_owner_key_control_candidate_index": 2,
        "predictive_owner_wrong_weighted_code_correlation": (
            0.2 if variant == PREDICTIVE_VARIANT else None
        ),
    }


def _complete_path_summaries() -> list[dict]:
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
                        llr=-10.0,
                    ),
                    _summary(
                        prompt_id,
                        seed_id,
                        variant,
                        "wrong_owner_key",
                        0.0,
                        llr=10.0,
                    ),
                ]
            )
        for role, margin in (
            ("wrong_owner_spatial_key_only", 0.8 if index < 3 else -0.1),
            ("wrong_owner_temporal_code_only", 0.7 if index < 3 else -0.2),
        ):
            summaries.append(
                _summary(
                    prompt_id,
                    seed_id,
                    PREDICTIVE_VARIANT,
                    role,
                    signed_margin - margin,
                    llr=100.0,
                )
            )
    return summaries


def test_predictive_gate_uses_signed_path_not_endpoint_llr():
    summaries = _complete_path_summaries()
    pairs = build_predictive_pair_records(summaries)
    decision = build_predictive_decision(
        summaries,
        pairs,
        [],
        _config(),
    )
    assert len(pairs) == 20
    assert decision["predictive_correct_over_wrong_path_fraction"] == 0.75
    assert (
        decision["predictive_over_nonnegative_path_margin_fraction"] == 0.75
    )
    assert (
        decision["predictive_correct_over_spatial_only_wrong_fraction"]
        == 0.75
    )
    assert (
        decision["predictive_correct_over_temporal_only_wrong_fraction"]
        == 0.75
    )
    assert decision["predictive_path_evidence_ready"] is True
    assert decision["predictive_trajectory_gate_ready"] is True
    assert decision["predictive_endpoint_llr_role"] == (
        "diagnostic_only_not_gate"
    )
    assert decision["endpoint_gate_executed"] is False
    assert decision["state_space_posterior_executed"] is False
    assert decision["stage_progression_allowed"] is False


def test_predictive_gate_fails_closed_when_code_correlation_is_missing():
    summaries = _complete_path_summaries()
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


def test_predictive_gate_fails_closed_when_path_context_is_incomplete():
    summaries = _complete_path_summaries()
    summaries[0][
        "predictive_replay_path_quadrature_context_complete"
    ] = False
    decision = build_predictive_decision(
        summaries,
        build_predictive_pair_records(summaries),
        [],
        _config(),
    )
    assert decision["coverage_ready"] is True
    assert decision["predictive_path_evidence_ready"] is False
    assert decision["predictive_trajectory_gate_ready"] is False


def test_predictive_gate_requires_temporal_only_wrong_control():
    summaries = _complete_path_summaries()
    for row in summaries:
        if row["candidate_key_role"] == "wrong_owner_temporal_code_only":
            row["predictive_replay_path_score"] = 100.0
    decision = build_predictive_decision(
        summaries,
        build_predictive_pair_records(summaries),
        [],
        _config(),
    )
    assert (
        decision["predictive_correct_over_temporal_only_wrong_fraction"]
        == 0.0
    )
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
        != "predictive_signed_over_nonnegative_path_margin"
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


def test_predictive_request_dispatches_replay_only_resume(
    tmp_path: Path,
):
    drive_root = tmp_path / "drive" / "SSTW"
    input_root = drive_root / "inputs"
    input_root.mkdir(parents=True)
    source_zip = input_root / "controlled.zip"
    resume_zip = input_root / "predictive_result.zip"
    with ZipFile(source_zip, "w") as archive:
        archive.writestr(
            "controlled/records/generation_records.jsonl",
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
    with ZipFile(resume_zip, "w") as archive:
        for name in (
            "prior/artifacts/predictive_trajectory_smoke_manifest.json",
            "prior/artifacts/predictive_trajectory_smoke_decision.json",
        ):
            archive.writestr(name, "{}")
        for name in (
            "prior/records/predictive_trajectory_generation_plan.jsonl",
            "prior/records/generation_records.jsonl",
            "prior/records/trajectory_trace.jsonl",
        ):
            archive.writestr(name, "{}\n")
        archive.writestr("prior/videos/video.mp4", "video")
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
                    "run_series_id": "predictive_path_replay",
                    "source_package_path": str(source_zip),
                    "resume_package_path": str(resume_zip),
                },
            }
        ),
        encoding="utf-8",
    )
    observed = {}

    def fake_runner(
        _source: Path,
        output: Path,
        replay_source: Path,
    ) -> dict:
        observed["replay_source"] = replay_source
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
    assert observed["replay_source"].name == "prior"
    assert Path(result["drive_result_zip"]).is_file()
    assert result["diagnostic_decision"]["formal_result"] is False


def test_predictive_request_accepts_replay_only_resume(tmp_path: Path):
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
    loaded = load_colab_test_request(request_path, project_root=drive_root)
    assert loaded["resume_package_path"] == str(resume.resolve())
