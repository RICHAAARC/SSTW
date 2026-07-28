from __future__ import annotations

from dataclasses import replace
import gc
import inspect
import json
import os
from pathlib import Path
import weakref

import numpy as np
import pytest

from evaluation.protocol.patch_relation_phase_response_preflight_contract import (
    DEFAULT_CONFIG_PATH,
    FROZEN_PROTOCOL_DIGEST,
    NEXT_DETERMINISTIC_SCALE,
    load_patch_relation_phase_response_preflight_config,
    protocol_digest,
)
from experiments.generative_video_model_probe.patch_relation_phase_response_preflight import (
    PhaseResponsePreflightBatch,
    ScopedPatchRelationPhaseResponsePreflight,
    _tuple_delta_observer,
    classify_phase_response_preflight,
    run_patch_relation_phase_response_preflight,
    validate_phase_response_preflight_batch,
)
from evaluation.protocol.patch_relation_gate0_contract import (
    load_patch_relation_gate0_config,
)
from main.methods.state_space_watermark.patch_relation_carrier import (
    ROPE_TUPLE_SHAPE,
    build_public_patch_relation_descriptor,
)
import experiments.generative_video_model_probe.patch_relation_gate0_construction as gate_runner
import experiments.generative_video_model_probe.patch_relation_phase_response_preflight as preflight
import main.methods.state_space_watermark.patch_relation_wan_runtime as runtime


pytestmark = pytest.mark.quick


def _record(
    *,
    base_exact: bool = True,
    control_exact: tuple[bool, bool] = (True, True),
    control_equals_base: bool = False,
    candidate_norm: float = 1.0,
    feasible: bool = False,
) -> dict[str, object]:
    return {
        "zero_rope_delta_norms": [0.0] * 4,
        "zero_rope_delta_matches_expected": [True] * 4,
        "active_rope_delta_norms": [1e-5] * 8,
        "expected_active_rope_delta_norms": [1e-5] * 8,
        "active_rope_delta_matches_expected": [True] * 8,
        "base_cfg_repeat_delta_norm": 0.0 if base_exact else 1.0,
        "control_repeat_delta_norm_by_sign": [
            0.0 if control_exact[0] else 1.0,
            0.0 if control_exact[1] else 1.0,
        ],
        "control_base_cfg_delta_norm_by_sign_and_repeat": [1.0] * 4,
        "base_cfg_repeats_byte_exact": base_exact,
        "control_cfg_repeats_byte_exact_by_sign": list(control_exact),
        "control_cfg_equals_zero_base_by_sign_and_repeat": [
            control_equals_base
        ] * 4,
        "candidate_evaluations": [
            {"actual_delta_norm": candidate_norm, "feasible": feasible}
            for _ in range(4)
        ],
    }


def test_preflight_config_freezes_next_scale_and_all_evidence_boundaries() -> None:
    config = load_patch_relation_phase_response_preflight_config()
    assert config["protocol_digest"] == FROZEN_PROTOCOL_DIGEST
    assert (
        float(
            config["protocol_contract"]["forward_plan"]["candidate_scale"]
        )
        == NEXT_DETERMINISTIC_SCALE
    )
    assert config["protocol_contract"]["forward_plan"][
        "real_scheduler_step_call_count"
    ] == 0
    assert not config["authorization_boundary"]["gate0_execution_allowed"]
    assert not config["authorization_boundary"]["formal_result"]
    assert not config["authorization_boundary"]["stage_progression_allowed"]


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("forward_plan", "candidate_scale"), "0.0001"),
        (("forward_plan", "candidate_repeat_count_per_sign"), 3),
        (("forward_plan", "real_scheduler_step_call_count"), 1),
        (("classification_contract", "plateau_ratio_threshold"), "0.1"),
        (
            ("classification_contract", "classification_order"),
            ["indeterminate_multiple_candidates"],
        ),
        (
            ("measurement_contract", "candidate_scalar_revalidation"),
            "caller_reported_guards_are_trusted",
        ),
        (
            ("measurement_contract", "cfg_digest_boolean_binding"),
            "caller_reported_booleans_are_trusted",
        ),
        (("result_boundary", "full_eight_video_rerun_allowed"), True),
    ],
)
def test_rehashed_preflight_contract_mutation_fails_closed(
    tmp_path: Path,
    path: tuple[str, str],
    value: object,
) -> None:
    config = json.loads(Path(DEFAULT_CONFIG_PATH).read_text(encoding="utf-8"))
    config["protocol_contract"][path[0]][path[1]] = value
    config["protocol_digest"] = protocol_digest(config["protocol_contract"])
    target = tmp_path / "mutated.json"
    target.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError):
        load_patch_relation_phase_response_preflight_config(target)


def test_preflight_authorization_boundary_rejects_extra_true_field(
    tmp_path: Path,
) -> None:
    config = json.loads(Path(DEFAULT_CONFIG_PATH).read_text(encoding="utf-8"))
    config["authorization_boundary"]["full_eight_video_rerun_allowed"] = True
    target = tmp_path / "authorization_mutation.json"
    target.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="authorization boundary"):
        load_patch_relation_phase_response_preflight_config(target)


def test_phase_response_classification_separates_repeat_floor_dead_zone_and_feasible() -> None:
    repeat_floor = classify_phase_response_preflight(
        _record(base_exact=False)
    )
    assert repeat_floor["diagnostic_classification"] == (
        "forward_repeatability_floor_candidate"
    )

    dead_zone = classify_phase_response_preflight(
        _record(control_equals_base=True, candidate_norm=0.0)
    )
    assert dead_zone["diagnostic_classification"] == (
        "quantization_dead_zone_candidate"
    )

    feasible = classify_phase_response_preflight(
        _record(candidate_norm=1.0, feasible=True)
    )
    assert feasible["diagnostic_classification"] == (
        "feasible_nonzero_phase_region_candidate"
    )
    assert feasible["gate0_pass"] is False
    assert feasible["full_eight_video_rerun_allowed"] is False


def test_repeatability_floor_uses_response_relative_ratio_not_byte_identity() -> None:
    below_floor = _record(
        base_exact=False,
        candidate_norm=1.0,
        feasible=True,
    )
    below_floor["base_cfg_repeat_delta_norm"] = 0.01
    classified = classify_phase_response_preflight(below_floor)
    assert classified["base_repeatability_floor_ratio"] == pytest.approx(0.01)
    assert classified["repeatability_floor_detected"] is False
    assert classified["diagnostic_classification"] == (
        "feasible_nonzero_phase_region_candidate"
    )

    above_floor = dict(below_floor)
    above_floor["base_cfg_repeat_delta_norm"] = 0.100001
    classified = classify_phase_response_preflight(above_floor)
    assert classified["repeatability_floor_detected"] is True
    assert classified["diagnostic_classification"] == (
        "forward_repeatability_floor_candidate"
    )


def test_phase_response_classification_detects_scale_application_and_plateau() -> None:
    broken = _record()
    broken["zero_rope_delta_norms"] = [0.0, 0.0, 1e-8, 0.0]
    assert classify_phase_response_preflight(broken)[
        "diagnostic_classification"
    ] == "scale_application_failure"

    wrong_active_scale = _record()
    wrong_active_scale["expected_active_rope_delta_norms"] = [2e-5] * 8
    assert classify_phase_response_preflight(wrong_active_scale)[
        "diagnostic_classification"
    ] == "scale_application_failure"

    plateau = classify_phase_response_preflight(
        _record(candidate_norm=19.0)
    )
    assert plateau["diagnostic_classification"] == (
        "bf16_or_attention_piecewise_plateau_candidate"
    )


def test_rope_tuple_observer_rejects_equal_norm_wrong_support() -> None:
    descriptor = build_public_patch_relation_descriptor()
    rope = _FakeRope()
    expected_cosine, expected_sine = runtime.apply_wan_rotary_phase_runtime(
        rope.cosine,
        rope.sine,
        descriptor=descriptor,
        signed_coefficient=1,
        phase_projection_scale=NEXT_DETERMINISTIC_SCALE,
    )
    wrong_cosine = np.ascontiguousarray(
        rope.cosine
        + np.roll(expected_cosine - rope.cosine, shift=1, axis=1),
        dtype="<f4",
    )
    wrong_sine = np.ascontiguousarray(
        rope.sine
        + np.roll(expected_sine - rope.sine, shift=1, axis=1),
        dtype="<f4",
    )
    observations = []
    observer = _tuple_delta_observer(
        observations,
        descriptor=descriptor,
        signed_coefficient=1,
        phase_projection_scale=NEXT_DETERMINISTIC_SCALE,
    )
    observer(
        rope.cosine,
        rope.sine,
        wrong_cosine,
        wrong_sine,
    )
    assert len(observations) == 1
    observation = observations[0]
    assert observation.actual_delta_norm == pytest.approx(
        observation.expected_delta_norm
    )
    assert observation.actual_matches_expected is False
    assert observation.actual_delta_digest != observation.expected_delta_digest

    record = _record()
    record["active_rope_delta_matches_expected"][0] = False
    assert classify_phase_response_preflight(record)[
        "diagnostic_classification"
    ] == "scale_application_failure"


class _FakeRope:
    def __init__(self) -> None:
        self.cosine = np.ones(ROPE_TUPLE_SHAPE, dtype="<f4")
        self.sine = np.zeros(ROPE_TUPLE_SHAPE, dtype="<f4")

    def forward(self, hidden_states: np.ndarray):
        return self.cosine, self.sine

    def __call__(self, hidden_states: np.ndarray):
        return self.forward(hidden_states)


class _FakeTransformer:
    def __init__(self, velocity_shape: tuple[int, ...]) -> None:
        self.rope = _FakeRope()
        self.velocity_shape = velocity_shape
        self.forward_count = 0

    def forward(
        self,
        *,
        hidden_states: np.ndarray,
        timestep: np.ndarray,
        encoder_hidden_states: np.ndarray,
        **kwargs,
    ):
        del timestep, kwargs
        self.forward_count += 1
        _cosine, sine = self.rope(hidden_states)
        relation = float(sine[0, 2221, 0, 1])
        branch = float(encoder_hidden_states[0])
        value = np.float32(1.0 + branch * 0.01 + relation * 0.1)
        return (np.full(self.velocity_shape, value, dtype="<f4"),)


class _FakeScheduler:
    def __init__(self, sigmas: tuple[float, ...]) -> None:
        self.sigmas = np.asarray(sigmas, dtype="<f4")
        self.timesteps = np.ascontiguousarray(
            self.sigmas[:-1] * np.float32(1000.0),
            dtype="<f4",
        )
        self._step_index = None
        self.original_step_count = 0

    @property
    def step_index(self):
        return self._step_index

    def step(self, *args, **kwargs):
        self.original_step_count += 1
        raise AssertionError("real scheduler must not be called")


class _StepInstallFailingScheduler(_FakeScheduler):
    def __init__(self, sigmas: tuple[float, ...]) -> None:
        super().__init__(sigmas)
        self.fail_step_install = True

    def __setattr__(self, name: str, value) -> None:
        if name == "step" and getattr(self, "fail_step_install", False):
            raise RuntimeError("synthetic scheduler install failure")
        super().__setattr__(name, value)


class _ForwardCleanupFailingTransformer(_FakeTransformer):
    def __init__(self, velocity_shape: tuple[int, ...]) -> None:
        super().__init__(velocity_shape)
        self.fail_forward_cleanup = True

    def __delattr__(self, name: str) -> None:
        if name == "forward" and self.fail_forward_cleanup:
            raise RuntimeError("synthetic transformer cleanup failure")
        super().__delattr__(name)


def _preflight_adapter(
    transformer: _FakeTransformer,
    scheduler: _FakeScheduler,
) -> ScopedPatchRelationPhaseResponsePreflight:
    config = load_patch_relation_gate0_config()
    schedule = config["protocol_contract"]["gate0_runtime_execution_contract"]
    return ScopedPatchRelationPhaseResponsePreflight(
        transformer,
        scheduler,
        descriptor=build_public_patch_relation_descriptor(),
        probe_id="patch_relation_phase_response_preflight",
        sigma_grid=tuple(
            float(value) for value in schedule["sigma_grid_decimal"]
        ),
        delta_sigma_by_step=tuple(
            float(value)
            for value in schedule["delta_sigma_by_step_decimal"]
        ),
        timestep_by_step=tuple(
            float(value) for value in schedule["timestep_by_step_decimal"]
        ),
    )


def _complete_fake_preflight_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    ScopedPatchRelationPhaseResponsePreflight,
    PhaseResponsePreflightBatch,
]:
    small_shape = (1, 1, 1, 1, 2)
    monkeypatch.setattr(runtime, "TRANSFORMER_HIDDEN_SHAPE", small_shape)
    monkeypatch.setattr(runtime, "SCHEDULER_VELOCITY_SHAPE", small_shape)
    monkeypatch.setattr(gate_runner, "SCHEDULER_VELOCITY_SHAPE", small_shape)
    config = load_patch_relation_gate0_config()
    schedule = config["protocol_contract"]["gate0_runtime_execution_contract"]
    sigmas = tuple(float(value) for value in schedule["sigma_grid_decimal"])
    timesteps = tuple(
        float(value) for value in schedule["timestep_by_step_decimal"]
    )
    transformer = _FakeTransformer(small_shape)
    scheduler = _FakeScheduler(sigmas)
    adapter = _preflight_adapter(transformer, scheduler)
    hidden = np.zeros(small_shape, dtype="<f4")
    timestep = np.asarray([timesteps[0]], dtype="<f4")
    with adapter:
        conditional = transformer.forward(
            hidden_states=hidden,
            timestep=timestep,
            encoder_hidden_states=np.asarray([2.0], dtype="<f4"),
            return_dict=False,
        )[0]
        unconditional = transformer.forward(
            hidden_states=hidden,
            timestep=timestep,
            encoder_hidden_states=np.asarray([1.0], dtype="<f4"),
            return_dict=False,
        )[0]
        cfg = np.ascontiguousarray(
            unconditional
            + np.float32(5.0)
            * np.subtract(conditional, unconditional, dtype=np.float32),
            dtype="<f4",
        )
        scheduler.step(
            cfg,
            np.asarray(timesteps[0], dtype="<f4"),
            hidden,
            return_dict=False,
        )
    return (
        adapter,
        adapter.batch(generator_state_digest_random="a" * 64),
    )


def _complete_fake_preflight_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> PhaseResponsePreflightBatch:
    return _complete_fake_preflight_scope(monkeypatch)[1]


def _run_fake_preflight_with_intercepted_completion(
    monkeypatch: pytest.MonkeyPatch,
    *,
    raise_after_completion: bool,
) -> tuple[ScopedPatchRelationPhaseResponsePreflight, BaseException | None]:
    small_shape = (1, 1, 1, 1, 2)
    monkeypatch.setattr(runtime, "TRANSFORMER_HIDDEN_SHAPE", small_shape)
    monkeypatch.setattr(runtime, "SCHEDULER_VELOCITY_SHAPE", small_shape)
    monkeypatch.setattr(gate_runner, "SCHEDULER_VELOCITY_SHAPE", small_shape)
    config = load_patch_relation_gate0_config()
    schedule = config["protocol_contract"]["gate0_runtime_execution_contract"]
    sigmas = tuple(float(value) for value in schedule["sigma_grid_decimal"])
    timestep = np.asarray(
        [float(schedule["timestep_by_step_decimal"][0])],
        dtype="<f4",
    )
    transformer = _FakeTransformer(small_shape)
    scheduler = _FakeScheduler(sigmas)
    adapter = _preflight_adapter(transformer, scheduler)
    hidden = np.zeros(small_shape, dtype="<f4")
    observed_error: BaseException | None = None
    try:
        with adapter:
            conditional = transformer.forward(
                hidden_states=hidden,
                timestep=timestep,
                encoder_hidden_states=np.asarray([2.0], dtype="<f4"),
                return_dict=False,
            )[0]
            unconditional = transformer.forward(
                hidden_states=hidden,
                timestep=timestep,
                encoder_hidden_states=np.asarray([1.0], dtype="<f4"),
                return_dict=False,
            )[0]
            cfg = np.ascontiguousarray(
                unconditional
                + np.float32(5.0)
                * np.subtract(
                    conditional,
                    unconditional,
                    dtype=np.float32,
                ),
                dtype="<f4",
            )
            try:
                scheduler.step(
                    cfg,
                    np.asarray(timestep[0], dtype="<f4"),
                    hidden,
                    return_dict=False,
                )
            except preflight._PreflightComplete:
                if raise_after_completion:
                    raise RuntimeError(
                        "synthetic error after intercepted completion"
                    )
    except BaseException as error:
        observed_error = error
    return adapter, observed_error


def test_single_step_scope_repeats_real_forwards_and_never_advances_scheduler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    small_shape = (1, 1, 1, 1, 2)
    monkeypatch.setattr(runtime, "TRANSFORMER_HIDDEN_SHAPE", small_shape)
    monkeypatch.setattr(runtime, "SCHEDULER_VELOCITY_SHAPE", small_shape)
    monkeypatch.setattr(gate_runner, "SCHEDULER_VELOCITY_SHAPE", small_shape)
    config = load_patch_relation_gate0_config()
    schedule = config["protocol_contract"]["gate0_runtime_execution_contract"]
    sigmas = tuple(float(value) for value in schedule["sigma_grid_decimal"])
    deltas = tuple(
        float(value) for value in schedule["delta_sigma_by_step_decimal"]
    )
    timesteps = tuple(
        float(value) for value in schedule["timestep_by_step_decimal"]
    )
    transformer = _FakeTransformer(small_shape)
    scheduler = _FakeScheduler(sigmas)
    original_forward = transformer.forward
    original_step = scheduler.step
    adapter = _preflight_adapter(transformer, scheduler)
    hidden = np.zeros(small_shape, dtype="<f4")
    timestep = np.asarray([timesteps[0]], dtype="<f4")
    with adapter:
        conditional = transformer.forward(
            hidden_states=hidden,
            timestep=timestep,
            encoder_hidden_states=np.asarray([2.0], dtype="<f4"),
            return_dict=False,
        )[0]
        unconditional = transformer.forward(
            hidden_states=hidden,
            timestep=timestep,
            encoder_hidden_states=np.asarray([1.0], dtype="<f4"),
            return_dict=False,
        )[0]
        cfg = np.ascontiguousarray(
            unconditional
            + np.float32(5.0)
            * np.subtract(conditional, unconditional, dtype=np.float32),
            dtype="<f4",
        )
        scheduler.step(
            cfg,
            np.asarray(timesteps[0], dtype="<f4"),
            hidden,
            return_dict=False,
        )
    batch = adapter.batch(generator_state_digest_random="a" * 64)
    assert batch.transformer_forward_count == 12
    assert batch.scheduler_step_call_count == 0
    assert scheduler.original_step_count == 0
    assert transformer.forward.__func__ is original_forward.__func__
    assert transformer.forward.__self__ is original_forward.__self__
    assert scheduler.step.__func__ is original_step.__func__
    assert scheduler.step.__self__ is original_step.__self__
    assert batch.record["real_scheduler_step_call_count"] == 0
    assert batch.record["decode_executed"] is False
    assert batch.record["video_export_executed"] is False
    assert batch.record["gate0_executed"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        "conditional_encoder",
        "conditional_public_kwargs",
        "scheduler_sample",
        "transformer_timestep",
    ],
)
def test_preflight_reforward_recomputes_complete_runtime_input_binding(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    small_shape = (1, 1, 1, 1, 2)
    monkeypatch.setattr(runtime, "TRANSFORMER_HIDDEN_SHAPE", small_shape)
    monkeypatch.setattr(runtime, "SCHEDULER_VELOCITY_SHAPE", small_shape)
    monkeypatch.setattr(gate_runner, "SCHEDULER_VELOCITY_SHAPE", small_shape)
    config = load_patch_relation_gate0_config()
    schedule = config["protocol_contract"]["gate0_runtime_execution_contract"]
    sigmas = tuple(float(value) for value in schedule["sigma_grid_decimal"])
    frozen_timestep = float(schedule["timestep_by_step_decimal"][0])
    transformer = _FakeTransformer(small_shape)
    scheduler = _FakeScheduler(sigmas)
    adapter = _preflight_adapter(transformer, scheduler)
    hidden = np.zeros(small_shape, dtype="<f4")
    transformer_timestep = np.asarray(
        [
            frozen_timestep
            + (1.0 if mutation == "transformer_timestep" else 0.0)
        ],
        dtype="<f4",
    )
    conditional_encoder = np.asarray([2.0], dtype="<f4")
    conditional_attention = {"scale": 1.0}
    with pytest.raises(RuntimeError, match="漂移|冻结step0"):
        with adapter:
            conditional = transformer.forward(
                hidden_states=hidden,
                timestep=transformer_timestep,
                encoder_hidden_states=conditional_encoder,
                attention_kwargs=conditional_attention,
                return_dict=False,
            )[0]
            unconditional = transformer.forward(
                hidden_states=hidden,
                timestep=transformer_timestep,
                encoder_hidden_states=np.asarray([1.0], dtype="<f4"),
                attention_kwargs={"scale": 1.0},
                return_dict=False,
            )[0]
            cfg = np.ascontiguousarray(
                unconditional
                + np.float32(5.0)
                * np.subtract(
                    conditional,
                    unconditional,
                    dtype=np.float32,
                ),
                dtype="<f4",
            )
            if mutation == "conditional_encoder":
                conditional_encoder[0] = np.float32(3.0)
            elif mutation == "conditional_public_kwargs":
                conditional_attention["scale"] = 2.0
            scheduler_sample = (
                np.ones(small_shape, dtype="<f4")
                if mutation == "scheduler_sample"
                else hidden
            )
            scheduler.step(
                cfg,
                np.asarray(frozen_timestep, dtype="<f4"),
                scheduler_sample,
                return_dict=False,
            )
    with pytest.raises(RuntimeError, match="未完整完成"):
        adapter.batch(generator_state_digest_random="a" * 64)


def test_preflight_rejects_equal_raw_conditional_unconditional_encoder_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    small_shape = (1, 1, 1, 1, 2)
    monkeypatch.setattr(runtime, "TRANSFORMER_HIDDEN_SHAPE", small_shape)
    monkeypatch.setattr(runtime, "SCHEDULER_VELOCITY_SHAPE", small_shape)
    monkeypatch.setattr(gate_runner, "SCHEDULER_VELOCITY_SHAPE", small_shape)
    config = load_patch_relation_gate0_config()
    schedule = config["protocol_contract"]["gate0_runtime_execution_contract"]
    sigmas = tuple(float(value) for value in schedule["sigma_grid_decimal"])
    timestep = np.asarray(
        [float(schedule["timestep_by_step_decimal"][0])],
        dtype="<f4",
    )
    transformer = _FakeTransformer(small_shape)
    scheduler = _FakeScheduler(sigmas)
    adapter = _preflight_adapter(transformer, scheduler)
    hidden = np.zeros(small_shape, dtype="<f4")
    shared_encoder = np.asarray([2.0], dtype="<f4")
    with pytest.raises(RuntimeError, match="raw encoder values"):
        with adapter:
            transformer.forward(
                hidden_states=hidden,
                timestep=timestep,
                encoder_hidden_states=shared_encoder,
                return_dict=False,
            )
            transformer.forward(
                hidden_states=hidden,
                timestep=timestep,
                encoder_hidden_states=shared_encoder.copy(),
                return_dict=False,
            )
    with pytest.raises(RuntimeError, match="未完整完成"):
        adapter.batch(generator_state_digest_random="a" * 64)


def test_preflight_scope_rolls_back_partial_enter_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    small_shape = (1, 1, 1, 1, 2)
    monkeypatch.setattr(runtime, "TRANSFORMER_HIDDEN_SHAPE", small_shape)
    monkeypatch.setattr(runtime, "SCHEDULER_VELOCITY_SHAPE", small_shape)
    monkeypatch.setattr(gate_runner, "SCHEDULER_VELOCITY_SHAPE", small_shape)
    transformer = _FakeTransformer(small_shape)
    config = load_patch_relation_gate0_config()
    sigmas = tuple(
        float(value)
        for value in config["protocol_contract"][
            "gate0_runtime_execution_contract"
        ]["sigma_grid_decimal"]
    )
    scheduler = _StepInstallFailingScheduler(sigmas)
    original_forward = transformer.forward
    with pytest.raises(RuntimeError, match="install failure"):
        _preflight_adapter(transformer, scheduler).__enter__()
    assert transformer.forward.__func__ is original_forward.__func__
    assert transformer.forward.__self__ is original_forward.__self__
    assert "forward" not in transformer.__dict__


def test_preflight_scope_restores_after_downstream_failure_and_rejects_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    small_shape = (1, 1, 1, 1, 2)
    monkeypatch.setattr(runtime, "TRANSFORMER_HIDDEN_SHAPE", small_shape)
    monkeypatch.setattr(runtime, "SCHEDULER_VELOCITY_SHAPE", small_shape)
    monkeypatch.setattr(gate_runner, "SCHEDULER_VELOCITY_SHAPE", small_shape)
    config = load_patch_relation_gate0_config()
    schedule = config["protocol_contract"]["gate0_runtime_execution_contract"]
    sigmas = tuple(float(value) for value in schedule["sigma_grid_decimal"])
    transformer = _FakeTransformer(small_shape)
    scheduler = _FakeScheduler(sigmas)
    original_forward = transformer.forward
    original_step = scheduler.step
    adapter = _preflight_adapter(transformer, scheduler)
    hidden = np.zeros(small_shape, dtype="<f4")
    with pytest.raises(RuntimeError, match="downstream failure"):
        with adapter:
            transformer.forward(
                hidden_states=hidden,
                timestep=np.asarray(
                    [float(schedule["timestep_by_step_decimal"][0])],
                    dtype="<f4",
                ),
                encoder_hidden_states=np.asarray([2.0], dtype="<f4"),
            )
            raise RuntimeError("synthetic downstream failure after RoPE")
    assert transformer.forward.__func__ is original_forward.__func__
    assert transformer.forward.__self__ is original_forward.__self__
    assert scheduler.step.__func__ is original_step.__func__
    assert scheduler.step.__self__ is original_step.__self__
    with pytest.raises(RuntimeError, match="未完整完成"):
        adapter.batch(generator_state_digest_random="a" * 64)


def test_preflight_scope_cleanup_failure_restores_other_binding_and_rejects_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    small_shape = (1, 1, 1, 1, 2)
    monkeypatch.setattr(runtime, "TRANSFORMER_HIDDEN_SHAPE", small_shape)
    monkeypatch.setattr(runtime, "SCHEDULER_VELOCITY_SHAPE", small_shape)
    monkeypatch.setattr(gate_runner, "SCHEDULER_VELOCITY_SHAPE", small_shape)
    config = load_patch_relation_gate0_config()
    schedule = config["protocol_contract"]["gate0_runtime_execution_contract"]
    sigmas = tuple(float(value) for value in schedule["sigma_grid_decimal"])
    transformer = _ForwardCleanupFailingTransformer(small_shape)
    scheduler = _FakeScheduler(sigmas)
    original_step = scheduler.step
    adapter = _preflight_adapter(transformer, scheduler)
    hidden = np.zeros(small_shape, dtype="<f4")
    with pytest.raises(RuntimeError, match="synthetic body failure") as error:
        with adapter:
            raise RuntimeError("synthetic body failure")
    assert any(
        "cleanup also failed" in note
        for note in getattr(error.value, "__notes__", ())
    )
    assert scheduler.step.__func__ is original_step.__func__
    assert scheduler.step.__self__ is original_step.__self__
    with pytest.raises(RuntimeError, match="未完整完成"):
        adapter.batch(generator_state_digest_random="a" * 64)
    transformer.fail_forward_cleanup = False
    if "forward" in transformer.__dict__:
        delattr(transformer, "forward")


def test_preflight_repeated_exit_permanently_revokes_completed_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, completed_batch = _complete_fake_preflight_scope(monkeypatch)
    assert completed_batch.transformer_forward_count == 12
    with pytest.raises(RuntimeError, match="重复退出"):
        adapter.__exit__(None, None, None)
    with pytest.raises(RuntimeError, match="未完整完成"):
        adapter.batch(generator_state_digest_random="a" * 64)


def test_preflight_release_drops_runtime_references_but_preserves_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, batch = _complete_fake_preflight_scope(monkeypatch)
    transformer_reference = weakref.ref(adapter.transformer)
    scheduler_reference = weakref.ref(adapter.scheduler)
    adapter.release_runtime_references()
    gc.collect()
    assert adapter.transformer is None
    assert adapter.scheduler is None
    assert adapter.descriptor is None
    assert adapter._branch_kwargs == {}
    assert adapter._base == {}
    assert adapter._original_forward is None
    assert adapter._original_scheduler_step is None
    assert transformer_reference() is None
    assert scheduler_reference() is None
    validated = validate_phase_response_preflight_batch(batch)
    assert validated["formal_result"] is False
    adapter.release_runtime_references()


def test_real_preflight_releases_adapter_before_pipeline_and_cuda_cleanup() -> None:
    source = inspect.getsource(
        preflight.execute_real_patch_relation_phase_response_preflight
    )
    release_index = source.index(
        "adapter_to_release.release_runtime_references"
    )
    hook_index = source.index(
        'getattr(pipe, "maybe_free_model_hooks", None)'
    )
    gc_index = source.index("gc.collect,")
    cuda_index = source.index("torch.cuda.empty_cache,")
    assert release_index < hook_index < gc_index < cuda_index
    assert "finally:" in source


def test_runtime_cleanup_attempts_all_steps_without_masking_primary_error() -> None:
    primary_error = RuntimeError("synthetic primary runtime failure")
    calls: list[str] = []

    def successful_cleanup(label: str):
        def callback() -> None:
            calls.append(label)

        return callback

    def failing_cleanup(label: str, error: BaseException):
        def callback() -> None:
            calls.append(label)
            raise error

        return callback

    def exercise() -> None:
        try:
            raise primary_error
        finally:
            cleanup_errors: list[tuple[str, BaseException]] = []
            preflight._attempt_runtime_cleanup(
                cleanup_errors,
                "adapter_release",
                successful_cleanup("adapter_release"),
            )
            preflight._attempt_runtime_cleanup(
                cleanup_errors,
                "pipeline_model_hook_cleanup",
                successful_cleanup("pipeline_model_hook_cleanup"),
            )
            preflight._attempt_runtime_cleanup(
                cleanup_errors,
                "gc_collect",
                failing_cleanup("gc_collect", RuntimeError("gc failed")),
            )
            preflight._attempt_runtime_cleanup(
                cleanup_errors,
                "cuda_empty_cache",
                failing_cleanup(
                    "cuda_empty_cache",
                    ValueError("empty cache failed"),
                ),
            )
            preflight._finish_runtime_cleanup(
                primary_error,
                cleanup_errors,
            )

    with pytest.raises(RuntimeError) as observed:
        exercise()
    assert observed.value is primary_error
    assert calls == [
        "adapter_release",
        "pipeline_model_hook_cleanup",
        "gc_collect",
        "cuda_empty_cache",
    ]
    notes = getattr(primary_error, "__notes__", ())
    assert any("gc_collect=RuntimeError" in note for note in notes)
    assert any("cuda_empty_cache=ValueError" in note for note in notes)


def test_preflight_caught_completion_sentinel_never_issues_success_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, observed_error = _run_fake_preflight_with_intercepted_completion(
        monkeypatch,
        raise_after_completion=False,
    )
    assert observed_error is None
    with pytest.raises(RuntimeError, match="未完整完成"):
        adapter.batch(generator_state_digest_random="a" * 64)


def test_preflight_error_after_caught_completion_never_issues_success_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, observed_error = _run_fake_preflight_with_intercepted_completion(
        monkeypatch,
        raise_after_completion=True,
    )
    assert isinstance(observed_error, RuntimeError)
    assert "after intercepted completion" in str(observed_error)
    with pytest.raises(RuntimeError, match="未完整完成"):
        adapter.batch(generator_state_digest_random="a" * 64)
    adapter.release_runtime_references()
    assert adapter.transformer is None
    assert adapter.scheduler is None
    assert adapter._branch_kwargs == {}
    assert adapter._base == {}


def test_preflight_runner_writes_only_non_gate_diagnostic_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = _complete_fake_preflight_batch(monkeypatch)

    def fake_executor(config, gate0_config):
        return batch

    decision = run_patch_relation_phase_response_preflight(
        tmp_path / "run",
        runtime_executor=fake_executor,
    )
    assert decision["gate0_pass"] is False
    assert decision["full_eight_video_rerun_allowed"] is False
    assert decision["formal_result"] is False
    assert decision["stage_progression_allowed"] is False
    assert sorted(path.name for path in (tmp_path / "run").iterdir()) == [
        "patch_relation_phase_response_preflight_decision.json",
        "patch_relation_phase_response_preflight_manifest.json",
        "phase_response_preflight_record.json",
    ]


@pytest.mark.parametrize(
    ("field_name", "forged_value"),
    [
        ("formal_result", True),
        ("stage_progression_allowed", True),
        ("diagnostic_classification", "forged_pass"),
        ("diagnostic_candidates", ["forged_pass"]),
    ],
)
def test_preflight_batch_validator_rejects_forged_evidence_and_classification(
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
    forged_value: object,
) -> None:
    batch = _complete_fake_preflight_batch(monkeypatch)
    forged_record = dict(batch.record)
    forged_record[field_name] = forged_value
    forged_batch = PhaseResponsePreflightBatch(
        record=forged_record,
        candidate_evaluations=batch.candidate_evaluations,
        transformer_forward_count=batch.transformer_forward_count,
        scheduler_step_call_count=batch.scheduler_step_call_count,
        initial_hidden_state_digest_random=(
            batch.initial_hidden_state_digest_random
        ),
        generator_state_digest_random=batch.generator_state_digest_random,
    )
    with pytest.raises(ValueError):
        validate_phase_response_preflight_batch(forged_batch)


@pytest.mark.parametrize(
    "mutation",
    [
        "over_norm",
        "over_energy",
        "direction_failure",
        "energy_square_mismatch",
        "signed_exposure_mismatch",
    ],
)
def test_preflight_batch_recomputes_candidate_budget_and_direction_facts(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    batch = _complete_fake_preflight_batch(monkeypatch)
    record = dict(batch.record)
    evaluations = [
        dict(value) for value in record["candidate_evaluations"]
    ]
    candidate = evaluations[0]
    if mutation == "over_norm":
        candidate["actual_delta_norm"] = candidate["norm_budget"] * 10.0
        candidate["norm_guard_passed"] = True
    elif mutation == "over_energy":
        state_norm = float(np.sqrt(candidate["remaining_flow_energy"] + 1.0))
        candidate["state_update_delta_norm"] = state_norm
        candidate["energy_increment"] = state_norm**2
        candidate["signed_state_update_exposure"] = state_norm
        candidate["energy_guard_passed"] = True
    elif mutation == "direction_failure":
        candidate["direction_cosine"] = 0.0
        candidate["direction_guard_passed"] = True
    elif mutation == "energy_square_mismatch":
        candidate["energy_increment"] += 1.0
    elif mutation == "signed_exposure_mismatch":
        candidate["signed_state_update_exposure"] += 1.0
    else:  # pragma: no cover - parametrization is frozen above
        raise AssertionError(mutation)
    candidate["feasible"] = True
    record["candidate_evaluations"] = evaluations
    record.update(classify_phase_response_preflight(record))
    forged_batch = PhaseResponsePreflightBatch(
        record=record,
        candidate_evaluations=batch.candidate_evaluations,
        transformer_forward_count=batch.transformer_forward_count,
        scheduler_step_call_count=batch.scheduler_step_call_count,
        initial_hidden_state_digest_random=(
            batch.initial_hidden_state_digest_random
        ),
        generator_state_digest_random=batch.generator_state_digest_random,
    )
    with pytest.raises(ValueError, match="candidate"):
        validate_phase_response_preflight_batch(forged_batch)


def test_preflight_batch_requires_original_runtime_issued_candidate_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = _complete_fake_preflight_batch(monkeypatch)
    forged_batch = PhaseResponsePreflightBatch(
        record=batch.record,
        candidate_evaluations=tuple(
            replace(value) for value in batch.candidate_evaluations
        ),
        transformer_forward_count=batch.transformer_forward_count,
        scheduler_step_call_count=batch.scheduler_step_call_count,
        initial_hidden_state_digest_random=(
            batch.initial_hidden_state_digest_random
        ),
        generator_state_digest_random=batch.generator_state_digest_random,
    )
    with pytest.raises(ValueError, match="raw-array"):
        validate_phase_response_preflight_batch(forged_batch)


@pytest.mark.parametrize("mutation", ["coordinated_scalars", "delta_sigma"])
def test_preflight_batch_rejects_coordinated_candidate_source_forgery(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    batch = _complete_fake_preflight_batch(monkeypatch)
    record = dict(batch.record)
    evaluations = [
        dict(value) for value in record["candidate_evaluations"]
    ]
    for candidate in evaluations:
        if mutation == "coordinated_scalars":
            candidate.update(
                {
                    "base_velocity_norm": 1_000_000.0,
                    "norm_budget": 2400.0,
                    "actual_delta_norm": 1.0,
                    "reference_energy_increment": 1_000_000.0,
                    "projected_reference_energy": 8_000_000.0,
                    "total_flow_energy_budget": 120.0,
                    "remaining_flow_energy": 120.0,
                    "state_update_delta_norm": 1.0,
                    "energy_increment": 1.0,
                    "signed_state_update_exposure": float(
                        candidate["signed_coefficient"]
                    ),
                    "direction_cosine": 1.0,
                    "norm_guard_passed": True,
                    "energy_guard_passed": True,
                    "direction_guard_passed": True,
                    "feasible": True,
                }
            )
        elif mutation == "delta_sigma":
            candidate["delta_sigma"] = -999.0
        else:  # pragma: no cover - parametrization is frozen above
            raise AssertionError(mutation)
    record["candidate_evaluations"] = evaluations
    record.update(classify_phase_response_preflight(record))
    forged_batch = PhaseResponsePreflightBatch(
        record=record,
        candidate_evaluations=batch.candidate_evaluations,
        transformer_forward_count=batch.transformer_forward_count,
        scheduler_step_call_count=batch.scheduler_step_call_count,
        initial_hidden_state_digest_random=(
            batch.initial_hidden_state_digest_random
        ),
        generator_state_digest_random=batch.generator_state_digest_random,
    )
    with pytest.raises(ValueError, match="runtime-issued|identity"):
        validate_phase_response_preflight_batch(forged_batch)


@pytest.mark.parametrize(
    "mutation",
    [
        "base_bool",
        "control_repeat_bool",
        "control_base_bool",
        "digest_equal_nonzero_base_norm",
        "digest_equal_nonzero_control_repeat_norm",
        "coordinated_control_base_digest_bool",
    ],
)
def test_preflight_batch_recomputes_cfg_digest_boolean_and_zero_norm_relations(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    batch = _complete_fake_preflight_batch(monkeypatch)
    record = dict(batch.record)
    record["base_cfg_digests"] = list(record["base_cfg_digests"])
    record["control_cfg_digests_by_sign_and_repeat"] = list(
        record["control_cfg_digests_by_sign_and_repeat"]
    )
    record["control_cfg_repeats_byte_exact_by_sign"] = list(
        record["control_cfg_repeats_byte_exact_by_sign"]
    )
    record["control_cfg_equals_zero_base_by_sign_and_repeat"] = list(
        record["control_cfg_equals_zero_base_by_sign_and_repeat"]
    )
    if mutation == "base_bool":
        record["base_cfg_repeats_byte_exact"] = False
    elif mutation == "control_repeat_bool":
        record["control_cfg_repeats_byte_exact_by_sign"][0] = False
    elif mutation == "control_base_bool":
        record["control_cfg_equals_zero_base_by_sign_and_repeat"][0] = True
    elif mutation == "digest_equal_nonzero_base_norm":
        record["base_cfg_repeat_delta_norm"] = 1.0
    elif mutation == "digest_equal_nonzero_control_repeat_norm":
        record["control_repeat_delta_norm_by_sign"] = list(
            record["control_repeat_delta_norm_by_sign"]
        )
        record["control_repeat_delta_norm_by_sign"][0] = 1.0
    elif mutation == "coordinated_control_base_digest_bool":
        record["control_cfg_digests_by_sign_and_repeat"][0] = (
            record["base_cfg_digests"][0]
        )
        record["control_cfg_equals_zero_base_by_sign_and_repeat"][0] = True
    else:  # pragma: no cover - parametrization is frozen above
        raise AssertionError(mutation)
    forged_batch = PhaseResponsePreflightBatch(
        record=record,
        candidate_evaluations=batch.candidate_evaluations,
        transformer_forward_count=batch.transformer_forward_count,
        scheduler_step_call_count=batch.scheduler_step_call_count,
        initial_hidden_state_digest_random=(
            batch.initial_hidden_state_digest_random
        ),
        generator_state_digest_random=batch.generator_state_digest_random,
    )
    with pytest.raises(ValueError, match="digest"):
        validate_phase_response_preflight_batch(forged_batch)


def test_preflight_byte_exact_digest_semantics_accept_ieee_signed_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    positive_zero = np.asarray([0.0], dtype="<f4")
    negative_zero = np.asarray([-0.0], dtype="<f4")
    assert np.array_equal(positive_zero, negative_zero)
    assert preflight._float64_l2(
        np.subtract(positive_zero, negative_zero, dtype=np.float32)
    ) == 0.0
    assert preflight._array_digest(positive_zero) != preflight._array_digest(
        negative_zero
    )

    batch = _complete_fake_preflight_batch(monkeypatch)
    record = dict(batch.record)
    record["base_cfg_digests"] = list(record["base_cfg_digests"])
    record["base_cfg_digests"][1] = "f" * 64
    record["base_cfg_repeats_byte_exact"] = False
    signed_zero_batch = PhaseResponsePreflightBatch(
        record=record,
        candidate_evaluations=batch.candidate_evaluations,
        transformer_forward_count=batch.transformer_forward_count,
        scheduler_step_call_count=batch.scheduler_step_call_count,
        initial_hidden_state_digest_random=(
            batch.initial_hidden_state_digest_random
        ),
        generator_state_digest_random=batch.generator_state_digest_random,
    )
    validated = validate_phase_response_preflight_batch(signed_zero_batch)
    assert validated["base_cfg_repeat_delta_norm"] == 0.0
    assert validated["base_cfg_repeats_byte_exact"] is False


def test_preflight_runner_recovers_without_record_or_manifest_for_forged_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = _complete_fake_preflight_batch(monkeypatch)
    record = dict(batch.record)
    evaluations = [
        dict(value) for value in record["candidate_evaluations"]
    ]
    for candidate in evaluations:
        candidate.update(
            {
                "base_velocity_norm": 1_000_000.0,
                "norm_budget": 2400.0,
                "actual_delta_norm": 1.0,
                "reference_energy_increment": 1_000_000.0,
                "projected_reference_energy": 8_000_000.0,
                "total_flow_energy_budget": 120.0,
                "remaining_flow_energy": 120.0,
                "state_update_delta_norm": 1.0,
                "energy_increment": 1.0,
                "signed_state_update_exposure": float(
                    candidate["signed_coefficient"]
                ),
                "direction_cosine": 1.0,
                "norm_guard_passed": True,
                "energy_guard_passed": True,
                "direction_guard_passed": True,
                "feasible": True,
            }
        )
    record["candidate_evaluations"] = evaluations
    record.update(classify_phase_response_preflight(record))
    forged_batch = PhaseResponsePreflightBatch(
        record=record,
        candidate_evaluations=batch.candidate_evaluations,
        transformer_forward_count=batch.transformer_forward_count,
        scheduler_step_call_count=batch.scheduler_step_call_count,
        initial_hidden_state_digest_random=(
            batch.initial_hidden_state_digest_random
        ),
        generator_state_digest_random=batch.generator_state_digest_random,
    )
    output = tmp_path / "forged_budget"
    with pytest.raises(ValueError, match="candidate"):
        run_patch_relation_phase_response_preflight(
            output,
            runtime_executor=lambda config, gate0_config: forged_batch,
        )
    decision = json.loads(
        (
            output
            / "patch_relation_phase_response_preflight_decision.json"
        ).read_text(encoding="utf-8")
    )
    assert decision["formal_result"] is False
    assert decision["stage_progression_allowed"] is False
    assert decision["phase_response_preflight_decision"] == (
        "runtime_or_contract_failure_recovery_only"
    )
    assert not (output / "phase_response_preflight_record.json").exists()
    assert not (
        output / "patch_relation_phase_response_preflight_manifest.json"
    ).exists()


@pytest.mark.parametrize(
    "failed_success_artifact",
    [
        "patch_relation_phase_response_preflight_decision.json",
        "patch_relation_phase_response_preflight_manifest.json",
    ],
)
def test_preflight_success_artifact_write_failure_is_recovery_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_success_artifact: str,
) -> None:
    batch = _complete_fake_preflight_batch(monkeypatch)
    original_write_artifact = (
        preflight._write_success_artifact_by_directory_fd
    )
    injected = False

    def flaky_write_artifact(
        directory_fd: int,
        filename: str,
        value: dict[str, object],
    ) -> None:
        nonlocal injected
        if not injected and filename == failed_success_artifact:
            injected = True
            raise OSError("synthetic success artifact writer failure")
        original_write_artifact(directory_fd, filename, value)

    monkeypatch.setattr(
        preflight,
        "_write_success_artifact_by_directory_fd",
        flaky_write_artifact,
    )
    output = tmp_path / f"writer_failure_{failed_success_artifact}"
    with pytest.raises(OSError, match="writer failure"):
        run_patch_relation_phase_response_preflight(
            output,
            runtime_executor=lambda config, gate0_config: batch,
        )
    assert injected is True
    assert sorted(path.name for path in output.iterdir()) == [
        "patch_relation_phase_response_preflight_decision.json"
    ]
    decision = json.loads(
        (
            output
            / "patch_relation_phase_response_preflight_decision.json"
        ).read_text(encoding="utf-8")
    )
    assert decision["phase_response_preflight_decision"] == (
        "runtime_or_contract_failure_recovery_only"
    )
    assert decision["formal_result"] is False
    assert decision["stage_progression_allowed"] is False
    assert not list(
        output.parent.glob(
            f".{output.name}.phase_response_success_staging.*"
        )
    )


@pytest.mark.parametrize("collision_kind", ["directory", "directory_symlink"])
def test_preflight_staging_collision_preserves_external_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    collision_kind: str,
) -> None:
    batch = _complete_fake_preflight_batch(monkeypatch)
    token = "c" * 32
    monkeypatch.setattr(preflight.secrets, "token_hex", lambda size: token)
    output = tmp_path / f"staging_collision_{collision_kind}"
    collision = output.with_name(
        f".{output.name}.phase_response_success_staging.{token}"
    )
    if collision_kind == "directory":
        protected_root = collision
        protected_root.mkdir()
    else:
        protected_root = tmp_path / "protected_symlink_target"
        protected_root.mkdir()
        collision.symlink_to(protected_root, target_is_directory=True)
    sentinel = protected_root / "sentinel.txt"
    sentinel.write_text("must remain unchanged", encoding="utf-8")

    with pytest.raises(FileExistsError):
        run_patch_relation_phase_response_preflight(
            output,
            runtime_executor=lambda config, gate0_config: batch,
        )
    assert collision.exists()
    if collision_kind == "directory_symlink":
        assert collision.is_symlink()
    assert sentinel.read_text(encoding="utf-8") == "must remain unchanged"
    assert sorted(path.name for path in output.iterdir()) == [
        "patch_relation_phase_response_preflight_decision.json"
    ]
    decision = json.loads(
        (
            output
            / "patch_relation_phase_response_preflight_decision.json"
        ).read_text(encoding="utf-8")
    )
    assert decision["phase_response_preflight_decision"] == (
        "runtime_or_contract_failure_recovery_only"
    )


def test_preflight_post_promote_symlink_swap_is_recovery_only_and_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = _complete_fake_preflight_batch(monkeypatch)
    output = tmp_path / "post_promote_symlink_swap"
    external_target = tmp_path / "external_target"
    external_target.mkdir()
    sentinel = external_target / "sentinel.txt"
    sentinel.write_text("must remain unchanged", encoding="utf-8")
    displaced_owned_directory = tmp_path / "displaced_owned_staging"
    original_replace = Path.replace
    original_os_open = preflight.os.open
    original_os_close = preflight.os.close
    owned_directory_fds: list[int] = []
    artifact_fds: list[int] = []
    injected = False

    def track_owned_directory_fd(*args, **kwargs) -> int:
        descriptor = original_os_open(*args, **kwargs)
        if kwargs.get("dir_fd") is None:
            owned_directory_fds.append(descriptor)
        else:
            artifact_fds.append(descriptor)
        return descriptor

    def close_then_report_failure(descriptor: int) -> None:
        original_os_close(descriptor)
        if descriptor in owned_directory_fds:
            raise OSError("synthetic owned directory fd close failure")

    def replace_with_symlink(source: Path, target: Path) -> Path:
        nonlocal injected
        source_path = Path(source)
        target_path = Path(target)
        if (
            not injected
            and ".phase_response_success_staging." in source_path.name
            and target_path == output
        ):
            injected = True
            original_replace(source_path, displaced_owned_directory)
            source_path.symlink_to(
                external_target,
                target_is_directory=True,
            )
        return original_replace(source_path, target_path)

    monkeypatch.setattr(preflight.os, "open", track_owned_directory_fd)
    monkeypatch.setattr(preflight.os, "close", close_then_report_failure)
    monkeypatch.setattr(Path, "replace", replace_with_symlink)
    with pytest.raises(
        RuntimeError,
        match="promoted output ownership",
    ) as observed:
        run_patch_relation_phase_response_preflight(
            output,
            runtime_executor=lambda config, gate0_config: batch,
        )
    assert injected is True
    assert owned_directory_fds
    assert artifact_fds
    assert any(
        "OSError" in note
        for note in getattr(observed.value, "__notes__", ())
    )
    for descriptor in (*owned_directory_fds, *artifact_fds):
        with pytest.raises(OSError):
            os.fstat(descriptor)
    assert sentinel.read_text(encoding="utf-8") == "must remain unchanged"
    assert output.is_dir()
    assert not output.is_symlink()
    assert sorted(path.name for path in output.iterdir()) == [
        "patch_relation_phase_response_preflight_decision.json"
    ]
    decision = json.loads(
        (
            output
            / "patch_relation_phase_response_preflight_decision.json"
        ).read_text(encoding="utf-8")
    )
    assert decision["phase_response_preflight_decision"] == (
        "runtime_or_contract_failure_recovery_only"
    )
    assert decision["formal_result"] is False
    assert decision["stage_progression_allowed"] is False
    assert displaced_owned_directory.is_dir()
    assert list(displaced_owned_directory.iterdir()) == []


def test_preflight_prewrite_symlink_swap_never_writes_external_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = _complete_fake_preflight_batch(monkeypatch)
    output = tmp_path / "prewrite_symlink_swap"
    external_target = tmp_path / "prewrite_external_target"
    external_target.mkdir()
    sentinel = external_target / "sentinel.txt"
    sentinel.write_text("must remain unchanged", encoding="utf-8")
    displaced_owned_directory = tmp_path / "prewrite_displaced_owned_staging"
    original_write_artifact = (
        preflight._write_success_artifact_by_directory_fd
    )
    original_os_open = preflight.os.open
    opened_fds: list[int] = []
    displaced_staging_path: Path | None = None

    def track_open(*args, **kwargs) -> int:
        descriptor = original_os_open(*args, **kwargs)
        opened_fds.append(descriptor)
        return descriptor

    def swap_before_first_write(
        directory_fd: int,
        filename: str,
        value: dict[str, object],
    ) -> None:
        nonlocal displaced_staging_path
        if displaced_staging_path is None:
            candidates = list(
                output.parent.glob(
                    f".{output.name}.phase_response_success_staging.*"
                )
            )
            assert len(candidates) == 1
            displaced_staging_path = candidates[0]
            displaced_staging_path.replace(displaced_owned_directory)
            displaced_staging_path.symlink_to(
                external_target,
                target_is_directory=True,
            )
        original_write_artifact(directory_fd, filename, value)

    monkeypatch.setattr(preflight.os, "open", track_open)
    monkeypatch.setattr(
        preflight,
        "_write_success_artifact_by_directory_fd",
        swap_before_first_write,
    )
    with pytest.raises(RuntimeError, match="staging ownership"):
        run_patch_relation_phase_response_preflight(
            output,
            runtime_executor=lambda config, gate0_config: batch,
        )
    assert displaced_staging_path is not None
    assert displaced_staging_path.is_symlink()
    assert sorted(path.name for path in external_target.iterdir()) == [
        "sentinel.txt"
    ]
    assert sentinel.read_text(encoding="utf-8") == "must remain unchanged"
    assert displaced_owned_directory.is_dir()
    assert list(displaced_owned_directory.iterdir()) == []
    assert sorted(path.name for path in output.iterdir()) == [
        "patch_relation_phase_response_preflight_decision.json"
    ]
    decision = json.loads(
        (
            output
            / "patch_relation_phase_response_preflight_decision.json"
        ).read_text(encoding="utf-8")
    )
    assert decision["phase_response_preflight_decision"] == (
        "runtime_or_contract_failure_recovery_only"
    )
    assert decision["formal_result"] is False
    assert decision["stage_progression_allowed"] is False
    assert opened_fds
    for descriptor in opened_fds:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_preflight_success_closes_owned_staging_directory_fd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = _complete_fake_preflight_batch(monkeypatch)
    original_os_open = preflight.os.open
    opened_fds: list[int] = []

    def track_owned_directory_fd(*args, **kwargs) -> int:
        descriptor = original_os_open(*args, **kwargs)
        opened_fds.append(descriptor)
        return descriptor

    monkeypatch.setattr(preflight.os, "open", track_owned_directory_fd)
    output = tmp_path / "success_directory_fd"
    decision = run_patch_relation_phase_response_preflight(
        output,
        runtime_executor=lambda config, gate0_config: batch,
    )
    assert decision["phase_response_preflight_decision"] == (
        "single_step_diagnostic_completed_no_gate_decision"
    )
    assert opened_fds
    for descriptor in opened_fds:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_preflight_bad_self_digest_leaves_only_recovery_decision(
    tmp_path: Path,
) -> None:
    config = json.loads(Path(DEFAULT_CONFIG_PATH).read_text(encoding="utf-8"))
    config["protocol_digest"] = "0" * 64
    config_path = tmp_path / "bad_self_digest.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "bad_self_digest_run"
    with pytest.raises(ValueError, match="self digest"):
        run_patch_relation_phase_response_preflight(
            output,
            config_path=config_path,
            runtime_executor=lambda config, gate0_config: pytest.fail(
                "runtime executor must not run"
            ),
        )
    assert sorted(path.name for path in output.iterdir()) == [
        "patch_relation_phase_response_preflight_decision.json"
    ]
    decision = json.loads(
        (
            output
            / "patch_relation_phase_response_preflight_decision.json"
        ).read_text(encoding="utf-8")
    )
    assert decision["phase_response_preflight_decision"] == (
        "runtime_or_contract_failure_recovery_only"
    )
    assert decision["formal_result"] is False
    assert decision["stage_progression_allowed"] is False


def test_preflight_bad_source_binding_leaves_only_recovery_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate0_config = json.loads(
        json.dumps(load_patch_relation_gate0_config())
    )
    gate0_config["protocol_digest"] = "f" * 64
    monkeypatch.setattr(
        preflight,
        "load_patch_relation_gate0_config",
        lambda path: gate0_config,
    )
    output = tmp_path / "bad_source_binding_run"
    with pytest.raises(RuntimeError, match="source Gate0"):
        run_patch_relation_phase_response_preflight(
            output,
            runtime_executor=lambda config, gate0_config: pytest.fail(
                "runtime executor must not run"
            ),
        )
    assert sorted(path.name for path in output.iterdir()) == [
        "patch_relation_phase_response_preflight_decision.json"
    ]
    decision = json.loads(
        (
            output
            / "patch_relation_phase_response_preflight_decision.json"
        ).read_text(encoding="utf-8")
    )
    assert decision["phase_response_preflight_decision"] == (
        "runtime_or_contract_failure_recovery_only"
    )
    assert decision["formal_result"] is False
    assert decision["stage_progression_allowed"] is False


def test_preflight_runner_turns_forged_batch_into_nonformal_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = _complete_fake_preflight_batch(monkeypatch)
    forged_record = dict(batch.record)
    forged_record.update(
        {
            "formal_result": True,
            "stage_progression_allowed": True,
            "diagnostic_classification": "forged_pass",
            "diagnostic_candidates": ["forged_pass"],
        }
    )
    forged_batch = PhaseResponsePreflightBatch(
        record=forged_record,
        candidate_evaluations=batch.candidate_evaluations,
        transformer_forward_count=12,
        scheduler_step_call_count=0,
        initial_hidden_state_digest_random="1" * 64,
        generator_state_digest_random="2" * 64,
    )

    with pytest.raises(ValueError):
        run_patch_relation_phase_response_preflight(
            tmp_path / "forged",
            runtime_executor=lambda config, gate0_config: forged_batch,
        )
    decision = json.loads(
        (
            tmp_path
            / "forged"
            / "patch_relation_phase_response_preflight_decision.json"
        ).read_text(encoding="utf-8")
    )
    assert decision["formal_result"] is False
    assert decision["stage_progression_allowed"] is False
    assert decision["phase_response_preflight_decision"] == (
        "runtime_or_contract_failure_recovery_only"
    )
    assert not (
        tmp_path / "forged" / "phase_response_preflight_record.json"
    ).exists()
