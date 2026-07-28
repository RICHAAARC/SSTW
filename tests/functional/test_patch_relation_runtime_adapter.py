from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from evaluation.protocol.patch_relation_gate0_contract import (
    DEFAULT_CONFIG_PATH,
    EXPECTED_WAN_RUNTIME_ADAPTER_CONTRACT,
    FROZEN_PROTOCOL_DIGEST,
    load_patch_relation_gate0_config,
    protocol_digest,
)
from main.methods.state_space_watermark.patch_relation_carrier import (
    ROPE_TUPLE_SHAPE,
    build_relation_phase_delta,
    build_public_patch_relation_descriptor,
)
from main.methods.state_space_watermark.patch_relation_wan_runtime import (
    CfgRopeApplicationPair,
    PatchRelationRuntimeGuardError,
    SCHEDULER_VELOCITY_SHAPE,
    ScopedWanRopeOutputAdapter,
    TRANSFORMER_HIDDEN_SHAPE,
    RUNTIME_ADAPTER_PROTOCOL_DIGEST,
    apply_wan_rotary_phase_runtime,
    measure_cfg_state_update_numpy,
    validate_cfg_state_update_measurement_numpy,
    validate_cfg_rope_application_pair,
)
import main.methods.state_space_watermark.patch_relation_wan_runtime as runtime_module
import experiments.generative_video_model_probe.patch_relation_gate0_construction as runner_module
from experiments.generative_video_model_probe.patch_relation_gate0_construction import (
    ScopedPatchRelationWanProbeAdapter,
)


pytestmark = pytest.mark.quick

_BINDING_DIGEST = "1" * 64


def test_runtime_adapter_config_is_exact_and_execution_stays_closed() -> None:
    config = load_patch_relation_gate0_config()
    contract = config["protocol_contract"]
    assert protocol_digest(contract) == FROZEN_PROTOCOL_DIGEST
    assert FROZEN_PROTOCOL_DIGEST == RUNTIME_ADAPTER_PROTOCOL_DIGEST
    assert (
        contract["wan_runtime_adapter_contract"]
        == EXPECTED_WAN_RUNTIME_ADAPTER_CONTRACT
    )
    assert contract["method_scope"][
        "local_wan_rope_runtime_adapter_implemented"
    ]
    assert contract["method_scope"][
        "local_cfg_state_update_measurement_adapter_implemented"
    ]
    assert config["authorization_boundary"][
        "runtime_implementation_authorized"
    ]
    assert config["authorization_boundary"]["gpu_execution_allowed"]
    assert config["authorization_boundary"]["colab_execution_allowed"]
    assert not config["authorization_boundary"]["formal_result"]
    assert not config["authorization_boundary"]["stage_progression_allowed"]


def test_rehashed_official_rope_storage_dtype_mutation_fails_closed(
    tmp_path: Path,
) -> None:
    config = json.loads(Path(DEFAULT_CONFIG_PATH).read_text(encoding="utf-8"))
    config["protocol_contract"]["wan_runtime_adapter_contract"][
        "official_rope_output_dtype_non_mps"
    ] = "float64"
    config["protocol_digest"] = protocol_digest(config["protocol_contract"])
    path = tmp_path / "runtime_dtype_mutation.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="adapter boundary"):
        load_patch_relation_gate0_config(path)


@pytest.mark.parametrize(
    ("section", "field", "mutated"),
    [
        ("scoped_rope_output_contract", "expected_successful_rope_calls", 2),
        (
            "scoped_rope_output_contract",
            "record_requires_clean_body_and_completed_cleanup",
            False,
        ),
        (
            "cfg_branch_contract",
            "same_relation_control_on_conditional_and_unconditional",
            False,
        ),
        (
            "scheduler_state_update_measurement_contract",
            "minimum_direction_cosine_decimal",
            "0.0",
        ),
        (
            "scheduler_state_update_measurement_contract",
            "local_measurement_is_execution_evidence",
            True,
        ),
    ],
)
def test_rehashed_runtime_contract_mutations_fail_closed(
    tmp_path: Path,
    section: str,
    field: str,
    mutated,
) -> None:
    config = json.loads(Path(DEFAULT_CONFIG_PATH).read_text(encoding="utf-8"))
    config["protocol_contract"]["wan_runtime_adapter_contract"][section][
        field
    ] = mutated
    config["protocol_digest"] = protocol_digest(config["protocol_contract"])
    path = tmp_path / "runtime_mutation.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="adapter boundary"):
        load_patch_relation_gate0_config(path)


class _FakeRope:
    def __init__(self) -> None:
        self.cosine = np.ones(ROPE_TUPLE_SHAPE, dtype="<f4")
        self.sine = np.zeros(ROPE_TUPLE_SHAPE, dtype="<f4")
        self.call_count = 0

    def forward(self, hidden_states: np.ndarray):
        self.call_count += 1
        return self.cosine, self.sine

    def __call__(self, hidden_states: np.ndarray):
        return self.forward(hidden_states)


class _FailingRope(_FakeRope):
    def forward(self, hidden_states: np.ndarray):
        self.call_count += 1
        raise RuntimeError("synthetic rope failure")


class _CleanupFailingRope(_FakeRope):
    def __init__(self) -> None:
        super().__init__()
        self.fail_scope_cleanup = True

    def __delattr__(self, name: str) -> None:
        if (
            name == "_sstw_patch_relation_rope_scope_active"
            and self.fail_scope_cleanup
        ):
            raise RuntimeError("synthetic scope cleanup failure")
        super().__delattr__(name)


class _FakeTransformer:
    def __init__(self, rope: _FakeRope | None = None) -> None:
        self.rope = rope or _FakeRope()


def _hidden_states() -> np.ndarray:
    return np.zeros(TRANSFORMER_HIDDEN_SHAPE, dtype="<f4")


def _run_branch(
    transformer: _FakeTransformer,
    *,
    control_role: str,
    branch_role: str,
    coefficient: int,
    probe_id: str = "patch_relation_probe",
    step_index: int = 3,
    binding_digest: str = _BINDING_DIGEST,
):
    descriptor = build_public_patch_relation_descriptor()
    scope = ScopedWanRopeOutputAdapter(
        transformer,
        descriptor=descriptor,
        signed_coefficient=coefficient,
        probe_id=probe_id,
        step_index=step_index,
        control_role=control_role,
        cfg_branch_role=branch_role,
        input_binding_digest=binding_digest,
    )
    with scope:
        result = transformer.rope(_hidden_states())
    return result, scope.record()


def _run_pair(
    *,
    control_role: str,
    coefficient: int,
    probe_id: str = "patch_relation_probe",
    step_index: int = 3,
    binding_digest: str = _BINDING_DIGEST,
) -> CfgRopeApplicationPair:
    transformer = _FakeTransformer()
    _, conditional = _run_branch(
        transformer,
        control_role=control_role,
        branch_role="conditional",
        coefficient=coefficient,
        probe_id=probe_id,
        step_index=step_index,
        binding_digest=binding_digest,
    )
    _, unconditional = _run_branch(
        transformer,
        control_role=control_role,
        branch_role="unconditional",
        coefficient=coefficient,
        probe_id=probe_id,
        step_index=step_index,
        binding_digest=binding_digest,
    )
    return validate_cfg_rope_application_pair(conditional, unconditional)


def _velocity(value: float) -> np.ndarray:
    return np.full(SCHEDULER_VELOCITY_SHAPE, value, dtype="<f4")


def _scheduler_state_arguments(
    base_conditional: np.ndarray,
    base_unconditional: np.ndarray,
    controlled_conditional: np.ndarray,
    controlled_unconditional: np.ndarray,
    *,
    delta_sigma: float,
) -> dict[str, np.ndarray]:
    base_cfg = np.ascontiguousarray(
        base_unconditional
        + np.float32(5.0) * (base_conditional - base_unconditional),
        dtype="<f4",
    )
    controlled_cfg = np.ascontiguousarray(
        controlled_unconditional
        + np.float32(5.0)
        * (controlled_conditional - controlled_unconditional),
        dtype="<f4",
    )
    sample = np.ones_like(base_cfg, dtype="<f4")
    base_next = np.ascontiguousarray(
        sample + np.float32(delta_sigma) * base_cfg,
        dtype="<f4",
    )
    controlled_next = np.ascontiguousarray(
        sample + np.float32(delta_sigma) * controlled_cfg,
        dtype="<f4",
    )
    return {
        "scheduler_consumed_velocity": controlled_cfg,
        "scheduler_sample": sample,
        "scheduler_base_next_state": base_next,
        "scheduler_controlled_next_state": controlled_next,
    }


def _round_float32_to_bfloat16_values(values) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    bits = array.view(np.uint32)
    rounding = np.uint32(0x7FFF) + (
        (bits >> np.uint32(16)) & np.uint32(1)
    )
    return ((bits + rounding) & np.uint32(0xFFFF0000)).view(np.float32)


class _FakeProbeTransformer:
    def __init__(self, velocity_shape: tuple[int, ...]) -> None:
        self.rope = _FakeRope()
        self.velocity_shape = velocity_shape
        self.dtype = np.dtype("<f4")
        self.is_cache_enabled = False
        self._cache_config = None
        self.original_forward_count = 0

    def forward(
        self,
        *,
        hidden_states: np.ndarray,
        timestep: np.ndarray,
        encoder_hidden_states: np.ndarray,
        **kwargs,
    ):
        self.original_forward_count += 1
        _cosine, sine = self.rope(hidden_states)
        relation_signal = float(sine[0, 2221, 0, 1])
        branch_offset = float(encoder_hidden_states.reshape(-1)[0]) * 0.01
        value = np.float32(1.0 + branch_offset + relation_signal * 1e-3)
        return (np.full(self.velocity_shape, value, dtype="<f4"),)


class _FakeProbeScheduler:
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

    def step(
        self,
        model_output: np.ndarray,
        timestep: np.ndarray,
        sample: np.ndarray,
        *args,
        **kwargs,
    ):
        if self._step_index is None:
            self._step_index = 0
        index = self._step_index
        self.original_step_count += 1
        delta = np.float32(self.sigmas[index + 1] - self.sigmas[index])
        result = (
            np.ascontiguousarray(
                sample + delta * model_output,
                dtype="<f4",
            ),
        )
        self._step_index += 1
        return result


class _StepInstallFailingScheduler(_FakeProbeScheduler):
    def __init__(self, sigmas: tuple[float, ...]) -> None:
        super().__init__(sigmas)
        self.fail_step_install = True

    def __setattr__(self, name: str, value) -> None:
        if (
            name == "step"
            and getattr(self, "fail_step_install", False)
        ):
            raise RuntimeError("synthetic scheduler step install failure")
        super().__setattr__(name, value)


class _ScopeCleanupFailingScheduler(_FakeProbeScheduler):
    def __init__(self, sigmas: tuple[float, ...]) -> None:
        super().__init__(sigmas)
        self.fail_scope_cleanup = True

    def __delattr__(self, name: str) -> None:
        if (
            name
            == ScopedPatchRelationWanProbeAdapter._ACTIVE_SCHEDULER_ATTRIBUTE
            and self.fail_scope_cleanup
        ):
            raise RuntimeError("synthetic probe scope cleanup failure")
        super().__delattr__(name)


class _StepDriftScheduler(_FakeProbeScheduler):
    def step(self, *args, **kwargs):
        result = super().step(*args, **kwargs)
        self._step_index += 1
        return result


def _probe_adapter(
    transformer: _FakeProbeTransformer,
    scheduler: _FakeProbeScheduler,
    *,
    coefficient: int = 1,
) -> ScopedPatchRelationWanProbeAdapter:
    config = load_patch_relation_gate0_config()
    schedule = config["protocol_contract"]["gate0_runtime_execution_contract"]
    return ScopedPatchRelationWanProbeAdapter(
        transformer,
        scheduler,
        descriptor=build_public_patch_relation_descriptor(),
        probe_id="patch_relation_probe",
        identity_id="patch_relation_test_identity",
        signed_coefficient=coefficient,
        sigma_grid=tuple(
            float(value) for value in schedule["sigma_grid_decimal"]
        ),
        delta_sigma_by_step=tuple(
            float(value)
            for value in schedule["delta_sigma_by_step_decimal"]
        ),
        timestep_by_step=tuple(
            float(value)
            for value in schedule["timestep_by_step_decimal"]
        ),
    )


def _run_fake_probe_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    *,
    coefficient: int,
    timestep_values: tuple[float, ...] | None = None,
    scheduler_type: type[_FakeProbeScheduler] = _FakeProbeScheduler,
):
    small_shape = (1, 1, 1, 1, 2)
    monkeypatch.setattr(
        runtime_module,
        "TRANSFORMER_HIDDEN_SHAPE",
        small_shape,
    )
    monkeypatch.setattr(
        runtime_module,
        "SCHEDULER_VELOCITY_SHAPE",
        small_shape,
    )
    monkeypatch.setattr(
        runner_module,
        "SCHEDULER_VELOCITY_SHAPE",
        small_shape,
    )
    config = load_patch_relation_gate0_config()
    schedule = config["protocol_contract"]["gate0_runtime_execution_contract"]
    sigmas = tuple(float(value) for value in schedule["sigma_grid_decimal"])
    deltas = tuple(
        float(value) for value in schedule["delta_sigma_by_step_decimal"]
    )
    transformer = _FakeProbeTransformer(small_shape)
    scheduler = scheduler_type(sigmas)
    adapter = ScopedPatchRelationWanProbeAdapter(
        transformer,
        scheduler,
        descriptor=build_public_patch_relation_descriptor(),
        probe_id="patch_relation_probe",
        identity_id="patch_relation_test_identity",
        signed_coefficient=coefficient,
        sigma_grid=sigmas,
        delta_sigma_by_step=deltas,
        timestep_by_step=tuple(
            float(value)
            for value in schedule["timestep_by_step_decimal"]
        ),
    )
    sample = np.zeros(small_shape, dtype="<f4")
    conditional_encoder = np.asarray([2.0], dtype="<f4")
    unconditional_encoder = np.asarray([1.0], dtype="<f4")
    with adapter:
        for step_index, sigma in enumerate(sigmas[:-1]):
            timestep_value = (
                sigma * 1000.0
                if timestep_values is None
                else timestep_values[step_index]
            )
            timestep = np.asarray([timestep_value], dtype="<f4")
            conditional = transformer.forward(
                hidden_states=sample,
                timestep=timestep,
                encoder_hidden_states=conditional_encoder,
                return_dict=False,
            )[0]
            unconditional = transformer.forward(
                hidden_states=sample,
                timestep=timestep,
                encoder_hidden_states=unconditional_encoder,
                return_dict=False,
            )[0]
            controlled_cfg = np.ascontiguousarray(
                unconditional
                + np.float32(5.0)
                * np.subtract(
                    conditional,
                    unconditional,
                    dtype=np.float32,
                ),
                dtype="<f4",
            )
            sample = scheduler.step(
                controlled_cfg,
                np.asarray(sigma * 1000.0, dtype="<f4"),
                sample,
                return_dict=False,
            )[0]
    return transformer, scheduler, adapter.records()


@pytest.mark.quick
def test_governed_probe_adapter_runs_four_forwards_and_controlled_scheduler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transformer, scheduler, records = _run_fake_probe_pipeline(
        monkeypatch,
        coefficient=1,
    )
    assert transformer.original_forward_count == 32
    assert scheduler.original_step_count == 8
    assert len(records) == 8
    assert [record.step_index for record in records] == list(range(8))
    assert [record.remaining_step_count for record in records] == list(
        range(8, 0, -1)
    )
    assert all(record.actual_delta_norm > 0.0 for record in records)
    assert all(record.norm_guard_passed for record in records)
    assert all(record.energy_guard_passed for record in records)
    assert all(record.direction_guard_passed is True for record in records)
    assert all(
        record.signed_state_update_exposure > 0.0 for record in records
    )
    assert all(
        record.conditional_encoder_digest
        != record.unconditional_encoder_digest
        for record in records
    )
    assert "forward" not in transformer.__dict__
    assert "step" not in scheduler.__dict__


@pytest.mark.quick
def test_governed_probe_adapter_clean_uses_same_full_path_and_exact_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transformer, scheduler, records = _run_fake_probe_pipeline(
        monkeypatch,
        coefficient=0,
    )
    assert transformer.original_forward_count == 32
    assert scheduler.original_step_count == 8
    assert all(record.clean_exact_noop for record in records)
    assert all(record.actual_delta_norm == 0.0 for record in records)
    assert all(
        record.signed_state_update_exposure == 0.0 for record in records
    )


@pytest.mark.quick
def test_governed_probe_adapter_rejects_repeated_or_wrong_timestep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_patch_relation_gate0_config()
    schedule = config["protocol_contract"]["gate0_runtime_execution_contract"]
    timesteps = tuple(
        float(value) for value in schedule["timestep_by_step_decimal"]
    )
    repeated = list(timesteps)
    repeated[1] = repeated[0]
    with pytest.raises(RuntimeError, match="timestep"):
        _run_fake_probe_pipeline(
            monkeypatch,
            coefficient=1,
            timestep_values=tuple(repeated),
        )


@pytest.mark.quick
def test_governed_probe_adapter_rejects_internal_step_index_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(RuntimeError, match="internal step index after"):
        _run_fake_probe_pipeline(
            monkeypatch,
            coefficient=1,
            scheduler_type=_StepDriftScheduler,
        )


@pytest.mark.quick
def test_governed_probe_adapter_rejects_transformer_cache() -> None:
    config = load_patch_relation_gate0_config()
    schedule = config["protocol_contract"]["gate0_runtime_execution_contract"]
    transformer = _FakeProbeTransformer((1, 1, 1, 1, 2))
    transformer.is_cache_enabled = True
    scheduler = _FakeProbeScheduler(
        tuple(float(value) for value in schedule["sigma_grid_decimal"])
    )
    adapter = ScopedPatchRelationWanProbeAdapter(
        transformer,
        scheduler,
        descriptor=build_public_patch_relation_descriptor(),
        probe_id="patch_relation_probe",
        identity_id="patch_relation_test_identity",
        signed_coefficient=1,
        sigma_grid=tuple(
            float(value) for value in schedule["sigma_grid_decimal"]
        ),
        delta_sigma_by_step=tuple(
            float(value)
            for value in schedule["delta_sigma_by_step_decimal"]
        ),
        timestep_by_step=tuple(
            float(value)
            for value in schedule["timestep_by_step_decimal"]
        ),
    )
    with pytest.raises(RuntimeError, match="cache disabled"):
        adapter.__enter__()


@pytest.mark.quick
def test_probe_scope_rolls_back_transformer_when_scheduler_patch_fails() -> None:
    config = load_patch_relation_gate0_config()
    schedule = config["protocol_contract"]["gate0_runtime_execution_contract"]
    sigmas = tuple(float(value) for value in schedule["sigma_grid_decimal"])
    transformer = _FakeProbeTransformer((1, 1, 1, 1, 2))
    scheduler = _StepInstallFailingScheduler(sigmas)
    adapter = _probe_adapter(transformer, scheduler)
    with pytest.raises(RuntimeError, match="step install failure"):
        adapter.__enter__()
    assert "forward" not in transformer.__dict__
    assert "step" not in scheduler.__dict__
    assert not hasattr(
        transformer,
        adapter._ACTIVE_TRANSFORMER_ATTRIBUTE,
    )
    assert not hasattr(
        scheduler,
        adapter._ACTIVE_SCHEDULER_ATTRIBUTE,
    )
    with pytest.raises(RuntimeError, match="完整退出"):
        adapter.records()


@pytest.mark.quick
def test_probe_scope_rejects_records_after_downstream_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    small_shape = (1, 1, 1, 1, 2)
    monkeypatch.setattr(
        runtime_module,
        "TRANSFORMER_HIDDEN_SHAPE",
        small_shape,
    )
    monkeypatch.setattr(
        runtime_module,
        "SCHEDULER_VELOCITY_SHAPE",
        small_shape,
    )
    monkeypatch.setattr(
        runner_module,
        "SCHEDULER_VELOCITY_SHAPE",
        small_shape,
    )
    config = load_patch_relation_gate0_config()
    schedule = config["protocol_contract"]["gate0_runtime_execution_contract"]
    sigmas = tuple(float(value) for value in schedule["sigma_grid_decimal"])
    transformer = _FakeProbeTransformer(small_shape)
    scheduler = _FakeProbeScheduler(sigmas)
    adapter = _probe_adapter(transformer, scheduler)
    sample = np.zeros(small_shape, dtype="<f4")
    with pytest.raises(RuntimeError, match="downstream probe failure"):
        with adapter:
            transformer.forward(
                hidden_states=sample,
                timestep=np.asarray([1000.0], dtype="<f4"),
                encoder_hidden_states=np.asarray([2.0], dtype="<f4"),
                return_dict=False,
            )
            raise RuntimeError("synthetic downstream probe failure")
    assert "forward" not in transformer.__dict__
    assert "step" not in scheduler.__dict__
    assert not hasattr(
        transformer,
        adapter._ACTIVE_TRANSFORMER_ATTRIBUTE,
    )
    assert not hasattr(
        scheduler,
        adapter._ACTIVE_SCHEDULER_ATTRIBUTE,
    )
    with pytest.raises(RuntimeError, match="完整退出"):
        adapter.records()


@pytest.mark.quick
def test_probe_scope_runs_all_cleanup_after_one_cleanup_failure() -> None:
    config = load_patch_relation_gate0_config()
    schedule = config["protocol_contract"]["gate0_runtime_execution_contract"]
    sigmas = tuple(float(value) for value in schedule["sigma_grid_decimal"])
    transformer = _FakeProbeTransformer((1, 1, 1, 1, 2))
    scheduler = _ScopeCleanupFailingScheduler(sigmas)
    adapter = _probe_adapter(transformer, scheduler)
    with pytest.raises(RuntimeError, match="probe scope cleanup failure"):
        with adapter:
            pass
    assert "forward" not in transformer.__dict__
    assert "step" not in scheduler.__dict__
    assert not hasattr(
        transformer,
        adapter._ACTIVE_TRANSFORMER_ATTRIBUTE,
    )
    assert hasattr(
        scheduler,
        adapter._ACTIVE_SCHEDULER_ATTRIBUTE,
    )
    with pytest.raises(RuntimeError, match="完整退出"):
        adapter.records()
    scheduler.fail_scope_cleanup = False
    delattr(scheduler, adapter._ACTIVE_SCHEDULER_ATTRIBUTE)


def test_scoped_numpy_adapter_changes_only_active_pair_and_restores() -> None:
    transformer = _FakeTransformer()
    rope = transformer.rope
    assert "forward" not in rope.__dict__
    original_cosine = rope.cosine.copy()
    original_sine = rope.sine.copy()
    result, record = _run_branch(
        transformer,
        control_role="controlled",
        branch_role="conditional",
        coefficient=1,
    )
    assert "forward" not in rope.__dict__
    assert not hasattr(rope, "_sstw_patch_relation_rope_scope_active")
    assert np.array_equal(rope.cosine, original_cosine)
    assert np.array_equal(rope.sine, original_sine)
    assert result[0] is not rope.cosine
    assert result[1] is not rope.sine
    phase_delta = build_relation_phase_delta(
        build_public_patch_relation_descriptor(),
        signed_coefficient=1,
    )
    active_tokens = np.flatnonzero(phase_delta[0, :, 0, 0])
    assert active_tokens.size == 6
    expected_changed = np.zeros(ROPE_TUPLE_SHAPE, dtype=bool)
    expected_changed[0, active_tokens, 0, 0] = True
    expected_changed[0, active_tokens, 0, 1] = True
    cosine_changed = result[0] != original_cosine
    sine_changed = result[1] != original_sine
    assert np.array_equal(cosine_changed, expected_changed)
    assert np.array_equal(sine_changed, expected_changed)
    assert np.count_nonzero(cosine_changed) == 12
    assert np.count_nonzero(sine_changed) == 12
    assert (
        np.count_nonzero(cosine_changed) + np.count_nonzero(sine_changed)
        == 24
    )
    assert np.array_equal(
        result[0][0, active_tokens, 0, 0],
        np.cos(phase_delta[0, active_tokens, 0, 0]).astype("<f4"),
    )
    assert np.array_equal(
        result[1][0, active_tokens, 0, 1],
        np.sin(phase_delta[0, active_tokens, 0, 1]).astype("<f4"),
    )
    assert record.cfg_branch_role == "conditional"
    assert record.cfg_branch_order_index == 0
    assert record.rope_call_attempt_count == 1
    assert record.successful_rope_call_count == 1
    assert record.scope_completed_successfully
    assert not record.clean_exact_noop
    assert record.local_contract_only
    assert not record.execution_evidence_allowed


def test_clean_scope_returns_original_tuple_and_exact_noop() -> None:
    transformer = _FakeTransformer()
    result, record = _run_branch(
        transformer,
        control_role="base",
        branch_role="unconditional",
        coefficient=0,
    )
    assert result[0] is transformer.rope.cosine
    assert result[1] is transformer.rope.sine
    assert result[0].dtype == np.dtype("<f4")
    assert result[1].dtype == np.dtype("<f4")
    assert record.clean_exact_noop
    assert record.cfg_branch_order_index == 1


def test_numpy_runtime_rejects_nonofficial_float64_tuple_with_dtype_details() -> None:
    descriptor = build_public_patch_relation_descriptor()
    cosine = np.ones(ROPE_TUPLE_SHAPE, dtype="<f8")
    sine = np.zeros(ROPE_TUPLE_SHAPE, dtype="<f8")
    with pytest.raises(
        ValueError,
        match=r"expected=float32, observed=float64",
    ):
        apply_wan_rotary_phase_runtime(
            cosine,
            sine,
            descriptor=descriptor,
            signed_coefficient=0,
        )


def test_positive_negative_runtime_phase_is_exactly_opposite() -> None:
    descriptor = build_public_patch_relation_descriptor()
    phase_delta = build_relation_phase_delta(
        descriptor,
        signed_coefficient=1,
    )
    active_tokens = np.flatnonzero(phase_delta[0, :, 0, 0])
    cosine = np.ones(ROPE_TUPLE_SHAPE, dtype="<f4")
    sine = np.zeros(ROPE_TUPLE_SHAPE, dtype="<f4")
    positive = apply_wan_rotary_phase_runtime(
        cosine,
        sine,
        descriptor=descriptor,
        signed_coefficient=1,
    )
    negative = apply_wan_rotary_phase_runtime(
        cosine,
        sine,
        descriptor=descriptor,
        signed_coefficient=-1,
    )
    assert positive[0].dtype == np.dtype("<f4")
    assert positive[1].dtype == np.dtype("<f4")
    assert negative[0].dtype == np.dtype("<f4")
    assert negative[1].dtype == np.dtype("<f4")
    assert np.array_equal(positive[0], negative[0])
    assert np.array_equal(positive[1], -negative[1])
    assert np.array_equal(
        positive[0][0, active_tokens, 0, 0],
        negative[0][0, active_tokens, 0, 0],
    )
    assert np.array_equal(
        positive[1][0, active_tokens, 0, 1],
        -negative[1][0, active_tokens, 0, 1],
    )


def test_scope_restores_after_exception_without_masking() -> None:
    rope = _FailingRope()
    transformer = _FakeTransformer(rope)
    scope = ScopedWanRopeOutputAdapter(
        transformer,
        descriptor=build_public_patch_relation_descriptor(),
        signed_coefficient=1,
        probe_id="patch_relation_probe",
        step_index=0,
        control_role="controlled",
        cfg_branch_role="conditional",
        input_binding_digest=_BINDING_DIGEST,
    )
    with pytest.raises(RuntimeError, match="synthetic rope failure"):
        with scope:
            transformer.rope(_hidden_states())
    assert "forward" not in rope.__dict__
    assert not hasattr(rope, "_sstw_patch_relation_rope_scope_active")
    with pytest.raises(RuntimeError, match="clean exit"):
        scope.record()


def test_scope_rejects_record_after_downstream_transformer_failure() -> None:
    transformer = _FakeTransformer()
    rope = transformer.rope
    scope = ScopedWanRopeOutputAdapter(
        transformer,
        descriptor=build_public_patch_relation_descriptor(),
        signed_coefficient=1,
        probe_id="patch_relation_probe",
        step_index=0,
        control_role="controlled",
        cfg_branch_role="conditional",
        input_binding_digest=_BINDING_DIGEST,
    )
    with pytest.raises(RuntimeError, match="downstream transformer failure"):
        with scope:
            transformer.rope(_hidden_states())
            raise RuntimeError("downstream transformer failure")
    assert "forward" not in rope.__dict__
    assert not hasattr(rope, "_sstw_patch_relation_rope_scope_active")
    with pytest.raises(RuntimeError, match="clean exit"):
        scope.record()


def test_scope_rejects_record_after_cleanup_failure() -> None:
    rope = _CleanupFailingRope()
    transformer = _FakeTransformer(rope)
    scope = ScopedWanRopeOutputAdapter(
        transformer,
        descriptor=build_public_patch_relation_descriptor(),
        signed_coefficient=1,
        probe_id="patch_relation_probe",
        step_index=0,
        control_role="controlled",
        cfg_branch_role="conditional",
        input_binding_digest=_BINDING_DIGEST,
    )
    with pytest.raises(RuntimeError, match="synthetic scope cleanup failure"):
        with scope:
            transformer.rope(_hidden_states())
    assert "forward" not in rope.__dict__
    with pytest.raises(RuntimeError, match="clean exit"):
        scope.record()
    rope.fail_scope_cleanup = False
    delattr(rope, "_sstw_patch_relation_rope_scope_active")


def test_scope_repeated_exit_permanently_rejects_record() -> None:
    transformer = _FakeTransformer()
    scope = ScopedWanRopeOutputAdapter(
        transformer,
        descriptor=build_public_patch_relation_descriptor(),
        signed_coefficient=1,
        probe_id="patch_relation_probe",
        step_index=0,
        control_role="controlled",
        cfg_branch_role="conditional",
        input_binding_digest=_BINDING_DIGEST,
    )
    with scope:
        transformer.rope(_hidden_states())
    assert scope.record().scope_completed_successfully
    with pytest.raises(RuntimeError, match="不得重复退出"):
        scope.__exit__(None, None, None)
    with pytest.raises(RuntimeError, match="clean exit"):
        scope.record()


def test_scope_rejects_nested_or_multiple_forward_calls() -> None:
    transformer = _FakeTransformer()
    descriptor = build_public_patch_relation_descriptor()
    outer = ScopedWanRopeOutputAdapter(
        transformer,
        descriptor=descriptor,
        signed_coefficient=1,
        probe_id="patch_relation_probe",
        step_index=0,
        control_role="controlled",
        cfg_branch_role="conditional",
        input_binding_digest=_BINDING_DIGEST,
    )
    with pytest.raises(RuntimeError, match="精确一次"):
        with outer:
            with pytest.raises(RuntimeError, match="嵌套"):
                with ScopedWanRopeOutputAdapter(
                    transformer,
                    descriptor=descriptor,
                    signed_coefficient=1,
                    probe_id="patch_relation_probe",
                    step_index=0,
                    control_role="controlled",
                    cfg_branch_role="conditional",
                    input_binding_digest=_BINDING_DIGEST,
                ):
                    pass
            transformer.rope(_hidden_states())
            with pytest.raises(RuntimeError, match="超额"):
                transformer.rope(_hidden_states())
    with pytest.raises(RuntimeError, match="clean exit"):
        outer.record()


def test_cfg_pair_requires_same_control_on_both_official_branches() -> None:
    pair = _run_pair(control_role="controlled", coefficient=1)
    assert pair.conditional.cfg_branch_role == "conditional"
    assert pair.unconditional.cfg_branch_role == "unconditional"
    with pytest.raises(ValueError, match="不一致"):
        validate_cfg_rope_application_pair(
            pair.conditional,
            replace(pair.unconditional, signed_coefficient=-1),
        )
    with pytest.raises(ValueError, match="role/order"):
        validate_cfg_rope_application_pair(
            replace(
                pair.conditional,
                cfg_branch_role="unconditional",
                cfg_branch_order_index=1,
            ),
            pair.unconditional,
        )
    with pytest.raises(ValueError, match="boundary"):
        validate_cfg_rope_application_pair(
            replace(pair.conditional, rope_call_attempt_count=2),
            pair.unconditional,
        )
    with pytest.raises(ValueError, match="boundary"):
        validate_cfg_rope_application_pair(
            replace(pair.conditional, scope_completed_successfully=False),
            pair.unconditional,
        )
    with pytest.raises(ValueError, match="identity/layout"):
        validate_cfg_rope_application_pair(
            replace(pair.conditional, descriptor_digest="2" * 64),
            replace(pair.unconditional, descriptor_digest="2" * 64),
        )
    with pytest.raises(ValueError, match="identity/layout"):
        validate_cfg_rope_application_pair(
            replace(pair.conditional, clean_exact_noop=True),
            replace(pair.unconditional, clean_exact_noop=True),
        )


def test_cfg_state_update_recomputes_actual_fp32_delta_and_exposure() -> None:
    base_pair = _run_pair(control_role="base", coefficient=0)
    controlled_pair = _run_pair(control_role="controlled", coefficient=1)
    base_conditional = _velocity(1.0)
    base_unconditional = _velocity(0.5)
    controlled_conditional = base_conditional.copy()
    controlled_unconditional = base_unconditional.copy()
    controlled_conditional.reshape(-1)[0] += np.float32(1e-4)
    controlled_unconditional.reshape(-1)[0] += np.float32(1e-4)
    result = measure_cfg_state_update_numpy(
        base_pair=base_pair,
        controlled_pair=controlled_pair,
        base_conditional_velocity=base_conditional,
        base_unconditional_velocity=base_unconditional,
        controlled_conditional_velocity=controlled_conditional,
        controlled_unconditional_velocity=controlled_unconditional,
        **_scheduler_state_arguments(
            base_conditional,
            base_unconditional,
            controlled_conditional,
            controlled_unconditional,
            delta_sigma=-0.1,
        ),
        delta_sigma=-0.1,
        cumulative_reference_energy_before_step=0.0,
        cumulative_control_energy_before_step=0.0,
        remaining_step_count=8,
    )
    assert result.actual_delta_norm > 0.0
    assert result.state_update_delta_norm > 0.0
    assert result.direction_cosine == pytest.approx(1.0)
    assert result.signed_state_update_exposure == pytest.approx(
        result.state_update_delta_norm
    )
    assert result.norm_guard_passed
    assert result.energy_guard_passed
    assert result.direction_guard_passed
    assert not result.clean_exact_noop
    assert result.local_contract_only
    assert not result.execution_evidence_allowed
    assert (
        validate_cfg_state_update_measurement_numpy(
            result,
            base_pair=base_pair,
            controlled_pair=controlled_pair,
        )
        is result
    )
    factory_parameters = inspect.signature(
        runner_module._governed_step_from_measurement
    ).parameters
    assert not {
        "scheduler_consumed_velocity_digest",
        "scheduler_sample_digest",
        "scheduler_base_next_state_digest",
        "scheduler_controlled_next_state_digest",
        "actual_state_update_digest",
    }.intersection(factory_parameters)
    governed = runner_module._governed_step_from_measurement(
        result,
        base_pair=base_pair,
        controlled_pair=controlled_pair,
        conditional_encoder_digest="2" * 64,
        unconditional_encoder_digest="3" * 64,
    )
    assert governed.scheduler_consumed_velocity_digest == sha256(
        result.controlled_cfg_velocity.tobytes(order="C")
    ).hexdigest()
    assert governed.scheduler_sample_digest == sha256(
        result.scheduler_sample.tobytes(order="C")
    ).hexdigest()
    assert governed.scheduler_base_next_state_digest == sha256(
        result.base_next_state.tobytes(order="C")
    ).hexdigest()
    assert governed.scheduler_controlled_next_state_digest == sha256(
        result.controlled_next_state.tobytes(order="C")
    ).hexdigest()
    assert governed.actual_state_update_digest == sha256(
        result.actual_state_update_delta.tobytes(order="C")
    ).hexdigest()
    forged_measurement = replace(
        result,
        base_velocity_norm=1000.0,
        intended_delta_norm=0.5,
        actual_delta_norm=0.5,
        base_state_update_norm=0.052457,
        intended_state_update_norm=5.2457e-8,
        state_update_delta_norm=5.2457e-8,
        state_update_direction_dot=(5.2457e-8) ** 2,
        direction_actual_norm=5.2457e-8,
        direction_intended_norm=5.2457e-8,
        norm_budget=2.4,
        energy_increment=(5.2457e-8) ** 2,
        direction_cosine=1.0,
        signed_state_update_exposure=5.2457e-8,
        norm_guard_passed=True,
        energy_guard_passed=True,
        direction_guard_passed=True,
    )
    with pytest.raises(ValueError, match="transition数组重建"):
        validate_cfg_state_update_measurement_numpy(
            forged_measurement,
            base_pair=base_pair,
            controlled_pair=controlled_pair,
        )
    with pytest.raises(ValueError, match="transition数组重建"):
        runner_module._governed_step_from_measurement(
            forged_measurement,
            base_pair=base_pair,
            controlled_pair=controlled_pair,
            conditional_encoder_digest="2" * 64,
            unconditional_encoder_digest="3" * 64,
        )

    negative_pair = _run_pair(control_role="controlled", coefficient=-1)
    negative_conditional = base_conditional.copy()
    negative_unconditional = base_unconditional.copy()
    negative_conditional.reshape(-1)[0] -= np.float32(1e-4)
    negative_unconditional.reshape(-1)[0] -= np.float32(1e-4)
    negative_result = measure_cfg_state_update_numpy(
        base_pair=base_pair,
        controlled_pair=negative_pair,
        base_conditional_velocity=base_conditional,
        base_unconditional_velocity=base_unconditional,
        controlled_conditional_velocity=negative_conditional,
        controlled_unconditional_velocity=negative_unconditional,
        **_scheduler_state_arguments(
            base_conditional,
            base_unconditional,
            negative_conditional,
            negative_unconditional,
            delta_sigma=-0.1,
        ),
        delta_sigma=-0.1,
        cumulative_reference_energy_before_step=0.0,
        cumulative_control_energy_before_step=0.0,
        remaining_step_count=8,
    )
    assert negative_result.signed_state_update_exposure < 0.0
    assert negative_result.signed_state_update_exposure == pytest.approx(
        -negative_result.state_update_delta_norm
    )


def test_bfloat16_branch_rounding_requires_pre_cfg_float32_cast() -> None:
    conditional = _round_float32_to_bfloat16_values([0.1])
    unconditional = _round_float32_to_bfloat16_values([0.13177258])
    legacy_bfloat16_cfg = _round_float32_to_bfloat16_values(
        unconditional
        + _round_float32_to_bfloat16_values(
            np.float32(5.0)
            * _round_float32_to_bfloat16_values(
                conditional - unconditional
            )
        )
    )
    frozen_float32_cfg = np.ascontiguousarray(
        unconditional
        + np.float32(5.0) * (conditional - unconditional),
        dtype="<f4",
    )
    assert legacy_bfloat16_cfg.item() == pytest.approx(-0.0263671875)
    assert frozen_float32_cfg.item() == pytest.approx(-0.02685546875)
    assert not np.array_equal(legacy_bfloat16_cfg, frozen_float32_cfg)


def test_scheduler_postcast_quantization_cannot_masquerade_as_actual_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    small_shape = (1, 1, 1, 1, 2)
    monkeypatch.setattr(
        runtime_module,
        "SCHEDULER_VELOCITY_SHAPE",
        small_shape,
    )
    base_pair = _run_pair(control_role="base", coefficient=0)
    controlled_pair = _run_pair(control_role="controlled", coefficient=1)
    base_conditional = np.full(small_shape, 1.0, dtype="<f4")
    base_unconditional = np.full(small_shape, 0.5, dtype="<f4")
    controlled_conditional = base_conditional.copy()
    controlled_unconditional = base_unconditional.copy()
    controlled_conditional.reshape(-1)[0] += np.float32(1e-4)
    controlled_unconditional.reshape(-1)[0] += np.float32(1e-4)
    state = _scheduler_state_arguments(
        base_conditional,
        base_unconditional,
        controlled_conditional,
        controlled_unconditional,
        delta_sigma=-0.1,
    )
    state["scheduler_controlled_next_state"] = np.ascontiguousarray(
        _round_float32_to_bfloat16_values(
            state["scheduler_controlled_next_state"]
        ),
        dtype="<f4",
    )
    with pytest.raises(ValueError, match="next-state"):
        measure_cfg_state_update_numpy(
            base_pair=base_pair,
            controlled_pair=controlled_pair,
            base_conditional_velocity=base_conditional,
            base_unconditional_velocity=base_unconditional,
            controlled_conditional_velocity=controlled_conditional,
            controlled_unconditional_velocity=controlled_unconditional,
            **state,
            delta_sigma=-0.1,
            cumulative_reference_energy_before_step=0.0,
            cumulative_control_energy_before_step=0.0,
            remaining_step_count=8,
        )


def test_clean_cfg_state_update_is_exact_zero_on_same_numeric_path() -> None:
    base_pair = _run_pair(control_role="base", coefficient=0)
    controlled_pair = _run_pair(control_role="controlled", coefficient=0)
    conditional = _velocity(1.0)
    unconditional = _velocity(0.5)
    result = measure_cfg_state_update_numpy(
        base_pair=base_pair,
        controlled_pair=controlled_pair,
        base_conditional_velocity=conditional,
        base_unconditional_velocity=unconditional,
        controlled_conditional_velocity=conditional.copy(),
        controlled_unconditional_velocity=unconditional.copy(),
        **_scheduler_state_arguments(
            conditional,
            unconditional,
            conditional,
            unconditional,
            delta_sigma=-0.1,
        ),
        delta_sigma=-0.1,
        cumulative_reference_energy_before_step=0.0,
        cumulative_control_energy_before_step=0.0,
        remaining_step_count=8,
    )
    assert result.clean_exact_noop
    assert result.actual_delta_norm == 0.0
    assert result.signed_state_update_exposure == 0.0
    assert np.count_nonzero(result.actual_delta_velocity) == 0
    assert np.count_nonzero(result.actual_state_update_delta) == 0


def test_velocity_boundary_rejects_forgery_and_unbound_inputs() -> None:
    base_pair = _run_pair(control_role="base", coefficient=0)
    controlled_pair = _run_pair(control_role="controlled", coefficient=1)
    conditional = _velocity(1.0)
    unconditional = _velocity(0.5)
    mismatched_controlled_pair = replace(
        controlled_pair,
        input_binding_digest="2" * 64,
        conditional=replace(
            controlled_pair.conditional,
            input_binding_digest="2" * 64,
        ),
        unconditional=replace(
            controlled_pair.unconditional,
            input_binding_digest="2" * 64,
        ),
    )
    with pytest.raises(ValueError, match="binding"):
        measure_cfg_state_update_numpy(
            base_pair=base_pair,
            controlled_pair=mismatched_controlled_pair,
            base_conditional_velocity=conditional,
            base_unconditional_velocity=unconditional,
            controlled_conditional_velocity=conditional,
            controlled_unconditional_velocity=unconditional,
            **_scheduler_state_arguments(
                conditional,
                unconditional,
                conditional,
                unconditional,
                delta_sigma=-0.1,
            ),
            delta_sigma=-0.1,
            cumulative_reference_energy_before_step=0.0,
            cumulative_control_energy_before_step=0.0,
            remaining_step_count=8,
        )
    with pytest.raises(ValueError, match="dtype"):
        measure_cfg_state_update_numpy(
            base_pair=base_pair,
            controlled_pair=controlled_pair,
            base_conditional_velocity=conditional.astype(np.float64),
            base_unconditional_velocity=unconditional,
            controlled_conditional_velocity=conditional,
            controlled_unconditional_velocity=unconditional,
            **_scheduler_state_arguments(
                conditional,
                unconditional,
                conditional,
                unconditional,
                delta_sigma=-0.1,
            ),
            delta_sigma=-0.1,
            cumulative_reference_energy_before_step=0.0,
            cumulative_control_energy_before_step=0.0,
            remaining_step_count=8,
        )
    nonfinite = conditional.copy()
    nonfinite.reshape(-1)[0] = np.nan
    with pytest.raises(ValueError, match="有限"):
        measure_cfg_state_update_numpy(
            base_pair=base_pair,
            controlled_pair=controlled_pair,
            base_conditional_velocity=nonfinite,
            base_unconditional_velocity=unconditional,
            controlled_conditional_velocity=conditional,
            controlled_unconditional_velocity=unconditional,
            **_scheduler_state_arguments(
                conditional,
                unconditional,
                conditional,
                unconditional,
                delta_sigma=-0.1,
            ),
            delta_sigma=-0.1,
            cumulative_reference_energy_before_step=0.0,
            cumulative_control_energy_before_step=0.0,
            remaining_step_count=8,
        )
    with pytest.raises(ValueError, match="delta_sigma"):
        measure_cfg_state_update_numpy(
            base_pair=base_pair,
            controlled_pair=controlled_pair,
            base_conditional_velocity=conditional,
            base_unconditional_velocity=unconditional,
            controlled_conditional_velocity=conditional,
            controlled_unconditional_velocity=unconditional,
            **_scheduler_state_arguments(
                conditional,
                unconditional,
                conditional,
                unconditional,
                delta_sigma=0.1,
            ),
            delta_sigma=0.1,
            cumulative_reference_energy_before_step=0.0,
            cumulative_control_energy_before_step=0.0,
            remaining_step_count=8,
        )


def test_active_zero_delta_and_budget_excess_fail_closed() -> None:
    base_pair = _run_pair(control_role="base", coefficient=0)
    controlled_pair = _run_pair(control_role="controlled", coefficient=-1)
    conditional = _velocity(1.0)
    unconditional = _velocity(0.5)
    with pytest.raises(PatchRelationRuntimeGuardError, match="actual_delta_norm"):
        measure_cfg_state_update_numpy(
            base_pair=base_pair,
            controlled_pair=controlled_pair,
            base_conditional_velocity=conditional,
            base_unconditional_velocity=unconditional,
            controlled_conditional_velocity=conditional.copy(),
            controlled_unconditional_velocity=unconditional.copy(),
            **_scheduler_state_arguments(
                conditional,
                unconditional,
                conditional,
                unconditional,
                delta_sigma=-0.1,
            ),
            delta_sigma=-0.1,
            cumulative_reference_energy_before_step=0.0,
            cumulative_control_energy_before_step=0.0,
            remaining_step_count=8,
        )
    controlled_conditional = conditional.copy()
    controlled_unconditional = unconditional.copy()
    controlled_conditional.reshape(-1)[0] += np.float32(0.1)
    controlled_unconditional.reshape(-1)[0] += np.float32(0.1)
    with pytest.raises(PatchRelationRuntimeGuardError, match="energy_guard"):
        measure_cfg_state_update_numpy(
            base_pair=base_pair,
            controlled_pair=controlled_pair,
            base_conditional_velocity=conditional,
            base_unconditional_velocity=unconditional,
            controlled_conditional_velocity=controlled_conditional,
            controlled_unconditional_velocity=controlled_unconditional,
            **_scheduler_state_arguments(
                conditional,
                unconditional,
                controlled_conditional,
                controlled_unconditional,
                delta_sigma=-0.1,
            ),
            delta_sigma=-0.1,
            cumulative_reference_energy_before_step=0.0,
            cumulative_control_energy_before_step=1e9,
            remaining_step_count=8,
        )
