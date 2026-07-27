from __future__ import annotations

from dataclasses import replace
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
    validate_cfg_rope_application_pair,
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
    assert not config["authorization_boundary"][
        "runtime_implementation_authorized"
    ]
    assert not config["authorization_boundary"]["gpu_execution_allowed"]


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
        self.cosine = np.ones(ROPE_TUPLE_SHAPE, dtype="<f8")
        self.sine = np.zeros(ROPE_TUPLE_SHAPE, dtype="<f8")
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
    assert np.allclose(
        result[0][0, active_tokens, 0, 0],
        np.cos(phase_delta[0, active_tokens, 0, 0]),
        rtol=0.0,
        atol=1e-15,
    )
    assert np.allclose(
        result[1][0, active_tokens, 0, 1],
        np.sin(phase_delta[0, active_tokens, 0, 1]),
        rtol=0.0,
        atol=1e-15,
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
    assert record.clean_exact_noop
    assert record.cfg_branch_order_index == 1


def test_positive_negative_runtime_phase_is_exactly_opposite() -> None:
    descriptor = build_public_patch_relation_descriptor()
    phase_delta = build_relation_phase_delta(
        descriptor,
        signed_coefficient=1,
    )
    active_tokens = np.flatnonzero(phase_delta[0, :, 0, 0])
    cosine = np.ones(ROPE_TUPLE_SHAPE, dtype="<f8")
    sine = np.zeros(ROPE_TUPLE_SHAPE, dtype="<f8")
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
    assert np.array_equal(positive[0], negative[0])
    assert np.allclose(positive[1], -negative[1], rtol=0.0, atol=1e-18)
    assert np.allclose(
        positive[0][0, active_tokens, 0, 0],
        negative[0][0, active_tokens, 0, 0],
        rtol=0.0,
        atol=1e-18,
    )
    assert np.allclose(
        positive[1][0, active_tokens, 0, 1],
        -negative[1][0, active_tokens, 0, 1],
        rtol=0.0,
        atol=1e-18,
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
        delta_sigma=-0.1,
        cumulative_reference_energy_before_step=0.0,
        cumulative_control_energy_before_step=0.0,
        remaining_step_count=8,
    )
    assert result.actual_delta_norm > 0.0
    assert result.state_update_delta_norm > 0.0
    assert result.direction_cosine == pytest.approx(1.0)
    assert result.signed_state_update_exposure == pytest.approx(
        0.1 * result.actual_delta_norm
    )
    assert result.norm_guard_passed
    assert result.energy_guard_passed
    assert result.direction_guard_passed
    assert not result.clean_exact_noop
    assert result.local_contract_only
    assert not result.execution_evidence_allowed

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
        delta_sigma=-0.1,
        cumulative_reference_energy_before_step=0.0,
        cumulative_control_energy_before_step=0.0,
        remaining_step_count=8,
    )
    assert negative_result.signed_state_update_exposure < 0.0
    assert negative_result.signed_state_update_exposure == pytest.approx(
        -0.1 * negative_result.actual_delta_norm
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
            delta_sigma=-0.1,
            cumulative_reference_energy_before_step=0.0,
            cumulative_control_energy_before_step=1e9,
            remaining_step_count=8,
        )
