"""验证 P3 endpoint-aware minimum-energy approximation 的真实控制语义。"""

from __future__ import annotations

import pytest

from main.methods.state_space_watermark.velocity_field_constraint import (
    VelocityControlContext,
    VelocityFieldConstraintConfig,
    apply_velocity_field_constraint,
)


torch = pytest.importorskip("torch")


def _formal_context(*, cumulative_control_energy: float = 0.0) -> VelocityControlContext:
    return VelocityControlContext(
        delta_sigma=-0.1,
        cumulative_control_energy=cumulative_control_energy,
        cumulative_reference_energy=0.0,
        remaining_step_count=10,
    )


@pytest.mark.quick
def test_endpoint_control_uses_finite_difference_minimum_energy_and_budget() -> None:
    model_output = torch.tensor([1.0, 0.0])
    sample = torch.tensor([1.0, 0.0])
    key_direction = torch.tensor([0.0, 1.0])
    config = VelocityFieldConstraintConfig(endpoint_target_margin=0.0001)

    constrained, record = apply_velocity_field_constraint(
        model_output,
        sample,
        key_direction,
        flow_phase=0.5,
        config=config,
        control_context=_formal_context(),
    )

    assert record["endpoint_control_formal_context_complete"] is True
    assert record["endpoint_control_policy"] == (
        "finite_difference_endpoint_minimum_energy_approximation"
    )
    assert record["endpoint_minimum_energy_control_status"] == (
        "minimum_energy_control_applied"
    )
    assert record["endpoint_controllability_gain"] > 0.0
    assert record["endpoint_quality_energy_guard_passed"] is True
    assert record["endpoint_control_energy_increment"] <= (
        record["endpoint_remaining_energy_budget_before_step"] + 1e-10
    )
    assert record["velocity_constraint_delta_ratio"] <= (
        config.velocity_norm_ratio_budget * config.lambda_max + 1e-8
    )
    assert not torch.equal(constrained, model_output)


@pytest.mark.quick
def test_endpoint_control_spends_no_energy_after_target_is_reached() -> None:
    model_output = torch.tensor([1.0, 0.0])
    sample = torch.tensor([0.0, 1.0])
    key_direction = torch.tensor([0.0, 1.0])

    constrained, record = apply_velocity_field_constraint(
        model_output,
        sample,
        key_direction,
        flow_phase=0.5,
        control_context=_formal_context(),
    )

    assert record["endpoint_minimum_energy_control_status"] == (
        "endpoint_target_already_reached"
    )
    assert record["velocity_constraint_delta_norm"] == 0.0
    assert torch.equal(constrained, model_output)


@pytest.mark.quick
def test_missing_time_energy_context_cannot_support_formal_p3_claim() -> None:
    model_output = torch.tensor([1.0, 0.0])
    sample = torch.tensor([1.0, 0.0])
    key_direction = torch.tensor([0.0, 1.0])

    _constrained, record = apply_velocity_field_constraint(
        model_output,
        sample,
        key_direction,
        flow_phase=0.5,
    )

    assert record["endpoint_control_formal_context_complete"] is False
    assert record["endpoint_minimum_energy_control_status"] == (
        "compatibility_heuristic_missing_time_and_energy_context"
    )
    assert record["endpoint_quality_energy_guard_passed"] is False
