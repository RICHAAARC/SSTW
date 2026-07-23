from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

from evaluation.protocol.record_writer import write_json, write_jsonl
from experiments.generative_video_model_probe.minimal_signed_trajectory_state_space_smoke import (
    CLEAN_VARIANT,
    NONNEGATIVE_VARIANT,
    PROFILE_ID,
    SIGNED_VARIANT,
    build_minimal_signed_trajectory_generation_plan,
    build_signed_trajectory_decision,
    build_signed_trajectory_pair_records,
    validate_controlled_embedding_source_result,
    validate_minimal_signed_trajectory_config,
)
from experiments.generative_video_model_probe.colab_runtime import (
    WAN21_PRIMARY_MODEL_ID,
)
from main.methods.state_space_watermark.signed_trajectory_carrier import (
    SignedTrajectoryCarrierConfig,
    apply_signed_trajectory_two_channel_constraint,
    build_signed_trajectory_schedule,
    select_dc_scale_for_ac_direction_retention,
    select_signed_trajectory_joint_scale,
)
from main.methods.state_space_watermark.flow_tubelet_key_code import (
    FlowTubeletKeyCodeConfig,
    flow_phase_weight,
)
from main.methods.state_space_watermark.velocity_field_constraint import (
    VelocityControlContext,
    VelocityFieldConstraintConfig,
    apply_velocity_field_constraint,
)
from workflows.colab_test_request import (
    MINIMAL_SIGNED_TRAJECTORY_STATE_SPACE_SMOKE_TEST_ID,
    load_colab_test_request,
    run_colab_test_request,
)

pytestmark = pytest.mark.quick


CONFIG_PATH = Path(
    "configs/protocol/sstw_minimal_signed_trajectory_state_space_smoke.json"
)


class _NumpyTensor:
    """覆盖 velocity constraint 所需运算的轻量 CPU tensor 测试替身。"""

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

    def __rtruediv__(self, other):
        return _NumpyTensor(self._values(other) / self.values)

    def __matmul__(self, other):
        return _NumpyTensor(self.values @ self._values(other))

    @staticmethod
    def _values(value):
        return value.values if isinstance(value, _NumpyTensor) else value


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _fake_validated_source() -> dict:
    prompts = [
        {
            "prompt_id": f"prompt_{index}",
            "prompt_text": f"prompt text {index}",
            "prompt_category": "motion",
            "prompt_suite_role": "probe_paper",
            "motion_pattern_id": f"motion_{index}",
            "motion_claim_role": "positive_motion",
            "motion_calibration_role": None,
            "prompt_negative_text": "",
        }
        for index in (1, 2)
    ]
    seeds = [
        {
            "seed_id": f"seed_{index}",
            "seed_suite_role": "probe_paper",
            "seed_value": 1000 + index,
        }
        for index in (1, 2)
    ]
    generation_rows = [
        {
            "prompt_id": prompt["prompt_id"],
            "seed_id": seed["seed_id"],
            "generation_seed_random": seed["seed_value"],
            "prompt_suite_role": "probe_paper",
            "seed_suite_role": "probe_paper",
        }
        for prompt in prompts
        for seed in seeds
    ]
    return {
        "prompt_suite": {"prompts": prompts, "seeds": seeds},
        "reference_generation_rows": generation_rows,
    }


def _write_fake_controlled_result(root: Path) -> None:
    config = _config()
    write_json(
        root
        / "artifacts"
        / "controlled_embedding_strength_diagnostic_decision.json",
        {
            "profile_id": (
                "sstw_controlled_embedding_strength_diagnostic"
            ),
            "controlled_embedding_strength_diagnostic_decision": (
                "lambda_increase_did_not_repair_path_signal_stop"
            ),
            "generation_record_count": 16,
            "generation_success_count": 16,
            "summary_record_count": 96,
            "pair_record_count": 84,
            "failure_record_count": 0,
            "lambda_increase_path_signal_repair_observed": False,
            "attacked_phase_executed": False,
            "attacked_phase_execution_allowed": False,
            "fixed_fpr_evaluation_executed": False,
            "external_baseline_execution_executed": False,
            "stage_progression_allowed": False,
            "formal_result": False,
            "strength_grid_diagnostics": {
                "reference_default:20": {
                    "coverage_ready": True,
                    "correct_over_wrong_endpoint_fraction": 0.75,
                    "correct_over_wrong_path_fraction": 0.25,
                    "correct_over_wrong_trajectory_fraction": 0.25,
                    "lambda_max": 0.12,
                    "replay_grid_step_count": 20,
                }
            },
        },
    )
    write_json(
        root
        / "artifacts"
        / "controlled_embedding_strength_diagnostic_manifest.json",
        {
            "profile_id": config["required_source_profile_id"],
            "formal_result": False,
            "stage_progression_allowed": False,
        },
    )
    rows = []
    for prompt_index in (1, 2):
        for seed_index in (1, 2):
            rows.append(
                {
                    "generation_status": "success",
                    "generation_model_id": WAN21_PRIMARY_MODEL_ID,
                    "embedding_strength_level_id": "reference_default",
                    "method_variant": "sstw_full_method",
                    "prompt_id": f"prompt_{prompt_index}",
                    "seed_id": f"seed_{seed_index}",
                    "generation_seed_random": 1000 + seed_index,
                    "prompt_suite_role": "probe_paper",
                    "seed_suite_role": "probe_paper",
                }
            )
    while len(rows) < 16:
        rows.append(
            {
                **rows[len(rows) % 4],
                "embedding_strength_level_id": "moderate_increase",
            }
        )
    write_jsonl(root / "records" / "generation_records.jsonl", rows)
    write_jsonl(
        root
        / "records"
        / "controlled_embedding_strength_summary_records.jsonl",
        [{"index": index} for index in range(96)],
    )
    write_jsonl(
        root
        / "records"
        / "controlled_embedding_strength_pair_records.jsonl",
        [{"index": index} for index in range(84)],
    )
    write_jsonl(
        root
        / "records"
        / "controlled_embedding_strength_failure_records.jsonl",
        [],
    )
    write_json(
        root / "datasets" / "prompt_seed_suite.json",
        _fake_validated_source()["prompt_suite"],
    )
    write_jsonl(
        root
        / "records"
        / "trajectory_replay_smoke_likelihood_calibrations.jsonl",
        [
            {
                "replay_relative_observation_noise_standard_deviation": 0.1,
                "replay_minimum_observation_noise_variance": 1e-6,
                "replay_likelihood_model_id": "test",
                "replay_likelihood_calibration_protocol": "test",
                "replay_likelihood_calibration_cluster_count": 4,
            }
        ],
    )


def _summary(
    prompt_id: str,
    seed_id: str,
    variant: str,
    role: str,
    *,
    path: float,
    endpoint: float,
    reliability: float = 0.2,
) -> dict:
    return {
        "prompt_id": prompt_id,
        "seed_id": seed_id,
        "trajectory_carrier_variant_id": variant,
        "candidate_key_role": role,
        "trajectory_path_projection": path,
        "trajectory_static_aggregation_score": path,
        "endpoint_score": endpoint,
        "trajectory_global_reliability": reliability,
    }


def test_signed_schedule_is_deterministic_key_bound_and_weighted_zero_mean():
    kwargs = {
        "key_text": "owner-key-a",
        "key_context_digest": "a" * 64,
        "flow_phases": [0.1, 0.3, 0.5, 0.7, 0.9],
        "active_weights": [0.0, 0.2, 0.5, 0.3, 0.0],
    }
    first = build_signed_trajectory_schedule(**kwargs)
    second = build_signed_trajectory_schedule(**kwargs)
    other = build_signed_trajectory_schedule(
        **{**kwargs, "key_text": "owner-key-b"}
    )
    assert first == second
    assert first.schedule_digest != other.schedule_digest
    assert abs(
        sum(
            weight * code
            for weight, code in zip(
                first.active_weights,
                first.codes,
                strict=True,
            )
        )
    ) <= 1e-10
    assert any(value > 0.0 for value in first.codes)
    assert any(value < 0.0 for value in first.codes)


def test_generation_and_replay_grids_share_the_same_phase_bin_signs():
    common = {
        "key_text": "owner-key-a",
        "key_context_digest": "b" * 64,
    }
    schedules = []
    for step_count in (8, 20, 40):
        phases = [
            (step_index + 0.5) / step_count
            for step_index in range(step_count)
        ]
        weights = [
            1.0 if 0.25 <= phase <= 0.75 else 0.0
            for phase in phases
        ]
        schedules.append(
            build_signed_trajectory_schedule(
                **common,
                flow_phases=phases,
                active_weights=weights,
            )
        )
    raw_sign_by_bin: dict[int, int] = {}
    for schedule in schedules:
        assert any(code > 0.0 for code in schedule.codes)
        assert any(code < 0.0 for code in schedule.codes)
        for phase_bin, raw_sign, weight in zip(
            schedule.phase_bins,
            schedule.raw_signs,
            schedule.active_weights,
            strict=True,
        ):
            if weight <= 0.0:
                continue
            if phase_bin in raw_sign_by_bin:
                assert raw_sign == raw_sign_by_bin[phase_bin]
            raw_sign_by_bin[phase_bin] = raw_sign
    assert len({schedule.schedule_digest for schedule in schedules}) == 3
    assert len({schedule.phase_function_digest for schedule in schedules}) == 1


def test_wrong_key_changes_continuous_phase_bin_sign_function():
    phases = [(index + 0.5) / 40 for index in range(40)]
    weights = [
        1.0 if 0.25 <= phase <= 0.75 else 0.0
        for phase in phases
    ]
    owner = build_signed_trajectory_schedule(
        key_text="owner-key-a",
        key_context_digest="b" * 64,
        flow_phases=phases,
        active_weights=weights,
    )
    wrong = build_signed_trajectory_schedule(
        key_text="wrong-key-z",
        key_context_digest="b" * 64,
        flow_phases=phases,
        active_weights=weights,
    )
    assert owner.phase_function_digest != wrong.phase_function_digest
    assert owner.raw_signs != wrong.raw_signs


def test_signed_carrier_freezes_original_lambda_budget_partition():
    carrier = SignedTrajectoryCarrierConfig()
    assert carrier.ac_allocation == 0.75
    assert carrier.dc_allocation == 0.25
    assert carrier.minimum_ac_direction_retained_cosine == 0.25
    with pytest.raises(ValueError):
        SignedTrajectoryCarrierConfig(ac_allocation=0.8, dc_allocation=0.3)
    with pytest.raises(ValueError):
        SignedTrajectoryCarrierConfig(
            minimum_ac_direction_retained_cosine=0.0
        )


def test_joint_scale_selects_the_stricter_norm_or_energy_budget_without_torch():
    norm_limited = select_signed_trajectory_joint_scale(
        observed_delta_norm=4.0,
        joint_norm_budget=1.0,
        energy_limited_delta_norm=2.0,
    )
    assert norm_limited == {
        "norm_scale": 0.25,
        "energy_scale": 0.5,
        "joint_scale": 0.25,
    }
    energy_limited = select_signed_trajectory_joint_scale(
        observed_delta_norm=4.0,
        joint_norm_budget=3.0,
        energy_limited_delta_norm=0.5,
    )
    assert energy_limited["joint_scale"] == 0.125
    assert 4.0 * energy_limited["joint_scale"] <= 3.0
    assert 4.0 * energy_limited["joint_scale"] <= 0.5


def test_dc_scale_caps_an_opposing_channel_to_preserve_ac_direction():
    selection = select_dc_scale_for_ac_direction_retention(
        ac_delta_norm=0.125,
        dc_delta_norm=0.25,
        ac_dc_dot=-0.03125,
        minimum_retained_cosine=0.25,
    )
    assert 0.0 <= selection["dc_scale"] < 0.5
    assert selection["candidate_joint_ac_cosine"] < 0.0
    assert selection["selected_joint_ac_cosine"] >= 0.25 - 1e-10


def test_small_negative_ac_code_is_not_reversed_by_dc_without_torch():
    phases = [(index + 0.5) / 8 for index in range(8)]
    tubelet = FlowTubeletKeyCodeConfig()
    weights = [
        flow_phase_weight(phase, tubelet) / 8.0
        for phase in phases
    ]
    schedule = build_signed_trajectory_schedule(
        key_text="key5",
        key_context_digest="a" * 64,
        flow_phases=phases,
        active_weights=weights,
    )
    active_negative = [
        (phase, code)
        for phase, code, weight in zip(
            phases,
            schedule.codes,
            weights,
            strict=True,
        )
        if weight > 0.0 and -0.2 < code < 0.0
    ]
    assert active_negative
    flow_phase, ac_code = active_negative[0]

    generator = np.random.default_rng(seed=0)
    model_output = _NumpyTensor(generator.normal(size=(1, 2, 2, 2, 2)))
    sample = _NumpyTensor(generator.normal(size=(1, 2, 2, 2, 2)))
    direction = _NumpyTensor(generator.normal(size=(1, 2, 2, 2, 2)))
    direction = direction / direction.norm()
    context = VelocityControlContext(
        delta_sigma=-0.2,
        cumulative_control_energy=0.0,
        cumulative_reference_energy=10.0,
        remaining_step_count=4,
    )
    velocity = VelocityFieldConstraintConfig()
    ac_only, _record = apply_velocity_field_constraint(
        model_output,
        sample,
        direction * -1.0,
        flow_phase=flow_phase,
        config=replace(
            velocity,
            lambda_max=(
                velocity.lambda_max * 0.75 * abs(ac_code)
            ),
        ),
        tubelet_config=tubelet,
        endpoint_control_enabled=False,
        control_context=context,
    )
    constrained, record = apply_signed_trajectory_two_channel_constraint(
        model_output,
        sample,
        direction,
        ac_code=ac_code,
        flow_phase=flow_phase,
        config=velocity,
        tubelet_config=tubelet,
        carrier_config=SignedTrajectoryCarrierConfig(),
        control_context=context,
    )
    ac_delta = ac_only - model_output
    joint_delta = constrained - model_output
    ac_joint_dot = float(
        (
            ac_delta.reshape(-1)
            @ joint_delta.reshape(-1)
        ).item()
    )
    ac_joint_cosine = ac_joint_dot / (
        float(ac_delta.norm().item())
        * float(joint_delta.norm().item())
    )
    replay_signed_direction = direction * -1.0
    state_path_delta = joint_delta * context.delta_sigma
    replay_path_projection = float(
        (
            state_path_delta.reshape(-1)
            @ replay_signed_direction.reshape(-1)
        ).item()
    )
    assert ac_joint_dot > 0.0
    assert ac_joint_cosine >= 0.25 - 1e-10
    assert replay_path_projection > 0.0
    assert record["signed_trajectory_dc_direction_guard_scale"] < 1.0
    assert record["signed_trajectory_ac_direction_guard_passed"] is True
    assert (
        record["signed_trajectory_final_joint_ac_direction_cosine"]
        >= 0.25 - 1e-10
    )


@pytest.mark.parametrize(
    ("step_count", "expected_inactive_count"),
    [(8, 4), (20, 10)],
)
def test_complete_signed_schedule_handles_inactive_and_active_steps_without_torch(
    step_count: int,
    expected_inactive_count: int,
):
    phases = [
        (step_index + 0.5) / step_count
        for step_index in range(step_count)
    ]
    tubelet = FlowTubeletKeyCodeConfig()
    weights = [
        flow_phase_weight(phase, tubelet) / step_count
        for phase in phases
    ]
    schedule = build_signed_trajectory_schedule(
        key_text="owner-key-a",
        key_context_digest="c" * 64,
        flow_phases=phases,
        active_weights=weights,
    )
    generator = np.random.default_rng(seed=17 + step_count)
    model_output = _NumpyTensor(
        generator.normal(size=(1, 2, 2, 2, 2))
    )
    sample = _NumpyTensor(generator.normal(size=(1, 2, 2, 2, 2)))
    direction = _NumpyTensor(
        generator.normal(size=(1, 2, 2, 2, 2))
    )
    direction = direction / direction.norm()
    cumulative_control_energy = 0.0
    cumulative_reference_energy = 0.0
    records = []
    for step_index, (phase, code) in enumerate(
        zip(phases, schedule.codes, strict=True)
    ):
        context = VelocityControlContext(
            delta_sigma=-1.0 / step_count,
            cumulative_control_energy=cumulative_control_energy,
            cumulative_reference_energy=cumulative_reference_energy,
            remaining_step_count=step_count - step_index,
        )
        constrained, record = (
            apply_signed_trajectory_two_channel_constraint(
                model_output,
                sample,
                direction,
                ac_code=code,
                flow_phase=phase,
                config=VelocityFieldConstraintConfig(),
                tubelet_config=tubelet,
                carrier_config=SignedTrajectoryCarrierConfig(),
                control_context=context,
            )
        )
        records.append(record)
        cumulative_control_energy = float(
            record["endpoint_control_cumulative_energy_after"]
        )
        cumulative_reference_energy = float(
            record["endpoint_reference_cumulative_energy_after"]
        )
        if abs(code) <= 1e-12:
            assert constrained is model_output
            assert record["velocity_field_constraint_status"] == (
                "inactive_flow_phase"
            )
            assert record["velocity_constraint_delta_norm"] == 0.0
            assert record["endpoint_control_energy_increment"] == 0.0
            assert record["signed_trajectory_inactive_phase_noop"] is True
            assert (
                record[
                    "signed_trajectory_inactive_phase_noop_context_complete"
                ]
                is True
            )
            assert (
                record["signed_trajectory_ac_direction_guard_applicable"]
                is False
            )
            assert record["signed_trajectory_ac_direction_guard_passed"] is None
        else:
            assert record["signed_trajectory_inactive_phase_noop"] is False
            assert (
                record["signed_trajectory_ac_direction_guard_applicable"]
                is True
            )
            assert record["signed_trajectory_ac_direction_guard_passed"] is True
            assert record["signed_trajectory_joint_norm_guard_passed"] is True
            assert (
                record["signed_trajectory_joint_energy_guard_passed"]
                is True
            )
    assert sum(
        record["signed_trajectory_inactive_phase_noop"] is True
        for record in records
    ) == expected_inactive_count
    assert sum(
        record["signed_trajectory_ac_direction_guard_passed"] is True
        for record in records
    ) == step_count - expected_inactive_count


def test_two_channel_constraint_record_matches_final_delta_without_torch():
    generator = np.random.default_rng(seed=7)
    model_output = _NumpyTensor(generator.normal(size=(1, 2, 2, 2, 2)))
    sample = _NumpyTensor(generator.normal(size=(1, 2, 2, 2, 2)))
    direction = _NumpyTensor(generator.normal(size=(1, 2, 2, 2, 2)))
    direction = direction / direction.norm()
    context = VelocityControlContext(
        delta_sigma=-0.2,
        cumulative_control_energy=0.0,
        cumulative_reference_energy=10.0,
        remaining_step_count=4,
    )
    constrained, record = apply_signed_trajectory_two_channel_constraint(
        model_output,
        sample,
        direction,
        ac_code=-0.8,
        flow_phase=0.5,
        config=VelocityFieldConstraintConfig(),
        tubelet_config=FlowTubeletKeyCodeConfig(),
        carrier_config=SignedTrajectoryCarrierConfig(),
        control_context=context,
    )
    actual_delta_norm = float((constrained - model_output).norm().item())
    assert record["velocity_constraint_delta_norm"] == pytest.approx(
        actual_delta_norm,
        abs=1e-6,
    )
    assert record["velocity_constraint_lambda"] == 0.12
    assert record["signed_trajectory_joint_norm_guard_passed"] is True
    assert record["signed_trajectory_joint_energy_guard_passed"] is True
    assert record["signed_trajectory_ac_direction_guard_passed"] is True
    assert (
        actual_delta_norm
        <= record["signed_trajectory_joint_norm_budget"] + 1e-10
    )
    assert (
        record["endpoint_control_energy_increment"]
        <= record["endpoint_remaining_energy_budget_before_step"] + 1e-10
    )
    assert record["velocity_norm_before_constraint"] == pytest.approx(
        float(model_output.norm().item()),
        abs=1e-6,
    )
    assert record["velocity_norm_after_constraint"] == pytest.approx(
        float(constrained.norm().item()),
        abs=1e-6,
    )
    assert (
        record["signed_trajectory_ac_semantic_projection_status"]
        == "prompt_velocity_tangent_projection_applied"
    )
    assert (
        record[
            "signed_trajectory_dc_semantic_projection_status_before_joint_guard"
        ]
        == "prompt_velocity_tangent_projection_applied"
    )


def test_two_channel_constraint_stays_within_joint_flow_energy_budget():
    torch = pytest.importorskip("torch")
    generator = torch.Generator().manual_seed(7)
    model_output = torch.randn((1, 2, 2, 2, 2), generator=generator)
    sample = torch.randn((1, 2, 2, 2, 2), generator=generator)
    direction = torch.randn((1, 2, 2, 2, 2), generator=generator)
    direction = direction / direction.norm()
    context = VelocityControlContext(
        delta_sigma=-0.2,
        cumulative_control_energy=0.0,
        cumulative_reference_energy=10.0,
        remaining_step_count=4,
    )
    constrained, record = apply_signed_trajectory_two_channel_constraint(
        model_output,
        sample,
        direction,
        ac_code=-0.8,
        flow_phase=0.5,
        config=VelocityFieldConstraintConfig(),
        tubelet_config=FlowTubeletKeyCodeConfig(),
        carrier_config=SignedTrajectoryCarrierConfig(),
        control_context=context,
    )
    assert constrained.shape == model_output.shape
    assert record["signed_trajectory_ac_allocation"] == 0.75
    assert record["signed_trajectory_dc_allocation"] == 0.25
    assert record["velocity_constraint_lambda"] == 0.12
    assert record["signed_trajectory_joint_energy_guard_passed"] is True
    assert record["signed_trajectory_joint_norm_guard_passed"] is True
    assert (
        record["velocity_constraint_delta_norm"]
        <= record["signed_trajectory_joint_norm_budget"] + 1e-6
    )
    assert (
        record["endpoint_control_energy_increment"]
        <= record["endpoint_remaining_energy_budget_before_step"] + 1e-10
    )
    assert record["velocity_norm_before_constraint"] == pytest.approx(
        float(model_output.detach().float().norm().item()),
        abs=1e-6,
    )
    assert record["velocity_norm_after_constraint"] == pytest.approx(
        float(constrained.detach().float().norm().item()),
        abs=1e-6,
    )
    assert (
        record["signed_trajectory_ac_semantic_projection_status"]
        is not None
    )
    assert (
        record[
            "signed_trajectory_dc_semantic_projection_status_before_joint_guard"
        ]
        is not None
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("lambda_max", 0.24),
        ("replay_step_count", 40),
        ("prompt_limit", 3),
        ("phase_bin_count", 16),
        ("minimum_ac_direction_retained_cosine", 0.0),
        ("minimum_signed_correct_over_wrong_trajectory_fraction", 0.5),
        ("minimum_signed_over_nonnegative_path_margin_fraction", 0.5),
        ("minimum_replay_reliability", 0.0),
        ("endpoint_reference_default_fraction", 0.5),
        ("maximum_endpoint_fraction_drop_from_reference", 0.5),
        ("attacked_phase_execution_allowed", True),
        ("fixed_fpr_evaluation_allowed", True),
        ("stage_progression_allowed", True),
        ("claim_support_status", "formal_paper_evidence"),
    ],
)
def test_config_mutations_fail_closed(field: str, value: object):
    config = _config()
    config[field] = value
    with pytest.raises(ValueError):
        validate_minimal_signed_trajectory_config(config)


def test_generation_plan_is_exactly_four_identities_by_three_variants():
    config = _config()
    validated = _fake_validated_source()
    plan = build_minimal_signed_trajectory_generation_plan(validated, config)
    assert len(plan) == 12
    assert len(
        {
            (row["prompt_id"], row["seed_id"])
            for row in plan
        }
    ) == 4
    assert {
        row["trajectory_carrier_variant_id"] for row in plan
    } == {SIGNED_VARIANT, NONNEGATIVE_VARIANT, CLEAN_VARIANT}
    assert {row["lambda_max"] for row in plan} == {0.12}
    assert all(row["stage_progression_allowed"] is False for row in plan)
    assert all(
        row["attacked_phase_execution_allowed"] is False for row in plan
    )


def test_controlled_source_result_accepts_only_reviewed_negative_decision(
    tmp_path: Path,
):
    _write_fake_controlled_result(tmp_path)
    validated = validate_controlled_embedding_source_result(
        tmp_path,
        _config(),
    )
    assert len(validated["generation_rows"]) == 16
    assert len(validated["reference_generation_rows"]) == 4


def test_controlled_source_result_rejects_relaxed_or_positive_decision(
    tmp_path: Path,
):
    _write_fake_controlled_result(tmp_path)
    path = (
        tmp_path
        / "artifacts"
        / "controlled_embedding_strength_diagnostic_decision.json"
    )
    decision = json.loads(path.read_text(encoding="utf-8"))
    decision["controlled_embedding_strength_diagnostic_decision"] = (
        "path_signal_repaired"
    )
    write_json(path, decision)
    with pytest.raises(ValueError, match="source decision"):
        validate_controlled_embedding_source_result(tmp_path, _config())


def test_gate_pass_requires_three_of_four_for_both_path_comparisons():
    config = _config()
    summaries = []
    identities = [
        (f"prompt_{index // 2}", f"seed_{index % 2}")
        for index in range(4)
    ]
    for index, (prompt_id, seed_id) in enumerate(identities):
        for variant in (SIGNED_VARIANT, NONNEGATIVE_VARIANT, CLEAN_VARIANT):
            if variant == SIGNED_VARIANT:
                correct_path = 1.0 if index < 3 else -1.0
                correct_endpoint = 1.0 if index < 2 else -1.0
            elif variant == NONNEGATIVE_VARIANT:
                correct_path = 0.1 if index < 3 else -0.5
                correct_endpoint = 0.2
            else:
                correct_path = 0.0
                correct_endpoint = 0.0
            summaries.extend(
                [
                    _summary(
                        prompt_id,
                        seed_id,
                        variant,
                        "correct_owner_key",
                        path=correct_path,
                        endpoint=correct_endpoint,
                    ),
                    _summary(
                        prompt_id,
                        seed_id,
                        variant,
                        "wrong_owner_key",
                        path=0.0,
                        endpoint=0.0,
                    ),
                ]
            )
    pairs = build_signed_trajectory_pair_records(summaries, config)
    decision = build_signed_trajectory_decision(
        summaries,
        pairs,
        [],
        config,
    )
    assert decision["profile_id"] == PROFILE_ID
    assert decision["summary_record_count"] == 24
    assert decision["pair_record_count"] == 16
    assert decision["signed_correct_over_wrong_trajectory_fraction"] == 0.75
    assert decision["signed_over_nonnegative_path_margin_fraction"] == 0.75
    assert decision["signed_correct_over_wrong_endpoint_fraction"] == 0.5
    assert decision["signed_trajectory_carrier_gate_ready"] is True
    assert decision["stage_progression_allowed"] is False
    assert decision["formal_result"] is False


def test_gate_failure_stops_claim_without_relaxing_thresholds():
    config = _config()
    summaries = []
    for index in range(4):
        prompt_id = f"prompt_{index // 2}"
        seed_id = f"seed_{index % 2}"
        for variant in (SIGNED_VARIANT, NONNEGATIVE_VARIANT, CLEAN_VARIANT):
            summaries.extend(
                [
                    _summary(
                        prompt_id,
                        seed_id,
                        variant,
                        "correct_owner_key",
                        path=0.1 if index == 0 else -0.1,
                        endpoint=0.1,
                    ),
                    _summary(
                        prompt_id,
                        seed_id,
                        variant,
                        "wrong_owner_key",
                        path=0.0,
                        endpoint=0.0,
                    ),
                ]
            )
    pairs = build_signed_trajectory_pair_records(summaries, config)
    decision = build_signed_trajectory_decision(
        summaries,
        pairs,
        [],
        config,
    )
    assert decision["signed_trajectory_carrier_gate_ready"] is False
    assert (
        decision["minimal_signed_trajectory_smoke_decision"]
        == "signed_trajectory_carrier_gate_failed_stop_claim"
    )
    assert decision["attacked_phase_execution_allowed"] is False
    assert decision["fixed_fpr_evaluation_executed"] is False


def test_variant_order_and_claim_fields_are_frozen():
    config = _config()
    altered = deepcopy(config)
    altered["trajectory_carrier_variant_ids"] = [
        NONNEGATIVE_VARIANT,
        SIGNED_VARIANT,
        CLEAN_VARIANT,
    ]
    with pytest.raises(ValueError):
        validate_minimal_signed_trajectory_config(altered)


def test_fixed_colab_handler_dispatches_signed_smoke_without_notebook_change(
    tmp_path: Path,
):
    drive_root = tmp_path / "drive" / "SSTW"
    source_zip = (
        drive_root
        / "inputs"
        / "minimal_signed_trajectory_state_space"
        / "controlled_embedding_result.zip"
    )
    source_zip.parent.mkdir(parents=True)
    generation_row = {
        "generation_status": "success",
        "generation_model_id": (
            "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
        ),
    }
    with ZipFile(source_zip, "w") as archive:
        archive.writestr(
            "bundle/records/generation_records.jsonl",
            json.dumps(generation_row) + "\n",
        )
    request_path = drive_root / "requests" / "colab_test_request.json"
    request_path.parent.mkdir(parents=True)
    request_path.write_text(
        json.dumps(
            {
                "request_schema_version": "sstw_colab_test_request_v1",
                "test_id": (
                    MINIMAL_SIGNED_TRAJECTORY_STATE_SPACE_SMOKE_TEST_ID
                ),
                "repository": {
                    "url": "https://github.com/RICHAAARC/SSTW.git",
                    "ref": "main",
                },
                "parameters": {
                    "phase": "no_attack",
                    "run_series_id": "minimal_signed_trajectory_smoke_001",
                    "source_package_path": str(source_zip),
                    "resume_package_path": "",
                },
            }
        ),
        encoding="utf-8",
    )

    calls: list[tuple[Path, Path]] = []

    def fake_runner(source_root: Path, output_root: Path) -> dict:
        calls.append((source_root, output_root))
        write_json(
            output_root / "artifacts" / "decision.json",
            {
                "minimal_signed_trajectory_smoke_decision": (
                    "signed_trajectory_carrier_gate_failed_stop_claim"
                )
            },
        )
        return {
            "minimal_signed_trajectory_smoke_decision": (
                "signed_trajectory_carrier_gate_failed_stop_claim"
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
        minimal_signed_trajectory_runner=fake_runner,
    )
    assert len(calls) == 1
    assert result["test_id"] == (
        MINIMAL_SIGNED_TRAJECTORY_STATE_SPACE_SMOKE_TEST_ID
    )
    assert Path(result["drive_result_zip"]).is_file()
    assert Path(result["drive_result_manifest"]).is_file()


def test_signed_smoke_request_rejects_resume_package(tmp_path: Path):
    drive_root = tmp_path / "drive" / "SSTW"
    inputs = drive_root / "inputs"
    inputs.mkdir(parents=True)
    source = inputs / "source.zip"
    resume = inputs / "resume.zip"
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
                    MINIMAL_SIGNED_TRAJECTORY_STATE_SPACE_SMOKE_TEST_ID
                ),
                "repository": {
                    "url": "https://github.com/RICHAAARC/SSTW.git",
                    "ref": "main",
                },
                "parameters": {
                    "phase": "no_attack",
                    "run_series_id": "minimal_signed_trajectory_smoke_001",
                    "source_package_path": str(source),
                    "resume_package_path": str(resume),
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="不接受 resume"):
        load_colab_test_request(request_path, project_root=drive_root)
