from __future__ import annotations

from dataclasses import replace
import inspect
import json
import math
from pathlib import Path

import numpy as np
import pytest

import main.methods.state_space_watermark.prompt_orthogonal_replay as prompt_replay_module
from main.methods.state_space_watermark.prompt_orthogonal_replay import (
    PROMPT_ORTHOGONAL_REPLAY_RELIABILITY_MODE,
)
from main.methods.state_space_watermark.continuous_trajectory_code import (
    PROMPT_ORTHOGONAL_TRAJECTORY_CODE_DIMENSION,
    PROMPT_ORTHOGONAL_TRAJECTORY_CODE_ID,
    build_continuous_trajectory_schedule,
    derive_continuous_trajectory_codeword,
    weighted_continuous_code_correlation,
)
from main.methods.state_space_watermark.flow_tubelet_key_code import (
    FlowTubeletKeyContext,
    FlowTubeletKeyCodeConfig,
    flow_phase_weight,
)
from main.methods.state_space_watermark.flow_velocity_runtime import (
    FlowVelocityConstraintRuntime,
    FlowVelocityRuntimeMechanismConfig,
)
from main.methods.state_space_watermark.replay_innovation import (
    FLOW_MATCH_EULER_TRANSITION_ADAPTER_ID,
    compute_flow_match_euler_replay_innovation,
)
from main.methods.state_space_watermark.replay_inversion import (
    FlowSchedulePoint,
    ReplayGaussianLikelihoodConfig,
)
from main.methods.state_space_watermark.state_rotation_operator import (
    PROMPT_ORTHOGONAL_CODE_SCHEMA_ID,
    PROMPT_ORTHOGONAL_LATENT_LAYOUT_ID,
    PROMPT_ORTHOGONAL_METHOD_DOMAIN,
    PROMPT_ORTHOGONAL_MINIMUM_RETAINED_RATIO,
    PROMPT_ORTHOGONAL_CANONICAL_PLANE_CONSTRUCTION_DEVICE,
    PROMPT_ORTHOGONAL_OPERATOR_SCHEMA_ID,
    PROMPT_ORTHOGONAL_PROJECTION_TOLERANCE,
    PROMPT_ORTHOGONAL_OPERATOR_RANK,
    PromptOrthogonalKeyDomainConfig,
    build_prompt_orthogonal_state_direction,
    derive_prompt_orthogonal_subkeys,
    low_rank_rotation_tangent_values,
)
from main.methods.state_space_watermark.state_trajectory_injection import (
    PROMPT_ORTHOGONAL_DIRECTION_COSINE_MINIMUM,
    PROMPT_ORTHOGONAL_FLOW_ENERGY_BUDGET_RATIO,
    PROMPT_ORTHOGONAL_LAMBDA_MAX,
    PROMPT_ORTHOGONAL_SCHEDULER_CONTROL_DTYPE,
    PROMPT_ORTHOGONAL_VELOCITY_NORM_RATIO_BUDGET,
    PromptOrthogonalInjectionConfig,
    apply_prompt_orthogonal_state_trajectory_injection,
)
from main.methods.state_space_watermark.trajectory_vector_demodulation import (
    PROMPT_ORTHOGONAL_DEMODULATION_SOLVER_ID,
    PROMPT_ORTHOGONAL_SMOKE_WHITENING_MODE,
    demodulate_trajectory_responses,
)
from main.methods.state_space_watermark.velocity_field_constraint import (
    VelocityControlContext,
)
from main.methods.state_space_watermark.watermark_key_derivation import (
    derive_prompt_orthogonal_master_key_text,
    derive_prompt_orthogonal_wrong_candidate_master_key_text,
    derive_wrong_prompt_orthogonal_master_key_text,
)


pytestmark = pytest.mark.quick
CONFIG_PATH = Path(
    "configs/protocol/"
    "sstw_prompt_orthogonal_state_trajectory_primitives.json"
)


class _NumpyTensor:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float64)

    @property
    def shape(self):
        return self.values.shape

    @property
    def dtype(self):
        return self.values.dtype

    @property
    def device(self):
        return "cpu"

    def detach(self):
        return self

    def clone(self):
        return _NumpyTensor(self.values.copy())

    def float(self):
        return self

    def cpu(self):
        return self

    def norm(self):
        return _NumpyTensor(np.linalg.norm(self.values))

    def square(self):
        return _NumpyTensor(np.square(self.values))

    def pow(self, exponent):
        return _NumpyTensor(np.power(self.values, exponent))

    def mean(self):
        return _NumpyTensor(np.mean(self.values))

    def sum(self):
        return _NumpyTensor(np.sum(self.values))

    def reshape(self, *shape):
        return _NumpyTensor(self.values.reshape(*shape))

    def clamp_min(self, minimum):
        return _NumpyTensor(np.maximum(self.values, float(minimum)))

    def to(self, *args, **kwargs):
        if kwargs.get("dtype") == "quantized_test_dtype":
            return _NumpyTensor(np.round(self.values, decimals=1))
        return self

    def item(self):
        return self.values.item()

    def __add__(self, other):
        return _NumpyTensor(self.values + self._values(other))

    def __sub__(self, other):
        return _NumpyTensor(self.values - self._values(other))

    def __truediv__(self, other):
        return _NumpyTensor(self.values / self._values(other))

    def __mul__(self, other):
        return _NumpyTensor(self.values * self._values(other))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __matmul__(self, other):
        return _NumpyTensor(self.values @ self._values(other))

    @staticmethod
    def _values(value):
        return value.values if isinstance(value, _NumpyTensor) else value


class _QuantizedNumpyTensor(_NumpyTensor):
    @property
    def dtype(self):
        return "quantized_test_dtype"


def _subkeys(master_key: str = "owner-master-key-material"):
    return derive_prompt_orthogonal_subkeys(master_key)


def test_fixed_replay_trace_is_key_free_and_has_exact_base_call_count(
    monkeypatch: pytest.MonkeyPatch,
):
    class _Pipeline:
        vae = object()
        scheduler = object()

    class _BaseVelocity:
        device = "cpu"

        def __call__(self, state, timestep, step_index):
            del timestep, step_index
            return _NumpyTensor(np.zeros_like(state.values))

    monkeypatch.setattr(
        prompt_replay_module,
        "encode_video_to_wan_endpoint_latent",
        lambda _vae, _path: (_NumpyTensor([1.0, -2.0]), {"ready": True}),
    )
    monkeypatch.setattr(
        prompt_replay_module,
        "WanPromptConditionedVelocity",
        lambda *_args, **_kwargs: _BaseVelocity(),
    )
    monkeypatch.setattr(
        prompt_replay_module,
        "build_flow_schedule_points",
        lambda _scheduler, *, num_inference_steps, device: [
            FlowSchedulePoint(
                timestep=index,
                sigma=1.0 - index / num_inference_steps,
            )
            for index in range(num_inference_steps + 1)
        ],
    )
    result = (
        prompt_replay_module.build_wan_prompt_orthogonal_fixed_replay_trace(
            _Pipeline(),
            "video.mp4",
            prompt="held-out prompt",
        )
    )
    assert len(result.schedule) == 21
    assert len(result.reverse_states) == 21
    assert result.base_model_velocity_call_count == 20
    assert result.key_independent_trace_complete is True


def test_prompt_orthogonal_primitives_protocol_freezes_implemented_core():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config == {
        "claim_support_status": (
            "core_primitives_construction_only_not_method_evidence"
        ),
        "flow_energy_budget_ratio": (
            PROMPT_ORTHOGONAL_FLOW_ENERGY_BUDGET_RATIO
        ),
        "formal_result": False,
        "lambda_max": PROMPT_ORTHOGONAL_LAMBDA_MAX,
        "paper_result_level": (
            "prompt_orthogonal_core_primitives_construction"
        ),
        "profile_id": (
            "sstw_prompt_orthogonal_state_trajectory_primitives"
        ),
        "prompt_orthogonal_code_schema_id": (
            PROMPT_ORTHOGONAL_CODE_SCHEMA_ID
        ),
        "prompt_orthogonal_demodulation_solver_id": (
            PROMPT_ORTHOGONAL_DEMODULATION_SOLVER_ID
        ),
        "prompt_orthogonal_innovation_adapter_id": (
            FLOW_MATCH_EULER_TRANSITION_ADAPTER_ID
        ),
        "prompt_orthogonal_latent_layout_id": (
            PROMPT_ORTHOGONAL_LATENT_LAYOUT_ID
        ),
        "prompt_orthogonal_method_domain": (
            PROMPT_ORTHOGONAL_METHOD_DOMAIN
        ),
        "prompt_orthogonal_minimum_direction_cosine": (
            PROMPT_ORTHOGONAL_DIRECTION_COSINE_MINIMUM
        ),
        "prompt_orthogonal_minimum_retained_ratio": (
            PROMPT_ORTHOGONAL_MINIMUM_RETAINED_RATIO
        ),
        "prompt_orthogonal_operator_rank": (
            PROMPT_ORTHOGONAL_OPERATOR_RANK
        ),
        "prompt_orthogonal_plane_construction_device": (
            PROMPT_ORTHOGONAL_CANONICAL_PLANE_CONSTRUCTION_DEVICE
        ),
        "prompt_orthogonal_operator_schema_id": (
            PROMPT_ORTHOGONAL_OPERATOR_SCHEMA_ID
        ),
        "prompt_orthogonal_projection_tolerance": (
            PROMPT_ORTHOGONAL_PROJECTION_TOLERANCE
        ),
        "prompt_orthogonal_replay_reliability_mode": (
            PROMPT_ORTHOGONAL_REPLAY_RELIABILITY_MODE
        ),
        "prompt_orthogonal_scheduler_control_dtype": (
            PROMPT_ORTHOGONAL_SCHEDULER_CONTROL_DTYPE
        ),
        "prompt_orthogonal_smoke_whitening_mode": (
            PROMPT_ORTHOGONAL_SMOKE_WHITENING_MODE
        ),
        "prompt_orthogonal_trajectory_code_dimension": (
            PROMPT_ORTHOGONAL_TRAJECTORY_CODE_DIMENSION
        ),
        "prompt_orthogonal_trajectory_code_id": (
            PROMPT_ORTHOGONAL_TRAJECTORY_CODE_ID
        ),
        "stage_progression_allowed": False,
        "velocity_norm_ratio_budget": (
            PROMPT_ORTHOGONAL_VELOCITY_NORM_RATIO_BUDGET
        ),
    }


def _schedule(master_key: str, step_count: int):
    subkeys = _subkeys(master_key)
    phases = tuple((index + 0.5) / step_count for index in range(step_count))
    tubelet = FlowTubeletKeyCodeConfig()
    weights = tuple(
        flow_phase_weight(phase, tubelet) / step_count for phase in phases
    )
    return build_continuous_trajectory_schedule(
        trajectory_code_subkey=subkeys.trajectory_code_subkey,
        flow_phases=phases,
        active_weights=weights,
    )


def test_prompt_orthogonal_kdf_api_excludes_prompt_seed_model_and_grid():
    parameter_names = set(
        inspect.signature(derive_prompt_orthogonal_subkeys).parameters
    )
    assert parameter_names == {"master_key_text", "config"}
    owner = _subkeys()
    repeated = _subkeys()
    wrong = _subkeys("wrong-owner-master-key-material")
    assert owner == repeated
    assert owner.state_operator_subkey != owner.trajectory_code_subkey
    assert owner.state_operator_subkey != wrong.state_operator_subkey
    assert owner.trajectory_code_subkey != wrong.trajectory_code_subkey


def test_owner_master_key_derivation_is_prompt_and_seed_independent_by_api():
    parameter_names = set(
        inspect.signature(
            derive_prompt_orthogonal_master_key_text
        ).parameters
    )
    assert parameter_names == {"authentication_key", "key_id"}
    secret = b"x" * 32
    owner = derive_prompt_orthogonal_master_key_text(
        secret,
        key_id="owner",
    )
    repeated = derive_prompt_orthogonal_master_key_text(
        secret,
        key_id="owner",
    )
    wrong = derive_wrong_prompt_orthogonal_master_key_text(
        secret,
        key_id="owner",
    )
    assert owner == repeated
    assert owner != wrong
    assert "x" * 16 not in owner


def test_wrong_owner_candidate_sequence_is_fixed_unique_and_validated():
    assert set(
        inspect.signature(
            derive_prompt_orthogonal_wrong_candidate_master_key_text
        ).parameters
    ) == {"authentication_key", "key_id", "candidate_index"}
    secret = b"x" * 32
    candidates = [
        derive_prompt_orthogonal_wrong_candidate_master_key_text(
            secret,
            key_id="owner",
            candidate_index=index,
        )
        for index in range(8)
    ]
    assert len(set(candidates)) == 8
    with pytest.raises(ValueError, match="至少需要32字节"):
        derive_prompt_orthogonal_wrong_candidate_master_key_text(
            b"short",
            key_id="owner",
            candidate_index=0,
        )
    with pytest.raises(ValueError, match="不能为负数"):
        derive_prompt_orthogonal_wrong_candidate_master_key_text(
            secret,
            key_id="owner",
            candidate_index=-1,
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("method_domain", "other_method"),
        ("latent_layout_id", "other_layout"),
        ("operator_schema_id", "other_operator"),
        ("code_schema_id", "other_code"),
    ],
)
def test_prompt_orthogonal_key_domain_config_is_frozen(field_name, value):
    with pytest.raises(ValueError, match="未冻结"):
        replace(PromptOrthogonalKeyDomainConfig(), **{field_name: value})


def test_low_rank_rotation_reference_is_tangent_to_state():
    inverse_sqrt_two = 1.0 / math.sqrt(2.0)
    state = (2.0, -1.0, 0.5)
    plane_a = (inverse_sqrt_two, inverse_sqrt_two, 0.0)
    plane_b = (0.0, 0.0, 1.0)
    tangent = low_rank_rotation_tangent_values(state, plane_a, plane_b)
    assert sum(
        left * right for left, right in zip(state, tangent, strict=True)
    ) == pytest.approx(0.0, abs=1e-12)
    assert tangent == pytest.approx(
        (
            0.5 * inverse_sqrt_two,
            0.5 * inverse_sqrt_two,
            -inverse_sqrt_two,
        )
    )


def test_continuous_code_uses_one_function_across_8_20_40_grids():
    schedules = [_schedule("owner-master-key-material", count) for count in (8, 20, 40)]
    assert len(
        {schedule.continuous_function_digest for schedule in schedules}
    ) == 1
    assert len(
        {schedule.schedule_projection_digest for schedule in schedules}
    ) == 3
    for schedule in schedules:
        assert abs(schedule.weighted_residual) <= 1e-12
        assert schedule.active_phase_count >= 4
        assert schedule.weighted_code_energy > 0.0
        assert any(value > 0.0 for value in schedule.codes)
        assert any(value < 0.0 for value in schedule.codes)


def test_continuous_code_is_smooth_under_phase_perturbation():
    codeword = derive_continuous_trajectory_codeword(
        _subkeys().trajectory_code_subkey
    )
    left_phase = 0.499999
    right_phase = 0.500001
    left_basis = (
        math.sin(2.0 * math.pi * left_phase),
        math.cos(2.0 * math.pi * left_phase),
    )
    right_basis = (
        math.sin(2.0 * math.pi * right_phase),
        math.cos(2.0 * math.pi * right_phase),
    )
    left = sum(a * b for a, b in zip(codeword, left_basis, strict=True))
    right = sum(a * b for a, b in zip(codeword, right_basis, strict=True))
    assert abs(left - right) < 2e-5


def test_continuous_wrong_key_changes_codeword_and_function():
    owner = _schedule("owner-master-key-material", 20)
    wrong = _schedule("wrong-owner-master-key-material", 20)
    assert owner.codeword != wrong.codeword
    assert owner.continuous_function_digest != wrong.continuous_function_digest
    correlation = weighted_continuous_code_correlation(owner, wrong)
    assert math.isfinite(correlation)
    assert -1.0 <= correlation <= 1.0


def test_flow_match_euler_innovation_removes_base_transition():
    current = _NumpyTensor([1.0, -2.0])
    velocity = _NumpyTensor([0.4, 0.2])
    injected = _NumpyTensor([0.03, -0.01])
    delta_sigma = -0.25
    following = current + (velocity + injected) * delta_sigma
    result = compute_flow_match_euler_replay_innovation(
        current,
        following,
        velocity,
        delta_sigma=delta_sigma,
    )
    assert result.adapter_id == FLOW_MATCH_EULER_TRANSITION_ADAPTER_ID
    assert result.scheduler_transition_context_complete is True
    assert result.innovation.values == pytest.approx(injected.values)
    assert result.innovation_norm == pytest.approx(
        np.linalg.norm(injected.values)
    )


def test_budgeted_injection_is_ac_only_and_respects_joint_guards():
    from main.methods.state_space_watermark.state_rotation_operator import (
        PromptOrthogonalDirection,
    )

    model_output = _NumpyTensor([3.0, 4.0])
    state = _NumpyTensor([1.0, -1.0])
    direction = PromptOrthogonalDirection(
        direction=_NumpyTensor([1.0, 0.0]),
        operator_plane_digest="a" * 64,
        operator_rank=2,
        state_tangent_norm=1.0,
        projected_tangent_norm=1.0,
        projection_retained_ratio=1.0,
        state_orthogonality_residual=0.0,
        velocity_orthogonality_residual=0.0,
        active=True,
    )
    context = VelocityControlContext(
        delta_sigma=-0.1,
        cumulative_control_energy=0.0,
        cumulative_reference_energy=0.0,
        remaining_step_count=4,
    )
    constrained, record = (
        apply_prompt_orthogonal_state_trajectory_injection(
            model_output,
            state,
            direction,
            continuous_code=-0.5,
            flow_phase=0.5,
            control_context=context,
        )
    )
    observed_delta = constrained.values - model_output.values
    assert observed_delta[0] < 0.0
    assert observed_delta[1] == pytest.approx(0.0)
    assert record.status == "prompt_orthogonal_state_trajectory_applied"
    assert record.norm_guard_passed is True
    assert record.energy_guard_passed is True
    assert record.direction_guard_passed is True
    assert record.endpoint_control_enabled is False
    assert record.delta_norm <= record.joint_norm_budget + 1e-12


def test_budgeted_injection_inactive_phase_is_exact_noop():
    from main.methods.state_space_watermark.state_rotation_operator import (
        PromptOrthogonalDirection,
    )

    model_output = _NumpyTensor([3.0, 4.0])
    direction = PromptOrthogonalDirection(
        direction=_NumpyTensor([1.0, 0.0]),
        operator_plane_digest="a" * 64,
        operator_rank=2,
        state_tangent_norm=1.0,
        projected_tangent_norm=1.0,
        projection_retained_ratio=1.0,
        state_orthogonality_residual=0.0,
        velocity_orthogonality_residual=0.0,
        active=True,
    )
    result, record = apply_prompt_orthogonal_state_trajectory_injection(
        model_output,
        _NumpyTensor([1.0, -1.0]),
        direction,
        continuous_code=0.0,
        flow_phase=0.1,
        control_context=VelocityControlContext(
            delta_sigma=-0.1,
            cumulative_control_energy=0.0,
            cumulative_reference_energy=0.0,
            remaining_step_count=4,
        ),
    )
    assert result is model_output
    assert record.inactive_phase_noop is True
    assert record.endpoint_control_enabled is False
    assert record.norm_guard_passed is None


def test_budgeted_injection_keeps_scheduler_control_in_float32():
    from main.methods.state_space_watermark.state_rotation_operator import (
        PromptOrthogonalDirection,
    )

    direction = PromptOrthogonalDirection(
        direction=_NumpyTensor([1.0, 0.0]),
        operator_plane_digest="a" * 64,
        operator_rank=2,
        state_tangent_norm=1.0,
        projected_tangent_norm=1.0,
        projection_retained_ratio=1.0,
        state_orthogonality_residual=0.0,
        velocity_orthogonality_residual=0.0,
        active=True,
    )
    constrained, record = (
        apply_prompt_orthogonal_state_trajectory_injection(
            _QuantizedNumpyTensor([3.0, 4.0]),
            _NumpyTensor([1.0, -1.0]),
            direction,
            continuous_code=0.5,
            flow_phase=0.5,
            control_context=VelocityControlContext(
                delta_sigma=-0.1,
                cumulative_control_energy=0.0,
                cumulative_reference_energy=0.0,
                remaining_step_count=4,
            ),
        )
    )
    assert constrained.values[0] > 3.0
    assert record.delta_norm > 0.0
    assert record.direction_guard_passed is True


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("carrier_id", "other"),
        ("lambda_max", 0.13),
        ("velocity_norm_ratio_budget", 0.03),
        ("flow_energy_budget_ratio", 0.00002),
        ("minimum_direction_cosine", 0.9),
    ],
)
def test_prompt_orthogonal_injection_config_is_frozen(field_name, value):
    with pytest.raises(ValueError, match="未冻结"):
        replace(PromptOrthogonalInjectionConfig(), **{field_name: value})


def test_vector_demodulation_retains_two_channels_and_prefers_owner_code():
    schedule = _schedule("owner-master-key-material", 20)
    owner = schedule.codeword
    wrong = _schedule("wrong-owner-master-key-material", 20).codeword
    responses = [
        value for value in schedule.codes
    ]
    reliability = [1.0] * len(responses)
    owner_result = demodulate_trajectory_responses(
        step_responses=responses,
        basis_values=schedule.centered_basis_values,
        scheduler_weights=schedule.active_weights,
        reliability_weights=reliability,
        candidate_codeword=owner,
    )
    wrong_result = demodulate_trajectory_responses(
        step_responses=responses,
        basis_values=schedule.centered_basis_values,
        scheduler_weights=schedule.active_weights,
        reliability_weights=reliability,
        candidate_codeword=wrong,
    )
    assert owner_result.whitening_mode == (
        PROMPT_ORTHOGONAL_SMOKE_WHITENING_MODE
    )
    assert owner_result.vector_context_complete is True
    assert len(owner_result.demodulation_vector) == 2
    normalized_vector = tuple(
        value / math.hypot(*owner_result.demodulation_vector)
        for value in owner_result.demodulation_vector
    )
    assert normalized_vector == pytest.approx(owner)
    assert owner_result.matched_cosine_score == pytest.approx(1.0)
    assert owner_result.demodulation_solver_id == (
        PROMPT_ORTHOGONAL_DEMODULATION_SOLVER_ID
    )
    assert owner_result.matched_cosine_score > (
        wrong_result.matched_cosine_score
    )


def test_vector_demodulation_rejects_candidate_specific_or_invalid_weights():
    schedule = _schedule("owner-master-key-material", 20)
    with pytest.raises(ValueError, match="非负"):
        demodulate_trajectory_responses(
            step_responses=[1.0] * len(schedule.codes),
            basis_values=schedule.basis_values,
            scheduler_weights=schedule.active_weights,
            reliability_weights=[-1.0] * len(schedule.codes),
            candidate_codeword=schedule.codeword,
        )
    with pytest.raises(ValueError, match="未冻结"):
        demodulate_trajectory_responses(
            step_responses=[1.0] * len(schedule.codes),
            basis_values=schedule.basis_values,
            scheduler_weights=schedule.active_weights,
            reliability_weights=[1.0] * len(schedule.codes),
            candidate_codeword=schedule.codeword,
            whitening_mode="fit_on_current_test",
        )


def test_torch_state_direction_is_state_and_velocity_orthogonal():
    torch = pytest.importorskip("torch")
    generator = torch.Generator().manual_seed(7)
    state = torch.randn((2, 3, 2, 3, 3), generator=generator)
    velocity = torch.randn((2, 3, 2, 3, 3), generator=generator)
    result = build_prompt_orthogonal_state_direction(
        state,
        velocity,
        state_operator_subkey=_subkeys().state_operator_subkey,
        minimum_retained_ratio=0.0,
    )
    assert result.operator_rank == PROMPT_ORTHOGONAL_OPERATOR_RANK
    assert result.active is True
    assert result.direction.shape == state.shape
    flat_direction = result.direction.float().reshape(2, -1)
    flat_state = state.float().reshape(2, -1)
    flat_velocity = velocity.float().reshape(2, -1)
    assert torch.max(torch.abs((flat_direction * flat_state).sum(dim=1))) < 1e-5
    assert (
        torch.max(torch.abs((flat_direction * flat_velocity).sum(dim=1)))
        < 1e-5
    )


def test_torch_state_plane_is_canonical_across_runtime_devices():
    torch = pytest.importorskip("torch")
    subkey = _subkeys().state_operator_subkey
    reference = torch.zeros((1, 2, 1, 2, 2), dtype=torch.float32)
    cpu_a, cpu_b, cpu_digest = (
        prompt_replay_module.build_state_rotation_plane_like(
            reference,
            state_operator_subkey=subkey,
        )
    )
    assert cpu_a.device.type == "cpu"
    assert cpu_b.device.type == "cpu"
    low_precision_a, low_precision_b, low_precision_digest = (
        prompt_replay_module.build_state_rotation_plane_like(
            reference.to(dtype=torch.bfloat16),
            state_operator_subkey=subkey,
        )
    )
    assert low_precision_digest == cpu_digest
    assert torch.equal(low_precision_a, cpu_a)
    assert torch.equal(low_precision_b, cpu_b)
    if not torch.cuda.is_available():
        return
    cuda_a, cuda_b, cuda_digest = (
        prompt_replay_module.build_state_rotation_plane_like(
            reference.cuda(),
            state_operator_subkey=subkey,
        )
    )
    assert cuda_digest == cpu_digest
    assert torch.equal(cuda_a.cpu(), cpu_a)
    assert torch.equal(cuda_b.cpu(), cpu_b)


class FlowMatchPromptOrthogonalTestScheduler:
    def __init__(self, sigmas):
        self.sigmas = sigmas
        self.index = 0
        self.received_dtypes = []

    def step(self, model_output, _timestep, sample, *args, **kwargs):
        self.received_dtypes.append(getattr(model_output, "dtype", None))
        interval = self.sigmas[self.index + 1] - self.sigmas[self.index]
        self.index += 1
        return (sample + interval * model_output,)


def _flow_key_context():
    return FlowTubeletKeyContext(
        prompt_digest="a" * 64,
        sampler_signature="flow_match_test_scheduler",
    )


def test_prompt_orthogonal_runtime_rejects_endpoint_or_partial_configuration():
    scheduler = FlowMatchPromptOrthogonalTestScheduler(
        [1.0, 0.5, 0.0]
    )
    with pytest.raises(ValueError, match="master key 与 injection config"):
        with FlowVelocityConstraintRuntime(
            scheduler,
            key_text="legacy-key-unused-by-new-carrier",
            total_steps=2,
            key_context=_flow_key_context(),
            mechanism_config=FlowVelocityRuntimeMechanismConfig(
                endpoint_control_enabled=False
            ),
            prompt_orthogonal_master_key_text=(
                "prompt-orthogonal-owner-master-key-material"
            ),
        ):
            pass
    with pytest.raises(ValueError, match="禁止 endpoint/terminal"):
        with FlowVelocityConstraintRuntime(
            scheduler,
            key_text="legacy-key-unused-by-new-carrier",
            total_steps=2,
            key_context=_flow_key_context(),
            prompt_orthogonal_master_key_text=(
                "prompt-orthogonal-owner-master-key-material"
            ),
            prompt_orthogonal_injection_config=(
                PromptOrthogonalInjectionConfig()
            ),
        ):
            pass


def test_torch_prompt_orthogonal_runtime_executes_real_scheduler_hook():
    torch = pytest.importorskip("torch")
    scheduler = FlowMatchPromptOrthogonalTestScheduler(
        torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0])
    )
    generator = torch.Generator().manual_seed(19)
    state = torch.randn(
        (1, 2, 1, 2, 2),
        generator=generator,
    ).to(torch.bfloat16)
    with FlowVelocityConstraintRuntime(
        scheduler,
        key_text="legacy-key-unused-by-new-carrier",
        total_steps=4,
        key_context=_flow_key_context(),
        mechanism_config=FlowVelocityRuntimeMechanismConfig(
            velocity_constraint_enabled=True,
            endpoint_control_enabled=False,
            terminal_endpoint_perturbation_enabled=False,
        ),
        prompt_orthogonal_master_key_text=(
            "prompt-orthogonal-owner-master-key-material"
        ),
        prompt_orthogonal_injection_config=(
            PromptOrthogonalInjectionConfig()
        ),
        prompt_orthogonal_scheduler_float32_control=True,
    ) as runtime:
        for index in range(4):
            velocity = torch.randn(
                state.shape,
                generator=generator,
            ).to(torch.bfloat16)
            state = scheduler.step(
                velocity,
                torch.tensor(float(index)),
                state,
            )[0]
    records = runtime.step_records
    assert scheduler.received_dtypes == [torch.float32] * 4
    assert state.dtype == torch.float32
    assert len(records) == 4
    assert sum(
        record["prompt_orthogonal_inactive_phase_noop"] is True
        for record in records
    ) == 2
    assert all(
        record["endpoint_control_enabled"] is False for record in records
    )
    assert all(
        record["prompt_orthogonal_scheduler_control_dtype"] == "float32"
        for record in records
    )
    assert all(
        record["flow_runtime_step_formal_context_complete"] is True
        for record in records
    )
    active = [
        record
        for record in records
        if record["prompt_orthogonal_inactive_phase_noop"] is False
    ]
    assert active
    assert all(
        record["prompt_orthogonal_norm_guard_passed"] is True
        and record["prompt_orthogonal_energy_guard_passed"] is True
        and record["prompt_orthogonal_direction_guard_passed"] is True
        for record in active
    )


def test_torch_prompt_orthogonal_clean_control_uses_same_scheduler_dtype():
    torch = pytest.importorskip("torch")
    scheduler = FlowMatchPromptOrthogonalTestScheduler(
        torch.tensor([1.0, 0.5, 0.0])
    )
    state = torch.ones((1, 2, 1, 2, 2), dtype=torch.bfloat16)
    with FlowVelocityConstraintRuntime(
        scheduler,
        key_text="legacy-key-unused-by-new-carrier",
        total_steps=2,
        key_context=_flow_key_context(),
        mechanism_config=FlowVelocityRuntimeMechanismConfig(
            velocity_constraint_enabled=False,
            endpoint_control_enabled=False,
            terminal_endpoint_perturbation_enabled=False,
        ),
        prompt_orthogonal_scheduler_float32_control=True,
    ) as runtime:
        for index in range(2):
            state = scheduler.step(
                torch.ones_like(state),
                torch.tensor(float(index)),
                state,
            )[0]
    assert scheduler.received_dtypes == [torch.float32] * 2
    assert state.dtype == torch.float32
    assert all(
        row["velocity_constraint_enabled"] is False
        and row["prompt_orthogonal_scheduler_control_dtype"] == "float32"
        for row in runtime.step_records
    )


def test_torch_replay_calls_base_model_once_per_step_for_all_candidates(
    monkeypatch,
):
    torch = pytest.importorskip("torch")
    schedule = tuple(
        FlowSchedulePoint(
            timestep=torch.tensor(float(index)),
            sigma=1.0 - index / 8.0,
        )
        for index in range(9)
    )
    phases = tuple((index + 0.5) / 8.0 for index in range(8))
    tubelet = FlowTubeletKeyCodeConfig()
    scheduler_weights = tuple(
        0.125 * flow_phase_weight(phase, tubelet) for phase in phases
    )
    owner_master = "prompt-orthogonal-owner-master-key-material"
    owner_subkeys = derive_prompt_orthogonal_subkeys(owner_master)
    owner_schedule = build_continuous_trajectory_schedule(
        trajectory_code_subkey=owner_subkeys.trajectory_code_subkey,
        flow_phases=phases,
        active_weights=scheduler_weights,
    )
    generator = torch.Generator().manual_seed(23)
    base_velocity = torch.randn(
        (1, 2, 1, 3, 3),
        generator=generator,
    )
    current = torch.randn(
        base_velocity.shape,
        generator=generator,
    )
    plane = prompt_replay_module.build_state_rotation_plane_like(
        current,
        state_operator_subkey=owner_subkeys.state_operator_subkey,
    )
    states = [current]
    for index in range(8):
        direction = build_prompt_orthogonal_state_direction(
            current,
            base_velocity,
            state_operator_subkey=owner_subkeys.state_operator_subkey,
            minimum_retained_ratio=0.0,
            state_rotation_plane=plane,
        )
        interval = schedule[index + 1].sigma - schedule[index].sigma
        current = current + interval * (
            base_velocity
            + direction.direction * owner_schedule.codes[index] * 0.1
        )
        states.append(current)
    calls = []

    class FakeWanVelocity:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, _state, _timestep, index):
            calls.append(index)
            return base_velocity

    monkeypatch.setattr(
        prompt_replay_module,
        "WanPromptConditionedVelocity",
        FakeWanVelocity,
    )
    evaluation = (
        prompt_replay_module
        .evaluate_wan_prompt_orthogonal_candidates_on_fixed_trace(
            object(),
            tuple(states),
            schedule,
            prompt="held-out prompt",
            candidate_master_keys={
                "owner": owner_master,
                "wrong": (
                    "prompt-orthogonal-wrong-owner-master-key-material"
                ),
            },
            likelihood_config=ReplayGaussianLikelihoodConfig(
                relative_observation_noise_standard_deviation=0.1,
                calibration_protocol="test",
                calibration_cluster_count=2,
            ),
        )
    )
    assert calls == list(range(8))
    assert evaluation.base_model_velocity_call_count == 8
    assert evaluation.candidate_count == 2
    assert evaluation.key_independent_trace_complete is True
    scores = {
        value.candidate_role: value.demodulation.matched_cosine_score
        for value in evaluation.candidate_scores
    }
    assert scores["owner"] > scores["wrong"]
