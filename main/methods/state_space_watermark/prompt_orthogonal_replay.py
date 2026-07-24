"""Key-independent Wan replay trace and prompt-orthogonal candidate scoring."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from main.methods.state_space_watermark.continuous_trajectory_code import (
    ContinuousTrajectorySchedule,
    build_continuous_trajectory_schedule,
)
from main.methods.state_space_watermark.endpoint_latent_detector import (
    encode_video_to_wan_endpoint_latent,
)
from main.methods.state_space_watermark.flow_tubelet_key_code import (
    FlowTubeletKeyCodeConfig,
    flow_phase_weight,
)
from main.methods.state_space_watermark.flow_velocity_runtime import (
    normalized_flow_phase_from_sigma_interval,
)
from main.methods.state_space_watermark.replay_innovation import (
    ReplayInnovationStep,
    compute_flow_match_euler_replay_innovation,
)
from main.methods.state_space_watermark.replay_inversion import (
    FlowSchedulePoint,
    ReplayGaussianLikelihoodConfig,
    gaussian_replay_residual_likelihood,
    reverse_flow_trajectory,
)
from main.methods.state_space_watermark.state_rotation_operator import (
    build_prompt_orthogonal_state_direction,
    build_state_rotation_plane_like,
    derive_prompt_orthogonal_subkeys,
)
from main.methods.state_space_watermark.trajectory_vector_demodulation import (
    TrajectoryVectorDemodulation,
    demodulate_trajectory_responses,
)
from main.methods.state_space_watermark.wan_flow_replay_backend import (
    WanPromptConditionedVelocity,
    build_flow_schedule_points,
)

PROMPT_ORTHOGONAL_REPLAY_RELIABILITY_MODE = (
    "base_predicted_transition_residual_key_independent"
)


@dataclass(frozen=True)
class PromptOrthogonalReplayStepSummary:
    """Candidate-independent scalar context; no latent or key is serialized."""

    step_index: int
    flow_phase: float
    delta_sigma: float
    scheduler_weight: float
    reliability_weight: float
    innovation_norm: float
    innovation_relative_norm: float
    base_transition_norm: float
    scheduler_transition_context_complete: bool


@dataclass(frozen=True)
class PromptOrthogonalCandidateReplayScore:
    """One candidate's vector evidence on the shared replay trace."""

    candidate_role: str
    continuous_schedule: ContinuousTrajectorySchedule
    demodulation: TrajectoryVectorDemodulation
    step_responses: tuple[float, ...]
    operator_plane_digest: str
    minimum_projection_retained_ratio: float
    maximum_state_orthogonality_residual: float
    maximum_velocity_orthogonality_residual: float
    candidate_context_complete: bool


@dataclass(frozen=True)
class PromptOrthogonalReplayEvaluation:
    """Shared trace summaries plus all candidate results."""

    step_summaries: tuple[PromptOrthogonalReplayStepSummary, ...]
    candidate_scores: tuple[PromptOrthogonalCandidateReplayScore, ...]
    base_model_velocity_call_count: int
    replay_state_count: int
    candidate_count: int
    key_independent_trace_complete: bool


@dataclass(frozen=True)
class PromptOrthogonalFixedReplayTrace:
    """Prompt-conditioned, key-independent reverse trace."""

    reverse_states: tuple[Any, ...]
    schedule: tuple[FlowSchedulePoint, ...]
    endpoint_metadata: Mapping[str, Any]
    base_model_velocity_call_count: int
    key_independent_trace_complete: bool


def build_wan_prompt_orthogonal_fixed_replay_trace(
    pipeline: Any,
    video_path: str,
    *,
    prompt: str,
    num_inference_steps: int = 20,
    negative_prompt: str | None = None,
    guidance_scale: float = 5.0,
) -> PromptOrthogonalFixedReplayTrace:
    """Build reverse states without constructing any candidate key."""

    if int(num_inference_steps) != 20:
        raise ValueError("prompt-orthogonal smoke 固定使用20-step replay")
    endpoint_latent, endpoint_metadata = encode_video_to_wan_endpoint_latent(
        pipeline.vae,
        video_path,
    )
    base_velocity = WanPromptConditionedVelocity(
        pipeline,
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
    )
    schedule = tuple(
        build_flow_schedule_points(
            pipeline.scheduler,
            num_inference_steps=int(num_inference_steps),
            device=base_velocity.device,
        )
    )
    velocity_call_count = 0

    def counted_base_velocity(
        state: Any,
        timestep: Any,
        step_index: int,
    ) -> Any:
        nonlocal velocity_call_count
        velocity_call_count += 1
        return base_velocity(state, timestep, step_index)

    reverse_states = tuple(
        reverse_flow_trajectory(
            endpoint_latent,
            schedule,
            counted_base_velocity,
        )
    )
    expected_calls = int(num_inference_steps)
    return PromptOrthogonalFixedReplayTrace(
        reverse_states=reverse_states,
        schedule=schedule,
        endpoint_metadata=dict(endpoint_metadata),
        base_model_velocity_call_count=velocity_call_count,
        key_independent_trace_complete=bool(
            len(reverse_states) == len(schedule)
            and velocity_call_count == expected_calls
        ),
    )


def _schedule_context(
    schedule: Sequence[FlowSchedulePoint],
    tubelet_config: FlowTubeletKeyCodeConfig,
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    sigmas = tuple(float(point.sigma) for point in schedule)
    phases: list[float] = []
    intervals: list[float] = []
    weights: list[float] = []
    for index in range(len(schedule) - 1):
        interval = sigmas[index + 1] - sigmas[index]
        if not math.isfinite(interval) or abs(interval) <= 1e-12:
            raise ValueError("prompt-orthogonal replay 包含非法 sigma interval")
        phase = normalized_flow_phase_from_sigma_interval(sigmas, index)
        phases.append(phase)
        intervals.append(interval)
        weights.append(
            abs(interval) * flow_phase_weight(phase, tubelet_config)
        )
    return tuple(phases), tuple(intervals), tuple(weights)


def evaluate_wan_prompt_orthogonal_candidates_on_fixed_trace(
    pipeline: Any,
    reverse_states: Sequence[Any],
    schedule: Sequence[FlowSchedulePoint],
    *,
    prompt: str,
    candidate_master_keys: Mapping[str, str],
    likelihood_config: ReplayGaussianLikelihoodConfig,
    negative_prompt: str | None = None,
    guidance_scale: float = 5.0,
    tubelet_config: FlowTubeletKeyCodeConfig | None = None,
) -> PromptOrthogonalReplayEvaluation:
    """Call Wan once per step and score every candidate on the same trace."""

    if not candidate_master_keys:
        raise ValueError("prompt-orthogonal replay 至少需要一个 candidate")
    roles = tuple(str(value) for value in candidate_master_keys)
    if any(not value for value in roles) or len(set(roles)) != len(roles):
        raise ValueError("prompt-orthogonal candidate roles 必须非空且唯一")
    states = tuple(reverse_states)
    if len(states) != len(schedule) or len(states) < 3:
        raise ValueError("prompt-orthogonal replay states/schedule 覆盖不完整")
    tubelet_config = tubelet_config or FlowTubeletKeyCodeConfig()
    phases, intervals, scheduler_weights = _schedule_context(
        schedule,
        tubelet_config,
    )
    candidate_subkeys = {
        role: derive_prompt_orthogonal_subkeys(master_key)
        for role, master_key in candidate_master_keys.items()
    }
    candidate_schedules = {
        role: build_continuous_trajectory_schedule(
            trajectory_code_subkey=subkeys.trajectory_code_subkey,
            flow_phases=phases,
            active_weights=scheduler_weights,
        )
        for role, subkeys in candidate_subkeys.items()
    }
    reference_cpu = states[0].detach().float().cpu()
    candidate_planes: dict[str, tuple[Any, Any, str]] = {}
    for role, subkeys in candidate_subkeys.items():
        plane_a, plane_b, digest = build_state_rotation_plane_like(
            reference_cpu,
            state_operator_subkey=subkeys.state_operator_subkey,
        )
        candidate_planes[role] = (
            plane_a,
            plane_b,
            digest,
        )
    base_velocity = WanPromptConditionedVelocity(
        pipeline,
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
    )
    responses = {role: [] for role in roles}
    retained_ratios = {role: [] for role in roles}
    state_residuals = {role: [] for role in roles}
    velocity_residuals = {role: [] for role in roles}
    step_summaries: list[PromptOrthogonalReplayStepSummary] = []
    velocity_call_count = 0
    for index, (phase, interval, scheduler_weight) in enumerate(
        zip(phases, intervals, scheduler_weights, strict=True)
    ):
        state = states[index]
        velocity = base_velocity(
            state,
            schedule[index].timestep,
            index,
        )
        velocity_call_count += 1
        innovation: ReplayInnovationStep = (
            compute_flow_match_euler_replay_innovation(
                state,
                states[index + 1],
                velocity,
                delta_sigma=interval,
            )
        )
        base_predicted_next = (
            state.detach().float()
            + float(interval) * velocity.detach().float()
        )
        reliability_likelihood = gaussian_replay_residual_likelihood(
            base_predicted_next,
            base_predicted_next,
            states[index + 1],
            config=likelihood_config,
        )
        normalized_residual = (
            reliability_likelihood.null_residual_mean_squared_error
            / max(
                reliability_likelihood.observation_noise_variance,
                1e-12,
            )
        )
        reliability = max(
            0.0,
            min(1.0, math.exp(-0.5 * normalized_residual)),
        )
        step_summaries.append(
            PromptOrthogonalReplayStepSummary(
                step_index=index,
                flow_phase=phase,
                delta_sigma=interval,
                scheduler_weight=scheduler_weight,
                reliability_weight=reliability,
                innovation_norm=innovation.innovation_norm,
                innovation_relative_norm=(
                    innovation.innovation_relative_norm
                ),
                base_transition_norm=innovation.base_transition_norm,
                scheduler_transition_context_complete=(
                    innovation.scheduler_transition_context_complete
                ),
            )
        )
        for role in roles:
            if scheduler_weight <= 1e-12:
                responses[role].append(0.0)
                retained_ratios[role].append(1.0)
                state_residuals[role].append(0.0)
                velocity_residuals[role].append(0.0)
                continue
            plane_a_cpu, plane_b_cpu, plane_digest = candidate_planes[role]
            plane = (
                plane_a_cpu.to(
                    device=state.device,
                ),
                plane_b_cpu.to(
                    device=state.device,
                ),
                plane_digest,
            )
            direction = build_prompt_orthogonal_state_direction(
                state,
                velocity,
                state_operator_subkey=(
                    candidate_subkeys[role].state_operator_subkey
                ),
                state_rotation_plane=plane,
            )
            if not direction.active:
                raise RuntimeError(
                    f"prompt-orthogonal candidate {role} active direction 退化"
                )
            innovation_flat = innovation.innovation.detach().float().reshape(
                -1
            )
            direction_flat = direction.direction.detach().float().reshape(-1)
            response = float(
                (
                    innovation_flat
                    @ direction_flat
                    / direction_flat.norm().clamp_min(1e-12)
                ).item()
            )
            responses[role].append(response)
            retained_ratios[role].append(
                direction.projection_retained_ratio
            )
            state_residuals[role].append(
                direction.state_orthogonality_residual
            )
            velocity_residuals[role].append(
                direction.velocity_orthogonality_residual
            )
    reliability_weights = tuple(
        value.reliability_weight for value in step_summaries
    )
    candidate_results: list[PromptOrthogonalCandidateReplayScore] = []
    for role in roles:
        active_schedule = candidate_schedules[role]
        demodulation = demodulate_trajectory_responses(
            step_responses=responses[role],
            basis_values=active_schedule.centered_basis_values,
            scheduler_weights=scheduler_weights,
            reliability_weights=reliability_weights,
            candidate_codeword=active_schedule.codeword,
        )
        candidate_results.append(
            PromptOrthogonalCandidateReplayScore(
                candidate_role=role,
                continuous_schedule=active_schedule,
                demodulation=demodulation,
                step_responses=tuple(responses[role]),
                operator_plane_digest=candidate_planes[role][2],
                minimum_projection_retained_ratio=min(
                    retained_ratios[role]
                ),
                maximum_state_orthogonality_residual=max(
                    state_residuals[role]
                ),
                maximum_velocity_orthogonality_residual=max(
                    velocity_residuals[role]
                ),
                candidate_context_complete=True,
            )
        )
    return PromptOrthogonalReplayEvaluation(
        step_summaries=tuple(step_summaries),
        candidate_scores=tuple(candidate_results),
        base_model_velocity_call_count=velocity_call_count,
        replay_state_count=len(states),
        candidate_count=len(candidate_results),
        key_independent_trace_complete=bool(
            velocity_call_count == len(states) - 1
            and all(
                step.scheduler_transition_context_complete
                for step in step_summaries
            )
            and all(
                score.candidate_context_complete
                for score in candidate_results
            )
        ),
    )
