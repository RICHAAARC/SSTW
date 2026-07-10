"""将 Wan2.1 官方 Transformer、VAE 与 Flow scheduler 接入真实 replay。"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from main.methods.state_space_watermark.endpoint_latent_detector import (
    EndpointLatentEvidence,
    compute_endpoint_latent_evidence,
    encode_video_to_wan_endpoint_latent,
)
from main.methods.state_space_watermark.flow_tubelet_key_code import (
    FlowTubeletKeyCodeConfig,
    build_flow_tubelet_key_direction_like,
)
from main.methods.state_space_watermark.path_observation import (
    aggregate_path_observations,
    compute_path_step_observation,
)
from main.methods.state_space_watermark.replay_inversion import (
    FlowSchedulePoint,
    ReplayTrajectory,
    ReplayUncertainty,
    estimate_replay_uncertainty,
    evaluate_candidate_on_fixed_inversion,
    run_key_independent_inversion_hypothesis,
)
from main.methods.state_space_watermark.velocity_field_constraint import (
    VelocityFieldConstraintConfig,
    apply_velocity_field_constraint,
)


@dataclass(frozen=True)
class WanFlowReplayResult:
    """保存攻击后视频的 endpoint、路径与 replay 不确定性证据。"""

    endpoint_evidence: EndpointLatentEvidence
    path_evidence: dict[str, float | int | None]
    replay_uncertainty: ReplayUncertainty
    replay_trajectories: tuple[ReplayTrajectory, ...]
    endpoint_metadata: dict[str, Any]
    replay_step_counts: tuple[int, ...]
    endpoint_latent: Any
    primary_schedule: tuple[FlowSchedulePoint, ...]
    primary_replay_index: int


def build_flow_schedule_points(scheduler: Any, *, num_inference_steps: int, device: Any) -> list[FlowSchedulePoint]:
    """从官方 Flow scheduler 提取 timestep 与 sigma 网格。"""

    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = list(scheduler.timesteps)
    sigmas = list(scheduler.sigmas)
    if len(sigmas) != len(timesteps) + 1:
        raise RuntimeError("Flow scheduler 的 sigmas 必须比 timesteps 多一个 endpoint")
    points = [
        FlowSchedulePoint(timestep=timestep, sigma=float(sigmas[index]))
        for index, timestep in enumerate(timesteps)
    ]
    endpoint_timestep = timesteps[-1].new_zeros(()) if hasattr(timesteps[-1], "new_zeros") else 0.0
    points.append(FlowSchedulePoint(timestep=endpoint_timestep, sigma=float(sigmas[-1])))
    return points


class WanPromptConditionedVelocity:
    """使用 Wan 官方 prompt encoder 与 Transformer 计算真实条件 velocity。"""

    def __init__(
        self,
        pipeline: Any,
        *,
        prompt: str,
        negative_prompt: str | None = None,
        guidance_scale: float = 5.0,
    ) -> None:
        self.pipeline = pipeline
        self.guidance_scale = float(guidance_scale)
        self.device = pipeline._execution_device
        self.model = pipeline.transformer
        if self.model is None:
            raise RuntimeError("Wan replay 当前要求 pipeline.transformer 可用")
        self.model_dtype = self.model.dtype
        self.prompt_embeds, self.negative_prompt_embeds = pipeline.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.guidance_scale > 1.0,
            num_videos_per_prompt=1,
            device=self.device,
            dtype=self.model_dtype,
        )
        self.prompt_embeds = self.prompt_embeds.to(self.model_dtype)
        if self.negative_prompt_embeds is not None:
            self.negative_prompt_embeds = self.negative_prompt_embeds.to(self.model_dtype)

    def __call__(self, latent: Any, timestep: Any, step_index: int) -> Any:
        """返回与官方 WanPipeline denoising loop 同口径的 CFG velocity。"""

        import torch

        del step_index
        latent_input = latent.to(device=self.device, dtype=self.model_dtype)
        if hasattr(timestep, "to"):
            timestep_value = timestep.to(device=self.device)
        else:
            timestep_value = torch.tensor(timestep, device=self.device)
        timestep_batch = timestep_value.reshape(-1)[0].expand(latent_input.shape[0])
        cond_context = self.model.cache_context("cond") if hasattr(self.model, "cache_context") else nullcontext()
        with torch.inference_mode(), cond_context:
            conditional = self.model(
                hidden_states=latent_input,
                timestep=timestep_batch,
                encoder_hidden_states=self.prompt_embeds,
                attention_kwargs=None,
                return_dict=False,
            )[0]
        if self.guidance_scale <= 1.0:
            return conditional.to(dtype=latent.dtype)
        uncond_context = self.model.cache_context("uncond") if hasattr(self.model, "cache_context") else nullcontext()
        with torch.inference_mode(), uncond_context:
            unconditional = self.model(
                hidden_states=latent_input,
                timestep=timestep_batch,
                encoder_hidden_states=self.negative_prompt_embeds,
                attention_kwargs=None,
                return_dict=False,
            )[0]
        guided = unconditional + self.guidance_scale * (conditional - unconditional)
        return guided.to(dtype=latent.dtype)


class WanKeyConditionedVelocity:
    """在真实 Wan velocity 上复现生成阶段的同源 key 弱约束。"""

    def __init__(
        self,
        base_velocity: WanPromptConditionedVelocity,
        *,
        key_text: str,
        total_steps: int,
        tubelet_config: FlowTubeletKeyCodeConfig | None = None,
        velocity_config: VelocityFieldConstraintConfig | None = None,
    ) -> None:
        self.base_velocity = base_velocity
        self.key_text = key_text
        self.total_steps = int(total_steps)
        self.tubelet_config = tubelet_config or FlowTubeletKeyCodeConfig()
        self.velocity_config = velocity_config or VelocityFieldConstraintConfig()
        self._direction: Any | None = None

    def __call__(self, latent: Any, timestep: Any, step_index: int) -> Any:
        """返回 base model velocity 与生成阶段弱约束的合成结果。"""

        base = self.base_velocity(latent, timestep, step_index)
        if self._direction is None or tuple(self._direction.shape) != tuple(latent.shape):
            self._direction, _metadata = build_flow_tubelet_key_direction_like(
                latent,
                key_text=self.key_text,
                config=self.tubelet_config,
            )
        phase = int(step_index) / max(1, self.total_steps - 2)
        constrained, _record = apply_velocity_field_constraint(
            base,
            latent,
            self._direction.to(device=latent.device, dtype=latent.dtype),
            flow_phase=phase,
            config=self.velocity_config,
            tubelet_config=self.tubelet_config,
            endpoint_control_enabled=True,
        )
        return constrained


def _path_evidence_from_replay(
    trajectory: ReplayTrajectory,
    *,
    key_text: str,
    tubelet_config: FlowTubeletKeyCodeConfig,
    schedule: Sequence[FlowSchedulePoint],
) -> dict[str, float | int | None]:
    """在 key 无关固定反演路径上计算候选 key 投影证据。"""

    states = trajectory.reverse_states
    if len(states) != len(schedule):
        raise RuntimeError("replay states 与 Flow schedule 长度不一致")
    direction, _metadata = build_flow_tubelet_key_direction_like(
        states[0],
        key_text=key_text,
        config=tubelet_config,
    )
    records: list[dict[str, Any]] = []
    for step_index in range(len(states) - 1):
        delta_sigma = float(schedule[step_index + 1].sigma) - float(schedule[step_index].sigma)
        if abs(delta_sigma) <= 1e-12:
            continue
        phase = step_index / max(1, len(states) - 2)
        velocity = (states[step_index + 1] - states[step_index]) / delta_sigma
        records.append(compute_path_step_observation(
            states[step_index],
            states[step_index + 1],
            velocity,
            direction,
            flow_phase=phase,
        ).as_dict())
    return aggregate_path_observations(records)


def score_replay_trajectory_for_key(
    trajectory: ReplayTrajectory,
    schedule: Sequence[FlowSchedulePoint],
    *,
    key_text: str,
    tubelet_config: FlowTubeletKeyCodeConfig | None = None,
) -> dict[str, float | int | None]:
    """在不重复模型推理的情况下为另一把 key 重算 replay 路径证据。

    固定反演路径只由 attacked-video endpoint 和基础 Wan velocity 决定。候选 key
    仅用于读取该路径的投影, 因而 clean negative 的多 key 校准不会重新构造观测。
    """

    tubelet_config = tubelet_config or FlowTubeletKeyCodeConfig()
    states = trajectory.reverse_states
    if len(states) != len(schedule):
        raise RuntimeError("replay states 与 Flow schedule 长度不一致")
    direction, _metadata = build_flow_tubelet_key_direction_like(
        states[0],
        key_text=key_text,
        config=tubelet_config,
    )
    records: list[dict[str, Any]] = []
    for step_index in range(len(states) - 1):
        delta_sigma = float(schedule[step_index + 1].sigma) - float(schedule[step_index].sigma)
        if abs(delta_sigma) <= 1e-12:
            continue
        velocity = (states[step_index + 1] - states[step_index]) / delta_sigma
        phase = step_index / max(1, len(states) - 2)
        records.append(compute_path_step_observation(
            states[step_index],
            states[step_index + 1],
            velocity,
            direction,
            flow_phase=phase,
        ).as_dict())
    return aggregate_path_observations(records)


def evaluate_fixed_wan_replay_hypothesis_for_key(
    pipeline: Any,
    replay: WanFlowReplayResult,
    *,
    prompt: str,
    key_text: str,
    negative_prompt: str | None = None,
    guidance_scale: float = 5.0,
    tubelet_config: FlowTubeletKeyCodeConfig | None = None,
) -> tuple[ReplayTrajectory, dict[str, float | int | None]]:
    """在固定 attacked-video 反演观测上评估另一把候选 key。"""

    tubelet_config = tubelet_config or FlowTubeletKeyCodeConfig()
    base_velocity = WanPromptConditionedVelocity(
        pipeline,
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
    )
    keyed_velocity = WanKeyConditionedVelocity(
        base_velocity,
        key_text=key_text,
        total_steps=len(replay.primary_schedule),
        tubelet_config=tubelet_config,
    )
    fixed = replay.replay_trajectories[replay.primary_replay_index]
    hypothesis = evaluate_candidate_on_fixed_inversion(
        replay.endpoint_latent,
        replay.primary_schedule,
        fixed,
        keyed_velocity,
    )
    return hypothesis, score_replay_trajectory_for_key(
        hypothesis,
        replay.primary_schedule,
        key_text=key_text,
        tubelet_config=tubelet_config,
    )


def run_wan_control_replay(
    pipeline: Any,
    endpoint_latent: Any,
    *,
    prompt: str,
    key_text: str,
    num_inference_steps: int,
    scheduler: Any | None = None,
    negative_prompt: str | None = None,
    guidance_scale: float = 5.0,
    tubelet_config: FlowTubeletKeyCodeConfig | None = None,
) -> tuple[ReplayTrajectory, tuple[FlowSchedulePoint, ...], dict[str, float | int | None]]:
    """使用显式 prompt 或 scheduler 执行一个可审计的 replay control。"""

    tubelet_config = tubelet_config or FlowTubeletKeyCodeConfig()
    velocity = WanPromptConditionedVelocity(
        pipeline,
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
    )
    active_scheduler = scheduler or pipeline.scheduler
    schedule = build_flow_schedule_points(
        active_scheduler,
        num_inference_steps=int(num_inference_steps),
        device=velocity.device,
    )
    keyed_velocity = WanKeyConditionedVelocity(
        velocity,
        key_text=key_text,
        total_steps=len(schedule),
        tubelet_config=tubelet_config,
    )
    trajectory = run_key_independent_inversion_hypothesis(
        endpoint_latent,
        schedule,
        velocity,
        keyed_velocity,
    )
    path_evidence = _path_evidence_from_replay(
        trajectory,
        key_text=key_text,
        tubelet_config=tubelet_config,
        schedule=schedule,
    )
    return trajectory, tuple(schedule), path_evidence


def run_wan_attacked_video_replay(
    pipeline: Any,
    video_path: str | Path,
    *,
    prompt: str,
    key_text: str,
    negative_prompt: str | None = None,
    guidance_scale: float = 5.0,
    replay_step_counts: Sequence[int] = (16, 20, 24),
    tubelet_config: FlowTubeletKeyCodeConfig | None = None,
) -> WanFlowReplayResult:
    """从攻击后视频执行多时间网格 Wan inversion/replay 并返回正式证据。"""

    tubelet_config = tubelet_config or FlowTubeletKeyCodeConfig()
    endpoint_latent, endpoint_metadata = encode_video_to_wan_endpoint_latent(pipeline.vae, video_path)
    endpoint_evidence = compute_endpoint_latent_evidence(
        endpoint_latent,
        key_text=key_text,
        tubelet_config=tubelet_config,
    )
    base_velocity = WanPromptConditionedVelocity(
        pipeline,
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
    )
    replay_rows: list[ReplayTrajectory] = []
    schedules: list[list[FlowSchedulePoint]] = []
    for step_count in replay_step_counts:
        if int(step_count) < 2:
            raise ValueError("replay step count 必须至少为2")
        schedule = build_flow_schedule_points(
            pipeline.scheduler,
            num_inference_steps=int(step_count),
            device=base_velocity.device,
        )
        schedules.append(schedule)
        keyed_velocity = WanKeyConditionedVelocity(
            base_velocity,
            key_text=key_text,
            total_steps=len(schedule),
            tubelet_config=tubelet_config,
        )
        replay_rows.append(run_key_independent_inversion_hypothesis(
            endpoint_latent,
            schedule,
            base_velocity,
            keyed_velocity,
        ))
    uncertainty = estimate_replay_uncertainty(replay_rows)
    # 主网格固定为预注册列表的中间项, 禁止按结果挑选最有利网格。
    primary_index = len(replay_rows) // 2
    path_evidence = _path_evidence_from_replay(
        replay_rows[primary_index],
        key_text=key_text,
        tubelet_config=tubelet_config,
        schedule=schedules[primary_index],
    )
    return WanFlowReplayResult(
        endpoint_evidence=endpoint_evidence,
        path_evidence=path_evidence,
        replay_uncertainty=uncertainty,
        replay_trajectories=tuple(replay_rows),
        endpoint_metadata=endpoint_metadata,
        replay_step_counts=tuple(int(value) for value in replay_step_counts),
        endpoint_latent=endpoint_latent,
        primary_schedule=tuple(schedules[primary_index]),
        primary_replay_index=primary_index,
    )
