"""将 LTX-Video 官方 Transformer、VAE 与 Flow scheduler 接入真实 replay。"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from main.methods.state_space_watermark.endpoint_latent_detector import (
    EndpointLatentEvidence,
    _retrieve_vae_latent,
    compute_endpoint_latent_evidence,
    load_video_tensor_for_wan_vae,
)
from main.methods.state_space_watermark.flow_latent_layout import PackedTokenFlowLatentLayout
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
    replay_step_reliability_weight,
    run_key_independent_inversion_hypothesis,
)
from main.methods.state_space_watermark.velocity_field_constraint import (
    VelocityFieldConstraintConfig,
    apply_velocity_field_constraint,
)


@dataclass(frozen=True)
class LTXFlowReplayResult:
    """保存 LTX attacked video 的 endpoint、路径与 replay 不确定性证据。"""

    endpoint_evidence: EndpointLatentEvidence
    path_evidence: dict[str, float | int | None]
    replay_uncertainty: ReplayUncertainty
    replay_trajectories: tuple[ReplayTrajectory, ...]
    endpoint_metadata: dict[str, Any]
    replay_step_counts: tuple[int, ...]
    endpoint_latent: Any
    canonical_endpoint_latent: Any
    latent_layout: PackedTokenFlowLatentLayout
    primary_schedule: tuple[FlowSchedulePoint, ...]
    primary_replay_index: int


def build_ltx_latent_layout(
    pipeline: Any,
    *,
    num_frames: int,
    height: int,
    width: int,
) -> PackedTokenFlowLatentLayout:
    """根据 LTX VAE 压缩率和 Transformer patch 配置构造可逆布局。"""

    latent_frames = (int(num_frames) - 1) // int(pipeline.vae_temporal_compression_ratio) + 1
    latent_height = int(height) // int(pipeline.vae_spatial_compression_ratio)
    latent_width = int(width) // int(pipeline.vae_spatial_compression_ratio)
    return PackedTokenFlowLatentLayout(
        num_frames=latent_frames,
        height=latent_height,
        width=latent_width,
        spatial_patch_size=int(pipeline.transformer_spatial_patch_size),
        temporal_patch_size=int(pipeline.transformer_temporal_patch_size),
        layout_id="ltx_packed_token_flow_latent",
    )


def _normalise_ltx_vae_latent(vae: Any, latent: Any) -> Any:
    """复现 LTX pipeline 在 Transformer 输入前使用的 VAE latent 归一化。"""

    import torch

    latent = latent.to(dtype=torch.float32)
    mean_values = getattr(vae, "latents_mean", None)
    std_values = getattr(vae, "latents_std", None)
    if mean_values is None or std_values is None:
        raise RuntimeError("LTX VAE 缺少 latents_mean 或 latents_std buffer")
    mean_tensor = mean_values.to(device=latent.device, dtype=latent.dtype).view(1, -1, 1, 1, 1)
    std_tensor = std_values.to(device=latent.device, dtype=latent.dtype).view(1, -1, 1, 1, 1)
    scaling_factor = float(getattr(vae.config, "scaling_factor", 1.0))
    return (latent - mean_tensor) * scaling_factor / std_tensor.clamp_min(1e-8)


def encode_video_to_ltx_endpoint_latent(
    pipeline: Any,
    video_path: str | Path,
) -> tuple[Any, Any, PackedTokenFlowLatentLayout, dict[str, Any]]:
    """使用 LTX 官方 VAE 把攻击后视频重建为生成时的 packed endpoint latent。"""

    import torch

    device = pipeline._execution_device
    dtype = pipeline.vae.dtype
    video, frame_count = load_video_tensor_for_wan_vae(video_path, device=device, dtype=dtype)
    with torch.inference_mode():
        canonical = _retrieve_vae_latent(pipeline.vae.encode(video))
    canonical = _normalise_ltx_vae_latent(pipeline.vae, canonical)
    layout = build_ltx_latent_layout(
        pipeline,
        num_frames=frame_count,
        height=int(video.shape[-2]),
        width=int(video.shape[-1]),
    )
    packed = layout.from_canonical(canonical)
    return packed, canonical, layout, {
        "endpoint_video_frame_count": frame_count,
        "endpoint_vae_model_class": type(pipeline.vae).__name__,
        "endpoint_vae_encode_status": "ready",
        "endpoint_latent_shape": list(canonical.shape),
        "endpoint_native_latent_shape": list(packed.shape),
        "endpoint_evidence_source": "ltx_vae_reencoded_video_latent",
        **layout.as_dict(),
    }


def build_ltx_flow_schedule_points(
    scheduler: Any,
    *,
    num_inference_steps: int,
    device: Any,
    latent_layout: PackedTokenFlowLatentLayout,
) -> list[FlowSchedulePoint]:
    """按 LTX pipeline 的 sequence-length shift 规则建立真实 sigma 网格。"""

    step_count = int(num_inference_steps)
    if step_count < 2:
        raise ValueError("LTX replay step count 必须至少为2")
    sigmas = [
        1.0 + index * ((1.0 / step_count) - 1.0) / (step_count - 1)
        for index in range(step_count)
    ]
    sequence_length = latent_layout.num_frames * latent_layout.height * latent_layout.width
    config = scheduler.config
    base_sequence_length = int(config.get("base_image_seq_len", 256))
    maximum_sequence_length = int(config.get("max_image_seq_len", 4096))
    base_shift = float(config.get("base_shift", 0.5))
    maximum_shift = float(config.get("max_shift", 1.15))
    slope = (maximum_shift - base_shift) / max(1, maximum_sequence_length - base_sequence_length)
    mu = sequence_length * slope + (base_shift - slope * base_sequence_length)
    scheduler.set_timesteps(
        num_inference_steps=step_count,
        device=device,
        sigmas=sigmas,
        mu=mu,
    )
    timesteps = list(scheduler.timesteps)
    scheduler_sigmas = list(scheduler.sigmas)
    if len(scheduler_sigmas) != len(timesteps) + 1:
        raise RuntimeError("LTX Flow scheduler 的 sigmas 必须比 timesteps 多一个 endpoint")
    points = [
        FlowSchedulePoint(timestep=timestep, sigma=float(scheduler_sigmas[index]))
        for index, timestep in enumerate(timesteps)
    ]
    endpoint_timestep = timesteps[-1].new_zeros(()) if hasattr(timesteps[-1], "new_zeros") else 0.0
    points.append(FlowSchedulePoint(timestep=endpoint_timestep, sigma=float(scheduler_sigmas[-1])))
    return points


class LTXPromptConditionedVelocity:
    """使用 LTX 官方 prompt encoder 与 Transformer 计算真实条件 velocity。"""

    def __init__(
        self,
        pipeline: Any,
        *,
        prompt: str,
        latent_layout: PackedTokenFlowLatentLayout,
        negative_prompt: str | None = None,
        guidance_scale: float = 3.0,
        frame_rate: int = 8,
    ) -> None:
        self.pipeline = pipeline
        self.latent_layout = latent_layout
        self.guidance_scale = float(guidance_scale)
        self.frame_rate = int(frame_rate)
        self.device = pipeline._execution_device
        self.model = pipeline.transformer
        if self.model is None:
            raise RuntimeError("LTX replay 要求 pipeline.transformer 可用")
        self.model_dtype = self.model.dtype
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = pipeline.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.guidance_scale > 1.0,
            num_videos_per_prompt=1,
            device=self.device,
            dtype=self.model_dtype,
        )
        if self.guidance_scale > 1.0:
            prompt_embeds = self._concatenate(negative_prompt_embeds, prompt_embeds)
            prompt_attention_mask = self._concatenate(
                negative_prompt_attention_mask,
                prompt_attention_mask,
            )
        self.prompt_embeds = prompt_embeds.to(self.model_dtype)
        self.prompt_attention_mask = prompt_attention_mask

    @staticmethod
    def _concatenate(first: Any, second: Any) -> Any:
        import torch

        if first is None or second is None:
            raise RuntimeError("LTX classifier-free guidance 缺少正向或负向 prompt 表示")
        return torch.cat([first, second], dim=0)

    def __call__(self, latent: Any, timestep: Any, step_index: int) -> Any:
        """返回与官方 LTX denoising loop 同口径的 CFG velocity。"""

        import torch

        del step_index
        latent_input = latent.to(device=self.device, dtype=self.model_dtype)
        if self.guidance_scale > 1.0:
            latent_input = torch.cat([latent_input, latent_input], dim=0)
        if hasattr(timestep, "to"):
            timestep_value = timestep.to(device=self.device)
        else:
            timestep_value = torch.tensor(timestep, device=self.device)
        timestep_batch = timestep_value.reshape(-1)[0].expand(latent_input.shape[0])
        rope_interpolation_scale = (
            float(self.pipeline.vae_temporal_compression_ratio) / self.frame_rate,
            float(self.pipeline.vae_spatial_compression_ratio),
            float(self.pipeline.vae_spatial_compression_ratio),
        )
        cache_context = (
            self.model.cache_context("cond_uncond")
            if hasattr(self.model, "cache_context")
            else nullcontext()
        )
        with torch.inference_mode(), cache_context:
            velocity = self.model(
                hidden_states=latent_input,
                encoder_hidden_states=self.prompt_embeds,
                timestep=timestep_batch,
                encoder_attention_mask=self.prompt_attention_mask,
                num_frames=self.latent_layout.num_frames,
                height=self.latent_layout.height,
                width=self.latent_layout.width,
                rope_interpolation_scale=rope_interpolation_scale,
                attention_kwargs=None,
                return_dict=False,
            )[0].float()
        if self.guidance_scale > 1.0:
            unconditional, conditional = velocity.chunk(2)
            velocity = unconditional + self.guidance_scale * (conditional - unconditional)
        return velocity.to(dtype=latent.dtype)


class LTXKeyConditionedVelocity:
    """在真实 LTX velocity 上复现生成阶段的同源 SSTW 弱约束。"""

    def __init__(
        self,
        base_velocity: LTXPromptConditionedVelocity,
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
        """返回 LTX base velocity 与五维 tubelet 弱约束的合成结果。"""

        base = self.base_velocity(latent, timestep, step_index)
        if self._direction is None or tuple(self._direction.shape) != tuple(latent.shape):
            canonical = self.base_velocity.latent_layout.to_canonical(latent)
            canonical_direction, _metadata = build_flow_tubelet_key_direction_like(
                canonical,
                key_text=self.key_text,
                config=self.tubelet_config,
            )
            self._direction = self.base_velocity.latent_layout.from_canonical(canonical_direction)
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


def _packed_key_direction(
    latent: Any,
    *,
    latent_layout: PackedTokenFlowLatentLayout,
    key_text: str,
    tubelet_config: FlowTubeletKeyCodeConfig,
) -> Any:
    canonical = latent_layout.to_canonical(latent)
    canonical_direction, _metadata = build_flow_tubelet_key_direction_like(
        canonical,
        key_text=key_text,
        config=tubelet_config,
    )
    return latent_layout.from_canonical(canonical_direction)


def score_ltx_replay_trajectory_for_key(
    trajectory: ReplayTrajectory,
    schedule: Sequence[FlowSchedulePoint],
    *,
    latent_layout: PackedTokenFlowLatentLayout,
    key_text: str,
    tubelet_config: FlowTubeletKeyCodeConfig | None = None,
) -> dict[str, float | int | None]:
    """在固定 LTX packed replay 路径上计算候选 key 的路径投影。"""

    tubelet_config = tubelet_config or FlowTubeletKeyCodeConfig()
    states = trajectory.reverse_states
    if len(states) != len(schedule):
        raise RuntimeError("LTX replay states 与 Flow schedule 长度不一致")
    direction = _packed_key_direction(
        states[0],
        latent_layout=latent_layout,
        key_text=key_text,
        tubelet_config=tubelet_config,
    )
    records: list[dict[str, Any]] = []
    for step_index in range(len(states) - 1):
        delta_sigma = float(schedule[step_index + 1].sigma) - float(schedule[step_index].sigma)
        if abs(delta_sigma) <= 1e-12:
            continue
        phase = step_index / max(1, len(states) - 2)
        velocity = (states[step_index + 1] - states[step_index]) / delta_sigma
        step_record = compute_path_step_observation(
            states[step_index],
            states[step_index + 1],
            velocity,
            direction,
            flow_phase=phase,
        ).as_dict()
        step_record["replay_reliability_weight"] = replay_step_reliability_weight(
            trajectory,
            step_index + 1,
        )
        records.append(step_record)
    return aggregate_path_observations(records)


def compute_ltx_endpoint_evidence_for_key(
    replay: LTXFlowReplayResult,
    *,
    key_text: str,
    tubelet_config: FlowTubeletKeyCodeConfig | None = None,
) -> EndpointLatentEvidence:
    """在 LTX 五维 VAE endpoint 上计算候选 key 证据。"""

    return compute_endpoint_latent_evidence(
        replay.canonical_endpoint_latent,
        key_text=key_text,
        tubelet_config=tubelet_config,
    )


def evaluate_fixed_ltx_replay_hypothesis_for_key(
    pipeline: Any,
    replay: LTXFlowReplayResult,
    *,
    prompt: str,
    key_text: str,
    negative_prompt: str | None = None,
    guidance_scale: float = 3.0,
    frame_rate: int = 8,
    tubelet_config: FlowTubeletKeyCodeConfig | None = None,
) -> tuple[ReplayTrajectory, dict[str, float | int | None]]:
    """在固定 attacked-video LTX 反演观测上评估另一把候选 key。"""

    tubelet_config = tubelet_config or FlowTubeletKeyCodeConfig()
    base_velocity = LTXPromptConditionedVelocity(
        pipeline,
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        frame_rate=frame_rate,
        latent_layout=replay.latent_layout,
    )
    keyed_velocity = LTXKeyConditionedVelocity(
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
    return hypothesis, score_ltx_replay_trajectory_for_key(
        hypothesis,
        replay.primary_schedule,
        latent_layout=replay.latent_layout,
        key_text=key_text,
        tubelet_config=tubelet_config,
    )


def run_ltx_control_replay(
    pipeline: Any,
    endpoint_latent: Any,
    *,
    latent_layout: PackedTokenFlowLatentLayout,
    prompt: str,
    key_text: str,
    num_inference_steps: int,
    scheduler: Any | None = None,
    fixed_trajectory: ReplayTrajectory | None = None,
    negative_prompt: str | None = None,
    guidance_scale: float = 3.0,
    frame_rate: int = 8,
    tubelet_config: FlowTubeletKeyCodeConfig | None = None,
) -> tuple[ReplayTrajectory, tuple[FlowSchedulePoint, ...], dict[str, float | int | None]]:
    """使用显式 prompt 或 scheduler 执行一个可审计的 LTX replay control。

    传入 fixed_trajectory 时, wrong condition 只改变 forward hypothesis,
    reverse observation 和 null replay 均保持为正确条件下的固定结果。
    """

    tubelet_config = tubelet_config or FlowTubeletKeyCodeConfig()
    velocity = LTXPromptConditionedVelocity(
        pipeline,
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        frame_rate=frame_rate,
        latent_layout=latent_layout,
    )
    schedule = build_ltx_flow_schedule_points(
        scheduler or pipeline.scheduler,
        num_inference_steps=int(num_inference_steps),
        device=velocity.device,
        latent_layout=latent_layout,
    )
    keyed_velocity = LTXKeyConditionedVelocity(
        velocity,
        key_text=key_text,
        total_steps=len(schedule),
        tubelet_config=tubelet_config,
    )
    trajectory = (
        evaluate_candidate_on_fixed_inversion(
            endpoint_latent,
            schedule,
            fixed_trajectory,
            keyed_velocity,
        )
        if fixed_trajectory is not None
        else run_key_independent_inversion_hypothesis(
            endpoint_latent,
            schedule,
            velocity,
            keyed_velocity,
        )
    )
    path_evidence = score_ltx_replay_trajectory_for_key(
        trajectory,
        schedule,
        latent_layout=latent_layout,
        key_text=key_text,
        tubelet_config=tubelet_config,
    )
    return trajectory, tuple(schedule), path_evidence


def run_ltx_attacked_video_replay(
    pipeline: Any,
    video_path: str | Path,
    *,
    prompt: str,
    key_text: str,
    negative_prompt: str | None = None,
    guidance_scale: float = 3.0,
    frame_rate: int = 8,
    replay_step_counts: Sequence[int] = (16, 20, 24),
    tubelet_config: FlowTubeletKeyCodeConfig | None = None,
) -> LTXFlowReplayResult:
    """从攻击后视频执行多时间网格 LTX inversion/replay 并返回正式证据。"""

    tubelet_config = tubelet_config or FlowTubeletKeyCodeConfig()
    endpoint_latent, canonical_endpoint, layout, metadata = encode_video_to_ltx_endpoint_latent(
        pipeline,
        video_path,
    )
    endpoint_evidence = compute_endpoint_latent_evidence(
        canonical_endpoint,
        key_text=key_text,
        tubelet_config=tubelet_config,
    )
    base_velocity = LTXPromptConditionedVelocity(
        pipeline,
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        frame_rate=frame_rate,
        latent_layout=layout,
    )
    replays: list[ReplayTrajectory] = []
    schedules: list[list[FlowSchedulePoint]] = []
    for step_count in replay_step_counts:
        schedule = build_ltx_flow_schedule_points(
            pipeline.scheduler,
            num_inference_steps=int(step_count),
            device=base_velocity.device,
            latent_layout=layout,
        )
        schedules.append(schedule)
        keyed_velocity = LTXKeyConditionedVelocity(
            base_velocity,
            key_text=key_text,
            total_steps=len(schedule),
            tubelet_config=tubelet_config,
        )
        replays.append(run_key_independent_inversion_hypothesis(
            endpoint_latent,
            schedule,
            base_velocity,
            keyed_velocity,
        ))
    uncertainty = estimate_replay_uncertainty(replays)
    primary_index = len(replays) // 2
    path_evidence = score_ltx_replay_trajectory_for_key(
        replays[primary_index],
        schedules[primary_index],
        latent_layout=layout,
        key_text=key_text,
        tubelet_config=tubelet_config,
    )
    return LTXFlowReplayResult(
        endpoint_evidence=endpoint_evidence,
        path_evidence=path_evidence,
        replay_uncertainty=uncertainty,
        replay_trajectories=tuple(replays),
        endpoint_metadata=metadata,
        replay_step_counts=tuple(int(value) for value in replay_step_counts),
        endpoint_latent=endpoint_latent,
        canonical_endpoint_latent=canonical_endpoint,
        latent_layout=layout,
        primary_schedule=tuple(schedules[primary_index]),
        primary_replay_index=primary_index,
    )
