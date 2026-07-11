"""把真实速度场约束接入 Diffusers Flow scheduler 的运行时包装器。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from main.methods.state_space_watermark.flow_latent_layout import (
    FiveDimensionalFlowLatentLayout,
    FlowLatentLayout,
)
from main.methods.state_space_watermark.flow_tubelet_key_code import (
    FlowTubeletKeyCodeConfig,
    build_flow_tubelet_key_direction_like,
)
from main.methods.state_space_watermark.path_observation import compute_path_step_observation
from main.methods.state_space_watermark.velocity_field_constraint import (
    VelocityFieldConstraintConfig,
    apply_velocity_field_constraint,
)


@dataclass(frozen=True)
class FlowVelocityRuntimeMechanismConfig:
    """定义 scheduler 包装器启用的核心速度场机制。

    该配置只包含可组合的运行原语。论文中的受控实验如何选择这些参数，
    由外层实验模块负责，核心运行时不识别实验变体名称。
    """

    velocity_constraint_enabled: bool = True
    endpoint_control_enabled: bool = True
    terminal_endpoint_perturbation_enabled: bool = False


@dataclass
class FlowVelocityConstraintRuntime:
    """在 context 中包装 scheduler.step, 并在退出时恢复原实现。

    Diffusers WanPipeline 会把经过 classifier-free guidance 的 `noise_pred` 作为
    `scheduler.step` 第一个参数。当前包装器在该位置修改模型输出, 因而不需要复制
    第三方 pipeline 的完整 denoising loop, 同时能保持官方版本升级边界清晰。
    """

    scheduler: Any
    key_text: str
    total_steps: int
    mechanism_config: FlowVelocityRuntimeMechanismConfig = field(
        default_factory=FlowVelocityRuntimeMechanismConfig
    )
    velocity_config: VelocityFieldConstraintConfig = field(default_factory=VelocityFieldConstraintConfig)
    tubelet_config: FlowTubeletKeyCodeConfig = field(default_factory=FlowTubeletKeyCodeConfig)
    latent_layout: FlowLatentLayout = field(default_factory=FiveDimensionalFlowLatentLayout)
    require_flow_scheduler: bool = True

    def __post_init__(self) -> None:
        self._original_step: Any | None = None
        self._key_direction: Any | None = None
        self._key_metadata: dict[str, Any] = {}
        self._step_records: list[dict[str, Any]] = []
        self._endpoint_latent: Any | None = None
        self._step_index = 0

    @property
    def step_records(self) -> list[dict[str, Any]]:
        """返回真实 scheduler 更新产生的逐步证据副本。"""

        return [dict(record) for record in self._step_records]

    @property
    def endpoint_latent(self) -> Any | None:
        """返回最后一个 scheduler step 的模型原生 endpoint latent。"""

        return self._endpoint_latent

    @property
    def canonical_endpoint_latent(self) -> Any | None:
        """返回最后一个 scheduler step 的 SSTW 五维规范 endpoint latent。"""

        if self._endpoint_latent is None:
            return None
        return self.latent_layout.to_canonical(self._endpoint_latent)

    @property
    def key_metadata(self) -> dict[str, Any]:
        """返回当前运行使用的 tubelet key code 摘要。"""

        return dict(self._key_metadata)

    def __enter__(self) -> "FlowVelocityConstraintRuntime":
        scheduler_name = type(self.scheduler).__name__
        if self.require_flow_scheduler and "FlowMatch" not in scheduler_name:
            raise RuntimeError(f"正式 velocity constraint 需要 FlowMatch scheduler, 当前为 {scheduler_name}")
        if self.total_steps <= 0:
            raise ValueError("total_steps 必须为正整数")
        self._original_step = self.scheduler.step

        def constrained_step(model_output: Any, timestep: Any, sample: Any, *args: Any, **kwargs: Any):
            return self._run_step(model_output, timestep, sample, *args, **kwargs)

        self.scheduler.step = constrained_step
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._original_step is not None:
            self.scheduler.step = self._original_step

    def _direction(self, sample: Any) -> Any:
        canonical_sample = self.latent_layout.to_canonical(sample)
        if self._key_direction is None or tuple(self._key_direction.shape) != tuple(sample.shape):
            canonical_direction, self._key_metadata = build_flow_tubelet_key_direction_like(
                canonical_sample,
                key_text=self.key_text,
                config=self.tubelet_config,
            )
            self._key_direction = self.latent_layout.from_canonical(canonical_direction)
            self._key_metadata.update(self.latent_layout.as_dict())
        return self._key_direction.to(device=sample.device, dtype=sample.dtype)

    @staticmethod
    def _replace_sample(step_output: Any, replacement: Any) -> Any:
        """替换 scheduler 的 tuple 输出首项, 正式 Wan 路径使用该返回结构。"""

        if isinstance(step_output, tuple):
            return (replacement, *step_output[1:])
        if hasattr(step_output, "prev_sample"):
            step_output.prev_sample = replacement
            return step_output
        raise TypeError("不支持的 scheduler.step 返回类型")

    @staticmethod
    def _first_sample(step_output: Any) -> Any:
        if isinstance(step_output, tuple):
            return step_output[0]
        if hasattr(step_output, "prev_sample"):
            return step_output.prev_sample
        raise TypeError("不支持的 scheduler.step 返回类型")

    def _run_step(self, model_output: Any, timestep: Any, sample: Any, *args: Any, **kwargs: Any):
        if self._original_step is None:
            raise RuntimeError("FlowVelocityConstraintRuntime 尚未进入 context")
        direction = self._direction(sample)
        flow_phase = self._step_index / max(1, self.total_steps - 1)
        if self.mechanism_config.velocity_constraint_enabled:
            constrained_velocity, velocity_record = apply_velocity_field_constraint(
                model_output,
                sample,
                direction,
                flow_phase=flow_phase,
                config=self.velocity_config,
                tubelet_config=self.tubelet_config,
                endpoint_control_enabled=self.mechanism_config.endpoint_control_enabled,
            )
        else:
            constrained_velocity = model_output
            velocity_record = {
                "velocity_field_constraint_status": "disabled_by_mechanism_config",
                "velocity_field_source": "scheduler_model_output_before_flow_match_step",
                "flow_phase": round(float(flow_phase), 8),
                "flow_phase_weight": 0.0,
                "velocity_constraint_lambda": 0.0,
                "velocity_constraint_delta_norm": 0.0,
                "velocity_constraint_delta_ratio": 0.0,
                "endpoint_control_enabled": self.mechanism_config.endpoint_control_enabled,
            }

        step_output = self._original_step(constrained_velocity, timestep, sample, *args, **kwargs)
        sample_after = self._first_sample(step_output)
        if (
            self.mechanism_config.terminal_endpoint_perturbation_enabled
            and self._step_index == self.total_steps - 1
        ):
            endpoint_delta_norm = (
                float(sample_after.detach().float().norm().item())
                * self.velocity_config.velocity_norm_ratio_budget
                * self.velocity_config.lambda_max
            )
            sample_after = sample_after + direction * endpoint_delta_norm
            step_output = self._replace_sample(step_output, sample_after)
            velocity_record["terminal_endpoint_perturbation_delta_norm"] = round(
                endpoint_delta_norm,
                6,
            )

        path_record = compute_path_step_observation(
            sample,
            sample_after,
            constrained_velocity,
            direction,
            flow_phase=flow_phase,
        ).as_dict()
        timestep_value = float(timestep.detach().float().reshape(-1)[0].item()) if hasattr(timestep, "detach") else float(timestep)
        self._step_records.append({
            "trajectory_step_index": self._step_index,
            "trajectory_timestep": timestep_value,
            "flow_scheduler_runtime_verified": True,
            "velocity_constraint_enabled": (
                self.mechanism_config.velocity_constraint_enabled
            ),
            "terminal_endpoint_perturbation_enabled": (
                self.mechanism_config.terminal_endpoint_perturbation_enabled
            ),
            **self._key_metadata,
            **velocity_record,
            **path_record,
        })
        self._endpoint_latent = sample_after.detach().clone()
        self._step_index += 1
        return step_output
