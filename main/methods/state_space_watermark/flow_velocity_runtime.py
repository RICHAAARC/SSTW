"""把真实速度场约束接入 Diffusers Flow scheduler 的运行时包装器。"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import math
from typing import Any

from main.methods.state_space_watermark.flow_latent_layout import (
    FiveDimensionalFlowLatentLayout,
    FlowLatentLayout,
)
from main.methods.state_space_watermark.flow_tubelet_key_code import (
    FlowTubeletKeyContext,
    FlowTubeletKeyCodeConfig,
    build_flow_tubelet_key_direction_like,
    flow_phase_weight,
    flow_tubelet_key_context_digest,
)
from main.methods.state_space_watermark.path_observation import compute_path_step_observation
from main.methods.state_space_watermark.signed_trajectory_carrier import (
    SignedTrajectoryCarrierConfig,
    SignedTrajectorySchedule,
    apply_signed_trajectory_two_channel_constraint,
    build_signed_trajectory_schedule,
)
from main.methods.state_space_watermark.predictive_trajectory_carrier import (
    PredictiveTrajectoryCarrierConfig,
    PredictiveTrajectorySchedule,
    apply_predictive_trajectory_constraint,
    build_predictive_trajectory_schedule,
)
from main.methods.state_space_watermark.continuous_trajectory_code import (
    ContinuousTrajectorySchedule,
    build_continuous_trajectory_schedule,
)
from main.methods.state_space_watermark.state_rotation_operator import (
    PROMPT_ORTHOGONAL_CANONICAL_PLANE_CONSTRUCTION_DEVICE,
    PromptOrthogonalDirection,
    PromptOrthogonalSubkeys,
    build_prompt_orthogonal_state_direction,
    build_state_rotation_plane_like,
    derive_prompt_orthogonal_subkeys,
)
from main.methods.state_space_watermark.state_trajectory_injection import (
    PROMPT_ORTHOGONAL_SCHEDULER_CONTROL_DTYPE,
    PromptOrthogonalInjectionConfig,
    apply_prompt_orthogonal_state_trajectory_injection,
)
from main.methods.state_space_watermark.velocity_field_constraint import (
    VelocityControlContext,
    VelocityFieldConstraintConfig,
    apply_velocity_field_constraint,
)


def normalized_flow_phase_from_sigma_interval(
    sigma_grid: Any,
    step_index: int,
) -> float:
    """把真实 scheduler 的相邻 sigma 区间映射为规范 Flow phase。

    该实现使用区间中点而不是离散 step 编号。因此，同一连续 sigma 路径在不同
    网格密度下会得到一致的 phase 语义，可由生成、replay 和 endpoint 积分共同复用。
    """

    sigmas = [float(value.detach().float().item()) if hasattr(value, "detach") else float(value) for value in sigma_grid]
    index = int(step_index)
    if len(sigmas) < 2 or index < 0 or index + 1 >= len(sigmas):
        raise ValueError("Flow phase 需要包含当前区间的完整 sigma 网格")
    span = sigmas[-1] - sigmas[0]
    if not math.isfinite(span) or abs(span) <= 1e-12:
        raise ValueError("Flow phase 的 sigma 网格总跨度必须有限且非零")
    midpoint = 0.5 * (sigmas[index] + sigmas[index + 1])
    phase = (midpoint - sigmas[0]) / span
    return max(0.0, min(1.0, float(phase)))


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
    key_context: FlowTubeletKeyContext | None = None
    signed_trajectory_carrier_config: SignedTrajectoryCarrierConfig | None = None
    predictive_trajectory_carrier_config: (
        PredictiveTrajectoryCarrierConfig | None
    ) = None
    prompt_orthogonal_master_key_text: str | None = None
    prompt_orthogonal_injection_config: (
        PromptOrthogonalInjectionConfig | None
    ) = None
    prompt_orthogonal_scheduler_float32_control: bool = False
    require_flow_scheduler: bool = True

    def __post_init__(self) -> None:
        self._original_step: Any | None = None
        self._key_direction: Any | None = None
        self._canonical_key_direction: Any | None = None
        self._key_metadata: dict[str, Any] = {}
        self._step_records: list[dict[str, Any]] = []
        self._endpoint_latent: Any | None = None
        self._integrated_direction_accumulator: Any | None = None
        self._integrated_phase_count = 0
        self._integrated_weight_sum = 0.0
        self._integrated_phase_schedule_bindings: list[str] = []
        self._flow_phases: list[float] = []
        self._integration_weights: list[float] = []
        self._cumulative_control_energy = 0.0
        self._cumulative_reference_energy = 0.0
        self._step_index = 0
        self._signed_trajectory_schedule: SignedTrajectorySchedule | None = None
        self._predictive_trajectory_schedule: (
            PredictiveTrajectorySchedule | None
        ) = None
        self._prompt_orthogonal_subkeys: PromptOrthogonalSubkeys | None = None
        self._prompt_orthogonal_schedule: (
            ContinuousTrajectorySchedule | None
        ) = None
        self._prompt_orthogonal_state_rotation_plane: (
            tuple[Any, Any, str] | None
        ) = None
        self._prompt_orthogonal_direction_result: (
            PromptOrthogonalDirection | None
        ) = None
        self._canonical_endpoint_key_direction: Any | None = None

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
    def canonical_integrated_key_direction(self) -> Any | None:
        """返回按真实 sigma 网格积分后的规范 endpoint key 方向。

        该方向只累计实际启用速度约束的 phase 区间。权重由
        ``|delta_sigma| * flow_phase_weight`` 决定，不读取输出视频或检测分数。
        """

        if self._integrated_direction_accumulator is None:
            return None
        norm = self._integrated_direction_accumulator.detach().float().norm()
        if float(norm.item()) <= 1e-8:
            return None
        return self._integrated_direction_accumulator / norm

    @property
    def flow_phases(self) -> tuple[float, ...]:
        """返回 endpoint 积分使用的规范 phase 网格。"""

        return tuple(self._flow_phases)

    @property
    def integration_weights(self) -> tuple[float, ...]:
        """返回 endpoint 积分使用的非负 schedule 权重。"""

        return tuple(self._integration_weights)

    @property
    def key_metadata(self) -> dict[str, Any]:
        """返回当前运行使用的 tubelet key code 摘要。"""

        return {
            **self._key_metadata,
            **self._integrated_key_metadata(),
        }

    def __enter__(self) -> "FlowVelocityConstraintRuntime":
        scheduler_name = type(self.scheduler).__name__
        if self.require_flow_scheduler and "FlowMatch" not in scheduler_name:
            raise RuntimeError(f"正式 velocity constraint 需要 FlowMatch scheduler, 当前为 {scheduler_name}")
        if self.total_steps <= 0:
            raise ValueError("total_steps 必须为正整数")
        prompt_orthogonal_enabled = bool(
            self.prompt_orthogonal_master_key_text is not None
            or self.prompt_orthogonal_injection_config is not None
        )
        if (
            (
                self.signed_trajectory_carrier_config is not None
                or self.predictive_trajectory_carrier_config is not None
                or prompt_orthogonal_enabled
            )
            and self.key_context is None
        ):
            raise ValueError("trajectory carrier runtime 要求完整 key context")
        enabled_carrier_count = sum(
            (
                self.signed_trajectory_carrier_config is not None,
                self.predictive_trajectory_carrier_config is not None,
                prompt_orthogonal_enabled,
            )
        )
        if enabled_carrier_count > 1:
            raise ValueError("同一次 runtime 不得同时启用多种 trajectory carrier")
        if prompt_orthogonal_enabled:
            if (
                self.prompt_orthogonal_master_key_text is None
                or self.prompt_orthogonal_injection_config is None
            ):
                raise ValueError(
                    "prompt-orthogonal runtime 要求 master key 与 injection config"
                )
            if (
                self.mechanism_config.endpoint_control_enabled
                or self.mechanism_config.terminal_endpoint_perturbation_enabled
            ):
                raise ValueError(
                    "prompt-orthogonal runtime 禁止 endpoint/terminal control"
                )
            self._prompt_orthogonal_subkeys = (
                derive_prompt_orthogonal_subkeys(
                    self.prompt_orthogonal_master_key_text
                )
            )
        if self.prompt_orthogonal_scheduler_float32_control:
            if (
                self.signed_trajectory_carrier_config is not None
                or self.predictive_trajectory_carrier_config is not None
                or self.mechanism_config.endpoint_control_enabled
                or self.mechanism_config.terminal_endpoint_perturbation_enabled
            ):
                raise ValueError(
                    "prompt-orthogonal FP32 scheduler control "
                    "不得与旧 carrier/endpoint control 混用"
                )
        self._original_step = self.scheduler.step

        def constrained_step(model_output: Any, timestep: Any, sample: Any, *args: Any, **kwargs: Any):
            return self._run_step(model_output, timestep, sample, *args, **kwargs)

        self.scheduler.step = constrained_step
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._original_step is not None:
            self.scheduler.step = self._original_step

    def _key_context_digest(self) -> str | None:
        """返回不随 phase 变化的生成上下文摘要。"""

        if self.key_context is None:
            return None
        return flow_tubelet_key_context_digest(self.key_context)

    def _integrated_key_metadata(self) -> dict[str, Any]:
        """返回可随 generation record 保存的 schedule 积分摘要。"""

        integrated = self.canonical_integrated_key_direction
        integrated_digest = (
            sha256(
                "||".join(self._integrated_phase_schedule_bindings).encode("utf-8")
            ).hexdigest()
            if self._integrated_phase_schedule_bindings
            else None
        )
        prompt_orthogonal_formal_context_complete = bool(
            self.prompt_orthogonal_injection_config is not None
            and self._step_records
            and all(
                record.get("flow_runtime_step_formal_context_complete") is True
                for record in self._step_records
            )
        )
        return {
            "flow_tubelet_key_context_digest": self._key_context_digest(),
            "flow_integrated_phase_count": self._integrated_phase_count,
            "flow_integrated_weight_sum": round(self._integrated_weight_sum, 10),
            "flow_integrated_key_direction_digest": integrated_digest,
            "flow_integrated_key_direction_ready": integrated is not None,
            "flow_runtime_formal_context_complete": bool(
                prompt_orthogonal_formal_context_complete
                or (
                    self.key_context is not None
                    and integrated is not None
                    and self._step_records
                    and all(
                        record.get("flow_runtime_step_formal_context_complete")
                        is True
                        for record in self._step_records
                    )
                )
            ),
        }

    def _formal_sigma_interval(self) -> tuple[float, float] | None:
        """读取当前真实 scheduler 区间，完整上下文缺失时保持旧诊断语义。"""

        if self.key_context is None:
            return None
        sigma_grid = getattr(self.scheduler, "sigmas", None)
        if sigma_grid is None:
            raise RuntimeError("正式 Flow runtime 缺少 scheduler.sigmas")
        if len(sigma_grid) != self.total_steps + 1:
            raise RuntimeError(
                "正式 Flow runtime 的 sigma 网格长度与 total_steps 不一致: "
                f"sigmas={len(sigma_grid)}, total_steps={self.total_steps}"
            )
        if self._step_index + 1 >= len(sigma_grid):
            raise RuntimeError("正式 Flow runtime 的 scheduler step 超出 sigma 网格")
        sigma_before = (
            float(sigma_grid[self._step_index].detach().float().item())
            if hasattr(sigma_grid[self._step_index], "detach")
            else float(sigma_grid[self._step_index])
        )
        sigma_after = (
            float(sigma_grid[self._step_index + 1].detach().float().item())
            if hasattr(sigma_grid[self._step_index + 1], "detach")
            else float(sigma_grid[self._step_index + 1])
        )
        delta_sigma = sigma_after - sigma_before
        if not math.isfinite(delta_sigma) or abs(delta_sigma) <= 1e-12:
            raise RuntimeError("正式 Flow runtime 要求有限且非零的真实 delta_sigma")
        phase = normalized_flow_phase_from_sigma_interval(
            sigma_grid,
            self._step_index,
        )
        return delta_sigma, phase

    def _ensure_signed_trajectory_schedule(self) -> None:
        """在 pipeline 已物化 timesteps 后绑定真实 generation schedule。"""

        if (
            self.signed_trajectory_carrier_config is None
            or self._signed_trajectory_schedule is not None
        ):
            return
        if self.key_context is None:
            raise RuntimeError("signed trajectory runtime 缺少 key context")
        sigma_grid = getattr(self.scheduler, "sigmas", None)
        if sigma_grid is None or len(sigma_grid) != self.total_steps + 1:
            raise RuntimeError(
                "signed trajectory runtime 要求完整 generation sigma grid"
            )
        phases: list[float] = []
        weights: list[float] = []
        for step_index in range(self.total_steps):
            phase = normalized_flow_phase_from_sigma_interval(
                sigma_grid,
                step_index,
            )
            before = (
                float(sigma_grid[step_index].detach().float().item())
                if hasattr(sigma_grid[step_index], "detach")
                else float(sigma_grid[step_index])
            )
            after = (
                float(
                    sigma_grid[step_index + 1].detach().float().item()
                )
                if hasattr(sigma_grid[step_index + 1], "detach")
                else float(sigma_grid[step_index + 1])
            )
            phases.append(phase)
            weights.append(
                abs(after - before)
                * flow_phase_weight(phase, self.tubelet_config)
            )
        self._signed_trajectory_schedule = build_signed_trajectory_schedule(
            key_text=self.key_text,
            key_context_digest=flow_tubelet_key_context_digest(
                self.key_context
            ),
            flow_phases=phases,
            active_weights=weights,
        )

    def _ensure_predictive_trajectory_schedule(self) -> None:
        """在 pipeline 已物化 timesteps 后绑定预测同步真实网格。"""

        if (
            self.predictive_trajectory_carrier_config is None
            or self._predictive_trajectory_schedule is not None
        ):
            return
        if self.key_context is None:
            raise RuntimeError("predictive trajectory runtime 缺少 key context")
        sigma_grid = getattr(self.scheduler, "sigmas", None)
        if sigma_grid is None or len(sigma_grid) != self.total_steps + 1:
            raise RuntimeError(
                "predictive trajectory runtime 要求完整 generation sigma grid"
            )
        phases: list[float] = []
        weights: list[float] = []
        for step_index in range(self.total_steps):
            phase = normalized_flow_phase_from_sigma_interval(
                sigma_grid,
                step_index,
            )
            before = (
                float(sigma_grid[step_index].detach().float().item())
                if hasattr(sigma_grid[step_index], "detach")
                else float(sigma_grid[step_index])
            )
            after = (
                float(sigma_grid[step_index + 1].detach().float().item())
                if hasattr(sigma_grid[step_index + 1], "detach")
                else float(sigma_grid[step_index + 1])
            )
            phases.append(phase)
            weights.append(
                abs(after - before)
                * flow_phase_weight(phase, self.tubelet_config)
            )
        self._predictive_trajectory_schedule = (
            build_predictive_trajectory_schedule(
                key_text=self.key_text,
                key_context_digest=flow_tubelet_key_context_digest(
                    self.key_context
                ),
                flow_phases=phases,
                active_weights=weights,
            )
        )

    def _ensure_prompt_orthogonal_schedule(self) -> None:
        """Bind the prompt-independent continuous function to real sigmas."""

        if (
            self.prompt_orthogonal_injection_config is None
            or self._prompt_orthogonal_schedule is not None
        ):
            return
        if self._prompt_orthogonal_subkeys is None:
            raise RuntimeError("prompt-orthogonal runtime 缺少派生子 key")
        sigma_grid = getattr(self.scheduler, "sigmas", None)
        if sigma_grid is None or len(sigma_grid) != self.total_steps + 1:
            raise RuntimeError(
                "prompt-orthogonal runtime 要求完整 generation sigma grid"
            )
        phases: list[float] = []
        weights: list[float] = []
        for step_index in range(self.total_steps):
            phase = normalized_flow_phase_from_sigma_interval(
                sigma_grid,
                step_index,
            )
            before_value = sigma_grid[step_index]
            after_value = sigma_grid[step_index + 1]
            before = (
                float(before_value.detach().float().item())
                if hasattr(before_value, "detach")
                else float(before_value)
            )
            after = (
                float(after_value.detach().float().item())
                if hasattr(after_value, "detach")
                else float(after_value)
            )
            phases.append(phase)
            weights.append(
                abs(after - before)
                * flow_phase_weight(phase, self.tubelet_config)
            )
        self._prompt_orthogonal_schedule = (
            build_continuous_trajectory_schedule(
                trajectory_code_subkey=(
                    self._prompt_orthogonal_subkeys.trajectory_code_subkey
                ),
                flow_phases=phases,
                active_weights=weights,
            )
        )

    def _prompt_orthogonal_direction(
        self,
        sample: Any,
        model_output: Any,
    ) -> Any:
        if self._prompt_orthogonal_subkeys is None:
            raise RuntimeError("prompt-orthogonal runtime 缺少派生子 key")
        if self._prompt_orthogonal_state_rotation_plane is None:
            self._prompt_orthogonal_state_rotation_plane = (
                build_state_rotation_plane_like(
                    sample.detach().float(),
                    state_operator_subkey=(
                        self._prompt_orthogonal_subkeys.state_operator_subkey
                    ),
                )
            )
        result = build_prompt_orthogonal_state_direction(
            sample,
            model_output,
            state_operator_subkey=(
                self._prompt_orthogonal_subkeys.state_operator_subkey
            ),
            state_rotation_plane=(
                self._prompt_orthogonal_state_rotation_plane
            ),
        )
        self._prompt_orthogonal_direction_result = result
        schedule = self._prompt_orthogonal_schedule
        if schedule is None:
            raise RuntimeError("prompt-orthogonal runtime 缺少连续 schedule")
        self._key_direction = result.direction.to(
            device=sample.device,
            dtype=sample.dtype,
        )
        self._canonical_key_direction = self.latent_layout.to_canonical(
            self._key_direction
        )
        self._key_metadata = {
            "prompt_orthogonal_domain_separation_digest": (
                self._prompt_orthogonal_subkeys.domain_separation_digest
            ),
            "prompt_orthogonal_operator_plane_digest": (
                result.operator_plane_digest
            ),
            "prompt_orthogonal_plane_construction_device": (
                PROMPT_ORTHOGONAL_CANONICAL_PLANE_CONSTRUCTION_DEVICE
            ),
            "prompt_orthogonal_operator_rank": result.operator_rank,
            "prompt_orthogonal_projection_retained_ratio": (
                result.projection_retained_ratio
            ),
            "prompt_orthogonal_state_orthogonality_residual": (
                result.state_orthogonality_residual
            ),
            "prompt_orthogonal_velocity_orthogonality_residual": (
                result.velocity_orthogonality_residual
            ),
            "prompt_orthogonal_continuous_function_digest": (
                schedule.continuous_function_digest
            ),
            "prompt_orthogonal_schedule_projection_digest": (
                schedule.schedule_projection_digest
            ),
            "prompt_orthogonal_weighted_code_residual": (
                schedule.weighted_residual
            ),
            "prompt_orthogonal_weighted_code_energy": (
                schedule.weighted_code_energy
            ),
            "prompt_orthogonal_minimum_active_code_magnitude": (
                schedule.minimum_active_code_magnitude
            ),
        }
        return self._key_direction

    def _direction(self, sample: Any, *, flow_phase: float) -> Any:
        canonical_sample = self.latent_layout.to_canonical(sample)
        canonical_direction, phase_metadata = build_flow_tubelet_key_direction_like(
            canonical_sample,
            key_text=self.key_text,
            config=self.tubelet_config,
            flow_phase=(flow_phase if self.key_context is not None else None),
            key_context=self.key_context,
            phase_code_override=(
                1.0
                if self.signed_trajectory_carrier_config is not None
                else None
            ),
        )
        self._canonical_endpoint_key_direction = canonical_direction
        active_schedule = (
            self._signed_trajectory_schedule
            or self._predictive_trajectory_schedule
        )
        if active_schedule is not None:
            ac_code = active_schedule.codes[self._step_index]
            canonical_direction = canonical_direction * (
                1.0 if ac_code >= 0.0 else -1.0
            )
            phase_metadata.update(
                active_schedule.metadata_for_step(
                    self._step_index
                )
            )
        self._canonical_key_direction = canonical_direction
        self._key_direction = self.latent_layout.from_canonical(
            canonical_direction
        )
        self._key_metadata = {
            **phase_metadata,
            **self.latent_layout.as_dict(),
            "flow_tubelet_key_context_digest": self._key_context_digest(),
        }
        return self._key_direction.to(device=sample.device, dtype=sample.dtype)

    def _accumulate_integrated_direction(
        self,
        *,
        flow_phase: float,
        delta_sigma: float,
        additional_weight: float = 0.0,
        canonical_direction: Any | None = None,
    ) -> None:
        """累计可由同一 scheduler 网格重建的 endpoint joint code。"""

        active_direction = (
            canonical_direction
            if canonical_direction is not None
            else self._canonical_key_direction
        )
        if self.key_context is None or active_direction is None:
            return
        schedule_weight = (
            abs(float(delta_sigma))
            * flow_phase_weight(flow_phase, self.tubelet_config)
            + max(0.0, float(additional_weight))
        )
        if schedule_weight <= 0.0:
            return
        direction = active_direction.detach().float()
        if self._integrated_direction_accumulator is None:
            self._integrated_direction_accumulator = direction * 0.0
        self._integrated_direction_accumulator = (
            self._integrated_direction_accumulator + direction * schedule_weight
        )
        self._flow_phases.append(float(flow_phase))
        self._integration_weights.append(float(schedule_weight))
        self._integrated_phase_count += 1
        self._integrated_weight_sum += schedule_weight
        self._integrated_phase_schedule_bindings.append(
            f"{self._key_metadata['flow_key_direction_digest']}::"
            f"{float(flow_phase):.17g}::{float(schedule_weight):.17g}"
        )

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
        formal_interval = self._formal_sigma_interval()
        if formal_interval is None:
            delta_sigma = None
            flow_phase = self._step_index / max(1, self.total_steps - 1)
            control_context = None
        else:
            delta_sigma, flow_phase = formal_interval
            control_context = VelocityControlContext(
                delta_sigma=delta_sigma,
                cumulative_control_energy=self._cumulative_control_energy,
                cumulative_reference_energy=self._cumulative_reference_energy,
                remaining_step_count=self.total_steps - self._step_index,
            )
        self._ensure_signed_trajectory_schedule()
        self._ensure_predictive_trajectory_schedule()
        self._ensure_prompt_orthogonal_schedule()
        if self.prompt_orthogonal_injection_config is not None:
            direction = self._prompt_orthogonal_direction(
                sample,
                model_output,
            )
        else:
            direction = self._direction(sample, flow_phase=flow_phase)
        if (
            self.mechanism_config.velocity_constraint_enabled
            and self.prompt_orthogonal_injection_config is not None
        ):
            if (
                control_context is None
                or self._prompt_orthogonal_schedule is None
                or self._prompt_orthogonal_direction_result is None
            ):
                raise RuntimeError(
                    "prompt-orthogonal runtime 缺少正式 schedule/control context"
                )
            constrained_velocity, injection_record = (
                apply_prompt_orthogonal_state_trajectory_injection(
                    model_output,
                    sample,
                    self._prompt_orthogonal_direction_result,
                    continuous_code=(
                        self._prompt_orthogonal_schedule.codes[
                            self._step_index
                        ]
                    ),
                    flow_phase=flow_phase,
                    control_context=control_context,
                    injection_config=(
                        self.prompt_orthogonal_injection_config
                    ),
                    tubelet_config=self.tubelet_config,
                    velocity_config=self.velocity_config,
                )
            )
            velocity_record = {
                "velocity_field_constraint_status": injection_record.status,
                "velocity_field_source": (
                    "scheduler_model_output_before_flow_match_step"
                ),
                "flow_phase": round(float(flow_phase), 8),
                "flow_phase_weight": injection_record.flow_phase_weight,
                "velocity_constraint_lambda": (
                    self.prompt_orthogonal_injection_config.lambda_max
                    * injection_record.flow_phase_weight
                ),
                "velocity_norm_before_constraint": (
                    injection_record.velocity_norm_before
                ),
                "velocity_norm_after_constraint": (
                    injection_record.velocity_norm_after
                ),
                "velocity_constraint_delta_norm": (
                    injection_record.delta_norm
                ),
                "endpoint_control_enabled": False,
                "endpoint_control_formal_context_complete": False,
                "prompt_orthogonal_continuous_code": (
                    injection_record.continuous_code
                ),
                "prompt_orthogonal_joint_norm_budget": (
                    injection_record.joint_norm_budget
                ),
                "prompt_orthogonal_energy_limited_delta_norm": (
                    injection_record.energy_limited_delta_norm
                ),
                "prompt_orthogonal_joint_scale": (
                    injection_record.joint_scale
                ),
                "prompt_orthogonal_direction_cosine": (
                    injection_record.direction_cosine
                ),
                "prompt_orthogonal_norm_guard_passed": (
                    injection_record.norm_guard_passed
                ),
                "prompt_orthogonal_energy_guard_passed": (
                    injection_record.energy_guard_passed
                ),
                "prompt_orthogonal_direction_guard_passed": (
                    injection_record.direction_guard_passed
                ),
                "prompt_orthogonal_control_energy_increment": (
                    injection_record.control_energy_increment
                ),
                "prompt_orthogonal_control_cumulative_energy_after": (
                    injection_record.control_cumulative_energy_after
                ),
                "prompt_orthogonal_reference_energy_increment": (
                    injection_record.reference_energy_increment
                ),
                "prompt_orthogonal_reference_cumulative_energy_after": (
                    injection_record.reference_cumulative_energy_after
                ),
                "prompt_orthogonal_inactive_phase_noop": (
                    injection_record.inactive_phase_noop
                ),
                "endpoint_quality_energy_guard_passed": bool(
                    injection_record.inactive_phase_noop
                    or injection_record.energy_guard_passed is True
                ),
            }
            self._cumulative_control_energy = (
                injection_record.control_cumulative_energy_after
            )
            self._cumulative_reference_energy = (
                injection_record.reference_cumulative_energy_after
            )
        elif (
            self.mechanism_config.velocity_constraint_enabled
            and self.predictive_trajectory_carrier_config is not None
        ):
            if (
                control_context is None
                or self._predictive_trajectory_schedule is None
            ):
                raise RuntimeError(
                    "predictive trajectory runtime 缺少正式 schedule/control context"
                )
            constrained_velocity, velocity_record = (
                apply_predictive_trajectory_constraint(
                    model_output,
                    sample,
                    direction,
                    ac_code=self._predictive_trajectory_schedule.codes[
                        self._step_index
                    ],
                    flow_phase=flow_phase,
                    config=self.velocity_config,
                    tubelet_config=self.tubelet_config,
                    carrier_config=(
                        self.predictive_trajectory_carrier_config
                    ),
                    control_context=control_context,
                )
            )
            self._cumulative_control_energy = float(
                velocity_record[
                    "predictive_trajectory_control_cumulative_energy_after"
                ]
            )
            self._cumulative_reference_energy = float(
                velocity_record[
                    "predictive_trajectory_reference_cumulative_energy_after"
                ]
            )
        elif (
            self.mechanism_config.velocity_constraint_enabled
            and self.signed_trajectory_carrier_config is not None
        ):
            if control_context is None or self._signed_trajectory_schedule is None:
                raise RuntimeError(
                    "signed trajectory runtime 缺少正式 schedule/control context"
                )
            endpoint_direction = self.latent_layout.from_canonical(
                self._canonical_endpoint_key_direction
            ).to(device=sample.device, dtype=sample.dtype)
            constrained_velocity, velocity_record = (
                apply_signed_trajectory_two_channel_constraint(
                    model_output,
                    sample,
                    endpoint_direction,
                    ac_code=self._signed_trajectory_schedule.codes[
                        self._step_index
                    ],
                    flow_phase=flow_phase,
                    config=self.velocity_config,
                    tubelet_config=self.tubelet_config,
                    carrier_config=self.signed_trajectory_carrier_config,
                    control_context=control_context,
                )
            )
            if "endpoint_control_cumulative_energy_after" in velocity_record:
                self._cumulative_control_energy = float(
                    velocity_record[
                        "endpoint_control_cumulative_energy_after"
                    ]
                )
            if "endpoint_reference_cumulative_energy_after" in velocity_record:
                self._cumulative_reference_energy = float(
                    velocity_record[
                        "endpoint_reference_cumulative_energy_after"
                    ]
                )
        elif self.mechanism_config.velocity_constraint_enabled:
            constrained_velocity, velocity_record = apply_velocity_field_constraint(
                model_output,
                sample,
                direction,
                flow_phase=flow_phase,
                config=self.velocity_config,
                tubelet_config=self.tubelet_config,
                endpoint_control_enabled=self.mechanism_config.endpoint_control_enabled,
                control_context=control_context,
            )
            if "endpoint_control_cumulative_energy_after" in velocity_record:
                self._cumulative_control_energy = float(
                    velocity_record["endpoint_control_cumulative_energy_after"]
                )
            if "endpoint_reference_cumulative_energy_after" in velocity_record:
                self._cumulative_reference_energy = float(
                    velocity_record["endpoint_reference_cumulative_energy_after"]
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
                "endpoint_control_formal_context_complete": False,
                "endpoint_minimum_energy_control_status": "disabled_velocity_constraint_ablation",
                "endpoint_quality_energy_guard_passed": True,
            }

        if self.prompt_orthogonal_scheduler_float32_control:
            constrained_velocity = constrained_velocity.detach().float()
        step_output = self._original_step(
            constrained_velocity,
            timestep,
            sample,
            *args,
            **kwargs,
        )
        sample_after = self._first_sample(step_output)
        terminal_integration_weight = 0.0
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
            terminal_integration_weight = endpoint_delta_norm

        if (
            delta_sigma is not None
            and self.prompt_orthogonal_injection_config is None
        ):
            self._accumulate_integrated_direction(
                flow_phase=flow_phase,
                delta_sigma=delta_sigma,
                additional_weight=terminal_integration_weight,
                canonical_direction=(
                    self._canonical_endpoint_key_direction
                    if self.signed_trajectory_carrier_config is not None
                    else self._canonical_key_direction
                    if self.predictive_trajectory_carrier_config is not None
                    else None
                ),
            )

        path_record = compute_path_step_observation(
            sample,
            sample_after,
            constrained_velocity,
            direction,
            flow_phase=flow_phase,
            delta_sigma=delta_sigma,
        ).as_dict()
        timestep_value = float(timestep.detach().float().reshape(-1)[0].item()) if hasattr(timestep, "detach") else float(timestep)
        step_record = {
            "trajectory_step_index": self._step_index,
            "trajectory_timestep": timestep_value,
            "trajectory_delta_sigma": (
                None if delta_sigma is None else round(float(delta_sigma), 10)
            ),
            "flow_scheduler_runtime_verified": True,
            "velocity_constraint_enabled": (
                self.mechanism_config.velocity_constraint_enabled
            ),
            "terminal_endpoint_perturbation_enabled": (
                self.mechanism_config.terminal_endpoint_perturbation_enabled
            ),
            "prompt_orthogonal_scheduler_control_dtype": (
                PROMPT_ORTHOGONAL_SCHEDULER_CONTROL_DTYPE
                if self.prompt_orthogonal_scheduler_float32_control
                else "not_applicable"
            ),
            **self._key_metadata,
            **velocity_record,
            **path_record,
        }
        if self.prompt_orthogonal_injection_config is not None:
            carrier_context_complete = bool(
                step_record.get("prompt_orthogonal_inactive_phase_noop")
                is True
                or (
                    step_record.get(
                        "prompt_orthogonal_norm_guard_passed"
                    )
                    is True
                    and step_record.get(
                        "prompt_orthogonal_energy_guard_passed"
                    )
                    is True
                    and step_record.get(
                        "prompt_orthogonal_direction_guard_passed"
                    )
                    is True
                )
            )
        elif self.predictive_trajectory_carrier_config is not None:
            carrier_context_complete = bool(
                step_record.get("predictive_trajectory_inactive_phase_noop")
                is True
                or (
                    step_record.get(
                        "predictive_trajectory_norm_guard_passed"
                    )
                    is True
                    and step_record.get(
                        "predictive_trajectory_energy_guard_passed"
                    )
                    is True
                )
            )
        else:
            carrier_context_complete = bool(
                self.mechanism_config.endpoint_control_enabled
                and (
                    (
                        self.signed_trajectory_carrier_config is not None
                        and step_record.get(
                            "signed_trajectory_inactive_phase_noop"
                        )
                        is True
                        and step_record.get(
                            "signed_trajectory_inactive_phase_noop_context_complete"
                        )
                        is True
                    )
                    or step_record.get(
                        "endpoint_control_formal_context_complete"
                    )
                    is True
                    or (
                        self.signed_trajectory_carrier_config is not None
                        and step_record.get(
                            "signed_trajectory_joint_energy_guard_passed"
                        )
                        is True
                        and step_record.get(
                            "signed_trajectory_joint_norm_guard_passed"
                        )
                        is True
                        and step_record.get(
                            "signed_trajectory_ac_direction_guard_passed"
                        )
                        is True
                    )
                )
            )
        step_record["flow_runtime_step_formal_context_complete"] = bool(
            self.key_context is not None
            and delta_sigma is not None
            and (
                self.prompt_orthogonal_injection_config is not None
                or step_record.get("flow_tubelet_formal_context_complete")
                is True
            )
            and step_record.get("path_quadrature_context_complete") is True
            and self.mechanism_config.velocity_constraint_enabled
            and carrier_context_complete
        )
        self._step_records.append(step_record)
        self._endpoint_latent = sample_after.detach().clone()
        self._step_index += 1
        return step_output
