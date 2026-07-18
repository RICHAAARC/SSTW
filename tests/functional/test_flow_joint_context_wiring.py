from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="requires optional method-runtime dependency")

from main.methods.state_space_watermark.endpoint_latent_detector import (
    compute_endpoint_latent_evidence,
)
from main.methods.state_space_watermark.flow_latent_layout import (
    PackedTokenFlowLatentLayout,
)
from main.methods.state_space_watermark.flow_tubelet_key_code import (
    FlowTubeletKeyContext,
    build_integrated_flow_tubelet_key_direction_like,
)
from main.methods.state_space_watermark.flow_velocity_runtime import (
    FlowVelocityConstraintRuntime,
    normalized_flow_phase_from_sigma_interval,
)
from main.methods.state_space_watermark.ltx_flow_replay_backend import (
    LTXKeyConditionedVelocity,
)
from main.methods.state_space_watermark.replay_inversion import FlowSchedulePoint
from main.methods.state_space_watermark.wan_flow_replay_backend import (
    WanKeyConditionedVelocity,
)


class FormalFlowMatchFakeScheduler:
    """用真实 sigma 差执行 Euler 更新，隔离测试 runtime 接线。"""

    def __init__(self) -> None:
        self.sigmas = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0])
        self._index = 0

    def step(self, model_output, timestep, sample, *args, **kwargs):
        del timestep, args, kwargs
        delta_sigma = self.sigmas[self._index + 1] - self.sigmas[self._index]
        self._index += 1
        return (sample + delta_sigma * model_output,)


class ConstantVelocity:
    """提供无需模型权重的确定性 prompt-conditioned velocity。"""

    def __init__(self, *, latent_layout=None) -> None:
        self.latent_layout = latent_layout

    def __call__(self, latent, timestep, step_index):
        del timestep, step_index
        return torch.ones_like(latent)


def _context() -> FlowTubeletKeyContext:
    return FlowTubeletKeyContext(
        prompt_digest="a" * 64,
        sampler_signature="FlowMatchEulerDiscreteScheduler:test-grid-v1",
    )


def _schedule() -> tuple[FlowSchedulePoint, ...]:
    return tuple(
        FlowSchedulePoint(timestep=float(4 - index), sigma=1.0 - index * 0.25)
        for index in range(5)
    )


@pytest.mark.quick
def test_sigma_interval_phase_uses_continuous_grid_midpoint() -> None:
    """规范 phase 必须由 sigma 区间决定，而不是依赖离散 step 编号。"""

    coarse = [1.0, 0.5, 0.0]
    refined = [1.0, 0.75, 0.5, 0.25, 0.0]

    assert normalized_flow_phase_from_sigma_interval(coarse, 0) == pytest.approx(0.25)
    assert normalized_flow_phase_from_sigma_interval(refined, 1) == pytest.approx(0.375)
    assert normalized_flow_phase_from_sigma_interval(refined, 2) == pytest.approx(0.625)


@pytest.mark.quick
def test_formal_runtime_binds_joint_context_real_delta_and_integrated_endpoint() -> None:
    """生成 runtime 必须闭合 phase code、真实 delta-sigma、能量状态和 endpoint 积分。"""

    scheduler = FormalFlowMatchFakeScheduler()
    context = _context()
    sample = torch.zeros((1, 2, 4, 8, 8), dtype=torch.float32)
    model_output = torch.ones_like(sample)

    with FlowVelocityConstraintRuntime(
        scheduler,
        key_text="owner-key",
        total_steps=4,
        key_context=context,
    ) as runtime:
        current = sample
        for step_index in range(4):
            current = scheduler.step(
                model_output,
                torch.tensor(float(4 - step_index)),
                current,
                return_dict=False,
            )[0]

    records = runtime.step_records
    assert len(records) == 4
    assert all(record["trajectory_delta_sigma"] == pytest.approx(-0.25) for record in records)
    assert all(record["path_quadrature_context_complete"] is True for record in records)
    assert all(record["flow_tubelet_formal_context_complete"] is True for record in records)
    assert all(record["endpoint_control_formal_context_complete"] is True for record in records)
    assert len({record["flow_key_direction_digest"] for record in records}) == 4
    assert runtime.key_metadata["flow_runtime_formal_context_complete"] is True
    assert len(runtime.key_metadata["flow_tubelet_key_context_digest"]) == 64
    assert runtime.flow_phases == pytest.approx((0.375, 0.625))
    assert runtime.canonical_integrated_key_direction is not None

    rebuilt, rebuilt_metadata = build_integrated_flow_tubelet_key_direction_like(
        sample,
        key_text="owner-key",
        key_context=context,
        flow_phases=runtime.flow_phases,
        integration_weights=runtime.integration_weights,
    )
    assert torch.allclose(runtime.canonical_integrated_key_direction, rebuilt, atol=1e-6)
    assert (
        runtime.key_metadata["flow_integrated_key_direction_digest"]
        == rebuilt_metadata["flow_key_direction_digest"]
    )

    endpoint = rebuilt * 5.0
    evidence = compute_endpoint_latent_evidence(
        endpoint,
        key_text="owner-key",
        key_context=context,
        flow_phases=runtime.flow_phases,
        integration_weights=runtime.integration_weights,
    )
    assert evidence.formal_context_complete is True
    assert evidence.projection > 0.999


@pytest.mark.quick
def test_old_runtime_and_endpoint_api_remain_diagnostic_only() -> None:
    """缺少 prompt/sampler/payload 上下文的旧 API 不得支持正式 claim。"""

    reference = torch.zeros((1, 1, 2, 4, 4), dtype=torch.float32)
    evidence = compute_endpoint_latent_evidence(reference + 0.1, key_text="legacy-key")

    assert evidence.formal_context_complete is False
    assert evidence.as_dict()["endpoint_formal_context_complete"] is False


@pytest.mark.quick
def test_wan_candidate_replay_reuses_schedule_phase_and_energy_context() -> None:
    """Wan candidate hypothesis 必须逐步复现 generation 的 joint code 和控制状态。"""

    schedule = _schedule()
    velocity = WanKeyConditionedVelocity(
        ConstantVelocity(),
        key_text="owner-key",
        total_steps=len(schedule),
        key_context=_context(),
        schedule=schedule,
    )
    state = torch.zeros((1, 2, 4, 8, 8), dtype=torch.float32)

    for step_index in range(len(schedule) - 1):
        velocity(state, schedule[step_index].timestep, step_index)

    assert len(velocity.step_records) == 4
    assert all(record["replay_joint_context_complete"] is True for record in velocity.step_records)
    assert len({record["flow_key_direction_digest"] for record in velocity.step_records}) == 4
    assert velocity.step_records[-1]["endpoint_reference_cumulative_energy_after"] > 0.0


@pytest.mark.quick
def test_ltx_candidate_replay_uses_same_joint_code_in_packed_layout() -> None:
    """LTX packed token 路径必须复用同一个五维 phase joint code。"""

    layout = PackedTokenFlowLatentLayout(
        num_frames=2,
        height=4,
        width=4,
        spatial_patch_size=1,
        temporal_patch_size=1,
    )
    canonical = torch.zeros((1, 3, 2, 4, 4), dtype=torch.float32)
    packed = layout.from_canonical(canonical)
    schedule = _schedule()
    velocity = LTXKeyConditionedVelocity(
        ConstantVelocity(latent_layout=layout),
        key_text="owner-key",
        total_steps=len(schedule),
        key_context=_context(),
        schedule=schedule,
    )

    for step_index in range(len(schedule) - 1):
        velocity(packed, schedule[step_index].timestep, step_index)

    assert len(velocity.step_records) == 4
    assert all(record["replay_joint_context_complete"] is True for record in velocity.step_records)
    assert all(record["flow_tubelet_formal_context_complete"] is True for record in velocity.step_records)
