"""从真实 Flow 状态更新构造时间重参数化不变路径证据。"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class PathStepObservation:
    """保存单个 Flow 更新步的速度与路径观测。"""

    flow_phase: float
    path_projection: float
    path_projection_normalized: float
    velocity_projection_normalized: float
    path_step_norm: float
    path_velocity_consistency: float

    def as_dict(self) -> dict[str, float]:
        """转换为可写入 trajectory records 的字段。"""

        return {
            "flow_phase": round(self.flow_phase, 8),
            "path_projection": round(self.path_projection, 8),
            "path_projection_normalized": round(self.path_projection_normalized, 8),
            "velocity_projection_normalized": round(self.velocity_projection_normalized, 8),
            "path_step_norm": round(self.path_step_norm, 8),
            "path_velocity_consistency": round(self.path_velocity_consistency, 8),
        }


def _normalized_projection(value: Any, direction: Any) -> tuple[float, float]:
    """返回原始投影和按状态增量范数归一化后的投影。"""

    flat = value.detach().float().reshape(-1)
    direction_flat = direction.detach().float().reshape(-1)
    raw = float((flat @ direction_flat).item())
    norm = float(flat.norm().item())
    return raw, raw / max(norm, 1e-8)


def compute_path_step_observation(
    sample_before: Any,
    sample_after: Any,
    constrained_velocity: Any,
    key_direction: Any,
    *,
    flow_phase: float,
) -> PathStepObservation:
    """从 scheduler 前后 latent 和真实模型输出计算路径观测。"""

    displacement = sample_after - sample_before
    path_raw, path_normalized = _normalized_projection(displacement, key_direction)
    # FlowMatchEulerDiscreteScheduler 沿递减 sigma 网格更新, 因而状态位移方向与
    # model velocity 的符号相反。将 velocity 翻转到实际路径方向后, endpoint、
    # path 和 velocity 三类证据才具有统一的“越大越像水印”方向。
    path_oriented_velocity = -constrained_velocity
    _velocity_raw, velocity_normalized = _normalized_projection(path_oriented_velocity, key_direction)
    displacement_flat = displacement.detach().float().reshape(-1)
    velocity_flat = path_oriented_velocity.detach().float().reshape(-1)
    cosine = float((
        displacement_flat @ velocity_flat
        / (displacement_flat.norm().clamp_min(1e-8) * velocity_flat.norm().clamp_min(1e-8))
    ).item())
    consistency = max(0.0, min(1.0, 0.5 + 0.5 * cosine))
    return PathStepObservation(
        flow_phase=float(flow_phase),
        path_projection=path_raw,
        path_projection_normalized=path_normalized,
        velocity_projection_normalized=velocity_normalized,
        path_step_norm=float(displacement.detach().float().norm().item()),
        path_velocity_consistency=consistency,
    )


def aggregate_path_observations(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, float | int | bool | None]:
    """聚合 trajectory step records, 形成不确定性感知的路径证据。

    生成阶段的 step record 没有 replay 不确定性, 因而默认可靠性为1。攻击后
    replay 可以逐步提供 ``replay_reliability_weight``; 该权重直接衰减每一步
    对路径积分的贡献, 而不是只作为检测器的旁路特征。
    """

    rows = [record for record in records if record.get("path_projection_normalized") is not None]
    if not rows:
        return {
            "path_observation_step_count": 0,
            "S_path_inv": None,
            "S_velocity": None,
            "path_velocity_consistency_mean": None,
            "S_path_inv_unweighted": None,
            "path_replay_reliability_weight_mean": None,
            "path_replay_weighted_aggregation_applied": False,
        }
    phase_weights = [
        max(0.0, 1.0 - abs(float(row.get("flow_phase", 0.5)) - 0.5) * 2.0)
        for row in rows
    ]
    phase_weight_sum = sum(phase_weights)
    if phase_weight_sum <= 1e-8:
        phase_weights = [1.0 for _ in rows]
        phase_weight_sum = float(len(rows))
    replay_weights = [
        max(0.0, min(1.0, float(row.get("replay_reliability_weight", 1.0))))
        for row in rows
    ]
    weighted_path_weights = [
        phase_weight * replay_weight
        for phase_weight, replay_weight in zip(phase_weights, replay_weights)
    ]
    unweighted_path_score = sum(
        weight * float(row["path_projection_normalized"])
        for weight, row in zip(phase_weights, rows)
    ) / phase_weight_sum
    path_score = sum(
        weight * float(row["path_projection_normalized"])
        for weight, row in zip(weighted_path_weights, rows)
    ) / phase_weight_sum
    velocity_score = sum(
        weight * float(row.get("velocity_projection_normalized") or 0.0)
        for weight, row in zip(phase_weights, rows)
    ) / phase_weight_sum
    consistency_score = sum(
        weight * float(row.get("path_velocity_consistency") or 0.0)
        for weight, row in zip(weighted_path_weights, rows)
    ) / phase_weight_sum
    return {
        "path_observation_step_count": len(rows),
        "S_path_inv": round(path_score, 8),
        "S_path_inv_unweighted": round(unweighted_path_score, 8),
        "S_velocity": round(velocity_score, 8),
        "path_velocity_consistency_mean": round(consistency_score, 8),
        "path_replay_reliability_weight_mean": round(mean(replay_weights), 8),
        "path_replay_weighted_aggregation_applied": any(
            "replay_reliability_weight" in row for row in rows
        ),
    }
