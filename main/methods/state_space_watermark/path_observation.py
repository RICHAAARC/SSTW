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
    path_sigma_measure: float | None = None
    path_arc_length: float = 0.0
    path_projection_integrand: float | None = None
    path_quadrature_contribution: float | None = None
    path_quadrature_context_complete: bool = False

    def as_dict(self) -> dict[str, float | bool | str | None]:
        """转换为可写入 trajectory records 的字段。"""

        return {
            "flow_phase": round(self.flow_phase, 8),
            "path_projection": round(self.path_projection, 8),
            "path_projection_normalized": round(self.path_projection_normalized, 8),
            "velocity_projection_normalized": round(self.velocity_projection_normalized, 8),
            "path_step_norm": round(self.path_step_norm, 8),
            "path_velocity_consistency": round(self.path_velocity_consistency, 8),
            "path_sigma_measure": (
                None
                if self.path_sigma_measure is None
                else round(self.path_sigma_measure, 10)
            ),
            "path_arc_length": round(self.path_arc_length, 10),
            "path_projection_integrand": (
                None
                if self.path_projection_integrand is None
                else round(self.path_projection_integrand, 10)
            ),
            "path_quadrature_contribution": (
                None
                if self.path_quadrature_contribution is None
                else round(self.path_quadrature_contribution, 10)
            ),
            "path_quadrature_context_complete": self.path_quadrature_context_complete,
            "path_quadrature_rule": (
                "delta_sigma_arc_length_discrete_line_integral"
                if self.path_quadrature_context_complete
                else "compatibility_normalized_step_observation"
            ),
        }


def _normalized_projection(value: Any, direction: Any) -> tuple[float, float]:
    """返回原始投影和按状态增量范数归一化后的投影。"""

    flat = value.detach().float().reshape(-1)
    direction_flat = direction.detach().float().reshape(-1)
    direction_unit = direction_flat / direction_flat.norm().clamp_min(1e-8)
    raw = float((flat @ direction_unit).item())
    norm = float(flat.norm().item())
    return raw, raw / max(norm, 1e-8)


def compute_path_step_observation(
    sample_before: Any,
    sample_after: Any,
    constrained_velocity: Any,
    key_direction: Any,
    *,
    flow_phase: float,
    delta_sigma: float | None = None,
) -> PathStepObservation:
    """从 scheduler 前后 latent 和真实模型输出计算路径观测。

    正式路径应显式传入相邻 scheduler 点的 ``delta_sigma``。单步投影先除以
    ``|delta_sigma|`` 得到积分密度，再乘同一测度形成求积贡献；聚合时以真实
    弧长归一化。因此把一个区间细分成多个同路径子区间不会改变路径统计量。
    """

    displacement = sample_after - sample_before
    path_raw, path_normalized = _normalized_projection(displacement, key_direction)
    path_step_norm = float(displacement.detach().float().norm().item())
    sigma_measure: float | None = None
    projection_integrand: float | None = None
    quadrature_contribution: float | None = None
    quadrature_complete = delta_sigma is not None
    if delta_sigma is not None:
        sigma_measure = abs(float(delta_sigma))
        if sigma_measure <= 1e-12:
            raise ValueError("正式路径求积要求非零 delta_sigma")
        projection_integrand = path_raw / sigma_measure
        quadrature_contribution = projection_integrand * sigma_measure
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
        path_step_norm=path_step_norm,
        path_velocity_consistency=consistency,
        path_sigma_measure=sigma_measure,
        path_arc_length=path_step_norm,
        path_projection_integrand=projection_integrand,
        path_quadrature_contribution=quadrature_contribution,
        path_quadrature_context_complete=quadrature_complete,
    )


def _phase_quadrature_weight(flow_phase: float) -> float:
    """返回定义在规范 Flow phase 上的连续三角窗求积权重。"""

    phase = max(0.0, min(1.0, float(flow_phase)))
    return max(0.0, 1.0 - abs(phase - 0.5) * 2.0)


def _aggregate_quadrature_path_observations(
    rows: list[Mapping[str, Any]],
    replay_weights: list[float],
) -> dict[str, float | int | bool | str | None]:
    """使用 delta-sigma 求积贡献与弧长测度聚合完整路径观测。"""

    phase_weights = [_phase_quadrature_weight(float(row["flow_phase"])) for row in rows]
    if sum(phase_weights) <= 1e-12:
        phase_weights = [1.0 for _ in rows]
    arc_lengths = [max(0.0, float(row["path_arc_length"])) for row in rows]
    if sum(arc_lengths) <= 1e-12:
        raise ValueError("正式路径求积的总弧长必须为正数")
    sigma_measures = [float(row["path_sigma_measure"]) for row in rows]
    if any(value <= 0.0 for value in sigma_measures):
        raise ValueError("正式路径求积的 sigma 测度必须为正数")
    contributions = [float(row["path_quadrature_contribution"]) for row in rows]

    path_measure = sum(
        phase_weight * arc_length
        for phase_weight, arc_length in zip(phase_weights, arc_lengths)
    )
    if path_measure <= 1e-12:
        # 若全部有效弧长恰好位于三角窗端点，则改用完整弧长测度，不改变路径方向。
        phase_weights = [1.0 for _ in rows]
        path_measure = sum(arc_lengths)
    unweighted_path_score = sum(
        phase_weight * contribution
        for phase_weight, contribution in zip(phase_weights, contributions)
    ) / path_measure
    path_score = sum(
        phase_weight * replay_weight * contribution
        for phase_weight, replay_weight, contribution in zip(
            phase_weights,
            replay_weights,
            contributions,
        )
    ) / path_measure

    sigma_measure = sum(
        phase_weight * value
        for phase_weight, value in zip(phase_weights, sigma_measures)
    )
    velocity_score = sum(
        phase_weight
        * sigma_value
        * float(row.get("velocity_projection_normalized") or 0.0)
        for phase_weight, sigma_value, row in zip(
            phase_weights,
            sigma_measures,
            rows,
        )
    ) / max(sigma_measure, 1e-12)
    consistency_score = sum(
        phase_weight
        * arc_length
        * replay_weight
        * float(row.get("path_velocity_consistency") or 0.0)
        for phase_weight, arc_length, replay_weight, row in zip(
            phase_weights,
            arc_lengths,
            replay_weights,
            rows,
        )
    ) / path_measure
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
        "path_quadrature_context_complete": True,
        "path_quadrature_rule": "delta_sigma_arc_length_discrete_line_integral",
        "path_total_sigma_measure": round(sum(sigma_measures), 10),
        "path_total_arc_length": round(sum(arc_lengths), 10),
        "path_reparameterization_stability_semantics": (
            "stable_under_partition_refinement_of_the_same_piecewise_linear_path"
        ),
    }


def aggregate_path_observations(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, float | int | bool | str | None]:
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
            "path_quadrature_context_complete": False,
            "path_quadrature_rule": "missing_path_observations",
            "path_total_sigma_measure": None,
            "path_total_arc_length": None,
            "path_reparameterization_stability_semantics": "not_evaluable",
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
    quadrature_ready = all(
        row.get("path_quadrature_context_complete") is True
        and row.get("path_sigma_measure") is not None
        and row.get("path_arc_length") is not None
        and row.get("path_quadrature_contribution") is not None
        for row in rows
    )
    if quadrature_ready:
        return _aggregate_quadrature_path_observations(rows, replay_weights)
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
        "path_quadrature_context_complete": False,
        "path_quadrature_rule": "compatibility_normalized_step_average",
        "path_total_sigma_measure": None,
        "path_total_arc_length": round(
            sum(max(0.0, float(row.get("path_step_norm") or 0.0)) for row in rows),
            10,
        ),
        "path_reparameterization_stability_semantics": "not_formal_without_delta_sigma",
    }
