"""定义 B4 trajectory trace 的轻量结构。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrajectoryTrace:
    """表示一个可审计的轨迹 trace 元数据。"""

    trajectory_trace_id: str
    trajectory_source: str
    trajectory_source_status: str
    trajectory_time_grid_id: str
    trajectory_num_steps: int
    trajectory_time_points: tuple[float, ...]
    velocity_estimator_id: str
    velocity_projection_operator_id: str
    trajectory_runtime_sec: float


def build_trajectory_trace(sample_id: str, config: dict) -> TrajectoryTrace:
    """根据样本 ID 和配置构建轨迹 trace。"""
    return TrajectoryTrace(
        trajectory_trace_id=f"trajectory_trace_{sample_id}",
        trajectory_source=config["trajectory_source"],
        trajectory_source_status=config["trajectory_source_status"],
        trajectory_time_grid_id=config["trajectory_time_grid_id"],
        trajectory_num_steps=int(config["trajectory_num_steps"]),
        trajectory_time_points=tuple(float(value) for value in config["trajectory_time_points"]),
        velocity_estimator_id=config["velocity_estimator_id"],
        velocity_projection_operator_id=config["velocity_projection_operator_id"],
        trajectory_runtime_sec=0.002,
    )
