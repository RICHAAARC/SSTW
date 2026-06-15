"""提供 B4 trajectory replay / reconstruction 接口。"""

from __future__ import annotations


def reconstruct_trajectory_status(trajectory_source: str) -> str:
    """返回轨迹重建状态。"""
    return "pass" if trajectory_source in {"recorded_sampling_trace", "approximate_inversion_trace", "latent_replay_trace", "synthetic_surrogate_trace"} else "unavailable"
