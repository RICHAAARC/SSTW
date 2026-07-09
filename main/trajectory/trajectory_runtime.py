"""记录 trajectory_observation_core_probe trajectory runtime 开销。"""

from __future__ import annotations


def runtime_overhead_status(runtime_sec: float, blocking_sec: float) -> str:
    """判断 trajectory runtime 是否阻断。"""
    return "PASS" if runtime_sec < blocking_sec else "BLOCKING"
