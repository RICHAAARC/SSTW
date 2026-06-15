"""提供 trajectory-state adapter 的轻量接口。"""

from __future__ import annotations


def trajectory_state_adapter_status(trajectory_enabled: bool, trajectory_source_status: str) -> str:
    """返回 trajectory-state adapter 状态。"""
    return "pass" if trajectory_enabled and trajectory_source_status in {"valid", "approximate", "surrogate"} else "disabled_or_unavailable"
