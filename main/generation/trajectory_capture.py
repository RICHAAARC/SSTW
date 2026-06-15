"""记录生成 trajectory 捕获状态。"""

from __future__ import annotations


def trajectory_capture_record(runnable_status: str) -> dict:
    """根据模型可运行性给出 trajectory 捕获状态。"""
    if runnable_status != "runnable":
        return {"trajectory_capture_status": "not_run", "trajectory_capture_failure_reason": "generation_model_not_runnable"}
    return {"trajectory_capture_status": "pending_runtime", "trajectory_capture_failure_reason": "none"}
