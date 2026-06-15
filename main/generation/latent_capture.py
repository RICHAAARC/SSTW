"""记录生成模型 latent 捕获状态。"""

from __future__ import annotations


def latent_capture_record(runnable_status: str) -> dict:
    """根据模型可运行性给出 latent 捕获状态。"""
    if runnable_status != "runnable":
        return {"latent_capture_status": "not_run", "latent_capture_failure_reason": "generation_model_not_runnable"}
    return {"latent_capture_status": "pending_runtime", "latent_capture_failure_reason": "none"}
