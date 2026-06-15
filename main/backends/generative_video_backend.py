"""生成式视频 backend 的轻量状态接口。"""

from __future__ import annotations


def backend_status(generation_model_runnable_status: str) -> dict:
    """返回 backend 是否可以执行真实视频生成。"""
    if generation_model_runnable_status != "runnable":
        return {"generation_backend_status": "blocked", "generation_backend_reason": "generation_model_not_runnable"}
    return {"generation_backend_status": "ready", "generation_backend_reason": "none"}
