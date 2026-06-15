"""封装生成式视频模型的可运行性检查。"""

from __future__ import annotations

import shutil


def detect_gpu_status() -> dict:
    """检查本地是否存在 nvidia-smi, 该检查只用于决定是否可运行真实 GPU 生成链路。"""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"gpu_validation_status": "not_run", "gpu_validation_reason": "nvidia_smi_not_found"}
    return {"gpu_validation_status": "available", "gpu_validation_reason": "nvidia_smi_found"}


def generation_runnable_status(gpu_status: dict, requires_gpu: bool = True) -> dict:
    """根据 GPU 状态给出生成模型运行状态, 不把未运行伪装为成功。"""
    if requires_gpu and gpu_status.get("gpu_validation_status") != "available":
        return {"generation_model_runnable_status": "not_runnable", "generation_model_not_run_reason": gpu_status.get("gpu_validation_reason", "gpu_not_available")}
    return {"generation_model_runnable_status": "runnable", "generation_model_not_run_reason": "none"}
