"""B5 生成运行状态构建。"""

from __future__ import annotations

from main.backends.generative_video_backend import backend_status
from main.generation.latent_capture import latent_capture_record
from main.generation.trajectory_capture import trajectory_capture_record
from main.generation.video_generator import detect_gpu_status, generation_runnable_status


def build_generation_runtime_status(requires_gpu: bool = True) -> dict:
    """汇总 GPU、生成模型、latent 和 trajectory 捕获状态。"""
    gpu = detect_gpu_status()
    runnable = generation_runnable_status(gpu, requires_gpu=requires_gpu)
    return {**gpu, **runnable, **backend_status(runnable["generation_model_runnable_status"]), **latent_capture_record(runnable["generation_model_runnable_status"]), **trajectory_capture_record(runnable["generation_model_runnable_status"])}
