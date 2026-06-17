"""Wan2.1 Flow adapter preflight Colab Notebook 的路径和命令编排工具。"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
import subprocess
import sys


DEFAULT_DRIVE_PROJECT_ROOT = "/content/drive/MyDrive/SSTW"
DEFAULT_WAN21_PREFLIGHT_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


def build_drive_layout(drive_project_root: str = DEFAULT_DRIVE_PROJECT_ROOT) -> dict[str, str]:
    """构造真实 Wan2.1 GPU preflight 在 Google Drive 中的输出目录布局。

    该函数属于通用工程写法。Notebook 只负责设置路径和调用仓库模块,
    正式 records、artifacts 和 reports 必须由 `experiments/` 中的代码写出。
    """
    root = PurePosixPath(drive_project_root)
    return {
        "drive_project_root": root.as_posix(),
        "drive_run_root": (root / "runs" / "wan21_flow_adapter_preflight").as_posix(),
        "drive_log_dir": (root / "logs" / "wan21_flow_adapter_preflight").as_posix(),
    }


def ensure_drive_layout(drive_project_root: str = DEFAULT_DRIVE_PROJECT_ROOT) -> dict[str, str]:
    """创建真实 Wan2.1 GPU preflight 的 Google Drive 目录。"""
    layout = build_drive_layout(drive_project_root)
    for key, value in layout.items():
        if key.endswith("_dir") or key.endswith("_root"):
            Path(value).mkdir(parents=True, exist_ok=True)
    return layout


def build_wan21_flow_adapter_preflight_command(
    layout: dict[str, str],
    model_id: str = DEFAULT_WAN21_PREFLIGHT_MODEL_ID,
    num_inference_steps: int = 4,
    num_frames: int = 33,
    height: int = 320,
    width: int = 512,
) -> list[str]:
    """构造真实 Wan2.1 GPU preflight 命令。

    该命令只检查 adapter 能力, 不执行 sampling-time constraint 小实验,
    也不生成 full generative video probe 的正式结论。
    """
    return [
        sys.executable,
        "-m",
        "experiments.flow_model_adapter_preflight.wan21_preflight",
        "--output-root",
        layout["drive_run_root"],
        "--model-id",
        model_id,
        "--num-inference-steps",
        str(num_inference_steps),
        "--num-frames",
        str(num_frames),
        "--height",
        str(height),
        "--width",
        str(width),
    ]


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    """执行 Notebook 编排命令, 正式输出由仓库模块负责生成。"""
    return subprocess.run(command, text=True, capture_output=True)
