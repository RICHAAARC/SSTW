"""Wan2.1 Flow adapter preflight Colab Notebook 的路径和命令编排工具。"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
import subprocess
import sys

from workflows.stage_package_sync import stage_zip_handoff_enabled
from workflows.streaming_command import run_streaming_command


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
        "drive_package_dir": (root / "helper").as_posix(),
        "drive_log_dir": (root / "logs" / "wan21_flow_adapter_preflight").as_posix(),
        "workflow_profile": "wan21_flow_adapter_preflight",
        "runtime_profile": "wan21_flow_adapter_preflight",
        "result_tier": "preflight",
        "notebook_role": "wan21_flow_adapter_preflight",
    }


def ensure_drive_layout(drive_project_root: str = DEFAULT_DRIVE_PROJECT_ROOT) -> dict[str, str]:
    """创建真实 Wan2.1 GPU preflight 的 Google Drive 目录。"""
    layout = build_drive_layout(drive_project_root)
    if stage_zip_handoff_enabled():
        # local_zip 模式下, helper Notebook 的中间结果只写入 /content 本地 workspace。
        # Drive 上只保留最终 helper 阶段 zip, 避免产生空的 runs / logs 目录。
        Path(layout["drive_project_root"]).mkdir(parents=True, exist_ok=True)
        return layout
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


def _stage_zip_mode_uses_unified_package(layout: dict[str, str]) -> bool:
    """判断是否跳过历史 drive packager, 避免 Drive 重复归档。"""

    mode = str(layout.get("stage_package_handoff_mode") or os.environ.get("SSTW_COLAB_STAGE_IO_MODE", ""))
    return mode.strip().lower() in {"local_zip", "stage_zip", "zip_handoff"}


def _build_legacy_drive_packaging_noop_command(packager_name: str) -> list[str]:
    """构造可执行 no-op 命令, 由阶段 zip 发布逻辑承担实际落盘。"""

    message = (
        "SSTW stage zip handoff is active; skip legacy drive packager "
        f"{packager_name}. Unified output is written by publish_colab_stage_package."
    )
    return [sys.executable, "-c", f"print({message!r})"]


def build_drive_packaging_command(layout: dict[str, str]) -> list[str]:
    """构造真实 Wan2.1 GPU preflight 的 Google Drive 打包命令。

    该命令只打包已生成的 governed outputs, 不补写实验结果。
    """
    if _stage_zip_mode_uses_unified_package(layout):
        return _build_legacy_drive_packaging_noop_command("wan21_flow_adapter_preflight_drive_packager.py")

    return [
        sys.executable,
        "scripts/package_results/wan21_flow_adapter_preflight_drive_packager.py",
        "--run-root",
        layout["drive_run_root"],
        "--drive-package-dir",
        layout["drive_package_dir"],
    ]


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    """执行 Notebook 编排命令, 并实时显示 repository runner 进度。"""
    return run_streaming_command(command)
