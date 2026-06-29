"""B6 sampling-time constraint Colab Notebook 的路径和命令编排工具。"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
import subprocess
import sys

from paper_workflow.notebook_utils.streaming_command import run_streaming_command

DEFAULT_DRIVE_PROJECT_ROOT = "/content/drive/MyDrive/SSTW"
DEFAULT_SSTW_TC_PRIMARY_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


def build_drive_layout(drive_project_root: str = DEFAULT_DRIVE_PROJECT_ROOT) -> dict[str, str]:
    """构造 B6 Colab 在 Google Drive 中的输出目录布局。"""
    root = PurePosixPath(drive_project_root)
    return {
        "drive_project_root": root.as_posix(),
        "drive_dataset_root": (root / "datasets" / "generative_video_prompt_suite").as_posix(),
        "drive_run_root": (root / "runs" / "sampling_time_constraint_colab").as_posix(),
        "drive_package_dir": (root / "packages" / "sampling_time_constraint").as_posix(),
        "drive_log_dir": (root / "logs" / "sampling_time_constraint").as_posix(),
        "prompt_suite_path": (root / "datasets" / "generative_video_prompt_suite" / "prompt_seed_suite.json").as_posix(),
        "workflow_profile": "sampling_time_constraint",
        "runtime_profile": "recommended",
        "result_tier": "sampling_time_constraint_colab_probe",
        "notebook_role": "sampling_time_constraint",
    }


def ensure_drive_layout(drive_project_root: str = DEFAULT_DRIVE_PROJECT_ROOT) -> dict[str, str]:
    """创建 B6 Colab 目标目录并返回路径布局。"""
    layout = build_drive_layout(drive_project_root)
    for key, value in layout.items():
        if key.endswith("_dir") or key.endswith("_root"):
            Path(value).mkdir(parents=True, exist_ok=True)
    return layout


def build_prompt_suite_command(layout: dict[str, str]) -> list[str]:
    """构造 prompt suite 数据集命令, 该步骤不执行 GPU 测试。"""
    return [sys.executable, "scripts/prepare_generative_video_prompt_suite.py", "--output-root", layout["drive_dataset_root"]]


def build_sampling_constraint_colab_runtime_command(
    layout: dict[str, str],
    profile: str,
    model_id: str = DEFAULT_SSTW_TC_PRIMARY_MODEL_ID,
) -> list[str]:
    """构造 B6 Colab GPU runtime 命令。"""
    return [
        sys.executable,
        "-m",
        "experiments.sampling_time_constraint.colab_runtime",
        "--output-root",
        layout["drive_run_root"],
        "--prompt-suite-path",
        layout["prompt_suite_path"],
        "--profile",
        profile,
        "--model-id",
        model_id,
        "--constraint-config-path",
        "configs/generation/sampling_constraint.json",
        "--lambda-schedules-path",
        "configs/generation/lambda_schedules.json",
    ]


def build_formal_metric_command(
    layout: dict[str, str],
    semantic_model_id: str = "openai/clip-vit-base-patch32",
    semantic_frame_limit: int = 8,
) -> list[str]:
    """构造 B6 真实视频质量、运动和语义 metric 命令。"""
    return [
        sys.executable,
        "-m",
        "experiments.generative_video_model_probe.formal_metric_runner",
        "--run-root",
        layout["drive_run_root"],
        "--prompt-suite-path",
        layout["prompt_suite_path"],
        "--semantic-model-id",
        semantic_model_id,
        "--semantic-frame-limit",
        str(semantic_frame_limit),
    ]


def build_postprocess_command(layout: dict[str, str]) -> list[str]:
    """构造 B6 Colab 后处理命令。"""
    return [
        sys.executable,
        "-m",
        "experiments.sampling_time_constraint.postprocess_runner",
        "--run-root",
        layout["drive_run_root"],
    ]


def build_result_check_command(layout: dict[str, str]) -> list[str]:
    """构造 B6 Colab 结果检查命令, 用于在打包前显式核对证据状态。"""
    return [
        sys.executable,
        "scripts/check_results/sampling_time_constraint_colab_result_checker.py",
        "--run-root",
        layout["drive_run_root"],
    ]


def _stage_zip_mode_uses_unified_package(layout: dict[str, str]) -> bool:
    """判断是否跳过旧版 packages/ 打包, 避免 Drive 重复归档。"""

    mode = str(layout.get("stage_package_handoff_mode") or os.environ.get("SSTW_COLAB_STAGE_IO_MODE", ""))
    if mode.strip().lower() not in {"local_zip", "stage_zip", "zip_handoff"}:
        return False
    return os.environ.get("SSTW_WRITE_LEGACY_DRIVE_PACKAGE_IN_STAGE_ZIP_MODE", "false").strip().lower() not in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


def _build_legacy_drive_packaging_noop_command(packager_name: str) -> list[str]:
    """构造可执行 no-op 命令, 由阶段 zip 发布逻辑承担实际落盘。"""

    message = (
        "SSTW stage zip handoff is active; skip legacy drive packager "
        f"{packager_name}. Unified output is written by publish_colab_stage_package."
    )
    return [sys.executable, "-c", f"print({message!r})"]


def build_drive_packaging_command(layout: dict[str, str], include_videos: bool = True) -> list[str]:
    """构造 B6 Google Drive 打包命令。"""
    if _stage_zip_mode_uses_unified_package(layout):
        return _build_legacy_drive_packaging_noop_command("sampling_time_constraint_drive_packager.py")

    command = [
        sys.executable,
        "scripts/package_results/sampling_time_constraint_drive_packager.py",
        "--run-root",
        layout["drive_run_root"],
        "--drive-package-dir",
        layout["drive_package_dir"],
    ]
    if not include_videos:
        command.append("--exclude-videos")
    return command


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    """执行 Notebook 编排命令, 并实时显示 repository runner 进度。"""
    return run_streaming_command(command)
