"""B5 生成式视频 Colab Notebook 的路径和命令编排工具。"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
import subprocess
import sys


DEFAULT_DRIVE_PROJECT_ROOT = "/content/drive/MyDrive/SSTW"


def build_drive_layout(drive_project_root: str = DEFAULT_DRIVE_PROJECT_ROOT) -> dict[str, str]:
    """构造 Colab 与 Google Drive 共享的 SSTW 输出目录布局。"""
    root = PurePosixPath(drive_project_root)
    return {
        "drive_project_root": root.as_posix(),
        "drive_dataset_root": (root / "datasets" / "generative_video_prompt_suite").as_posix(),
        "drive_run_root": (root / "runs" / "generative_video_model_probe_colab").as_posix(),
        "drive_package_dir": (root / "packages" / "generative_video_model_probe").as_posix(),
        "drive_log_dir": (root / "logs" / "generative_video_model_probe").as_posix(),
        "prompt_suite_path": (root / "datasets" / "generative_video_prompt_suite" / "prompt_seed_suite.json").as_posix(),
    }


def ensure_drive_layout(drive_project_root: str = DEFAULT_DRIVE_PROJECT_ROOT) -> dict[str, str]:
    """创建 Google Drive 目标目录并返回路径布局。"""
    layout = build_drive_layout(drive_project_root)
    for key, value in layout.items():
        if key.endswith("_dir") or key.endswith("_root"):
            Path(value).mkdir(parents=True, exist_ok=True)
    return layout


def build_prompt_suite_command(layout: dict[str, str]) -> list[str]:
    """构造 prompt suite 数据集命令, 该命令不执行 GPU 模型测试。"""
    return [sys.executable, "scripts/prepare_generative_video_prompt_suite.py", "--output-root", layout["drive_dataset_root"]]


def build_colab_runtime_command(layout: dict[str, str], profile: str, model_id: str, cross_model_id: str = "") -> list[str]:
    """构造 B5 Colab GPU 运行命令。"""
    command = [
        sys.executable,
        "-m",
        "experiments.generative_video_model_probe.colab_runtime",
        "--output-root",
        layout["drive_run_root"],
        "--prompt-suite-path",
        layout["prompt_suite_path"],
        "--profile",
        profile,
        "--model-id",
        model_id,
    ]
    if cross_model_id:
        command.extend(["--cross-model-id", cross_model_id])
    return command


def build_formal_metric_command(
    layout: dict[str, str],
    semantic_model_id: str = "openai/clip-vit-base-patch32",
    semantic_frame_limit: int = 8,
    disable_semantic_metric: bool = False,
) -> list[str]:
    """构造 B5 正式质量、运动与语义 metric 命令, 从实际 mp4 文件生成 governed records。"""
    command = [
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
    if disable_semantic_metric:
        command.append("--disable-semantic-metric")
    return command


def build_mechanism_postprocess_command(layout: dict[str, str]) -> list[str]:
    """构造 B5 Colab 机制后处理命令, 从已有 governed records 重建后处理 artifacts。"""
    return [
        sys.executable,
        "-m",
        "experiments.generative_video_model_probe.postprocess_runner",
        "--run-root",
        layout["drive_run_root"],
    ]


def build_drive_packaging_command(layout: dict[str, str], include_videos: bool = True) -> list[str]:
    """构造 Google Drive 打包命令。"""
    command = [
        sys.executable,
        "scripts/package_results/generative_video_drive_packager.py",
        "--run-root",
        layout["drive_run_root"],
        "--drive-package-dir",
        layout["drive_package_dir"],
    ]
    if not include_videos:
        command.append("--exclude-videos")
    return command


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    """执行 Notebook 编排命令, 输出仍由仓库脚本负责生成。"""
    return subprocess.run(command, text=True, capture_output=True)
