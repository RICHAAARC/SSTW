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


def build_motion_threshold_calibration_command(layout: dict[str, str]) -> list[str]:
    """构造 formal motion threshold calibration 命令, 从 formal motion records 冻结或报告阈值状态。"""
    return [
        sys.executable,
        "-m",
        "experiments.generative_video_model_probe.motion_threshold_calibration",
        "--run-root",
        layout["drive_run_root"],
    ]


def build_mechanism_postprocess_command(layout: dict[str, str]) -> list[str]:
    """构造 B5 Colab 机制后处理命令, 从已有 governed records 重建后处理 artifacts。"""
    return [
        sys.executable,
        "-m",
        "experiments.generative_video_model_probe.postprocess_runner",
        "--run-root",
        layout["drive_run_root"],
    ]


def build_pilot_matrix_postprocess_command(layout: dict[str, str]) -> list[str]:
    """构造 small-scale pilot matrix postprocess 命令, 从 generation 与 trajectory records 补齐 pilot 矩阵。"""
    return [
        sys.executable,
        "-m",
        "experiments.generative_video_model_probe.pilot_matrix_postprocess",
        "--run-root",
        layout["drive_run_root"],
    ]


def build_runtime_attack_command(layout: dict[str, str]) -> list[str]:
    """构造 runtime video-file attack 命令, 对真实 mp4 生成 attacked videos 与 governed records。"""
    return [
        sys.executable,
        "-m",
        "experiments.generative_video_model_probe.attack_runner",
        "--run-root",
        layout["drive_run_root"],
    ]


def build_runtime_detection_command(layout: dict[str, str]) -> list[str]:
    """构造 runtime attacked video detection 命令, 把 attacked videos 接入检测评分 records。"""
    return [
        sys.executable,
        "-m",
        "experiments.generative_video_model_probe.detection_runner",
        "--run-root",
        layout["drive_run_root"],
    ]



def build_external_baseline_comparison_command(layout: dict[str, str]) -> list[str]:
    """构建 external_baseline adapter comparison 命令, 从 runtime detection records 生成 baseline 对比结果。"""
    return [
        sys.executable,
        "-m",
        "experiments.generative_video_model_probe.external_baseline_runner",
        "--run-root",
        layout["drive_run_root"],
        "--mode",
        "comparison",
    ]

def build_validation_internal_ablation_command(layout: dict[str, str]) -> list[str]:
    """构造 validation-scale 内部消融矩阵后处理命令。"""
    return [
        sys.executable,
        "-m",
        "experiments.generative_video_model_probe.validation_internal_ablation",
        "--run-root",
        layout["drive_run_root"],
    ]


def build_adaptive_attack_command(layout: dict[str, str]) -> list[str]:
    """构造 validation-scale adaptive attack proxy 命令。"""
    return [
        sys.executable,
        "-m",
        "experiments.generative_video_model_probe.adaptive_attack_runner",
        "--run-root",
        layout["drive_run_root"],
    ]


def build_replay_and_sketch_gate_command(layout: dict[str, str]) -> list[str]:
    """构造 replay/sketch gate validation proxy 命令。"""
    return [
        sys.executable,
        "-m",
        "experiments.generative_video_model_probe.replay_and_sketch_gate",
        "--run-root",
        layout["drive_run_root"],
    ]


def build_claim3_downgrade_command(layout: dict[str, str]) -> list[str]:
    """构造 Claim-3 downgrade gate 命令, 明确 replay/sketch 未闭合时的 claim 边界。"""
    return [
        sys.executable,
        "-m",
        "experiments.generative_video_model_probe.claim3_downgrade",
        "--run-root",
        layout["drive_run_root"],
    ]


def build_statistical_confidence_interval_command(layout: dict[str, str]) -> list[str]:
    """构造 validation-scale 统计置信区间报告命令。"""
    return [
        sys.executable,
        "-m",
        "experiments.generative_video_model_probe.statistical_confidence_interval",
        "--run-root",
        layout["drive_run_root"],
    ]


def build_validation_artifact_rebuild_dry_run_command(layout: dict[str, str]) -> list[str]:
    """构造 validation-scale artifact rebuild dry-run 命令。"""
    return [
        sys.executable,
        "-m",
        "experiments.generative_video_model_probe.validation_artifact_rebuild",
        "--run-root",
        layout["drive_run_root"],
    ]


def build_small_scale_claim_pilot_gate_command(layout: dict[str, str]) -> list[str]:
    """构造 small-scale claim pilot gate 命令, 从 governed records 汇总 pilot 状态。"""
    return [
        sys.executable,
        "-m",
        "experiments.generative_video_model_probe.pilot_claim_gate",
        "--run-root",
        layout["drive_run_root"],
        "--write-outputs",
    ]


def build_validation_scale_gate_command(layout: dict[str, str]) -> list[str]:
    """构造 validation-scale gate 命令, 防止从 pilot 直接跳到 full_paper。"""
    return [
        sys.executable,
        "-m",
        "experiments.generative_video_model_probe.validation_scale_gate",
        "--run-root",
        layout["drive_run_root"],
        "--write-outputs",
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
