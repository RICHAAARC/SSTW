"""B5 生成式视频 Colab Notebook 的路径和命令编排工具。"""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
from typing import Mapping


DEFAULT_DRIVE_PROJECT_ROOT = "/content/drive/MyDrive/SSTW"
DEFAULT_VALIDATION_SCALE_CONFIG = "configs/protocol/validation_scale_generative_probe.json"
DEFAULT_FPR01_PILOT_CONFIG = "configs/protocol/fpr01_pilot_generative_probe.json"
PAPER_GATE_PROFILES = {"validation_scale", "pilot_paper", "fpr01_pilot"}
EXTERNAL_BASELINE_COLAB_PREFLIGHT_DECISION = "artifacts/external_baseline_colab_preflight_decision.json"


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


def _read_json(path: str | Path) -> dict:
    """读取 Notebook helper 需要的轻量 JSON 配置, 并兼容 UTF-8 BOM。"""
    input_path = Path(path)
    if not input_path.exists():
        return {}
    payload = json.loads(input_path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON 顶层必须是对象: {input_path}")
    return payload


def _write_json(path: str | Path, payload: Mapping[str, object]) -> None:
    """写出 Colab preflight artifact, 使冷启动失败也能在 Google Drive 中审计原因。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _config_path_for_profile(profile: str) -> str:
    """根据运行 profile 选择现代 baseline 要求来源。"""
    if profile in {"pilot_paper", "fpr01_pilot"}:
        return DEFAULT_FPR01_PILOT_CONFIG
    return DEFAULT_VALIDATION_SCALE_CONFIG


def external_baseline_command_env_var_for(baseline_id: str) -> str:
    """由 baseline_id 推导 Colab 中对应的官方命令环境变量名。

    该函数属于通用工程写法。项目特定约定是所有现代视频水印 baseline
    都通过 `SSTW_<BASELINE_ID>_EVAL_COMMAND` 注入官方 detector / scorer 命令。
    """
    return f"SSTW_{baseline_id.upper()}_EVAL_COMMAND"


def required_modern_external_baseline_command_requirements(
    profile: str,
    config_path: str | Path | None = None,
) -> list[dict[str, str]]:
    """从 protocol config 中读取现代 baseline command 要求。

    Notebook 不应手写 baseline 清单。该函数把 `validation_scale`、`pilot_paper`
    和 `fpr01_pilot` 的 hard gate 要求统一收敛到 helper, 防止配置已更新而
    Notebook cell 仍保留旧 baseline 列表。
    """
    config = _read_json(config_path or _config_path_for_profile(profile))
    baseline_ids = [
        str(name)
        for name in config.get("required_modern_external_baseline_adapter_names", [])
        if str(name)
    ]
    return [
        {
            "baseline_id": baseline_id,
            "external_baseline_command_env_var": external_baseline_command_env_var_for(baseline_id),
        }
        for baseline_id in baseline_ids
    ]


def build_modern_baseline_command_env(
    profile: str,
    command_templates: Mapping[str, str],
    config_path: str | Path | None = None,
) -> dict[str, str]:
    """构造现代 baseline command 环境变量映射。

    `command_templates` 可以使用 baseline_id 作为 key, 也可以直接使用环境变量名
    作为 key。这样 Notebook 只需要维护用户可编辑的短变量, 具体 hard gate 清单
    始终来自 protocol config。
    """
    env: dict[str, str] = {}
    for requirement in required_modern_external_baseline_command_requirements(profile, config_path):
        baseline_id = requirement["baseline_id"]
        env_var = requirement["external_baseline_command_env_var"]
        env[env_var] = str(command_templates.get(baseline_id) or command_templates.get(env_var) or "")
    return env


def build_external_baseline_colab_preflight_decision(
    layout: dict[str, str],
    *,
    profile: str,
    command_env: Mapping[str, str],
    require_modern_baseline_commands_for_paper_gate: bool,
    run_external_baseline_source_clone: bool,
    evidence_paths: list[str] | tuple[str, ...] | None = None,
    config_path: str | Path | None = None,
) -> dict[str, object]:
    """构造 external baseline Colab preflight 决策。

    该决策只检查真实 GPU 运行前是否具备现代 baseline command 配置, 不运行第三方
    baseline, 也不把配置存在解释为论文 claim。其价值在于: Colab 冷启动失败时,
    Google Drive 中仍保留可审计的阻断原因。
    """
    requirements = required_modern_external_baseline_command_requirements(profile, config_path)
    required_env_vars = [item["external_baseline_command_env_var"] for item in requirements]
    configured_env_vars = [
        env_var for env_var in required_env_vars
        if str(command_env.get(env_var) or "").strip()
    ]
    missing_env_vars = [env_var for env_var in required_env_vars if env_var not in configured_env_vars]
    paper_gate_profile = profile in PAPER_GATE_PROFILES
    hard_required = paper_gate_profile and require_modern_baseline_commands_for_paper_gate
    decision = "FAIL" if hard_required and missing_env_vars else "PASS"
    if not paper_gate_profile:
        status = "not_required_for_profile"
    elif not require_modern_baseline_commands_for_paper_gate:
        status = "requirement_disabled"
    elif missing_env_vars:
        status = "commands_missing_for_paper_gate"
    else:
        status = "commands_configured_for_paper_gate"
    return {
        "artifact_name": "external_baseline_colab_preflight_decision.json",
        "manifest_kind": "external_baseline_colab_preflight",
        "profile": profile,
        "run_root": layout["drive_run_root"],
        "external_baseline_colab_preflight_decision": decision,
        "external_baseline_colab_preflight_status": status,
        "paper_gate_profile": paper_gate_profile,
        "require_modern_baseline_commands_for_paper_gate": bool(require_modern_baseline_commands_for_paper_gate),
        "run_external_baseline_source_clone": bool(run_external_baseline_source_clone),
        "required_modern_external_baseline_adapter_names": [item["baseline_id"] for item in requirements],
        "external_baseline_colab_preflight_required_env_vars": required_env_vars,
        "external_baseline_colab_preflight_configured_env_vars": configured_env_vars,
        "external_baseline_colab_preflight_missing_env_vars": missing_env_vars,
        "external_baseline_colab_preflight_missing_env_var_count": len(missing_env_vars),
        "external_baseline_colab_preflight_required_env_var_count": len(required_env_vars),
        "external_baseline_colab_preflight_configured_env_var_count": len(configured_env_vars),
        "external_baseline_evidence_path_count": len(evidence_paths or []),
        "evidence_paths": list(evidence_paths or []),
        "claim_support_status": "external_baseline_colab_preflight_only_not_claim_evidence",
    }


def write_external_baseline_colab_preflight_decision(
    layout: dict[str, str],
    *,
    profile: str,
    command_env: Mapping[str, str],
    require_modern_baseline_commands_for_paper_gate: bool,
    run_external_baseline_source_clone: bool,
    evidence_paths: list[str] | tuple[str, ...] | None = None,
    config_path: str | Path | None = None,
) -> dict[str, object]:
    """写出 external baseline Colab preflight 决策 artifact。"""
    decision = build_external_baseline_colab_preflight_decision(
        layout,
        profile=profile,
        command_env=command_env,
        require_modern_baseline_commands_for_paper_gate=require_modern_baseline_commands_for_paper_gate,
        run_external_baseline_source_clone=run_external_baseline_source_clone,
        evidence_paths=evidence_paths,
        config_path=config_path,
    )
    _write_json(Path(layout["drive_run_root"]) / EXTERNAL_BASELINE_COLAB_PREFLIGHT_DECISION, decision)
    return decision


def validate_modern_baseline_commands_for_profile(preflight_decision: Mapping[str, object]) -> None:
    """在 paper gate profile 缺少现代 baseline command 时抛出明确错误。"""
    if preflight_decision.get("external_baseline_colab_preflight_decision") == "FAIL":
        missing = preflight_decision.get("external_baseline_colab_preflight_missing_env_vars")
        raise RuntimeError(
            "当前 PROFILE 是 paper gate 或 paper gate 前最后门禁, 必须先在 Colab 配置现代视频水印 baseline command。"
            f" 缺失: {missing}"
        )


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


def build_external_baseline_source_intake_command(layout: dict[str, str], execute_clone: bool = False) -> list[str]:
    """构造 external baseline source intake 命令, 写出源码、adapter 和命令配置治理清单。

    `execute_clone` 仅应在 Colab 冷启动且用户已经确认第三方源码 URL 可访问时启用。
    默认不访问网络, 这样本地测试和 harness 审计不会被外部仓库状态影响。
    """
    command = [
        sys.executable,
        "scripts/build_external_baseline_source_intake.py",
        "--output-root",
        f"{layout['drive_run_root']}/artifacts",
        "--repo-root",
        ".",
    ]
    if execute_clone:
        command.append("--execute-clone")
    return command



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


def build_fpr01_pilot_gate_command(layout: dict[str, str]) -> list[str]:
    """构建 pilot_paper FPR=0.01 gate 命令, 只汇总已落盘 records 并写出冻结阈值。"""
    return [
        sys.executable,
        "-m",
        "experiments.generative_video_model_probe.fpr01_pilot_gate",
        "--run-root",
        layout["drive_run_root"],
        "--write-outputs",
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
    """构造 validation-scale gate 命令, 防止从 small-scale pilot 直接跳到 pilot_paper 或 full_paper。"""
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
