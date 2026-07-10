"""独立 Drive packager 的阶段归档路径工具。

该模块把历史独立 packager 的默认输出位置统一到当前 Colab 阶段 zip
归档结构。通用工程写法是让 CLI 默认值集中在一个 helper 中, 项目特定约束是
不再默认写入旧版 `SSTW/packages/` 目录, 防止用户手动运行脚本时重新产生旧目录。
"""

from __future__ import annotations

from pathlib import Path

from evaluation.protocol.package_naming import build_package_file_stem, sanitize_filename_token
from workflows.stage_package_sync import stage_package_dir


DEFAULT_DRIVE_PROJECT_ROOT = "/content/drive/MyDrive/SSTW"


def resolve_stage_package_output_dir(
    drive_project_root: str | Path,
    workflow_profile: str,
    stage_package_id: str,
) -> Path:
    """解析独立 packager 的默认 Drive 输出目录。

    该函数复用 `stage_package_dir` 作为唯一目录规则来源。这样即使用户绕过
    Notebook 直接运行 packager, 输出仍会进入 `motion_threshold/`、
    `<workflow_profile>/<stage_package_id>/`、
    `<workflow_profile>/external_baseline_official_reference/` 或 `helper/` 等
    当前正式归档目录, 不会回退到旧版 `packages/`。
    """

    return stage_package_dir(drive_project_root, workflow_profile, stage_package_id)


def build_packager_file_stem(
    run_root_name: str,
    package_utc_time: str,
    package_short_commit: str,
    *,
    workflow_profile: str | None = None,
    stage_package_id: str | None = None,
) -> str:
    """生成独立 packager 的文件名前缀。

    当传入 workflow profile 和 stage package ID 时, 使用当前阶段包命名规则:
    `<workflow_profile>_<stage_package_id>_<YYYYMMDD_HHMMSS>_<git_short_commit>`。
    未传入这些字段时保留历史函数调用行为, 便于既有轻量单元测试和临时本地打包
    继续以 run_root 名称作为前缀。
    """

    if workflow_profile and stage_package_id:
        return build_package_file_stem(
            f"{workflow_profile}_{stage_package_id}",
            package_utc_time,
            package_short_commit,
        )
    return build_package_file_stem(run_root_name, package_utc_time, package_short_commit)


def packager_manifest_filename(package_file_stem: str, *, stage_package_naming: bool) -> str:
    """返回 manifest 文件名。

    阶段包命名模式使用 `_manifest.json`, 与 `publish_colab_stage_package` 保持一致。
    历史函数调用模式保留 `_package_manifest.json`, 避免破坏已有本地测试和临时脚本。
    """

    suffix = "_manifest.json" if stage_package_naming else "_package_manifest.json"
    return f"{package_file_stem}{suffix}"


def archive_run_root_for_stage(
    run_root_name: str,
    *,
    workflow_profile: str | None = None,
    stage_package_id: str | None = None,
) -> str:
    """返回 zip 内 run_root 应使用的规范相对路径。

    直接运行独立 packager 时, 若文件名已经使用阶段包命名, zip 内容也应尽量接近
    `publish_colab_stage_package` 的结构。否则后续 Notebook 恢复该包时会把文件解压到
    错误目录。未传入阶段字段时保留历史 `run_root.name/...` 结构。
    """

    if not workflow_profile or not stage_package_id:
        return sanitize_filename_token(run_root_name)

    profile = sanitize_filename_token(workflow_profile)
    package_id = sanitize_filename_token(stage_package_id)
    if package_id in {
        "formal_comparison_scoring_colab",
        "generative_video_generation_colab",
        "generative_video_quality_scoring_colab",
        "runtime_attack_colab",
        "runtime_detection_colab",
        "paper_evidence_postprocess_colab",
        "paper_gate_and_package_colab",
    }:
        return f"runs/generative_video_model_probe/{profile}"
    if package_id == "motion_threshold_calibration_colab":
        return "runs/generative_video_model_probe/motion_calibration"
    if package_id == "wan21_flow_adapter_preflight_colab":
        return "runs/wan21_flow_adapter_preflight"
    return f"runs/{package_id}"
