"""Colab 阶段 zip 交接工具。

该模块把 Notebook 的大批量小文件 I/O 从 Google Drive 切换到 Colab 本地磁盘:

1. Notebook 启动后只从 Drive 复制少量阶段 zip。
2. zip 解压到 `/content` 本地 workspace 后, 后续 runner 都读写本地路径。
3. 阶段完成后先在本地生成 zip, 再把单个 zip 和 manifest 复制回 Drive。

该实现属于项目特定治理写法。通用工程原则是“本地热路径 + 远端冷归档”,
项目特定约束是 zip 中必须保存 governed records、artifacts、tables、reports、
videos 和 external baseline official bundle, 以便后续 Notebook 只依赖冻结包。
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import time
import zipfile
from typing import Any, Mapping

from main.protocol.package_naming import (
    build_package_batch_id,
    build_package_file_stem,
    current_short_commit,
    current_utc_time_for_filename,
    sanitize_filename_token,
)
from paper_workflow.colab_utils.notebook_run_timing import (
    finalize_notebook_runtime_report_for_package,
    initialize_notebook_runtime_session,
)
from experiments.generative_video_model_probe.paper_result_artifact_builders import (
    PAPER_RESULT_ARTIFACT_RELPATHS,
)


DEFAULT_LOCAL_WORKSPACE_ROOT = "/content/SSTW_stage_workspace"
DEFAULT_LOCAL_PACKAGE_CACHE_ROOT = "/content/SSTW_stage_packages"
STAGE_PACKAGE_MANIFEST_KIND = "colab_stage_zip_handoff_manifest"
STAGE_PACKAGE_MANIFEST_VERSION = "2026_07_01_profile_stage_timestamp_zip_v2"
EXTERNAL_BASELINE_FORMAL_REFERENCE_DECISION_KIND = "modern_external_baseline_formal_reference_decision"
EXTERNAL_BASELINE_REFERENCE_DIR_NAME = "external_baseline_official_reference"
MOTION_THRESHOLD_PACKAGE_DIR_NAME = "motion_threshold"
HELPER_PACKAGE_DIR_NAME = "helper"

MODERN_EXTERNAL_BASELINE_IDS = (
    "revmark",
    "videoseal",
    "vidsig",
    "videoshield",
    "wam_frame",
)

FORMAL_COMPARISON_SCORING_PACKAGE_RELPATHS = (
    "artifacts/external_baseline_colab_preflight_decision.json",
    "artifacts/external_baseline_command_template_summary.json",
    "artifacts/external_baseline_official_bridge_preflight_decision.json",
    "artifacts/external_baseline_official_result_bundle_preflight_decision.json",
    "artifacts/external_baseline_execution_manifest.json",
    "artifacts/external_baseline_status_decision.json",
    "artifacts/external_baseline_comparison_decision.json",
    "artifacts/external_baseline_self_containment_decision.json",
    "artifacts/fair_detection_calibration_decision.json",
    "artifacts/formal_method_baseline_comparison_decision.json",
    "artifacts/formal_baseline_difference_interval_decision.json",
    "artifacts/notebook_runtime_report.json",
    "artifacts/notebook_run_timing_manifest.json",
    "artifacts/formal_comparison_external_baseline_environment_decision.json",
    "artifacts/stage_package_restore_decision.json",
    "artifacts/sstw_measured_formal_decision.json",
    "records/external_baseline_records.jsonl",
    "records/external_baseline_score_records.jsonl",
    "records/external_baseline_self_containment_records.jsonl",
    "records/fair_detection_calibration_records.jsonl",
    "records/formal_method_baseline_comparison_records.jsonl",
    "records/formal_baseline_difference_interval_records.jsonl",
    "records/notebook_stage_timing_records.jsonl",
    "records/sstw_measured_formal_records.jsonl",
    "reports/external_baseline_status_report.md",
    "reports/external_baseline_comparison_report.md",
    "reports/external_baseline_self_containment_report.md",
    "reports/fair_detection_calibration_report.md",
    "reports/formal_method_baseline_comparison_report.md",
    "reports/formal_baseline_difference_interval_report.md",
    "reports/sstw_measured_formal_report.md",
    "tables/external_baseline_status_table.csv",
    "tables/external_baseline_comparison_table.csv",
    "tables/external_baseline_self_containment_table.csv",
    "tables/fair_detection_calibration_table.csv",
    "tables/formal_method_baseline_comparison_table.csv",
    "tables/formal_baseline_difference_interval_table.csv",
    "tables/sstw_measured_formal_table.csv",
)

PAPER_EVIDENCE_POSTPROCESS_PACKAGE_RELPATHS = (
    "artifacts/adaptive_attack_decision.json",
    "artifacts/claim3_downgrade_decision.json",
    "artifacts/data_split_and_leakage_guard_decision.json",
    "artifacts/low_fpr_formal_statistics_decision.json",
    "artifacts/low_fpr_curve_decision.json",
    "artifacts/motion_consistency_exclusion_decision.json",
    "artifacts/motion_threshold_calibration_decision.json",
    "artifacts/motion_threshold_reuse_decision.json",
    "artifacts/notebook_runtime_report.json",
    "artifacts/notebook_run_timing_manifest.json",
    "artifacts/paper_result_artifact_skeleton_decision.json",
    "artifacts/efficiency_metric_decision.json",
    "artifacts/real_adaptive_attack_decision.json",
    "artifacts/real_world_attack_decision.json",
    "artifacts/replay_and_sketch_gate_decision.json",
    "artifacts/stage_package_restore_decision.json",
    "artifacts/statistical_confidence_interval_decision.json",
    "artifacts/validation_internal_ablation_decision.json",
    "artifacts/video_quality_metric_decision.json",
    "artifacts/validation_scale_formal_internal_ablation_decision.json",
    "records/adaptive_attack_records.jsonl",
    "records/claim3_downgrade_records.jsonl",
    "records/data_split_and_leakage_guard_records.jsonl",
    "records/low_fpr_formal_statistics_records.jsonl",
    "records/low_fpr_curve_records.jsonl",
    "records/motion_consistency_exclusion_records.jsonl",
    "records/notebook_stage_timing_records.jsonl",
    "records/efficiency_metric_records.jsonl",
    "records/real_adaptive_attack_records.jsonl",
    "records/real_world_attack_records.jsonl",
    "records/replay_uncertainty_records.jsonl",
    "records/statistical_confidence_interval_records.jsonl",
    "records/trajectory_sketch_verification_records.jsonl",
    "records/validation_internal_ablation_records.jsonl",
    "records/validation_scale_formal_internal_ablation_records.jsonl",
    "records/video_quality_metric_records.jsonl",
    "records/wrong_prompt_replay_records.jsonl",
    "records/wrong_sampler_replay_records.jsonl",
    "reports/adaptive_attack_report.md",
    "reports/claim3_downgrade_report.md",
    "reports/data_split_and_leakage_guard_report.md",
    "reports/low_fpr_formal_statistics_report.md",
    "reports/low_fpr_curve_report.md",
    "reports/motion_consistency_exclusion_report.md",
    "reports/paper_result_artifact_skeleton_report.md",
    "reports/efficiency_metric_report.md",
    "reports/real_adaptive_attack_report.md",
    "reports/real_world_attack_report.md",
    "reports/replay_and_sketch_gate_report.md",
    "reports/statistical_confidence_interval_report.md",
    "reports/validation_internal_ablation_report.md",
    "reports/validation_scale_formal_internal_ablation_report.md",
    "reports/video_quality_metric_report.md",
    "tables/adaptive_attack_table.csv",
    "tables/claim3_downgrade_table.csv",
    "tables/data_split_and_leakage_guard_table.csv",
    "tables/low_fpr_formal_statistics_table.csv",
    "tables/low_fpr_curve_table.csv",
    "tables/motion_consistency_exclusion_table.csv",
    "tables/efficiency_metric_table.csv",
    "tables/real_adaptive_attack_table.csv",
    "tables/real_world_attack_table.csv",
    "tables/replay_verification_table.csv",
    "tables/statistical_confidence_interval_table.csv",
    "tables/validation_internal_ablation_table.csv",
    "tables/validation_scale_formal_internal_ablation_table.csv",
    "tables/video_quality_metric_table.csv",
    "figures/efficiency_comparison_figure.json",
    "figures/low_fpr_curve_figure.json",
    "figures/real_adaptive_attack_robustness_figure.json",
    "figures/real_world_attack_robustness_figure.json",
    "figures/video_quality_robustness_tradeoff_figure.json",
)

OBSOLETE_STAGE_PAYLOAD_NAME_FRAGMENTS = (
    "small_scale_claim_pilot",
)

OBSOLETE_EXTERNAL_BASELINE_EVIDENCE_IDS = {
    "spdmark",
}

MAIN_STAGE_PACKAGE_IDS = {
    "formal_comparison_scoring_colab",
    "generative_video_runtime_colab",
    "paper_evidence_postprocess_colab",
    "paper_gate_and_package_colab",
}

HELPER_WORKFLOW_PROFILES = {
    "sampling_time_constraint",
    "wan21_flow_adapter_preflight",
}


def stage_zip_handoff_enabled() -> bool:
    """判断当前 Notebook 是否启用阶段 zip 本地化交接。

    默认保持关闭, 便于本地单元测试和非 Colab 环境继续使用普通路径。所有正式
    Colab Notebook 会显式设置 `SSTW_COLAB_STAGE_IO_MODE=local_zip`。
    """

    return os.environ.get("SSTW_COLAB_STAGE_IO_MODE", "").strip().lower() in {
        "local_zip",
        "stage_zip",
        "zip_handoff",
    }


def stage_package_id_for_notebook(
    notebook_role: str,
    *,
    baseline_id: str | None = None,
) -> str:
    """根据 Notebook role 生成稳定的阶段包 ID。"""

    role = sanitize_filename_token(notebook_role)
    if role == "external_baseline_formal_scoring" and baseline_id:
        return f"external_baseline_formal_reference_{sanitize_filename_token(baseline_id)}"
    if role == "generative_video_runtime":
        return "generative_video_runtime_colab"
    if role == "formal_comparison_scoring":
        return "formal_comparison_scoring_colab"
    if role == "paper_evidence_postprocess":
        return "paper_evidence_postprocess_colab"
    if role == "motion_threshold_calibration":
        return "motion_threshold_calibration_colab"
    if role == "paper_gate_and_package":
        return "paper_gate_and_package_colab"
    return f"{role}_colab" if not role.endswith("_colab") else role


def _as_posix(path_text: str | Path) -> str:
    """把路径转换为 POSIX 风格字符串, 便于 Colab 与 Windows 测试共用。"""

    return str(path_text).replace("\\", "/")


def _relative_to_drive_project(path_text: str, drive_project_root: str) -> PurePosixPath | None:
    """若 path 位于 Drive 项目根目录下, 返回其相对路径。"""

    path = PurePosixPath(_as_posix(path_text))
    root = PurePosixPath(_as_posix(drive_project_root))
    try:
        return path.relative_to(root)
    except ValueError:
        return None


def _localize_drive_path(path_text: str, drive_project_root: str, local_project_root: Path) -> str:
    """把 Drive 项目内路径映射到 Colab 本地 workspace。"""

    relative = _relative_to_drive_project(path_text, drive_project_root)
    if relative is None:
        return str(local_project_root / sanitize_filename_token(Path(path_text).name or "path"))
    return str(local_project_root / Path(relative.as_posix()))


def stage_package_dir(
    drive_project_root: str | Path,
    workflow_profile: str,
    stage_package_id: str,
) -> Path:
    """返回某个阶段包在 Google Drive 中的归档目录。"""

    root = Path(drive_project_root)
    profile = sanitize_filename_token(workflow_profile)
    package_id = sanitize_filename_token(stage_package_id)
    if package_id == "motion_threshold_calibration_colab":
        return root / MOTION_THRESHOLD_PACKAGE_DIR_NAME
    if _is_external_baseline_stage_package(package_id):
        return root / profile / EXTERNAL_BASELINE_REFERENCE_DIR_NAME
    if profile in HELPER_WORKFLOW_PROFILES or package_id not in MAIN_STAGE_PACKAGE_IDS:
        return root / HELPER_PACKAGE_DIR_NAME
    return root / profile / package_id


def _env_flag(name: str, default: bool = False) -> bool:
    """读取布尔环境变量, 统一处理 Colab Notebook 中的开关。"""

    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _is_external_baseline_stage_package(stage_package_id: str) -> bool:
    """判断阶段包是否属于 modern external baseline 官方参考结果。"""

    return sanitize_filename_token(stage_package_id).startswith("external_baseline_formal_reference_")


def _stage_package_manifest_path_for_zip(package_zip: Path) -> Path:
    """根据 zip 名称推导同阶段 manifest 路径。"""

    return package_zip.with_name(f"{package_zip.stem}_manifest.json")


def _load_notebook_timing_metadata(layout: Mapping[str, str]) -> dict[str, Any]:
    """读取 Notebook 总耗时摘要, 写入阶段包 manifest 便于在 Drive 侧快速查看。"""

    run_root = Path(str(layout.get("drive_run_root") or ""))
    report_path = run_root / "artifacts" / "notebook_runtime_report.json"
    legacy_path = run_root / "artifacts" / "notebook_run_timing_manifest.json"
    timing_path = report_path
    if not timing_path.exists():
        timing_path = legacy_path
    if not timing_path.exists():
        return {
            "notebook_runtime_report_path": str(report_path),
            "notebook_run_timing_manifest_path": str(legacy_path),
            "notebook_timing_status": "missing",
        }
    try:
        timing = json.loads(timing_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {
            "notebook_runtime_report_path": str(report_path),
            "notebook_run_timing_manifest_path": str(legacy_path),
            "notebook_timing_status": "invalid_json",
        }
    return {
        "notebook_runtime_report_path": str(report_path),
        "notebook_run_timing_manifest_path": str(legacy_path),
        "notebook_run_id": str(timing.get("notebook_run_id") or ""),
        "notebook_elapsed_sec": timing.get("notebook_elapsed_sec"),
        "notebook_elapsed_min": timing.get("notebook_elapsed_min"),
        "notebook_timing_status": str(timing.get("notebook_timing_status") or ""),
        "notebook_timing_scope": str(timing.get("notebook_timing_scope") or ""),
        "notebook_timing_start_source": str(timing.get("notebook_timing_start_source") or ""),
        "notebook_timing_coverage_status": str(timing.get("notebook_timing_coverage_status") or ""),
        "notebook_stage_timing_record_count": timing.get("notebook_stage_timing_record_count"),
        "notebook_stage_timing_records_path": str(timing.get("notebook_stage_timing_records_path") or ""),
        "stage_package_publish_included_in_notebook_elapsed": bool(
            timing.get("stage_package_publish_included_in_notebook_elapsed", False)
        ),
    }


def _stage_package_zip_allowed_for_restore(package_zip: Path, stage_package_id: str) -> bool:
    """判断候选 zip 是否允许作为恢复输入。"""

    if not _is_external_baseline_stage_package(stage_package_id):
        return True
    manifest_path = _stage_package_manifest_path_for_zip(package_zip)
    if not manifest_path.exists():
        return _env_flag("SSTW_ALLOW_LEGACY_EXTERNAL_BASELINE_STAGE_PACKAGE_RESTORE", default=False)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if manifest.get("stage_package_publish_status") != "published":
        return False
    if manifest.get("formal_reference_decision") != "PASS":
        return False
    return True


def latest_stage_package_zip(
    drive_project_root: str | Path,
    workflow_profile: str,
    stage_package_id: str,
) -> Path | None:
    """查找 Drive 中某个阶段最新的完整时间戳 zip。"""

    package_dir = stage_package_dir(drive_project_root, workflow_profile, stage_package_id)
    profile = sanitize_filename_token(workflow_profile)
    package_id = sanitize_filename_token(stage_package_id)
    candidates = sorted(
        package_dir.glob(f"{profile}_{package_id}_*.zip"),
        key=lambda path: path.name,
        reverse=True,
    )
    for candidate in candidates:
        if _stage_package_zip_allowed_for_restore(candidate, stage_package_id):
            return candidate
    return None


def _stage_package_source_workflow_profile(
    layout: Mapping[str, str],
    stage_package_id: str,
) -> str:
    """返回恢复某个阶段包时应使用的来源 profile。

    通用写法是从当前 Notebook 的 workflow profile 读取前置阶段包。项目特定例外是
    `motion_threshold_calibration_colab`: 它属于独立 calibration split, 后续
    validation / pilot / full profile 都必须复用 `motion_calibration` 中冻结的阈值,
    不能到当前 evaluation profile 下查找或重新估计阈值。
    """

    if sanitize_filename_token(stage_package_id) == "motion_threshold_calibration_colab":
        return "motion_calibration"
    workflow_profile = str(layout.get("workflow_profile") or layout.get("runtime_profile") or "default")
    if sanitize_filename_token(stage_package_id) == "paper_gate_and_package_colab":
        if workflow_profile == "pilot_paper":
            return "validation_scale"
        if workflow_profile == "full_paper":
            return "pilot_paper"
    return workflow_profile


def activate_local_stage_layout(
    layout: Mapping[str, str],
    *,
    notebook_role: str,
    baseline_id: str | None = None,
    local_workspace_root: str | Path | None = None,
) -> dict[str, str]:
    """把 workflow layout 的热路径切换到 Colab 本地磁盘。

    `drive_package_dir` 会指向当前阶段的 Drive 归档目录, 只接收最终 zip。
    `external_baseline_resource_root` 若存在资源 zip, 会在后续 prepare 步骤切换到本地
    解包目录; 若没有资源包, 则保留原始远端目录作为显式兼容路径。
    """

    drive_project_root = str(layout.get("drive_project_root") or "/content/drive/MyDrive/SSTW")
    workflow_profile = str(layout.get("workflow_profile") or layout.get("runtime_profile") or "default")
    stage_package_id = stage_package_id_for_notebook(notebook_role, baseline_id=baseline_id)
    local_project_root = Path(local_workspace_root or os.environ.get("SSTW_LOCAL_STAGE_WORKSPACE_ROOT", DEFAULT_LOCAL_WORKSPACE_ROOT))
    local_project_root.mkdir(parents=True, exist_ok=True)

    localized = dict(layout)
    localized["stage_package_handoff_mode"] = "local_zip"
    localized["stage_package_id"] = stage_package_id
    localized["local_stage_workspace_root"] = str(local_project_root)
    localized["local_stage_package_cache_root"] = os.environ.get(
        "SSTW_LOCAL_STAGE_PACKAGE_CACHE_ROOT",
        DEFAULT_LOCAL_PACKAGE_CACHE_ROOT,
    )
    resolved_stage_package_dir = stage_package_dir(drive_project_root, workflow_profile, stage_package_id)
    localized["stage_package_dir"] = str(resolved_stage_package_dir)
    localized["drive_package_dir"] = str(resolved_stage_package_dir)
    if str(layout.get("external_baseline_resource_root") or ""):
        localized["external_baseline_resource_root_remote"] = str(layout["external_baseline_resource_root"])
        localized["external_baseline_resource_root_local"] = str(local_project_root / "resources" / "external_baseline")

    for key in (
        "drive_dataset_root",
        "drive_run_root",
        "drive_log_dir",
        "prompt_suite_path",
        "motion_threshold_artifact_run_root",
        "external_baseline_official_result_bundle_root",
    ):
        value = str(layout.get(key) or "")
        if not value:
            continue
        localized[f"{key}_remote"] = value
        localized[key] = _localize_drive_path(value, drive_project_root, local_project_root)
        if key.endswith("_root") or key.endswith("_dir"):
            Path(localized[key]).mkdir(parents=True, exist_ok=True)
        else:
            Path(localized[key]).parent.mkdir(parents=True, exist_ok=True)

    return localized


def _resource_package_candidates(remote_resource_root: Path) -> list[Path]:
    """列出 Drive resources 中可一次性复制到 Colab 的资源 zip 包。"""

    if not remote_resource_root.exists():
        return []
    candidates: list[Path] = []
    for pattern in ("*.zip", "*/*.zip"):
        candidates.extend(path for path in remote_resource_root.glob(pattern) if path.is_file())
    return sorted(set(candidates), key=lambda path: path.as_posix())


def _strip_resource_archive_prefix(relative: PurePosixPath) -> PurePosixPath:
    """兼容资源包内可能包含的 resources/external_baseline 前缀。"""

    parts = relative.parts
    if len(parts) >= 2 and parts[0] == "resources" and parts[1] == "external_baseline":
        return PurePosixPath(*parts[2:])
    if parts and parts[0] == "external_baseline":
        return PurePosixPath(*parts[1:])
    return relative


def _extract_resource_package(package_zip: Path, target_root: Path) -> int:
    """安全解压单个 external baseline 资源包, 返回解压文件数。"""

    count = 0
    with zipfile.ZipFile(package_zip) as archive:
        members = _safe_archive_members(archive)
        for name in members:
            member = PurePosixPath(name)
            if name.endswith("/") or not member.name:
                continue
            relative = _strip_resource_archive_prefix(member)
            if not relative.parts:
                continue
            output = target_root / Path(relative.as_posix())
            output.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(name) as source, output.open("wb") as target:
                shutil.copyfileobj(source, target)
            count += 1
    return count


def hydrate_external_baseline_resource_packages(layout: Mapping[str, str]) -> dict[str, Any]:
    """把 Drive resources 下的压缩资源包复制到本地并解压。

    该函数只在检测到 zip 包时切换 `external_baseline_resource_root`。若用户仍使用
    Drive 上的松散 checkpoint 文件, 当前 Notebook 会继续使用原远端目录, 避免
    因未打包资源而破坏既有流程。
    """

    remote_root_text = str(layout.get("external_baseline_resource_root_remote") or layout.get("external_baseline_resource_root") or "")
    local_root_text = str(layout.get("external_baseline_resource_root_local") or "")
    if not remote_root_text or not local_root_text:
        return {
            "external_baseline_resource_package_restore_status": "not_configured",
            "resource_package_count": 0,
            "extracted_resource_file_count": 0,
        }
    remote_root = Path(remote_root_text)
    local_root = Path(local_root_text)
    packages = _resource_package_candidates(remote_root)
    if not packages:
        return {
            "external_baseline_resource_package_restore_status": "no_resource_package_zip_found",
            "external_baseline_resource_root": remote_root_text,
            "external_baseline_resource_root_remote": remote_root_text,
            "external_baseline_resource_root_local": local_root_text,
            "resource_package_count": 0,
            "extracted_resource_file_count": 0,
        }

    local_root.mkdir(parents=True, exist_ok=True)
    cache_root = Path(str(layout.get("local_stage_package_cache_root") or DEFAULT_LOCAL_PACKAGE_CACHE_ROOT)) / "resources"
    cache_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    extracted_count = 0
    for package_zip in packages:
        local_zip = cache_root / package_zip.name
        _copy_file(package_zip, local_zip)
        file_count = _extract_resource_package(local_zip, local_root)
        extracted_count += file_count
        rows.append({
            "resource_package_zip": str(package_zip),
            "local_resource_package_zip": str(local_zip),
            "extracted_resource_file_count": file_count,
            "resource_package_sha256": _sha256_file(local_zip),
        })
    return {
        "external_baseline_resource_package_restore_status": "restored",
        "external_baseline_resource_root": str(local_root),
        "external_baseline_resource_root_remote": remote_root_text,
        "external_baseline_resource_root_local": str(local_root),
        "resource_package_count": len(packages),
        "extracted_resource_file_count": extracted_count,
        "resource_package_restore_rows": rows,
    }


def _safe_archive_members(archive: zipfile.ZipFile) -> list[str]:
    """校验 zip 条目, 防止解压逃逸到 workspace 之外。"""

    names = archive.namelist()
    for name in names:
        normalized = PurePosixPath(name)
        if normalized.is_absolute() or ".." in normalized.parts:
            raise ValueError(f"unsafe_stage_package_member:{name}")
    return names


def _sha256_file(path: Path) -> str:
    """计算文件 sha256。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _windows_long_path(path: Path) -> str:
    """在 Windows 本地测试中为长路径添加长路径前缀。"""

    resolved = str(path.resolve())
    if os.name != "nt" or resolved.startswith("\\\\?\\"):
        return resolved
    return "\\\\?\\" + resolved


def _copy_file(source: Path, target: Path) -> None:
    """复制文件并兼容 Windows 长路径。"""

    target.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        shutil.copy2(_windows_long_path(source), _windows_long_path(target))
    else:
        shutil.copy2(source, target)


def hydrate_stage_package(
    layout: Mapping[str, str],
    stage_package_id: str,
    *,
    required: bool = True,
    source_workflow_profile: str | None = None,
) -> dict[str, Any]:
    """从 Drive 复制单个阶段 zip 到本地并解压。"""

    drive_project_root = str(layout.get("drive_project_root") or "/content/drive/MyDrive/SSTW")
    target_workflow_profile = str(layout.get("workflow_profile") or layout.get("runtime_profile") or "default")
    source_profile = source_workflow_profile or _stage_package_source_workflow_profile(layout, stage_package_id)
    local_cache_root = Path(str(layout.get("local_stage_package_cache_root") or DEFAULT_LOCAL_PACKAGE_CACHE_ROOT))
    local_workspace_root = Path(str(layout.get("local_stage_workspace_root") or DEFAULT_LOCAL_WORKSPACE_ROOT))
    source_zip = latest_stage_package_zip(drive_project_root, source_profile, stage_package_id)
    if source_zip is None:
        if required:
            raise FileNotFoundError(
                "缺少阶段 zip 包, 为避免 Google Drive 小文件循环读取, 请先完成并发布阶段包: "
                f"{stage_package_id}; source_workflow_profile={source_profile}"
            )
        return {
            "stage_package_id": stage_package_id,
            "stage_package_restore_status": "missing_optional",
            "stage_package_source_workflow_profile": source_profile,
            "stage_package_target_workflow_profile": target_workflow_profile,
            "required": False,
        }

    local_cache_dir = local_cache_root / source_profile / sanitize_filename_token(stage_package_id)
    local_cache_dir.mkdir(parents=True, exist_ok=True)
    local_zip = local_cache_dir / source_zip.name
    _copy_file(source_zip, local_zip)

    with zipfile.ZipFile(local_zip) as archive:
        members = _safe_archive_members(archive)
        archive.extractall(local_workspace_root)

    return {
        "stage_package_id": stage_package_id,
        "stage_package_restore_status": "restored",
        "stage_package_source_kind": "stage_package",
        "stage_package_source_workflow_profile": source_profile,
        "stage_package_target_workflow_profile": target_workflow_profile,
        "drive_stage_package_zip": str(source_zip),
        "local_stage_package_zip": str(local_zip),
        "local_stage_workspace_root": str(local_workspace_root),
        "stage_package_entry_count": len(members),
        "stage_package_archive_sha256": _sha256_file(local_zip),
        "legacy_package_normalized": False,
    }


def _default_required_stage_packages(layout: Mapping[str, str], notebook_role: str) -> list[str]:
    """返回某个 Notebook role 必须先恢复的阶段包。"""

    profile = str(layout.get("workflow_profile") or "")
    role = sanitize_filename_token(notebook_role)
    if role == "external_baseline_formal_scoring":
        return ["generative_video_runtime_colab"]
    if role == "formal_comparison_scoring":
        required = ["generative_video_runtime_colab"]
        required.extend(
            stage_package_id_for_notebook("external_baseline_formal_scoring", baseline_id=baseline)
            for baseline in MODERN_EXTERNAL_BASELINE_IDS
        )
        return required
    if role == "paper_evidence_postprocess":
        required = ["generative_video_runtime_colab"]
        if profile != "motion_calibration":
            required.append("motion_threshold_calibration_colab")
            required.append("formal_comparison_scoring_colab")
        return required
    if role == "paper_gate_and_package":
        required = ["generative_video_runtime_colab"]
        if profile != "motion_calibration":
            required.append("motion_threshold_calibration_colab")
            required.append("formal_comparison_scoring_colab")
            required.append("paper_evidence_postprocess_colab")
            if profile in {"pilot_paper", "full_paper"}:
                required.append("paper_gate_and_package_colab")
        return required
    if role == "generative_video_runtime" and profile != "motion_calibration":
        return ["motion_threshold_calibration_colab"]
    return []


def _default_optional_stage_packages(notebook_role: str, baseline_id: str | None = None) -> list[str]:
    """返回某个 Notebook role 可恢复但不强制存在的阶段包。"""

    role = sanitize_filename_token(notebook_role)
    return []


def hydrate_stage_packages_for_notebook(
    layout: Mapping[str, str],
    *,
    notebook_role: str,
    baseline_id: str | None = None,
) -> dict[str, Any]:
    """根据 Notebook role 恢复所需阶段包。"""

    if not stage_zip_handoff_enabled():
        return {
            "stage_package_handoff_mode": "disabled",
            "stage_package_restore_status": "skipped",
            "restored_stage_package_count": 0,
        }

    required = _default_required_stage_packages(layout, notebook_role)
    optional = _default_optional_stage_packages(notebook_role, baseline_id=baseline_id)
    rows: list[dict[str, Any]] = []
    for stage_package_id in required:
        rows.append(
            hydrate_stage_package(
                layout,
                stage_package_id,
                required=True,
                source_workflow_profile=_stage_package_source_workflow_profile(layout, stage_package_id),
            )
        )
    for stage_package_id in optional:
        rows.append(
            hydrate_stage_package(
                layout,
                stage_package_id,
                required=False,
                source_workflow_profile=_stage_package_source_workflow_profile(layout, stage_package_id),
            )
        )
    restore_status = "restored_required_packages" if required or optional else "no_required_stage_packages"
    return {
        "stage_package_handoff_mode": "local_zip",
        "stage_package_restore_status": restore_status,
        "required_stage_package_ids": required,
        "optional_stage_package_ids": optional,
        "restored_stage_package_count": sum(1 for row in rows if row.get("stage_package_restore_status") == "restored"),
        "stage_package_restore_rows": rows,
    }


def prepare_colab_stage_layout(
    layout: Mapping[str, str],
    *,
    notebook_role: str,
    baseline_id: str | None = None,
) -> dict[str, str]:
    """为 Notebook 准备本地化 layout, 并恢复前置阶段包。"""

    if not stage_zip_handoff_enabled():
        return initialize_notebook_runtime_session(
            dict(layout),
            notebook_role=notebook_role,
            baseline_id=baseline_id,
        )
    localized = activate_local_stage_layout(layout, notebook_role=notebook_role, baseline_id=baseline_id)
    localized = initialize_notebook_runtime_session(
        localized,
        notebook_role=notebook_role,
        baseline_id=baseline_id,
    )
    resource_restore = hydrate_external_baseline_resource_packages(localized)
    if resource_restore.get("external_baseline_resource_package_restore_status") == "restored":
        localized["external_baseline_resource_root"] = str(resource_restore["external_baseline_resource_root"])
    restore = hydrate_stage_packages_for_notebook(localized, notebook_role=notebook_role, baseline_id=baseline_id)
    restore_path = Path(localized["drive_run_root"]) / "artifacts" / "stage_package_restore_decision.json"
    restore_path.parent.mkdir(parents=True, exist_ok=True)
    restore = {
        **restore,
        "external_baseline_resource_package_restore": resource_restore,
    }
    restore_path.write_text(json.dumps(restore, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    localized["stage_package_restore_decision_path"] = str(restore_path)
    os.environ["SSTW_STAGE_PACKAGE_HANDOFF_MODE"] = "local_zip"
    os.environ["SSTW_EXTERNAL_BASELINE_RESOURCE_ROOT"] = localized.get(
        "external_baseline_resource_root",
        os.environ.get("SSTW_EXTERNAL_BASELINE_RESOURCE_ROOT", ""),
    )
    os.environ["SSTW_EXTERNAL_BASELINE_OFFICIAL_RESULT_BUNDLE_ROOT"] = localized.get(
        "external_baseline_official_result_bundle_root",
        os.environ.get("SSTW_EXTERNAL_BASELINE_OFFICIAL_RESULT_BUNDLE_ROOT", ""),
    )
    return localized


def _iter_package_sources(
    layout: Mapping[str, str],
    *,
    notebook_role: str,
    baseline_id: str | None,
) -> list[tuple[Path, str]]:
    """列出阶段包需要归档的本地源目录。"""

    sources: list[tuple[Path, str]] = []
    role = sanitize_filename_token(notebook_role)
    if role == "external_baseline_formal_scoring":
        run_root = Path(str(layout.get("drive_run_root") or ""))
        bundle_root = Path(str(layout.get("external_baseline_official_result_bundle_root") or ""))
        if baseline_id:
            baseline_token = sanitize_filename_token(baseline_id)
            run_archive_root = _archive_root_for_layout_path(layout, "drive_run_root")
            decision_path = (
                run_root
                / "artifacts"
                / "external_baseline_formal_reference"
                / f"{baseline_token}_formal_reference_decision.json"
            )
            if decision_path.exists():
                sources.append((
                    decision_path,
                    f"{run_archive_root}/artifacts/external_baseline_formal_reference",
                ))
            runtime_report_path = run_root / "artifacts" / "notebook_runtime_report.json"
            if runtime_report_path.exists():
                sources.append((
                    runtime_report_path,
                    f"{run_archive_root}/artifacts",
                ))
            timing_manifest_path = run_root / "artifacts" / "notebook_run_timing_manifest.json"
            if timing_manifest_path.exists():
                sources.append((
                    timing_manifest_path,
                    f"{run_archive_root}/artifacts",
                ))
            timing_records_path = run_root / "records" / "notebook_stage_timing_records.jsonl"
            if timing_records_path.exists():
                sources.append((
                    timing_records_path,
                    f"{run_archive_root}/records",
                ))
            evidence_dir = run_root / "artifacts" / "external_baseline_evidence" / baseline_token
            if evidence_dir.exists():
                sources.append((
                    evidence_dir,
                    f"{run_archive_root}/artifacts/external_baseline_evidence/{baseline_token}",
                ))
            baseline_bundle = bundle_root / baseline_token
            if baseline_bundle.exists():
                sources.append(
                    (
                        baseline_bundle,
                        f"{_archive_root_for_layout_path(layout, 'external_baseline_official_result_bundle_root')}/{baseline_token}",
                    )
                )
        elif bundle_root.exists():
            if run_root.exists():
                artifacts_dir = run_root / "artifacts"
                if artifacts_dir.exists():
                    sources.append((artifacts_dir, f"{_archive_root_for_layout_path(layout, 'drive_run_root')}/artifacts"))
            sources.append((bundle_root, _archive_root_for_layout_path(layout, "external_baseline_official_result_bundle_root")))
        return sources

    if role == "formal_comparison_scoring":
        run_root = Path(str(layout.get("drive_run_root") or ""))
        run_archive_root = _archive_root_for_layout_path(layout, "drive_run_root")
        _append_existing_run_relpaths(
            sources,
            run_root,
            run_archive_root,
            FORMAL_COMPARISON_SCORING_PACKAGE_RELPATHS,
        )
        return sources

    if role == "paper_evidence_postprocess":
        run_root = Path(str(layout.get("drive_run_root") or ""))
        run_archive_root = _archive_root_for_layout_path(layout, "drive_run_root")
        _append_existing_run_relpaths(
            sources,
            run_root,
            run_archive_root,
            PAPER_EVIDENCE_POSTPROCESS_PACKAGE_RELPATHS,
        )
        return sources

    keys = ["drive_run_root"]
    if role in {"generative_video_runtime", "motion_threshold_calibration"}:
        keys.append("drive_dataset_root")
    for key in keys:
        path_text = str(layout.get(key) or "")
        if not path_text:
            continue
        path = Path(path_text)
        if not path.exists():
            continue
        sources.append((path, _archive_root_for_layout_path(layout, key)))
    return sources


def _append_existing_run_relpaths(
    sources: list[tuple[Path, str]],
    run_root: Path,
    archive_root: str,
    relpaths: tuple[str, ...],
) -> None:
    """把指定 run_root 相对文件加入阶段包源列表。

    该函数用于轻量后处理阶段。此类阶段会恢复上游大包, 但它自己的阶段 zip
    只应保存本阶段新生成的 governed 产物, 不能把上游视频、帧图或 official
    bundle 再次重复打包。
    """

    if not run_root.exists():
        return
    for relpath in relpaths:
        source = run_root / relpath
        if not source.exists():
            continue
        sources.append((source, f"{archive_root}/{PurePosixPath(relpath).parent.as_posix()}"))


def _archive_root_for_layout_path(layout: Mapping[str, str], key: str) -> str:
    """根据 layout key 生成 zip 内的规范根路径。"""

    remote_key = f"{key}_remote"
    remote_path = str(layout.get(remote_key) or layout.get(key) or "")
    drive_project_root = str(layout.get("drive_project_root") or "")
    relative = _relative_to_drive_project(remote_path, drive_project_root) if drive_project_root else None
    if relative is not None:
        return relative.as_posix()
    path = Path(remote_path)
    if key == "drive_run_root":
        return f"runs/{path.name}"
    if key == "drive_dataset_root":
        return f"datasets/{path.name}"
    if key == "external_baseline_official_result_bundle_root":
        return f"external_baseline_official_result_bundles/{path.name}"
    return sanitize_filename_token(path.name or key)


def _write_source_to_archive(
    archive: zipfile.ZipFile,
    source_root: Path,
    archive_root: str,
    *,
    include_videos: bool,
) -> int:
    """把 source_root 写入 zip, 返回写入文件数。"""

    count = 0
    if source_root.is_file():
        relative = Path(source_root.name)
        if _should_skip_obsolete_stage_payload(relative):
            return 0
        if not include_videos and _should_skip_video_payload(relative):
            return 0
        archive.write(source_root, arcname=f"{archive_root}/{relative.as_posix()}")
        return 1
    for file_path in sorted(path for path in source_root.rglob("*") if path.is_file()):
        relative = file_path.relative_to(source_root)
        if _should_skip_obsolete_stage_payload(relative):
            continue
        if not include_videos and _should_skip_video_payload(relative):
            continue
        archive.write(file_path, arcname=f"{archive_root}/{relative.as_posix()}")
        count += 1
    return count


def _should_skip_obsolete_stage_payload(relative: Path) -> bool:
    """判断是否应从阶段 zip 中排除历史兼容产物。

    该函数只影响归档内容, 不删除本地运行目录中的文件。这样既能保留旧代码路径的
    本地调试能力, 又能防止 paper gate package 把已经退出主实验门禁的历史 artifact
    当作当前规则的一部分发布。
    """

    relative_posix = PurePosixPath(relative.as_posix())
    relative_text = relative_posix.as_posix()
    if any(fragment in relative_text for fragment in OBSOLETE_STAGE_PAYLOAD_NAME_FRAGMENTS):
        return True
    parts = relative_posix.parts
    for index, part in enumerate(parts[:-1]):
        if part != "external_baseline_evidence":
            continue
        baseline_id = sanitize_filename_token(parts[index + 1])
        if baseline_id in OBSOLETE_EXTERNAL_BASELINE_EVIDENCE_IDS:
            return True
    return False


def _should_skip_video_payload(relative: Path) -> bool:
    """判断文件是否属于可选视频或帧级大文件。"""

    parts = {sanitize_filename_token(part) for part in relative.parts}
    suffix = relative.suffix.lower()
    if parts & {"videos", "attacked_videos"}:
        return True
    if suffix in {".mp4", ".mov", ".avi", ".webm", ".mkv"}:
        return True
    if "frames" in parts and suffix in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
        return True
    return False


def _external_baseline_decision_path(layout: Mapping[str, str], baseline_id: str | None) -> Path | None:
    """返回单个 external baseline 官方参考决策文件路径。"""

    if not baseline_id:
        return None
    run_root = Path(str(layout.get("drive_run_root") or ""))
    if not str(run_root):
        return None
    baseline_token = sanitize_filename_token(baseline_id)
    return run_root / "artifacts" / "external_baseline_formal_reference" / f"{baseline_token}_formal_reference_decision.json"


def _load_external_baseline_decision(layout: Mapping[str, str], baseline_id: str | None) -> dict[str, Any] | None:
    """读取 external baseline 决策文件, 缺失或损坏时返回 None。"""

    decision_path = _external_baseline_decision_path(layout, baseline_id)
    if decision_path is None or not decision_path.exists():
        return None
    try:
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if decision.get("manifest_kind") != EXTERNAL_BASELINE_FORMAL_REFERENCE_DECISION_KIND:
        return None
    return decision


def _external_baseline_publish_blocker(
    layout: Mapping[str, str],
    *,
    stage_package_id: str,
    notebook_role: str,
    baseline_id: str | None,
) -> dict[str, Any] | None:
    """判断 external baseline 阶段是否允许发布完整 zip。"""

    is_external_reference = stage_package_id.startswith("external_baseline_formal_reference_")
    if not is_external_reference:
        return None

    decision = _load_external_baseline_decision(layout, baseline_id)
    decision_path = _external_baseline_decision_path(layout, baseline_id)
    if decision is None:
        return {
            "stage_package_publish_status": "blocked_missing_external_baseline_formal_reference_decision",
            "formal_reference_decision_path": str(decision_path or ""),
            "formal_reference_decision": "MISSING",
            "formal_reference_status": "missing_or_invalid_decision_record",
        }
    if decision.get("formal_reference_decision") != "PASS":
        return {
            "stage_package_publish_status": "skipped_failed_external_baseline_reference",
            "formal_reference_decision_path": str(decision_path or ""),
            "formal_reference_decision": decision.get("formal_reference_decision", "UNKNOWN"),
            "formal_reference_status": decision.get("formal_reference_status", "unknown_failure"),
        }
    return None


def _validation_scale_paper_gate_publish_blocker(
    layout: Mapping[str, str],
    *,
    stage_package_id: str,
    notebook_role: str,
    workflow_profile: str,
) -> dict[str, Any] | None:
    """判断 validation_scale paper gate 包是否允许发布完整 zip。

    该门禁属于项目特定写法。普通阶段 zip 只是文件交接容器, 但
    validation_scale 的 paper gate zip 同时承担“进入 pilot_paper 前最终证据包”
    的职责。若 package manifest 缺失或失败, 说明公平比较、artifact rebuild、
    stage transition 等闭环没有被当前运行证明, 因此只能写阻断 manifest, 不能
    发布可被误用的完整 zip。
    """

    role = sanitize_filename_token(notebook_role)
    profile = sanitize_filename_token(workflow_profile)
    package_id = sanitize_filename_token(stage_package_id)
    if role != "paper_gate_and_package" or profile != "validation_scale" or package_id != "paper_gate_and_package_colab":
        return None

    run_root = Path(str(layout.get("drive_run_root") or ""))
    manifest_path = run_root / "manifests" / "validation_scale_package_manifest.json"
    if not manifest_path.exists():
        return {
            "stage_package_publish_status": "blocked_missing_validation_scale_package_manifest",
            "validation_scale_package_manifest_path": str(manifest_path),
            "validation_scale_package_manifest_decision": "MISSING",
            "validation_scale_gate_decision": "UNKNOWN",
            "validation_scale_to_pilot_paper_transition_decision": "UNKNOWN",
            "stage_package_publish_block_reason": "missing_validation_scale_package_manifest",
        }
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        return {
            "stage_package_publish_status": "blocked_invalid_validation_scale_package_manifest",
            "validation_scale_package_manifest_path": str(manifest_path),
            "validation_scale_package_manifest_decision": "INVALID",
            "validation_scale_gate_decision": "UNKNOWN",
            "validation_scale_to_pilot_paper_transition_decision": "UNKNOWN",
            "stage_package_publish_block_reason": f"invalid_validation_scale_package_manifest_json:{exc}",
        }
    if not isinstance(manifest, dict):
        return {
            "stage_package_publish_status": "blocked_invalid_validation_scale_package_manifest",
            "validation_scale_package_manifest_path": str(manifest_path),
            "validation_scale_package_manifest_decision": "INVALID",
            "validation_scale_gate_decision": "UNKNOWN",
            "validation_scale_to_pilot_paper_transition_decision": "UNKNOWN",
            "stage_package_publish_block_reason": "validation_scale_package_manifest_top_level_not_object",
        }

    decision = str(manifest.get("validation_scale_package_manifest_decision") or "UNKNOWN")
    gate_decision = str(manifest.get("validation_scale_gate_decision") or "UNKNOWN")
    transition_decision = str(manifest.get("validation_scale_to_pilot_paper_transition_decision") or "UNKNOWN")
    if decision == "PASS" and gate_decision == "PASS" and transition_decision == "PASS":
        return None
    return {
        "stage_package_publish_status": "blocked_failed_validation_scale_package_manifest",
        "validation_scale_package_manifest_path": str(manifest_path),
        "validation_scale_package_manifest_decision": decision,
        "validation_scale_gate_decision": gate_decision,
        "validation_scale_to_pilot_paper_transition_decision": transition_decision,
        "missing_artifact_count": manifest.get("missing_artifact_count", 0),
        "missing_artifact_relpaths": manifest.get("missing_artifact_relpaths", []),
        "stage_package_publish_block_reason": "validation_scale_package_manifest_not_pass",
    }


def _prune_remote_stage_package_snapshots(remote_package_dir: Path, workflow_profile: str, stage_package_id: str) -> list[str]:
    """按显式开关删除当前阶段目录下的历史时间戳包。"""

    removed: list[str] = []
    profile = sanitize_filename_token(workflow_profile)
    package_id = sanitize_filename_token(stage_package_id)
    for pattern in (f"{profile}_{package_id}_*.zip", f"{profile}_{package_id}_*_manifest.json"):
        for path in sorted(remote_package_dir.glob(pattern)):
            if not path.is_file():
                continue
            path.unlink()
            removed.append(str(path))
    return removed


def _write_manifest_only_stage_package(
    *,
    layout: Mapping[str, str],
    remote_package_dir: Path,
    remote_manifest: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """只写阶段 manifest, 不写 zip。

    external baseline 未通过时, 这个 manifest 是阻断记录, 不能被后续门禁当成可恢复
    阶段包使用。该设计避免失败运行把数百 MB 或数 GB 的中间视频继续写入 Drive。
    """

    remote_package_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        **manifest,
        "drive_stage_package_zip": "",
        "latest_drive_stage_package_zip": "",
        "stage_package_archive_sha256": "",
        "stage_package_entry_count": 0,
        "stage_package_source_root_count": 0,
        "stage_package_source_roots": [],
        "keep_timestamp_snapshot": False,
        "removed_latest_stage_package_zip": False,
        "removed_timestamp_stage_package_snapshots": [],
        "claim_support_status": "stage_package_blocked_not_claim_evidence",
    }
    remote_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        **manifest,
        "stage_package_manifest_path": str(remote_manifest),
        "latest_stage_package_manifest_path": "",
    }


def publish_colab_stage_package(
    layout: Mapping[str, str],
    *,
    notebook_role: str,
    baseline_id: str | None = None,
    include_videos: bool = True,
) -> dict[str, Any]:
    """将当前 Notebook 阶段的本地结果发布为 Drive 阶段 zip。"""

    if not stage_zip_handoff_enabled():
        return {
            "stage_package_handoff_mode": "disabled",
            "stage_package_publish_status": "skipped",
        }
    stage_package_publish_start = time.perf_counter()

    stage_package_id = str(layout.get("stage_package_id") or stage_package_id_for_notebook(notebook_role, baseline_id=baseline_id))
    workflow_profile = str(layout.get("workflow_profile") or layout.get("runtime_profile") or "default")
    drive_project_root = str(layout.get("drive_project_root") or "/content/drive/MyDrive/SSTW")
    local_cache_root = Path(str(layout.get("local_stage_package_cache_root") or DEFAULT_LOCAL_PACKAGE_CACHE_ROOT))
    local_package_dir = local_cache_root / workflow_profile / stage_package_id
    local_package_dir.mkdir(parents=True, exist_ok=True)
    remote_package_dir = stage_package_dir(drive_project_root, workflow_profile, stage_package_id)
    remote_package_dir.mkdir(parents=True, exist_ok=True)

    package_utc_time = current_utc_time_for_filename()
    package_short_commit = current_short_commit()
    package_batch_id = build_package_batch_id(package_utc_time, package_short_commit)
    stem = build_package_file_stem(f"{workflow_profile}_{stage_package_id}", package_utc_time, package_short_commit)
    local_zip = local_package_dir / f"local_{package_batch_id}.zip"
    local_manifest = local_package_dir / f"local_{package_batch_id}_manifest.json"
    remote_zip = remote_package_dir / f"{stem}.zip"
    remote_manifest = remote_package_dir / f"{stem}_manifest.json"
    keep_timestamp_snapshot = _env_flag("SSTW_STAGE_PACKAGE_KEEP_TIMESTAMP_SNAPSHOT", default=True)
    prune_timestamp_snapshots = _env_flag("SSTW_STAGE_PACKAGE_PRUNE_TIMESTAMP_SNAPSHOTS", default=False)
    finalize_notebook_runtime_report_for_package(
        layout,
        notebook_role=notebook_role,
        baseline_id=baseline_id,
        notebook_timing_status="completed_before_stage_package_publish",
        extra={
            "stage_package_id": stage_package_id,
            "stage_package_publish_timing_policy": (
                "stage_package_publish_elapsed_is_recorded_in_stage_package_manifest_not_in_runtime_report"
            ),
            "stage_package_publish_included_in_notebook_elapsed": False,
        },
    )
    external_decision = _load_external_baseline_decision(layout, baseline_id) if _is_external_baseline_stage_package(stage_package_id) else None
    external_decision_path = _external_baseline_decision_path(layout, baseline_id) if _is_external_baseline_stage_package(stage_package_id) else None
    external_decision_metadata = (
        {
            "formal_reference_decision_path": str(external_decision_path or ""),
            "formal_reference_decision": external_decision.get("formal_reference_decision", "UNKNOWN"),
            "formal_reference_status": external_decision.get("formal_reference_status", "unknown_status"),
        }
        if external_decision is not None
        else (
            {
                "formal_reference_decision_path": str(external_decision_path or ""),
                "formal_reference_decision": "MISSING",
                "formal_reference_status": "missing_or_invalid_decision_record",
            }
            if _is_external_baseline_stage_package(stage_package_id)
            else {}
        )
    )
    notebook_timing_metadata = _load_notebook_timing_metadata(layout)

    base_manifest: dict[str, Any] = {
        "manifest_kind": STAGE_PACKAGE_MANIFEST_KIND,
        "manifest_version": STAGE_PACKAGE_MANIFEST_VERSION,
        "stage_package_id": stage_package_id,
        "notebook_role": notebook_role,
        "baseline_id": baseline_id or "",
        "workflow_profile": workflow_profile,
        "stage_package_handoff_mode": "local_zip",
        "stage_package_publish_status": "pending",
        "package_batch_id": package_batch_id,
        "package_utc_time": package_utc_time,
        "package_short_commit": package_short_commit,
        "stage_package_file_stem": stem,
        "include_videos": bool(include_videos),
        "local_stage_workspace_root": str(layout.get("local_stage_workspace_root") or ""),
        "local_stage_package_zip": str(local_zip),
        "local_stage_package_manifest": str(local_manifest),
        "retention_policy": "timestamp_snapshots_retained_by_default",
        "keep_timestamp_snapshot": keep_timestamp_snapshot,
        "prune_timestamp_snapshots": prune_timestamp_snapshots,
        **notebook_timing_metadata,
        **external_decision_metadata,
    }

    blocker = _external_baseline_publish_blocker(
        layout,
        stage_package_id=stage_package_id,
        notebook_role=notebook_role,
        baseline_id=baseline_id,
    )
    if blocker is not None:
        return _write_manifest_only_stage_package(
            layout=layout,
            remote_package_dir=remote_package_dir,
            remote_manifest=remote_manifest,
            manifest={**base_manifest, **blocker},
        )
    blocker = _validation_scale_paper_gate_publish_blocker(
        layout,
        stage_package_id=stage_package_id,
        notebook_role=notebook_role,
        workflow_profile=workflow_profile,
    )
    if blocker is not None:
        return _write_manifest_only_stage_package(
            layout=layout,
            remote_package_dir=remote_package_dir,
            remote_manifest=remote_manifest,
            manifest={**base_manifest, **blocker},
        )

    sources = _iter_package_sources(layout, notebook_role=notebook_role, baseline_id=baseline_id)
    entry_count = 0
    with zipfile.ZipFile(local_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source_root, archive_root in sources:
            entry_count += _write_source_to_archive(
                archive,
                source_root,
                archive_root,
                include_videos=include_videos,
            )

    archive_sha256 = _sha256_file(local_zip)
    removed_timestamp_snapshots = (
        _prune_remote_stage_package_snapshots(remote_package_dir, workflow_profile, stage_package_id)
        if prune_timestamp_snapshots and not keep_timestamp_snapshot
        else []
    )
    drive_stage_package_zip = str(remote_zip)
    stage_package_manifest_path = str(remote_manifest)

    manifest = {
        **base_manifest,
        "stage_package_publish_status": "published",
        "drive_stage_package_zip": drive_stage_package_zip,
        "latest_drive_stage_package_zip": "",
        "stage_package_archive_sha256": archive_sha256,
        "stage_package_entry_count": entry_count,
        "stage_package_source_root_count": len(sources),
        "stage_package_source_roots": [
            {"source_root": str(source_root), "archive_root": archive_root}
            for source_root, archive_root in sources
        ],
        "removed_timestamp_stage_package_snapshots": removed_timestamp_snapshots,
        "claim_support_status": "stage_package_handoff_container_not_claim_evidence",
    }
    _copy_file(local_zip, remote_zip)
    manifest = {
        **manifest,
        "stage_package_publish_elapsed_sec": round(time.perf_counter() - stage_package_publish_start, 3),
    }
    local_manifest.parent.mkdir(parents=True, exist_ok=True)
    local_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _copy_file(local_manifest, remote_manifest)
    return {
        **manifest,
        "stage_package_publish_status": "published",
        "stage_package_manifest_path": stage_package_manifest_path,
        "latest_stage_package_manifest_path": "",
    }
