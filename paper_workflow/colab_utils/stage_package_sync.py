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
import zipfile
from typing import Any, Iterable, Mapping

from main.protocol.package_naming import (
    build_package_batch_id,
    current_short_commit,
    current_utc_time_for_filename,
    sanitize_filename_token,
)


DEFAULT_LOCAL_WORKSPACE_ROOT = "/content/SSTW_stage_workspace"
DEFAULT_LOCAL_PACKAGE_CACHE_ROOT = "/content/SSTW_stage_packages"
STAGE_PACKAGE_ROOT_RELATIVE = "stage_packages"
STAGE_PACKAGE_MANIFEST_KIND = "colab_stage_zip_handoff_manifest"
STAGE_PACKAGE_MANIFEST_VERSION = "2026_06_27_local_zip_handoff_v1"
EXTERNAL_BASELINE_FORMAL_REFERENCE_DECISION_KIND = "modern_external_baseline_formal_reference_decision"

MODERN_EXTERNAL_BASELINE_IDS = (
    "videoseal",
    "vidsig",
    "videomark",
    "videoshield",
    "spdmark",
    "sigmark",
)


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

    return (
        Path(drive_project_root)
        / STAGE_PACKAGE_ROOT_RELATIVE
        / sanitize_filename_token(workflow_profile)
        / sanitize_filename_token(stage_package_id)
    )


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

    if package_zip.name == "stage_package_latest.zip":
        return package_zip.with_name("stage_package_latest_manifest.json")
    if package_zip.name.startswith("stage_package__") and package_zip.suffix == ".zip":
        return package_zip.with_name(f"{package_zip.stem}_stage_package_manifest.json")
    return package_zip.with_suffix(".json")


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
    """查找 Drive 中某个阶段的 latest zip。

    默认只接受 `stage_package_latest.zip`。历史时间戳 zip 只能在显式设置
    `SSTW_STAGE_PACKAGE_ALLOW_TIMESTAMP_FALLBACK=true` 后作为兼容输入。这样做的
    主要原因是失败的 external baseline 重跑不能误用旧的成功或失败归档。
    """

    package_dir = stage_package_dir(drive_project_root, workflow_profile, stage_package_id)
    latest = package_dir / "stage_package_latest.zip"
    if latest.exists() and _stage_package_zip_allowed_for_restore(latest, stage_package_id):
        return latest
    if not _env_flag("SSTW_STAGE_PACKAGE_ALLOW_TIMESTAMP_FALLBACK", default=False):
        return None
    candidates = sorted(package_dir.glob("stage_package__*.zip"), key=lambda path: path.stat().st_mtime, reverse=True)
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
    return str(layout.get("workflow_profile") or layout.get("runtime_profile") or "default")


def activate_local_stage_layout(
    layout: Mapping[str, str],
    *,
    notebook_role: str,
    baseline_id: str | None = None,
    local_workspace_root: str | Path | None = None,
) -> dict[str, str]:
    """把 workflow layout 的热路径切换到 Colab 本地磁盘。

    `drive_package_dir` 和 `external_baseline_resource_root` 仍保留在 Drive:
    前者只接收最终 zip, 后者通常是较大的 checkpoint 资源, 不是小文件循环读写热点。
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
    localized["stage_package_dir"] = str(stage_package_dir(drive_project_root, workflow_profile, stage_package_id))

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


def _copy_tree_contents(source: Path, target: Path) -> None:
    """把一个目录内容复制到目标目录, 已存在文件会被覆盖。"""

    if not source.exists():
        return
    for item in source.rglob("*"):
        if not item.is_file():
            continue
        relative = item.relative_to(source)
        output = target / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, output)


def _normalize_legacy_generative_package(layout: Mapping[str, str], extract_root: Path) -> dict[str, Any]:
    """兼容历史 drive_packager 生成的 `<profile>/records/...` zip。

    这一兼容层只在本地解压目录中移动文件, 不回退到 Drive 小文件读取。它使旧 zip
    可以作为新阶段包机制的初始输入。
    """

    run_root = Path(str(layout.get("drive_run_root") or ""))
    profile = str(layout.get("workflow_profile") or run_root.name)
    legacy_root = extract_root / profile
    if legacy_root.is_dir() and (legacy_root / "records").exists():
        _copy_tree_contents(legacy_root, run_root)
        return {
            "legacy_package_normalized": True,
            "legacy_package_source_root": str(legacy_root),
            "legacy_package_target_run_root": str(run_root),
        }
    return {"legacy_package_normalized": False}


def hydrate_stage_package(
    layout: Mapping[str, str],
    stage_package_id: str,
    *,
    required: bool = True,
    allow_legacy_drive_package: bool = False,
    source_workflow_profile: str | None = None,
) -> dict[str, Any]:
    """从 Drive 复制单个阶段 zip 到本地并解压。"""

    drive_project_root = str(layout.get("drive_project_root") or "/content/drive/MyDrive/SSTW")
    target_workflow_profile = str(layout.get("workflow_profile") or layout.get("runtime_profile") or "default")
    source_profile = source_workflow_profile or _stage_package_source_workflow_profile(layout, stage_package_id)
    local_cache_root = Path(str(layout.get("local_stage_package_cache_root") or DEFAULT_LOCAL_PACKAGE_CACHE_ROOT))
    local_workspace_root = Path(str(layout.get("local_stage_workspace_root") or DEFAULT_LOCAL_WORKSPACE_ROOT))
    source_zip = latest_stage_package_zip(drive_project_root, source_profile, stage_package_id)
    source_kind = "stage_package"

    if source_zip is None and allow_legacy_drive_package:
        package_dir = Path(str(layout.get("drive_package_dir") or ""))
        legacy_candidates = sorted(package_dir.glob("*.zip"), key=lambda path: path.stat().st_mtime, reverse=True) if package_dir.exists() else []
        source_zip = legacy_candidates[0] if legacy_candidates else None
        source_kind = "legacy_drive_package" if source_zip else source_kind

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
    shutil.copy2(source_zip, local_zip)

    with zipfile.ZipFile(local_zip) as archive:
        members = _safe_archive_members(archive)
        archive.extractall(local_workspace_root)

    legacy_normalization = (
        _normalize_legacy_generative_package(layout, local_workspace_root)
        if source_kind == "legacy_drive_package"
        else {"legacy_package_normalized": False}
    )
    return {
        "stage_package_id": stage_package_id,
        "stage_package_restore_status": "restored",
        "stage_package_source_kind": source_kind,
        "stage_package_source_workflow_profile": source_profile,
        "stage_package_target_workflow_profile": target_workflow_profile,
        "drive_stage_package_zip": str(source_zip),
        "local_stage_package_zip": str(local_zip),
        "local_stage_workspace_root": str(local_workspace_root),
        "stage_package_entry_count": len(members),
        "stage_package_archive_sha256": _sha256_file(local_zip),
        **legacy_normalization,
    }


def _default_required_stage_packages(layout: Mapping[str, str], notebook_role: str) -> list[str]:
    """返回某个 Notebook role 必须先恢复的阶段包。"""

    profile = str(layout.get("workflow_profile") or "")
    role = sanitize_filename_token(notebook_role)
    if role == "external_baseline_formal_scoring":
        return ["generative_video_runtime_colab"]
    if role == "paper_gate_and_package":
        required = ["generative_video_runtime_colab"]
        if profile != "motion_calibration":
            required.append("motion_threshold_calibration_colab")
        return required
    if role == "generative_video_runtime" and profile != "motion_calibration":
        return ["motion_threshold_calibration_colab"]
    return []


def _default_optional_stage_packages(notebook_role: str, baseline_id: str | None = None) -> list[str]:
    """返回某个 Notebook role 可恢复但不强制存在的阶段包。"""

    role = sanitize_filename_token(notebook_role)
    if role in {"external_baseline_formal_scoring", "paper_gate_and_package"}:
        return [
            stage_package_id_for_notebook("external_baseline_formal_scoring", baseline_id=baseline)
            for baseline in MODERN_EXTERNAL_BASELINE_IDS
            if baseline != baseline_id
        ]
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
    allow_legacy = os.environ.get("SSTW_ALLOW_LEGACY_DRIVE_PACKAGE_RESTORE", "true").lower() == "true"
    rows: list[dict[str, Any]] = []
    for stage_package_id in required:
        rows.append(
            hydrate_stage_package(
                layout,
                stage_package_id,
                required=True,
                allow_legacy_drive_package=allow_legacy and stage_package_id == "generative_video_runtime_colab",
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
        return dict(layout)
    localized = activate_local_stage_layout(layout, notebook_role=notebook_role, baseline_id=baseline_id)
    restore = hydrate_stage_packages_for_notebook(localized, notebook_role=notebook_role, baseline_id=baseline_id)
    restore_path = Path(localized["drive_run_root"]) / "artifacts" / "stage_package_restore_decision.json"
    restore_path.parent.mkdir(parents=True, exist_ok=True)
    restore_path.write_text(json.dumps(restore, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    localized["stage_package_restore_decision_path"] = str(restore_path)
    os.environ["SSTW_STAGE_PACKAGE_HANDOFF_MODE"] = "local_zip"
    os.environ["SSTW_EXTERNAL_BASELINE_OFFICIAL_RESULT_BUNDLE_ROOT"] = localized.get(
        "external_baseline_official_result_bundle_root",
        os.environ.get("SSTW_EXTERNAL_BASELINE_OFFICIAL_RESULT_BUNDLE_ROOT", ""),
    )
    return localized


def _iter_package_sources(layout: Mapping[str, str], *, include_videos: bool) -> list[tuple[Path, str]]:
    """列出阶段包需要归档的本地源目录。"""

    sources: list[tuple[Path, str]] = []
    for key in ("drive_run_root", "drive_dataset_root", "external_baseline_official_result_bundle_root"):
        path_text = str(layout.get(key) or "")
        if not path_text:
            continue
        path = Path(path_text)
        if not path.exists():
            continue
        if not include_videos and key == "drive_run_root":
            sources.append((path, _archive_root_for_layout_path(layout, key)))
            continue
        sources.append((path, _archive_root_for_layout_path(layout, key)))
    return sources


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
    for file_path in sorted(path for path in source_root.rglob("*") if path.is_file()):
        relative = file_path.relative_to(source_root)
        if not include_videos and relative.parts and relative.parts[0] in {"videos", "attacked_videos"}:
            continue
        archive.write(file_path, arcname=f"{archive_root}/{relative.as_posix()}")
        count += 1
    return count


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

    role = sanitize_filename_token(notebook_role)
    is_external_reference = role == "external_baseline_formal_scoring" or stage_package_id.startswith(
        "external_baseline_formal_reference_"
    )
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


def _prune_remote_stage_package_snapshots(remote_package_dir: Path) -> list[str]:
    """删除当前阶段目录下的历史时间戳包, 保留 latest 作为唯一默认入口。"""

    removed: list[str] = []
    for pattern in ("stage_package__*.zip", "stage_package__*_stage_package_manifest.json"):
        for path in sorted(remote_package_dir.glob(pattern)):
            if not path.is_file():
                continue
            path.unlink()
            removed.append(str(path))
    return removed


def _remove_path_if_file(path: Path) -> bool:
    """若文件存在则删除, 用于清理失败阶段遗留的 latest zip。"""

    if path.exists() and path.is_file():
        path.unlink()
        return True
    return False


def _write_manifest_only_stage_package(
    *,
    layout: Mapping[str, str],
    remote_package_dir: Path,
    latest_zip: Path,
    latest_manifest: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """只写阶段 manifest, 不写 zip。

    external baseline 未通过时, 这个 manifest 是阻断记录, 不能被后续门禁当成可恢复
    阶段包使用。该设计避免失败运行把数百 MB 或数 GB 的中间视频继续写入 Drive。
    """

    remote_package_dir.mkdir(parents=True, exist_ok=True)
    removed_timestamp_snapshots = _prune_remote_stage_package_snapshots(remote_package_dir)
    removed_latest_zip = _remove_path_if_file(latest_zip)
    manifest = {
        **manifest,
        "drive_stage_package_zip": "",
        "latest_drive_stage_package_zip": "",
        "stage_package_archive_sha256": "",
        "stage_package_entry_count": 0,
        "stage_package_source_root_count": 0,
        "stage_package_source_roots": [],
        "keep_timestamp_snapshot": False,
        "removed_latest_stage_package_zip": removed_latest_zip,
        "removed_timestamp_stage_package_snapshots": removed_timestamp_snapshots,
        "claim_support_status": "stage_package_blocked_not_claim_evidence",
    }
    latest_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        **manifest,
        "stage_package_manifest_path": str(latest_manifest),
        "latest_stage_package_manifest_path": str(latest_manifest),
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
    stem = f"stage_package__{package_batch_id}"
    local_zip = local_package_dir / f"{stem}.zip"
    local_manifest = local_package_dir / f"{stem}_stage_package_manifest.json"
    remote_zip = remote_package_dir / local_zip.name
    remote_manifest = remote_package_dir / local_manifest.name
    latest_zip = remote_package_dir / "stage_package_latest.zip"
    latest_manifest = remote_package_dir / "stage_package_latest_manifest.json"
    keep_timestamp_snapshot = _env_flag("SSTW_STAGE_PACKAGE_KEEP_TIMESTAMP_SNAPSHOT", default=False)
    prune_timestamp_snapshots = _env_flag("SSTW_STAGE_PACKAGE_PRUNE_TIMESTAMP_SNAPSHOTS", default=True)
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
        "include_videos": bool(include_videos),
        "local_stage_workspace_root": str(layout.get("local_stage_workspace_root") or ""),
        "local_stage_package_zip": str(local_zip),
        "local_stage_package_manifest": str(local_manifest),
        "retention_policy": "latest_only_by_default",
        "keep_timestamp_snapshot": keep_timestamp_snapshot,
        "prune_timestamp_snapshots": prune_timestamp_snapshots,
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
            latest_zip=latest_zip,
            latest_manifest=latest_manifest,
            manifest={**base_manifest, **blocker},
        )

    sources = _iter_package_sources(layout, include_videos=include_videos)
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
        _prune_remote_stage_package_snapshots(remote_package_dir)
        if prune_timestamp_snapshots and not keep_timestamp_snapshot
        else []
    )
    if keep_timestamp_snapshot:
        drive_stage_package_zip = str(remote_zip)
        stage_package_manifest_path = str(remote_manifest)
    else:
        drive_stage_package_zip = str(latest_zip)
        stage_package_manifest_path = str(latest_manifest)

    manifest = {
        **base_manifest,
        "stage_package_publish_status": "published",
        "drive_stage_package_zip": drive_stage_package_zip,
        "latest_drive_stage_package_zip": str(latest_zip),
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
    local_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if keep_timestamp_snapshot:
        shutil.copy2(local_zip, remote_zip)
        shutil.copy2(local_manifest, remote_manifest)
    shutil.copy2(local_zip, latest_zip)
    shutil.copy2(local_manifest, latest_manifest)
    return {
        **manifest,
        "stage_package_publish_status": "published",
        "stage_package_manifest_path": stage_package_manifest_path,
        "latest_stage_package_manifest_path": str(latest_manifest),
    }
