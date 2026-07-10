"""将 Wan2.1 Flow adapter preflight 结果打包到 Google Drive。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.protocol.package_naming import build_package_batch_id, current_short_commit, current_utc_time_for_filename
from scripts.package_results.drive_package_paths import (
    DEFAULT_DRIVE_PROJECT_ROOT,
    archive_run_root_for_stage,
    build_packager_file_stem,
    packager_manifest_filename,
    resolve_stage_package_output_dir,
)


DEFAULT_WORKFLOW_PROFILE = "wan21_flow_adapter_preflight"
DEFAULT_STAGE_PACKAGE_ID = "wan21_flow_adapter_preflight_colab"


def _write_tree_to_archive(archive: zipfile.ZipFile, run_root: Path, tree_path: Path, archive_run_root: str) -> None:
    """将一个结果子目录写入 zip, 不存在的目录会被跳过。"""
    if not tree_path.exists():
        return
    for file_path in sorted(path for path in tree_path.rglob("*") if path.is_file()):
        archive.write(file_path, arcname=f"{archive_run_root}/{file_path.relative_to(run_root).as_posix()}")


def _read_json_if_exists(path: Path) -> dict:
    """读取可选 JSON 文件, 不存在时返回空对象。"""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def package_wan21_flow_adapter_preflight_run(
    run_root: str | Path,
    drive_package_dir: str | Path,
    workflow_profile: str | None = None,
    stage_package_id: str | None = None,
) -> dict:
    """打包 Wan2.1 Flow adapter preflight run_root。

    该函数属于通用工程写法。它只复制和压缩已经由 preflight runtime 写出的 governed
    records 与 artifacts, 不创建新的实验结论, 因此适合 Colab 断开前固化结果。
    """
    run_root_path = Path(run_root)
    if not run_root_path.exists():
        raise FileNotFoundError(run_root_path)
    package_dir = Path(drive_package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)
    package_utc_time = current_utc_time_for_filename()
    package_short_commit = current_short_commit()
    package_batch_id = build_package_batch_id(package_utc_time, package_short_commit)
    stage_package_naming = bool(workflow_profile and stage_package_id)
    package_file_stem = build_packager_file_stem(
        run_root_path.name,
        package_utc_time,
        package_short_commit,
        workflow_profile=workflow_profile,
        stage_package_id=stage_package_id,
    )
    archive_path = package_dir / f"{package_file_stem}.zip"
    package_manifest_path = package_dir / packager_manifest_filename(
        package_file_stem,
        stage_package_naming=stage_package_naming,
    )
    archive_run_root = archive_run_root_for_stage(
        run_root_path.name,
        workflow_profile=workflow_profile,
        stage_package_id=stage_package_id,
    )
    decision = _read_json_if_exists(run_root_path / "artifacts" / "wan21_flow_adapter_preflight_decision.json")

    with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for subdir_name in ("records", "tables", "reports", "thresholds", "artifacts"):
            _write_tree_to_archive(archive, run_root_path, run_root_path / subdir_name, archive_run_root)

    package_manifest = {
        "artifact_id": "wan21_flow_adapter_preflight_drive_package",
        "artifact_type": "package_manifest",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root_path),
        "drive_package_dir": str(package_dir),
        "archive_path": str(archive_path),
        "package_manifest_path": str(package_manifest_path),
        "package_batch_id": package_batch_id,
        "package_utc_time": package_utc_time,
        "package_short_commit": package_short_commit,
        "workflow_profile": workflow_profile or "",
        "stage_package_id": stage_package_id or "",
        "archive_run_root": archive_run_root,
        "input_paths": [str(run_root_path)],
        "output_paths": [str(archive_path), str(package_manifest_path)],
        "decision_summary": {
            "stage_id": decision.get("stage_id"),
            "adapter_preflight_decision": decision.get("adapter_preflight_decision"),
            "model_load_status": decision.get("model_load_status"),
            "callback_latent_capture_status": decision.get("callback_latent_capture_status"),
            "time_grid_capture_status": decision.get("time_grid_capture_status"),
            "sampler_signature_status": decision.get("sampler_signature_status"),
            "velocity_proxy_status": decision.get("velocity_proxy_status"),
            "gpu_name": decision.get("gpu_name"),
            "gpu_memory_mb": decision.get("gpu_memory_mb"),
        },
    }
    package_manifest_path.write_text(json.dumps(package_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return package_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="打包 Wan2.1 Flow adapter preflight 结果到 Google Drive。")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--drive-project-root", default=os.environ.get("SSTW_DRIVE_PROJECT_ROOT", DEFAULT_DRIVE_PROJECT_ROOT))
    parser.add_argument("--workflow-profile", default=os.environ.get("SSTW_WORKFLOW_PROFILE", DEFAULT_WORKFLOW_PROFILE))
    parser.add_argument("--stage-package-id", default=os.environ.get("SSTW_STAGE_PACKAGE_ID", DEFAULT_STAGE_PACKAGE_ID))
    parser.add_argument("--drive-package-dir", default=None)
    args = parser.parse_args()
    drive_package_dir = args.drive_package_dir or resolve_stage_package_output_dir(
        args.drive_project_root,
        args.workflow_profile,
        args.stage_package_id,
    )
    payload = package_wan21_flow_adapter_preflight_run(
        args.run_root,
        drive_package_dir,
        workflow_profile=args.workflow_profile,
        stage_package_id=args.stage_package_id,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
