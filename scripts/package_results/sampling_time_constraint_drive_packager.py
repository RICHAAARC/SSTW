"""将 B6 sampling-time constraint Colab probe 结果打包到 Google Drive。"""

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

from main.protocol.package_naming import build_package_batch_id, current_short_commit, current_utc_time_for_filename
from scripts.package_results.drive_package_paths import (
    DEFAULT_DRIVE_PROJECT_ROOT,
    archive_run_root_for_stage,
    build_packager_file_stem,
    packager_manifest_filename,
    resolve_stage_package_output_dir,
)

DEFAULT_WORKFLOW_PROFILE = "sampling_time_constraint"
DEFAULT_STAGE_PACKAGE_ID = "sampling_time_constraint_colab"


def _write_tree_to_archive(archive: zipfile.ZipFile, run_root: Path, tree_path: Path, archive_run_root: str) -> None:
    """将结果子目录写入 zip, 不存在的目录会跳过。"""
    if not tree_path.exists():
        return
    for file_path in sorted(path for path in tree_path.rglob("*") if path.is_file()):
        archive.write(file_path, arcname=f"{archive_run_root}/{file_path.relative_to(run_root).as_posix()}")


def _read_json_if_exists(path: Path) -> dict:
    """读取可选 JSON 文件。"""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def package_sampling_time_constraint_colab_run(
    run_root: str | Path,
    drive_package_dir: str | Path,
    include_videos: bool = True,
    workflow_profile: str | None = None,
    stage_package_id: str | None = None,
) -> dict:
    """打包 B6 Colab run_root。"""
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
    runtime_decision = _read_json_if_exists(run_root_path / "artifacts" / "sampling_time_constraint_colab_runtime_decision.json")
    postprocess_decision = _read_json_if_exists(run_root_path / "artifacts" / "sampling_time_constraint_colab_postprocess_decision.json")
    formal_metric_decision = _read_json_if_exists(run_root_path / "artifacts" / "formal_quality_motion_semantic_decision.json")
    generation_manifest = _read_json_if_exists(run_root_path / "artifacts" / "generation_manifest.json")

    with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for subdir_name in ("records", "tables", "reports", "thresholds", "artifacts"):
            _write_tree_to_archive(archive, run_root_path, run_root_path / subdir_name, archive_run_root)
        if include_videos:
            _write_tree_to_archive(archive, run_root_path, run_root_path / "videos", archive_run_root)

    package_manifest = {
        "artifact_id": "sampling_time_constraint_colab_drive_package",
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
        "include_videos": include_videos,
        "input_paths": [str(run_root_path)],
        "output_paths": [str(archive_path), str(package_manifest_path)],
        "decision_summary": {
            "runtime_stage_id": runtime_decision.get("stage_id"),
            "implementation_decision": runtime_decision.get("implementation_decision"),
            "runtime_mechanism_decision": runtime_decision.get("mechanism_decision"),
            "postprocess_stage_id": postprocess_decision.get("stage_id"),
            "mechanism_postprocess_decision": postprocess_decision.get("mechanism_postprocess_decision"),
            "postprocess_mechanism_decision": postprocess_decision.get("mechanism_decision"),
            "postprocess_formal_claim_status": postprocess_decision.get("details", {}).get("formal_claim_status"),
            "formal_visual_motion_ready": formal_metric_decision.get("formal_visual_motion_ready"),
            "formal_semantic_ready": formal_metric_decision.get("formal_semantic_ready"),
            "formal_metric_claim_status": formal_metric_decision.get("formal_metric_claim_status"),
        },
        "generation_manifest_status": "present" if generation_manifest else "missing",
    }
    package_manifest_path.write_text(json.dumps(package_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return package_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="打包 B6 sampling-time constraint Colab probe。")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--drive-project-root", default=os.environ.get("SSTW_DRIVE_PROJECT_ROOT", DEFAULT_DRIVE_PROJECT_ROOT))
    parser.add_argument("--workflow-profile", default=os.environ.get("SSTW_WORKFLOW_PROFILE", DEFAULT_WORKFLOW_PROFILE))
    parser.add_argument("--stage-package-id", default=os.environ.get("SSTW_STAGE_PACKAGE_ID", DEFAULT_STAGE_PACKAGE_ID))
    parser.add_argument("--drive-package-dir", default=None)
    parser.add_argument("--exclude-videos", action="store_true")
    args = parser.parse_args()
    drive_package_dir = args.drive_package_dir or resolve_stage_package_output_dir(
        args.drive_project_root,
        args.workflow_profile,
        args.stage_package_id,
    )
    payload = package_sampling_time_constraint_colab_run(
        args.run_root,
        drive_package_dir,
        include_videos=not args.exclude_videos,
        workflow_profile=args.workflow_profile,
        stage_package_id=args.stage_package_id,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
