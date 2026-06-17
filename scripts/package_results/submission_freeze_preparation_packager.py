"""打包 submission freeze preparation governed artifacts。"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import zipfile


PACKAGE_SUBDIRS = ("records", "tables", "reports", "artifacts")


def _sha256_file(path: Path) -> str:
    """计算文件 sha256 摘要。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_package_files(run_root: Path) -> list[Path]:
    """列出允许进入 submission preparation package 的 governed artifact 文件。"""
    files: list[Path] = []
    for subdir_name in PACKAGE_SUBDIRS:
        subdir = run_root / subdir_name
        if not subdir.exists():
            continue
        files.extend(sorted(path for path in subdir.rglob("*") if path.is_file()))
    return files


def build_submission_freeze_preparation_package(
    run_root: str | Path,
    package_dir: str | Path,
    package_name: str = "submission_freeze_preparation_package",
) -> dict:
    """构建 submission freeze preparation package 并返回 package manifest。

    该函数属于通用工程写法。它只复制 governed artifacts, 不把 `outputs/` 中的其他临时运行目录、审计缓存或视频大文件混入 release 候选包。
    """
    run_root = Path(run_root)
    package_dir = Path(package_dir)
    if not run_root.exists():
        raise FileNotFoundError(run_root)
    package_dir.mkdir(parents=True, exist_ok=True)

    files = _iter_package_files(run_root)
    if not files:
        raise RuntimeError(f"没有可打包的 governed artifacts: {run_root}")

    archive_path = package_dir / f"{package_name}.zip"
    package_manifest_path = package_dir / f"{package_name}_manifest.json"
    file_records = []
    with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in files:
            relative = file_path.relative_to(run_root).as_posix()
            archive_name = f"{run_root.name}/{relative}"
            archive.write(file_path, archive_name)
            file_records.append({
                "relative_path": relative,
                "archive_name": archive_name,
                "size_bytes": file_path.stat().st_size,
                "sha256": _sha256_file(file_path),
            })

    package_manifest = {
        "artifact_id": "submission_freeze_preparation_package_manifest",
        "artifact_type": "package_manifest",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root),
        "package_dir": str(package_dir),
        "archive_path": str(archive_path),
        "package_manifest_path": str(package_manifest_path),
        "package_file_count": len(file_records),
        "package_size_bytes": archive_path.stat().st_size,
        "package_digest": _sha256_file(archive_path),
        "included_subdirs": list(PACKAGE_SUBDIRS),
        "excluded_asset_policy": "exclude_runtime_cache_audit_reports_outputs_siblings_and_large_video_assets",
        "file_records": file_records,
        "rebuild_command": "python scripts/package_results/submission_freeze_preparation_packager.py",
    }
    package_manifest_path.write_text(json.dumps(package_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return package_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="打包 submission freeze preparation governed artifacts。")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--package-dir", required=True)
    parser.add_argument("--package-name", default="submission_freeze_preparation_package")
    args = parser.parse_args()
    payload = build_submission_freeze_preparation_package(args.run_root, args.package_dir, args.package_name)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
