"""将 Wan2.1 Flow adapter preflight 结果打包到 Google Drive。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import zipfile


DEFAULT_DRIVE_PROJECT_ROOT = "/content/drive/MyDrive/SSTW"


def _write_tree_to_archive(archive: zipfile.ZipFile, run_root: Path, tree_path: Path) -> None:
    """将一个结果子目录写入 zip, 不存在的目录会被跳过。"""
    if not tree_path.exists():
        return
    for file_path in sorted(path for path in tree_path.rglob("*") if path.is_file()):
        archive.write(file_path, arcname=f"{run_root.name}/{file_path.relative_to(run_root).as_posix()}")


def _read_json_if_exists(path: Path) -> dict:
    """读取可选 JSON 文件, 不存在时返回空对象。"""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def package_wan21_flow_adapter_preflight_run(run_root: str | Path, drive_package_dir: str | Path) -> dict:
    """打包 Wan2.1 Flow adapter preflight run_root。

    该函数属于通用工程写法。它只复制和压缩已经由 preflight runtime 写出的 governed
    records 与 artifacts, 不创建新的实验结论, 因此适合 Colab 断开前固化结果。
    """
    run_root_path = Path(run_root)
    if not run_root_path.exists():
        raise FileNotFoundError(run_root_path)
    package_dir = Path(drive_package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive_path = package_dir / f"{run_root_path.name}_{timestamp}.zip"
    package_manifest_path = package_dir / f"{run_root_path.name}_{timestamp}_package_manifest.json"
    decision = _read_json_if_exists(run_root_path / "artifacts" / "wan21_flow_adapter_preflight_decision.json")

    with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for subdir_name in ("records", "tables", "reports", "thresholds", "artifacts"):
            _write_tree_to_archive(archive, run_root_path, run_root_path / subdir_name)

    package_manifest = {
        "artifact_id": "wan21_flow_adapter_preflight_drive_package",
        "artifact_type": "package_manifest",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root_path),
        "drive_package_dir": str(package_dir),
        "archive_path": str(archive_path),
        "package_manifest_path": str(package_manifest_path),
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
    parser.add_argument("--drive-package-dir", default=f"{DEFAULT_DRIVE_PROJECT_ROOT}/packages/wan21_flow_adapter_preflight")
    args = parser.parse_args()
    payload = package_wan21_flow_adapter_preflight_run(args.run_root, args.drive_package_dir)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
