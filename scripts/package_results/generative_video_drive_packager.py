"""将 B5 Colab 生成式视频探测结果打包到 Google Drive。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main.protocol.package_naming import build_package_batch_id, build_package_file_stem, current_short_commit, current_utc_time_for_filename


DEFAULT_DRIVE_PROJECT_ROOT = "/content/drive/MyDrive/SSTW"


def _write_tree_to_archive(archive: zipfile.ZipFile, run_root: Path, tree_path: Path) -> None:
    """将一个结果子目录写入 zip, 不存在的子目录会被跳过。"""
    if not tree_path.exists():
        return
    for file_path in sorted(path for path in tree_path.rglob("*") if path.is_file()):
        archive.write(file_path, arcname=f"{run_root.name}/{file_path.relative_to(run_root).as_posix()}")


def _read_json_if_exists(path: Path) -> dict:
    """读取可选 JSON 文件, 不存在时返回空对象。"""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def package_generative_video_colab_run(
    run_root: str | Path,
    drive_package_dir: str | Path,
    include_videos: bool = True,
) -> dict:
    """将 B5 Colab run_root 打包为 zip 和 sidecar manifest。

    该函数只移动与打包已有 governed outputs, 不生成新的实验结果。
    """
    run_root_path = Path(run_root)
    if not run_root_path.exists():
        raise FileNotFoundError(run_root_path)
    package_dir = Path(drive_package_dir)
    package_dir.mkdir(parents=True, exist_ok=True)
    package_utc_time = current_utc_time_for_filename()
    package_short_commit = current_short_commit()
    package_batch_id = build_package_batch_id(package_utc_time, package_short_commit)
    package_file_stem = build_package_file_stem(run_root_path.name, package_utc_time, package_short_commit)
    archive_path = package_dir / f"{package_file_stem}.zip"
    package_manifest_path = package_dir / f"{package_file_stem}_package_manifest.json"
    decision_path = run_root_path / "artifacts" / "generative_video_colab_runtime_decision.json"
    postprocess_decision_path = run_root_path / "artifacts" / "generative_video_mechanism_postprocess_decision.json"
    formal_metric_decision_path = run_root_path / "artifacts" / "formal_quality_motion_semantic_decision.json"
    pilot_gate_decision_path = run_root_path / "artifacts" / "small_scale_claim_pilot_gate_decision.json"
    pilot_matrix_decision_path = run_root_path / "artifacts" / "small_scale_claim_pilot_matrix_decision.json"
    runtime_attack_decision_path = run_root_path / "artifacts" / "runtime_attack_decision.json"
    generation_manifest_path = run_root_path / "artifacts" / "generation_manifest.json"

    with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for subdir_name in ("records", "tables", "reports", "thresholds", "artifacts"):
            _write_tree_to_archive(archive, run_root_path, run_root_path / subdir_name)
        if include_videos:
            _write_tree_to_archive(archive, run_root_path, run_root_path / "videos")
            _write_tree_to_archive(archive, run_root_path, run_root_path / "attacked_videos")

    decision = _read_json_if_exists(decision_path)
    postprocess_decision = _read_json_if_exists(postprocess_decision_path)
    formal_metric_decision = _read_json_if_exists(formal_metric_decision_path)
    pilot_gate_decision = _read_json_if_exists(pilot_gate_decision_path)
    pilot_matrix_decision = _read_json_if_exists(pilot_matrix_decision_path)
    runtime_attack_decision = _read_json_if_exists(runtime_attack_decision_path)
    generation_manifest = _read_json_if_exists(generation_manifest_path)
    package_manifest = {
        "artifact_id": "generative_video_colab_drive_package",
        "artifact_type": "package_manifest",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root_path),
        "drive_package_dir": str(package_dir),
        "archive_path": str(archive_path),
        "package_manifest_path": str(package_manifest_path),
        "package_batch_id": package_batch_id,
        "package_utc_time": package_utc_time,
        "package_short_commit": package_short_commit,
        "include_videos": include_videos,
        "input_paths": [str(run_root_path)],
        "output_paths": [str(archive_path), str(package_manifest_path)],
        "decision_summary": {
            "stage_id": decision.get("stage_id"),
            "implementation_decision": decision.get("implementation_decision"),
            "mechanism_decision": decision.get("mechanism_decision"),
            "mechanism_postprocess_decision": postprocess_decision.get("mechanism_postprocess_decision"),
            "postprocess_mechanism_decision": postprocess_decision.get("mechanism_decision"),
            "postprocess_formal_claim_status": postprocess_decision.get("details", {}).get("formal_claim_status"),
            "formal_visual_motion_ready": formal_metric_decision.get("formal_visual_motion_ready"),
            "formal_semantic_ready": formal_metric_decision.get("formal_semantic_ready"),
            "formal_metric_claim_status": formal_metric_decision.get("formal_metric_claim_status"),
            "small_scale_pilot_gate_decision": pilot_gate_decision.get("pilot_gate_decision"),
            "small_scale_pilot_claim_support_status": pilot_gate_decision.get("claim_support_status"),
            "small_scale_pilot_missing_requirement_count": pilot_gate_decision.get("pilot_missing_requirement_count"),
            "small_scale_pilot_matrix_postprocess_decision": pilot_matrix_decision.get("pilot_matrix_postprocess_decision"),
            "small_scale_pilot_matrix_record_count": pilot_matrix_decision.get("pilot_matrix_record_count"),
            "runtime_attack_decision": runtime_attack_decision.get("runtime_attack_decision"),
            "runtime_attack_record_count": runtime_attack_decision.get("runtime_attack_record_count"),
            "runtime_attack_ready_count": runtime_attack_decision.get("runtime_attack_ready_count"),
        },
        "generation_manifest_status": "present" if generation_manifest else "missing",
    }
    package_manifest_path.write_text(json.dumps(package_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return package_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="将 B5 Colab 生成式视频探测结果打包到 Google Drive。")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--drive-package-dir", default=f"{DEFAULT_DRIVE_PROJECT_ROOT}/packages/generative_video_model_probe")
    parser.add_argument("--exclude-videos", action="store_true")
    args = parser.parse_args()
    payload = package_generative_video_colab_run(args.run_root, args.drive_package_dir, include_videos=not args.exclude_videos)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
