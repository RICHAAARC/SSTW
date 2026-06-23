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
    """读取可选 JSON 文件, 不存在时返回空对象。

    Google Drive 本地映射中的文件可能被 Windows PowerShell 以 UTF-8 BOM 重写。
    使用 `utf-8-sig` 可以同时读取有 BOM 和无 BOM 的 JSON。
    """
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _effective_mechanism_decision_summary(runtime_decision: dict, postprocess_decision: dict, pilot_gate_decision: dict) -> dict:
    """汇总 package 级机制状态, 避免把 runtime 原始 FAIL 与后处理 PASS 混在同一个字段里。

    runtime 直接判定只说明生成阶段尚未完成机制证明。small-scale pilot 或 postprocess
    后处理通过时, package 级 summary 应显式给出 effective 结果和来源, 同时保留
    runtime_mechanism_decision 供审计回溯。
    """
    runtime_mechanism_decision = runtime_decision.get("mechanism_decision")
    postprocess_mechanism_decision = postprocess_decision.get("mechanism_decision")
    pilot_gate_status = pilot_gate_decision.get("pilot_gate_decision")
    if pilot_gate_status == "PASS":
        effective_decision = "PASS"
        decision_source = "small_scale_claim_pilot_gate"
    elif postprocess_mechanism_decision == "PASS":
        effective_decision = "PASS"
        decision_source = "postprocess_mechanism_artifact"
    else:
        effective_decision = runtime_mechanism_decision
        decision_source = "runtime_mechanism_artifact"
    return {
        "mechanism_decision": effective_decision,
        "effective_mechanism_decision": effective_decision,
        "mechanism_decision_source": decision_source,
        "runtime_mechanism_decision": runtime_mechanism_decision,
        "postprocess_mechanism_decision": postprocess_mechanism_decision,
    }


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
    runtime_detection_decision_path = run_root_path / "artifacts" / "runtime_detection_decision.json"
    external_baseline_decision_path = run_root_path / "artifacts" / "external_baseline_status_decision.json"
    external_baseline_comparison_decision_path = run_root_path / "artifacts" / "external_baseline_comparison_decision.json"
    internal_ablation_decision_path = run_root_path / "artifacts" / "validation_internal_ablation_decision.json"
    adaptive_attack_decision_path = run_root_path / "artifacts" / "adaptive_attack_decision.json"
    claim3_downgrade_decision_path = run_root_path / "artifacts" / "claim3_downgrade_decision.json"
    confidence_interval_decision_path = run_root_path / "artifacts" / "statistical_confidence_interval_decision.json"
    artifact_rebuild_decision_path = run_root_path / "artifacts" / "validation_artifact_rebuild_dry_run_decision.json"
    validation_scale_decision_path = run_root_path / "artifacts" / "validation_scale_gate_decision.json"
    motion_threshold_calibration_decision_path = run_root_path / "artifacts" / "motion_threshold_calibration_decision.json"
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
    runtime_detection_decision = _read_json_if_exists(runtime_detection_decision_path)
    external_baseline_decision = _read_json_if_exists(external_baseline_decision_path)
    external_baseline_comparison_decision = _read_json_if_exists(external_baseline_comparison_decision_path)
    internal_ablation_decision = _read_json_if_exists(internal_ablation_decision_path)
    adaptive_attack_decision = _read_json_if_exists(adaptive_attack_decision_path)
    claim3_downgrade_decision = _read_json_if_exists(claim3_downgrade_decision_path)
    confidence_interval_decision = _read_json_if_exists(confidence_interval_decision_path)
    artifact_rebuild_decision = _read_json_if_exists(artifact_rebuild_decision_path)
    validation_scale_decision = _read_json_if_exists(validation_scale_decision_path)
    motion_threshold_calibration_decision = _read_json_if_exists(motion_threshold_calibration_decision_path)
    generation_manifest = _read_json_if_exists(generation_manifest_path)
    mechanism_summary = _effective_mechanism_decision_summary(decision, postprocess_decision, pilot_gate_decision)
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
            **mechanism_summary,
            "mechanism_postprocess_decision": postprocess_decision.get("mechanism_postprocess_decision"),
            "postprocess_formal_claim_status": postprocess_decision.get("details", {}).get("formal_claim_status"),
            "formal_visual_motion_ready": formal_metric_decision.get("formal_visual_motion_ready"),
            "formal_semantic_ready": formal_metric_decision.get("formal_semantic_ready"),
            "formal_metric_claim_status": formal_metric_decision.get("formal_metric_claim_status"),
            "small_scale_pilot_gate_decision": pilot_gate_decision.get("pilot_gate_decision"),
            "small_scale_pilot_claim_support_status": pilot_gate_decision.get("claim_support_status"),
            "small_scale_pilot_missing_requirement_count": pilot_gate_decision.get("pilot_missing_requirement_count"),
            "formal_motion_claim_status": pilot_gate_decision.get("formal_motion_claim_status"),
            "motion_claim_eligible_generation_count": pilot_gate_decision.get("motion_claim_eligible_generation_count"),
            "motion_claim_excluded_generation_count": pilot_gate_decision.get("motion_claim_excluded_generation_count"),
            "motion_claim_runtime_attack_ready_count": pilot_gate_decision.get("runtime_attack_ready_count"),
            "motion_claim_runtime_detection_ready_count": pilot_gate_decision.get("runtime_detection_ready_count"),
            "small_scale_pilot_matrix_postprocess_decision": pilot_matrix_decision.get("pilot_matrix_postprocess_decision"),
            "small_scale_pilot_matrix_record_count": pilot_matrix_decision.get("pilot_matrix_record_count"),
            "runtime_attack_decision": runtime_attack_decision.get("runtime_attack_decision"),
            "runtime_attack_record_count": runtime_attack_decision.get("runtime_attack_record_count"),
            "runtime_attack_ready_count": runtime_attack_decision.get("runtime_attack_ready_count"),
            "runtime_detection_decision": runtime_detection_decision.get("runtime_detection_decision"),
            "runtime_detection_record_count": runtime_detection_decision.get("runtime_detection_record_count"),
            "runtime_detection_ready_count": runtime_detection_decision.get("runtime_detection_ready_count"),
            "external_baseline_status_decision": external_baseline_decision.get("external_baseline_status_decision"),
            "external_baseline_record_count": external_baseline_decision.get("external_baseline_record_count"),
            "modern_external_baseline_record_count": external_baseline_decision.get("modern_external_baseline_record_count"),
            "modern_external_baseline_main_comparison_ready_count": external_baseline_decision.get("modern_external_baseline_main_comparison_ready_count"),
            "external_baseline_claim_support_status": external_baseline_decision.get("external_baseline_claim_support_status"),
            "external_baseline_comparison_decision": external_baseline_comparison_decision.get("external_baseline_comparison_decision"),
            "external_baseline_comparison_record_count": external_baseline_comparison_decision.get("external_baseline_comparison_record_count"),
            "external_baseline_comparison_ready_count": external_baseline_comparison_decision.get("external_baseline_comparison_ready_count"),
            "external_baseline_measured_adapter_count": external_baseline_comparison_decision.get("external_baseline_measured_adapter_count"),
            "external_baseline_comparison_status": external_baseline_comparison_decision.get("external_baseline_comparison_status"),
            "external_baseline_comparison_table_status": external_baseline_comparison_decision.get("external_baseline_comparison_table_status"),
            "validation_internal_ablation_decision": internal_ablation_decision.get("validation_internal_ablation_decision"),
            "validation_internal_ablation_record_count": internal_ablation_decision.get("internal_ablation_record_count"),
            "adaptive_attack_decision": adaptive_attack_decision.get("adaptive_attack_decision"),
            "adaptive_attack_record_count": adaptive_attack_decision.get("adaptive_attack_record_count"),
            "adaptive_robustness_claim_allowed": adaptive_attack_decision.get("adaptive_robustness_claim_allowed"),
            "claim3_downgrade_decision": claim3_downgrade_decision.get("claim3_downgrade_decision"),
            "claim3_downgraded": claim3_downgrade_decision.get("claim3_downgraded"),
            "claim3_full_support_allowed": claim3_downgrade_decision.get("claim3_full_support_allowed"),
            "replay_or_sketch_status": claim3_downgrade_decision.get("replay_or_sketch_status"),
            "statistical_confidence_interval_decision": confidence_interval_decision.get("statistical_confidence_interval_decision"),
            "statistical_confidence_interval_total_count": confidence_interval_decision.get("ci_total_count"),
            "validation_artifact_rebuild_dry_run_decision": artifact_rebuild_decision.get("validation_artifact_rebuild_dry_run_decision"),
            "validation_artifact_rebuild_missing_count": artifact_rebuild_decision.get("artifact_rebuild_missing_count"),
            "validation_scale_gate_decision": validation_scale_decision.get("validation_scale_gate_decision"),
            "validation_scale_claim_support_status": validation_scale_decision.get("claim_support_status"),
            "validation_missing_requirement_count": validation_scale_decision.get("validation_missing_requirement_count"),
            "validation_generation_record_count": validation_scale_decision.get("validation_generation_record_count"),
            "validation_prompt_count": validation_scale_decision.get("validation_prompt_count"),
            "validation_seed_per_prompt_min": validation_scale_decision.get("validation_seed_per_prompt_min"),
            "motion_threshold_calibration_decision": motion_threshold_calibration_decision.get("motion_threshold_calibration_decision"),
            "motion_threshold_id": motion_threshold_calibration_decision.get("motion_threshold_id"),
            "motion_threshold_source_split": motion_threshold_calibration_decision.get("motion_threshold_source_split"),
            "motion_threshold_calibration_required": motion_threshold_calibration_decision.get("motion_threshold_calibration_required"),
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
