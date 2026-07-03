"""将 B5 Colab 生成式视频探测结果打包到 Google Drive。"""

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


DEFAULT_WORKFLOW_PROFILE = "validation_scale"
DEFAULT_STAGE_PACKAGE_ID = "generative_video_runtime_colab"


def _write_tree_to_archive(archive: zipfile.ZipFile, run_root: Path, tree_path: Path, archive_run_root: str) -> None:
    """将一个结果子目录写入 zip, 不存在的子目录会被跳过。"""
    if not tree_path.exists():
        return
    for file_path in sorted(path for path in tree_path.rglob("*") if path.is_file()):
        archive.write(file_path, arcname=f"{archive_run_root}/{file_path.relative_to(run_root).as_posix()}")


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
    workflow_profile: str | None = None,
    stage_package_id: str | None = None,
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
    decision_path = run_root_path / "artifacts" / "generative_video_colab_runtime_decision.json"
    postprocess_decision_path = run_root_path / "artifacts" / "generative_video_mechanism_postprocess_decision.json"
    formal_metric_decision_path = run_root_path / "artifacts" / "formal_quality_motion_semantic_decision.json"
    pilot_gate_decision_path = run_root_path / "artifacts" / "small_scale_claim_pilot_gate_decision.json"
    pilot_matrix_decision_path = run_root_path / "artifacts" / "small_scale_claim_pilot_matrix_decision.json"
    runtime_attack_decision_path = run_root_path / "artifacts" / "runtime_attack_decision.json"
    runtime_detection_decision_path = run_root_path / "artifacts" / "runtime_detection_decision.json"
    sstw_measured_formal_decision_path = run_root_path / "artifacts" / "sstw_measured_formal_decision.json"
    external_baseline_decision_path = run_root_path / "artifacts" / "external_baseline_status_decision.json"
    external_baseline_comparison_decision_path = run_root_path / "artifacts" / "external_baseline_comparison_decision.json"
    formal_method_baseline_comparison_decision_path = run_root_path / "artifacts" / "formal_method_baseline_comparison_decision.json"
    external_baseline_execution_manifest_path = run_root_path / "artifacts" / "external_baseline_execution_manifest.json"
    internal_ablation_decision_path = run_root_path / "artifacts" / "validation_internal_ablation_decision.json"
    adaptive_attack_decision_path = run_root_path / "artifacts" / "adaptive_attack_decision.json"
    replay_and_sketch_decision_path = run_root_path / "artifacts" / "replay_and_sketch_gate_decision.json"
    claim3_downgrade_decision_path = run_root_path / "artifacts" / "claim3_downgrade_decision.json"
    confidence_interval_decision_path = run_root_path / "artifacts" / "statistical_confidence_interval_decision.json"
    pilot_paper_decision_path = run_root_path / "artifacts" / "pilot_paper_gate_decision.json"
    artifact_rebuild_decision_path = run_root_path / "artifacts" / "validation_artifact_rebuild_dry_run_decision.json"
    validation_scale_decision_path = run_root_path / "artifacts" / "validation_scale_gate_decision.json"
    motion_threshold_calibration_decision_path = run_root_path / "artifacts" / "motion_threshold_calibration_decision.json"
    generation_manifest_path = run_root_path / "artifacts" / "generation_manifest.json"

    with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for subdir_name in ("records", "tables", "reports", "thresholds", "artifacts"):
            _write_tree_to_archive(archive, run_root_path, run_root_path / subdir_name, archive_run_root)
        if include_videos:
            _write_tree_to_archive(archive, run_root_path, run_root_path / "videos", archive_run_root)
            _write_tree_to_archive(archive, run_root_path, run_root_path / "attacked_videos", archive_run_root)

    decision = _read_json_if_exists(decision_path)
    postprocess_decision = _read_json_if_exists(postprocess_decision_path)
    formal_metric_decision = _read_json_if_exists(formal_metric_decision_path)
    pilot_gate_decision = _read_json_if_exists(pilot_gate_decision_path)
    pilot_matrix_decision = _read_json_if_exists(pilot_matrix_decision_path)
    runtime_attack_decision = _read_json_if_exists(runtime_attack_decision_path)
    runtime_detection_decision = _read_json_if_exists(runtime_detection_decision_path)
    sstw_measured_formal_decision = _read_json_if_exists(sstw_measured_formal_decision_path)
    external_baseline_decision = _read_json_if_exists(external_baseline_decision_path)
    external_baseline_comparison_decision = _read_json_if_exists(external_baseline_comparison_decision_path)
    formal_method_baseline_comparison_decision = _read_json_if_exists(formal_method_baseline_comparison_decision_path)
    external_baseline_execution_manifest = _read_json_if_exists(external_baseline_execution_manifest_path)
    internal_ablation_decision = _read_json_if_exists(internal_ablation_decision_path)
    adaptive_attack_decision = _read_json_if_exists(adaptive_attack_decision_path)
    replay_and_sketch_decision = _read_json_if_exists(replay_and_sketch_decision_path)
    claim3_downgrade_decision = _read_json_if_exists(claim3_downgrade_decision_path)
    confidence_interval_decision = _read_json_if_exists(confidence_interval_decision_path)
    pilot_paper_decision = _read_json_if_exists(pilot_paper_decision_path)
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
        "workflow_profile": workflow_profile or "",
        "stage_package_id": stage_package_id or "",
        "archive_run_root": archive_run_root,
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
            "sstw_measured_formal_decision": sstw_measured_formal_decision.get("sstw_measured_formal_decision"),
            "sstw_measured_formal_record_count": sstw_measured_formal_decision.get("sstw_measured_formal_record_count"),
            "sstw_measured_formal_score_mean": sstw_measured_formal_decision.get("sstw_measured_formal_score_mean"),
            "sstw_measured_formal_detectable_rate": sstw_measured_formal_decision.get("sstw_measured_formal_detectable_rate"),
            "external_baseline_status_decision": external_baseline_decision.get("external_baseline_status_decision"),
            "external_baseline_record_count": external_baseline_decision.get("external_baseline_record_count"),
            "modern_external_baseline_record_count": external_baseline_decision.get("modern_external_baseline_record_count"),
            "modern_external_baseline_main_comparison_ready_count": external_baseline_decision.get("modern_external_baseline_main_comparison_ready_count"),
            "external_baseline_claim_support_status": external_baseline_decision.get("external_baseline_claim_support_status"),
            "external_baseline_comparison_decision": external_baseline_comparison_decision.get("external_baseline_comparison_decision"),
            "external_baseline_comparison_record_count": external_baseline_comparison_decision.get("external_baseline_comparison_record_count"),
            "external_baseline_comparison_ready_count": external_baseline_comparison_decision.get("external_baseline_comparison_ready_count"),
            "external_baseline_measured_adapter_count": external_baseline_comparison_decision.get("external_baseline_measured_adapter_count"),
            "modern_external_baseline_formal_measured_adapter_count": external_baseline_comparison_decision.get("modern_external_baseline_formal_measured_adapter_count"),
            "external_baseline_execution_manifest_status": "present" if external_baseline_execution_manifest else "missing",
            "external_baseline_formal_evidence_status": external_baseline_execution_manifest.get("formal_evidence_status"),
            "external_baseline_evidence_path_count": external_baseline_execution_manifest.get("evidence_path_count"),
            "external_baseline_comparison_status": external_baseline_comparison_decision.get("external_baseline_comparison_status"),
            "external_baseline_comparison_table_status": external_baseline_comparison_decision.get("external_baseline_comparison_table_status"),
            "formal_method_baseline_comparison_decision": formal_method_baseline_comparison_decision.get("formal_method_baseline_comparison_decision"),
            "formal_comparison_ready_method_count": formal_method_baseline_comparison_decision.get("formal_comparison_ready_method_count"),
            "formal_comparison_modern_baseline_ready_count": formal_method_baseline_comparison_decision.get("formal_comparison_modern_baseline_ready_count"),
            "formal_comparison_missing_method_count": formal_method_baseline_comparison_decision.get("formal_comparison_missing_method_count"),
            "validation_internal_ablation_decision": internal_ablation_decision.get("validation_internal_ablation_decision"),
            "validation_internal_ablation_record_count": internal_ablation_decision.get("internal_ablation_record_count"),
            "adaptive_attack_decision": adaptive_attack_decision.get("adaptive_attack_decision"),
            "adaptive_attack_record_count": adaptive_attack_decision.get("adaptive_attack_record_count"),
            "adaptive_robustness_claim_allowed": adaptive_attack_decision.get("adaptive_robustness_claim_allowed"),
            "replay_and_sketch_gate_decision": replay_and_sketch_decision.get("replay_and_sketch_gate_decision"),
            "replay_and_sketch_evidence_level": replay_and_sketch_decision.get("replay_and_sketch_evidence_level"),
            "trajectory_sketch_verified_count": replay_and_sketch_decision.get("trajectory_sketch_verified_count"),
            "replay_uncertainty_ready_count": replay_and_sketch_decision.get("replay_uncertainty_ready_count"),
            "wrong_sampler_replay_rejected_count": replay_and_sketch_decision.get("wrong_sampler_replay_rejected_count"),
            "wrong_prompt_replay_rejected_count": replay_and_sketch_decision.get("wrong_prompt_replay_rejected_count"),
            "replay_and_sketch_claim3_full_support_allowed": replay_and_sketch_decision.get("claim3_full_support_allowed"),
            "claim3_downgrade_decision": claim3_downgrade_decision.get("claim3_downgrade_decision"),
            "claim3_downgraded": claim3_downgrade_decision.get("claim3_downgraded"),
            "claim3_full_support_allowed": claim3_downgrade_decision.get("claim3_full_support_allowed"),
            "replay_or_sketch_status": claim3_downgrade_decision.get("replay_or_sketch_status"),
            "statistical_confidence_interval_decision": confidence_interval_decision.get("statistical_confidence_interval_decision"),
            "statistical_confidence_interval_total_count": confidence_interval_decision.get("ci_total_count"),
            "pilot_paper_gate_decision": pilot_paper_decision.get("pilot_paper_gate_decision"),
            "pilot_paper_claim_support_status": pilot_paper_decision.get("claim_support_status"),
            "pilot_paper_result_level": pilot_paper_decision.get("paper_result_level"),
            "pilot_paper_protocol_level": pilot_paper_decision.get("paper_protocol_level"),
            "pilot_paper_protocol_difference_from_full_paper": pilot_paper_decision.get("paper_protocol_difference_from_full_paper"),
            "pilot_paper_protocol_matches_full_paper": pilot_paper_decision.get("pilot_paper_protocol_matches_full_paper"),
            "pilot_paper_claim_allowed": pilot_paper_decision.get("pilot_paper_claim_allowed"),
            "pilot_paper_external_baseline_trace_count": pilot_paper_decision.get("pilot_paper_external_baseline_trace_count"),
            "pilot_paper_external_baseline_trace_count_min": pilot_paper_decision.get("pilot_paper_external_baseline_trace_count_min"),
            "pilot_paper_modern_external_baseline_formal_measured_adapter_count": pilot_paper_decision.get("modern_external_baseline_formal_measured_adapter_count"),
            "pilot_paper_internal_ablation_trace_count_min": pilot_paper_decision.get("pilot_paper_internal_ablation_trace_count_min"),
            "pilot_paper_missing_external_baseline_adapter_names": pilot_paper_decision.get("missing_external_baseline_adapter_names"),
            "pilot_paper_missing_modern_external_baseline_formal_adapter_names": pilot_paper_decision.get("missing_modern_external_baseline_formal_adapter_names"),
            "pilot_paper_missing_internal_ablation_variants": pilot_paper_decision.get("missing_internal_ablation_variants"),
            "pilot_paper_missing_requirement_count": pilot_paper_decision.get("pilot_paper_missing_requirement_count"),
            "pilot_paper_threshold_protocol": pilot_paper_decision.get("threshold_protocol"),
            "pilot_paper_threshold_source_split": pilot_paper_decision.get("threshold_source_split"),
            "pilot_paper_test_time_threshold_update_blocked": pilot_paper_decision.get("test_time_threshold_update_blocked"),
            "pilot_paper_target_fpr": pilot_paper_decision.get("target_fpr"),
            "pilot_paper_tpr_at_target_fpr": pilot_paper_decision.get("tpr_at_target_fpr"),
            "pilot_paper_target_fpr_claim_allowed": pilot_paper_decision.get("target_fpr_claim_allowed"),
            "pilot_paper_blocked_target_fpr": pilot_paper_decision.get("blocked_target_fpr"),
            "pilot_paper_blocked_target_fpr_claim_allowed": pilot_paper_decision.get("blocked_target_fpr_claim_allowed"),
            "pilot_paper_tpr_at_fpr_01": pilot_paper_decision.get("tpr_at_fpr_01"),
            "pilot_paper_calibration_negative_fpr_at_threshold": pilot_paper_decision.get("calibration_negative_fpr_at_threshold"),
            "pilot_paper_heldout_negative_fpr_at_threshold": pilot_paper_decision.get("heldout_negative_fpr_at_threshold"),
            "pilot_paper_observed_negative_fpr_at_threshold": pilot_paper_decision.get("observed_negative_fpr_at_threshold"),
            "pilot_paper_calibration_negative_event_count": pilot_paper_decision.get("calibration_negative_event_count"),
            "pilot_paper_heldout_test_negative_event_count": pilot_paper_decision.get("heldout_test_negative_event_count"),
            "pilot_paper_heldout_negative_event_count": pilot_paper_decision.get("heldout_negative_event_count"),
            "pilot_paper_heldout_attacked_positive_event_count": pilot_paper_decision.get("heldout_attacked_positive_event_count"),
            "pilot_paper_attacked_positive_event_count": pilot_paper_decision.get("attacked_positive_event_count"),
            "pilot_paper_tpr_at_fpr_01_pilot_claim_allowed": pilot_paper_decision.get("tpr_at_fpr_01_pilot_claim_allowed"),
            "pilot_paper_tpr_at_fpr_001_claim_allowed": pilot_paper_decision.get("tpr_at_fpr_001_claim_allowed"),
            "validation_artifact_rebuild_dry_run_decision": artifact_rebuild_decision.get("validation_artifact_rebuild_dry_run_decision"),
            "validation_artifact_rebuild_missing_count": artifact_rebuild_decision.get("artifact_rebuild_missing_count"),
            "validation_scale_gate_decision": validation_scale_decision.get("validation_scale_gate_decision"),
            "validation_scale_claim_support_status": validation_scale_decision.get("claim_support_status"),
            "validation_scale_target_fpr": validation_scale_decision.get("target_fpr"),
            "validation_scale_result_level": validation_scale_decision.get("paper_result_level"),
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
    payload = package_generative_video_colab_run(
        args.run_root,
        drive_package_dir,
        include_videos=not args.exclude_videos,
        workflow_profile=args.workflow_profile,
        stage_package_id=args.stage_package_id,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
