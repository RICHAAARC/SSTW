"""检查 B6 sampling-time constraint Colab probe 结果是否可进入后续审计。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable
import zipfile


def _read_json(path: Path) -> dict:
    """读取 JSON 文件, 文件不存在时返回空对象。"""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    """读取 JSONL records, 文件不存在时返回空列表。"""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256_file(path: Path) -> str | None:
    """计算文件 sha256, 文件不存在时返回 None。"""
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count_successful_generations(records: Iterable[dict]) -> int:
    """统计 generation_status 为 success 的记录数量。"""
    return sum(1 for record in records if record.get("generation_status") == "success")


def _video_integrity_records(run_root: Path, generation_records: list[dict]) -> list[dict]:
    """检查 generation records 中登记的视频路径、大小和哈希。"""
    checks: list[dict] = []
    for record in generation_records:
        video_path_text = str(record.get("video_path") or "")
        candidate_paths = []
        if video_path_text:
            candidate_paths.append(Path(video_path_text))
            candidate_paths.append(run_root / "videos" / Path(video_path_text).name)
        existing_path = next((path for path in candidate_paths if path.exists()), None)
        actual_sha256 = _sha256_file(existing_path) if existing_path else None
        expected_sha256 = record.get("video_sha256")
        checks.append({
            "generation_model_id": record.get("generation_model_id"),
            "method_variant": record.get("method_variant"),
            "prompt_id": record.get("prompt_id"),
            "seed_id": record.get("seed_id"),
            "video_path": str(existing_path) if existing_path else video_path_text,
            "video_exists": existing_path is not None,
            "video_size_bytes": existing_path.stat().st_size if existing_path else 0,
            "video_sha256_match": bool(actual_sha256 and expected_sha256 and actual_sha256 == expected_sha256),
            "expected_video_sha256": expected_sha256,
            "actual_video_sha256": actual_sha256,
        })
    return checks


def _latest_package_manifest(package_dir: Path) -> Path | None:
    """返回 package 目录下最近生成的 package manifest。"""
    candidates = sorted(package_dir.glob("*_package_manifest.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _inspect_archive_members(archive_path: Path) -> dict:
    """检查 zip 包中是否包含 B6 关键 artifacts。"""
    if not archive_path.exists():
        return {
            "archive_path": str(archive_path),
            "archive_exists": False,
            "archive_member_count": 0,
            "required_archive_members_present": False,
            "missing_archive_member_patterns": ["archive_missing"],
        }
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
    required_patterns = [
        "/records/generation_records.jsonl",
        "/records/constraint_records.jsonl",
        "/records/formal_quality_motion_semantic_records.jsonl",
        "/records/constraint_variant_summary_records.jsonl",
        "/artifacts/sampling_time_constraint_colab_runtime_decision.json",
        "/artifacts/sampling_time_constraint_colab_postprocess_decision.json",
        "/artifacts/formal_quality_motion_semantic_decision.json",
    ]
    missing_patterns = [
        pattern
        for pattern in required_patterns
        if not any(name.endswith(pattern) for name in names)
    ]
    return {
        "archive_path": str(archive_path),
        "archive_exists": True,
        "archive_member_count": len(names),
        "required_archive_members_present": not missing_patterns,
        "missing_archive_member_patterns": missing_patterns,
    }


def _resolve_colab_drive_path(path_text: str, local_project_root: Path) -> Path:
    """将 Colab Drive 绝对路径映射到本地同步的 SSTW 项目目录。

    该函数属于项目特定写法。Colab package manifest 会记录 `/content/drive/MyDrive/SSTW/...` 路径, 但 Windows 本地审阅时对应目录通常是 `G:\\我的云端硬盘\\SSTW`。因此检查器需要在不修改 manifest 的前提下, 将 Colab 路径映射为当前 package_dir 推断出的本地项目根目录。
    """
    normalized = path_text.replace("\\", "/")
    colab_root = "/content/drive/MyDrive/SSTW"
    if normalized.startswith(colab_root):
        relative_text = normalized[len(colab_root):].lstrip("/")
        return local_project_root / Path(relative_text)
    return Path(path_text)


def _variant_names(records: list[dict]) -> set[str]:
    """提取 records 中存在的 method_variant 名称。"""
    return {str(record.get("method_variant")) for record in records if record.get("method_variant")}


def check_sampling_time_constraint_colab_results(run_root: str | Path) -> dict:
    """检查 B6 Colab run 目录, 并区分实现证据、机制证据和最终 claim 边界。"""
    run_root = Path(run_root)
    generation_records = _read_jsonl(run_root / "records" / "generation_records.jsonl")
    trajectory_records = _read_jsonl(run_root / "records" / "trajectory_trace.jsonl")
    constraint_records = _read_jsonl(run_root / "records" / "constraint_records.jsonl")
    formal_metric_records = _read_jsonl(run_root / "records" / "formal_quality_motion_semantic_records.jsonl")
    variant_summary_records = _read_jsonl(run_root / "records" / "constraint_variant_summary_records.jsonl")
    runtime_decision = _read_json(run_root / "artifacts" / "sampling_time_constraint_colab_runtime_decision.json")
    postprocess_decision = _read_json(run_root / "artifacts" / "sampling_time_constraint_colab_postprocess_decision.json")
    formal_metric_decision = _read_json(run_root / "artifacts" / "formal_quality_motion_semantic_decision.json")
    manifest = _read_json(run_root / "artifacts" / "generation_manifest.json")
    video_checks = _video_integrity_records(run_root, generation_records)

    successful_generation_count = _count_successful_generations(generation_records)
    generation_trace_ids = {record.get("trajectory_trace_id") for record in generation_records if record.get("trajectory_trace_id")}
    trajectory_trace_ids = {record.get("trajectory_trace_id") for record in trajectory_records if record.get("trajectory_trace_id")}
    generation_constraint_ids = {record.get("constraint_trace_id") for record in generation_records if record.get("constraint_trace_id")}
    captured_constraint_ids = {record.get("constraint_trace_id") for record in constraint_records if record.get("constraint_trace_id")}
    trajectory_capture_ready = bool(trajectory_records) and generation_trace_ids.issubset(trajectory_trace_ids)
    constraint_capture_ready = bool(constraint_records) and generation_constraint_ids.issubset(captured_constraint_ids)
    successful_video_checks = [
        check
        for check, record in zip(video_checks, generation_records)
        if record.get("generation_status") == "success"
    ]
    all_success_videos_valid = all(item["video_exists"] and item["video_sha256_match"] for item in successful_video_checks) if successful_video_checks else False

    method_variants = _variant_names(generation_records)
    constraint_variants = _variant_names(constraint_records)
    formal_variants = _variant_names(formal_metric_records)
    keyed_records = [record for record in constraint_records if record.get("method_variant") == "keyed_state_trajectory_constraint"]
    baseline_records = [record for record in constraint_records if record.get("method_variant") == "key_conditioned_state_space_with_trajectory"]
    keyed_applied_count = sum(1 for record in keyed_records if record.get("constraint_apply_status") == "applied")
    baseline_applied_count = sum(1 for record in baseline_records if record.get("constraint_apply_status") == "applied")
    keyed_gain = float(postprocess_decision.get("details", {}).get("keyed_constraint_alignment_gain_mean", 0.0) or 0.0)
    baseline_gain = float(postprocess_decision.get("details", {}).get("baseline_alignment_gain_mean", 0.0) or 0.0)
    formal_visual_motion_ready_count = sum(
        1 for record in formal_metric_records
        if record.get("formal_visual_quality_ready") is True and record.get("formal_motion_consistency_ready") is True
    )
    formal_semantic_ready_count = sum(1 for record in formal_metric_records if record.get("formal_semantic_consistency_ready") is True)

    implementation_evidence_status = "PASS" if all([
        run_root.exists(),
        runtime_decision.get("implementation_decision") == "PASS",
        successful_generation_count > 0,
        all_success_videos_valid,
        trajectory_capture_ready,
        constraint_capture_ready,
        bool(manifest),
    ]) else "FAIL"

    missing_mechanism_requirements = []
    if len(method_variants) < 4 or len(constraint_variants) < 4:
        missing_mechanism_requirements.append("insufficient_method_variant_coverage")
    if keyed_applied_count <= 0:
        missing_mechanism_requirements.append("keyed_constraint_not_applied")
    if baseline_applied_count != 0:
        missing_mechanism_requirements.append("unconstrained_baseline_should_not_apply_constraint")
    if keyed_gain <= baseline_gain:
        missing_mechanism_requirements.append("keyed_constraint_gain_not_above_baseline")
    if postprocess_decision.get("details", {}).get("formal_quality_semantic_ready") is not True:
        missing_mechanism_requirements.append("formal_quality_semantic_not_ready")
    if formal_metric_decision.get("formal_quality_motion_semantic_ready") is not True:
        missing_mechanism_requirements.append("formal_metric_decision_not_ready")

    mechanism_evidence_status = "PASS" if (
        postprocess_decision.get("mechanism_decision") == "PASS"
        and not missing_mechanism_requirements
    ) else "FAIL"

    return {
        "run_root": str(run_root),
        "generation_record_count": len(generation_records),
        "successful_generation_count": successful_generation_count,
        "trajectory_record_count": len(trajectory_records),
        "constraint_record_count": len(constraint_records),
        "formal_metric_record_count": len(formal_metric_records),
        "constraint_variant_summary_record_count": len(variant_summary_records),
        "method_variant_count": len(method_variants),
        "constraint_variant_count": len(constraint_variants),
        "formal_metric_variant_count": len(formal_variants),
        "keyed_constraint_applied_step_count": keyed_applied_count,
        "baseline_constraint_applied_step_count": baseline_applied_count,
        "keyed_constraint_alignment_gain_mean": keyed_gain,
        "baseline_alignment_gain_mean": baseline_gain,
        "formal_visual_motion_ready_count": formal_visual_motion_ready_count,
        "formal_semantic_ready_count": formal_semantic_ready_count,
        "trajectory_capture_ready": trajectory_capture_ready,
        "constraint_capture_ready": constraint_capture_ready,
        "video_checks": video_checks,
        "implementation_evidence_status": implementation_evidence_status,
        "mechanism_evidence_status": mechanism_evidence_status,
        "missing_mechanism_requirements": missing_mechanism_requirements,
        "decision_summary": {
            "runtime_stage_id": runtime_decision.get("stage_id"),
            "implementation_decision": runtime_decision.get("implementation_decision"),
            "runtime_mechanism_decision": runtime_decision.get("mechanism_decision"),
            "postprocess_stage_id": postprocess_decision.get("stage_id"),
            "mechanism_postprocess_decision": postprocess_decision.get("mechanism_postprocess_decision"),
            "postprocess_mechanism_decision": postprocess_decision.get("mechanism_decision"),
            "postprocess_formal_claim_status": postprocess_decision.get("details", {}).get("formal_claim_status"),
            "formal_metric_claim_status": formal_metric_decision.get("formal_metric_claim_status"),
        },
        "claim_boundary": "real_sampling_probe_not_final_b6_submission_claim",
        "next_recommended_action": "run_recommended_profile_on_l4" if mechanism_evidence_status == "PASS" else "rerun_smoke_after_fixing_outputs",
    }


def check_latest_sampling_time_constraint_package(package_dir: str | Path) -> dict:
    """检查 B6 package 目录中的最新 manifest, 并在可访问 run_root 时联动检查 run 目录。"""
    package_dir = Path(package_dir)
    local_project_root = package_dir.parent.parent
    manifest_path = _latest_package_manifest(package_dir)
    if manifest_path is None:
        return {
            "package_dir": str(package_dir),
            "package_manifest_status": "missing",
            "implementation_evidence_status": "FAIL",
            "mechanism_evidence_status": "FAIL",
            "missing_mechanism_requirements": ["package_manifest_missing"],
            "next_recommended_action": "run_colab_notebook_and_package_results",
        }
    package_manifest = _read_json(manifest_path)
    archive_path = _resolve_colab_drive_path(str(package_manifest.get("archive_path") or ""), local_project_root)
    archive_inspection = _inspect_archive_members(archive_path)
    run_root_text = str(package_manifest.get("run_root") or "")
    run_root = _resolve_colab_drive_path(run_root_text, local_project_root) if run_root_text else None
    run_check = check_sampling_time_constraint_colab_results(run_root) if run_root and run_root.exists() else {}
    return {
        "package_dir": str(package_dir),
        "package_manifest_status": "present",
        "package_manifest_path": str(manifest_path),
        "package_manifest": package_manifest,
        "archive_inspection": archive_inspection,
        "run_check_status": "present" if run_check else "run_root_not_accessible",
        "run_check": run_check,
        "implementation_evidence_status": run_check.get("implementation_evidence_status", "PASS" if archive_inspection["required_archive_members_present"] else "FAIL"),
        "mechanism_evidence_status": run_check.get("mechanism_evidence_status", "package_only_not_enough_for_mechanism_evidence"),
        "missing_mechanism_requirements": run_check.get("missing_mechanism_requirements", [] if archive_inspection["required_archive_members_present"] else archive_inspection["missing_archive_member_patterns"]),
        "next_recommended_action": run_check.get("next_recommended_action", "inspect_run_root_or_unpack_archive_for_full_check"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 B6 sampling-time constraint Colab probe 结果。")
    parser.add_argument("--run-root", default="")
    parser.add_argument("--package-dir", default="")
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()
    if not args.run_root and not args.package_dir:
        raise SystemExit("必须提供 --run-root 或 --package-dir")
    payload = (
        check_sampling_time_constraint_colab_results(args.run_root)
        if args.run_root
        else check_latest_sampling_time_constraint_package(args.package_dir)
    )
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
