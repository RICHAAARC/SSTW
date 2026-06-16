"""检查 B5 Colab 生成式视频探测结果是否可进入后续机制审计。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable


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
    """统计 generation_status 为 success 的记录数。"""
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


def check_generative_video_colab_results(run_root: str | Path) -> dict:
    """检查 B5 Colab run 目录, 并明确区分 implementation evidence 与 mechanism evidence。"""
    run_root = Path(run_root)
    generation_records = _read_jsonl(run_root / "records" / "generation_records.jsonl")
    trajectory_records = _read_jsonl(run_root / "records" / "trajectory_trace.jsonl")
    quality_records = _read_jsonl(run_root / "records" / "quality_motion_semantic_records.jsonl")
    external_records = _read_jsonl(run_root / "records" / "external_baseline_records.jsonl")
    decision = _read_json(run_root / "artifacts" / "generative_video_colab_runtime_decision.json")
    postprocess_decision = _read_json(run_root / "artifacts" / "generative_video_mechanism_postprocess_decision.json")
    manifest = _read_json(run_root / "artifacts" / "generation_manifest.json")
    mechanism_records = _read_jsonl(run_root / "records" / "mechanism_score_records.jsonl")
    controlled_negative_records = _read_jsonl(run_root / "records" / "controlled_negative_records.jsonl")
    quality_proxy_records = _read_jsonl(run_root / "records" / "quality_motion_semantic_proxy_records.jsonl")
    formal_metric_records = _read_jsonl(run_root / "records" / "formal_quality_motion_semantic_records.jsonl")
    video_checks = _video_integrity_records(run_root, generation_records)

    successful_generation_count = _count_successful_generations(generation_records)
    captured_trace_ids = {record.get("trajectory_trace_id") for record in trajectory_records if record.get("trajectory_trace_id")}
    generation_trace_ids = {record.get("trajectory_trace_id") for record in generation_records if record.get("trajectory_trace_id")}
    all_success_videos_valid = all(item["video_exists"] and item["video_sha256_match"] for item in video_checks) if video_checks else False
    trajectory_capture_ready = bool(trajectory_records) and generation_trace_ids.issubset(captured_trace_ids)
    external_baseline_runnable_count = sum(1 for record in external_records if record.get("external_baseline_runnable_status") == "runnable")
    quality_metric_ready_count = sum(1 for record in quality_records if record.get("quality_metric_status") not in {"not_run", "disabled", None})
    formal_visual_motion_ready_count = sum(
        1 for record in formal_metric_records
        if record.get("formal_visual_quality_ready") is True and record.get("formal_motion_consistency_ready") is True
    )
    formal_semantic_ready_count = sum(1 for record in formal_metric_records if record.get("formal_semantic_consistency_ready") is True)

    implementation_evidence_status = "PASS" if all([
        run_root.exists(),
        successful_generation_count > 0,
        all_success_videos_valid,
        trajectory_capture_ready,
        bool(manifest),
    ]) else "FAIL"

    missing_mechanism_requirements = []
    postprocess_details = postprocess_decision.get("details", {})
    if (
        decision.get("details", {}).get("fixed_low_fpr_audit_pass") is not True
        and postprocess_details.get("fixed_low_fpr_proxy_pass") is not True
    ):
        missing_mechanism_requirements.append("fixed_low_fpr_audit_not_passed")
    if (
        decision.get("details", {}).get("trajectory_observation_gain_confirmed") is not True
        and postprocess_details.get("trajectory_gain_confirmed_by_proxy") is not True
    ):
        missing_mechanism_requirements.append("trajectory_observation_gain_not_confirmed")
    if decision.get("details", {}).get("quality_motion_semantic_consistency_pass") is not True:
        if postprocess_details.get("formal_quality_semantic_ready") is True:
            pass
        elif postprocess_details.get("formal_visual_motion_ready") is True and postprocess_details.get("formal_semantic_ready") is not True:
            missing_mechanism_requirements.append("formal_semantic_metric_missing")
        elif postprocess_details.get("quality_motion_semantic_proxy_pass") is True:
            missing_mechanism_requirements.append("formal_quality_semantic_metrics_missing")
        else:
            missing_mechanism_requirements.append("quality_motion_semantic_consistency_not_passed")
    if external_baseline_runnable_count < 1:
        missing_mechanism_requirements.append("external_baseline_not_runnable")
    if successful_generation_count < 4:
        missing_mechanism_requirements.append("insufficient_prompt_seed_coverage_for_b5")

    formal_mechanism_pass = (
        decision.get("mechanism_decision") == "PASS"
        or postprocess_decision.get("mechanism_decision") == "PASS"
    )
    mechanism_evidence_status = "PASS" if formal_mechanism_pass and not missing_mechanism_requirements else "FAIL"
    return {
        "run_root": str(run_root),
        "generation_record_count": len(generation_records),
        "successful_generation_count": successful_generation_count,
        "trajectory_record_count": len(trajectory_records),
        "quality_record_count": len(quality_records),
        "external_baseline_record_count": len(external_records),
        "external_baseline_runnable_count": external_baseline_runnable_count,
        "mechanism_score_record_count": len(mechanism_records),
        "controlled_negative_record_count": len(controlled_negative_records),
        "quality_proxy_record_count": len(quality_proxy_records),
        "formal_metric_record_count": len(formal_metric_records),
        "formal_visual_motion_ready_count": formal_visual_motion_ready_count,
        "formal_semantic_ready_count": formal_semantic_ready_count,
        "mechanism_postprocess_status": postprocess_decision.get("mechanism_postprocess_decision", "missing"),
        "quality_metric_ready_count": quality_metric_ready_count,
        "video_checks": video_checks,
        "implementation_evidence_status": implementation_evidence_status,
        "mechanism_evidence_status": mechanism_evidence_status,
        "missing_mechanism_requirements": missing_mechanism_requirements,
        "decision_summary": {
            "stage_id": decision.get("stage_id"),
            "implementation_decision": decision.get("implementation_decision"),
            "mechanism_decision": decision.get("mechanism_decision"),
            "postprocess_stage_id": postprocess_decision.get("stage_id"),
            "mechanism_postprocess_decision": postprocess_decision.get("mechanism_postprocess_decision"),
            "postprocess_mechanism_decision": postprocess_decision.get("mechanism_decision"),
        },
        "next_recommended_profile": "recommended_on_l4_or_a100" if implementation_evidence_status == "PASS" else "rerun_smoke_after_fixing_outputs",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 B5 Colab 生成式视频探测结果。")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-json", default="")
    args = parser.parse_args()
    payload = check_generative_video_colab_results(args.run_root)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
