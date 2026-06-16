"""对 B5 Colab 生成视频执行文件级正式质量与运动度量。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from main.analysis.video_file_metrics import compute_video_file_metrics
from main.protocol.record_writer import write_json, write_jsonl
from main.protocol.table_builder import write_csv


def _read_jsonl(path: Path) -> list[dict]:
    """读取 JSONL records, 文件不存在时返回空列表。"""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _resolve_video_path(run_root: Path, record: dict) -> Path:
    """解析 generation record 中的视频路径, 兼容 Colab 绝对路径与本地 Drive 同步路径。"""
    video_path_text = str(record.get("video_path") or "")
    direct_path = Path(video_path_text)
    if direct_path.exists():
        return direct_path
    return run_root / "videos" / direct_path.name


def build_formal_metric_records(run_root: str | Path) -> list[dict]:
    """从 generation records 与实际 mp4 文件构造正式质量/运动 metric records。"""
    run_root = Path(run_root)
    generation_records = _read_jsonl(run_root / "records" / "generation_records.jsonl")
    records: list[dict] = []
    for generation_record in generation_records:
        video_path = _resolve_video_path(run_root, generation_record)
        metrics = compute_video_file_metrics(video_path)
        visual_ready = metrics.get("visual_quality_metric_status") == "ready"
        motion_ready = metrics.get("motion_consistency_metric_status") == "ready"
        records.append({
            "record_version": "generative_video_formal_quality_motion_semantic_v1",
            "generation_model_id": generation_record.get("generation_model_id"),
            "prompt_id": generation_record.get("prompt_id"),
            "seed_id": generation_record.get("seed_id"),
            "trajectory_trace_id": generation_record.get("trajectory_trace_id"),
            "video_path": str(video_path),
            "video_decode_status": metrics.get("video_decode_status"),
            "video_metric_failure_reason": metrics.get("video_metric_failure_reason"),
            **{key: value for key, value in metrics.items() if key not in {"video_decode_status", "video_metric_failure_reason"}},
            "formal_visual_quality_ready": visual_ready,
            "formal_motion_consistency_ready": motion_ready,
            "semantic_metric_name": "not_configured",
            "semantic_metric_status": "not_configured",
            "semantic_consistency_score": None,
            "semantic_metric_failure_reason": "clip_or_vlm_text_video_metric_not_configured",
            "formal_semantic_consistency_ready": False,
            "formal_metric_result_used_for_claim": False,
        })
    return records


def audit_formal_metrics(records: list[dict]) -> dict:
    """审计文件级正式 metric records 的就绪状态。"""
    visual_ready_count = sum(1 for record in records if record.get("formal_visual_quality_ready") is True)
    motion_ready_count = sum(1 for record in records if record.get("formal_motion_consistency_ready") is True)
    semantic_ready_count = sum(1 for record in records if record.get("formal_semantic_consistency_ready") is True)
    all_visual_motion_ready = bool(records) and visual_ready_count == len(records) and motion_ready_count == len(records)
    all_semantic_ready = bool(records) and semantic_ready_count == len(records)
    return {
        "stage_id": "generative_video_formal_quality_motion_semantic_metrics",
        "formal_metric_record_count": len(records),
        "formal_visual_quality_ready_count": visual_ready_count,
        "formal_motion_consistency_ready_count": motion_ready_count,
        "formal_semantic_consistency_ready_count": semantic_ready_count,
        "formal_visual_motion_ready": all_visual_motion_ready,
        "formal_semantic_ready": all_semantic_ready,
        "formal_quality_motion_semantic_ready": all_visual_motion_ready and all_semantic_ready,
        "formal_metric_claim_status": "blocked_until_semantic_metric_configured" if not all_semantic_ready else "ready",
    }


def run_formal_metric_audit(run_root: str | Path) -> dict:
    """执行 B5 文件级正式 metric 计算并写出 governed artifacts。"""
    run_root = Path(run_root)
    records = build_formal_metric_records(run_root)
    audit = audit_formal_metrics(records)
    write_jsonl(run_root / "records" / "formal_quality_motion_semantic_records.jsonl", records)
    write_csv(run_root / "tables" / "formal_quality_motion_semantic_table.csv", records)
    write_json(run_root / "artifacts" / "formal_quality_motion_semantic_decision.json", audit)
    report = (
        "# Formal Quality Motion Semantic Metrics Report\n\n"
        "该报告基于实际 mp4 文件解码结果生成质量与运动指标。语义一致性指标尚未配置 "
        "CLIP / VLM text-video metric, 因此不能支撑正式机制 claim。\n\n"
        f"- formal_visual_motion_ready: {audit['formal_visual_motion_ready']}\n"
        f"- formal_semantic_ready: {audit['formal_semantic_ready']}\n"
        f"- formal_metric_claim_status: {audit['formal_metric_claim_status']}\n"
    )
    report_path = run_root / "reports" / "formal_quality_motion_semantic_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="对 B5 Colab 生成视频执行文件级正式质量与运动度量。")
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    payload = run_formal_metric_audit(args.run_root)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
