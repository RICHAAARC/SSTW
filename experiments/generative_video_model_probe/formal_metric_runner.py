"""对 generative_video_model_probe Colab 生成视频执行文件级质量、运动与语义度量。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.metrics.semantic_video_metrics import DEFAULT_CLIP_MODEL_ID, DEFAULT_SEMANTIC_THRESHOLD, compute_clip_text_video_similarity
from evaluation.metrics.video_file_metrics import (
    compute_paired_video_quality_metrics,
    compute_video_file_metrics,
)
from runtime.core.progress import ProgressReporter
from evaluation.protocol.flow_evidence_fields import with_flow_evidence_protocol_defaults
from evaluation.protocol.record_writer import write_json, write_jsonl
from evaluation.protocol.table_builder import write_csv


def _read_json(path: Path) -> dict:
    """读取 JSON 文件, 文件不存在或为空时返回空对象。"""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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


def _candidate_project_roots(run_root: Path) -> list[Path]:
    """从 run_root 推断可能的 SSTW 项目落盘根目录。"""
    candidates: list[Path] = []
    parents = list(run_root.parents)
    if len(parents) >= 2:
        candidates.append(parents[1])
    if len(parents) >= 1:
        candidates.append(parents[0])
    candidates.extend([
        Path("/content/drive/MyDrive/SSTW"),
        Path(r"G:\我的云端硬盘\SSTW"),
    ])
    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def _resolve_prompt_suite_path(run_root: Path, prompt_suite_path: str | Path | None = None) -> Path | None:
    """解析 prompt suite 路径。

    该函数属于项目特定写法。Colab 记录的 manifest 可能包含云端绝对路径, 本地 Windows 审计时该路径不一定存在,
    因此需要按 SSTW Drive 目录约定进行回退解析。
    """
    candidates: list[Path] = []
    if prompt_suite_path:
        candidates.append(Path(prompt_suite_path))

    manifest = _read_json(run_root / "artifacts" / "generation_manifest.json")
    for input_path in manifest.get("input_paths", []):
        if input_path:
            candidates.append(Path(str(input_path)))

    for project_root in _candidate_project_roots(run_root):
        candidates.append(project_root / "datasets" / "generative_video_prompt_suite" / "prompt_seed_suite.json")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_prompt_metadata(prompt_suite_path: Path | None) -> tuple[dict[str, dict], str]:
    """读取 prompt 元数据, 同时返回 prompt suite 来源状态。

    该函数属于通用工程写法。调用方不只需要 prompt_text 做语义 metric, 还需要
    motion_calibration_role 判断样本在 formal motion gate 中的职责。
    """
    if prompt_suite_path is None:
        return {}, "prompt_suite_missing"
    payload = _read_json(prompt_suite_path)
    prompts = payload.get("prompts", [])
    prompt_metadata = {
        str(item.get("prompt_id")): dict(item)
        for item in prompts
        if item.get("prompt_id")
    }
    return prompt_metadata, str(prompt_suite_path)


def _infer_motion_claim_role(generation_record: dict, prompt_metadata: dict) -> str:
    """推断 formal motion gate 中样本承担的运动职责。

    通用工程写法是优先读取显式字段。项目特定写法是把 calibration 中的
    negative_static 和 ambiguous_low_motion 视为负样本或边界样本, 它们允许低运动,
    但不能用来支撑正向 trajectory / velocity claim。
    """
    for field_name in ("motion_claim_role", "motion_calibration_role"):
        for source in (generation_record, prompt_metadata):
            role = source.get(field_name)
            if role:
                return str(role)
    prompt_suite_role = str(generation_record.get("prompt_suite_role") or prompt_metadata.get("prompt_suite_role") or "")
    if "negative_static" in prompt_suite_role:
        return "negative_static"
    if "ambiguous_low_motion" in prompt_suite_role:
        return "ambiguous_low_motion"
    if "positive_motion" in prompt_suite_role:
        return "positive_motion"
    return "positive_motion"


def _role_aware_motion_gate(metrics: dict, motion_claim_role: str) -> dict:
    """根据样本角色解释文件级 motion metric。

    对 positive_motion, 低运动会阻断正向 motion claim。对 negative_static 和
    ambiguous_low_motion, 低运动是预期现象, 不应阻断 formal metric；但高闪烁仍会阻断,
    因为高闪烁表示视频文件本身不稳定, 不能作为可靠边界样本。
    """
    raw_status = metrics.get("motion_consistency_metric_status")
    raw_reason = str(metrics.get("motion_consistency_failure_reason") or "none")
    low_motion_expected = motion_claim_role in {"negative_static", "ambiguous_low_motion"}
    reason_parts = set(part for part in raw_reason.split(";") if part and part != "none")
    only_low_motion_failure = reason_parts == {"motion_delta_below_min"}
    if low_motion_expected and only_low_motion_failure:
        return {
            "formal_motion_consistency_ready": True,
            "formal_motion_gate_policy": "low_motion_allowed_for_boundary_role",
            "formal_motion_gate_failure_reason": "none",
            "low_motion_expected_for_role": True,
        }
    ready = raw_status == "ready"
    return {
        "formal_motion_consistency_ready": ready,
        "formal_motion_gate_policy": "positive_motion_requires_min_delta" if not low_motion_expected else "boundary_role_blocks_only_non_low_motion_failures",
        "formal_motion_gate_failure_reason": "none" if ready else raw_reason,
        "low_motion_expected_for_role": low_motion_expected,
    }


def _build_disabled_semantic_metrics(model_id: str, frame_limit: int, reason: str) -> dict:
    """构造语义 metric 未运行时的显式状态字段。"""
    return {
        "semantic_metric_name": "clip_text_video_similarity",
        "semantic_model_id": model_id,
        "semantic_metric_status": "disabled" if reason == "semantic_metric_disabled" else reason,
        "semantic_metric_failure_reason": reason,
        "semantic_consistency_score": None,
        "semantic_consistency_mean_score": None,
        "semantic_consistency_max_score": None,
        "semantic_sampled_frame_count": 0,
        "semantic_frame_limit": frame_limit,
        "semantic_metric_device": "not_run",
    }


def _build_prompt_missing_semantic_metrics(model_id: str, frame_limit: int) -> dict:
    """构造缺少 prompt 文本时的语义 metric 状态字段。"""
    return {
        "semantic_metric_name": "clip_text_video_similarity",
        "semantic_model_id": model_id,
        "semantic_metric_status": "prompt_text_missing",
        "semantic_metric_failure_reason": "prompt_text_missing",
        "semantic_consistency_score": None,
        "semantic_consistency_mean_score": None,
        "semantic_consistency_max_score": None,
        "semantic_sampled_frame_count": 0,
        "semantic_frame_limit": frame_limit,
        "semantic_metric_device": "not_run",
    }


def _formal_metric_blocking_reason(
    visual_ready: bool,
    motion_ready: bool,
    semantic_ready: bool,
    paired_quality_ready: bool = True,
) -> str:
    """把 formal gate 的阻塞来源压缩成一个稳定字符串。"""
    reasons: list[str] = []
    if not visual_ready:
        reasons.append("formal_visual_quality_not_ready")
    if not motion_ready:
        reasons.append("formal_motion_consistency_not_ready")
    if not semantic_ready:
        reasons.append("formal_semantic_consistency_not_ready")
    if not paired_quality_ready:
        reasons.append("formal_paired_video_quality_not_ready")
    return "none" if not reasons else ";".join(reasons)


def build_formal_metric_records(
    run_root: str | Path,
    prompt_suite_path: str | Path | None = None,
    semantic_model_id: str = DEFAULT_CLIP_MODEL_ID,
    semantic_frame_limit: int = 8,
    semantic_consistency_threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
    enable_semantic_metric: bool = True,
) -> list[dict]:
    """从 generation records 与实际 mp4 文件构造正式质量、运动和语义 metric records。"""
    run_root = Path(run_root)
    generation_records = _read_jsonl(run_root / "records" / "generation_records.jsonl")
    resolved_prompt_suite_path = _resolve_prompt_suite_path(run_root, prompt_suite_path)
    prompt_metadata_by_id, semantic_prompt_source = _load_prompt_metadata(resolved_prompt_suite_path)
    clean_reference_by_identity = {
        (
            str(record.get("generation_model_id") or ""),
            str(record.get("prompt_id") or ""),
            str(record.get("seed_id") or ""),
        ): record
        for record in generation_records
        if record.get("sample_role") == "clean_negative"
        and record.get("generation_status") == "success"
    }
    records: list[dict] = []
    progress = ProgressReporter("formal_metric_runtime_video_scan", len(generation_records), "runtime_video")
    for index, generation_record in enumerate(generation_records):
        progress.update(
            index + 1,
            f"prompt={generation_record.get('prompt_id')} seed={generation_record.get('seed_id')}",
        )
        video_path = _resolve_video_path(run_root, generation_record)
        metrics = compute_video_file_metrics(video_path)
        visual_ready = metrics.get("visual_quality_metric_status") == "ready"
        prompt_id = str(generation_record.get("prompt_id") or "")
        prompt_metadata = prompt_metadata_by_id.get(prompt_id, {})
        prompt_text = str(prompt_metadata.get("prompt_text") or "")
        motion_claim_role = _infer_motion_claim_role(generation_record, prompt_metadata)
        motion_gate = _role_aware_motion_gate(metrics, motion_claim_role)
        motion_ready = motion_gate["formal_motion_consistency_ready"]
        if not enable_semantic_metric:
            semantic_metrics = _build_disabled_semantic_metrics(semantic_model_id, semantic_frame_limit, "semantic_metric_disabled")
        elif not prompt_text:
            semantic_metrics = _build_prompt_missing_semantic_metrics(semantic_model_id, semantic_frame_limit)
        else:
            semantic_metrics = compute_clip_text_video_similarity(
                video_path,
                prompt_text,
                model_id=semantic_model_id,
                frame_limit=semantic_frame_limit,
            )
        semantic_score = semantic_metrics.get("semantic_consistency_score")
        semantic_ready = (
            semantic_metrics.get("semantic_metric_status") == "ready"
            and semantic_score is not None
            and float(semantic_score) >= semantic_consistency_threshold
        )
        paired_quality_required = (
            generation_record.get("method_variant") == "sstw_full_method"
            and generation_record.get("sample_role") == "attacked_positive_source"
        )
        paired_reference = clean_reference_by_identity.get((
            str(generation_record.get("generation_model_id") or ""),
            str(generation_record.get("prompt_id") or ""),
            str(generation_record.get("seed_id") or ""),
        ))
        if paired_quality_required and paired_reference is not None:
            paired_reference_path = _resolve_video_path(run_root, paired_reference)
            paired_metrics = compute_paired_video_quality_metrics(
                paired_reference_path,
                video_path,
            )
            paired_metrics["paired_reference_video_path"] = str(
                paired_reference_path
            )
        elif paired_quality_required:
            paired_metrics = {
                "paired_video_quality_status": "missing_reference",
                "paired_video_quality_failure_reason": (
                    "same_model_prompt_seed_clean_reference_missing"
                ),
                "paired_reference_video_path": None,
            }
        else:
            paired_metrics = {
                "paired_video_quality_status": "not_required_for_non_primary_variant",
                "paired_video_quality_failure_reason": "none",
                "paired_reference_video_path": None,
            }
        paired_quality_ready = (
            not paired_quality_required
            or paired_metrics.get("paired_video_quality_status") == "ready"
        )
        formal_metric_ready = bool(
            visual_ready and motion_ready and semantic_ready and paired_quality_ready
        )
        records.append(with_flow_evidence_protocol_defaults({
            "record_version": "generative_video_formal_quality_motion_semantic_v1",
            "generation_model_id": generation_record.get("generation_model_id"),
            "method_variant": generation_record.get("method_variant"),
            "sample_role": generation_record.get("sample_role"),
            "prompt_id": generation_record.get("prompt_id"),
            "seed_id": generation_record.get("seed_id"),
            "trajectory_trace_id": generation_record.get("trajectory_trace_id"),
            "motion_claim_role": motion_claim_role,
            "video_path": str(video_path),
            "video_decode_status": metrics.get("video_decode_status"),
            "video_metric_failure_reason": metrics.get("video_metric_failure_reason"),
            **{key: value for key, value in metrics.items() if key not in {"video_decode_status", "video_metric_failure_reason"}},
            "formal_visual_quality_ready": visual_ready,
            **motion_gate,
            "semantic_prompt_source": semantic_prompt_source,
            "semantic_consistency_threshold": semantic_consistency_threshold,
            **semantic_metrics,
            "formal_semantic_consistency_ready": semantic_ready,
            "paired_video_quality_required": paired_quality_required,
            **paired_metrics,
            "formal_paired_video_quality_ready": paired_quality_ready,
            "formal_metric_blocking_reason": _formal_metric_blocking_reason(
                visual_ready,
                motion_ready,
                semantic_ready,
                paired_quality_ready,
            ),
            "formal_metric_result_used_for_claim": formal_metric_ready,
        },
            trajectory_source_level="formal_video_metric_with_generation_trace_reference",
            flow_state_admissibility_status="formal_metric_ready" if formal_metric_ready else "formal_metric_blocked",
            claim_support_status="formal_metric_evidence_only" if formal_metric_ready else "formal_metric_blocked",
        ))
    ready_count = sum(1 for record in records if record.get("formal_metric_result_used_for_claim") is True)
    progress.finish(f"formal_ready={ready_count} blocked={len(records) - ready_count}")
    return records


def audit_formal_metrics(records: list[dict]) -> dict:
    """审计文件级正式 metric records 的就绪状态。"""
    visual_ready_count = sum(1 for record in records if record.get("formal_visual_quality_ready") is True)
    motion_ready_count = sum(1 for record in records if record.get("formal_motion_consistency_ready") is True)
    semantic_ready_count = sum(1 for record in records if record.get("formal_semantic_consistency_ready") is True)
    paired_required_records = [
        record for record in records
        if record.get("paired_video_quality_required") is True
    ]
    paired_ready_count = sum(
        record.get("formal_paired_video_quality_ready") is True
        for record in paired_required_records
    )
    visual_blocked_count = len(records) - visual_ready_count
    motion_blocked_count = len(records) - motion_ready_count
    semantic_blocked_count = len(records) - semantic_ready_count
    all_visual_motion_ready = bool(records) and visual_ready_count == len(records) and motion_ready_count == len(records)
    all_semantic_ready = bool(records) and semantic_ready_count == len(records)
    all_paired_ready = (
        not paired_required_records
        or paired_ready_count == len(paired_required_records)
    )
    formal_ready = all_visual_motion_ready and all_semantic_ready and all_paired_ready
    if formal_ready:
        claim_status = "ready"
    elif visual_blocked_count > 0:
        claim_status = "blocked_by_formal_visual_quality"
    elif motion_blocked_count > 0:
        claim_status = "blocked_by_formal_motion_consistency"
    elif semantic_blocked_count > 0:
        claim_status = "blocked_until_semantic_metric_ready"
    elif not all_paired_ready:
        claim_status = "blocked_until_paired_video_quality_ready"
    else:
        claim_status = "blocked_until_formal_quality_motion_semantic_metrics"
    return {
        "stage_id": "generative_video_formal_quality_motion_semantic_metrics",
        "formal_metric_record_count": len(records),
        "formal_visual_quality_ready_count": visual_ready_count,
        "formal_motion_consistency_ready_count": motion_ready_count,
        "formal_semantic_consistency_ready_count": semantic_ready_count,
        "formal_visual_quality_blocked_count": visual_blocked_count,
        "formal_motion_consistency_blocked_count": motion_blocked_count,
        "formal_semantic_consistency_blocked_count": semantic_blocked_count,
        "formal_paired_video_quality_required_count": len(paired_required_records),
        "formal_paired_video_quality_ready_count": paired_ready_count,
        "formal_visual_motion_ready": all_visual_motion_ready,
        "formal_semantic_ready": all_semantic_ready,
        "formal_paired_video_quality_ready": all_paired_ready,
        "formal_quality_motion_semantic_ready": formal_ready,
        "formal_metric_claim_status": claim_status,
    }


def run_formal_metric_audit(
    run_root: str | Path,
    prompt_suite_path: str | Path | None = None,
    semantic_model_id: str = DEFAULT_CLIP_MODEL_ID,
    semantic_frame_limit: int = 8,
    semantic_consistency_threshold: float = DEFAULT_SEMANTIC_THRESHOLD,
    enable_semantic_metric: bool = True,
) -> dict:
    """执行 generative_video_model_probe 文件级正式 metric 计算并写出 governed artifacts。"""
    run_root = Path(run_root)
    records = build_formal_metric_records(
        run_root,
        prompt_suite_path=prompt_suite_path,
        semantic_model_id=semantic_model_id,
        semantic_frame_limit=semantic_frame_limit,
        semantic_consistency_threshold=semantic_consistency_threshold,
        enable_semantic_metric=enable_semantic_metric,
    )
    audit = audit_formal_metrics(records)
    write_jsonl(run_root / "records" / "formal_quality_motion_semantic_records.jsonl", records)
    write_csv(run_root / "tables" / "formal_quality_motion_semantic_table.csv", records)
    write_json(run_root / "artifacts" / "formal_quality_motion_semantic_decision.json", audit)
    report = (
        "# Formal Quality Motion Semantic Metrics Report\n\n"
        "该报告基于实际 mp4 文件解码结果生成质量、运动与 CLIP 文本-视频语义一致性指标。"
        "若语义模型依赖、prompt suite 或模型权重不可用, records 会显式记录阻断原因, 不会伪造 positive claim。\n\n"
        f"- formal_visual_motion_ready: {audit['formal_visual_motion_ready']}\n"
        f"- formal_semantic_ready: {audit['formal_semantic_ready']}\n"
        f"- formal_metric_claim_status: {audit['formal_metric_claim_status']}\n"
        f"- formal_visual_quality_blocked_count: {audit['formal_visual_quality_blocked_count']}\n"
        f"- formal_motion_consistency_blocked_count: {audit['formal_motion_consistency_blocked_count']}\n"
        f"- formal_semantic_consistency_blocked_count: {audit['formal_semantic_consistency_blocked_count']}\n"
    )
    report_path = run_root / "reports" / "formal_quality_motion_semantic_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="对 generative_video_model_probe Colab 生成视频执行文件级质量、运动与语义度量。")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--prompt-suite-path", default="")
    parser.add_argument("--semantic-model-id", default=DEFAULT_CLIP_MODEL_ID)
    parser.add_argument("--semantic-frame-limit", type=int, default=8)
    parser.add_argument("--semantic-consistency-threshold", type=float, default=DEFAULT_SEMANTIC_THRESHOLD)
    parser.add_argument("--disable-semantic-metric", action="store_true")
    args = parser.parse_args()
    payload = run_formal_metric_audit(
        args.run_root,
        prompt_suite_path=args.prompt_suite_path or None,
        semantic_model_id=args.semantic_model_id,
        semantic_frame_limit=args.semantic_frame_limit,
        semantic_consistency_threshold=args.semantic_consistency_threshold,
        enable_semantic_metric=not args.disable_semantic_metric,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
