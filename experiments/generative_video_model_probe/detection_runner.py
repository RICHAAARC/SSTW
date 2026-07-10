"""generative_video_model_probe 正式视频内容检测记录构建。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

from main.core.progress import ProgressReporter
from main.protocol.flow_evidence_fields import with_flow_evidence_protocol_defaults
from main.methods.state_space_watermark.video_content_detector import (
    FORMAL_CLEAN_NEGATIVE_EVIDENCE_LEVEL,
    FORMAL_VIDEO_DETECTOR_EVIDENCE_LEVEL,
    FORMAL_VIDEO_DETECTOR_INPUT_CONTRACT,
    FORMAL_VIDEO_DETECTOR_SCORE_SEMANTICS,
    build_sstw_detector_key,
    extract_video_content_features,
    score_video_features,
    score_video_content,
)
from main.protocol.record_writer import write_json, write_jsonl
from main.protocol.table_builder import write_csv


def build_detection_records(generation_records: list[dict], attack_records: list[dict]) -> list[dict]:
    """把生成记录与攻击矩阵合并为 detection records, 未运行时不产生正向分数。

    该函数属于早期 readiness 路径的通用占位写法, 用于在真实生成模型不可运行时保留完整矩阵形状。
    它不会读取视频文件, 也不会生成可支持论文 claim 的检测分数。
    """
    records = []
    for generation_record in generation_records:
        for attack_record in attack_records:
            records.append(with_flow_evidence_protocol_defaults({
                "record_version": "generative_video_model_probe_v1",
                "generation_model_id": generation_record["generation_model_id"],
                "prompt_id": generation_record["prompt_id"],
                "seed_id": generation_record["seed_id"],
                "method_variant": "key_conditioned_state_space_with_trajectory",
                "attack_name": attack_record["attack_name"],
                "decision": "not_run",
                "decision_reason": "generation_model_not_runnable",
                "S_final": None,
                "S_trajectory_observation": None,
                "trajectory_gain_over_state_space": None,
                "trajectory_negative_leakage_delta": None,
                "negative_state_over_threshold_count": None,
            },
                trajectory_source_level="not_captured",
                flow_state_admissibility_status="not_evaluated",
                claim_support_status="not_supported_generation_model_not_runnable",
            ))
    return records


def _read_jsonl(path: Path) -> list[dict]:
    """读取 JSONL records, 文件不存在时返回空列表。"""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256_file(path: Path) -> str:
    """计算文件 sha256, 用于确认 detection 输入与 attack 输出没有断链。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """把检测分数裁剪到稳定区间, 避免异常视频导致下游表格不可比较。"""
    return max(lower, min(upper, value))


def _load_video_frame_count(video_path: Path) -> tuple[str, int, str]:
    """读取视频帧数。

    该函数只做文件级可解码性检查。正式检测分数由
    `score_video_content` 另行读取视频内容后给出。
    """
    try:
        import imageio.v3 as iio

        frame_count = sum(1 for _frame in iio.imiter(video_path))
        return "decoded", frame_count, "none"
    except Exception as exc:  # pragma: no cover - 依赖具体视频编解码后端
        return "decode_failed", 0, str(exc)

def _formal_runtime_detection_score(runtime_attack_record: dict, attacked_video_path: Path) -> dict[str, Any]:
    """对 attacked video 执行 SSTW 正式视频内容检测。

    该函数不读取 trajectory trace, 也不使用 latent callback 统计量。它只消费
    attacked video 文件和由 prompt / seed 派生的项目水印 key, 因而可以与
    external baseline 的 official detector score 放在同一 fixed-FPR 校准层比较。
    """

    detector_key = build_sstw_detector_key(runtime_attack_record)
    result = score_video_content(attacked_video_path, detector_key=detector_key)
    score = round(_clip(result.score), 6)
    return {
        "sstw_raw_detector_score": score,
        "raw_detector_score": score,
        "S_runtime_attack_detection": score,
        "S_final_conservative": score,
        "sstw_detector_score_semantics": FORMAL_VIDEO_DETECTOR_SCORE_SEMANTICS,
        "sstw_score_orientation": "higher_is_more_watermarked",
        "sstw_detector_evidence_level": FORMAL_VIDEO_DETECTOR_EVIDENCE_LEVEL,
        "sstw_detector_input_contract": FORMAL_VIDEO_DETECTOR_INPUT_CONTRACT,
        "sstw_detector_key_digest": result.detector_key_digest,
        "sstw_content_feature_count": result.content_feature_count,
        "sstw_detector_sampled_frame_count": result.sampled_frame_count,
        "trajectory_trace_used_for_score": False,
        "runtime_detection_claim_level": "formal_paper_detector",
        "attacked_video_detectable": score >= 0.5,
    }


def _is_clean_negative_generation_record(record: dict[str, Any]) -> bool:
    """判断 generation record 是否是未嵌入水印的 clean negative 视频。"""

    role_values = {
        str(record.get("sample_role") or "").lower(),
        str(record.get("generation_sample_role") or "").lower(),
        str(record.get("watermark_embedding_status") or "").lower(),
        str(record.get("method_variant") or "").lower(),
    }
    return bool(
        role_values
        & {
            "clean_negative",
            "disabled_clean_negative",
            "clean_unwatermarked_reference",
            "sstw_clean_unwatermarked_reference",
        }
    )


def _load_clean_negative_context(config_path: str | Path | None) -> dict[str, int]:
    """读取 clean negative 事件规模要求。

    通用工程写法是让检测 runner 直接读取 protocol config, 而不是在 Notebook
    cell 中硬写不同 profile 的样本数量。这样 `probe_paper`、`pilot_paper` 和
    `full_paper` 只需要切换 workflow profile 即可复用同一执行路径。
    """

    defaults = {
        "minimum_clean_negative_count": 0,
        "minimum_calibration_negative_event_count": 0,
        "minimum_heldout_test_negative_event_count": 0,
    }
    if not config_path:
        return defaults
    path = Path(config_path)
    if not path.exists():
        return defaults
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return {
        "minimum_clean_negative_count": int(payload.get("minimum_clean_negative_count") or 0),
        "minimum_calibration_negative_event_count": int(payload.get("minimum_calibration_negative_event_count") or 0),
        "minimum_heldout_test_negative_event_count": int(payload.get("minimum_heldout_test_negative_event_count") or 0),
    }


def _clean_negative_required_events_for_split(context: dict[str, int], split_name: str, clean_video_count: int) -> int:
    """返回某个 split 至少需要的 clean negative event 数。"""

    if split_name == "calibration":
        configured = context.get("minimum_calibration_negative_event_count", 0)
    elif split_name == "test":
        configured = context.get("minimum_heldout_test_negative_event_count", 0)
    else:
        configured = 0
    if configured:
        return configured
    total = int(context.get("minimum_clean_negative_count") or 0)
    if total and clean_video_count:
        return max(clean_video_count, math.ceil(total / 2))
    return clean_video_count


def _clean_negative_key_trial_count(context: dict[str, int], split_name: str, clean_video_count: int) -> int:
    """计算每个 clean video 需要派生多少个正式 key trial。"""

    if clean_video_count <= 0:
        return 0
    required = _clean_negative_required_events_for_split(context, split_name, clean_video_count)
    return max(1, math.ceil(required / clean_video_count))


def _clean_negative_detector_key(generation_record: dict[str, Any], trial_index: int) -> str:
    """为 clean negative event 派生独立 detector key。

    这些 key trial 仍然读取真实 clean video 文件并使用 SSTW 正式视频内容检测器。
    它们的作用是扩大 unwatermarked negative 分布, 用于 fixed-FPR 阈值估计。
    """

    base_key = build_sstw_detector_key(generation_record)
    return f"{base_key}::clean_negative_key_trial::{trial_index:04d}"


def build_sstw_clean_negative_score_records(
    run_root: str | Path,
    config_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """从 clean generation videos 构建 SSTW clean negative detector records。"""

    run_root = Path(run_root)
    context = _load_clean_negative_context(config_path)
    generation_records = [
        record
        for record in _read_jsonl(run_root / "records" / "generation_records.jsonl")
        if record.get("generation_status") == "success" and _is_clean_negative_generation_record(record)
    ]
    records: list[dict[str, Any]] = []
    split_video_counts: dict[str, int] = {}
    for generation_record in generation_records:
        split_name = str(generation_record.get("split") or generation_record.get("protocol_split") or "main")
        split_video_counts[split_name] = split_video_counts.get(split_name, 0) + 1
    total_trials = sum(
        _clean_negative_key_trial_count(context, split_name, count) * count
        for split_name, count in split_video_counts.items()
    )
    progress = ProgressReporter("sstw_clean_negative_video_detector", total_trials, "clean_negative_event")
    progress_index = 0
    for video_index, generation_record in enumerate(generation_records):
        video_path = Path(str(generation_record.get("video_path") or ""))
        if not video_path.exists() and video_path.name:
            video_path = run_root / "videos" / video_path.name
        split_name = str(generation_record.get("split") or generation_record.get("protocol_split") or "main")
        key_trial_count = _clean_negative_key_trial_count(context, split_name, split_video_counts.get(split_name, 0))
        try:
            if not video_path.exists():
                raise FileNotFoundError("clean_negative_video_not_found")
            features, sampled_frame_count = extract_video_content_features(video_path)
            for trial_index in range(key_trial_count):
                progress_index += 1
                progress.update(
                    progress_index,
                    (
                        f"prompt={generation_record.get('prompt_id')} "
                        f"seed={generation_record.get('seed_id')} split={split_name} "
                        f"trial={trial_index + 1}/{key_trial_count}"
                    ),
                )
                detector_key = _clean_negative_detector_key(generation_record, trial_index)
                result = score_video_features(features, detector_key=detector_key, sampled_frame_count=sampled_frame_count)
                score = round(_clip(result.score), 6)
                control_name = f"clean_negative_key_trial_{split_name}_{trial_index:04d}"
                records.append(with_flow_evidence_protocol_defaults({
                    "record_version": "sstw_clean_negative_score_v1",
                    "generation_model_id": generation_record.get("generation_model_id"),
                    "prompt_id": generation_record.get("prompt_id"),
                    "seed_id": generation_record.get("seed_id"),
                    "trajectory_trace_id": generation_record.get("trajectory_trace_id"),
                    "split": generation_record.get("split"),
                    "protocol_split": generation_record.get("protocol_split"),
                    "colab_runtime_profile": generation_record.get("colab_runtime_profile"),
                    "sample_role": "clean_negative",
                    "method_variant": "sstw_clean_unwatermarked_reference",
                    "clean_negative_video_path": str(video_path),
                    "clean_negative_video_sha256": _sha256_file(video_path),
                    "clean_negative_evidence_level": FORMAL_CLEAN_NEGATIVE_EVIDENCE_LEVEL,
                    "clean_negative_event_source": "same_prompt_seed_clean_video_key_trial",
                    "clean_negative_key_trial_index": trial_index,
                    "clean_negative_key_trial_count_for_video": key_trial_count,
                    "clean_negative_source_video_index": video_index,
                    "clean_negative_unit_id": (
                        f"sstw_clean_{generation_record.get('prompt_id')}_"
                        f"{generation_record.get('seed_id')}_{split_name}_{trial_index:04d}"
                    ),
                    "negative_family": control_name,
                    "control_name": control_name,
                    "sstw_detector_input_contract": FORMAL_VIDEO_DETECTOR_INPUT_CONTRACT,
                    "trajectory_trace_used_for_score": False,
                    "metric_status": "measured_formal",
                    "clean_negative_status": "ready",
                    "clean_negative_failure_reason": "none",
                    "sstw_clean_negative_score": score,
                    "clean_negative_score": score,
                    "sstw_raw_detector_score": score,
                    "raw_detector_score": score,
                    "sstw_score": score,
                    "sstw_clean_negative_score_semantics": FORMAL_VIDEO_DETECTOR_SCORE_SEMANTICS,
                    "sstw_score_semantics": FORMAL_VIDEO_DETECTOR_SCORE_SEMANTICS,
                    "sstw_score_orientation": "higher_is_more_watermarked",
                    "sstw_detector_key_digest": result.detector_key_digest,
                    "sstw_content_feature_count": result.content_feature_count,
                    "sstw_detector_sampled_frame_count": result.sampled_frame_count,
                    "claim_support_status": "sstw_clean_negative_video_detector_ready",
                }, trajectory_source_level="project_owned_sstw_clean_video_content_detector", claim_support_status="sstw_clean_negative_video_detector_ready"))
        except Exception as exc:  # pragma: no cover - 依赖实际视频文件和编解码后端
            progress_index += max(1, key_trial_count)
            records.append(with_flow_evidence_protocol_defaults({
                "record_version": "sstw_clean_negative_score_v1",
                "generation_model_id": generation_record.get("generation_model_id"),
                "prompt_id": generation_record.get("prompt_id"),
                "seed_id": generation_record.get("seed_id"),
                "trajectory_trace_id": generation_record.get("trajectory_trace_id"),
                "split": generation_record.get("split"),
                "protocol_split": generation_record.get("protocol_split"),
                "colab_runtime_profile": generation_record.get("colab_runtime_profile"),
                "sample_role": "clean_negative",
                "method_variant": "sstw_clean_unwatermarked_reference",
                "clean_negative_video_path": str(video_path),
                "clean_negative_evidence_level": FORMAL_CLEAN_NEGATIVE_EVIDENCE_LEVEL,
                "trajectory_trace_used_for_score": False,
                "metric_status": "missing",
                "clean_negative_status": "failed",
                "clean_negative_failure_reason": str(exc),
                "claim_support_status": "sstw_clean_negative_video_detector_blocked",
            }, trajectory_source_level="project_owned_sstw_clean_video_content_detector", claim_support_status="sstw_clean_negative_video_detector_blocked"))
    ready_count = sum(1 for record in records if record.get("clean_negative_status") == "ready")
    progress.finish(f"ready={ready_count} failed={len(records) - ready_count}")
    return records


def build_runtime_detection_records(run_root: str | Path) -> list[dict]:
    """读取 runtime attack outputs 并构造 attacked video detection records。"""
    run_root = Path(run_root)
    runtime_attack_records = _read_jsonl(run_root / "records" / "runtime_attack_records.jsonl")
    records: list[dict] = []
    progress = ProgressReporter("runtime_detection_attacked_video_scan", len(runtime_attack_records), "attacked_video")

    for index, attack_record in enumerate(runtime_attack_records):
        progress.update(
            index + 1,
            f"prompt={attack_record.get('prompt_id')} seed={attack_record.get('seed_id')} attack={attack_record.get('attack_name')}",
        )
        detection_record = with_flow_evidence_protocol_defaults({
            "record_version": "generative_video_runtime_detection_v1",
            "generation_model_id": attack_record.get("generation_model_id"),
            "prompt_id": attack_record.get("prompt_id"),
            "seed_id": attack_record.get("seed_id"),
            "trajectory_trace_id": attack_record.get("trajectory_trace_id"),
            "split": attack_record.get("split"),
            "protocol_split": attack_record.get("protocol_split"),
            "colab_runtime_profile": attack_record.get("colab_runtime_profile"),
                "method_variant": "key_conditioned_state_space_with_trajectory",
                "attack_name": attack_record.get("attack_name"),
                "runtime_detection_evidence_level": FORMAL_VIDEO_DETECTOR_EVIDENCE_LEVEL,
                "runtime_detection_status": "failed",
                "runtime_detection_failure_reason": "not_run",
            "source_video_path": attack_record.get("source_video_path"),
            "source_video_sha256": attack_record.get("source_video_sha256"),
            "attacked_video_path": attack_record.get("attacked_video_path"),
            "attacked_video_sha256": attack_record.get("attacked_video_sha256"),
            "attacked_video_decode_status": "not_run",
            "attacked_video_decode_failure_reason": "not_run",
            "source_frame_count": attack_record.get("source_frame_count", 0),
            "attacked_frame_count": attack_record.get("attacked_frame_count", 0),
                "attacked_video_decoded_frame_count": 0,
                "trajectory_trace_used_for_score": False,
                "runtime_detection_claim_level": "formal_paper_detector",
                "decision": "not_run",
                "decision_reason": "runtime_attack_not_ready",
                "claim_support_status": "sstw_formal_video_detector_ready",
            },
                negative_family=attack_record.get("negative_family"),
            trajectory_source_level="runtime_attacked_video_file_only",
                flow_state_admissibility_status="not_evaluated",
            claim_support_status="sstw_formal_video_detector_ready",
        )
        try:
            if attack_record.get("attack_runtime_status") != "ready":
                raise RuntimeError(str(attack_record.get("attack_runtime_failure_reason") or "runtime_attack_not_ready"))
            attacked_video_path = Path(str(attack_record.get("attacked_video_path") or ""))
            if not attacked_video_path.exists():
                raise FileNotFoundError("attacked_video_not_found")
            actual_digest = _sha256_file(attacked_video_path)
            if attack_record.get("attacked_video_sha256") and actual_digest != attack_record.get("attacked_video_sha256"):
                raise RuntimeError("attacked_video_sha256_mismatch")
            decode_status, decoded_frame_count, decode_reason = _load_video_frame_count(attacked_video_path)
            if decode_status != "decoded":
                raise RuntimeError(decode_reason)
            score_payload = _formal_runtime_detection_score(attack_record, attacked_video_path)
            detection_record.update({
                "runtime_detection_status": "ready",
                "runtime_detection_failure_reason": "none",
                "attacked_video_decode_status": decode_status,
                "attacked_video_decode_failure_reason": decode_reason,
                "attacked_video_decoded_frame_count": decoded_frame_count,
                "decision": "sstw_formal_detector_positive" if score_payload["attacked_video_detectable"] else "sstw_formal_detector_below_threshold",
                "decision_reason": "runtime_attacked_video_scored_by_sstw_video_content_detector",
                "flow_state_admissibility_status": "formal_video_detector_admissible"
                if score_payload["attacked_video_detectable"]
                else "formal_video_detector_below_threshold",
                **score_payload,
            })
        except Exception as exc:  # pragma: no cover - 依赖实际落盘文件和编解码后端
            detection_record.update({
                "runtime_detection_status": "failed",
                "runtime_detection_failure_reason": str(exc),
                "decision": "runtime_detection_failed",
                "decision_reason": str(exc),
            })
        records.append(detection_record)
    ready_count = sum(1 for record in records if record.get("runtime_detection_status") == "ready")
    progress.finish(f"ready={ready_count} failed={len(records) - ready_count}")
    return records


def audit_runtime_detection_records(
    records: list[dict],
    clean_negative_records: list[dict] | None = None,
    *,
    clean_negative_required: bool = False,
) -> dict:
    """审计 runtime detection records 是否完成正式视频内容检测闭环。"""
    ready_records = [record for record in records if record.get("runtime_detection_status") == "ready"]
    detectable_records = [record for record in ready_records if record.get("attacked_video_detectable") is True]
    formal_records = [
        record
        for record in ready_records
        if record.get("sstw_detector_evidence_level") == FORMAL_VIDEO_DETECTOR_EVIDENCE_LEVEL
        and record.get("trajectory_trace_used_for_score") is False
        and record.get("runtime_detection_claim_level") == "formal_paper_detector"
    ]
    attack_names = {str(record.get("attack_name")) for record in ready_records if record.get("attack_name")}
    score_values = [float(record["sstw_raw_detector_score"]) for record in ready_records if record.get("sstw_raw_detector_score") is not None]
    missing_formal_count = len(ready_records) - len(formal_records)
    clean_negative_records = list(clean_negative_records or [])
    clean_negative_ready_records = [
        record
        for record in clean_negative_records
        if record.get("metric_status") == "measured_formal"
        and record.get("clean_negative_status") == "ready"
        and record.get("clean_negative_evidence_level") == FORMAL_CLEAN_NEGATIVE_EVIDENCE_LEVEL
    ]
    clean_negative_missing_count = len(clean_negative_records) - len(clean_negative_ready_records)
    clean_negative_requirement_met = (
        not clean_negative_required
        or (bool(clean_negative_records) and clean_negative_missing_count == 0)
    )
    decision_passed = (
        bool(records)
        and len(ready_records) == len(records)
        and missing_formal_count == 0
        and clean_negative_requirement_met
    )
    return {
        "stage_id": "generative_video_runtime_detection_runner",
        "runtime_detection_decision": "PASS" if decision_passed else "FAIL",
        "runtime_detection_record_count": len(records),
        "runtime_detection_ready_count": len(ready_records),
        "runtime_detection_detectable_count": len(detectable_records),
        "runtime_detection_attack_count": len(attack_names),
        "runtime_detection_score_mean": round(mean(score_values), 6) if score_values else None,
        "runtime_detection_evidence_level": FORMAL_VIDEO_DETECTOR_EVIDENCE_LEVEL,
        "runtime_detection_formal_detector_ready_count": len(formal_records),
        "runtime_detection_formal_detector_missing_count": missing_formal_count,
        "sstw_clean_negative_record_count": len(clean_negative_records),
        "sstw_clean_negative_ready_count": len(clean_negative_ready_records),
        "sstw_clean_negative_missing_count": clean_negative_missing_count,
        "sstw_clean_negative_required": clean_negative_required,
        "sstw_clean_negative_requirement_met": clean_negative_requirement_met,
        "claim_support_status": "sstw_formal_video_detector_ready" if decision_passed else "sstw_formal_video_detector_blocked",
    }


def run_runtime_detection(run_root: str | Path, config_path: str | Path | None = None) -> dict:
    """执行 runtime attacked video detection 并写出 governed artifacts。"""
    run_root = Path(run_root)
    records = build_runtime_detection_records(run_root)
    clean_negative_records = build_sstw_clean_negative_score_records(run_root, config_path=config_path)
    clean_context = _load_clean_negative_context(config_path)
    clean_negative_required = any(value > 0 for value in clean_context.values())
    audit = audit_runtime_detection_records(
        records,
        clean_negative_records,
        clean_negative_required=clean_negative_required,
    )
    write_jsonl(run_root / "records" / "runtime_detection_records.jsonl", records)
    write_jsonl(run_root / "records" / "sstw_clean_negative_score_records.jsonl", clean_negative_records)
    write_csv(run_root / "tables" / "runtime_detection_table.csv", records)
    write_csv(run_root / "tables" / "sstw_clean_negative_score_table.csv", clean_negative_records)
    write_json(run_root / "artifacts" / "runtime_detection_decision.json", audit)
    report = (
        "# Runtime Detection Runner Report\n\n"
        "该报告记录 attacked videos 进入 SSTW 正式视频内容检测器后的 governed evidence。"
        "检测分数只来自 attacked video 文件和项目水印 key, 不读取 generation trajectory trace。\n\n"
        f"- runtime_detection_decision: {audit['runtime_detection_decision']}\n"
        f"- runtime_detection_record_count: {audit['runtime_detection_record_count']}\n"
        f"- runtime_detection_ready_count: {audit['runtime_detection_ready_count']}\n"
        f"- runtime_detection_formal_detector_ready_count: {audit['runtime_detection_formal_detector_ready_count']}\n"
        f"- runtime_detection_detectable_count: {audit['runtime_detection_detectable_count']}\n"
        f"- sstw_clean_negative_ready_count: {audit['sstw_clean_negative_ready_count']}\n"
        f"- sstw_clean_negative_required: {str(audit['sstw_clean_negative_required']).lower()}\n"
        f"- runtime_detection_score_mean: {audit['runtime_detection_score_mean']}\n"
        f"- claim_support_status: {audit['claim_support_status']}\n"
    )
    report_path = run_root / "reports" / "runtime_detection_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="对 runtime attacked videos 执行 SSTW 正式视频内容检测。")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--config-path", default="")
    args = parser.parse_args()
    payload = run_runtime_detection(args.run_root, config_path=args.config_path or None)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
