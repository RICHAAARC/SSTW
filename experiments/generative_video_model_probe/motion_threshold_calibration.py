"""B5 formal motion threshold calibration runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from main.analysis.video_file_metrics import MOTION_DELTA_MIN
from main.protocol.record_writer import write_json, write_jsonl
from main.protocol.table_builder import write_csv

CALIBRATED_MOTION_THRESHOLD_ID = "motion_delta_calibrated_v1"
HEURISTIC_MOTION_THRESHOLD_ID = "motion_delta_heuristic_v1"
DEFAULT_TARGET_STATIC_FPR = 0.01
DEFAULT_MIN_NEGATIVE_STATIC_COUNT = 128
DEFAULT_MIN_POSITIVE_MOTION_COUNT = 64
DEFAULT_MIN_AMBIGUOUS_LOW_MOTION_COUNT = 32
DEFAULT_MIN_POSITIVE_PASS_RATE = 0.80
DEFAULT_MARGIN = 0.000001
NEGATIVE_STATIC_CONTAMINATION_IQR_MULTIPLIER = 3.0


def _read_jsonl(path: Path) -> list[dict]:
    """读取 JSONL records, 文件不存在时返回空列表。"""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _key(record: dict) -> tuple[str, str, str]:
    """构造 generation record 与 formal metric record 的稳定连接键。"""
    return (
        str(record.get("generation_model_id") or ""),
        str(record.get("prompt_id") or ""),
        str(record.get("seed_id") or ""),
    )


def _infer_motion_calibration_role(generation_record: dict, formal_record: dict) -> str:
    """推断 motion calibration 中的样本角色。

    通用工程写法是优先读取显式字段。项目特定写法是: 旧 pilot records 没有 split 字段时,
    只把名称中明确包含 static / still / no_motion 的样本视为 negative_static。
    不能把普通 main motion prompt 自动当作 calibration negative。
    """
    explicit_role = generation_record.get("motion_calibration_role") or formal_record.get("motion_calibration_role")
    if explicit_role:
        return str(explicit_role)
    prompt_suite_role = str(generation_record.get("prompt_suite_role") or "")
    if prompt_suite_role in {"calibration_negative_static", "negative_static_calibration"}:
        return "negative_static"
    joined = " ".join(
        str(generation_record.get(field) or "")
        for field in ("prompt_id", "prompt_category", "motion_pattern_id")
    ).lower()
    if any(token in joined for token in ("negative_static", "static", "still", "no_motion")):
        return "negative_static"
    if generation_record.get("generation_status") == "success":
        return "positive_motion"
    return "unknown"


def _infer_source_split(generation_record: dict, formal_record: dict) -> str:
    """推断 calibration source split, 不存在显式 split 时标记为 pilot_main。"""
    for field in ("motion_threshold_source_split", "threshold_source_split", "split"):
        value = formal_record.get(field) or generation_record.get(field)
        if value:
            return str(value)
    prompt_suite_role = str(generation_record.get("prompt_suite_role") or "")
    if "calibration" in prompt_suite_role:
        return "calibration"
    return "pilot_main"


def build_motion_calibration_records(run_root: str | Path) -> list[dict]:
    """从 generation records 与 formal motion records 构造 calibration records。"""
    run_root = Path(run_root)
    generation_records = _read_jsonl(run_root / "records" / "generation_records.jsonl")
    formal_records = _read_jsonl(run_root / "records" / "formal_quality_motion_semantic_records.jsonl")
    generation_by_key = {_key(record): record for record in generation_records}
    records: list[dict] = []
    for formal_record in formal_records:
        generation_record = generation_by_key.get(_key(formal_record), {})
        source_split = _infer_source_split(generation_record, formal_record)
        role = _infer_motion_calibration_role(generation_record, formal_record)
        motion_delta = formal_record.get("motion_delta_score")
        temporal_flicker = formal_record.get("temporal_flicker_score")
        usable = motion_delta is not None and formal_record.get("video_decode_status") == "ready"
        records.append({
            "record_version": "motion_threshold_calibration_v1",
            "generation_model_id": formal_record.get("generation_model_id"),
            "prompt_id": formal_record.get("prompt_id"),
            "seed_id": formal_record.get("seed_id"),
            "trajectory_trace_id": formal_record.get("trajectory_trace_id"),
            "motion_calibration_source_split": source_split,
            "motion_calibration_role": role,
            "motion_delta_score": motion_delta,
            "temporal_flicker_score": temporal_flicker,
            "motion_calibration_record_status": "usable" if usable else "not_usable",
            "motion_calibration_record_failure_reason": "none" if usable else "missing_motion_delta_or_video_decode_not_ready",
            "previous_motion_threshold_id": HEURISTIC_MOTION_THRESHOLD_ID,
            "previous_motion_delta_threshold": MOTION_DELTA_MIN,
        })
    return records


def _select_threshold_from_negative_tail(scores: list[float], target_static_fpr: float, margin: float) -> float:
    """使用 calibration negative tail 选择冻结阈值。

    该函数属于项目特定写法。为了避免小样本低估 false positive tail, 当前使用 max negative + margin。
    target_static_fpr 保留在函数签名中, 便于后续替换为分位数或带置信区间的估计器。
    """
    _ = target_static_fpr
    if not scores:
        raise ValueError("missing_negative_static_scores")
    return round(max(scores) + margin, 6)


def _quantile(sorted_scores: list[float], fraction: float) -> float:
    """从已排序分数中取近似分位数。"""
    if not sorted_scores:
        raise ValueError("missing_scores")
    index = round((len(sorted_scores) - 1) * fraction)
    return sorted_scores[int(index)]


def _negative_static_contamination_cutoff(scores: list[float]) -> float | None:
    """估计 negative_static 中的异常高运动污染阈值。

    该实现属于项目特定写法。negative_static 的职责是估计静止负样本尾部; 如果其中少量样本
    出现明显高运动, 这些样本应进入 contamination audit, 不应直接把主阈值抬高。
    """
    if len(scores) < 4:
        return None
    sorted_scores = sorted(scores)
    q1 = _quantile(sorted_scores, 0.25)
    q3 = _quantile(sorted_scores, 0.75)
    iqr = max(q3 - q1, 0.0)
    return round(max(MOTION_DELTA_MIN, q3 + NEGATIVE_STATIC_CONTAMINATION_IQR_MULTIPLIER * iqr), 6)


def _split_negative_static_records(records: list[dict]) -> tuple[list[dict], list[dict], float | None]:
    """把 negative_static records 分为 clean tail 和疑似污染 tail。"""
    scores = [float(record["motion_delta_score"]) for record in records]
    cutoff = _negative_static_contamination_cutoff(scores)
    if cutoff is None:
        return records, [], None
    clean_records = [record for record in records if float(record["motion_delta_score"]) <= cutoff]
    contaminated_records = [record for record in records if float(record["motion_delta_score"]) > cutoff]
    return clean_records, contaminated_records, cutoff


def _calibration_decision(missing_reasons: list[str], count_missing_reasons: list[str], calibration_ready: bool) -> str:
    """把 calibration 状态压缩为稳定决策字符串。"""
    if calibration_ready:
        return "PASS"
    if count_missing_reasons:
        return "INSUFFICIENT_SAMPLE"
    if missing_reasons:
        return "FAIL_NOT_SEPARABLE"
    return "INSUFFICIENT_SAMPLE"


def _recommended_action(missing_reasons: list[str]) -> str:
    """根据失败原因给出下一步动作, 便于 notebook 直接展示。"""
    if any(reason.endswith("count_below_min") for reason in missing_reasons):
        return "collect_required_calibration_sample_counts"
    if "positive_motion_pass_rate_below_min" in missing_reasons or "positive_negative_motion_score_overlap" in missing_reasons:
        return "revise_positive_motion_prompts_or_motion_metric_then_rerun_calibration"
    return "freeze_motion_threshold_for_downstream_pilot"


def audit_motion_threshold_calibration(
    records: list[dict],
    target_static_fpr: float = DEFAULT_TARGET_STATIC_FPR,
    min_negative_static_count: int = DEFAULT_MIN_NEGATIVE_STATIC_COUNT,
    min_positive_motion_count: int = DEFAULT_MIN_POSITIVE_MOTION_COUNT,
    min_ambiguous_low_motion_count: int = DEFAULT_MIN_AMBIGUOUS_LOW_MOTION_COUNT,
    min_positive_pass_rate: float = DEFAULT_MIN_POSITIVE_PASS_RATE,
    margin: float = DEFAULT_MARGIN,
) -> dict:
    """根据 calibration records 生成 threshold artifact 与 calibration decision。"""
    usable_records = [record for record in records if record.get("motion_calibration_record_status") == "usable"]
    negative_static_records = [
        record for record in usable_records
        if record.get("motion_calibration_role") == "negative_static"
        and record.get("motion_calibration_source_split") == "calibration"
    ]
    positive_motion_records = [
        record for record in usable_records
        if record.get("motion_calibration_role") == "positive_motion"
        and record.get("motion_calibration_source_split") == "calibration"
    ]
    ambiguous_low_motion_records = [
        record for record in usable_records
        if record.get("motion_calibration_role") == "ambiguous_low_motion"
        and record.get("motion_calibration_source_split") == "calibration"
    ]

    clean_negative_static_records, contaminated_negative_static_records, contamination_cutoff = _split_negative_static_records(negative_static_records)
    negative_scores = [float(record["motion_delta_score"]) for record in negative_static_records]
    clean_negative_scores = [float(record["motion_delta_score"]) for record in clean_negative_static_records]
    contaminated_negative_scores = [float(record["motion_delta_score"]) for record in contaminated_negative_static_records]
    positive_scores = [float(record["motion_delta_score"]) for record in positive_motion_records]
    ambiguous_scores = [float(record["motion_delta_score"]) for record in ambiguous_low_motion_records]

    count_missing_reasons: list[str] = []
    if len(negative_static_records) < min_negative_static_count:
        count_missing_reasons.append("negative_static_calibration_count_below_min")
    if len(positive_motion_records) < min_positive_motion_count:
        count_missing_reasons.append("positive_motion_calibration_count_below_min")
    if len(ambiguous_low_motion_records) < min_ambiguous_low_motion_count:
        count_missing_reasons.append("ambiguous_low_motion_calibration_count_below_min")

    if not clean_negative_scores:
        threshold_value = MOTION_DELTA_MIN
        threshold_id = HEURISTIC_MOTION_THRESHOLD_ID
        threshold_source_split = "heuristic_precalibration"
    else:
        threshold_value = _select_threshold_from_negative_tail(clean_negative_scores, target_static_fpr, margin)
        threshold_id = CALIBRATED_MOTION_THRESHOLD_ID
        threshold_source_split = "calibration"

    conservative_threshold_value = _select_threshold_from_negative_tail(negative_scores, target_static_fpr, margin) if negative_scores else MOTION_DELTA_MIN
    false_positive_count = sum(1 for score in clean_negative_scores if score >= threshold_value)
    estimated_static_fpr = 0.0 if not clean_negative_scores else false_positive_count / len(clean_negative_scores)
    false_positive_count_including_contaminated = sum(1 for score in negative_scores if score >= threshold_value)
    estimated_static_fpr_including_contaminated = 0.0 if not negative_scores else false_positive_count_including_contaminated / len(negative_scores)
    positive_pass_count = sum(1 for score in positive_scores if score >= threshold_value)
    positive_pass_rate = 0.0 if not positive_scores else positive_pass_count / len(positive_scores)
    positive_negative_margin = round(min(positive_scores) - max(clean_negative_scores), 6) if positive_scores and clean_negative_scores else None

    separability_missing_reasons: list[str] = []
    if positive_scores and positive_pass_rate < min_positive_pass_rate:
        separability_missing_reasons.append("positive_motion_pass_rate_below_min")
    if positive_negative_margin is not None and positive_negative_margin <= 0:
        separability_missing_reasons.append("positive_negative_motion_score_overlap")

    missing_reasons = count_missing_reasons + separability_missing_reasons
    calibration_ready = (
        not missing_reasons
        and threshold_id == CALIBRATED_MOTION_THRESHOLD_ID
        and estimated_static_fpr <= target_static_fpr
        and positive_pass_rate >= min_positive_pass_rate
    )
    decision = _calibration_decision(missing_reasons, count_missing_reasons, calibration_ready)

    return {
        "stage_id": "motion_threshold_calibration",
        "motion_threshold_calibration_decision": decision,
        "motion_threshold_calibration_ready": calibration_ready,
        "motion_threshold_id": threshold_id,
        "motion_delta_threshold": threshold_value,
        "conservative_motion_delta_threshold": conservative_threshold_value,
        "motion_threshold_source_split": threshold_source_split,
        "threshold_source_split": threshold_source_split,
        "motion_threshold_selection_strategy": "robust_negative_tail_excluding_contamination" if contaminated_negative_static_records else "negative_tail_max_plus_margin",
        "target_static_fpr": target_static_fpr,
        "estimated_static_fpr": round(estimated_static_fpr, 6),
        "estimated_static_fpr_including_contaminated": round(estimated_static_fpr_including_contaminated, 6),
        "negative_static_contamination_status": "suspected" if contaminated_negative_static_records else "none_detected",
        "negative_static_contamination_rule": "q3_plus_3iqr",
        "negative_static_contamination_cutoff": contamination_cutoff,
        "negative_static_contamination_count": len(contaminated_negative_static_records),
        "negative_static_contaminated_prompt_count": len({record.get("prompt_id") for record in contaminated_negative_static_records}),
        "negative_static_clean_calibration_count": len(clean_negative_static_records),
        "negative_static_calibration_count": len(negative_static_records),
        "positive_motion_calibration_count": len(positive_motion_records),
        "ambiguous_low_motion_calibration_count": len(ambiguous_low_motion_records),
        "usable_motion_calibration_record_count": len(usable_records),
        "motion_calibration_record_count": len(records),
        "negative_static_motion_delta_max": round(max(negative_scores), 6) if negative_scores else None,
        "negative_static_motion_delta_mean": round(mean(negative_scores), 6) if negative_scores else None,
        "negative_static_clean_motion_delta_max": round(max(clean_negative_scores), 6) if clean_negative_scores else None,
        "negative_static_contaminated_motion_delta_min": round(min(contaminated_negative_scores), 6) if contaminated_negative_scores else None,
        "positive_motion_delta_min": round(min(positive_scores), 6) if positive_scores else None,
        "positive_motion_delta_mean": round(mean(positive_scores), 6) if positive_scores else None,
        "ambiguous_low_motion_delta_min": round(min(ambiguous_scores), 6) if ambiguous_scores else None,
        "ambiguous_low_motion_delta_mean": round(mean(ambiguous_scores), 6) if ambiguous_scores else None,
        "positive_motion_pass_rate_at_threshold": round(positive_pass_rate, 6),
        "minimum_positive_motion_pass_rate_at_threshold": min_positive_pass_rate,
        "positive_negative_motion_delta_margin": positive_negative_margin,
        "minimum_negative_static_calibration_count": min_negative_static_count,
        "minimum_positive_motion_calibration_count": min_positive_motion_count,
        "minimum_ambiguous_low_motion_calibration_count": min_ambiguous_low_motion_count,
        "motion_threshold_calibration_missing_reasons": missing_reasons,
        "motion_threshold_calibration_required": not calibration_ready,
        "motion_threshold_calibration_recommended_action": _recommended_action(missing_reasons),
        "test_time_threshold_update_blocked": True,
        "claim_support_status": "motion_threshold_calibrated" if calibration_ready else "blocked_until_motion_threshold_calibration",
    }


def run_motion_threshold_calibration(
    run_root: str | Path,
    target_static_fpr: float = DEFAULT_TARGET_STATIC_FPR,
    min_negative_static_count: int = DEFAULT_MIN_NEGATIVE_STATIC_COUNT,
    min_positive_motion_count: int = DEFAULT_MIN_POSITIVE_MOTION_COUNT,
    min_ambiguous_low_motion_count: int = DEFAULT_MIN_AMBIGUOUS_LOW_MOTION_COUNT,
    min_positive_pass_rate: float = DEFAULT_MIN_POSITIVE_PASS_RATE,
) -> dict:
    """执行 motion threshold calibration 并写出 governed records / threshold / report。"""
    run_root = Path(run_root)
    records = build_motion_calibration_records(run_root)
    audit = audit_motion_threshold_calibration(
        records,
        target_static_fpr=target_static_fpr,
        min_negative_static_count=min_negative_static_count,
        min_positive_motion_count=min_positive_motion_count,
        min_ambiguous_low_motion_count=min_ambiguous_low_motion_count,
        min_positive_pass_rate=min_positive_pass_rate,
    )
    write_jsonl(run_root / "records" / "motion_threshold_calibration_records.jsonl", records)
    write_csv(run_root / "tables" / "motion_threshold_calibration_table.csv", records)
    write_json(run_root / "thresholds" / "motion_threshold_calibration_threshold.json", audit)
    write_json(run_root / "artifacts" / "motion_threshold_calibration_decision.json", audit)
    report = (
        "# Motion Threshold Calibration Report\n\n"
        "该报告从 governed formal motion records 构造 motion threshold calibration artifact。只有显式 calibration split 中的 "
        "negative_static / positive_motion / ambiguous_low_motion 样本数量充足, 且 positive_motion 在冻结阈值下仍有足够通过率时, "
        "才允许把 threshold 标记为可支撑后续 claim。\n\n"
        f"- motion_threshold_calibration_decision: {audit['motion_threshold_calibration_decision']}\n"
        f"- motion_threshold_id: {audit['motion_threshold_id']}\n"
        f"- motion_delta_threshold: {audit['motion_delta_threshold']}\n"
        f"- conservative_motion_delta_threshold: {audit['conservative_motion_delta_threshold']}\n"
        f"- positive_motion_pass_rate_at_threshold: {audit['positive_motion_pass_rate_at_threshold']}\n"
        f"- minimum_positive_motion_pass_rate_at_threshold: {audit['minimum_positive_motion_pass_rate_at_threshold']}\n"
        f"- positive_negative_motion_delta_margin: {audit['positive_negative_motion_delta_margin']}\n"
        f"- motion_threshold_selection_strategy: {audit['motion_threshold_selection_strategy']}\n"
        f"- negative_static_contamination_status: {audit['negative_static_contamination_status']}\n"
        f"- negative_static_contamination_count: {audit['negative_static_contamination_count']}\n"
        f"- motion_threshold_source_split: {audit['motion_threshold_source_split']}\n"
        f"- negative_static_calibration_count: {audit['negative_static_calibration_count']}\n"
        f"- positive_motion_calibration_count: {audit['positive_motion_calibration_count']}\n"
        f"- ambiguous_low_motion_calibration_count: {audit['ambiguous_low_motion_calibration_count']}\n"
        f"- missing_reasons: {', '.join(audit['motion_threshold_calibration_missing_reasons']) if audit['motion_threshold_calibration_missing_reasons'] else 'none'}\n"
        f"- recommended_action: {audit['motion_threshold_calibration_recommended_action']}\n"
        f"- claim_support_status: {audit['claim_support_status']}\n"
    )
    report_path = run_root / "reports" / "motion_threshold_calibration_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="执行 formal motion threshold calibration。")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--target-static-fpr", type=float, default=DEFAULT_TARGET_STATIC_FPR)
    parser.add_argument("--min-negative-static-count", type=int, default=DEFAULT_MIN_NEGATIVE_STATIC_COUNT)
    parser.add_argument("--min-positive-motion-count", type=int, default=DEFAULT_MIN_POSITIVE_MOTION_COUNT)
    parser.add_argument("--min-ambiguous-low-motion-count", type=int, default=DEFAULT_MIN_AMBIGUOUS_LOW_MOTION_COUNT)
    parser.add_argument("--min-positive-pass-rate", type=float, default=DEFAULT_MIN_POSITIVE_PASS_RATE)
    args = parser.parse_args()
    payload = run_motion_threshold_calibration(
        args.run_root,
        target_static_fpr=args.target_static_fpr,
        min_negative_static_count=args.min_negative_static_count,
        min_positive_motion_count=args.min_positive_motion_count,
        min_ambiguous_low_motion_count=args.min_ambiguous_low_motion_count,
        min_positive_pass_rate=args.min_positive_pass_rate,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
