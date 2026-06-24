"""B5 formal motion threshold calibration runner.

该模块只负责 motion observability / prompt validity 层面的阈值校准。
污染过滤不得读取或依赖最终水印检测分数, 例如 S_final、S_final_conservative 或 watermark_detection_score。
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from main.analysis.video_file_metrics import MOTION_DELTA_MIN
from main.protocol.flow_evidence_fields import with_flow_evidence_protocol_defaults
from main.protocol.record_writer import write_json, write_jsonl
from main.protocol.table_builder import write_csv

CALIBRATED_MOTION_THRESHOLD_ID = "motion_delta_calibrated_v1"
HEURISTIC_MOTION_THRESHOLD_ID = "motion_delta_heuristic_v1"
DEFAULT_TARGET_STATIC_FPR = 0.05
DEFAULT_THRESHOLD_QUANTILE = 0.95
DEFAULT_MIN_NEGATIVE_STATIC_COUNT = 128
DEFAULT_MIN_POSITIVE_MOTION_COUNT = 64
DEFAULT_MIN_AMBIGUOUS_LOW_MOTION_COUNT = 32
DEFAULT_MIN_POSITIVE_PASS_RATE = 0.80
DEFAULT_MIN_POSITIVE_PASS_RATE_WILSON_LOWER = 0.70
DEFAULT_MIN_VISUAL_READY_RATE = 0.98
DEFAULT_MIN_MOTION_READY_RATE = 0.98
DEFAULT_PROMPT_PASS_LIKE_RATIO_MAX = 0.50
DEFAULT_MARGIN = 0.000001
NEGATIVE_STATIC_CONTAMINATION_IQR_MULTIPLIER = 3.0
CONTAMINATION_DECISION_SOURCE = "motion_observability_score_only"
MOTION_SCORE_VERSION = "focus_only_v0"
CALIBRATION_SCORE_ROLE = "engineering_prompt_audit"


def _read_jsonl(path: Path) -> list[dict]:
    """读取 JSONL records, 文件不存在时返回空列表。"""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _key(record: dict) -> tuple[str, str, str]:
    """构造 generation record 与 formal metric record 的稳定连接键。"""
    return (str(record.get("generation_model_id") or ""), str(record.get("prompt_id") or ""), str(record.get("seed_id") or ""))


def _infer_motion_calibration_role(generation_record: dict, formal_record: dict) -> str:
    """推断 motion calibration 中的样本角色。"""
    explicit_role = generation_record.get("motion_calibration_role") or formal_record.get("motion_calibration_role")
    if explicit_role:
        return str(explicit_role)
    prompt_suite_role = str(generation_record.get("prompt_suite_role") or "")
    if prompt_suite_role in {"calibration_negative_static", "negative_static_calibration"}:
        return "negative_static"
    joined = " ".join(str(generation_record.get(field) or "") for field in ("prompt_id", "prompt_category", "motion_pattern_id")).lower()
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
    if "calibration" in str(generation_record.get("prompt_suite_role") or ""):
        return "calibration"
    return "pilot_main"


def _motion_calibration_score(formal_record: dict) -> tuple[float | None, str]:
    """选择 motion calibration 使用的主分数。

    该分数只表示 prompt validity / motion observability / calibration eligibility,
    不表示最终水印检测分数。
    """
    if formal_record.get("motion_delta_focus_score") is not None:
        return float(formal_record["motion_delta_focus_score"]), "motion_delta_focus_score"
    if formal_record.get("motion_delta_score") is not None:
        return float(formal_record["motion_delta_score"]), "motion_delta_score_backward_fallback"
    return None, "missing_motion_score"


def _safe_bool(value: Any, default: bool = True) -> bool:
    """读取历史 records 中可能缺失的布尔字段。"""
    if value is None:
        return default
    return value is True


def build_motion_calibration_records(run_root: str | Path) -> list[dict]:
    """从 generation records 与 formal motion records 构造 calibration records。"""
    run_root = Path(run_root)
    generation_records = _read_jsonl(run_root / "records" / "generation_records.jsonl")
    formal_records = _read_jsonl(run_root / "records" / "formal_quality_motion_semantic_records.jsonl")
    generation_by_key = {_key(record): record for record in generation_records}
    records: list[dict] = []
    for formal_record in formal_records:
        generation_record = generation_by_key.get(_key(formal_record), {})
        motion_score, motion_score_name = _motion_calibration_score(formal_record)
        usable = motion_score is not None and formal_record.get("video_decode_status") == "ready"
        records.append(with_flow_evidence_protocol_defaults({
            "record_version": "motion_threshold_calibration_v2",
            "generation_model_id": formal_record.get("generation_model_id"),
            "prompt_id": formal_record.get("prompt_id"),
            "seed_id": formal_record.get("seed_id"),
            "trajectory_trace_id": formal_record.get("trajectory_trace_id"),
            "negative_family": _infer_motion_calibration_role(generation_record, formal_record)
            if _infer_motion_calibration_role(generation_record, formal_record) in {"negative_static", "ambiguous_low_motion"}
            else None,
            "motion_calibration_source_split": _infer_source_split(generation_record, formal_record),
            "motion_calibration_role": _infer_motion_calibration_role(generation_record, formal_record),
            "motion_delta_score": formal_record.get("motion_delta_score"),
            "motion_delta_focus_score": formal_record.get("motion_delta_focus_score"),
            "motion_delta_p90_score": formal_record.get("motion_delta_p90_score"),
            "motion_delta_top10_mean_score": formal_record.get("motion_delta_top10_mean_score"),
            "motion_calibration_score": motion_score,
            "motion_calibration_score_name": motion_score_name,
            "motion_score_version": MOTION_SCORE_VERSION,
            "calibration_score_role": CALIBRATION_SCORE_ROLE,
            "not_watermark_detection_score": True,
            "temporal_flicker_score": formal_record.get("temporal_flicker_score"),
            "formal_visual_quality_ready": formal_record.get("formal_visual_quality_ready"),
            "formal_motion_consistency_ready": formal_record.get("formal_motion_consistency_ready"),
            "motion_calibration_record_status": "usable" if usable else "not_usable",
            "motion_calibration_record_failure_reason": "none" if usable else "missing_motion_score_or_video_decode_not_ready",
            "previous_motion_threshold_id": HEURISTIC_MOTION_THRESHOLD_ID,
            "previous_motion_delta_threshold": MOTION_DELTA_MIN,
            "contamination_decision_source": CONTAMINATION_DECISION_SOURCE,
            "final_detection_score_filtering_blocked": True,
            "no_final_detection_score_used_for_filtering": True,
        },
            trajectory_source_level="formal_motion_metric_calibration",
            flow_state_admissibility_status="motion_calibration_usable" if usable else "motion_calibration_not_usable",
            claim_support_status="motion_threshold_calibration_record_only",
        ))
    return records


def _sorted_scores(scores: list[float]) -> list[float]:
    return sorted(float(score) for score in scores)


def _quantile(sorted_scores: list[float], fraction: float) -> float:
    """从已排序分数中取确定性的 nearest-rank 近似分位数。"""
    if not sorted_scores:
        raise ValueError("missing_scores")
    fraction = min(max(fraction, 0.0), 1.0)
    return float(sorted_scores[round((len(sorted_scores) - 1) * fraction)])


def _median_abs_deviation(scores: list[float]) -> float:
    """计算 MAD, 用于 robust prompt-level contamination audit。"""
    if not scores:
        return 0.0
    center = median(scores)
    return float(median([abs(score - center) for score in scores]))


def _select_threshold_from_negative_tail(scores: list[float], target_static_fpr: float = DEFAULT_TARGET_STATIC_FPR) -> float:
    """使用 fixed-FPR quantile 选择工程阶段冻结阈值。"""
    if not scores:
        raise ValueError("missing_negative_static_scores")
    return round(_quantile(_sorted_scores(scores), 1.0 - target_static_fpr), 6)


def _negative_static_contamination_cutoff(scores: list[float]) -> float | None:
    """估计 record-level high-motion contamination cutoff。"""
    if len(scores) < 4:
        return None
    sorted_scores = _sorted_scores(scores)
    q1 = _quantile(sorted_scores, 0.25)
    q3 = _quantile(sorted_scores, 0.75)
    return round(max(MOTION_DELTA_MIN, q3 + NEGATIVE_STATIC_CONTAMINATION_IQR_MULTIPLIER * max(q3 - q1, 0.0)), 6)


def _records_by_prompt(records: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("prompt_id") or "unknown_prompt")].append(record)
    return dict(grouped)

def _build_prompt_contamination_audit(
    negative_static_records: list[dict],
    pass_like_ratio_max: float = DEFAULT_PROMPT_PASS_LIKE_RATIO_MAX,
) -> tuple[list[dict], set[str], dict[str, Any]]:
    """构造 prompt-level contamination audit。

    污染判定只使用 motion_calibration_score, 不能使用 S_final 或最终检测分数。
    被污染 prompt 不参与 threshold estimation, 但保留进入 stress negative audit。
    """
    scores = [float(record["motion_calibration_score"]) for record in negative_static_records]
    if not scores:
        return [], set(), {"global_negative_median": None, "global_negative_mad": None, "preliminary_cutoff": None}
    global_median = float(median(scores))
    global_mad = _median_abs_deviation(scores)
    preliminary_cutoff = round(max(global_median + 3.0 * global_mad, global_median + DEFAULT_MARGIN), 6)
    p95_cutoff = round(_quantile(_sorted_scores(scores), 0.95), 6)
    audits: list[dict] = []
    contaminated_prompt_ids: set[str] = set()
    for prompt_id, prompt_records in sorted(_records_by_prompt(negative_static_records).items()):
        prompt_scores = [float(record["motion_calibration_score"]) for record in prompt_records]
        prompt_median = float(median(prompt_scores))
        prompt_mad = _median_abs_deviation(prompt_scores)
        pass_like_count = sum(1 for score in prompt_scores if score > preliminary_cutoff)
        pass_like_ratio = pass_like_count / len(prompt_scores) if prompt_scores else 0.0
        enough_seeds = len(prompt_scores) >= 4
        reasons: list[str] = []
        if enough_seeds and prompt_median > preliminary_cutoff:
            reasons.append("prompt_median_above_global_median_plus_3mad")
        if enough_seeds and pass_like_ratio >= pass_like_ratio_max:
            reasons.append("prompt_pass_like_ratio_above_max")
        contaminated = bool(reasons)
        if contaminated:
            contaminated_prompt_ids.add(prompt_id)
        audits.append({
            "record_version": "prompt_contamination_audit_v1",
            "prompt_id": prompt_id,
            "sample_role": "negative_static",
            "seed_count": len(prompt_scores),
            "prompt_median_motion_score": round(prompt_median, 6),
            "prompt_mad_motion_score": round(prompt_mad, 6),
            "prompt_pass_like_count": pass_like_count,
            "prompt_pass_like_ratio": round(pass_like_ratio, 6),
            "global_negative_median_motion_score": round(global_median, 6),
            "global_negative_mad_motion_score": round(global_mad, 6),
            "preliminary_cutoff": preliminary_cutoff,
            "preliminary_cutoff_source": "calibration_negative_motion_score_median_plus_3mad",
            "preliminary_p95_cutoff": p95_cutoff,
            "contamination_status": "contaminated_prompt" if contaminated else "clean_prompt",
            "contamination_reason": "none" if not reasons else ";".join(reasons),
            "contamination_rule_id": "prompt_median_3mad_or_pass_like_ratio_v1",
            "contamination_decision_source": CONTAMINATION_DECISION_SOURCE,
            "excluded_from_threshold": contaminated,
            "excluded_from_threshold_estimation": contaminated,
            "included_in_contamination_audit": True,
            "included_in_stress_eval": contaminated,
            "included_in_stress_negative_eval": contaminated,
            "not_final_detection_score_based": True,
            "final_detection_score_filtering_blocked": True,
        })
    return audits, contaminated_prompt_ids, {
        "global_negative_median": round(global_median, 6),
        "global_negative_mad": round(global_mad, 6),
        "preliminary_cutoff": preliminary_cutoff,
        "preliminary_cutoff_source": "calibration_negative_motion_score_median_plus_3mad",
        "preliminary_p95_cutoff": p95_cutoff,
    }


def _split_negative_static_records(records: list[dict], prompt_contaminated_ids: set[str]) -> tuple[list[dict], list[dict], list[dict], float | None]:
    """把 negative_static records 分为 threshold-clean、record-contaminated 和 prompt-contaminated。"""
    prompt_contaminated_records = [record for record in records if str(record.get("prompt_id") or "unknown_prompt") in prompt_contaminated_ids]
    prompt_clean_records = [record for record in records if str(record.get("prompt_id") or "unknown_prompt") not in prompt_contaminated_ids]
    scores = [float(record["motion_calibration_score"]) for record in prompt_clean_records]
    cutoff = _negative_static_contamination_cutoff(scores)
    if cutoff is None:
        clean_records = prompt_clean_records
        record_contaminated_records: list[dict] = []
    else:
        clean_records = [record for record in prompt_clean_records if float(record["motion_calibration_score"]) <= cutoff]
        record_contaminated_records = [record for record in prompt_clean_records if float(record["motion_calibration_score"]) > cutoff]
    record_contaminated_ids = {id(record) for record in record_contaminated_records}
    for record in records:
        prompt_id = str(record.get("prompt_id") or "unknown_prompt")
        prompt_contaminated = prompt_id in prompt_contaminated_ids
        record_contaminated = id(record) in record_contaminated_ids
        excluded = prompt_contaminated or record_contaminated
        reason = "prompt_level_contamination" if prompt_contaminated else "record_level_high_motion_tail" if record_contaminated else "none"
        record.update({
            "prompt_contamination_status": "contaminated_prompt" if prompt_contaminated else "clean_prompt",
            "prompt_contamination_reason": "prompt_level_contamination" if prompt_contaminated else "none",
            "prompt_contamination_score": record.get("motion_calibration_score"),
            "contamination_rule_id": "prompt_median_3mad_or_pass_like_ratio_v1;record_iqr_tail_v1",
            "contamination_decision_source": CONTAMINATION_DECISION_SOURCE,
            "excluded_from_threshold_estimation": excluded,
            "threshold_estimation_exclusion_reason": reason,
            "included_in_contamination_audit": True,
            "included_in_stress_negative_eval": excluded,
            "not_final_detection_score_based": True,
            "final_detection_score_filtering_blocked": True,
        })
    return clean_records, record_contaminated_records, prompt_contaminated_records, cutoff


def _wilson_lower_bound(success_count: int, total_count: int, z: float = 1.96) -> float:
    """计算 Wilson 置信区间下界, 用于小样本 positive pass rate 审计。"""
    if total_count <= 0:
        return 0.0
    p_hat = success_count / total_count
    denominator = 1.0 + z * z / total_count
    center = p_hat + z * z / (2.0 * total_count)
    spread = z * math.sqrt((p_hat * (1.0 - p_hat) + z * z / (4.0 * total_count)) / total_count)
    return max(0.0, (center - spread) / denominator)


def _ready_rate(records: list[dict], field: str) -> float:
    """计算 formal quality / motion ready rate, 历史缺失字段默认视为 ready。"""
    if not records:
        return 0.0
    return sum(1 for record in records if _safe_bool(record.get(field), default=True)) / len(records)


def _leave_one_prompt_out_delta(clean_records: list[dict], threshold_value: float, target_static_fpr: float) -> tuple[float, list[str]]:
    """计算 leave-one-prompt-out threshold 最大变化。"""
    grouped = _records_by_prompt(clean_records)
    deltas: list[tuple[float, str]] = []
    for prompt_id in grouped:
        remaining = [float(record["motion_calibration_score"]) for record in clean_records if str(record.get("prompt_id") or "unknown_prompt") != prompt_id]
        if remaining:
            deltas.append((abs(_select_threshold_from_negative_tail(remaining, target_static_fpr) - threshold_value), prompt_id))
    if not deltas:
        return 0.0, []
    max_delta = max(delta for delta, _prompt_id in deltas)
    return round(max_delta, 6), [prompt_id for delta, prompt_id in sorted(deltas, reverse=True) if delta == max_delta][:5]


def _bootstrap_threshold_ci(scores: list[float], target_static_fpr: float, iterations: int = 200) -> tuple[float | None, float | None]:
    """用确定性 bootstrap 估计 threshold 稳定区间。"""
    if not scores:
        return None, None
    rng = random.Random(1337)
    thresholds = [_select_threshold_from_negative_tail([scores[rng.randrange(len(scores))] for _item in scores], target_static_fpr) for _ in range(iterations)]
    sorted_thresholds = sorted(thresholds)
    return round(_quantile(sorted_thresholds, 0.025), 6), round(_quantile(sorted_thresholds, 0.975), 6)


def _build_threshold_stability_audit(threshold_value: float, clean_negative_static_records: list[dict], negative_static_records: list[dict], prompt_contamination_audits: list[dict], target_static_fpr: float) -> dict:
    """构造 threshold stability audit artifact。"""
    clean_scores = [float(record["motion_calibration_score"]) for record in clean_negative_static_records]
    ci_low, ci_high = _bootstrap_threshold_ci(clean_scores, target_static_fpr)
    loo_delta, loo_prompts = _leave_one_prompt_out_delta(clean_negative_static_records, threshold_value, target_static_fpr)
    tail_records = [record for record in clean_negative_static_records if float(record["motion_calibration_score"]) >= threshold_value]
    tail_counts = Counter(str(record.get("prompt_id") or "unknown_prompt") for record in tail_records)
    contaminated_prompt_count = sum(1 for item in prompt_contamination_audits if item.get("contamination_status") == "contaminated_prompt")
    return {
        "record_version": "threshold_stability_audit_v1",
        "threshold_value": threshold_value,
        "threshold_quantile": DEFAULT_THRESHOLD_QUANTILE,
        "target_static_fpr_engineering": target_static_fpr,
        "motion_threshold_evidence_level": "engineering_calibration",
        "not_final_paper_fpr_0_01": True,
        "negative_count_before_filter": len(negative_static_records),
        "negative_count_after_filter": len(clean_negative_static_records),
        "contaminated_prompt_count": contaminated_prompt_count,
        "leave_one_prompt_out_threshold_delta_max": loo_delta,
        "leave_one_prompt_out_tail_dominant_prompt_ids": loo_prompts,
        "bootstrap_threshold_ci_low": ci_low,
        "bootstrap_threshold_ci_high": ci_high,
        "tail_dominant_prompt_ids": [prompt_id for prompt_id, _count in tail_counts.most_common(5)],
        "contamination_decision_source": CONTAMINATION_DECISION_SOURCE,
        "not_final_detection_score_based": True,
        "final_detection_score_filtering_blocked": True,
    }

def _calibration_decision(count_missing_reasons: list[str], calibration_ready: bool) -> str:
    """把 calibration 状态压缩为稳定决策字符串。"""
    if calibration_ready:
        return "PASS"
    if count_missing_reasons:
        return "INSUFFICIENT_SAMPLE"
    return "FAIL_NOT_SEPARABLE"


def _recommended_action(missing_reasons: list[str]) -> str:
    """根据失败原因给出下一步动作, 便于 notebook 直接展示。"""
    if any(reason.endswith("count_below_min") for reason in missing_reasons):
        return "collect_required_calibration_sample_counts"
    if "positive_motion_pass_rate_below_min" in missing_reasons or "positive_motion_pass_rate_wilson_lower_below_min" in missing_reasons:
        return "revise_positive_motion_prompts_or_motion_metric_then_rerun_calibration"
    if "formal_visual_quality_ready_rate_below_min" in missing_reasons:
        return "revise_prompt_suite_or_quality_gate_then_rerun_calibration"
    return "freeze_engineering_motion_threshold_for_downstream_pilot"


def audit_motion_threshold_calibration(
    records: list[dict],
    target_static_fpr: float = DEFAULT_TARGET_STATIC_FPR,
    min_negative_static_count: int = DEFAULT_MIN_NEGATIVE_STATIC_COUNT,
    min_positive_motion_count: int = DEFAULT_MIN_POSITIVE_MOTION_COUNT,
    min_ambiguous_low_motion_count: int = DEFAULT_MIN_AMBIGUOUS_LOW_MOTION_COUNT,
    min_positive_pass_rate: float = DEFAULT_MIN_POSITIVE_PASS_RATE,
    min_positive_pass_rate_wilson_lower: float = DEFAULT_MIN_POSITIVE_PASS_RATE_WILSON_LOWER,
    min_visual_ready_rate: float = DEFAULT_MIN_VISUAL_READY_RATE,
    min_motion_ready_rate: float = DEFAULT_MIN_MOTION_READY_RATE,
) -> dict:
    """根据 calibration records 生成 threshold artifact 与 calibration decision。"""
    usable_records = [record for record in records if record.get("motion_calibration_record_status") == "usable"]
    negative_static_records = [record for record in usable_records if record.get("motion_calibration_role") == "negative_static" and record.get("motion_calibration_source_split") == "calibration"]
    positive_motion_records = [record for record in usable_records if record.get("motion_calibration_role") == "positive_motion" and record.get("motion_calibration_source_split") == "calibration"]
    ambiguous_low_motion_records = [record for record in usable_records if record.get("motion_calibration_role") == "ambiguous_low_motion" and record.get("motion_calibration_source_split") == "calibration"]

    prompt_audits, prompt_contaminated_ids, prompt_context = _build_prompt_contamination_audit(negative_static_records)
    clean_negative_records, record_contaminated_records, prompt_contaminated_records, contamination_cutoff = _split_negative_static_records(negative_static_records, prompt_contaminated_ids)

    negative_scores = [float(record["motion_calibration_score"]) for record in negative_static_records]
    clean_negative_scores = [float(record["motion_calibration_score"]) for record in clean_negative_records]
    record_contaminated_scores = [float(record["motion_calibration_score"]) for record in record_contaminated_records]
    prompt_contaminated_scores = [float(record["motion_calibration_score"]) for record in prompt_contaminated_records]
    contaminated_scores = record_contaminated_scores + prompt_contaminated_scores
    positive_scores = [float(record["motion_calibration_score"]) for record in positive_motion_records]
    ambiguous_scores = [float(record["motion_calibration_score"]) for record in ambiguous_low_motion_records]

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
        threshold_value = _select_threshold_from_negative_tail(clean_negative_scores, target_static_fpr)
        threshold_id = CALIBRATED_MOTION_THRESHOLD_ID
        threshold_source_split = "calibration"

    conservative_threshold_value = _select_threshold_from_negative_tail(negative_scores, target_static_fpr) if negative_scores else MOTION_DELTA_MIN
    false_positive_count = sum(1 for score in clean_negative_scores if score > threshold_value)
    estimated_static_fpr = 0.0 if not clean_negative_scores else false_positive_count / len(clean_negative_scores)
    fp_including_contaminated = sum(1 for score in negative_scores if score > threshold_value)
    estimated_static_fpr_including_contaminated = 0.0 if not negative_scores else fp_including_contaminated / len(negative_scores)
    positive_pass_count = sum(1 for score in positive_scores if score >= threshold_value)
    positive_pass_rate = 0.0 if not positive_scores else positive_pass_count / len(positive_scores)
    positive_wilson_lower = _wilson_lower_bound(positive_pass_count, len(positive_scores))
    positive_negative_margin = round(min(positive_scores) - max(clean_negative_scores), 6) if positive_scores and clean_negative_scores else None
    visual_ready_rate = _ready_rate(usable_records, "formal_visual_quality_ready")
    motion_ready_rate = _ready_rate(usable_records, "formal_motion_consistency_ready")

    missing_reasons = list(count_missing_reasons)
    if positive_scores and positive_pass_rate < min_positive_pass_rate:
        missing_reasons.append("positive_motion_pass_rate_below_min")
    if positive_scores and positive_wilson_lower < min_positive_pass_rate_wilson_lower:
        missing_reasons.append("positive_motion_pass_rate_wilson_lower_below_min")
    if positive_scores and positive_pass_rate < min_positive_pass_rate and positive_negative_margin is not None and positive_negative_margin <= 0:
        missing_reasons.append("positive_negative_motion_score_overlap")
    if usable_records and visual_ready_rate < min_visual_ready_rate:
        missing_reasons.append("formal_visual_quality_ready_rate_below_min")
    if usable_records and motion_ready_rate < min_motion_ready_rate:
        missing_reasons.append("formal_motion_consistency_ready_rate_below_min")

    engineering_ready = (
        not missing_reasons
        and threshold_id == CALIBRATED_MOTION_THRESHOLD_ID
        and estimated_static_fpr <= target_static_fpr
        and positive_pass_rate >= min_positive_pass_rate
        and positive_wilson_lower >= min_positive_pass_rate_wilson_lower
        and visual_ready_rate >= min_visual_ready_rate
        and motion_ready_rate >= min_motion_ready_rate
    )
    decision = _calibration_decision(count_missing_reasons, engineering_ready)
    threshold_stability = _build_threshold_stability_audit(threshold_value, clean_negative_records, negative_static_records, prompt_audits, target_static_fpr)

    return {
        "stage_id": "motion_threshold_calibration",
        "motion_threshold_calibration_decision": decision,
        "engineering_motion_threshold_calibration_decision": decision,
        "paper_fixed_fpr_calibration_decision": "NOT_READY_REQUIRES_LARGE_HELDOUT_NEGATIVE_SPLIT",
        "motion_threshold_calibration_ready": engineering_ready,
        "engineering_motion_threshold_calibration_ready": engineering_ready,
        "paper_fixed_fpr_calibration_ready": False,
        "motion_threshold_id": threshold_id,
        "motion_delta_threshold": threshold_value,
        "motion_calibration_score_name": "motion_delta_focus_score_preferred",
        "motion_score_version": MOTION_SCORE_VERSION,
        "calibration_score_role": CALIBRATION_SCORE_ROLE,
        "not_watermark_detection_score": True,
        "conservative_motion_delta_threshold": conservative_threshold_value,
        "motion_threshold_source_split": threshold_source_split,
        "threshold_source_split": threshold_source_split,
        "motion_threshold_selection_strategy": "prompt_aware_robust_quantile_p95",
        "threshold_quantile": DEFAULT_THRESHOLD_QUANTILE,
        "target_static_fpr": target_static_fpr,
        "target_static_fpr_engineering": target_static_fpr,
        "motion_threshold_evidence_level": "engineering_calibration",
        "not_final_paper_fpr_0_01": True,
        "paper_fpr_0_01_negative_count_required": 1000,
        "estimated_static_fpr": round(estimated_static_fpr, 6),
        "estimated_static_fpr_including_contaminated": round(estimated_static_fpr_including_contaminated, 6),
        "negative_static_contamination_status": "suspected" if contaminated_scores else "none_detected",
        "negative_static_contamination_rule": "prompt_median_3mad_or_pass_like_ratio_plus_record_iqr",
        "negative_static_contamination_cutoff": contamination_cutoff,
        "negative_static_contamination_count": len(contaminated_scores),
        "negative_static_record_contamination_count": len(record_contaminated_records),
        "negative_static_prompt_contamination_count": len(prompt_contaminated_records),
        "negative_static_contaminated_prompt_count": len(prompt_contaminated_ids),
        "negative_static_clean_calibration_count": len(clean_negative_records),
        "negative_static_calibration_count": len(negative_static_records),
        "positive_motion_calibration_count": len(positive_motion_records),
        "ambiguous_low_motion_calibration_count": len(ambiguous_low_motion_records),
        "usable_motion_calibration_record_count": len(usable_records),
        "motion_calibration_record_count": len(records),
        "negative_static_motion_delta_max": round(max(negative_scores), 6) if negative_scores else None,
        "negative_static_motion_delta_mean": round(mean(negative_scores), 6) if negative_scores else None,
        "negative_static_clean_motion_delta_max": round(max(clean_negative_scores), 6) if clean_negative_scores else None,
        "negative_static_contaminated_motion_delta_min": round(min(contaminated_scores), 6) if contaminated_scores else None,
        "positive_motion_delta_min": round(min(positive_scores), 6) if positive_scores else None,
        "positive_motion_delta_mean": round(mean(positive_scores), 6) if positive_scores else None,
        "ambiguous_low_motion_delta_min": round(min(ambiguous_scores), 6) if ambiguous_scores else None,
        "ambiguous_low_motion_delta_mean": round(mean(ambiguous_scores), 6) if ambiguous_scores else None,
        "positive_motion_pass_count_at_threshold": positive_pass_count,
        "positive_motion_pass_rate_at_threshold": round(positive_pass_rate, 6),
        "positive_motion_pass_rate_wilson_lower": round(positive_wilson_lower, 6),
        "minimum_positive_motion_pass_rate_at_threshold": min_positive_pass_rate,
        "minimum_positive_motion_pass_rate_wilson_lower": min_positive_pass_rate_wilson_lower,
        "positive_negative_motion_delta_margin": positive_negative_margin,
        "formal_visual_quality_ready_rate": round(visual_ready_rate, 6),
        "formal_motion_consistency_ready_rate": round(motion_ready_rate, 6),
        "minimum_formal_visual_quality_ready_rate": min_visual_ready_rate,
        "minimum_formal_motion_consistency_ready_rate": min_motion_ready_rate,
        "minimum_negative_static_calibration_count": min_negative_static_count,
        "minimum_positive_motion_calibration_count": min_positive_motion_count,
        "minimum_ambiguous_low_motion_calibration_count": min_ambiguous_low_motion_count,
        "motion_threshold_calibration_missing_reasons": missing_reasons,
        "motion_threshold_calibration_required": not engineering_ready,
        "motion_threshold_calibration_recommended_action": _recommended_action(missing_reasons),
        "test_time_threshold_update_blocked": True,
        "contamination_decision_source": CONTAMINATION_DECISION_SOURCE,
        "final_detection_score_filtering_blocked": True,
        "no_final_detection_score_used_for_filtering": True,
        "prompt_contamination_audit": prompt_audits,
        "prompt_contamination_context": prompt_context,
        "threshold_stability_audit": threshold_stability,
        "claim_support_status": "motion_threshold_engineering_calibrated" if engineering_ready else "blocked_until_motion_threshold_calibration",
    }

def _write_prompt_contamination_outputs(run_root: Path, audit: dict) -> None:
    """写出 prompt-level contamination audit records / table / artifact。"""
    prompt_audit = [
        with_flow_evidence_protocol_defaults(
            record,
            trajectory_source_level="prompt_level_motion_calibration_audit",
            flow_state_admissibility_status="not_evaluated",
            claim_support_status="prompt_contamination_audit_record_only",
        )
        for record in audit.get("prompt_contamination_audit", [])
    ]
    write_jsonl(run_root / "records" / "prompt_contamination_audit_records.jsonl", prompt_audit)
    write_csv(run_root / "tables" / "prompt_contamination_audit_table.csv", prompt_audit)
    write_json(run_root / "artifacts" / "prompt_contamination_audit.json", {
        "stage_id": "prompt_contamination_audit",
        "prompt_contamination_audit_record_count": len(prompt_audit),
        "contaminated_prompt_count": sum(1 for row in prompt_audit if row.get("contamination_status") == "contaminated_prompt"),
        "contamination_decision_source": CONTAMINATION_DECISION_SOURCE,
        "final_detection_score_filtering_blocked": True,
        "no_final_detection_score_used_for_filtering": True,
        "records": prompt_audit,
    })


def _serializable_audit(audit: dict) -> dict:
    """移除大块内联 audit records, 避免主 decision 文件过大。"""
    payload = dict(audit)
    payload.pop("prompt_contamination_audit", None)
    return payload


def run_motion_threshold_calibration(
    run_root: str | Path,
    target_static_fpr: float = DEFAULT_TARGET_STATIC_FPR,
    min_negative_static_count: int = DEFAULT_MIN_NEGATIVE_STATIC_COUNT,
    min_positive_motion_count: int = DEFAULT_MIN_POSITIVE_MOTION_COUNT,
    min_ambiguous_low_motion_count: int = DEFAULT_MIN_AMBIGUOUS_LOW_MOTION_COUNT,
    min_positive_pass_rate: float = DEFAULT_MIN_POSITIVE_PASS_RATE,
    min_positive_pass_rate_wilson_lower: float = DEFAULT_MIN_POSITIVE_PASS_RATE_WILSON_LOWER,
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
        min_positive_pass_rate_wilson_lower=min_positive_pass_rate_wilson_lower,
    )
    serializable_audit = _serializable_audit(audit)
    write_jsonl(run_root / "records" / "motion_threshold_calibration_records.jsonl", records)
    write_csv(run_root / "tables" / "motion_threshold_calibration_table.csv", records)
    write_json(run_root / "thresholds" / "motion_threshold_calibration_threshold.json", serializable_audit)
    write_json(run_root / "artifacts" / "motion_threshold_calibration_decision.json", serializable_audit)
    write_json(run_root / "artifacts" / "threshold_stability_audit.json", audit["threshold_stability_audit"])
    _write_prompt_contamination_outputs(run_root, audit)
    report = (
        "# Motion Threshold Calibration Report\n\n"
        "该报告从 governed formal motion records 构造 engineering motion threshold calibration artifact。"
        "污染过滤只使用 motion observability score, 不使用 S_final 或最终水印检测分数。\n\n"
        f"- motion_threshold_calibration_decision: {audit['motion_threshold_calibration_decision']}\n"
        f"- motion_threshold_evidence_level: {audit['motion_threshold_evidence_level']}\n"
        f"- paper_fixed_fpr_calibration_decision: {audit['paper_fixed_fpr_calibration_decision']}\n"
        f"- motion_threshold_id: {audit['motion_threshold_id']}\n"
        f"- motion_delta_threshold: {audit['motion_delta_threshold']}\n"
        f"- threshold_quantile: {audit['threshold_quantile']}\n"
        f"- target_static_fpr_engineering: {audit['target_static_fpr_engineering']}\n"
        f"- positive_motion_pass_rate_at_threshold: {audit['positive_motion_pass_rate_at_threshold']}\n"
        f"- positive_motion_pass_rate_wilson_lower: {audit['positive_motion_pass_rate_wilson_lower']}\n"
        f"- formal_visual_quality_ready_rate: {audit['formal_visual_quality_ready_rate']}\n"
        f"- formal_motion_consistency_ready_rate: {audit['formal_motion_consistency_ready_rate']}\n"
        f"- negative_static_contaminated_prompt_count: {audit['negative_static_contaminated_prompt_count']}\n"
        f"- negative_static_contamination_count: {audit['negative_static_contamination_count']}\n"
        f"- no_final_detection_score_used_for_filtering: {audit['no_final_detection_score_used_for_filtering']}\n"
        f"- missing_reasons: {', '.join(audit['motion_threshold_calibration_missing_reasons']) if audit['motion_threshold_calibration_missing_reasons'] else 'none'}\n"
        f"- recommended_action: {audit['motion_threshold_calibration_recommended_action']}\n"
        f"- claim_support_status: {audit['claim_support_status']}\n"
    )
    report_path = run_root / "reports" / "motion_threshold_calibration_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return serializable_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="执行 formal motion threshold calibration。")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--target-static-fpr", type=float, default=DEFAULT_TARGET_STATIC_FPR)
    parser.add_argument("--min-negative-static-count", type=int, default=DEFAULT_MIN_NEGATIVE_STATIC_COUNT)
    parser.add_argument("--min-positive-motion-count", type=int, default=DEFAULT_MIN_POSITIVE_MOTION_COUNT)
    parser.add_argument("--min-ambiguous-low-motion-count", type=int, default=DEFAULT_MIN_AMBIGUOUS_LOW_MOTION_COUNT)
    parser.add_argument("--min-positive-pass-rate", type=float, default=DEFAULT_MIN_POSITIVE_PASS_RATE)
    parser.add_argument("--min-positive-pass-rate-wilson-lower", type=float, default=DEFAULT_MIN_POSITIVE_PASS_RATE_WILSON_LOWER)
    args = parser.parse_args()
    payload = run_motion_threshold_calibration(
        args.run_root,
        target_static_fpr=args.target_static_fpr,
        min_negative_static_count=args.min_negative_static_count,
        min_positive_motion_count=args.min_positive_motion_count,
        min_ambiguous_low_motion_count=args.min_ambiguous_low_motion_count,
        min_positive_pass_rate=args.min_positive_pass_rate,
        min_positive_pass_rate_wilson_lower=args.min_positive_pass_rate_wilson_lower,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
