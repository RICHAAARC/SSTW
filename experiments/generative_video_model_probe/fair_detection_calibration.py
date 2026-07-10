"""基于方法自身 clean negative 分布的公平检测校准。

该模块实现 probe_paper 必须闭合的公平比较机制: 每个方法先在自己的
clean negative 分布上校准到同一个 target FPR, 再在同一 prompt / seed / attack
锚点下比较 attacked positive 的 TPR。Notebook 只调用本模块命令, 不在 cell 中
手写阈值、TPR 或论文表格。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from experiments.generative_video_model_probe.external_baseline_runner import (
    formal_clean_negative_score_record_ready_for_calibration,
    formal_score_record_ready_for_claim,
)
from experiments.generative_video_model_probe.sstw_formal_result import (
    formal_sstw_clean_negative_record_ready_for_calibration,
    formal_sstw_score_record_ready_for_claim,
)
from runtime.core.digest import build_stable_digest
from evaluation.attacks.video_runtime_attack_protocol import (
    load_protocol_config_with_shared_attack_protocol,
    required_runtime_attack_names_from_config,
)
from evaluation.protocol.flow_evidence_fields import with_flow_evidence_protocol_defaults
from evaluation.protocol.record_writer import write_json, write_jsonl
from evaluation.protocol.table_builder import write_csv


DEFAULT_PROTOCOL_CONFIG = "configs/protocol/probe_paper_generative_probe.json"
SSTW_METHOD_ID = "sstw_key_conditioned_flow_trajectory"
DEFAULT_REQUIRED_BASELINES = ("videoshield", "vidsig", "videoseal", "revmark", "wam_frame")
SPLIT_FIELDS = ("split", "protocol_split", "data_split", "sample_split")
CALIBRATION_SPLITS = {"calibration", "calibration_negative", "calibration_split"}
HELDOUT_TEST_SPLITS = {"test", "heldout", "heldout_test", "test_split"}
SPLIT_THRESHOLD_PROTOCOL = "calibration_split_to_frozen_threshold_to_heldout_test_split"


def _read_json(path: Path) -> dict[str, Any]:
    """读取 JSON 对象, 文件不存在时返回空对象。"""

    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON 顶层必须是对象: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL records, 文件不存在时返回空列表。"""

    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _safe_float(value: object) -> float | None:
    """将 record 中的分数字段安全转换为 float。"""

    if value is None or value == "" or value == "unsupported":
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _load_profile_context(config_path: str | Path) -> dict[str, Any]:
    """读取当前 workflow profile 的公平比较协议。"""

    config_path = Path(config_path)
    config = load_protocol_config_with_shared_attack_protocol(config_path)
    required_runtime_attack_names = (
        list(required_runtime_attack_names_from_config(config))
        if "required_runtime_attack_names" in config or "shared_attack_protocol_config_path" in config
        else []
    )
    if "target_fpr" not in config:
        raise KeyError(f"protocol config 缺少 target_fpr: {config_path}")
    return {
        "paper_result_level": str(config.get("paper_result_level") or "probe_paper"),
        "target_fpr": float(config["target_fpr"]),
        "target_fpr_source_config_path": str(config_path),
        "minimum_clean_negative_count": int(config.get("minimum_clean_negative_count") or 0),
        "minimum_calibration_negative_event_count": int(config.get("minimum_calibration_negative_event_count") or 0),
        "minimum_heldout_test_negative_event_count": int(config.get("minimum_heldout_test_negative_event_count") or 0),
        "minimum_heldout_attacked_positive_event_count": int(config.get("minimum_heldout_attacked_positive_event_count") or 0),
        "threshold_protocol": str(config.get("threshold_protocol") or ""),
        "threshold_source_split": str(config.get("threshold_source_split") or "calibration"),
        "required_modern_external_baseline_adapter_names": [
            str(item)
            for item in config.get("required_modern_external_baseline_adapter_names", DEFAULT_REQUIRED_BASELINES)
            if str(item)
        ],
        "required_runtime_attack_names": required_runtime_attack_names,
        "fair_comparison_protocol": str(
            config.get("fair_comparison_protocol")
            or "method_specific_clean_negative_calibration_to_target_fpr"
        ),
    }


def _role_values(record: Mapping[str, Any]) -> set[str]:
    """汇总一个 record 中可表达 positive / negative 角色的字段。"""

    fields = (
        "sample_role",
        "negative_family",
        "calibration_sample_role",
        "comparison_sample_role",
        "split_role",
        "motion_calibration_role",
    )
    return {str(record.get(field) or "").strip().lower() for field in fields if record.get(field) not in {None, ""}}


def _is_clean_negative_record(record: Mapping[str, Any]) -> bool:
    """判断 record 是否代表 clean negative 样本。"""

    values = _role_values(record)
    return any("negative" in value for value in values) or any(value in {"clean", "clean_negative"} for value in values)


def _is_attacked_positive_record(record: Mapping[str, Any]) -> bool:
    """判断 record 是否代表 attacked positive 样本。"""

    if _is_clean_negative_record(record):
        return False
    attack_name = str(record.get("attack_name") or "").strip()
    return bool(attack_name)


def _canonical_split(value: object) -> str:
    """把不同 split 写法归一到 calibration / heldout_test 等论文协议名称。"""

    normalized = str(value or "").strip().lower()
    if normalized in CALIBRATION_SPLITS:
        return "calibration"
    if normalized in HELDOUT_TEST_SPLITS:
        return "heldout_test"
    return normalized


def _prompt_seed_split_map(generation_records: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str], str]:
    """从 generation records 建立 prompt / seed 到 split 的映射。"""

    mapping: dict[tuple[str, str], str] = {}
    for record in generation_records:
        prompt_id = str(record.get("prompt_id") or "")
        seed_id = str(record.get("seed_id") or "")
        if not prompt_id or not seed_id:
            continue
        split = _record_split(record, {})
        if split:
            mapping[(prompt_id, seed_id)] = split
    return mapping


def _record_split(record: Mapping[str, Any], split_lookup: Mapping[tuple[str, str], str]) -> str:
    """读取 record split, 缺失时按 prompt / seed 从 generation split 映射补齐。"""

    for field_name in SPLIT_FIELDS:
        split = _canonical_split(record.get(field_name))
        if split:
            return split
    prompt_id = str(record.get("prompt_id") or "")
    seed_id = str(record.get("seed_id") or "")
    return split_lookup.get((prompt_id, seed_id), "")


def _first_score(record: Mapping[str, Any], field_names: tuple[str, ...]) -> tuple[float | None, str]:
    """按候选字段顺序提取第一个可用分数。"""

    for field_name in field_names:
        value = _safe_float(record.get(field_name))
        if value is not None:
            return value, field_name
    return None, "missing_score"


def _deduplicated_score_rows(
    records: Iterable[Mapping[str, Any]],
    *,
    score_fields: tuple[str, ...],
    negative_embedded_fields: tuple[str, ...] = (),
    negative_record_ready_predicate: Callable[[Mapping[str, Any]], bool] | None = None,
    embedded_negative_record_ready_predicate: Callable[[Mapping[str, Any]], bool] | None = None,
    split_lookup: Mapping[tuple[str, str], str] | None = None,
) -> list[dict[str, Any]]:
    """提取 clean negative 分数, 并按视频路径或 prompt / seed 去重。

    部分 baseline 会把同一个 clean negative 分数随每个 attack positive record
    重复写出。此处用 clean video path、prompt、seed 和字段名去重, 避免阈值校准
    被重复攻击数放大。
    """

    rows: list[dict[str, Any]] = []
    split_lookup = split_lookup or {}
    seen: set[tuple[str, str, str, str]] = set()
    for record in records:
        is_clean_negative = _is_clean_negative_record(record)
        candidate_fields = score_fields if is_clean_negative else negative_embedded_fields
        if not candidate_fields:
            continue
        score, field_name = _first_score(record, candidate_fields)
        if score is None:
            continue
        if is_clean_negative and negative_record_ready_predicate is not None:
            if not negative_record_ready_predicate(record):
                continue
        if not is_clean_negative and embedded_negative_record_ready_predicate is not None:
            if not embedded_negative_record_ready_predicate(record):
                continue
        path = str(
            record.get("clean_negative_video_path")
            or record.get("external_baseline_clean_negative_video_path")
            or record.get("source_video_path")
            or ""
        )
        negative_unit_id = str(
            record.get("clean_negative_unit_id")
            or record.get("control_name")
            or record.get("negative_family")
            or ""
        )
        key = (
            path,
            str(record.get("prompt_id") or ""),
            str(record.get("seed_id") or ""),
            field_name,
            negative_unit_id,
        )
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "score": float(score),
            "score_field": field_name,
            "prompt_id": record.get("prompt_id"),
            "seed_id": record.get("seed_id"),
            "video_path": path,
            "split": _record_split(record, split_lookup),
        })
    return rows


def _positive_score_rows(
    records: Iterable[Mapping[str, Any]],
    *,
    score_fields: tuple[str, ...],
    positive_record_ready_predicate: Callable[[Mapping[str, Any]], bool] | None = None,
    split_lookup: Mapping[tuple[str, str], str] | None = None,
) -> list[dict[str, Any]]:
    """提取具备完整 prompt / seed / attack anchor 的 attacked positive 分数。

    公平比较的核心约束是每个方法都在同一 prompt / seed / attack 单元上比较
    attacked positive TPR。缺少 prompt_id、seed_id 或 attack_name 的记录不能被
    隐式拼成 `None` anchor, 否则不同方法可能在非同源样本上被误判为已对齐。
    """

    rows: list[dict[str, Any]] = []
    split_lookup = split_lookup or {}
    for record in records:
        if record.get("metric_status") != "measured_formal" or not _is_attacked_positive_record(record):
            continue
        if positive_record_ready_predicate is not None and not positive_record_ready_predicate(record):
            continue
        score, field_name = _first_score(record, score_fields)
        if score is None:
            continue
        prompt_id = record.get("prompt_id")
        seed_id = record.get("seed_id")
        attack_name = record.get("attack_name")
        if prompt_id in {None, ""} or seed_id in {None, ""} or attack_name in {None, ""}:
            continue
        comparison_anchor_key = f"{prompt_id}::{seed_id}::{attack_name}"
        rows.append({
            "score": float(score),
            "score_field": field_name,
            "prompt_id": prompt_id,
            "seed_id": seed_id,
            "attack_name": attack_name,
            "split": _record_split(record, split_lookup),
            "comparison_anchor_key": comparison_anchor_key,
            "comparison_unit_id": record.get("runtime_comparison_unit_id")
            or record.get("external_baseline_score_record_id")
            or record.get("sstw_measured_formal_record_id"),
        })
    return rows


def _positive_score_missing_anchor_count(
    records: Iterable[Mapping[str, Any]],
    *,
    score_fields: tuple[str, ...],
) -> int:
    """统计已测量 attacked positive 记录中缺少完整 anchor 的数量。

    该计数用于 fail closed: 只要存在带分数但缺少 prompt_id / seed_id /
    attack_name 的正式 positive 记录, 当前方法的公平校准就不能进入 ready 状态。
    """

    missing_count = 0
    for record in records:
        if record.get("metric_status") != "measured_formal" or not _is_attacked_positive_record(record):
            continue
        score, _field_name = _first_score(record, score_fields)
        if score is None:
            continue
        if record.get("prompt_id") in {None, ""} or record.get("seed_id") in {None, ""} or record.get("attack_name") in {None, ""}:
            missing_count += 1
    return missing_count


def _positive_formal_evidence_missing_count(
    records: Iterable[Mapping[str, Any]],
    *,
    score_fields: tuple[str, ...],
    positive_record_ready_predicate: Callable[[Mapping[str, Any]], bool] | None,
) -> int:
    """统计带分数但未满足 formal evidence 要求的 attacked positive 记录数量。"""

    if positive_record_ready_predicate is None:
        return 0
    missing_count = 0
    for record in records:
        if record.get("metric_status") != "measured_formal" or not _is_attacked_positive_record(record):
            continue
        score, _field_name = _first_score(record, score_fields)
        if score is None:
            continue
        if not positive_record_ready_predicate(record):
            missing_count += 1
    return missing_count


def _negative_formal_evidence_missing_count(
    records: Iterable[Mapping[str, Any]],
    *,
    score_fields: tuple[str, ...],
    embedded_negative_score_fields: tuple[str, ...],
    negative_record_ready_predicate: Callable[[Mapping[str, Any]], bool] | None,
    embedded_negative_record_ready_predicate: Callable[[Mapping[str, Any]], bool] | None,
) -> int:
    """统计带 clean negative 分数但未满足 official evidence 要求的记录数量。"""

    missing_count = 0
    for record in records:
        is_clean_negative = _is_clean_negative_record(record)
        candidate_fields = score_fields if is_clean_negative else embedded_negative_score_fields
        if not candidate_fields:
            continue
        score, _field_name = _first_score(record, candidate_fields)
        if score is None:
            continue
        if is_clean_negative and negative_record_ready_predicate is not None:
            if not negative_record_ready_predicate(record):
                missing_count += 1
        elif not is_clean_negative and embedded_negative_record_ready_predicate is not None:
            if not embedded_negative_record_ready_predicate(record):
                missing_count += 1
    return missing_count


def _fpr_at_threshold(scores: list[float], threshold: float) -> float:
    """计算 higher-is-more-watermarked 方向下的 FPR。"""

    if not scores:
        return 0.0
    return sum(1 for score in scores if score >= threshold) / len(scores)


def _negative_cluster_scores(rows: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    """按 clean source video 聚合多个 key trial, 每簇采用最大分数。"""

    grouped: dict[str, float] = {}
    for row in rows:
        cluster_id = f"{row.get('prompt_id')}::{row.get('seed_id')}"
        score = float(row["score"])
        grouped[cluster_id] = max(grouped.get(cluster_id, float("-inf")), score)
    return grouped


def _threshold_for_target_fpr(scores: list[float], target_fpr: float) -> tuple[float | None, float | None, str]:
    """从 clean negative 分布选择满足 target FPR 的阈值。"""

    if not scores:
        return None, None, "missing_clean_negative_scores"
    unique_scores = sorted({float(score) for score in scores})
    epsilon = max(1e-12, (max(unique_scores) - min(unique_scores, default=0.0)) * 1e-9)
    candidates = [unique_scores[-1] + epsilon, *unique_scores]
    candidates = sorted({float(value) for value in candidates})
    selected_threshold: float | None = None
    selected_fpr: float | None = None
    for threshold in candidates:
        fpr = _fpr_at_threshold(scores, threshold)
        if fpr <= target_fpr:
            selected_threshold = threshold
            selected_fpr = fpr
            break
    if selected_threshold is None:
        selected_threshold = unique_scores[-1] + epsilon
        selected_fpr = 0.0
    return round(float(selected_threshold), 12), round(float(selected_fpr), 6), "empirical_clean_negative_quantile"


def _wilson_interval(success_count: int, total_count: int, z: float = 1.96) -> tuple[float | None, float | None]:
    """计算二项比例 Wilson 95% 置信区间。"""

    if total_count <= 0:
        return None, None
    phat = success_count / total_count
    denom = 1.0 + z * z / total_count
    centre = phat + z * z / (2 * total_count)
    radius = z * math.sqrt((phat * (1.0 - phat) + z * z / (4 * total_count)) / total_count)
    return round((centre - radius) / denom, 6), round((centre + radius) / denom, 6)


def _calibrated_method_record(
    *,
    method_id: str,
    method_role: str,
    records: list[dict[str, Any]],
    positive_score_fields: tuple[str, ...],
    negative_score_fields: tuple[str, ...],
    embedded_negative_score_fields: tuple[str, ...],
    score_semantics_field: str,
    default_score_semantics: str,
    context: dict[str, Any],
    positive_record_ready_predicate: Callable[[Mapping[str, Any]], bool] | None = None,
    negative_record_ready_predicate: Callable[[Mapping[str, Any]], bool] | None = None,
    embedded_negative_record_ready_predicate: Callable[[Mapping[str, Any]], bool] | None = None,
    split_lookup: Mapping[tuple[str, str], str] | None = None,
) -> dict[str, Any]:
    """构建单个方法的公平校准记录。"""

    split_lookup = split_lookup or {}
    positive_rows = _positive_score_rows(
        records,
        score_fields=positive_score_fields,
        positive_record_ready_predicate=positive_record_ready_predicate,
        split_lookup=split_lookup,
    )
    positive_missing_anchor_count = _positive_score_missing_anchor_count(records, score_fields=positive_score_fields)
    positive_formal_evidence_missing_count = _positive_formal_evidence_missing_count(
        records,
        score_fields=positive_score_fields,
        positive_record_ready_predicate=positive_record_ready_predicate,
    )
    negative_formal_evidence_missing_count = _negative_formal_evidence_missing_count(
        records,
        score_fields=negative_score_fields,
        embedded_negative_score_fields=embedded_negative_score_fields,
        negative_record_ready_predicate=negative_record_ready_predicate,
        embedded_negative_record_ready_predicate=embedded_negative_record_ready_predicate,
    )
    negative_rows = _deduplicated_score_rows(
        records,
        score_fields=negative_score_fields,
        negative_embedded_fields=embedded_negative_score_fields,
        negative_record_ready_predicate=negative_record_ready_predicate,
        embedded_negative_record_ready_predicate=embedded_negative_record_ready_predicate,
        split_lookup=split_lookup,
    )
    split_required = context.get("threshold_protocol") == SPLIT_THRESHOLD_PROTOCOL
    calibration_negative_rows = [row for row in negative_rows if row.get("split") == "calibration"]
    heldout_negative_rows = [row for row in negative_rows if row.get("split") == "heldout_test"]
    heldout_positive_rows = [row for row in positive_rows if row.get("split") == "heldout_test"]
    threshold_negative_rows = calibration_negative_rows if split_required else negative_rows
    fpr_negative_rows = heldout_negative_rows if split_required else negative_rows
    eval_positive_rows = heldout_positive_rows if split_required else positive_rows
    calibration_negative_cluster_scores = _negative_cluster_scores(threshold_negative_rows)
    heldout_negative_cluster_scores = _negative_cluster_scores(fpr_negative_rows)
    negative_scores = list(calibration_negative_cluster_scores.values())
    heldout_negative_scores = list(heldout_negative_cluster_scores.values())
    positive_scores = [row["score"] for row in eval_positive_rows]
    positive_attack_names = sorted({str(row["attack_name"]) for row in eval_positive_rows if row.get("attack_name")})
    required_attack_names = [str(item) for item in context.get("required_runtime_attack_names", []) if str(item)]
    missing_required_attack_names = sorted(set(required_attack_names) - set(positive_attack_names))
    threshold, calibration_fpr, threshold_policy = _threshold_for_target_fpr(negative_scores, float(context["target_fpr"]))
    heldout_fpr = _fpr_at_threshold(heldout_negative_scores, threshold) if threshold is not None and heldout_negative_scores else None
    if heldout_fpr is not None:
        heldout_fpr = round(float(heldout_fpr), 6)
    detected_count = 0
    positive_detection_units: list[dict[str, Any]] = []
    negative_detection_units: list[dict[str, Any]] = []
    if threshold is not None:
        detected_count = sum(1 for score in positive_scores if score >= threshold)
        positive_detection_units = [
            {
                "comparison_anchor_key": row["comparison_anchor_key"],
                "prompt_id": row.get("prompt_id"),
                "seed_id": row.get("seed_id"),
                "attack_name": row.get("attack_name"),
                "split": row.get("split"),
                "score": row["score"],
                "detected_at_target_fpr": row["score"] >= threshold,
            }
            for row in eval_positive_rows
        ]
        negative_detection_units = [
            {
                "statistical_cluster_id": cluster_id,
                "cluster_maximum_score": score,
                "false_positive_at_target_fpr": score >= threshold,
            }
            for cluster_id, score in sorted(heldout_negative_cluster_scores.items())
        ]
    tpr = round(detected_count / len(positive_scores), 6) if positive_scores else None
    ci_lower, ci_upper = _wilson_interval(detected_count, len(positive_scores))
    semantics = next(
        (
            str(record.get(score_semantics_field))
            for record in records
            if record.get(score_semantics_field) not in {None, ""}
        ),
        default_score_semantics,
    )
    missing_reasons: list[str] = []
    all_negative_cluster_count = len(_negative_cluster_scores(negative_rows))
    if all_negative_cluster_count < int(context["minimum_clean_negative_count"]):
        missing_reasons.append("clean_negative_score_count_below_minimum")
    if split_required:
        if len(calibration_negative_cluster_scores) < int(context.get("minimum_calibration_negative_event_count") or 0):
            missing_reasons.append("calibration_clean_negative_score_count_below_minimum")
        if len(heldout_negative_cluster_scores) < int(context.get("minimum_heldout_test_negative_event_count") or 0):
            missing_reasons.append("heldout_clean_negative_score_count_below_minimum")
        if len(heldout_positive_rows) < int(context.get("minimum_heldout_attacked_positive_event_count") or 0):
            missing_reasons.append("heldout_attacked_positive_score_count_below_minimum")
    if not positive_rows:
        missing_reasons.append("attacked_positive_scores_missing")
    if split_required and not eval_positive_rows:
        missing_reasons.append("heldout_attacked_positive_scores_missing")
    if positive_missing_anchor_count:
        missing_reasons.append("positive_anchor_fields_missing")
    if positive_formal_evidence_missing_count:
        missing_reasons.append("positive_formal_evidence_missing")
    if negative_formal_evidence_missing_count:
        missing_reasons.append("clean_negative_formal_evidence_missing")
    if missing_required_attack_names:
        missing_reasons.append("required_runtime_attack_coverage_missing")
    if any(str(record.get("external_baseline_score_orientation") or "higher_is_more_watermarked") != "higher_is_more_watermarked" for record in records):
        missing_reasons.append("unsupported_score_orientation")
    calibration_status = "ready" if not missing_reasons and threshold is not None and tpr is not None else "blocked"
    payload = {
        "method_id": method_id,
        "method_role": method_role,
        "metric_status": "measured_formal" if calibration_status == "ready" else "missing",
        "fair_comparison_status": calibration_status,
        "fair_comparison_missing_reasons": missing_reasons,
        "fair_comparison_protocol": context["fair_comparison_protocol"],
        "score_semantics": semantics,
        "score_orientation": "higher_is_more_watermarked",
        "positive_score_field": positive_rows[0]["score_field"] if positive_rows else positive_score_fields[0],
        "clean_negative_score_field": negative_rows[0]["score_field"] if negative_rows else (embedded_negative_score_fields or negative_score_fields or ("missing",))[0],
        "clean_negative_score_count": all_negative_cluster_count,
        "clean_negative_raw_trial_count": len(negative_rows),
        "calibration_clean_negative_score_count": len(calibration_negative_cluster_scores),
        "calibration_clean_negative_raw_trial_count": len(calibration_negative_rows),
        "heldout_clean_negative_score_count": len(heldout_negative_cluster_scores),
        "heldout_clean_negative_raw_trial_count": len(heldout_negative_rows),
        "attacked_positive_score_count": len(positive_rows),
        "heldout_attacked_positive_score_count": len(heldout_positive_rows),
        "threshold_protocol": context.get("threshold_protocol"),
        "threshold_source_split": "calibration" if split_required else "all_clean_negative",
        "calibration_fpr_at_calibrated_threshold": calibration_fpr,
        "positive_anchor_missing_count": positive_missing_anchor_count,
        "positive_formal_evidence_missing_count": positive_formal_evidence_missing_count,
        "negative_formal_evidence_missing_count": negative_formal_evidence_missing_count,
        "positive_anchor_count": len({str(row["comparison_anchor_key"]) for row in positive_rows}),
        "positive_anchor_keys": sorted({str(row["comparison_anchor_key"]) for row in positive_rows}),
        "positive_attack_names": positive_attack_names,
        "required_runtime_attack_names": required_attack_names,
        "missing_required_runtime_attack_names": missing_required_attack_names,
        "missing_required_runtime_attack_count": len(missing_required_attack_names),
        "positive_detection_units_at_target_fpr": sorted(
            positive_detection_units,
            key=lambda item: str(item["comparison_anchor_key"]),
        ),
        "negative_detection_units_at_target_fpr": negative_detection_units,
        "statistical_independent_unit": "source_video_prompt_seed",
        "calibrated_threshold": threshold,
        "threshold_selection_policy": threshold_policy,
        "heldout_fpr_at_calibrated_threshold": heldout_fpr,
        "detected_positive_count_at_target_fpr": detected_count,
        "tpr_at_target_fpr": tpr,
        "tpr_ci_confidence_level": 0.95,
        "tpr_ci_lower": ci_lower,
        "tpr_ci_upper": ci_upper,
        "prompt_count": len({str(row["prompt_id"]) for row in positive_rows if row.get("prompt_id")}),
        "attack_count": len({str(row["attack_name"]) for row in positive_rows if row.get("attack_name")}),
        "claim_support_status": "fair_detection_calibration_paper_profile_ready"
        if calibration_status == "ready"
        else "fair_detection_calibration_blocked",
        **{
            key: value
            for key, value in context.items()
            if key not in {"required_modern_external_baseline_adapter_names", "required_runtime_attack_names"}
        },
    }
    digest = build_stable_digest(payload)
    return with_flow_evidence_protocol_defaults({
        "record_version": "fair_detection_calibration_v1",
        "fair_detection_calibration_record_id": f"fair_detection_calibration_{digest[:16]}",
        **payload,
    }, trajectory_source_level="fair_detection_calibration_from_measured_formal_records", claim_support_status=payload["claim_support_status"])


def build_fair_detection_calibration_records(
    run_root: str | Path,
    config_path: str | Path = DEFAULT_PROTOCOL_CONFIG,
) -> list[dict[str, Any]]:
    """构建 SSTW 和现代 baseline 的公平校准 records。"""

    run_root = Path(run_root)
    context = _load_profile_context(config_path)
    split_lookup = _prompt_seed_split_map(_read_jsonl(run_root / "records" / "generation_records.jsonl"))
    rows: list[dict[str, Any]] = []
    sstw_records = _read_jsonl(run_root / "records" / "sstw_measured_formal_records.jsonl")
    rows.append(_calibrated_method_record(
        method_id=SSTW_METHOD_ID,
        method_role="proposed_method",
        records=sstw_records,
        positive_score_fields=("sstw_raw_detector_score", "sstw_score", "raw_detector_score"),
        negative_score_fields=("sstw_clean_negative_score", "clean_negative_score", "sstw_raw_detector_score", "sstw_score", "raw_detector_score"),
        embedded_negative_score_fields=("sstw_clean_negative_score", "clean_negative_score"),
        score_semantics_field="sstw_score_semantics",
        default_score_semantics="calibrated_probability_posterior_with_fixed_fpr_threshold",
        context=context,
        positive_record_ready_predicate=formal_sstw_score_record_ready_for_claim,
        negative_record_ready_predicate=formal_sstw_clean_negative_record_ready_for_calibration,
        embedded_negative_record_ready_predicate=formal_sstw_score_record_ready_for_claim,
        split_lookup=split_lookup,
    ))
    external_records = _read_jsonl(run_root / "records" / "external_baseline_score_records.jsonl")
    records_by_baseline: dict[str, list[dict[str, Any]]] = {}
    for record in external_records:
        if record.get("metric_status") != "measured_formal":
            continue
        baseline_id = str(record.get("external_baseline_name") or "")
        if baseline_id:
            records_by_baseline.setdefault(baseline_id, []).append(record)
    for baseline_id in context["required_modern_external_baseline_adapter_names"]:
        rows.append(_calibrated_method_record(
            method_id=baseline_id,
            method_role="modern_external_baseline",
            records=records_by_baseline.get(baseline_id, []),
            positive_score_fields=("external_baseline_raw_detector_score", "external_baseline_score", "raw_detector_score"),
            negative_score_fields=("external_baseline_clean_negative_score", "clean_negative_score", "external_baseline_raw_detector_score", "external_baseline_score"),
            embedded_negative_score_fields=("external_baseline_clean_negative_score", "clean_negative_score"),
            score_semantics_field="external_baseline_score_semantics",
            default_score_semantics="external_baseline_detector_score",
            context=context,
            positive_record_ready_predicate=formal_score_record_ready_for_claim,
            negative_record_ready_predicate=formal_clean_negative_score_record_ready_for_calibration,
            embedded_negative_record_ready_predicate=formal_score_record_ready_for_claim,
            split_lookup=split_lookup,
        ))
    return rows


def audit_fair_detection_calibration_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """审计公平校准是否覆盖 SSTW 与所有现代 baseline。"""

    ready_records = [record for record in records if record.get("fair_comparison_status") == "ready"]
    missing_method_ids = [str(record.get("method_id")) for record in records if record.get("fair_comparison_status") != "ready"]
    decision = "PASS" if records and len(ready_records) == len(records) else "FAIL"
    return {
        "stage_id": "fair_detection_calibration",
        "fair_detection_calibration_decision": decision,
        "claim_support_status": "fair_detection_calibration_paper_profile_ready" if decision == "PASS" else "fair_detection_calibration_blocked",
        "paper_result_level": records[0].get("paper_result_level") if records else None,
        "target_fpr": records[0].get("target_fpr") if records else None,
        "fair_comparison_protocol": records[0].get("fair_comparison_protocol") if records else None,
        "fair_detection_calibration_method_count": len(records),
        "fair_detection_calibration_ready_count": len(ready_records),
        "fair_detection_calibration_missing_method_ids": missing_method_ids,
        "fair_detection_calibration_missing_method_count": len(missing_method_ids),
    }


def run_fair_detection_calibration(
    run_root: str | Path,
    config_path: str | Path = DEFAULT_PROTOCOL_CONFIG,
) -> dict[str, Any]:
    """写出公平校准 records、table、decision 和 report。"""

    run_root = Path(run_root)
    records = build_fair_detection_calibration_records(run_root, config_path)
    audit = audit_fair_detection_calibration_records(records)
    write_jsonl(run_root / "records" / "fair_detection_calibration_records.jsonl", records)
    write_csv(run_root / "tables" / "fair_detection_calibration_table.csv", records)
    write_json(run_root / "artifacts" / "fair_detection_calibration_decision.json", audit)
    report = (
        "# Fair Detection Calibration Report\n\n"
        "该报告在每个方法自身 clean negative 分布上校准到相同 target FPR, 再统计 attacked positive "
        "TPR。probe_paper 必须通过该门禁后才允许进入 pilot_paper, 但 paper_profile "
        "本身仍不支持最终效果大小主张。\n\n"
        f"- fair_detection_calibration_decision: {audit['fair_detection_calibration_decision']}\n"
        f"- target_fpr: {audit['target_fpr']}\n"
        f"- fair_detection_calibration_ready_count: {audit['fair_detection_calibration_ready_count']}\n"
        f"- fair_detection_calibration_missing_method_ids: {', '.join(audit['fair_detection_calibration_missing_method_ids']) if audit['fair_detection_calibration_missing_method_ids'] else 'none'}\n"
        f"- claim_support_status: {audit['claim_support_status']}\n"
    )
    report_path = run_root / "reports" / "fair_detection_calibration_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="生成基于 clean negative calibration 的公平比较记录。")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--config-path", default=DEFAULT_PROTOCOL_CONFIG)
    args = parser.parse_args()
    payload = run_fair_detection_calibration(args.run_root, args.config_path)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
