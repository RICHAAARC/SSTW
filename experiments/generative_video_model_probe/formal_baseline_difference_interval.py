"""SSTW 与现代 external baseline 的差值置信区间报告。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from evaluation.statistics.clustered_inference import clustered_mean_interval
from evaluation.attacks.video_runtime_attack_protocol import (
    load_protocol_config_with_shared_attack_protocol,
    required_runtime_attack_names_from_config,
)
from evaluation.protocol.flow_evidence_fields import with_flow_evidence_protocol_defaults
from evaluation.protocol.record_writer import write_json, write_jsonl
from evaluation.protocol.table_builder import write_csv


DEFAULT_PROTOCOL_CONFIG = "configs/protocol/probe_paper_generative_probe.json"
SSTW_METHOD_ID = "sstw_key_conditioned_flow_trajectory"
DEFAULT_REQUIRED_BASELINES = ("videoshield", "vidsig", "videoseal", "videomark", "wam_frame")


def _read_json(path: Path) -> dict[str, Any]:
    """读取 JSON 配置或 artifact, 文件不存在时返回空对象。"""
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
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _safe_float(value: object) -> float | None:
    """将 record 中的数值字段安全转换为 float。"""
    if value is None or value == "" or value == "unsupported":
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _load_profile_context(config_path: str | Path) -> dict[str, Any]:
    """读取当前 profile 的差值统计语义。"""
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
        "required_modern_external_baseline_adapter_names": [
            str(item)
            for item in config.get("required_modern_external_baseline_adapter_names", DEFAULT_REQUIRED_BASELINES)
            if str(item)
        ],
        "required_runtime_attack_names": required_runtime_attack_names,
        "allow_effect_size_claims": bool(config.get("allow_effect_size_claims", False)),
    }


def _mean(values: list[float]) -> float | None:
    """计算均值, 空列表返回 None。"""
    return round(mean(values), 6) if values else None


def _paired_detection_difference_interval(
    reference_units: dict[str, bool],
    baseline_units: dict[str, bool],
) -> tuple[float | None, float | None, float | None, int, str]:
    """基于同一 prompt / seed / attack anchor 计算配对检测差值区间。

    这里的单元差值为 `SSTW_detected - baseline_detected`, 取值只能是 -1、0 或 1。
    该实现比两个非配对 TPR 的差值更严格, 因为它只使用双方都存在的同一
    comparison anchor。
    """

    shared_keys = sorted(set(reference_units) & set(baseline_units))
    if not shared_keys:
        return None, None, None, 0, "missing_paired_detection_units"
    differences_by_cluster: dict[str, list[float]] = {}
    for key in shared_keys:
        parts = key.split("::")
        if len(parts) < 2:
            continue
        cluster_id = "::".join(parts[:2])
        differences_by_cluster.setdefault(cluster_id, []).append(
            (1.0 if reference_units[key] else 0.0)
            - (1.0 if baseline_units[key] else 0.0)
        )
    if len(differences_by_cluster) < 2:
        return None, None, None, len(shared_keys), "insufficient_independent_source_video_clusters"
    estimate = clustered_mean_interval(
        differences_by_cluster,
        purpose="sstw_baseline_paired_detection_difference",
    )
    return (
        round(estimate.estimate, 6),
        round(estimate.confidence_interval_lower, 6),
        round(estimate.confidence_interval_upper, 6),
        estimate.observation_count,
        "paired_source_video_cluster_bootstrap_detection_difference",
    )


def _detection_units_by_anchor(record: dict[str, Any]) -> dict[str, bool]:
    """从 fair calibration record 中读取 anchor -> detected 映射。"""

    units: dict[str, bool] = {}
    for unit in record.get("positive_detection_units_at_target_fpr") or []:
        if not isinstance(unit, dict):
            continue
        key = str(unit.get("comparison_anchor_key") or "")
        if not key:
            continue
        units[key] = bool(unit.get("detected_at_target_fpr"))
    return units


def _attack_names_from_unit_keys(unit_keys: Iterable[str]) -> list[str]:
    """从 prompt / seed / attack comparison key 中解析 attack 名称。"""

    names: set[str] = set()
    for key in unit_keys:
        parts = str(key or "").split("::")
        if len(parts) >= 3 and parts[-1]:
            names.add(parts[-1])
    return sorted(names)


def _unit_key(record: dict[str, Any]) -> tuple[str, str, str]:
    """构造 prompt / seed / attack 锚点, 用于审计是否存在可配对比较单元。"""
    return (
        str(record.get("prompt_id") or ""),
        str(record.get("seed_id") or ""),
        str(record.get("attack_name") or ""),
    )


def _score_values(records: Iterable[dict[str, Any]], score_field: str) -> list[float]:
    """提取 measured_formal records 中的分数列表。"""
    values: list[float] = []
    for record in records:
        if record.get("metric_status") != "measured_formal":
            continue
        value = _safe_float(record.get(score_field))
        if value is not None:
            values.append(value)
    return values


def _target_fpr_matches(record: dict[str, Any] | None, expected_target_fpr: float) -> bool:
    """检查上游 fair calibration record 是否来自当前 protocol config。"""

    if not record:
        return False
    try:
        return abs(float(record.get("target_fpr")) - float(expected_target_fpr)) <= 1e-12
    except (TypeError, ValueError):
        return False


def build_formal_baseline_difference_interval_records(
    run_root: str | Path,
    config_path: str | Path = DEFAULT_PROTOCOL_CONFIG,
) -> list[dict[str, Any]]:
    """构建 SSTW 相对 5 个现代 baseline 的分数差值 CI records。"""
    run_root = Path(run_root)
    profile_context = _load_profile_context(config_path)
    fair_records = _read_jsonl(run_root / "records" / "fair_detection_calibration_records.jsonl")
    fair_by_method = {str(record.get("method_id") or ""): record for record in fair_records if record.get("method_id")}
    raw_sstw_record = fair_by_method.get(SSTW_METHOD_ID, {})
    sstw_record = raw_sstw_record if _target_fpr_matches(raw_sstw_record, float(profile_context["target_fpr"])) else {}
    sstw_tpr = _safe_float(sstw_record.get("tpr_at_target_fpr"))
    sstw_count = int(sstw_record.get("attacked_positive_score_count") or 0)
    sstw_units = _detection_units_by_anchor(sstw_record)

    rows: list[dict[str, Any]] = []
    claim_support_status = (
        "formal_baseline_difference_interval_paper_profile_claim_evidence"
        if profile_context["allow_effect_size_claims"]
        else "formal_baseline_difference_interval_paper_profile_only"
    )
    for baseline_id in profile_context["required_modern_external_baseline_adapter_names"]:
        raw_baseline_record = fair_by_method.get(baseline_id, {})
        baseline_record = raw_baseline_record if _target_fpr_matches(raw_baseline_record, float(profile_context["target_fpr"])) else {}
        baseline_tpr = _safe_float(baseline_record.get("tpr_at_target_fpr"))
        baseline_count = int(baseline_record.get("attacked_positive_score_count") or 0)
        baseline_units = _detection_units_by_anchor(baseline_record)
        paired_delta, ci_lower, ci_upper, paired_count, interval_method = _paired_detection_difference_interval(
            sstw_units,
            baseline_units,
        )
        unpaired_reference_count = len(set(sstw_units) - set(baseline_units))
        unpaired_baseline_count = len(set(baseline_units) - set(sstw_units))
        paired_anchor_keys = sorted(set(sstw_units) & set(baseline_units))
        paired_attack_names = _attack_names_from_unit_keys(paired_anchor_keys)
        required_attack_names = [str(item) for item in profile_context.get("required_runtime_attack_names", []) if str(item)]
        missing_required_attack_names = sorted(set(required_attack_names) - set(paired_attack_names))
        anchor_alignment_status = (
            "aligned_with_sstw_reference_anchors"
            if sstw_units and not unpaired_reference_count and not unpaired_baseline_count and not missing_required_attack_names
            else "anchor_set_mismatch_with_sstw"
        )
        delta = paired_delta
        if delta is None and sstw_tpr is not None and baseline_tpr is not None:
            delta = round(float(sstw_tpr) - float(baseline_tpr), 6)
        interval_status = (
            "ready"
            if ci_lower is not None
            and ci_upper is not None
            and anchor_alignment_status == "aligned_with_sstw_reference_anchors"
            else "missing_or_unaligned_paired_anchors"
        )
        rows.append(with_flow_evidence_protocol_defaults({
            "record_version": "formal_baseline_difference_interval_v1",
            "reference_method_id": SSTW_METHOD_ID,
            "baseline_method_id": baseline_id,
            "difference_metric_name": "tpr_at_target_fpr_difference",
            "metric_status": "measured_formal" if interval_status == "ready" else "missing",
            "comparison_scope": "fair_detection_calibration_at_target_fpr",
            "reference_score_field": "tpr_at_target_fpr",
            "baseline_score_field": "tpr_at_target_fpr",
            "reference_source_fair_detection_target_fpr": raw_sstw_record.get("target_fpr") if raw_sstw_record else None,
            "baseline_source_fair_detection_target_fpr": raw_baseline_record.get("target_fpr") if raw_baseline_record else None,
            "reference_record_count": sstw_count,
            "baseline_record_count": baseline_count,
            "paired_comparison_unit_count": paired_count,
            "paired_comparison_anchor_keys": paired_anchor_keys,
            "paired_attack_names": paired_attack_names,
            "required_runtime_attack_names": required_attack_names,
            "missing_required_runtime_attack_names": missing_required_attack_names,
            "missing_required_runtime_attack_count": len(missing_required_attack_names),
            "reference_anchor_count": len(sstw_units),
            "baseline_anchor_count": len(baseline_units),
            "unpaired_reference_anchor_count": unpaired_reference_count,
            "unpaired_baseline_anchor_count": unpaired_baseline_count,
            "comparison_anchor_alignment_status": anchor_alignment_status,
            "reference_score_mean": None,
            "baseline_score_mean": None,
            "reference_tpr_at_target_fpr": sstw_tpr,
            "baseline_tpr_at_target_fpr": baseline_tpr,
            "tpr_at_target_fpr_difference": delta,
            "score_mean_difference": None,
            "difference_ci_confidence_level": 0.95,
            "difference_ci_lower": ci_lower,
            "difference_ci_upper": ci_upper,
            "difference_interval_method": interval_method,
            "difference_interval_status": interval_status,
            "significance_claim_status": "paper_profile_interval_not_significance_claim"
            if not profile_context["allow_effect_size_claims"]
            else "paper_profile_interval_ready_requires_claim_audit",
            "claim_support_status": claim_support_status if interval_status == "ready" else "formal_baseline_difference_interval_blocked",
            **{
                key: value
                for key, value in profile_context.items()
                if key not in {"required_modern_external_baseline_adapter_names", "required_runtime_attack_names"}
            },
        }, trajectory_source_level="formal_baseline_difference_interval", claim_support_status=claim_support_status if interval_status == "ready" else "formal_baseline_difference_interval_blocked"))
    return rows


def audit_formal_baseline_difference_interval_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """审计差值 CI 是否覆盖所有现代 baseline。"""
    ready_records = [record for record in records if record.get("difference_interval_status") == "ready"]
    missing_baseline_ids = [
        str(record.get("baseline_method_id"))
        for record in records
        if record.get("difference_interval_status") != "ready"
    ]
    decision = "PASS" if records and len(ready_records) == len(records) else "FAIL"
    ready_claim_statuses = {str(record.get("claim_support_status")) for record in ready_records}
    if decision == "PASS" and ready_claim_statuses == {"formal_baseline_difference_interval_paper_profile_claim_evidence"}:
        claim_support_status = "formal_baseline_difference_interval_paper_profile_claim_evidence"
        significance_claim_status = "paper_profile_interval_ready_requires_claim_audit"
    elif decision == "PASS":
        claim_support_status = "formal_baseline_difference_interval_paper_profile_only"
        significance_claim_status = "paper_profile_interval_not_significance_claim"
    else:
        claim_support_status = "formal_baseline_difference_interval_blocked"
        significance_claim_status = "formal_baseline_difference_interval_blocked"
    return {
        "stage_id": "formal_baseline_difference_interval",
        "formal_baseline_difference_interval_decision": decision,
        "claim_support_status": claim_support_status,
        "paper_result_level": records[0].get("paper_result_level") if records else None,
        "target_fpr": records[0].get("target_fpr") if records else None,
        "difference_interval_record_count": len(records),
        "difference_interval_ready_count": len(ready_records),
        "difference_interval_missing_baseline_ids": missing_baseline_ids,
        "difference_interval_missing_baseline_count": len(missing_baseline_ids),
        "significance_claim_status": significance_claim_status,
    }


def run_formal_baseline_difference_interval(
    run_root: str | Path,
    config_path: str | Path = DEFAULT_PROTOCOL_CONFIG,
) -> dict[str, Any]:
    """写出差值 CI records、table、decision 和 report。"""
    run_root = Path(run_root)
    records = build_formal_baseline_difference_interval_records(run_root, config_path)
    audit = audit_formal_baseline_difference_interval_records(records)
    write_jsonl(run_root / "records" / "formal_baseline_difference_interval_records.jsonl", records)
    write_csv(run_root / "tables" / "formal_baseline_difference_interval_table.csv", records)
    write_json(run_root / "artifacts" / "formal_baseline_difference_interval_decision.json", audit)
    report = (
        "# Formal Baseline Difference Interval Report\n\n"
        "该报告计算 SSTW 相对 5 个现代 external baseline 的 TPR@target FPR 差值及 95% 置信区间。"
        "当 protocol config 启用 allow_effect_size_claims 时, probe_paper 差值区间用于支撑 target_fpr=0.1 "
        "的FPR=0.1 条件下的完整优势结论, 且仍需由 claim audit 限定其不能外推到更低 FPR 或更大样本结论。\n\n"
        f"- formal_baseline_difference_interval_decision: {audit['formal_baseline_difference_interval_decision']}\n"
        f"- paper_result_level: {audit['paper_result_level']}\n"
        f"- target_fpr: {audit['target_fpr']}\n"
        f"- difference_interval_record_count: {audit['difference_interval_record_count']}\n"
        f"- difference_interval_ready_count: {audit['difference_interval_ready_count']}\n"
        f"- difference_interval_missing_baseline_ids: {', '.join(audit['difference_interval_missing_baseline_ids']) if audit['difference_interval_missing_baseline_ids'] else 'none'}\n"
        f"- significance_claim_status: {audit['significance_claim_status']}\n"
        f"- claim_support_status: {audit['claim_support_status']}\n"
    )
    report_path = run_root / "reports" / "formal_baseline_difference_interval_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 SSTW 与现代 external baseline 的差值置信区间报告。")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--config-path", default=DEFAULT_PROTOCOL_CONFIG)
    args = parser.parse_args()
    payload = run_formal_baseline_difference_interval(args.run_root, args.config_path)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
