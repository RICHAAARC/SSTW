"""SSTW 与现代 external baseline 的差值置信区间报告。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from main.protocol.flow_evidence_fields import with_flow_evidence_protocol_defaults
from main.protocol.record_writer import write_json, write_jsonl
from main.protocol.table_builder import write_csv


DEFAULT_PROTOCOL_CONFIG = "configs/protocol/validation_scale_generative_probe.json"
SSTW_METHOD_ID = "sstw_key_conditioned_flow_trajectory"
DEFAULT_REQUIRED_BASELINES = ("videoshield", "sigmark", "videomark", "vidsig", "videoseal")


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
    config = _read_json(config_path)
    if "target_fpr" not in config:
        raise KeyError(f"protocol config 缺少 target_fpr: {config_path}")
    return {
        "paper_result_level": str(config.get("paper_result_level") or "validation_scale"),
        "target_fpr": float(config["target_fpr"]),
        "target_fpr_source_config_path": str(config_path),
        "required_modern_external_baseline_adapter_names": [
            str(item)
            for item in config.get("required_modern_external_baseline_adapter_names", DEFAULT_REQUIRED_BASELINES)
            if str(item)
        ],
        "allow_effect_size_claims": bool(config.get("allow_effect_size_claims", False)),
    }


def _mean(values: list[float]) -> float | None:
    """计算均值, 空列表返回 None。"""
    return round(mean(values), 6) if values else None


def _sample_variance(values: list[float]) -> float:
    """计算样本方差, 样本不足时返回 0。"""
    if len(values) < 2:
        return 0.0
    value_mean = mean(values)
    return sum((value - value_mean) ** 2 for value in values) / (len(values) - 1)


def _normal_difference_interval(reference_scores: list[float], baseline_scores: list[float]) -> tuple[float | None, float | None, str]:
    """计算两个均值差的轻量 95% 正态近似区间。"""
    if not reference_scores or not baseline_scores:
        return None, None, "missing_scores"
    delta = float(mean(reference_scores) - mean(baseline_scores))
    variance = _sample_variance(reference_scores) / len(reference_scores) + _sample_variance(baseline_scores) / len(baseline_scores)
    if variance <= 0:
        return round(delta, 6), round(delta, 6), "degenerate_interval_singleton_or_zero_variance"
    half_width = 1.96 * math.sqrt(variance)
    return round(delta - half_width, 6), round(delta + half_width, 6), "normal_approx_difference_of_means"


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


def build_formal_baseline_difference_interval_records(
    run_root: str | Path,
    config_path: str | Path = DEFAULT_PROTOCOL_CONFIG,
) -> list[dict[str, Any]]:
    """构建 SSTW 相对 5 个现代 baseline 的分数差值 CI records。"""
    run_root = Path(run_root)
    profile_context = _load_profile_context(config_path)
    sstw_records = [
        record
        for record in _read_jsonl(run_root / "records" / "sstw_measured_formal_records.jsonl")
        if record.get("metric_status") == "measured_formal"
    ]
    sstw_scores = _score_values(sstw_records, "sstw_score")
    sstw_units = {_unit_key(record) for record in sstw_records}
    external_records = _read_jsonl(run_root / "records" / "external_baseline_score_records.jsonl")
    records_by_baseline: dict[str, list[dict[str, Any]]] = {}
    for record in external_records:
        if record.get("metric_status") != "measured_formal":
            continue
        baseline_id = str(record.get("external_baseline_name") or "")
        if baseline_id:
            records_by_baseline.setdefault(baseline_id, []).append(record)

    rows: list[dict[str, Any]] = []
    claim_support_status = (
        "formal_baseline_difference_interval_paper_profile_claim_candidate"
        if profile_context["allow_effect_size_claims"]
        else "formal_baseline_difference_interval_validation_scale_only"
    )
    for baseline_id in profile_context["required_modern_external_baseline_adapter_names"]:
        baseline_records = records_by_baseline.get(baseline_id, [])
        baseline_scores = _score_values(baseline_records, "external_baseline_score")
        baseline_units = {_unit_key(record) for record in baseline_records}
        paired_count = len({unit for unit in sstw_units & baseline_units if all(unit)})
        ci_lower, ci_upper, interval_method = _normal_difference_interval(sstw_scores, baseline_scores)
        delta = None
        if sstw_scores and baseline_scores:
            delta = round(float(mean(sstw_scores) - mean(baseline_scores)), 6)
        interval_status = "ready" if ci_lower is not None and ci_upper is not None else "missing_scores"
        rows.append(with_flow_evidence_protocol_defaults({
            "record_version": "formal_baseline_difference_interval_v1",
            "reference_method_id": SSTW_METHOD_ID,
            "baseline_method_id": baseline_id,
            "difference_metric_name": "score_mean_difference",
            "metric_status": "measured_formal" if interval_status == "ready" else "missing",
            "comparison_scope": "paper_protocol_formal_adapter",
            "reference_score_field": "sstw_score",
            "baseline_score_field": "external_baseline_score",
            "reference_record_count": len(sstw_scores),
            "baseline_record_count": len(baseline_scores),
            "paired_comparison_unit_count": paired_count,
            "reference_score_mean": _mean(sstw_scores),
            "baseline_score_mean": _mean(baseline_scores),
            "score_mean_difference": delta,
            "difference_ci_confidence_level": 0.95,
            "difference_ci_lower": ci_lower,
            "difference_ci_upper": ci_upper,
            "difference_interval_method": interval_method,
            "difference_interval_status": interval_status,
            "significance_claim_status": "validation_scale_interval_not_significance_claim"
            if not profile_context["allow_effect_size_claims"]
            else "paper_profile_interval_ready_requires_claim_audit",
            "claim_support_status": claim_support_status if interval_status == "ready" else "formal_baseline_difference_interval_blocked",
            **{key: value for key, value in profile_context.items() if key != "required_modern_external_baseline_adapter_names"},
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
    return {
        "stage_id": "formal_baseline_difference_interval",
        "formal_baseline_difference_interval_decision": decision,
        "claim_support_status": "formal_baseline_difference_interval_validation_scale_only" if decision == "PASS" else "formal_baseline_difference_interval_blocked",
        "paper_result_level": records[0].get("paper_result_level") if records else None,
        "target_fpr": records[0].get("target_fpr") if records else None,
        "difference_interval_record_count": len(records),
        "difference_interval_ready_count": len(ready_records),
        "difference_interval_missing_baseline_ids": missing_baseline_ids,
        "difference_interval_missing_baseline_count": len(missing_baseline_ids),
        "significance_claim_status": "validation_scale_interval_not_significance_claim",
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
        "该报告计算 SSTW 相对 5 个现代 external baseline 的分数均值差及 95% 置信区间。"
        "validation_scale 样本量只用于验证统计产物闭环, 不作为显著性或最终效果主张。\n\n"
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
