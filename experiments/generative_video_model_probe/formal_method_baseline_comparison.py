"""SSTW 与现代 external baseline 的同协议 measured_formal 统计表。"""

from __future__ import annotations

import argparse
import json
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


def _truthy_positive(value: object) -> bool | None:
    """把 detector 输出中的布尔或数值 positive 状态规整为布尔值。"""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value) > 0.0
    text = str(value).strip().lower()
    if text in {"true", "yes", "1", "positive", "detected"}:
        return True
    if text in {"false", "no", "0", "negative", "not_detected"}:
        return False
    return None


def _load_profile_context(config_path: str | Path) -> dict[str, Any]:
    """读取当前 profile 的对比统计语义。"""
    config_path = Path(config_path)
    config = _read_json(config_path)
    if "target_fpr" not in config:
        raise KeyError(f"protocol config 缺少 target_fpr: {config_path}")
    required_baselines = [
        str(item)
        for item in config.get("required_modern_external_baseline_adapter_names", DEFAULT_REQUIRED_BASELINES)
        if str(item)
    ]
    return {
        "paper_result_level": str(config.get("paper_result_level") or "validation_scale"),
        "target_fpr": float(config["target_fpr"]),
        "target_fpr_source_config_path": str(config_path),
        "required_modern_external_baseline_adapter_names": required_baselines,
        "required_runtime_attack_names": [
            str(item)
            for item in config.get("required_runtime_attack_names", [])
            if str(item)
        ],
        "allow_effect_size_claims": bool(config.get("allow_effect_size_claims", False)),
    }


def _mean(values: Iterable[float | None]) -> float | None:
    """计算非空数值均值。"""
    numeric_values = [float(value) for value in values if value is not None]
    return round(mean(numeric_values), 6) if numeric_values else None


def _target_fpr_matches(record: dict[str, Any] | None, expected_target_fpr: float) -> bool:
    """检查上游 fair calibration record 是否来自当前 protocol config。"""

    if not record:
        return False
    try:
        return abs(float(record.get("target_fpr")) - float(expected_target_fpr)) <= 1e-12
    except (TypeError, ValueError):
        return False


def _method_row(
    *,
    method_id: str,
    method_role: str,
    record: dict[str, Any] | None,
    profile_context: dict[str, Any],
    missing_reason: str = "",
    reference_anchor_keys: set[str] | None = None,
) -> dict[str, Any]:
    """将 fair calibration record 转成同协议比较行。"""
    record_target_matches = _target_fpr_matches(record, float(profile_context["target_fpr"]))
    fair_ready = bool(record) and record.get("fair_comparison_status") == "ready" and record_target_matches
    anchor_keys = {str(item) for item in (record or {}).get("positive_anchor_keys", []) if str(item)}
    comparison_attack_names = sorted({str(item) for item in (record or {}).get("positive_attack_names", []) if str(item)})
    if not comparison_attack_names:
        comparison_attack_names = sorted({key.split("::")[-1] for key in anchor_keys if "::" in key})
    required_attack_names = [str(item) for item in profile_context.get("required_runtime_attack_names", []) if str(item)]
    missing_required_attack_names = sorted(set(required_attack_names) - set(comparison_attack_names))
    if record and not record_target_matches:
        missing_reason = "fair_detection_calibration_target_fpr_mismatch"
    if method_role == "proposed_method":
        anchor_alignment_status = "reference_method_anchor_set_ready" if fair_ready and anchor_keys else "reference_method_anchor_set_missing"
        anchor_aligned = fair_ready and bool(anchor_keys)
    else:
        reference_anchor_keys = reference_anchor_keys or set()
        missing_reference_anchor_keys = sorted(reference_anchor_keys - anchor_keys)
        extra_anchor_keys = sorted(anchor_keys - reference_anchor_keys)
        anchor_aligned = fair_ready and bool(reference_anchor_keys) and not missing_reference_anchor_keys and not extra_anchor_keys
        anchor_alignment_status = "aligned_with_sstw_reference_anchors" if anchor_aligned else "anchor_set_mismatch_with_sstw"
    metric_status = "measured_formal" if fair_ready and anchor_aligned and not missing_required_attack_names else "missing"
    if metric_status == "missing" and fair_ready and not anchor_aligned:
        missing_reason = "anchor_set_mismatch_with_sstw"
    if metric_status == "missing" and fair_ready and anchor_aligned and missing_required_attack_names:
        missing_reason = "required_runtime_attack_coverage_missing"
    claim_support_status = (
        "formal_method_baseline_comparison_paper_profile_claim_candidate"
        if profile_context["allow_effect_size_claims"] and metric_status == "measured_formal"
        else "formal_method_baseline_comparison_validation_scale_only"
        if metric_status == "measured_formal"
        else "formal_method_baseline_comparison_missing_measured_formal"
    )
    return with_flow_evidence_protocol_defaults({
        "record_version": "formal_method_baseline_comparison_v1",
        "method_id": method_id,
        "method_role": method_role,
        "metric_status": metric_status,
        "comparison_scope": "fair_detection_calibration_at_target_fpr",
        "comparison_primary_metric_name": "tpr_at_target_fpr",
        "comparison_primary_metric_value": record.get("tpr_at_target_fpr") if record else None,
        "source_fair_detection_target_fpr": record.get("target_fpr") if record else None,
        "comparison_score_field": record.get("positive_score_field") if record else "missing_fair_detection_calibration",
        "comparison_record_count": record.get("attacked_positive_score_count") if record else 0,
        "comparison_anchor_count": len(anchor_keys),
        "comparison_anchor_keys": sorted(anchor_keys),
        "reference_anchor_count": len(reference_anchor_keys or anchor_keys),
        "missing_reference_anchor_count": 0 if method_role == "proposed_method" else len((reference_anchor_keys or set()) - anchor_keys),
        "extra_anchor_count": 0 if method_role == "proposed_method" else len(anchor_keys - (reference_anchor_keys or set())),
        "comparison_anchor_alignment_status": anchor_alignment_status,
        "comparison_prompt_count": record.get("prompt_count") if record else 0,
        "comparison_attack_count": record.get("attack_count") if record else 0,
        "comparison_attack_names": comparison_attack_names,
        "required_runtime_attack_names": required_attack_names,
        "missing_required_runtime_attack_names": missing_required_attack_names,
        "missing_required_runtime_attack_count": len(missing_required_attack_names),
        "comparison_positive_count": record.get("detected_positive_count_at_target_fpr") if record else None,
        "comparison_positive_rate": record.get("tpr_at_target_fpr") if record else None,
        "comparison_score_mean": None,
        "calibrated_threshold": record.get("calibrated_threshold") if record else None,
        "heldout_fpr_at_calibrated_threshold": record.get("heldout_fpr_at_calibrated_threshold") if record else None,
        "clean_negative_score_count": record.get("clean_negative_score_count") if record else 0,
        "score_semantics": record.get("score_semantics") if record else None,
        "comparison_missing_reason": missing_reason if metric_status == "missing" else "none",
        "claim_support_status": claim_support_status,
        **{
            key: value
            for key, value in profile_context.items()
            if key not in {"required_modern_external_baseline_adapter_names", "required_runtime_attack_names"}
        },
    }, trajectory_source_level="formal_method_baseline_comparison", claim_support_status=claim_support_status)


def build_formal_method_baseline_comparison_records(
    run_root: str | Path,
    config_path: str | Path = DEFAULT_PROTOCOL_CONFIG,
) -> list[dict[str, Any]]:
    """构建 SSTW 与 5 个现代 baseline 的同协议统计 records。"""
    run_root = Path(run_root)
    profile_context = _load_profile_context(config_path)
    rows: list[dict[str, Any]] = []
    fair_records = _read_jsonl(run_root / "records" / "fair_detection_calibration_records.jsonl")
    fair_by_method = {str(record.get("method_id") or ""): record for record in fair_records if record.get("method_id")}
    sstw_anchor_keys = {
        str(item)
        for item in fair_by_method.get(SSTW_METHOD_ID, {}).get("positive_anchor_keys", [])
        if str(item)
    }
    rows.append(_method_row(
        method_id=SSTW_METHOD_ID,
        method_role="proposed_method",
        record=fair_by_method.get(SSTW_METHOD_ID),
        profile_context=profile_context,
        missing_reason="missing_or_blocked_sstw_fair_detection_calibration_record",
        reference_anchor_keys=sstw_anchor_keys,
    ))

    for baseline_id in profile_context["required_modern_external_baseline_adapter_names"]:
        rows.append(_method_row(
            method_id=baseline_id,
            method_role="modern_external_baseline",
            record=fair_by_method.get(baseline_id),
            profile_context=profile_context,
            missing_reason="missing_or_blocked_external_baseline_fair_detection_calibration_record",
            reference_anchor_keys=sstw_anchor_keys,
        ))
    return rows


def audit_formal_method_baseline_comparison_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """审计同协议 measured_formal 比较表是否覆盖 SSTW 与所有现代 baseline。"""
    required_method_count = len(records)
    ready_rows = [record for record in records if record.get("metric_status") == "measured_formal"]
    missing_method_ids = [
        str(record.get("method_id"))
        for record in records
        if record.get("metric_status") != "measured_formal"
    ]
    sstw_ready = any(
        record.get("method_id") == SSTW_METHOD_ID and record.get("metric_status") == "measured_formal"
        for record in records
    )
    baseline_ready_count = sum(
        1
        for record in records
        if record.get("method_role") == "modern_external_baseline"
        and record.get("metric_status") == "measured_formal"
    )
    decision = "PASS" if records and sstw_ready and not missing_method_ids else "FAIL"
    return {
        "stage_id": "formal_method_baseline_comparison",
        "formal_method_baseline_comparison_decision": decision,
        "claim_support_status": "formal_method_baseline_comparison_validation_scale_only" if decision == "PASS" else "formal_method_baseline_comparison_blocked",
        "paper_result_level": records[0].get("paper_result_level") if records else None,
        "target_fpr": records[0].get("target_fpr") if records else None,
        "formal_comparison_required_method_count": required_method_count,
        "formal_comparison_ready_method_count": len(ready_rows),
        "formal_comparison_modern_baseline_ready_count": baseline_ready_count,
        "formal_comparison_sstw_ready": sstw_ready,
        "formal_comparison_missing_method_ids": missing_method_ids,
        "formal_comparison_missing_method_count": len(missing_method_ids),
    }


def run_formal_method_baseline_comparison(
    run_root: str | Path,
    config_path: str | Path = DEFAULT_PROTOCOL_CONFIG,
) -> dict[str, Any]:
    """写出同协议 measured_formal 比较 records、table、decision 和 report。"""
    run_root = Path(run_root)
    records = build_formal_method_baseline_comparison_records(run_root, config_path)
    audit = audit_formal_method_baseline_comparison_records(records)
    write_jsonl(run_root / "records" / "formal_method_baseline_comparison_records.jsonl", records)
    write_csv(run_root / "tables" / "formal_method_baseline_comparison_table.csv", records)
    write_json(run_root / "artifacts" / "formal_method_baseline_comparison_decision.json", audit)
    report = (
        "# Formal Method Baseline Comparison Report\n\n"
        "该报告只聚合已经通过 clean negative calibration 的 fair_detection_calibration records, "
        "主指标为 `tpr_at_target_fpr`。这保证 SSTW 与 5 个现代 external baseline 处在同 FPR、"
        "同攻击锚点、同证据层级的统计表中。validation_scale 结果仍不支持最终效果主张。\n\n"
        f"- formal_method_baseline_comparison_decision: {audit['formal_method_baseline_comparison_decision']}\n"
        f"- paper_result_level: {audit['paper_result_level']}\n"
        f"- target_fpr: {audit['target_fpr']}\n"
        f"- formal_comparison_ready_method_count: {audit['formal_comparison_ready_method_count']}\n"
        f"- formal_comparison_modern_baseline_ready_count: {audit['formal_comparison_modern_baseline_ready_count']}\n"
        f"- formal_comparison_missing_method_ids: {', '.join(audit['formal_comparison_missing_method_ids']) if audit['formal_comparison_missing_method_ids'] else 'none'}\n"
        f"- claim_support_status: {audit['claim_support_status']}\n"
    )
    report_path = run_root / "reports" / "formal_method_baseline_comparison_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 SSTW 与现代 external baseline 的同协议 measured_formal 比较表。")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--config-path", default=DEFAULT_PROTOCOL_CONFIG)
    args = parser.parse_args()
    payload = run_formal_method_baseline_comparison(args.run_root, args.config_path)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
