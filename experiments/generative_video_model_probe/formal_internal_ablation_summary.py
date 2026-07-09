"""paper profile 级内部消融汇总层。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from experiments.generative_video_model_probe.validation_internal_ablation import VALIDATION_ABLATION_VARIANTS
from main.protocol.flow_evidence_fields import with_flow_evidence_protocol_defaults
from main.protocol.record_writer import write_json, write_jsonl
from main.protocol.table_builder import write_csv


FULL_METHOD_VARIANT = "sstw_full_method"


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


def _variant_config_by_name() -> dict[str, dict[str, Any]]:
    """按 method_variant 索引内部消融配置。"""
    return {str(row["method_variant"]): dict(row) for row in VALIDATION_ABLATION_VARIANTS}


def _mean_or_none(values: list[float]) -> float | None:
    """计算均值, 空列表返回 None。"""
    return round(mean(values), 6) if values else None


def build_formal_internal_ablation_summary_records(run_root: str | Path) -> list[dict[str, Any]]:
    """把 SSTW formal full-method 结果和 validation proxy 消融矩阵合成 paper_profile 消融表。

    该函数的核心作用是消除“只有 proxy ablation, 没有与 SSTW formal 结果绑定”的文档缺口。
    `sstw_full_method` 行来自 SSTW measured_formal records; 其余 component-removal 行仍来自
    paper_profile proxy ablation, 因此只能支持小样本论文闭合检查, 不能支持 full-paper
    正式消融效果主张。
    """
    run_root = Path(run_root)
    sstw_records = [
        record
        for record in _read_jsonl(run_root / "records" / "sstw_measured_formal_records.jsonl")
        if record.get("metric_status") == "measured_formal"
    ]
    proxy_ablation_records = [
        record
        for record in _read_jsonl(run_root / "records" / "validation_internal_ablation_records.jsonl")
        if record.get("ablation_status") == "ready"
    ]
    variant_config = _variant_config_by_name()
    full_scores = [
        value
        for value in (_safe_float(record.get("sstw_score")) for record in sstw_records)
        if value is not None
    ]
    full_score_mean = _mean_or_none(full_scores)
    rows: list[dict[str, Any]] = []
    for variant_name, config in variant_config.items():
        if variant_name == FULL_METHOD_VARIANT:
            variant_scores = full_scores
            metric_status = "measured_formal" if variant_scores else "missing"
            evidence_level = "sstw_measured_formal_full_method"
            source_record_family = "sstw_measured_formal_records"
        else:
            variant_scores = [
                value
                for value in (
                    _safe_float(record.get("validation_ablation_proxy_score"))
                    for record in proxy_ablation_records
                    if record.get("method_variant") == variant_name
                )
                if value is not None
            ]
            metric_status = "measured_proxy" if variant_scores else "missing"
            evidence_level = "paper_profile_proxy_component_removal"
            source_record_family = "validation_internal_ablation_records"
        score_mean = _mean_or_none(variant_scores)
        delta = round(score_mean - full_score_mean, 6) if score_mean is not None and full_score_mean is not None else None
        claim_support_status = (
            "formal_internal_ablation_summary_ready_for_target_fpr_0_1_claim_context"
            if metric_status != "missing"
            else "formal_internal_ablation_summary_missing_variant"
        )
        rows.append(with_flow_evidence_protocol_defaults({
            "record_version": "formal_internal_ablation_summary_v1",
            "method_variant": variant_name,
            "ablation_family": config.get("ablation_family"),
            "ablation_removed_component": config.get("ablation_removed_component"),
            "metric_status": metric_status,
            "formal_internal_ablation_evidence_level": evidence_level,
            "formal_internal_ablation_source_record_family": source_record_family,
            "formal_internal_ablation_record_count": len(variant_scores),
            "formal_internal_ablation_score_mean": score_mean,
            "formal_internal_ablation_full_method_score_mean": full_score_mean,
            "formal_internal_ablation_delta_vs_full_method": delta,
            "claim_support_status": claim_support_status,
        }, trajectory_source_level="formal_internal_ablation_summary", claim_support_status=claim_support_status))
    return rows


def audit_formal_internal_ablation_summary_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """审计 paper_profile 级内部消融汇总是否完整。"""
    expected_variants = {str(row["method_variant"]) for row in VALIDATION_ABLATION_VARIANTS}
    ready_variants = {
        str(record.get("method_variant"))
        for record in records
        if record.get("metric_status") in {"measured_formal", "measured_proxy"}
    }
    full_formal_ready = any(
        record.get("method_variant") == FULL_METHOD_VARIANT and record.get("metric_status") == "measured_formal"
        for record in records
    )
    missing_variants = sorted(expected_variants - ready_variants)
    decision = "PASS" if full_formal_ready and not missing_variants else "FAIL"
    return {
        "stage_id": "formal_internal_ablation_summary",
        "formal_internal_ablation_summary_decision": decision,
        "claim_support_status": "formal_internal_ablation_summary_ready_for_target_fpr_0_1_claim_context"
        if decision == "PASS"
        else "formal_internal_ablation_summary_blocked",
        "formal_internal_ablation_variant_count": len(ready_variants),
        "formal_internal_ablation_expected_variant_count": len(expected_variants),
        "formal_internal_ablation_full_method_formal_ready": full_formal_ready,
        "formal_internal_ablation_missing_variants": missing_variants,
        "formal_internal_ablation_missing_variant_count": len(missing_variants),
    }


def run_formal_internal_ablation_summary(run_root: str | Path) -> dict[str, Any]:
    """写出 paper_profile 级内部消融 records、table、decision 和 report。"""
    run_root = Path(run_root)
    records = build_formal_internal_ablation_summary_records(run_root)
    audit = audit_formal_internal_ablation_summary_records(records)
    write_jsonl(run_root / "records" / "formal_internal_ablation_summary_records.jsonl", records)
    write_csv(run_root / "tables" / "formal_internal_ablation_summary_table.csv", records)
    write_json(run_root / "artifacts" / "formal_internal_ablation_summary_decision.json", audit)
    report = (
        "# Formal Internal Ablation Summary Report\n\n"
        "该报告把 SSTW full-method measured_formal 结果与 paper profile proxy component-removal "
        "消融矩阵绑定, 用于确认内部消融产物在 paper_profile 阶段闭环。除 full-method 行外, "
        "component-removal 行用于支撑 target_fpr=0.1 小样本机制解释, 但不能外推为 full_paper 规模正式消融结论。\n\n"
        f"- formal_internal_ablation_summary_decision: {audit['formal_internal_ablation_summary_decision']}\n"
        f"- formal_internal_ablation_variant_count: {audit['formal_internal_ablation_variant_count']}\n"
        f"- formal_internal_ablation_full_method_formal_ready: {str(audit['formal_internal_ablation_full_method_formal_ready']).lower()}\n"
        f"- formal_internal_ablation_missing_variants: {', '.join(audit['formal_internal_ablation_missing_variants']) if audit['formal_internal_ablation_missing_variants'] else 'none'}\n"
        f"- claim_support_status: {audit['claim_support_status']}\n"
    )
    report_path = run_root / "reports" / "formal_internal_ablation_summary_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 paper_profile 级内部消融汇总。")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--config-path", default="", help="保留 profile config provenance, 当前阶段不读取该配置。")
    args = parser.parse_args()
    payload = run_formal_internal_ablation_summary(args.run_root)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
