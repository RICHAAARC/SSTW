"""paper profile 统计置信区间报告。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluation.statistics.clustered_inference import (
    clustered_mean_interval,
    one_sided_binomial_upper_bound,
)
from evaluation.attacks.video_runtime_attack_protocol import load_protocol_config_with_shared_attack_protocol
from evaluation.protocol.flow_evidence_fields import with_flow_evidence_protocol_defaults
from evaluation.protocol.record_writer import write_json, write_jsonl
from evaluation.protocol.table_builder import write_csv


DEFAULT_PROTOCOL_CONFIG = "configs/protocol/probe_paper_generative_probe.json"


def _read_json(path: Path) -> dict[str, Any]:
    """读取 JSON config 或 artifact, 并兼容 UTF-8 BOM。"""
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON 顶层必须是对象: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict]:
    """读取 JSONL records, 文件不存在时返回空列表。"""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_profile_context(config_path: str | Path) -> dict[str, Any]:
    """从 protocol config 读取当前 CI 报告所属的 profile 语义。"""
    config = load_protocol_config_with_shared_attack_protocol(config_path)
    if "target_fpr" not in config:
        raise KeyError(f"protocol config 缺少 target_fpr: {config_path}")
    return {
        "target_fpr": float(config["target_fpr"]),
        "paper_result_level": config.get("paper_result_level", "probe_paper"),
        "target_fpr_source_config_path": str(config_path),
        "required_runtime_attack_names": [
            str(value)
            for value in config.get("required_runtime_attack_names", [])
            if str(value).strip()
        ],
        "minimum_sstw_worst_attack_tpr_ci_lower": float(
            config["minimum_sstw_worst_attack_tpr_ci_lower"]
        ),
    }


def _format_fpr(value: float | None) -> str:
    """把 FPR 数值格式化为报告中的稳定短文本。"""
    if value is None:
        return "未配置"
    return f"{float(value):g}"


def _cluster_values(source: dict[str, Any]) -> dict[str, list[float]]:
    """把同一 prompt/seed 下的不同攻击视为簇内重复测量。"""

    grouped: dict[str, list[float]] = {}
    for unit in source.get("positive_detection_units_at_target_fpr") or []:
        if not isinstance(unit, dict):
            continue
        anchor = str(unit.get("comparison_anchor_key") or "")
        parts = anchor.split("::")
        if len(parts) < 2:
            continue
        cluster_id = "::".join(parts[:2])
        grouped.setdefault(cluster_id, []).append(
            float(bool(unit.get("detected_at_target_fpr")))
        )
    return grouped


def _attack_cluster_values(
    source: dict[str, Any],
    attack_name: str,
) -> dict[str, list[float]]:
    """提取一个预注册攻击的逐视频检测结果，禁止由其他攻击均值补偿。"""

    grouped: dict[str, list[float]] = {}
    for unit in source.get("positive_detection_units_at_target_fpr") or []:
        if not isinstance(unit, dict):
            continue
        if str(unit.get("attack_name") or "") != attack_name:
            continue
        prompt_id = str(unit.get("prompt_id") or "")
        seed_id = str(unit.get("seed_id") or "")
        if not prompt_id or not seed_id:
            continue
        grouped.setdefault(f"{prompt_id}::{seed_id}", []).append(
            float(bool(unit.get("detected_at_target_fpr")))
        )
    return grouped


def _negative_cluster_outcomes(source: dict[str, Any]) -> list[bool]:
    """读取 fair calibration 已按视频最大分数聚合的 held-out FPR 单元。"""

    return [
        bool(unit.get("false_positive_at_target_fpr"))
        for unit in source.get("negative_detection_units_at_target_fpr") or []
        if isinstance(unit, dict) and unit.get("statistical_cluster_id")
    ]


def build_statistical_confidence_interval_records(
    run_root: str | Path,
    config_path: str | Path = DEFAULT_PROTOCOL_CONFIG,
) -> list[dict]:
    """从公平校准 records 构建 paper profile 置信区间 records。"""
    run_root = Path(run_root)
    profile_context = _load_profile_context(config_path)
    fair_records = [
        record for record in _read_jsonl(run_root / "records" / "fair_detection_calibration_records.jsonl")
        if record.get("fair_comparison_status") == "ready"
        and record.get("metric_status") == "measured_formal"
    ]
    rows: list[dict] = []
    for source in fair_records:
        clustered = clustered_mean_interval(
            _cluster_values(source),
            purpose=f"paper_profile_tpr:{source.get('method_id')}",
        )
        total_count = clustered.observation_count
        success_count = int(source.get("detected_positive_count_at_target_fpr") or 0)
        claim_status = "formal_cluster_bootstrap_ci_from_fair_detection_calibration"
        rows.append(with_flow_evidence_protocol_defaults({
            "record_version": "formal_statistical_confidence_interval_v2",
            "statistical_confidence_interval_family": "tpr_at_target_fpr",
            "method_id": source.get("method_id"),
            "method_role": source.get("method_role"),
            **profile_context,
            "ci_success_count": success_count,
            "ci_total_count": total_count,
            "ci_point_estimate": round(clustered.estimate, 6),
            "ci_cluster_bootstrap_lower": round(clustered.confidence_interval_lower, 6),
            "ci_cluster_bootstrap_upper": round(clustered.confidence_interval_upper, 6),
            "ci_statistical_cluster_count": clustered.cluster_count,
            "ci_bootstrap_resample_count": clustered.bootstrap_resample_count,
            "ci_confidence_level": 0.95,
            "ci_evidence_level": "fair_detection_calibration_measured_formal",
            "cluster_by_video_interval_status": "source_video_cluster_bootstrap_complete",
            "paper_low_fpr_ci_status": "formal_target_fpr_ci_ready",
            "claim_support_status": claim_status,
        }, trajectory_source_level="fair_detection_calibration_records", claim_support_status=claim_status))
        for attack_name in profile_context["required_runtime_attack_names"]:
            attack_cluster_values = _attack_cluster_values(source, attack_name)
            if attack_cluster_values:
                attack_interval = clustered_mean_interval(
                    attack_cluster_values,
                    purpose=(
                        f"paper_profile_tpr_per_attack:{source.get('method_id')}::"
                        f"{attack_name}"
                    ),
                )
                attack_total_count = attack_interval.observation_count
                attack_success_count = sum(
                    int(value)
                    for values in attack_cluster_values.values()
                    for value in values
                )
                attack_claim_status = "formal_per_attack_cluster_bootstrap_ci_ready"
            else:
                attack_interval = None
                attack_total_count = 0
                attack_success_count = 0
                attack_claim_status = "formal_per_attack_cluster_bootstrap_ci_missing"
            rows.append(with_flow_evidence_protocol_defaults({
                "record_version": "formal_statistical_confidence_interval_v2",
                "statistical_confidence_interval_family": (
                    "tpr_at_target_fpr_per_attack"
                ),
                "method_id": source.get("method_id"),
                "method_role": source.get("method_role"),
                "attack_name": attack_name,
                **profile_context,
                "ci_success_count": attack_success_count,
                "ci_total_count": attack_total_count,
                "ci_point_estimate": (
                    round(attack_interval.estimate, 6)
                    if attack_interval is not None
                    else None
                ),
                "ci_cluster_bootstrap_lower": (
                    round(attack_interval.confidence_interval_lower, 6)
                    if attack_interval is not None
                    else None
                ),
                "ci_cluster_bootstrap_upper": (
                    round(attack_interval.confidence_interval_upper, 6)
                    if attack_interval is not None
                    else None
                ),
                "ci_statistical_cluster_count": (
                    attack_interval.cluster_count
                    if attack_interval is not None
                    else 0
                ),
                "ci_bootstrap_resample_count": (
                    attack_interval.bootstrap_resample_count
                    if attack_interval is not None
                    else 0
                ),
                "ci_confidence_level": 0.95,
                "ci_evidence_level": (
                    "fair_detection_calibration_measured_formal"
                    if attack_interval is not None
                    else "missing"
                ),
                "cluster_by_video_interval_status": (
                    "source_video_cluster_bootstrap_complete"
                    if attack_interval is not None
                    else "missing"
                ),
                "paper_low_fpr_ci_status": (
                    "formal_target_fpr_ci_ready"
                    if attack_interval is not None
                    else "missing"
                ),
                "claim_support_status": attack_claim_status,
            }, trajectory_source_level="fair_detection_calibration_records", claim_support_status=attack_claim_status))
        negative_outcomes = _negative_cluster_outcomes(source)
        false_positive_count = sum(negative_outcomes)
        heldout_negative_count = len(negative_outcomes)
        fpr_upper = one_sided_binomial_upper_bound(
            false_positive_count,
            heldout_negative_count,
        ) if heldout_negative_count else None
        fpr_claim_status = (
            "formal_heldout_fpr_exact_interval_ready"
            if fpr_upper is not None
            else "formal_heldout_fpr_exact_interval_missing"
        )
        rows.append(with_flow_evidence_protocol_defaults({
            "record_version": "formal_statistical_confidence_interval_v2",
            "statistical_confidence_interval_family": "heldout_fpr_at_frozen_threshold",
            "method_id": source.get("method_id"),
            "method_role": source.get("method_role"),
            **profile_context,
            "ci_success_count": false_positive_count,
            "ci_total_count": heldout_negative_count,
            "ci_point_estimate": round(false_positive_count / heldout_negative_count, 8) if heldout_negative_count else None,
            "ci_one_sided_exact_upper": round(fpr_upper, 8) if fpr_upper is not None else None,
            "ci_statistical_cluster_count": heldout_negative_count,
            "ci_confidence_level": 0.95,
            "ci_evidence_level": "fair_detection_calibration_measured_formal",
            "cluster_by_video_interval_status": "source_video_exact_binomial_complete" if fpr_upper is not None else "missing",
            "paper_low_fpr_ci_status": "formal_target_fpr_ci_ready" if fpr_upper is not None else "missing",
            "claim_support_status": fpr_claim_status,
        }, trajectory_source_level="fair_detection_calibration_records", claim_support_status=fpr_claim_status))
    return rows


def audit_statistical_confidence_interval_records(records: list[dict]) -> dict[str, Any]:
    """审计 paper profile CI records 是否可用于后续 gate。"""
    ready_records = [
        record
        for record in records
        if int(record.get("ci_total_count") or 0) > 0
        and (
            (
                record.get("statistical_confidence_interval_family") == "tpr_at_target_fpr"
                and record.get("ci_cluster_bootstrap_lower") is not None
                and record.get("ci_cluster_bootstrap_upper") is not None
            )
            or (
                record.get("statistical_confidence_interval_family")
                == "tpr_at_target_fpr_per_attack"
                and record.get("ci_cluster_bootstrap_lower") is not None
                and record.get("ci_cluster_bootstrap_upper") is not None
            )
            or (
                record.get("statistical_confidence_interval_family") == "heldout_fpr_at_frozen_threshold"
                and record.get("ci_one_sided_exact_upper") is not None
            )
        )
        and record.get("ci_evidence_level") == "fair_detection_calibration_measured_formal"
    ]
    record = ready_records[0] if ready_records else (records[0] if records else {})
    tpr_records = [row for row in ready_records if row.get("statistical_confidence_interval_family") == "tpr_at_target_fpr"]
    per_attack_tpr_records = [
        row for row in ready_records
        if row.get("statistical_confidence_interval_family")
        == "tpr_at_target_fpr_per_attack"
    ]
    fpr_records = [row for row in ready_records if row.get("statistical_confidence_interval_family") == "heldout_fpr_at_frozen_threshold"]
    total_count = sum(int(row.get("ci_total_count") or 0) for row in tpr_records)
    success_count = sum(int(row.get("ci_success_count") or 0) for row in tpr_records)
    target_fpr = float(record.get("target_fpr") or 0.0)
    maximum_fpr_upper = max((float(row["ci_one_sided_exact_upper"]) for row in fpr_records), default=None)
    fpr_point_closed = bool(fpr_records) and all(
        float(row.get("ci_point_estimate") or 0.0) <= target_fpr
        for row in fpr_records
    )
    confidence_bound_available = maximum_fpr_upper is not None
    confidence_upper_within_target = bool(
        maximum_fpr_upper is not None and maximum_fpr_upper <= target_fpr
    )
    required_attacks = {
        str(value)
        for value in record.get("required_runtime_attack_names", [])
        if str(value).strip()
    }
    method_ids = {
        str(row.get("method_id") or "") for row in tpr_records
    } - {""}
    per_attack_scope_map = {
        (str(row.get("method_id") or ""), str(row.get("attack_name") or "")): row
        for row in per_attack_tpr_records
    }
    missing_per_attack_scopes = [
        f"{method_id}::{attack_name}"
        for method_id in sorted(method_ids)
        for attack_name in sorted(required_attacks)
        if (method_id, attack_name) not in per_attack_scope_map
    ]
    sstw_attack_rows = [
        row for row in per_attack_tpr_records
        if row.get("method_id") == "sstw_key_conditioned_flow_trajectory"
    ]
    worst_attack_row = min(
        sstw_attack_rows,
        key=lambda row: float(row["ci_cluster_bootstrap_lower"]),
        default=None,
    )
    worst_attack_lower = (
        float(worst_attack_row["ci_cluster_bootstrap_lower"])
        if worst_attack_row is not None
        else None
    )
    minimum_worst_attack_lower = float(
        record.get("minimum_sstw_worst_attack_tpr_ci_lower") or 0.0
    )
    worst_attack_ready = bool(
        worst_attack_lower is not None
        and worst_attack_lower >= minimum_worst_attack_lower
        and len(sstw_attack_rows) == len(required_attacks)
    )
    decision = "PASS" if (
        ready_records
        and len(ready_records) == len(records)
        and fpr_point_closed
        and confidence_bound_available
        and confidence_upper_within_target
        and not missing_per_attack_scopes
        and worst_attack_ready
    ) else "FAIL"
    return {
        "stage_id": "statistical_confidence_interval_reporter",
        "statistical_confidence_interval_decision": decision,
        "claim_support_status": "formal_cluster_bootstrap_ci_from_fair_detection_calibration" if decision == "PASS" else "formal_ci_blocked",
        "ci_record_count": len(records),
        "paper_result_level": record.get("paper_result_level"),
        "target_fpr": record.get("target_fpr"),
        "target_fpr_source_config_path": record.get("target_fpr_source_config_path"),
        "ci_total_count": total_count,
        "ci_success_count": success_count,
        "ci_point_estimate": round(success_count / total_count, 6) if total_count else None,
        "ci_cluster_bootstrap_lower": record.get("ci_cluster_bootstrap_lower"),
        "ci_cluster_bootstrap_upper": record.get("ci_cluster_bootstrap_upper"),
        "ci_statistical_cluster_count": record.get("ci_statistical_cluster_count"),
        "heldout_fpr_ci_record_count": len(fpr_records),
        "per_attack_tpr_ci_record_count": len(per_attack_tpr_records),
        "per_attack_tpr_ci_missing_scopes": missing_per_attack_scopes,
        "per_attack_tpr_ci_ready": not missing_per_attack_scopes,
        "worst_attack_name": (
            worst_attack_row.get("attack_name") if worst_attack_row else None
        ),
        "worst_attack_tpr_ci_lower": worst_attack_lower,
        "minimum_sstw_worst_attack_tpr_ci_lower": minimum_worst_attack_lower,
        "heldout_fpr_one_sided_exact_upper_maximum": maximum_fpr_upper,
        "heldout_fpr_point_target_closed": fpr_point_closed,
        "heldout_fpr_confidence_bound_available": confidence_bound_available,
        "heldout_fpr_confidence_upper_within_target": confidence_upper_within_target,
        "paper_low_fpr_ci_status": record.get("paper_low_fpr_ci_status", "missing"),
        "cluster_by_video_interval_status": record.get("cluster_by_video_interval_status", "missing"),
    }


def run_statistical_confidence_interval_reporter(
    run_root: str | Path,
    config_path: str | Path = DEFAULT_PROTOCOL_CONFIG,
) -> dict[str, Any]:
    """写出 paper profile CI records、table、decision 和 report。"""
    run_root = Path(run_root)
    records = build_statistical_confidence_interval_records(run_root, config_path)
    audit = audit_statistical_confidence_interval_records(records)
    write_jsonl(run_root / "records" / "statistical_confidence_interval_records.jsonl", records)
    write_csv(run_root / "tables" / "statistical_confidence_interval_table.csv", records)
    write_json(run_root / "artifacts" / "statistical_confidence_interval_decision.json", audit)
    target_fpr_text = _format_fpr(audit.get("target_fpr"))
    report = (
        "# Statistical Confidence Interval Report\n\n"
        "该报告基于 fair_detection_calibration_records 中的 measured_formal TPR@target FPR "
        "按 source-video cluster 计算配对 bootstrap 区间。"
        f"当前 profile 的 target_fpr={target_fpr_text}, 该数值来自 protocol config。"
        "本报告不会自动替代更低 FPR profile 的正式大规模统计报告。\n\n"
        f"- statistical_confidence_interval_decision: {audit['statistical_confidence_interval_decision']}\n"
        f"- paper_result_level: {audit['paper_result_level']}\n"
        f"- target_fpr: {target_fpr_text}\n"
        f"- ci_total_count: {audit['ci_total_count']}\n"
        f"- ci_point_estimate: {audit['ci_point_estimate']}\n"
        f"- ci_cluster_bootstrap_lower: {audit['ci_cluster_bootstrap_lower']}\n"
        f"- ci_cluster_bootstrap_upper: {audit['ci_cluster_bootstrap_upper']}\n"
        f"- claim_support_status: {audit['claim_support_status']}\n"
    )
    report_path = run_root / "reports" / "statistical_confidence_interval_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 paper profile 统计置信区间报告。")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--config-path", default=DEFAULT_PROTOCOL_CONFIG)
    args = parser.parse_args()
    payload = run_statistical_confidence_interval_reporter(args.run_root, args.config_path)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
