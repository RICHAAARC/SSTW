"""论文级结果图表与补充实验产物的统一构建器。

该模块的作用是把 validation_scale、pilot_paper 和 full_paper 共享的论文产物
结构固定下来。Notebook 与服务器 CLI 只调用本模块命令, 不在入口层手写图表、
低 FPR 曲线、效率指标或 adaptive / real-world 攻击汇总逻辑。

这些构建器只从已有 governed records 派生 tables / figures / reports。它们不会
补造检测分数, 也不会把 proxy 协议伪装成 measured claim。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping

from main.protocol.flow_evidence_fields import with_flow_evidence_protocol_defaults
from main.protocol.record_writer import write_json, write_jsonl
from main.protocol.table_builder import write_csv


DEFAULT_PROTOCOL_CONFIG = "configs/protocol/validation_scale_generative_probe.json"
SSTW_METHOD_ID = "sstw_key_conditioned_flow_trajectory"
DEFAULT_TARGET_FPR_LEVELS = (0.1, 0.01, 0.001)
REAL_WORLD_ATTACK_PROTOCOLS = (
    "platform_transcode_proxy_runtime",
    "generative_recompression_or_regeneration_attack",
    "screen_recording_or_capture_protocol",
)

PAPER_RESULT_ARTIFACT_RELPATHS = (
    "records/video_quality_metric_records.jsonl",
    "tables/video_quality_metric_table.csv",
    "artifacts/video_quality_metric_decision.json",
    "reports/video_quality_metric_report.md",
    "figures/video_quality_robustness_tradeoff_figure.json",
    "records/efficiency_metric_records.jsonl",
    "tables/efficiency_metric_table.csv",
    "artifacts/efficiency_metric_decision.json",
    "reports/efficiency_metric_report.md",
    "figures/efficiency_comparison_figure.json",
    "records/low_fpr_curve_records.jsonl",
    "tables/low_fpr_curve_table.csv",
    "artifacts/low_fpr_curve_decision.json",
    "reports/low_fpr_curve_report.md",
    "figures/low_fpr_curve_figure.json",
    "records/real_adaptive_attack_records.jsonl",
    "tables/real_adaptive_attack_table.csv",
    "artifacts/real_adaptive_attack_decision.json",
    "reports/real_adaptive_attack_report.md",
    "figures/real_adaptive_attack_robustness_figure.json",
    "records/real_world_attack_records.jsonl",
    "tables/real_world_attack_table.csv",
    "artifacts/real_world_attack_decision.json",
    "reports/real_world_attack_report.md",
    "figures/real_world_attack_robustness_figure.json",
    "artifacts/paper_result_artifact_skeleton_decision.json",
    "reports/paper_result_artifact_skeleton_report.md",
)


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
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _safe_float(value: Any) -> float | None:
    """把字段安全转换为 float, 无法转换时返回 None。"""

    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean_optional(values: Iterable[Any]) -> float | None:
    """对可解析数值求平均值, 没有有效数值时返回 None。"""

    parsed = [float(value) for value in (_safe_float(item) for item in values) if value is not None]
    if not parsed:
        return None
    return round(mean(parsed), 6)


def _load_protocol_context(config_path: str | Path) -> dict[str, Any]:
    """读取当前论文 profile 的产物构建上下文。"""

    path = Path(config_path)
    config = _read_json(path)
    if "target_fpr" not in config:
        raise KeyError(f"protocol config 缺少 target_fpr: {path}")
    return {
        "paper_result_level": str(config.get("paper_result_level") or "validation_scale"),
        "target_fpr": float(config["target_fpr"]),
        "protocol_config_path": str(path),
        "required_runtime_attack_names": [
            str(item) for item in config.get("required_runtime_attack_names", []) if str(item)
        ],
        "required_non_runtime_attack_protocols": [
            str(item) for item in config.get("required_non_runtime_attack_protocols", []) if str(item)
        ],
        "target_fpr_levels": [float(item) for item in config.get("target_fpr_levels", DEFAULT_TARGET_FPR_LEVELS)],
        "claim_support_status": str(
            config.get("claim_support_status") or "paper_profile_artifact_skeleton_not_claim_evidence"
        ),
    }


def _claim_status_for_current_profile(context: Mapping[str, Any], artifact_status: str) -> str:
    """生成不冒充最终论文结论的 claim_support_status。"""

    if artifact_status == "ready":
        return f"{context['paper_result_level']}_paper_result_artifact_ready"
    return f"{context['paper_result_level']}_paper_result_artifact_blocked"


def _target_fpr_matches(record: Mapping[str, Any], target_fpr: float) -> bool:
    """判断 record 的 target_fpr 是否等于当前 protocol config。"""

    value = _safe_float(record.get("target_fpr"))
    return value is not None and abs(value - float(target_fpr)) <= 1e-12


def build_video_quality_metric_records(run_root: str | Path, context: Mapping[str, Any]) -> list[dict[str, Any]]:
    """从正式视频质量 records 构建论文质量指标汇总 records。"""

    run_root = Path(run_root)
    formal_records = _read_jsonl(run_root / "records" / "formal_quality_motion_semantic_records.jsonl")
    records: list[dict[str, Any]] = []
    scopes = sorted({str(item.get("motion_claim_role") or "unspecified") for item in formal_records})
    for role in scopes:
        scoped = [item for item in formal_records if str(item.get("motion_claim_role") or "unspecified") == role]
        ready_count = sum(1 for item in scoped if item.get("formal_metric_result_used_for_claim") is True)
        status = "ready" if ready_count > 0 else "blocked"
        claim_status = _claim_status_for_current_profile(context, status)
        records.append(with_flow_evidence_protocol_defaults({
            "record_version": "paper_video_quality_metric_v1",
            "paper_result_level": context["paper_result_level"],
            "target_fpr": context["target_fpr"],
            "quality_metric_scope": role,
            "quality_metric_source_record_count": len(scoped),
            "formal_metric_ready_count": ready_count,
            "formal_metric_blocked_count": len(scoped) - ready_count,
            "mean_brightness": _mean_optional(item.get("mean_brightness") for item in scoped),
            "mean_contrast": _mean_optional(item.get("mean_contrast") for item in scoped),
            "mean_motion_delta": _mean_optional(item.get("motion_delta_mean") for item in scoped),
            "mean_temporal_flicker": _mean_optional(item.get("temporal_flicker_mean") for item in scoped),
            "mean_semantic_consistency_score": _mean_optional(item.get("semantic_consistency_score") for item in scoped),
            "video_quality_metric_status": status,
            "metric_status": "measured_formal" if status == "ready" else "missing",
            "claim_support_status": claim_status,
        }, trajectory_source_level="paper_quality_metrics_from_formal_video_records", claim_support_status=claim_status))
    return records


def audit_video_quality_metric_records(records: list[dict[str, Any]], context: Mapping[str, Any]) -> dict[str, Any]:
    """审计视频质量指标派生产物是否可用于构图。"""

    ready_count = sum(1 for item in records if item.get("video_quality_metric_status") == "ready")
    decision = "PASS" if records and ready_count > 0 else "FAIL"
    return {
        "stage_id": "video_quality_metric_builder",
        "paper_result_level": context["paper_result_level"],
        "target_fpr": context["target_fpr"],
        "video_quality_metric_decision": decision,
        "video_quality_metric_record_count": len(records),
        "video_quality_metric_ready_count": ready_count,
        "claim_support_status": _claim_status_for_current_profile(context, "ready" if decision == "PASS" else "blocked"),
    }


def _fair_rows_for_current_target(run_root: Path, target_fpr: float) -> list[dict[str, Any]]:
    """读取当前 target_fpr 下 ready 的公平校准 records。"""

    return [
        record
        for record in _read_jsonl(run_root / "records" / "fair_detection_calibration_records.jsonl")
        if record.get("fair_comparison_status") == "ready" and _target_fpr_matches(record, target_fpr)
    ]


def build_low_fpr_curve_records(run_root: str | Path, context: Mapping[str, Any]) -> list[dict[str, Any]]:
    """构建当前 profile 的 TPR@target_fpr 曲线点记录。"""

    run_root = Path(run_root)
    current_target = float(context["target_fpr"])
    fair_rows = _fair_rows_for_current_target(run_root, current_target)
    records: list[dict[str, Any]] = []
    for row in fair_rows:
        status = "ready" if row.get("metric_status") == "measured_formal" else "blocked"
        claim_status = _claim_status_for_current_profile(context, status)
        records.append(with_flow_evidence_protocol_defaults({
            "record_version": "paper_low_fpr_curve_point_v1",
            "paper_result_level": context["paper_result_level"],
            "method_id": row.get("method_id"),
            "method_role": row.get("method_role"),
            "target_fpr": current_target,
            "curve_point_fpr_level": current_target,
            "curve_point_status": status,
            "calibrated_threshold": row.get("calibrated_threshold"),
            "heldout_fpr_at_calibrated_threshold": row.get("heldout_fpr_at_calibrated_threshold"),
            "tpr_at_target_fpr": row.get("tpr_at_target_fpr"),
            "tpr_ci_lower": row.get("tpr_ci_lower"),
            "tpr_ci_upper": row.get("tpr_ci_upper"),
            "clean_negative_score_count": row.get("clean_negative_score_count"),
            "attacked_positive_score_count": row.get("attacked_positive_score_count"),
            "metric_status": row.get("metric_status"),
            "claim_support_status": claim_status,
        }, trajectory_source_level="low_fpr_curve_from_fair_detection_calibration", claim_support_status=claim_status))
    observed_methods = {str(item.get("method_id")) for item in records if item.get("method_id")}
    for fpr_level in context.get("target_fpr_levels", DEFAULT_TARGET_FPR_LEVELS):
        if abs(float(fpr_level) - current_target) <= 1e-12:
            continue
        records.append(with_flow_evidence_protocol_defaults({
            "record_version": "paper_low_fpr_curve_point_v1",
            "paper_result_level": context["paper_result_level"],
            "method_id": "all_methods",
            "method_role": "profile_scope_marker",
            "target_fpr": current_target,
            "curve_point_fpr_level": float(fpr_level),
            "curve_point_status": "not_run_for_current_profile",
            "curve_point_scope_note": "该 FPR 等级需要切换到对应 workflow profile 后使用同一 calibrator 重新生成。",
            "covered_method_count_at_current_target_fpr": len(observed_methods),
            "metric_status": "missing",
            "claim_support_status": "other_target_fpr_requires_matching_profile_run",
        }, trajectory_source_level="low_fpr_curve_profile_scope_marker", claim_support_status="other_target_fpr_requires_matching_profile_run"))
    return records


def audit_low_fpr_curve_records(records: list[dict[str, Any]], context: Mapping[str, Any]) -> dict[str, Any]:
    """审计当前 profile 的低 FPR 曲线点是否就绪。"""

    current_target = float(context["target_fpr"])
    ready_records = [
        item for item in records
        if item.get("curve_point_status") == "ready"
        and _safe_float(item.get("curve_point_fpr_level")) == current_target
    ]
    decision = "PASS" if ready_records else "FAIL"
    return {
        "stage_id": "low_fpr_curve_builder",
        "paper_result_level": context["paper_result_level"],
        "target_fpr": current_target,
        "low_fpr_curve_decision": decision,
        "low_fpr_curve_record_count": len(records),
        "low_fpr_curve_ready_method_count": len({str(item.get("method_id")) for item in ready_records}),
        "available_curve_target_fpr_levels": sorted({float(item["curve_point_fpr_level"]) for item in records if item.get("curve_point_fpr_level") is not None}),
        "claim_support_status": _claim_status_for_current_profile(context, "ready" if decision == "PASS" else "blocked"),
    }


def build_efficiency_metric_records(run_root: str | Path, context: Mapping[str, Any]) -> list[dict[str, Any]]:
    """从 Notebook / CLI 阶段耗时 records 构建效率指标。"""

    run_root = Path(run_root)
    timing_records = _read_jsonl(run_root / "records" / "notebook_stage_timing_records.jsonl")
    runtime_report = _read_json(run_root / "artifacts" / "notebook_runtime_report.json")
    records: list[dict[str, Any]] = []
    for record in timing_records:
        elapsed_sec = _safe_float(record.get("stage_elapsed_sec"))
        status = "ready" if elapsed_sec is not None else "blocked"
        records.append(with_flow_evidence_protocol_defaults({
            "record_version": "paper_efficiency_metric_v1",
            "paper_result_level": context["paper_result_level"],
            "target_fpr": context["target_fpr"],
            "efficiency_metric_scope": "stage_runtime",
            "notebook_role": record.get("notebook_role"),
            "stage_name": record.get("stage_name"),
            "stage_execution_status": record.get("stage_execution_status"),
            "stage_elapsed_sec": elapsed_sec,
            "stage_elapsed_min": _safe_float(record.get("stage_elapsed_min")),
            "efficiency_metric_status": status,
            "metric_status": "measured_formal" if status == "ready" else "missing",
            "claim_support_status": "efficiency_runtime_estimation_only_not_effect_claim",
        }, trajectory_source_level="efficiency_metrics_from_stage_timing_records", claim_support_status="efficiency_runtime_estimation_only_not_effect_claim"))
    if runtime_report:
        elapsed_sec = _safe_float(runtime_report.get("notebook_elapsed_sec"))
        status = "ready" if elapsed_sec is not None else "blocked"
        records.append(with_flow_evidence_protocol_defaults({
            "record_version": "paper_efficiency_metric_v1",
            "paper_result_level": context["paper_result_level"],
            "target_fpr": context["target_fpr"],
            "efficiency_metric_scope": "total_runtime",
            "notebook_role": runtime_report.get("notebook_role"),
            "stage_name": "total_notebook_or_server_role_runtime",
            "stage_execution_status": runtime_report.get("notebook_timing_status"),
            "stage_elapsed_sec": elapsed_sec,
            "stage_elapsed_min": _safe_float(runtime_report.get("notebook_elapsed_min")),
            "efficiency_metric_status": status,
            "metric_status": "measured_formal" if status == "ready" else "missing",
            "claim_support_status": "efficiency_runtime_estimation_only_not_effect_claim",
        }, trajectory_source_level="efficiency_metrics_from_notebook_runtime_report", claim_support_status="efficiency_runtime_estimation_only_not_effect_claim"))
    return records


def audit_efficiency_metric_records(records: list[dict[str, Any]], context: Mapping[str, Any]) -> dict[str, Any]:
    """审计效率指标是否有可绘图记录。"""

    ready_count = sum(1 for item in records if item.get("efficiency_metric_status") == "ready")
    decision = "PASS" if ready_count > 0 else "FAIL"
    return {
        "stage_id": "efficiency_metric_builder",
        "paper_result_level": context["paper_result_level"],
        "target_fpr": context["target_fpr"],
        "efficiency_metric_decision": decision,
        "efficiency_metric_record_count": len(records),
        "efficiency_metric_ready_count": ready_count,
        "claim_support_status": "efficiency_runtime_estimation_only_not_effect_claim" if decision == "PASS" else "efficiency_metric_blocked",
    }


def build_real_adaptive_attack_records(run_root: str | Path, context: Mapping[str, Any]) -> list[dict[str, Any]]:
    """把 adaptive / non-runtime 协议 records 规整为论文图表输入。"""

    run_root = Path(run_root)
    source_records = _read_jsonl(run_root / "records" / "adaptive_attack_records.jsonl")
    by_protocol: dict[str, list[dict[str, Any]]] = {}
    for record in source_records:
        protocol = str(record.get("non_runtime_attack_protocol") or record.get("adaptive_attack_name") or "")
        if protocol:
            by_protocol.setdefault(protocol, []).append(record)
    required = [str(item) for item in context.get("required_non_runtime_attack_protocols", []) if str(item)]
    records: list[dict[str, Any]] = []
    for protocol in required:
        scoped = by_protocol.get(protocol, [])
        measured_count = sum(1 for item in scoped if item.get("metric_status") == "measured_formal")
        proxy_count = sum(1 for item in scoped if "proxy" in str(item.get("claim_support_status") or ""))
        status = "measured_ready" if measured_count else ("proxy_protocol_ready" if scoped else "missing")
        claim_status = "real_adaptive_attack_measured_ready" if measured_count else "real_adaptive_attack_governed_protocol_record_only"
        records.append(with_flow_evidence_protocol_defaults({
            "record_version": "paper_real_adaptive_attack_v1",
            "paper_result_level": context["paper_result_level"],
            "target_fpr": context["target_fpr"],
            "non_runtime_attack_protocol": protocol,
            "adaptive_attack_record_count": len(scoped),
            "adaptive_attack_measured_formal_count": measured_count,
            "adaptive_attack_proxy_record_count": proxy_count,
            "real_adaptive_attack_status": status,
            "metric_status": "measured_formal" if measured_count else "missing",
            "claim_support_status": claim_status,
        }, trajectory_source_level="real_adaptive_attack_summary_from_governed_records", claim_support_status=claim_status))
    return records


def audit_real_adaptive_attack_records(records: list[dict[str, Any]], context: Mapping[str, Any]) -> dict[str, Any]:
    """审计 adaptive 协议是否覆盖配置要求。"""

    missing = [str(item.get("non_runtime_attack_protocol")) for item in records if item.get("real_adaptive_attack_status") == "missing"]
    decision = "PASS" if records and not missing else "FAIL"
    return {
        "stage_id": "real_adaptive_attack_builder",
        "paper_result_level": context["paper_result_level"],
        "target_fpr": context["target_fpr"],
        "real_adaptive_attack_decision": decision,
        "real_adaptive_attack_record_count": len(records),
        "real_adaptive_attack_missing_protocols": missing,
        "real_adaptive_attack_measured_protocol_count": sum(1 for item in records if item.get("real_adaptive_attack_status") == "measured_ready"),
        "real_adaptive_attack_proxy_protocol_count": sum(1 for item in records if item.get("real_adaptive_attack_status") == "proxy_protocol_ready"),
        "claim_support_status": "real_adaptive_attack_protocol_coverage_ready" if decision == "PASS" else "real_adaptive_attack_protocol_coverage_blocked",
    }


def build_real_world_attack_records(run_root: str | Path, context: Mapping[str, Any]) -> list[dict[str, Any]]:
    """构建真实平台、屏幕录制和重生成攻击图表的治理记录。"""

    run_root = Path(run_root)
    runtime_records = _read_jsonl(run_root / "records" / "runtime_attack_records.jsonl")
    adaptive_records = _read_jsonl(run_root / "records" / "adaptive_attack_records.jsonl")
    runtime_names = {str(item.get("attack_name") or "") for item in runtime_records if item.get("attack_name")}
    adaptive_names = {
        str(item.get("non_runtime_attack_protocol") or item.get("adaptive_attack_name") or "")
        for item in adaptive_records
        if item.get("non_runtime_attack_protocol") or item.get("adaptive_attack_name")
    }
    records: list[dict[str, Any]] = []
    for protocol in REAL_WORLD_ATTACK_PROTOCOLS:
        if protocol in runtime_names:
            source_kind = "runtime_attack_record"
            status = "proxy_or_runtime_ready"
            source_count = sum(1 for item in runtime_records if item.get("attack_name") == protocol)
        elif protocol in adaptive_names:
            source_kind = "non_runtime_adaptive_protocol_record"
            status = "proxy_or_runtime_ready"
            source_count = sum(1 for item in adaptive_records if (item.get("non_runtime_attack_protocol") or item.get("adaptive_attack_name")) == protocol)
        else:
            source_kind = "not_configured_in_current_protocol"
            status = "governed_not_available_for_current_profile"
            source_count = 0
        claim_status = (
            "real_world_attack_protocol_record_ready"
            if status == "proxy_or_runtime_ready"
            else "real_world_attack_not_available_recorded_not_claim_evidence"
        )
        records.append(with_flow_evidence_protocol_defaults({
            "record_version": "paper_real_world_attack_v1",
            "paper_result_level": context["paper_result_level"],
            "target_fpr": context["target_fpr"],
            "real_world_attack_protocol": protocol,
            "real_world_attack_source_kind": source_kind,
            "real_world_attack_source_record_count": source_count,
            "real_world_attack_status": status,
            "metric_status": "measured_formal" if status == "proxy_or_runtime_ready" else "missing",
            "claim_support_status": claim_status,
        }, trajectory_source_level="real_world_attack_summary_from_protocol_records", claim_support_status=claim_status))
    return records


def audit_real_world_attack_records(records: list[dict[str, Any]], context: Mapping[str, Any]) -> dict[str, Any]:
    """审计真实世界攻击图表输入是否至少覆盖可运行代理。"""

    ready_count = sum(1 for item in records if item.get("real_world_attack_status") == "proxy_or_runtime_ready")
    decision = "PASS" if ready_count > 0 else "FAIL"
    return {
        "stage_id": "real_world_attack_builder",
        "paper_result_level": context["paper_result_level"],
        "target_fpr": context["target_fpr"],
        "real_world_attack_decision": decision,
        "real_world_attack_record_count": len(records),
        "real_world_attack_ready_count": ready_count,
        "real_world_attack_missing_or_not_configured_protocols": [
            str(item.get("real_world_attack_protocol"))
            for item in records
            if item.get("real_world_attack_status") != "proxy_or_runtime_ready"
        ],
        "claim_support_status": "real_world_attack_proxy_coverage_ready" if decision == "PASS" else "real_world_attack_coverage_blocked",
    }


def _write_figure(
    path: Path,
    *,
    figure_id: str,
    title: str,
    rows: list[dict[str, Any]],
    x: str,
    y: str,
    color: str | None,
    context: Mapping[str, Any],
    source_paths: list[str],
) -> dict[str, Any]:
    """写出统一的轻量 figure manifest。"""

    encoding: dict[str, str] = {"x": x, "y": y}
    if color:
        encoding["color"] = color
    manifest = {
        "artifact_type": "figure_manifest",
        "figure_id": figure_id,
        "figure_title": title,
        "paper_result_level": context["paper_result_level"],
        "target_fpr": context["target_fpr"],
        "source_artifact_paths": source_paths,
        "encoding": encoding,
        "figure_rows": rows,
        "claim_support_status": "paper_figure_manifest_from_governed_records",
    }
    write_json(path, manifest)
    return manifest


def _write_markdown_report(path: Path, title: str, audit: Mapping[str, Any]) -> None:
    """把 decision 摘要写成简短 Markdown 报告。"""

    lines = [f"# {title}", "", "该报告由 governed records 自动派生, 不在报告中手工补造实验结果。", ""]
    for key, value in audit.items():
        if isinstance(value, (list, tuple)):
            rendered = ", ".join(str(item) for item in value) if value else "none"
        else:
            rendered = str(value)
        lines.append(f"- {key}: {rendered}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _present_relpaths(run_root: Path) -> list[str]:
    """返回已经成功写出的论文结果 artifact 相对路径。"""

    return [relpath for relpath in PAPER_RESULT_ARTIFACT_RELPATHS if (run_root / relpath).exists()]


def run_paper_result_artifact_builders(
    run_root: str | Path,
    config_path: str | Path = DEFAULT_PROTOCOL_CONFIG,
) -> dict[str, Any]:
    """运行全部论文结果补充产物构建器。"""

    run_root = Path(run_root)
    context = _load_protocol_context(config_path)

    quality_records = build_video_quality_metric_records(run_root, context)
    quality_audit = audit_video_quality_metric_records(quality_records, context)
    write_jsonl(run_root / "records" / "video_quality_metric_records.jsonl", quality_records)
    write_csv(run_root / "tables" / "video_quality_metric_table.csv", quality_records)
    write_json(run_root / "artifacts" / "video_quality_metric_decision.json", quality_audit)
    _write_markdown_report(run_root / "reports" / "video_quality_metric_report.md", "Video Quality Metric Report", quality_audit)

    low_fpr_records = build_low_fpr_curve_records(run_root, context)
    low_fpr_audit = audit_low_fpr_curve_records(low_fpr_records, context)
    write_jsonl(run_root / "records" / "low_fpr_curve_records.jsonl", low_fpr_records)
    write_csv(run_root / "tables" / "low_fpr_curve_table.csv", low_fpr_records)
    write_json(run_root / "artifacts" / "low_fpr_curve_decision.json", low_fpr_audit)
    _write_markdown_report(run_root / "reports" / "low_fpr_curve_report.md", "Low FPR Curve Report", low_fpr_audit)

    efficiency_records = build_efficiency_metric_records(run_root, context)
    efficiency_audit = audit_efficiency_metric_records(efficiency_records, context)
    write_jsonl(run_root / "records" / "efficiency_metric_records.jsonl", efficiency_records)
    write_csv(run_root / "tables" / "efficiency_metric_table.csv", efficiency_records)
    write_json(run_root / "artifacts" / "efficiency_metric_decision.json", efficiency_audit)
    _write_markdown_report(run_root / "reports" / "efficiency_metric_report.md", "Efficiency Metric Report", efficiency_audit)

    adaptive_records = build_real_adaptive_attack_records(run_root, context)
    adaptive_audit = audit_real_adaptive_attack_records(adaptive_records, context)
    write_jsonl(run_root / "records" / "real_adaptive_attack_records.jsonl", adaptive_records)
    write_csv(run_root / "tables" / "real_adaptive_attack_table.csv", adaptive_records)
    write_json(run_root / "artifacts" / "real_adaptive_attack_decision.json", adaptive_audit)
    _write_markdown_report(run_root / "reports" / "real_adaptive_attack_report.md", "Real Adaptive Attack Report", adaptive_audit)

    real_world_records = build_real_world_attack_records(run_root, context)
    real_world_audit = audit_real_world_attack_records(real_world_records, context)
    write_jsonl(run_root / "records" / "real_world_attack_records.jsonl", real_world_records)
    write_csv(run_root / "tables" / "real_world_attack_table.csv", real_world_records)
    write_json(run_root / "artifacts" / "real_world_attack_decision.json", real_world_audit)
    _write_markdown_report(run_root / "reports" / "real_world_attack_report.md", "Real World Attack Report", real_world_audit)

    _write_figure(
        run_root / "figures" / "video_quality_robustness_tradeoff_figure.json",
        figure_id="video_quality_robustness_tradeoff_figure",
        title="Video quality versus robustness trade-off",
        rows=[
            {
                "quality_metric_scope": item.get("quality_metric_scope"),
                "mean_semantic_consistency_score": item.get("mean_semantic_consistency_score"),
                "formal_metric_ready_count": item.get("formal_metric_ready_count"),
            }
            for item in quality_records
        ],
        x="mean_semantic_consistency_score",
        y="formal_metric_ready_count",
        color="quality_metric_scope",
        context=context,
        source_paths=["records/video_quality_metric_records.jsonl", "records/formal_quality_motion_semantic_records.jsonl"],
    )
    _write_figure(
        run_root / "figures" / "low_fpr_curve_figure.json",
        figure_id="low_fpr_curve_figure",
        title="TPR at calibrated FPR levels",
        rows=low_fpr_records,
        x="curve_point_fpr_level",
        y="tpr_at_target_fpr",
        color="method_id",
        context=context,
        source_paths=["records/low_fpr_curve_records.jsonl", "records/fair_detection_calibration_records.jsonl"],
    )
    _write_figure(
        run_root / "figures" / "efficiency_comparison_figure.json",
        figure_id="efficiency_comparison_figure",
        title="Workflow stage runtime efficiency",
        rows=efficiency_records,
        x="stage_name",
        y="stage_elapsed_min",
        color="efficiency_metric_scope",
        context=context,
        source_paths=["records/efficiency_metric_records.jsonl", "records/notebook_stage_timing_records.jsonl"],
    )
    _write_figure(
        run_root / "figures" / "real_adaptive_attack_robustness_figure.json",
        figure_id="real_adaptive_attack_robustness_figure",
        title="Adaptive attack protocol coverage",
        rows=adaptive_records,
        x="non_runtime_attack_protocol",
        y="adaptive_attack_record_count",
        color="real_adaptive_attack_status",
        context=context,
        source_paths=["records/real_adaptive_attack_records.jsonl", "records/adaptive_attack_records.jsonl"],
    )
    _write_figure(
        run_root / "figures" / "real_world_attack_robustness_figure.json",
        figure_id="real_world_attack_robustness_figure",
        title="Real-world attack protocol coverage",
        rows=real_world_records,
        x="real_world_attack_protocol",
        y="real_world_attack_source_record_count",
        color="real_world_attack_status",
        context=context,
        source_paths=["records/real_world_attack_records.jsonl", "records/runtime_attack_records.jsonl", "records/adaptive_attack_records.jsonl"],
    )

    component_decisions = {
        "video_quality_metric_decision": quality_audit["video_quality_metric_decision"],
        "efficiency_metric_decision": efficiency_audit["efficiency_metric_decision"],
        "low_fpr_curve_decision": low_fpr_audit["low_fpr_curve_decision"],
        "real_adaptive_attack_decision": adaptive_audit["real_adaptive_attack_decision"],
        "real_world_attack_decision": real_world_audit["real_world_attack_decision"],
    }
    missing_relpaths = [
        relpath for relpath in PAPER_RESULT_ARTIFACT_RELPATHS
        if relpath not in {
            "artifacts/paper_result_artifact_skeleton_decision.json",
            "reports/paper_result_artifact_skeleton_report.md",
        }
        and not (run_root / relpath).exists()
    ]
    skeleton_ready = all(value == "PASS" for value in component_decisions.values()) and not missing_relpaths
    skeleton = {
        "stage_id": "paper_result_artifact_skeleton",
        "paper_result_level": context["paper_result_level"],
        "target_fpr": context["target_fpr"],
        "paper_result_artifact_skeleton_decision": "PASS" if skeleton_ready else "FAIL",
        "component_decisions": component_decisions,
        "present_artifact_count": len(_present_relpaths(run_root)),
        "required_artifact_count": len(PAPER_RESULT_ARTIFACT_RELPATHS),
        "missing_artifact_relpaths": missing_relpaths,
        "claim_support_status": "paper_result_artifact_skeleton_ready" if skeleton_ready else "paper_result_artifact_skeleton_blocked",
    }
    write_json(run_root / "artifacts" / "paper_result_artifact_skeleton_decision.json", skeleton)
    _write_markdown_report(run_root / "reports" / "paper_result_artifact_skeleton_report.md", "Paper Result Artifact Skeleton Report", skeleton)
    return skeleton


def main() -> None:
    parser = argparse.ArgumentParser(description="构建论文级结果图表和补充实验产物。")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--config-path", default=DEFAULT_PROTOCOL_CONFIG)
    args = parser.parse_args()
    payload = run_paper_result_artifact_builders(args.run_root, args.config_path)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
