"""论文级结果图表与补充实验产物的统一构建器。

该模块的作用是把 probe_paper、pilot_paper 和 full_paper 共享的论文产物
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

from evaluation.attacks.video_runtime_attack_protocol import (
    load_protocol_config_with_shared_attack_protocol,
    required_non_runtime_attack_protocols_from_config,
    required_runtime_attack_names_from_config,
    target_fpr_levels_from_config,
)
from evaluation.metrics.video_file_metrics import compute_paired_video_quality_metrics
from evaluation.metrics.semantic_video_metrics import (
    DEFAULT_CLIP_MODEL_ID,
    DEFAULT_SEMANTIC_THRESHOLD,
    compute_clip_text_video_similarity,
)
from evaluation.protocol.flow_evidence_fields import with_flow_evidence_protocol_defaults
from evaluation.protocol.record_writer import write_json, write_jsonl
from evaluation.protocol.table_builder import write_csv
from external_baseline.runtime_trace_io import (
    NATIVE_GENERATION_COMPARISON_DESIGN,
    SAME_SOURCE_POSTHOC_COMPARISON_DESIGN,
    baseline_comparison_design,
)


DEFAULT_PROTOCOL_CONFIG = "configs/protocol/probe_paper_generative_probe.json"
SSTW_METHOD_ID = "sstw_key_conditioned_flow_trajectory"
DEFAULT_TARGET_FPR_LEVELS = (0.1, 0.01, 0.001)
REAL_WORLD_ATTACK_PROTOCOLS = (
    "platform_transcode_runtime",
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
NATIVE_GENERATION_MAXIMUM_MEAN_SEMANTIC_DEGRADATION = 0.05

COMPLETE_MECHANISM_ARTIFACT_RELPATHS = (
    "records/trajectory_sketch_records.jsonl",
    "records/formal_flow_evidence_records.jsonl",
    "records/paired_path_evidence_gain_records.jsonl",
    "records/paired_velocity_causal_evidence_records.jsonl",
    "records/wrong_key_replay_records.jsonl",
    "records/heldout_posterior_calibration_records.jsonl",
    "records/formal_adaptive_attack_query_budget_checkpoint_records.jsonl",
    "thresholds/formal_flow_detector_thresholds.jsonl",
    "thresholds/replay_gaussian_likelihood_calibrations.jsonl",
    "tables/formal_flow_detection_table.csv",
    "tables/paired_path_evidence_gain_table.csv",
    "tables/paired_velocity_causal_evidence_table.csv",
    "tables/heldout_posterior_calibration_table.csv",
    "tables/formal_adaptive_attack_query_budget_checkpoint_table.csv",
    "tables/cross_model_generalization_table.csv",
    "artifacts/formal_flow_evidence_decision.json",
    "artifacts/three_layer_mechanism_evidence_decision.json",
    "artifacts/cross_model_generalization_decision.json",
    "artifacts/replay_and_sketch_gate_decision.json",
    "artifacts/heldout_posterior_calibration_decision.json",
    "artifacts/complete_paper_mechanism_claim_decision.json",
    "reports/formal_flow_evidence_report.md",
    "reports/replay_and_sketch_gate_report.md",
    "reports/heldout_posterior_calibration_report.md",
    "reports/complete_paper_mechanism_claim_report.md",
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
    config = load_protocol_config_with_shared_attack_protocol(path)
    if "target_fpr" not in config:
        raise KeyError(f"protocol config 缺少 target_fpr: {path}")
    return {
        "paper_result_level": str(config.get("paper_result_level") or "probe_paper"),
        "target_fpr": float(config["target_fpr"]),
        "protocol_config_path": str(path),
        "required_runtime_attack_names": list(required_runtime_attack_names_from_config(config)),
        "required_non_runtime_attack_protocols": list(required_non_runtime_attack_protocols_from_config(config)),
        "target_fpr_levels": list(target_fpr_levels_from_config(config)),
        "claim_support_status": str(
            config.get("claim_support_status") or "paper_profile_artifact_skeleton_not_claim_evidence"
        ),
        "require_complete_paper_mechanism_contract": bool(
            config.get("require_complete_paper_mechanism_contract", False)
        ),
        "require_baseline_matched_video_quality_metrics": bool(
            config.get("require_baseline_matched_video_quality_metrics", False)
        ),
        "required_modern_external_baseline_adapter_names": [
            str(value)
            for value in config.get(
                "required_modern_external_baseline_adapter_names",
                [],
            )
            if str(value)
        ],
        "video_quality_comparison_protocol": str(
            config.get("video_quality_comparison_protocol")
            or "same_clean_reference_to_method_own_watermarked_source_paired_metrics"
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


def _resolve_quality_video_path(
    run_root: Path,
    raw_path: Any,
    *,
    allow_run_video_basename_fallback: bool = False,
) -> Path | None:
    """解析质量计算使用的视频路径, 不允许用不存在的路径继续聚合。

    通用工程写法是优先使用 record 中的原始路径。项目特定写法是兼容 Colab
    绝对路径同步到服务器后仅保留在本地阶段工作区的情况。回退只按明确的
    `videos/`、`artifacts/` 或 official bundle 相对后缀执行, 不递归搜索并误配同名视频。
    """

    text = str(raw_path or "").strip()
    if not text:
        return None
    direct = Path(text)
    candidates = [direct]
    if not direct.is_absolute():
        candidates.append(run_root / direct)
    if allow_run_video_basename_fallback:
        candidates.append(run_root / "videos" / direct.name)
    normalized_parts = list(direct.parts)
    for anchor in (
        "videos",
        "artifacts",
        "external_baseline_official_result_bundles",
    ):
        matching_indices = [
            index
            for index, part in enumerate(normalized_parts)
            if str(part).lower() == anchor.lower()
        ]
        if not matching_indices:
            continue
        suffix = Path(*normalized_parts[matching_indices[-1]:])
        if anchor in {"videos", "artifacts"}:
            candidates.append(run_root / suffix)
        else:
            # 阶段 zip 会把 official bundle 恢复到本地项目工作区, 它与
            # `runs/<run_id>` 是同级树, 因此只尝试 run root 的祖先目录。
            candidates.extend(parent / suffix for parent in run_root.parents)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _candidate_project_roots(run_root: Path) -> list[Path]:
    """从运行目录推断 prompt suite 可能所在的项目根目录。"""

    candidates = [*run_root.parents, Path("/content/drive/MyDrive/SSTW"), Path(r"G:\我的云端硬盘\SSTW")]
    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def _prompt_text_by_id(run_root: Path) -> dict[str, str]:
    """读取正式 prompt 文本, 供 native-generation 语义质量评测使用。"""

    manifest = _read_json(run_root / "artifacts" / "generation_manifest.json")
    candidates = [
        Path(str(path))
        for path in manifest.get("input_paths", [])
        if str(path or "").strip()
    ]
    for project_root in _candidate_project_roots(run_root):
        candidates.append(
            project_root
            / "datasets"
            / "generative_video_prompt_suite"
            / "prompt_seed_suite.json"
        )
    prompt_suite = next((path for path in candidates if path.is_file()), None)
    if prompt_suite is None:
        return {}
    payload = _read_json(prompt_suite)
    return {
        str(row.get("prompt_id")): str(row.get("prompt_text"))
        for row in payload.get("prompts", [])
        if isinstance(row, Mapping)
        and str(row.get("prompt_id") or "")
        and str(row.get("prompt_text") or "")
    }


def _native_generation_semantic_quality_metrics(
    *,
    prompt_id: str,
    prompt_text: str,
    clean_source_path: Path,
    watermarked_source_path: Path,
) -> dict[str, Any]:
    """用真实 CLIP 文本-视频分数评价 native-generation baseline。

    native-generation 方法没有与 SSTW 共用的逐像素源视频, 因而禁止计算跨生成器
    PSNR / SSIM。这里分别评价其官方 clean 与 watermarked 视频对同一 prompt 的
    语义一致性, 并记录均值差。该指标不冒充逐像素失真或 FVD。
    """

    clean = compute_clip_text_video_similarity(
        clean_source_path,
        prompt_text,
        model_id=DEFAULT_CLIP_MODEL_ID,
    )
    watermarked = compute_clip_text_video_similarity(
        watermarked_source_path,
        prompt_text,
        model_id=DEFAULT_CLIP_MODEL_ID,
    )
    clean_score = _safe_float(clean.get("semantic_consistency_score"))
    watermarked_score = _safe_float(watermarked.get("semantic_consistency_score"))
    ready = (
        clean.get("semantic_metric_status") == "ready"
        and watermarked.get("semantic_metric_status") == "ready"
        and clean_score is not None
        and watermarked_score is not None
    )
    return {
        "prompt_id": prompt_id,
        "native_generation_quality_status": "ready" if ready else "blocked",
        "native_generation_quality_failure_reason": (
            "none"
            if ready
            else ";".join(
                str(value)
                for value in (
                    clean.get("semantic_metric_failure_reason"),
                    watermarked.get("semantic_metric_failure_reason"),
                )
                if value not in {None, "", "none"}
            )
            or "semantic_metric_not_ready"
        ),
        "native_generation_semantic_metric_name": "clip_text_video_similarity",
        "native_generation_semantic_model_id": DEFAULT_CLIP_MODEL_ID,
        "native_generation_clean_semantic_consistency_score": clean_score,
        "native_generation_watermarked_semantic_consistency_score": watermarked_score,
        "native_generation_semantic_consistency_delta": (
            round(watermarked_score - clean_score, 8)
            if clean_score is not None and watermarked_score is not None
            else None
        ),
        "native_generation_clean_source_video_path": str(clean_source_path),
        "native_generation_watermarked_source_video_path": str(watermarked_source_path),
        "paired_video_quality_status": "not_applicable_native_generation",
        "paired_watermark_psnr": None,
        "paired_watermark_ssim": None,
        "paired_temporal_delta_error": None,
    }


def _fair_quality_robustness_by_method(
    run_root: Path,
    target_fpr: float,
) -> dict[str, dict[str, Any]]:
    """读取同一 target FPR 下每个方法唯一的公平鲁棒性记录。"""

    by_method: dict[str, list[dict[str, Any]]] = {}
    for record in _fair_rows_for_current_target(run_root, target_fpr):
        method_id = str(record.get("method_id") or "")
        if method_id:
            by_method.setdefault(method_id, []).append(record)
    return {
        method_id: rows[0]
        for method_id, rows in by_method.items()
        if len(rows) == 1
    }


def _quality_record_payload(
    *,
    method_id: str,
    method_role: str,
    source_kind: str,
    source_record_count: int,
    unit_metrics: list[Mapping[str, Any]],
    failure_reasons: list[str],
    robustness_record: Mapping[str, Any] | None,
    context: Mapping[str, Any],
    quality_protocol: str,
) -> dict[str, Any]:
    """按方法适用的质量协议聚合逐视频结果。"""

    if quality_protocol == "paired_same_source_distortion":
        ready_units = [
            item
            for item in unit_metrics
            if item.get("paired_video_quality_status") == "ready"
            and all(
                _safe_float(item.get(field_name)) is not None
                for field_name in (
                    "paired_watermark_psnr",
                    "paired_watermark_ssim",
                    "paired_temporal_delta_error",
                )
            )
        ]
    elif quality_protocol == "native_generation_semantic_quality":
        ready_units = [
            item
            for item in unit_metrics
            if item.get("native_generation_quality_status") == "ready"
            and all(
                _safe_float(item.get(field_name)) is not None
                for field_name in (
                    "native_generation_clean_semantic_consistency_score",
                    "native_generation_watermarked_semantic_consistency_score",
                    "native_generation_semantic_consistency_delta",
                )
            )
            and item.get("paired_video_quality_status")
            == "not_applicable_native_generation"
        ]
    else:
        raise ValueError(f"unsupported_video_quality_protocol:{quality_protocol}")
    robustness_ready = bool(
        robustness_record
        and robustness_record.get("fair_comparison_status") == "ready"
        and all(
            _safe_float(robustness_record.get(field_name)) is not None
            for field_name in (
                "tpr_at_target_fpr",
                "tpr_ci_lower",
                "tpr_ci_upper",
            )
        )
    )
    normalized_failures = sorted({str(value) for value in failure_reasons if str(value)})
    mean_clean_semantic = _mean_optional(
        item.get("native_generation_clean_semantic_consistency_score")
        for item in ready_units
    )
    mean_watermarked_semantic = _mean_optional(
        item.get("native_generation_watermarked_semantic_consistency_score")
        for item in ready_units
    )
    mean_semantic_delta = _mean_optional(
        item.get("native_generation_semantic_consistency_delta")
        for item in ready_units
    )
    if quality_protocol == "native_generation_semantic_quality" and ready_units:
        if (
            mean_clean_semantic is None
            or mean_clean_semantic < DEFAULT_SEMANTIC_THRESHOLD
        ):
            normalized_failures.append("native_generation_clean_semantic_quality_below_minimum")
        if (
            mean_watermarked_semantic is None
            or mean_watermarked_semantic < DEFAULT_SEMANTIC_THRESHOLD
        ):
            normalized_failures.append("native_generation_watermarked_semantic_quality_below_minimum")
        if (
            mean_semantic_delta is None
            or mean_semantic_delta
            < -NATIVE_GENERATION_MAXIMUM_MEAN_SEMANTIC_DEGRADATION
        ):
            normalized_failures.append("native_generation_mean_semantic_degradation_above_maximum")
    if not robustness_ready:
        normalized_failures.append("fair_robustness_record_missing_or_not_unique")
    status = (
        "ready"
        if source_record_count > 0
        and unit_metrics
        and len(ready_units) == len(unit_metrics)
        and not normalized_failures
        and robustness_ready
        else "blocked"
    )
    claim_status = _claim_status_for_current_profile(context, status)
    return with_flow_evidence_protocol_defaults({
        "record_version": "paper_video_quality_metric_v2",
        "paper_result_level": context["paper_result_level"],
        "target_fpr": context["target_fpr"],
        "method_id": method_id,
        "method_role": method_role,
        "quality_metric_scope": "method_watermark_embedding_quality",
        "quality_metric_source_kind": source_kind,
        "quality_metric_source_record_count": source_record_count,
        "quality_evaluation_unit_count": len(unit_metrics),
        "quality_evaluation_ready_count": len(ready_units),
        "quality_evaluation_blocked_count": len(unit_metrics) - len(ready_units),
        "paired_quality_unit_count": (
            len(unit_metrics) if quality_protocol == "paired_same_source_distortion" else 0
        ),
        "paired_quality_ready_count": (
            len(ready_units) if quality_protocol == "paired_same_source_distortion" else 0
        ),
        "paired_quality_blocked_count": (
            len(unit_metrics) - len(ready_units)
            if quality_protocol == "paired_same_source_distortion"
            else 0
        ),
        "native_generation_semantic_quality_unit_count": (
            len(unit_metrics) if quality_protocol == "native_generation_semantic_quality" else 0
        ),
        "native_generation_semantic_quality_ready_count": (
            len(ready_units) if quality_protocol == "native_generation_semantic_quality" else 0
        ),
        "mean_paired_watermark_psnr": _mean_optional(
            item.get("paired_watermark_psnr") for item in ready_units
        ),
        "mean_paired_watermark_ssim": _mean_optional(
            item.get("paired_watermark_ssim") for item in ready_units
        ),
        "mean_paired_temporal_delta_error": _mean_optional(
            item.get("paired_temporal_delta_error") for item in ready_units
        ),
        "mean_native_generation_clean_semantic_consistency": mean_clean_semantic,
        "mean_native_generation_watermarked_semantic_consistency": mean_watermarked_semantic,
        "mean_native_generation_semantic_consistency_delta": mean_semantic_delta,
        "native_generation_semantic_consistency_minimum": (
            DEFAULT_SEMANTIC_THRESHOLD
            if quality_protocol == "native_generation_semantic_quality"
            else None
        ),
        "native_generation_maximum_mean_semantic_degradation": (
            NATIVE_GENERATION_MAXIMUM_MEAN_SEMANTIC_DEGRADATION
            if quality_protocol == "native_generation_semantic_quality"
            else None
        ),
        "robustness_tpr_at_target_fpr": (
            _safe_float(robustness_record.get("tpr_at_target_fpr"))
            if robustness_record
            else None
        ),
        "robustness_tpr_ci_lower": (
            _safe_float(robustness_record.get("tpr_ci_lower"))
            if robustness_record
            else None
        ),
        "robustness_tpr_ci_upper": (
            _safe_float(robustness_record.get("tpr_ci_upper"))
            if robustness_record
            else None
        ),
        "video_quality_comparison_protocol": quality_protocol,
        "cross_generator_pixel_metric_prohibited": (
            quality_protocol == "native_generation_semantic_quality"
        ),
        "quality_metric_failure_reasons": sorted(set(normalized_failures)),
        "video_quality_metric_status": status,
        "metric_status": "measured_formal" if status == "ready" else "missing",
        "claim_support_status": claim_status,
    }, trajectory_source_level="paper_method_appropriate_quality_from_method_own_videos", claim_support_status=claim_status)


def _sstw_paired_quality_record(
    formal_records: list[dict[str, Any]],
    robustness_record: Mapping[str, Any] | None,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    """聚合 SSTW 完整方法已经计算完成的 clean-reference 配对质量。"""

    scoped = [
        record
        for record in formal_records
        if record.get("method_variant") == "sstw_full_method"
        and record.get("paired_video_quality_required") is True
    ]
    unit_metrics: list[dict[str, Any]] = []
    failures: list[str] = []
    for record in scoped:
        unit_metrics.append({
            field_name: record.get(field_name)
            for field_name in (
                "paired_video_quality_status",
                "paired_watermark_psnr",
                "paired_watermark_ssim",
                "paired_temporal_delta_error",
            )
        })
        if record.get("formal_metric_result_used_for_claim") is not True:
            failures.append("sstw_formal_quality_motion_semantic_record_not_ready")
        if record.get("paired_video_quality_status") != "ready":
            failures.append("sstw_paired_video_quality_not_ready")
    if not scoped:
        failures.append("sstw_paired_quality_records_missing")
    return _quality_record_payload(
        method_id=SSTW_METHOD_ID,
        method_role="proposed_method",
        source_kind="sstw_formal_quality_motion_semantic_records",
        source_record_count=len(scoped),
        unit_metrics=unit_metrics,
        failure_reasons=failures,
        robustness_record=robustness_record,
        context=context,
        quality_protocol="paired_same_source_distortion",
    )


def _baseline_paired_quality_record(
    run_root: Path,
    baseline_id: str,
    score_records: list[dict[str, Any]],
    robustness_record: Mapping[str, Any] | None,
    context: Mapping[str, Any],
    prompt_text_by_id: Mapping[str, str],
) -> dict[str, Any]:
    """按 baseline 生成设计选择适用质量协议并去除 attack 重复。"""

    scoped = [
        record
        for record in score_records
        if record.get("external_baseline_name") == baseline_id
        and record.get("metric_status") == "measured_formal"
        and record.get("external_baseline_result_used_for_claim") is True
    ]
    failures: list[str] = []
    comparison_design = baseline_comparison_design(baseline_id)
    quality_protocol = (
        "paired_same_source_distortion"
        if comparison_design == SAME_SOURCE_POSTHOC_COMPARISON_DESIGN
        else "native_generation_semantic_quality"
    )
    unique_pairs: dict[tuple[str, str, str], tuple[Path, Path]] = {}
    identities_to_paths: dict[tuple[str, str, str], set[tuple[str, str]]] = {}
    for record in scoped:
        observed_design = str(record.get("external_baseline_comparison_design") or "")
        observed_quality_protocol = str(
            record.get("external_baseline_quality_comparison_protocol") or ""
        )
        reference_status = str(record.get("baseline_clean_reference_status") or "")
        input_policy = str(record.get("baseline_input_source_policy") or "")
        reference_raw = (
            record.get("baseline_quality_reference_video_path")
            or record.get("external_baseline_clean_source_video_path")
            or record.get("baseline_clean_reference_video_path")
        )
        source_raw = record.get("external_baseline_source_video_path")
        reference_path = _resolve_quality_video_path(
            run_root,
            reference_raw,
            allow_run_video_basename_fallback=True,
        )
        source_path = _resolve_quality_video_path(run_root, source_raw)
        if observed_design != comparison_design:
            failures.append("baseline_comparison_design_mismatch")
        if observed_quality_protocol != quality_protocol:
            failures.append("baseline_quality_comparison_protocol_mismatch")
        if comparison_design == SAME_SOURCE_POSTHOC_COMPARISON_DESIGN:
            if reference_status != "same_source_posthoc_clean_reference":
                failures.append("baseline_same_source_clean_reference_status_invalid")
            if input_policy != "baseline_embeds_own_watermark_into_clean_reference":
                failures.append("baseline_same_source_input_policy_invalid")
        else:
            if reference_status != "native_generation_own_model_clean_reference":
                failures.append("baseline_native_generation_clean_reference_status_invalid")
            if input_policy != "baseline_uses_official_native_generation_model":
                failures.append("baseline_native_generation_input_policy_invalid")
        if reference_path is None:
            failures.append("baseline_clean_reference_video_missing")
        if source_path is None:
            failures.append("baseline_watermarked_source_video_missing")
        if reference_path is None or source_path is None:
            continue
        identity = (
            str(
                record.get("external_baseline_generation_model_id")
                or record.get("generation_model_id")
                or ""
            ),
            str(record.get("prompt_id") or ""),
            str(record.get("seed_id") or ""),
        )
        if not all(identity):
            failures.append("baseline_quality_identity_incomplete")
            continue
        path_pair = (str(reference_path.resolve()), str(source_path.resolve()))
        identities_to_paths.setdefault(identity, set()).add(path_pair)
        unique_pairs[identity] = (reference_path, source_path)
    if not scoped:
        failures.append("required_baseline_measured_formal_records_missing")
    if any(len(path_pairs) != 1 for path_pairs in identities_to_paths.values()):
        failures.append("baseline_identity_maps_to_multiple_watermarked_sources")
    unit_metrics: list[dict[str, Any]] = []
    for identity, (reference_path, source_path) in unique_pairs.items():
        if comparison_design == SAME_SOURCE_POSTHOC_COMPARISON_DESIGN:
            metrics = compute_paired_video_quality_metrics(reference_path, source_path)
            if metrics.get("paired_video_quality_status") != "ready":
                failures.append(
                    "baseline_paired_video_quality_"
                    + str(metrics.get("paired_video_quality_status") or "unknown")
                )
        else:
            prompt_id = identity[1]
            prompt_text = str(prompt_text_by_id.get(prompt_id) or "")
            if not prompt_text:
                failures.append("baseline_native_generation_prompt_text_missing")
                continue
            metrics = _native_generation_semantic_quality_metrics(
                prompt_id=prompt_id,
                prompt_text=prompt_text,
                clean_source_path=reference_path,
                watermarked_source_path=source_path,
            )
            if metrics.get("native_generation_quality_status") != "ready":
                failures.append(
                    "baseline_native_generation_semantic_quality_"
                    + str(metrics.get("native_generation_quality_status") or "unknown")
                )
        unit_metrics.append({
            **metrics,
            "quality_identity_generation_model_id": identity[0],
            "quality_identity_prompt_id": identity[1],
            "quality_identity_seed_id": identity[2],
        })
    return _quality_record_payload(
        method_id=baseline_id,
        method_role="modern_external_baseline",
        source_kind=(
            "external_baseline_same_source_posthoc_quality_pairs"
            if comparison_design == SAME_SOURCE_POSTHOC_COMPARISON_DESIGN
            else "external_baseline_native_generation_semantic_quality_units"
        ),
        source_record_count=len(scoped),
        unit_metrics=unit_metrics,
        failure_reasons=failures,
        robustness_record=robustness_record,
        context=context,
        quality_protocol=quality_protocol,
    )


def build_video_quality_metric_records(run_root: str | Path, context: Mapping[str, Any]) -> list[dict[str, Any]]:
    """构建 SSTW 与全部正式 baseline 的同口径配对质量汇总 records。"""

    run_root = Path(run_root)
    formal_records = _read_jsonl(
        run_root / "records" / "formal_quality_motion_semantic_records.jsonl"
    )
    external_records = _read_jsonl(
        run_root / "records" / "external_baseline_score_records.jsonl"
    )
    robustness_by_method = _fair_quality_robustness_by_method(
        run_root,
        float(context["target_fpr"]),
    )
    prompt_texts = _prompt_text_by_id(run_root)
    records = [
        _sstw_paired_quality_record(
            formal_records,
            robustness_by_method.get(SSTW_METHOD_ID),
            context,
        )
    ]
    for baseline_id in context.get("required_modern_external_baseline_adapter_names", []):
        records.append(_baseline_paired_quality_record(
            run_root,
            str(baseline_id),
            external_records,
            robustness_by_method.get(str(baseline_id)),
            context,
            prompt_texts,
        ))
    return records


def audit_video_quality_metric_records(records: list[dict[str, Any]], context: Mapping[str, Any]) -> dict[str, Any]:
    """审计 SSTW 与5个 baseline 是否具备适用于各自生成设计的质量证据。"""

    required_baselines = {
        str(value)
        for value in context.get("required_modern_external_baseline_adapter_names", [])
        if str(value)
    }
    required_methods = {SSTW_METHOD_ID, *required_baselines}
    ready_methods = {
        str(item.get("method_id"))
        for item in records
        if item.get("video_quality_metric_status") == "ready"
    }
    observed_methods = {
        str(item.get("method_id"))
        for item in records
        if item.get("method_id")
    }
    sstw_ready = SSTW_METHOD_ID in ready_methods
    baseline_ready_methods = ready_methods & required_baselines
    baseline_ready = bool(required_baselines) and baseline_ready_methods == required_baselines
    protocol_matches = all(
        item.get("video_quality_comparison_protocol")
        == (
            "paired_same_source_distortion"
            if item.get("method_id") == SSTW_METHOD_ID
            or baseline_comparison_design(str(item.get("method_id") or ""))
            == SAME_SOURCE_POSTHOC_COMPARISON_DESIGN
            else "native_generation_semantic_quality"
        )
        for item in records
        if item.get("method_id")
    )
    decision = (
        "PASS"
        if records
        and observed_methods == required_methods
        and ready_methods == required_methods
        and sstw_ready
        and baseline_ready
        and protocol_matches
        else "FAIL"
    )
    return {
        "stage_id": "video_quality_metric_builder",
        "paper_result_level": context["paper_result_level"],
        "target_fpr": context["target_fpr"],
        "video_quality_comparison_protocol": "method_appropriate_same_source_or_native_generation_quality",
        "video_quality_metric_decision": decision,
        "video_quality_metric_record_count": len(records),
        "video_quality_metric_ready_count": len(ready_methods),
        "video_quality_required_method_ids": sorted(required_methods),
        "video_quality_ready_method_ids": sorted(ready_methods),
        "video_quality_missing_method_ids": sorted(required_methods - ready_methods),
        "sstw_paired_video_quality_ready": sstw_ready,
        "baseline_matched_video_quality_ready": baseline_ready,
        "baseline_matched_video_quality_ready_method_ids": sorted(baseline_ready_methods),
        "baseline_matched_video_quality_missing_method_ids": sorted(
            required_baselines - baseline_ready_methods
        ),
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
        formal_scoped = [
            item for item in scoped
            if item.get("metric_status") == "measured_formal"
            and item.get("adaptive_attack_evidence_level") == "formal_adaptive_attack_execution"
            and item.get("adaptive_attack_execution_granularity")
            == "per_video_frozen_flow_detector_adaptive_execution"
            and item.get("adaptive_robustness_claim_allowed") is True
        ]
        measured_count = len(formal_scoped)
        non_formal_count = len(scoped) - measured_count
        status = "measured_ready" if measured_count and non_formal_count == 0 else ("non_formal_records_blocked" if scoped else "missing")
        claim_status = "real_adaptive_attack_measured_ready" if measured_count else "real_adaptive_attack_governed_protocol_record_only"
        records.append(with_flow_evidence_protocol_defaults({
            "record_version": "paper_real_adaptive_attack_v1",
            "paper_result_level": context["paper_result_level"],
            "target_fpr": context["target_fpr"],
            "non_runtime_attack_protocol": protocol,
            "adaptive_attack_record_count": len(scoped),
            "adaptive_attack_measured_formal_count": measured_count,
            "adaptive_attack_non_formal_record_count": non_formal_count,
            "real_adaptive_attack_status": status,
            "metric_status": "measured_formal" if status == "measured_ready" else "missing",
            "claim_support_status": claim_status,
        }, trajectory_source_level="real_adaptive_attack_summary_from_governed_records", claim_support_status=claim_status))
    return records


def audit_real_adaptive_attack_records(records: list[dict[str, Any]], context: Mapping[str, Any]) -> dict[str, Any]:
    """审计 adaptive 协议是否覆盖配置要求。"""

    blocked = [str(item.get("non_runtime_attack_protocol")) for item in records if item.get("real_adaptive_attack_status") != "measured_ready"]
    decision = "PASS" if records and not blocked else "FAIL"
    return {
        "stage_id": "real_adaptive_attack_builder",
        "paper_result_level": context["paper_result_level"],
        "target_fpr": context["target_fpr"],
        "real_adaptive_attack_decision": decision,
        "real_adaptive_attack_record_count": len(records),
        "real_adaptive_attack_missing_protocols": blocked,
        "real_adaptive_attack_measured_protocol_count": sum(1 for item in records if item.get("real_adaptive_attack_status") == "measured_ready"),
        "real_adaptive_attack_non_formal_protocol_count": sum(1 for item in records if item.get("real_adaptive_attack_status") == "non_formal_records_blocked"),
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
            scoped_runtime = [item for item in runtime_records if item.get("attack_name") == protocol]
            source_count = len(scoped_runtime)
            formal_count = sum(1 for item in scoped_runtime if item.get("runtime_attack_formal_evidence_level") == "formal_runtime_video_transform")
            status = "formal_runtime_ready" if formal_count == source_count and source_count else "non_formal_records_blocked"
        elif protocol in adaptive_names:
            source_kind = "non_runtime_adaptive_protocol_record"
            scoped_adaptive = [
                item for item in adaptive_records
                if (item.get("non_runtime_attack_protocol") or item.get("adaptive_attack_name")) == protocol
            ]
            source_count = len(scoped_adaptive)
            formal_count = sum(
                1
                for item in scoped_adaptive
                if item.get("adaptive_attack_evidence_level")
                == "formal_adaptive_attack_execution"
                and item.get("adaptive_attack_execution_granularity")
                == "per_video_frozen_flow_detector_adaptive_execution"
            )
            status = "formal_non_runtime_ready" if formal_count == source_count and source_count else "non_formal_records_blocked"
        else:
            source_kind = "not_configured_in_current_protocol"
            status = "governed_not_available_for_current_profile"
            source_count = 0
        claim_status = (
            "real_world_attack_protocol_record_ready"
            if status in {"formal_runtime_ready", "formal_non_runtime_ready"}
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
            "metric_status": "measured_formal" if status in {"formal_runtime_ready", "formal_non_runtime_ready"} else "missing",
            "claim_support_status": claim_status,
        }, trajectory_source_level="real_world_attack_summary_from_protocol_records", claim_support_status=claim_status))
    return records


def audit_real_world_attack_records(records: list[dict[str, Any]], context: Mapping[str, Any]) -> dict[str, Any]:
    """审计真实世界攻击图表输入是否至少覆盖正式攻击记录。"""

    ready_count = sum(1 for item in records if item.get("real_world_attack_status") in {"formal_runtime_ready", "formal_non_runtime_ready"})
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
            if item.get("real_world_attack_status") not in {"formal_runtime_ready", "formal_non_runtime_ready"}
        ],
        "claim_support_status": "real_world_attack_formal_coverage_ready" if decision == "PASS" else "real_world_attack_coverage_blocked",
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
    alternate_encodings: list[Mapping[str, str]] | None = None,
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
        "alternate_encodings": [dict(item) for item in (alternate_encodings or [])],
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


def _present_relpaths(run_root: Path, required_relpaths: Iterable[str] = PAPER_RESULT_ARTIFACT_RELPATHS) -> list[str]:
    """返回已经成功写出的论文结果 artifact 相对路径。"""

    return [relpath for relpath in required_relpaths if (run_root / relpath).exists()]


def run_video_quality_metric_artifact_builder(
    run_root: str | Path,
    config_path: str | Path = DEFAULT_PROTOCOL_CONFIG,
    *,
    reuse_ready_records: bool = False,
) -> dict[str, Any]:
    """构建跨方法配对质量产物, 并返回 fail-closed decision。

    正式 workflow 在 baseline 证据包仍处于本地工作区时执行本函数。后续论文
    后处理阶段只复用已经通过审计的 governed records, 从而不需要再次恢复包含
    全部攻击视频的5个大型 baseline 包。
    """

    root = Path(run_root)
    context = _load_protocol_context(config_path)
    records_path = root / "records" / "video_quality_metric_records.jsonl"
    existing_records = _read_jsonl(records_path) if reuse_ready_records else []
    existing_audit = (
        audit_video_quality_metric_records(existing_records, context)
        if existing_records
        else {}
    )
    if existing_audit.get("video_quality_metric_decision") == "PASS":
        quality_records = existing_records
        quality_audit = existing_audit
    else:
        quality_records = build_video_quality_metric_records(root, context)
        quality_audit = audit_video_quality_metric_records(quality_records, context)
    write_jsonl(records_path, quality_records)
    write_csv(root / "tables" / "video_quality_metric_table.csv", quality_records)
    write_json(root / "artifacts" / "video_quality_metric_decision.json", quality_audit)
    _write_markdown_report(
        root / "reports" / "video_quality_metric_report.md",
        "Video Quality Metric Report",
        quality_audit,
    )
    _write_figure(
        root / "figures" / "video_quality_robustness_tradeoff_figure.json",
        figure_id="video_quality_robustness_tradeoff_figure",
        title="Paired video quality versus fixed-FPR robustness trade-off",
        rows=[
            {
                "method_id": item.get("method_id"),
                "method_role": item.get("method_role"),
                "mean_paired_watermark_psnr": item.get("mean_paired_watermark_psnr"),
                "mean_paired_watermark_ssim": item.get("mean_paired_watermark_ssim"),
                "mean_paired_temporal_delta_error": item.get("mean_paired_temporal_delta_error"),
                "mean_native_generation_clean_semantic_consistency": item.get(
                    "mean_native_generation_clean_semantic_consistency"
                ),
                "mean_native_generation_watermarked_semantic_consistency": item.get(
                    "mean_native_generation_watermarked_semantic_consistency"
                ),
                "mean_native_generation_semantic_consistency_delta": item.get(
                    "mean_native_generation_semantic_consistency_delta"
                ),
                "video_quality_comparison_protocol": item.get(
                    "video_quality_comparison_protocol"
                ),
                "robustness_tpr_at_target_fpr": item.get("robustness_tpr_at_target_fpr"),
                "robustness_tpr_ci_lower": item.get("robustness_tpr_ci_lower"),
                "robustness_tpr_ci_upper": item.get("robustness_tpr_ci_upper"),
            }
            for item in quality_records
            if item.get("video_quality_metric_status") == "ready"
        ],
        x="mean_paired_watermark_psnr",
        y="robustness_tpr_at_target_fpr",
        color="method_id",
        context=context,
        source_paths=[
            "records/video_quality_metric_records.jsonl",
            "records/formal_quality_motion_semantic_records.jsonl",
            "records/external_baseline_score_records.jsonl",
            "records/fair_detection_calibration_records.jsonl",
        ],
        alternate_encodings=[
            {
                "panel_id": "paired_ssim_vs_fixed_fpr_robustness",
                "x": "mean_paired_watermark_ssim",
                "y": "robustness_tpr_at_target_fpr",
                "color": "method_id",
                "quality_protocol": "paired_same_source_distortion",
            },
            {
                "panel_id": "native_generation_semantic_quality_vs_fixed_fpr_robustness",
                "x": "mean_native_generation_watermarked_semantic_consistency",
                "y": "robustness_tpr_at_target_fpr",
                "color": "method_id",
                "quality_protocol": "native_generation_semantic_quality",
            },
        ],
    )
    return quality_audit


def run_paper_result_artifact_builders(
    run_root: str | Path,
    config_path: str | Path = DEFAULT_PROTOCOL_CONFIG,
) -> dict[str, Any]:
    """运行全部论文结果补充产物构建器。"""

    run_root = Path(run_root)
    context = _load_protocol_context(config_path)

    quality_audit = run_video_quality_metric_artifact_builder(
        run_root,
        config_path,
        reuse_ready_records=True,
    )

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
    required_artifact_relpaths = PAPER_RESULT_ARTIFACT_RELPATHS + (
        COMPLETE_MECHANISM_ARTIFACT_RELPATHS
        if context["require_complete_paper_mechanism_contract"]
        else ()
    )
    missing_relpaths = [
        relpath for relpath in required_artifact_relpaths
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
        "present_artifact_count": len(_present_relpaths(run_root, required_artifact_relpaths)),
        "required_artifact_count": len(required_artifact_relpaths),
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
    parser.add_argument(
        "--quality-only",
        action="store_true",
        help="仅在 baseline 视频仍可访问的正式比较阶段构建跨方法配对质量产物。",
    )
    args = parser.parse_args()
    if args.quality_only:
        payload = run_video_quality_metric_artifact_builder(
            args.run_root,
            args.config_path,
            reuse_ready_records=False,
        )
    else:
        payload = run_paper_result_artifact_builders(args.run_root, args.config_path)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
