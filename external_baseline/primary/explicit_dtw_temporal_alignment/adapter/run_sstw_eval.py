"""把显式 DTW 时间对齐 control 适配到 SSTW 外部 baseline 比较协议。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from external_baseline.runtime_trace_io import (
    build_comparison_unit_id,
    build_observed_sequence,
    build_reference_sequence,
    comparable_detection_records,
    load_trace_groups,
    safe_float,
)
from main.external_baselines.explicit_dtw_temporal_alignment import compute_dtw_alignment_cost
from main.core.progress import ProgressReporter
from main.protocol.flow_evidence_fields import with_flow_evidence_protocol_defaults


ADAPTER_NAME = "explicit_dtw_temporal_alignment"
ADAPTER_PATH = "external_baseline/primary/explicit_dtw_temporal_alignment/adapter/run_sstw_eval.py"


def adapter_status() -> dict[str, Any]:
    """返回显式 DTW adapter 的受治理状态。

    该状态说明 adapter 已经放在 `external_baseline/` 边界内, 可以从 run_root 的 governed records
    生成 comparison records。它仍然不是现代视频水印 baseline, 因而不能支持正向论文 claim。
    """
    return {
        "external_baseline_runnable_status": "runnable",
        "external_baseline_adapter_status": "ready",
        "external_baseline_adapter_path": ADAPTER_PATH,
        "external_baseline_input_compatibility_status": "runtime_detection_and_callback_trace_ready",
        "external_baseline_output_record_status": "governed_records_written",
        "external_baseline_threshold_policy_compatible": False,
        "external_baseline_attack_manifest_compatible": True,
        "external_baseline_not_run_reason": "none",
        "external_baseline_result_used_for_claim": False,
    }


def _unsupported_record(baseline_record: Mapping[str, Any], detection_record: Mapping[str, Any], reason: str) -> dict[str, Any]:
    """构造单条 adapter 无法评分时的 governed unsupported record。"""
    return with_flow_evidence_protocol_defaults({
        "record_version": "external_baseline_score_v1",
        "external_baseline_score_record_id": build_comparison_unit_id(ADAPTER_NAME, detection_record),
        "external_baseline_name": ADAPTER_NAME,
        "external_baseline_family": baseline_record.get("external_baseline_family"),
        "external_baseline_layer": baseline_record.get("external_baseline_layer"),
        "external_baseline_adapter_path": ADAPTER_PATH,
        "generation_model_id": detection_record.get("generation_model_id"),
        "prompt_id": detection_record.get("prompt_id"),
        "seed_id": detection_record.get("seed_id"),
        "trajectory_trace_id": detection_record.get("trajectory_trace_id"),
        "attack_name": detection_record.get("attack_name"),
        "metric_status": "unsupported",
        "external_baseline_score_status": "unsupported",
        "external_baseline_score_source": "callback_trace_and_runtime_video_metadata",
        "external_baseline_score_failure_reason": reason,
        "external_baseline_reference_sequence_length": 0,
        "external_baseline_observed_sequence_length": 0,
        "external_baseline_distance": None,
        "external_baseline_score": None,
        "baseline_score_margin": None,
        "external_baseline_result_used_for_claim": False,
        "claim_support_status": "external_baseline_proxy_comparison_not_claim_supporting",
    }, trajectory_source_level="external_baseline_adapter_runtime_trace_proxy", claim_support_status="external_baseline_proxy_comparison_not_claim_supporting")


def build_score_records(run_root: str | Path, baseline_record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """从 SSTW runtime records 生成 DTW baseline score records。

    这一 adapter 不读取最终检测分数 `S_final`, 只读取 callback latent 轨迹与 runtime video metadata。
    因此它可以作为显式同步 control 的工程比较结果, 但仍需要真实外部 detector 接入后才能支持论文主 claim。
    """
    detection_records = comparable_detection_records(run_root)
    trace_groups = load_trace_groups(run_root)
    records: list[dict[str, Any]] = []
    progress = ProgressReporter(f"external_baseline_proxy_scoring:{ADAPTER_NAME}", len(detection_records), "runtime_video")
    for index, detection_record in enumerate(detection_records):
        progress.update(
            index + 1,
            f"baseline={ADAPTER_NAME} prompt={detection_record.get('prompt_id')} seed={detection_record.get('seed_id')} attack={detection_record.get('attack_name')}",
        )
        trace_id = str(detection_record.get("trajectory_trace_id") or "")
        reference_sequence = build_reference_sequence(trace_groups.get(trace_id, []))
        observed_sequence = build_observed_sequence(reference_sequence, detection_record)
        if len(reference_sequence) < 2 or len(observed_sequence) < 1:
            records.append(_unsupported_record(baseline_record, detection_record, "missing_comparable_trajectory_sequence"))
            continue
        distance = compute_dtw_alignment_cost(reference_sequence, observed_sequence)
        score = round(1.0 / (1.0 + distance), 6)
        method_score = safe_float(detection_record.get("S_runtime_attack_detection"), 0.0)
        records.append(with_flow_evidence_protocol_defaults({
            "record_version": "external_baseline_score_v1",
            "external_baseline_score_record_id": build_comparison_unit_id(ADAPTER_NAME, detection_record),
            "external_baseline_name": ADAPTER_NAME,
            "external_baseline_family": baseline_record.get("external_baseline_family"),
            "external_baseline_layer": baseline_record.get("external_baseline_layer"),
            "external_baseline_adapter_path": ADAPTER_PATH,
            "generation_model_id": detection_record.get("generation_model_id"),
            "prompt_id": detection_record.get("prompt_id"),
            "seed_id": detection_record.get("seed_id"),
            "trajectory_trace_id": trace_id,
            "attack_name": detection_record.get("attack_name"),
            "metric_status": "measured_proxy",
            "external_baseline_score_status": "measured_proxy",
            "external_baseline_score_source": "callback_trace_and_runtime_video_metadata",
            "external_baseline_score_failure_reason": "none",
            "external_baseline_reference_sequence_length": len(reference_sequence),
            "external_baseline_observed_sequence_length": len(observed_sequence),
            "external_baseline_distance": round(distance, 6),
            "external_baseline_score": score,
            "baseline_score_margin": round(method_score - score, 6),
            "external_baseline_result_used_for_claim": False,
            "claim_support_status": "external_baseline_proxy_comparison_not_claim_supporting",
        }, trajectory_source_level="external_baseline_adapter_runtime_trace_proxy", claim_support_status="external_baseline_proxy_comparison_not_claim_supporting"))
    measured_count = sum(1 for record in records if record.get("external_baseline_score_status") == "measured_proxy")
    progress.finish(f"measured={measured_count} unsupported={len(records) - measured_count}")
    return records
