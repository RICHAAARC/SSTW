"""SSTW 本方法 measured_formal 结果转写器。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from main.core.digest import build_stable_digest
from main.methods.state_space_watermark.video_content_detector import (
    FORMAL_CLEAN_NEGATIVE_EVIDENCE_LEVEL,
    FORMAL_VIDEO_DETECTOR_EVIDENCE_LEVEL,
    FORMAL_VIDEO_DETECTOR_SCORE_SEMANTICS,
)
from main.protocol.flow_evidence_fields import with_flow_evidence_protocol_defaults
from main.protocol.record_writer import write_json, write_jsonl
from main.protocol.table_builder import write_csv


DEFAULT_PROTOCOL_CONFIG = "configs/protocol/probe_paper_generative_probe.json"
SSTW_METHOD_ID = "sstw_key_conditioned_flow_trajectory"
FORMAL_SSTW_DETECTOR_EVIDENCE_LEVELS = {
    FORMAL_VIDEO_DETECTOR_EVIDENCE_LEVEL,
    "attacked_video_wan_vae_model_velocity_replay",
}
FORMAL_SSTW_CLEAN_NEGATIVE_EVIDENCE_LEVELS = {
    FORMAL_CLEAN_NEGATIVE_EVIDENCE_LEVEL,
    "attacked_video_wan_vae_model_velocity_replay",
}


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


def _load_profile_context(config_path: str | Path) -> dict[str, Any]:
    """从 protocol config 读取 SSTW formal 结果的 profile 语义。"""
    config_path = Path(config_path)
    config = _read_json(config_path)
    if "target_fpr" not in config:
        raise KeyError(f"protocol config 缺少 target_fpr: {config_path}")
    return {
        "paper_result_level": str(config.get("paper_result_level") or "probe_paper"),
        "target_fpr": float(config["target_fpr"]),
        "target_fpr_source_config_path": str(config_path),
        "minimum_clean_negative_count": int(config.get("minimum_clean_negative_count") or 0),
        "allow_effect_size_claims": bool(config.get("allow_effect_size_claims", False)),
    }


def _safe_float(value: object) -> float | None:
    """把 record 中可能为空的数值字段转换为 float。"""
    if value is None or value == "" or value == "unsupported":
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _score_from_runtime_detection(record: dict[str, Any]) -> tuple[float | None, str]:
    """从 runtime detection record 中选择 SSTW 正式转写分数。"""
    for field_name in ("sstw_raw_detector_score", "raw_detector_score"):
        value = _safe_float(record.get(field_name))
        if value is not None:
            return round(value, 6), field_name
    return None, "missing_formal_video_detector_score"


def formal_sstw_score_record_ready_for_claim(record: dict[str, Any]) -> bool:
    """判断 SSTW positive record 是否具备正式论文级检测证据。"""

    return (
        record.get("metric_status") == "measured_formal"
        and str(record.get("sample_role") or "").lower() not in {"clean_negative", "controlled_negative"}
        and record.get("sstw_detector_evidence_level") in FORMAL_SSTW_DETECTOR_EVIDENCE_LEVELS
        and record.get("trajectory_trace_used_for_score") is False
        and record.get("runtime_detection_claim_level") == "formal_paper_detector"
        and _safe_float(record.get("sstw_raw_detector_score")) is not None
    )


def formal_sstw_clean_negative_record_ready_for_calibration(record: dict[str, Any]) -> bool:
    """判断 SSTW clean negative record 是否可用于 fixed-FPR 校准。"""

    return (
        record.get("metric_status") == "measured_formal"
        and str(record.get("sample_role") or "").lower() == "clean_negative"
        and record.get("clean_negative_evidence_level") in FORMAL_SSTW_CLEAN_NEGATIVE_EVIDENCE_LEVELS
        and record.get("trajectory_trace_used_for_score") is False
        and _safe_float(record.get("sstw_clean_negative_score")) is not None
    )


def _runtime_detection_record_ready(record: dict[str, Any]) -> bool:
    """判断 runtime detection record 是否来自正式视频内容检测器。"""

    return (
        record.get("runtime_detection_status") == "ready"
        and str(record.get("method_variant") or "sstw_full_method") in {
            "sstw_full_method",
            "key_conditioned_state_space_with_trajectory",
        }
        and record.get("sstw_detector_evidence_level") in FORMAL_SSTW_DETECTOR_EVIDENCE_LEVELS
        and record.get("trajectory_trace_used_for_score") is False
        and record.get("runtime_detection_claim_level") == "formal_paper_detector"
    )


def _score_from_control_record(record: dict[str, Any]) -> tuple[float | None, str]:
    """从 SSTW 受控负样本 record 中选择 clean negative 校准分数。

    正式论文级 clean negative 必须来自 clean video content detector, 不能来自
    latent trajectory control 或其他受控 proxy 记录。
    """

    for field_name in ("sstw_clean_negative_score", "clean_negative_score", "sstw_raw_detector_score", "raw_detector_score"):
        value = _safe_float(record.get(field_name))
        if value is not None:
            return round(value, 6), field_name
    return None, "missing_formal_clean_negative_score"


def build_sstw_measured_formal_records(
    run_root: str | Path,
    config_path: str | Path = DEFAULT_PROTOCOL_CONFIG,
) -> list[dict[str, Any]]:
    """从 SSTW runtime detection records 构建本方法 measured_formal records。

    该函数属于项目特定转写层。它不重新运行 GPU, 只把本项目已经完成的
    generation -> attack -> detection 链路转成与 external baseline 对齐的
    `metric_status: measured_formal` 记录形状。当 protocol config 启用
    `allow_effect_size_claims` 且 target_fpr 为0.1 时, probe_paper 产物用于支撑
    fpr=0.1 论文设定下的小样本结论候选, 但不能外推到 pilot_paper 或 full_paper 的更低 FPR。
    """
    run_root = Path(run_root)
    profile_context = _load_profile_context(config_path)
    detection_records = _read_jsonl(run_root / "records" / "runtime_detection_records.jsonl")
    control_records = _read_jsonl(run_root / "records" / "sstw_clean_negative_score_records.jsonl")
    if not control_records:
        control_records = _read_jsonl(run_root / "records" / "controlled_negative_records.jsonl")
    formal_records: list[dict[str, Any]] = []
    claim_support_status = (
        "sstw_measured_formal_paper_profile_claim_candidate"
        if profile_context["allow_effect_size_claims"]
        else "sstw_measured_formal_paper_profile_only"
    )
    for index, detection_record in enumerate(detection_records):
        if not _runtime_detection_record_ready(detection_record):
            continue
        score, score_field = _score_from_runtime_detection(detection_record)
        if score is None:
            continue
        payload = {
            "method_id": SSTW_METHOD_ID,
            "method_role": "proposed_method",
            "generation_model_id": detection_record.get("generation_model_id"),
            "prompt_id": detection_record.get("prompt_id"),
            "seed_id": detection_record.get("seed_id"),
            "trajectory_trace_id": detection_record.get("trajectory_trace_id"),
            "split": detection_record.get("split"),
            "protocol_split": detection_record.get("protocol_split"),
            "colab_runtime_profile": detection_record.get("colab_runtime_profile"),
            "attack_name": detection_record.get("attack_name"),
            "source_video_path": detection_record.get("source_video_path"),
            "source_video_sha256": detection_record.get("source_video_sha256"),
            "attacked_video_path": detection_record.get("attacked_video_path"),
            "attacked_video_sha256": detection_record.get("attacked_video_sha256"),
            "sample_role": "attacked_positive",
            "sstw_score": score,
            "sstw_raw_detector_score": score,
            "raw_detector_score": score,
            "sstw_detected": bool(detection_record.get("attacked_video_detectable")),
            "sstw_score_semantics": FORMAL_VIDEO_DETECTOR_SCORE_SEMANTICS,
            "sstw_score_orientation": "higher_is_more_watermarked",
            "sstw_detection_score_field": score_field,
            "runtime_detection_evidence_level": detection_record.get("runtime_detection_evidence_level"),
            "sstw_detector_evidence_level": detection_record.get("sstw_detector_evidence_level"),
            "sstw_detector_input_contract": detection_record.get("sstw_detector_input_contract"),
            "sstw_detector_key_digest": detection_record.get("sstw_detector_key_digest"),
            "trajectory_trace_used_for_score": False,
            "runtime_detection_claim_level": detection_record.get("runtime_detection_claim_level"),
            "source_runtime_detection_record_index": index,
            **profile_context,
        }
        digest = build_stable_digest(payload)
        formal_records.append(with_flow_evidence_protocol_defaults({
            "record_version": "sstw_measured_formal_v1",
            "sstw_measured_formal_record_id": f"sstw_measured_formal_{digest[:16]}",
            "metric_status": "measured_formal",
            "comparison_scope": "paper_protocol_formal_adapter",
            "sstw_measured_formal_status": "ready",
            "claim_support_status": claim_support_status,
            **payload,
        }, trajectory_source_level="project_owned_sstw_video_content_detector", claim_support_status=claim_support_status))
    for index, control_record in enumerate(control_records):
        if str(control_record.get("sample_role") or "").lower() != "clean_negative":
            continue
        if str(control_record.get("method_variant") or "sstw_full_method") != "sstw_full_method":
            continue
        if control_record.get("clean_negative_evidence_level") not in FORMAL_SSTW_CLEAN_NEGATIVE_EVIDENCE_LEVELS:
            continue
        if control_record.get("trajectory_trace_used_for_score") is not False:
            continue
        score, score_field = _score_from_control_record(control_record)
        if score is None:
            continue
        control_name = str(control_record.get("control_name") or control_record.get("negative_family") or "sstw_controlled_negative")
        payload = {
            "method_id": SSTW_METHOD_ID,
            "method_role": "proposed_method",
            "generation_model_id": control_record.get("generation_model_id"),
            "prompt_id": control_record.get("prompt_id"),
            "seed_id": control_record.get("seed_id"),
            "trajectory_trace_id": control_record.get("trajectory_trace_id"),
            "split": control_record.get("split"),
            "protocol_split": control_record.get("protocol_split"),
            "colab_runtime_profile": control_record.get("colab_runtime_profile"),
            "attack_name": "",
            "sample_role": "clean_negative",
            "negative_family": control_name,
            "control_name": control_name,
            "clean_negative_unit_id": f"sstw_{control_name}_{control_record.get('prompt_id')}_{control_record.get('seed_id')}",
            "sstw_score": score,
            "sstw_raw_detector_score": score,
            "raw_detector_score": score,
            "sstw_clean_negative_score": score,
            "clean_negative_score": score,
            "sstw_score_semantics": FORMAL_VIDEO_DETECTOR_SCORE_SEMANTICS,
            "sstw_clean_negative_score_semantics": FORMAL_VIDEO_DETECTOR_SCORE_SEMANTICS,
            "sstw_score_orientation": "higher_is_more_watermarked",
            "sstw_detection_score_field": score_field,
            "clean_negative_evidence_level": control_record.get("clean_negative_evidence_level"),
            "clean_negative_video_path": control_record.get("clean_negative_video_path") or control_record.get("source_video_path"),
            "sstw_detector_input_contract": control_record.get("sstw_detector_input_contract"),
            "sstw_detector_key_digest": control_record.get("sstw_detector_key_digest"),
            "trajectory_trace_used_for_score": False,
            "clean_negative_source_record_family": "sstw_clean_negative_score_records",
            "source_controlled_negative_record_index": index,
            **profile_context,
        }
        digest = build_stable_digest(payload)
        formal_records.append(with_flow_evidence_protocol_defaults({
            "record_version": "sstw_measured_formal_v1",
            "sstw_measured_formal_record_id": f"sstw_measured_formal_{digest[:16]}",
            "metric_status": "measured_formal",
            "comparison_scope": "paper_protocol_formal_adapter",
            "sstw_measured_formal_status": "ready",
            "claim_support_status": claim_support_status,
            **payload,
        }, trajectory_source_level="project_owned_sstw_clean_video_content_detector", claim_support_status=claim_support_status))
    if formal_records:
        all_fields = sorted({field for record in formal_records for field in record})
        for record in formal_records:
            for field in all_fields:
                record.setdefault(field, None)
    return formal_records


def audit_sstw_measured_formal_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """审计 SSTW measured_formal records 是否可以进入同协议对比表。"""
    positive_records = [
        record
        for record in records
        if str(record.get("sample_role") or "").lower() not in {"clean_negative", "controlled_negative"}
    ]
    clean_negative_records = [
        record
        for record in records
        if str(record.get("sample_role") or "").lower() in {"clean_negative", "controlled_negative"}
    ]
    attacks = {str(record.get("attack_name")) for record in positive_records if record.get("attack_name")}
    prompts = {str(record.get("prompt_id")) for record in positive_records if record.get("prompt_id")}
    scores = [float(record["sstw_score"]) for record in positive_records if record.get("sstw_score") is not None]
    clean_negative_scores = [
        float(record["sstw_clean_negative_score"])
        for record in clean_negative_records
        if record.get("sstw_clean_negative_score") is not None
    ]
    detected_count = sum(1 for record in positive_records if record.get("sstw_detected") is True)
    metric_statuses = {str(record.get("metric_status")) for record in records if record.get("metric_status")}
    formal_positive_count = sum(1 for record in positive_records if formal_sstw_score_record_ready_for_claim(record))
    formal_clean_negative_count = sum(1 for record in clean_negative_records if formal_sstw_clean_negative_record_ready_for_calibration(record))
    minimum_clean_negative_count = int(records[0].get("minimum_clean_negative_count") or 0) if records else 0
    missing_requirements: list[str] = []
    if not scores:
        missing_requirements.append("sstw_attacked_positive_measured_formal_records")
    if len(clean_negative_scores) < minimum_clean_negative_count:
        missing_requirements.append("sstw_clean_negative_measured_formal_records")
    if metric_statuses != {"measured_formal"}:
        missing_requirements.append("sstw_metric_status_measured_formal_only")
    if formal_positive_count != len(positive_records):
        missing_requirements.append("sstw_positive_formal_video_detector_evidence")
    if formal_clean_negative_count != len(clean_negative_records):
        missing_requirements.append("sstw_clean_negative_formal_video_detector_evidence")
    decision = "PASS" if not missing_requirements else "FAIL"
    return {
        "stage_id": "sstw_measured_formal_result",
        "sstw_measured_formal_decision": decision,
        "claim_support_status": records[0].get("claim_support_status", "sstw_measured_formal_blocked") if decision == "PASS" else "sstw_measured_formal_blocked",
        "paper_result_level": records[0].get("paper_result_level") if records else None,
        "target_fpr": records[0].get("target_fpr") if records else None,
        "sstw_measured_formal_record_count": len(records),
        "sstw_measured_formal_ready_count": len(records) if decision == "PASS" else 0,
        "sstw_measured_formal_positive_record_count": len(positive_records),
        "sstw_measured_formal_clean_negative_record_count": len(clean_negative_records),
        "sstw_measured_formal_clean_negative_score_count": len(clean_negative_scores),
        "minimum_clean_negative_count": minimum_clean_negative_count,
        "sstw_measured_formal_prompt_count": len(prompts),
        "sstw_measured_formal_attack_count": len(attacks),
        "sstw_measured_formal_detected_count": detected_count,
        "sstw_formal_video_detector_positive_count": formal_positive_count,
        "sstw_formal_video_detector_clean_negative_count": formal_clean_negative_count,
        "sstw_measured_formal_detectable_rate": round(detected_count / len(positive_records), 6) if positive_records else None,
        "sstw_measured_formal_score_mean": round(mean(scores), 6) if scores else None,
        "sstw_measured_formal_clean_negative_score_mean": round(mean(clean_negative_scores), 6) if clean_negative_scores else None,
        "sstw_measured_formal_metric_status": "measured_formal" if decision == "PASS" else "missing",
        "missing_sstw_measured_formal_requirements": missing_requirements,
        "sstw_measured_formal_missing_requirement_count": len(missing_requirements),
        "comparison_scope": "paper_protocol_formal_adapter",
    }


def run_sstw_measured_formal_result(
    run_root: str | Path,
    config_path: str | Path = DEFAULT_PROTOCOL_CONFIG,
) -> dict[str, Any]:
    """写出 SSTW 本方法 measured_formal records、table、decision 和 report。"""
    run_root = Path(run_root)
    records = build_sstw_measured_formal_records(run_root, config_path)
    audit = audit_sstw_measured_formal_records(records)
    write_jsonl(run_root / "records" / "sstw_measured_formal_records.jsonl", records)
    write_csv(run_root / "tables" / "sstw_measured_formal_table.csv", records)
    write_json(run_root / "artifacts" / "sstw_measured_formal_decision.json", audit)
    report = (
        "# SSTW Measured Formal Result Report\n\n"
        "该报告把本项目 SSTW generation -> attack -> detection 链路转写为与 external baseline "
        "同层级的 measured_formal 记录。paper_profile 在 target_fpr=0.1 配置下用于支撑 "
        "fpr=0.1 论文设定的小样本结论候选, 但不能外推到 pilot_paper 或 full_paper 的更低 FPR。\n\n"
        f"- sstw_measured_formal_decision: {audit['sstw_measured_formal_decision']}\n"
        f"- paper_result_level: {audit['paper_result_level']}\n"
        f"- target_fpr: {audit['target_fpr']}\n"
        f"- sstw_measured_formal_record_count: {audit['sstw_measured_formal_record_count']}\n"
        f"- sstw_measured_formal_positive_record_count: {audit['sstw_measured_formal_positive_record_count']}\n"
        f"- sstw_measured_formal_clean_negative_score_count: {audit['sstw_measured_formal_clean_negative_score_count']}\n"
        f"- sstw_measured_formal_attack_count: {audit['sstw_measured_formal_attack_count']}\n"
        f"- sstw_measured_formal_score_mean: {audit['sstw_measured_formal_score_mean']}\n"
        f"- sstw_measured_formal_clean_negative_score_mean: {audit['sstw_measured_formal_clean_negative_score_mean']}\n"
        f"- sstw_measured_formal_detectable_rate: {audit['sstw_measured_formal_detectable_rate']}\n"
        f"- missing_sstw_measured_formal_requirements: {', '.join(audit['missing_sstw_measured_formal_requirements']) if audit['missing_sstw_measured_formal_requirements'] else 'none'}\n"
        f"- claim_support_status: {audit['claim_support_status']}\n"
    )
    report_path = run_root / "reports" / "sstw_measured_formal_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 SSTW 本方法 measured_formal 结果。")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--config-path", default=DEFAULT_PROTOCOL_CONFIG)
    args = parser.parse_args()
    payload = run_sstw_measured_formal_result(args.run_root, args.config_path)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
