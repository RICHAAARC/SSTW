"""SSTW 本方法 measured_formal 结果转写器。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from main.core.digest import build_stable_digest
from main.protocol.flow_evidence_fields import with_flow_evidence_protocol_defaults
from main.protocol.record_writer import write_json, write_jsonl
from main.protocol.table_builder import write_csv


DEFAULT_PROTOCOL_CONFIG = "configs/protocol/validation_scale_generative_probe.json"
SSTW_METHOD_ID = "sstw_key_conditioned_flow_trajectory"


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
        "paper_result_level": str(config.get("paper_result_level") or "validation_scale"),
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
    for field_name in ("S_final_conservative", "S_runtime_attack_detection", "S_path_inv", "S_velocity"):
        value = _safe_float(record.get(field_name))
        if value is not None:
            return round(value, 6), field_name
    return None, "missing_runtime_detection_score"


def _score_from_control_record(record: dict[str, Any]) -> tuple[float | None, str]:
    """从 SSTW 受控负样本 record 中选择 clean negative 校准分数。

    通用工程写法是让下游公平校准只消费一种稳定的 `sstw_clean_negative_score`
    字段。项目特定写法是当前 validation_scale 的 SSTW clean negative 来自
    `controlled_negative_records.jsonl`, 它们由同一条 latent trajectory 的方向破坏
    控制构造, 用于在 paper 级前验证阈值校准闭环。
    """

    for field_name in ("S_final_conservative", "S_final", "S_path_inv", "S_velocity"):
        value = _safe_float(record.get(field_name))
        if value is not None:
            return round(value, 6), field_name
    return None, "missing_control_negative_score"


def build_sstw_measured_formal_records(
    run_root: str | Path,
    config_path: str | Path = DEFAULT_PROTOCOL_CONFIG,
) -> list[dict[str, Any]]:
    """从 SSTW runtime detection records 构建本方法 measured_formal records。

    该函数属于项目特定转写层。它不重新运行 GPU, 只把本项目已经完成的
    generation -> attack -> detection 链路转成与 external baseline 对齐的
    `metric_status: measured_formal` 记录形状。validation_scale 产物仍只能说明
    小样本全流程可打通, 不支持最终效果大小 claim。
    """
    run_root = Path(run_root)
    profile_context = _load_profile_context(config_path)
    detection_records = _read_jsonl(run_root / "records" / "runtime_detection_records.jsonl")
    control_records = _read_jsonl(run_root / "records" / "controlled_negative_records.jsonl")
    formal_records: list[dict[str, Any]] = []
    claim_support_status = (
        "sstw_measured_formal_paper_profile_claim_candidate"
        if profile_context["allow_effect_size_claims"]
        else "sstw_measured_formal_validation_scale_only"
    )
    for index, detection_record in enumerate(detection_records):
        if detection_record.get("runtime_detection_status") != "ready":
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
            "attack_name": detection_record.get("attack_name"),
            "source_video_path": detection_record.get("source_video_path"),
            "source_video_sha256": detection_record.get("source_video_sha256"),
            "attacked_video_path": detection_record.get("attacked_video_path"),
            "attacked_video_sha256": detection_record.get("attacked_video_sha256"),
            "sample_role": "attacked_positive",
            "sstw_score": score,
            "sstw_detected": bool(detection_record.get("attacked_video_detectable")),
            "sstw_score_semantics": "sstw_conservative_detector_score",
            "sstw_score_orientation": "higher_is_more_watermarked",
            "sstw_detection_score_field": score_field,
            "runtime_detection_evidence_level": detection_record.get("runtime_detection_evidence_level"),
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
        }, trajectory_source_level="project_owned_sstw_runtime_detection", claim_support_status=claim_support_status))
    for index, control_record in enumerate(control_records):
        if str(control_record.get("sample_role") or "").lower() not in {"controlled_negative", "clean_negative"}:
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
            "attack_name": "",
            "sample_role": "clean_negative",
            "negative_family": control_name,
            "control_name": control_name,
            "clean_negative_unit_id": f"sstw_{control_name}_{control_record.get('prompt_id')}_{control_record.get('seed_id')}",
            "sstw_score": score,
            "sstw_clean_negative_score": score,
            "clean_negative_score": score,
            "sstw_score_semantics": "sstw_conservative_detector_score",
            "sstw_clean_negative_score_semantics": "sstw_conservative_detector_score",
            "sstw_score_orientation": "higher_is_more_watermarked",
            "sstw_detection_score_field": score_field,
            "clean_negative_evidence_level": "project_owned_sstw_controlled_negative_record",
            "clean_negative_source_record_family": "controlled_negative_records",
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
        }, trajectory_source_level="project_owned_sstw_clean_negative_control", claim_support_status=claim_support_status))
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
    minimum_clean_negative_count = int(records[0].get("minimum_clean_negative_count") or 0) if records else 0
    missing_requirements: list[str] = []
    if not scores:
        missing_requirements.append("sstw_attacked_positive_measured_formal_records")
    if len(clean_negative_scores) < minimum_clean_negative_count:
        missing_requirements.append("sstw_clean_negative_measured_formal_records")
    if metric_statuses != {"measured_formal"}:
        missing_requirements.append("sstw_metric_status_measured_formal_only")
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
        "同层级的 measured_formal 记录。validation_scale 结果只用于小样本全流程打通验证, "
        "不能作为最终论文效果大小结论。\n\n"
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
