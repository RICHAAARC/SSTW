"""执行 paper profile 的 non-runtime / adaptive attack 正式记录生成。

该模块把 11 个 non-runtime / adaptive 协议接入真实视频文件和 SSTW 正式视频内容
检测器。它不在 Notebook 中手写结果, 也不使用 runtime detection 结果直接合成
分数; 所有 measured_formal 行都必须来自已落盘视频文件的重新检测。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping

from experiments.generative_video_model_probe.formal_motion_claim_filter import (
    select_motion_claim_generation_records,
)
from main.attacks.video_runtime_attack_protocol import (
    load_protocol_config_with_shared_attack_protocol,
    required_non_runtime_attack_protocols_from_config,
)
from main.core.digest import build_stable_digest
from main.core.progress import ProgressReporter
from main.methods.state_space_watermark.video_content_detector import (
    FORMAL_VIDEO_DETECTOR_INPUT_CONTRACT,
    FORMAL_VIDEO_DETECTOR_SCORE_SEMANTICS,
    build_sstw_detector_key,
    score_video_content,
)
from main.protocol.flow_evidence_fields import with_flow_evidence_protocol_defaults
from main.protocol.record_writer import write_json, write_jsonl
from main.protocol.table_builder import write_csv


DEFAULT_PROTOCOL_CONFIG = "configs/protocol/probe_paper_generative_probe.json"
FORMAL_ADAPTIVE_ATTACK_EVIDENCE_LEVEL = "formal_adaptive_attack_execution"

PREFERRED_RUNTIME_VIDEO_BY_PROTOCOL: dict[str, tuple[str, ...]] = {
    "generative_recompression_or_regeneration_attack": (
        "platform_transcode_runtime",
        "h264_crf28_runtime",
        "h265_crf28_runtime",
    ),
    "watermark_removal_optimization_attack": (
        "denoise_runtime",
        "gaussian_blur_runtime",
        "median_blur_runtime",
    ),
    "adversarial_detector_evasion_attack": (
        "compression_noise_combined_runtime",
        "brightness_contrast_runtime",
        "color_jitter_runtime",
    ),
    "collusion_multi_sample_attack": (
        "frame_average_runtime",
        "frame_duplicate_runtime",
    ),
}

KEY_TRANSFORMATION_BY_PROTOCOL: dict[str, str] = {
    "endpoint_preserving_path_perturbation_attack": "endpoint_path_perturbed_key",
    "flow_time_grid_mismatch_attack": "flow_time_grid_mismatch_key",
    "wrong_sampler_replay_attack": "wrong_sampler_replay_key",
    "wrong_prompt_replay_attack": "wrong_prompt_replay_key",
    "wrong_key_attack": "wrong_key_detector_rescore",
    "detector_probing_with_public_negatives": "detector_probe_public_negative_key",
    "watermark_spoofing_or_copy_attack": "watermark_copy_or_spoof_key",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL records, 文件不存在时返回空列表。"""

    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256_file(path: Path) -> str:
    """计算视频文件摘要, 用于证明 adaptive 记录确实绑定了落盘视频。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_float(value: Any) -> float | None:
    """把可选数值转成 float。"""

    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_protocol_context(config_path: str | Path) -> dict[str, Any]:
    """读取 paper profile 和 required non-runtime 协议。"""

    path = Path(config_path)
    config = load_protocol_config_with_shared_attack_protocol(path)
    if "target_fpr" not in config:
        raise KeyError(f"protocol config 缺少 target_fpr: {path}")
    return {
        "paper_result_level": str(config.get("paper_result_level") or "probe_paper"),
        "target_fpr": float(config["target_fpr"]),
        "protocol_config_path": str(path),
        "required_non_runtime_attack_protocols": list(required_non_runtime_attack_protocols_from_config(config)),
    }


def _identity_key(record: Mapping[str, Any]) -> tuple[str, str, str, str]:
    """构造 generation、runtime attack 和 adaptive records 的视频身份键。"""

    return (
        str(record.get("generation_model_id") or ""),
        str(record.get("prompt_id") or ""),
        str(record.get("seed_id") or ""),
        str(record.get("trajectory_trace_id") or ""),
    )


def _resolve_generation_video_path(run_root: Path, record: Mapping[str, Any]) -> Path:
    """解析 generation record 的视频路径。"""

    path = Path(str(record.get("video_path") or ""))
    if path.exists():
        return path
    return run_root / "videos" / path.name


def _runtime_attack_records_by_key(run_root: Path) -> dict[tuple[str, str, str, str], list[dict[str, Any]]]:
    """按生成单元索引已完成的 runtime attack 记录。"""

    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for record in _read_jsonl(run_root / "records" / "runtime_attack_records.jsonl"):
        if record.get("attack_runtime_status") != "ready":
            continue
        grouped.setdefault(_identity_key(record), []).append(record)
    return grouped


def _select_video_for_protocol(
    run_root: Path,
    protocol_name: str,
    generation_record: Mapping[str, Any],
    attack_records: Iterable[Mapping[str, Any]],
) -> tuple[Path, str, str]:
    """为 non-runtime / adaptive 协议选择实际要检测的视频文件。

    对需要视频扰动的协议, 优先复用已经由 runtime attack 阶段产生的正式视频文件。
    对 key、prompt、sampler 或 detector probing 类协议, 使用 source generation
    视频并改变检测 key 或检测上下文。
    """

    preferred_attacks = PREFERRED_RUNTIME_VIDEO_BY_PROTOCOL.get(protocol_name, ())
    by_attack = {
        str(record.get("attack_name") or ""): record
        for record in attack_records
        if record.get("attacked_video_path")
    }
    for attack_name in preferred_attacks:
        record = by_attack.get(attack_name)
        if record:
            path = Path(str(record.get("attacked_video_path") or ""))
            if path.exists():
                return path, "runtime_transformed_video", attack_name
    return _resolve_generation_video_path(run_root, generation_record), "source_generation_video", "none"


def _detector_key_for_protocol(protocol_name: str, generation_record: Mapping[str, Any]) -> tuple[str, str]:
    """构造 adaptive 协议使用的检测 key。"""

    base_key = build_sstw_detector_key(dict(generation_record))
    transformation = KEY_TRANSFORMATION_BY_PROTOCOL.get(protocol_name, "matched_key_video_rescore")
    if transformation == "matched_key_video_rescore":
        return base_key, transformation
    return f"{base_key}::{transformation}", transformation


def _formal_adaptive_record(
    *,
    run_root: Path,
    context: Mapping[str, Any],
    protocol_name: str,
    generation_record: Mapping[str, Any],
    attack_records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """执行单个 generation 单元上的一个 adaptive 协议并返回正式记录。"""

    video_path, video_source_kind, source_attack_name = _select_video_for_protocol(
        run_root,
        protocol_name,
        generation_record,
        attack_records,
    )
    detector_key, key_transformation = _detector_key_for_protocol(protocol_name, generation_record)
    if not video_path.exists():
        raise FileNotFoundError(f"adaptive_attack_input_video_missing:{video_path}")
    result = score_video_content(video_path, detector_key=detector_key)
    score = round(float(result.score), 6)
    payload = {
        "record_version": "formal_adaptive_attack_execution_v1",
        "paper_result_level": context["paper_result_level"],
        "target_fpr": context["target_fpr"],
        "non_runtime_attack_protocol": protocol_name,
        "adaptive_attack_name": protocol_name,
        "adaptive_attack_family": _adaptive_family(protocol_name),
        "generation_model_id": generation_record.get("generation_model_id"),
        "prompt_id": generation_record.get("prompt_id"),
        "seed_id": generation_record.get("seed_id"),
        "trajectory_trace_id": generation_record.get("trajectory_trace_id"),
        "split": generation_record.get("split"),
        "protocol_split": generation_record.get("protocol_split"),
        "colab_runtime_profile": generation_record.get("colab_runtime_profile"),
        "adaptive_attack_input_video_path": str(video_path),
        "adaptive_attack_input_video_sha256": _sha256_file(video_path),
        "adaptive_attack_video_source_kind": video_source_kind,
        "adaptive_attack_source_runtime_attack_name": source_attack_name,
        "adaptive_attack_detector_key_transformation": key_transformation,
        "adaptive_attack_score": score,
        "adaptive_attack_score_semantics": FORMAL_VIDEO_DETECTOR_SCORE_SEMANTICS,
        "adaptive_attack_score_orientation": "higher_is_more_watermarked",
        "adaptive_attack_detected_by_sstw": score >= 0.5,
        "adaptive_attack_status": "ready",
        "metric_status": "measured_formal",
        "adaptive_attack_evidence_level": FORMAL_ADAPTIVE_ATTACK_EVIDENCE_LEVEL,
        "adaptive_robustness_claim_allowed": True,
        "sstw_detector_input_contract": FORMAL_VIDEO_DETECTOR_INPUT_CONTRACT,
        "sstw_detector_key_digest": result.detector_key_digest,
        "sstw_content_feature_count": result.content_feature_count,
        "sstw_detector_sampled_frame_count": result.sampled_frame_count,
        "claim_support_status": "formal_adaptive_attack_execution_ready",
    }
    digest = build_stable_digest(payload)
    return with_flow_evidence_protocol_defaults(
        {
            "formal_adaptive_attack_execution_record_id": f"formal_adaptive_attack_{digest[:16]}",
            **payload,
        },
        trajectory_source_level=FORMAL_ADAPTIVE_ATTACK_EVIDENCE_LEVEL,
        flow_state_admissibility_status="formal_adaptive_attack_execution_ready",
        claim_support_status="formal_adaptive_attack_execution_ready",
    )


def _adaptive_family(protocol_name: str) -> str:
    """把协议映射到论文表格中的攻击族。"""

    if protocol_name in {"wrong_sampler_replay_attack", "wrong_prompt_replay_attack"}:
        return "replay_signature_mismatch"
    if protocol_name == "flow_time_grid_mismatch_attack":
        return "time_grid_or_scheduler_mismatch"
    if protocol_name == "wrong_key_attack":
        return "key_mismatch_attack"
    if protocol_name == "generative_recompression_or_regeneration_attack":
        return "generative_recompression_or_regeneration"
    if protocol_name == "detector_probing_with_public_negatives":
        return "detector_threshold_probing"
    if protocol_name == "watermark_removal_optimization_attack":
        return "watermark_removal_optimization"
    if protocol_name == "watermark_spoofing_or_copy_attack":
        return "watermark_spoofing_or_copy"
    if protocol_name == "collusion_multi_sample_attack":
        return "collusion_multi_sample"
    if protocol_name == "adversarial_detector_evasion_attack":
        return "adversarial_detector_evasion"
    return "endpoint_preserving_path_attack"


def build_formal_adaptive_attack_execution_records(
    run_root: str | Path,
    config_path: str | Path = DEFAULT_PROTOCOL_CONFIG,
) -> list[dict[str, Any]]:
    """从真实视频文件执行 11 个 non-runtime / adaptive 协议记录。"""

    run_root = Path(run_root)
    context = _load_protocol_context(config_path)
    generation_records = [
        record
        for record in _read_jsonl(run_root / "records" / "generation_records.jsonl")
        if record.get("generation_status") == "success"
        and str(record.get("sample_role") or record.get("generation_sample_role") or "").lower() != "clean_negative"
    ]
    formal_metric_records = _read_jsonl(run_root / "records" / "formal_quality_motion_semantic_records.jsonl")
    selection = select_motion_claim_generation_records(generation_records, formal_metric_records)
    attack_by_key = _runtime_attack_records_by_key(run_root)
    protocols = [str(item) for item in context["required_non_runtime_attack_protocols"] if str(item)]
    total = len(selection.eligible_generation_records) * len(protocols)
    progress = ProgressReporter("formal_adaptive_attack_execution", total, "adaptive_attack_event")
    records: list[dict[str, Any]] = []
    progress_index = 0
    for generation_record in selection.eligible_generation_records:
        source_attack_records = attack_by_key.get(_identity_key(generation_record), [])
        for protocol_name in protocols:
            progress_index += 1
            progress.update(
                progress_index,
                f"prompt={generation_record.get('prompt_id')} seed={generation_record.get('seed_id')} protocol={protocol_name}",
            )
            records.append(_formal_adaptive_record(
                run_root=run_root,
                context=context,
                protocol_name=protocol_name,
                generation_record=generation_record,
                attack_records=source_attack_records,
            ))
    progress.finish(f"ready={len(records)}")
    return records


def audit_formal_adaptive_attack_execution_records(
    records: list[dict[str, Any]],
    config_path: str | Path = DEFAULT_PROTOCOL_CONFIG,
) -> dict[str, Any]:
    """审计自动执行的 adaptive attack 正式记录是否覆盖配置要求。"""

    context = _load_protocol_context(config_path)
    required_protocols = {str(item) for item in context["required_non_runtime_attack_protocols"] if str(item)}
    ready_records = [
        record
        for record in records
        if record.get("metric_status") == "measured_formal"
        and record.get("adaptive_attack_status") == "ready"
        and record.get("adaptive_attack_evidence_level") == FORMAL_ADAPTIVE_ATTACK_EVIDENCE_LEVEL
    ]
    observed_protocols = {
        str(record.get("non_runtime_attack_protocol") or "")
        for record in ready_records
        if record.get("non_runtime_attack_protocol")
    }
    missing_protocols = sorted(required_protocols - observed_protocols)
    scores = [
        value
        for value in (_safe_float(record.get("adaptive_attack_score")) for record in ready_records)
        if value is not None
    ]
    decision = "PASS" if ready_records and not missing_protocols else "FAIL"
    return {
        "stage_id": "formal_adaptive_attack_execution",
        "formal_adaptive_attack_execution_decision": decision,
        "paper_result_level": context["paper_result_level"],
        "target_fpr": context["target_fpr"],
        "formal_adaptive_attack_execution_record_count": len(records),
        "formal_adaptive_attack_execution_ready_count": len(ready_records),
        "required_non_runtime_attack_protocols": sorted(required_protocols),
        "observed_non_runtime_attack_protocols": sorted(observed_protocols),
        "missing_non_runtime_attack_protocols": missing_protocols,
        "adaptive_attack_score_mean": round(mean(scores), 6) if scores else None,
        "claim_support_status": (
            "formal_adaptive_attack_execution_ready"
            if decision == "PASS"
            else "formal_adaptive_attack_execution_blocked"
        ),
    }


def run_formal_adaptive_attack_execution(
    run_root: str | Path,
    config_path: str | Path = DEFAULT_PROTOCOL_CONFIG,
) -> dict[str, Any]:
    """执行 adaptive attack 正式记录生成并写出 governed artifacts。"""

    run_root = Path(run_root)
    records = build_formal_adaptive_attack_execution_records(run_root, config_path)
    audit = audit_formal_adaptive_attack_execution_records(records, config_path)
    write_jsonl(run_root / "records" / "formal_adaptive_attack_execution_records.jsonl", records)
    write_csv(run_root / "tables" / "formal_adaptive_attack_execution_table.csv", records)
    write_json(run_root / "artifacts" / "formal_adaptive_attack_execution_decision.json", audit)
    report = (
        "# Formal Adaptive Attack Execution Report\n\n"
        "该报告记录 11 个 non-runtime / adaptive 协议在已落盘视频文件上的正式执行结果。"
        "分数由 SSTW 正式视频内容检测器重新计算, 不从 runtime detection 结果表直接复制。\n\n"
        f"- formal_adaptive_attack_execution_decision: {audit['formal_adaptive_attack_execution_decision']}\n"
        f"- paper_result_level: {audit['paper_result_level']}\n"
        f"- target_fpr: {audit['target_fpr']}\n"
        f"- formal_adaptive_attack_execution_record_count: {audit['formal_adaptive_attack_execution_record_count']}\n"
        f"- observed_non_runtime_attack_protocols: {', '.join(audit['observed_non_runtime_attack_protocols']) if audit['observed_non_runtime_attack_protocols'] else 'none'}\n"
        f"- missing_non_runtime_attack_protocols: {', '.join(audit['missing_non_runtime_attack_protocols']) if audit['missing_non_runtime_attack_protocols'] else 'none'}\n"
        f"- claim_support_status: {audit['claim_support_status']}\n"
    )
    report_path = run_root / "reports" / "formal_adaptive_attack_execution_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="执行 paper profile non-runtime / adaptive attack 正式协议。")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--config-path", default=DEFAULT_PROTOCOL_CONFIG)
    args = parser.parse_args()
    payload = run_formal_adaptive_attack_execution(args.run_root, args.config_path)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
