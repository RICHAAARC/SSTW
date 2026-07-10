"""从真实 attacked video 构建 endpoint、path、replay 与固定 FPR 检测记录。"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from statistics import pstdev
from typing import Any, Iterable, Mapping

from experiments.generative_video_model_probe.colab_runtime import _load_video_generation_pipeline, _select_dtype
from main.core.digest import build_stable_digest
from main.methods.state_space_watermark.endpoint_latent_detector import compute_endpoint_latent_evidence
from main.methods.state_space_watermark.formal_detector import (
    FORMAL_METHOD_VARIANTS,
    apply_frozen_flow_detector,
    fit_flow_evidence_calibration,
)
from main.methods.state_space_watermark.wan_flow_replay_backend import (
    WanFlowReplayResult,
    run_wan_attacked_video_replay,
    run_wan_control_replay,
    score_replay_trajectory_for_key,
)
from main.protocol.flow_evidence_fields import with_flow_evidence_protocol_defaults
from main.protocol.record_writer import write_json, write_jsonl
from main.protocol.table_builder import write_csv


FORMAL_FLOW_EVIDENCE_LEVEL = "attacked_video_wan_vae_model_velocity_replay"
FORMAL_FLOW_DETECTOR_INPUT_CONTRACT = "video_file_prompt_key_model_scheduler_and_frozen_calibration"


def _read_json(path: str | Path) -> dict[str, Any]:
    """读取 UTF-8 JSON 对象。"""

    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """读取 JSONL; 文件不存在时返回空列表。"""

    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _resolve_video_path(run_root: Path, raw_path: Any, *, fallback_dir: str) -> Path:
    """解析记录中的视频路径, 并兼容跨机器复制后的相对文件名。"""

    path = Path(str(raw_path or ""))
    if path.exists():
        return path
    candidate = run_root / fallback_dir / path.name
    return candidate


def _prompt_text_by_id(prompt_suite: Mapping[str, Any]) -> dict[str, str]:
    """构造 prompt ID 到实际条件文本的映射。"""

    return {
        str(item["prompt_id"]): str(item["prompt_text"])
        for item in prompt_suite.get("prompts") or []
        if item.get("prompt_id") and item.get("prompt_text")
    }


def _generation_key(record: Mapping[str, Any]) -> str:
    """生成与嵌入阶段完全一致的项目水印 key。"""

    return f"{record.get('generation_model_id')}::{record.get('prompt_id')}::{record.get('seed_id')}"


def _path_endpoint_consistency(endpoint_projection: float, path_projection: float) -> float:
    """比较 endpoint 与积分路径在同一 key 方向上的一致性。"""

    return max(0.0, min(1.0, 1.0 - abs(float(endpoint_projection) - float(path_projection))))


def _time_grid_reliability(result: WanFlowReplayResult) -> float:
    """根据多时间网格循环误差离散程度计算独立的 time-grid 可靠性。"""

    errors = [float(row.cycle_relative_error) for row in result.replay_trajectories]
    dispersion = pstdev(errors) if len(errors) > 1 else 0.0
    return math.exp(-max(0.0, dispersion))


def _scheduler_signature(scheduler: Any) -> str:
    """生成与 generation sketch 相同口径的 scheduler 配置签名。"""

    payload = json.dumps(dict(scheduler.config), ensure_ascii=False, sort_keys=True, default=str)
    return f"{type(scheduler).__name__}:{sha256(payload.encode('utf-8')).hexdigest()}"


def build_flow_evidence_payload(
    replay: WanFlowReplayResult,
    *,
    key_text: str,
    method_variant: str,
) -> dict[str, Any]:
    """把模型 replay 结果转换为统一正式证据字段。"""

    endpoint = replay.endpoint_evidence.as_dict()
    path = dict(replay.path_evidence)
    uncertainty = replay.replay_uncertainty.as_dict()
    return {
        **endpoint,
        **path,
        **uncertainty,
        **replay.endpoint_metadata,
        "formal_flow_evidence_level": FORMAL_FLOW_EVIDENCE_LEVEL,
        "formal_flow_detector_input_contract": FORMAL_FLOW_DETECTOR_INPUT_CONTRACT,
        "method_variant": method_variant,
        "detector_key_digest": sha256(key_text.encode("utf-8")).hexdigest(),
        "path_endpoint_consistency": round(_path_endpoint_consistency(
            replay.endpoint_evidence.projection,
            float(path.get("S_path_inv") or 0.0),
        ), 8),
        "time_grid_reliability": round(_time_grid_reliability(replay), 8),
        "flow_phase": 0.5,
        "replay_step_counts": list(replay.replay_step_counts),
        "replay_primary_step_count": replay.replay_step_counts[replay.primary_replay_index],
        "trajectory_trace_used_for_score": False,
        "metric_status": "measured_formal",
    }


def _control_payload(
    pipeline: Any,
    replay: WanFlowReplayResult,
    *,
    prompt: str,
    wrong_prompt: str,
    key_text: str,
) -> dict[str, Any]:
    """执行 wrong key、wrong prompt 与 wrong sampler/time-grid 真实对照。"""

    wrong_key = f"{key_text}::wrong_key_control"
    wrong_key_endpoint = compute_endpoint_latent_evidence(replay.endpoint_latent, key_text=wrong_key)

    primary_steps = int(replay.replay_step_counts[replay.primary_replay_index])
    wrong_key_trajectory, _wrong_key_schedule, wrong_key_path = run_wan_control_replay(
        pipeline,
        replay.endpoint_latent,
        prompt=prompt,
        key_text=wrong_key,
        num_inference_steps=primary_steps,
    )
    wrong_prompt_trajectory, _wrong_prompt_schedule, wrong_prompt_path = run_wan_control_replay(
        pipeline,
        replay.endpoint_latent,
        prompt=wrong_prompt,
        key_text=key_text,
        num_inference_steps=primary_steps,
    )

    scheduler_class = type(pipeline.scheduler)
    scheduler_config = dict(pipeline.scheduler.config)
    original_shift = float(scheduler_config.get("shift", 1.0))
    wrong_scheduler = scheduler_class.from_config(
        pipeline.scheduler.config,
        shift=original_shift + 1.0,
    )
    wrong_sampler_trajectory, _wrong_sampler_schedule, wrong_sampler_path = run_wan_control_replay(
        pipeline,
        replay.endpoint_latent,
        prompt=prompt,
        key_text=key_text,
        num_inference_steps=max(2, primary_steps + 3),
        scheduler=wrong_scheduler,
    )
    matched_path = float(replay.path_evidence.get("S_path_inv") or 0.0)
    matched_reliability = float(replay.replay_uncertainty.replay_reliability)
    matched_path_reliability_score = (0.5 + 0.5 * matched_path) * matched_reliability
    wrong_key_path_reliability_score = (
        0.5 + 0.5 * float(wrong_key_path.get("S_path_inv") or 0.0)
    ) * math.exp(-max(0.0, wrong_key_trajectory.cycle_relative_error))
    wrong_prompt_path_reliability_score = (
        0.5 + 0.5 * float(wrong_prompt_path.get("S_path_inv") or 0.0)
    ) * math.exp(-max(0.0, wrong_prompt_trajectory.cycle_relative_error))
    wrong_sampler_path_reliability_score = (
        0.5 + 0.5 * float(wrong_sampler_path.get("S_path_inv") or 0.0)
    ) * math.exp(-max(0.0, wrong_sampler_trajectory.cycle_relative_error))
    return {
        "wrong_key_endpoint_score": round(wrong_key_endpoint.score, 8),
        "wrong_key_S_path_inv": wrong_key_path.get("S_path_inv"),
        "wrong_key_replay_cycle_error": round(wrong_key_trajectory.cycle_relative_error, 8),
        "wrong_key_control_margin": round(
            (replay.endpoint_evidence.score + matched_path_reliability_score)
            - (wrong_key_endpoint.score + wrong_key_path_reliability_score),
            8,
        ),
        "wrong_prompt_replay_cycle_error": round(wrong_prompt_trajectory.cycle_relative_error, 8),
        "wrong_prompt_S_path_inv": wrong_prompt_path.get("S_path_inv"),
        "wrong_prompt_control_margin": round(
            matched_path_reliability_score - wrong_prompt_path_reliability_score,
            8,
        ),
        "wrong_sampler_replay_cycle_error": round(wrong_sampler_trajectory.cycle_relative_error, 8),
        "wrong_sampler_S_path_inv": wrong_sampler_path.get("S_path_inv"),
        "wrong_sampler_control_margin": round(
            matched_path_reliability_score - wrong_sampler_path_reliability_score,
            8,
        ),
        "wrong_sampler_control_shift": original_shift + 1.0,
        "replay_control_execution_status": "measured_formal",
        "wrong_prompt_control_prompt_digest": sha256(wrong_prompt.encode("utf-8")).hexdigest(),
    }


def _minimum_negative_count(config: Mapping[str, Any], split: str) -> int:
    """读取 calibration 或 held-out test 所需 negative event 数。"""

    if split == "calibration":
        return int(config.get("minimum_calibration_negative_event_count") or config.get("minimum_clean_negative_count") or 0)
    return int(config.get("minimum_heldout_test_negative_event_count") or config.get("minimum_clean_negative_count") or 0)


def _clean_trial_count(config: Mapping[str, Any], split: str, source_count: int) -> int:
    """把协议要求的 negative event 数均匀分配到真实 clean videos。"""

    if source_count <= 0:
        return 0
    return max(1, math.ceil(_minimum_negative_count(config, split) / source_count))


def _base_record(source: Mapping[str, Any], *, sample_role: str, method_variant: str) -> dict[str, Any]:
    """构造 positive 与 negative evidence 的共享身份字段。"""

    return {
        "record_version": "formal_flow_evidence_v1",
        "generation_model_id": source.get("generation_model_id"),
        "prompt_id": source.get("prompt_id"),
        "seed_id": source.get("seed_id"),
        "trajectory_trace_id": source.get("trajectory_trace_id"),
        "split": source.get("split"),
        "protocol_split": source.get("protocol_split"),
        "colab_runtime_profile": source.get("colab_runtime_profile"),
        "sample_role": sample_role,
        "method_variant": method_variant,
        "attack_name": source.get("attack_name"),
    }


def _load_pipeline(model_id: str) -> Any:
    """在 CUDA 上加载与生成阶段相同的 Wan pipeline。"""

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("正式 Flow replay 需要可用 CUDA GPU")
    return _load_video_generation_pipeline(model_id, _select_dtype(torch))


def _score_records_with_frozen_calibration(
    evidence_records: Iterable[Mapping[str, Any]],
    *,
    target_fpr: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """按 method variant 冻结 calibration negative 后评分全部 records。"""

    rows = [dict(record) for record in evidence_records]
    calibration_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in rows:
        if record.get("sample_role") == "clean_negative" and record.get("split") == "calibration":
            calibration_rows[str(record.get("method_variant"))].append(record)
    calibrations = {
        variant: fit_flow_evidence_calibration(
            calibration_rows.get(variant, []),
            method_variant=variant,
            target_fpr=target_fpr,
        )
        for variant in FORMAL_METHOD_VARIANTS
    }
    scored: list[dict[str, Any]] = []
    for record in rows:
        variant = str(record.get("method_variant"))
        detection = apply_frozen_flow_detector(record, calibrations[variant])
        score = float(detection["S_final_conservative"])
        scored_record = {
            **record,
            **detection,
            "runtime_detection_status": "ready",
            "runtime_detection_claim_level": "formal_paper_detector",
            "sstw_detector_evidence_level": FORMAL_FLOW_EVIDENCE_LEVEL,
            "sstw_detector_input_contract": FORMAL_FLOW_DETECTOR_INPUT_CONTRACT,
            "sstw_detector_key_digest": record.get("detector_key_digest"),
            "sstw_raw_detector_score": score,
            "raw_detector_score": score,
            "sstw_score": score,
            "attacked_video_detectable": bool(detection["decision"]),
            "claim_support_status": "sstw_complete_flow_mechanism_measured_formal",
        }
        if record.get("sample_role") == "clean_negative":
            scored_record.update({
                "clean_negative_status": "ready",
                "clean_negative_evidence_level": FORMAL_FLOW_EVIDENCE_LEVEL,
                "sstw_clean_negative_score": score,
                "clean_negative_score": score,
            })
        scored.append(scored_record)
    return scored, [calibration.as_dict() for calibration in calibrations.values()], calibrations


def _paired_path_gain_records(
    scored_records: Iterable[Mapping[str, Any]],
    calibrations: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """在同一 full-method 视频证据上配对比较 full 与 endpoint-only 检测器。"""

    rows: list[dict[str, Any]] = []
    for record in scored_records:
        if (
            record.get("sample_role") != "attacked_positive"
            or record.get("method_variant") != "sstw_full_method"
            or record.get("split") != "test"
        ):
            continue
        endpoint_only = apply_frozen_flow_detector(record, calibrations["endpoint_only_control"])
        full_score = float(record["S_final_conservative"])
        endpoint_score = float(endpoint_only["S_final_conservative"])
        rows.append({
            "record_version": "paired_path_evidence_gain_v1",
            "generation_model_id": record.get("generation_model_id"),
            "prompt_id": record.get("prompt_id"),
            "seed_id": record.get("seed_id"),
            "trajectory_trace_id": record.get("trajectory_trace_id"),
            "attack_name": record.get("attack_name"),
            "target_fpr": record.get("target_fpr"),
            "paired_source_method_variant": "sstw_full_method",
            "paired_full_detector_score": full_score,
            "paired_endpoint_only_detector_score": endpoint_score,
            "paired_path_evidence_score_gain": round(full_score - endpoint_score, 8),
            "paired_full_detector_decision": bool(record.get("decision")),
            "paired_endpoint_only_detector_decision": bool(endpoint_only.get("decision")),
            "paired_path_evidence_detection_gain": int(bool(record.get("decision"))) - int(bool(endpoint_only.get("decision"))),
            "metric_status": "measured_formal",
            "claim_support_status": "claim2_same_video_paired_fixed_fpr_evidence",
        })
    return rows


def _audit_three_layer_mechanism(
    scored_records: Iterable[Mapping[str, Any]],
    paired_path_records: Iterable[Mapping[str, Any]],
    *,
    target_fpr: float,
) -> dict[str, Any]:
    """审计 Claim-1 与 Claim-2, Claim-3 的最终认证由 replay gate 继续完成。"""

    rows = list(scored_records)
    full_positive = [
        record for record in rows
        if record.get("sample_role") == "attacked_positive"
        and record.get("method_variant") == "sstw_full_method"
        and record.get("split") == "test"
    ]
    full_test_negative = [
        record for record in rows
        if record.get("sample_role") == "clean_negative"
        and record.get("method_variant") == "sstw_full_method"
        and record.get("split") == "test"
    ]
    negative_accept_count = sum(bool(record.get("decision")) for record in full_test_negative)
    positive_accept_count = sum(bool(record.get("decision")) for record in full_positive)
    empirical_fpr = negative_accept_count / max(1, len(full_test_negative))
    positive_tpr = positive_accept_count / max(1, len(full_positive))

    def wilson_interval(successes: int, count: int) -> tuple[float | None, float | None]:
        if count <= 0:
            return None, None
        z = 1.96
        probability = successes / count
        denominator = 1.0 + z * z / count
        center = (probability + z * z / (2.0 * count)) / denominator
        half_width = z * math.sqrt(
            probability * (1.0 - probability) / count + z * z / (4.0 * count * count)
        ) / denominator
        return max(0.0, center - half_width), min(1.0, center + half_width)

    fpr_ci_lower, fpr_ci_upper = wilson_interval(negative_accept_count, len(full_test_negative))
    tpr_ci_lower, tpr_ci_upper = wilson_interval(positive_accept_count, len(full_positive))
    paired = list(paired_path_records)
    gains = [float(record["paired_path_evidence_score_gain"]) for record in paired]
    detection_gains = [float(record["paired_path_evidence_detection_gain"]) for record in paired]
    gain_mean = sum(gains) / len(gains) if gains else 0.0
    gain_se = pstdev(gains) / math.sqrt(len(gains)) if len(gains) > 1 else math.inf
    gain_ci_lower = gain_mean - 1.96 * gain_se if math.isfinite(gain_se) else None
    gain_ci_upper = gain_mean + 1.96 * gain_se if math.isfinite(gain_se) else None
    detection_gain_mean = sum(detection_gains) / len(detection_gains) if detection_gains else 0.0
    detection_gain_se = (
        pstdev(detection_gains) / math.sqrt(len(detection_gains))
        if len(detection_gains) > 1
        else math.inf
    )
    detection_gain_ci_lower = (
        detection_gain_mean - 1.96 * detection_gain_se
        if math.isfinite(detection_gain_se)
        else None
    )
    detection_gain_ci_upper = (
        detection_gain_mean + 1.96 * detection_gain_se
        if math.isfinite(detection_gain_se)
        else None
    )
    claim1_pass = (
        bool(full_positive)
        and bool(full_test_negative)
        and empirical_fpr <= target_fpr
        and tpr_ci_lower is not None
        and tpr_ci_lower > target_fpr
    )
    claim2_pass = bool(paired) and detection_gain_ci_lower is not None and detection_gain_ci_lower > 0.0
    return {
        "stage_id": "sstw_three_layer_mechanism_evidence",
        "claim_1_velocity_constraint_detectable_watermark_decision": "PASS" if claim1_pass else "FAIL",
        "claim_1_heldout_positive_count": len(full_positive),
        "claim_1_heldout_negative_count": len(full_test_negative),
        "claim_1_empirical_fpr": round(empirical_fpr, 8),
        "claim_1_empirical_fpr_ci_95_lower": round(fpr_ci_lower, 8) if fpr_ci_lower is not None else None,
        "claim_1_empirical_fpr_ci_95_upper": round(fpr_ci_upper, 8) if fpr_ci_upper is not None else None,
        "claim_1_tpr_at_target_fpr": round(positive_tpr, 8),
        "claim_1_tpr_ci_95_lower": round(tpr_ci_lower, 8) if tpr_ci_lower is not None else None,
        "claim_1_tpr_ci_95_upper": round(tpr_ci_upper, 8) if tpr_ci_upper is not None else None,
        "claim_2_path_evidence_independent_gain_decision": "PASS" if claim2_pass else "FAIL",
        "claim_2_paired_comparison_count": len(paired),
        "claim_2_paired_score_gain_mean": round(gain_mean, 8),
        "claim_2_paired_score_gain_ci_95_lower": round(gain_ci_lower, 8) if gain_ci_lower is not None else None,
        "claim_2_paired_score_gain_ci_95_upper": round(gain_ci_upper, 8) if gain_ci_upper is not None else None,
        "claim_2_paired_detection_gain_mean": round(detection_gain_mean, 8),
        "claim_2_paired_detection_gain_ci_95_lower": round(detection_gain_ci_lower, 8)
        if detection_gain_ci_lower is not None
        else None,
        "claim_2_paired_detection_gain_ci_95_upper": round(detection_gain_ci_upper, 8)
        if detection_gain_ci_upper is not None
        else None,
        "claim_3_attacked_video_replay_posterior_decision": "PENDING_AUTHENTICATED_REPLAY_GATE",
        "target_fpr": target_fpr,
        "three_layer_mechanism_pre_replay_decision": "PASS" if claim1_pass and claim2_pass else "FAIL",
    }


def run_formal_flow_evidence(
    run_root: str | Path,
    prompt_suite_path: str | Path,
    config_path: str | Path,
    *,
    pipeline_loader: Any = _load_pipeline,
) -> dict[str, Any]:
    """执行完整 Flow evidence、真实 controls 与冻结 fixed-FPR 检测。"""

    run_root = Path(run_root)
    config = _read_json(config_path)
    prompt_map = _prompt_text_by_id(_read_json(prompt_suite_path))
    generation_records = _read_jsonl(run_root / "records" / "generation_records.jsonl")
    attack_records = _read_jsonl(run_root / "records" / "runtime_attack_records.jsonl")
    attack_records.extend(_read_jsonl(run_root / "records" / "cross_sample_adaptive_video_attack_records.jsonl"))
    successful_generation = [record for record in generation_records if record.get("generation_status") == "success"]
    clean_records = [record for record in successful_generation if record.get("sample_role") == "clean_negative"]
    ready_attacks = [record for record in attack_records if record.get("attack_runtime_status") == "ready"]
    if not ready_attacks:
        raise RuntimeError("缺少 ready runtime attack records, 不能执行正式 Flow replay")
    models = sorted({str(record.get("generation_model_id")) for record in ready_attacks if record.get("generation_model_id")})
    pipelines = {model_id: pipeline_loader(model_id) for model_id in models}
    all_prompts = list(dict.fromkeys(prompt_map.values()))
    evidence_records: list[dict[str, Any]] = []
    failure_records: list[dict[str, Any]] = []

    for source in ready_attacks:
        method_variant = str(source.get("method_variant") or "sstw_full_method")
        if method_variant not in FORMAL_METHOD_VARIANTS:
            continue
        try:
            prompt = prompt_map[str(source.get("prompt_id"))]
            wrong_prompt = next(value for value in all_prompts if value != prompt)
            pipeline = pipelines[str(source.get("generation_model_id"))]
            key_text = _generation_key(source)
            video_path = _resolve_video_path(run_root, source.get("attacked_video_path"), fallback_dir="attacked_videos")
            replay = run_wan_attacked_video_replay(
                pipeline,
                video_path,
                prompt=prompt,
                key_text=key_text,
            )
            payload = {
                **_base_record(source, sample_role="attacked_positive", method_variant=method_variant),
                "formal_flow_evidence_unit_id": build_stable_digest({
                    "trajectory_trace_id": source.get("trajectory_trace_id"),
                    "attack_name": source.get("attack_name"),
                    "method_variant": method_variant,
                }),
                "attacked_video_path": str(video_path),
                "attacked_video_sha256": source.get("attacked_video_sha256"),
                "replay_sampler_signature": _scheduler_signature(pipeline.scheduler),
                "authenticated_generation_time_grid_id": source.get("trajectory_time_grid_id"),
                "authenticated_generation_step_count": source.get("num_inference_steps"),
                "replay_prompt_digest": sha256(prompt.encode("utf-8")).hexdigest(),
                **build_flow_evidence_payload(replay, key_text=key_text, method_variant=method_variant),
                **_control_payload(
                    pipeline,
                    replay,
                    prompt=prompt,
                    wrong_prompt=wrong_prompt,
                    key_text=key_text,
                ),
            }
            evidence_records.append(with_flow_evidence_protocol_defaults(
                payload,
                trajectory_source_level="attacked_video_model_velocity_inversion_replay",
                flow_state_admissibility_status="pending_frozen_detector",
                claim_support_status="sstw_complete_flow_evidence_ready",
            ))
        except Exception as exc:  # pragma: no cover - 依赖真实 GPU、模型和视频文件
            failure_records.append({
                **_base_record(source, sample_role="attacked_positive", method_variant=method_variant),
                "formal_flow_evidence_status": "failed",
                "formal_flow_evidence_failure_reason": str(exc),
                "metric_status": "missing",
            })

    clean_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in clean_records:
        clean_by_split[str(record.get("split") or "test")].append(record)
    for split, sources in clean_by_split.items():
        trial_count = _clean_trial_count(config, split, len(sources))
        for source in sources:
            try:
                prompt = prompt_map[str(source.get("prompt_id"))]
                pipeline = pipelines.get(str(source.get("generation_model_id"))) or pipeline_loader(str(source.get("generation_model_id")))
                pipelines[str(source.get("generation_model_id"))] = pipeline
                video_path = _resolve_video_path(run_root, source.get("video_path"), fallback_dir="videos")
                base_replay = run_wan_attacked_video_replay(
                    pipeline,
                    video_path,
                    prompt=prompt,
                    key_text=f"{_generation_key(source)}::clean_replay_base",
                )
                primary = base_replay.replay_trajectories[base_replay.primary_replay_index]
                for method_variant in FORMAL_METHOD_VARIANTS:
                    for trial_index in range(trial_count):
                        trial_key = f"{_generation_key(source)}::clean_negative::{method_variant}::{trial_index:06d}"
                        endpoint = compute_endpoint_latent_evidence(base_replay.endpoint_latent, key_text=trial_key)
                        path = score_replay_trajectory_for_key(primary, base_replay.primary_schedule, key_text=trial_key)
                        payload = {
                            **_base_record(source, sample_role="clean_negative", method_variant=method_variant),
                            "formal_flow_evidence_unit_id": build_stable_digest({
                                "trajectory_trace_id": source.get("trajectory_trace_id"),
                                "method_variant": method_variant,
                                "clean_negative_trial_index": trial_index,
                            }),
                            "clean_negative_trial_index": trial_index,
                            "negative_family": f"clean_key_family_{trial_index % 4}",
                            "clean_negative_video_path": str(video_path),
                            **endpoint.as_dict(),
                            **path,
                            **base_replay.replay_uncertainty.as_dict(),
                            **base_replay.endpoint_metadata,
                            "formal_flow_evidence_level": FORMAL_FLOW_EVIDENCE_LEVEL,
                            "formal_flow_detector_input_contract": FORMAL_FLOW_DETECTOR_INPUT_CONTRACT,
                            "detector_key_digest": sha256(trial_key.encode("utf-8")).hexdigest(),
                            "path_endpoint_consistency": round(_path_endpoint_consistency(
                                endpoint.projection,
                                float(path.get("S_path_inv") or 0.0),
                            ), 8),
                            "time_grid_reliability": round(_time_grid_reliability(base_replay), 8),
                            "flow_phase": 0.5,
                            "trajectory_trace_used_for_score": False,
                            "metric_status": "measured_formal",
                        }
                        evidence_records.append(with_flow_evidence_protocol_defaults(
                            payload,
                            trajectory_source_level="clean_video_model_velocity_inversion_replay_key_trial",
                            flow_state_admissibility_status="pending_frozen_detector",
                            claim_support_status="sstw_clean_flow_evidence_ready",
                        ))
            except Exception as exc:  # pragma: no cover - 依赖真实 GPU、模型和视频文件
                failure_records.append({
                    **_base_record(source, sample_role="clean_negative", method_variant="all_variants"),
                    "formal_flow_evidence_status": "failed",
                    "formal_flow_evidence_failure_reason": str(exc),
                    "metric_status": "missing",
                })

    scored_records, threshold_records, calibrations = _score_records_with_frozen_calibration(
        evidence_records,
        target_fpr=float(config["target_fpr"]),
    )
    paired_path_records = _paired_path_gain_records(scored_records, calibrations)
    mechanism_audit = _audit_three_layer_mechanism(
        scored_records,
        paired_path_records,
        target_fpr=float(config["target_fpr"]),
    )
    positive_records = [record for record in scored_records if record.get("sample_role") == "attacked_positive"]
    negative_records = [record for record in scored_records if record.get("sample_role") == "clean_negative"]
    required_variants = set(FORMAL_METHOD_VARIANTS)
    observed_variants = {str(record.get("method_variant")) for record in positive_records}
    claim3_records = [
        record for record in positive_records
        if record.get("method_variant") == "sstw_full_method"
        and record.get("replay_control_execution_status") == "measured_formal"
    ]
    audit = {
        "stage_id": "formal_flow_evidence_runner",
        "formal_flow_evidence_decision": "PASS" if (
            bool(positive_records)
            and bool(negative_records)
            and not failure_records
            and required_variants.issubset(observed_variants)
            and bool(claim3_records)
            and mechanism_audit["three_layer_mechanism_pre_replay_decision"] == "PASS"
        ) else "FAIL",
        "formal_flow_evidence_record_count": len(scored_records),
        "formal_flow_positive_record_count": len(positive_records),
        "formal_flow_clean_negative_record_count": len(negative_records),
        "formal_flow_failure_record_count": len(failure_records),
        "formal_flow_observed_method_variants": sorted(observed_variants),
        "formal_flow_missing_method_variants": sorted(required_variants - observed_variants),
        "formal_flow_threshold_record_count": len(threshold_records),
        "claim3_real_replay_record_count": len(claim3_records),
        "claim_1_velocity_constraint_detectable_watermark_decision": mechanism_audit["claim_1_velocity_constraint_detectable_watermark_decision"],
        "claim_2_path_evidence_independent_gain_decision": mechanism_audit["claim_2_path_evidence_independent_gain_decision"],
        "target_fpr": float(config["target_fpr"]),
        "test_time_threshold_update_blocked": True,
        "claim_support_status": "sstw_complete_paper_mechanism_ready" if not failure_records else "sstw_complete_paper_mechanism_blocked",
    }
    write_jsonl(run_root / "records" / "formal_flow_evidence_records.jsonl", scored_records)
    write_jsonl(run_root / "records" / "formal_flow_evidence_failure_records.jsonl", failure_records)
    write_jsonl(run_root / "records" / "runtime_detection_records.jsonl", positive_records)
    write_jsonl(run_root / "records" / "sstw_clean_negative_score_records.jsonl", negative_records)
    write_jsonl(run_root / "records" / "paired_path_evidence_gain_records.jsonl", paired_path_records)
    write_jsonl(run_root / "thresholds" / "formal_flow_detector_thresholds.jsonl", threshold_records)
    write_csv(run_root / "tables" / "formal_flow_detection_table.csv", scored_records)
    write_csv(run_root / "tables" / "runtime_detection_table.csv", positive_records)
    write_csv(run_root / "tables" / "sstw_clean_negative_score_table.csv", negative_records)
    write_json(run_root / "artifacts" / "runtime_detection_decision.json", {
        "runtime_detection_decision": audit["formal_flow_evidence_decision"],
        "runtime_detection_record_count": len(positive_records),
        "runtime_detection_ready_count": len(positive_records),
        "runtime_detection_formal_detector_ready_count": len(positive_records),
        "runtime_detection_evidence_level": FORMAL_FLOW_EVIDENCE_LEVEL,
        "claim_support_status": audit["claim_support_status"],
    })
    write_json(run_root / "artifacts" / "formal_flow_evidence_decision.json", audit)
    write_json(run_root / "artifacts" / "three_layer_mechanism_evidence_decision.json", mechanism_audit)
    write_csv(run_root / "tables" / "paired_path_evidence_gain_table.csv", paired_path_records)
    report = (
        "# Formal Flow Evidence and Runtime Detection Report\n\n"
        "该报告由攻击后视频的 Wan VAE endpoint、key-conditioned model velocity replay、"
        "多时间网格不确定性和冻结 fixed-FPR detector 自动生成。\n\n"
        f"- formal_flow_evidence_decision: {audit['formal_flow_evidence_decision']}\n"
        f"- formal_flow_positive_record_count: {audit['formal_flow_positive_record_count']}\n"
        f"- formal_flow_clean_negative_record_count: {audit['formal_flow_clean_negative_record_count']}\n"
        f"- claim_1_decision: {audit['claim_1_velocity_constraint_detectable_watermark_decision']}\n"
        f"- claim_2_decision: {audit['claim_2_path_evidence_independent_gain_decision']}\n"
        f"- claim3_real_replay_record_count: {audit['claim3_real_replay_record_count']}\n"
    )
    for report_name in ("formal_flow_evidence_report.md", "runtime_detection_report.md"):
        report_path = run_root / "reports" / report_name
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="执行 attacked video 的 SSTW 完整 Flow 证据检测。")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--prompt-suite-path", required=True)
    parser.add_argument("--config-path", required=True)
    args = parser.parse_args()
    payload = run_formal_flow_evidence(args.run_root, args.prompt_suite_path, args.config_path)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
