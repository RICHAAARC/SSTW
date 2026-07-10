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

from evaluation.statistics.clustered_inference import (
    clustered_binary_rate_interval,
    paired_cluster_difference_interval,
)
from evaluation.attacks.video_runtime_attack_protocol import load_protocol_config_with_shared_attack_protocol

from experiments.generative_video_model_probe.colab_runtime import _load_video_generation_pipeline, _select_dtype
from runtime.core.digest import build_stable_digest
from main.methods.state_space_watermark.endpoint_latent_detector import compute_endpoint_latent_evidence
from main.methods.state_space_watermark.formal_detector import (
    FORMAL_METHOD_VARIANTS,
    apply_frozen_flow_detector,
    fit_flow_evidence_calibration,
)
from main.methods.state_space_watermark.wan_flow_replay_backend import (
    WanFlowReplayResult,
    evaluate_fixed_wan_replay_hypothesis_for_key,
    run_wan_attacked_video_replay,
    run_wan_control_replay,
)
from main.methods.state_space_watermark.ltx_flow_replay_backend import (
    LTXFlowReplayResult,
    compute_ltx_endpoint_evidence_for_key,
    evaluate_fixed_ltx_replay_hypothesis_for_key,
    run_ltx_attacked_video_replay,
    run_ltx_control_replay,
)
from evaluation.protocol.flow_evidence_fields import with_flow_evidence_protocol_defaults
from evaluation.protocol.record_writer import write_json, write_jsonl
from evaluation.protocol.table_builder import write_csv


FORMAL_FLOW_EVIDENCE_LEVEL = "attacked_video_key_independent_inversion_hypothesis_replay"
FORMAL_FLOW_DETECTOR_INPUT_CONTRACT = "video_file_prompt_key_model_scheduler_and_frozen_calibration"


def _run_attacked_video_replay_for_model(
    pipeline: Any,
    video_path: str | Path,
    *,
    prompt: str,
    key_text: str,
) -> WanFlowReplayResult | LTXFlowReplayResult:
    """按 pipeline 家族分派真实 attacked-video replay, 不允许回退到代理分数。"""

    if "LTX" in type(pipeline).__name__.upper():
        return run_ltx_attacked_video_replay(
            pipeline,
            video_path,
            prompt=prompt,
            key_text=key_text,
        )
    return run_wan_attacked_video_replay(
        pipeline,
        video_path,
        prompt=prompt,
        key_text=key_text,
    )


def _compute_replay_endpoint_evidence_for_key(
    replay: WanFlowReplayResult | LTXFlowReplayResult,
    *,
    key_text: str,
) -> Any:
    """在模型对应的五维 VAE endpoint 坐标上计算同源 key 证据。"""

    if isinstance(replay, LTXFlowReplayResult):
        return compute_ltx_endpoint_evidence_for_key(replay, key_text=key_text)
    return compute_endpoint_latent_evidence(replay.endpoint_latent, key_text=key_text)


def _run_control_replay_for_model(
    pipeline: Any,
    replay: WanFlowReplayResult | LTXFlowReplayResult,
    *,
    prompt: str,
    key_text: str,
    num_inference_steps: int,
    scheduler: Any | None = None,
) -> tuple[Any, tuple[Any, ...], dict[str, float | int | None]]:
    """按模型家族执行相同定义的 wrong-key、wrong-prompt 或 wrong-sampler 对照。"""

    if isinstance(replay, LTXFlowReplayResult):
        return run_ltx_control_replay(
            pipeline,
            replay.endpoint_latent,
            latent_layout=replay.latent_layout,
            prompt=prompt,
            key_text=key_text,
            num_inference_steps=num_inference_steps,
            scheduler=scheduler,
        )
    return run_wan_control_replay(
        pipeline,
        replay.endpoint_latent,
        prompt=prompt,
        key_text=key_text,
        num_inference_steps=num_inference_steps,
        scheduler=scheduler,
    )


def _evaluate_fixed_replay_hypothesis_for_key(
    pipeline: Any,
    replay: WanFlowReplayResult | LTXFlowReplayResult,
    *,
    prompt: str,
    key_text: str,
) -> tuple[Any, dict[str, float | int | None]]:
    """在同一 key 无关固定反演路径上评估候选 key, 防止循环构造观测。"""

    if isinstance(replay, LTXFlowReplayResult):
        return evaluate_fixed_ltx_replay_hypothesis_for_key(
            pipeline,
            replay,
            prompt=prompt,
            key_text=key_text,
        )
    return evaluate_fixed_wan_replay_hypothesis_for_key(
        pipeline,
        replay,
        prompt=prompt,
        key_text=key_text,
    )


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


def _time_grid_reliability(result: WanFlowReplayResult | LTXFlowReplayResult) -> float:
    """根据多时间网格循环误差离散程度计算独立的 time-grid 可靠性。"""

    errors = [float(row.cycle_relative_error) for row in result.replay_trajectories]
    dispersion = pstdev(errors) if len(errors) > 1 else 0.0
    return math.exp(-max(0.0, dispersion))


def _scheduler_signature(scheduler: Any) -> str:
    """生成与 generation sketch 相同口径的 scheduler 配置签名。"""

    payload = json.dumps(dict(scheduler.config), ensure_ascii=False, sort_keys=True, default=str)
    return f"{type(scheduler).__name__}:{sha256(payload.encode('utf-8')).hexdigest()}"


def build_flow_evidence_payload(
    replay: WanFlowReplayResult | LTXFlowReplayResult,
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
    replay: WanFlowReplayResult | LTXFlowReplayResult,
    *,
    prompt: str,
    wrong_prompt: str,
    key_text: str,
) -> dict[str, Any]:
    """执行 wrong key、wrong prompt 与 wrong sampler/time-grid 真实对照。"""

    wrong_key = f"{key_text}::wrong_key_control"
    wrong_key_endpoint = _compute_replay_endpoint_evidence_for_key(replay, key_text=wrong_key)

    primary_steps = int(replay.replay_step_counts[replay.primary_replay_index])
    wrong_key_trajectory, _wrong_key_schedule, wrong_key_path = _run_control_replay_for_model(
        pipeline,
        replay,
        prompt=prompt,
        key_text=wrong_key,
        num_inference_steps=primary_steps,
    )
    wrong_prompt_trajectory, _wrong_prompt_schedule, wrong_prompt_path = _run_control_replay_for_model(
        pipeline,
        replay,
        prompt=wrong_prompt,
        key_text=key_text,
        num_inference_steps=primary_steps,
    )

    scheduler_class = type(pipeline.scheduler)
    scheduler_config = dict(pipeline.scheduler.config)
    if isinstance(replay, LTXFlowReplayResult):
        original_shift = float(scheduler_config.get("base_shift", 0.5))
        wrong_shift = original_shift + 0.25
        wrong_scheduler = scheduler_class.from_config(
            pipeline.scheduler.config,
            base_shift=wrong_shift,
            max_shift=float(scheduler_config.get("max_shift", 1.15)) + 0.25,
        )
    else:
        original_shift = float(scheduler_config.get("shift", 1.0))
        wrong_shift = original_shift + 1.0
        wrong_scheduler = scheduler_class.from_config(
            pipeline.scheduler.config,
            shift=wrong_shift,
        )
    wrong_sampler_trajectory, _wrong_sampler_schedule, wrong_sampler_path = _run_control_replay_for_model(
        pipeline,
        replay,
        prompt=prompt,
        key_text=key_text,
        num_inference_steps=max(2, primary_steps + 3),
        scheduler=wrong_scheduler,
    )
    def hypothesis_support(trajectory: Any) -> float:
        likelihood_probability = 1.0 / (
            1.0 + math.exp(-float(trajectory.replay_log_likelihood_ratio))
        )
        return likelihood_probability * math.exp(
            -max(0.0, float(trajectory.candidate_cycle_relative_error))
        )

    matched_path = float(replay.path_evidence.get("S_path_inv") or 0.0)
    matched_reliability = float(replay.replay_uncertainty.replay_reliability)
    matched_trajectory = replay.replay_trajectories[replay.primary_replay_index]
    matched_path_reliability_score = (
        (0.5 + 0.5 * matched_path)
        * matched_reliability
        * hypothesis_support(matched_trajectory)
    )
    wrong_key_path_reliability_score = (
        0.5 + 0.5 * float(wrong_key_path.get("S_path_inv") or 0.0)
    ) * hypothesis_support(wrong_key_trajectory)
    wrong_prompt_path_reliability_score = (
        0.5 + 0.5 * float(wrong_prompt_path.get("S_path_inv") or 0.0)
    ) * hypothesis_support(wrong_prompt_trajectory)
    wrong_sampler_path_reliability_score = (
        0.5 + 0.5 * float(wrong_sampler_path.get("S_path_inv") or 0.0)
    ) * hypothesis_support(wrong_sampler_trajectory)
    return {
        "wrong_key_endpoint_score": round(wrong_key_endpoint.score, 8),
        "wrong_key_S_path_inv": wrong_key_path.get("S_path_inv"),
        "wrong_key_replay_cycle_error": round(wrong_key_trajectory.cycle_relative_error, 8),
        "wrong_key_replay_log_likelihood_ratio": round(
            wrong_key_trajectory.replay_log_likelihood_ratio,
            8,
        ),
        "wrong_key_control_margin": round(
            (replay.endpoint_evidence.score + matched_path_reliability_score)
            - (wrong_key_endpoint.score + wrong_key_path_reliability_score),
            8,
        ),
        "wrong_prompt_replay_cycle_error": round(wrong_prompt_trajectory.cycle_relative_error, 8),
        "wrong_prompt_replay_log_likelihood_ratio": round(
            wrong_prompt_trajectory.replay_log_likelihood_ratio,
            8,
        ),
        "wrong_prompt_S_path_inv": wrong_prompt_path.get("S_path_inv"),
        "wrong_prompt_control_margin": round(
            matched_path_reliability_score - wrong_prompt_path_reliability_score,
            8,
        ),
        "wrong_sampler_replay_cycle_error": round(wrong_sampler_trajectory.cycle_relative_error, 8),
        "wrong_sampler_replay_log_likelihood_ratio": round(
            wrong_sampler_trajectory.replay_log_likelihood_ratio,
            8,
        ),
        "wrong_sampler_S_path_inv": wrong_sampler_path.get("S_path_inv"),
        "wrong_sampler_control_margin": round(
            matched_path_reliability_score - wrong_sampler_path_reliability_score,
            8,
        ),
        "wrong_sampler_control_shift": wrong_shift,
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

    statistical_cluster_id = build_stable_digest({
        "generation_model_id": source.get("generation_model_id"),
        "generation_model_family": source.get("generation_model_family"),
        "cross_model_role": source.get("cross_model_role"),
        "prompt_id": source.get("prompt_id"),
        "seed_id": source.get("seed_id"),
        "split": source.get("split"),
    })
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
        "source_video_cluster_id": source.get("trajectory_trace_id"),
        "statistical_cluster_id": statistical_cluster_id,
        "statistical_independent_unit": "source_video_prompt_seed",
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
    calibration_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in rows:
        if record.get("split") == "calibration":
            calibration_rows[(
                str(record.get("generation_model_id")),
                str(record.get("method_variant")),
            )].append(record)
    calibrations = {
        (model_id, variant): fit_flow_evidence_calibration(
            calibration_rows.get((model_id, variant), []),
            method_variant=variant,
            target_fpr=target_fpr,
        )
        for model_id in sorted({str(record.get("generation_model_id")) for record in rows})
        for variant in FORMAL_METHOD_VARIANTS
    }
    scored: list[dict[str, Any]] = []
    for record in rows:
        variant = str(record.get("method_variant"))
        model_id = str(record.get("generation_model_id"))
        detection = apply_frozen_flow_detector(record, calibrations[(model_id, variant)])
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
    threshold_records = [
        {
            "generation_model_id": model_id,
            "model_specific_calibration": True,
            **calibration.as_dict(),
        }
        for (model_id, _variant), calibration in calibrations.items()
    ]
    return scored, threshold_records, calibrations


def _paired_path_gain_records(
    scored_records: Iterable[Mapping[str, Any]],
    calibrations: Mapping[tuple[str, str], Any],
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
        endpoint_only = apply_frozen_flow_detector(
            record,
            calibrations[(str(record.get("generation_model_id")), "endpoint_only_control")],
        )
        full_score = float(record["S_final_conservative"])
        endpoint_score = float(endpoint_only["S_final_conservative"])
        rows.append({
            "record_version": "paired_path_evidence_gain_v1",
            "generation_model_id": record.get("generation_model_id"),
            "cross_model_role": record.get("cross_model_role"),
            "prompt_id": record.get("prompt_id"),
            "seed_id": record.get("seed_id"),
            "trajectory_trace_id": record.get("trajectory_trace_id"),
            "attack_name": record.get("attack_name"),
            "statistical_cluster_id": record.get("statistical_cluster_id"),
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


def _paired_velocity_causal_records(
    scored_records: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """在相同 prompt/seed/attack 上比较完整方法与无速度约束对照。"""

    positives = [
        record
        for record in scored_records
        if record.get("sample_role") == "attacked_positive"
        and record.get("split") == "test"
        and record.get("method_variant")
        in {"sstw_full_method", "without_velocity_constraint"}
    ]
    by_identity: dict[tuple[str, str, str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for record in positives:
        identity = (
            str(record.get("generation_model_id") or ""),
            str(record.get("prompt_id") or ""),
            str(record.get("seed_id") or ""),
            str(record.get("attack_name") or ""),
        )
        by_identity[identity][str(record.get("method_variant"))] = record
    rows: list[dict[str, Any]] = []
    for identity, variants in by_identity.items():
        full = variants.get("sstw_full_method")
        control = variants.get("without_velocity_constraint")
        if full is None or control is None:
            continue
        full_score = float(full.get("S_final_conservative") or 0.0)
        control_score = float(control.get("S_final_conservative") or 0.0)
        rows.append({
            "record_version": "paired_velocity_causal_evidence_v1",
            "generation_model_id": identity[0],
            "cross_model_role": full.get("cross_model_role"),
            "prompt_id": identity[1],
            "seed_id": identity[2],
            "attack_name": identity[3],
            "statistical_cluster_id": full.get("statistical_cluster_id"),
            "target_fpr": full.get("target_fpr"),
            "paired_full_method_score": full_score,
            "paired_without_velocity_constraint_score": control_score,
            "paired_velocity_causal_score_gain": round(full_score - control_score, 8),
            "paired_full_method_decision": bool(full.get("decision")),
            "paired_without_velocity_constraint_decision": bool(control.get("decision")),
            "paired_velocity_causal_detection_gain": (
                int(bool(full.get("decision")))
                - int(bool(control.get("decision")))
            ),
            "metric_status": "measured_formal",
            "claim_support_status": "claim1_same_unit_velocity_constraint_causal_evidence",
        })
    return rows


def _audit_three_layer_mechanism(
    scored_records: Iterable[Mapping[str, Any]],
    paired_path_records: Iterable[Mapping[str, Any]],
    paired_velocity_records: Iterable[Mapping[str, Any]],
    *,
    target_fpr: float,
) -> dict[str, Any]:
    """审计 Claim-1 与 Claim-2, Claim-3 的最终认证由 replay gate 继续完成。"""

    rows = [
        record
        for record in scored_records
        if record.get("cross_model_role") != "cross_model_validation_model"
    ]
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
    fpr_estimate = clustered_binary_rate_interval(
        full_test_negative,
        outcome_field="decision",
        purpose="claim1_heldout_fpr",
    )
    tpr_estimate = clustered_binary_rate_interval(
        full_positive,
        outcome_field="decision",
        purpose="claim1_heldout_tpr",
    )
    paired = [
        record
        for record in paired_path_records
        if record.get("cross_model_role") != "cross_model_validation_model"
    ]
    path_score_gain = paired_cluster_difference_interval(
        paired,
        difference_field="paired_path_evidence_score_gain",
        purpose="claim2_path_score_gain",
    )
    path_detection_gain = paired_cluster_difference_interval(
        paired,
        difference_field="paired_path_evidence_detection_gain",
        purpose="claim2_path_detection_gain",
    )
    velocity_paired = [
        record
        for record in paired_velocity_records
        if record.get("cross_model_role") != "cross_model_validation_model"
    ]
    velocity_score_gain = paired_cluster_difference_interval(
        velocity_paired,
        difference_field="paired_velocity_causal_score_gain",
        purpose="claim1_velocity_score_gain",
    )
    velocity_detection_gain = paired_cluster_difference_interval(
        velocity_paired,
        difference_field="paired_velocity_causal_detection_gain",
        purpose="claim1_velocity_detection_gain",
    )
    claim1_pass = (
        bool(full_positive)
        and bool(full_test_negative)
        and bool(velocity_paired)
        and fpr_estimate.estimate <= target_fpr
        and tpr_estimate.confidence_interval_lower > target_fpr
        and velocity_score_gain.confidence_interval_lower > 0.0
        and velocity_detection_gain.confidence_interval_lower > 0.0
    )
    claim2_pass = (
        bool(paired)
        and path_score_gain.confidence_interval_lower > 0.0
        and path_detection_gain.confidence_interval_lower > 0.0
    )
    return {
        "stage_id": "sstw_three_layer_mechanism_evidence",
        "claim_1_velocity_constraint_detectable_watermark_decision": "PASS" if claim1_pass else "FAIL",
        "claim_1_heldout_positive_count": len(full_positive),
        "claim_1_heldout_negative_count": len(full_test_negative),
        "claim_1_empirical_fpr": round(fpr_estimate.estimate, 8),
        "claim_1_empirical_fpr_ci_95_lower": round(fpr_estimate.confidence_interval_lower, 8),
        "claim_1_empirical_fpr_ci_95_upper": round(fpr_estimate.confidence_interval_upper, 8),
        "claim_1_fpr_statistical_cluster_count": fpr_estimate.cluster_count,
        "claim_1_tpr_at_target_fpr": round(tpr_estimate.estimate, 8),
        "claim_1_tpr_ci_95_lower": round(tpr_estimate.confidence_interval_lower, 8),
        "claim_1_tpr_ci_95_upper": round(tpr_estimate.confidence_interval_upper, 8),
        "claim_1_tpr_statistical_cluster_count": tpr_estimate.cluster_count,
        "claim_1_velocity_causal_pair_count": len(velocity_paired),
        "claim_1_velocity_causal_score_gain_mean": round(velocity_score_gain.estimate, 8),
        "claim_1_velocity_causal_score_gain_ci_95_lower": round(velocity_score_gain.confidence_interval_lower, 8),
        "claim_1_velocity_causal_score_gain_ci_95_upper": round(velocity_score_gain.confidence_interval_upper, 8),
        "claim_1_velocity_causal_detection_gain_mean": round(velocity_detection_gain.estimate, 8),
        "claim_1_velocity_causal_detection_gain_ci_95_lower": round(velocity_detection_gain.confidence_interval_lower, 8),
        "claim_1_velocity_causal_detection_gain_ci_95_upper": round(velocity_detection_gain.confidence_interval_upper, 8),
        "claim_2_path_evidence_independent_gain_decision": "PASS" if claim2_pass else "FAIL",
        "claim_2_paired_comparison_count": len(paired),
        "claim_2_paired_score_gain_mean": round(path_score_gain.estimate, 8),
        "claim_2_paired_score_gain_ci_95_lower": round(path_score_gain.confidence_interval_lower, 8),
        "claim_2_paired_score_gain_ci_95_upper": round(path_score_gain.confidence_interval_upper, 8),
        "claim_2_paired_detection_gain_mean": round(path_detection_gain.estimate, 8),
        "claim_2_paired_detection_gain_ci_95_lower": round(path_detection_gain.confidence_interval_lower, 8),
        "claim_2_paired_detection_gain_ci_95_upper": round(path_detection_gain.confidence_interval_upper, 8),
        "claim_3_attacked_video_replay_posterior_decision": "PENDING_AUTHENTICATED_REPLAY_GATE",
        "target_fpr": target_fpr,
        "three_layer_mechanism_pre_replay_decision": "PASS" if claim1_pass and claim2_pass else "FAIL",
    }


def _audit_cross_model_generalization(
    scored_records: Iterable[Mapping[str, Any]],
    paired_path_records: Iterable[Mapping[str, Any]],
    paired_velocity_records: Iterable[Mapping[str, Any]],
    *,
    target_fpr: float,
) -> dict[str, Any]:
    """审计资源受限跨模型子集是否复现三层机制方向, 不冒充主固定 FPR 结论。"""

    cross_rows = [
        record
        for record in scored_records
        if record.get("cross_model_role") == "cross_model_validation_model"
    ]
    if not cross_rows:
        return {
            "cross_model_generalization_decision": "NOT_CONFIGURED",
            "cross_model_generalization_claim_scope": "supportive_not_primary_fixed_fpr_closure",
            "cross_model_generalization_model_ids": [],
            "cross_model_generalization_record_count": 0,
        }
    model_ids = sorted({str(record.get("generation_model_id")) for record in cross_rows})
    per_model: list[dict[str, Any]] = []
    for model_id in model_ids:
        model_rows = [record for record in cross_rows if str(record.get("generation_model_id")) == model_id]
        positives = [
            record
            for record in model_rows
            if record.get("method_variant") == "sstw_full_method"
            and record.get("sample_role") == "attacked_positive"
            and record.get("split") == "test"
        ]
        negatives = [
            record
            for record in model_rows
            if record.get("method_variant") == "sstw_full_method"
            and record.get("sample_role") == "clean_negative"
            and record.get("split") == "test"
        ]
        paths = [
            record
            for record in paired_path_records
            if str(record.get("generation_model_id")) == model_id
            and record.get("cross_model_role") == "cross_model_validation_model"
        ]
        velocities = [
            record
            for record in paired_velocity_records
            if str(record.get("generation_model_id")) == model_id
            and record.get("cross_model_role") == "cross_model_validation_model"
        ]
        fpr = clustered_binary_rate_interval(
            negatives,
            outcome_field="decision",
            cluster_field="statistical_cluster_id",
            purpose=f"cross_model_fpr::{model_id}",
        )
        tpr = clustered_binary_rate_interval(
            positives,
            outcome_field="decision",
            cluster_field="statistical_cluster_id",
            purpose=f"cross_model_tpr::{model_id}",
        )
        path_gain = paired_cluster_difference_interval(
            paths,
            difference_field="paired_path_evidence_score_gain",
            purpose=f"cross_model_path_gain::{model_id}",
        )
        velocity_gain = paired_cluster_difference_interval(
            velocities,
            difference_field="paired_velocity_causal_score_gain",
            purpose=f"cross_model_velocity_gain::{model_id}",
        )
        replay_count = sum(
            record.get("replay_control_execution_status") == "measured_formal"
            for record in positives
        )
        model_pass = (
            bool(positives)
            and bool(negatives)
            and bool(paths)
            and bool(velocities)
            and fpr.estimate <= target_fpr
            and tpr.estimate > fpr.estimate
            and path_gain.estimate > 0.0
            and velocity_gain.estimate > 0.0
            and replay_count == len(positives)
        )
        per_model.append({
            "generation_model_id": model_id,
            "cross_model_generalization_model_decision": "PASS" if model_pass else "FAIL",
            "cross_model_test_positive_cluster_count": tpr.cluster_count,
            "cross_model_test_negative_cluster_count": fpr.cluster_count,
            "cross_model_test_tpr": round(tpr.estimate, 8),
            "cross_model_test_fpr": round(fpr.estimate, 8),
            "cross_model_test_fpr_ci_95_upper": round(fpr.confidence_interval_upper, 8),
            "cross_model_path_pair_count": len(paths),
            "cross_model_path_score_gain_mean": round(path_gain.estimate, 8),
            "cross_model_velocity_pair_count": len(velocities),
            "cross_model_velocity_score_gain_mean": round(velocity_gain.estimate, 8),
            "cross_model_replay_control_record_count": replay_count,
        })
    return {
        "cross_model_generalization_decision": (
            "PASS"
            if per_model and all(row["cross_model_generalization_model_decision"] == "PASS" for row in per_model)
            else "FAIL"
        ),
        "cross_model_generalization_claim_scope": "supportive_not_primary_fixed_fpr_closure",
        "cross_model_generalization_model_ids": model_ids,
        "cross_model_generalization_record_count": len(cross_rows),
        "cross_model_generalization_per_model": per_model,
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
    config = load_protocol_config_with_shared_attack_protocol(config_path)
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
            replay = _run_attacked_video_replay_for_model(
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
        for source_index, source in enumerate(sources):
            try:
                prompt = prompt_map[str(source.get("prompt_id"))]
                pipeline = pipelines.get(str(source.get("generation_model_id"))) or pipeline_loader(str(source.get("generation_model_id")))
                pipelines[str(source.get("generation_model_id"))] = pipeline
                video_path = _resolve_video_path(run_root, source.get("video_path"), fallback_dir="videos")
                base_replay = _run_attacked_video_replay_for_model(
                    pipeline,
                    video_path,
                    prompt=prompt,
                    key_text=f"{_generation_key(source)}::clean_replay_base",
                )
                for method_variant in FORMAL_METHOD_VARIANTS:
                    for trial_index in range(trial_count):
                        trial_key = f"{_generation_key(source)}::clean_negative::{method_variant}::{trial_index:06d}"
                        endpoint = _compute_replay_endpoint_evidence_for_key(
                            base_replay,
                            key_text=trial_key,
                        )
                        hypothesis, path = _evaluate_fixed_replay_hypothesis_for_key(
                            pipeline,
                            base_replay,
                            prompt=prompt,
                            key_text=trial_key,
                        )
                        trial_reliability = math.exp(
                            -max(0.0, float(hypothesis.candidate_cycle_relative_error))
                        )
                        payload = {
                            **_base_record(source, sample_role="clean_negative", method_variant=method_variant),
                            "formal_flow_evidence_unit_id": build_stable_digest({
                                "trajectory_trace_id": source.get("trajectory_trace_id"),
                                "method_variant": method_variant,
                                "clean_negative_trial_index": trial_index,
                            }),
                            "clean_negative_trial_index": trial_index,
                            "statistical_within_cluster_trial_index": trial_index,
                            "negative_family": f"clean_key_family_{(source_index + trial_index) % 4}",
                            "clean_negative_video_path": str(video_path),
                            **endpoint.as_dict(),
                            **path,
                            "replay_inversion_status": "ready",
                            "replay_cycle_error_mean": round(hypothesis.candidate_cycle_relative_error, 8),
                            "replay_cycle_error_maximum": round(hypothesis.candidate_cycle_relative_error, 8),
                            "replay_null_cycle_error_mean": round(hypothesis.null_cycle_relative_error, 8),
                            "replay_log_likelihood_ratio_mean": round(hypothesis.replay_log_likelihood_ratio, 8),
                            "replay_log_likelihood_ratio_standard_deviation": 0.0,
                            "replay_endpoint_ensemble_variance": 0.0,
                            "replay_uncertainty_mean": round(1.0 - trial_reliability, 8),
                            "replay_reliability_weight": round(trial_reliability, 8),
                            "replay_ensemble_count": 1,
                            "replay_trajectory_source": "clean_video_fixed_key_independent_inversion_candidate_key_hypothesis",
                            **base_replay.endpoint_metadata,
                            "formal_flow_evidence_level": FORMAL_FLOW_EVIDENCE_LEVEL,
                            "formal_flow_detector_input_contract": FORMAL_FLOW_DETECTOR_INPUT_CONTRACT,
                            "detector_key_digest": sha256(trial_key.encode("utf-8")).hexdigest(),
                            "path_endpoint_consistency": round(_path_endpoint_consistency(
                                endpoint.projection,
                                float(path.get("S_path_inv") or 0.0),
                            ), 8),
                            "time_grid_reliability": round(trial_reliability, 8),
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
    paired_velocity_records = _paired_velocity_causal_records(scored_records)
    mechanism_audit = _audit_three_layer_mechanism(
        scored_records,
        paired_path_records,
        paired_velocity_records,
        target_fpr=float(config["target_fpr"]),
    )
    cross_model_audit = _audit_cross_model_generalization(
        scored_records,
        paired_path_records,
        paired_velocity_records,
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
    posterior_calibration_failures = [
        {
            "method_variant": record.get("method_variant"),
            "posterior_calibration_brier_score": record.get("posterior_calibration_brier_score"),
            "posterior_calibration_expected_calibration_error": record.get(
                "posterior_calibration_expected_calibration_error"
            ),
            "posterior_calibration_group_count": record.get("posterior_calibration_group_count"),
        }
        for record in threshold_records
        if float(
            record["posterior_calibration_brier_score"]
            if record.get("posterior_calibration_brier_score") is not None
            else math.inf
        )
        > float(config.get("maximum_posterior_brier_score", 0.25))
        or float(
            record["posterior_calibration_expected_calibration_error"]
            if record.get("posterior_calibration_expected_calibration_error") is not None
            else math.inf
        )
        > float(config.get("maximum_posterior_expected_calibration_error", 0.1))
        or int(record.get("posterior_calibration_group_count") or 0)
        < int(config.get("minimum_posterior_calibration_group_count", 2))
    ]
    audit = {
        "stage_id": "formal_flow_evidence_runner",
        "formal_flow_evidence_decision": "PASS" if (
            bool(positive_records)
            and bool(negative_records)
            and not failure_records
            and required_variants.issubset(observed_variants)
            and bool(claim3_records)
            and not posterior_calibration_failures
            and mechanism_audit["three_layer_mechanism_pre_replay_decision"] == "PASS"
            and cross_model_audit["cross_model_generalization_decision"] in {"PASS", "NOT_CONFIGURED"}
        ) else "FAIL",
        "formal_flow_evidence_record_count": len(scored_records),
        "formal_flow_positive_record_count": len(positive_records),
        "formal_flow_clean_negative_record_count": len(negative_records),
        "formal_flow_failure_record_count": len(failure_records),
        "formal_flow_observed_method_variants": sorted(observed_variants),
        "formal_flow_missing_method_variants": sorted(required_variants - observed_variants),
        "formal_flow_threshold_record_count": len(threshold_records),
        "claim3_real_replay_record_count": len(claim3_records),
        "posterior_probability_calibration_decision": (
            "PASS" if not posterior_calibration_failures else "FAIL"
        ),
        "posterior_probability_calibration_failures": posterior_calibration_failures,
        "claim_1_velocity_constraint_detectable_watermark_decision": mechanism_audit["claim_1_velocity_constraint_detectable_watermark_decision"],
        "claim_2_path_evidence_independent_gain_decision": mechanism_audit["claim_2_path_evidence_independent_gain_decision"],
        "cross_model_generalization_decision": cross_model_audit["cross_model_generalization_decision"],
        "cross_model_generalization_model_ids": cross_model_audit["cross_model_generalization_model_ids"],
        "cross_model_generalization_record_count": cross_model_audit["cross_model_generalization_record_count"],
        "target_fpr": float(config["target_fpr"]),
        "test_time_threshold_update_blocked": True,
        "claim_support_status": "sstw_complete_paper_mechanism_ready" if not failure_records else "sstw_complete_paper_mechanism_blocked",
    }
    write_jsonl(run_root / "records" / "formal_flow_evidence_records.jsonl", scored_records)
    write_jsonl(run_root / "records" / "formal_flow_evidence_failure_records.jsonl", failure_records)
    write_jsonl(run_root / "records" / "runtime_detection_records.jsonl", positive_records)
    write_jsonl(run_root / "records" / "sstw_clean_negative_score_records.jsonl", negative_records)
    write_jsonl(run_root / "records" / "paired_path_evidence_gain_records.jsonl", paired_path_records)
    write_jsonl(
        run_root / "records" / "paired_velocity_causal_evidence_records.jsonl",
        paired_velocity_records,
    )
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
    write_json(run_root / "artifacts" / "cross_model_generalization_decision.json", cross_model_audit)
    write_csv(
        run_root / "tables" / "cross_model_generalization_table.csv",
        cross_model_audit.get("cross_model_generalization_per_model") or [],
    )
    write_csv(run_root / "tables" / "paired_path_evidence_gain_table.csv", paired_path_records)
    write_csv(
        run_root / "tables" / "paired_velocity_causal_evidence_table.csv",
        paired_velocity_records,
    )
    report = (
        "# Formal Flow Evidence and Runtime Detection Report\n\n"
        "该报告由攻击后视频对应模型的真实 VAE endpoint、key-conditioned model velocity replay、"
        "多时间网格不确定性和冻结 fixed-FPR detector 自动生成。\n\n"
        f"- formal_flow_evidence_decision: {audit['formal_flow_evidence_decision']}\n"
        f"- formal_flow_positive_record_count: {audit['formal_flow_positive_record_count']}\n"
        f"- formal_flow_clean_negative_record_count: {audit['formal_flow_clean_negative_record_count']}\n"
        f"- claim_1_decision: {audit['claim_1_velocity_constraint_detectable_watermark_decision']}\n"
        f"- claim_2_decision: {audit['claim_2_path_evidence_independent_gain_decision']}\n"
        f"- claim3_real_replay_record_count: {audit['claim3_real_replay_record_count']}\n"
        f"- cross_model_generalization_decision: {audit['cross_model_generalization_decision']}\n"
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
