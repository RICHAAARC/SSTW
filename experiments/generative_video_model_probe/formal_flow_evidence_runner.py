"""从真实 attacked video 构建 endpoint、path、replay 与固定 FPR 检测记录。"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from statistics import pstdev
from typing import Any, Iterable, Mapping

from evaluation.statistics.clustered_inference import (
    clustered_binary_any_rate_interval,
    clustered_binary_rate_interval,
    paired_cluster_difference_interval,
)
from evaluation.attacks.video_runtime_attack_protocol import load_protocol_config_with_shared_attack_protocol

from experiments.generative_video_model_probe.colab_runtime import (
    _load_video_generation_pipeline,
    _select_dtype,
    validate_generation_model_provenance,
)
from runtime.core.digest import build_stable_digest
from main.methods.state_space_watermark.endpoint_latent_detector import compute_endpoint_latent_evidence
from main.methods.state_space_watermark.flow_tubelet_key_code import (
    FlowTubeletKeyContext,
    build_flow_tubelet_key_direction_like,
    flow_tubelet_key_context_digest,
)
from main.methods.state_space_watermark.flow_velocity_runtime import (
    normalized_flow_phase_from_sigma_interval,
)
from main.methods.state_space_watermark.path_observation import compute_path_step_observation
from main.methods.state_space_watermark.replay_inversion import (
    REPLAY_GAUSSIAN_LIKELIHOOD_MODEL_ID,
    ReplayGaussianLikelihoodConfig,
    ReplayTrajectory,
    fit_replay_gaussian_likelihood_config,
    gaussian_replay_residual_likelihood,
    replay_step_reliability_weight,
)
from main.methods.state_space_watermark.formal_detector import (
    FLOW_STATE_POSTERIOR_SCORE_SOURCE,
)
from experiments.generative_video_model_probe.formal_method_variants import (
    CLAIM2_PATH_NESTED_ABLATION_VARIANT,
    DETECTOR_ONLY_METHOD_VARIANTS,
    FORMAL_DETECTOR_VARIANTS,
    FORMAL_METHOD_VARIANTS,
    GENERATION_METHOD_VARIANTS,
    apply_frozen_flow_detector,
    fit_flow_evidence_calibration,
    frozen_flow_detector_calibration_artifact,
)
from experiments.generative_video_model_probe.heldout_posterior_calibration import (
    audit_heldout_posterior_calibration_records,
    build_heldout_posterior_calibration_records,
    write_heldout_posterior_calibration_artifacts,
)
from main.methods.state_space_watermark.flow_state_posterior import (
    FLOW_STATE_ADMISSIBILITY_THRESHOLD_NAMES,
    FLOW_STATE_POSTERIOR_CONTRACT_VERSION,
    FLOW_STATE_POSTERIOR_MODEL_TYPE,
)
from main.methods.state_space_watermark.watermark_key_derivation import (
    WATERMARK_KEY_DERIVATION_ID,
    derive_watermark_key_text,
    derive_wrong_key_control_text,
)
from main.methods.state_space_watermark.wan_flow_replay_backend import (
    WanFlowReplayResult,
    compute_wan_endpoint_evidence_for_key,
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
FORMAL_NEGATIVE_HYPOTHESIS_FAMILIES = frozenset({
    "clean_unwatermarked_candidate_key_hypothesis",
    "watermarked_video_wrong_key_hypothesis",
    "watermarked_video_wrong_prompt_hypothesis",
    "watermarked_video_wrong_sampler_time_grid_hypothesis",
})


def _run_attacked_video_replay_for_model(
    pipeline: Any,
    video_path: str | Path,
    *,
    prompt: str,
    key_text: str,
    key_context: FlowTubeletKeyContext,
    likelihood_config: ReplayGaussianLikelihoodConfig,
    replay_step_counts: tuple[int, ...] = (16, 20, 24),
) -> WanFlowReplayResult | LTXFlowReplayResult:
    """按 pipeline 家族分派真实 attacked-video replay, 不允许回退到代理分数。"""

    if "LTX" in type(pipeline).__name__.upper():
        return run_ltx_attacked_video_replay(
            pipeline,
            video_path,
            prompt=prompt,
            key_text=key_text,
            key_context=key_context,
            likelihood_config=likelihood_config,
            replay_step_counts=replay_step_counts,
        )
    return run_wan_attacked_video_replay(
        pipeline,
        video_path,
        prompt=prompt,
        key_text=key_text,
        key_context=key_context,
        likelihood_config=likelihood_config,
        replay_step_counts=replay_step_counts,
    )


def _compute_replay_endpoint_evidence_for_key(
    replay: WanFlowReplayResult | LTXFlowReplayResult,
    *,
    key_text: str,
    key_context: FlowTubeletKeyContext | None = None,
) -> Any:
    """在模型对应的五维 VAE endpoint 坐标上计算同源 key 证据。"""

    if isinstance(replay, LTXFlowReplayResult):
        return compute_ltx_endpoint_evidence_for_key(
            replay,
            key_text=key_text,
            key_context=key_context,
        )
    return compute_wan_endpoint_evidence_for_key(
        replay,
        key_text=key_text,
        key_context=key_context,
    )


def _run_control_replay_for_model(
    pipeline: Any,
    replay: WanFlowReplayResult | LTXFlowReplayResult,
    *,
    prompt: str,
    key_text: str,
    key_context: FlowTubeletKeyContext,
    num_inference_steps: int,
    scheduler: Any | None = None,
    fixed_trajectory: ReplayTrajectory | None = None,
) -> tuple[Any, tuple[Any, ...], dict[str, float | int | None]]:
    """按模型家族执行相同定义的 wrong-key、wrong-prompt 或 wrong-sampler 对照。"""

    if isinstance(replay, LTXFlowReplayResult):
        return run_ltx_control_replay(
            pipeline,
            replay.endpoint_latent,
            latent_layout=replay.latent_layout,
            prompt=prompt,
            key_text=key_text,
            key_context=key_context,
            num_inference_steps=num_inference_steps,
            scheduler=scheduler,
            fixed_trajectory=fixed_trajectory,
            likelihood_config=replay.replay_likelihood_config,
        )
    return run_wan_control_replay(
        pipeline,
        replay.endpoint_latent,
        prompt=prompt,
        key_text=key_text,
        key_context=key_context,
        num_inference_steps=num_inference_steps,
        scheduler=scheduler,
        fixed_trajectory=fixed_trajectory,
        likelihood_config=replay.replay_likelihood_config,
    )


def _evaluate_fixed_replay_hypothesis_for_key(
    pipeline: Any,
    replay: WanFlowReplayResult | LTXFlowReplayResult,
    *,
    prompt: str,
    key_text: str,
    key_context: FlowTubeletKeyContext,
) -> tuple[Any, dict[str, float | int | None]]:
    """在同一 key 无关固定反演路径上评估候选 key, 防止循环构造观测。"""

    if isinstance(replay, LTXFlowReplayResult):
        return evaluate_fixed_ltx_replay_hypothesis_for_key(
            pipeline,
            replay,
            prompt=prompt,
            key_text=key_text,
            key_context=key_context,
        )
    return evaluate_fixed_wan_replay_hypothesis_for_key(
        pipeline,
        replay,
        prompt=prompt,
        key_text=key_text,
        key_context=key_context,
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


def _owner_key_context(record: Mapping[str, Any]) -> tuple[bytes, str]:
    """读取所有者密钥，并核对 record 中公开的派生算法和 key ID。"""

    authentication_key = (
        os.environ.get("SSTW_TRAJECTORY_AUTHENTICATION_KEY") or ""
    ).encode("utf-8")
    key_id = (os.environ.get("SSTW_TRAJECTORY_AUTHENTICATION_KEY_ID") or "").strip()
    if len(authentication_key) < 32 or not key_id:
        raise RuntimeError(
            "正式检测要求至少32字节的 SSTW 所有者密钥和非空 key ID"
        )
    record_derivation_id = str(record.get("watermark_key_derivation_id") or "")
    if record_derivation_id and record_derivation_id != WATERMARK_KEY_DERIVATION_ID:
        raise RuntimeError("generation record 的水印 key 派生算法与当前实现不一致")
    record_key_id = str(record.get("watermark_key_id") or "").strip()
    if record_key_id and record_key_id != key_id:
        raise RuntimeError("generation record 的水印 key ID 与检测密钥不一致")
    return authentication_key, key_id


def _generation_key(
    record: Mapping[str, Any],
    *,
    extra_context: Mapping[str, Any] | None = None,
) -> str:
    """使用所有者秘密和公开生成上下文复算与嵌入阶段一致的水印 key。"""

    authentication_key, key_id = _owner_key_context(record)
    return derive_watermark_key_text(
        authentication_key,
        key_id=key_id,
        generation_model_id=str(record.get("generation_model_id") or ""),
        prompt_id=str(record.get("prompt_id") or ""),
        seed_id=str(record.get("seed_id") or ""),
        extra_context=extra_context,
    )


def _wrong_owner_generation_key(record: Mapping[str, Any]) -> str:
    """使用域分离的错误所有者秘密构造真正独立的 wrong-key 对照。"""

    authentication_key, key_id = _owner_key_context(record)
    return derive_wrong_key_control_text(
        authentication_key,
        key_id=key_id,
        generation_model_id=str(record.get("generation_model_id") or ""),
        prompt_id=str(record.get("prompt_id") or ""),
        seed_id=str(record.get("seed_id") or ""),
    )


def _path_endpoint_consistency(endpoint_projection: float, path_projection: float) -> float:
    """比较 endpoint 与积分路径在同一 key 方向上的一致性。"""

    return max(0.0, min(1.0, 1.0 - abs(float(endpoint_projection) - float(path_projection))))


def _time_grid_reliability(result: WanFlowReplayResult | LTXFlowReplayResult) -> float:
    """根据多时间网格循环误差离散程度计算独立的 time-grid 可靠性。"""

    errors = [
        float(row.candidate_cycle_relative_error)
        for row in result.replay_trajectories
    ]
    dispersion = pstdev(errors) if len(errors) > 1 else 0.0
    return math.exp(-max(0.0, dispersion))


def _scheduler_signature(scheduler: Any) -> str:
    """生成与 generation sketch 相同口径的 scheduler 配置签名。"""

    payload = json.dumps(dict(scheduler.config), ensure_ascii=False, sort_keys=True, default=str)
    return f"{type(scheduler).__name__}:{sha256(payload.encode('utf-8')).hexdigest()}"


def _flow_key_context(prompt: str, scheduler: Any) -> FlowTubeletKeyContext:
    """按生成阶段同一规则重建 prompt 与 sampler 联合 key context。"""

    return FlowTubeletKeyContext(
        prompt_digest=sha256(prompt.encode("utf-8")).hexdigest(),
        sampler_signature=_scheduler_signature(scheduler),
    )


def _validated_flow_key_context(
    source: Mapping[str, Any],
    *,
    prompt: str,
    scheduler: Any,
) -> FlowTubeletKeyContext:
    """重建并核验 generation 绑定的 tubelet context，防止检测期漂移。"""

    context = _flow_key_context(prompt, scheduler)
    observed_digest = str(source.get("flow_tubelet_key_context_digest") or "")
    expected_digest = flow_tubelet_key_context_digest(context)
    if not observed_digest:
        raise RuntimeError("正式 replay 输入缺少 flow tubelet key context 摘要")
    if observed_digest != expected_digest:
        raise RuntimeError(
            "正式 replay 的 prompt/sampler context 与生成记录不一致: "
            f"expected={expected_digest}, observed={observed_digest}"
        )
    return context


def _replay_native_key_direction(
    replay: WanFlowReplayResult | LTXFlowReplayResult,
    *,
    key_text: str,
    key_context: FlowTubeletKeyContext,
    flow_phase: float,
) -> Any:
    """在模型原生 replay 布局中构造同一 phase 的联合 tubelet code。"""

    reference = replay.replay_trajectories[replay.primary_replay_index].reverse_states[0]
    if isinstance(replay, LTXFlowReplayResult):
        canonical = replay.latent_layout.to_canonical(reference)
        canonical_direction, _metadata = build_flow_tubelet_key_direction_like(
            canonical,
            key_text=key_text,
            key_context=key_context,
            flow_phase=flow_phase,
        )
        return replay.latent_layout.from_canonical(canonical_direction)
    direction, _metadata = build_flow_tubelet_key_direction_like(
        reference,
        key_text=key_text,
        key_context=key_context,
        flow_phase=flow_phase,
    )
    return direction


def _normalized_projection(value: Any, direction: Any) -> float:
    """计算状态或位移在 key 方向上的范数归一化投影。"""

    value_flat = value.detach().float().reshape(-1)
    direction_flat = direction.detach().float().reshape(-1)
    denominator = value_flat.norm().clamp_min(1e-8) * direction_flat.norm().clamp_min(1e-8)
    return float((value_flat @ direction_flat / denominator).item())


def _root_mean_square(value: Any) -> float:
    """计算不含候选 key 的轨迹能量, 供 generic SSM 公平对照使用。"""

    return float(value.detach().float().pow(2).mean().sqrt().item())


def build_flow_state_observation_sequence(
    replay: WanFlowReplayResult | LTXFlowReplayResult,
    *,
    key_text: str,
    trajectory: ReplayTrajectory | None = None,
    schedule: Iterable[Any] | None = None,
    key_context: FlowTubeletKeyContext | None = None,
) -> list[dict[str, Any]]:
    """从固定 inversion 路径构造真实逐 phase 状态空间观测序列。"""

    active = trajectory or replay.replay_trajectories[replay.primary_replay_index]
    active_schedule = tuple(schedule) if schedule is not None else replay.primary_schedule
    states = active.reverse_states
    if len(active_schedule) != len(states):
        raise ValueError("状态空间观测的 Flow schedule 与固定 reverse path 长度不一致")
    if len(states) != len(active_schedule):
        raise RuntimeError("状态空间观测的 replay states 与 schedule 长度不一致")
    if len(active.forward_states) != len(states) or len(active.null_forward_states) != len(states):
        raise RuntimeError("候选、null 与固定 inversion 轨迹长度不一致")
    active_key_context = key_context or replay.key_context
    if active_key_context is None:
        raise RuntimeError("正式状态空间观测缺少 FlowTubeletKeyContext")
    sigma_grid = [float(point.sigma) for point in active_schedule]
    replay_reliability = float(replay.replay_uncertainty.replay_reliability)
    grid_reliability = _time_grid_reliability(replay)
    coverage = float(replay.endpoint_evidence.coverage_ratio)
    observations: list[dict[str, Any]] = []
    for step_index in range(len(states) - 1):
        delta_sigma = float(active_schedule[step_index + 1].sigma) - float(
            active_schedule[step_index].sigma
        )
        if abs(delta_sigma) <= 1e-12:
            continue
        phase = normalized_flow_phase_from_sigma_interval(
            sigma_grid,
            step_index,
        )
        direction = _replay_native_key_direction(
            replay,
            key_text=key_text,
            key_context=active_key_context,
            flow_phase=phase,
        ).to(device=states[0].device, dtype=states[0].dtype)
        displacement = states[step_index + 1] - states[step_index]
        velocity = displacement / delta_sigma
        path = compute_path_step_observation(
            states[step_index],
            states[step_index + 1],
            velocity,
            direction,
            flow_phase=phase,
            delta_sigma=delta_sigma,
        )
        state_projection = _normalized_projection(states[step_index + 1], direction)
        endpoint_score = max(0.0, min(1.0, 0.5 + 0.5 * state_projection))
        likelihood = gaussian_replay_residual_likelihood(
            active.forward_states[step_index + 1],
            active.null_forward_states[step_index + 1],
            states[step_index + 1],
            config=replay.replay_likelihood_config,
        )
        local_replay_reliability = replay_step_reliability_weight(
            active,
            step_index + 1,
            config=replay.replay_likelihood_config,
        )
        step_replay_reliability = max(
            0.0,
            min(1.0, replay_reliability * local_replay_reliability),
        )
        unweighted_path_score = float(path.path_projection_normalized)
        unweighted_path_endpoint_consistency = max(
            0.0,
            min(1.0, 1.0 - abs(state_projection - unweighted_path_score)),
        )
        observations.append({
            "flow_state_observation_step_index": step_index,
            "flow_phase": round(phase, 8),
            "trajectory_delta_sigma": round(delta_sigma, 10),
            "flow_tubelet_key_context_digest": flow_tubelet_key_context_digest(
                active_key_context
            ),
            "flow_tubelet_formal_context_complete": True,
            "path_quadrature_context_complete": (
                path.path_quadrature_context_complete
            ),
            "endpoint_score": round(endpoint_score, 8),
            "velocity_score": round(path.velocity_projection_normalized, 8),
            "path_score": round(
                unweighted_path_score * step_replay_reliability,
                8,
            ),
            "path_score_unweighted": round(unweighted_path_score, 8),
            "path_endpoint_consistency": round(
                unweighted_path_endpoint_consistency * step_replay_reliability,
                8,
            ),
            "path_endpoint_consistency_unweighted": round(
                unweighted_path_endpoint_consistency,
                8,
            ),
            "replay_log_likelihood_ratio": round(
                likelihood.log_likelihood_ratio_per_dimension,
                8,
            ),
            "replay_reliability": round(step_replay_reliability, 8),
            "replay_reliability_weight": round(step_replay_reliability, 8),
            "replay_step_reliability_weight": round(
                step_replay_reliability,
                8,
            ),
            "replay_step_likelihood_reliability": round(
                local_replay_reliability,
                8,
            ),
            "replay_global_reliability": round(replay_reliability, 8),
            "path_replay_uncertainty_weighting_status": (
                "global_multigrid_and_step_likelihood_weight_applied"
            ),
            "time_grid_reliability": round(grid_reliability, 8),
            "coverage_ratio": round(coverage, 8),
            "path_velocity_consistency": round(path.path_velocity_consistency, 8),
            "key_agnostic_endpoint_energy": round(_root_mean_square(states[step_index + 1]), 8),
            "key_agnostic_velocity_energy": round(_root_mean_square(velocity), 8),
            "key_agnostic_path_energy": round(_root_mean_square(displacement), 8),
            "replay_observation_noise_variance": round(
                likelihood.observation_noise_variance,
                10,
            ),
            "replay_likelihood_model_id": likelihood.likelihood_model_id,
        })
    if not observations:
        raise RuntimeError("固定 replay 路径未形成有效状态空间观测")
    return observations


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
    state_sequence = build_flow_state_observation_sequence(replay, key_text=key_text)
    state_context_complete = all(
        row.get("flow_tubelet_formal_context_complete") is True
        and row.get("path_quadrature_context_complete") is True
        for row in state_sequence
    )
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
        "flow_state_observation_sequence": state_sequence,
        "flow_state_observation_sequence_status": "measured_from_fixed_replay_path",
        "flow_state_observation_step_count": len(state_sequence),
        "flow_state_observation_formal_context_complete": state_context_complete,
        "flow_tubelet_key_context_digest": (
            flow_tubelet_key_context_digest(replay.key_context)
            if replay.key_context is not None
            else None
        ),
        "flow_state_transition_source": "calibration_fitted_linear_gaussian_dynamics",
    }


def _control_payload(
    pipeline: Any,
    replay: WanFlowReplayResult | LTXFlowReplayResult,
    *,
    prompt: str,
    wrong_prompt: str,
    key_text: str,
    wrong_key_text: str,
) -> dict[str, Any]:
    """执行 wrong key、wrong prompt 与 wrong sampler/time-grid 真实对照。"""

    wrong_key = wrong_key_text
    if replay.key_context is None:
        raise RuntimeError("正式 replay control 缺少正确生成 key context")
    correct_context = replay.key_context
    wrong_prompt_context = _flow_key_context(wrong_prompt, pipeline.scheduler)
    wrong_key_endpoint = _compute_replay_endpoint_evidence_for_key(
        replay,
        key_text=wrong_key,
        key_context=correct_context,
    )

    primary_steps = int(replay.replay_step_counts[replay.primary_replay_index])
    fixed_trajectory = replay.replay_trajectories[replay.primary_replay_index]
    wrong_key_trajectory, wrong_key_schedule, wrong_key_path = _run_control_replay_for_model(
        pipeline,
        replay,
        prompt=prompt,
        key_text=wrong_key,
        key_context=correct_context,
        num_inference_steps=primary_steps,
        fixed_trajectory=fixed_trajectory,
    )
    wrong_prompt_trajectory, wrong_prompt_schedule, wrong_prompt_path = _run_control_replay_for_model(
        pipeline,
        replay,
        prompt=wrong_prompt,
        key_text=key_text,
        key_context=wrong_prompt_context,
        num_inference_steps=primary_steps,
        fixed_trajectory=fixed_trajectory,
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
    wrong_sampler_trajectory, wrong_sampler_schedule, wrong_sampler_path = _run_control_replay_for_model(
        pipeline,
        replay,
        prompt=prompt,
        key_text=key_text,
        key_context=_flow_key_context(prompt, wrong_scheduler),
        num_inference_steps=primary_steps,
        scheduler=wrong_scheduler,
        fixed_trajectory=fixed_trajectory,
    )
    def hypothesis_support(trajectory: Any) -> float:
        likelihood_probability = 1.0 / (
            1.0 + math.exp(-float(trajectory.replay_log_likelihood_ratio))
        )
        gaussian_fit_probability = math.exp(-0.5 * (
            float(trajectory.candidate_residual_mean_squared_error)
            / max(float(trajectory.observation_noise_variance), 1e-12)
        ))
        return likelihood_probability * gaussian_fit_probability

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
    wrong_key_sequence = build_flow_state_observation_sequence(
        replay,
        key_text=wrong_key,
        trajectory=wrong_key_trajectory,
        schedule=wrong_key_schedule,
        key_context=correct_context,
    )
    wrong_prompt_sequence = build_flow_state_observation_sequence(
        replay,
        key_text=key_text,
        trajectory=wrong_prompt_trajectory,
        schedule=wrong_prompt_schedule,
        key_context=wrong_prompt_context,
    )
    wrong_sampler_sequence = build_flow_state_observation_sequence(
        replay,
        key_text=key_text,
        trajectory=wrong_sampler_trajectory,
        schedule=wrong_sampler_schedule,
        key_context=_flow_key_context(prompt, wrong_scheduler),
    )
    fixed_reverse_reused = all(
        trajectory.reverse_states is fixed_trajectory.reverse_states
        for trajectory in (
            wrong_key_trajectory,
            wrong_prompt_trajectory,
            wrong_sampler_trajectory,
        )
    )
    return {
        "wrong_key_endpoint_score": round(wrong_key_endpoint.score, 8),
        "wrong_key_S_path_inv": wrong_key_path.get("S_path_inv"),
        "wrong_key_replay_cycle_error": round(
            wrong_key_trajectory.candidate_cycle_relative_error,
            8,
        ),
        "wrong_key_replay_log_likelihood_ratio": round(
            wrong_key_trajectory.replay_log_likelihood_ratio,
            8,
        ),
        "wrong_key_control_margin": round(
            (replay.endpoint_evidence.score + matched_path_reliability_score)
            - (wrong_key_endpoint.score + wrong_key_path_reliability_score),
            8,
        ),
        "wrong_prompt_replay_cycle_error": round(
            wrong_prompt_trajectory.candidate_cycle_relative_error,
            8,
        ),
        "wrong_prompt_replay_log_likelihood_ratio": round(
            wrong_prompt_trajectory.replay_log_likelihood_ratio,
            8,
        ),
        "wrong_prompt_S_path_inv": wrong_prompt_path.get("S_path_inv"),
        "wrong_prompt_control_margin": round(
            matched_path_reliability_score - wrong_prompt_path_reliability_score,
            8,
        ),
        "wrong_sampler_replay_cycle_error": round(
            wrong_sampler_trajectory.candidate_cycle_relative_error,
            8,
        ),
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
        "wrong_key_flow_state_observation_sequence": wrong_key_sequence,
        "wrong_prompt_flow_state_observation_sequence": wrong_prompt_sequence,
        "wrong_sampler_flow_state_observation_sequence": wrong_sampler_sequence,
        "replay_control_fixed_reverse_path_reused": fixed_reverse_reused,
        "replay_control_execution_status": "measured_formal",
        "wrong_prompt_control_prompt_digest": sha256(wrong_prompt.encode("utf-8")).hexdigest(),
        "replay_control_correct_key_context_digest": (
            flow_tubelet_key_context_digest(correct_context)
        ),
        "wrong_prompt_key_context_digest": (
            flow_tubelet_key_context_digest(wrong_prompt_context)
        ),
        "wrong_sampler_key_context_digest": flow_tubelet_key_context_digest(
            _flow_key_context(prompt, wrong_scheduler)
        ),
        "replay_control_joint_context_complete": True,
    }


def _controlled_negative_records_from_positive(
    positive_record: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """把真实 wrong-condition replay 展开为三个独立负假设记录。"""

    identity_fields = (
        "generation_model_id",
        "generation_model_family",
        "generation_model_requested_revision",
        "generation_model_commit_or_hash",
        "generation_model_revision_source",
        "generation_model_revision_resolution_status",
        "cross_model_role",
        "prompt_id",
        "seed_id",
        "watermark_key_derivation_id",
        "watermark_key_id",
        "generation_seed_random",
        "generation_generator_state_digest_random",
        "velocity_causal_pair_id",
        "velocity_causal_intervention_status",
        "trajectory_trace_id",
        "split",
        "protocol_split",
        "colab_runtime_profile",
        "method_variant",
        "attack_name",
        "source_video_cluster_id",
        "statistical_cluster_id",
        "statistical_independent_unit",
        "generation_source_video_sha256",
        "attacked_video_path",
        "attacked_video_sha256",
        "replay_sampler_signature",
        "authenticated_generation_time_grid_id",
        "authenticated_generation_step_count",
        "formal_flow_detector_input_contract",
        "replay_likelihood_calibration_protocol",
        "replay_likelihood_calibration_cluster_count",
        "replay_relative_observation_noise_standard_deviation",
        "replay_minimum_observation_noise_variance",
    )
    common = {field: positive_record.get(field) for field in identity_fields}
    specs = (
        (
            "watermarked_video_wrong_key_hypothesis",
            "wrong_key",
            "wrong_key_flow_state_observation_sequence",
            "wrong_key_replay_log_likelihood_ratio",
            "wrong_key_S_path_inv",
        ),
        (
            "watermarked_video_wrong_prompt_hypothesis",
            "wrong_prompt",
            "wrong_prompt_flow_state_observation_sequence",
            "wrong_prompt_replay_log_likelihood_ratio",
            "wrong_prompt_S_path_inv",
        ),
        (
            "watermarked_video_wrong_sampler_time_grid_hypothesis",
            "wrong_sampler_time_grid",
            "wrong_sampler_flow_state_observation_sequence",
            "wrong_sampler_replay_log_likelihood_ratio",
            "wrong_sampler_S_path_inv",
        ),
    )
    records: list[dict[str, Any]] = []
    for negative_family, hypothesis_type, sequence_field, llr_field, path_field in specs:
        sequence = positive_record.get(sequence_field)
        if not isinstance(sequence, list) or len(sequence) < 2:
            raise RuntimeError(f"{negative_family} 缺少多步固定路径状态观测")
        records.append(with_flow_evidence_protocol_defaults(
            {
                **common,
                "record_version": "formal_flow_controlled_negative_v1",
                "formal_flow_evidence_unit_id": build_stable_digest({
                    "positive_unit_id": positive_record.get("formal_flow_evidence_unit_id"),
                    "negative_family": negative_family,
                }),
                "sample_role": "controlled_negative",
                "negative_family": negative_family,
                "controlled_negative_hypothesis_type": hypothesis_type,
                "formal_flow_evidence_level": FORMAL_FLOW_EVIDENCE_LEVEL,
                "replay_inversion_status": "ready",
                "replay_trajectory_source": (
                    "attacked_video_fixed_reverse_path_wrong_condition_forward_hypothesis"
                ),
                "replay_control_execution_status": "measured_formal",
                "replay_control_fixed_reverse_path_reused": positive_record.get(
                    "replay_control_fixed_reverse_path_reused"
                ),
                "replay_log_likelihood_ratio_mean": positive_record.get(llr_field),
                "replay_likelihood_model_id": REPLAY_GAUSSIAN_LIKELIHOOD_MODEL_ID,
                "S_path_inv": positive_record.get(path_field),
                "flow_state_observation_sequence": sequence,
                "flow_state_observation_sequence_status": (
                    "measured_from_fixed_replay_path"
                ),
                "flow_state_observation_step_count": len(sequence),
                "flow_state_observation_formal_context_complete": all(
                    row.get("flow_tubelet_formal_context_complete") is True
                    and row.get("path_quadrature_context_complete") is True
                    for row in sequence
                ),
                "flow_tubelet_key_context_digest": sequence[0].get(
                    "flow_tubelet_key_context_digest"
                ),
                "flow_tubelet_formal_context_complete": all(
                    row.get("flow_tubelet_formal_context_complete") is True
                    for row in sequence
                ),
                "path_quadrature_context_complete": all(
                    row.get("path_quadrature_context_complete") is True
                    for row in sequence
                ),
                "endpoint_formal_context_complete": True,
                "endpoint_evidence_source": (
                    "fixed_reverse_path_phase_conditioned_state_endpoint"
                ),
                "flow_state_transition_source": (
                    "calibration_fitted_linear_gaussian_dynamics"
                ),
                "trajectory_trace_used_for_score": False,
                "metric_status": "measured_formal",
            },
            negative_family=negative_family,
            trajectory_source_level=(
                "attacked_video_fixed_reverse_path_wrong_condition_forward_hypothesis"
            ),
            flow_state_admissibility_status="pending_frozen_detector",
            claim_support_status="sstw_controlled_negative_flow_evidence_ready",
        ))
    return records


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
        "generation_model_requested_revision": source.get(
            "generation_model_requested_revision"
        ),
        "generation_model_commit_or_hash": source.get(
            "generation_model_commit_or_hash"
        ),
        "generation_model_revision_source": source.get(
            "generation_model_revision_source"
        ),
        "generation_model_revision_resolution_status": source.get(
            "generation_model_revision_resolution_status"
        ),
        "cross_model_role": source.get("cross_model_role"),
        "prompt_id": source.get("prompt_id"),
        "seed_id": source.get("seed_id"),
        "watermark_key_derivation_id": source.get(
            "watermark_key_derivation_id"
        ),
        "watermark_key_id": source.get("watermark_key_id"),
        "split": source.get("split"),
    })
    return {
        "record_version": "formal_flow_evidence_v1",
        "generation_model_id": source.get("generation_model_id"),
        "generation_model_family": source.get("generation_model_family"),
        "cross_model_role": source.get("cross_model_role"),
        "prompt_id": source.get("prompt_id"),
        "seed_id": source.get("seed_id"),
        "generation_seed_random": source.get("generation_seed_random"),
        "generation_generator_state_digest_random": source.get(
            "generation_generator_state_digest_random"
        ),
        "velocity_causal_pair_id": source.get("velocity_causal_pair_id"),
        "velocity_causal_intervention_status": source.get(
            "velocity_causal_intervention_status"
        ),
        "trajectory_trace_id": source.get("trajectory_trace_id"),
        "split": source.get("split"),
        "protocol_split": source.get("protocol_split"),
        "colab_runtime_profile": source.get("colab_runtime_profile"),
        "sample_role": sample_role,
        "method_variant": method_variant,
        "attack_name": source.get("attack_name"),
        "source_video_cluster_id": source.get("trajectory_trace_id"),
        "generation_source_video_sha256": source.get("source_video_sha256"),
        "statistical_cluster_id": statistical_cluster_id,
        "statistical_independent_unit": "source_video_prompt_seed",
        "generation_record_digest": source.get("generation_record_digest"),
        "code_commit": source.get("code_commit"),
        "flow_runtime_formal_context_complete": source.get(
            "flow_runtime_formal_context_complete"
        ),
        "generation_endpoint_control_formal_context_complete": source.get(
            "generation_endpoint_control_formal_context_complete"
        ),
        "generation_endpoint_quality_energy_guard_passed": source.get(
            "generation_endpoint_quality_energy_guard_passed"
        ),
        "generation_endpoint_control_cumulative_energy_final": source.get(
            "generation_endpoint_control_cumulative_energy_final"
        ),
        "generation_endpoint_reference_cumulative_energy_final": source.get(
            "generation_endpoint_reference_cumulative_energy_final"
        ),
        "generation_velocity_constraint_delta_ratio_maximum": source.get(
            "generation_velocity_constraint_delta_ratio_maximum"
        ),
    }


def _load_pipeline(model_id: str, *, revision: str | None = None) -> Any:
    """在 CUDA 上加载与生成记录同一不可变 revision 的 pipeline。"""

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("正式 Flow replay 需要可用 CUDA GPU")
    return _load_video_generation_pipeline(
        model_id,
        _select_dtype(torch),
        revision=revision,
    )


def _invoke_pipeline_loader(
    pipeline_loader: Any,
    *,
    model_id: str,
    revision: str,
) -> Any:
    """调用可注入 loader, 默认运行时必须消费冻结 revision。

    轻量测试使用的单参数 loader 不加载真实模型, 因而保留兼容入口; 真实默认
    loader 和任何声明 `revision` 参数的服务器实现都会收到不可变 commit。
    """

    parameters = inspect.signature(pipeline_loader).parameters.values()
    supports_revision = any(
        parameter.name == "revision"
        or parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )
    if supports_revision:
        return pipeline_loader(model_id, revision=revision)
    return pipeline_loader(model_id)


def _score_records_with_frozen_calibration(
    evidence_records: Iterable[Mapping[str, Any]],
    *,
    target_fpr: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """按 method variant 冻结 calibration negative 后评分全部 records。"""

    source_rows = [dict(record) for record in evidence_records]
    unexpected_source_variants = sorted({
        str(record.get("method_variant"))
        for record in source_rows
        if str(record.get("method_variant")) not in GENERATION_METHOD_VARIANTS
    })
    if unexpected_source_variants:
        raise ValueError(
            "正式 Flow replay 输入只能包含真实生成机制变体; 检测器专用消融必须"
            "由 full-method 同视频证据派生: "
            + ", ".join(unexpected_source_variants)
        )
    rows: list[dict[str, Any]] = []
    for source in source_rows:
        rows.append(source)
        if source.get("method_variant") != "sstw_full_method":
            continue
        for detector_variant in DETECTOR_ONLY_METHOD_VARIANTS:
            derived = dict(source)
            derived.update({
                "method_variant": detector_variant,
                "detector_only_ablation": True,
                "detector_only_source_method_variant": "sstw_full_method",
                "formal_flow_evidence_unit_id": build_stable_digest({
                    "source_formal_flow_evidence_unit_id": source.get(
                        "formal_flow_evidence_unit_id"
                    ),
                    "detector_only_method_variant": detector_variant,
                }),
            })
            rows.append(derived)
    governed_unit_ids = [
        str(record["formal_flow_evidence_unit_id"])
        for record in rows
        if record.get("formal_flow_evidence_unit_id")
    ]
    if len(governed_unit_ids) != len(set(governed_unit_ids)):
        raise RuntimeError("检测器专用消融展开后出现重复 formal_flow_evidence_unit_id")
    calibration_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in rows:
        if record.get("split") == "calibration":
            calibration_rows[(
                str(record.get("generation_model_id")),
                str(record.get("method_variant")),
            )].append(record)
    calibration_source_variant = {
        variant: (
            "sstw_full_method"
            if variant in {
                CLAIM2_PATH_NESTED_ABLATION_VARIANT,
                *DETECTOR_ONLY_METHOD_VARIANTS,
            }
            else variant
        )
        for variant in FORMAL_DETECTOR_VARIANTS
    }
    calibrations = {
        (model_id, variant): fit_flow_evidence_calibration(
            calibration_rows.get(
                (model_id, calibration_source_variant[variant]),
                [],
            ),
            method_variant=variant,
            target_fpr=target_fpr,
        )
        for model_id in sorted({str(record.get("generation_model_id")) for record in rows})
        for variant in FORMAL_DETECTOR_VARIANTS
    }
    cross_model_role_by_model = {
        model_id: next(
            (
                record.get("cross_model_role")
                for record in rows
                if str(record.get("generation_model_id")) == model_id
            ),
            None,
        )
        for model_id, _variant in calibrations
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
        if record.get("sample_role") in {"clean_negative", "controlled_negative"}:
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
            "cross_model_role": cross_model_role_by_model.get(model_id),
            "model_specific_calibration": True,
            "method_variant": variant,
            "calibration_source_method_variant": calibration_source_variant[variant],
            "detector_only_nested_ablation": (
                variant
                in {
                    CLAIM2_PATH_NESTED_ABLATION_VARIANT,
                    *DETECTOR_ONLY_METHOD_VARIANTS,
                }
            ),
            **frozen_flow_detector_calibration_artifact(calibration),
        }
        for (model_id, variant), calibration in calibrations.items()
    ]
    return scored, threshold_records, calibrations


def _paired_path_gain_records(
    scored_records: Iterable[Mapping[str, Any]],
    calibrations: Mapping[tuple[str, str], Any],
) -> list[dict[str, Any]]:
    """在同一 full-method 视频上比较完整检测器与仅移除路径证据的检测器。

    两个检测器消费相同视频、replay 轨迹、速度证据、endpoint 证据和状态空间
    机制。唯一干预是把 path score 与 path-endpoint consistency 置零, 因而该
    配对差可以识别 Claim-2 的路径证据边际增益。
    """

    rows: list[dict[str, Any]] = []
    for record in scored_records:
        if (
            record.get("sample_role") != "attacked_positive"
            or record.get("method_variant") != "sstw_full_method"
            or record.get("split") != "test"
        ):
            continue
        without_path = apply_frozen_flow_detector(
            record,
            calibrations[
                (
                    str(record.get("generation_model_id")),
                    CLAIM2_PATH_NESTED_ABLATION_VARIANT,
                )
            ],
        )
        full_score = float(record["S_final_conservative"])
        without_path_score = float(without_path["S_final_conservative"])
        rows.append({
            "record_version": "paired_path_evidence_gain_v2",
            "generation_model_id": record.get("generation_model_id"),
            "cross_model_role": record.get("cross_model_role"),
            "prompt_id": record.get("prompt_id"),
            "seed_id": record.get("seed_id"),
            "trajectory_trace_id": record.get("trajectory_trace_id"),
            "attack_name": record.get("attack_name"),
            "statistical_cluster_id": record.get("statistical_cluster_id"),
            "target_fpr": record.get("target_fpr"),
            "paired_source_method_variant": "sstw_full_method",
            "paired_path_ablation_method_variant": (
                CLAIM2_PATH_NESTED_ABLATION_VARIANT
            ),
            "paired_path_nested_ablation_status": (
                "same_video_same_replay_only_path_features_removed"
            ),
            "paired_detector_threshold_source_split": "calibration",
            "paired_test_time_threshold_update_blocked": True,
            "paired_full_detector_target_fpr": record.get("target_fpr"),
            "paired_without_path_evidence_detector_target_fpr": without_path.get(
                "target_fpr"
            ),
            "paired_fpr_alignment_status": (
                "same_preregistered_target_fpr"
                if abs(
                    float(record.get("target_fpr") or -1.0)
                    - float(without_path.get("target_fpr") or -2.0)
                )
                <= 1e-12
                else "target_fpr_mismatch"
            ),
            "paired_full_detector_score": full_score,
            "paired_without_path_evidence_detector_score": without_path_score,
            "paired_path_evidence_score_gain": round(
                full_score - without_path_score,
                8,
            ),
            "paired_full_detector_decision": bool(record.get("decision")),
            "paired_without_path_evidence_detector_decision": bool(
                without_path.get("decision")
            ),
            "paired_path_evidence_detection_gain": (
                int(bool(record.get("decision")))
                - int(bool(without_path.get("decision")))
            ),
            "metric_status": "measured_formal",
            "claim_support_status": (
                "claim2_same_video_nested_path_ablation_fixed_fpr_evidence"
            ),
        })
    return rows


def _paired_velocity_causal_records(
    scored_records: Iterable[Mapping[str, Any]],
    calibrations: Mapping[tuple[str, str], Any],
) -> list[dict[str, Any]]:
    """用同一完整检测器比较受控随机种子下的速度约束干预。"""

    positives = [
        record
        for record in scored_records
        if record.get("sample_role") == "attacked_positive"
        and record.get("split") == "test"
        and record.get("method_variant")
        in {"sstw_full_method", "without_velocity_constraint"}
        and record.get("attack_name")
        == "no_attack_generation_causal_ablation"
    ]
    by_identity: dict[tuple[str, str, str, str, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for record in positives:
        identity = (
            str(record.get("generation_model_id") or ""),
            str(record.get("prompt_id") or ""),
            str(record.get("seed_id") or ""),
            str(record.get("attack_name") or ""),
            str(record.get("velocity_causal_pair_id") or ""),
        )
        by_identity[identity][str(record.get("method_variant"))] = record
    rows: list[dict[str, Any]] = []
    for identity, variants in by_identity.items():
        full = variants.get("sstw_full_method")
        control = variants.get("without_velocity_constraint")
        if full is None or control is None:
            continue
        matched_design = (
            bool(identity[4])
            and full.get("generation_seed_random") is not None
            and bool(full.get("generation_generator_state_digest_random"))
            and full.get("generation_generator_state_digest_random")
            == control.get("generation_generator_state_digest_random")
            and full.get("generation_seed_random") == control.get("generation_seed_random")
            and bool(full.get("replay_sampler_signature"))
            and full.get("replay_sampler_signature")
            == control.get("replay_sampler_signature")
            and bool(full.get("authenticated_generation_time_grid_id"))
            and full.get("authenticated_generation_time_grid_id")
            == control.get("authenticated_generation_time_grid_id")
            and bool(full.get("generation_source_video_sha256"))
            and bool(control.get("generation_source_video_sha256"))
            and full.get("generation_source_video_sha256")
            != control.get("generation_source_video_sha256")
            and full.get("velocity_causal_intervention_status")
            == "velocity_constraint_enabled"
            and full.get("flow_runtime_formal_context_complete") is True
            and full.get(
                "generation_endpoint_control_formal_context_complete"
            )
            is True
            and full.get("generation_endpoint_quality_energy_guard_passed")
            is True
            and control.get("velocity_causal_intervention_status")
            == "velocity_constraint_disabled"
        )
        base = {
            "record_version": "paired_velocity_causal_evidence_v1",
            "generation_model_id": identity[0],
            "cross_model_role": full.get("cross_model_role"),
            "prompt_id": identity[1],
            "seed_id": identity[2],
            "attack_name": identity[3],
            "velocity_causal_pair_id": identity[4],
            "statistical_cluster_id": full.get("statistical_cluster_id"),
            "target_fpr": full.get("target_fpr"),
            "paired_detector_method_variant": "sstw_full_method",
            "paired_detector_threshold_source_split": "calibration",
            "paired_test_time_threshold_update_blocked": True,
            "full_generation_generator_state_digest_random": full.get(
                "generation_generator_state_digest_random"
            ),
            "control_generation_generator_state_digest_random": control.get(
                "generation_generator_state_digest_random"
            ),
            "full_generation_source_video_sha256": full.get(
                "generation_source_video_sha256"
            ),
            "control_generation_source_video_sha256": control.get(
                "generation_source_video_sha256"
            ),
        }
        if not matched_design:
            rows.append({
                **base,
                "velocity_causal_pairing_status": "blocked_by_unmatched_generation_design",
                "metric_status": "missing",
                "claim_support_status": "claim1_velocity_causal_evidence_blocked",
            })
            continue
        frozen_full_detector = calibrations[(identity[0], "sstw_full_method")]
        full_detection = apply_frozen_flow_detector(full, frozen_full_detector)
        control_detection = apply_frozen_flow_detector(control, frozen_full_detector)
        full_score = float(full_detection["S_final_conservative"])
        control_score = float(control_detection["S_final_conservative"])
        rows.append({
            **base,
            "velocity_causal_pairing_status": "matched_single_intervention_design",
            "paired_detector_score_source": FLOW_STATE_POSTERIOR_SCORE_SOURCE,
            "paired_frozen_final_score_threshold": (
                frozen_full_detector.final_score_threshold
            ),
            "paired_detector_target_fpr": frozen_full_detector.target_fpr,
            "paired_full_method_score": full_score,
            "paired_without_velocity_constraint_score": control_score,
            "paired_velocity_causal_score_gain": round(full_score - control_score, 8),
            "paired_full_method_decision": bool(full_detection["decision"]),
            "paired_without_velocity_constraint_decision": bool(control_detection["decision"]),
            "paired_velocity_causal_detection_gain": (
                int(bool(full_detection["decision"]))
                - int(bool(control_detection["decision"]))
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
    minimum_velocity_causal_pair_count: int,
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
        if record.get("sample_role") in {"clean_negative", "controlled_negative"}
        and record.get("method_variant") == "sstw_full_method"
        and record.get("split") == "test"
    ]
    fpr_estimate = clustered_binary_any_rate_interval(
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
    path_pairing_failures = [
        record
        for record in paired
        if (
            record.get("paired_path_ablation_method_variant")
            != CLAIM2_PATH_NESTED_ABLATION_VARIANT
            or record.get("paired_path_nested_ablation_status")
            != "same_video_same_replay_only_path_features_removed"
            or record.get("paired_detector_threshold_source_split") != "calibration"
            or record.get("paired_test_time_threshold_update_blocked") is not True
            or record.get("paired_fpr_alignment_status")
            != "same_preregistered_target_fpr"
            or abs(float(record.get("target_fpr") or -1.0) - float(target_fpr))
            > 1e-12
        )
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
    velocity_pair_records = [
        record
        for record in paired_velocity_records
        if record.get("cross_model_role") != "cross_model_validation_model"
    ]
    velocity_paired = [
        record for record in velocity_pair_records
        if record.get("velocity_causal_pairing_status")
        == "matched_single_intervention_design"
        and record.get("metric_status") == "measured_formal"
    ]
    velocity_pairing_failures = [
        record for record in velocity_pair_records if record not in velocity_paired
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
        and len(velocity_paired) >= int(minimum_velocity_causal_pair_count)
        and not velocity_pairing_failures
        and fpr_estimate.estimate <= target_fpr
        and tpr_estimate.confidence_interval_lower > target_fpr
        and velocity_score_gain.confidence_interval_lower > 0.0
        and velocity_detection_gain.confidence_interval_lower > 0.0
    )
    claim2_pass = (
        bool(paired)
        and len(paired) == len(full_positive)
        and not path_pairing_failures
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
        "claim_1_velocity_causal_expected_pair_count": int(
            minimum_velocity_causal_pair_count
        ),
        "claim_1_velocity_causal_pairing_failure_count": len(velocity_pairing_failures),
        "claim_1_velocity_causal_pair_coverage": round(
            min(
                1.0,
                len(velocity_paired)
                / max(1, int(minimum_velocity_causal_pair_count)),
            ),
            8,
        ),
        "claim_1_velocity_causal_detector_protocol": (
            "same_frozen_full_method_detector_on_paired_generation_intervention"
        ),
        "claim_1_velocity_causal_score_gain_mean": round(velocity_score_gain.estimate, 8),
        "claim_1_velocity_causal_score_gain_ci_95_lower": round(velocity_score_gain.confidence_interval_lower, 8),
        "claim_1_velocity_causal_score_gain_ci_95_upper": round(velocity_score_gain.confidence_interval_upper, 8),
        "claim_1_velocity_causal_detection_gain_mean": round(velocity_detection_gain.estimate, 8),
        "claim_1_velocity_causal_detection_gain_ci_95_lower": round(velocity_detection_gain.confidence_interval_lower, 8),
        "claim_1_velocity_causal_detection_gain_ci_95_upper": round(velocity_detection_gain.confidence_interval_upper, 8),
        "claim_2_path_evidence_independent_gain_decision": "PASS" if claim2_pass else "FAIL",
        "claim_2_paired_comparison_count": len(paired),
        "claim_2_expected_paired_comparison_count": len(full_positive),
        "claim_2_pairing_failure_count": len(path_pairing_failures),
        "claim_2_paired_comparison_coverage": round(
            len(paired) / max(1, len(full_positive)),
            8,
        ),
        "claim_2_nested_ablation_method_variant": (
            CLAIM2_PATH_NESTED_ABLATION_VARIANT
        ),
        "claim_2_causal_comparison_protocol": (
            "same_video_same_replay_only_path_features_removed_with_frozen_calibration"
        ),
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
            and record.get("sample_role") in {"clean_negative", "controlled_negative"}
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
            and record.get("velocity_causal_pairing_status")
            == "matched_single_intervention_design"
            and record.get("metric_status") == "measured_formal"
        ]
        fpr = clustered_binary_any_rate_interval(
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


def _state_space_posterior_mechanism_failures(
    scored_records: Iterable[Mapping[str, Any]],
    threshold_records: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """拒绝单步代理后验、非概率 replay 分数和不可重建的状态模型。"""

    failures: list[dict[str, Any]] = []
    for record in threshold_records:
        identity = {
            "generation_model_id": record.get("generation_model_id"),
            "method_variant": record.get("method_variant"),
            "failure_scope": "frozen_calibration",
        }
        negative_model = record.get("posterior_negative_state_space_model")
        positive_model = record.get("posterior_positive_state_space_model")
        checks = {
            "posterior_model_type": (
                record.get("posterior_model_type") == FLOW_STATE_POSTERIOR_MODEL_TYPE
            ),
            "posterior_contract_version": (
                record.get("posterior_model_contract_version")
                == FLOW_STATE_POSTERIOR_CONTRACT_VERSION
            ),
            "phase_conditioned_transition_configured": (
                record.get("posterior_phase_conditioned_transition_configured")
                is True
            ),
            "reliability_heteroscedastic_observation_configured": (
                record.get(
                    "posterior_reliability_heteroscedastic_observation_configured"
                )
                is True
            ),
            "complete_p6_admissibility_context": (
                record.get("posterior_admissibility_context_complete") is True
                and isinstance(
                    record.get("posterior_admissibility_thresholds"), Mapping
                )
                and set(record["posterior_admissibility_thresholds"])
                == set(FLOW_STATE_ADMISSIBILITY_THRESHOLD_NAMES)
            ),
            "nested_group_probability_calibration": (
                record.get("posterior_probability_calibration_protocol")
                == "nested_source_video_group_cross_fitted_state_space_llr_and_platt"
                and int(
                    record.get(
                        "posterior_probability_calibration_outer_fold_count"
                    )
                    or 0
                )
                >= 2
                and int(
                    record.get(
                        "posterior_probability_calibration_inner_fold_minimum"
                    )
                    or 0
                )
                >= 2
            ),
            "group_heldout_fixed_fpr_threshold_scores": (
                record.get("fixed_fpr_threshold_score_source")
                == "outer_group_heldout_nested_cross_fitted_conservative_scores"
            ),
            "negative_state_model_reconstructable": isinstance(negative_model, Mapping),
            "positive_state_model_reconstructable": isinstance(positive_model, Mapping),
            "negative_state_transitions_fitted": (
                isinstance(negative_model, Mapping)
                and int(negative_model.get("training_transition_count") or 0) > 0
                and int(negative_model.get("training_transition_group_count") or 0) >= 2
                and int(negative_model.get("training_group_count") or 0) >= 2
                and negative_model.get("state_space_dynamics_contract")
                == "phase_conditioned_transition_with_reliability_heteroscedastic_observation"
                and bool(negative_model.get("phase_transition_matrix"))
                and float(
                    negative_model.get(
                        "reliability_observation_variance_scale"
                    )
                    or 0.0
                )
                > 0.0
            ),
            "positive_state_transitions_fitted": (
                isinstance(positive_model, Mapping)
                and int(positive_model.get("training_transition_count") or 0) > 0
                and int(positive_model.get("training_transition_group_count") or 0) >= 2
                and int(positive_model.get("training_group_count") or 0) >= 2
                and positive_model.get("state_space_dynamics_contract")
                == "phase_conditioned_transition_with_reliability_heteroscedastic_observation"
                and bool(positive_model.get("phase_transition_matrix"))
                and float(
                    positive_model.get(
                        "reliability_observation_variance_scale"
                    )
                    or 0.0
                )
                > 0.0
            ),
        }
        missing = [name for name, passed in checks.items() if not passed]
        if missing:
            failures.append({**identity, "failed_requirements": missing})

    for record in scored_records:
        sequence = record.get("flow_state_observation_sequence")
        step_count = int(record.get("flow_state_observation_step_count") or 0)
        filter_step_count = int(record.get("flow_state_filter_step_count") or 0)
        likelihood_values = (
            record.get("flow_state_positive_log_likelihood_per_step"),
            record.get("flow_state_negative_log_likelihood_per_step"),
            record.get("flow_state_log_likelihood_ratio"),
        )
        checks = {
            "measured_state_observation_sequence": (
                record.get("flow_state_observation_sequence_status")
                == "measured_from_fixed_replay_path"
            ),
            "multi_step_state_observation_sequence": (
                isinstance(sequence, list)
                and step_count >= 2
                and len(sequence) == step_count
            ),
            "joint_flow_context_complete": (
                record.get("flow_state_observation_formal_context_complete")
                is True
                and bool(record.get("flow_tubelet_key_context_digest"))
                and record.get("endpoint_formal_context_complete") is True
                and record.get("flow_tubelet_formal_context_complete") is True
                and record.get("path_quadrature_context_complete") is True
                and isinstance(sequence, list)
                and all(
                    row.get("flow_tubelet_formal_context_complete") is True
                    and row.get("path_quadrature_context_complete") is True
                    and row.get("trajectory_delta_sigma") is not None
                    and row.get("flow_tubelet_key_context_digest")
                    == record.get("flow_tubelet_key_context_digest")
                    for row in sequence
                )
            ),
            "kalman_filter_consumed_complete_sequence": (
                filter_step_count == step_count and filter_step_count >= 2
            ),
            "kalman_filter_ready": (
                record.get("flow_state_filtering_status") == "kalman_filter_ready"
            ),
            "rts_smoother_ready": (
                record.get("flow_state_smoothing_status")
                == "rauch_tung_striebel_smoother_ready"
            ),
            "state_marginal_likelihood_ready": all(
                value is not None and math.isfinite(float(value))
                for value in likelihood_values
            ),
            "gaussian_replay_likelihood_ready": (
                record.get("replay_likelihood_model_id")
                == REPLAY_GAUSSIAN_LIKELIHOOD_MODEL_ID
            ),
            "replay_likelihood_fitted_from_calibration_clean_videos": (
                record.get("replay_likelihood_calibration_protocol")
                == "calibration_clean_video_null_residual_cluster_equal_mle"
                and int(record.get("replay_likelihood_calibration_cluster_count") or 0)
                >= 2
                and math.isfinite(float(
                    record.get(
                        "replay_relative_observation_noise_standard_deviation"
                    )
                    or math.nan
                ))
                and float(
                    record.get(
                        "replay_relative_observation_noise_standard_deviation"
                    )
                    or 0.0
                )
                > 0.0
            ),
            "state_space_posterior_score_source_ready": (
                record.get("flow_detector_score_source")
                == FLOW_STATE_POSTERIOR_SCORE_SOURCE
            ),
        }
        missing = [name for name, passed in checks.items() if not passed]
        if missing:
            failures.append({
                "generation_model_id": record.get("generation_model_id"),
                "method_variant": record.get("method_variant"),
                "sample_role": record.get("sample_role"),
                "formal_flow_evidence_unit_id": record.get("formal_flow_evidence_unit_id"),
                "failure_scope": "scored_flow_sequence",
                "failed_requirements": missing,
            })
    return failures


def _negative_family_cluster_counts(
    records: Iterable[Mapping[str, Any]],
    *,
    split: str,
) -> dict[str, int]:
    """按 primary full-method source video 统计真实负假设 family 覆盖。"""

    groups: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if (
            record.get("sample_role") not in {"clean_negative", "controlled_negative"}
            or record.get("method_variant") != "sstw_full_method"
            or record.get("split") != split
            or record.get("cross_model_role") == "cross_model_validation_model"
        ):
            continue
        family = str(record.get("negative_family") or "")
        cluster_id = str(record.get("statistical_cluster_id") or "")
        if family and cluster_id:
            groups[family].add(cluster_id)
    return {family: len(cluster_ids) for family, cluster_ids in groups.items()}


REPLAY_LIKELIHOOD_CALIBRATION_SOURCE_SPLIT = "calibration"
REPLAY_LIKELIHOOD_BOOTSTRAP_MODEL_ID = (
    "replay_noise_calibration_bootstrap_discarded_not_claim_evidence"
)


def _replay_likelihood_calibration_cluster_id(source: Mapping[str, Any]) -> str:
    """为 clean-video 噪声拟合构造独立视频簇标识。

    优先复用生成阶段认证的 trajectory trace。旧记录缺少该字段时，使用模型、
    prompt 与 seed 的稳定摘要作为同一生成视频的簇标识，绝不把多个时间网格当成
    多个独立样本。
    """

    trace_id = str(source.get("trajectory_trace_id") or "").strip()
    if trace_id:
        return trace_id
    required_identity = {
        "generation_model_id": source.get("generation_model_id"),
        "prompt_id": source.get("prompt_id"),
        "seed": source.get("seed"),
        "split": source.get("split"),
    }
    if any(value in {None, ""} for value in required_identity.values()):
        raise ValueError("replay 噪声 calibration 记录缺少可构造独立视频簇的身份字段")
    return build_stable_digest(required_identity)


def _fit_model_specific_replay_likelihood_configs(
    run_root: Path,
    clean_records: Iterable[Mapping[str, Any]],
    prompt_map: Mapping[str, str],
    pipelines: Mapping[str, Any],
    *,
    minimum_clean_video_cluster_count: int,
    calibration_replay_step_count: int,
) -> tuple[dict[str, ReplayGaussianLikelihoodConfig], list[dict[str, Any]]]:
    """仅用 calibration clean videos 拟合逐模型 replay 高斯噪声。

    噪声拟合只运行预注册主网格20步，用于取得与候选 key 无关的 null residual。
    bootstrap 方差不参与状态积分，且其候选似然会被完整丢弃。正式三网格 replay
    随后使用冻结配置独立运行，因此既不长期缓存 GPU states，也不把 bootstrap
    概率写入 claim evidence。
    """

    bootstrap_config = ReplayGaussianLikelihoodConfig(
        relative_observation_noise_standard_deviation=1.0,
        likelihood_model_id=REPLAY_LIKELIHOOD_BOOTSTRAP_MODEL_ID,
        calibration_protocol=(
            "bootstrap_only_for_key_independent_reverse_and_null_states_discarded"
        ),
        calibration_cluster_count=0,
    )
    calibration_step_count = int(calibration_replay_step_count)
    if calibration_step_count < 2:
        raise ValueError("replay 噪声 calibration step count 必须至少为2")
    variances_by_model: dict[str, list[float]] = defaultdict(list)
    cluster_ids_by_model: dict[str, list[str]] = defaultdict(list)
    model_roles: dict[str, set[str]] = defaultdict(set)
    for source in clean_records:
        if str(source.get("split") or "") != REPLAY_LIKELIHOOD_CALIBRATION_SOURCE_SPLIT:
            continue
        model_id = str(source.get("generation_model_id") or "").strip()
        if model_id not in pipelines:
            raise RuntimeError(f"replay 噪声 calibration 缺少已冻结模型 pipeline: {model_id}")
        prompt_id = str(source.get("prompt_id") or "")
        if prompt_id not in prompt_map:
            raise RuntimeError(f"replay 噪声 calibration 缺少 prompt 文本: {prompt_id}")
        video_path = _resolve_video_path(
            run_root,
            source.get("video_path"),
            fallback_dir="videos",
        )
        calibration_key_text = _generation_key(
            source,
            extra_context={
                "negative_role": "replay_noise_calibration_bootstrap",
            },
        )
        replay = _run_attacked_video_replay_for_model(
            pipelines[model_id],
            video_path,
            prompt=prompt_map[prompt_id],
            key_text=calibration_key_text,
            key_context=_validated_flow_key_context(
                source,
                prompt=prompt_map[prompt_id],
                scheduler=pipelines[model_id].scheduler,
            ),
            likelihood_config=bootstrap_config,
            replay_step_counts=(calibration_step_count,),
        )
        endpoint_energy = float(
            replay.endpoint_latent.detach().float().pow(2).mean().item()
        )
        if not math.isfinite(endpoint_energy) or endpoint_energy <= 0.0:
            raise RuntimeError("replay 噪声 calibration 的 observed endpoint energy 非有限正数")
        cluster_id = _replay_likelihood_calibration_cluster_id(source)
        for trajectory in replay.replay_trajectories:
            normalized_variance = (
                float(trajectory.null_residual_mean_squared_error) / endpoint_energy
            )
            variances_by_model[model_id].append(normalized_variance)
            cluster_ids_by_model[model_id].append(cluster_id)
        model_roles[model_id].add(str(source.get("cross_model_role") or "primary_model"))

    required_models = set(pipelines)
    observed_models = set(variances_by_model)
    if observed_models != required_models:
        missing = sorted(required_models - observed_models)
        raise RuntimeError(
            "replay 噪声 calibration 未覆盖所有正式生成模型: " + ", ".join(missing)
        )

    fitted: dict[str, ReplayGaussianLikelihoodConfig] = {}
    records: list[dict[str, Any]] = []
    minimum_count = max(2, int(minimum_clean_video_cluster_count))
    for model_id in sorted(required_models):
        config = fit_replay_gaussian_likelihood_config(
            variances_by_model[model_id],
            cluster_ids_by_model[model_id],
        )
        if config.calibration_cluster_count < minimum_count:
            raise RuntimeError(
                "replay 噪声 calibration 的独立 clean-video 簇不足: "
                f"{model_id}={config.calibration_cluster_count}, required={minimum_count}"
            )
        fitted[model_id] = config
        role_values = sorted(model_roles[model_id])
        record = {
            "replay_likelihood_calibration_record_id": build_stable_digest({
                "generation_model_id": model_id,
                "source_split": REPLAY_LIKELIHOOD_CALIBRATION_SOURCE_SPLIT,
                **config.as_dict(),
            }),
            "generation_model_id": model_id,
            "cross_model_role": (
                role_values[0] if len(role_values) == 1 else "mixed_model_role_invalid"
            ),
            "replay_likelihood_calibration_source_split": (
                REPLAY_LIKELIHOOD_CALIBRATION_SOURCE_SPLIT
            ),
            "replay_likelihood_calibration_clean_video_cluster_count": (
                config.calibration_cluster_count
            ),
            "replay_likelihood_calibration_null_residual_observation_count": len(
                variances_by_model[model_id]
            ),
            "replay_likelihood_calibration_step_counts": [
                calibration_step_count
            ],
            "replay_likelihood_calibration_grid_policy": (
                "single_preregistered_primary_grid_for_noise_fit"
            ),
            **config.as_dict(),
            "replay_likelihood_calibration_status": (
                "fitted_from_model_specific_calibration_clean_videos"
            ),
            "test_time_likelihood_update_blocked": True,
        }
        if len(role_values) != 1:
            raise RuntimeError(f"同一模型的 replay 噪声 calibration 混用了角色: {model_id}")
        records.append(record)
    return fitted, records


def _generation_causal_ablation_replay_sources(
    generation_records: Iterable[Mapping[str, Any]],
    *,
    maximum_identity_count_per_model_split: int,
) -> list[dict[str, Any]]:
    """选择预注册同 prompt/seed 生成变体，形成 Claim-1 因果配对。

    runtime robustness 的46种攻击只施加到完整方法；改变生成轨迹的内部消融在
    同一 prompt、seed 和模型的原始输出上比较。该设计隔离了速度约束干预，避免
    将攻击随机性混入 Claim-1，同时按 profile 的统计规模限制昂贵 replay 数量。
    """

    candidates = [
        dict(record)
        for record in generation_records
        if record.get("generation_status") == "success"
        and record.get("method_variant") in GENERATION_METHOD_VARIANTS
        and str(record.get("split") or "") in {"calibration", "test"}
        and str(record.get("sample_role") or "") != "clean_negative"
    ]
    grouped: dict[tuple[str, str], dict[tuple[str, str], list[dict[str, Any]]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    for record in candidates:
        group_key = (
            str(record.get("generation_model_id") or ""),
            str(record.get("split") or ""),
        )
        identity = (
            str(record.get("prompt_id") or ""),
            str(record.get("seed_id") or ""),
        )
        grouped[group_key][identity].append(record)

    selected_sources: list[dict[str, Any]] = []
    required_variants = set(GENERATION_METHOD_VARIANTS)
    maximum_count = max(1, int(maximum_identity_count_per_model_split))
    for group_key in sorted(grouped):
        complete_identities = [
            identity
            for identity, rows in grouped[group_key].items()
            if {str(row.get("method_variant")) for row in rows}
            == required_variants
        ]
        ranked_identities = sorted(
            complete_identities,
            key=lambda identity: build_stable_digest({
                "selection_protocol": (
                    "sstw_generation_causal_ablation_subset_v1"
                ),
                "generation_model_id": group_key[0],
                "split": group_key[1],
                "prompt_id": identity[0],
                "seed_id": identity[1],
            }),
        )[:maximum_count]
        selection_digest = build_stable_digest({
            "generation_model_id": group_key[0],
            "split": group_key[1],
            "selected_prompt_seed_identities": ranked_identities,
        })
        for rank, identity in enumerate(ranked_identities, start=1):
            for source in grouped[group_key][identity]:
                selected_sources.append({
                    **source,
                    "sample_role": "attacked_positive",
                    "attack_name": "no_attack_generation_causal_ablation",
                    "attack_runtime_status": "ready",
                    "attacked_video_path": source.get("video_path"),
                    "attacked_video_sha256": source.get("video_sha256"),
                    "source_video_sha256": source.get("video_sha256"),
                    "generation_source_video_sha256": source.get(
                        "video_sha256"
                    ),
                    "causal_ablation_subset_rank": rank,
                    "causal_ablation_subset_digest": selection_digest,
                    "causal_ablation_selection_protocol": (
                        "prompt_seed_hash_before_detection_complete_variant_block"
                    ),
                })
    return selected_sources


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
    ready_attacks = [
        record
        for record in attack_records
        if record.get("attack_runtime_status") == "ready"
        and record.get("method_variant") == "sstw_full_method"
    ]
    ready_attacks.extend(
        _generation_causal_ablation_replay_sources(
            successful_generation,
            maximum_identity_count_per_model_split=int(
                config.get("minimum_internal_ablation_trace_count") or 1
            ),
        )
    )
    if not ready_attacks:
        raise RuntimeError("缺少 ready runtime attack records, 不能执行正式 Flow replay")
    if str(config.get("paper_result_level") or "") in {
        "probe_paper",
        "pilot_paper",
        "full_paper",
    }:
        for record in [*successful_generation, *ready_attacks]:
            validate_generation_model_provenance(record)
    model_revisions: dict[str, str] = {}
    for record in [*ready_attacks, *clean_records]:
        model_id = str(record.get("generation_model_id") or "")
        if not model_id:
            continue
        revision = validate_generation_model_provenance(record)
        previous = model_revisions.setdefault(model_id, revision)
        if previous != revision:
            raise RuntimeError(
                f"同一 generation model ID 混用了多个不可变 revision: {model_id}"
            )
    pipelines = {
        model_id: _invoke_pipeline_loader(
            pipeline_loader,
            model_id=model_id,
            revision=revision,
        )
        for model_id, revision in sorted(model_revisions.items())
    }
    (
        replay_likelihood_configs,
        replay_likelihood_calibration_records,
    ) = _fit_model_specific_replay_likelihood_configs(
        run_root,
        clean_records,
        prompt_map,
        pipelines,
        minimum_clean_video_cluster_count=int(
            config.get(
                "minimum_replay_likelihood_calibration_clean_video_cluster_count",
                2,
            )
        ),
        calibration_replay_step_count=int(
            config.get("replay_likelihood_calibration_step_count", 20)
        ),
    )
    all_prompts = list(dict.fromkeys(prompt_map.values()))
    evidence_records: list[dict[str, Any]] = []
    failure_records: list[dict[str, Any]] = []

    for source in ready_attacks:
        method_variant = str(source.get("method_variant") or "sstw_full_method")
        if method_variant not in GENERATION_METHOD_VARIANTS:
            continue
        try:
            prompt = prompt_map[str(source.get("prompt_id"))]
            wrong_prompt = next(value for value in all_prompts if value != prompt)
            pipeline = pipelines[str(source.get("generation_model_id"))]
            key_text = _generation_key(source)
            key_context = _validated_flow_key_context(
                source,
                prompt=prompt,
                scheduler=pipeline.scheduler,
            )
            video_path = _resolve_video_path(run_root, source.get("attacked_video_path"), fallback_dir="attacked_videos")
            replay = _run_attacked_video_replay_for_model(
                pipeline,
                video_path,
                prompt=prompt,
                key_text=key_text,
                key_context=key_context,
                likelihood_config=replay_likelihood_configs[
                    str(source.get("generation_model_id"))
                ],
            )
            control_payload = (
                _control_payload(
                    pipeline,
                    replay,
                    prompt=prompt,
                    wrong_prompt=wrong_prompt,
                    key_text=key_text,
                    wrong_key_text=_wrong_owner_generation_key(source),
                )
                if method_variant == "sstw_full_method"
                else {
                    "replay_control_execution_status": (
                        "not_applicable_internal_ablation_variant"
                    )
                }
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
                **control_payload,
            }
            formal_positive = with_flow_evidence_protocol_defaults(
                payload,
                trajectory_source_level="attacked_video_model_velocity_inversion_replay",
                flow_state_admissibility_status="pending_frozen_detector",
                claim_support_status="sstw_complete_flow_evidence_ready",
            )
            evidence_records.append(formal_positive)
            if method_variant == "sstw_full_method":
                evidence_records.extend(
                    _controlled_negative_records_from_positive(formal_positive)
                )
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
                source_model_id = str(source.get("generation_model_id"))
                source_revision = validate_generation_model_provenance(source)
                pipeline = pipelines.get(source_model_id) or _invoke_pipeline_loader(
                    pipeline_loader,
                    model_id=source_model_id,
                    revision=source_revision,
                )
                pipelines[source_model_id] = pipeline
                video_path = _resolve_video_path(run_root, source.get("video_path"), fallback_dir="videos")
                base_replay = _run_attacked_video_replay_for_model(
                    pipeline,
                    video_path,
                    prompt=prompt,
                    key_text=_generation_key(
                        source,
                        extra_context={"negative_role": "clean_replay_base"},
                    ),
                    key_context=_validated_flow_key_context(
                        source,
                        prompt=prompt,
                        scheduler=pipeline.scheduler,
                    ),
                    likelihood_config=replay_likelihood_configs[
                        source_model_id
                    ],
                )
                # 这里只为真实生成机制变体构造独立 candidate-key 负假设。
                # 检测器专用消融必须在评分阶段复用 full-method 的同一 replay、
                # 同一 candidate key 和同一统计簇, 否则会引入伪重复与随机性混杂。
                for method_variant in GENERATION_METHOD_VARIANTS:
                    for trial_index in range(trial_count):
                        trial_key = _generation_key(
                            source,
                            extra_context={
                                "negative_role": "clean_negative_candidate_key",
                                "method_variant": method_variant,
                                "trial_index": trial_index,
                            },
                        )
                        endpoint = _compute_replay_endpoint_evidence_for_key(
                            base_replay,
                            key_text=trial_key,
                            key_context=base_replay.key_context,
                        )
                        hypothesis, path = _evaluate_fixed_replay_hypothesis_for_key(
                            pipeline,
                            base_replay,
                            prompt=prompt,
                            key_text=trial_key,
                            key_context=base_replay.key_context,
                        )
                        trial_reliability = math.exp(-0.5 * (
                            float(hypothesis.candidate_residual_mean_squared_error)
                            / max(float(hypothesis.observation_noise_variance), 1e-12)
                        ))
                        state_sequence = build_flow_state_observation_sequence(
                            base_replay,
                            key_text=trial_key,
                            trajectory=hypothesis,
                            key_context=base_replay.key_context,
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
                            "negative_family": "clean_unwatermarked_candidate_key_hypothesis",
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
                            "replay_observation_noise_variance_mean": round(
                                hypothesis.observation_noise_variance,
                                10,
                            ),
                            "replay_candidate_log_likelihood_per_dimension_mean": round(
                                hypothesis.candidate_log_likelihood_per_dimension,
                                8,
                            ),
                            "replay_null_log_likelihood_per_dimension_mean": round(
                                hypothesis.null_log_likelihood_per_dimension,
                                8,
                            ),
                            "replay_likelihood_model_id": hypothesis.replay_likelihood_model_id,
                            **base_replay.replay_likelihood_config.as_dict(),
                            "flow_state_observation_sequence": state_sequence,
                            "flow_state_observation_sequence_status": (
                                "measured_from_fixed_replay_path"
                            ),
                            "flow_state_observation_step_count": len(state_sequence),
                            "flow_state_observation_formal_context_complete": all(
                                row.get("flow_tubelet_formal_context_complete")
                                is True
                                and row.get("path_quadrature_context_complete")
                                is True
                                for row in state_sequence
                            ),
                            "flow_tubelet_key_context_digest": (
                                flow_tubelet_key_context_digest(
                                    base_replay.key_context
                                )
                                if base_replay.key_context is not None
                                else None
                            ),
                            "flow_state_transition_source": (
                                "calibration_fitted_linear_gaussian_dynamics"
                            ),
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
    # Calibration split 只负责拟合状态模型、Platt 映射与 fixed-FPR 阈值。
    # Claim-3 的“可靠后验”必须另外在 test split 上评价，且不得更新任何参数。
    heldout_posterior_records = build_heldout_posterior_calibration_records(
        scored_records,
        config,
    )
    heldout_posterior_audit = audit_heldout_posterior_calibration_records(
        heldout_posterior_records,
        config,
    )
    paired_path_records = _paired_path_gain_records(scored_records, calibrations)
    paired_velocity_records = _paired_velocity_causal_records(
        scored_records,
        calibrations,
    )
    mechanism_audit = _audit_three_layer_mechanism(
        scored_records,
        paired_path_records,
        paired_velocity_records,
        target_fpr=float(config["target_fpr"]),
        minimum_velocity_causal_pair_count=int(
            config["minimum_internal_ablation_trace_count"]
        ),
    )
    cross_model_audit = _audit_cross_model_generalization(
        scored_records,
        paired_path_records,
        paired_velocity_records,
        target_fpr=float(config["target_fpr"]),
    )
    positive_records = [record for record in scored_records if record.get("sample_role") == "attacked_positive"]
    negative_records = [
        record for record in scored_records
        if record.get("sample_role") in {"clean_negative", "controlled_negative"}
    ]
    calibration_negative_family_counts = _negative_family_cluster_counts(
        negative_records,
        split="calibration",
    )
    heldout_negative_family_counts = _negative_family_cluster_counts(
        negative_records,
        split="test",
    )
    negative_family_mechanism_pass = (
        set(calibration_negative_family_counts) == FORMAL_NEGATIVE_HYPOTHESIS_FAMILIES
        and set(heldout_negative_family_counts) == FORMAL_NEGATIVE_HYPOTHESIS_FAMILIES
        and min(calibration_negative_family_counts.values(), default=0)
        >= int(config.get("minimum_calibration_negative_event_count_per_family", 1))
        and min(heldout_negative_family_counts.values(), default=0)
        >= int(config.get("minimum_heldout_negative_event_count_per_family", 1))
    )
    required_variants = set(FORMAL_METHOD_VARIANTS)
    observed_variants = {str(record.get("method_variant")) for record in positive_records}
    claim3_records = [
        record for record in positive_records
        if record.get("method_variant") == "sstw_full_method"
        and record.get("replay_control_execution_status") == "measured_formal"
        and record.get("replay_control_fixed_reverse_path_reused") is True
    ]
    watermark_key_derivation_failures = [
        {
            "formal_flow_evidence_unit_id": record.get(
                "formal_flow_evidence_unit_id"
            ),
            "watermark_key_derivation_id": record.get(
                "watermark_key_derivation_id"
            ),
            "watermark_key_id": record.get("watermark_key_id"),
        }
        for record in positive_records
        if record.get("watermark_key_derivation_id")
        != WATERMARK_KEY_DERIVATION_ID
        or not str(record.get("watermark_key_id") or "").strip()
    ]
    generation_model_provenance_failures: list[dict[str, Any]] = []
    for record in scored_records:
        try:
            validate_generation_model_provenance(record)
        except (KeyError, TypeError, ValueError) as exc:
            generation_model_provenance_failures.append({
                "formal_flow_evidence_unit_id": record.get(
                    "formal_flow_evidence_unit_id"
                ),
                "generation_model_id": record.get("generation_model_id"),
                "generation_model_commit_or_hash": record.get(
                    "generation_model_commit_or_hash"
                ),
                "generation_model_provenance_failure_reason": str(exc),
            })
    posterior_calibration_failures = [
        {
            "method_variant": record.get("method_variant"),
            "posterior_calibration_brier_score": record.get("posterior_calibration_brier_score"),
            "posterior_calibration_expected_calibration_error": record.get(
                "posterior_calibration_expected_calibration_error"
            ),
            "posterior_calibration_group_count": record.get("posterior_calibration_group_count"),
            "calibration_negative_cluster_count": record.get(
                "calibration_negative_cluster_count"
            ),
            "calibration_positive_cluster_count": record.get(
                "calibration_positive_cluster_count"
            ),
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
        or (
            record.get("cross_model_role") != "cross_model_validation_model"
            and int(record.get("calibration_negative_cluster_count") or 0)
            < int(config.get("minimum_calibration_unique_video_count", 2))
        )
        or (
            record.get("cross_model_role") != "cross_model_validation_model"
            and int(record.get("calibration_positive_cluster_count") or 0)
            < int(config.get("minimum_calibration_unique_video_count", 2))
        )
    ]
    state_space_posterior_mechanism_failures = _state_space_posterior_mechanism_failures(
        scored_records,
        threshold_records,
    )
    minimum_replay_calibration_clusters = max(
        2,
        int(
            config.get(
                "minimum_replay_likelihood_calibration_clean_video_cluster_count",
                2,
            )
        ),
    )
    replay_likelihood_calibration_failures = [
        {
            "generation_model_id": record.get("generation_model_id"),
            "replay_likelihood_calibration_status": record.get(
                "replay_likelihood_calibration_status"
            ),
            "replay_likelihood_calibration_clean_video_cluster_count": record.get(
                "replay_likelihood_calibration_clean_video_cluster_count"
            ),
        }
        for record in replay_likelihood_calibration_records
        if record.get("replay_likelihood_model_id")
        != REPLAY_GAUSSIAN_LIKELIHOOD_MODEL_ID
        or record.get("replay_likelihood_calibration_protocol")
        != "calibration_clean_video_null_residual_cluster_equal_mle"
        or record.get("replay_likelihood_calibration_source_split")
        != REPLAY_LIKELIHOOD_CALIBRATION_SOURCE_SPLIT
        or record.get("replay_likelihood_calibration_status")
        != "fitted_from_model_specific_calibration_clean_videos"
        or record.get("test_time_likelihood_update_blocked") is not True
        or record.get("replay_likelihood_calibration_grid_policy")
        != "single_preregistered_primary_grid_for_noise_fit"
        or record.get("replay_likelihood_calibration_step_counts")
        != [int(config.get("replay_likelihood_calibration_step_count", 20))]
        or int(
            record.get("replay_likelihood_calibration_clean_video_cluster_count")
            or 0
        )
        < minimum_replay_calibration_clusters
    ]
    calibrated_model_ids = {
        str(record.get("generation_model_id") or "")
        for record in replay_likelihood_calibration_records
    }
    if calibrated_model_ids != set(pipelines):
        replay_likelihood_calibration_failures.append({
            "failure_scope": "generation_model_coverage",
            "missing_generation_model_ids": sorted(set(pipelines) - calibrated_model_ids),
            "unexpected_generation_model_ids": sorted(calibrated_model_ids - set(pipelines)),
        })
    formal_flow_evidence_pass = (
        bool(positive_records)
        and bool(negative_records)
        and not failure_records
        and required_variants.issubset(observed_variants)
        and bool(claim3_records)
        and not watermark_key_derivation_failures
        and not generation_model_provenance_failures
        and not posterior_calibration_failures
        and not state_space_posterior_mechanism_failures
        and not replay_likelihood_calibration_failures
        and heldout_posterior_audit["heldout_posterior_calibration_decision"]
        == "PASS"
        and negative_family_mechanism_pass
        and mechanism_audit["three_layer_mechanism_pre_replay_decision"] == "PASS"
        and cross_model_audit["cross_model_generalization_decision"] in {"PASS", "NOT_CONFIGURED"}
    )
    audit = {
        "stage_id": "formal_flow_evidence_runner",
        "formal_flow_evidence_decision": "PASS" if formal_flow_evidence_pass else "FAIL",
        "formal_flow_evidence_record_count": len(scored_records),
        "formal_flow_positive_record_count": len(positive_records),
        "formal_flow_clean_negative_record_count": len(negative_records),
        "formal_negative_hypothesis_family_decision": (
            "PASS" if negative_family_mechanism_pass else "FAIL"
        ),
        "formal_required_negative_hypothesis_families": sorted(
            FORMAL_NEGATIVE_HYPOTHESIS_FAMILIES
        ),
        "formal_calibration_negative_family_cluster_counts": (
            calibration_negative_family_counts
        ),
        "formal_heldout_negative_family_cluster_counts": (
            heldout_negative_family_counts
        ),
        "formal_flow_failure_record_count": len(failure_records),
        "formal_flow_observed_method_variants": sorted(observed_variants),
        "formal_flow_missing_method_variants": sorted(required_variants - observed_variants),
        "formal_flow_threshold_record_count": len(threshold_records),
        "claim3_real_replay_record_count": len(claim3_records),
        "watermark_key_derivation_decision": (
            "PASS" if not watermark_key_derivation_failures else "FAIL"
        ),
        "watermark_key_derivation_failures": watermark_key_derivation_failures,
        "generation_model_provenance_decision": (
            "PASS" if not generation_model_provenance_failures else "FAIL"
        ),
        "generation_model_provenance_failures": (
            generation_model_provenance_failures
        ),
        "posterior_probability_calibration_decision": (
            "PASS" if not posterior_calibration_failures else "FAIL"
        ),
        "posterior_probability_calibration_failures": posterior_calibration_failures,
        "state_space_posterior_mechanism_decision": (
            "PASS" if not state_space_posterior_mechanism_failures else "FAIL"
        ),
        "state_space_posterior_mechanism_failures": (
            state_space_posterior_mechanism_failures
        ),
        "replay_likelihood_calibration_decision": (
            "PASS" if not replay_likelihood_calibration_failures else "FAIL"
        ),
        "replay_likelihood_calibration_record_count": len(
            replay_likelihood_calibration_records
        ),
        "replay_likelihood_calibration_failures": (
            replay_likelihood_calibration_failures
        ),
        "heldout_posterior_calibration_decision": heldout_posterior_audit[
            "heldout_posterior_calibration_decision"
        ],
        "heldout_posterior_calibration_record_count": heldout_posterior_audit[
            "heldout_posterior_calibration_record_count"
        ],
        "heldout_posterior_missing_scopes": heldout_posterior_audit[
            "heldout_posterior_missing_scopes"
        ],
        "heldout_posterior_blocked_scopes": heldout_posterior_audit[
            "heldout_posterior_blocked_scopes"
        ],
        "claim_1_velocity_constraint_detectable_watermark_decision": mechanism_audit["claim_1_velocity_constraint_detectable_watermark_decision"],
        "claim_2_path_evidence_independent_gain_decision": mechanism_audit["claim_2_path_evidence_independent_gain_decision"],
        "cross_model_generalization_decision": cross_model_audit["cross_model_generalization_decision"],
        "cross_model_generalization_model_ids": cross_model_audit["cross_model_generalization_model_ids"],
        "cross_model_generalization_record_count": cross_model_audit["cross_model_generalization_record_count"],
        "target_fpr": float(config["target_fpr"]),
        "test_time_threshold_update_blocked": True,
        "claim_support_status": (
            "sstw_complete_paper_mechanism_ready"
            if formal_flow_evidence_pass
            else "sstw_complete_paper_mechanism_blocked"
        ),
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
    write_jsonl(
        run_root
        / "thresholds"
        / "replay_gaussian_likelihood_calibrations.jsonl",
        replay_likelihood_calibration_records,
    )
    write_heldout_posterior_calibration_artifacts(
        run_root,
        heldout_posterior_records,
        heldout_posterior_audit,
    )
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
        f"- replay_likelihood_calibration_decision: {audit['replay_likelihood_calibration_decision']}\n"
        f"- heldout_posterior_calibration_decision: {audit['heldout_posterior_calibration_decision']}\n"
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
