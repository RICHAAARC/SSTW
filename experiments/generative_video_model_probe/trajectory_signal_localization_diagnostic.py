"""Stage 0-D：在既有4-source上定位 trajectory watermark 信号消失位置。

该入口只读取已冻结的 generation、attack 与 replay calibration artifacts。它不生成
视频、不拟合噪声或阈值、不选择时间网格，也不产生 paper-level evidence。
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from evaluation.protocol.record_writer import write_json, write_jsonl
from experiments.generative_video_model_probe.colab_runtime import (
    _configure_wan_flow_match_euler_scheduler,
    validate_generation_model_provenance,
)
from experiments.generative_video_model_probe.formal_flow_evidence_runner import (
    _compute_replay_endpoint_evidence_for_key,
    _evaluate_fixed_replay_hypothesis_for_key,
    _flow_key_context,
    _generation_key,
    _invoke_pipeline_loader,
    _load_pipeline,
    _prompt_text_by_id,
    _read_jsonl,
    _resolve_video_path,
    _run_attacked_video_replay_for_model,
    _validated_flow_key_context,
    _wrong_owner_generation_key,
    build_flow_state_observation_sequence,
)
from main.methods.state_space_watermark.flow_tubelet_key_code import (
    FlowTubeletKeyCodeConfig,
    build_integrated_flow_tubelet_key_direction_like,
)
from main.methods.state_space_watermark.replay_inversion import (
    ReplayGaussianLikelihoodConfig,
    estimate_replay_uncertainty,
)
from main.methods.state_space_watermark.wan_flow_replay_backend import (
    _endpoint_integration_grid,
    build_flow_schedule_points,
    score_replay_trajectory_for_key,
)


DEFAULT_CONFIG_PATH = (
    "configs/protocol/sstw_trajectory_signal_localization_diagnostic.json"
)
DIAGNOSTIC_RECORD_VERSION = "trajectory_signal_localization_diagnostic_v1"


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON 顶层必须是对象: {path}")
    return value


def _sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_digest(payload: Mapping[str, Any]) -> str:
    return sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def validate_signal_localization_config(config: Mapping[str, Any]) -> None:
    """冻结 Stage 0-D 范围与停止规则，拒绝范围漂移。"""

    prohibited = (
        "large_scale_generation_allowed",
        "external_baseline_execution_allowed",
        "cross_project_integration_allowed",
        "fixed_fpr_evaluation_allowed",
        "test_split_claims_allowed",
        "stage_progression_allowed",
        "time_grid_selection_allowed",
    )
    invalid = [name for name in prohibited if config.get(name) is not False]
    if invalid:
        raise ValueError("Stage 0-D 禁止项未冻结: " + ", ".join(invalid))
    if config.get("profile_id") != "sstw_trajectory_signal_localization_diagnostic":
        raise ValueError("Stage 0-D 必须使用独立 signal-localization profile")
    if [int(value) for value in config.get("replay_grid_step_counts") or []] != [8, 20, 40]:
        raise ValueError("Stage 0-D replay grids 必须固定为 [8,20,40]")
    if int(config.get("frozen_likelihood_calibration_step_count") or 0) != 20:
        raise ValueError("Stage 0-D 必须复用20步冻结 likelihood calibration")
    expected_variants = {
        "sstw_full_method",
        "endpoint_only_control",
        "sstw_clean_unwatermarked_reference",
    }
    if set(config.get("required_source_method_variants") or ()) != expected_variants:
        raise ValueError("Stage 0-D source variants 不完整")
    if config.get("owner_key_direction_preflight_required") is not True:
        raise ValueError("Stage 0-D 必须启用 owner-key direction preflight")
    latent_shape = config.get("expected_wan_endpoint_latent_shape")
    if [int(value) for value in latent_shape or []] != [1, 16, 9, 40, 64]:
        raise ValueError("Stage 0-D Wan endpoint latent shape 必须冻结为 [1,16,9,40,64]")


def _resolved_source_video(
    source_root: Path,
    record: Mapping[str, Any],
    *,
    attacked: bool,
) -> Path:
    return _resolve_video_path(
        source_root,
        record.get("attacked_video_path" if attacked else "video_path"),
        fallback_dir="attacked_videos" if attacked else "videos",
    )


def build_immutable_input_snapshot(
    source_root: str | Path,
    output_root: str | Path,
    config: Mapping[str, Any],
    *,
    phase: str,
) -> dict[str, Any]:
    """按显式阶段验证真实输入依赖，且禁止 source/output 混写。"""

    source = Path(source_root).resolve()
    output = Path(output_root).resolve()
    if source == output or source in output.parents:
        raise ValueError("Stage 0-D output 不得写入 source run tree")
    valid_phases = {"credential_preflight", "no_attack", "attacked", "decision"}
    if phase not in valid_phases:
        raise ValueError(f"Stage 0-D immutable input phase 不受支持: {phase}")
    generation_path = source / "records" / "generation_records.jsonl"
    prompt_suite_path = source / "datasets" / "prompt_seed_suite.json"
    attack_path = source / "records" / "trajectory_replay_smoke_attack_records.jsonl"
    calibration_path = (
        source / "records" / "trajectory_replay_smoke_likelihood_calibrations.jsonl"
    )
    decision_path = source / "artifacts" / "trajectory_replay_smoke_decision.json"
    manifest_path = source / "artifacts" / "trajectory_replay_smoke_manifest.json"
    required_files = [generation_path, prompt_suite_path]
    if phase != "credential_preflight":
        required_files.extend(
            (attack_path, calibration_path, decision_path, manifest_path)
        )
    for required in required_files:
        if not required.is_file():
            raise FileNotFoundError(f"Stage 0-D 缺少冻结输入: {required}")

    variants = set(config["required_source_method_variants"])
    generations = [
        row
        for row in _read_jsonl(generation_path)
        if row.get("generation_status") == "success"
        and row.get("method_variant") in variants
    ]
    expected_generation_count = int(config["required_source_video_count"]) * len(variants)
    if len(generations) != expected_generation_count:
        raise RuntimeError(
            "Stage 0-D original video coverage 不完整: "
            f"observed={len(generations)}, expected={expected_generation_count}"
        )
    for variant in variants:
        observed_variant_count = sum(
            row.get("method_variant") == variant for row in generations
        )
        if observed_variant_count != int(config["required_source_video_count"]):
            raise RuntimeError(
                "Stage 0-D generation variant coverage 不完整: "
                f"variant={variant}, observed={observed_variant_count}"
            )
    for row in generations:
        validate_generation_model_provenance(row)
    prompt_map = _prompt_text_by_id(_read_json(prompt_suite_path))
    if not prompt_map:
        raise RuntimeError("Stage 0-D prompt suite 未提供 prompt_id/prompt_text")
    missing_prompt_ids = sorted(
        {
            str(row.get("prompt_id") or "")
            for row in generations
            if str(row.get("prompt_id") or "") not in prompt_map
        }
    )
    if missing_prompt_ids:
        raise RuntimeError(
            "Stage 0-D generation records 缺少 prompt suite 绑定: "
            + ", ".join(missing_prompt_ids)
        )

    if phase == "credential_preflight":
        governed_files = [generation_path, prompt_suite_path]
        snapshot = {
            "record_version": DIAGNOSTIC_RECORD_VERSION,
            "profile_id": config["profile_id"],
            "immutable_input_preflight_status": "ready",
            "immutable_input_scope": (
                "credential_preflight_generation_and_prompt_only"
            ),
            "source_run_root": str(source),
            "output_run_root": str(output),
            "generation_record_count": len(generations),
            "prompt_suite_prompt_count": len(prompt_map),
            "attack_input_status": "not_applicable_for_credential_preflight",
            "attack_record_count": 0,
            "likelihood_calibration_input_status": (
                "not_applicable_for_credential_preflight"
            ),
            "frozen_likelihood_calibration_record_id": None,
            "governed_input_sha256": {
                str(path): _sha256_file(path) for path in governed_files
            },
            "video_inputs": [],
            "claim_support_status": config["claim_support_status"],
        }
        snapshot["immutable_input_snapshot_digest"] = _stable_digest(snapshot)
        return snapshot

    attacks = [
        row
        for row in _read_jsonl(attack_path)
        if row.get("attack_runtime_status") == "ready"
        and row.get("method_variant") in variants
        and row.get("attack_name") in set(config["required_attacked_video_condition_ids"])
    ]
    expected_attack_count = expected_generation_count * len(
        config["required_attacked_video_condition_ids"]
    )
    if len(attacks) != expected_attack_count:
        raise RuntimeError(
            "Stage 0-D attacked video coverage 不完整: "
            f"observed={len(attacks)}, expected={expected_attack_count}"
        )
    calibrations = _read_jsonl(calibration_path)
    if len(calibrations) != 1:
        raise RuntimeError("Stage 0-D 要求唯一冻结 replay likelihood calibration")
    calibration = calibrations[0]
    if [int(value) for value in calibration.get("replay_likelihood_calibration_step_counts") or []] != [20]:
        raise RuntimeError("冻结 replay likelihood calibration 不是20步主网格")
    relative_std = float(
        calibration.get("replay_relative_observation_noise_standard_deviation") or 0.0
    )
    if not math.isfinite(relative_std) or relative_std <= 0.0:
        raise RuntimeError("冻结 replay likelihood calibration 缺少有限正标准差")

    video_inputs: list[dict[str, Any]] = []
    for condition, rows, attacked in (
        (str(config["no_attack_video_condition_id"]), generations, False),
        ("existing_attacked_video", attacks, True),
    ):
        for row in rows:
            path = _resolved_source_video(source, row, attacked=attacked)
            if not path.is_file():
                raise FileNotFoundError(f"Stage 0-D 输入视频不存在: {path}")
            observed_hash = _sha256_file(path)
            recorded_hash = str(
                row.get("attacked_video_sha256" if attacked else "video_sha256") or ""
            )
            if recorded_hash and observed_hash != recorded_hash:
                raise RuntimeError(f"Stage 0-D 输入视频摘要漂移: {path}")
            video_inputs.append({
                "video_condition_id": (
                    str(row.get("attack_name")) if attacked else condition
                ),
                "trajectory_trace_id": row.get("trajectory_trace_id"),
                "method_variant": row.get("method_variant"),
                "video_path": str(path),
                "video_sha256": observed_hash,
            })
    governed_files = [
        generation_path,
        prompt_suite_path,
        attack_path,
        calibration_path,
        decision_path,
        manifest_path,
    ]
    snapshot = {
        "record_version": DIAGNOSTIC_RECORD_VERSION,
        "profile_id": config["profile_id"],
        "immutable_input_preflight_status": "ready",
        "immutable_input_scope": "full_replay_diagnostic_inputs",
        "source_run_root": str(source),
        "output_run_root": str(output),
        "generation_record_count": len(generations),
        "prompt_suite_prompt_count": len(prompt_map),
        "attack_input_status": "ready",
        "attack_record_count": len(attacks),
        "likelihood_calibration_input_status": "ready",
        "frozen_likelihood_calibration_record_id": calibration.get(
            "replay_likelihood_calibration_record_id"
        ),
        "governed_input_sha256": {
            str(path): _sha256_file(path) for path in governed_files
        },
        "video_inputs": sorted(
            video_inputs,
            key=lambda row: (
                str(row["video_condition_id"]),
                str(row["trajectory_trace_id"]),
                str(row["method_variant"]),
            ),
        ),
        "claim_support_status": config["claim_support_status"],
    }
    snapshot["immutable_input_snapshot_digest"] = _stable_digest(snapshot)
    return snapshot


def _load_owner_key_preflight_scheduler(
    *,
    model_id: str,
    revision: str,
) -> Any:
    """只加载 scheduler 配置，不加载 Transformer、VAE 或视频 pipeline。"""

    from diffusers import UniPCMultistepScheduler

    scheduler = UniPCMultistepScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
        revision=revision,
        local_files_only=True,
    )
    holder = type("OwnerKeySchedulerHolder", (), {})()
    holder.scheduler = scheduler
    _configure_wan_flow_match_euler_scheduler(holder)
    return holder.scheduler


def _rebuild_owner_key_direction_metadata(
    *,
    source_record: Mapping[str, Any],
    prompt: str,
    scheduler: Any,
    key_text: str,
    latent_shape: tuple[int, ...],
) -> dict[str, Any]:
    """在 CPU 上重建生成期 endpoint direction 的公开摘要元数据。"""

    import torch

    key_context = _flow_key_context(prompt, scheduler)
    schedule = build_flow_schedule_points(
        scheduler,
        num_inference_steps=int(source_record.get("num_inference_steps") or 0),
        device=torch.device("cpu"),
    )
    phases, weights = _endpoint_integration_grid(
        schedule,
        FlowTubeletKeyCodeConfig(),
    )
    reference = torch.zeros(latent_shape, dtype=torch.float32)
    _direction, metadata = build_integrated_flow_tubelet_key_direction_like(
        reference,
        key_text=key_text,
        key_context=key_context,
        flow_phases=phases,
        integration_weights=weights,
    )
    return {
        "endpoint_key_direction_digest": metadata["flow_key_direction_digest"],
        "endpoint_key_context_digest": metadata[
            "flow_tubelet_key_context_digest"
        ],
        "endpoint_integrated_phase_count": metadata["flow_integrated_phase_count"],
        "endpoint_integrated_weight_sum": metadata["flow_integrated_weight_sum"],
    }


def _owner_key_preflight_record(
    config: Mapping[str, Any],
    *,
    status: str,
    expected_count: int,
    watermark_key_id: str | None,
    failure_reason_code: str | None = None,
    direction_match_count: int = 0,
    context_match_count: int = 0,
    phase_grid_match_count: int = 0,
    direction_mismatch_trace_ids: Iterable[str] = (),
) -> dict[str, Any]:
    record = {
        "record_version": DIAGNOSTIC_RECORD_VERSION,
        "profile_id": config["profile_id"],
        "owner_key_direction_preflight_status": status,
        "owner_key_direction_preflight_failure_reason_code": failure_reason_code,
        "owner_key_direction_expected_count": expected_count,
        "owner_key_direction_match_count": direction_match_count,
        "owner_key_direction_all_match": bool(
            expected_count > 0 and direction_match_count == expected_count
        ),
        "owner_key_context_all_match": bool(
            expected_count > 0 and context_match_count == expected_count
        ),
        "owner_key_phase_grid_all_match": bool(
            expected_count > 0 and phase_grid_match_count == expected_count
        ),
        "owner_key_direction_mismatch_trace_ids": sorted(
            str(value) for value in direction_mismatch_trace_ids
        ),
        "watermark_key_id": watermark_key_id,
        "claim_support_status": config["claim_support_status"],
    }
    record["owner_key_direction_preflight_record_id"] = _stable_digest(record)
    return record


def build_owner_key_direction_preflight(
    source_root: str | Path,
    config: Mapping[str, Any],
    *,
    scheduler_loader: Any = _load_owner_key_preflight_scheduler,
    direction_metadata_builder: Any = _rebuild_owner_key_direction_metadata,
) -> dict[str, Any]:
    """行为核验 owner secret 是否能重建4个 full-source 生成方向。"""

    source = Path(source_root).resolve()
    full_rows = [
        row
        for row in _read_jsonl(source / "records" / "generation_records.jsonl")
        if row.get("generation_status") == "success"
        and row.get("method_variant") == "sstw_full_method"
    ]
    expected_count = int(config["required_source_video_count"])
    key_ids = {
        str(row.get("watermark_key_id") or "").strip()
        for row in full_rows
        if str(row.get("watermark_key_id") or "").strip()
    }
    watermark_key_id = next(iter(key_ids)) if len(key_ids) == 1 else None
    if len(full_rows) != expected_count or len(key_ids) != 1:
        return _owner_key_preflight_record(
            config,
            status="blocked",
            expected_count=expected_count,
            watermark_key_id=watermark_key_id,
            failure_reason_code="owner_key_preflight_source_coverage_invalid",
        )

    # 先完成凭据派生；凭据缺失时不得触发任何 scheduler/model 加载。
    try:
        derived_keys = [_generation_key(row) for row in full_rows]
    except Exception:
        return _owner_key_preflight_record(
            config,
            status="blocked",
            expected_count=expected_count,
            watermark_key_id=watermark_key_id,
            failure_reason_code="owner_key_credentials_unavailable_or_incompatible",
        )

    prompt_map = _prompt_text_by_id(
        _read_json(source / "datasets" / "prompt_seed_suite.json")
    )
    latent_shape = tuple(int(value) for value in config["expected_wan_endpoint_latent_shape"])
    try:
        revisions: dict[str, str] = {}
        for row in full_rows:
            model_id = str(row.get("generation_model_id") or "")
            revision = validate_generation_model_provenance(row)
            previous = revisions.setdefault(model_id, revision)
            if previous != revision:
                raise RuntimeError("同一 generation model 混用不同 revision")
        schedulers = {
            model_id: scheduler_loader(model_id=model_id, revision=revision)
            for model_id, revision in sorted(revisions.items())
        }
        direction_match_count = 0
        context_match_count = 0
        phase_grid_match_count = 0
        direction_mismatch_trace_ids: list[str] = []
        for row, key_text in zip(full_rows, derived_keys):
            trace_id = str(row.get("trajectory_trace_id") or "")
            prompt_id = str(row.get("prompt_id") or "")
            if prompt_id not in prompt_map:
                raise RuntimeError("owner-key preflight 缺少生成 prompt")
            observed = direction_metadata_builder(
                source_record=row,
                prompt=prompt_map[prompt_id],
                scheduler=schedulers[str(row.get("generation_model_id") or "")],
                key_text=key_text,
                latent_shape=latent_shape,
            )
            direction_matches = (
                str(observed.get("endpoint_key_direction_digest") or "")
                == str(row.get("endpoint_key_direction_digest") or "")
            )
            context_matches = (
                str(observed.get("endpoint_key_context_digest") or "")
                == str(
                    row.get("endpoint_key_context_digest")
                    or row.get("flow_tubelet_key_context_digest")
                    or ""
                )
            )
            phase_matches = int(
                observed.get("endpoint_integrated_phase_count") or 0
            ) == int(row.get("endpoint_integrated_phase_count") or 0)
            weight_matches = math.isclose(
                float(observed.get("endpoint_integrated_weight_sum") or 0.0),
                float(row.get("endpoint_integrated_weight_sum") or 0.0),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            direction_match_count += int(direction_matches)
            context_match_count += int(context_matches)
            phase_grid_match_count += int(phase_matches and weight_matches)
            if not direction_matches:
                direction_mismatch_trace_ids.append(trace_id)
        all_ready = (
            direction_match_count == expected_count
            and context_match_count == expected_count
            and phase_grid_match_count == expected_count
        )
        return _owner_key_preflight_record(
            config,
            status="ready" if all_ready else "mismatch",
            expected_count=expected_count,
            watermark_key_id=watermark_key_id,
            failure_reason_code=(
                None if all_ready else "owner_key_generation_direction_mismatch"
            ),
            direction_match_count=direction_match_count,
            context_match_count=context_match_count,
            phase_grid_match_count=phase_grid_match_count,
            direction_mismatch_trace_ids=direction_mismatch_trace_ids,
        )
    except Exception:
        return _owner_key_preflight_record(
            config,
            status="blocked",
            expected_count=expected_count,
            watermark_key_id=watermark_key_id,
            failure_reason_code="owner_key_direction_rebuild_failed",
        )


def _likelihood_config(source_root: Path) -> ReplayGaussianLikelihoodConfig:
    row = _read_jsonl(
        source_root / "records" / "trajectory_replay_smoke_likelihood_calibrations.jsonl"
    )[0]
    return ReplayGaussianLikelihoodConfig(
        relative_observation_noise_standard_deviation=float(
            row["replay_relative_observation_noise_standard_deviation"]
        ),
        minimum_observation_noise_variance=float(
            row["replay_minimum_observation_noise_variance"]
        ),
        likelihood_model_id=str(row["replay_likelihood_model_id"]),
        calibration_protocol=str(row["replay_likelihood_calibration_protocol"]),
        calibration_cluster_count=int(row["replay_likelihood_calibration_cluster_count"]),
    )


def _source_identity(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("generation_model_id") or ""),
        str(row.get("prompt_id") or ""),
        str(row.get("seed_id") or ""),
        str(row.get("video_condition_id") or row.get("attack_name") or ""),
    )


def _grid_role(config: Mapping[str, Any], step_count: int) -> str:
    if step_count == int(config["generation_aligned_replay_step_count"]):
        return "generation_aligned"
    if step_count == int(config["primary_replay_step_count"]):
        return "primary"
    return "fine_sensitivity"


def _step_records(
    base: Mapping[str, Any],
    observations: Iterable[Mapping[str, Any]],
    trajectory: Any,
    schedule: Iterable[Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    schedule_rows = tuple(schedule)
    for observation in observations:
        index = int(observation["flow_state_observation_step_index"])
        observed = trajectory.reverse_states[index + 1].detach().float()
        candidate = trajectory.forward_states[index + 1].detach().float()
        null = trajectory.null_forward_states[index + 1].detach().float()
        candidate_mse = float((candidate - observed).pow(2).mean().item())
        null_mse = float((null - observed).pow(2).mean().item())
        rows.append({
            **base,
            "trajectory_step_index": index,
            "trajectory_sigma_start": float(schedule_rows[index].sigma),
            "trajectory_sigma_end": float(schedule_rows[index + 1].sigma),
            "trajectory_step_candidate_mse": candidate_mse,
            "trajectory_step_null_mse": null_mse,
            "trajectory_step_candidate_mse_advantage": null_mse - candidate_mse,
            "trajectory_step_path_projection": observation.get("path_score"),
            "trajectory_step_path_projection_unweighted": observation.get(
                "path_score_unweighted"
            ),
            "trajectory_step_velocity_projection": observation.get("velocity_score"),
            "trajectory_step_endpoint_projection": (
                2.0 * float(observation.get("endpoint_score") or 0.5) - 1.0
            ),
            "trajectory_step_reliability": observation.get("replay_reliability"),
            "trajectory_step_local_likelihood_reliability": observation.get(
                "replay_step_likelihood_reliability"
            ),
            "trajectory_global_reliability": observation.get("replay_global_reliability"),
            "replay_log_likelihood_ratio": observation.get(
                "replay_log_likelihood_ratio"
            ),
            "claim_support_status": base["claim_support_status"],
        })
    return rows


def execute_condition(
    source_root: str | Path,
    output_root: str | Path,
    config: Mapping[str, Any],
    *,
    condition: str,
    pipeline_loader: Any = _load_pipeline,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """执行无攻击或既有攻击条件；每个视频一次 VAE encode、三个冻结 grid。"""

    source = Path(source_root).resolve()
    output = Path(output_root).resolve()
    generation_rows = [
        row for row in _read_jsonl(source / "records" / "generation_records.jsonl")
        if row.get("generation_status") == "success"
        and row.get("method_variant") in set(config["required_source_method_variants"])
    ]
    if condition == "no_attack":
        rows = [
            {**row, "video_condition_id": config["no_attack_video_condition_id"]}
            for row in generation_rows
        ]
        attacked = False
    elif condition == "attacked":
        rows = [
            {**row, "video_condition_id": row.get("attack_name")}
            for row in _read_jsonl(
                source / "records" / "trajectory_replay_smoke_attack_records.jsonl"
            )
            if row.get("attack_runtime_status") == "ready"
            and row.get("method_variant") in set(config["required_source_method_variants"])
            and row.get("attack_name") in set(config["required_attacked_video_condition_ids"])
        ]
        attacked = True
    else:
        raise ValueError("condition 必须是 no_attack 或 attacked")

    prompt_map = _prompt_text_by_id(
        _read_json(source / "datasets" / "prompt_seed_suite.json")
    )
    if not prompt_map:
        raise RuntimeError("Stage 0-D prompt suite 未提供 prompt_id/prompt_text")
    revisions: dict[str, str] = {}
    for row in rows:
        model_id = str(row.get("generation_model_id") or "")
        revision = validate_generation_model_provenance(row)
        previous = revisions.setdefault(model_id, revision)
        if previous != revision:
            raise RuntimeError("Stage 0-D 同一模型混用了不同不可变 revision")
    pipelines = {
        model_id: _invoke_pipeline_loader(
            pipeline_loader,
            model_id=model_id,
            revision=revision,
        )
        for model_id, revision in sorted(revisions.items())
    }
    likelihood = _likelihood_config(source)
    grid_counts = tuple(int(value) for value in config["replay_grid_step_counts"])
    checkpoint_prefix = "no_attack" if condition == "no_attack" else "attacked"
    checkpoint_summary_path = (
        output / "runtime" / f"trajectory_signal_{checkpoint_prefix}_summary_checkpoint.jsonl"
    )
    checkpoint_step_path = (
        output / "runtime" / f"trajectory_signal_{checkpoint_prefix}_step_checkpoint.jsonl"
    )
    checkpoint_failure_path = (
        output / "runtime" / f"trajectory_signal_{checkpoint_prefix}_failure_checkpoint.jsonl"
    )
    checkpoint_summary_path.parent.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = _read_jsonl(checkpoint_summary_path)
    steps: list[dict[str, Any]] = _read_jsonl(checkpoint_step_path)
    failures: list[dict[str, Any]] = _read_jsonl(checkpoint_failure_path)
    for source_row in rows:
        source_identity = (
            str(source_row.get("trajectory_trace_id") or ""),
            str(source_row.get("method_variant") or ""),
            str(source_row.get("video_condition_id") or ""),
        )
        existing_source_summaries = [
            row for row in summaries
            if (
                str(row.get("trajectory_trace_id") or ""),
                str(row.get("method_variant") or ""),
                str(row.get("video_condition_id") or ""),
            ) == source_identity
        ]
        existing_source_steps = [
            row for row in steps
            if (
                str(row.get("trajectory_trace_id") or ""),
                str(row.get("method_variant") or ""),
                str(row.get("video_condition_id") or ""),
            ) == source_identity
        ]
        expected_summary_count = len(grid_counts) * 2
        expected_step_count = sum(grid_counts) * 2
        if (
            len(existing_source_summaries) == expected_summary_count
            and len(existing_source_steps) == expected_step_count
        ):
            print(
                "stage0d source cached | "
                f"condition={source_identity[2]} | variant={source_identity[1]} | "
                f"trace={source_identity[0]}",
                flush=True,
            )
            continue
        summaries = [
            row for row in summaries
            if (
                str(row.get("trajectory_trace_id") or ""),
                str(row.get("method_variant") or ""),
                str(row.get("video_condition_id") or ""),
            ) != source_identity
        ]
        steps = [
            row for row in steps
            if (
                str(row.get("trajectory_trace_id") or ""),
                str(row.get("method_variant") or ""),
                str(row.get("video_condition_id") or ""),
            ) != source_identity
        ]
        failures = [
            row for row in failures
            if (
                str(row.get("trajectory_trace_id") or ""),
                str(row.get("method_variant") or ""),
                str(row.get("video_condition_id") or ""),
            ) != source_identity
        ]
        base_failure = {
            "record_version": DIAGNOSTIC_RECORD_VERSION,
            "profile_id": config["profile_id"],
            "trajectory_trace_id": source_row.get("trajectory_trace_id"),
            "method_variant": source_row.get("method_variant"),
            "video_condition_id": source_row.get("video_condition_id"),
            "claim_support_status": config["claim_support_status"],
        }
        try:
            model_id = str(source_row["generation_model_id"])
            prompt = prompt_map[str(source_row["prompt_id"])]
            pipeline = pipelines[model_id]
            key_context = _validated_flow_key_context(
                source_row,
                prompt=prompt,
                scheduler=pipeline.scheduler,
            )
            correct_key = _generation_key(source_row)
            wrong_key = _wrong_owner_generation_key(source_row)
            video_path = _resolved_source_video(source, source_row, attacked=attacked)
            replay = _run_attacked_video_replay_for_model(
                pipeline,
                video_path,
                prompt=prompt,
                key_text=correct_key,
                key_context=key_context,
                likelihood_config=likelihood,
                replay_step_counts=grid_counts,
            )
            if not getattr(replay, "replay_schedules", ()):
                raise RuntimeError("Stage 0-D replay backend 未导出所有冻结 schedule")
            for grid_index, step_count in enumerate(grid_counts):
                schedule = replay.replay_schedules[grid_index]
                correct_trajectory = replay.replay_trajectories[grid_index]
                phases, weights = _endpoint_integration_grid(
                    schedule,
                    FlowTubeletKeyCodeConfig(),
                )
                active = replace(
                    replay,
                    primary_schedule=schedule,
                    primary_replay_index=grid_index,
                    replay_uncertainty=estimate_replay_uncertainty((correct_trajectory,)),
                    endpoint_flow_phases=phases,
                    endpoint_integration_weights=weights,
                )
                correct_endpoint = _compute_replay_endpoint_evidence_for_key(
                    active,
                    key_text=correct_key,
                    key_context=key_context,
                )
                correct_path = score_replay_trajectory_for_key(
                    correct_trajectory,
                    schedule,
                    key_text=correct_key,
                    likelihood_config=likelihood,
                    key_context=key_context,
                )
                wrong_trajectory, wrong_path = _evaluate_fixed_replay_hypothesis_for_key(
                    pipeline,
                    active,
                    prompt=prompt,
                    key_text=wrong_key,
                    key_context=key_context,
                )
                wrong_endpoint = _compute_replay_endpoint_evidence_for_key(
                    active,
                    key_text=wrong_key,
                    key_context=key_context,
                )
                fixed_path_id = _stable_digest({
                    "trajectory_trace_id": source_row.get("trajectory_trace_id"),
                    "video_condition_id": source_row.get("video_condition_id"),
                    "replay_grid_step_count": step_count,
                    "video_sha256": _sha256_file(video_path),
                })
                for key_role, key_text, endpoint, trajectory, path in (
                    ("correct_owner_key", correct_key, correct_endpoint, correct_trajectory, correct_path),
                    ("wrong_owner_key", wrong_key, wrong_endpoint, wrong_trajectory, wrong_path),
                ):
                    summary_base = {
                        "record_version": DIAGNOSTIC_RECORD_VERSION,
                        "profile_id": config["profile_id"],
                        "generation_model_id": model_id,
                        "prompt_id": source_row.get("prompt_id"),
                        "seed_id": source_row.get("seed_id"),
                        "trajectory_trace_id": source_row.get("trajectory_trace_id"),
                        "method_variant": source_row.get("method_variant"),
                        "video_condition_id": source_row.get("video_condition_id"),
                        "candidate_key_role": key_role,
                        "replay_grid_step_count": step_count,
                        "replay_grid_role": _grid_role(config, step_count),
                        "generation_scheduler_step_count": int(
                            source_row.get("num_inference_steps") or 0
                        ),
                        "fixed_reverse_path_reference_id": fixed_path_id,
                        "fixed_reverse_path_reused_with_correct_key": (
                            True if key_role == "correct_owner_key" else (
                                trajectory.reverse_states is correct_trajectory.reverse_states
                            )
                        ),
                        "fixed_null_replay_reused_with_correct_key": (
                            True if key_role == "correct_owner_key" else (
                                trajectory.null_forward_states
                                is correct_trajectory.null_forward_states
                            )
                        ),
                        "claim_support_status": config["claim_support_status"],
                    }
                    summary = {
                        **summary_base,
                        "trajectory_signal_diagnostic_record_id": _stable_digest(summary_base),
                        "replay_candidate_endpoint_mse": trajectory.candidate_residual_mean_squared_error,
                        "replay_null_endpoint_mse": trajectory.null_residual_mean_squared_error,
                        "replay_candidate_mse_advantage": (
                            trajectory.null_residual_mean_squared_error
                            - trajectory.candidate_residual_mean_squared_error
                        ),
                        "trajectory_path_projection": path.get("S_path_inv"),
                        "trajectory_path_projection_unweighted": path.get(
                            "S_path_inv_unweighted"
                        ),
                        "trajectory_velocity_projection": path.get("S_velocity"),
                        "replay_log_likelihood_ratio": trajectory.replay_log_likelihood_ratio,
                        "trajectory_global_reliability": active.replay_uncertainty.replay_reliability,
                        **replay.endpoint_metadata,
                        # 必须保存当前 candidate 自己的方向摘要，wrong-key 行不能继承
                        # replay 初始 correct-key endpoint metadata。
                        **endpoint.as_dict(),
                        "metric_status": "measured_stage0d_diagnostic",
                    }
                    summaries.append(summary)
                    observations = build_flow_state_observation_sequence(
                        active,
                        key_text=key_text,
                        trajectory=trajectory,
                        schedule=schedule,
                        key_context=key_context,
                    )
                    steps.extend(
                        _step_records(
                            summary_base,
                            observations,
                            trajectory,
                            schedule,
                        )
                    )
            print(
                "stage0d source complete | "
                f"condition={source_row.get('video_condition_id')} | "
                f"variant={source_row.get('method_variant')} | "
                f"trace={source_row.get('trajectory_trace_id')}",
                flush=True,
            )
        except Exception as exc:  # pragma: no cover - 真实 GPU failure record
            failures.append({
                **base_failure,
                "trajectory_signal_diagnostic_status": "failed",
                "trajectory_signal_diagnostic_failure_reason": str(exc),
            })
        write_jsonl(checkpoint_summary_path, summaries)
        write_jsonl(checkpoint_step_path, steps)
        write_jsonl(checkpoint_failure_path, failures)
    return summaries, steps, failures


def build_pair_records(
    summaries: Iterable[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """从标量 summary 重建 correct/wrong 与 full/control 配对，不读取 tensor。"""

    rows = [dict(row) for row in summaries]
    by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = (
            *_source_identity(row),
            str(row.get("method_variant")),
            int(row.get("replay_grid_step_count") or 0),
            str(row.get("candidate_key_role")),
        )
        by_key[key] = row
    candidate_pairs: list[dict[str, Any]] = []
    margins: dict[tuple[Any, ...], dict[str, Any]] = {}
    variants = sorted(set(config["required_source_method_variants"]))
    identities = sorted({_source_identity(row) for row in rows})
    for identity in identities:
        for variant in variants:
            for grid in config["replay_grid_step_counts"]:
                correct = by_key.get((*identity, variant, int(grid), "correct_owner_key"))
                wrong = by_key.get((*identity, variant, int(grid), "wrong_owner_key"))
                if correct is None or wrong is None:
                    continue
                base = {
                    "record_version": DIAGNOSTIC_RECORD_VERSION,
                    "profile_id": config["profile_id"],
                    "generation_model_id": identity[0],
                    "prompt_id": identity[1],
                    "seed_id": identity[2],
                    "video_condition_id": identity[3],
                    "method_variant": variant,
                    "replay_grid_step_count": int(grid),
                    "trajectory_signal_comparison_kind": "correct_owner_key_over_wrong_owner_key",
                    "claim_support_status": config["claim_support_status"],
                }
                margin = {
                    **base,
                    "trajectory_signal_pair_record_id": _stable_digest(base),
                    "correct_over_wrong_path_margin": float(correct["trajectory_path_projection"]) - float(wrong["trajectory_path_projection"]),
                    "correct_over_wrong_likelihood_margin": float(correct["replay_log_likelihood_ratio"]) - float(wrong["replay_log_likelihood_ratio"]),
                    "correct_over_wrong_endpoint_margin": float(correct["endpoint_score"]) - float(wrong["endpoint_score"]),
                    "minimum_pair_reliability": min(
                        float(correct["trajectory_global_reliability"]),
                        float(wrong["trajectory_global_reliability"]),
                    ),
                }
                candidate_pairs.append(margin)
                margins[(*identity, variant, int(grid))] = margin
    for identity in identities:
        for grid in config["replay_grid_step_counts"]:
            full = margins.get((*identity, "sstw_full_method", int(grid)))
            if full is None:
                continue
            for control in (
                "endpoint_only_control",
                "sstw_clean_unwatermarked_reference",
            ):
                control_row = margins.get((*identity, control, int(grid)))
                if control_row is None:
                    continue
                base = {
                    "record_version": DIAGNOSTIC_RECORD_VERSION,
                    "profile_id": config["profile_id"],
                    "generation_model_id": identity[0],
                    "prompt_id": identity[1],
                    "seed_id": identity[2],
                    "video_condition_id": identity[3],
                    "method_variant": "sstw_full_method",
                    "control_method_variant": control,
                    "replay_grid_step_count": int(grid),
                    "trajectory_signal_comparison_kind": "full_over_control_path_margin_gain",
                    "claim_support_status": config["claim_support_status"],
                }
                candidate_pairs.append({
                    **base,
                    "trajectory_signal_pair_record_id": _stable_digest(base),
                    "full_over_control_path_margin_gain": (
                        float(full["correct_over_wrong_path_margin"])
                        - float(control_row["correct_over_wrong_path_margin"])
                    ),
                })
    return candidate_pairs


def _fraction(values: Iterable[bool]) -> float:
    rows = list(values)
    return sum(bool(value) for value in rows) / len(rows) if rows else 0.0


def build_diagnostic_decision(
    summaries: Iterable[Mapping[str, Any]],
    pairs: Iterable[Mapping[str, Any]],
    failures: Iterable[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    owner_key_preflight: Mapping[str, Any] | None = None,
    attacked_phase_requested: bool = False,
) -> dict[str, Any]:
    """只按预声明 gates 分类信号位置；永不授权 Stage 1。"""

    summary_rows = [dict(row) for row in summaries]
    pair_rows = [dict(row) for row in pairs]
    failure_rows = [dict(row) for row in failures]
    diagnostics: dict[str, Any] = {}
    condition_ids = [str(config["no_attack_video_condition_id"])]
    condition_ids.extend(str(value) for value in config["required_attacked_video_condition_ids"])
    for condition in condition_ids:
        condition_summary = [row for row in summary_rows if row.get("video_condition_id") == condition]
        if not condition_summary:
            continue
        for grid in config["replay_grid_step_counts"]:
            candidate = [
                row for row in pair_rows
                if row.get("video_condition_id") == condition
                and int(row.get("replay_grid_step_count") or 0) == int(grid)
                and row.get("trajectory_signal_comparison_kind") == "correct_owner_key_over_wrong_owner_key"
            ]
            full = [row for row in candidate if row.get("method_variant") == "sstw_full_method"]
            controls = [
                row for row in pair_rows
                if row.get("video_condition_id") == condition
                and int(row.get("replay_grid_step_count") or 0) == int(grid)
                and row.get("trajectory_signal_comparison_kind") == "full_over_control_path_margin_gain"
            ]
            endpoint_gains = [row for row in controls if row.get("control_method_variant") == "endpoint_only_control"]
            clean_gains = [row for row in controls if row.get("control_method_variant") == "sstw_clean_unwatermarked_reference"]
            path_fraction = _fraction(float(row["correct_over_wrong_path_margin"]) > 0.0 for row in full)
            llr_fraction = _fraction(float(row["correct_over_wrong_likelihood_margin"]) > 0.0 for row in full)
            endpoint_fraction = _fraction(float(row["full_over_control_path_margin_gain"]) > 0.0 for row in endpoint_gains)
            clean_fraction = _fraction(float(row["full_over_control_path_margin_gain"]) > 0.0 for row in clean_gains)
            reliability = min((float(row["minimum_pair_reliability"]) for row in candidate), default=0.0)
            coverage_ready = len(candidate) == int(config["required_source_video_count"]) * 3
            gate_ready = bool(
                coverage_ready
                and path_fraction >= float(config["minimum_full_correct_over_wrong_fraction"])
                and llr_fraction >= float(config["minimum_full_correct_over_wrong_fraction"])
                and endpoint_fraction >= float(config["minimum_full_path_margin_over_endpoint_fraction"])
                and clean_fraction >= float(config["minimum_full_path_margin_over_clean_fraction"])
                and reliability >= float(config["minimum_replay_reliability"])
            )
            diagnostics[f"{condition}:{int(grid)}"] = {
                "video_condition_id": condition,
                "replay_grid_step_count": int(grid),
                "coverage_ready": coverage_ready,
                "full_correct_over_wrong_path_fraction": path_fraction,
                "full_correct_over_wrong_likelihood_fraction": llr_fraction,
                "full_path_margin_over_endpoint_fraction": endpoint_fraction,
                "full_path_margin_over_clean_fraction": clean_fraction,
                "minimum_replay_reliability": reliability,
                "signal_separation_gate_ready": gate_ready,
            }
    no_attack = str(config["no_attack_video_condition_id"])
    grid_ready = {
        int(grid): bool(diagnostics.get(f"{no_attack}:{int(grid)}", {}).get("signal_separation_gate_ready"))
        for grid in config["replay_grid_step_counts"]
    }
    primary_ready = grid_ready[int(config["primary_replay_step_count"])]
    fine_ready = grid_ready[int(config["trajectory_signal_fine_replay_step_count"])]
    aligned_ready = grid_ready[int(config["generation_aligned_replay_step_count"])]
    no_attack_ready = primary_ready and fine_ready and not failure_rows
    attacked_executed = any(
        row.get("video_condition_id") in set(config["required_attacked_video_condition_ids"])
        for row in summary_rows
    )
    preflight_status = str(
        (owner_key_preflight or {}).get("owner_key_direction_preflight_status")
        or "not_run"
    )
    if preflight_status == "mismatch":
        classification = "owner_key_direction_mismatch_stop"
    elif preflight_status == "blocked":
        classification = "owner_key_direction_preflight_failure_stop"
    elif not summary_rows and not failure_rows:
        classification = "no_attack_replay_pending"
    elif failure_rows:
        classification = "runtime_or_input_failure_stop"
    elif primary_ready != fine_ready:
        classification = "replay_grid_sensitive_stop"
    elif not no_attack_ready and aligned_ready:
        classification = "generation_grid_alignment_sensitive_stop"
    elif not no_attack_ready:
        classification = "embedding_or_replay_signal_not_separated_stop"
    elif not attacked_executed:
        classification = "no_attack_signal_separated_attacked_diagnostic_allowed"
    else:
        attacked_ready = all(
            bool(diagnostics.get(f"{condition}:{grid}", {}).get("signal_separation_gate_ready"))
            for condition in config["required_attacked_video_condition_ids"]
            for grid in (
                int(config["primary_replay_step_count"]),
                int(config["trajectory_signal_fine_replay_step_count"]),
            )
        )
        classification = (
            "existing_standard_attack_carrier_diagnostic_supported"
            if attacked_ready
            else "standard_attack_erases_separable_signal_stop"
        )
    attacked_allowed = bool(
        preflight_status == "ready"
        and no_attack_ready
        and config.get("conditional_attacked_phase_allowed") is True
    )
    if (
        attacked_phase_requested
        and preflight_status == "ready"
        and not attacked_allowed
        and classification == "no_attack_replay_pending"
    ):
        classification = "attacked_phase_precondition_not_ready_stop"
    return {
        "record_version": DIAGNOSTIC_RECORD_VERSION,
        "profile_id": config["profile_id"],
        "trajectory_signal_diagnostic_decision": classification,
        "no_attack_signal_separation_ready": no_attack_ready,
        "attacked_phase_executed": attacked_executed,
        "attacked_phase_execution_allowed": attacked_allowed,
        "controlled_embedding_profile_construction_allowed": bool(
            summary_rows
            and preflight_status == "ready"
            and classification == "embedding_or_replay_signal_not_separated_stop"
        ),
        "stage_progression_allowed": False,
        "trajectory_signal_grid_diagnostics": diagnostics,
        "summary_record_count": len(summary_rows),
        "pair_record_count": len(pair_rows),
        "failure_record_count": len(failure_rows),
        "claim_support_status": config["claim_support_status"],
        "owner_key_direction_preflight_status": preflight_status,
        "owner_key_direction_all_match": bool(
            (owner_key_preflight or {}).get("owner_key_direction_all_match")
        ),
        "owner_key_context_all_match": bool(
            (owner_key_preflight or {}).get("owner_key_context_all_match")
        ),
        "owner_key_phase_grid_all_match": bool(
            (owner_key_preflight or {}).get("owner_key_phase_grid_all_match")
        ),
    }


def _write_report(path: Path, decision: Mapping[str, Any]) -> None:
    lines = [
        "# SSTW trajectory signal localization diagnostic",
        "",
        f"- Decision: `{decision['trajectory_signal_diagnostic_decision']}`",
        f"- Owner-key direction preflight: `{decision['owner_key_direction_preflight_status']}`",
        f"- Owner-key direction all match: `{decision['owner_key_direction_all_match']}`",
        f"- No-attack separation ready: `{decision['no_attack_signal_separation_ready']}`",
        f"- Attacked phase executed: `{decision['attacked_phase_executed']}`",
        f"- Stage progression allowed: `{decision['stage_progression_allowed']}`",
        f"- Summary records: `{decision['summary_record_count']}`",
        f"- Pair records: `{decision['pair_record_count']}`",
        f"- Failures: `{decision['failure_record_count']}`",
        "",
        "该报告仅为 Stage 0-D 诊断，不是 fixed-FPR 或论文证据。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_stage0d(
    source_root: str | Path,
    output_root: str | Path,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    phase: str,
    pipeline_loader: Any = _load_pipeline,
    scheduler_loader: Any = _load_owner_key_preflight_scheduler,
    direction_metadata_builder: Any = _rebuild_owner_key_direction_metadata,
) -> dict[str, Any]:
    """运行一个显式阶段并重建 decision/report/manifest。"""

    source = Path(source_root).resolve()
    output = Path(output_root).resolve()
    config = _read_json(config_path)
    validate_signal_localization_config(config)
    snapshot = build_immutable_input_snapshot(
        source,
        output,
        config,
        phase=phase,
    )
    snapshot_path = output / "artifacts" / "trajectory_signal_immutable_input_snapshot.json"
    if snapshot_path.exists() and _read_json(snapshot_path) != snapshot:
        raise RuntimeError("Stage 0-D immutable input snapshot 与既有运行不一致")
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    if not snapshot_path.exists():
        write_json(snapshot_path, snapshot)

    records_dir = output / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    summary_path = records_dir / "trajectory_signal_summary_records.jsonl"
    step_path = records_dir / "trajectory_signal_step_records.jsonl"
    failure_path = records_dir / "trajectory_signal_failure_records.jsonl"
    existing_summaries = _read_jsonl(summary_path)
    existing_steps = _read_jsonl(step_path)
    existing_failures = _read_jsonl(failure_path)
    # decision-only preflight 也必须物化空 governed records，保证 manifest 可重建。
    for path, rows in (
        (summary_path, existing_summaries),
        (step_path, existing_steps),
        (failure_path, existing_failures),
    ):
        if not path.exists():
            write_jsonl(path, rows)
    preflight_path = output / "artifacts" / "trajectory_signal_owner_key_preflight.json"
    owner_key_preflight = _read_json(preflight_path) if preflight_path.exists() else None
    if phase in {"credential_preflight", "no_attack", "attacked"}:
        runtime_checkpoint_rows_exist = any(
            _read_jsonl(path)
            for path in (output / "runtime").glob(
                "trajectory_signal_*_checkpoint.jsonl"
            )
        ) if (output / "runtime").is_dir() else False
        if owner_key_preflight is None and (
            existing_summaries
            or existing_steps
            or existing_failures
            or runtime_checkpoint_rows_exist
        ):
            raise RuntimeError(
                "Stage 0-D 既有 replay 记录缺少 owner-key preflight；必须使用新的 output root"
            )
        rebuilt_preflight = build_owner_key_direction_preflight(
            source,
            config,
            scheduler_loader=scheduler_loader,
            direction_metadata_builder=direction_metadata_builder,
        )
        if owner_key_preflight is not None and owner_key_preflight != rebuilt_preflight:
            raise RuntimeError(
                "Stage 0-D owner-key preflight 与既有运行不一致；必须使用新的 output root"
            )
        owner_key_preflight = rebuilt_preflight
        if not preflight_path.exists():
            write_json(preflight_path, owner_key_preflight)

    execution_allowed = bool(
        owner_key_preflight
        and owner_key_preflight.get("owner_key_direction_preflight_status") == "ready"
    )
    if phase == "attacked" and execution_allowed:
        existing_pairs = build_pair_records(existing_summaries, config)
        pre_execution_decision = build_diagnostic_decision(
            existing_summaries,
            existing_pairs,
            existing_failures,
            config,
            owner_key_preflight=owner_key_preflight,
            attacked_phase_requested=True,
        )
        execution_allowed = bool(
            pre_execution_decision["attacked_phase_execution_allowed"]
        )
    if phase in {"no_attack", "attacked"} and execution_allowed:
        new_summaries, new_steps, new_failures = execute_condition(
            source,
            output,
            config,
            condition=phase,
            pipeline_loader=pipeline_loader,
        )
        target_conditions = (
            {str(config["no_attack_video_condition_id"])}
            if phase == "no_attack"
            else set(config["required_attacked_video_condition_ids"])
        )
        existing_summaries = [row for row in existing_summaries if row.get("video_condition_id") not in target_conditions]
        existing_steps = [row for row in existing_steps if row.get("video_condition_id") not in target_conditions]
        existing_failures = [row for row in existing_failures if row.get("video_condition_id") not in target_conditions]
        existing_summaries.extend(new_summaries)
        existing_steps.extend(new_steps)
        existing_failures.extend(new_failures)
        write_jsonl(summary_path, existing_summaries)
        write_jsonl(step_path, existing_steps)
        write_jsonl(failure_path, existing_failures)
    pairs = build_pair_records(existing_summaries, config)
    pair_path = records_dir / "trajectory_signal_pair_records.jsonl"
    write_jsonl(pair_path, pairs)
    decision = build_diagnostic_decision(
        existing_summaries,
        pairs,
        existing_failures,
        config,
        owner_key_preflight=owner_key_preflight,
        attacked_phase_requested=phase == "attacked",
    )
    decision_path = output / "artifacts" / "trajectory_signal_diagnostic_decision.json"
    report_path = output / "reports" / "trajectory_signal_diagnostic_report.md"
    write_json(decision_path, decision)
    _write_report(report_path, decision)
    governed = [snapshot_path, summary_path, step_path, pair_path, failure_path, decision_path, report_path]
    if preflight_path.exists():
        governed.insert(1, preflight_path)
    manifest = {
        "artifact_id": "trajectory_signal_localization_diagnostic_manifest",
        "profile_id": config["profile_id"],
        "source_run_root": str(source),
        "output_run_root": str(output),
        "protocol_config_path": str(Path(config_path).resolve()),
        "protocol_config_sha256": _sha256_file(config_path),
        "immutable_input_snapshot_digest": snapshot["immutable_input_snapshot_digest"],
        "record_paths": [str(path) for path in governed],
        "output_sha256": {str(path): _sha256_file(path) for path in governed},
        "claim_support_status": config["claim_support_status"],
    }
    write_json(
        output / "artifacts" / "trajectory_signal_diagnostic_manifest.json",
        manifest,
    )
    return decision


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-root", required=True)
    parser.add_argument("--output-run-root", required=True)
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--phase",
        choices=("credential_preflight", "no_attack", "attacked", "decision"),
        required=True,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = run_stage0d(
        args.source_run_root,
        args.output_run_root,
        args.config_path,
        phase=args.phase,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
