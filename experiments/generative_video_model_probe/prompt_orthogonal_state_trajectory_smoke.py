"""Minimal no-attack smoke for prompt-orthogonal state trajectories."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping

from evaluation.protocol.record_writer import write_json, write_jsonl
from experiments.generative_video_model_probe.colab_runtime import (
    PROMPT_ORTHOGONAL_STATE_TRAJECTORY_SMOKE_PROFILE,
    WAN21_PRIMARY_MODEL_ID,
    run_colab_probe,
    validate_generation_model_provenance,
)
from experiments.generative_video_model_probe.formal_flow_evidence_runner import (
    _invoke_pipeline_loader,
    _load_pipeline,
    _prompt_text_by_id,
)
from experiments.generative_video_model_probe.minimal_signed_trajectory_state_space_smoke import (
    LIKELIHOOD_SUFFIX,
    PROMPT_SUITE_SUFFIX,
    _likelihood_config,
    _read_json,
    _read_jsonl,
    _stable_digest,
    validate_controlled_embedding_source_result,
)
from main.methods.state_space_watermark.prompt_orthogonal_replay import (
    PROMPT_ORTHOGONAL_REPLAY_RELIABILITY_MODE,
    build_wan_prompt_orthogonal_fixed_replay_trace,
    evaluate_wan_prompt_orthogonal_candidates_on_fixed_trace,
)
from main.methods.state_space_watermark.replay_inversion import (
    ReplayGaussianLikelihoodConfig,
)
from main.methods.state_space_watermark.state_trajectory_injection import (
    PROMPT_ORTHOGONAL_SCHEDULER_CONTROL_DTYPE,
    PromptOrthogonalInjectionConfig,
)
from main.methods.state_space_watermark.state_rotation_operator import (
    PROMPT_ORTHOGONAL_CANONICAL_PLANE_CONSTRUCTION_DEVICE,
)
from main.methods.state_space_watermark.velocity_field_constraint import (
    VelocityFieldConstraintConfig,
)
from main.methods.state_space_watermark.watermark_key_derivation import (
    derive_prompt_orthogonal_master_key_text,
    derive_prompt_orthogonal_wrong_candidate_master_key_text,
)


DEFAULT_CONFIG_PATH = (
    "configs/protocol/sstw_prompt_orthogonal_state_trajectory_smoke.json"
)
PROFILE_ID = "sstw_prompt_orthogonal_state_trajectory_smoke"
RECORD_VERSION = "prompt_orthogonal_state_trajectory_smoke"
TEST_ID = "prompt_orthogonal_state_trajectory_smoke"
WATERMARKED_VARIANT = "prompt_orthogonal_state_trajectory"
CLEAN_VARIANT = "clean_unwatermarked_control"
OWNER_ROLE = "correct_owner_key"
WRONG_ROLE = "wrong_owner_key"
TEMPORAL_DECISION_SUFFIX = (
    "artifacts/temporal_code_isolation_smoke_decision.json"
)
TEMPORAL_MANIFEST_SUFFIX = (
    "artifacts/temporal_code_isolation_smoke_manifest.json"
)
TEMPORAL_SUMMARY_SUFFIX = (
    "records/temporal_code_isolation_summary_records.jsonl"
)
TEMPORAL_PAIR_SUFFIX = (
    "records/temporal_code_isolation_pair_records.jsonl"
)
TEMPORAL_IDENTITY_SUFFIX = (
    "records/temporal_code_isolation_identity_records.jsonl"
)
TEMPORAL_FAILURE_SUFFIX = (
    "records/temporal_code_isolation_failure_records.jsonl"
)


def _find_unique(root: Path, suffix: str) -> Path:
    candidates = sorted(
        path.resolve()
        for path in root.rglob(Path(suffix).name)
        if path.as_posix().endswith(suffix)
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"输入必须唯一包含 {suffix}: observed={candidates}"
        )
    return candidates[0]


def validate_prompt_orthogonal_smoke_config(
    config: Mapping[str, Any],
) -> None:
    """Freeze the mechanism smoke before reading its generation results."""

    false_fields = (
        "attacked_phase_execution_allowed",
        "cross_project_integration_allowed",
        "external_baseline_execution_allowed",
        "fixed_fpr_evaluation_allowed",
        "large_scale_generation_allowed",
        "stage_progression_allowed",
        "test_split_claims_allowed",
    )
    allowed_fields = set(false_fields) | {
        "claim_support_status",
        "clean_identity_record_count",
        "generation_step_count",
        "heldout_prompt_ids",
        "heldout_seed_ids",
        "identity_record_count",
        "lambda_max",
        "maximum_clean_owner_top1_fraction",
        "maximum_prompt_pair_fraction_gap",
        "minimum_owner_over_wrong_pair_fraction",
        "minimum_owner_top1_identity_fraction",
        "minimum_replay_reliability",
        "minimum_watermarked_over_clean_owner_fraction",
        "no_attack_only",
        "pair_record_count",
        "paper_result_level",
        "profile_id",
        "prompt_orthogonal_replay_reliability_mode",
        "replay_step_count",
        "required_generation_model_id",
        "required_source_controlled_embedding_decision",
        "required_source_failure_record_count",
        "required_source_generation_record_count",
        "required_source_pair_record_count",
        "required_source_profile_id",
        "required_source_summary_record_count",
        "required_temporal_source_decision",
        "required_temporal_source_failure_record_count",
        "required_temporal_source_identity_record_count",
        "required_temporal_source_pair_record_count",
        "required_temporal_source_profile_id",
        "required_temporal_source_summary_record_count",
        "smoke_generation_record_count",
        "summary_record_count",
        "trajectory_carrier_variant_ids",
        "wrong_owner_candidate_count",
    }
    unknown = sorted(set(config) - allowed_fields)
    if unknown:
        raise ValueError(
            "prompt-orthogonal 配置包含未声明字段: "
            + ", ".join(unknown)
        )
    invalid = [name for name in false_fields if config.get(name) is not False]
    if invalid:
        raise ValueError(
            "prompt-orthogonal 禁止项未冻结: " + ", ".join(invalid)
        )
    exact = {
        "profile_id": PROFILE_ID,
        "paper_result_level": PROFILE_ID.removeprefix("sstw_"),
        "claim_support_status": (
            "prompt_orthogonal_state_trajectory_smoke_only_not_paper_evidence"
        ),
        "required_source_profile_id": (
            "sstw_controlled_embedding_strength_diagnostic"
        ),
        "required_source_controlled_embedding_decision": (
            "lambda_increase_did_not_repair_path_signal_stop"
        ),
        "required_source_generation_record_count": 16,
        "required_source_summary_record_count": 96,
        "required_source_pair_record_count": 84,
        "required_source_failure_record_count": 0,
        "required_temporal_source_profile_id": (
            "sstw_temporal_code_isolation_replay_smoke"
        ),
        "required_temporal_source_decision": (
            "temporal_code_isolation_failed_stop_method"
        ),
        "required_temporal_source_summary_record_count": 36,
        "required_temporal_source_pair_record_count": 32,
        "required_temporal_source_identity_record_count": 4,
        "required_temporal_source_failure_record_count": 0,
        "required_generation_model_id": WAN21_PRIMARY_MODEL_ID,
        "no_attack_only": True,
        "generation_step_count": 8,
        "replay_step_count": 20,
        "smoke_generation_record_count": 8,
        "summary_record_count": 72,
        "pair_record_count": 32,
        "identity_record_count": 4,
        "clean_identity_record_count": 4,
        "wrong_owner_candidate_count": 8,
        "prompt_orthogonal_replay_reliability_mode": (
            PROMPT_ORTHOGONAL_REPLAY_RELIABILITY_MODE
        ),
        "heldout_prompt_ids": [
            "probe_paper_paper_master_prompt_003",
            "probe_paper_paper_master_prompt_004",
        ],
        "heldout_seed_ids": [
            "probe_paper_paper_master_test_seed_01",
            "probe_paper_paper_master_test_seed_02",
        ],
        "trajectory_carrier_variant_ids": [
            WATERMARKED_VARIANT,
            CLEAN_VARIANT,
        ],
    }
    mismatches = [
        name for name, expected in exact.items()
        if config.get(name) != expected
    ]
    if mismatches:
        raise ValueError(
            "prompt-orthogonal 配置字段未冻结: "
            + ", ".join(mismatches)
        )
    numeric = {
        "lambda_max": 0.12,
        "minimum_owner_over_wrong_pair_fraction": 0.75,
        "minimum_owner_top1_identity_fraction": 0.75,
        "maximum_prompt_pair_fraction_gap": 0.25,
        "minimum_replay_reliability": 0.05,
        "minimum_watermarked_over_clean_owner_fraction": 0.75,
        "maximum_clean_owner_top1_fraction": 0.5,
    }
    changed = [
        name for name, expected in numeric.items()
        if not math.isclose(
            float(config.get(name, math.nan)),
            expected,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ]
    if changed:
        raise ValueError(
            "prompt-orthogonal 数值门槛未冻结: " + ", ".join(changed)
        )


def validate_temporal_failure_source(
    input_root: str | Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Require the completed temporal-code failure that motivates this route."""

    root = Path(input_root).resolve()
    decision_path = _find_unique(root, TEMPORAL_DECISION_SUFFIX)
    manifest_path = _find_unique(root, TEMPORAL_MANIFEST_SUFFIX)
    summary_path = _find_unique(root, TEMPORAL_SUMMARY_SUFFIX)
    pair_path = _find_unique(root, TEMPORAL_PAIR_SUFFIX)
    identity_path = _find_unique(root, TEMPORAL_IDENTITY_SUFFIX)
    failure_path = _find_unique(root, TEMPORAL_FAILURE_SUFFIX)
    decision = _read_json(decision_path)
    manifest = _read_json(manifest_path)
    summaries = _read_jsonl(summary_path)
    pairs = _read_jsonl(pair_path)
    identities = _read_jsonl(identity_path)
    failures = _read_jsonl(failure_path)
    required_decision = {
        "profile_id": config["required_temporal_source_profile_id"],
        "temporal_code_isolation_smoke_decision": config[
            "required_temporal_source_decision"
        ],
        "temporal_code_isolation_gate_ready": False,
        "generation_execution_allowed": False,
        "stage_progression_allowed": False,
        "formal_result": False,
        "summary_record_count": config[
            "required_temporal_source_summary_record_count"
        ],
        "pair_record_count": config[
            "required_temporal_source_pair_record_count"
        ],
        "identity_record_count": config[
            "required_temporal_source_identity_record_count"
        ],
    }
    if any(
        decision.get(name) != expected
        for name, expected in required_decision.items()
    ):
        raise ValueError("temporal failure decision 与新路线前提不一致")
    required_manifest = {
        "profile_id": config["required_temporal_source_profile_id"],
        "summary_record_count": config[
            "required_temporal_source_summary_record_count"
        ],
        "pair_record_count": config[
            "required_temporal_source_pair_record_count"
        ],
        "identity_record_count": config[
            "required_temporal_source_identity_record_count"
        ],
        "failure_record_count": config[
            "required_temporal_source_failure_record_count"
        ],
        "generation_execution_allowed": False,
        "stage_progression_allowed": False,
        "formal_result": False,
    }
    if any(
        manifest.get(name) != expected
        for name, expected in required_manifest.items()
    ):
        raise ValueError("temporal failure manifest 不完整或越界")
    observed_counts = {
        "summary_record_count": len(summaries),
        "pair_record_count": len(pairs),
        "identity_record_count": len(identities),
        "failure_record_count": len(failures),
    }
    if any(
        observed_counts[name] != required_manifest[name]
        for name in observed_counts
    ):
        raise ValueError("temporal failure records 与 manifest count 不一致")
    if any(
        row.get("profile_id")
        != config["required_temporal_source_profile_id"]
        for row in summaries + pairs + identities
    ):
        raise ValueError("temporal failure records profile 不一致")
    return {
        "decision_path": decision_path,
        "manifest_path": manifest_path,
        "summary_path": summary_path,
        "pair_path": pair_path,
        "identity_path": identity_path,
        "failure_path": failure_path,
        "decision": decision,
        "manifest": manifest,
    }


def build_prompt_orthogonal_generation_plan(
    validated: Mapping[str, Any],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build exactly two prompts x two seeds x watermarked/clean."""

    prompts = {
        str(row["prompt_id"]): dict(row)
        for row in validated["prompt_suite"].get("prompts") or []
    }
    seeds = {
        str(row["seed_id"]): dict(row)
        for row in validated["prompt_suite"].get("seeds") or []
    }
    if any(value not in prompts for value in config["heldout_prompt_ids"]):
        raise ValueError("prompt suite 缺少冻结 held-out prompt")
    if any(value not in seeds for value in config["heldout_seed_ids"]):
        raise ValueError("prompt suite 缺少冻结 held-out seed")
    source_prompt_ids = {
        str(row["prompt_id"]) for row in validated["generation_rows"]
    }
    source_seed_ids = {
        str(row["seed_id"]) for row in validated["generation_rows"]
    }
    if set(config["heldout_prompt_ids"]) & source_prompt_ids or set(
        config["heldout_seed_ids"]
    ) & source_seed_ids:
        raise ValueError("prompt-orthogonal held-out identity 与 source 重叠")
    variants = (
        (
            WATERMARKED_VARIANT,
            "sstw_prompt_orthogonal_state_trajectory",
            "controlled_embedding_positive_source",
            "flow_scheduler_velocity_constraint",
        ),
        (
            CLEAN_VARIANT,
            "sstw_clean_unwatermarked_reference",
            "clean_negative",
            "clean_unwatermarked_reference",
        ),
    )
    plan: list[dict[str, Any]] = []
    for prompt_id in config["heldout_prompt_ids"]:
        for seed_id in config["heldout_seed_ids"]:
            prompt = prompts[prompt_id]
            seed = seeds[seed_id]
            seed_value = int(
                seed.get("seed_value")
                if seed.get("seed_value") is not None
                else seed.get("generation_seed_random")
            )
            for variant, method, sample_role, embedding_status in variants:
                identity = {
                    "generation_model_id": config[
                        "required_generation_model_id"
                    ],
                    "prompt_id": prompt_id,
                    "seed_id": seed_id,
                    "trajectory_carrier_variant_id": variant,
                }
                plan.append(
                    {
                        **prompt,
                        **seed,
                        **identity,
                        "prompt_orthogonal_plan_record_id": _stable_digest(
                            identity
                        ),
                        "seed_value": seed_value,
                        "generation_seed_random": seed_value,
                        "cross_model_role": "main_generation_model",
                        "sample_role": sample_role,
                        "generation_sample_role": sample_role,
                        "method_variant": method,
                        "watermark_embedding_status": embedding_status,
                        "lambda_max": 0.12,
                        "prompt_suite_role": prompt["prompt_suite_role"],
                        "seed_suite_role": (
                            seed.get("seed_suite_role")
                            or seed["prompt_suite_role"]
                        ),
                        "formal_method_variant_execution": False,
                        "generation_execution_allowed": True,
                        "attacked_phase_execution_allowed": False,
                        "stage_progression_allowed": False,
                        "claim_support_status": config[
                            "claim_support_status"
                        ],
                    }
                )
    if len(plan) != 8:
        raise RuntimeError("prompt-orthogonal generation plan 必须为8条")
    return plan


def _run_generation(
    validated: Mapping[str, Any],
    output_root: Path,
    plan: list[dict[str, Any]],
    *,
    pipeline_cache: dict[str, Any],
) -> dict[str, Any]:
    velocity = VelocityFieldConstraintConfig()
    velocity_configs = {
        str(row["prompt_orthogonal_plan_record_id"]): velocity
        for row in plan
    }
    injection = PromptOrthogonalInjectionConfig()
    injections = {
        str(row["prompt_orthogonal_plan_record_id"]): (
            injection
            if row["trajectory_carrier_variant_id"] == WATERMARKED_VARIANT
            else None
        )
        for row in plan
    }
    return run_colab_probe(
        output_root,
        validated["prompt_suite_path"],
        PROMPT_ORTHOGONAL_STATE_TRAJECTORY_SMOKE_PROFILE,
        model_id=WAN21_PRIMARY_MODEL_ID,
        cross_model_id=None,
        generation_plan_override=plan,
        velocity_config_by_plan_record_id=velocity_configs,
        prompt_orthogonal_injection_by_plan_record_id=injections,
        pipeline_cache=pipeline_cache,
    )


def validate_prompt_orthogonal_generation_execution(
    output_root: str | Path,
    generation_result: Mapping[str, Any],
) -> None:
    root = Path(output_root)
    decision = generation_result.get("decision")
    if not isinstance(decision, Mapping) or (
        decision.get("implementation_decision") != "PASS"
        or decision.get("mechanism_decision")
        != "GENERATION_READY_NO_ATTACK_REPLAY_PENDING"
    ):
        raise RuntimeError(
            "prompt-orthogonal generation runtime decision 未就绪"
        )
    rows = _read_jsonl(root / "records" / "generation_records.jsonl")
    traces = _read_jsonl(root / "records" / "trajectory_trace.jsonl")
    if len(rows) != 8 or any(
        row.get("generation_status") != "success" for row in rows
    ):
        raise RuntimeError("prompt-orthogonal generation 未完整成功")
    watermarked_ids = {
        str(row["trajectory_trace_id"])
        for row in rows
        if row["trajectory_carrier_variant_id"] == WATERMARKED_VARIANT
    }
    generation_ids = {
        str(row["trajectory_trace_id"]) for row in rows
    }
    generation_traces = [
        row for row in traces
        if str(row.get("trajectory_trace_id")) in generation_ids
    ]
    if not generation_traces or any(
        row.get("prompt_orthogonal_scheduler_control_dtype")
        != PROMPT_ORTHOGONAL_SCHEDULER_CONTROL_DTYPE
        for row in generation_traces
    ):
        raise RuntimeError(
            "prompt-orthogonal watermarked/clean scheduler dtype 不同源"
        )
    watermarked_traces = [
        row for row in traces
        if str(row.get("trajectory_trace_id")) in watermarked_ids
    ]
    if not watermarked_traces or any(
        row.get("flow_runtime_step_formal_context_complete") is not True
        for row in watermarked_traces
    ):
        raise RuntimeError("prompt-orthogonal generation trace context 不完整")
    active = [
        row for row in watermarked_traces
        if row.get("prompt_orthogonal_inactive_phase_noop") is False
    ]
    if not active or any(
        row.get("prompt_orthogonal_norm_guard_passed") is not True
        or row.get("prompt_orthogonal_energy_guard_passed") is not True
        or row.get("prompt_orthogonal_direction_guard_passed") is not True
        or row.get("prompt_orthogonal_finite_precision_projection_status")
        not in {
            "direct_actual_delta_pass",
            "bounded_actual_delta_backoff_pass",
        }
        or not (
            0.0
            < float(
                row.get(
                    "prompt_orthogonal_finite_precision_projection_scale",
                    0.0,
                )
            )
            <= 1.0
        )
        or int(
            row.get(
                "prompt_orthogonal_finite_precision_projection_attempt_count",
                0,
            )
        )
        < 1
        or float(
            row.get("prompt_orthogonal_candidate_delta_norm", 0.0)
        )
        <= 0.0
        or float(
            row.get(
                "prompt_orthogonal_intended_delta_norm_before_projection",
                0.0,
            )
        )
        <= 0.0
        or float(row.get("velocity_constraint_delta_norm", 0.0)) <= 0.0
        or int(
            row.get(
                "prompt_orthogonal_finite_precision_backoff_count",
                -1,
            )
        )
        < 0
        or row.get("endpoint_control_enabled") is not False
        for row in active
    ):
        raise RuntimeError("prompt-orthogonal generation guard 未完整通过")


def _authentication_context() -> tuple[bytes, str]:
    secret = (
        os.environ.get("SSTW_TRAJECTORY_AUTHENTICATION_KEY") or ""
    ).encode("utf-8")
    key_id = (
        os.environ.get("SSTW_TRAJECTORY_AUTHENTICATION_KEY_ID") or ""
    ).strip()
    if len(secret) < 32 or not key_id:
        raise RuntimeError("prompt-orthogonal replay 缺少 owner credentials")
    return secret, key_id


def _execute_replay(
    output: Path,
    config: Mapping[str, Any],
    *,
    likelihood: ReplayGaussianLikelihoodConfig,
    pipeline_loader: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    generation_rows = [
        row for row in _read_jsonl(
            output / "records" / "generation_records.jsonl"
        )
        if row.get("generation_status") == "success"
    ]
    prompt_suite = _read_json(output / PROMPT_SUITE_SUFFIX)
    prompt_map = _prompt_text_by_id(prompt_suite)
    negative_prompt_map = {
        str(row["prompt_id"]): (
            None
            if row.get("prompt_negative_text") is None
            else str(row["prompt_negative_text"])
        )
        for row in prompt_suite.get("prompts") or []
    }
    revisions: dict[str, str] = {}
    for row in generation_rows:
        model_id = str(row["generation_model_id"])
        revision = validate_generation_model_provenance(row)
        if revisions.setdefault(model_id, revision) != revision:
            raise RuntimeError("prompt-orthogonal 同一模型混用 revision")
    pipelines = {
        model_id: _invoke_pipeline_loader(
            pipeline_loader,
            model_id=model_id,
            revision=revision,
        )
        for model_id, revision in revisions.items()
    }
    secret, key_id = _authentication_context()
    owner_master = derive_prompt_orthogonal_master_key_text(
        secret,
        key_id=key_id,
    )
    candidate_master_keys = {
        OWNER_ROLE: owner_master,
        **{
            f"{WRONG_ROLE}_{index:02d}": (
                derive_prompt_orthogonal_wrong_candidate_master_key_text(
                    secret,
                    key_id=key_id,
                    candidate_index=index,
                )
            )
            for index in range(int(config["wrong_owner_candidate_count"]))
        },
    }
    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for source in generation_rows:
        base = {
            "record_version": RECORD_VERSION,
            "profile_id": PROFILE_ID,
            "generation_model_id": source["generation_model_id"],
            "prompt_id": source["prompt_id"],
            "seed_id": source["seed_id"],
            "trajectory_trace_id": source["trajectory_trace_id"],
            "prompt_orthogonal_plan_record_id": source[
                "prompt_orthogonal_plan_record_id"
            ],
            "trajectory_carrier_variant_id": source[
                "trajectory_carrier_variant_id"
            ],
            "method_variant": source["method_variant"],
            "video_condition_id": (
                "no_attack_prompt_orthogonal_state_trajectory"
            ),
            "replay_grid_step_count": 20,
            "claim_support_status": config["claim_support_status"],
        }
        try:
            prompt = prompt_map[str(source["prompt_id"])]
            negative_prompt = negative_prompt_map.get(
                str(source["prompt_id"])
            )
            negative_prompt_digest = sha256(
                str(negative_prompt or "").encode("utf-8")
            ).hexdigest()
            if source.get(
                "prompt_orthogonal_negative_prompt_text_hash"
            ) != negative_prompt_digest:
                raise RuntimeError(
                    "prompt-orthogonal generation/replay negative prompt 不同源"
                )
            pipeline = pipelines[str(source["generation_model_id"])]
            video_path = Path(str(source["video_path"]))
            if not video_path.is_file():
                raise FileNotFoundError(f"生成视频不存在: {video_path}")
            fixed_trace = build_wan_prompt_orthogonal_fixed_replay_trace(
                pipeline,
                str(video_path),
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=20,
            )
            if not fixed_trace.key_independent_trace_complete:
                raise RuntimeError(
                    "prompt-orthogonal key-independent trace 构造不完整"
                )
            evaluation = (
                evaluate_wan_prompt_orthogonal_candidates_on_fixed_trace(
                    pipeline,
                    fixed_trace.reverse_states,
                    fixed_trace.schedule,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    candidate_master_keys=candidate_master_keys,
                    likelihood_config=likelihood,
                )
            )
            if (
                not evaluation.key_independent_trace_complete
                or evaluation.base_model_velocity_call_count != 20
            ):
                raise RuntimeError(
                    "prompt-orthogonal shared replay trace 不完整"
                )
            minimum_reliability = min(
                row.reliability_weight
                for row in evaluation.step_summaries
            )
            for candidate in evaluation.candidate_scores:
                role = candidate.candidate_role
                candidate_index = (
                    None
                    if role == OWNER_ROLE
                    else int(role.rsplit("_", 1)[1])
                )
                record_base = {
                    **base,
                    "prompt_orthogonal_candidate_role": (
                        OWNER_ROLE if role == OWNER_ROLE else WRONG_ROLE
                    ),
                    "prompt_orthogonal_wrong_candidate_index": (
                        candidate_index
                    ),
                }
                summaries.append(
                    {
                        **record_base,
                        "prompt_orthogonal_summary_record_id": (
                            _stable_digest(record_base)
                        ),
                        "prompt_orthogonal_continuous_function_digest": (
                            candidate.continuous_schedule
                            .continuous_function_digest
                        ),
                        "prompt_orthogonal_schedule_projection_digest": (
                            candidate.continuous_schedule
                            .schedule_projection_digest
                        ),
                        "prompt_orthogonal_operator_plane_digest": (
                            candidate.operator_plane_digest
                        ),
                        "prompt_orthogonal_negative_prompt_text_hash": (
                            negative_prompt_digest
                        ),
                        "prompt_orthogonal_plane_construction_device": (
                            PROMPT_ORTHOGONAL_CANONICAL_PLANE_CONSTRUCTION_DEVICE
                        ),
                        "prompt_orthogonal_demodulation_vector": list(
                            candidate.demodulation.demodulation_vector
                        ),
                        "prompt_orthogonal_matched_amplitude": (
                            candidate.demodulation.matched_amplitude
                        ),
                        "prompt_orthogonal_orthogonal_amplitude": (
                            candidate.demodulation.orthogonal_amplitude
                        ),
                        "prompt_orthogonal_matched_cosine_score": (
                            candidate.demodulation.matched_cosine_score
                        ),
                        "prompt_orthogonal_vector_context_complete": (
                            candidate.candidate_context_complete
                        ),
                        "prompt_orthogonal_minimum_projection_retained_ratio": (
                            candidate.minimum_projection_retained_ratio
                        ),
                        "minimum_replay_reliability": minimum_reliability,
                        "prompt_orthogonal_replay_reliability_mode": (
                            config[
                                "prompt_orthogonal_replay_reliability_mode"
                            ]
                        ),
                        "prompt_orthogonal_base_velocity_call_count": (
                            evaluation.base_model_velocity_call_count
                        ),
                        "prompt_orthogonal_trace_construction_base_velocity_call_count": (
                            fixed_trace.base_model_velocity_call_count
                        ),
                        "prompt_orthogonal_total_base_velocity_call_count": (
                            fixed_trace.base_model_velocity_call_count
                            + evaluation.base_model_velocity_call_count
                        ),
                        "prompt_orthogonal_candidate_count": (
                            evaluation.candidate_count
                        ),
                        "prompt_orthogonal_fixed_trace_key_independent": (
                            fixed_trace.key_independent_trace_complete
                        ),
                        "prompt_orthogonal_key_independent_trace_complete": (
                            evaluation.key_independent_trace_complete
                        ),
                    }
                )
        except Exception as error:  # pragma: no cover - Colab GPU path
            failure = {
                **base,
                "prompt_orthogonal_smoke_status": "failed",
                "prompt_orthogonal_smoke_failure_reason": str(error),
            }
            failure["prompt_orthogonal_failure_record_id"] = _stable_digest(
                failure
            )
            failures.append(failure)
    return summaries, failures


def build_prompt_orthogonal_pair_records(
    summaries: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in summaries]
    pairs: list[dict[str, Any]] = []
    identities = sorted(
        {
            (str(row["prompt_id"]), str(row["seed_id"]))
            for row in rows
            if row["trajectory_carrier_variant_id"] == WATERMARKED_VARIANT
        }
    )
    for prompt_id, seed_id in identities:
        selected = [
            row for row in rows
            if row["prompt_id"] == prompt_id
            and row["seed_id"] == seed_id
            and row["trajectory_carrier_variant_id"]
            == WATERMARKED_VARIANT
        ]
        owners = [
            row for row in selected
            if row["prompt_orthogonal_candidate_role"] == OWNER_ROLE
        ]
        wrongs = [
            row for row in selected
            if row["prompt_orthogonal_candidate_role"] == WRONG_ROLE
        ]
        if len(owners) != 1 or len(wrongs) != 8:
            continue
        owner_score = float(
            owners[0]["prompt_orthogonal_matched_cosine_score"]
        )
        for wrong in wrongs:
            margin = owner_score - float(
                wrong["prompt_orthogonal_matched_cosine_score"]
            )
            base = {
                "record_version": RECORD_VERSION,
                "profile_id": PROFILE_ID,
                "prompt_id": prompt_id,
                "seed_id": seed_id,
                "prompt_orthogonal_wrong_candidate_index": wrong[
                    "prompt_orthogonal_wrong_candidate_index"
                ],
                "prompt_orthogonal_owner_score": owner_score,
                "prompt_orthogonal_wrong_score": float(
                    wrong["prompt_orthogonal_matched_cosine_score"]
                ),
                "prompt_orthogonal_owner_over_wrong_margin": margin,
                "prompt_orthogonal_owner_over_wrong": margin > 0.0,
            }
            pairs.append(
                {
                    **base,
                    "prompt_orthogonal_pair_record_id": _stable_digest(base),
                }
            )
    return pairs


def build_prompt_orthogonal_identity_records(
    summaries: Iterable[Mapping[str, Any]],
    *,
    variant: str,
) -> list[dict[str, Any]]:
    rows = [
        dict(row) for row in summaries
        if row["trajectory_carrier_variant_id"] == variant
    ]
    records: list[dict[str, Any]] = []
    identities = sorted(
        {(str(row["prompt_id"]), str(row["seed_id"])) for row in rows}
    )
    for prompt_id, seed_id in identities:
        selected = [
            row for row in rows
            if row["prompt_id"] == prompt_id
            and row["seed_id"] == seed_id
        ]
        owners = [
            row for row in selected
            if row["prompt_orthogonal_candidate_role"] == OWNER_ROLE
        ]
        wrongs = [
            row for row in selected
            if row["prompt_orthogonal_candidate_role"] == WRONG_ROLE
        ]
        if len(owners) != 1 or len(wrongs) != 8:
            continue
        owner_score = float(
            owners[0]["prompt_orthogonal_matched_cosine_score"]
        )
        wrong_scores = [
            float(row["prompt_orthogonal_matched_cosine_score"])
            for row in wrongs
        ]
        top1 = owner_score > max(wrong_scores)
        base = {
            "record_version": RECORD_VERSION,
            "profile_id": PROFILE_ID,
            "prompt_id": prompt_id,
            "seed_id": seed_id,
            "trajectory_carrier_variant_id": variant,
            "prompt_orthogonal_owner_score": owner_score,
            "prompt_orthogonal_maximum_wrong_score": max(wrong_scores),
            "prompt_orthogonal_owner_top1": top1,
            "prompt_orthogonal_owner_rank": (
                1 + sum(score >= owner_score for score in wrong_scores)
            ),
        }
        records.append(
            {
                **base,
                "prompt_orthogonal_identity_record_id": _stable_digest(base),
            }
        )
    return records


def build_prompt_orthogonal_decision(
    summaries: Iterable[Mapping[str, Any]],
    pairs: Iterable[Mapping[str, Any]],
    identities: Iterable[Mapping[str, Any]],
    clean_identities: Iterable[Mapping[str, Any]],
    failures: Iterable[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    summary_rows = [dict(row) for row in summaries]
    pair_rows = [dict(row) for row in pairs]
    identity_rows = [dict(row) for row in identities]
    clean_rows = [dict(row) for row in clean_identities]
    failure_rows = [dict(row) for row in failures]
    pair_fraction = (
        sum(row["prompt_orthogonal_owner_over_wrong"] for row in pair_rows)
        / len(pair_rows)
        if pair_rows else 0.0
    )
    top1_fraction = (
        sum(row["prompt_orthogonal_owner_top1"] for row in identity_rows)
        / len(identity_rows)
        if identity_rows else 0.0
    )
    clean_top1_fraction = (
        sum(row["prompt_orthogonal_owner_top1"] for row in clean_rows)
        / len(clean_rows)
        if clean_rows else 1.0
    )
    prompt_fractions = {
        prompt_id: (
            sum(
                row["prompt_orthogonal_owner_over_wrong"]
                for row in pair_rows
                if row["prompt_id"] == prompt_id
            )
            / max(
                1,
                sum(row["prompt_id"] == prompt_id for row in pair_rows),
            )
        )
        for prompt_id in config["heldout_prompt_ids"]
    }
    prompt_gap = max(prompt_fractions.values()) - min(
        prompt_fractions.values()
    )
    clean_by_identity = {
        (row["prompt_id"], row["seed_id"]): row for row in clean_rows
    }
    owner_clean_wins = [
        float(row["prompt_orthogonal_owner_score"])
        > float(
            clean_by_identity[(row["prompt_id"], row["seed_id"])][
                "prompt_orthogonal_owner_score"
            ]
        )
        for row in identity_rows
        if (row["prompt_id"], row["seed_id"]) in clean_by_identity
    ]
    watermarked_over_clean_fraction = (
        sum(owner_clean_wins) / len(owner_clean_wins)
        if owner_clean_wins else 0.0
    )
    reliability = min(
        (
            float(row["minimum_replay_reliability"])
            for row in summary_rows
        ),
        default=0.0,
    )
    coverage_ready = bool(
        len(summary_rows) == int(config["summary_record_count"])
        and len(pair_rows) == int(config["pair_record_count"])
        and len(identity_rows) == int(config["identity_record_count"])
        and len(clean_rows) == int(config["clean_identity_record_count"])
        and not failure_rows
        and all(
            row.get("prompt_orthogonal_vector_context_complete") is True
            and row.get(
                "prompt_orthogonal_key_independent_trace_complete"
            )
            is True
            and row.get(
                "prompt_orthogonal_fixed_trace_key_independent"
            )
            is True
            and row.get("prompt_orthogonal_base_velocity_call_count") == 20
            and row.get(
                "prompt_orthogonal_trace_construction_base_velocity_call_count"
            )
            == 20
            and row.get("prompt_orthogonal_total_base_velocity_call_count")
            == 40
            and row.get("prompt_orthogonal_candidate_count") == 9
            and row.get(
                "prompt_orthogonal_replay_reliability_mode"
            )
            == config["prompt_orthogonal_replay_reliability_mode"]
            and row.get(
                "prompt_orthogonal_plane_construction_device"
            )
            == PROMPT_ORTHOGONAL_CANONICAL_PLANE_CONSTRUCTION_DEVICE
            for row in summary_rows
        )
    )
    gate_ready = bool(
        coverage_ready
        and pair_fraction
        >= float(config["minimum_owner_over_wrong_pair_fraction"])
        and top1_fraction
        >= float(config["minimum_owner_top1_identity_fraction"])
        and prompt_gap
        <= float(config["maximum_prompt_pair_fraction_gap"])
        and reliability >= float(config["minimum_replay_reliability"])
        and watermarked_over_clean_fraction
        >= float(config["minimum_watermarked_over_clean_owner_fraction"])
        and clean_top1_fraction
        <= float(config["maximum_clean_owner_top1_fraction"])
    )
    if failure_rows:
        classification = "runtime_or_input_failure_stop"
    elif gate_ready:
        classification = (
            "prompt_orthogonal_mechanism_smoke_passed_"
            "independent_calibration_design_allowed"
        )
    else:
        classification = (
            "prompt_orthogonal_mechanism_smoke_failed_stop_instance"
        )
    return {
        "record_version": RECORD_VERSION,
        "profile_id": PROFILE_ID,
        "prompt_orthogonal_smoke_decision": classification,
        "prompt_orthogonal_gate_ready": gate_ready,
        "coverage_ready": coverage_ready,
        "prompt_orthogonal_owner_over_wrong_pair_fraction": pair_fraction,
        "prompt_orthogonal_owner_top1_identity_fraction": top1_fraction,
        "prompt_orthogonal_pair_fraction_by_prompt": prompt_fractions,
        "prompt_orthogonal_prompt_pair_fraction_gap": prompt_gap,
        "prompt_orthogonal_watermarked_over_clean_owner_fraction": (
            watermarked_over_clean_fraction
        ),
        "prompt_orthogonal_clean_owner_top1_fraction": clean_top1_fraction,
        "minimum_replay_reliability": reliability,
        "summary_record_count": len(summary_rows),
        "pair_record_count": len(pair_rows),
        "identity_record_count": len(identity_rows),
        "clean_identity_record_count": len(clean_rows),
        "failure_record_count": len(failure_rows),
        "formal_result": False,
        "stage_progression_allowed": False,
        "claim_support_status": config["claim_support_status"],
    }


def run_prompt_orthogonal_state_trajectory_smoke(
    input_root: str | Path,
    output_root: str | Path,
    temporal_source_root: str | Path,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    generation_runner: Any = _run_generation,
    pipeline_loader: Any = _load_pipeline,
) -> dict[str, Any]:
    config = _read_json(config_path)
    validate_prompt_orthogonal_smoke_config(config)
    validated = validate_controlled_embedding_source_result(
        input_root,
        config,
    )
    temporal = validate_temporal_failure_source(
        temporal_source_root,
        config,
    )
    output = Path(output_root).resolve()
    if not output.is_dir() or any(output.iterdir()):
        raise FileExistsError(
            "prompt-orthogonal output root 必须是已创建的空目录"
        )
    plan = build_prompt_orthogonal_generation_plan(validated, config)
    write_jsonl(
        output / "records" / "prompt_orthogonal_generation_plan.jsonl",
        plan,
    )
    write_json(
        output / "artifacts" / "prompt_orthogonal_execution_decision.json",
        {
            "record_version": RECORD_VERSION,
            "profile_id": PROFILE_ID,
            "execution_preflight_status": "ready",
            "generation_execution_allowed": True,
            "generation_record_count": 8,
            "generation_step_count": 8,
            "replay_step_count": 20,
            "wrong_owner_candidate_count": 8,
            "lambda_max": 0.12,
            "no_attack_only": True,
            "endpoint_gate_execution_allowed": False,
            "state_space_posterior_execution_allowed": False,
            "attacked_phase_execution_allowed": False,
            "fixed_fpr_evaluation_allowed": False,
            "external_baseline_execution_allowed": False,
            "stage_progression_allowed": False,
            "formal_result": False,
            "claim_support_status": config["claim_support_status"],
        },
    )
    for suffix, source_path in (
        (PROMPT_SUITE_SUFFIX, validated["prompt_suite_path"]),
        (LIKELIHOOD_SUFFIX, validated["likelihood_path"]),
    ):
        target = output / suffix
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target)
    for source_path in (
        Path(validated["source_decision_path"]),
        Path(validated["source_manifest_path"]),
        Path(temporal["decision_path"]),
        Path(temporal["manifest_path"]),
        Path(temporal["summary_path"]),
        Path(temporal["pair_path"]),
        Path(temporal["identity_path"]),
        Path(temporal["failure_path"]),
    ):
        target = output / "inputs" / "source" / source_path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target)
    pipeline_cache: dict[str, Any] = {}
    generation_result = generation_runner(
        validated,
        output,
        plan,
        pipeline_cache=pipeline_cache,
    )
    validate_prompt_orthogonal_generation_execution(
        output,
        generation_result,
    )

    def cached_loader(model_id: str, *, revision: str | None = None) -> Any:
        del revision
        return pipeline_cache[model_id]

    summaries, failures = _execute_replay(
        output,
        config,
        likelihood=_likelihood_config(output / LIKELIHOOD_SUFFIX),
        pipeline_loader=cached_loader if pipeline_cache else pipeline_loader,
    )
    pairs = build_prompt_orthogonal_pair_records(summaries)
    identities = build_prompt_orthogonal_identity_records(
        summaries,
        variant=WATERMARKED_VARIANT,
    )
    clean_identities = build_prompt_orthogonal_identity_records(
        summaries,
        variant=CLEAN_VARIANT,
    )
    decision = build_prompt_orthogonal_decision(
        summaries,
        pairs,
        identities,
        clean_identities,
        failures,
        config,
    )
    write_jsonl(
        output / "records" / "prompt_orthogonal_summary_records.jsonl",
        summaries,
    )
    write_jsonl(
        output / "records" / "prompt_orthogonal_pair_records.jsonl",
        pairs,
    )
    write_jsonl(
        output / "records" / "prompt_orthogonal_identity_records.jsonl",
        identities,
    )
    write_jsonl(
        output / "records" / "prompt_orthogonal_clean_identity_records.jsonl",
        clean_identities,
    )
    write_jsonl(
        output / "records" / "prompt_orthogonal_failure_records.jsonl",
        failures,
    )
    write_json(
        output / "artifacts" / "prompt_orthogonal_smoke_decision.json",
        decision,
    )
    report = output / "reports" / "prompt_orthogonal_smoke_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "\n".join(
            [
                "# Prompt-orthogonal state-trajectory smoke",
                "",
                f"- Decision: `{decision['prompt_orthogonal_smoke_decision']}`",
                f"- Gate ready: `{decision['prompt_orthogonal_gate_ready']}`",
                f"- Owner/wrong fraction: `{decision['prompt_orthogonal_owner_over_wrong_pair_fraction']}`",
                f"- Owner top-1 fraction: `{decision['prompt_orthogonal_owner_top1_identity_fraction']}`",
                f"- Prompt gap: `{decision['prompt_orthogonal_prompt_pair_fraction_gap']}`",
                f"- Watermarked/clean owner fraction: `{decision['prompt_orthogonal_watermarked_over_clean_owner_fraction']}`",
                f"- Clean owner top-1 fraction: `{decision['prompt_orthogonal_clean_owner_top1_fraction']}`",
                "",
                "该结果仅为 no-attack mechanism smoke，不是论文证据。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    write_json(
        output / "artifacts" / "prompt_orthogonal_smoke_manifest.json",
        {
            "artifact_id": "prompt_orthogonal_smoke_manifest",
            "artifact_type": "manifest",
            "profile_id": PROFILE_ID,
            "generation_result": generation_result,
            "generation_record_count": 8,
            "summary_record_count": len(summaries),
            "pair_record_count": len(pairs),
            "identity_record_count": len(identities),
            "clean_identity_record_count": len(clean_identities),
            "failure_record_count": len(failures),
            "base_model_velocity_per_step_shared_across_candidates": True,
            "prompt_orthogonal_replay_reliability_mode": config[
                "prompt_orthogonal_replay_reliability_mode"
            ],
            "prompt_orthogonal_plane_construction_device": (
                PROMPT_ORTHOGONAL_CANONICAL_PLANE_CONSTRUCTION_DEVICE
            ),
            "fixed_trace_base_velocity_call_count_per_video": 20,
            "candidate_evaluation_base_velocity_call_count_per_video": 20,
            "total_base_velocity_call_count_per_video": 40,
            "endpoint_gate_executed": False,
            "state_space_posterior_executed": False,
            "attacked_phase_executed": False,
            "fixed_fpr_evaluation_executed": False,
            "external_baseline_execution_executed": False,
            "stage_progression_allowed": False,
            "formal_result": False,
            "claim_support_status": config["claim_support_status"],
        },
    )
    return decision


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-run-root", required=True)
    parser.add_argument("--temporal-source-root", required=True)
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    decision = run_prompt_orthogonal_state_trajectory_smoke(
        args.input_root,
        args.output_run_root,
        args.temporal_source_root,
        args.config_path,
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
