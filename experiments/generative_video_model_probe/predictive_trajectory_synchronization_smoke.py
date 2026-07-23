"""最小 predictive trajectory synchronization no-attack smoke。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping

from evaluation.protocol.record_writer import write_json, write_jsonl
from experiments.generative_video_model_probe.colab_runtime import (
    PREDICTIVE_TRAJECTORY_SYNCHRONIZATION_SMOKE_PROFILE,
    WAN21_PRIMARY_MODEL_ID,
    run_colab_probe,
    validate_generation_model_provenance,
)
from experiments.generative_video_model_probe.formal_flow_evidence_runner import (
    _evaluate_fixed_replay_hypothesis_for_key,
    _generation_key,
    _invoke_pipeline_loader,
    _load_pipeline,
    _prompt_text_by_id,
    _run_attacked_video_replay_for_model,
    _validated_flow_key_context,
    _wrong_owner_generation_key,
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
from main.methods.state_space_watermark.flow_tubelet_key_code import (
    FlowTubeletKeyCodeConfig,
)
from main.methods.state_space_watermark.predictive_trajectory_carrier import (
    PREDICTIVE_TRAJECTORY_CARRIER_ID,
    PREDICTIVE_TRAJECTORY_MAXIMUM_ABSOLUTE_CODE_CORRELATION,
    PREDICTIVE_TRAJECTORY_MINIMUM_ACTIVE_CODE_MAGNITUDE,
    PREDICTIVE_TRAJECTORY_MINIMUM_ACTIVE_PHASE_COUNT,
    PREDICTIVE_TRAJECTORY_MINIMUM_WEIGHTED_CODE_ENERGY,
    PREDICTIVE_TRAJECTORY_PHASE_SEGMENT_COUNT,
    PredictiveTrajectoryCarrierConfig,
    predictive_trajectory_weighted_code_correlation,
)
from main.methods.state_space_watermark.replay_inversion import (
    ReplayGaussianLikelihoodConfig,
)
from main.methods.state_space_watermark.velocity_field_constraint import (
    VelocityFieldConstraintConfig,
)
from main.methods.state_space_watermark.wan_flow_replay_backend import (
    predictive_schedule_for_replay,
)


DEFAULT_CONFIG_PATH = (
    "configs/protocol/sstw_predictive_trajectory_synchronization_smoke.json"
)
PROFILE_ID = "sstw_predictive_trajectory_synchronization_smoke"
RECORD_VERSION = "predictive_trajectory_synchronization_smoke"
PREDICTIVE_VARIANT = "predictive_signed_phase_code"
NONNEGATIVE_VARIANT = "nonnegative_phase_control"
WRONG_OWNER_KEY_CONTROL_CANDIDATE_COUNT = 32


def validate_predictive_trajectory_config(
    config: Mapping[str, Any],
) -> None:
    """冻结失败后独立 falsification smoke，不允许结果后调参。"""

    false_fields = (
        "attacked_phase_execution_allowed",
        "cross_project_integration_allowed",
        "external_baseline_execution_allowed",
        "fixed_fpr_evaluation_allowed",
        "large_scale_generation_allowed",
        "stage_progression_allowed",
        "test_split_claims_allowed",
    )
    invalid = [name for name in false_fields if config.get(name) is not False]
    if invalid:
        raise ValueError(
            "predictive trajectory 禁止项未冻结: " + ", ".join(invalid)
        )
    exact = {
        "profile_id": PROFILE_ID,
        "paper_result_level": "predictive_trajectory_synchronization_smoke",
        "claim_support_status": (
            "predictive_trajectory_synchronization_smoke_only_not_paper_evidence"
        ),
        "required_source_profile_id": (
            "sstw_controlled_embedding_strength_diagnostic"
        ),
        "required_source_controlled_embedding_decision": (
            "lambda_increase_did_not_repair_path_signal_stop"
        ),
        "required_generation_model_id": WAN21_PRIMARY_MODEL_ID,
        "predictive_trajectory_carrier_id": (
            PREDICTIVE_TRAJECTORY_CARRIER_ID
        ),
        "no_attack_only": True,
        "generation_step_count": 20,
        "replay_step_count": 20,
        "phase_segment_count": PREDICTIVE_TRAJECTORY_PHASE_SEGMENT_COUNT,
        "minimum_active_phase_count": (
            PREDICTIVE_TRAJECTORY_MINIMUM_ACTIVE_PHASE_COUNT
        ),
        "smoke_generation_record_count": 8,
        "required_source_generation_record_count": 16,
        "required_source_summary_record_count": 96,
        "required_source_pair_record_count": 84,
        "required_source_failure_record_count": 0,
        "heldout_prompt_ids": [
            "probe_paper_paper_master_prompt_003",
            "probe_paper_paper_master_prompt_004",
        ],
        "heldout_seed_ids": [
            "probe_paper_paper_master_test_seed_01",
            "probe_paper_paper_master_test_seed_02",
        ],
        "trajectory_carrier_variant_ids": [
            PREDICTIVE_VARIANT,
            NONNEGATIVE_VARIANT,
        ],
        "wrong_owner_key_control_candidate_count": (
            WRONG_OWNER_KEY_CONTROL_CANDIDATE_COUNT
        ),
    }
    mismatches = [
        name for name, expected in exact.items() if config.get(name) != expected
    ]
    if mismatches:
        raise ValueError(
            "predictive trajectory 配置字段未冻结: "
            + ", ".join(mismatches)
        )
    numeric = (
        ("lambda_max", 0.12),
        (
            "minimum_active_code_magnitude",
            PREDICTIVE_TRAJECTORY_MINIMUM_ACTIVE_CODE_MAGNITUDE,
        ),
        (
            "minimum_weighted_code_energy",
            PREDICTIVE_TRAJECTORY_MINIMUM_WEIGHTED_CODE_ENERGY,
        ),
        (
            "maximum_absolute_code_correlation",
            PREDICTIVE_TRAJECTORY_MAXIMUM_ABSOLUTE_CODE_CORRELATION,
        ),
        ("minimum_predictive_correct_over_wrong_fraction", 0.75),
        ("minimum_predictive_over_nonnegative_margin_fraction", 0.75),
        ("minimum_replay_reliability", 0.05),
    )
    allowed_fields = (
        set(false_fields)
        | set(exact)
        | {name for name, _expected in numeric}
    )
    unknown_fields = sorted(set(config) - allowed_fields)
    if unknown_fields:
        raise ValueError(
            "predictive trajectory 配置包含未声明字段: "
            + ", ".join(unknown_fields)
        )
    for name, expected in numeric:
        if not math.isclose(
            float(config.get(name) or 0.0),
            float(expected),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"predictive trajectory 数值字段未冻结: {name}"
            )


def build_predictive_trajectory_generation_plan(
    validated: Mapping[str, Any],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """使用未参与前一轮诊断的两个 prompt 与两个 test seed。"""

    prompt_rows = {
        str(row["prompt_id"]): dict(row)
        for row in validated["prompt_suite"].get("prompts") or []
    }
    seed_rows = {
        str(row["seed_id"]): dict(row)
        for row in validated["prompt_suite"].get("seeds") or []
    }
    missing_prompts = [
        value for value in config["heldout_prompt_ids"] if value not in prompt_rows
    ]
    missing_seeds = [
        value for value in config["heldout_seed_ids"] if value not in seed_rows
    ]
    if missing_prompts or missing_seeds:
        raise ValueError(
            "predictive trajectory prompt suite 缺少冻结 held-out 身份"
        )
    source_prompt_ids = {
        str(row["prompt_id"]) for row in validated["generation_rows"]
    }
    source_seed_ids = {
        str(row["seed_id"]) for row in validated["generation_rows"]
    }
    overlapping_prompts = sorted(
        set(config["heldout_prompt_ids"]) & source_prompt_ids
    )
    overlapping_seeds = sorted(
        set(config["heldout_seed_ids"]) & source_seed_ids
    )
    if overlapping_prompts or overlapping_seeds:
        raise ValueError(
            "predictive trajectory held-out 身份已出现在 source generation: "
            f"prompts={overlapping_prompts}, seeds={overlapping_seeds}"
        )
    variants = (
        (
            PREDICTIVE_VARIANT,
            "sstw_predictive_signed_phase_code",
        ),
        (
            NONNEGATIVE_VARIANT,
            "sstw_nonnegative_phase_control",
        ),
    )
    plan: list[dict[str, Any]] = []
    for prompt_id in config["heldout_prompt_ids"]:
        for seed_id in config["heldout_seed_ids"]:
            prompt = prompt_rows[prompt_id]
            seed = seed_rows[seed_id]
            seed_value = int(
                seed.get("seed_value")
                if seed.get("seed_value") is not None
                else seed.get("generation_seed_random")
            )
            for variant_id, method_variant in variants:
                identity = {
                    "generation_model_id": config[
                        "required_generation_model_id"
                    ],
                    "prompt_id": prompt_id,
                    "seed_id": seed_id,
                    "trajectory_carrier_variant_id": variant_id,
                }
                plan.append(
                    {
                        **prompt,
                        **seed,
                        **identity,
                        "predictive_trajectory_plan_record_id": (
                            _stable_digest(identity)
                        ),
                        "seed_value": seed_value,
                        "generation_seed_random": seed_value,
                        "cross_model_role": "main_generation_model",
                        "sample_role": "controlled_embedding_positive_source",
                        "generation_sample_role": (
                            "controlled_embedding_positive_source"
                        ),
                        "method_variant": method_variant,
                        "watermark_embedding_status": (
                            "flow_scheduler_velocity_constraint"
                        ),
                        "lambda_max": 0.12,
                        "predictive_trajectory_carrier_id": (
                            config["predictive_trajectory_carrier_id"]
                        ),
                        "predictive_trajectory_phase_segment_count": 8,
                        "predictive_trajectory_minimum_active_phase_count": 4,
                        "predictive_trajectory_maximum_absolute_code_correlation": (
                            0.75
                        ),
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
        raise RuntimeError("predictive trajectory generation plan 必须为8条")
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
        str(row["predictive_trajectory_plan_record_id"]): velocity
        for row in plan
    }
    carrier = PredictiveTrajectoryCarrierConfig()
    carriers = {
        str(row["predictive_trajectory_plan_record_id"]): (
            carrier
            if row["trajectory_carrier_variant_id"]
            == PREDICTIVE_VARIANT
            else None
        )
        for row in plan
    }
    return run_colab_probe(
        output_root,
        validated["prompt_suite_path"],
        PREDICTIVE_TRAJECTORY_SYNCHRONIZATION_SMOKE_PROFILE,
        model_id=WAN21_PRIMARY_MODEL_ID,
        cross_model_id=None,
        generation_plan_override=plan,
        velocity_config_by_plan_record_id=velocity_configs,
        predictive_trajectory_carrier_by_plan_record_id=carriers,
        pipeline_cache=pipeline_cache,
    )


def validate_predictive_generation_execution(
    output_root: str | Path,
    plan: Iterable[Mapping[str, Any]],
    generation_result: Mapping[str, Any],
) -> None:
    """在进入 replay 前区分完整生成与 runtime/GPU failure。"""

    root = Path(output_root)
    decision = generation_result.get("decision")
    if not isinstance(decision, Mapping):
        raise RuntimeError("predictive trajectory generation 缺少 runtime decision")
    if (
        decision.get("implementation_decision") != "PASS"
        or decision.get("mechanism_decision")
        != "GENERATION_READY_NO_ATTACK_REPLAY_PENDING"
    ):
        raise RuntimeError(
            "predictive trajectory generation runtime decision 未就绪"
        )
    if (
        int(generation_result.get("generation_record_count") or -1) != 8
        or int(generation_result.get("trajectory_record_count") or -1)
        != 160
    ):
        raise RuntimeError(
            "predictive trajectory generation runtime record count 不完整"
        )
    plan_rows = [dict(row) for row in plan]
    expected_plan_ids = {
        str(row["predictive_trajectory_plan_record_id"])
        for row in plan_rows
    }
    generation_rows = _read_jsonl(
        root / "records" / "generation_records.jsonl"
    )
    observed_plan_ids = {
        str(row.get("predictive_trajectory_plan_record_id") or "")
        for row in generation_rows
    }
    if (
        len(generation_rows) != 8
        or observed_plan_ids != expected_plan_ids
        or any(
            row.get("generation_status") != "success"
            or row.get("colab_runtime_profile")
            != PREDICTIVE_TRAJECTORY_SYNCHRONIZATION_SMOKE_PROFILE
            for row in generation_rows
        )
    ):
        raise RuntimeError(
            "predictive trajectory generation records 未完整成功"
        )
    trajectory_rows = _read_jsonl(
        root / "records" / "trajectory_trace.jsonl"
    )
    trace_counts = {
        plan_id: sum(
            str(row.get("predictive_trajectory_plan_record_id") or "")
            == plan_id
            for row in trajectory_rows
        )
        for plan_id in expected_plan_ids
    }
    if (
        len(trajectory_rows) != 160
        or set(trace_counts.values()) != {20}
        or any(
            row.get("endpoint_control_enabled") is not False
            for row in trajectory_rows
        )
    ):
        raise RuntimeError(
            "predictive trajectory generation step records 未保持20步 "
            "endpoint-disabled 边界"
        )
    predictive_plan_ids = {
        str(row["predictive_trajectory_plan_record_id"])
        for row in plan_rows
        if row["trajectory_carrier_variant_id"] == PREDICTIVE_VARIANT
    }
    predictive_steps = [
        row
        for row in trajectory_rows
        if str(row.get("predictive_trajectory_plan_record_id") or "")
        in predictive_plan_ids
    ]
    if len(predictive_steps) != 80 or any(
        row.get("predictive_trajectory_noncollapse_verified") is not True
        or (
            row.get("predictive_trajectory_inactive_phase_noop") is not True
            and (
                row.get("predictive_trajectory_norm_guard_passed") is not True
                or row.get("predictive_trajectory_energy_guard_passed")
                is not True
            )
        )
        for row in predictive_steps
    ):
        raise RuntimeError(
            "predictive trajectory generation carrier/guard 证据不完整"
        )


def _execute_replay(
    output_root: Path,
    config: Mapping[str, Any],
    *,
    likelihood: ReplayGaussianLikelihoodConfig,
    pipeline_loader: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    generation_rows = [
        row
        for row in _read_jsonl(
            output_root / "records" / "generation_records.jsonl"
        )
        if row.get("generation_status") == "success"
    ]
    prompt_map = _prompt_text_by_id(
        _read_json(output_root / PROMPT_SUITE_SUFFIX)
    )
    revisions: dict[str, str] = {}
    for row in generation_rows:
        model_id = str(row["generation_model_id"])
        revision = validate_generation_model_provenance(row)
        previous = revisions.setdefault(model_id, revision)
        if previous != revision:
            raise RuntimeError("predictive trajectory 同一模型混用 revision")
    pipelines = {
        model_id: _invoke_pipeline_loader(
            pipeline_loader,
            model_id=model_id,
            revision=revision,
        )
        for model_id, revision in revisions.items()
    }
    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    carrier = PredictiveTrajectoryCarrierConfig()
    tubelet = FlowTubeletKeyCodeConfig()
    for source in generation_rows:
        base = {
            "record_version": RECORD_VERSION,
            "profile_id": PROFILE_ID,
            "generation_model_id": source["generation_model_id"],
            "prompt_id": source["prompt_id"],
            "seed_id": source["seed_id"],
            "trajectory_trace_id": source["trajectory_trace_id"],
            "predictive_trajectory_plan_record_id": source[
                "predictive_trajectory_plan_record_id"
            ],
            "trajectory_carrier_variant_id": source[
                "trajectory_carrier_variant_id"
            ],
            "method_variant": source["method_variant"],
            "video_condition_id": (
                "no_attack_predictive_trajectory_synchronization"
            ),
            "replay_grid_step_count": 20,
            "claim_support_status": config["claim_support_status"],
        }
        try:
            prompt = prompt_map[str(source["prompt_id"])]
            pipeline = pipelines[str(source["generation_model_id"])]
            key_context = _validated_flow_key_context(
                source,
                prompt=prompt,
                scheduler=pipeline.scheduler,
            )
            correct_key = _generation_key(source)
            active_carrier = (
                carrier
                if source["trajectory_carrier_variant_id"]
                == PREDICTIVE_VARIANT
                else None
            )
            video_path = Path(str(source["video_path"]))
            if not video_path.is_file():
                raise FileNotFoundError(f"生成视频不存在: {video_path}")
            replay = _run_attacked_video_replay_for_model(
                pipeline,
                video_path,
                prompt=prompt,
                key_text=correct_key,
                key_context=key_context,
                likelihood_config=likelihood,
                replay_step_counts=(20,),
                predictive_trajectory_carrier_config=active_carrier,
                endpoint_control_enabled=False,
            )
            correct_trajectory = replay.replay_trajectories[0]
            correct_schedule = predictive_schedule_for_replay(
                replay.primary_schedule,
                key_text=correct_key,
                key_context=key_context,
                tubelet_config=tubelet,
            )
            wrong_key = ""
            selected_code_correlation: float | None = None
            wrong_key_candidate_index = -1
            for candidate_index in range(
                int(config["wrong_owner_key_control_candidate_count"])
            ):
                candidate_key = _wrong_owner_generation_key(
                    source,
                    extra_context={
                        "predictive_wrong_owner_key_control_candidate_index": (
                            candidate_index
                        )
                    },
                )
                candidate_schedule = predictive_schedule_for_replay(
                    replay.primary_schedule,
                    key_text=candidate_key,
                    key_context=key_context,
                    tubelet_config=tubelet,
                )
                candidate_correlation = (
                    predictive_trajectory_weighted_code_correlation(
                        correct_schedule,
                        candidate_schedule,
                    )
                )
                if abs(candidate_correlation) <= float(
                    config["maximum_absolute_code_correlation"]
                ) + 1e-12:
                    wrong_key = candidate_key
                    wrong_key_candidate_index = candidate_index
                    selected_code_correlation = candidate_correlation
                    break
            if not wrong_key:
                raise RuntimeError(
                    "predictive trajectory 无法构造冻结的低相关 "
                    "wrong-owner key control"
                )
            code_correlation = (
                selected_code_correlation
                if active_carrier is not None
                else None
            )
            wrong_trajectory, _wrong_path = (
                _evaluate_fixed_replay_hypothesis_for_key(
                    pipeline,
                    replay,
                    prompt=prompt,
                    key_text=wrong_key,
                    key_context=key_context,
                    predictive_trajectory_carrier_config=active_carrier,
                    endpoint_control_enabled=False,
                )
            )
            for key_role, trajectory in (
                ("correct_owner_key", correct_trajectory),
                ("wrong_owner_key", wrong_trajectory),
            ):
                summary_base = {
                    **base,
                    "candidate_key_role": key_role,
                }
                summaries.append(
                    {
                        **summary_base,
                        "predictive_trajectory_summary_record_id": (
                            _stable_digest(summary_base)
                        ),
                        "predictive_replay_log_likelihood_ratio": (
                            trajectory.replay_log_likelihood_ratio
                        ),
                        "predictive_replay_cycle_relative_error": (
                            trajectory.candidate_cycle_relative_error
                        ),
                        "predictive_null_cycle_relative_error": (
                            trajectory.null_cycle_relative_error
                        ),
                        "trajectory_global_reliability": (
                            replay.replay_uncertainty.replay_reliability
                        ),
                        "predictive_owner_wrong_weighted_code_correlation": (
                            code_correlation
                        ),
                        "predictive_wrong_owner_key_control_candidate_index": (
                            wrong_key_candidate_index
                        ),
                        "metric_status": (
                            "measured_predictive_trajectory_synchronization_smoke"
                        ),
                    }
                )
        except Exception as exc:  # pragma: no cover - real Colab GPU path
            failures.append(
                {
                    **base,
                    "predictive_trajectory_smoke_status": "failed",
                    "predictive_trajectory_smoke_failure_reason": str(exc),
                }
            )
    return summaries, failures


def _fraction(values: Iterable[bool]) -> float:
    rows = list(values)
    return sum(bool(value) for value in rows) / len(rows) if rows else 0.0


def build_predictive_pair_records(
    summaries: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in summaries]
    by_key = {
        (
            str(row["prompt_id"]),
            str(row["seed_id"]),
            str(row["trajectory_carrier_variant_id"]),
            str(row["candidate_key_role"]),
        ): row
        for row in rows
    }
    identities = sorted(
        {(str(row["prompt_id"]), str(row["seed_id"])) for row in rows}
    )
    pairs: list[dict[str, Any]] = []
    margins: dict[tuple[str, str, str], tuple[float, int]] = {}
    for prompt_id, seed_id in identities:
        for variant in (PREDICTIVE_VARIANT, NONNEGATIVE_VARIANT):
            correct = by_key.get(
                (prompt_id, seed_id, variant, "correct_owner_key")
            )
            wrong = by_key.get(
                (prompt_id, seed_id, variant, "wrong_owner_key")
            )
            if correct is None or wrong is None:
                continue
            candidate_index_values = [
                int(row["predictive_wrong_owner_key_control_candidate_index"])
                for row in (correct, wrong)
                if row.get(
                    "predictive_wrong_owner_key_control_candidate_index"
                )
                is not None
            ]
            if (
                len(candidate_index_values) != 2
                or len(set(candidate_index_values)) != 1
            ):
                continue
            candidate_index = candidate_index_values[0]
            margin = float(
                correct["predictive_replay_log_likelihood_ratio"]
            ) - float(wrong["predictive_replay_log_likelihood_ratio"])
            base = {
                "record_version": RECORD_VERSION,
                "profile_id": PROFILE_ID,
                "prompt_id": prompt_id,
                "seed_id": seed_id,
                "trajectory_carrier_variant_id": variant,
                "comparison_kind": (
                    "correct_owner_key_over_wrong_owner_key_predictive_llr"
                ),
            }
            pairs.append(
                {
                    **base,
                    "predictive_trajectory_pair_record_id": (
                        _stable_digest(base)
                    ),
                    "correct_over_wrong_predictive_llr_margin": margin,
                    "predictive_wrong_owner_key_control_candidate_index": (
                        candidate_index
                    ),
                    "minimum_pair_reliability": min(
                        float(correct["trajectory_global_reliability"]),
                        float(wrong["trajectory_global_reliability"]),
                    ),
                }
            )
            margins[(prompt_id, seed_id, variant)] = (
                margin,
                candidate_index,
            )
    for prompt_id, seed_id in identities:
        signed_entry = margins.get(
            (prompt_id, seed_id, PREDICTIVE_VARIANT)
        )
        control_entry = margins.get(
            (prompt_id, seed_id, NONNEGATIVE_VARIANT)
        )
        if signed_entry is None or control_entry is None:
            continue
        signed, signed_candidate_index = signed_entry
        control, control_candidate_index = control_entry
        if signed_candidate_index != control_candidate_index:
            continue
        base = {
            "record_version": RECORD_VERSION,
            "profile_id": PROFILE_ID,
            "prompt_id": prompt_id,
            "seed_id": seed_id,
            "trajectory_carrier_variant_id": PREDICTIVE_VARIANT,
            "control_trajectory_carrier_variant_id": NONNEGATIVE_VARIANT,
            "comparison_kind": (
                "predictive_signed_over_nonnegative_llr_margin"
            ),
        }
        pairs.append(
            {
                **base,
                "predictive_trajectory_pair_record_id": (
                    _stable_digest(base)
                ),
                "predictive_over_nonnegative_llr_margin_gain": (
                    signed - control
                ),
                "predictive_wrong_owner_key_control_candidate_index": (
                    signed_candidate_index
                ),
            }
        )
    return pairs


def build_predictive_decision(
    summaries: Iterable[Mapping[str, Any]],
    pairs: Iterable[Mapping[str, Any]],
    failures: Iterable[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    summary_rows = [dict(row) for row in summaries]
    pair_rows = [dict(row) for row in pairs]
    failure_rows = [dict(row) for row in failures]
    signed_pairs = [
        row
        for row in pair_rows
        if row.get("comparison_kind")
        == "correct_owner_key_over_wrong_owner_key_predictive_llr"
        and row.get("trajectory_carrier_variant_id")
        == PREDICTIVE_VARIANT
    ]
    gains = [
        row
        for row in pair_rows
        if row.get("comparison_kind")
        == "predictive_signed_over_nonnegative_llr_margin"
    ]
    correct_fraction = _fraction(
        float(row["correct_over_wrong_predictive_llr_margin"]) > 0.0
        for row in signed_pairs
    )
    gain_fraction = _fraction(
        float(row["predictive_over_nonnegative_llr_margin_gain"]) > 0.0
        for row in gains
    )
    reliability = min(
        (float(row["minimum_pair_reliability"]) for row in signed_pairs),
        default=0.0,
    )
    signed_correlations: dict[tuple[str, str], set[float]] = {}
    correlation_parse_failed = False
    for summary in summary_rows:
        if summary.get("trajectory_carrier_variant_id") != PREDICTIVE_VARIANT:
            continue
        identity = (str(summary["prompt_id"]), str(summary["seed_id"]))
        value = summary.get(
            "predictive_owner_wrong_weighted_code_correlation"
        )
        try:
            correlation = float(value)
        except (TypeError, ValueError):
            correlation_parse_failed = True
            continue
        signed_correlations.setdefault(identity, set()).add(correlation)
    expected_signed_identities = {
        (str(row["prompt_id"]), str(row["seed_id"])) for row in signed_pairs
    }
    code_separation_ready = bool(
        len(signed_pairs) == 4
        and not correlation_parse_failed
        and set(signed_correlations) == expected_signed_identities
        and all(len(values) == 1 for values in signed_correlations.values())
        and all(
            abs(next(iter(values)))
            <= float(config["maximum_absolute_code_correlation"]) + 1e-12
            for values in signed_correlations.values()
        )
    )
    coverage_ready = bool(
        len(summary_rows) == 16
        and len(pair_rows) == 12
        and len(signed_pairs) == 4
        and len(gains) == 4
        and not failure_rows
    )
    gate_ready = bool(
        coverage_ready
        and code_separation_ready
        and correct_fraction
        >= float(
            config["minimum_predictive_correct_over_wrong_fraction"]
        )
        and gain_fraction
        >= float(
            config[
                "minimum_predictive_over_nonnegative_margin_fraction"
            ]
        )
        and reliability >= float(config["minimum_replay_reliability"])
    )
    if failure_rows:
        classification = "runtime_or_input_failure_stop"
    elif gate_ready:
        classification = (
            "predictive_trajectory_gate_passed_calibration_design_allowed"
        )
    else:
        classification = "predictive_trajectory_gate_failed_stop_method"
    return {
        "record_version": RECORD_VERSION,
        "profile_id": PROFILE_ID,
        "predictive_trajectory_smoke_decision": classification,
        "predictive_trajectory_gate_ready": gate_ready,
        "coverage_ready": coverage_ready,
        "predictive_code_separation_ready": code_separation_ready,
        "predictive_correct_over_wrong_fraction": correct_fraction,
        "predictive_over_nonnegative_margin_fraction": gain_fraction,
        "minimum_replay_reliability": reliability,
        "predictive_wrong_owner_key_control_candidate_count": int(
            config["wrong_owner_key_control_candidate_count"]
        ),
        "summary_record_count": len(summary_rows),
        "pair_record_count": len(pair_rows),
        "failure_record_count": len(failure_rows),
        "detector_semantics": (
            "frozen_key_independent_inversion_keyed_forward_replay_llr"
        ),
        "endpoint_gate_executed": False,
        "state_space_posterior_executed": False,
        "attacked_phase_executed": False,
        "fixed_fpr_evaluation_executed": False,
        "external_baseline_execution_executed": False,
        "stage_progression_allowed": False,
        "formal_result": False,
        "claim_support_status": config["claim_support_status"],
    }


def run_predictive_trajectory_synchronization_smoke(
    input_root: str | Path,
    output_root: str | Path,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    generation_runner: Any = _run_generation,
    pipeline_loader: Any = _load_pipeline,
) -> dict[str, Any]:
    config = _read_json(config_path)
    validate_predictive_trajectory_config(config)
    validated = validate_controlled_embedding_source_result(
        input_root,
        config,
    )
    output = Path(output_root).resolve()
    if not output.is_dir() or any(output.iterdir()):
        raise FileExistsError(
            "predictive trajectory output root 必须是已创建的空目录"
        )
    plan = build_predictive_trajectory_generation_plan(validated, config)
    write_jsonl(
        output / "records" / "predictive_trajectory_generation_plan.jsonl",
        plan,
    )
    write_json(
        output / "artifacts" / "predictive_trajectory_execution_decision.json",
        {
            "record_version": RECORD_VERSION,
            "profile_id": PROFILE_ID,
            "execution_preflight_status": "ready",
            "generation_execution_allowed": True,
            "generation_record_count": 8,
            "generation_step_count": 20,
            "replay_step_count": 20,
            "lambda_max": 0.12,
            "no_attack_only": True,
            "predictive_wrong_owner_key_control_candidate_count": int(
                config["wrong_owner_key_control_candidate_count"]
            ),
            "endpoint_gate_execution_allowed": False,
            "state_space_posterior_execution_allowed": False,
            "attacked_phase_execution_allowed": False,
            "fixed_fpr_evaluation_allowed": False,
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
    for name in ("source_decision_path", "source_manifest_path"):
        source_path = Path(validated[name])
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
    validate_predictive_generation_execution(
        output,
        plan,
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
    pairs = build_predictive_pair_records(summaries)
    decision = build_predictive_decision(
        summaries,
        pairs,
        failures,
        config,
    )
    write_jsonl(
        output / "records" / "predictive_trajectory_summary_records.jsonl",
        summaries,
    )
    write_jsonl(
        output / "records" / "predictive_trajectory_pair_records.jsonl",
        pairs,
    )
    write_jsonl(
        output / "records" / "predictive_trajectory_failure_records.jsonl",
        failures,
    )
    write_json(
        output / "artifacts" / "predictive_trajectory_smoke_decision.json",
        decision,
    )
    report = output / "reports" / "predictive_trajectory_smoke_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "\n".join(
            [
                "# Predictive trajectory synchronization smoke",
                "",
                f"- Decision: `{decision['predictive_trajectory_smoke_decision']}`",
                f"- Gate ready: `{decision['predictive_trajectory_gate_ready']}`",
                f"- Correct-over-wrong fraction: `{decision['predictive_correct_over_wrong_fraction']}`",
                f"- Predictive-over-control fraction: `{decision['predictive_over_nonnegative_margin_fraction']}`",
                f"- Minimum replay reliability: `{decision['minimum_replay_reliability']}`",
                "",
                "该结果只判断 keyed forward replay 预测同步，不是论文证据。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    write_json(
        output / "artifacts" / "predictive_trajectory_smoke_manifest.json",
        {
            "artifact_id": "predictive_trajectory_smoke_manifest",
            "artifact_type": "manifest",
            "profile_id": PROFILE_ID,
            "generation_result": generation_result,
            "generation_record_count": 8,
            "summary_record_count": len(summaries),
            "pair_record_count": len(pairs),
            "failure_record_count": len(failures),
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
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    decision = run_predictive_trajectory_synchronization_smoke(
        args.input_root,
        args.output_run_root,
        args.config_path,
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
