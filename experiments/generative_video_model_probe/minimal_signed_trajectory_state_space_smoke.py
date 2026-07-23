"""最小 signed trajectory state-space no-attack smoke。

该入口只验证时间载体是否能从视频 replay 中恢复。它不拟合 posterior、不运行
攻击、fixed-FPR、baseline 或论文统计，也不授权阶段推进。
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import shutil
from statistics import mean
from typing import Any, Iterable, Mapping

from evaluation.protocol.record_writer import write_json, write_jsonl
from experiments.generative_video_model_probe.colab_runtime import (
    MINIMAL_SIGNED_TRAJECTORY_STATE_SPACE_SMOKE_PROFILE,
    WAN21_PRIMARY_MODEL_ID,
    run_colab_probe,
    validate_generation_model_provenance,
)
from experiments.generative_video_model_probe.formal_flow_evidence_runner import (
    _compute_replay_endpoint_evidence_for_key,
    _evaluate_fixed_replay_hypothesis_for_key,
    _generation_key,
    _invoke_pipeline_loader,
    _load_pipeline,
    _prompt_text_by_id,
    _validated_flow_key_context,
    _wrong_owner_generation_key,
    build_flow_state_observation_sequence,
)
from main.methods.state_space_watermark.flow_state_posterior import (
    FlowEvidenceObservation,
)
from main.methods.state_space_watermark.formal_detector import (
    flow_evidence_observation_sequence_from_mappings,
)
from main.methods.state_space_watermark.replay_inversion import (
    ReplayGaussianLikelihoodConfig,
)
from main.methods.state_space_watermark.signed_trajectory_carrier import (
    SIGNED_BALANCED_AC_CARRIER_ID,
    SignedTrajectoryCarrierConfig,
)
from main.methods.state_space_watermark.velocity_field_constraint import (
    VelocityFieldConstraintConfig,
)


DEFAULT_CONFIG_PATH = (
    "configs/protocol/sstw_minimal_signed_trajectory_state_space_smoke.json"
)
PROFILE_ID = "sstw_minimal_signed_trajectory_state_space_smoke"
RECORD_VERSION = "minimal_signed_trajectory_state_space_smoke_v1"
SOURCE_DECISION_SUFFIX = (
    "artifacts/controlled_embedding_strength_diagnostic_decision.json"
)
SOURCE_MANIFEST_SUFFIX = (
    "artifacts/controlled_embedding_strength_diagnostic_manifest.json"
)
GENERATION_RECORDS_SUFFIX = "records/generation_records.jsonl"
SOURCE_SUMMARY_SUFFIX = (
    "records/controlled_embedding_strength_summary_records.jsonl"
)
SOURCE_PAIR_SUFFIX = "records/controlled_embedding_strength_pair_records.jsonl"
SOURCE_FAILURE_SUFFIX = (
    "records/controlled_embedding_strength_failure_records.jsonl"
)
PROMPT_SUITE_SUFFIX = "datasets/prompt_seed_suite.json"
LIKELIHOOD_SUFFIX = (
    "records/trajectory_replay_smoke_likelihood_calibrations.jsonl"
)
SIGNED_VARIANT = "signed_balanced_ac"
NONNEGATIVE_VARIANT = "nonnegative_phase_control"
CLEAN_VARIANT = "clean_unwatermarked_control"


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON 顶层必须是对象: {path}")
    return value


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"JSONL 行必须是对象: {path}")
        rows.append(value)
    return rows


def _stable_digest(payload: Mapping[str, Any]) -> str:
    return sha256(
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _find_unique_file(root: Path, logical_suffix: str) -> Path:
    suffix_parts = Path(logical_suffix).parts
    candidates = [
        path.resolve()
        for path in root.rglob(suffix_parts[-1])
        if path.is_file()
        and not path.is_symlink()
        and path.parts[-len(suffix_parts) :] == suffix_parts
    ]
    unique = sorted(set(candidates))
    if len(unique) != 1:
        raise RuntimeError(
            "minimal signed trajectory input 必须唯一包含 "
            f"{logical_suffix}: {unique}"
        )
    return unique[0]


def validate_minimal_signed_trajectory_config(
    config: Mapping[str, Any],
) -> None:
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
            "minimal signed trajectory 禁止项未冻结: " + ", ".join(invalid)
        )
    exact = {
        "profile_id": PROFILE_ID,
        "paper_result_level": "minimal_signed_trajectory_state_space_smoke",
        "claim_support_status": (
            "minimal_signed_trajectory_state_space_smoke_only_not_paper_evidence"
        ),
        "required_source_profile_id": (
            "sstw_controlled_embedding_strength_diagnostic"
        ),
        "required_source_controlled_embedding_decision": (
            "lambda_increase_did_not_repair_path_signal_stop"
        ),
        "required_generation_model_id": WAN21_PRIMARY_MODEL_ID,
        "signed_trajectory_carrier_id": SIGNED_BALANCED_AC_CARRIER_ID,
        "state_space_control_evaluation_status": (
            "reserved_until_carrier_gate_passes"
        ),
        "no_attack_only": True,
        "generation_step_count": 8,
        "replay_step_count": 20,
        "prompt_limit": 2,
        "seed_limit": 2,
        "phase_bin_count": 8,
        "smoke_generation_record_count": 12,
        "required_source_generation_record_count": 16,
        "required_source_summary_record_count": 96,
        "required_source_pair_record_count": 84,
        "required_source_failure_record_count": 0,
        "trajectory_carrier_variant_ids": [
            SIGNED_VARIANT,
            NONNEGATIVE_VARIANT,
            CLEAN_VARIANT,
        ],
    }
    mismatches = [
        name for name, expected in exact.items() if config.get(name) != expected
    ]
    if mismatches:
        raise ValueError(
            "minimal signed trajectory 配置字段未冻结: "
            + ", ".join(mismatches)
        )
    for name, expected in (
        ("lambda_max", 0.12),
        ("ac_allocation", 0.75),
        ("dc_allocation", 0.25),
        ("minimum_ac_direction_retained_cosine", 0.25),
        ("minimum_signed_correct_over_wrong_trajectory_fraction", 0.75),
        ("minimum_signed_over_nonnegative_path_margin_fraction", 0.75),
        ("minimum_replay_reliability", 0.05),
        ("endpoint_reference_default_fraction", 0.75),
        ("maximum_endpoint_fraction_drop_from_reference", 0.25),
    ):
        if not math.isclose(
            float(config.get(name) or 0.0),
            expected,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"minimal signed trajectory 数值字段未冻结: {name}"
            )


def validate_controlled_embedding_source_result(
    input_root: str | Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """只接受真实完成且结论为“不再提高 lambda”的 controlled 结果。"""

    root = Path(input_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"controlled embedding result root 不存在: {root}")
    paths = {
        "source_decision_path": _find_unique_file(
            root,
            SOURCE_DECISION_SUFFIX,
        ),
        "source_manifest_path": _find_unique_file(
            root,
            SOURCE_MANIFEST_SUFFIX,
        ),
        "generation_records_path": _find_unique_file(
            root,
            GENERATION_RECORDS_SUFFIX,
        ),
        "source_summary_path": _find_unique_file(
            root,
            SOURCE_SUMMARY_SUFFIX,
        ),
        "source_pair_path": _find_unique_file(root, SOURCE_PAIR_SUFFIX),
        "source_failure_path": _find_unique_file(root, SOURCE_FAILURE_SUFFIX),
        "prompt_suite_path": _find_unique_file(root, PROMPT_SUITE_SUFFIX),
        "likelihood_path": _find_unique_file(root, LIKELIHOOD_SUFFIX),
    }
    decision = _read_json(paths["source_decision_path"])
    manifest = _read_json(paths["source_manifest_path"])
    generation_rows = _read_jsonl(paths["generation_records_path"])
    summary_rows = _read_jsonl(paths["source_summary_path"])
    pair_rows = _read_jsonl(paths["source_pair_path"])
    failure_rows = _read_jsonl(paths["source_failure_path"])
    required_decision = {
        "profile_id": config["required_source_profile_id"],
        "controlled_embedding_strength_diagnostic_decision": config[
            "required_source_controlled_embedding_decision"
        ],
        "generation_record_count": 16,
        "generation_success_count": 16,
        "summary_record_count": 96,
        "pair_record_count": 84,
        "failure_record_count": 0,
        "lambda_increase_path_signal_repair_observed": False,
        "attacked_phase_executed": False,
        "attacked_phase_execution_allowed": False,
        "fixed_fpr_evaluation_executed": False,
        "external_baseline_execution_executed": False,
        "stage_progression_allowed": False,
        "formal_result": False,
    }
    mismatches = [
        name
        for name, expected in required_decision.items()
        if decision.get(name) != expected
    ]
    if mismatches:
        raise ValueError(
            "controlled embedding source decision 不允许 signed smoke: "
            + ", ".join(mismatches)
        )
    if (
        manifest.get("profile_id")
        != config["required_source_profile_id"]
        or manifest.get("formal_result") is not False
        or manifest.get("stage_progression_allowed") is not False
    ):
        raise ValueError("controlled embedding source manifest 边界不匹配")
    expected_counts = (
        (generation_rows, 16, "generation"),
        (summary_rows, 96, "summary"),
        (pair_rows, 84, "pair"),
        (failure_rows, 0, "failure"),
    )
    for rows, expected, label in expected_counts:
        if len(rows) != expected:
            raise ValueError(
                f"controlled embedding source {label} count 不匹配"
            )
    if any(row.get("generation_status") != "success" for row in generation_rows):
        raise ValueError("controlled embedding source 包含失败 generation")
    model_ids = {
        str(row.get("generation_model_id") or "")
        for row in generation_rows
    }
    if model_ids != {config["required_generation_model_id"]}:
        raise ValueError("controlled embedding source generation model 不匹配")

    reference = decision.get("strength_grid_diagnostics", {}).get(
        "reference_default:20"
    )
    if not isinstance(reference, dict):
        raise ValueError("controlled embedding source 缺少 reference_default:20")
    expected_reference = {
        "coverage_ready": True,
        "correct_over_wrong_endpoint_fraction": 0.75,
        "correct_over_wrong_path_fraction": 0.25,
        "correct_over_wrong_trajectory_fraction": 0.25,
        "lambda_max": 0.12,
        "replay_grid_step_count": 20,
    }
    if any(reference.get(name) != expected for name, expected in expected_reference.items()):
        raise ValueError(
            "controlled embedding source reference default 与已审核结果不匹配"
        )
    reference_rows = [
        row
        for row in generation_rows
        if row.get("embedding_strength_level_id") == "reference_default"
        and row.get("method_variant") == "sstw_full_method"
    ]
    identities = {
        (str(row["prompt_id"]), str(row["seed_id"])) for row in reference_rows
    }
    if len(reference_rows) != 4 or len(identities) != 4:
        raise ValueError(
            "controlled embedding source 必须提供2 prompts x 2 seeds reference"
        )
    return {
        "input_root": str(root),
        **{name: str(path) for name, path in paths.items()},
        "source_decision": decision,
        "source_manifest": manifest,
        "generation_rows": generation_rows,
        "summary_rows": summary_rows,
        "pair_rows": pair_rows,
        "reference_generation_rows": reference_rows,
        "prompt_suite": _read_json(paths["prompt_suite_path"]),
    }


def build_minimal_signed_trajectory_generation_plan(
    validated: Mapping[str, Any],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    prompts = {
        str(row["prompt_id"]): dict(row)
        for row in validated["prompt_suite"].get("prompts") or []
    }
    seeds = {
        str(row["seed_id"]): dict(row)
        for row in validated["prompt_suite"].get("seeds") or []
    }
    variants = (
        (
            SIGNED_VARIANT,
            "sstw_signed_balanced_ac",
            "controlled_embedding_positive_source",
        ),
        (
            NONNEGATIVE_VARIANT,
            "sstw_nonnegative_phase_control",
            "controlled_embedding_positive_source",
        ),
        (
            CLEAN_VARIANT,
            "sstw_clean_unwatermarked_reference",
            "clean_negative",
        ),
    )
    plan: list[dict[str, Any]] = []
    for source in sorted(
        validated["reference_generation_rows"],
        key=lambda row: (str(row["prompt_id"]), str(row["seed_id"])),
    ):
        prompt = prompts[str(source["prompt_id"])]
        seed = seeds[str(source["seed_id"])]
        for variant_id, method_variant, sample_role in variants:
            identity = {
                "generation_model_id": config["required_generation_model_id"],
                "prompt_id": source["prompt_id"],
                "seed_id": source["seed_id"],
                "trajectory_carrier_variant_id": variant_id,
            }
            plan.append(
                {
                    **prompt,
                    **seed,
                    **identity,
                    "signed_trajectory_plan_record_id": _stable_digest(identity),
                    "seed_value": int(source["generation_seed_random"]),
                    "generation_seed_random": int(
                        source["generation_seed_random"]
                    ),
                    "cross_model_role": "main_generation_model",
                    "sample_role": sample_role,
                    "generation_sample_role": sample_role,
                    "method_variant": method_variant,
                    "watermark_embedding_status": (
                        "clean_unwatermarked_reference"
                        if variant_id == CLEAN_VARIANT
                        else "flow_scheduler_velocity_constraint"
                    ),
                    "lambda_max": 0.12,
                    "signed_trajectory_ac_allocation": 0.75,
                    "signed_trajectory_dc_allocation": 0.25,
                    "signed_trajectory_minimum_ac_direction_retained_cosine": (
                        0.25
                    ),
                    "prompt_suite_role": source["prompt_suite_role"],
                    "seed_suite_role": source["seed_suite_role"],
                    "formal_method_variant_execution": False,
                    "generation_execution_allowed": True,
                    "attacked_phase_execution_allowed": False,
                    "stage_progression_allowed": False,
                    "claim_support_status": config["claim_support_status"],
                }
            )
    if len(plan) != 12:
        raise RuntimeError("minimal signed trajectory generation plan 必须为12条")
    return plan


def _likelihood_config(path: str | Path) -> ReplayGaussianLikelihoodConfig:
    rows = _read_jsonl(path)
    if len(rows) != 1:
        raise ValueError("minimal signed trajectory 要求唯一 likelihood calibration")
    row = rows[0]
    return ReplayGaussianLikelihoodConfig(
        relative_observation_noise_standard_deviation=float(
            row["replay_relative_observation_noise_standard_deviation"]
        ),
        minimum_observation_noise_variance=float(
            row["replay_minimum_observation_noise_variance"]
        ),
        likelihood_model_id=str(row["replay_likelihood_model_id"]),
        calibration_protocol=str(row["replay_likelihood_calibration_protocol"]),
        calibration_cluster_count=int(
            row["replay_likelihood_calibration_cluster_count"]
        ),
    )


def _fraction(values: Iterable[bool]) -> float:
    rows = list(values)
    return sum(bool(value) for value in rows) / len(rows) if rows else 0.0


def _summary_identity(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("prompt_id") or ""),
        str(row.get("seed_id") or ""),
        str(row.get("trajectory_carrier_variant_id") or ""),
    )


def build_signed_trajectory_pair_records(
    summaries: Iterable[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in summaries]
    by_key = {
        (*_summary_identity(row), str(row["candidate_key_role"])): row
        for row in rows
    }
    pairs: list[dict[str, Any]] = []
    margins: dict[tuple[str, str, str], dict[str, Any]] = {}
    identities = sorted(
        {
            (str(row["prompt_id"]), str(row["seed_id"]))
            for row in rows
        }
    )
    for prompt_id, seed_id in identities:
        for variant in config["trajectory_carrier_variant_ids"]:
            correct = by_key.get(
                (prompt_id, seed_id, variant, "correct_owner_key")
            )
            wrong = by_key.get(
                (prompt_id, seed_id, variant, "wrong_owner_key")
            )
            if correct is None or wrong is None:
                continue
            base = {
                "record_version": RECORD_VERSION,
                "profile_id": PROFILE_ID,
                "prompt_id": prompt_id,
                "seed_id": seed_id,
                "trajectory_carrier_variant_id": variant,
                "comparison_kind": "correct_owner_key_over_wrong_owner_key",
                "claim_support_status": config["claim_support_status"],
            }
            pair = {
                **base,
                "signed_trajectory_pair_record_id": _stable_digest(base),
                "correct_over_wrong_path_margin": (
                    float(correct["trajectory_path_projection"])
                    - float(wrong["trajectory_path_projection"])
                ),
                "correct_over_wrong_trajectory_static_margin": (
                    float(correct["trajectory_static_aggregation_score"])
                    - float(wrong["trajectory_static_aggregation_score"])
                ),
                "correct_over_wrong_endpoint_margin": (
                    float(correct["endpoint_score"])
                    - float(wrong["endpoint_score"])
                ),
                "minimum_pair_reliability": min(
                    float(correct["trajectory_global_reliability"]),
                    float(wrong["trajectory_global_reliability"]),
                ),
            }
            pairs.append(pair)
            margins[(prompt_id, seed_id, variant)] = pair
    for prompt_id, seed_id in identities:
        signed = margins.get((prompt_id, seed_id, SIGNED_VARIANT))
        control = margins.get((prompt_id, seed_id, NONNEGATIVE_VARIANT))
        if signed is None or control is None:
            continue
        base = {
            "record_version": RECORD_VERSION,
            "profile_id": PROFILE_ID,
            "prompt_id": prompt_id,
            "seed_id": seed_id,
            "trajectory_carrier_variant_id": SIGNED_VARIANT,
            "control_trajectory_carrier_variant_id": NONNEGATIVE_VARIANT,
            "comparison_kind": "signed_over_nonnegative_path_margin",
            "claim_support_status": config["claim_support_status"],
        }
        pairs.append(
            {
                **base,
                "signed_trajectory_pair_record_id": _stable_digest(base),
                "signed_over_nonnegative_path_margin_gain": (
                    float(signed["correct_over_wrong_path_margin"])
                    - float(control["correct_over_wrong_path_margin"])
                ),
            }
        )
    return pairs


def build_signed_trajectory_decision(
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
        == "correct_owner_key_over_wrong_owner_key"
        and row.get("trajectory_carrier_variant_id") == SIGNED_VARIANT
    ]
    gains = [
        row
        for row in pair_rows
        if row.get("comparison_kind")
        == "signed_over_nonnegative_path_margin"
    ]
    path_fraction = _fraction(
        float(row["correct_over_wrong_trajectory_static_margin"]) > 0.0
        for row in signed_pairs
    )
    gain_fraction = _fraction(
        float(row["signed_over_nonnegative_path_margin_gain"]) > 0.0
        for row in gains
    )
    endpoint_fraction = _fraction(
        float(row["correct_over_wrong_endpoint_margin"]) > 0.0
        for row in signed_pairs
    )
    reliability = min(
        (float(row["minimum_pair_reliability"]) for row in signed_pairs),
        default=0.0,
    )
    endpoint_minimum = (
        float(config["endpoint_reference_default_fraction"])
        - float(config["maximum_endpoint_fraction_drop_from_reference"])
    )
    coverage_ready = (
        len(summary_rows) == 24
        and len(signed_pairs) == 4
        and len(gains) == 4
        and not failure_rows
    )
    gate_ready = bool(
        coverage_ready
        and path_fraction
        >= float(
            config[
                "minimum_signed_correct_over_wrong_trajectory_fraction"
            ]
        )
        and gain_fraction
        >= float(
            config[
                "minimum_signed_over_nonnegative_path_margin_fraction"
            ]
        )
        and reliability >= float(config["minimum_replay_reliability"])
        and endpoint_fraction >= endpoint_minimum
    )
    if failure_rows:
        classification = "runtime_or_input_failure_stop"
    elif gate_ready:
        classification = (
            "signed_trajectory_carrier_gate_passed_calibration_design_allowed"
        )
    else:
        classification = "signed_trajectory_carrier_gate_failed_stop_claim"
    return {
        "record_version": RECORD_VERSION,
        "profile_id": PROFILE_ID,
        "minimal_signed_trajectory_smoke_decision": classification,
        "signed_trajectory_carrier_gate_ready": gate_ready,
        "coverage_ready": coverage_ready,
        "signed_correct_over_wrong_trajectory_fraction": path_fraction,
        "signed_over_nonnegative_path_margin_fraction": gain_fraction,
        "minimum_replay_reliability": reliability,
        "signed_correct_over_wrong_endpoint_fraction": endpoint_fraction,
        "endpoint_reference_default_fraction": config[
            "endpoint_reference_default_fraction"
        ],
        "minimum_allowed_endpoint_fraction": endpoint_minimum,
        "summary_record_count": len(summary_rows),
        "pair_record_count": len(pair_rows),
        "failure_record_count": len(failure_rows),
        "endpoint_only_control_status": "measured_as_endpoint_key_margin",
        "trajectory_static_aggregation_status": (
            "measured_from_signed_step_observation_sequence"
        ),
        "sstw_state_posterior_status": (
            "reserved_not_fit_until_carrier_gate_passes_and_independent_identities_exist"
        ),
        "attacked_phase_executed": False,
        "attacked_phase_execution_allowed": False,
        "fixed_fpr_evaluation_executed": False,
        "external_baseline_execution_executed": False,
        "stage_progression_allowed": False,
        "formal_result": False,
        "claim_support_status": config["claim_support_status"],
    }


def _run_generation(
    validated: Mapping[str, Any],
    output_root: Path,
    plan: list[dict[str, Any]],
    *,
    pipeline_cache: dict[str, Any],
) -> dict[str, Any]:
    velocity = VelocityFieldConstraintConfig()
    velocity_configs = {
        str(row["signed_trajectory_plan_record_id"]): velocity for row in plan
    }
    carrier = SignedTrajectoryCarrierConfig()
    carriers = {
        str(row["signed_trajectory_plan_record_id"]): (
            carrier
            if row["trajectory_carrier_variant_id"] == SIGNED_VARIANT
            else None
        )
        for row in plan
    }
    return run_colab_probe(
        output_root,
        validated["prompt_suite_path"],
        MINIMAL_SIGNED_TRAJECTORY_STATE_SPACE_SMOKE_PROFILE,
        model_id=WAN21_PRIMARY_MODEL_ID,
        cross_model_id=None,
        generation_plan_override=plan,
        velocity_config_by_plan_record_id=velocity_configs,
        signed_trajectory_carrier_by_plan_record_id=carriers,
        pipeline_cache=pipeline_cache,
    )


def _execute_no_attack_replay(
    output_root: Path,
    config: Mapping[str, Any],
    *,
    likelihood: ReplayGaussianLikelihoodConfig,
    pipeline_loader: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
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
            raise RuntimeError("minimal signed smoke 同一模型混用 revision")
    pipelines = {
        model_id: _invoke_pipeline_loader(
            pipeline_loader,
            model_id=model_id,
            revision=revision,
        )
        for model_id, revision in revisions.items()
    }
    summaries: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    carrier = SignedTrajectoryCarrierConfig()
    for source in generation_rows:
        base = {
            "record_version": RECORD_VERSION,
            "profile_id": PROFILE_ID,
            "generation_model_id": source["generation_model_id"],
            "prompt_id": source["prompt_id"],
            "seed_id": source["seed_id"],
            "trajectory_trace_id": source["trajectory_trace_id"],
            "signed_trajectory_plan_record_id": source[
                "signed_trajectory_plan_record_id"
            ],
            "trajectory_carrier_variant_id": source[
                "trajectory_carrier_variant_id"
            ],
            "method_variant": source["method_variant"],
            "video_condition_id": "no_attack_minimal_signed_trajectory",
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
            wrong_key = _wrong_owner_generation_key(source)
            video_path = Path(str(source["video_path"]))
            if not video_path.is_file():
                raise FileNotFoundError(f"生成视频不存在: {video_path}")
            active_carrier = (
                carrier
                if source["trajectory_carrier_variant_id"]
                == SIGNED_VARIANT
                else None
            )
            replay = _run_attacked_video_replay_for_model(
                pipeline,
                video_path,
                prompt=prompt,
                key_text=correct_key,
                key_context=key_context,
                likelihood_config=likelihood,
                replay_step_counts=(20,),
                signed_trajectory_carrier_config=active_carrier,
            )
            correct_trajectory = replay.replay_trajectories[0]
            correct_path = replay.path_evidence
            correct_endpoint = _compute_replay_endpoint_evidence_for_key(
                replay,
                key_text=correct_key,
                key_context=key_context,
            )
            wrong_trajectory, wrong_path = (
                _evaluate_fixed_replay_hypothesis_for_key(
                    pipeline,
                    replay,
                    prompt=prompt,
                    key_text=wrong_key,
                    key_context=key_context,
                    signed_trajectory_carrier_config=active_carrier,
                )
            )
            wrong_endpoint = _compute_replay_endpoint_evidence_for_key(
                replay,
                key_text=wrong_key,
                key_context=key_context,
            )
            for key_role, key_text, endpoint, trajectory, path in (
                (
                    "correct_owner_key",
                    correct_key,
                    correct_endpoint,
                    correct_trajectory,
                    correct_path,
                ),
                (
                    "wrong_owner_key",
                    wrong_key,
                    wrong_endpoint,
                    wrong_trajectory,
                    wrong_path,
                ),
            ):
                observations = build_flow_state_observation_sequence(
                    replay,
                    key_text=key_text,
                    trajectory=trajectory,
                    schedule=replay.primary_schedule,
                    key_context=key_context,
                    signed_trajectory_carrier_config=active_carrier,
                )
                typed: list[FlowEvidenceObservation] = (
                    flow_evidence_observation_sequence_from_mappings(
                        observations
                    )
                )
                if len(typed) != 20:
                    raise RuntimeError(
                        "minimal signed smoke 必须形成20步 FlowEvidenceObservation"
                    )
                summary_base = {
                    **base,
                    "candidate_key_role": key_role,
                }
                summary = {
                    **summary_base,
                    "minimal_signed_trajectory_summary_record_id": (
                        _stable_digest(summary_base)
                    ),
                    "trajectory_path_projection": path["S_path_inv"],
                    "trajectory_velocity_projection": path["S_velocity"],
                    "trajectory_static_aggregation_score": mean(
                        float(row["path_score"]) for row in observations
                    ),
                    "replay_log_likelihood_ratio": (
                        trajectory.replay_log_likelihood_ratio
                    ),
                    "trajectory_global_reliability": (
                        replay.replay_uncertainty.replay_reliability
                    ),
                    "flow_evidence_observation_count": len(typed),
                    "flow_evidence_observation_contract_status": "ready",
                    "sstw_state_posterior_evaluation_status": (
                        "reserved_until_carrier_gate_passes"
                    ),
                    **replay.endpoint_metadata,
                    **endpoint.as_dict(),
                    "metric_status": (
                        "measured_minimal_signed_trajectory_smoke"
                    ),
                }
                summaries.append(summary)
                for observation in observations:
                    index = int(
                        observation["flow_state_observation_step_index"]
                    )
                    step_base = {
                        **summary_base,
                        "trajectory_step_index": index,
                    }
                    steps.append(
                        {
                            **step_base,
                            "minimal_signed_trajectory_step_record_id": (
                                _stable_digest(step_base)
                            ),
                            **observation,
                        }
                    )
        except Exception as exc:  # pragma: no cover - real Colab GPU path
            failures.append(
                {
                    **base,
                    "minimal_signed_trajectory_smoke_status": "failed",
                    "minimal_signed_trajectory_smoke_failure_reason": str(exc),
                }
            )
    return summaries, steps, failures


def _write_report(path: Path, decision: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# Minimal signed trajectory state-space smoke",
                "",
                f"- Decision: `{decision['minimal_signed_trajectory_smoke_decision']}`",
                f"- Carrier gate ready: `{decision['signed_trajectory_carrier_gate_ready']}`",
                f"- Signed correct-over-wrong trajectory fraction: `{decision['signed_correct_over_wrong_trajectory_fraction']}`",
                f"- Signed-over-control path fraction: `{decision['signed_over_nonnegative_path_margin_fraction']}`",
                f"- Minimum replay reliability: `{decision['minimum_replay_reliability']}`",
                f"- Endpoint fraction: `{decision['signed_correct_over_wrong_endpoint_fraction']}`",
                "",
                "该结果只判断时间载体可辨识性，不是 posterior、fixed-FPR 或论文证据。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def run_minimal_signed_trajectory_state_space_smoke(
    input_root: str | Path,
    output_root: str | Path,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    generation_runner: Any = _run_generation,
    pipeline_loader: Any = _load_pipeline,
) -> dict[str, Any]:
    config = _read_json(config_path)
    validate_minimal_signed_trajectory_config(config)
    validated = validate_controlled_embedding_source_result(
        input_root,
        config,
    )
    output = Path(output_root).resolve()
    if not output.is_dir() or any(output.iterdir()):
        raise FileExistsError(
            "minimal signed trajectory output root 必须是已创建的空目录"
        )
    plan = build_minimal_signed_trajectory_generation_plan(validated, config)
    write_jsonl(
        output / "records" / "minimal_signed_trajectory_generation_plan.jsonl",
        plan,
    )
    write_json(
        output / "artifacts" / "minimal_signed_trajectory_execution_decision.json",
        {
            "record_version": RECORD_VERSION,
            "profile_id": PROFILE_ID,
            "execution_preflight_status": "ready",
            "generation_execution_allowed": True,
            "generation_record_count": 12,
            "lambda_max": 0.12,
            "replay_step_count": 20,
            "no_attack_only": True,
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

    def cached_loader(model_id: str, *, revision: str | None = None) -> Any:
        del revision
        return pipeline_cache[model_id]

    summaries, steps, failures = _execute_no_attack_replay(
        output,
        config,
        likelihood=_likelihood_config(output / LIKELIHOOD_SUFFIX),
        pipeline_loader=(
            cached_loader if pipeline_cache else pipeline_loader
        ),
    )
    pairs = build_signed_trajectory_pair_records(summaries, config)
    decision = build_signed_trajectory_decision(
        summaries,
        pairs,
        failures,
        config,
    )
    write_jsonl(
        output / "records" / "minimal_signed_trajectory_summary_records.jsonl",
        summaries,
    )
    write_jsonl(
        output / "records" / "minimal_signed_trajectory_step_records.jsonl",
        steps,
    )
    write_jsonl(
        output / "records" / "minimal_signed_trajectory_pair_records.jsonl",
        pairs,
    )
    write_jsonl(
        output / "records" / "minimal_signed_trajectory_failure_records.jsonl",
        failures,
    )
    write_json(
        output / "artifacts" / "minimal_signed_trajectory_smoke_decision.json",
        decision,
    )
    _write_report(
        output / "reports" / "minimal_signed_trajectory_smoke_report.md",
        decision,
    )
    write_json(
        output / "artifacts" / "minimal_signed_trajectory_smoke_manifest.json",
        {
            "artifact_id": "minimal_signed_trajectory_smoke_manifest",
            "artifact_type": "manifest",
            "profile_id": PROFILE_ID,
            "generation_result": generation_result,
            "generation_record_count": 12,
            "summary_record_count": len(summaries),
            "step_record_count": len(steps),
            "pair_record_count": len(pairs),
            "failure_record_count": len(failures),
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
    decision = run_minimal_signed_trajectory_state_space_smoke(
        args.input_root,
        args.output_run_root,
        args.config_path,
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
