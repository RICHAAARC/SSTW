"""固定空间 basis 的 temporal-code 多候选 replay-only smoke。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping, Sequence

from evaluation.protocol.record_writer import write_json, write_jsonl
from experiments.generative_video_model_probe.colab_runtime import (
    WAN21_PRIMARY_MODEL_ID,
    validate_generation_model_provenance,
)
from experiments.generative_video_model_probe.formal_flow_evidence_runner import (
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
from experiments.generative_video_model_probe.predictive_trajectory_synchronization_smoke import (
    NONNEGATIVE_VARIANT,
    PATH_RELIABILITY_MODE,
    PREDICTIVE_VARIANT,
    PRIMARY_PATH_STATISTIC,
    PROFILE_ID as PREDICTIVE_SOURCE_PROFILE_ID,
    _predictive_summary_path_ready,
    _reuse_predictive_generation_for_replay_only,
    _validated_predictive_path_evidence,
    build_predictive_trajectory_generation_plan,
    validate_predictive_generation_execution,
)
from main.methods.state_space_watermark.flow_tubelet_key_code import (
    FlowTubeletKeyCodeConfig,
)
from main.methods.state_space_watermark.predictive_trajectory_carrier import (
    PREDICTIVE_TRAJECTORY_CARRIER_ID,
    PREDICTIVE_TRAJECTORY_MAXIMUM_ABSOLUTE_CODE_CORRELATION,
    PredictiveTrajectoryCarrierConfig,
    predictive_trajectory_weighted_code_correlation,
)
from main.methods.state_space_watermark.replay_inversion import (
    ReplayGaussianLikelihoodConfig,
)
from main.methods.state_space_watermark.wan_flow_replay_backend import (
    predictive_schedule_for_replay,
    score_replay_trajectory_for_key,
)


DEFAULT_CONFIG_PATH = (
    "configs/protocol/sstw_temporal_code_isolation_replay_smoke.json"
)
PROFILE_ID = "sstw_temporal_code_isolation_replay_smoke"
RECORD_VERSION = "temporal_code_isolation_replay_smoke"
TEST_ID = "temporal_code_isolation_replay_smoke"
OWNER_TEMPORAL_ROLE = "owner_temporal_code"
WRONG_TEMPORAL_ROLE = "wrong_temporal_code"
TEMPORAL_PAIR_COMPARISON = "owner_temporal_code_over_wrong_temporal_code"
TEMPORAL_WRONG_CANDIDATE_COUNT = 8
TEMPORAL_WRONG_CANDIDATE_POOL_COUNT = 32


def validate_temporal_code_isolation_config(
    config: Mapping[str, Any],
) -> None:
    """冻结失败后的时间码隔离诊断，不允许扩大为新方法网格。"""

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
            "temporal code isolation 禁止项未冻结: " + ", ".join(invalid)
        )
    exact = {
        "profile_id": PROFILE_ID,
        "paper_result_level": "temporal_code_isolation_replay_smoke",
        "claim_support_status": (
            "temporal_code_isolation_replay_smoke_only_not_paper_evidence"
        ),
        "required_source_profile_id": (
            "sstw_controlled_embedding_strength_diagnostic"
        ),
        "required_source_controlled_embedding_decision": (
            "lambda_increase_did_not_repair_path_signal_stop"
        ),
        "required_predictive_source_profile_id": (
            PREDICTIVE_SOURCE_PROFILE_ID
        ),
        "required_predictive_source_decision": (
            "predictive_trajectory_gate_failed_stop_method"
        ),
        "required_generation_model_id": WAN21_PRIMARY_MODEL_ID,
        "no_attack_only": True,
        "replay_step_count": 20,
        "source_generation_record_count": 8,
        "replay_generation_record_count": 4,
        "required_predictive_source_summary_record_count": 24,
        "required_predictive_source_pair_record_count": 20,
        "temporal_wrong_candidate_count": TEMPORAL_WRONG_CANDIDATE_COUNT,
        "temporal_wrong_candidate_pool_count": (
            TEMPORAL_WRONG_CANDIDATE_POOL_COUNT
        ),
        "summary_record_count": 36,
        "pair_record_count": 32,
        "identity_record_count": 4,
        "primary_predictive_path_statistic": PRIMARY_PATH_STATISTIC,
        "predictive_path_reliability_mode": PATH_RELIABILITY_MODE,
        "predictive_trajectory_carrier_id": (
            PREDICTIVE_TRAJECTORY_CARRIER_ID
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
            PREDICTIVE_VARIANT,
            NONNEGATIVE_VARIANT,
        ],
    }
    mismatches = [
        name for name, expected in exact.items() if config.get(name) != expected
    ]
    if mismatches:
        raise ValueError(
            "temporal code isolation 配置字段未冻结: "
            + ", ".join(mismatches)
        )
    numeric = (
        ("lambda_max", 0.12),
        (
            "maximum_absolute_code_correlation",
            PREDICTIVE_TRAJECTORY_MAXIMUM_ABSOLUTE_CODE_CORRELATION,
        ),
        ("minimum_temporal_owner_over_wrong_pair_fraction", 0.75),
        ("minimum_temporal_owner_top1_identity_fraction", 0.75),
        ("maximum_prompt_temporal_pair_fraction_gap", 0.25),
        ("minimum_replay_reliability", 0.05),
    )
    allowed = (
        set(false_fields)
        | set(exact)
        | {name for name, _expected in numeric}
    )
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise ValueError(
            "temporal code isolation 配置包含未声明字段: "
            + ", ".join(unknown)
        )
    for name, expected in numeric:
        if not math.isclose(
            float(config.get(name) or 0.0),
            float(expected),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"temporal code isolation 数值字段未冻结: {name}"
            )


def _find_unique_predictive_file(root: Path, relative: str) -> Path:
    matches = [
        path
        for path in root.rglob(relative)
        if path.is_file() and not path.is_symlink()
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "temporal code isolation 输入必须唯一包含 "
            f"{relative}; observed={matches}"
        )
    return matches[0]


def validate_predictive_replay_source_result(
    replay_source_root: str | Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """只接受修复后 S_path_inv 失败包作为独立隔离诊断输入。"""

    root = Path(replay_source_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"predictive replay source root 不存在: {root}"
        )
    decision_path = _find_unique_predictive_file(
        root,
        "artifacts/predictive_trajectory_smoke_decision.json",
    )
    manifest_path = _find_unique_predictive_file(
        root,
        "artifacts/predictive_trajectory_smoke_manifest.json",
    )
    summary_path = _find_unique_predictive_file(
        root,
        "records/predictive_trajectory_summary_records.jsonl",
    )
    pair_path = _find_unique_predictive_file(
        root,
        "records/predictive_trajectory_pair_records.jsonl",
    )
    failure_path = _find_unique_predictive_file(
        root,
        "records/predictive_trajectory_failure_records.jsonl",
    )
    decision = _read_json(decision_path)
    manifest = _read_json(manifest_path)
    summary_rows = _read_jsonl(summary_path)
    pair_rows = _read_jsonl(pair_path)
    failure_rows = _read_jsonl(failure_path)
    required_decision = {
        "profile_id": config["required_predictive_source_profile_id"],
        "predictive_trajectory_smoke_decision": config[
            "required_predictive_source_decision"
        ],
        "coverage_ready": True,
        "predictive_path_evidence_ready": True,
        "predictive_code_separation_ready": True,
        "summary_record_count": config[
            "required_predictive_source_summary_record_count"
        ],
        "pair_record_count": config[
            "required_predictive_source_pair_record_count"
        ],
        "failure_record_count": 0,
        "primary_predictive_path_statistic": PRIMARY_PATH_STATISTIC,
        "predictive_path_reliability_mode": PATH_RELIABILITY_MODE,
        "predictive_endpoint_llr_role": "diagnostic_only_not_gate",
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
            "predictive replay source decision 不允许 temporal isolation: "
            + ", ".join(mismatches)
        )
    generation_result = manifest.get("generation_result")
    if not isinstance(generation_result, Mapping):
        raise ValueError("predictive replay source manifest 缺 generation result")
    required_manifest = {
        "profile_id": config["required_predictive_source_profile_id"],
        "generation_record_count": 8,
        "summary_record_count": 24,
        "pair_record_count": 20,
        "failure_record_count": 0,
        "primary_predictive_path_statistic": PRIMARY_PATH_STATISTIC,
        "predictive_path_reliability_mode": PATH_RELIABILITY_MODE,
        "stage_progression_allowed": False,
        "formal_result": False,
    }
    manifest_mismatches = [
        name
        for name, expected in required_manifest.items()
        if manifest.get(name) != expected
    ]
    if (
        manifest_mismatches
        or generation_result.get("generation_reused_for_replay_only")
        is not True
    ):
        raise ValueError(
            "predictive replay source manifest 边界不匹配: "
            + ", ".join(manifest_mismatches)
        )
    if len(summary_rows) != 24 or len(pair_rows) != 20 or failure_rows:
        raise ValueError("predictive replay source records count 不匹配")
    return {
        "input_root": str(root),
        "decision_path": str(decision_path),
        "manifest_path": str(manifest_path),
        "decision": decision,
        "manifest": manifest,
    }


def _candidate_code_signature(schedule: Any) -> str:
    return _stable_digest(
        {
            "raw_signs": list(schedule.raw_signs),
            "codes": [format(float(value), ".17g") for value in schedule.codes],
        }
    )


def select_temporal_wrong_candidates(
    source: Mapping[str, Any],
    replay_schedule: Sequence[Any],
    *,
    correct_key: str,
    key_context: Any,
    config: Mapping[str, Any],
    tubelet_config: FlowTubeletKeyCodeConfig,
) -> tuple[Any, list[dict[str, Any]]]:
    """从固定32候选池选前8个低相关且时间码序列互异的候选。"""

    owner_schedule = predictive_schedule_for_replay(
        replay_schedule,
        key_text=correct_key,
        key_context=key_context,
        tubelet_config=tubelet_config,
    )
    owner_signature = _candidate_code_signature(owner_schedule)
    seen = {owner_signature}
    selected: list[dict[str, Any]] = []
    for candidate_index in range(
        int(config["temporal_wrong_candidate_pool_count"])
    ):
        candidate_key = _wrong_owner_generation_key(
            source,
            extra_context={
                "temporal_code_isolation_candidate_index": candidate_index
            },
        )
        candidate_schedule = predictive_schedule_for_replay(
            replay_schedule,
            key_text=candidate_key,
            key_context=key_context,
            tubelet_config=tubelet_config,
        )
        signature = _candidate_code_signature(candidate_schedule)
        correlation = predictive_trajectory_weighted_code_correlation(
            owner_schedule,
            candidate_schedule,
        )
        if (
            signature in seen
            or abs(correlation)
            > float(config["maximum_absolute_code_correlation"]) + 1e-12
        ):
            continue
        seen.add(signature)
        selected.append(
            {
                "candidate_index": candidate_index,
                "candidate_key": candidate_key,
                "weighted_code_correlation": correlation,
                "temporal_code_signature_digest": signature,
                "temporal_phase_function_digest": (
                    candidate_schedule.phase_function_digest
                ),
            }
        )
        if len(selected) == int(config["temporal_wrong_candidate_count"]):
            break
    if len(selected) != int(config["temporal_wrong_candidate_count"]):
        raise RuntimeError(
            "temporal code isolation 无法从冻结池构造8个互异低相关时间码"
        )
    return owner_schedule, selected


def _execute_temporal_isolation_replay(
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
        and row.get("trajectory_carrier_variant_id") == PREDICTIVE_VARIANT
    ]
    if len(generation_rows) != int(
        config["replay_generation_record_count"]
    ):
        raise RuntimeError("temporal code isolation 必须正好 replay 4个 signed 视频")
    prompt_map = _prompt_text_by_id(
        _read_json(output_root / PROMPT_SUITE_SUFFIX)
    )
    revisions: dict[str, str] = {}
    for row in generation_rows:
        model_id = str(row["generation_model_id"])
        revision = validate_generation_model_provenance(row)
        previous = revisions.setdefault(model_id, revision)
        if previous != revision:
            raise RuntimeError("temporal code isolation 同一模型混用 revision")
    pipelines = {
        model_id: _invoke_pipeline_loader(
            pipeline_loader,
            model_id=model_id,
            revision=revision,
        )
        for model_id, revision in revisions.items()
    }
    carrier = PredictiveTrajectoryCarrierConfig()
    tubelet = FlowTubeletKeyCodeConfig()
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
            "predictive_trajectory_plan_record_id": source[
                "predictive_trajectory_plan_record_id"
            ],
            "trajectory_carrier_variant_id": PREDICTIVE_VARIANT,
            "method_variant": source["method_variant"],
            "video_condition_id": "no_attack_temporal_code_isolation",
            "replay_grid_step_count": 20,
            "claim_support_status": config["claim_support_status"],
        }
        try:
            pipeline = pipelines[str(source["generation_model_id"])]
            prompt = prompt_map[str(source["prompt_id"])]
            key_context = _validated_flow_key_context(
                source,
                prompt=prompt,
                scheduler=pipeline.scheduler,
            )
            correct_key = _generation_key(source)
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
                predictive_trajectory_carrier_config=carrier,
                endpoint_control_enabled=False,
            )
            trajectory = replay.replay_trajectories[0]
            owner_schedule, wrong_candidates = (
                select_temporal_wrong_candidates(
                    source,
                    replay.primary_schedule,
                    correct_key=correct_key,
                    key_context=key_context,
                    config=config,
                    tubelet_config=tubelet,
                )
            )
            shared = {
                "tubelet_config": tubelet,
                "likelihood_config": likelihood,
                "key_context": key_context,
                "predictive_trajectory_carrier_config": carrier,
                "path_reliability_mode": config[
                    "predictive_path_reliability_mode"
                ],
            }
            owner_path = score_replay_trajectory_for_key(
                trajectory,
                replay.primary_schedule,
                key_text=correct_key,
                trajectory_carrier_key_text=correct_key,
                **shared,
            )
            candidates = [
                {
                    "candidate_role": OWNER_TEMPORAL_ROLE,
                    "candidate_index": None,
                    "weighted_code_correlation": 1.0,
                    "temporal_code_signature_digest": (
                        _candidate_code_signature(owner_schedule)
                    ),
                    "temporal_phase_function_digest": (
                        owner_schedule.phase_function_digest
                    ),
                    "path_evidence": owner_path,
                }
            ]
            for candidate in wrong_candidates:
                candidates.append(
                    {
                        **{
                            name: value
                            for name, value in candidate.items()
                            if name != "candidate_key"
                        },
                        "candidate_role": WRONG_TEMPORAL_ROLE,
                        "path_evidence": score_replay_trajectory_for_key(
                            trajectory,
                            replay.primary_schedule,
                            key_text=correct_key,
                            trajectory_carrier_key_text=str(
                                candidate["candidate_key"]
                            ),
                            **shared,
                        ),
                    }
                )
            for candidate in candidates:
                identity = {
                    **base,
                    "temporal_code_candidate_role": candidate[
                        "candidate_role"
                    ],
                    "temporal_code_candidate_index": candidate[
                        "candidate_index"
                    ],
                    "temporal_code_signature_digest": candidate[
                        "temporal_code_signature_digest"
                    ],
                }
                summaries.append(
                    {
                        **identity,
                        "temporal_code_isolation_summary_record_id": (
                            _stable_digest(identity)
                        ),
                        **_validated_predictive_path_evidence(
                            candidate["path_evidence"],
                            config,
                        ),
                        "temporal_code_weighted_correlation_to_owner": (
                            candidate["weighted_code_correlation"]
                        ),
                        "temporal_code_phase_function_digest": candidate[
                            "temporal_phase_function_digest"
                        ],
                        "temporal_code_spatial_key_role": (
                            "owner_key_fixed"
                        ),
                        "trajectory_global_reliability": (
                            replay.replay_uncertainty.replay_reliability
                        ),
                        "metric_status": (
                            "measured_temporal_code_isolation_replay_smoke"
                        ),
                    }
                )
        except Exception as exc:  # pragma: no cover - real Colab GPU path
            failures.append(
                {
                    **base,
                    "temporal_code_isolation_status": "failed",
                    "temporal_code_isolation_failure_reason": str(exc),
                }
            )
    return summaries, failures


def _fraction(values: Iterable[bool]) -> float:
    rows = list(values)
    return sum(bool(value) for value in rows) / len(rows) if rows else 0.0


def build_temporal_isolation_pair_and_identity_records(
    summaries: Iterable[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = [dict(row) for row in summaries]
    identities = sorted(
        {
            (str(row["prompt_id"]), str(row["seed_id"]))
            for row in rows
        }
    )
    pairs: list[dict[str, Any]] = []
    identity_rows: list[dict[str, Any]] = []
    for prompt_id, seed_id in identities:
        current = [
            row
            for row in rows
            if row["prompt_id"] == prompt_id and row["seed_id"] == seed_id
        ]
        owner = [
            row
            for row in current
            if row.get("temporal_code_candidate_role")
            == OWNER_TEMPORAL_ROLE
        ]
        wrong = [
            row
            for row in current
            if row.get("temporal_code_candidate_role")
            == WRONG_TEMPORAL_ROLE
        ]
        if len(owner) != 1 or len(wrong) != int(
            config["temporal_wrong_candidate_count"]
        ):
            continue
        owner_row = owner[0]
        owner_score = float(owner_row["predictive_replay_path_score"])
        margins: list[float] = []
        for wrong_row in sorted(
            wrong,
            key=lambda row: int(row["temporal_code_candidate_index"]),
        ):
            wrong_score = float(wrong_row["predictive_replay_path_score"])
            margin = owner_score - wrong_score
            margins.append(margin)
            base = {
                "record_version": RECORD_VERSION,
                "profile_id": PROFILE_ID,
                "prompt_id": prompt_id,
                "seed_id": seed_id,
                "comparison_kind": TEMPORAL_PAIR_COMPARISON,
                "temporal_code_candidate_index": int(
                    wrong_row["temporal_code_candidate_index"]
                ),
                "temporal_code_signature_digest": wrong_row[
                    "temporal_code_signature_digest"
                ],
                "claim_support_status": config["claim_support_status"],
            }
            pairs.append(
                {
                    **base,
                    "temporal_code_isolation_pair_record_id": (
                        _stable_digest(base)
                    ),
                    "owner_over_wrong_temporal_path_margin": margin,
                    "temporal_code_weighted_correlation_to_owner": wrong_row[
                        "temporal_code_weighted_correlation_to_owner"
                    ],
                    "minimum_pair_reliability": min(
                        float(owner_row["trajectory_global_reliability"]),
                        float(wrong_row["trajectory_global_reliability"]),
                    ),
                }
            )
        identity = {
            "record_version": RECORD_VERSION,
            "profile_id": PROFILE_ID,
            "prompt_id": prompt_id,
            "seed_id": seed_id,
            "claim_support_status": config["claim_support_status"],
        }
        positive_count = sum(value > 0.0 for value in margins)
        identity_rows.append(
            {
                **identity,
                "temporal_code_isolation_identity_record_id": (
                    _stable_digest(identity)
                ),
                "temporal_owner_path_score": owner_score,
                "maximum_wrong_temporal_path_score": max(
                    owner_score - value for value in margins
                ),
                "minimum_owner_over_wrong_temporal_path_margin": min(
                    margins
                ),
                "temporal_owner_rank": (
                    1 + sum(value <= 0.0 for value in margins)
                ),
                "temporal_owner_top1": all(
                    value > 0.0 for value in margins
                ),
                "temporal_owner_percentile": (
                    positive_count / len(margins)
                ),
                "temporal_owner_over_wrong_candidate_fraction": (
                    positive_count / len(margins)
                ),
            }
        )
    return pairs, identity_rows


def build_temporal_isolation_decision(
    summaries: Iterable[Mapping[str, Any]],
    pairs: Iterable[Mapping[str, Any]],
    identities: Iterable[Mapping[str, Any]],
    failures: Iterable[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    summary_rows = [dict(row) for row in summaries]
    pair_rows = [dict(row) for row in pairs]
    identity_rows = [dict(row) for row in identities]
    failure_rows = [dict(row) for row in failures]
    pair_fraction = _fraction(
        float(row["owner_over_wrong_temporal_path_margin"]) > 0.0
        for row in pair_rows
    )
    top1_fraction = _fraction(
        row.get("temporal_owner_top1") is True for row in identity_rows
    )
    prompt_fractions: dict[str, float] = {}
    for prompt_id in config["heldout_prompt_ids"]:
        prompt_rows = [
            row for row in pair_rows if row["prompt_id"] == prompt_id
        ]
        prompt_fractions[prompt_id] = _fraction(
            float(row["owner_over_wrong_temporal_path_margin"]) > 0.0
            for row in prompt_rows
        )
    prompt_gap = (
        max(prompt_fractions.values()) - min(prompt_fractions.values())
        if prompt_fractions
        else 1.0
    )
    reliability = min(
        (float(row["minimum_pair_reliability"]) for row in pair_rows),
        default=0.0,
    )
    path_ready = bool(
        len(summary_rows) == int(config["summary_record_count"])
        and all(
            _predictive_summary_path_ready(row, config)
            for row in summary_rows
        )
    )
    candidate_separation_ready = True
    for identity in identity_rows:
        owner_rows = [
            row
            for row in summary_rows
            if row["prompt_id"] == identity["prompt_id"]
            and row["seed_id"] == identity["seed_id"]
            and row.get("temporal_code_candidate_role")
            == OWNER_TEMPORAL_ROLE
        ]
        matching = [
            row
            for row in summary_rows
            if row["prompt_id"] == identity["prompt_id"]
            and row["seed_id"] == identity["seed_id"]
            and row.get("temporal_code_candidate_role")
            == WRONG_TEMPORAL_ROLE
        ]
        indexes = {
            int(row["temporal_code_candidate_index"]) for row in matching
        }
        signatures = {
            str(row["temporal_code_signature_digest"]) for row in matching
        }
        correlations = [
            float(row["temporal_code_weighted_correlation_to_owner"])
            for row in matching
        ]
        candidate_separation_ready = bool(
            candidate_separation_ready
            and len(owner_rows) == 1
            and len(matching) == int(config["temporal_wrong_candidate_count"])
            and len(indexes) == len(matching)
            and len(signatures) == len(matching)
            and str(owner_rows[0]["temporal_code_signature_digest"])
            not in signatures
            and all(
                math.isfinite(value)
                and abs(value)
                <= float(config["maximum_absolute_code_correlation"]) + 1e-12
                for value in correlations
            )
        )
    expected_identities = {
        (prompt_id, seed_id)
        for prompt_id in config["heldout_prompt_ids"]
        for seed_id in config["heldout_seed_ids"]
    }
    observed_identities = {
        (str(row["prompt_id"]), str(row["seed_id"]))
        for row in identity_rows
    }
    coverage_ready = bool(
        len(summary_rows) == int(config["summary_record_count"])
        and len(pair_rows) == int(config["pair_record_count"])
        and len(identity_rows) == int(config["identity_record_count"])
        and observed_identities == expected_identities
        and len(prompt_fractions) == 2
        and all(
            sum(row["prompt_id"] == prompt_id for row in pair_rows) == 16
            for prompt_id in config["heldout_prompt_ids"]
        )
        and not failure_rows
    )
    gate_ready = bool(
        coverage_ready
        and path_ready
        and candidate_separation_ready
        and pair_fraction
        >= float(config["minimum_temporal_owner_over_wrong_pair_fraction"])
        and top1_fraction
        >= float(config["minimum_temporal_owner_top1_identity_fraction"])
        and prompt_gap
        <= float(config["maximum_prompt_temporal_pair_fraction_gap"])
        and reliability >= float(config["minimum_replay_reliability"])
    )
    if failure_rows:
        classification = "runtime_or_input_failure_stop"
    elif gate_ready:
        classification = (
            "temporal_code_isolation_passed_reduced_generation_design_allowed"
        )
    else:
        classification = "temporal_code_isolation_failed_stop_method"
    return {
        "record_version": RECORD_VERSION,
        "profile_id": PROFILE_ID,
        "temporal_code_isolation_smoke_decision": classification,
        "temporal_code_isolation_gate_ready": gate_ready,
        "coverage_ready": coverage_ready,
        "temporal_path_evidence_ready": path_ready,
        "temporal_candidate_separation_ready": candidate_separation_ready,
        "temporal_owner_over_wrong_pair_fraction": pair_fraction,
        "temporal_owner_top1_identity_fraction": top1_fraction,
        "temporal_owner_over_wrong_pair_fraction_by_prompt": (
            prompt_fractions
        ),
        "temporal_owner_over_wrong_pair_fraction_prompt_gap": prompt_gap,
        "minimum_replay_reliability": reliability,
        "summary_record_count": len(summary_rows),
        "pair_record_count": len(pair_rows),
        "identity_record_count": len(identity_rows),
        "failure_record_count": len(failure_rows),
        "replayed_video_count": 4,
        "spatial_key_fixed_to_owner": True,
        "generation_execution_allowed": False,
        "generation_reuse_for_replay_only": True,
        "primary_predictive_path_statistic": PRIMARY_PATH_STATISTIC,
        "predictive_path_reliability_mode": PATH_RELIABILITY_MODE,
        "endpoint_gate_executed": False,
        "state_space_posterior_executed": False,
        "attacked_phase_executed": False,
        "fixed_fpr_evaluation_executed": False,
        "external_baseline_execution_executed": False,
        "stage_progression_allowed": False,
        "formal_result": False,
        "claim_support_status": config["claim_support_status"],
    }


def run_temporal_code_isolation_replay_smoke(
    input_root: str | Path,
    output_root: str | Path,
    replay_source_root: str | Path,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    pipeline_loader: Any = _load_pipeline,
) -> dict[str, Any]:
    config = _read_json(config_path)
    validate_temporal_code_isolation_config(config)
    validated = validate_controlled_embedding_source_result(
        input_root,
        config,
    )
    predictive_source = validate_predictive_replay_source_result(
        replay_source_root,
        config,
    )
    output = Path(output_root).resolve()
    if not output.is_dir() or any(output.iterdir()):
        raise FileExistsError(
            "temporal code isolation output root 必须是已创建的空目录"
        )
    plan = build_predictive_trajectory_generation_plan(validated, config)
    write_jsonl(
        output / "records" / "predictive_trajectory_generation_plan.jsonl",
        plan,
    )
    execution = {
        "record_version": RECORD_VERSION,
        "profile_id": PROFILE_ID,
        "execution_preflight_status": "ready",
        "generation_execution_allowed": False,
        "generation_reuse_for_replay_only": True,
        "source_generation_record_count": 8,
        "replayed_video_count": 4,
        "replay_step_count": 20,
        "temporal_wrong_candidate_count": 8,
        "temporal_wrong_candidate_pool_count": 32,
        "spatial_key_fixed_to_owner": True,
        "primary_predictive_path_statistic": PRIMARY_PATH_STATISTIC,
        "predictive_path_reliability_mode": PATH_RELIABILITY_MODE,
        "endpoint_gate_execution_allowed": False,
        "state_space_posterior_execution_allowed": False,
        "attacked_phase_execution_allowed": False,
        "fixed_fpr_evaluation_allowed": False,
        "stage_progression_allowed": False,
        "formal_result": False,
        "claim_support_status": config["claim_support_status"],
    }
    write_json(
        output / "artifacts" / "temporal_code_isolation_execution_decision.json",
        execution,
    )
    for suffix, source_path in (
        (PROMPT_SUITE_SUFFIX, validated["prompt_suite_path"]),
        (LIKELIHOOD_SUFFIX, validated["likelihood_path"]),
    ):
        target = output / suffix
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target)
    for label, source_path in (
        ("controlled_source_decision", validated["source_decision_path"]),
        ("controlled_source_manifest", validated["source_manifest_path"]),
        ("predictive_source_decision", predictive_source["decision_path"]),
        ("predictive_source_manifest", predictive_source["manifest_path"]),
    ):
        source = Path(source_path)
        target = output / "inputs" / label / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    generation_result = _reuse_predictive_generation_for_replay_only(
        replay_source_root,
        output,
        plan,
    )
    validate_predictive_generation_execution(
        output,
        plan,
        generation_result,
    )
    summaries, failures = _execute_temporal_isolation_replay(
        output,
        config,
        likelihood=_likelihood_config(output / LIKELIHOOD_SUFFIX),
        pipeline_loader=pipeline_loader,
    )
    pairs, identities = build_temporal_isolation_pair_and_identity_records(
        summaries,
        config,
    )
    decision = build_temporal_isolation_decision(
        summaries,
        pairs,
        identities,
        failures,
        config,
    )
    write_jsonl(
        output / "records" / "temporal_code_isolation_summary_records.jsonl",
        summaries,
    )
    write_jsonl(
        output / "records" / "temporal_code_isolation_pair_records.jsonl",
        pairs,
    )
    write_jsonl(
        output / "records" / "temporal_code_isolation_identity_records.jsonl",
        identities,
    )
    write_jsonl(
        output / "records" / "temporal_code_isolation_failure_records.jsonl",
        failures,
    )
    write_json(
        output / "artifacts" / "temporal_code_isolation_smoke_decision.json",
        decision,
    )
    report = output / "reports" / "temporal_code_isolation_smoke_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "\n".join(
            [
                "# Temporal code isolation replay smoke",
                "",
                f"- Decision: `{decision['temporal_code_isolation_smoke_decision']}`",
                f"- Gate ready: `{decision['temporal_code_isolation_gate_ready']}`",
                f"- Owner-over-wrong pair fraction: `{decision['temporal_owner_over_wrong_pair_fraction']}`",
                f"- Owner top-1 identity fraction: `{decision['temporal_owner_top1_identity_fraction']}`",
                f"- Prompt fraction gap: `{decision['temporal_owner_over_wrong_pair_fraction_prompt_gap']}`",
                f"- Minimum replay reliability: `{decision['minimum_replay_reliability']}`",
                "",
                "该结果只隔离时间状态码；空间 key 固定为 owner，不是论文证据。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    write_json(
        output / "artifacts" / "temporal_code_isolation_smoke_manifest.json",
        {
            "artifact_id": "temporal_code_isolation_smoke_manifest",
            "artifact_type": "manifest",
            "profile_id": PROFILE_ID,
            "generation_result": generation_result,
            "source_generation_record_count": 8,
            "replayed_video_count": 4,
            "summary_record_count": len(summaries),
            "pair_record_count": len(pairs),
            "identity_record_count": len(identities),
            "failure_record_count": len(failures),
            "spatial_key_fixed_to_owner": True,
            "generation_execution_allowed": False,
            "generation_reuse_for_replay_only": True,
            "primary_predictive_path_statistic": PRIMARY_PATH_STATISTIC,
            "predictive_path_reliability_mode": PATH_RELIABILITY_MODE,
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
    parser.add_argument("--replay-source-root", required=True)
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    decision = run_temporal_code_isolation_replay_smoke(
        args.input_root,
        args.output_run_root,
        args.replay_source_root,
        args.config_path,
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
