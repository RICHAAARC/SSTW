"""执行固定 16-record controlled embedding no-attack 强度诊断。

该入口显式消费 construction decision、plan 与 manifest，另行产生仅限本次
no-attack 诊断的执行决定。它不修改 construction profile 的 false 授权，不运行
attack、fixed-FPR、baseline 或阶段推进。
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import shutil
from typing import Any, Callable, Iterable, Mapping

from evaluation.protocol.record_writer import write_json, write_jsonl
from experiments.generative_video_model_probe.colab_runtime import (
    CONTROLLED_EMBEDDING_STRENGTH_DIAGNOSTIC_PROFILE,
    WAN21_PRIMARY_MODEL_ID,
    run_colab_probe,
)
from experiments.generative_video_model_probe.controlled_embedding_strength_profile import (
    CONTROLLED_EMBEDDING_PROFILE_ID,
    build_controlled_embedding_generation_plan,
    build_velocity_constraint_config_for_strength_level,
    validate_controlled_embedding_strength_profile,
    validate_source_trajectory_signal_bundle,
    _recorded_digest_for_relocated_path,
)
from experiments.generative_video_model_probe.trajectory_signal_localization_diagnostic import (
    execute_condition,
)


DEFAULT_CONFIG_PATH = (
    "configs/protocol/sstw_controlled_embedding_strength_diagnostic.json"
)
DEFAULT_CONSTRUCTION_CONFIG_PATH = (
    "configs/protocol/sstw_controlled_embedding_strength_profile.json"
)
DIAGNOSTIC_RECORD_VERSION = "controlled_embedding_strength_diagnostic_v1"
CONTROLLED_EMBEDDING_DIAGNOSTIC_PROFILE_ID = (
    "sstw_controlled_embedding_strength_diagnostic"
)
CONSTRUCTION_DECISION_SUFFIX = (
    "artifacts/controlled_embedding_profile_construction_decision.json"
)
CONSTRUCTION_MANIFEST_SUFFIX = (
    "artifacts/controlled_embedding_profile_construction_manifest.json"
)
CONSTRUCTION_PLAN_SUFFIX = (
    "records/controlled_embedding_generation_plan.jsonl"
)
SOURCE_DECISION_SUFFIX = (
    "artifacts/trajectory_signal_diagnostic_decision.json"
)
SOURCE_SNAPSHOT_SUFFIX = (
    "artifacts/trajectory_signal_immutable_input_snapshot.json"
)
SOURCE_MANIFEST_SUFFIX = (
    "artifacts/trajectory_signal_diagnostic_manifest.json"
)
PROMPT_SUITE_SUFFIX = "datasets/prompt_seed_suite.json"
LIKELIHOOD_CALIBRATION_SUFFIX = (
    "records/trajectory_replay_smoke_likelihood_calibrations.jsonl"
)


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON 顶层必须是对象: {path}")
    return value


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise TypeError(f"JSONL 每行必须是对象: {path}")
    return [dict(row) for row in rows]


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
            f"controlled embedding input 必须唯一包含 {logical_suffix}: {unique}"
        )
    return unique[0]


def validate_controlled_embedding_diagnostic_config(
    config: Mapping[str, Any],
) -> None:
    prohibited = (
        "attacked_phase_execution_allowed",
        "cross_project_integration_allowed",
        "external_baseline_execution_allowed",
        "fixed_fpr_evaluation_allowed",
        "large_scale_generation_allowed",
        "stage_progression_allowed",
        "test_split_claims_allowed",
        "time_grid_selection_allowed",
    )
    invalid = [name for name in prohibited if config.get(name) is not False]
    if invalid:
        raise ValueError(
            "controlled embedding diagnostic 禁止项未冻结: "
            + ", ".join(invalid)
        )
    if config.get("profile_id") != CONTROLLED_EMBEDDING_DIAGNOSTIC_PROFILE_ID:
        raise ValueError("controlled embedding diagnostic profile_id 不受支持")
    if config.get("stage_id") != (
        "controlled_embedding_strength_no_attack_diagnostic"
    ):
        raise ValueError("controlled embedding diagnostic stage_id 不受支持")
    if config.get("execution_scope") != (
        "predeclared_16_record_no_attack_single_factor_lambda_max_diagnostic"
    ):
        raise ValueError("controlled embedding diagnostic execution scope 不受支持")
    if config.get("claim_support_status") != (
        "controlled_embedding_strength_diagnostic_only_not_paper_evidence"
    ):
        raise ValueError("controlled embedding diagnostic claim 语义不受支持")
    if config.get("paper_result_level") != (
        "controlled_embedding_strength_diagnostic"
    ):
        raise ValueError("controlled embedding diagnostic paper result level 不受支持")
    if config.get("required_construction_profile_id") != (
        CONTROLLED_EMBEDDING_PROFILE_ID
    ):
        raise ValueError("controlled embedding diagnostic construction profile 不受支持")
    if config.get("required_construction_decision") != (
        "profile_constructed_generation_not_authorized"
    ):
        raise ValueError("controlled embedding diagnostic construction decision 不受支持")
    if int(config.get("required_construction_plan_record_count") or 0) != 16:
        raise ValueError("controlled embedding diagnostic 必须精确消费16条计划")
    if [int(value) for value in config.get("replay_grid_step_counts") or []] != [
        8,
        20,
        40,
    ]:
        raise ValueError("controlled embedding diagnostic replay grid 必须冻结为8/20/40")
    if [
        int(value)
        for value in config.get("required_decision_replay_step_counts") or []
    ] != [20, 40]:
        raise ValueError(
            "controlled embedding diagnostic decision grid 必须冻结为20/40"
        )
    for name, expected in (
        ("generation_aligned_replay_step_count", 8),
        ("primary_replay_step_count", 20),
        ("trajectory_signal_fine_replay_step_count", 40),
    ):
        if int(config.get(name) or 0) != expected:
            raise ValueError(
                f"controlled embedding diagnostic grid role 未冻结: {name}"
            )
    if config.get("no_attack_video_condition_id") != (
        "no_attack_controlled_embedding_strength"
    ):
        raise ValueError(
            "controlled embedding diagnostic no-attack condition ID 不受支持"
        )
    for name, expected in (
        ("minimum_correct_over_wrong_fraction", 0.75),
        ("minimum_path_margin_gain_over_clean_fraction", 0.5),
        ("minimum_replay_reliability", 0.05),
    ):
        observed = float(config.get(name) or 0.0)
        if not math.isclose(
            observed,
            expected,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"controlled embedding diagnostic 判定阈值未冻结: {name}"
            )
    if set(config.get("required_source_method_variants") or []) != {
        "sstw_full_method",
        "sstw_clean_unwatermarked_reference",
    }:
        raise ValueError("controlled embedding diagnostic method variants 不受支持")
    if config.get("required_generation_model_id") != WAN21_PRIMARY_MODEL_ID:
        raise ValueError("controlled embedding diagnostic generation model 不受支持")
    if config.get("required_embedding_strength_level_ids") != [
        "reference_default",
        "moderate_increase",
        "high_increase",
        "clean_unwatermarked_control",
    ]:
        raise ValueError("controlled embedding diagnostic strength ladder 不受支持")


def _require_recorded_digest(
    recorded_hashes: Mapping[str, Any],
    path: Path,
    logical_suffix: str,
    *,
    label: str,
) -> None:
    expected = _recorded_digest_for_relocated_path(
        recorded_hashes,
        path,
        logical_suffix=logical_suffix,
    )
    if expected is None or _sha256_file(path) != expected:
        raise ValueError(f"{label} digest 不匹配: {logical_suffix}")


def validate_controlled_embedding_execution_input(
    input_root: str | Path,
    *,
    diagnostic_config_path: str | Path = DEFAULT_CONFIG_PATH,
    construction_config_path: str | Path = DEFAULT_CONSTRUCTION_CONFIG_PATH,
) -> dict[str, Any]:
    """验证 portable input bundle，并重建 construction plan 防止路径替换。"""

    root = Path(input_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"controlled embedding input root 不存在: {root}")
    paths = {
        "construction_decision_path": _find_unique_file(
            root, CONSTRUCTION_DECISION_SUFFIX
        ),
        "construction_manifest_path": _find_unique_file(
            root, CONSTRUCTION_MANIFEST_SUFFIX
        ),
        "construction_plan_path": _find_unique_file(
            root, CONSTRUCTION_PLAN_SUFFIX
        ),
        "source_decision_path": _find_unique_file(root, SOURCE_DECISION_SUFFIX),
        "source_snapshot_path": _find_unique_file(root, SOURCE_SNAPSHOT_SUFFIX),
        "source_manifest_path": _find_unique_file(root, SOURCE_MANIFEST_SUFFIX),
        "prompt_suite_path": _find_unique_file(root, PROMPT_SUITE_SUFFIX),
        "likelihood_calibration_path": _find_unique_file(
            root, LIKELIHOOD_CALIBRATION_SUFFIX
        ),
    }
    diagnostic_config_path = Path(diagnostic_config_path).resolve()
    construction_config_path = Path(construction_config_path).resolve()
    diagnostic_config = _read_json(diagnostic_config_path)
    construction_config = _read_json(construction_config_path)
    validate_controlled_embedding_diagnostic_config(diagnostic_config)
    validate_controlled_embedding_strength_profile(construction_config)

    construction_decision = _read_json(paths["construction_decision_path"])
    construction_manifest = _read_json(paths["construction_manifest_path"])
    construction_plan = _read_jsonl(paths["construction_plan_path"])
    source_decision = _read_json(paths["source_decision_path"])
    source_snapshot = _read_json(paths["source_snapshot_path"])
    source_manifest = _read_json(paths["source_manifest_path"])
    prompt_suite = _read_json(paths["prompt_suite_path"])

    expected_construction = {
        "profile_id": diagnostic_config["required_construction_profile_id"],
        "controlled_embedding_profile_construction_status": "ready",
        "controlled_embedding_profile_construction_decision": diagnostic_config[
            "required_construction_decision"
        ],
        "generation_execution_allowed": False,
        "stage_progression_allowed": False,
        "controlled_embedding_plan_record_count": 16,
        "controlled_embedding_strength_level_count": 3,
    }
    mismatches = [
        name
        for name, expected in expected_construction.items()
        if construction_decision.get(name) != expected
    ]
    if mismatches:
        raise ValueError(
            "construction decision 不允许 controlled embedding execution: "
            + ", ".join(mismatches)
        )
    if construction_manifest.get("profile_id") != CONTROLLED_EMBEDDING_PROFILE_ID:
        raise ValueError("construction manifest profile_id 不匹配")
    if construction_manifest.get("config_digest") != _sha256_file(
        construction_config_path
    ):
        raise ValueError("construction manifest 与仓库 construction config 不匹配")

    for path_key, suffix in (
        ("construction_decision_path", CONSTRUCTION_DECISION_SUFFIX),
        ("construction_plan_path", CONSTRUCTION_PLAN_SUFFIX),
    ):
        _require_recorded_digest(
            dict(construction_manifest.get("output_sha256") or {}),
            paths[path_key],
            suffix,
            label="construction output",
        )
    for path_key, suffix in (
        ("source_decision_path", SOURCE_DECISION_SUFFIX),
        ("source_snapshot_path", SOURCE_SNAPSHOT_SUFFIX),
        ("source_manifest_path", SOURCE_MANIFEST_SUFFIX),
        ("prompt_suite_path", PROMPT_SUITE_SUFFIX),
    ):
        _require_recorded_digest(
            dict(construction_manifest.get("input_sha256") or {}),
            paths[path_key],
            suffix,
            label="construction input",
        )
    _require_recorded_digest(
        dict(construction_manifest.get("input_sha256") or {}),
        construction_config_path,
        "configs/protocol/sstw_controlled_embedding_strength_profile.json",
        label="construction config",
    )

    validate_source_trajectory_signal_bundle(
        source_decision,
        source_snapshot,
        source_manifest,
        construction_config,
        decision_path=paths["source_decision_path"],
        snapshot_path=paths["source_snapshot_path"],
        prompt_suite_path=paths["prompt_suite_path"],
    )
    _require_recorded_digest(
        dict(source_snapshot.get("governed_input_sha256") or {}),
        paths["likelihood_calibration_path"],
        LIKELIHOOD_CALIBRATION_SUFFIX,
        label="frozen likelihood calibration",
    )
    rebuilt_plan = build_controlled_embedding_generation_plan(
        prompt_suite,
        construction_config,
    )
    if construction_plan != rebuilt_plan:
        raise ValueError("construction plan 无法由绑定 prompt suite 与 config 精确重建")
    if construction_decision.get(
        "source_trajectory_signal_decision_sha256"
    ) != _sha256_file(paths["source_decision_path"]):
        raise ValueError("construction decision 未绑定 source trajectory decision")
    if any(
        row.get("generation_execution_allowed") is not False
        or row.get("stage_progression_allowed") is not False
        for row in construction_plan
    ):
        raise ValueError("construction plan 自身不得包含执行或阶段推进授权")

    return {
        "input_root": str(root),
        **{name: str(path) for name, path in paths.items()},
        "diagnostic_config_path": str(diagnostic_config_path),
        "construction_config_path": str(construction_config_path),
        "diagnostic_config": diagnostic_config,
        "construction_config": construction_config,
        "construction_decision": construction_decision,
        "construction_manifest": construction_manifest,
        "construction_plan": construction_plan,
        "prompt_suite": prompt_suite,
    }


def build_controlled_embedding_execution_decision(
    validated: Mapping[str, Any],
) -> dict[str, Any]:
    """在 construction false 授权之外，产生本次 no-attack-only 执行许可。"""

    return {
        "record_version": DIAGNOSTIC_RECORD_VERSION,
        "profile_id": CONTROLLED_EMBEDDING_DIAGNOSTIC_PROFILE_ID,
        "stage_id": "controlled_embedding_strength_no_attack_diagnostic",
        "controlled_embedding_execution_preflight_status": "ready",
        "controlled_embedding_execution_allowed": True,
        "execution_scope": (
            "predeclared_16_record_no_attack_single_factor_lambda_max_diagnostic"
        ),
        "source_construction_generation_execution_allowed": False,
        "source_construction_decision_sha256": _sha256_file(
            validated["construction_decision_path"]
        ),
        "source_construction_plan_sha256": _sha256_file(
            validated["construction_plan_path"]
        ),
        "source_construction_manifest_sha256": _sha256_file(
            validated["construction_manifest_path"]
        ),
        "controlled_embedding_plan_record_count": 16,
        "attacked_phase_executed": False,
        "attacked_phase_execution_allowed": False,
        "fixed_fpr_evaluation_allowed": False,
        "external_baseline_execution_allowed": False,
        "stage_progression_allowed": False,
        "formal_result": False,
        "claim_support_status": (
            "controlled_embedding_strength_diagnostic_only_not_paper_evidence"
        ),
    }


def _enriched_generation_plan(
    validated: Mapping[str, Any],
) -> list[dict[str, Any]]:
    prompts = {
        str(row["prompt_id"]): dict(row)
        for row in validated["prompt_suite"].get("prompts") or []
    }
    seeds = {
        str(row["seed_id"]): dict(row)
        for row in validated["prompt_suite"].get("seeds") or []
    }
    enriched: list[dict[str, Any]] = []
    for plan_row in validated["construction_plan"]:
        prompt = prompts[str(plan_row["prompt_id"])]
        seed = seeds[str(plan_row["seed_id"])]
        item = {
            **prompt,
            **seed,
            **dict(plan_row),
            "seed_value": int(plan_row["generation_seed_random"]),
            "cross_model_role": "main_generation_model",
            "sample_role": plan_row["sample_role"],
            "generation_sample_role": plan_row["sample_role"],
            "watermark_embedding_status": (
                "clean_unwatermarked_reference"
                if plan_row["method_variant"]
                == "sstw_clean_unwatermarked_reference"
                else "flow_scheduler_velocity_constraint"
            ),
            "formal_method_variant_execution": False,
        }
        item["prompt_suite_role"] = plan_row["prompt_suite_role"]
        item["seed_suite_role"] = plan_row["seed_suite_role"]
        enriched.append(item)
    return enriched


def _velocity_configs(
    validated: Mapping[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    construction_config = validated["construction_config"]
    for row in validated["construction_plan"]:
        level_id = str(row["embedding_strength_level_id"])
        if level_id == "clean_unwatermarked_control":
            config = build_velocity_constraint_config_for_strength_level(
                construction_config,
                "reference_default",
            )
        else:
            config = build_velocity_constraint_config_for_strength_level(
                construction_config,
                level_id,
            )
        result[str(row["controlled_embedding_plan_record_id"])] = config
    return result


def _materialize_bound_inputs(
    validated: Mapping[str, Any],
    output_root: Path,
) -> None:
    copies = {
        output_root / "inputs" / "construction" / Path(
            validated["construction_decision_path"]
        ).name: validated["construction_decision_path"],
        output_root / "inputs" / "construction" / Path(
            validated["construction_manifest_path"]
        ).name: validated["construction_manifest_path"],
        output_root / CONSTRUCTION_PLAN_SUFFIX: validated[
            "construction_plan_path"
        ],
        output_root / "inputs" / "source" / Path(
            validated["source_decision_path"]
        ).name: validated["source_decision_path"],
        output_root / "inputs" / "source" / Path(
            validated["source_snapshot_path"]
        ).name: validated["source_snapshot_path"],
        output_root / "inputs" / "source" / Path(
            validated["source_manifest_path"]
        ).name: validated["source_manifest_path"],
        output_root / PROMPT_SUITE_SUFFIX: validated["prompt_suite_path"],
        output_root / LIKELIHOOD_CALIBRATION_SUFFIX: validated[
            "likelihood_calibration_path"
        ],
    }
    for target, source in copies.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _default_generation_runner(
    validated: Mapping[str, Any],
    output_root: Path,
    *,
    pipeline_cache: dict[str, Any],
) -> dict[str, Any]:
    return run_colab_probe(
        output_root,
        validated["prompt_suite_path"],
        CONTROLLED_EMBEDDING_STRENGTH_DIAGNOSTIC_PROFILE,
        WAN21_PRIMARY_MODEL_ID,
        None,
        generation_plan_override=_enriched_generation_plan(validated),
        velocity_config_by_plan_record_id=_velocity_configs(validated),
        pipeline_cache=pipeline_cache,
    )


def _fraction(values: Iterable[bool]) -> float:
    rows = list(values)
    return sum(bool(value) for value in rows) / len(rows) if rows else 0.0


def build_controlled_embedding_strength_pair_records(
    summaries: Iterable[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """构造每档 correct/wrong 分离及同 prompt/seed strength-over-clean 增益。"""

    rows = [dict(row) for row in summaries]
    by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row.get("generation_model_id") or ""),
            str(row.get("prompt_id") or ""),
            str(row.get("seed_id") or ""),
            str(row.get("embedding_strength_level_id") or ""),
            int(row.get("replay_grid_step_count") or 0),
            str(row.get("candidate_key_role") or ""),
        )
        if key in by_key:
            raise ValueError(f"controlled embedding summary identity 重复: {key}")
        by_key[key] = row
    identities = sorted(
        {
            (key[0], key[1], key[2])
            for key in by_key
            if key[3] in set(config["required_embedding_strength_level_ids"])
        }
    )
    candidate_pairs: list[dict[str, Any]] = []
    margins: dict[tuple[Any, ...], dict[str, Any]] = {}
    for identity in identities:
        for level_id in config["required_embedding_strength_level_ids"]:
            for grid in config["replay_grid_step_counts"]:
                correct = by_key.get(
                    (*identity, str(level_id), int(grid), "correct_owner_key")
                )
                wrong = by_key.get(
                    (*identity, str(level_id), int(grid), "wrong_owner_key")
                )
                if correct is None or wrong is None:
                    continue
                base = {
                    "record_version": DIAGNOSTIC_RECORD_VERSION,
                    "profile_id": config["profile_id"],
                    "generation_model_id": identity[0],
                    "prompt_id": identity[1],
                    "seed_id": identity[2],
                    "embedding_strength_level_id": level_id,
                    "lambda_max": float(correct.get("lambda_max") or 0.0),
                    "replay_grid_step_count": int(grid),
                    "controlled_embedding_comparison_kind": (
                        "correct_owner_key_over_wrong_owner_key"
                    ),
                    "claim_support_status": config["claim_support_status"],
                }
                margin = {
                    **base,
                    "controlled_embedding_pair_record_id": _stable_digest(base),
                    "correct_over_wrong_trajectory_margin": (
                        float(correct["trajectory_velocity_projection"])
                        - float(wrong["trajectory_velocity_projection"])
                    ),
                    "correct_over_wrong_path_margin": (
                        float(correct["trajectory_path_projection"])
                        - float(wrong["trajectory_path_projection"])
                    ),
                    "correct_over_wrong_likelihood_margin": (
                        float(correct["replay_log_likelihood_ratio"])
                        - float(wrong["replay_log_likelihood_ratio"])
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
                candidate_pairs.append(margin)
                margins[(*identity, str(level_id), int(grid))] = margin

    clean_level = "clean_unwatermarked_control"
    for identity in identities:
        for level_id in config["required_embedding_strength_level_ids"]:
            if level_id == clean_level:
                continue
            for grid in config["replay_grid_step_counts"]:
                strength = margins.get((*identity, str(level_id), int(grid)))
                clean = margins.get((*identity, clean_level, int(grid)))
                if strength is None or clean is None:
                    continue
                base = {
                    "record_version": DIAGNOSTIC_RECORD_VERSION,
                    "profile_id": config["profile_id"],
                    "generation_model_id": identity[0],
                    "prompt_id": identity[1],
                    "seed_id": identity[2],
                    "embedding_strength_level_id": level_id,
                    "control_embedding_strength_level_id": clean_level,
                    "lambda_max": strength["lambda_max"],
                    "replay_grid_step_count": int(grid),
                    "controlled_embedding_comparison_kind": (
                        "strength_over_same_prompt_seed_clean_margin_gain"
                    ),
                    "claim_support_status": config["claim_support_status"],
                }
                candidate_pairs.append(
                    {
                        **base,
                        "controlled_embedding_pair_record_id": _stable_digest(base),
                        "strength_over_clean_trajectory_margin_gain": (
                            strength["correct_over_wrong_trajectory_margin"]
                            - clean["correct_over_wrong_trajectory_margin"]
                        ),
                        "strength_over_clean_path_margin_gain": (
                            strength["correct_over_wrong_path_margin"]
                            - clean["correct_over_wrong_path_margin"]
                        ),
                        "strength_over_clean_likelihood_margin_gain": (
                            strength["correct_over_wrong_likelihood_margin"]
                            - clean["correct_over_wrong_likelihood_margin"]
                        ),
                        "strength_over_clean_endpoint_margin_gain": (
                            strength["correct_over_wrong_endpoint_margin"]
                            - clean["correct_over_wrong_endpoint_margin"]
                        ),
                    }
                )
    return candidate_pairs


def build_controlled_embedding_strength_diagnostic_decision(
    generation_records: Iterable[Mapping[str, Any]],
    summaries: Iterable[Mapping[str, Any]],
    pairs: Iterable[Mapping[str, Any]],
    failures: Iterable[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    generation_rows = [dict(row) for row in generation_records]
    summary_rows = [dict(row) for row in summaries]
    pair_rows = [dict(row) for row in pairs]
    failure_rows = [dict(row) for row in failures]
    success_count = sum(
        row.get("generation_status") == "success" for row in generation_rows
    )
    diagnostics: dict[str, dict[str, Any]] = {}
    ready_by_level: dict[str, bool] = {}
    for level_id in config["required_embedding_strength_level_ids"]:
        if level_id == "clean_unwatermarked_control":
            continue
        level_ready = True
        for grid in config["replay_grid_step_counts"]:
            margins = [
                row
                for row in pair_rows
                if row.get("controlled_embedding_comparison_kind")
                == "correct_owner_key_over_wrong_owner_key"
                and row.get("embedding_strength_level_id") == level_id
                and int(row.get("replay_grid_step_count") or 0) == int(grid)
            ]
            gains = [
                row
                for row in pair_rows
                if row.get("controlled_embedding_comparison_kind")
                == "strength_over_same_prompt_seed_clean_margin_gain"
                and row.get("embedding_strength_level_id") == level_id
                and int(row.get("replay_grid_step_count") or 0) == int(grid)
            ]
            coverage_ready = len(margins) == len(gains) == 4
            trajectory_fraction = _fraction(
                row["correct_over_wrong_trajectory_margin"] > 0.0
                for row in margins
            )
            path_fraction = _fraction(
                row["correct_over_wrong_path_margin"] > 0.0 for row in margins
            )
            likelihood_fraction = _fraction(
                row["correct_over_wrong_likelihood_margin"] > 0.0
                for row in margins
            )
            endpoint_fraction = _fraction(
                row["correct_over_wrong_endpoint_margin"] > 0.0
                for row in margins
            )
            path_gain_fraction = _fraction(
                row["strength_over_clean_path_margin_gain"] > 0.0
                for row in gains
            )
            minimum_reliability = min(
                (
                    float(row["minimum_pair_reliability"])
                    for row in margins
                ),
                default=0.0,
            )
            path_signal_repair_gate_ready = bool(
                coverage_ready
                and path_fraction
                >= float(config["minimum_correct_over_wrong_fraction"])
                and likelihood_fraction
                >= float(config["minimum_correct_over_wrong_fraction"])
                and path_gain_fraction
                >= float(
                    config["minimum_path_margin_gain_over_clean_fraction"]
                )
                and minimum_reliability
                >= float(config["minimum_replay_reliability"])
            )
            diagnostics[f"{level_id}:{int(grid)}"] = {
                "embedding_strength_level_id": level_id,
                "lambda_max": (
                    float(margins[0]["lambda_max"]) if margins else None
                ),
                "replay_grid_step_count": int(grid),
                "coverage_ready": coverage_ready,
                "correct_over_wrong_trajectory_fraction": trajectory_fraction,
                "correct_over_wrong_path_fraction": path_fraction,
                "correct_over_wrong_likelihood_fraction": likelihood_fraction,
                "correct_over_wrong_endpoint_fraction": endpoint_fraction,
                "path_margin_gain_over_clean_fraction": path_gain_fraction,
                "minimum_replay_reliability": minimum_reliability,
                "path_signal_repair_gate_ready": path_signal_repair_gate_ready,
            }
            if int(grid) in {
                int(value)
                for value in config["required_decision_replay_step_counts"]
            }:
                level_ready = level_ready and path_signal_repair_gate_ready
        ready_by_level[str(level_id)] = level_ready

    runtime_ready = bool(
        len(generation_rows) == success_count == 16
        and len(summary_rows) == 96
        and len(pair_rows) == 84
        and not failure_rows
    )
    reference_ready = ready_by_level.get("reference_default", False)
    increased_ready = [
        level_id
        for level_id in ("moderate_increase", "high_increase")
        if ready_by_level.get(level_id, False)
    ]
    if not runtime_ready:
        classification = "runtime_or_input_failure_stop"
    elif reference_ready:
        classification = "reference_strength_path_signal_already_separated"
    elif increased_ready:
        classification = "lambda_increase_repaired_path_signal"
    else:
        classification = "lambda_increase_did_not_repair_path_signal_stop"
    return {
        "record_version": DIAGNOSTIC_RECORD_VERSION,
        "profile_id": config["profile_id"],
        "stage_id": config["stage_id"],
        "controlled_embedding_strength_diagnostic_decision": classification,
        "lambda_increase_path_signal_repair_observed": bool(
            runtime_ready and not reference_ready and increased_ready
        ),
        "path_signal_separated_strength_level_ids": [
            level_id
            for level_id in (
                "reference_default",
                "moderate_increase",
                "high_increase",
            )
            if ready_by_level.get(level_id, False)
        ],
        "strength_grid_diagnostics": diagnostics,
        "generation_record_count": len(generation_rows),
        "generation_success_count": success_count,
        "summary_record_count": len(summary_rows),
        "pair_record_count": len(pair_rows),
        "failure_record_count": len(failure_rows),
        "attacked_phase_executed": False,
        "attacked_phase_execution_allowed": False,
        "fixed_fpr_evaluation_executed": False,
        "external_baseline_execution_executed": False,
        "stage_progression_allowed": False,
        "formal_result": False,
        "claim_support_status": config["claim_support_status"],
    }


def _write_report(path: Path, decision: Mapping[str, Any]) -> None:
    lines = [
        "# SSTW controlled embedding strength diagnostic",
        "",
        f"- Decision: `{decision['controlled_embedding_strength_diagnostic_decision']}`",
        (
            "- Lambda increase repaired path signal: "
            f"`{decision['lambda_increase_path_signal_repair_observed']}`"
        ),
        (
            "- Separated strength levels: "
            f"`{decision['path_signal_separated_strength_level_ids']}`"
        ),
        f"- Generation success: `{decision['generation_success_count']}/16`",
        f"- Replay summaries: `{decision['summary_record_count']}`",
        f"- Pair records: `{decision['pair_record_count']}`",
        f"- Failures: `{decision['failure_record_count']}`",
        "- Attacked phase executed: `False`",
        "- Stage progression allowed: `False`",
        "",
        "该结果仅回答固定 lambda_max 阶梯是否修复 no-attack path 信号，",
        "不是 fixed-FPR、attack robustness、baseline 或论文证据。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_controlled_embedding_strength_diagnostic(
    input_root: str | Path,
    output_root: str | Path,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    construction_config_path: str | Path = DEFAULT_CONSTRUCTION_CONFIG_PATH,
    *,
    generation_runner: Callable[[Mapping[str, Any], Path], Mapping[str, Any]]
    | None = None,
    replay_runner: Callable[..., tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]]
    | None = None,
) -> dict[str, Any]:
    """执行 generation + no-attack replay，并始终保持非正式停止边界。"""

    output = Path(output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("controlled embedding diagnostic 要求新的空 output root")
    output.mkdir(parents=True, exist_ok=True)
    validated = validate_controlled_embedding_execution_input(
        input_root,
        diagnostic_config_path=config_path,
        construction_config_path=construction_config_path,
    )
    _materialize_bound_inputs(validated, output)
    execution_decision = build_controlled_embedding_execution_decision(
        validated
    )
    write_json(
        output
        / "artifacts"
        / "controlled_embedding_execution_decision.json",
        execution_decision,
    )

    pipeline_cache: dict[str, Any] = {}
    if generation_runner is None:
        generation_result = _default_generation_runner(
            validated,
            output,
            pipeline_cache=pipeline_cache,
        )
    else:
        generation_result = dict(generation_runner(validated, output))
    generation_records = _read_jsonl(
        output / "records" / "generation_records.jsonl"
    )
    generation_ready = bool(
        len(generation_records) == 16
        and all(
            row.get("generation_status") == "success"
            for row in generation_records
        )
    )
    config = validated["diagnostic_config"]
    if generation_ready:
        if replay_runner is None:
            def cached_pipeline_loader(
                model_id: str,
                revision: str | None = None,
            ) -> Any:
                del revision
                return pipeline_cache[model_id]

            summaries, steps, failures = execute_condition(
                output,
                output,
                config,
                condition="no_attack",
                pipeline_loader=cached_pipeline_loader,
            )
        else:
            summaries, steps, failures = replay_runner(
                output,
                output,
                config,
                condition="no_attack",
            )
    else:
        summaries, steps, failures = [], [], [
            {
                "record_version": DIAGNOSTIC_RECORD_VERSION,
                "profile_id": config["profile_id"],
                "controlled_embedding_strength_diagnostic_status": "failed",
                "controlled_embedding_strength_diagnostic_failure_reason": (
                    "generation_coverage_incomplete"
                ),
                "claim_support_status": config["claim_support_status"],
            }
        ]
    pairs = build_controlled_embedding_strength_pair_records(
        summaries,
        config,
    )
    decision = build_controlled_embedding_strength_diagnostic_decision(
        generation_records,
        summaries,
        pairs,
        failures,
        config,
    )
    write_jsonl(
        output
        / "records"
        / "controlled_embedding_strength_summary_records.jsonl",
        summaries,
    )
    write_jsonl(
        output
        / "records"
        / "controlled_embedding_strength_step_records.jsonl",
        steps,
    )
    write_jsonl(
        output
        / "records"
        / "controlled_embedding_strength_pair_records.jsonl",
        pairs,
    )
    write_jsonl(
        output
        / "records"
        / "controlled_embedding_strength_failure_records.jsonl",
        failures,
    )
    decision_path = (
        output
        / "artifacts"
        / "controlled_embedding_strength_diagnostic_decision.json"
    )
    write_json(decision_path, decision)
    write_json(
        output
        / "artifacts"
        / "controlled_embedding_strength_diagnostic_manifest.json",
        {
            "artifact_id": "controlled_embedding_strength_diagnostic_manifest",
            "artifact_type": "manifest",
            "profile_id": config["profile_id"],
            "source_construction_decision_sha256": execution_decision[
                "source_construction_decision_sha256"
            ],
            "source_construction_plan_sha256": execution_decision[
                "source_construction_plan_sha256"
            ],
            "generation_model_ids": [WAN21_PRIMARY_MODEL_ID],
            "generation_result": generation_result,
            "record_paths": [
                str(
                    output
                    / "records"
                    / "controlled_embedding_strength_summary_records.jsonl"
                ),
                str(
                    output
                    / "records"
                    / "controlled_embedding_strength_pair_records.jsonl"
                ),
                str(
                    output
                    / "records"
                    / "controlled_embedding_strength_failure_records.jsonl"
                ),
                str(decision_path),
            ],
            "attacked_phase_executed": False,
            "stage_progression_allowed": False,
            "formal_result": False,
            "claim_support_status": config["claim_support_status"],
        },
    )
    _write_report(
        output
        / "reports"
        / "controlled_embedding_strength_diagnostic_report.md",
        decision,
    )
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(
        description="执行固定 controlled embedding no-attack 强度诊断"
    )
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--construction-config-path",
        default=DEFAULT_CONSTRUCTION_CONFIG_PATH,
    )
    args = parser.parse_args()
    decision = run_controlled_embedding_strength_diagnostic(
        args.input_root,
        args.output_root,
        args.config_path,
        args.construction_config_path,
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
