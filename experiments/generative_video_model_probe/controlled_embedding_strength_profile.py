"""构建 Stage 0-D 失败后允许的受控 embedding 强度阶梯计划。

该模块只构建、验证和落盘 governed generation plan。它不加载模型、不生成视频，
也不把 ``controlled_embedding_profile_construction_allowed`` 扩张为 GPU 执行授权。
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
import math
from pathlib import Path, PurePosixPath
import shlex
from typing import Any, Mapping

from evaluation.protocol.package_naming import current_short_commit
from evaluation.protocol.record_writer import write_json, write_jsonl
from main.methods.state_space_watermark.velocity_field_constraint import (
    VelocityFieldConstraintConfig,
)


DEFAULT_CONFIG_PATH = (
    "configs/protocol/sstw_controlled_embedding_strength_profile.json"
)
CONTROLLED_EMBEDDING_PLAN_RECORD_VERSION = (
    "controlled_embedding_strength_plan_v1"
)
CONTROLLED_EMBEDDING_PROFILE_ID = "sstw_controlled_embedding_strength_profile"
SOURCE_TRAJECTORY_SIGNAL_PROFILE_ID = (
    "sstw_trajectory_signal_localization_diagnostic"
)
SOURCE_TRAJECTORY_SIGNAL_DECISION = (
    "embedding_or_replay_signal_not_separated_stop"
)


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


def _strength_levels(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    levels = config.get("strength_levels")
    if not isinstance(levels, list) or not levels:
        raise ValueError("controlled embedding profile 必须声明非空 strength_levels")
    if not all(isinstance(level, dict) for level in levels):
        raise TypeError("controlled embedding strength level 必须是对象")
    return [dict(level) for level in levels]


def validate_controlled_embedding_strength_profile(
    config: Mapping[str, Any],
) -> None:
    """冻结 construction-only profile 的范围、单因子阶梯和停止边界。"""

    required_false = (
        "attacked_phase_execution_allowed",
        "cross_project_integration_allowed",
        "external_baseline_execution_allowed",
        "fixed_fpr_evaluation_allowed",
        "generation_execution_allowed",
        "large_scale_generation_allowed",
        "stage_progression_allowed",
        "test_split_claims_allowed",
        "time_grid_selection_allowed",
    )
    invalid_false = [name for name in required_false if config.get(name) is not False]
    if invalid_false:
        raise ValueError(
            "controlled embedding construction 禁止项未冻结: "
            + ", ".join(invalid_false)
        )
    if config.get("profile_id") != CONTROLLED_EMBEDDING_PROFILE_ID:
        raise ValueError("controlled embedding profile_id 不受支持")
    if config.get("stage_id") != "controlled_embedding_strength_profile_construction":
        raise ValueError("controlled embedding stage_id 必须保持 construction 语义")
    if config.get("required_source_trajectory_signal_profile_id") != (
        SOURCE_TRAJECTORY_SIGNAL_PROFILE_ID
    ):
        raise ValueError("controlled embedding profile 未绑定 Stage 0-D source profile")
    if config.get("required_source_trajectory_signal_diagnostic_decision") != (
        SOURCE_TRAJECTORY_SIGNAL_DECISION
    ):
        raise ValueError("controlled embedding profile 未绑定 Stage 0-D failure decision")
    if config.get("required_source_controlled_embedding_profile_construction_allowed") is not True:
        raise ValueError("controlled embedding profile 必须要求显式 construction allowance")
    if config.get("required_source_owner_key_direction_preflight_status") != "ready":
        raise ValueError("controlled embedding profile 必须要求 ready owner-key preflight")
    if config.get("embedding_strength_selection_policy") != (
        "predeclared_single_factor_lambda_max_ladder_no_result_adaptive_selection"
    ):
        raise ValueError("controlled embedding strength selection policy 不受支持")
    if config.get("seed_source_split") != "calibration":
        raise ValueError("controlled embedding construction 只能使用 calibration seeds")
    if int(config.get("prompt_limit") or 0) != 2 or int(
        config.get("seed_limit") or 0
    ) != 2:
        raise ValueError("controlled embedding construction 必须冻结为2 prompts x 2 seeds")
    if int(config.get("num_inference_steps") or 0) != 8:
        raise ValueError("controlled embedding construction 必须保持8步 generation grid")
    if int(config.get("required_source_summary_record_count") or 0) != 72:
        raise ValueError("controlled embedding construction 必须绑定72条 Stage 0-D summaries")
    if int(config.get("required_source_pair_record_count") or 0) != 60:
        raise ValueError("controlled embedding construction 必须绑定60条 Stage 0-D pairs")
    if [float(value) for value in config.get("lambda_time_window") or []] != [
        0.25,
        0.75,
    ]:
        raise ValueError("controlled embedding construction 必须保持原冻结 phase window")

    levels = _strength_levels(config)
    if [level.get("embedding_strength_level_id") for level in levels] != [
        "reference_default",
        "moderate_increase",
        "high_increase",
    ]:
        raise ValueError("controlled embedding strength level 顺序或语义不受支持")
    default = VelocityFieldConstraintConfig()
    reference = levels[0]
    for field_name, expected in (
        ("lambda_max", default.lambda_max),
        ("velocity_norm_ratio_budget", default.velocity_norm_ratio_budget),
        ("flow_energy_budget_ratio", default.flow_energy_budget_ratio),
    ):
        if not math.isclose(
            float(reference.get(field_name, -1.0)),
            float(expected),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"reference_default 必须精确复现当前 runtime {field_name}"
            )
    multipliers = [float(level.get("embedding_strength_multiplier") or 0.0) for level in levels]
    lambda_values = [float(level.get("lambda_max") or 0.0) for level in levels]
    if multipliers != [1.0, 2.0, 4.0]:
        raise ValueError("controlled embedding multiplier ladder 必须冻结为 [1,2,4]")
    if any(
        not math.isclose(
            lambda_value,
            default.lambda_max * multiplier,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for lambda_value, multiplier in zip(lambda_values, multipliers, strict=True)
    ):
        raise ValueError("controlled embedding ladder 只能单因子缩放 lambda_max")
    if len({float(level["velocity_norm_ratio_budget"]) for level in levels}) != 1:
        raise ValueError("controlled embedding ladder 不得同时改变 velocity norm budget")
    if len({float(level["flow_energy_budget_ratio"]) for level in levels}) != 1:
        raise ValueError("controlled embedding ladder 不得同时改变 flow energy budget")
    maximum_delta_ratio = float(
        config.get("maximum_planned_velocity_constraint_delta_ratio") or 0.0
    )
    if maximum_delta_ratio <= 0.0 or any(
        float(level["lambda_max"])
        * float(level["velocity_norm_ratio_budget"])
        > maximum_delta_ratio + 1e-12
        for level in levels
    ):
        raise ValueError("controlled embedding strength level 超过预声明 delta ratio guard")


def build_velocity_constraint_config_for_strength_level(
    config: Mapping[str, Any],
    strength_level_id: str,
) -> VelocityFieldConstraintConfig:
    """把预声明强度映射到真实 core runtime config，不接受动态数值。"""

    validate_controlled_embedding_strength_profile(config)
    levels = {
        str(level["embedding_strength_level_id"]): level
        for level in _strength_levels(config)
    }
    try:
        level = levels[strength_level_id]
    except KeyError as exc:
        raise ValueError(f"未知 controlled embedding strength level: {strength_level_id}") from exc
    base = VelocityFieldConstraintConfig()
    return replace(
        base,
        lambda_max=float(level["lambda_max"]),
        velocity_norm_ratio_budget=float(level["velocity_norm_ratio_budget"]),
        flow_energy_budget_ratio=float(level["flow_energy_budget_ratio"]),
    )


def validate_source_trajectory_signal_decision(
    decision: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    """仅接受正确 owner key 下无攻击信号未分离的 Stage 0-D 决策。"""

    required = {
        "profile_id": config["required_source_trajectory_signal_profile_id"],
        "trajectory_signal_diagnostic_decision": config[
            "required_source_trajectory_signal_diagnostic_decision"
        ],
        "controlled_embedding_profile_construction_allowed": True,
        "owner_key_direction_preflight_status": config[
            "required_source_owner_key_direction_preflight_status"
        ],
        "owner_key_direction_all_match": True,
        "owner_key_context_all_match": True,
        "owner_key_phase_grid_all_match": True,
        "no_attack_signal_separation_ready": False,
        "attacked_phase_executed": False,
        "stage_progression_allowed": False,
        "failure_record_count": 0,
        "summary_record_count": int(config["required_source_summary_record_count"]),
        "pair_record_count": int(config["required_source_pair_record_count"]),
    }
    mismatches = [
        name for name, expected in required.items() if decision.get(name) != expected
    ]
    if mismatches:
        raise ValueError(
            "source Stage 0-D decision 不允许构建 controlled embedding profile: "
            + ", ".join(mismatches)
        )


def _recorded_digest_for_relocated_path(
    recorded_hashes: Mapping[str, Any],
    path: Path,
    *,
    logical_suffix: str,
) -> str | None:
    """Resolve a governed digest after a packaged artifact changes roots."""

    suffix_parts = PurePosixPath(logical_suffix).parts
    matches: list[str] = []
    for raw_path, digest in dict(recorded_hashes).items():
        if Path(raw_path).resolve() == path.resolve():
            matches.append(str(digest))
            continue
        recorded_parts = PurePosixPath(str(raw_path).replace("\\", "/")).parts
        if len(recorded_parts) >= len(suffix_parts) and (
            recorded_parts[-len(suffix_parts) :] == suffix_parts
        ):
            matches.append(str(digest))
    unique = sorted(set(matches))
    if len(unique) > 1:
        raise ValueError(
            f"governed artifact suffix 解析到冲突 digest: {logical_suffix}"
        )
    return unique[0] if unique else None


def validate_source_trajectory_signal_bundle(
    decision: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    decision_path: Path,
    snapshot_path: Path,
    prompt_suite_path: Path,
) -> None:
    """绑定 source decision、immutable snapshot、manifest 与 prompt suite。"""

    validate_source_trajectory_signal_decision(decision, config)
    snapshot_expected = {
        "profile_id": SOURCE_TRAJECTORY_SIGNAL_PROFILE_ID,
        "immutable_input_preflight_status": "ready",
        "immutable_input_scope": "full_replay_diagnostic_inputs",
        "generation_record_count": 12,
        "attack_record_count": 24,
        "likelihood_calibration_input_status": "ready",
    }
    snapshot_mismatches = [
        name
        for name, expected in snapshot_expected.items()
        if snapshot.get(name) != expected
    ]
    if snapshot_mismatches:
        raise ValueError(
            "source Stage 0-D immutable snapshot 不完整: "
            + ", ".join(snapshot_mismatches)
        )
    recorded_snapshot_digest = str(
        snapshot.get("immutable_input_snapshot_digest") or ""
    )
    snapshot_payload = dict(snapshot)
    snapshot_payload.pop("immutable_input_snapshot_digest", None)
    if not recorded_snapshot_digest or _stable_digest(snapshot_payload) != (
        recorded_snapshot_digest
    ):
        raise ValueError("source Stage 0-D immutable snapshot digest 不匹配")
    if manifest.get("profile_id") != SOURCE_TRAJECTORY_SIGNAL_PROFILE_ID:
        raise ValueError("source Stage 0-D manifest profile_id 不匹配")
    if manifest.get("immutable_input_snapshot_digest") != recorded_snapshot_digest:
        raise ValueError("source Stage 0-D manifest 未绑定 immutable snapshot digest")
    governed_outputs = dict(manifest.get("output_sha256") or {})
    for path, logical_suffix in (
        (
            decision_path,
            "artifacts/trajectory_signal_diagnostic_decision.json",
        ),
        (
            snapshot_path,
            "artifacts/trajectory_signal_immutable_input_snapshot.json",
        ),
    ):
        expected = _recorded_digest_for_relocated_path(
            governed_outputs,
            path,
            logical_suffix=logical_suffix,
        )
        if expected is None or _sha256_file(path) != expected:
            raise ValueError(
                f"source Stage 0-D manifest output digest 不匹配: {path.name}"
            )
    governed_inputs = dict(snapshot.get("governed_input_sha256") or {})
    expected_prompt_digest = _recorded_digest_for_relocated_path(
        governed_inputs,
        prompt_suite_path,
        logical_suffix="datasets/prompt_seed_suite.json",
    )
    if expected_prompt_digest is None or _sha256_file(prompt_suite_path) != (
        expected_prompt_digest
    ):
        raise ValueError("prompt suite 未被 source Stage 0-D immutable snapshot 绑定")


def _select_prompt_seed_inputs(
    prompt_suite: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    roles = set(str(value) for value in config.get("prompt_suite_roles") or [])
    prompts = [
        dict(item)
        for item in prompt_suite.get("prompts") or []
        if item.get("prompt_suite_role") in roles
    ][: int(config["prompt_limit"])]
    seeds = [
        dict(item)
        for item in prompt_suite.get("seeds") or []
        if item.get("prompt_suite_role") in roles
        and item.get("split") == config["seed_source_split"]
    ][: int(config["seed_limit"])]
    if len(prompts) != int(config["prompt_limit"]):
        raise ValueError("prompt suite 无法满足 controlled embedding prompt coverage")
    if len(seeds) != int(config["seed_limit"]):
        raise ValueError("prompt suite 无法满足 controlled embedding calibration seed coverage")
    required_prompt_fields = ("prompt_id", "prompt_text")
    required_seed_fields = ("seed_id", "seed_value")
    if any(not all(item.get(name) is not None for name in required_prompt_fields) for item in prompts):
        raise ValueError("controlled embedding prompt 缺少身份或文本")
    if any(not all(item.get(name) is not None for name in required_seed_fields) for item in seeds):
        raise ValueError("controlled embedding seed 缺少身份或数值")
    return prompts, seeds


def _plan_record_id(record: Mapping[str, Any]) -> str:
    identity = {
        name: record.get(name)
        for name in (
            "profile_id",
            "generation_model_id",
            "prompt_id",
            "seed_id",
            "method_variant",
            "embedding_strength_level_id",
        )
    }
    return _stable_digest(identity)


def build_controlled_embedding_generation_plan(
    prompt_suite: Mapping[str, Any],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """构建固定4-source的三强度加同源 clean 对照计划，但不授权执行。"""

    validate_controlled_embedding_strength_profile(config)
    prompts, seeds = _select_prompt_seed_inputs(prompt_suite, config)
    levels = _strength_levels(config)
    plan: list[dict[str, Any]] = []
    common = {
        "record_version": CONTROLLED_EMBEDDING_PLAN_RECORD_VERSION,
        "profile_id": config["profile_id"],
        "claim_support_status": config["claim_support_status"],
        "generation_execution_allowed": False,
        "generation_model_id": config["generation_model_id"],
        "num_inference_steps": int(config["num_inference_steps"]),
        "video_length_frames": int(config["num_frames"]),
        "video_resolution": f"{int(config['width'])}x{int(config['height'])}",
        "lambda_time_window": [
            float(value) for value in config["lambda_time_window"]
        ],
        "embedding_strength_selection_policy": config[
            "embedding_strength_selection_policy"
        ],
        "stage_progression_allowed": False,
    }
    for prompt in prompts:
        for seed in seeds:
            identity = {
                "prompt_id": prompt["prompt_id"],
                "prompt_text_hash": sha256(
                    str(prompt["prompt_text"]).encode("utf-8")
                ).hexdigest(),
                "seed_id": seed["seed_id"],
                "generation_seed_random": int(seed["seed_value"]),
                "split": seed["split"],
                "prompt_suite_role": prompt["prompt_suite_role"],
                "seed_suite_role": seed["prompt_suite_role"],
            }
            for level in levels:
                runtime_config = build_velocity_constraint_config_for_strength_level(
                    config,
                    str(level["embedding_strength_level_id"]),
                )
                record = {
                    **common,
                    **identity,
                    "sample_role": "controlled_embedding_positive_source",
                    "method_variant": "sstw_full_method",
                    "embedding_strength_control_role": "strength_ladder_candidate",
                    "embedding_strength_level_id": level[
                        "embedding_strength_level_id"
                    ],
                    "embedding_strength_multiplier": float(
                        level["embedding_strength_multiplier"]
                    ),
                    "lambda_max": runtime_config.lambda_max,
                    "velocity_norm_ratio_budget": (
                        runtime_config.velocity_norm_ratio_budget
                    ),
                    "flow_energy_budget_ratio": runtime_config.flow_energy_budget_ratio,
                    "maximum_planned_velocity_constraint_delta_ratio": float(
                        config["maximum_planned_velocity_constraint_delta_ratio"]
                    ),
                }
                record["controlled_embedding_plan_record_id"] = _plan_record_id(record)
                plan.append(record)
            clean = {
                **common,
                **identity,
                "sample_role": "clean_negative",
                "method_variant": "sstw_clean_unwatermarked_reference",
                "embedding_strength_control_role": "same_prompt_seed_clean_reference",
                "embedding_strength_level_id": "clean_unwatermarked_control",
                "embedding_strength_multiplier": 0.0,
                "lambda_max": 0.0,
                "velocity_norm_ratio_budget": VelocityFieldConstraintConfig().velocity_norm_ratio_budget,
                "flow_energy_budget_ratio": VelocityFieldConstraintConfig().flow_energy_budget_ratio,
                "maximum_planned_velocity_constraint_delta_ratio": float(
                    config["maximum_planned_velocity_constraint_delta_ratio"]
                ),
            }
            clean["controlled_embedding_plan_record_id"] = _plan_record_id(clean)
            plan.append(clean)
    return plan


def _assert_fresh_output_root(output_root: Path) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            "controlled embedding construction 要求新的空 output root"
        )


def construct_controlled_embedding_strength_profile(
    source_decision_path: str | Path,
    source_snapshot_path: str | Path,
    source_manifest_path: str | Path,
    prompt_suite_path: str | Path,
    output_run_root: str | Path,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """验证 Stage 0-D source decision 并物化 construction-only plan。"""

    source_path = Path(source_decision_path).resolve()
    snapshot_path = Path(source_snapshot_path).resolve()
    source_manifest_path = Path(source_manifest_path).resolve()
    prompt_path = Path(prompt_suite_path).resolve()
    protocol_path = Path(config_path).resolve()
    output = Path(output_run_root).resolve()
    _assert_fresh_output_root(output)
    config = _read_json(protocol_path)
    validate_controlled_embedding_strength_profile(config)
    source_decision = _read_json(source_path)
    source_snapshot = _read_json(snapshot_path)
    source_manifest = _read_json(source_manifest_path)
    validate_source_trajectory_signal_bundle(
        source_decision,
        source_snapshot,
        source_manifest,
        config,
        decision_path=source_path,
        snapshot_path=snapshot_path,
        prompt_suite_path=prompt_path,
    )
    prompt_suite = _read_json(prompt_path)
    plan = build_controlled_embedding_generation_plan(prompt_suite, config)

    plan_path = output / "records" / "controlled_embedding_generation_plan.jsonl"
    decision_path = (
        output
        / "artifacts"
        / "controlled_embedding_profile_construction_decision.json"
    )
    manifest_path = (
        output / "artifacts" / "controlled_embedding_profile_construction_manifest.json"
    )
    write_jsonl(plan_path, plan)
    strength_level_count = len(_strength_levels(config))
    decision = {
        "record_version": CONTROLLED_EMBEDDING_PLAN_RECORD_VERSION,
        "profile_id": config["profile_id"],
        "stage_id": config["stage_id"],
        "controlled_embedding_profile_construction_status": "ready",
        "controlled_embedding_profile_construction_decision": (
            "profile_constructed_generation_not_authorized"
        ),
        "generation_execution_allowed": False,
        "stage_progression_allowed": False,
        "controlled_embedding_strength_level_count": strength_level_count,
        "controlled_embedding_plan_record_count": len(plan),
        "source_trajectory_signal_diagnostic_decision": source_decision[
            "trajectory_signal_diagnostic_decision"
        ],
        "source_trajectory_signal_decision_sha256": _sha256_file(source_path),
        "immutable_input_snapshot_digest": source_snapshot[
            "immutable_input_snapshot_digest"
        ],
        "claim_support_status": config["claim_support_status"],
    }
    write_json(decision_path, decision)
    output_hashes = {
        str(path): _sha256_file(path) for path in (plan_path, decision_path)
    }
    repository_root = Path(__file__).resolve().parents[2]
    builder_source_path = Path(__file__).resolve()
    manifest = {
        "artifact_id": "controlled_embedding_profile_construction_manifest",
        "artifact_type": "manifest",
        "profile_id": config["profile_id"],
        "input_paths": [
            str(source_path),
            str(snapshot_path),
            str(source_manifest_path),
            str(prompt_path),
            str(protocol_path),
        ],
        "input_sha256": {
            str(path): _sha256_file(path)
            for path in (
                source_path,
                snapshot_path,
                source_manifest_path,
                prompt_path,
                protocol_path,
            )
        },
        "output_paths": [str(plan_path), str(decision_path)],
        "output_sha256": output_hashes,
        "config_digest": _sha256_file(protocol_path),
        "code_version": current_short_commit(repository_root),
        "controlled_embedding_profile_builder_source_path": str(
            builder_source_path
        ),
        "controlled_embedding_profile_builder_source_sha256": _sha256_file(
            builder_source_path
        ),
        "rebuild_command": (
            "python -m experiments.generative_video_model_probe."
            "controlled_embedding_strength_profile "
            f"--source-decision-path {shlex.quote(str(source_path))} "
            f"--source-snapshot-path {shlex.quote(str(snapshot_path))} "
            f"--source-manifest-path {shlex.quote(str(source_manifest_path))} "
            f"--prompt-suite-path {shlex.quote(str(prompt_path))} "
            f"--output-run-root {shlex.quote(str(output))} "
            f"--config-path {shlex.quote(str(protocol_path))}"
        ),
        "record_paths": [str(plan_path), str(decision_path)],
        "claim_support_status": config["claim_support_status"],
    }
    write_json(manifest_path, manifest)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(
        description="构建 construction-only controlled embedding strength profile"
    )
    parser.add_argument("--source-decision-path", required=True)
    parser.add_argument("--source-snapshot-path", required=True)
    parser.add_argument("--source-manifest-path", required=True)
    parser.add_argument("--prompt-suite-path", required=True)
    parser.add_argument("--output-run-root", required=True)
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    decision = construct_controlled_embedding_strength_profile(
        args.source_decision_path,
        args.source_snapshot_path,
        args.source_manifest_path,
        args.prompt_suite_path,
        args.output_run_root,
        args.config_path,
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
