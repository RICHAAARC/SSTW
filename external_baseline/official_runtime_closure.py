"""现代 external baseline 真实运行闭合要求预检。

该模块的职责是把 5 个主实验现代视频水印 baseline 的真实运行条件收敛为一个
可由 Colab Notebook 自动执行的 governed artifact。它不运行重型第三方模型, 也不
生成 baseline 分数; 它只检查当前 run_root 是否已经具备 source、requirements、
runtime videos、官方资源、项目内 official bundle cache 或可由后续 Notebook
调用的项目内官方 runner。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any, Mapping

from external_baseline.official_eval_adapters.common import (
    official_bundle_candidate_paths,
    official_result_bundle_roots,
    read_json,
    validate_repository_generated_bundle,
    validate_score_payload,
)
from external_baseline.runtime_trace_io import comparable_detection_records, read_jsonl


DEFAULT_RUNTIME_CLOSURE_REQUIREMENTS = Path("configs/external_baselines/official_runtime_closure_requirements.json")
MODERN_BASELINE_IDS = (
    "videoseal",
    "vidsig",
    "videomark",
    "videoshield",
    "sigmark",
)


def _read_json(path: str | Path) -> dict[str, Any]:
    """读取 JSON 对象, 并兼容 UTF-8 BOM。"""

    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"json_payload_must_be_object:{path}")
    return payload


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """写出稳定 JSON artifact。"""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def _safe_path_token(value: Any) -> str:
    """把 prompt、seed、attack 等字段转换为 bundle 路径 token。"""

    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "unknown"))
    return text.strip("_") or "unknown"


def _as_bundle_args(record: Mapping[str, Any]) -> SimpleNamespace:
    """把 runtime detection record 转换为 official bundle 候选路径参数。"""

    return SimpleNamespace(
        attack_name=str(record.get("attack_name") or ""),
        prompt_id=str(record.get("prompt_id") or ""),
        seed_id=str(record.get("seed_id") or ""),
        trajectory_trace_id=str(record.get("trajectory_trace_id") or ""),
    )


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    """按字符串形式去重路径, 保持原始顺序。"""

    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def _drive_project_root_from_run_root(run_root: str | Path) -> Path:
    """从 profile run_root 推导 Google Drive 项目根目录。"""

    root = Path(run_root)
    try:
        if root.name and root.parent.name == "generative_video_model_probe" and root.parents[1].name == "runs":
            return root.parents[2]
    except IndexError:
        pass
    try:
        return root.parents[2]
    except IndexError:
        return root


def _resolve_resource_root(run_root: Path, resource_root: str | Path | None) -> Path:
    """解析 external baseline 官方资源目录。"""

    if resource_root:
        return Path(resource_root)
    env_value = os.environ.get("SSTW_EXTERNAL_BASELINE_RESOURCE_ROOT", "").strip()
    if env_value:
        return Path(env_value)
    return _drive_project_root_from_run_root(run_root) / "resources" / "external_baseline"


def _resolve_bundle_root(run_root: Path, official_result_bundle_root: str | Path | None) -> Path:
    """解析当前 workflow profile 的 official bundle root。"""

    if official_result_bundle_root:
        return Path(official_result_bundle_root)
    env_value = os.environ.get("SSTW_EXTERNAL_BASELINE_OFFICIAL_RESULT_BUNDLE_ROOT", "").strip()
    if env_value:
        return Path(env_value)
    profile = run_root.name or "validation_scale"
    return _drive_project_root_from_run_root(run_root) / "external_baseline_official_result_bundles" / profile


def _path_with_repo_root(path_text: str, repo_root: Path) -> Path:
    """把配置中的绝对或仓库相对路径转换为 Path。"""

    path = Path(path_text)
    if path.is_absolute():
        return path
    return repo_root / path


def _runtime_input_audit(run_root: Path) -> dict[str, Any]:
    """检查 run_root 中 runtime records 与视频输入是否存在。"""

    detection_record_path = run_root / "records" / "runtime_detection_records.jsonl"
    generation_record_path = run_root / "records" / "generation_records.jsonl"
    trajectory_record_path = run_root / "records" / "trajectory_trace.jsonl"
    videos_dir = run_root / "videos"
    attacked_videos_dir = run_root / "attacked_videos"
    detection_records = comparable_detection_records(run_root)
    generation_records = read_jsonl(generation_record_path)
    missing_source_paths = []
    missing_attacked_paths = []
    for record in detection_records:
        source_path = Path(str(record.get("source_video_path") or ""))
        attacked_path = Path(str(record.get("attacked_video_path") or ""))
        if not source_path.exists():
            missing_source_paths.append(str(source_path))
        if not attacked_path.exists():
            missing_attacked_paths.append(str(attacked_path))
    missing_requirements: list[str] = []
    if not detection_record_path.exists():
        missing_requirements.append("records/runtime_detection_records.jsonl")
    if not generation_record_path.exists():
        missing_requirements.append("records/generation_records.jsonl")
    if not videos_dir.exists():
        missing_requirements.append("videos")
    if not attacked_videos_dir.exists():
        missing_requirements.append("attacked_videos")
    if not detection_records:
        missing_requirements.append("non_empty_comparable_runtime_detection_records")
    if missing_source_paths:
        missing_requirements.append("referenced_source_video_paths_exist")
    if missing_attacked_paths:
        missing_requirements.append("referenced_attacked_video_paths_exist")
    ready = not missing_requirements
    return {
        "runtime_input_ready": ready,
        "runtime_input_status": "ready" if ready else "missing_or_incomplete",
        "run_root": str(run_root),
        "runtime_detection_record_path": str(detection_record_path),
        "runtime_detection_record_path_exists": detection_record_path.exists(),
        "runtime_detection_record_count": len(detection_records),
        "generation_record_path": str(generation_record_path),
        "generation_record_path_exists": generation_record_path.exists(),
        "generation_record_count": len(generation_records),
        "trajectory_record_path": str(trajectory_record_path),
        "trajectory_record_path_exists": trajectory_record_path.exists(),
        "videos_dir": str(videos_dir),
        "videos_dir_exists": videos_dir.exists(),
        "attacked_videos_dir": str(attacked_videos_dir),
        "attacked_videos_dir_exists": attacked_videos_dir.exists(),
        "missing_source_video_path_count": len(missing_source_paths),
        "missing_attacked_video_path_count": len(missing_attacked_paths),
        "missing_source_video_path_examples": missing_source_paths[:10],
        "missing_attacked_video_path_examples": missing_attacked_paths[:10],
        "missing_runtime_requirements": missing_requirements,
    }


def _source_audit(row: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    """检查 baseline 官方源码目录与关键源码文件。"""

    baseline_id = str(row.get("baseline_id") or "")
    env_source_dir = os.environ.get(f"SSTW_{baseline_id.upper()}_OFFICIAL_SOURCE_DIR", "").strip()
    candidate_texts = []
    if env_source_dir:
        candidate_texts.append(env_source_dir)
    candidate_texts.extend(str(item) for item in row.get("official_source_dir_candidates", []) if item)
    if row.get("source_dir"):
        candidate_texts.append(str(row["source_dir"]))
    candidates = _dedupe_paths([_path_with_repo_root(text, repo_root) for text in candidate_texts])
    required_files = [str(item) for item in row.get("required_source_files", [])]
    selected_source_dir: Path | None = None
    selected_missing_files: list[str] = []
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        missing_files = [relative for relative in required_files if not (candidate / relative).exists()]
        if not missing_files:
            selected_source_dir = candidate
            selected_missing_files = []
            break
        if selected_source_dir is None:
            selected_source_dir = candidate
            selected_missing_files = missing_files
    ready = selected_source_dir is not None and not selected_missing_files
    return {
        "official_source_ready": ready,
        "official_source_status": "ready" if ready else "missing_or_incomplete",
        "official_source_dir": str(selected_source_dir) if selected_source_dir else "",
        "official_source_dir_candidates": [str(path) for path in candidates],
        "required_source_files": required_files,
        "missing_required_source_files": selected_missing_files if selected_source_dir else required_files,
    }


def _requirements_file_audit(row: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    """检查 baseline requirements 文件是否存在。"""

    requirements_file = _path_with_repo_root(str(row.get("requirements_file") or ""), repo_root)
    return {
        "requirements_file_path": str(requirements_file),
        "requirements_file_exists": requirements_file.exists(),
        "requirements_file_status": "ready" if requirements_file.exists() else "missing",
    }


def _find_default_resource_path(resource_root: Path, requirement: Mapping[str, Any]) -> Path | None:
    """按配置中的默认路径或 glob 查找 Google Drive 官方资源。"""

    relative_path = str(requirement.get("default_resource_relative_path") or "").strip()
    if relative_path:
        candidate = resource_root / relative_path
        if candidate.exists():
            return candidate
    pattern = str(requirement.get("default_resource_glob_pattern") or "").strip()
    if pattern and resource_root.exists():
        matches = sorted(path for path in resource_root.glob(pattern) if path.exists())
        if matches:
            return matches[0]
    return None


def _resource_audit(row: Mapping[str, Any], resource_root: Path) -> tuple[dict[str, Any], dict[str, str]]:
    """检查单个 baseline 的官方资源环境变量与默认 Drive 路径。"""

    resource_rows: list[dict[str, Any]] = []
    environment_updates: dict[str, str] = {}
    required_missing: list[str] = []
    for requirement in row.get("resource_env_vars", []):
        if not isinstance(requirement, Mapping):
            continue
        env_var = str(requirement.get("env_var") or "")
        env_value = os.environ.get(env_var, "").strip()
        env_path = Path(env_value).expanduser() if env_value else None
        env_path_exists = bool(env_path and env_path.exists())
        default_path = _find_default_resource_path(resource_root, requirement)
        if not env_path_exists and default_path is not None:
            environment_updates[env_var] = str(default_path)
        effective_path = env_path if env_path_exists else default_path
        effective_exists = bool(effective_path and effective_path.exists())
        required_for_default_adapter = bool(requirement.get("required_for_repository_default_adapter"))
        if required_for_default_adapter and not effective_exists:
            required_missing.append(env_var)
        resource_rows.append({
            "env_var": env_var,
            "resource_role": requirement.get("resource_role"),
            "required_for_repository_default_adapter": required_for_default_adapter,
            "project_owned_resource_required": bool(requirement.get("project_owned_resource_required")),
            "env_path": str(env_path) if env_path else "",
            "env_path_exists": env_path_exists,
            "default_resource_relative_path": requirement.get("default_resource_relative_path"),
            "default_resource_glob_pattern": requirement.get("default_resource_glob_pattern"),
            "default_resource_path": str(default_path) if default_path else "",
            "default_resource_path_exists": default_path is not None,
            "effective_resource_path": str(effective_path) if effective_path else "",
            "effective_resource_path_exists": effective_exists,
            "resource_policy_status": "project_owned_resource_required_not_external_supplement",
        })
    all_required_present = not required_missing
    return {
        "resource_root": str(resource_root),
        "resource_requirements": resource_rows,
        "required_resource_env_vars": [
            row_item["env_var"]
            for row_item in resource_rows
            if row_item.get("required_for_repository_default_adapter")
        ],
        "missing_required_resource_env_vars": required_missing,
        "required_resource_ready": all_required_present,
        "required_resource_status": "ready" if all_required_present else "missing",
    }, environment_updates


def _native_command_audit(row: Mapping[str, Any]) -> dict[str, Any]:
    """检查外层、官方内部与原生命令环境变量是否已配置。"""

    env_names = [
        str(row.get("external_baseline_command_env_var") or ""),
        str(row.get("official_baseline_command_env_var") or ""),
        str(row.get("native_command_env_var") or ""),
    ]
    status = {}
    for env_name in env_names:
        if env_name:
            status[env_name] = bool(os.environ.get(env_name, "").strip())
    return {
        "command_env_status": status,
        "external_baseline_command_configured": bool(status.get(str(row.get("external_baseline_command_env_var") or ""))),
        "official_baseline_command_configured": bool(status.get(str(row.get("official_baseline_command_env_var") or ""))),
        "native_command_configured": bool(status.get(str(row.get("native_command_env_var") or ""))),
    }


def _project_owned_reference_runner_audit(row: Mapping[str, Any]) -> dict[str, Any]:
    """检查项目内官方参考 runner 是否可作为 formal reference 尝试路径。

    此处只判断“是否允许进入真实运行尝试”, 不把 runner 存在解释为已得到
    measured_formal 结果。以 SIGMark 为例, `SSTW_SIGMARK_BIT_ACCURACY_NPZ`
    是 Hunyuan gen->extract 完成后的输出产物, 不能在预检阶段被当作运行前
    必备输入而阻断官方 runner。
    """

    baseline_id = str(row.get("baseline_id") or "")
    candidate_keys = [
        "project_owned_formal_reference_runner_module",
        f"project_owned_{baseline_id}_runner_module" if baseline_id else "",
        "project_owned_hunyuan_runner_module",
        "project_owned_vidsig_runner_module",
        "project_owned_videomark_runner_module",
        "project_owned_videoshield_runner_module",
    ]
    runner_module = ""
    runner_module_key = ""
    for key in candidate_keys:
        if not key:
            continue
        value = str(row.get(key) or "").strip()
        if value:
            runner_module = value
            runner_module_key = key
            break

    enabled = bool(runner_module)
    enable_env_var = ""
    enable_env_value = ""
    if baseline_id == "sigmark":
        enable_env_var = "SSTW_RUN_SIGMARK_OFFICIAL_HUNYUAN_PIPELINE"
        enable_env_value = os.environ.get(enable_env_var, "true").strip()
        enabled = bool(runner_module) and enable_env_value.lower() not in {"0", "false", "no", "off"}

    return {
        "project_owned_reference_runner_module": runner_module,
        "project_owned_reference_runner_module_key": runner_module_key,
        "project_owned_reference_runner_available": bool(runner_module),
        "project_owned_reference_runner_enable_env_var": enable_env_var,
        "project_owned_reference_runner_enable_env_value": enable_env_value,
        "project_owned_reference_runner_ready_to_attempt": enabled,
        "project_owned_reference_runner_policy": (
            "runner_may_generate_required_outputs_during_formal_reference_notebook"
            if enabled
            else "runner_not_configured_or_disabled"
        ),
    }


def _bundle_candidate_paths_for_roots(
    baseline_id: str,
    record: Mapping[str, Any],
    roots: list[Path],
) -> list[Path]:
    """构造单条 comparison unit 的 official bundle 候选路径。"""

    attack = _safe_path_token(record.get("attack_name"))
    prompt = _safe_path_token(record.get("prompt_id"))
    seed = _safe_path_token(record.get("seed_id"))
    trace = _safe_path_token(record.get("trajectory_trace_id"))
    candidates: list[Path] = []
    for root in roots:
        baseline_root = root / baseline_id
        candidates.extend([
            baseline_root / "records" / f"{prompt}__{seed}__{attack}.json",
            baseline_root / "records" / f"{trace}__{attack}.json",
            baseline_root / prompt / seed / f"{attack}.json",
            baseline_root / trace / f"{attack}.json",
        ])
    return candidates


def _bundle_audit(
    baseline_id: str,
    detection_records: list[dict[str, Any]],
    official_result_bundle_root: Path,
) -> dict[str, Any]:
    """检查项目内 official bundle cache 是否覆盖当前 comparison units。"""

    existing_roots = official_result_bundle_roots()
    roots = _dedupe_paths([official_result_bundle_root, *existing_roots])
    present_count = 0
    missing_examples: list[dict[str, Any]] = []
    invalid_examples: list[dict[str, Any]] = []
    for record in detection_records:
        valid_path: Path | None = None
        last_invalid_reason = ""
        env_candidates = official_bundle_candidate_paths(baseline_id=baseline_id, args=_as_bundle_args(record))
        candidates = _dedupe_paths([*env_candidates, *_bundle_candidate_paths_for_roots(baseline_id, record, roots)])
        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                payload = read_json(candidate)
                validate_score_payload(payload)
                validate_repository_generated_bundle(payload, candidate)
            except Exception as exc:  # noqa: BLE001 - artifact 需要保留具体阻断原因。
                last_invalid_reason = str(exc)
                continue
            valid_path = candidate
            break
        if valid_path is not None:
            present_count += 1
        elif last_invalid_reason and len(invalid_examples) < 10:
            invalid_examples.append({
                "baseline_id": baseline_id,
                "prompt_id": record.get("prompt_id"),
                "seed_id": record.get("seed_id"),
                "attack_name": record.get("attack_name"),
                "invalid_bundle_reason": last_invalid_reason,
                "candidate_paths": [str(path) for path in candidates[:8]],
            })
        elif len(missing_examples) < 10:
            missing_examples.append({
                "baseline_id": baseline_id,
                "prompt_id": record.get("prompt_id"),
                "seed_id": record.get("seed_id"),
                "attack_name": record.get("attack_name"),
                "candidate_paths": [str(path) for path in candidates[:8]],
            })
    expected_count = len(detection_records)
    complete = bool(detection_records) and present_count == expected_count
    return {
        "official_result_bundle_root": str(official_result_bundle_root),
        "official_result_bundle_roots": [str(root) for root in roots],
        "bundle_expected_count": expected_count,
        "bundle_present_count": present_count,
        "bundle_missing_count": max(0, expected_count - present_count),
        "bundle_complete": complete,
        "bundle_status": "complete" if complete else "missing_or_incomplete",
        "missing_bundle_examples": missing_examples,
        "invalid_bundle_examples": invalid_examples,
    }


def _baseline_runtime_row(
    row: Mapping[str, Any],
    *,
    repo_root: Path,
    resource_root: Path,
    official_result_bundle_root: Path,
    runtime_input: Mapping[str, Any],
    detection_records: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, str]]:
    """构建单个 baseline 的真实运行闭合要求行。"""

    baseline_id = str(row.get("baseline_id") or "")
    source = _source_audit(row, repo_root)
    requirements_file = _requirements_file_audit(row, repo_root)
    resources, environment_updates = _resource_audit(row, resource_root)
    commands = _native_command_audit(row)
    project_runner = _project_owned_reference_runner_audit(row)
    bundle = _bundle_audit(baseline_id, detection_records, official_result_bundle_root)
    runtime_ready = bool(runtime_input.get("runtime_input_ready"))
    source_ready = bool(source.get("official_source_ready"))
    requirements_ready = bool(requirements_file.get("requirements_file_exists"))
    resource_ready = bool(resources.get("required_resource_ready"))
    bundle_complete = bool(bundle.get("bundle_complete"))
    auto_supported = bool(row.get("automatic_bundle_generation_supported_by_sstw"))
    native_command_configured = bool(commands.get("native_command_configured"))
    official_command_configured = bool(commands.get("official_baseline_command_configured"))
    default_adapter_can_attempt = resource_ready and bool(resources.get("required_resource_env_vars"))
    project_runner_can_attempt = bool(project_runner.get("project_owned_reference_runner_ready_to_attempt"))
    ready_to_attempt = (
        runtime_ready
        and source_ready
        and requirements_ready
        and (
            bundle_complete
            or native_command_configured
            or official_command_configured
            or auto_supported
            or default_adapter_can_attempt
            or project_runner_can_attempt
        )
    )
    missing: list[str] = []
    if not runtime_ready:
        missing.append("runtime_inputs")
    if not source_ready:
        missing.append("official_source_required_files")
    if not requirements_ready:
        missing.append("requirements_file")
    if not (
        bundle_complete
        or native_command_configured
        or official_command_configured
        or auto_supported
        or default_adapter_can_attempt
        or project_runner_can_attempt
    ):
        missing.append("official_bundle_or_native_command_or_required_resources")
    if ready_to_attempt:
        status = "ready_to_attempt_formal_reference"
    elif bundle_complete:
        status = "official_bundle_complete_but_runtime_or_source_incomplete"
    else:
        status = "blocked_missing_requirements"
    return {
        "baseline_id": baseline_id,
        "baseline_name": row.get("baseline_name"),
        "runtime_support_mode": row.get("runtime_support_mode"),
        "automatic_bundle_generation_supported_by_sstw": auto_supported,
        "colab_default_can_attempt_without_user_files": bool(row.get("colab_default_can_attempt_without_user_files")),
        "external_supplemental_result_bundle_allowed": bool(row.get("external_supplemental_result_bundle_allowed")),
        "self_containment_rule": "project_clone_build_run_adapt_record_required_no_external_supplement",
        "source_requirement": source,
        "requirements_file_requirement": requirements_file,
        "resource_requirement": resources,
        "command_requirement": commands,
        "project_owned_reference_runner_requirement": project_runner,
        "official_bundle_requirement": bundle,
        "runtime_closure_ready_to_attempt": ready_to_attempt,
        "runtime_closure_status": status,
        "runtime_closure_missing_requirements": missing,
        "missing_resource_action": row.get("missing_resource_action"),
        "official_bundle_policy": row.get("official_bundle_policy"),
        "claim_support_status": "runtime_requirements_preflight_only_not_measured_formal",
    }, environment_updates


def load_official_runtime_closure_requirements(
    config_path: str | Path = DEFAULT_RUNTIME_CLOSURE_REQUIREMENTS,
) -> dict[str, Any]:
    """读取现代 baseline 真实运行闭合要求配置。"""

    return _read_json(config_path)


def build_official_runtime_closure_requirements(
    run_root: str | Path,
    *,
    config_path: str | Path = DEFAULT_RUNTIME_CLOSURE_REQUIREMENTS,
    repo_root: str | Path = ".",
    resource_root: str | Path | None = None,
    official_result_bundle_root: str | Path | None = None,
    baseline_id: str | None = None,
) -> dict[str, Any]:
    """构建 external baseline 真实运行闭合要求预检结果。"""

    resolved_run_root = Path(run_root)
    resolved_repo_root = Path(repo_root).resolve()
    resolved_resource_root = _resolve_resource_root(resolved_run_root, resource_root)
    resolved_bundle_root = _resolve_bundle_root(resolved_run_root, official_result_bundle_root)
    if official_result_bundle_root:
        os.environ["SSTW_EXTERNAL_BASELINE_OFFICIAL_RESULT_BUNDLE_ROOT"] = str(resolved_bundle_root)
    config = load_official_runtime_closure_requirements(config_path)
    runtime_input = _runtime_input_audit(resolved_run_root)
    detection_records = comparable_detection_records(resolved_run_root)
    required_baseline_ids = set(MODERN_BASELINE_IDS)
    selected_rows = [
        row for row in config.get("baseline_runtime_requirements", [])
        if isinstance(row, Mapping)
        and (
            str(row.get("baseline_id") or "") == baseline_id
            if baseline_id is not None
            else str(row.get("baseline_id") or "") in required_baseline_ids
        )
    ]
    baseline_rows: list[dict[str, Any]] = []
    environment_updates: dict[str, str] = {
        "SSTW_EXTERNAL_BASELINE_RESOURCE_ROOT": str(resolved_resource_root),
        "SSTW_EXTERNAL_BASELINE_OFFICIAL_RESULT_BUNDLE_ROOT": str(resolved_bundle_root),
    }
    for row in selected_rows:
        baseline_row, row_updates = _baseline_runtime_row(
            row,
            repo_root=resolved_repo_root,
            resource_root=resolved_resource_root,
            official_result_bundle_root=resolved_bundle_root,
            runtime_input=runtime_input,
            detection_records=detection_records,
        )
        baseline_rows.append(baseline_row)
        environment_updates.update(row_updates)
    ready_rows = [row for row in baseline_rows if row.get("runtime_closure_ready_to_attempt")]
    blocked_rows = [row for row in baseline_rows if not row.get("runtime_closure_ready_to_attempt")]
    bundle_complete_rows = [row for row in baseline_rows if row.get("official_bundle_requirement", {}).get("bundle_complete")]
    decision = "PASS" if baseline_rows and not blocked_rows else "FAIL"
    return {
        "artifact_name": "external_baseline_official_runtime_closure_requirements.json",
        "manifest_kind": "external_baseline_official_runtime_closure_requirements",
        "config_path": str(config_path),
        "config_version": config.get("config_version"),
        "run_root": str(resolved_run_root),
        "repo_root": str(resolved_repo_root),
        "resource_root": str(resolved_resource_root),
        "official_result_bundle_root": str(resolved_bundle_root),
        "baseline_filter": baseline_id or "all_modern_external_baselines",
        "official_runtime_closure_decision": decision,
        "official_runtime_closure_status": "ready_to_attempt_all_selected_baselines" if decision == "PASS" else "blocked_missing_runtime_requirements",
        "runtime_input_audit": runtime_input,
        "baseline_count": len(baseline_rows),
        "runtime_closure_ready_count": len(ready_rows),
        "runtime_closure_blocked_count": len(blocked_rows),
        "official_bundle_complete_baseline_count": len(bundle_complete_rows),
        "runtime_closure_ready_baselines": [str(row.get("baseline_id")) for row in ready_rows],
        "runtime_closure_blocked_baselines": [str(row.get("baseline_id")) for row in blocked_rows],
        "official_bundle_complete_baselines": [str(row.get("baseline_id")) for row in bundle_complete_rows],
        "missing_requirement_summary": {
            str(row.get("baseline_id")): row.get("runtime_closure_missing_requirements", [])
            for row in blocked_rows
        },
        "baseline_runtime_rows": baseline_rows,
        "environment_updates": environment_updates,
        "claim_support_status": "official_runtime_requirements_preflight_only_not_claim_evidence",
    }


def write_official_runtime_closure_requirements(
    run_root: str | Path,
    *,
    output_json: str | Path | None = None,
    config_path: str | Path = DEFAULT_RUNTIME_CLOSURE_REQUIREMENTS,
    repo_root: str | Path = ".",
    resource_root: str | Path | None = None,
    official_result_bundle_root: str | Path | None = None,
    baseline_id: str | None = None,
) -> dict[str, Any]:
    """写出 external baseline 真实运行闭合要求 artifact。"""

    payload = build_official_runtime_closure_requirements(
        run_root,
        config_path=config_path,
        repo_root=repo_root,
        resource_root=resource_root,
        official_result_bundle_root=official_result_bundle_root,
        baseline_id=baseline_id,
    )
    output_path = Path(output_json) if output_json else Path(run_root) / "artifacts" / "external_baseline_official_runtime_closure_requirements.json"
    _write_json(output_path, payload)
    return payload


def main() -> None:
    """CLI 入口。"""

    parser = argparse.ArgumentParser(description="检查现代 external baseline 真实运行闭合要求。")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--config-path", default=str(DEFAULT_RUNTIME_CLOSURE_REQUIREMENTS))
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--resource-root", default="")
    parser.add_argument("--official-result-bundle-root", default="")
    parser.add_argument("--baseline-id", default="")
    args = parser.parse_args()
    payload = write_official_runtime_closure_requirements(
        args.run_root,
        output_json=args.output_json or None,
        config_path=args.config_path,
        repo_root=args.repo_root,
        resource_root=args.resource_root or None,
        official_result_bundle_root=args.official_result_bundle_root or None,
        baseline_id=args.baseline_id or None,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
