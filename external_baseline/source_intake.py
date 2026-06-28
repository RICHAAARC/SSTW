"""外部 baseline 源码接入、检查和运行证据清单构造工具。"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping


DEFAULT_SOURCE_REGISTRY_PATH = Path("external_baseline/source_registry.json")
DEFAULT_OUTPUT_ROOT = Path("external_baseline")
TABLE_PLAN_RELATIVE_PATH = Path("plans/external_baseline_table_plan.json")
INTAKE_MANIFEST_NAME = "external_baseline_intake_manifest.json"
SOURCE_INSPECTION_NAME = "external_baseline_source_inspection.json"
CLONE_RESULTS_NAME = "external_baseline_clone_results.json"
OFFICIAL_COMMAND_EVIDENCE_RELATIVE_ROOT = Path("artifacts/external_baseline_evidence")

MODERN_BASELINE_IDS = {
    "videoshield",
    "sigmark",
    "spdmark",
    "videomark",
    "vidsig",
    "videoseal",
}

SOURCE_CANDIDATE_FILE_NAMES = (
    "README.md",
    "readme.md",
    "requirements.txt",
    "environment.yml",
    "pyproject.toml",
    "setup.py",
    "LICENSE",
    "LICENSE.md",
    "license",
)

ENTRYPOINT_CANDIDATE_PREFIXES = (
    "run",
    "eval",
    "evaluate",
    "detect",
    "infer",
)


def read_json(path: str | Path) -> dict[str, Any]:
    """读取 UTF-8 或带 BOM 的 JSON 对象。"""
    input_path = Path(path)
    payload = json.loads(input_path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON 顶层必须是对象: {input_path}")
    return payload


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """写出 UTF-8 JSON 文件, 用于跨 Notebook、脚本和审计工具稳定交接。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _safe_list(value: Any) -> list[dict[str, Any]]:
    """把 registry 中的 baseline_sources 字段转成对象列表。"""
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def load_source_registry(path: str | Path = DEFAULT_SOURCE_REGISTRY_PATH) -> dict[str, Any]:
    """读取 external_baseline source registry。"""
    return read_json(path)


def _is_cloneable_url(url: str) -> bool:
    """判断 source URL 是否可以被 `git clone` 直接处理。"""
    normalized = url.strip().lower()
    if not normalized:
        return False
    return normalized.endswith(".git") or "github.com/" in normalized or "gitlab.com/" in normalized


def _command_env_var_for(baseline_id: str) -> str:
    """根据 baseline_id 推导现代 baseline 官方命令环境变量名。"""
    return f"SSTW_{baseline_id.upper()}_EVAL_COMMAND"


def _list_source_files(source_dir: Path) -> list[Path]:
    """列出 source 目录下的普通文件, 排除 Python 缓存和 Git 元数据。"""
    if not source_dir.is_dir():
        return []
    files: list[Path] = []
    for path in source_dir.rglob("*"):
        if not path.is_file():
            continue
        parts = set(path.relative_to(source_dir).parts)
        if ".git" in parts or "__pycache__" in parts:
            continue
        files.append(path)
    return sorted(files)


def _source_dir_summary(repo_root: Path, source_dir_value: str) -> dict[str, Any]:
    """汇总单个 source 目录是否存在以及包含多少可审计文件。"""
    source_dir = repo_root / source_dir_value
    files = _list_source_files(source_dir)
    return {
        "source_dir": source_dir_value,
        "source_dir_exists": source_dir.is_dir(),
        "source_dir_file_count": len(files),
        "source_dir_top_level_entries": sorted(item.name for item in source_dir.iterdir()) if source_dir.is_dir() else [],
    }


def _source_intake_status(entry: Mapping[str, Any], source_exists: bool, command_configured: bool) -> tuple[str, str]:
    """根据 registry、source 目录和命令配置状态给出 source intake 状态。"""
    source_status = str(entry.get("source_status") or "")
    baseline_id = str(entry.get("baseline_id") or "")
    repository_url = str(entry.get("official_repository_url") or "")
    if source_status == "repository_local_algorithm":
        return "local_repository_algorithm", "none"
    if source_exists:
        return "source_snapshot_available", "inspect_and_configure_adapter_command"
    if command_configured:
        return "official_command_configured", "run_formal_command_adapter"
    if _is_cloneable_url(repository_url):
        return "source_clone_required", "run_source_intake_clone_with_execute"
    if baseline_id in MODERN_BASELINE_IDS:
        return "manual_source_or_command_required", "provide_official_source_or_eval_command"
    return "source_not_required", "none"


def build_source_intake_manifest(
    registry_path: str | Path = DEFAULT_SOURCE_REGISTRY_PATH,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    """构造外部 baseline source intake manifest。

    该函数属于治理层实现。它不会下载第三方源码, 只把当前 registry、adapter、source
    目录和命令环境变量状态汇总为可落盘 manifest。这样 Colab 冷启动、Windows 本地和 CI
    可以用同一份规则判断 baseline 是否已经达到可运行边界。
    """
    root = Path(repo_root)
    registry = load_source_registry(registry_path)
    baseline_sources = _safe_list(registry.get("baseline_sources"))
    rows: list[dict[str, Any]] = []
    for entry in baseline_sources:
        baseline_id = str(entry.get("baseline_id") or "")
        source_dir_value = str(entry.get("source_dir") or "")
        adapter_path = str(entry.get("adapter_path") or "")
        command_env_var = _command_env_var_for(baseline_id)
        command_configured = bool(os.environ.get(command_env_var))
        source_summary = _source_dir_summary(root, source_dir_value)
        intake_status, action_required = _source_intake_status(
            entry,
            bool(source_summary["source_dir_exists"]),
            command_configured,
        )
        rows.append({
            "baseline_id": baseline_id,
            "baseline_name": entry.get("baseline_name"),
            "baseline_family": entry.get("baseline_family"),
            "comparison_group": entry.get("comparison_group"),
            "paper_claim_support": bool(entry.get("paper_claim_support")),
            "official_repository_url": entry.get("official_repository_url"),
            "official_repository_branch": entry.get("official_repository_branch"),
            "official_repository_commit": entry.get("official_repository_commit"),
            "source_status": entry.get("source_status"),
            "source_intake_status": intake_status,
            "source_intake_action_required": action_required,
            "source_cloneable": _is_cloneable_url(str(entry.get("official_repository_url") or "")),
            "adapter_path": adapter_path,
            "adapter_exists": (root / adapter_path).is_file(),
            "adapter_status": entry.get("adapter_status"),
            "result_status": entry.get("result_status"),
            "external_baseline_command_env_var": command_env_var if baseline_id in MODERN_BASELINE_IDS else "not_applicable",
            "external_baseline_command_config_status": "configured" if command_configured else ("missing" if baseline_id in MODERN_BASELINE_IDS else "not_applicable"),
            **source_summary,
        })
    modern_rows = [row for row in rows if row["baseline_id"] in MODERN_BASELINE_IDS]
    source_ready_rows = [
        row for row in rows
        if row["source_intake_status"] in {"local_repository_algorithm", "source_snapshot_available", "official_command_configured"}
    ]
    return {
        "artifact_name": INTAKE_MANIFEST_NAME,
        "manifest_kind": "external_baseline_source_intake",
        "registry_path": str(registry_path),
        "registry_name": registry.get("registry_name"),
        "source_root": registry.get("source_root"),
        "baseline_source_count": len(rows),
        "modern_external_baseline_source_count": len(modern_rows),
        "source_intake_ready_count": len(source_ready_rows),
        "source_intake_missing_count": len(rows) - len(source_ready_rows),
        "modern_external_baseline_source_ready_count": sum(1 for row in modern_rows if row in source_ready_rows),
        "external_baseline_source_intake_decision": "PASS" if rows else "FAIL",
        "claim_support_status": "source_intake_manifest_only_not_claim_evidence",
        "baseline_sources": rows,
    }


def _candidate_files(source_dir: Path) -> dict[str, list[str]]:
    """在 source 目录中查找常见依赖、入口和许可证候选文件。"""
    files = _list_source_files(source_dir)
    candidate_names = {name.lower() for name in SOURCE_CANDIDATE_FILE_NAMES}
    metadata_files: list[str] = []
    entrypoint_files: list[str] = []
    for file_path in files:
        relative = file_path.relative_to(source_dir).as_posix()
        lower_name = file_path.name.lower()
        stem = file_path.stem.lower()
        if lower_name in candidate_names:
            metadata_files.append(relative)
        if file_path.suffix == ".py" and any(stem.startswith(prefix) for prefix in ENTRYPOINT_CANDIDATE_PREFIXES):
            entrypoint_files.append(relative)
    return {
        "metadata_files": sorted(metadata_files),
        "entrypoint_candidate_files": sorted(entrypoint_files),
    }


def build_source_inspection_manifest(
    registry_path: str | Path = DEFAULT_SOURCE_REGISTRY_PATH,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    """构造外部 baseline source inspection manifest。"""
    root = Path(repo_root)
    registry = load_source_registry(registry_path)
    inspection_rows: list[dict[str, Any]] = []
    for entry in _safe_list(registry.get("baseline_sources")):
        source_dir_value = str(entry.get("source_dir") or "")
        source_dir = root / source_dir_value
        candidates = _candidate_files(source_dir)
        source_exists = source_dir.is_dir()
        inspection_rows.append({
            "baseline_id": entry.get("baseline_id"),
            "source_dir": source_dir_value,
            "source_dir_exists": source_exists,
            "source_inspection_status": "inspected" if source_exists else "source_missing",
            "source_dir_file_count": len(_list_source_files(source_dir)),
            **candidates,
        })
    inspected_count = sum(1 for row in inspection_rows if row["source_inspection_status"] == "inspected")
    return {
        "artifact_name": SOURCE_INSPECTION_NAME,
        "manifest_kind": "external_baseline_source_inspection",
        "registry_path": str(registry_path),
        "source_inspection_record_count": len(inspection_rows),
        "source_inspection_ready_count": inspected_count,
        "source_inspection_missing_count": len(inspection_rows) - inspected_count,
        "source_inspection_decision": "PASS" if inspection_rows else "FAIL",
        "claim_support_status": "source_inspection_manifest_only_not_claim_evidence",
        "source_inspections": inspection_rows,
    }


def _run_git_command(command: list[str], cwd: Path | None = None) -> dict[str, Any]:
    """执行 Git 命令并返回受治理的结果摘要。"""
    completed = subprocess.run(command, cwd=str(cwd) if cwd else None, check=False, text=True, capture_output=True)
    return {
        "command": command,
        "return_code": completed.returncode,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def _concrete_git_reference(value: Any) -> str | None:
    """返回可 checkout 的 branch 或 commit, 排除 registry 中的占位语义值。"""
    text = str(value or "").strip()
    if not text or text in {"user_configured_or_source_default", "not_applicable"}:
        return None
    return text


def _checkout_pinned_source_reference(entry: Mapping[str, Any], source_dir: Path) -> list[dict[str, Any]]:
    """在 source 仓库存在时 checkout registry 中冻结的 branch / commit。

    通用工程写法是先 clone 或 fetch, 再 checkout 明确版本。项目特定要求是
    validation-scale 与 pilot_paper 的外部 baseline 必须能记录精确上游 commit,
    因此若 registry 提供 `official_repository_commit`, 该 commit 优先于 branch。
    """
    branch = _concrete_git_reference(entry.get("official_repository_branch"))
    commit = _concrete_git_reference(entry.get("official_repository_commit"))
    git_results: list[dict[str, Any]] = []
    if branch:
        git_results.append(_run_git_command(["git", "-C", str(source_dir), "checkout", branch]))
    if commit:
        git_results.append(_run_git_command(["git", "-C", str(source_dir), "checkout", commit]))
    return git_results


def _clone_or_update_one(entry: Mapping[str, Any], repo_root: Path, execute_clone: bool) -> dict[str, Any]:
    """对单个 baseline 生成或执行 source clone 操作。"""
    baseline_id = str(entry.get("baseline_id") or "")
    source_dir_value = str(entry.get("source_dir") or "")
    source_dir = repo_root / source_dir_value
    repository_url = str(entry.get("official_repository_url") or "")
    branch = str(entry.get("official_repository_branch") or "")
    commit = str(entry.get("official_repository_commit") or "")
    cloneable = _is_cloneable_url(repository_url)
    if str(entry.get("source_status") or "") == "repository_local_algorithm":
        return {
            "baseline_id": baseline_id,
            "source_dir": source_dir_value,
            "clone_operation_status": "not_applicable",
            "clone_failure_reason": "repository_local_algorithm",
            "source_dir_exists": source_dir.is_dir(),
        }
    if not cloneable:
        return {
            "baseline_id": baseline_id,
            "source_dir": source_dir_value,
            "clone_operation_status": "not_cloneable",
            "clone_failure_reason": "official_repository_url_not_git_cloneable",
            "source_dir_exists": source_dir.is_dir(),
        }
    if not execute_clone:
        return {
            "baseline_id": baseline_id,
            "source_dir": source_dir_value,
            "clone_operation_status": "planned_not_executed",
            "clone_failure_reason": "execute_clone_false",
            "source_dir_exists": source_dir.is_dir(),
            "planned_repository_url": repository_url,
            "target_repository_branch": branch,
            "target_repository_commit": commit,
        }
    source_dir.parent.mkdir(parents=True, exist_ok=True)
    if source_dir.exists() and (source_dir / ".git").is_dir():
        fetch_result = _run_git_command(["git", "-C", str(source_dir), "fetch", "--all", "--prune"])
        checkout_results = _checkout_pinned_source_reference(entry, source_dir)
        checkout_failed = any(result["return_code"] != 0 for result in checkout_results)
        return {
            "baseline_id": baseline_id,
            "source_dir": source_dir_value,
            "clone_operation_status": "updated" if fetch_result["return_code"] == 0 and not checkout_failed else "failed",
            "clone_failure_reason": "none" if fetch_result["return_code"] == 0 and not checkout_failed else ("git_checkout_failed" if checkout_failed else "git_fetch_failed"),
            "source_dir_exists": source_dir.is_dir(),
            "target_repository_branch": branch,
            "target_repository_commit": commit,
            "git_results": [fetch_result, *checkout_results],
        }
    if source_dir.exists():
        return {
            "baseline_id": baseline_id,
            "source_dir": source_dir_value,
            "clone_operation_status": "failed",
            "clone_failure_reason": "source_dir_exists_but_not_git_repository",
            "source_dir_exists": True,
        }
    command = ["git", "clone"]
    concrete_branch = _concrete_git_reference(branch)
    if concrete_branch:
        command.extend(["--branch", concrete_branch])
    command.extend([repository_url, str(source_dir)])
    clone_result = _run_git_command(command)
    checkout_results = _checkout_pinned_source_reference(entry, source_dir) if clone_result["return_code"] == 0 else []
    checkout_failed = any(result["return_code"] != 0 for result in checkout_results)
    return {
        "baseline_id": baseline_id,
        "source_dir": source_dir_value,
        "clone_operation_status": "cloned" if clone_result["return_code"] == 0 and not checkout_failed else "failed",
        "clone_failure_reason": "none" if clone_result["return_code"] == 0 and not checkout_failed else ("git_checkout_failed" if checkout_failed else "git_clone_failed"),
        "source_dir_exists": source_dir.is_dir(),
        "target_repository_branch": branch,
        "target_repository_commit": commit,
        "git_results": [clone_result, *checkout_results],
    }


def build_clone_results_manifest(
    registry_path: str | Path = DEFAULT_SOURCE_REGISTRY_PATH,
    repo_root: str | Path = ".",
    execute_clone: bool = False,
) -> dict[str, Any]:
    """构造或执行外部 baseline source clone 计划。"""
    root = Path(repo_root)
    registry = load_source_registry(registry_path)
    results = [_clone_or_update_one(entry, root, execute_clone) for entry in _safe_list(registry.get("baseline_sources"))]
    executed_results = [item for item in results if item["clone_operation_status"] in {"cloned", "updated", "failed"}]
    failed_results = [item for item in results if item["clone_operation_status"] == "failed"]
    return {
        "artifact_name": CLONE_RESULTS_NAME,
        "manifest_kind": "external_baseline_source_clone_results",
        "registry_path": str(registry_path),
        "execute_clone": bool(execute_clone),
        "clone_result_count": len(results),
        "clone_executed_count": len(executed_results),
        "clone_failed_count": len(failed_results),
        "clone_results_decision": "FAIL" if failed_results else "PASS",
        "claim_support_status": "source_clone_manifest_only_not_claim_evidence",
        "clone_results": results,
    }


def build_external_baseline_table_plan(
    registry_path: str | Path = DEFAULT_SOURCE_REGISTRY_PATH,
) -> dict[str, Any]:
    """由 source registry 构造 external baseline table plan。"""
    registry = load_source_registry(registry_path)
    methods: list[dict[str, Any]] = []
    for entry in _safe_list(registry.get("baseline_sources")):
        baseline_id = str(entry.get("baseline_id") or "")
        layer = "modern_external_baseline" if baseline_id in MODERN_BASELINE_IDS else "explicit_synchronization_control"
        methods.append({
            "method_id": baseline_id,
            "display_name": entry.get("baseline_name"),
            "table_role": "primary_modern_video_watermark_baseline" if layer == "modern_external_baseline" else "synchronization_control",
            "comparison_layer": layer,
            "source_url": entry.get("official_repository_url"),
            "local_source_root": entry.get("source_dir"),
            "adapter_path": entry.get("adapter_path"),
            "integration_status": entry.get("adapter_status"),
            "paper_claim_support": bool(entry.get("paper_claim_support")),
            "claim_boundary": "formal_measured_required" if layer == "modern_external_baseline" else "control_only_not_positive_claim",
        })
    return {
        "artifact_name": "external_baseline_table_plan.json",
        "manifest_kind": "external_baseline_table_plan",
        "registry_path": str(registry_path),
        "method_count": len(methods),
        "modern_external_baseline_count": sum(1 for item in methods if item["comparison_layer"] == "modern_external_baseline"),
        "explicit_synchronization_control_count": sum(1 for item in methods if item["comparison_layer"] == "explicit_synchronization_control"),
        "claim_support_status": "table_plan_only_not_claim_evidence",
        "methods": methods,
    }


def write_source_intake_artifacts(
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    registry_path: str | Path = DEFAULT_SOURCE_REGISTRY_PATH,
    repo_root: str | Path = ".",
    execute_clone: bool = False,
) -> dict[str, Any]:
    """写出 source intake、inspection、clone results 和 table plan 四类治理文件。"""
    output = Path(output_root)
    clone_manifest = build_clone_results_manifest(registry_path, repo_root, execute_clone=execute_clone)
    intake_manifest = build_source_intake_manifest(registry_path, repo_root)
    inspection_manifest = build_source_inspection_manifest(registry_path, repo_root)
    table_plan = build_external_baseline_table_plan(registry_path)
    write_json(output / INTAKE_MANIFEST_NAME, intake_manifest)
    write_json(output / SOURCE_INSPECTION_NAME, inspection_manifest)
    write_json(output / CLONE_RESULTS_NAME, clone_manifest)
    write_json(output / TABLE_PLAN_RELATIVE_PATH, table_plan)
    return {
        "external_baseline_source_intake_decision": intake_manifest["external_baseline_source_intake_decision"],
        "source_intake_manifest_path": str(output / INTAKE_MANIFEST_NAME),
        "source_inspection_manifest_path": str(output / SOURCE_INSPECTION_NAME),
        "clone_results_manifest_path": str(output / CLONE_RESULTS_NAME),
        "table_plan_path": str(output / TABLE_PLAN_RELATIVE_PATH),
        "source_intake_ready_count": intake_manifest["source_intake_ready_count"],
        "source_intake_missing_count": intake_manifest["source_intake_missing_count"],
        "modern_external_baseline_source_ready_count": intake_manifest["modern_external_baseline_source_ready_count"],
    }


def existing_evidence_paths(raw_value: str | None = None) -> list[str]:
    """解析外部 baseline evidence paths, 只保留当前文件系统中存在的路径。"""
    value = raw_value if raw_value is not None else os.environ.get("SSTW_EXTERNAL_BASELINE_EVIDENCE_PATHS", "")
    candidates = [item for item in value.split(os.pathsep) if item.strip()]
    return [str(Path(item).resolve()) for item in candidates if Path(item).exists()]


def persisted_official_command_evidence_paths(run_root: str | Path) -> list[str]:
    """收集现代 baseline command adapter 在 run_root 中持久化的官方输出证据。

    该函数属于项目特定写法。它把 Colab 中真实执行第三方 baseline 后留下的
    `official_output.json`、stdout / stderr 和 command manifest 绑定到 execution manifest,
    避免 measured_formal records 只剩聚合分数。
    """
    evidence_root = Path(run_root) / OFFICIAL_COMMAND_EVIDENCE_RELATIVE_ROOT
    if not evidence_root.is_dir():
        return []
    paths = [
        path
        for path in evidence_root.rglob("*")
        if path.is_file() and path.name in {
            "official_output.json",
            "official_stdout.txt",
            "official_stderr.txt",
            "official_command_manifest.json",
        }
    ]
    return [str(path.resolve()) for path in sorted(paths)]


def summarize_records_for_execution_manifest(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """从 comparison records 汇总 external baseline execution manifest 所需字段。"""
    materialized = [dict(record) for record in records]
    measured = [record for record in materialized if record.get("metric_status") in {"measured_proxy", "measured_formal"}]
    formal = [record for record in materialized if record.get("metric_status") == "measured_formal"]
    modern_formal_names = sorted({
        str(record.get("external_baseline_name"))
        for record in formal
        if record.get("external_baseline_layer") == "modern_external_baseline"
    })
    return {
        "external_baseline_comparison_record_count": len(materialized),
        "external_baseline_measured_adapter_count": len({str(record.get("external_baseline_name")) for record in measured if record.get("external_baseline_name")}),
        "external_baseline_formal_measured_adapter_count": len({str(record.get("external_baseline_name")) for record in formal if record.get("external_baseline_name")}),
        "modern_external_baseline_formal_measured_adapter_count": len(modern_formal_names),
        "modern_external_baseline_formal_measured_adapter_names": modern_formal_names,
        "external_baseline_result_used_for_claim": any(bool(record.get("external_baseline_result_used_for_claim")) for record in formal),
    }


def build_execution_manifest(
    records: Iterable[Mapping[str, Any]],
    *,
    run_root: str | Path,
    config_path: str | Path,
    source_intake_manifest_path: str | Path | None = None,
    evidence_paths: Iterable[str] | None = None,
) -> dict[str, Any]:
    """构造 external baseline execution manifest。

    该 manifest 记录本次 run_root 中 baseline comparison records 的执行边界。它不把
    unsupported 或 proxy 结果伪装为论文证据; 现代 baseline 只有 measured_formal 且有外部证据
    绑定时, 才能在后续 claim gate 中升级为正式主表证据。
    """
    summary = summarize_records_for_execution_manifest(records)
    configured_evidence_paths = list(evidence_paths if evidence_paths is not None else existing_evidence_paths())
    persisted_evidence_paths = persisted_official_command_evidence_paths(run_root)
    materialized_evidence_paths = []
    seen_paths: set[str] = set()
    for path in [*configured_evidence_paths, *persisted_evidence_paths]:
        if path not in seen_paths:
            materialized_evidence_paths.append(path)
            seen_paths.add(path)
    formal_rows_present = summary["external_baseline_formal_measured_adapter_count"] > 0
    return {
        "artifact_name": "external_baseline_execution_manifest.json",
        "manifest_kind": "external_baseline_execution",
        "producer_id": "sstw_external_baseline_adapter_runner",
        "producer_role": "external_baseline_adapter_execution",
        "run_root": str(run_root),
        "config_path": str(config_path),
        "source_intake_manifest_path": str(source_intake_manifest_path) if source_intake_manifest_path else None,
        "formal_result_claim": bool(formal_rows_present and materialized_evidence_paths),
        "formal_evidence_status": "evidence_paths_bound" if materialized_evidence_paths else ("formal_rows_without_external_evidence_paths" if formal_rows_present else "no_formal_rows"),
        "evidence_paths": materialized_evidence_paths,
        "evidence_path_count": len(materialized_evidence_paths),
        "execution_boundary": "governed_adapter_records_from_run_root",
        "claim_support_status": "external_baseline_execution_manifest_written",
        **summary,
    }
