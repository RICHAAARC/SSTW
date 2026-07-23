"""执行由 Google Drive 请求文件驱动的受控 Colab 测试。

Notebook 只读取仓库地址与 ref，并把请求路径交给服务器 CLI。本模块负责
白名单分派、输入包校验、本地热路径执行以及结果 zip/manifest 回写 Drive。
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
from typing import Any, Callable, Mapping
from zipfile import ZIP_DEFLATED, ZipFile

from evaluation.protocol.record_writer import write_json


REQUEST_SCHEMA_VERSION = "sstw_colab_test_request_v1"
TRAJECTORY_SIGNAL_TEST_ID = "trajectory_signal_localization_diagnostic"
SUPPORTED_TEST_IDS = (TRAJECTORY_SIGNAL_TEST_ID,)
SUPPORTED_TRAJECTORY_PHASES = (
    "no_attack",
    "attacked",
    "decision",
)
EXPECTED_REPOSITORY_URL = "https://github.com/RICHAAARC/SSTW.git"
_SAFE_REPOSITORY_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
_SAFE_RUN_SERIES_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{2,63}$")


def _require_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} 必须是 JSON 对象")
    return dict(value)


def _reject_unknown_fields(
    value: Mapping[str, Any],
    allowed: set[str],
    label: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{label} 包含未授权字段: {', '.join(unknown)}")


def _path_within_project_root(
    raw_path: object,
    project_root: Path,
    label: str,
    *,
    required: bool,
) -> Path | None:
    text = str(raw_path or "").strip()
    if not text:
        if required:
            raise ValueError(f"{label} 不能为空")
        return None
    path = Path(text).expanduser().resolve()
    try:
        path.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(f"{label} 必须位于 Drive SSTW 根目录内: {path}") from exc
    if path.suffix.lower() != ".zip":
        raise ValueError(f"{label} 必须是 zip 文件: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} 不存在: {path}")
    return path


def load_colab_test_request(
    request_path: str | Path,
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    """读取并 fail-closed 校验 Drive 请求；不允许命令、模块或脚本注入。"""

    root = Path(project_root).expanduser().resolve()
    path = Path(request_path).expanduser().resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Colab 请求文件必须位于 Drive SSTW 根目录内: {path}") from exc
    payload = _require_mapping(
        json.loads(path.read_text(encoding="utf-8-sig")),
        "Colab test request",
    )
    _reject_unknown_fields(
        payload,
        {"request_schema_version", "test_id", "repository", "parameters"},
        "Colab test request",
    )
    if payload.get("request_schema_version") != REQUEST_SCHEMA_VERSION:
        raise ValueError(
            f"request_schema_version 必须是 {REQUEST_SCHEMA_VERSION}"
        )
    test_id = str(payload.get("test_id") or "").strip()
    if test_id not in SUPPORTED_TEST_IDS:
        raise ValueError(
            f"test_id 不在仓库白名单中: {test_id!r}; 允许值: {SUPPORTED_TEST_IDS}"
        )

    repository = _require_mapping(payload.get("repository"), "repository")
    _reject_unknown_fields(repository, {"url", "ref"}, "repository")
    repository_url = str(repository.get("url") or "").strip()
    repository_ref = str(repository.get("ref") or "").strip()
    if repository_url != EXPECTED_REPOSITORY_URL:
        raise ValueError(f"repository.url 必须是 {EXPECTED_REPOSITORY_URL}")
    if (
        not _SAFE_REPOSITORY_REF.fullmatch(repository_ref)
        or ".." in repository_ref
        or repository_ref.endswith("/")
    ):
        raise ValueError("repository.ref 不是安全的 Git revision")

    parameters = _require_mapping(payload.get("parameters"), "parameters")
    _reject_unknown_fields(
        parameters,
        {
            "phase",
            "run_series_id",
            "source_package_path",
            "resume_package_path",
        },
        "parameters",
    )
    phase = str(parameters.get("phase") or "").strip()
    if phase not in SUPPORTED_TRAJECTORY_PHASES:
        raise ValueError(
            f"trajectory diagnostic phase 不受支持: {phase!r}; "
            f"允许值: {SUPPORTED_TRAJECTORY_PHASES}"
        )
    run_series_id = str(parameters.get("run_series_id") or "").strip()
    if not _SAFE_RUN_SERIES_ID.fullmatch(run_series_id):
        raise ValueError(
            "run_series_id 必须是3到64位小写字母、数字、下划线或连字符"
        )
    source_package = _path_within_project_root(
        parameters.get("source_package_path"),
        root,
        "source_package_path",
        required=True,
    )
    resume_package = _path_within_project_root(
        parameters.get("resume_package_path"),
        root,
        "resume_package_path",
        required=phase in {"attacked", "decision"},
    )
    return {
        "request_path": str(path),
        "request": payload,
        "test_id": test_id,
        "repository_url": repository_url,
        "repository_ref": repository_ref,
        "phase": phase,
        "run_series_id": run_series_id,
        "source_package_path": str(source_package),
        "resume_package_path": str(resume_package) if resume_package else "",
    }


def _safe_extract_zip(package_path: Path, destination: Path) -> None:
    """拒绝路径穿越和符号链接后解压受控输入包。"""

    with ZipFile(package_path) as archive:
        for member in archive.infolist():
            name = PurePosixPath(member.filename)
            mode = member.external_attr >> 16
            if name.is_absolute() or ".." in name.parts:
                raise ValueError(f"zip 包含不安全路径: {member.filename}")
            if stat.S_ISLNK(mode):
                raise ValueError(f"zip 包含不允许的符号链接: {member.filename}")
        destination.mkdir(parents=True, exist_ok=False)
        archive.extractall(destination)


def _discover_stage0d_source_root(extracted_root: Path) -> Path:
    candidates: list[Path] = []
    for generation_path in extracted_root.rglob("records/generation_records.jsonl"):
        candidate = generation_path.parent.parent
        if (candidate / "datasets" / "prompt_seed_suite.json").is_file():
            candidates.append(candidate.resolve())
    unique = sorted(set(candidates))
    if len(unique) != 1:
        raise RuntimeError(
            "输入 zip 必须唯一包含 records/generation_records.jsonl 与 "
            f"datasets/prompt_seed_suite.json；observed_roots={unique}"
        )
    return unique[0]


def _repository_commit(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
    ).strip()


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def build_colab_test_runtime_preflight_decision(
    *,
    project_root: str | Path,
    local_workspace_root: str | Path,
    local_package_cache_root: str | Path,
    cuda_available: bool | None = None,
    hf_home: str | Path | None = None,
    hf_hub_cache: str | Path | None = None,
) -> dict[str, Any]:
    """只检查 GPU 可用性和 Drive/本地路径边界，不执行论文环境锁。"""

    drive_root = Path(project_root).expanduser().resolve()
    workspace_root = Path(local_workspace_root).expanduser().resolve()
    package_cache_root = Path(local_package_cache_root).expanduser().resolve()
    resolved_hf_home = str(hf_home or os.environ.get("HF_HOME") or "").strip()
    resolved_hf_hub_cache = str(
        hf_hub_cache or os.environ.get("HF_HUB_CACHE") or ""
    ).strip()
    failures: list[str] = []

    if cuda_available is None:
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
        except Exception:
            cuda_available = False
    if not cuda_available:
        failures.append("cuda_unavailable")
    if _path_is_within(workspace_root, drive_root):
        failures.append("local_workspace_root_on_drive")
    if _path_is_within(package_cache_root, drive_root):
        failures.append("local_package_cache_root_on_drive")
    if not resolved_hf_home:
        failures.append("hf_home_not_configured")
    elif _path_is_within(Path(resolved_hf_home), drive_root):
        failures.append("hf_home_on_drive")
    if not resolved_hf_hub_cache:
        failures.append("hf_hub_cache_not_configured")
    elif _path_is_within(Path(resolved_hf_hub_cache), drive_root):
        failures.append("hf_hub_cache_on_drive")

    return {
        "runtime_environment_preflight_kind": "colab_test_lightweight",
        "runtime_environment_preflight_decision": (
            "PASS" if not failures else "FAIL"
        ),
        "runtime_environment_preflight_failures": failures,
        "cuda_available": bool(cuda_available),
        "local_workspace_root": str(workspace_root),
        "local_package_cache_root": str(package_cache_root),
        "hf_home": resolved_hf_home,
        "hf_hub_cache": resolved_hf_hub_cache,
        "formal_runtime_lock_checked": False,
        "claim_support_status": "runtime_check_only_not_claim_evidence",
    }


def _write_zip(source_root: Path, package_path: Path) -> None:
    package_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(package_path, "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(source_root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source_root).as_posix())


def _default_trajectory_runner(
    source_root: Path,
    output_root: Path,
    *,
    phase: str,
) -> dict[str, Any]:
    from experiments.generative_video_model_probe.trajectory_signal_localization_diagnostic import (
        run_stage0d,
    )

    return run_stage0d(source_root, output_root, phase=phase)


def _source_generation_model_ids(source_root: Path) -> list[str]:
    """读取模型地址列表；有效性校验仍由测试 handler 负责。"""

    generation_path = source_root / "records" / "generation_records.jsonl"
    rows = [
        json.loads(line)
        for line in generation_path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    return sorted(
        {
            str(row.get("generation_model_id") or "").strip()
            for row in rows
            if row.get("generation_status") == "success"
            and str(row.get("generation_model_id") or "").strip()
        }
    )


def build_colab_test_dry_run_plan(
    request_path: str | Path,
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    """返回已校验的白名单 dry-run 计划，不解压或执行 GPU 工作。"""

    resolved = load_colab_test_request(request_path, project_root=project_root)
    return {
        "notebook_role": "colab_test",
        "test_id": resolved["test_id"],
        "phase": resolved["phase"],
        "run_series_id": resolved["run_series_id"],
        "request_path": resolved["request_path"],
        "source_package_path": resolved["source_package_path"],
        "resume_package_path": resolved["resume_package_path"],
        "stage_execution_kind": "allowlisted_colab_test_request",
        "claim_support_status": "diagnostic_only_not_paper_evidence",
    }


def run_colab_test_request(
    request_path: str | Path,
    *,
    project_root: str | Path,
    repo_root: str | Path,
    local_workspace_root: str | Path,
    local_package_cache_root: str | Path,
    trajectory_runner: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """执行一个白名单测试，并把唯一结果 zip 与 manifest 回写 Drive。"""

    root = Path(project_root).expanduser().resolve()
    repository_root = Path(repo_root).expanduser().resolve()
    workspace_root = Path(local_workspace_root).expanduser().resolve()
    cache_root = Path(local_package_cache_root).expanduser().resolve()
    resolved = load_colab_test_request(request_path, project_root=root)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    repository_commit = _repository_commit(repository_root)
    run_id = f"{timestamp}_{repository_commit[:8]}"

    source_package = Path(resolved["source_package_path"])
    cached_source = cache_root / f"source_{resolved['run_series_id']}.zip"
    cached_source.parent.mkdir(parents=True, exist_ok=True)
    if not cached_source.exists():
        shutil.copy2(source_package, cached_source)
    source_extract_root = workspace_root / "inputs" / resolved["run_series_id"]
    if not source_extract_root.exists():
        _safe_extract_zip(cached_source, source_extract_root)
    source_root = _discover_stage0d_source_root(source_extract_root)
    generation_model_ids = _source_generation_model_ids(source_root)

    output_root = (
        workspace_root
        / "runs"
        / resolved["test_id"]
        / resolved["run_series_id"]
    )
    if resolved["resume_package_path"]:
        resume_package = Path(resolved["resume_package_path"])
        cached_resume = cache_root / (
            f"resume_{resolved['run_series_id']}_{resolved['phase']}.zip"
        )
        shutil.copy2(resume_package, cached_resume)
        if not output_root.exists():
            _safe_extract_zip(cached_resume, output_root)
    else:
        output_root.mkdir(parents=True, exist_ok=False)

    runner = trajectory_runner or _default_trajectory_runner
    diagnostic_decision = runner(
        source_root,
        output_root,
        phase=resolved["phase"],
    )
    drive_output_root = (
        root / "diagnostic_tests" / resolved["test_id"] / run_id
    )
    package_name = f"{resolved['test_id']}_{resolved['phase']}_{run_id}.zip"
    local_result_root = cache_root / "results" / run_id
    local_package_path = local_result_root / package_name
    _write_zip(output_root, local_package_path)
    drive_package_path = drive_output_root / package_name
    manifest_path = drive_output_root / "colab_test_package_manifest.json"

    manifest = {
        "manifest_kind": "sstw_colab_test_package_manifest",
        "request_schema_version": REQUEST_SCHEMA_VERSION,
        "test_id": resolved["test_id"],
        "phase": resolved["phase"],
        "run_series_id": resolved["run_series_id"],
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_url": resolved["repository_url"],
        "repository_ref": resolved["repository_ref"],
        "repository_commit": repository_commit,
        "generation_model_ids": generation_model_ids,
        "request_path": resolved["request_path"],
        "source_package_path": str(source_package),
        "resume_package_path": resolved["resume_package_path"],
        "drive_result_zip": str(drive_package_path),
        "diagnostic_decision": diagnostic_decision,
        "claim_support_status": "diagnostic_only_not_paper_evidence",
    }
    local_manifest_path = local_result_root / "colab_test_package_manifest.json"
    write_json(local_manifest_path, manifest)

    # Drive 上的最终目录只在本地 ZIP 和 manifest 都完成后才创建。
    drive_output_root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(local_package_path, drive_package_path)
    shutil.copy2(local_manifest_path, manifest_path)
    return {
        "notebook_role": "colab_test",
        "test_id": resolved["test_id"],
        "phase": resolved["phase"],
        "run_series_id": resolved["run_series_id"],
        "run_id": run_id,
        "colab_test_status": "completed",
        "drive_result_zip": str(drive_package_path),
        "drive_result_manifest": str(manifest_path),
        "diagnostic_decision": diagnostic_decision,
        "claim_support_status": "diagnostic_only_not_paper_evidence",
    }
