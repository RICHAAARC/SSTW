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
SERVER_WORKFLOW_DECISION_MANIFEST_KIND = (
    "generative_video_server_workflow_decision"
)
TRAJECTORY_REPLAY_SOURCE_BUILD_TEST_ID = "trajectory_replay_smoke_source_build"
TRAJECTORY_SIGNAL_TEST_ID = "trajectory_signal_localization_diagnostic"
CONTROLLED_EMBEDDING_STRENGTH_TEST_ID = (
    "controlled_embedding_strength_diagnostic"
)
MINIMAL_SIGNED_TRAJECTORY_STATE_SPACE_SMOKE_TEST_ID = (
    "minimal_signed_trajectory_state_space_smoke"
)
SUPPORTED_TEST_IDS = (
    TRAJECTORY_REPLAY_SOURCE_BUILD_TEST_ID,
    TRAJECTORY_SIGNAL_TEST_ID,
    CONTROLLED_EMBEDDING_STRENGTH_TEST_ID,
    MINIMAL_SIGNED_TRAJECTORY_STATE_SPACE_SMOKE_TEST_ID,
)
TRAJECTORY_REPLAY_SOURCE_BUILD_PHASE = "source_build"
SUPPORTED_TRAJECTORY_PHASES = (
    "no_attack",
    "attacked",
    "decision",
)
SUPPORTED_CONTROLLED_EMBEDDING_PHASES = ("no_attack",)
SUPPORTED_MINIMAL_SIGNED_TRAJECTORY_PHASES = ("no_attack",)
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
    if test_id == TRAJECTORY_REPLAY_SOURCE_BUILD_TEST_ID:
        supported_phases = (TRAJECTORY_REPLAY_SOURCE_BUILD_PHASE,)
    elif test_id == CONTROLLED_EMBEDDING_STRENGTH_TEST_ID:
        supported_phases = SUPPORTED_CONTROLLED_EMBEDDING_PHASES
    elif test_id == MINIMAL_SIGNED_TRAJECTORY_STATE_SPACE_SMOKE_TEST_ID:
        supported_phases = SUPPORTED_MINIMAL_SIGNED_TRAJECTORY_PHASES
    else:
        supported_phases = SUPPORTED_TRAJECTORY_PHASES
    if phase not in supported_phases:
        raise ValueError(
            f"Colab test phase 不受支持: {phase!r}; "
            f"允许值: {supported_phases}"
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
    if test_id == TRAJECTORY_REPLAY_SOURCE_BUILD_TEST_ID and resume_package:
        raise ValueError("trajectory replay source build 不接受 resume package")
    if test_id == CONTROLLED_EMBEDDING_STRENGTH_TEST_ID and resume_package:
        raise ValueError(
            "controlled embedding strength diagnostic 不接受 resume package"
        )
    if (
        test_id == MINIMAL_SIGNED_TRAJECTORY_STATE_SPACE_SMOKE_TEST_ID
        and resume_package
    ):
        raise ValueError(
            "minimal signed trajectory smoke 不接受 resume package"
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
            if path.is_file() and not path.is_symlink():
                archive.write(path, path.relative_to(source_root).as_posix())


def package_colab_test_recovery_bundle(
    request_path: str | Path,
    *,
    project_root: str | Path,
    repo_root: str | Path,
    local_runtime_root: str | Path,
    local_workspace_root: str | Path,
    local_package_cache_root: str | Path,
    run_decision_path: str | Path | None = None,
) -> dict[str, Any]:
    """把失败后仍存在的本地 checkpoint 打成非正式排障包并写入 Drive。"""

    root = Path(project_root).expanduser().resolve()
    repository_root = Path(repo_root).expanduser().resolve()
    runtime_root = Path(local_runtime_root).expanduser().resolve()
    workspace_root = Path(local_workspace_root).expanduser().resolve()
    cache_root = Path(local_package_cache_root).expanduser().resolve()
    if not runtime_root.is_dir():
        raise FileNotFoundError(
            f"Colab recovery 可信本地运行根不存在: {runtime_root}"
        )
    for path, label in (
        (workspace_root, "local_workspace_root"),
        (cache_root, "local_package_cache_root"),
    ):
        if path == runtime_root or not _path_is_within(path, runtime_root):
            raise ValueError(
                f"{label} 必须位于可信本地运行根内且不能等于根目录: {path}"
            )
        if _path_is_within(path, root):
            raise ValueError(f"{label} 不得位于 Drive SSTW 根目录内: {path}")

    resolved = load_colab_test_request(request_path, project_root=root)
    output_root = (
        workspace_root
        / "runs"
        / str(resolved["test_id"])
        / str(resolved["run_series_id"])
    )
    validation_root = (
        workspace_root / "validation" / str(resolved["run_series_id"])
    )
    decision_path: Path | None = None
    if run_decision_path is not None and str(run_decision_path).strip():
        unresolved_decision_path = Path(run_decision_path).expanduser()
        if unresolved_decision_path.is_symlink():
            raise ValueError(
                "run_decision_path 必须是非 symlink 普通 JSON 文件"
            )
        decision_path = unresolved_decision_path.resolve()
        if not _path_is_within(decision_path, runtime_root):
            raise ValueError(
                "run_decision_path 必须位于可信本地运行根内: "
                f"{decision_path}"
            )
        if _path_is_within(decision_path, root):
            raise ValueError(
                f"run_decision_path 不得位于 Drive SSTW 根目录内: {decision_path}"
            )
        if decision_path.suffix.lower() != ".json" or not decision_path.is_file():
            raise ValueError(
                "run_decision_path 必须是非 symlink 普通 JSON 文件: "
                f"{decision_path}"
            )
        try:
            run_decision = json.loads(
                decision_path.read_text(encoding="utf-8-sig")
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"run_decision_path 不是有效 JSON: {decision_path}"
            ) from exc
        if not isinstance(run_decision, dict):
            raise ValueError("run_decision_path JSON 顶层必须是对象")
        expected_decision_fields = {
            "manifest_kind": SERVER_WORKFLOW_DECISION_MANIFEST_KIND,
            "workflow_profile": "colab_test",
            "pipeline": "colab_test",
            "server_workflow_decision": "FAIL",
        }
        mismatches = [
            field
            for field, expected in expected_decision_fields.items()
            if run_decision.get(field) != expected
        ]
        if mismatches:
            raise ValueError(
                "run_decision_path 必须是明确失败的 colab_test server workflow "
                "decision; mismatched_fields="
                + ",".join(mismatches)
            )
    recoverable_files: list[tuple[Path, str]] = []
    for source_root, archive_prefix in (
        (output_root, "partial_run"),
        (validation_root, "partial_validation"),
    ):
        if not source_root.is_dir():
            continue
        for path in sorted(source_root.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            archive_name = (
                Path(archive_prefix) / path.relative_to(source_root)
            ).as_posix()
            recoverable_files.append((path, archive_name))
    if (
        decision_path is not None
        and decision_path.is_file()
        and not decision_path.is_symlink()
    ):
        recoverable_files.append(
            (decision_path, "diagnostics/server_workflow_decision.json")
        )
    if not recoverable_files:
        raise FileNotFoundError(
            "Colab recovery 未找到 partial run、validation 或 failure decision"
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    repository_commit = _repository_commit(repository_root)
    run_id = f"{timestamp}_{repository_commit[:8]}"
    package_name = (
        f"colab_test_recovery_{resolved['test_id']}_{resolved['phase']}_{run_id}.zip"
    )
    local_result_root = cache_root / "recovery" / run_id
    local_package_path = local_result_root / package_name
    local_package_path.parent.mkdir(parents=True, exist_ok=False)
    included_entries: list[str] = []
    with ZipFile(local_package_path, "w", compression=ZIP_DEFLATED) as archive:
        for source_path, archive_name in recoverable_files:
            archive.write(source_path, archive_name)
            included_entries.append(archive_name)

    manifest = {
        "manifest_kind": "sstw_colab_test_recovery_manifest",
        "request_schema_version": REQUEST_SCHEMA_VERSION,
        "test_id": resolved["test_id"],
        "phase": resolved["phase"],
        "run_series_id": resolved["run_series_id"],
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_url": resolved["repository_url"],
        "repository_ref": resolved["repository_ref"],
        "repository_commit": repository_commit,
        "request_path": resolved["request_path"],
        "local_runtime_root": str(runtime_root),
        "local_output_root": str(output_root),
        "local_validation_root": str(validation_root),
        "run_decision_path": "" if decision_path is None else str(decision_path),
        "included_entries": included_entries,
        "formal_result": False,
        "stage_progression_allowed": False,
        "claim_support_status": "failure_recovery_only_not_claim_evidence",
    }
    local_manifest_path = local_result_root / "colab_test_recovery_manifest.json"
    write_json(local_manifest_path, manifest)

    drive_output_root = (
        root
        / "diagnostic_tests"
        / str(resolved["test_id"])
        / "recovery"
        / run_id
    )
    drive_package_path = drive_output_root / package_name
    drive_manifest_path = drive_output_root / "colab_test_recovery_manifest.json"
    drive_output_root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(local_package_path, drive_package_path)
    shutil.copy2(local_manifest_path, drive_manifest_path)
    return {
        "notebook_role": "colab_test_recovery",
        "test_id": resolved["test_id"],
        "phase": resolved["phase"],
        "run_series_id": resolved["run_series_id"],
        "run_id": run_id,
        "recovery_package_status": "partial_diagnostic_packaged",
        "drive_result_zip": str(drive_package_path),
        "drive_result_manifest": str(drive_manifest_path),
        "formal_result": False,
        "stage_progression_allowed": False,
        "claim_support_status": "failure_recovery_only_not_claim_evidence",
    }


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


def _default_trajectory_source_builder(
    upstream_package: Path,
    output_root: Path,
) -> dict[str, Any]:
    from experiments.generative_video_model_probe.trajectory_replay_smoke import (
        run_trajectory_replay_smoke,
    )

    repository_root = Path(__file__).resolve().parents[1]
    return run_trajectory_replay_smoke(
        upstream_package,
        output_root,
        repository_root / "configs/protocol/sstw_minimal_trajectory_paper.json",
    )


def _default_trajectory_source_validator(
    source_root: Path,
    validation_output_root: Path,
) -> dict[str, Any]:
    from experiments.generative_video_model_probe.trajectory_signal_localization_diagnostic import (
        build_immutable_input_snapshot,
    )

    repository_root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (
            repository_root
            / "configs/protocol/sstw_trajectory_signal_localization_diagnostic.json"
        ).read_text(encoding="utf-8-sig")
    )
    return build_immutable_input_snapshot(
        source_root,
        validation_output_root,
        config,
        phase="no_attack",
    )


def _default_controlled_embedding_runner(
    source_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    from experiments.generative_video_model_probe.controlled_embedding_strength_diagnostic import (
        run_controlled_embedding_strength_diagnostic,
    )

    return run_controlled_embedding_strength_diagnostic(
        source_root,
        output_root,
    )


def _default_minimal_signed_trajectory_runner(
    source_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    from experiments.generative_video_model_probe.minimal_signed_trajectory_state_space_smoke import (
        run_minimal_signed_trajectory_state_space_smoke,
    )

    return run_minimal_signed_trajectory_state_space_smoke(
        source_root,
        output_root,
    )


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


def _controlled_embedding_generation_model_ids(
    source_root: Path,
) -> list[str]:
    candidates = list(
        source_root.rglob("records/controlled_embedding_generation_plan.jsonl")
    )
    if len(candidates) != 1:
        raise RuntimeError(
            "controlled embedding input 必须唯一包含 construction plan"
        )
    rows = [
        json.loads(line)
        for line in candidates[0].read_text(
            encoding="utf-8-sig"
        ).splitlines()
        if line.strip()
    ]
    return sorted(
        {
            str(row.get("generation_model_id") or "").strip()
            for row in rows
            if str(row.get("generation_model_id") or "").strip()
        }
    )


def _minimal_signed_trajectory_generation_model_ids(
    source_root: Path,
) -> list[str]:
    candidates = list(source_root.rglob("records/generation_records.jsonl"))
    if len(candidates) != 1:
        raise RuntimeError(
            "minimal signed trajectory input 必须唯一包含 generation records"
        )
    rows = [
        json.loads(line)
        for line in candidates[0].read_text(
            encoding="utf-8-sig"
        ).splitlines()
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
    claim_support_status = (
        "trajectory_replay_source_build_only_not_paper_evidence"
        if resolved["test_id"] == TRAJECTORY_REPLAY_SOURCE_BUILD_TEST_ID
        else "diagnostic_only_not_paper_evidence"
    )
    return {
        "notebook_role": "colab_test",
        "test_id": resolved["test_id"],
        "phase": resolved["phase"],
        "run_series_id": resolved["run_series_id"],
        "request_path": resolved["request_path"],
        "source_package_path": resolved["source_package_path"],
        "resume_package_path": resolved["resume_package_path"],
        "stage_execution_kind": "allowlisted_colab_test_request",
        "claim_support_status": claim_support_status,
    }


def _run_trajectory_replay_source_build(
    resolved: Mapping[str, Any],
    *,
    project_root: Path,
    workspace_root: Path,
    cache_root: Path,
    repository_commit: str,
    run_id: str,
    source_builder: Callable[..., dict[str, Any]],
    source_validator: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    """在本地重建 Stage 0-D source，通过完整性校验后才发布 Drive。"""

    drive_output_root = (
        project_root / "inputs" / "trajectory_signal_localization"
    )
    drive_package_path = drive_output_root / "stage0d_source.zip"
    manifest_path = drive_output_root / "stage0d_source_manifest.json"
    existing_targets = [
        path for path in (drive_package_path, manifest_path) if path.exists()
    ]
    if existing_targets:
        raise FileExistsError(
            "Stage 0-D source Drive 目标已存在，拒绝覆盖: "
            + ", ".join(str(path) for path in existing_targets)
        )

    upstream_package = Path(str(resolved["source_package_path"]))
    cached_upstream = cache_root / (
        f"source_build_upstream_{resolved['run_series_id']}.zip"
    )
    cached_upstream.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(upstream_package, cached_upstream)

    output_root = (
        workspace_root
        / "runs"
        / str(resolved["test_id"])
        / str(resolved["run_series_id"])
    )
    if output_root.exists():
        raise FileExistsError(
            f"trajectory replay source build 需要新的本地 output root: {output_root}"
        )
    source_build_decision = source_builder(cached_upstream, output_root)
    validation_output_root = (
        workspace_root
        / "validation"
        / str(resolved["run_series_id"])
    )
    snapshot = source_validator(output_root, validation_output_root)
    required_snapshot = {
        "immutable_input_preflight_status": "ready",
        "immutable_input_scope": "full_replay_diagnostic_inputs",
        "generation_record_count": 12,
        "attack_record_count": 24,
        "likelihood_calibration_input_status": "ready",
    }
    mismatches = [
        name
        for name, expected in required_snapshot.items()
        if snapshot.get(name) != expected
    ]
    if mismatches:
        raise RuntimeError(
            "trajectory replay source build 输入快照未就绪: "
            + ", ".join(mismatches)
        )

    generation_model_ids = _source_generation_model_ids(output_root)
    local_result_root = cache_root / "results" / run_id
    local_package_path = local_result_root / "stage0d_source.zip"
    _write_zip(output_root, local_package_path)
    manifest = {
        "manifest_kind": "sstw_colab_test_source_package_manifest",
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
        "source_package_path": str(upstream_package),
        "resume_package_path": "",
        "drive_result_zip": str(drive_package_path),
        "immutable_input_preflight_status": snapshot.get(
            "immutable_input_preflight_status"
        ),
        "immutable_input_scope": snapshot.get("immutable_input_scope"),
        "generation_record_count": snapshot.get("generation_record_count"),
        "attack_record_count": snapshot.get("attack_record_count"),
        "likelihood_calibration_input_status": snapshot.get(
            "likelihood_calibration_input_status"
        ),
        "diagnostic_decision": source_build_decision,
        "claim_support_status": (
            "trajectory_replay_source_build_only_not_paper_evidence"
        ),
    }
    local_manifest_path = local_result_root / "stage0d_source_manifest.json"
    write_json(local_manifest_path, manifest)

    drive_output_root.mkdir(parents=True, exist_ok=True)
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
        "diagnostic_decision": source_build_decision,
        "claim_support_status": (
            "trajectory_replay_source_build_only_not_paper_evidence"
        ),
    }


def run_colab_test_request(
    request_path: str | Path,
    *,
    project_root: str | Path,
    repo_root: str | Path,
    local_workspace_root: str | Path,
    local_package_cache_root: str | Path,
    trajectory_runner: Callable[..., dict[str, Any]] | None = None,
    trajectory_source_builder: Callable[..., dict[str, Any]] | None = None,
    trajectory_source_validator: Callable[..., dict[str, Any]] | None = None,
    controlled_embedding_runner: Callable[..., dict[str, Any]] | None = None,
    minimal_signed_trajectory_runner: (
        Callable[..., dict[str, Any]] | None
    ) = None,
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

    if resolved["test_id"] == TRAJECTORY_REPLAY_SOURCE_BUILD_TEST_ID:
        return _run_trajectory_replay_source_build(
            resolved,
            project_root=root,
            workspace_root=workspace_root,
            cache_root=cache_root,
            repository_commit=repository_commit,
            run_id=run_id,
            source_builder=(
                trajectory_source_builder or _default_trajectory_source_builder
            ),
            source_validator=(
                trajectory_source_validator
                or _default_trajectory_source_validator
            ),
        )

    source_package = Path(resolved["source_package_path"])
    cached_source = cache_root / f"source_{resolved['run_series_id']}.zip"
    cached_source.parent.mkdir(parents=True, exist_ok=True)
    if not cached_source.exists():
        shutil.copy2(source_package, cached_source)
    source_extract_root = workspace_root / "inputs" / resolved["run_series_id"]
    if not source_extract_root.exists():
        _safe_extract_zip(cached_source, source_extract_root)
    if resolved["test_id"] == CONTROLLED_EMBEDDING_STRENGTH_TEST_ID:
        source_root = source_extract_root
        generation_model_ids = _controlled_embedding_generation_model_ids(
            source_root
        )
    elif (
        resolved["test_id"]
        == MINIMAL_SIGNED_TRAJECTORY_STATE_SPACE_SMOKE_TEST_ID
    ):
        source_root = source_extract_root
        generation_model_ids = (
            _minimal_signed_trajectory_generation_model_ids(source_root)
        )
    else:
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

    if resolved["test_id"] == CONTROLLED_EMBEDDING_STRENGTH_TEST_ID:
        runner = (
            controlled_embedding_runner
            or _default_controlled_embedding_runner
        )
        diagnostic_decision = runner(source_root, output_root)
    elif (
        resolved["test_id"]
        == MINIMAL_SIGNED_TRAJECTORY_STATE_SPACE_SMOKE_TEST_ID
    ):
        runner = (
            minimal_signed_trajectory_runner
            or _default_minimal_signed_trajectory_runner
        )
        diagnostic_decision = runner(source_root, output_root)
    else:
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
