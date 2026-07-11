"""验证服务器与 Colab 共用的论文运行环境, 并在执行前失败关闭。"""

from __future__ import annotations

from hashlib import sha256
from importlib import metadata
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping


DEFAULT_RUNTIME_LOCK_PATH = Path("requirements/paper_runtime_environment_lock.json")
RUNTIME_ENVIRONMENT_PREFLIGHT_ARTIFACT_RELPATH = Path(
    "artifacts/paper_runtime_environment_preflight_decision.json"
)
IMMUTABLE_COMMIT_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")


def _read_json_object(path: Path) -> dict[str, Any]:
    """读取 JSON 对象, 缺失或类型错误时立即失败。"""

    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"运行环境锁顶层必须是对象: {path}")
    return payload


def load_runtime_environment_lock(
    repo_root: str | Path,
    lock_path: str | Path = DEFAULT_RUNTIME_LOCK_PATH,
) -> tuple[dict[str, Any], Path, str]:
    """读取环境锁并返回内容、绝对路径与 SHA-256 摘要。

    摘要会进入服务器运行清单, 使 Colab 与普通 GPU 服务器能够证明使用了
    同一份依赖和硬件约束, 而不是只比较一个容易漂移的文件名。
    """

    root = Path(repo_root).expanduser().resolve()
    requested = Path(lock_path).expanduser()
    resolved = requested.resolve() if requested.is_absolute() else (root / requested).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"缺少论文运行环境锁: {resolved}")
    raw = resolved.read_bytes()
    return _read_json_object(resolved), resolved, sha256(raw).hexdigest()


def _normalized_distribution_version(value: str) -> str:
    """移除 wheel 的本地构建后缀, 保留受锁定的上游版本。"""

    return str(value).strip().split("+", 1)[0]


def _installed_distribution_versions(names: list[str]) -> dict[str, str | None]:
    """读取已安装 distribution 版本, 不导入重型模型库。"""

    result: dict[str, str | None] = {}
    for name in names:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def _git_command(repo_root: Path, *arguments: str) -> str | None:
    """执行只读 Git 命令; 抽离包没有 Git 元数据时返回 ``None``。"""

    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _repository_provenance(repo_root: Path) -> dict[str, Any]:
    """读取开发仓库或抽离包的不可变代码来源。"""

    commit = _git_command(repo_root, "rev-parse", "HEAD")
    status = _git_command(repo_root, "status", "--porcelain")
    if commit and IMMUTABLE_COMMIT_PATTERN.fullmatch(commit):
        return {
            "repository_provenance_source": "git_worktree",
            "repository_commit": commit.lower(),
            "repository_tree_clean": status == "",
        }

    manifest_path = repo_root / "extraction_manifest.json"
    if not manifest_path.is_file():
        return {
            "repository_provenance_source": "unresolved",
            "repository_commit": None,
            "repository_tree_clean": False,
        }
    manifest = _read_json_object(manifest_path)
    extracted_commit = str(manifest.get("source_git_commit") or "").strip()
    return {
        "repository_provenance_source": "extraction_manifest",
        "repository_commit": extracted_commit.lower()
        if IMMUTABLE_COMMIT_PATTERN.fullmatch(extracted_commit)
        else None,
        "repository_tree_clean": manifest.get("source_git_tree_clean") is True,
    }


def resolve_huggingface_model_commit(
    model_id: str,
    requested_revision: str | None,
    *,
    hf_token: str | None = None,
) -> dict[str, str]:
    """把模型 revision 解析为不可变的40位 Hugging Face commit。

    显式 commit 可以离线复用。branch、tag 或空 revision 必须通过 Hub 元数据解析,
    解析失败时不得继续正式运行。
    """

    revision = str(requested_revision or "").strip()
    if IMMUTABLE_COMMIT_PATTERN.fullmatch(revision):
        return {
            "model_id": model_id,
            "requested_revision": revision.lower(),
            "resolved_commit": revision.lower(),
            "revision_resolution_source": "configured_immutable_commit",
        }
    try:
        from huggingface_hub import model_info

        info = model_info(model_id, revision=revision or None, token=hf_token)
    except Exception as exc:  # pragma: no cover - 真实网络与认证路径
        raise RuntimeError(f"无法解析模型 revision: {model_id}") from exc
    resolved = str(getattr(info, "sha", "") or "").strip()
    if not IMMUTABLE_COMMIT_PATTERN.fullmatch(resolved):
        raise RuntimeError(f"模型 revision 未解析为不可变 commit: {model_id}")
    return {
        "model_id": model_id,
        "requested_revision": revision or "default",
        "resolved_commit": resolved.lower(),
        "revision_resolution_source": "huggingface_hub_metadata",
    }


def _inspect_gpu() -> dict[str, Any]:
    """读取 CUDA 与首张 GPU 的最小能力信息。"""

    try:
        import torch
    except Exception as exc:
        return {
            "torch_import_status": "failed",
            "torch_import_failure_reason": f"{type(exc).__name__}: {exc}",
            "cuda_available": False,
        }
    cuda_available = bool(torch.cuda.is_available())
    result: dict[str, Any] = {
        "torch_import_status": "ready",
        "torch_runtime_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda or ""),
        "cuda_available": cuda_available,
        "cuda_device_count": int(torch.cuda.device_count()) if cuda_available else 0,
    }
    if not cuda_available:
        return result
    properties = torch.cuda.get_device_properties(0)
    capability = tuple(int(item) for item in torch.cuda.get_device_capability(0))
    result.update({
        "gpu_name": str(torch.cuda.get_device_name(0)),
        "gpu_memory_gib": round(float(properties.total_memory) / (1024 ** 3), 3),
        "gpu_compute_capability": list(capability),
    })
    return result


def build_runtime_environment_preflight_decision(
    *,
    repo_root: str | Path,
    lock_path: str | Path = DEFAULT_RUNTIME_LOCK_PATH,
    require_gpu: bool,
    model_requests: Mapping[str, str | None] | None = None,
    hf_token: str | None = None,
    installed_versions: Mapping[str, str | None] | None = None,
    python_major_minor: str | None = None,
    gpu_observation: Mapping[str, Any] | None = None,
    repository_observation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """构建 fail-closed 环境预检记录。

    可选 observation 参数只用于轻量测试注入, 正式服务器入口不会绕过真实检查。
    """

    root = Path(repo_root).expanduser().resolve()
    lock, resolved_lock_path, lock_digest = load_runtime_environment_lock(root, lock_path)
    required = dict(lock.get("required_distributions") or {})
    observed_versions = dict(installed_versions) if installed_versions is not None else _installed_distribution_versions(list(required))
    observed_python = python_major_minor or f"{sys.version_info.major}.{sys.version_info.minor}"
    gpu = dict(gpu_observation) if gpu_observation is not None else _inspect_gpu()
    repository = dict(repository_observation) if repository_observation is not None else _repository_provenance(root)
    failures: list[str] = []

    if observed_python != str(lock.get("python_major_minor") or ""):
        failures.append("python_major_minor_mismatch")
    dependency_mismatches: list[dict[str, Any]] = []
    for name, expected in sorted(required.items()):
        observed = observed_versions.get(name)
        if observed is None or _normalized_distribution_version(observed) != str(expected):
            dependency_mismatches.append({
                "distribution_name": name,
                "expected_version": str(expected),
                "observed_version": observed,
            })
    if dependency_mismatches:
        failures.append("locked_dependency_mismatch")

    if repository.get("repository_commit") is None:
        failures.append("repository_commit_unresolved")
    if repository.get("repository_tree_clean") is not True:
        failures.append("repository_tree_not_clean")

    if require_gpu:
        if gpu.get("cuda_available") is not True:
            failures.append("cuda_unavailable")
        expected_cuda = str(lock.get("torch_cuda_version") or "")
        if str(gpu.get("torch_cuda_version") or "") != expected_cuda:
            failures.append("torch_cuda_version_mismatch")
        minimum_memory = float(lock.get("minimum_gpu_memory_gib") or 0.0)
        if float(gpu.get("gpu_memory_gib") or 0.0) < minimum_memory:
            failures.append("gpu_memory_insufficient")
        minimum_capability = tuple(int(item) for item in lock.get("minimum_cuda_compute_capability") or [0, 0])
        observed_capability = tuple(int(item) for item in gpu.get("gpu_compute_capability") or [0, 0])
        if observed_capability < minimum_capability:
            failures.append("gpu_compute_capability_insufficient")

    registered_models = dict(lock.get("registered_generation_models") or {})
    resolved_models: list[dict[str, str]] = []
    model_revision_check_status = "blocked_by_runtime_environment"
    if not failures:
        model_revision_check_status = "completed"
        for model_id, requested_revision in dict(model_requests or {}).items():
            if model_id not in registered_models:
                failures.append(f"generation_model_not_registered:{model_id}")
                continue
            try:
                resolved_models.append(resolve_huggingface_model_commit(
                    model_id,
                    requested_revision,
                    hf_token=hf_token,
                ))
            except RuntimeError:
                failures.append(f"generation_model_revision_unresolved:{model_id}")

    return {
        "manifest_kind": "paper_runtime_environment_preflight_decision",
        "lock_schema_version": lock.get("lock_schema_version"),
        "runtime_environment_lock_id": lock.get("lock_id"),
        "runtime_environment_lock_path": resolved_lock_path.as_posix(),
        "runtime_environment_lock_sha256": lock_digest,
        "python_major_minor": observed_python,
        "required_gpu": bool(require_gpu),
        "installed_distribution_versions": observed_versions,
        "dependency_mismatches": dependency_mismatches,
        "gpu_observation": gpu,
        **repository,
        "generation_model_revision_check_status": model_revision_check_status,
        "resolved_generation_models": resolved_models,
        "runtime_environment_preflight_failures": sorted(set(failures)),
        "runtime_environment_preflight_decision": "PASS" if not failures else "FAIL",
        "claim_support_status": "environment_provenance_only_not_claim_evidence",
    }


def resolved_model_commit(decision: Mapping[str, Any], model_id: str) -> str:
    """从已通过的预检记录中读取指定模型 commit。"""

    for row in decision.get("resolved_generation_models") or []:
        if isinstance(row, Mapping) and row.get("model_id") == model_id:
            commit = str(row.get("resolved_commit") or "")
            if IMMUTABLE_COMMIT_PATTERN.fullmatch(commit):
                return commit.lower()
    raise KeyError(f"预检记录缺少模型 commit: {model_id}")


def write_runtime_environment_preflight_artifact(
    run_root: str | Path,
    decision: Mapping[str, Any],
) -> Path:
    """把环境来源记录写入当前 stage run root, 供阶段包归档。

    该 artifact 只证明运行环境和模型 revision 来源, 不参与任何论文 claim 判定。
    """

    if decision.get("runtime_environment_preflight_decision") != "PASS":
        raise ValueError("只有已通过的环境预检可以进入正式 stage 包")
    output_path = Path(run_root) / RUNTIME_ENVIRONMENT_PREFLIGHT_ARTIFACT_RELPATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(dict(decision), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path
