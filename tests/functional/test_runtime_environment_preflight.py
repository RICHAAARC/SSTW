"""验证服务器与 Colab 共用的论文环境锁和失败关闭行为。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from workflows.runtime_environment_preflight import (
    build_runtime_environment_preflight_decision,
    load_runtime_environment_lock,
    resolved_model_commit,
    write_runtime_environment_preflight_artifact,
)
from workflows.stage_package_sync import (
    FORMAL_COMPARISON_SCORING_PACKAGE_RELPATHS,
    GEN_VIDEO_GENERATION_PACKAGE_RELPATHS,
    PAPER_EVIDENCE_POSTPROCESS_PACKAGE_RELPATHS,
    RUNTIME_DETECTION_PACKAGE_RELPATHS,
)


def _runtime_lock_payload() -> dict:
    """读取测试所需的运行环境锁, 避免复制第二份环境契约。"""

    return json.loads(
        Path("requirements/paper_runtime_environment_lock.json").read_text(
            encoding="utf-8"
        )
    )


def _locked_versions() -> dict[str, str]:
    """读取锁定依赖版本。"""

    return dict(_runtime_lock_payload()["required_distributions"])


def _locked_python_major_minor() -> str:
    """读取锁定 Python major.minor。"""

    return str(_runtime_lock_payload()["python_major_minor"])


def _passing_gpu_observation() -> dict:
    """构造不导入真实 CUDA 的轻量硬件观测。"""

    return {
        "torch_import_status": "ready",
        "torch_runtime_version": "2.6.0+cu124",
        "torch_cuda_version": "12.4",
        "cuda_available": True,
        "cuda_device_count": 1,
        "gpu_name": "test_gpu",
        "gpu_memory_gib": 24.0,
        "gpu_compute_capability": [8, 0],
    }


@pytest.mark.quick
def test_runtime_preflight_freezes_models_and_accepts_exact_locked_environment() -> None:
    """完整预检必须同时固定依赖、代码来源、GPU 与模型 commit。"""

    main_commit = "1" * 40
    cross_commit = "2" * 40
    decision = build_runtime_environment_preflight_decision(
        repo_root=Path.cwd(),
        require_gpu=True,
        model_requests={
            "Wan-AI/Wan2.1-T2V-1.3B-Diffusers": main_commit,
            "Lightricks/LTX-Video": cross_commit,
        },
        installed_versions=_locked_versions(),
        python_major_minor=_locked_python_major_minor(),
        gpu_observation=_passing_gpu_observation(),
        repository_observation={
            "repository_provenance_source": "git_worktree",
            "repository_commit": "3" * 40,
            "repository_tree_clean": True,
        },
    )

    assert decision["runtime_environment_preflight_decision"] == "PASS"
    assert decision["runtime_environment_preflight_failures"] == []
    assert resolved_model_commit(
        decision, "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    ) == main_commit
    assert resolved_model_commit(decision, "Lightricks/LTX-Video") == cross_commit
    assert len(decision["runtime_environment_lock_sha256"]) == 64


@pytest.mark.quick
def test_runtime_preflight_fails_closed_on_dependency_gpu_or_dirty_tree() -> None:
    """任一关键环境条件漂移时都不得开始正式论文 workflow。"""

    versions = _locked_versions()
    versions["diffusers"] = "0.0.0"
    gpu = _passing_gpu_observation()
    gpu["gpu_memory_gib"] = 8.0
    decision = build_runtime_environment_preflight_decision(
        repo_root=Path.cwd(),
        require_gpu=True,
        installed_versions=versions,
        python_major_minor=_locked_python_major_minor(),
        gpu_observation=gpu,
        repository_observation={
            "repository_provenance_source": "git_worktree",
            "repository_commit": "3" * 40,
            "repository_tree_clean": False,
        },
    )

    assert decision["runtime_environment_preflight_decision"] == "FAIL"
    assert "locked_dependency_mismatch" in decision["runtime_environment_preflight_failures"]
    assert "gpu_memory_insufficient" in decision["runtime_environment_preflight_failures"]
    assert "repository_tree_not_clean" in decision["runtime_environment_preflight_failures"]


@pytest.mark.quick
def test_runtime_lock_is_present_and_machine_readable() -> None:
    """服务器重建包依赖的环境锁必须可由统一 helper 读取。"""

    lock, path, digest = load_runtime_environment_lock(Path.cwd())

    assert path.name == "paper_runtime_environment_lock.json"
    assert lock["lock_schema_version"] == "sstw_paper_runtime_environment_lock_v1"
    assert lock["lock_id"] == "sstw_paper_runtime_python_3_12_torch_2_6_cuda_12_4"
    assert lock["python_major_minor"] == "3.12"
    assert lock["registered_generation_models"]
    assert len(digest) == 64


@pytest.mark.quick
def test_passed_preflight_is_archived_with_each_formal_stage(tmp_path: Path) -> None:
    """环境与模型来源必须进入阶段包, 不能只停留在 Notebook stdout。"""

    decision = build_runtime_environment_preflight_decision(
        repo_root=Path.cwd(),
        require_gpu=False,
        installed_versions=_locked_versions(),
        python_major_minor=_locked_python_major_minor(),
        gpu_observation={"cuda_available": False},
        repository_observation={
            "repository_provenance_source": "git_worktree",
            "repository_commit": "3" * 40,
            "repository_tree_clean": True,
        },
    )
    output_path = write_runtime_environment_preflight_artifact(tmp_path, decision)
    relpath = "artifacts/paper_runtime_environment_preflight_decision.json"

    assert output_path == tmp_path / relpath
    assert json.loads(output_path.read_text(encoding="utf-8"))[
        "runtime_environment_preflight_decision"
    ] == "PASS"
    assert relpath in GEN_VIDEO_GENERATION_PACKAGE_RELPATHS
    assert relpath in RUNTIME_DETECTION_PACKAGE_RELPATHS
    assert relpath in FORMAL_COMPARISON_SCORING_PACKAGE_RELPATHS
    assert relpath in PAPER_EVIDENCE_POSTPROCESS_PACKAGE_RELPATHS
