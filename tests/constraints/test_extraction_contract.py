"""验证论文附件抽离契约。"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.extract_minimal_paper_package import extract_profile
from tools.harness.audits.audit_dependency_boundaries import run_audit as run_dependency_boundary_audit
from tools.harness.audits.audit_release_extraction_contract import run_audit as run_release_extraction_audit


@pytest.mark.constraint
def test_dependency_boundaries_pass_for_template() -> None:
    """模板自身必须保持核心方法层可抽离。"""
    report = run_dependency_boundary_audit(Path.cwd())
    assert report["decision"] == "pass"


@pytest.mark.constraint
def test_release_extraction_contract_pass_for_template() -> None:
    """模板必须提供最小论文附件抽离规则。"""
    report = run_release_extraction_audit(Path.cwd())
    assert report["decision"] == "pass"


@pytest.mark.constraint
def test_minimal_method_package_dry_run_excludes_governance_layer(tmp_path: Path) -> None:
    """最小方法包抽离清单不得包含外层治理目录。"""
    manifest = extract_profile(Path.cwd(), tmp_path / "minimal_method_package", "minimal_method_package", dry_run=True)
    copied_files = manifest["copied_files"]
    assert copied_files
    assert all(not path.startswith(".codex/") for path in copied_files)
    assert all(not path.startswith("tools/") for path in copied_files)
    assert all(not path.startswith("tests/") for path in copied_files)
    assert all(not path.startswith("experiments/") for path in copied_files)
    assert all(not path.startswith("paper_workflow/") for path in copied_files)
    assert "configs/protocol/shared_generative_video_attack_protocol.json" in copied_files


@pytest.mark.constraint
def test_paper_artifact_rebuild_package_keeps_workflow_configs(tmp_path: Path) -> None:
    """服务器重建包必须保留 configs 下的 workflow 配置, 但排除 Notebook 外层。"""

    manifest = extract_profile(
        Path.cwd(),
        tmp_path / "paper_artifact_rebuild_package",
        "paper_artifact_rebuild_package",
        dry_run=True,
    )
    copied_files = manifest["copied_files"]
    assert "configs/paper_workflow/generative_video_notebook_workflows.json" in copied_files
    assert all(not path.startswith("paper_workflow/") for path in copied_files)
