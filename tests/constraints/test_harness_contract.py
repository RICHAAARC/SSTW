"""验证模板仓库的 harness 基础契约。"""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.harness.run_all_audits import run_all_audits
from tools.harness.audits.audit_notebook_thin_entrypoints import run_audit as run_notebook_thin_entrypoint_audit
from tools.harness.lib.file_scanner import should_skip_path


@pytest.mark.constraint
def test_project_contract_exists() -> None:
    """项目契约必须存在, 作为所有修改前置依据。"""
    assert Path(".codex/project_contract.md").exists()


@pytest.mark.constraint
def test_external_agent_context_directory_is_not_governed_source() -> None:
    """.agents 是外部 agent 上下文目录, 不应进入项目正式命名审计。"""
    assert should_skip_path(Path(".agents"))


@pytest.mark.constraint
def test_local_environment_and_editor_directories_are_not_governed_source() -> None:
    """Local environment and editor state are outside governed source."""

    assert should_skip_path(Path(".conda"))
    assert should_skip_path(Path(".vscode"))


@pytest.mark.constraint
def test_harness_audits_pass_for_template() -> None:
    """模板仓库自身必须通过内置 harness 审计。"""
    summary = run_all_audits(Path.cwd())
    assert summary["overall_decision"] == "pass"


@pytest.mark.constraint
def test_notebooks_are_governed_thin_entrypoints() -> None:
    """全部 Notebook 必须委托仓库 workflow, 不得内嵌方法或产物 writer。"""

    report = run_notebook_thin_entrypoint_audit(Path.cwd())
    assert report["decision"] == "pass", report["violations"]


@pytest.mark.constraint
def test_main_core_package_exists() -> None:
    """论文研究项目模板必须使用 main 作为核心包目录。"""
    assert Path("main/__init__.py").exists()
    assert not Path("src").exists()
