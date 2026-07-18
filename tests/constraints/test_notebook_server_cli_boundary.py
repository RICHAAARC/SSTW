"""约束 Notebook、服务器 CLI 与内层包之间的单向依赖。"""

from __future__ import annotations

import json
from pathlib import Path
import re

import pytest


NOTEBOOK_ROOT = Path("paper_workflow/colab_notebooks")
INNER_ROOTS = (
    Path("main"),
    Path("runtime"),
    Path("evaluation"),
    Path("external_baseline"),
    Path("experiments"),
    Path("workflows"),
    Path("scripts"),
)


def _notebook_source(path: Path) -> str:
    """读取 Notebook 所有代码单元的连续文本。"""

    payload = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in payload.get("cells", [])
        if cell.get("cell_type") == "code"
    )


@pytest.mark.constraint
def test_all_colab_notebooks_delegate_only_to_server_cli() -> None:
    """每个 Colab 入口都必须复用普通 GPU 服务器的同一条 CLI。"""

    notebook_paths = sorted(NOTEBOOK_ROOT.glob("*.ipynb"))
    assert notebook_paths
    for path in notebook_paths:
        source = _notebook_source(path)
        assert "scripts/run_generative_video_server_workflow.py" in source, path
        assert "from workflows.streaming_command import run_streaming_command" in source, path
        assert "result = run_streaming_command(server_command)" in source, path
        assert "%pip install --requirement requirements/paper_runtime_lock.txt" in source, path
        assert "run_configured_colab_stage_plan" not in source, path
        assert "prepare_colab_stage_layout" not in source, path
        assert "publish_colab_stage_package" not in source, path
        assert "%pip install -U" not in source, path
        assert "git+https://" not in source, path
        assert not re.search(
            r"(?m)^\s*(?:from|import)\s+(?:main|runtime|evaluation|experiments|external_baseline)(?:\.|\s|$)",
            source,
        ), path


@pytest.mark.constraint
def test_colab_notebook_python_metadata_matches_runtime_lock() -> None:
    """所有 Colab 入口声明的 Python 版本必须与运行环境锁一致。"""

    lock = json.loads(
        Path("requirements/paper_runtime_environment_lock.json").read_text(
            encoding="utf-8"
        )
    )
    expected_python = str(lock["python_major_minor"])
    notebook_paths = sorted(NOTEBOOK_ROOT.glob("*.ipynb"))

    assert notebook_paths
    for path in notebook_paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["metadata"]["language_info"]["version"] == expected_python, path


@pytest.mark.constraint
def test_inner_layers_do_not_import_paper_workflow() -> None:
    """服务器可执行内层不得反向依赖最外层 Notebook 包。"""

    pattern = re.compile(r"(?m)^\s*(?:from|import)\s+paper_workflow(?:\.|\s|$)")
    violations: list[str] = []
    for root in INNER_ROOTS:
        for path in root.rglob("*.py"):
            if pattern.search(path.read_text(encoding="utf-8-sig")):
                violations.append(path.as_posix())
    assert violations == []
