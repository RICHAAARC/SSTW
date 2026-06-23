"""约束 external_baseline 文档不得引用其他本地项目名称。"""

from __future__ import annotations

from pathlib import Path

import pytest


GOVERNED_DOCUMENT_GLOBS = (
    "docs/builds/**/*.md",
    "external_baseline/**/*.md",
    "external_baseline/**/*.json",
)


@pytest.mark.constraint
def test_external_baseline_docs_do_not_reference_other_local_projects() -> None:
    """external_baseline 文档应描述本项目自己的接入契约, 不以其他本地项目名称作为依据。"""
    root = Path.cwd()
    forbidden_tokens = (
        "SLM" + "-WM",
        "D:" + "\\" + "Code" + "\\" + "SLM",
        "D:/" + "Code" + "/" + "SLM",
    )
    checked_paths: list[Path] = []
    violations: list[str] = []
    for pattern in GOVERNED_DOCUMENT_GLOBS:
        for path in root.glob(pattern):
            if path.is_file():
                checked_paths.append(path)
                text = path.read_text(encoding="utf-8")
                for token in forbidden_tokens:
                    if token in text:
                        violations.append(path.relative_to(root).as_posix())
                        break

    assert checked_paths
    assert violations == []
