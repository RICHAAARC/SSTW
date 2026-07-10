"""提供受治理文本文件扫描能力。"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

SKIP_DIRECTORY_NAMES = {
    ".git",
    ".gitnexus",
    ".agents",
    "__pycache__",
    ".pytest_cache",
    ".venv",
    "outputs",
    "audit_reports",
    "dist",
    "build",
}

SKIP_FILE_NAMES = {
    "acl.txt",
}

SKIP_FILE_SUFFIXES = {
    ".tmp",
}

BINARY_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".zip", ".tar", ".gz", ".7z", ".exe", ".dll", ".so", ".pyc", ".pyd",
}

DEFAULT_GOVERNED_SCAN_ROOTS = (
    "AGENTS.md",
    "README.md",
    "pyproject.toml",
    ".codex",
    "configs",
    "docs",
    "main",
    "main",
    "tools",
    "tests",
    "scripts",
    "experiments",
    "paper_workflow",
)


def should_skip_path(path: str | Path) -> bool:
    """判断路径是否属于缓存、输出或构建产物。"""
    candidate = Path(path)
    parts = candidate.parts
    # 第三方官方 baseline 源码是 Colab 冷启动 clone 的 runtime cache, 已由
    # source intake 单独记录 commit 与入口文件。它不属于本仓库命名治理对象,
    # 否则上游仓库自带的 LICENSE、README 或示例文件会让本项目 harness 误失败。
    is_external_official_source_snapshot = (
        len(parts) >= 4
        and parts[0] == "external_baseline"
        and parts[1] == "primary"
        and parts[3] == "source"
    )
    return (
        is_external_official_source_snapshot
        or
        any(part in SKIP_DIRECTORY_NAMES for part in candidate.parts)
        or candidate.name in SKIP_FILE_NAMES
        or candidate.suffix.lower() in SKIP_FILE_SUFFIXES
    )


def iter_text_files(root: str | Path) -> Iterator[Path]:
    """遍历目录下的文本候选文件。"""
    root_path = Path(root)
    if not root_path.exists():
        return
    for path in root_path.rglob("*"):
        if not path.is_file() or should_skip_path(path):
            continue
        if path.suffix.lower() in BINARY_SUFFIXES:
            continue
        yield path


def iter_governed_text_files(root: str | Path) -> Iterator[Path]:
    """按默认治理根遍历文本文件。"""
    root_path = Path(root)
    for relative_root in DEFAULT_GOVERNED_SCAN_ROOTS:
        candidate = root_path / relative_root
        if not candidate.exists() or should_skip_path(candidate):
            continue
        if candidate.is_file():
            yield candidate
        else:
            yield from iter_text_files(candidate)
