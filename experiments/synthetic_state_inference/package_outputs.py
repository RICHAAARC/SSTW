"""第一阶段输出打包占位接口。"""

from __future__ import annotations

from pathlib import Path


def list_package_candidates(output_root: str | Path) -> list[Path]:
    """列出可进入本地包的轻量产物路径。"""
    root = Path(output_root)
    return [path for path in root.rglob("*") if path.is_file()]
