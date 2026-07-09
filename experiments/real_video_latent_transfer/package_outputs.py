"""real_video_latent_transfer_check 输出打包候选枚举。"""

from __future__ import annotations

from pathlib import Path


def list_package_candidates(output_root: str | Path) -> list[Path]:
    """列出 real_video_latent_transfer_check 本地输出目录中的文件。"""
    root = Path(output_root)
    return [path for path in root.rglob("*") if path.is_file()]
