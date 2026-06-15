"""读取生成采样 scheduler 配置。"""

from __future__ import annotations

from pathlib import Path
import json


def load_scheduler_config(path: str | Path) -> dict:
    """读取 scheduler 配置并返回可写入 records 的字段。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))
