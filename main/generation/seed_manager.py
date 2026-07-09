"""读取 generative_video_model_probe seed 配置。"""

from __future__ import annotations

from pathlib import Path
import json


def load_seeds(path: str | Path) -> list[dict]:
    """读取确定性 seed 列表。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(data.get("seeds", []))
