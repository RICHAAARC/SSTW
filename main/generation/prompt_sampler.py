"""读取 B5 prompt 配置。"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import json


def load_prompts(path: str | Path) -> list[dict]:
    """读取 prompt 并用 hash 记录文本, 避免 records 中保存过长 prompt。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    prompts = []
    for item in data.get("prompts", []):
        prompt_text = item.get("prompt_text", "")
        prompts.append({
            "prompt_id": item["prompt_id"],
            "prompt_text_hash": sha256(prompt_text.encode("utf-8")).hexdigest()[:16],
            "prompt_category": item.get("prompt_category", "unspecified"),
        })
    return prompts
