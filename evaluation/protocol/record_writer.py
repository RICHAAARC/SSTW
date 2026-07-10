"""提供 JSONL records 的读写工具。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def write_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    """将 records 写入 JSONL 文件。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    """从 JSONL 文件读取 records。"""
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: str | Path, value: dict | list) -> None:
    """将 JSON 兼容对象写入文件。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
