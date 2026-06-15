"""管理 B5 生成式视频模型候选项。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class GenerationModelSpec:
    """保存一个生成模型候选项的可审计元数据。"""

    generation_model_id: str
    generation_model_name: str
    generation_model_family: str
    generation_model_version: str
    generation_model_commit_or_hash: str | None
    generation_model_license_status: str
    generation_backend_id: str
    vae_backend_id: str
    trajectory_capture_mode: str
    trajectory_availability_status: str


def load_generation_models(path: str | Path) -> list[GenerationModelSpec]:
    """从配置文件读取生成模型候选项, 不在此处执行模型下载或推理。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [GenerationModelSpec(**item) for item in data.get("models", [])]


def registry_snapshot(specs: list[GenerationModelSpec]) -> dict:
    """生成可写入 artifacts 的模型注册快照。"""
    return {"model_count": len(specs), "models": [spec.__dict__ for spec in specs]}
