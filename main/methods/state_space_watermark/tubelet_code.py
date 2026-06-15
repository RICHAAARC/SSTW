"""实现 key-conditioned tubelet code 的最小可复用结构。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TubeletCodeConfig:
    """描述 tubelet 编码的关键参数。"""

    tubelet_length: int = 4
    tubelet_spatial_patch: int = 8
    tubelet_stride_t: int = 2
    tubelet_stride_xy: int = 8
    watermark_alpha: float = 0.15
    payload_code_id: str = "payload_code_synthetic"
    sync_code_id: str = "sync_code_synthetic"
    joint_code_mode: str = "payload_times_sync"
    embedding_mode: str = "projection_margin"


def build_tubelet_code_config() -> TubeletCodeConfig:
    """返回第一阶段默认 tubelet code 配置。"""
    return TubeletCodeConfig()
