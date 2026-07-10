"""构造 SSTW 在 Flow latent 上使用的密钥条件 tubelet code。"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import sin, pi
from typing import Any


@dataclass(frozen=True)
class FlowTubeletKeyCodeConfig:
    """定义 Flow latent 的时空 tubelet 划分和相位窗口。"""

    temporal_size: int = 2
    spatial_height: int = 8
    spatial_width: int = 8
    phase_window_start: float = 0.25
    phase_window_end: float = 0.75


def _stable_seed(*parts: object) -> int:
    """把密钥和 tubelet 坐标转换成跨进程稳定的 PyTorch 种子。"""

    text = "::".join(str(part) for part in parts)
    digest = sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**63 - 1)


def flow_phase_weight(flow_phase: float, config: FlowTubeletKeyCodeConfig) -> float:
    """计算仅在中段 Flow phase 激活的平滑权重。"""

    phase = max(0.0, min(1.0, float(flow_phase)))
    start = float(config.phase_window_start)
    end = float(config.phase_window_end)
    if not 0.0 <= start < end <= 1.0:
        raise ValueError("Flow phase 窗口必须满足 0 <= start < end <= 1")
    if phase <= start or phase >= end:
        return 0.0
    normalized = (phase - start) / (end - start)
    return float(sin(pi * normalized) ** 2)


def build_flow_tubelet_key_direction_like(
    reference: Any,
    *,
    key_text: str,
    config: FlowTubeletKeyCodeConfig | None = None,
) -> tuple[Any, dict[str, Any]]:
    """生成与五维 Flow latent 同形状的密钥条件 tubelet 方向。

    方向在每个 tubelet 内独立归一化, 再执行全局归一化。payload bit 已经作为
    正负号写入方向, 因而生成端 velocity constraint、endpoint detector 和路径
    观测可以复用同一个方向, 避免各阶段使用互不相关的随机投影。
    """

    import torch

    config = config or FlowTubeletKeyCodeConfig()
    if not isinstance(key_text, str) or not key_text:
        raise ValueError("key_text 不能为空")
    if getattr(reference, "ndim", None) != 5:
        raise ValueError("Flow latent 必须使用 [B, C, T, H, W] 五维张量")
    if min(config.temporal_size, config.spatial_height, config.spatial_width) <= 0:
        raise ValueError("tubelet 尺寸必须为正整数")

    batch, channels, frames, height, width = (int(value) for value in reference.shape)
    direction = torch.zeros(reference.shape, device=reference.device, dtype=torch.float32)
    generator = torch.Generator(device=reference.device)
    generator.manual_seed(_stable_seed("sstw_flow_tubelet_direction", key_text, tuple(reference.shape)))
    base = torch.randn(reference.shape, device=reference.device, dtype=torch.float32, generator=generator)

    tubelet_count = 0
    positive_payload_count = 0
    for batch_index in range(batch):
        for frame_start in range(0, frames, config.temporal_size):
            frame_end = min(frames, frame_start + config.temporal_size)
            for top in range(0, height, config.spatial_height):
                bottom = min(height, top + config.spatial_height)
                for left in range(0, width, config.spatial_width):
                    right = min(width, left + config.spatial_width)
                    payload_positive = bool(
                        _stable_seed("sstw_payload", key_text, batch_index, frame_start, top, left) & 1
                    )
                    payload_sign = 1.0 if payload_positive else -1.0
                    block = base[
                        batch_index : batch_index + 1,
                        :channels,
                        frame_start:frame_end,
                        top:bottom,
                        left:right,
                    ]
                    block = block / block.norm().clamp_min(1e-8)
                    direction[
                        batch_index : batch_index + 1,
                        :channels,
                        frame_start:frame_end,
                        top:bottom,
                        left:right,
                    ] = block * payload_sign
                    tubelet_count += 1
                    positive_payload_count += int(payload_positive)

    direction = direction / direction.norm().clamp_min(1e-8)
    metadata = {
        "flow_tubelet_key_code_status": "ready",
        "flow_tubelet_count": tubelet_count,
        "flow_tubelet_temporal_size": config.temporal_size,
        "flow_tubelet_spatial_height": config.spatial_height,
        "flow_tubelet_spatial_width": config.spatial_width,
        "flow_payload_positive_count": positive_payload_count,
        "flow_payload_negative_count": tubelet_count - positive_payload_count,
        "flow_key_direction_norm": round(float(direction.norm().item()), 6),
        "flow_key_direction_digest": sha256(
            f"{key_text}::{tuple(reference.shape)}::{tubelet_count}".encode("utf-8")
        ).hexdigest(),
    }
    return direction.to(dtype=reference.dtype), metadata


def iter_tubelet_slices(shape: tuple[int, ...], config: FlowTubeletKeyCodeConfig | None = None):
    """按与 key direction 相同的规则枚举 tubelet 切片。"""

    config = config or FlowTubeletKeyCodeConfig()
    if len(shape) != 5:
        raise ValueError("Flow latent shape 必须包含5个维度")
    batch, _channels, frames, height, width = (int(value) for value in shape)
    for batch_index in range(batch):
        for frame_start in range(0, frames, config.temporal_size):
            frame_end = min(frames, frame_start + config.temporal_size)
            for top in range(0, height, config.spatial_height):
                bottom = min(height, top + config.spatial_height)
                for left in range(0, width, config.spatial_width):
                    right = min(width, left + config.spatial_width)
                    yield (
                        slice(batch_index, batch_index + 1),
                        slice(None),
                        slice(frame_start, frame_end),
                        slice(top, bottom),
                        slice(left, right),
                    )
