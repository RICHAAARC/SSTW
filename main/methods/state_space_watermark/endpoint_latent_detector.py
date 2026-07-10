"""从最终视频重建 Wan endpoint latent 并提取同源密钥证据。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from main.methods.state_space_watermark.flow_tubelet_key_code import (
    FlowTubeletKeyCodeConfig,
    build_flow_tubelet_key_direction_like,
    iter_tubelet_slices,
)


@dataclass(frozen=True)
class EndpointLatentEvidence:
    """保存 endpoint latent 的密钥投影与 payload 恢复结果。"""

    score: float
    projection: float
    bit_accuracy: float
    tubelet_count: int
    coverage_ratio: float
    endpoint_latent_norm: float

    def as_dict(self) -> dict[str, float | int | str]:
        """转换为正式 detection record 字段。"""

        return {
            "endpoint_evidence_status": "ready",
            "endpoint_score": round(self.score, 8),
            "endpoint_projection": round(self.projection, 8),
            "endpoint_bit_accuracy": round(self.bit_accuracy, 8),
            "endpoint_tubelet_count": self.tubelet_count,
            "endpoint_coverage_ratio": round(self.coverage_ratio, 8),
            "endpoint_latent_norm": round(self.endpoint_latent_norm, 6),
            "endpoint_evidence_source": "wan_vae_reencoded_video_latent",
        }


def compute_endpoint_latent_evidence(
    endpoint_latent: Any,
    *,
    key_text: str,
    tubelet_config: FlowTubeletKeyCodeConfig | None = None,
    minimum_block_norm: float = 1e-8,
) -> EndpointLatentEvidence:
    """使用生成阶段同一 tubelet key code 计算 endpoint 证据。"""

    tubelet_config = tubelet_config or FlowTubeletKeyCodeConfig()
    direction, _metadata = build_flow_tubelet_key_direction_like(
        endpoint_latent,
        key_text=key_text,
        config=tubelet_config,
    )
    latent_flat = endpoint_latent.detach().float().reshape(-1)
    direction_flat = direction.detach().float().reshape(-1)
    denominator = latent_flat.norm().clamp_min(1e-8) * direction_flat.norm().clamp_min(1e-8)
    projection = float((latent_flat @ direction_flat / denominator).item())

    bit_matches: list[float] = []
    for block_slice in iter_tubelet_slices(tuple(endpoint_latent.shape), tubelet_config):
        latent_block = endpoint_latent[block_slice].detach().float().reshape(-1)
        direction_block = direction[block_slice].detach().float().reshape(-1)
        if float(latent_block.norm().item()) <= minimum_block_norm:
            continue
        bit_matches.append(float((latent_block @ direction_block).item() > 0.0))
    tubelet_count = sum(1 for _ in iter_tubelet_slices(tuple(endpoint_latent.shape), tubelet_config))
    valid_count = len(bit_matches)
    bit_accuracy = mean(bit_matches) if bit_matches else 0.0
    score = max(0.0, min(1.0, 0.5 + projection * 0.5))
    return EndpointLatentEvidence(
        score=score,
        projection=projection,
        bit_accuracy=bit_accuracy,
        tubelet_count=tubelet_count,
        coverage_ratio=valid_count / max(1, tubelet_count),
        endpoint_latent_norm=float(endpoint_latent.detach().float().norm().item()),
    )


def _retrieve_vae_latent(encoded: Any) -> Any:
    """兼容 Diffusers 不同 VAE encode 返回结构并选择确定性 mode。"""

    value = encoded[0] if isinstance(encoded, tuple) else encoded
    if hasattr(value, "latent_dist"):
        value = value.latent_dist
    if hasattr(value, "mode"):
        return value.mode()
    if hasattr(value, "sample") and not callable(value.sample):
        return value.sample
    if hasattr(value, "latents"):
        return value.latents
    return value


def load_video_tensor_for_wan_vae(video_path: str | Path, *, device: Any, dtype: Any) -> tuple[Any, int]:
    """把 mp4 解码成 Wan VAE 使用的 `[B, C, T, H, W]` 张量。"""

    import imageio.v3 as iio
    import numpy as np
    import torch

    frames = [np.asarray(frame) for frame in iio.imiter(Path(video_path))]
    if not frames:
        raise RuntimeError("视频没有可解码帧")
    array = np.stack(frames, axis=0)
    if array.ndim != 4 or array.shape[-1] < 3:
        raise ValueError("视频帧必须为 RGB 格式")
    tensor = torch.from_numpy(array[..., :3]).permute(3, 0, 1, 2).unsqueeze(0).float()
    tensor = tensor / 127.5 - 1.0
    return tensor.to(device=device, dtype=dtype), len(frames)


def encode_video_to_wan_endpoint_latent(
    vae: Any,
    video_path: str | Path,
) -> tuple[Any, dict[str, Any]]:
    """调用真实 Wan VAE 把攻击后视频重建为生成坐标系中的 endpoint latent。"""

    import torch

    device = next(vae.parameters()).device
    dtype = vae.dtype
    video, frame_count = load_video_tensor_for_wan_vae(video_path, device=device, dtype=dtype)
    with torch.inference_mode():
        latent = _retrieve_vae_latent(vae.encode(video))
    latent = latent.to(device=device, dtype=torch.float32)
    mean_values = getattr(vae.config, "latents_mean", None)
    std_values = getattr(vae.config, "latents_std", None)
    if mean_values is None or std_values is None:
        raise RuntimeError("Wan VAE 配置缺少 latents_mean 或 latents_std")
    latent_mean = torch.tensor(mean_values, device=device, dtype=torch.float32).view(1, -1, 1, 1, 1)
    latent_std = 1.0 / torch.tensor(std_values, device=device, dtype=torch.float32).view(1, -1, 1, 1, 1)
    normalized = (latent - latent_mean) * latent_std
    return normalized, {
        "endpoint_video_frame_count": frame_count,
        "endpoint_vae_model_class": type(vae).__name__,
        "endpoint_vae_encode_status": "ready",
        "endpoint_latent_shape": list(normalized.shape),
    }


class WanEndpointLatentDetector:
    """缓存 Wan VAE, 为大量 attacked video 提供统一 endpoint 检测。"""

    def __init__(self, vae: Any, *, tubelet_config: FlowTubeletKeyCodeConfig | None = None):
        self.vae = vae
        self.tubelet_config = tubelet_config or FlowTubeletKeyCodeConfig()

    @classmethod
    def from_pretrained(cls, model_id: str, *, device: str = "cuda", torch_dtype: Any | None = None):
        """从主线 Wan 模型加载官方 VAE 权重。"""

        import torch
        from diffusers import AutoencoderKLWan

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("正式 endpoint latent 检测需要可用 CUDA GPU")
        dtype = torch_dtype or (torch.bfloat16 if device.startswith("cuda") else torch.float32)
        vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype)
        vae.to(device)
        vae.eval()
        if hasattr(vae, "enable_tiling"):
            vae.enable_tiling()
        return cls(vae)

    def score_video(self, video_path: str | Path, *, key_text: str) -> tuple[EndpointLatentEvidence, dict[str, Any]]:
        """对单个视频执行 VAE endpoint 重建和同源 key 检测。"""

        latent, metadata = encode_video_to_wan_endpoint_latent(self.vae, video_path)
        evidence = compute_endpoint_latent_evidence(
            latent,
            key_text=key_text,
            tubelet_config=self.tubelet_config,
        )
        return evidence, metadata
