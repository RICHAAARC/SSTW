"""提供 real_video_latent_transfer_check 视频 VAE backend 的轻量代理。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VideoVaeBackend:
    """记录 VAE backend 的可审计配置。"""

    vae_backend_id: str
    vae_model_name: str
    vae_model_version: str
    vae_encode_dtype: str
    vae_decode_dtype: str


def build_vae_backend(config: dict) -> VideoVaeBackend:
    """根据配置构造 VAE backend 元数据对象。"""
    return VideoVaeBackend(
        vae_backend_id=config["vae_backend_id"],
        vae_model_name=config["vae_model_name"],
        vae_model_version=config["vae_model_version"],
        vae_encode_dtype=config["vae_encode_dtype"],
        vae_decode_dtype=config["vae_decode_dtype"],
    )
