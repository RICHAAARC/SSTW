"""真实 Wan VAE endpoint encode 显存边界；默认 pytest 明确排除。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.mark.integration
@pytest.mark.slow
def test_real_wan_vae_encodes_33_frame_video_within_governed_peak() -> None:
    """在目标 GPU 上验证 FP32 VAE streaming 的真实 activation 峰值。"""

    torch = pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    video_path = Path(os.environ.get("WAN_VAE_INTEGRATION_VIDEO_PATH", ""))
    if not video_path.is_file():
        pytest.skip("set WAN_VAE_INTEGRATION_VIDEO_PATH to a real 33-frame video")
    model_id = os.environ.get(
        "WAN_VAE_INTEGRATION_MODEL_ID",
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    )
    revision = os.environ.get("WAN_VAE_INTEGRATION_MODEL_REVISION") or None

    from main.methods.state_space_watermark.endpoint_latent_detector import (
        WanVAEEncodeMemoryConfig,
        configure_wan_vae_encode_memory,
        encode_video_to_wan_endpoint_latent,
    )

    vae = diffusers.AutoencoderKLWan.from_pretrained(
        model_id,
        subfolder="vae",
        revision=revision,
        torch_dtype=torch.float32,
    ).to("cuda")
    vae.eval()
    config = WanVAEEncodeMemoryConfig()
    configure_wan_vae_encode_memory(vae, config)
    latent, metadata = encode_video_to_wan_endpoint_latent(vae, video_path)

    assert int(metadata["endpoint_video_frame_count"]) == 33
    assert metadata["endpoint_vae_encode_strategy"] == (
        "cpu_resident_spatiotemporal_streaming"
    )
    assert metadata["endpoint_vae_memory_preflight_status"] == "ready"
    assert (
        float(metadata["endpoint_vae_cuda_incremental_peak_allocated_gib"])
        <= config.maximum_incremental_cuda_peak_gib
    )
    assert latent.device.type == "cuda"
    assert torch.isfinite(latent).all()
