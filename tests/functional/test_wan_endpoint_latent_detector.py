"""验证 Wan endpoint VAE 的轻量设备与归一化语义。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch", reason="requires optional method-runtime dependency")

from main.methods.state_space_watermark.endpoint_latent_detector import (
    WanVAEEncodeMemoryConfig,
    configure_wan_vae_encode_memory,
    encode_video_to_wan_endpoint_latent,
)


class _LatentDistribution:
    def __init__(self, value: torch.Tensor) -> None:
        self.value = value

    def mode(self) -> torch.Tensor:
        return self.value


class _CpuOffloadedVAE:
    dtype = torch.float32
    config = SimpleNamespace(latents_mean=[0.0] * 4, latents_std=[1.0] * 4)
    _hf_hook = SimpleNamespace(execution_device=torch.device("cpu:1"))

    def __init__(self) -> None:
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        return iter((self.weight,))

    def encode(self, video: torch.Tensor) -> SimpleNamespace:
        latent = torch.ones((1, 4, 1, 1, 1), device=video.device)
        return SimpleNamespace(latent_dist=_LatentDistribution(latent))


class _StreamingWanVAE:
    dtype = torch.float32
    config = SimpleNamespace(
        latents_mean=[0.0] * 4,
        latents_std=[1.0] * 4,
        patch_size=None,
    )
    spatial_compression_ratio = 1

    def __init__(self) -> None:
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self._enc_feat_map = []
        self._enc_conv_idx = [0]
        self.temporal_chunks: list[int] = []

        def encoder(video, *, feat_cache, feat_idx):
            del feat_cache, feat_idx
            self.temporal_chunks.append(int(video.shape[2]))
            base = video.mean(dim=1, keepdim=True)
            return base.repeat(1, 8, 1, 1, 1)

        self.encoder = encoder
        self.quant_conv = lambda value: value

    def parameters(self):
        return iter((self.weight,))

    def clear_cache(self) -> None:
        self._enc_feat_map = []

    def enable_tiling(self, **kwargs) -> None:
        self.tiling = dict(kwargs)

    @staticmethod
    def blend_v(_above, current, _blend_extent):
        return current

    @staticmethod
    def blend_h(_left, current, _blend_extent):
        return current


@pytest.mark.quick
def test_wan_vae_input_uses_cpu_offload_execution_device(monkeypatch, tmp_path) -> None:
    """VAE 参数暂驻 CPU 时，输入仍须送到 offload hook 的实际执行设备。"""

    observed: dict[str, object] = {}

    def fake_load_video(_path, *, device, dtype):
        observed["device"] = device
        return torch.zeros((1, 3, 1, 2, 2), dtype=dtype), 1

    monkeypatch.setattr(
        "main.methods.state_space_watermark.endpoint_latent_detector.load_video_tensor_for_wan_vae",
        fake_load_video,
    )

    latent, metadata = encode_video_to_wan_endpoint_latent(
        _CpuOffloadedVAE(),
        tmp_path / "attacked.mp4",
    )

    assert observed["device"] == torch.device("cpu:1")
    assert torch.equal(latent, torch.ones_like(latent))
    assert metadata["endpoint_vae_encode_status"] == "ready"
    assert metadata["endpoint_vae_encode_strategy"] == "compatibility_full_tensor_encode"


@pytest.mark.quick
def test_wan_vae_streams_native_temporal_chunks_from_cpu(monkeypatch, tmp_path) -> None:
    """33帧类输入必须留在 CPU，并按 Wan 原生首1帧后4帧因果块编码。"""

    observed: dict[str, object] = {}

    def fake_load_video(_path, *, device, dtype):
        observed["device"] = device
        return torch.ones((1, 3, 9, 8, 8), device=device, dtype=dtype), 9

    monkeypatch.setattr(
        "main.methods.state_space_watermark.endpoint_latent_detector.load_video_tensor_for_wan_vae",
        fake_load_video,
    )
    vae = _StreamingWanVAE()
    configure_wan_vae_encode_memory(
        vae,
        WanVAEEncodeMemoryConfig(
            tile_sample_height=8,
            tile_sample_width=8,
            tile_sample_stride_height=4,
            tile_sample_stride_width=4,
            minimum_cuda_free_gib=0.0,
        ),
    )

    latent, metadata = encode_video_to_wan_endpoint_latent(
        vae,
        tmp_path / "source.mp4",
    )

    assert observed["device"] == torch.device("cpu")
    assert vae.temporal_chunks == [1, 4, 4, 1, 4, 4, 1, 4, 4, 1, 4, 4]
    assert tuple(latent.shape) == (1, 4, 9, 8, 8)
    assert metadata["endpoint_vae_encode_strategy"] == (
        "cpu_resident_spatiotemporal_streaming"
    )
    assert metadata["endpoint_vae_memory_preflight_status"] == (
        "ready_non_cuda_diagnostic"
    )


@pytest.mark.quick
def test_wan_vae_memory_config_rejects_non_native_temporal_chunk() -> None:
    """不得以显存优化为由悄悄改变 Wan 的原生因果时间分块。"""

    with pytest.raises(ValueError, match="固定按4帧"):
        WanVAEEncodeMemoryConfig(temporal_chunk_frame_count=2).validate()
