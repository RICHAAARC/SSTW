"""验证 Wan endpoint VAE 的轻量设备与归一化语义。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch", reason="requires optional method-runtime dependency")

from main.methods.state_space_watermark.endpoint_latent_detector import (
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
