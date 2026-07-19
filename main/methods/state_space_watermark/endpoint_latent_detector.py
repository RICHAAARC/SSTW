"""从最终视频重建 Wan endpoint latent 并提取同源密钥证据。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

from main.methods.state_space_watermark.flow_tubelet_key_code import (
    FlowTubeletKeyContext,
    FlowTubeletKeyCodeConfig,
    build_flow_tubelet_key_direction_like,
    build_integrated_flow_tubelet_key_direction_like,
    flow_tubelet_key_context_digest,
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
    formal_context_complete: bool = False
    key_direction_semantics: str = "static_compatibility_tubelet_direction"
    key_direction_digest: str | None = None
    key_context_digest: str | None = None
    integrated_phase_count: int = 0
    integrated_weight_sum: float = 0.0

    def as_dict(self) -> dict[str, float | int | str | bool | None]:
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
            "endpoint_formal_context_complete": self.formal_context_complete,
            "endpoint_key_direction_semantics": self.key_direction_semantics,
            "endpoint_key_direction_digest": self.key_direction_digest,
            "endpoint_key_context_digest": self.key_context_digest,
            "endpoint_integrated_phase_count": self.integrated_phase_count,
            "endpoint_integrated_weight_sum": round(self.integrated_weight_sum, 10),
        }


@dataclass(frozen=True)
class WanVAEEncodeMemoryConfig:
    """Wan VAE endpoint encode 的受控显存边界。"""

    temporal_chunk_frame_count: int = 4
    tile_sample_height: int = 128
    tile_sample_width: int = 128
    tile_sample_stride_height: int = 96
    tile_sample_stride_width: int = 96
    maximum_incremental_cuda_peak_gib: float = 16.0
    minimum_cuda_free_gib: float = 12.0

    def validate(self) -> None:
        """拒绝会改变 Wan 原生因果时间网格或产生无重叠 tile 的配置。"""

        if self.temporal_chunk_frame_count != 4:
            raise ValueError("Wan VAE 原生因果缓存要求首帧后固定按4帧分块")
        for name in (
            "tile_sample_height",
            "tile_sample_width",
            "tile_sample_stride_height",
            "tile_sample_stride_width",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} 必须为正整数")
        if self.tile_sample_stride_height >= self.tile_sample_height:
            raise ValueError("Wan VAE 纵向 tile stride 必须小于 tile height")
        if self.tile_sample_stride_width >= self.tile_sample_width:
            raise ValueError("Wan VAE 横向 tile stride 必须小于 tile width")
        if self.maximum_incremental_cuda_peak_gib <= 0.0:
            raise ValueError("Wan VAE CUDA peak budget 必须为正数")
        if self.minimum_cuda_free_gib < 0.0:
            raise ValueError("Wan VAE minimum CUDA free memory 不得为负")


def configure_wan_vae_encode_memory(
    vae: Any,
    config: WanVAEEncodeMemoryConfig | None = None,
) -> WanVAEEncodeMemoryConfig:
    """在不改变 FP32 权重边界的前提下登记 SSTW endpoint encode 策略。"""

    resolved = config or WanVAEEncodeMemoryConfig()
    resolved.validate()
    setattr(vae, "_sstw_encode_memory_config", resolved)
    if hasattr(vae, "enable_tiling"):
        vae.enable_tiling(
            tile_sample_min_height=resolved.tile_sample_height,
            tile_sample_min_width=resolved.tile_sample_width,
            tile_sample_stride_height=resolved.tile_sample_stride_height,
            tile_sample_stride_width=resolved.tile_sample_stride_width,
        )
    return resolved


def compute_endpoint_latent_evidence(
    endpoint_latent: Any,
    *,
    key_text: str,
    tubelet_config: FlowTubeletKeyCodeConfig | None = None,
    minimum_block_norm: float = 1e-8,
    key_context: FlowTubeletKeyContext | None = None,
    flow_phases: Sequence[float] | None = None,
    integration_weights: Sequence[float] | None = None,
    integrated_key_direction: Any | None = None,
    integrated_direction_formal_context_complete: bool = False,
) -> EndpointLatentEvidence:
    """使用生成阶段同一 schedule 积分 tubelet code 计算 endpoint 证据。

    正式路径必须提供 ``key_context`` 以及成对的 ``flow_phases`` 和
    ``integration_weights``，由本函数重建生成阶段的累计 joint code。旧调用仍可
    使用静态方向进行诊断，但会显式返回 ``formal_context_complete=False``。
    """

    tubelet_config = tubelet_config or FlowTubeletKeyCodeConfig()
    phases_supplied = flow_phases is not None
    weights_supplied = integration_weights is not None
    if phases_supplied != weights_supplied:
        raise ValueError("endpoint integrated direction 要求同时提供 phase 与 weight")
    if integrated_key_direction is not None:
        if tuple(integrated_key_direction.shape) != tuple(endpoint_latent.shape):
            raise ValueError("显式 endpoint integrated direction 必须与 latent 同形")
        direction = integrated_key_direction.to(
            device=endpoint_latent.device,
            dtype=endpoint_latent.dtype,
        )
        direction_norm = direction.detach().float().norm()
        if float(direction_norm.item()) <= 1e-8:
            raise ValueError("显式 endpoint integrated direction 的范数必须为正")
        direction = direction / direction_norm.to(dtype=direction.dtype)
        if phases_supplied:
            if key_context is None:
                raise ValueError("核验显式 endpoint direction 时缺少 FlowTubeletKeyContext")
            expected, metadata = build_integrated_flow_tubelet_key_direction_like(
                endpoint_latent,
                key_text=key_text,
                key_context=key_context,
                flow_phases=tuple(float(value) for value in flow_phases or ()),
                integration_weights=tuple(
                    float(value) for value in integration_weights or ()
                ),
                config=tubelet_config,
            )
            mismatch = float(
                (direction.detach().float() - expected.detach().float()).norm().item()
            )
            if mismatch > 1e-5:
                raise ValueError("显式 endpoint direction 与 phase 网格重建结果不一致")
            formal_context_complete = bool(
                integrated_direction_formal_context_complete
                and metadata["flow_tubelet_formal_context_complete"]
            )
            direction_semantics = (
                "verified_explicit_schedule_integrated_joint_tubelet_direction"
            )
            direction_digest = str(metadata["flow_key_direction_digest"])
            integrated_phase_count = len(tuple(flow_phases or ()))
            integrated_weight_sum = sum(float(value) for value in integration_weights or ())
        else:
            formal_context_complete = False
            direction_semantics = "unverified_explicit_integrated_direction"
            direction_digest = None
            integrated_phase_count = 0
            integrated_weight_sum = 0.0
    elif phases_supplied:
        if key_context is None:
            raise ValueError("正式 endpoint phase 积分缺少 FlowTubeletKeyContext")
        phases = tuple(float(value) for value in flow_phases or ())
        weights = tuple(float(value) for value in integration_weights or ())
        direction, metadata = build_integrated_flow_tubelet_key_direction_like(
            endpoint_latent,
            key_text=key_text,
            key_context=key_context,
            flow_phases=phases,
            integration_weights=weights,
            config=tubelet_config,
        )
        formal_context_complete = bool(
            metadata["flow_tubelet_formal_context_complete"]
        )
        direction_semantics = str(metadata["flow_tubelet_code_semantics"])
        direction_digest = str(metadata["flow_key_direction_digest"])
        integrated_phase_count = len(phases)
        integrated_weight_sum = sum(weights)
    else:
        if key_context is not None:
            raise ValueError("提供 key_context 时必须同时提供 endpoint phase 积分网格")
        direction, metadata = build_flow_tubelet_key_direction_like(
            endpoint_latent,
            key_text=key_text,
            config=tubelet_config,
        )
        formal_context_complete = False
        direction_semantics = "static_compatibility_tubelet_direction"
        direction_digest = str(metadata["flow_key_direction_digest"])
        integrated_phase_count = 0
        integrated_weight_sum = 0.0
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
        formal_context_complete=formal_context_complete,
        key_direction_semantics=direction_semantics,
        key_direction_digest=direction_digest,
        key_context_digest=(
            flow_tubelet_key_context_digest(key_context)
            if key_context is not None
            else None
        ),
        integrated_phase_count=integrated_phase_count,
        integrated_weight_sum=integrated_weight_sum,
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


def _vae_execution_device(vae: Any) -> Any:
    """返回 VAE forward 实际使用的设备，兼容 Accelerate CPU offload。"""

    import torch

    hook = getattr(vae, "_hf_hook", None)
    execution_device = getattr(hook, "execution_device", None)
    if execution_device is not None:
        return torch.device(execution_device)
    return next(vae.parameters()).device


def _wan_vae_memory_preflight(
    vae: Any,
    config: WanVAEEncodeMemoryConfig,
) -> dict[str, Any]:
    """在分配视频 activation 前验证设备、策略与可用显存。"""

    import torch

    config.validate()
    device = _vae_execution_device(vae)
    metadata: dict[str, Any] = {
        "endpoint_vae_encode_strategy": "cpu_resident_spatiotemporal_streaming",
        "endpoint_vae_temporal_chunk_frame_count": config.temporal_chunk_frame_count,
        "endpoint_vae_tile_sample_height": config.tile_sample_height,
        "endpoint_vae_tile_sample_width": config.tile_sample_width,
        "endpoint_vae_tile_sample_stride_height": config.tile_sample_stride_height,
        "endpoint_vae_tile_sample_stride_width": config.tile_sample_stride_width,
        "endpoint_vae_maximum_incremental_cuda_peak_gib": (
            config.maximum_incremental_cuda_peak_gib
        ),
        "endpoint_vae_minimum_cuda_free_gib": config.minimum_cuda_free_gib,
        "endpoint_vae_execution_device": str(device),
    }
    if device.type != "cuda":
        metadata.update(
            endpoint_vae_memory_preflight_status="ready_non_cuda_diagnostic",
            endpoint_vae_cuda_free_gib=None,
            endpoint_vae_cuda_total_gib=None,
        )
        return metadata
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    gib = float(1024**3)
    free_gib = float(free_bytes) / gib
    metadata.update(
        endpoint_vae_cuda_free_gib=round(free_gib, 6),
        endpoint_vae_cuda_total_gib=round(float(total_bytes) / gib, 6),
    )
    if free_gib < config.minimum_cuda_free_gib:
        raise RuntimeError(
            "Wan VAE encode 显存预检失败: "
            f"free={free_gib:.3f} GiB, required={config.minimum_cuda_free_gib:.3f} GiB"
        )
    metadata["endpoint_vae_memory_preflight_status"] = "ready"
    return metadata


def _supports_wan_streaming_encode(vae: Any) -> bool:
    """仅对具备 Diffusers Wan 因果缓存接口的 VAE 启用受控流式路径。"""

    return all(
        hasattr(vae, name)
        for name in (
            "encoder",
            "quant_conv",
            "clear_cache",
            "_enc_feat_map",
            "blend_v",
            "blend_h",
            "spatial_compression_ratio",
        )
    )


def _stream_wan_vae_moments_from_cpu(
    vae: Any,
    video_cpu: Any,
    *,
    device: Any,
    dtype: Any,
    config: WanVAEEncodeMemoryConfig,
) -> tuple[Any, dict[str, Any]]:
    """复刻 Wan tiled_encode 语义，同时只把当前时空 tile 放入 GPU。"""

    import torch

    if video_cpu.device.type != "cpu":
        raise ValueError("Wan VAE streaming 输入必须保留在 CPU")
    if getattr(vae.config, "patch_size", None) is not None:
        raise RuntimeError("SSTW streaming encode 尚不支持启用 patch_size 的 Wan VAE")
    _, _, num_frames, height, width = video_cpu.shape
    ratio = int(vae.spatial_compression_ratio)
    if ratio <= 0:
        raise RuntimeError("Wan VAE spatial compression ratio 必须为正")

    tile_h = config.tile_sample_height
    tile_w = config.tile_sample_width
    stride_h = config.tile_sample_stride_height
    stride_w = config.tile_sample_stride_width
    latent_height = height // ratio
    latent_width = width // ratio
    latent_tile_h = tile_h // ratio
    latent_tile_w = tile_w // ratio
    latent_stride_h = stride_h // ratio
    latent_stride_w = stride_w // ratio
    if min(latent_tile_h, latent_tile_w, latent_stride_h, latent_stride_w) <= 0:
        raise RuntimeError("Wan VAE tile 与 compression ratio 不兼容")

    hook = getattr(vae, "_hf_hook", None)
    if hook is not None and hasattr(hook, "pre_forward"):
        hook.pre_forward(vae)

    cuda_enabled = device.type == "cuda"
    baseline_allocated = 0
    if cuda_enabled:
        torch.cuda.synchronize(device)
        baseline_allocated = int(torch.cuda.memory_allocated(device))
        torch.cuda.reset_peak_memory_stats(device)

    rows: list[list[Any]] = []
    try:
        for i in range(0, height, stride_h):
            row: list[Any] = []
            for j in range(0, width, stride_w):
                vae.clear_cache()
                time_tiles: list[Any] = []
                frame_range = 1 + (num_frames - 1) // config.temporal_chunk_frame_count
                for k in range(frame_range):
                    vae._enc_conv_idx = [0]
                    if k == 0:
                        start, end = 0, 1
                    else:
                        start = 1 + config.temporal_chunk_frame_count * (k - 1)
                        end = min(
                            num_frames,
                            1 + config.temporal_chunk_frame_count * k,
                        )
                    tile = video_cpu[:, :, start:end, i : i + tile_h, j : j + tile_w]
                    tile = tile.to(device=device, dtype=dtype)
                    encoded = vae.encoder(
                        tile,
                        feat_cache=vae._enc_feat_map,
                        feat_idx=vae._enc_conv_idx,
                    )
                    encoded = vae.quant_conv(encoded)
                    time_tiles.append(encoded.to(device="cpu"))
                    del tile, encoded
                row.append(torch.cat(time_tiles, dim=2))
            rows.append(row)
        vae.clear_cache()

        result_rows: list[Any] = []
        blend_h = latent_tile_h - latent_stride_h
        blend_w = latent_tile_w - latent_stride_w
        for i, row in enumerate(rows):
            result_row: list[Any] = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = vae.blend_v(rows[i - 1][j], tile, blend_h)
                if j > 0:
                    tile = vae.blend_h(row[j - 1], tile, blend_w)
                result_row.append(tile[:, :, :, :latent_stride_h, :latent_stride_w])
            result_rows.append(torch.cat(result_row, dim=-1))
        moments = torch.cat(result_rows, dim=3)[
            :, :, :, :latent_height, :latent_width
        ]
    finally:
        vae.clear_cache()

    metadata: dict[str, Any] = {}
    if cuda_enabled:
        torch.cuda.synchronize(device)
        absolute_peak = int(torch.cuda.max_memory_allocated(device))
        incremental_peak = max(0, absolute_peak - baseline_allocated)
        gib = float(1024**3)
        incremental_peak_gib = float(incremental_peak) / gib
        metadata.update(
            endpoint_vae_cuda_baseline_allocated_gib=round(
                float(baseline_allocated) / gib, 6
            ),
            endpoint_vae_cuda_absolute_peak_allocated_gib=round(
                float(absolute_peak) / gib, 6
            ),
            endpoint_vae_cuda_incremental_peak_allocated_gib=round(
                incremental_peak_gib, 6
            ),
        )
        if incremental_peak_gib > config.maximum_incremental_cuda_peak_gib:
            del moments
            raise RuntimeError(
                "Wan VAE encode 超过受治理显存峰值: "
                f"peak={incremental_peak_gib:.3f} GiB, "
                f"budget={config.maximum_incremental_cuda_peak_gib:.3f} GiB"
            )
    else:
        metadata.update(
            endpoint_vae_cuda_baseline_allocated_gib=None,
            endpoint_vae_cuda_absolute_peak_allocated_gib=None,
            endpoint_vae_cuda_incremental_peak_allocated_gib=None,
        )
    return moments, metadata


def encode_video_to_wan_endpoint_latent(
    vae: Any,
    video_path: str | Path,
) -> tuple[Any, dict[str, Any]]:
    """调用真实 Wan VAE 把攻击后视频重建为生成坐标系中的 endpoint latent。"""

    import torch

    device = _vae_execution_device(vae)
    dtype = vae.dtype
    memory_config = getattr(vae, "_sstw_encode_memory_config", None)
    if memory_config is None:
        memory_config = configure_wan_vae_encode_memory(vae)
    preflight = _wan_vae_memory_preflight(vae, memory_config)
    streaming_ready = _supports_wan_streaming_encode(vae)
    video_device = torch.device("cpu") if streaming_ready else device
    video, frame_count = load_video_tensor_for_wan_vae(
        video_path,
        device=video_device,
        dtype=dtype,
    )
    with torch.inference_mode():
        if streaming_ready:
            moments, peak_metadata = _stream_wan_vae_moments_from_cpu(
                vae,
                video,
                device=device,
                dtype=dtype,
                config=memory_config,
            )
            latent = moments.chunk(2, dim=1)[0]
            del moments
        else:
            latent = _retrieve_vae_latent(vae.encode(video))
            peak_metadata = {
                "endpoint_vae_encode_strategy": "compatibility_full_tensor_encode",
                "endpoint_vae_cuda_baseline_allocated_gib": None,
                "endpoint_vae_cuda_absolute_peak_allocated_gib": None,
                "endpoint_vae_cuda_incremental_peak_allocated_gib": None,
            }
    latent = latent.to(device=device, dtype=torch.float32)
    mean_values = getattr(vae.config, "latents_mean", None)
    std_values = getattr(vae.config, "latents_std", None)
    if mean_values is None or std_values is None:
        raise RuntimeError("Wan VAE 配置缺少 latents_mean 或 latents_std")
    latent_mean = torch.tensor(mean_values, device=device, dtype=torch.float32).view(1, -1, 1, 1, 1)
    latent_std = 1.0 / torch.tensor(std_values, device=device, dtype=torch.float32).view(1, -1, 1, 1, 1)
    normalized = (latent - latent_mean) * latent_std
    return normalized, {
        **preflight,
        **peak_metadata,
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
        configure_wan_vae_encode_memory(vae)
        return cls(vae)

    def score_video(
        self,
        video_path: str | Path,
        *,
        key_text: str,
        key_context: FlowTubeletKeyContext | None = None,
        flow_phases: Sequence[float] | None = None,
        integration_weights: Sequence[float] | None = None,
    ) -> tuple[EndpointLatentEvidence, dict[str, Any]]:
        """对单个视频执行 VAE endpoint 重建和同源 key 检测。"""

        latent, metadata = encode_video_to_wan_endpoint_latent(self.vae, video_path)
        evidence = compute_endpoint_latent_evidence(
            latent,
            key_text=key_text,
            tubelet_config=self.tubelet_config,
            key_context=key_context,
            flow_phases=flow_phases,
            integration_weights=integration_weights,
        )
        return evidence, metadata
