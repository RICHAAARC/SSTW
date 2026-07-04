"""视频水印 runtime attack 协议与轻量帧级实现。

该模块的职责是把论文协议中的 attack 名称、分层覆盖要求和实际帧级变换
集中管理。这样 Notebook、SSTW 主流程和 external baseline official reference
可以共享同一套 attack 语义, 避免各文件各自硬写 attack 列表。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class RuntimeAttackSpec:
    """描述一个 runtime attack 的协议语义。

    `implementation_level` 用于区分当前仓库已实现的轻量运行时变换与 full paper
    级真实平台转码 / 生成式编辑攻击。当前字段只描述产物性质, 不直接支持论文
    效果 claim。
    """

    attack_name: str
    attack_family: str
    attack_transform: str
    attack_strength: str
    runtime_attack_expected_effect: str
    implementation_level: str = "repository_lightweight_runtime_transform"
    video_writer_codec: str | None = None
    video_writer_output_params: tuple[str, ...] = ()


VALIDATION_SCALE_RUNTIME_ATTACKS = (
    "video_compression_runtime",
    "temporal_crop_runtime",
    "frame_rate_resampling_runtime",
)

PILOT_PAPER_RUNTIME_ATTACKS = (
    "video_compression_runtime",
    "temporal_crop_runtime",
    "frame_rate_resampling_runtime",
    "frame_drop_uniform_runtime",
    "spatial_resize_runtime",
    "spatial_crop_resize_runtime",
    "gaussian_blur_runtime",
    "gaussian_noise_runtime",
)

FULL_PAPER_RUNTIME_ATTACKS = (
    "h264_crf23_runtime",
    "h264_crf33_runtime",
    "h265_crf28_runtime",
    "mpeg4_crf28_runtime",
    "temporal_crop_runtime",
    "frame_rate_resampling_runtime",
    "frame_drop_uniform_runtime",
    "frame_insert_duplicate_runtime",
    "frame_swap_adjacent_runtime",
    "frame_average_runtime",
    "spatial_resize_runtime",
    "spatial_crop_resize_runtime",
    "rotation_runtime",
    "perspective_runtime",
    "gaussian_noise_runtime",
    "gaussian_blur_runtime",
    "median_blur_runtime",
    "brightness_contrast_runtime",
    "compression_crop_combined_runtime",
    "compression_brightness_combined_runtime",
    "compression_temporal_combined_runtime",
)


RUNTIME_ATTACK_SPECS: dict[str, RuntimeAttackSpec] = {
    "video_compression_runtime": RuntimeAttackSpec(
        "video_compression_runtime",
        "compression",
        "decode_reencode",
        "runtime_reencode_default_quality",
        "codec_quantization_or_container_rewrite",
        video_writer_codec="libx264",
        video_writer_output_params=("-crf", "28"),
    ),
    "h264_crf23_runtime": RuntimeAttackSpec(
        "h264_crf23_runtime",
        "compression",
        "h264_reencode",
        "crf_23",
        "h264_codec_quantization",
        video_writer_codec="libx264",
        video_writer_output_params=("-crf", "23"),
    ),
    "h264_crf33_runtime": RuntimeAttackSpec(
        "h264_crf33_runtime",
        "compression",
        "h264_reencode",
        "crf_33",
        "h264_codec_quantization",
        video_writer_codec="libx264",
        video_writer_output_params=("-crf", "33"),
    ),
    "h265_crf28_runtime": RuntimeAttackSpec(
        "h265_crf28_runtime",
        "compression",
        "h265_reencode",
        "crf_28",
        "h265_codec_quantization",
        video_writer_codec="libx265",
        video_writer_output_params=("-crf", "28"),
    ),
    "mpeg4_crf28_runtime": RuntimeAttackSpec(
        "mpeg4_crf28_runtime",
        "compression",
        "mpeg4_reencode",
        "crf_28",
        "mpeg4_codec_quantization",
        video_writer_codec="mpeg4",
        video_writer_output_params=("-q:v", "5"),
    ),
    "temporal_crop_runtime": RuntimeAttackSpec(
        "temporal_crop_runtime",
        "temporal",
        "drop_first_and_last_frame_when_possible",
        "crop_boundary_frames",
        "temporal_boundary_shift",
    ),
    "frame_rate_resampling_runtime": RuntimeAttackSpec(
        "frame_rate_resampling_runtime",
        "temporal",
        "keep_every_second_frame_when_possible",
        "fps_downsample_by_2_proxy",
        "time_grid_resampling",
    ),
    "frame_drop_uniform_runtime": RuntimeAttackSpec(
        "frame_drop_uniform_runtime",
        "temporal",
        "drop_every_third_frame_when_possible",
        "uniform_frame_drop_1_over_3",
        "temporal_frame_loss",
    ),
    "frame_insert_duplicate_runtime": RuntimeAttackSpec(
        "frame_insert_duplicate_runtime",
        "temporal",
        "duplicate_middle_frame_when_possible",
        "single_frame_insert_duplicate",
        "temporal_frame_insertion",
    ),
    "frame_swap_adjacent_runtime": RuntimeAttackSpec(
        "frame_swap_adjacent_runtime",
        "temporal",
        "swap_middle_adjacent_frames_when_possible",
        "single_adjacent_swap",
        "temporal_order_disturbance",
    ),
    "frame_average_runtime": RuntimeAttackSpec(
        "frame_average_runtime",
        "temporal",
        "average_adjacent_frames",
        "adjacent_frame_average",
        "temporal_smoothing",
    ),
    "spatial_resize_runtime": RuntimeAttackSpec(
        "spatial_resize_runtime",
        "spatial_geometry",
        "downsample_then_restore_resolution",
        "resize_ratio_0_75",
        "spatial_resampling",
    ),
    "spatial_crop_resize_runtime": RuntimeAttackSpec(
        "spatial_crop_resize_runtime",
        "spatial_geometry",
        "center_crop_then_restore_resolution",
        "center_crop_ratio_0_80",
        "spatial_crop_and_rescale",
    ),
    "rotation_runtime": RuntimeAttackSpec(
        "rotation_runtime",
        "spatial_geometry",
        "small_angle_rotate_then_restore_canvas",
        "rotation_5_degrees",
        "spatial_geometric_desynchronization",
    ),
    "perspective_runtime": RuntimeAttackSpec(
        "perspective_runtime",
        "spatial_geometry",
        "lightweight_affine_roll_proxy",
        "horizontal_roll_2_pixels",
        "perspective_like_spatial_desynchronization",
    ),
    "gaussian_noise_runtime": RuntimeAttackSpec(
        "gaussian_noise_runtime",
        "visual_degradation",
        "deterministic_gaussian_like_noise",
        "sigma_4_uint8",
        "pixel_value_noise",
    ),
    "gaussian_blur_runtime": RuntimeAttackSpec(
        "gaussian_blur_runtime",
        "visual_degradation",
        "gaussian_blur",
        "radius_1",
        "local_low_pass_filtering",
    ),
    "median_blur_runtime": RuntimeAttackSpec(
        "median_blur_runtime",
        "visual_degradation",
        "median_filter",
        "kernel_3",
        "impulse_noise_suppression_like_filtering",
    ),
    "brightness_contrast_runtime": RuntimeAttackSpec(
        "brightness_contrast_runtime",
        "visual_degradation",
        "brightness_contrast_adjustment",
        "brightness_1_08_contrast_1_10",
        "global_color_value_shift",
    ),
    "compression_crop_combined_runtime": RuntimeAttackSpec(
        "compression_crop_combined_runtime",
        "combined",
        "decode_reencode_plus_center_crop_resize",
        "h264_crf_28_and_crop_ratio_0_80",
        "combined_codec_and_spatial_desynchronization",
        video_writer_codec="libx264",
        video_writer_output_params=("-crf", "28"),
    ),
    "compression_brightness_combined_runtime": RuntimeAttackSpec(
        "compression_brightness_combined_runtime",
        "combined",
        "decode_reencode_plus_brightness_contrast",
        "h264_crf_28_and_brightness_contrast",
        "combined_codec_and_value_shift",
        video_writer_codec="libx264",
        video_writer_output_params=("-crf", "28"),
    ),
    "compression_temporal_combined_runtime": RuntimeAttackSpec(
        "compression_temporal_combined_runtime",
        "combined",
        "decode_reencode_plus_frame_drop",
        "h264_crf_28_and_uniform_frame_drop",
        "combined_codec_and_temporal_frame_loss",
        video_writer_codec="libx264",
        video_writer_output_params=("-crf", "28"),
    ),
}


def runtime_attack_names_for_profile(profile_name: str) -> tuple[str, ...]:
    """按 workflow profile 返回默认 runtime attack 列表。"""

    normalized = str(profile_name or "").strip().lower()
    if normalized == "full_paper":
        return FULL_PAPER_RUNTIME_ATTACKS
    if normalized == "pilot_paper":
        return PILOT_PAPER_RUNTIME_ATTACKS
    return VALIDATION_SCALE_RUNTIME_ATTACKS


def required_runtime_attack_names_from_config(config: Mapping[str, Any]) -> tuple[str, ...]:
    """从 protocol config 读取 required runtime attack 名称集合。"""

    explicit = config.get("required_runtime_attack_names")
    if isinstance(explicit, list) and explicit:
        return tuple(str(item) for item in explicit if str(item))
    return runtime_attack_names_for_profile(str(config.get("paper_result_level") or "validation_scale"))


def runtime_attack_spec(attack_name: str) -> RuntimeAttackSpec:
    """读取单个 attack 的规范定义, 未注册名称直接失败。"""

    normalized = str(attack_name or "").strip()
    if normalized not in RUNTIME_ATTACK_SPECS:
        raise ValueError(f"unsupported_runtime_attack:{attack_name}")
    return RUNTIME_ATTACK_SPECS[normalized]


def _to_numpy(frame: Any) -> Any:
    """把一帧转换为 numpy array, 避免在模块导入时强依赖 numpy。"""

    import numpy as np

    return np.asarray(frame)


def _clip_like_uint8(frame: Any) -> Any:
    """将帧裁剪到 uint8 图像范围。"""

    import numpy as np

    return np.clip(frame, 0, 255).astype(np.uint8)


def _resize_frame(frame: Any, size: tuple[int, int]) -> Any:
    """使用 PIL 对单帧做 resize。"""

    import numpy as np
    from PIL import Image

    array = _to_numpy(frame)
    image = Image.fromarray(array.astype(np.uint8))
    resized = image.resize(size, Image.BILINEAR)
    return np.asarray(resized).astype(array.dtype)


def _blur_frame(frame: Any, attack_name: str) -> Any:
    """对单帧做 blur 或 median filter。"""

    import numpy as np
    from PIL import Image, ImageFilter

    array = _to_numpy(frame)
    image = Image.fromarray(array.astype(np.uint8))
    if attack_name == "median_blur_runtime":
        filtered = image.filter(ImageFilter.MedianFilter(size=3))
    else:
        filtered = image.filter(ImageFilter.GaussianBlur(radius=1))
    return np.asarray(filtered).astype(array.dtype)


def _spatial_resize(frames: list[Any], ratio: float = 0.75) -> list[Any]:
    """先缩小再恢复到原始分辨率, 模拟 resize 类攻击。"""

    if not frames:
        return []
    first = _to_numpy(frames[0])
    height, width = int(first.shape[0]), int(first.shape[1])
    small = (max(1, int(width * ratio)), max(1, int(height * ratio)))
    restored = (width, height)
    return [_resize_frame(_resize_frame(frame, small), restored) for frame in frames]


def _spatial_crop_resize(frames: list[Any], crop_ratio: float = 0.80) -> list[Any]:
    """中心裁剪后恢复到原始分辨率。"""

    import numpy as np

    attacked: list[Any] = []
    for frame in frames:
        array = _to_numpy(frame)
        height, width = int(array.shape[0]), int(array.shape[1])
        crop_h = max(1, int(height * crop_ratio))
        crop_w = max(1, int(width * crop_ratio))
        top = max(0, (height - crop_h) // 2)
        left = max(0, (width - crop_w) // 2)
        cropped = array[top : top + crop_h, left : left + crop_w]
        attacked.append(_resize_frame(cropped, (width, height)).astype(array.dtype))
    return attacked


def _rotate_frames(frames: list[Any]) -> list[Any]:
    """小角度旋转帧, 用于空间同步扰动。"""

    import numpy as np
    from PIL import Image

    attacked: list[Any] = []
    for frame in frames:
        array = _to_numpy(frame)
        image = Image.fromarray(array.astype(np.uint8))
        rotated = image.rotate(5, resample=Image.BILINEAR)
        attacked.append(np.asarray(rotated).astype(array.dtype))
    return attacked


def _brightness_contrast(frames: list[Any]) -> list[Any]:
    """调整亮度和对比度。"""

    import numpy as np

    attacked: list[Any] = []
    for frame in frames:
        array = _to_numpy(frame).astype(np.float32)
        mean = array.mean(axis=(0, 1), keepdims=True)
        adjusted = (array - mean) * 1.10 + mean
        adjusted = adjusted * 1.08
        attacked.append(_clip_like_uint8(adjusted).astype(_to_numpy(frame).dtype))
    return attacked


def _gaussian_noise(frames: list[Any]) -> list[Any]:
    """添加确定性高斯近似噪声, 保证测试可复现。"""

    import numpy as np

    rng = np.random.default_rng(20260704)
    attacked: list[Any] = []
    for frame in frames:
        array = _to_numpy(frame)
        noise = rng.normal(loc=0.0, scale=4.0, size=array.shape)
        attacked.append(_clip_like_uint8(array.astype(np.float32) + noise).astype(array.dtype))
    return attacked


def apply_runtime_attack_to_frames(frames: list[Any], attack_name: str) -> tuple[list[Any], dict[str, Any]]:
    """对解码帧执行指定 runtime attack。

    返回值中的 metadata 是 governed records 的协议证据字段。该函数只做轻量
    可复现变换, 不声称等同于社交媒体真实平台转码或生成式重生成攻击。
    """

    if not frames:
        raise ValueError("no_decodable_frames")
    spec = runtime_attack_spec(attack_name)
    normalized = spec.attack_name
    attacked = list(frames)

    if normalized in {
        "video_compression_runtime",
        "h264_crf23_runtime",
        "h264_crf33_runtime",
        "h265_crf28_runtime",
        "mpeg4_crf28_runtime",
    }:
        attacked = list(frames)
    elif normalized == "temporal_crop_runtime":
        attacked = frames[1:-1] if len(frames) >= 4 else list(frames)
    elif normalized == "frame_rate_resampling_runtime":
        attacked = frames[::2] if len(frames) >= 3 else list(frames)
    elif normalized == "frame_drop_uniform_runtime":
        attacked = [frame for index, frame in enumerate(frames) if (index + 1) % 3 != 0] or list(frames)
    elif normalized == "frame_insert_duplicate_runtime":
        midpoint = max(0, len(frames) // 2)
        attacked = list(frames[:midpoint]) + [frames[midpoint]] + list(frames[midpoint:])
    elif normalized == "frame_swap_adjacent_runtime":
        attacked = list(frames)
        if len(attacked) >= 4:
            midpoint = len(attacked) // 2
            attacked[midpoint - 1], attacked[midpoint] = attacked[midpoint], attacked[midpoint - 1]
    elif normalized == "frame_average_runtime":
        import numpy as np

        attacked = []
        for index, frame in enumerate(frames):
            if index == 0:
                attacked.append(frame)
            else:
                averaged = (_to_numpy(frames[index - 1]).astype(np.float32) + _to_numpy(frame).astype(np.float32)) / 2.0
                attacked.append(_clip_like_uint8(averaged).astype(_to_numpy(frame).dtype))
    elif normalized == "spatial_resize_runtime":
        attacked = _spatial_resize(frames)
    elif normalized == "spatial_crop_resize_runtime":
        attacked = _spatial_crop_resize(frames)
    elif normalized == "rotation_runtime":
        attacked = _rotate_frames(frames)
    elif normalized == "perspective_runtime":
        import numpy as np

        attacked = [np.roll(_to_numpy(frame), shift=2, axis=1).astype(_to_numpy(frame).dtype) for frame in frames]
    elif normalized == "gaussian_noise_runtime":
        attacked = _gaussian_noise(frames)
    elif normalized in {"gaussian_blur_runtime", "median_blur_runtime"}:
        attacked = [_blur_frame(frame, normalized) for frame in frames]
    elif normalized == "brightness_contrast_runtime":
        attacked = _brightness_contrast(frames)
    elif normalized == "compression_crop_combined_runtime":
        attacked = _spatial_crop_resize(frames)
    elif normalized == "compression_brightness_combined_runtime":
        attacked = _brightness_contrast(frames)
    elif normalized == "compression_temporal_combined_runtime":
        attacked = [frame for index, frame in enumerate(frames) if (index + 1) % 3 != 0] or list(frames)
    else:  # pragma: no cover - runtime_attack_spec 已经先行阻断未知名称。
        raise ValueError(f"unsupported_runtime_attack:{attack_name}")

    metadata: dict[str, Any] = {
        "attack_family": spec.attack_family,
        "attack_transform": spec.attack_transform,
        "attack_strength": spec.attack_strength,
        "runtime_attack_expected_effect": spec.runtime_attack_expected_effect,
        "runtime_attack_implementation_level": spec.implementation_level,
    }
    if spec.video_writer_codec:
        metadata["video_writer_codec"] = spec.video_writer_codec
    if spec.video_writer_output_params:
        metadata["video_writer_output_params"] = list(spec.video_writer_output_params)
    return attacked, metadata


def apply_runtime_attack_to_video_tensor(video: Any, attack_name: str) -> Any:
    """对 T-first 视频张量执行轻量 runtime attack。

    该函数用于 VideoSeal 这类以张量形式返回视频的 official runtime。若遇到
    空间或像素值攻击, 会尽量保持输入类型不变; 复杂几何攻击采用轻量近似。
    """

    spec = runtime_attack_spec(attack_name)
    normalized = spec.attack_name
    if normalized in {
        "video_compression_runtime",
        "h264_crf23_runtime",
        "h264_crf33_runtime",
        "h265_crf28_runtime",
        "mpeg4_crf28_runtime",
    }:
        return video
    if normalized == "temporal_crop_runtime":
        return video[1:-1] if video.shape[0] >= 4 else video
    if normalized in {"frame_rate_resampling_runtime", "compression_temporal_combined_runtime"}:
        return video[::2] if video.shape[0] >= 3 else video
    if normalized == "frame_drop_uniform_runtime":
        indices = [index for index in range(int(video.shape[0])) if (index + 1) % 3 != 0]
        return video[indices] if indices else video
    if normalized == "frame_insert_duplicate_runtime":
        midpoint = max(0, int(video.shape[0]) // 2)
        return _tensor_concat([video[:midpoint], video[midpoint : midpoint + 1], video[midpoint:]])
    if normalized == "frame_swap_adjacent_runtime":
        output = video.clone() if hasattr(video, "clone") else video.copy()
        if output.shape[0] >= 4:
            midpoint = int(output.shape[0]) // 2
            before = output[midpoint - 1].clone() if hasattr(output[midpoint - 1], "clone") else output[midpoint - 1].copy()
            output[midpoint - 1] = output[midpoint]
            output[midpoint] = before
        return output
    if normalized == "frame_average_runtime":
        output = video.clone() if hasattr(video, "clone") else video.copy()
        if output.shape[0] >= 2:
            output[1:] = (output[:-1] + output[1:]) / 2
        return output
    if normalized in {"spatial_resize_runtime", "spatial_crop_resize_runtime", "compression_crop_combined_runtime"}:
        return _tensor_center_crop(video, crop_ratio=0.80 if "crop" in normalized else 0.75)
    if normalized in {"rotation_runtime", "perspective_runtime"}:
        return _tensor_roll(video, shift=2)
    if normalized == "gaussian_noise_runtime":
        return _tensor_add_noise(video)
    if normalized in {"gaussian_blur_runtime", "median_blur_runtime"}:
        return _tensor_average_blur(video)
    if normalized in {"brightness_contrast_runtime", "compression_brightness_combined_runtime"}:
        return _tensor_brightness_contrast(video)
    raise ValueError(f"unsupported_runtime_attack:{attack_name}")


def _tensor_concat(parts: list[Any]) -> Any:
    """按输入类型拼接张量或数组。"""

    if not parts:
        return parts
    if hasattr(parts[0], "new_empty"):
        import torch

        return torch.cat(parts, dim=0)
    import numpy as np

    return np.concatenate(parts, axis=0)


def _tensor_center_crop(video: Any, crop_ratio: float) -> Any:
    """对 T-first 张量做中心裁剪并恢复原分辨率。"""

    shape = video.shape
    if len(shape) < 4:
        return video
    height_axis, width_axis = (-2, -1) if shape[1] in {1, 3, 4} else (1, 2)
    height, width = int(shape[height_axis]), int(shape[width_axis])
    crop_h = max(1, int(height * crop_ratio))
    crop_w = max(1, int(width * crop_ratio))
    top = max(0, (height - crop_h) // 2)
    left = max(0, (width - crop_w) // 2)
    if height_axis == -2:
        cropped = video[..., top : top + crop_h, left : left + crop_w]
    else:
        cropped = video[:, top : top + crop_h, left : left + crop_w, ...]
    if hasattr(video, "new_empty"):
        import torch.nn.functional as F

        channels_first = cropped if height_axis == -2 else cropped.permute(0, 3, 1, 2)
        resized = F.interpolate(channels_first.float(), size=(height, width), mode="bilinear", align_corners=False)
        resized = resized.to(dtype=video.dtype)
        return resized if height_axis == -2 else resized.permute(0, 2, 3, 1)
    import numpy as np

    frames = [frame for frame in cropped]
    restored = [_resize_frame(frame, (width, height)) for frame in frames]
    return np.asarray(restored).astype(video.dtype)


def _tensor_roll(video: Any, shift: int) -> Any:
    """对张量做水平 roll, 作为轻量几何扰动。"""

    if hasattr(video, "roll"):
        return video.roll(shifts=shift, dims=-1)
    import numpy as np

    return np.roll(video, shift=shift, axis=-2 if video.shape[-1] in {1, 3, 4} else -1)


def _tensor_add_noise(video: Any) -> Any:
    """对张量添加小幅噪声。"""

    if hasattr(video, "new_empty"):
        import torch

        generator = torch.Generator(device=video.device)
        generator.manual_seed(20260704)
        noise = torch.randn(video.shape, generator=generator, device=video.device, dtype=video.dtype) * 0.015
        return (video + noise).clamp(0.0, 1.0)
    import numpy as np

    rng = np.random.default_rng(20260704)
    return np.clip(video.astype("float32") + rng.normal(0.0, 4.0, size=video.shape), 0, 255).astype(video.dtype)


def _tensor_average_blur(video: Any) -> Any:
    """使用平均池化近似 blur。"""

    if hasattr(video, "new_empty"):
        import torch.nn.functional as F

        channels_first = video if video.shape[1] in {1, 3, 4} else video.permute(0, 3, 1, 2)
        blurred = F.avg_pool2d(channels_first.float(), kernel_size=3, stride=1, padding=1).to(dtype=video.dtype)
        return blurred if video.shape[1] in {1, 3, 4} else blurred.permute(0, 2, 3, 1)
    return video


def _tensor_brightness_contrast(video: Any) -> Any:
    """调整张量亮度和对比度。"""

    if hasattr(video, "new_empty"):
        mean = video.mean()
        return (((video - mean) * 1.10 + mean) * 1.08).clamp(0.0, 1.0)
    import numpy as np

    mean = video.mean()
    return np.clip(((video.astype("float32") - mean) * 1.10 + mean) * 1.08, 0, 255).astype(video.dtype)
