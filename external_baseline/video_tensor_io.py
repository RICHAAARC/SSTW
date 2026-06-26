"""external baseline 视频张量 I/O。

该模块提供不依赖 `torchvision.io.read_video` / `write_video` 的视频读写能力。
这是对 Colab 新版 torchvision 环境的适配: 部分 Colab 镜像中已经移除了
torchvision 的旧视频 I/O API, 因此正式 baseline 运行不能依赖该接口。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _normalize_frame_rgb(frame: Any) -> Any:
    """把 imageio 读出的单帧统一成 HWC RGB uint8 数组。"""
    import numpy as np

    array = np.asarray(frame)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    if array.ndim != 3:
        raise RuntimeError(f"unsupported_video_frame_shape:{array.shape}")
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    elif array.shape[2] >= 4:
        array = array[:, :, :3]
    elif array.shape[2] != 3:
        raise RuntimeError(f"unsupported_video_channel_count:{array.shape[2]}")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def _read_video_fps(video_path: str | Path, default_fps: float = 8.0) -> float:
    """从视频 metadata 读取 fps, 失败时返回协议默认值。"""
    import imageio.v3 as iio

    try:
        metadata = dict(iio.immeta(video_path))
    except Exception:
        return float(default_fps)
    for key in ("fps", "video_fps"):
        value = metadata.get(key)
        if value:
            try:
                fps = float(value)
            except (TypeError, ValueError):
                continue
            if fps > 0:
                return fps
    return float(default_fps)


def read_video_tchw_uint8(video_path: str | Path, *, empty_error: str = "video_empty") -> tuple[Any, dict[str, Any]]:
    """读取视频文件并返回 `[T, C, H, W]` uint8 tensor。

    通用工程写法是把文件级视频 I/O 收敛到一个独立模块, 避免不同 baseline wrapper
    直接绑定某个第三方库。项目特定考虑是: Colab 环境中的 torchvision 视频接口不稳定,
    因此这里使用 imageio v3 作为默认视频 I/O 后端。
    """
    import imageio.v3 as iio
    import numpy as np
    import torch

    frames = [_normalize_frame_rgb(frame) for frame in iio.imiter(video_path)]
    if not frames:
        raise RuntimeError(empty_error)
    array = np.stack(frames, axis=0)
    tensor = torch.from_numpy(array).permute(0, 3, 1, 2).contiguous()
    return tensor, {
        "video_fps": _read_video_fps(video_path),
        "video_frame_count": int(tensor.shape[0]),
        "video_io_backend": "imageio_v3",
    }


def video_tensor_to_uint8_thwc(video: Any) -> Any:
    """把 `[T, C, H, W]` 或 `[T, H, W, C]` 视频张量转换为 uint8 THWC 数组。"""
    import torch

    if video.ndim == 4 and video.shape[1] in {1, 3, 4}:
        video = video[:, :3].permute(0, 2, 3, 1)
    elif video.ndim == 4 and video.shape[-1] in {1, 3, 4}:
        video = video[..., :3]
    else:
        raise RuntimeError(f"unsupported_video_tensor_shape:{tuple(video.shape)}")
    if video.shape[-1] == 1:
        video = video.repeat(1, 1, 1, 3)
    video = video.detach().cpu()
    if not torch.is_floating_point(video):
        video = video.float() / 255.0
    else:
        video = video.float()
    video = video.clamp(0.0, 1.0)
    return (video * 255.0).round().to(torch.uint8).numpy()


def write_video_tchw(video_path: str | Path, video: Any, *, fps: float = 8.0) -> dict[str, Any]:
    """把视频张量写为 mp4 文件, 并返回 I/O 后端摘要。"""
    import imageio.v3 as iio

    output_path = Path(video_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    array = video_tensor_to_uint8_thwc(video)
    iio.imwrite(output_path, array, fps=float(fps))
    return {
        "video_path": str(output_path),
        "video_fps": float(fps),
        "video_frame_count": int(array.shape[0]),
        "video_io_backend": "imageio_v3",
    }
