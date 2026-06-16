"""提供生成视频文件级质量与运动指标。"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _load_sampled_frames(video_path: Path, max_frames: int = 24) -> list[Any]:
    """从视频文件中读取最多 max_frames 帧。

    该函数属于通用工程写法, 用于在无 GPU 的审阅环境中快速检查视频文件是否可解码、
    是否存在明显黑屏/白屏以及是否具有非零运动变化。
    """
    import imageio.v3 as iio

    frames: list[Any] = []
    for frame_index, frame in enumerate(iio.imiter(video_path)):
        if frame_index >= max_frames:
            break
        frames.append(frame)
    return frames


def compute_video_file_metrics(video_path: str | Path, max_frames: int = 24) -> dict:
    """计算视频文件级质量与运动指标。

    这些指标来自真实 mp4 文件解码结果, 可用于正式记录视频是否可读、亮度是否坍缩、
    以及相邻帧是否存在运动变化。它们不能替代 CLIP / VLM 类语义一致性指标。
    """
    path = Path(video_path)
    if not path.exists():
        return {
            "video_decode_status": "missing",
            "video_metric_failure_reason": "video_file_not_found",
        }

    try:
        frames = _load_sampled_frames(path, max_frames=max_frames)
    except Exception as exc:  # pragma: no cover - 依赖具体视频解码后端
        return {
            "video_decode_status": "failed",
            "video_metric_failure_reason": str(exc),
        }

    if not frames:
        return {
            "video_decode_status": "failed",
            "video_metric_failure_reason": "no_decodable_frames",
        }

    import numpy as np

    arrays = [np.asarray(frame).astype("float32") / 255.0 for frame in frames]
    gray_frames = [array.mean(axis=2) if array.ndim == 3 else array for array in arrays]
    brightness_values = [float(gray.mean()) for gray in gray_frames]
    contrast_values = [float(gray.std()) for gray in gray_frames]
    frame_deltas = [
        float(np.mean(np.abs(gray_frames[index] - gray_frames[index - 1])))
        for index in range(1, len(gray_frames))
    ]
    dark_pixel_ratio = float(np.mean([float((gray < 0.03).mean()) for gray in gray_frames]))
    bright_pixel_ratio = float(np.mean([float((gray > 0.97).mean()) for gray in gray_frames]))
    mean_brightness = float(np.mean(brightness_values))
    mean_contrast = float(np.mean(contrast_values))
    temporal_flicker_score = float(np.std(brightness_values))
    motion_delta_score = float(np.mean(frame_deltas)) if frame_deltas else 0.0

    visual_quality_pass = (
        0.03 <= mean_brightness <= 0.97
        and mean_contrast >= 0.01
        and dark_pixel_ratio < 0.95
        and bright_pixel_ratio < 0.95
    )
    motion_consistency_pass = motion_delta_score >= 0.0005 and temporal_flicker_score < 0.25

    return {
        "video_decode_status": "ready",
        "video_metric_failure_reason": "none",
        "decoded_frame_count": len(frames),
        "sampled_frame_count": len(frames),
        "mean_brightness": round(mean_brightness, 6),
        "mean_contrast": round(mean_contrast, 6),
        "dark_pixel_ratio": round(dark_pixel_ratio, 6),
        "bright_pixel_ratio": round(bright_pixel_ratio, 6),
        "temporal_flicker_score": round(temporal_flicker_score, 6),
        "motion_delta_score": round(motion_delta_score, 6),
        "visual_quality_metric_status": "ready" if visual_quality_pass else "failed",
        "motion_consistency_metric_status": "ready" if motion_consistency_pass else "failed",
    }
