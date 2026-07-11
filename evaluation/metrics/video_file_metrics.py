"""提供生成视频文件级质量与运动指标。"""

from __future__ import annotations

from pathlib import Path
from typing import Any


VISUAL_BRIGHTNESS_MIN = 0.03
VISUAL_BRIGHTNESS_MAX = 0.97
VISUAL_CONTRAST_MIN = 0.01
VISUAL_EXTREME_PIXEL_RATIO_MAX = 0.95
MOTION_DELTA_MIN = 0.0005
TEMPORAL_FLICKER_MAX = 0.25


def _load_sampled_frames(video_path: Path, max_frames: int = 24) -> list[Any]:
    """从视频文件中读取最多 max_frames 帧。

    该函数属于通用工程写法, 用于在无 GPU 的审计环境中快速检查视频文件是否可解码,
    以及是否存在明显黑屏、白屏或非零运动变化。其他项目也可以复用这一结构做轻量视频健康检查。
    """
    import imageio.v3 as iio

    frames: list[Any] = []
    for frame_index, frame in enumerate(iio.imiter(video_path)):
        if frame_index >= max_frames:
            break
        frames.append(frame)
    return frames


def _visual_quality_failure_reason(
    mean_brightness: float,
    mean_contrast: float,
    dark_pixel_ratio: float,
    bright_pixel_ratio: float,
) -> str:
    """根据视觉质量阈值生成可审计的失败原因。"""
    reasons: list[str] = []
    if mean_brightness < VISUAL_BRIGHTNESS_MIN:
        reasons.append("mean_brightness_below_min")
    if mean_brightness > VISUAL_BRIGHTNESS_MAX:
        reasons.append("mean_brightness_above_max")
    if mean_contrast < VISUAL_CONTRAST_MIN:
        reasons.append("mean_contrast_below_min")
    if dark_pixel_ratio >= VISUAL_EXTREME_PIXEL_RATIO_MAX:
        reasons.append("dark_pixel_ratio_above_max")
    if bright_pixel_ratio >= VISUAL_EXTREME_PIXEL_RATIO_MAX:
        reasons.append("bright_pixel_ratio_above_max")
    return "none" if not reasons else ";".join(reasons)


def _motion_consistency_failure_reason(motion_delta_score: float, temporal_flicker_score: float) -> str:
    """根据运动一致性阈值生成可审计的失败原因。

    当前 formal motion gate 的用途不是判断视频是否“好看”, 而是阻止几乎静止的视频支撑
    motion-related claim。因此失败原因需要区分低运动量和高闪烁。
    """
    reasons: list[str] = []
    if motion_delta_score < MOTION_DELTA_MIN:
        reasons.append("motion_delta_below_min")
    if temporal_flicker_score >= TEMPORAL_FLICKER_MAX:
        reasons.append("temporal_flicker_above_max")
    return "none" if not reasons else ";".join(reasons)


def _motion_delta_statistics(gray_frames: list[Any]) -> dict[str, float]:
    """计算多粒度帧间运动统计量。

    通用工程写法是先保留整帧平均差分 `motion_delta_score`, 便于兼容历史 records。
    项目特定写法是新增 `motion_delta_focus_score`: 它使用每对相邻帧的高差分区域均值减去中位差分,
    用来削弱全局曝光漂移或均匀闪烁, 同时提高对局部物体运动的敏感度。该指标仍是文件级 proxy,
    不能替代光流或人工语义审计, 但更适合 motion threshold calibration。
    """
    import numpy as np

    if len(gray_frames) < 2:
        return {
            "motion_delta_score": 0.0,
            "motion_delta_p90_score": 0.0,
            "motion_delta_top10_mean_score": 0.0,
            "motion_delta_focus_score": 0.0,
            "motion_delta_focus_to_mean_ratio": 0.0,
        }

    mean_deltas: list[float] = []
    p90_deltas: list[float] = []
    top10_deltas: list[float] = []
    focus_deltas: list[float] = []
    for index in range(1, len(gray_frames)):
        diff = np.abs(gray_frames[index] - gray_frames[index - 1])
        mean_delta = float(np.mean(diff))
        median_delta = float(np.median(diff))
        p90_delta = float(np.quantile(diff, 0.90))
        top10_mean_delta = float(np.mean(diff[diff >= p90_delta])) if diff.size else 0.0
        focus_delta = max(top10_mean_delta - median_delta, 0.0)
        mean_deltas.append(mean_delta)
        p90_deltas.append(p90_delta)
        top10_deltas.append(top10_mean_delta)
        focus_deltas.append(focus_delta)

    motion_delta_score = float(np.mean(mean_deltas))
    motion_delta_focus_score = float(np.mean(focus_deltas))
    ratio = motion_delta_focus_score / motion_delta_score if motion_delta_score > 0 else 0.0
    return {
        "motion_delta_score": motion_delta_score,
        "motion_delta_p90_score": float(np.mean(p90_deltas)),
        "motion_delta_top10_mean_score": float(np.mean(top10_deltas)),
        "motion_delta_focus_score": motion_delta_focus_score,
        "motion_delta_focus_to_mean_ratio": ratio,
    }


def compute_video_file_metrics(video_path: str | Path, max_frames: int = 24) -> dict:
    """计算视频文件级质量与运动指标。

    这些指标来自真实 mp4 文件解码结果, 可用于正式记录视频是否可读、亮度是否健康,
    以及相邻帧是否存在足够运动变化。它们不能替代 CLIP / VLM 类语义一致性指标。
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
    motion_stats = _motion_delta_statistics(gray_frames)
    dark_pixel_ratio = float(np.mean([float((gray < 0.03).mean()) for gray in gray_frames]))
    bright_pixel_ratio = float(np.mean([float((gray > 0.97).mean()) for gray in gray_frames]))
    mean_brightness = float(np.mean(brightness_values))
    mean_contrast = float(np.mean(contrast_values))
    temporal_flicker_score = float(np.std(brightness_values))
    motion_delta_score = motion_stats["motion_delta_score"]

    visual_failure_reason = _visual_quality_failure_reason(
        mean_brightness,
        mean_contrast,
        dark_pixel_ratio,
        bright_pixel_ratio,
    )
    motion_failure_reason = _motion_consistency_failure_reason(motion_delta_score, temporal_flicker_score)
    visual_quality_pass = visual_failure_reason == "none"
    motion_consistency_pass = motion_failure_reason == "none"

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
        "motion_delta_p90_score": round(motion_stats["motion_delta_p90_score"], 6),
        "motion_delta_top10_mean_score": round(motion_stats["motion_delta_top10_mean_score"], 6),
        "motion_delta_focus_score": round(motion_stats["motion_delta_focus_score"], 6),
        "motion_delta_focus_to_mean_ratio": round(motion_stats["motion_delta_focus_to_mean_ratio"], 6),
        "motion_calibration_score_name": "motion_delta_focus_score",
        "visual_brightness_min": VISUAL_BRIGHTNESS_MIN,
        "visual_brightness_max": VISUAL_BRIGHTNESS_MAX,
        "visual_contrast_min": VISUAL_CONTRAST_MIN,
        "visual_extreme_pixel_ratio_max": VISUAL_EXTREME_PIXEL_RATIO_MAX,
        "motion_delta_threshold": MOTION_DELTA_MIN,
        "temporal_flicker_threshold": TEMPORAL_FLICKER_MAX,
        "visual_quality_metric_status": "ready" if visual_quality_pass else "failed",
        "motion_consistency_metric_status": "ready" if motion_consistency_pass else "failed",
        "visual_quality_failure_reason": visual_failure_reason,
        "motion_consistency_failure_reason": motion_failure_reason,
    }


def compute_paired_video_quality_metrics(
    reference_video_path: str | Path,
    candidate_video_path: str | Path,
    *,
    max_frames: int = 24,
) -> dict[str, Any]:
    """计算 clean-reference 与 watermarked 视频的配对失真指标。

    PSNR、SSIM 和 temporal-delta error 都逐帧使用同一个 prompt/seed 的 clean
    reference。它们衡量水印引入的实际失真, 不再用单视频亮度或对比度替代质量。
    """

    import math
    import numpy as np
    from PIL import Image
    from skimage.metrics import structural_similarity

    reference_path = Path(reference_video_path)
    candidate_path = Path(candidate_video_path)
    if not reference_path.exists() or not candidate_path.exists():
        return {
            "paired_video_quality_status": "missing_video",
            "paired_video_quality_failure_reason": "reference_or_candidate_missing",
        }
    try:
        reference_frames = _load_sampled_frames(reference_path, max_frames=max_frames)
        candidate_frames = _load_sampled_frames(candidate_path, max_frames=max_frames)
    except Exception as exc:  # pragma: no cover - 依赖具体 codec
        return {
            "paired_video_quality_status": "decode_failed",
            "paired_video_quality_failure_reason": str(exc),
        }
    count = min(len(reference_frames), len(candidate_frames))
    if count < 2:
        return {
            "paired_video_quality_status": "insufficient_frames",
            "paired_video_quality_failure_reason": "paired_metric_requires_two_frames",
        }
    references: list[Any] = []
    candidates: list[Any] = []
    for index in range(count):
        reference = np.asarray(reference_frames[index])[..., :3].astype(np.float64)
        candidate = np.asarray(candidate_frames[index])[..., :3].astype(np.float64)
        if candidate.shape != reference.shape:
            candidate = np.asarray(
                Image.fromarray(candidate.astype(np.uint8)).resize(
                    (reference.shape[1], reference.shape[0]),
                    Image.Resampling.BICUBIC,
                ),
                dtype=np.float64,
            )
        references.append(reference)
        candidates.append(candidate)
    squared_errors = [float(np.mean((left - right) ** 2)) for left, right in zip(references, candidates)]
    mse = float(np.mean(squared_errors))
    psnr = 99.0 if mse <= 1e-12 else 10.0 * math.log10((255.0**2) / mse)
    ssim_values: list[float] = []
    for left, right in zip(references, candidates):
        ssim_values.append(
            float(
                structural_similarity(
                    left,
                    right,
                    data_range=255.0,
                    channel_axis=2,
                    gaussian_weights=True,
                    use_sample_covariance=False,
                )
            )
        )
    temporal_errors = []
    for index in range(1, count):
        reference_delta = references[index] - references[index - 1]
        candidate_delta = candidates[index] - candidates[index - 1]
        temporal_errors.append(
            float(np.mean(np.abs(reference_delta - candidate_delta)) / 255.0)
        )
    return {
        "paired_video_quality_status": "ready",
        "paired_video_quality_failure_reason": "none",
        "paired_quality_frame_count": count,
        "paired_watermark_psnr": round(float(psnr), 8),
        "paired_watermark_ssim": round(float(np.mean(ssim_values)), 8),
        "paired_watermark_ssim_protocol": (
            "local_gaussian_window_multichannel_ssim_data_range_255"
        ),
        "paired_temporal_delta_error": round(float(np.mean(temporal_errors)), 8),
    }
