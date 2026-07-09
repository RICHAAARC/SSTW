"""SSTW 正式视频内容检测器。

该模块只读取视频文件本身与项目登记的水印 key, 不读取 generation callback
trajectory trace。它的职责是为 probe_paper、pilot_paper 和 full_paper 提供
统一的 attacked-video / clean-video detector score, 使 SSTW 本方法与 external
baseline 一样在视频内容层完成 fixed-FPR 校准与 TPR 统计。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FORMAL_VIDEO_DETECTOR_EVIDENCE_LEVEL = "attacked_video_content_detector"
FORMAL_CLEAN_NEGATIVE_EVIDENCE_LEVEL = "project_owned_clean_video_content_detector"
FORMAL_VIDEO_DETECTOR_INPUT_CONTRACT = "video_file_plus_project_watermark_key"
FORMAL_VIDEO_DETECTOR_SCORE_SEMANTICS = "sstw_key_conditioned_video_content_detector_score"


@dataclass(frozen=True)
class VideoContentDetectorResult:
    """SSTW 视频内容检测结果。

    该结果对象只保存可进入 governed records 的字段。`score` 的方向固定为
    higher-is-more-watermarked, 下游公平校准可以直接在 clean negative 分布上选阈值。
    """

    score: float
    frame_count: int
    sampled_frame_count: int
    content_feature_count: int
    detector_key_digest: str
    detector_status: str


def _stable_seed(key_text: str) -> int:
    """从项目水印 key 派生可复现种子。"""

    digest = hashlib.sha256(key_text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**63 - 1)


def _read_video_frames(video_path: Path, max_frames: int = 32) -> list[Any]:
    """从视频中等间隔抽取有限帧, 控制 Colab 与服务器 CPU 检测成本。"""

    import imageio.v3 as iio

    all_frames = [frame for frame in iio.imiter(video_path)]
    if not all_frames:
        return []
    if len(all_frames) <= max_frames:
        return all_frames
    import numpy as np

    indices = np.linspace(0, len(all_frames) - 1, num=max_frames).round().astype(int)
    return [all_frames[int(index)] for index in indices]


def _frame_block_features(frame: Any, grid_size: int = 8) -> list[float]:
    """把单帧转换为低频块均值特征。

    这一实现属于通用视频水印检测写法: 检测器不直接依赖像素位置的单点值,
    而是在低频块统计上做 key-conditioned projection, 以提高对压缩、缩放和
    轻量视觉退化的稳定性。
    """

    import numpy as np

    array = np.asarray(frame).astype(np.float32)
    if array.ndim >= 3:
        # 使用 luma 近似, 避免颜色空间变化导致检测分数过度波动。
        array = array[..., 0] * 0.299 + array[..., 1] * 0.587 + array[..., 2] * 0.114
    height, width = int(array.shape[0]), int(array.shape[1])
    block_h = max(1, height // grid_size)
    block_w = max(1, width // grid_size)
    values: list[float] = []
    for y_index in range(grid_size):
        top = min(height - 1, y_index * block_h)
        bottom = height if y_index == grid_size - 1 else min(height, (y_index + 1) * block_h)
        for x_index in range(grid_size):
            left = min(width - 1, x_index * block_w)
            right = width if x_index == grid_size - 1 else min(width, (x_index + 1) * block_w)
            patch = array[top:bottom, left:right]
            values.append(float(patch.mean()) / 255.0 if patch.size else 0.0)
    return values


def _video_features(frames: list[Any]) -> list[float]:
    """提取视频级低频时空特征。"""

    values: list[float] = []
    for frame in frames:
        values.extend(_frame_block_features(frame))
    return values


def _key_projection_score(features: list[float], key_text: str) -> float:
    """计算 key-conditioned projection score。

    当前分数是正式视频内容检测统计量, 不是 trajectory proxy。为了让不同分辨率、
    不同帧数的视频可比较, 该函数先对低频块特征标准化, 再与 key 派生的固定投影
    方向做相关性, 最后映射到 [0, 1]。
    """

    import numpy as np

    if not features:
        return 0.0
    vector = np.asarray(features, dtype=np.float32)
    vector = vector - float(vector.mean())
    std = float(vector.std())
    if std > 1e-8:
        vector = vector / std
    rng = np.random.default_rng(_stable_seed(key_text))
    direction = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=vector.shape)
    correlation = float(np.dot(vector, direction) / max(1.0, float(vector.size)))
    # 使用对称相关强度作为检测统计量, 分数方向固定为越大越像携带该 key 的水印。
    score = 0.5 + min(0.5, abs(correlation) / 2.0)
    return round(float(max(0.0, min(1.0, score))), 6)


def build_sstw_detector_key(record: dict[str, Any]) -> str:
    """从生成与攻击 record 构造 SSTW detector key。

    key 只包含协议锚点和项目水印身份, 不包含检测分数或攻击后结果。这样同一
    prompt / seed / method 的 source、attacked 和 clean negative 可以在同一密钥
    语义下复现检测。
    """

    key_parts = [
        "sstw_key_conditioned_flow_trajectory",
        str(record.get("generation_model_id") or "unknown_model"),
        str(record.get("prompt_id") or "unknown_prompt"),
        str(record.get("seed_id") or "unknown_seed"),
    ]
    return "::".join(key_parts)


def score_video_content(
    video_path: str | Path,
    *,
    detector_key: str,
    max_frames: int = 32,
) -> VideoContentDetectorResult:
    """对单个视频文件执行 SSTW 正式视频内容检测。"""

    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"video_not_found:{path}")
    frames = _read_video_frames(path, max_frames=max_frames)
    if not frames:
        raise RuntimeError("video_has_no_decodable_frames")
    features = _video_features(frames)
    score = _key_projection_score(features, detector_key)
    return VideoContentDetectorResult(
        score=score,
        frame_count=len(frames),
        sampled_frame_count=len(frames),
        content_feature_count=len(features),
        detector_key_digest=hashlib.sha256(detector_key.encode("utf-8")).hexdigest()[:16],
        detector_status="ready",
    )
