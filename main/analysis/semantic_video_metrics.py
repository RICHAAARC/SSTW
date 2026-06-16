"""提供生成视频的文本-视频语义一致性指标。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any


DEFAULT_CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
DEFAULT_SEMANTIC_THRESHOLD = 0.18


def _format_exception(prefix: str, exc: Exception, max_length: int = 240) -> str:
    """格式化异常原因, 使正式 records 能保留可诊断信息且避免写入过长堆栈。

    该函数属于通用工程写法。它不会吞掉失败状态, 只把异常类型和短消息写入 governed records, 便于 Colab
    远程运行后在本地审阅失败原因。
    """
    message = str(exc).replace("\n", " ").replace("\r", " ").strip()
    if len(message) > max_length:
        message = message[:max_length] + "...truncated"
    return f"{prefix}:{type(exc).__name__}:{message}" if message else f"{prefix}:{type(exc).__name__}"


def _move_batch_to_device(batch: Any, device: str) -> dict:
    """将 processor 产出的 batch 显式移动到目标设备。

    这一实现属于通用工程写法。不同 Transformers 版本中 processor 返回对象可能是 BatchEncoding、
    BatchFeature 或普通 dict。显式遍历 tensor 字段比直接调用 `batch.to(device)` 更稳健, 可避免
    Colab 环境中出现 `AttributeError`。
    """
    items = batch.items() if hasattr(batch, "items") else dict(batch).items()
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in items
    }


def _load_sampled_rgb_frames(video_path: Path, frame_limit: int) -> list[Any]:
    """从视频中采样 RGB 帧。

    该函数属于通用工程写法。它只负责读取有限数量的视频帧, 避免在 Colab 或本地审计时把完整视频一次性加载到内存中。
    """
    import imageio.v3 as iio
    import numpy as np

    frames: list[Any] = []
    for frame_index, frame in enumerate(iio.imiter(video_path)):
        if frame_index >= frame_limit:
            break
        array = np.asarray(frame)
        if array.ndim == 2:
            array = np.stack([array, array, array], axis=-1)
        if array.shape[-1] > 3:
            array = array[..., :3]
        frames.append(array.astype("uint8"))
    return frames



@lru_cache(maxsize=4)
def _load_clip_model_and_processor(model_id: str, device: str):
    """缓存 CLIP 模型与 processor, 避免同一批视频重复下载或重复加载权重。

    该函数属于通用工程写法。正式 metric runner 会对多个生成视频逐条计算语义分数, 如果每条视频都重新加载模型, CPU 和 Colab 环境都会产生不必要的时间开销。
    """
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    model.eval()
    return model, processor

def compute_clip_text_video_similarity(
    video_path: str | Path,
    prompt_text: str,
    model_id: str = DEFAULT_CLIP_MODEL_ID,
    frame_limit: int = 8,
    device: str | None = None,
) -> dict:
    """使用 CLIP 计算 prompt 文本与采样视频帧之间的语义一致性。

    这一实现属于项目特定写法: B5 阶段需要把真实生成视频的语义检查落到 governed records 中, 因此函数返回完整状态字段, 而不是只返回一个浮点数。
    在其他项目中可复用的部分包括: 有界帧采样、可选模型依赖、失败原因显式记录、文本与图像 embedding 的余弦相似度计算。
    """
    path = Path(video_path)
    if not path.exists():
        return {
            "semantic_metric_name": "clip_text_video_similarity",
            "semantic_model_id": model_id,
            "semantic_metric_status": "video_missing",
            "semantic_metric_failure_reason": "video_file_not_found",
            "semantic_consistency_score": None,
            "semantic_consistency_mean_score": None,
            "semantic_consistency_max_score": None,
            "semantic_sampled_frame_count": 0,
            "semantic_frame_limit": frame_limit,
            "semantic_metric_device": device or "auto",
        }
    if not prompt_text:
        return {
            "semantic_metric_name": "clip_text_video_similarity",
            "semantic_model_id": model_id,
            "semantic_metric_status": "prompt_text_missing",
            "semantic_metric_failure_reason": "prompt_text_missing",
            "semantic_consistency_score": None,
            "semantic_consistency_mean_score": None,
            "semantic_consistency_max_score": None,
            "semantic_sampled_frame_count": 0,
            "semantic_frame_limit": frame_limit,
            "semantic_metric_device": device or "auto",
        }

    try:
        import torch
        import transformers  # noqa: F401
    except Exception as exc:  # pragma: no cover - 依赖是否存在由 Colab 环境决定
        return {
            "semantic_metric_name": "clip_text_video_similarity",
            "semantic_model_id": model_id,
            "semantic_metric_status": "dependency_missing",
            "semantic_metric_failure_reason": _format_exception("clip_dependency_missing", exc),
            "semantic_consistency_score": None,
            "semantic_consistency_mean_score": None,
            "semantic_consistency_max_score": None,
            "semantic_sampled_frame_count": 0,
            "semantic_frame_limit": frame_limit,
            "semantic_metric_device": device or "auto",
        }

    try:
        frames = _load_sampled_rgb_frames(path, frame_limit=max(1, frame_limit))
    except Exception as exc:  # pragma: no cover - 依赖具体视频解码后端
        return {
            "semantic_metric_name": "clip_text_video_similarity",
            "semantic_model_id": model_id,
            "semantic_metric_status": "video_decode_failed",
            "semantic_metric_failure_reason": _format_exception("video_decode_failed", exc),
            "semantic_consistency_score": None,
            "semantic_consistency_mean_score": None,
            "semantic_consistency_max_score": None,
            "semantic_sampled_frame_count": 0,
            "semantic_frame_limit": frame_limit,
            "semantic_metric_device": device or "auto",
        }
    if not frames:
        return {
            "semantic_metric_name": "clip_text_video_similarity",
            "semantic_model_id": model_id,
            "semantic_metric_status": "video_decode_failed",
            "semantic_metric_failure_reason": "no_decodable_frames",
            "semantic_consistency_score": None,
            "semantic_consistency_mean_score": None,
            "semantic_consistency_max_score": None,
            "semantic_sampled_frame_count": 0,
            "semantic_frame_limit": frame_limit,
            "semantic_metric_device": device or "auto",
        }

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model, processor = _load_clip_model_and_processor(model_id, resolved_device)
    except Exception as exc:  # pragma: no cover - 网络、缓存或 gated 模型均可能导致失败
        return {
            "semantic_metric_name": "clip_text_video_similarity",
            "semantic_model_id": model_id,
            "semantic_metric_status": "model_load_failed",
            "semantic_metric_failure_reason": _format_exception("model_load_failed", exc),
            "semantic_consistency_score": None,
            "semantic_consistency_mean_score": None,
            "semantic_consistency_max_score": None,
            "semantic_sampled_frame_count": len(frames),
            "semantic_frame_limit": frame_limit,
            "semantic_metric_device": resolved_device,
        }

    try:
        text_inputs = _move_batch_to_device(
            processor(text=[prompt_text], return_tensors="pt", padding=True, truncation=True),
            resolved_device,
        )
        image_inputs = _move_batch_to_device(
            processor(images=frames, return_tensors="pt", padding=True),
            resolved_device,
        )
        with torch.no_grad():
            text_features = model.get_text_features(**text_inputs)
            image_features = model.get_image_features(**image_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarities = (image_features @ text_features.T).squeeze(-1).detach().cpu()
        mean_score = float(similarities.mean().item())
        max_score = float(similarities.max().item())
    except Exception as exc:  # pragma: no cover - 设备显存或张量后端错误
        return {
            "semantic_metric_name": "clip_text_video_similarity",
            "semantic_model_id": model_id,
            "semantic_metric_status": "inference_failed",
            "semantic_metric_failure_reason": _format_exception("clip_inference_failed", exc),
            "semantic_consistency_score": None,
            "semantic_consistency_mean_score": None,
            "semantic_consistency_max_score": None,
            "semantic_sampled_frame_count": len(frames),
            "semantic_frame_limit": frame_limit,
            "semantic_metric_device": resolved_device,
        }

    return {
        "semantic_metric_name": "clip_text_video_similarity",
        "semantic_model_id": model_id,
        "semantic_metric_status": "ready",
        "semantic_metric_failure_reason": "none",
        "semantic_consistency_score": round(mean_score, 6),
        "semantic_consistency_mean_score": round(mean_score, 6),
        "semantic_consistency_max_score": round(max_score, 6),
        "semantic_sampled_frame_count": len(frames),
        "semantic_frame_limit": frame_limit,
        "semantic_metric_device": resolved_device,
    }
