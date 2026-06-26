"""VideoSeal 官方源码的 SSTW 评测 wrapper。

该 wrapper 使用官方 `videoseal` Python API 对输入视频执行检测。若用户需要完全复现
VideoSeal 自带的评测脚本, 可以通过 `SSTW_VIDEOSEAL_NATIVE_EVAL_COMMAND` 覆盖默认实现。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any

from external_baseline.official_eval_adapters.common import (
    read_official_result_bundle_if_available,
    run_adapter_main,
    safe_float,
)
from external_baseline.video_tensor_io import read_video_tchw_uint8
from external_baseline.videoseal_official_runtime import (
    ensure_videoseal_official_runtime_layout,
    videoseal_official_source_cwd,
)


BASELINE_ID = "videoseal"
REQUIRED_SOURCE_FILES = ("videoseal/__init__.py", "inference_streaming.py")


def _sigmoid_mean(values: Any) -> float:
    """把官方 detector logits 或概率统一成 [0, 1] 置信度。"""
    import torch

    tensor = values.detach().float().cpu().reshape(-1)
    if tensor.numel() == 0:
        return 0.0
    if torch.all((tensor >= 0.0) & (tensor <= 1.0)):
        return float(tensor.mean().item())
    return float(torch.sigmoid(tensor).mean().item())


def _load_reference_bits(path: str | Path | None) -> list[int] | None:
    """读取 VideoSeal 生成的 message txt, 不存在时返回 None。"""
    if not path:
        return None
    input_path = Path(path)
    if not input_path.exists():
        return None
    bits = [int(ch) for ch in input_path.read_text(encoding="utf-8").strip() if ch in {"0", "1"}]
    return bits or None


def _bit_accuracy(pred_bits: Any, reference_bits: list[int] | None) -> float | None:
    """根据官方 detector 输出和 reference message 计算 bit accuracy。"""
    if not reference_bits:
        return None
    import torch

    pred = pred_bits.detach().float().cpu().reshape(-1)
    if pred.numel() == 0:
        return None
    pred = (pred > 0).int().tolist()
    length = min(len(pred), len(reference_bits))
    if length == 0:
        return None
    return sum(1 for index in range(length) if int(pred[index]) == int(reference_bits[index])) / length


def _run_default(args: argparse.Namespace, source_dir: Path, output_json_path: Path) -> dict[str, Any]:
    """使用官方 VideoSeal API 执行视频检测。

    通用工程写法是优先读取已安装包, 其次把官方源码目录加入 `sys.path`。
    项目特定约束是: 分数来自 `model.detect(...)` 的官方输出, 不读取 SSTW 检测分数。
    """
    bundled = read_official_result_bundle_if_available(
        baseline_id=BASELINE_ID,
        args=args,
        source_dir=source_dir,
        output_json_path=output_json_path,
    )
    if bundled is not None:
        return bundled
    source_layout_audit = ensure_videoseal_official_runtime_layout(source_dir)
    sys.path.insert(0, str(source_dir))

    import torch
    import videoseal

    device = os.environ.get("SSTW_VIDEOSEAL_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    model_name = os.environ.get("SSTW_VIDEOSEAL_MODEL_NAME", "videoseal")
    with videoseal_official_source_cwd(source_dir):
        video_model = videoseal.load(model_name)
    video_model.eval()
    video_model.to(device)

    video, info = read_video_tchw_uint8(args.attacked_video, empty_error="videoseal_attacked_video_empty")
    if video.numel() == 0:
        raise RuntimeError("videoseal_attacked_video_empty")
    video = video.float().to(device) / 255.0
    max_frames = int(os.environ.get("SSTW_VIDEOSEAL_MAX_FRAMES", "0") or "0")
    if max_frames > 0:
        video = video[:max_frames]

    with torch.no_grad():
        outputs = video_model.detect(video, is_video=True)
    preds = outputs.get("preds")
    if preds is None:
        raise RuntimeError("videoseal_detect_output_missing_preds")
    detection_column = preds[:, 0] if preds.ndim >= 2 and preds.shape[-1] > 1 else preds
    message_logits = preds[:, 1:] if preds.ndim >= 2 and preds.shape[-1] > 1 else preds
    confidence = _sigmoid_mean(detection_column)

    reference_path = os.environ.get("SSTW_VIDEOSEAL_REFERENCE_MESSAGE_PATH", "").strip()
    if not reference_path:
        attacked_sidecar = Path(str(args.attacked_video)).with_suffix(".txt")
        source_sidecar = Path(str(args.source_video)).with_suffix(".txt")
        if attacked_sidecar.exists():
            reference_path = str(attacked_sidecar)
        elif source_sidecar.exists():
            reference_path = str(source_sidecar)
    bit_acc = _bit_accuracy(message_logits.mean(dim=0), _load_reference_bits(reference_path))
    score = bit_acc if bit_acc is not None else confidence
    threshold = safe_float(os.environ.get("SSTW_VIDEOSEAL_DETECTION_THRESHOLD"), 0.5)
    return {
        "external_baseline_score": round(float(score), 6),
        "confidence": round(float(confidence), 6),
        "bit_accuracy": round(float(bit_acc), 6) if bit_acc is not None else None,
        "detected": confidence >= threshold,
        "threshold": threshold,
        "official_adapter_status": "measured_by_videoseal_official_api",
        "official_adapter_baseline_id": BASELINE_ID,
        "official_source_dir": str(source_dir),
        "official_source_layout_status": source_layout_audit["layout_status"],
        "official_source_runtime_cwd": source_layout_audit["required_working_directory"],
        "official_video_io_backend": info.get("video_io_backend"),
        "official_model_name": model_name,
        "official_video_frame_count": int(video.shape[0]),
        "official_video_fps": float(info.get("video_fps") or 0.0),
        "official_adapter_protocol_note": (
            "score 来自 VideoSeal 官方 detector。若要支撑主表公平比较, 输入视频应来自 "
            "VideoSeal 官方 embed 或等价正式 baseline 生成链路。"
        ),
        "official_output_json_path": str(output_json_path),
    }


def main() -> None:
    """CLI 入口。"""
    run_adapter_main(
        baseline_id=BASELINE_ID,
        description="VideoSeal 官方检测 wrapper。",
        required_source_files=REQUIRED_SOURCE_FILES,
        default_runner=_run_default,
    )


if __name__ == "__main__":
    main()
