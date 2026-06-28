"""SPDMark 官方源码的 SSTW 评测 wrapper。

默认实现复用官方 `assets.evaluate_robustness_full` 中的 extractor 解码和
alignment / detection 逻辑。官方 extractor checkpoint 和 ground-truth message bits
必须由 Colab / Google Drive 提供。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any

from external_baseline.official_eval_adapters.common import (
    raise_missing_official_artifacts,
    read_official_result_bundle_if_available,
    resolve_existing_env_file,
    run_adapter_main,
    safe_float,
)


BASELINE_ID = "spdmark"
REQUIRED_SOURCE_FILES = (
    "assets/evaluate_robustness_full.py",
    "assets/utils.py",
    "ModelScope/generate_watermarked_txt2videos.py",
)


def _read_video_tensor(video_path: str | Path) -> Any:
    """读取视频并转换为 SPDMark extractor 需要的 `[1, T, C, H, W]` tensor。"""
    import imageio.v3 as iio
    import numpy as np
    import torch

    frames = [frame for frame in iio.imiter(video_path)]
    if not frames:
        raise RuntimeError("spdmark_attacked_video_empty")
    array = np.stack(frames, axis=0)
    tensor = torch.from_numpy(array).permute(0, 3, 1, 2).float() / 127.5 - 1.0
    return tensor.unsqueeze(0)


def _run_default(args: argparse.Namespace, source_dir: Path, output_json_path: Path) -> dict[str, Any]:
    """调用 SPDMark 官方 extractor 与 detection 逻辑。"""
    bundled = read_official_result_bundle_if_available(
        baseline_id=BASELINE_ID,
        args=args,
        source_dir=source_dir,
        output_json_path=output_json_path,
    )
    if bundled is not None:
        return bundled
    extractor_path = resolve_existing_env_file("SSTW_SPDMARK_EXTRACTOR_PATH")
    gt_bits_path = resolve_existing_env_file("SSTW_SPDMARK_GT_BITS_PATH")
    if extractor_path is None:
        raise_missing_official_artifacts(BASELINE_ID, "missing file SSTW_SPDMARK_EXTRACTOR_PATH")
    if gt_bits_path is None:
        raise_missing_official_artifacts(BASELINE_ID, "missing file SSTW_SPDMARK_GT_BITS_PATH")

    sys.path.insert(0, str(source_dir))
    sys.path.insert(0, str(source_dir / "assets"))

    import torch
    from assets import evaluate_robustness_full as eval_utils
    from assets import utils as spd_utils

    device = os.environ.get("SSTW_SPDMARK_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    total_slots = int(os.environ.get("SSTW_SPDMARK_TOTAL_SLOTS", "14"))
    bit_len = int(os.environ.get("SSTW_SPDMARK_BIT_LEN", str(total_slots * 2)))
    extractor = spd_utils.FrameWiseExtractor(total_slots)
    state_dict = torch.load(str(extractor_path), map_location="cpu")
    extractor.load_state_dict(state_dict)
    extractor.to(device)
    extractor.eval()

    gt_bits = torch.load(str(gt_bits_path), map_location="cpu").int()
    if gt_bits.ndim == 2:
        gt_bits = gt_bits.unsqueeze(0)
    video_tensor = _read_video_tensor(args.attacked_video).to(device)
    with torch.no_grad():
        pred_bits = eval_utils.decode_bits_per_frame(video_tensor, extractor).cpu().int()
    det = eval_utils.eval_alignment_and_detection(
        gt_bits.cpu().int(),
        pred_bits.cpu().int(),
        M_target=bit_len,
        fpr_frame=safe_float(os.environ.get("SSTW_SPDMARK_FPR_FRAME"), 0.01),
        fpr_video=safe_float(os.environ.get("SSTW_SPDMARK_FPR_VIDEO"), 0.01),
    )
    score = float(det.get("avg_valid_sim") or det.get("avg_match_sim") or 0.0)
    return {
        "external_baseline_score": round(score, 6),
        "bit_accuracy": round(score, 6),
        "detected": bool(det.get("is_watermarked")),
        "threshold": det.get("K_star"),
        "external_baseline_distance": None,
        "official_adapter_status": "measured_by_spdmark_official_extractor",
        "official_adapter_baseline_id": BASELINE_ID,
        "official_source_dir": str(source_dir),
        "official_extractor_path": str(extractor_path),
        "official_gt_bits_path": str(gt_bits_path),
        "official_detection": {
            key: value
            for key, value in det.items()
            if key not in {"pairs_all", "pairs_valid"}
        },
        "official_output_json_path": str(output_json_path),
    }


def main() -> None:
    """CLI 入口。"""
    run_adapter_main(
        baseline_id=BASELINE_ID,
        description="SPDMark 官方检测 wrapper。",
        required_source_files=REQUIRED_SOURCE_FILES,
        default_runner=_run_default,
    )


if __name__ == "__main__":
    main()
