"""VidSig 官方源码的 SSTW 评测 wrapper。

默认实现调用官方 `src/attack.py` 和官方 message decoder checkpoint。checkpoint 不属于
SSTW 仓库, 必须由 Colab / Google Drive 提供。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from external_baseline.official_eval_adapters.common import raise_missing_official_artifacts, run_adapter_main, safe_float


BASELINE_ID = "vidsig"
REQUIRED_SOURCE_FILES = ("src/attack.py", "src/generate_ms.py")
DEFAULT_KEY = "100011100001001101101100100011111101111110000000"


def _write_frame_array(video_path: str | Path, frame_array_dir: Path) -> Path:
    """把 mp4 转成 VidSig 官方 `attack.py` 可读取的 `.npy` 帧数组。"""
    import imageio.v3 as iio
    import numpy as np

    frames = [frame for frame in iio.imiter(video_path)]
    if not frames:
        raise RuntimeError("vidsig_attacked_video_empty")
    frame_array_dir.mkdir(parents=True, exist_ok=True)
    frame_path = frame_array_dir / "sstw_attacked_video.npy"
    np.save(frame_path, np.stack(frames, axis=0))
    return frame_path


def _parse_vidsig_log(log_path: Path) -> tuple[float, bool]:
    """从 VidSig 官方日志中提取 `fpr = 1e-2` 的检测结果。"""
    if not log_path.exists():
        raise FileNotFoundError(f"vidsig_log_missing:{log_path}")
    score = None
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "fpr = 1e-2" in line and ":" in line:
            score = safe_float(line.rsplit(":", 1)[-1].strip(), 0.0)
    if score is None:
        raise RuntimeError("vidsig_detection_score_missing_in_log")
    return float(score), bool(score > 0.0)


def _run_default(args: argparse.Namespace, source_dir: Path, output_json_path: Path) -> dict[str, Any]:
    """调用 VidSig 官方 `src/attack.py` 执行检测。"""
    decoder_path = Path(os.environ.get("SSTW_VIDSIG_MSG_DECODER_PATH", "")).expanduser()
    if not decoder_path.exists():
        raise_missing_official_artifacts(BASELINE_ID, "missing SSTW_VIDSIG_MSG_DECODER_PATH")

    work_dir = output_json_path.parent / "vidsig_official_work"
    frame_array_dir = work_dir / "frame_arrays"
    _write_frame_array(args.attacked_video, frame_array_dir)
    official_output_dir = work_dir / "official_attack_output"
    official_output_dir.mkdir(parents=True, exist_ok=True)

    attack_type = os.environ.get("SSTW_VIDSIG_ATTACK_TYPE", "clean")
    factor = os.environ.get("SSTW_VIDSIG_ATTACK_FACTOR", "2.0")
    key = os.environ.get("SSTW_VIDSIG_KEY", DEFAULT_KEY)
    command = [
        sys.executable,
        str(source_dir / "src" / "attack.py"),
        "--output_dir",
        str(official_output_dir),
        "--attack_type",
        attack_type,
        "--factor",
        str(factor),
        "--frame_array_path",
        str(frame_array_dir),
        "--msg_decoder_path",
        str(decoder_path),
        "--key",
        key,
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(source_dir) + os.pathsep + env.get("PYTHONPATH", "")
    timeout_sec = safe_float(os.environ.get("SSTW_VIDSIG_OFFICIAL_TIMEOUT_SEC"), 3600.0)
    completed = subprocess.run(command, cwd=str(source_dir), env=env, text=True, capture_output=True, timeout=timeout_sec)
    if completed.returncode != 0:
        raise RuntimeError(f"vidsig_official_attack_failed:{completed.returncode}:{completed.stderr[-1000:]}")
    detection_score, detected = _parse_vidsig_log(official_output_dir / "log.txt")
    return {
        "external_baseline_score": round(detection_score, 6),
        "detection_score": round(detection_score, 6),
        "detected": detected,
        "threshold": 0.5,
        "official_adapter_status": "measured_by_vidsig_official_attack_py",
        "official_adapter_baseline_id": BASELINE_ID,
        "official_source_dir": str(source_dir),
        "official_msg_decoder_path": str(decoder_path),
        "official_log_path": str(official_output_dir / "log.txt"),
        "official_stdout_tail": completed.stdout[-1000:],
        "official_stderr_tail": completed.stderr[-1000:],
        "official_output_json_path": str(output_json_path),
    }


def main() -> None:
    """CLI 入口。"""
    run_adapter_main(
        baseline_id=BASELINE_ID,
        description="VidSig 官方检测 wrapper。",
        required_source_files=REQUIRED_SOURCE_FILES,
        default_runner=_run_default,
    )


if __name__ == "__main__":
    main()

