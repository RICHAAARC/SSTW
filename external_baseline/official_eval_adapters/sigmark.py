"""SIGMark 官方源码的 SSTW 评测 wrapper。

SIGMark 官方流程需要先运行官方 `main.py --mode gen`, 再在同一输出目录中用
`main.py --mode extract` 写出 bit accuracy。默认 wrapper 读取该官方 bit accuracy
产物; 若用户希望由 wrapper 自动运行完整官方流程, 可配置
`SSTW_SIGMARK_NATIVE_EVAL_COMMAND`。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from external_baseline.official_eval_adapters.common import (
    raise_missing_official_artifacts,
    read_official_result_bundle_if_available,
    resolve_existing_env_file,
    run_adapter_main,
)


BASELINE_ID = "sigmark"
REQUIRED_SOURCE_FILES = ("main.py", "watermarks/sigmark.py", "apply_disturbances.py")


def _resolve_sigmark_bit_accuracy_npz() -> Path | None:
    """定位 SIGMark 官方提取阶段产出的 bit accuracy npz 文件。

    官方 `main.py --mode extract` 产物名称包含实验设置前缀, 典型形式是
    `HunyuanVideo-...-sigmark-...-bit_accuracy.npz`。因此除了显式
    `SSTW_SIGMARK_BIT_ACCURACY_NPZ`, 还允许通过 `SSTW_SIGMARK_OUTPUT_DIR`
    指向官方 output 目录并按 glob 自动发现。
    """

    explicit_path = resolve_existing_env_file("SSTW_SIGMARK_BIT_ACCURACY_NPZ")
    if explicit_path is not None:
        return explicit_path
    output_dir_value = os.environ.get("SSTW_SIGMARK_OUTPUT_DIR", "").strip()
    if not output_dir_value:
        return None
    output_dir = Path(output_dir_value).expanduser()
    if not output_dir.is_dir():
        return None
    pattern = os.environ.get("SSTW_SIGMARK_BIT_ACCURACY_GLOB", "**/*bit_accuracy*.npz").strip()
    if not pattern:
        pattern = "**/*bit_accuracy*.npz"
    matches = sorted(path for path in output_dir.glob(pattern) if path.is_file())
    return matches[0] if matches else None


def _mean_numeric(values: list[Any]) -> float:
    """计算官方 npz 中数值数组的均值。"""
    import numpy as np

    flattened = []
    for value in values:
        array = np.asarray(value, dtype=float).reshape(-1)
        flattened.extend(float(item) for item in array)
    if not flattened:
        raise RuntimeError("sigmark_bit_accuracy_npz_empty")
    return float(np.mean(flattened))


def _run_default(args: argparse.Namespace, source_dir: Path, output_json_path: Path) -> dict[str, Any]:
    """读取 SIGMark 官方 bit accuracy npz。"""
    bundled = read_official_result_bundle_if_available(
        baseline_id=BASELINE_ID,
        args=args,
        source_dir=source_dir,
        output_json_path=output_json_path,
    )
    if bundled is not None:
        return bundled
    bit_accuracy_npz = _resolve_sigmark_bit_accuracy_npz()
    if bit_accuracy_npz is None:
        raise_missing_official_artifacts(
            BASELINE_ID,
            "missing file SSTW_SIGMARK_BIT_ACCURACY_NPZ, "
            "SSTW_SIGMARK_OUTPUT_DIR/**/*bit_accuracy*.npz, or SSTW_SIGMARK_NATIVE_EVAL_COMMAND",
        )

    import numpy as np

    payload = dict(np.load(str(bit_accuracy_npz), allow_pickle=True))
    preferred_key = os.environ.get("SSTW_SIGMARK_RESULT_KEY", "").strip()
    if preferred_key:
        if preferred_key not in payload:
            raise KeyError(f"sigmark_result_key_missing:{preferred_key}")
        score = _mean_numeric([payload[preferred_key]])
    else:
        score = _mean_numeric(list(payload.values()))
    threshold = float(os.environ.get("SSTW_SIGMARK_BIT_ACCURACY_THRESHOLD", "0.5"))
    return {
        "external_baseline_score": round(score, 6),
        "bit_accuracy": round(score, 6),
        "detected": score >= threshold,
        "threshold": threshold,
        "official_adapter_status": "measured_from_sigmark_official_bit_accuracy_npz",
        "official_adapter_baseline_id": BASELINE_ID,
        "official_source_dir": str(source_dir),
        "official_bit_accuracy_npz_path": str(bit_accuracy_npz),
        "official_result_key": preferred_key or "mean_over_npz_entries",
        "official_output_json_path": str(output_json_path),
    }


def main() -> None:
    """CLI 入口。"""
    run_adapter_main(
        baseline_id=BASELINE_ID,
        description="SIGMark 官方 bit accuracy wrapper。",
        required_source_files=REQUIRED_SOURCE_FILES,
        default_runner=_run_default,
    )


if __name__ == "__main__":
    main()
