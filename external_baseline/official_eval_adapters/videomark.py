"""VideoMark 官方源码的 SSTW 评测 wrapper。

VideoMark 的官方脚本会在 `temporal_results.json` 中记录 `decode_acc` 和
`frames_acc`。默认 wrapper 读取该官方产物; 自动生成和反演流程应通过
`SSTW_VIDEOMARK_NATIVE_EVAL_COMMAND` 显式配置。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from external_baseline.official_eval_adapters.common import (
    raise_missing_official_artifacts,
    read_json,
    read_official_result_bundle_if_available,
    resolve_existing_env_file,
    run_adapter_main,
    safe_float,
)


BASELINE_ID = "videomark"
REQUIRED_SOURCE_FILES = ("embedding_and_extraction.py", "temporal_tamper.py", "src/prc.py")


def _collect_decode_acc(payload: Any) -> list[float]:
    """递归收集 VideoMark 官方结果中的 decode_acc。"""
    values: list[float] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == "decode_acc":
                values.append(safe_float(value, 0.0))
            else:
                values.extend(_collect_decode_acc(value))
    elif isinstance(payload, list):
        for item in payload:
            values.extend(_collect_decode_acc(item))
    return values


def _run_default(args: argparse.Namespace, source_dir: Path, output_json_path: Path) -> dict[str, Any]:
    """读取 VideoMark 官方 temporal_results JSON。"""
    bundled = read_official_result_bundle_if_available(
        baseline_id=BASELINE_ID,
        args=args,
        source_dir=source_dir,
        output_json_path=output_json_path,
    )
    if bundled is not None:
        return bundled
    result_json = resolve_existing_env_file("SSTW_VIDEOMARK_TEMPORAL_RESULTS_JSON")
    if result_json is None:
        raise_missing_official_artifacts(
            BASELINE_ID,
            "missing file SSTW_VIDEOMARK_TEMPORAL_RESULTS_JSON or SSTW_VIDEOMARK_NATIVE_EVAL_COMMAND",
        )
    payload = read_json(result_json)
    values = _collect_decode_acc(payload)
    if not values:
        raise RuntimeError("videomark_decode_acc_missing")
    score = sum(values) / len(values)
    threshold = safe_float(os.environ.get("SSTW_VIDEOMARK_DECODE_ACC_THRESHOLD"), 0.5)
    return {
        "external_baseline_score": round(score, 6),
        "bit_accuracy": round(score, 6),
        "detected": score >= threshold,
        "threshold": threshold,
        "official_adapter_status": "measured_from_videomark_official_temporal_results_json",
        "official_adapter_baseline_id": BASELINE_ID,
        "official_source_dir": str(source_dir),
        "official_temporal_results_json_path": str(result_json),
        "official_decode_acc_count": len(values),
        "official_output_json_path": str(output_json_path),
    }


def main() -> None:
    """CLI 入口。"""
    run_adapter_main(
        baseline_id=BASELINE_ID,
        description="VideoMark 官方 temporal results wrapper。",
        required_source_files=REQUIRED_SOURCE_FILES,
        default_runner=_run_default,
    )


if __name__ == "__main__":
    main()
