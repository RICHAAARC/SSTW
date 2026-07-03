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
    clean_negative_payload: dict[str, Any] = {}
    clean_result_json = resolve_existing_env_file("SSTW_VIDEOMARK_CLEAN_NEGATIVE_RESULTS_JSON")
    if clean_result_json is not None:
        clean_payload = read_json(clean_result_json)
        clean_values = _collect_decode_acc(clean_payload)
        if not clean_values:
            raise RuntimeError("videomark_clean_negative_decode_acc_missing")
        clean_score = sum(clean_values) / len(clean_values)
        clean_negative_payload = {
            "external_baseline_clean_negative_score": round(clean_score, 6),
            "external_baseline_clean_negative_score_semantics": "payload_bit_accuracy_extraction_score",
            "external_baseline_clean_negative_video_path": str(clean_result_json),
            "official_clean_negative_results_json_path": str(clean_result_json),
        }
    return {
        "external_baseline_score": round(score, 6),
        "raw_detector_score": round(score, 6),
        "payload_bit_accuracy": round(score, 6),
        "bit_accuracy": round(score, 6),
        "detected": score >= threshold,
        "threshold": threshold,
        "score_semantics": "payload_bit_accuracy_extraction_score",
        "score_orientation": "higher_is_more_watermarked",
        "official_score_extraction_policy": "videomark_official_decode_acc_temporal_result_mean",
        "official_reference_protocol_anchor": "same_prompt_seed_attack_runtime_comparison_unit",
        "attack_protocol_status": "videomark_official_temporal_results_project_runtime_attack_anchor",
        "official_adapter_status": "measured_from_videomark_official_temporal_results_json",
        "official_adapter_baseline_id": BASELINE_ID,
        "official_source_dir": str(source_dir),
        "official_temporal_results_json_path": str(result_json),
        "official_decode_acc_count": len(values),
        "official_output_json_path": str(output_json_path),
        **clean_negative_payload,
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
