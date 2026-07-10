"""把 VideoMark 官方实现适配到 SSTW external_baseline 正式比较协议。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from external_baseline.modern_command_adapter import (
    ModernBaselineCommandConfig,
    adapter_status_for,
    build_modern_score_records,
)


ADAPTER_NAME = "videomark"
ADAPTER_PATH = "external_baseline/primary/videomark/adapter/run_sstw_eval.py"
ADAPTER_CONFIG = ModernBaselineCommandConfig(
    baseline_name=ADAPTER_NAME,
    baseline_family="training_free_generative_video_watermark_baseline",
    adapter_path=ADAPTER_PATH,
    env_var="SSTW_VIDEOMARK_EVAL_COMMAND",
    default_source_script="external_baseline/primary/videomark/source/run_sstw_eval.py",
    score_source="videomark_official_adapter_output",
)


def adapter_status() -> dict[str, Any]:
    """返回 VideoMark adapter 的正式接入状态。"""

    return adapter_status_for(ADAPTER_CONFIG)


def build_score_records(
    run_root: str | Path,
    baseline_record: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """调用 VideoMark official bundle wrapper 并写出统一 score records。"""

    return build_modern_score_records(run_root, baseline_record, ADAPTER_CONFIG)
