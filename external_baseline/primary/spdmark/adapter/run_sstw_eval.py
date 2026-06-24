"""把 SPDMark 官方实现适配到 SSTW external_baseline 正式比较协议。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from external_baseline.modern_command_adapter import (
    ModernBaselineCommandConfig,
    adapter_status_for,
    build_modern_score_records,
)


ADAPTER_NAME = "spdmark"
ADAPTER_PATH = "external_baseline/primary/spdmark/adapter/run_sstw_eval.py"
ADAPTER_CONFIG = ModernBaselineCommandConfig(
    baseline_name=ADAPTER_NAME,
    baseline_family="parameter_or_adapter_video_watermark_baseline",
    adapter_path=ADAPTER_PATH,
    env_var="SSTW_SPDMARK_EVAL_COMMAND",
    default_source_script="external_baseline/primary/spdmark/source/run_sstw_eval.py",
    score_source="spdmark_official_adapter_output",
)


def adapter_status() -> dict[str, Any]:
    """返回 SPDMark adapter 的正式接入状态。

    该 adapter 是现代视频水印 baseline 的正式边界。若官方命令或 source script 未配置,
    状态会保持 not_runnable, 使 pilot_paper gate 明确失败, 防止用 proxy control 替代现代 baseline。
    """
    return adapter_status_for(ADAPTER_CONFIG)


def build_score_records(run_root: str | Path, baseline_record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """调用 SPDMark 官方命令并写出统一 comparison score records。"""
    return build_modern_score_records(run_root, baseline_record, ADAPTER_CONFIG)
