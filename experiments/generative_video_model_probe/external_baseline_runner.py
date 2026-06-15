"""B5 外部 baseline 状态构建。"""

from __future__ import annotations

from main.external_baselines.baseline_registry import build_external_baseline_records


def run_external_baseline_status(config_path: str) -> list[dict]:
    """返回外部 baseline limitation records。"""
    return build_external_baseline_records(config_path)
