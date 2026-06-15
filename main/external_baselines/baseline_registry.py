"""读取外部 baseline 配置并生成状态记录。"""

from __future__ import annotations

from pathlib import Path
import json


def load_external_baselines(path: str | Path) -> list[dict]:
    """读取外部 baseline 配置。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(data.get("baselines", []))


def build_external_baseline_records(path: str | Path) -> list[dict]:
    """生成外部 baseline 的 limitation records。"""
    records = []
    for item in load_external_baselines(path):
        records.append({
            **item,
            "external_baseline_runnable_status": "not_runnable",
            "external_baseline_not_run_reason": "external_dependency_not_configured",
            "external_baseline_result_used_for_claim": False,
        })
    return records
