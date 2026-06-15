"""rivagan_adapter 的可审计状态适配器。"""

from __future__ import annotations


def adapter_status() -> dict:
    """返回当前外部 baseline 的状态, 不执行第三方模型。"""
    return {"external_baseline_runnable_status": "not_runnable", "external_baseline_not_run_reason": "external_dependency_not_configured"}
