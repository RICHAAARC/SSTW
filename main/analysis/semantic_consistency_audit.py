"""语义一致性审计状态。"""

from __future__ import annotations


def semantic_consistency_status(runnable_status: str) -> dict:
    """在未生成视频时返回 not_run, 避免伪造语义一致性结果。"""
    if runnable_status != "runnable":
        return {"semantic_metric_name": "disabled", "semantic_metric_status": "not_run", "semantic_consistency_score": None, "metric_failure_reason": "generation_model_not_runnable"}
    return {"semantic_metric_name": "pending_runtime", "semantic_metric_status": "pending_runtime", "semantic_consistency_score": None, "metric_failure_reason": "none"}
