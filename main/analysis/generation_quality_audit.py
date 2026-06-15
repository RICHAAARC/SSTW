"""生成视频质量审计状态。"""

from __future__ import annotations


def generation_quality_status(runnable_status: str) -> dict:
    """在未生成视频时返回 not_run, 避免伪造质量指标。"""
    if runnable_status != "runnable":
        return {"visual_quality_score": None, "quality_metric_name": "disabled", "quality_metric_status": "not_run", "metric_failure_reason": "generation_model_not_runnable"}
    return {"visual_quality_score": None, "quality_metric_name": "pending_runtime", "quality_metric_status": "pending_runtime", "metric_failure_reason": "none"}
