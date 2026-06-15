"""运动一致性审计状态。"""

from __future__ import annotations


def motion_consistency_status(runnable_status: str) -> dict:
    """在未生成视频时返回 not_run, 避免伪造运动一致性结果。"""
    if runnable_status != "runnable":
        return {"motion_consistency_score": None, "motion_artifact_score": None, "motion_metric_status": "not_run", "metric_failure_reason": "generation_model_not_runnable"}
    return {"motion_consistency_score": None, "motion_artifact_score": None, "motion_metric_status": "pending_runtime", "metric_failure_reason": "none"}
