"""generative_video_model_probe 表格构建。"""

from __future__ import annotations


def build_status_table(decision: dict) -> list[dict]:
    """从 decision 构建状态表, 不手工生成正向结果。"""
    details = decision["details"]
    return [{
        "stage_id": decision["stage_id"],
        "implementation_decision": decision["implementation_decision"],
        "mechanism_decision": decision["mechanism_decision"],
        "gpu_validation_status": details["gpu_validation_status"],
        "generation_model_runnable_status": details["generation_model_runnable_status"],
        "formal_claim_status": details["formal_claim_status"],
        "top_conference_generative_video_model_probe_gate": details["top_conference_generative_video_model_probe_gate"],
    }]
