"""Prompt 与 seed 泛化审计。"""

from __future__ import annotations


def cross_prompt_seed_status(runnable_status: str) -> dict:
    """在未生成视频时阻断泛化正向结论。"""
    blocked = runnable_status != "runnable"
    return {
        "cross_prompt_generalization_pass": False if blocked else None,
        "cross_seed_generalization_pass": False if blocked else None,
        "cross_motion_generalization_pass": False if blocked else None,
        "cross_length_generalization_pass": False if blocked else None,
        "cross_prompt_seed_generalization_pass": False if blocked else None,
        "generalization_failure_reason": "generation_model_not_runnable" if blocked else "pending_runtime",
    }
