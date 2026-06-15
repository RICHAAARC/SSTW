"""B5 泛化状态构建。"""

from __future__ import annotations

from main.analysis.cross_prompt_seed_audit import cross_prompt_seed_status


def build_generalization_status(runnable_status: str) -> dict:
    """返回 prompt、seed、motion 与 length 泛化状态。"""
    return cross_prompt_seed_status(runnable_status)
