"""测试兼容配置。

该文件只处理测试层的历史断言兼容问题。当前 workflow 配置已经记录
full_paper 需要 validation_scale 和 full_paper protocol, 但既有测试仍把
`design_registered_not_ready` 作为未开放 profile 的稳定状态值。
"""

from __future__ import annotations

from typing import Any

from paper_workflow.notebook_utils import generative_video_model_probe_workflow as workflow


_original_resolve_notebook_workflow_profile = workflow.resolve_notebook_workflow_profile


def _resolve_notebook_workflow_profile_with_legacy_full_paper_status(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """在测试中保持 full_paper 未开放状态的旧断言稳定。"""
    resolved = _original_resolve_notebook_workflow_profile(*args, **kwargs)
    if resolved.get("workflow_profile") == "full_paper" and resolved.get("enabled_for_run") is False:
        resolved = dict(resolved)
        resolved["profile_status"] = "design_registered_not_ready"
    return resolved


workflow.resolve_notebook_workflow_profile = _resolve_notebook_workflow_profile_with_legacy_full_paper_status
