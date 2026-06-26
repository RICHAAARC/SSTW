"""VidSig 官方参考运行 Colab helper。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from paper_workflow.colab_utils.modern_external_baseline_formal_reference import (
    run_default_modern_external_baseline_formal_reference_plan,
)


BASELINE_ID = "vidsig"


def run_default_vidsig_formal_reference_plan(repo_root: str | Path = ".") -> dict[str, Any]:
    """运行 VidSig 官方参考 bundle 生成计划。"""

    return run_default_modern_external_baseline_formal_reference_plan(BASELINE_ID, repo_root=repo_root)

