"""验证真实 Wan2.1 GPU preflight Colab 入口。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_workflow.notebook_utils.flow_model_adapter_preflight_workflow import (
    DEFAULT_WAN21_PREFLIGHT_MODEL_ID,
    build_drive_layout,
    build_drive_packaging_command,
    build_wan21_flow_adapter_preflight_command,
)


@pytest.mark.quick
def test_wan21_preflight_workflow_uses_dedicated_drive_layout() -> None:
    """真实 Wan2.1 GPU preflight 必须写入独立 run 目录, 不得混入 B6 输出。"""
    layout = build_drive_layout()
    command = build_wan21_flow_adapter_preflight_command(layout)
    package_command = build_drive_packaging_command(layout)

    assert layout["drive_run_root"] == "/content/drive/MyDrive/SSTW/runs/wan21_flow_adapter_preflight"
    assert layout["drive_package_dir"] == "/content/drive/MyDrive/SSTW/packages/wan21_flow_adapter_preflight"
    assert "experiments.flow_model_adapter_preflight.wan21_preflight" in command
    assert DEFAULT_WAN21_PREFLIGHT_MODEL_ID in command
    assert "--num-inference-steps" in command
    assert "4" in command
    assert "scripts/package_results/wan21_flow_adapter_preflight_drive_packager.py" in package_command


@pytest.mark.quick
def test_wan21_preflight_colab_notebook_calls_repository_module() -> None:
    """preflight Notebook 只能作为入口, 必须调用仓库模块生成正式输出。"""
    notebook_path = Path("paper_workflow/colab_notebooks/wan21_flow_adapter_preflight_colab.ipynb")
    assert notebook_path.exists()
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "/content/drive/MyDrive/SSTW" in source
    assert "drive.mount('/content/drive')" in source
    assert "REPO_URL = 'https://github.com/RICHAAARC/SSTW.git'" in source
    assert "git clone" in source
    assert "git pull --ff-only" in source
    assert "git+https://github.com/huggingface/diffusers" in source
    assert "flow_model_adapter_preflight_workflow" in source
    assert "SSTW_COLAB_STAGE_IO_MODE" in source
    assert "prepare_colab_stage_layout" in source
    assert "publish_colab_stage_package" in source
    assert "active_local_layout" in source
    assert "Wan-AI/Wan2.1-T2V-1.3B-Diffusers" in source or "DEFAULT_WAN21_PREFLIGHT_MODEL_ID" in source
    assert "HF_TOKEN" in source
    assert "pytest -q" in source
    assert "tools/harness/run_all_audits.py" in source
    assert "adapter_preflight_decision" in source
    assert "不得进入 B6 sampling-time constraint" in source
    assert "build_drive_packaging_command" in source
    assert "打包到 Google Drive packages/" not in source
    assert "package_dir = Path(layout['drive_package_dir'])" not in source
    assert "stage_packages/wan21_flow_adapter_preflight" in source
    assert "stage_package_dir = Path(layout['stage_package_dir'])" in source
    assert "stage_package_latest.zip" in source
    assert "stage_package_latest_manifest.json" in source

    helper_text = Path("paper_workflow/notebook_utils/flow_model_adapter_preflight_workflow.py").read_text(encoding="utf-8")
    assert "experiments.flow_model_adapter_preflight.wan21_preflight" in helper_text
    assert "wan21_flow_adapter_preflight" in helper_text
    assert "wan21_flow_adapter_preflight_drive_packager.py" in helper_text
