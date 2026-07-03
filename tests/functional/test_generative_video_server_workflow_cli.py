"""验证无 Notebook 的 GPU 服务器 workflow 命令行入口。"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.run_generative_video_server_workflow import PIPELINE_ROLE_ORDER


@pytest.mark.quick
def test_server_workflow_cli_dry_run_exposes_paper_gate_stage_plan(tmp_path: Path) -> None:
    """服务器 CLI dry-run 必须复用统一 stage plan, 不执行 GPU-heavy 任务。"""

    command = [
        sys.executable,
        "scripts/run_generative_video_server_workflow.py",
        "--project-root",
        str(tmp_path / "sstw_server_run"),
        "--pipeline",
        "paper_gate_and_package",
        "--dry-run",
        "--exclude-videos",
    ]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    payload = json.loads(completed.stdout)
    stage_names = [
        row["stage_name"]
        for row in payload["pipeline_results"][0]["stage_plan"]
    ]

    assert payload["server_workflow_decision"] == "DRY_RUN"
    assert payload["pipeline"] == "paper_gate_and_package"
    assert payload["include_videos"] is False
    assert "motion_consistency_exclusion_report" in stage_names
    assert "sstw_measured_formal_result" in stage_names
    assert "fair_detection_calibration" in stage_names
    assert "formal_method_baseline_comparison" in stage_names
    assert "formal_baseline_difference_interval" in stage_names
    assert "validation_scale_formal_internal_ablation" in stage_names
    assert "low_fpr_formal_statistics" in stage_names


@pytest.mark.quick
def test_server_workflow_complete_pipeline_order_matches_notebook_handoff_model() -> None:
    """完整服务器 pipeline 必须覆盖 calibration、runtime、baseline 和 paper gate。"""

    assert PIPELINE_ROLE_ORDER["paper_protocol_complete"] == (
        "motion_threshold_calibration",
        "generative_video_runtime",
        "external_baseline_formal_scoring",
        "paper_gate_and_package",
    )
    assert PIPELINE_ROLE_ORDER["validation_scale_complete"] == PIPELINE_ROLE_ORDER["paper_protocol_complete"]


@pytest.mark.quick
def test_server_workflow_cli_dry_run_exposes_all_modern_baseline_references(tmp_path: Path) -> None:
    """服务器 CLI 必须能脱离 Notebook 调度 5 个 modern baseline 官方参考入口。"""

    command = [
        sys.executable,
        "scripts/run_generative_video_server_workflow.py",
        "--project-root",
        str(tmp_path / "sstw_server_run"),
        "--pipeline",
        "external_baseline_references",
        "--dry-run",
    ]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    payload = json.loads(completed.stdout)
    reference_plan = payload["pipeline_results"][0]

    assert payload["server_workflow_decision"] == "DRY_RUN"
    assert reference_plan["notebook_role"] == "external_baseline_formal_scoring"
    assert reference_plan["stage_execution_kind"] == "external_baseline_formal_reference_helper"
    assert reference_plan["baseline_ids"] == ["videoseal", "vidsig", "videomark", "videoshield", "sigmark"]


@pytest.mark.quick
def test_server_workflow_cli_contains_sigmark_notebook_equivalent_defaults() -> None:
    """SIGMark 的 Colab cell 默认值必须迁移到服务器 CLI, 避免入口语义不一致。"""

    source = Path("scripts/run_generative_video_server_workflow.py").read_text(encoding="utf-8")

    assert "SSTW_SIGMARK_AUTO_DOWNLOAD_HF_MODEL" in source
    assert "SSTW_SIGMARK_WIDTH" in source
    assert "SSTW_SIGMARK_HEIGHT" in source
    assert "SSTW_SIGMARK_NUM_FRAMES" in source
    assert "SSTW_SIGMARK_PRECISION" in source
    assert "SSTW_SIGMARK_NUM_VIDEOS_PER_PROMPT" not in source
    assert "run_default_modern_external_baseline_formal_reference_plan" in source
