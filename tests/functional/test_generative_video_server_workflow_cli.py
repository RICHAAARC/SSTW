"""验证无 Notebook 的 GPU 服务器 workflow 命令行入口。"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.run_generative_video_server_workflow import PIPELINE_ROLE_ORDER


@pytest.mark.quick
def test_server_workflow_cli_dry_run_exposes_split_stage_plans(tmp_path: Path) -> None:
    """服务器 CLI dry-run 必须区分 formal comparison、paper evidence 与最终 paper gate。"""

    scoring_command = [
        sys.executable,
        "scripts/run_generative_video_server_workflow.py",
        "--project-root",
        str(tmp_path / "sstw_server_run"),
        "--pipeline",
        "formal_comparison_scoring",
        "--dry-run",
        "--exclude-videos",
    ]
    completed = subprocess.run(scoring_command, check=True, text=True, capture_output=True)
    scoring_payload = json.loads(completed.stdout)
    scoring_stage_names = [
        row["stage_name"]
        for row in scoring_payload["pipeline_results"][0]["stage_plan"]
    ]

    assert scoring_payload["server_workflow_decision"] == "DRY_RUN"
    assert scoring_payload["pipeline"] == "formal_comparison_scoring"
    assert scoring_payload["include_videos"] is False
    assert "sstw_measured_formal_result" in scoring_stage_names
    assert "external_baseline_comparison" in scoring_stage_names
    assert "fair_detection_calibration" in scoring_stage_names
    assert "formal_method_baseline_comparison" in scoring_stage_names
    assert "formal_baseline_difference_interval" in scoring_stage_names
    assert "formal_internal_ablation_summary" not in scoring_stage_names

    generation_command = [
        sys.executable,
        "scripts/run_generative_video_server_workflow.py",
        "--project-root",
        str(tmp_path / "sstw_server_run"),
        "--pipeline",
        "generative_video_generation",
        "--dry-run",
        "--exclude-videos",
    ]
    completed = subprocess.run(generation_command, check=True, text=True, capture_output=True)
    generation_payload = json.loads(completed.stdout)

    assert generation_payload["server_workflow_decision"] == "DRY_RUN"
    assert generation_payload["pipeline"] == "generative_video_generation"
    assert generation_payload["pipeline_results"][0]["notebook_role"] == "generative_video_generation"
    assert generation_payload["pipeline_results"][0]["stage_plan"][0]["stage_name"] == "prepare_prompt_suite"

    evidence_command = [
        sys.executable,
        "scripts/run_generative_video_server_workflow.py",
        "--project-root",
        str(tmp_path / "sstw_server_run"),
        "--pipeline",
        "paper_evidence_postprocess",
        "--dry-run",
        "--exclude-videos",
    ]
    completed = subprocess.run(evidence_command, check=True, text=True, capture_output=True)
    evidence_payload = json.loads(completed.stdout)
    evidence_stage_names = [
        row["stage_name"]
        for row in evidence_payload["pipeline_results"][0]["stage_plan"]
    ]

    assert evidence_payload["server_workflow_decision"] == "DRY_RUN"
    assert evidence_payload["pipeline"] == "paper_evidence_postprocess"
    assert "motion_consistency_exclusion_report" in evidence_stage_names
    assert "validation_internal_ablation" in evidence_stage_names
    assert "formal_internal_ablation_summary" in evidence_stage_names
    assert "low_fpr_formal_statistics" in evidence_stage_names
    assert "paper_profile_gate" not in evidence_stage_names

    gate_command = [
        sys.executable,
        "scripts/run_generative_video_server_workflow.py",
        "--project-root",
        str(tmp_path / "sstw_server_run"),
        "--pipeline",
        "paper_gate_and_package",
        "--dry-run",
        "--exclude-videos",
    ]
    completed = subprocess.run(gate_command, check=True, text=True, capture_output=True)
    gate_payload = json.loads(completed.stdout)
    gate_stage_names = [
        row["stage_name"]
        for row in gate_payload["pipeline_results"][0]["stage_plan"]
    ]

    assert gate_payload["server_workflow_decision"] == "DRY_RUN"
    assert gate_payload["pipeline"] == "paper_gate_and_package"
    assert "motion_consistency_exclusion_report" not in gate_stage_names
    assert "external_baseline_comparison" not in gate_stage_names
    assert "fair_detection_calibration" not in gate_stage_names
    assert "formal_internal_ablation_summary" not in gate_stage_names
    assert "low_fpr_formal_statistics" not in gate_stage_names
    assert "paper_profile_gate" in gate_stage_names
    assert "paper_profile_package_manifest_builder" in gate_stage_names


@pytest.mark.quick
def test_server_workflow_complete_pipeline_order_matches_notebook_handoff_model() -> None:
    """完整服务器 pipeline 必须覆盖 calibration、runtime、baseline 和 paper gate。"""

    assert PIPELINE_ROLE_ORDER["paper_protocol_complete"] == (
        "motion_threshold_calibration",
        "generative_video_generation",
        "generative_video_quality_scoring",
        "sstw_mechanism_postprocess",
        "runtime_attack",
        "runtime_detection",
        "external_baseline_formal_scoring",
        "formal_comparison_scoring",
        "paper_evidence_postprocess",
        "paper_gate_and_package",
    )
    assert "generative_video_runtime" not in PIPELINE_ROLE_ORDER
    assert "probe_paper_complete" not in PIPELINE_ROLE_ORDER
