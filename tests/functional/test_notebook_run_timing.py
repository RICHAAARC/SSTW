"""验证 Notebook 运行时间报告由共享入口和打包层统一生成。"""

from __future__ import annotations

import json
from pathlib import Path
import zipfile

import pytest

from paper_workflow.colab_utils.notebook_run_timing import (
    initialize_notebook_runtime_session,
    start_notebook_run_timer,
)
from paper_workflow.colab_utils.stage_package_sync import publish_colab_stage_package


@pytest.mark.quick
def test_notebook_run_timer_writes_report_manifest_and_stage_records(tmp_path: Path) -> None:
    """Notebook 入口计时器必须写出 runtime report、兼容 manifest 和阶段 JSONL。"""

    layout = {"drive_run_root": str(tmp_path / "run")}
    timer = start_notebook_run_timer(
        layout,
        notebook_role="generative_video_generation",
        workflow_profile="probe_paper",
        enabled_stage_plan=["prepare_prompt_suite"],
        repo_root=".",
    )

    result = timer.run_stage(
        "prepare_prompt_suite",
        "python_helper",
        lambda: {"stage_status": "PASS"},
    )
    manifest = timer.finish("completed")

    assert result["stage_status"] == "PASS"
    assert manifest["manifest_kind"] == "notebook_runtime_report"
    assert manifest["notebook_role"] == "generative_video_generation"
    assert manifest["workflow_profile"] == "probe_paper"
    assert manifest["notebook_elapsed_sec"] >= 0
    assert manifest["notebook_timing_coverage_status"] == "repository_stage_plan_only_excludes_manual_colab_setup"
    assert manifest["claim_support_status"] == "notebook_runtime_estimation_only_not_claim_evidence"

    report_path = tmp_path / "run" / "artifacts" / "notebook_runtime_report.json"
    manifest_path = tmp_path / "run" / "artifacts" / "notebook_run_timing_manifest.json"
    records_path = tmp_path / "run" / "records" / "notebook_stage_timing_records.jsonl"
    assert report_path.exists()
    assert manifest_path.exists()
    assert records_path.exists()
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["stage_name"] == "prepare_prompt_suite"
    assert records[0]["stage_execution_status"] == "completed"
    assert records[0]["stage_elapsed_sec"] >= 0


@pytest.mark.quick
def test_notebook_run_timer_can_use_shared_layout_start_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """共享 layout 入口写入的起点应覆盖 helper 初始化起点。"""

    monkeypatch.setenv("SSTW_NOTEBOOK_STARTED_AT_UTC", "2026-07-04T00:00:00+00:00")
    monkeypatch.setenv("SSTW_NOTEBOOK_STARTED_AT_PERF_COUNTER", "0.0")
    timer = start_notebook_run_timer(
        {"drive_run_root": str(tmp_path / "run")},
        notebook_role="generative_video_generation",
        workflow_profile="probe_paper",
        repo_root=".",
    )
    manifest = timer.finish("completed")

    assert manifest["notebook_started_at_utc"] == "2026-07-04T00:00:00+00:00"
    assert manifest["notebook_timing_start_source"] == "shared_colab_stage_layout_environment"
    assert manifest["notebook_timing_coverage_status"] == "shared_colab_stage_layout_to_repository_stage_plan_finish"


@pytest.mark.quick
def test_shared_layout_initializes_runtime_session_without_notebook_edits(tmp_path: Path) -> None:
    """共享入口层应能向 layout 注入计时起点, 不要求 Notebook 逐个写代码。"""

    layout = initialize_notebook_runtime_session(
        {
            "drive_run_root": str(tmp_path / "run"),
            "workflow_profile": "probe_paper",
        },
        notebook_role="generative_video_generation",
    )

    assert layout["notebook_runtime_started_at_utc"]
    assert float(layout["notebook_runtime_start_perf_counter"]) >= 0
    assert layout["notebook_runtime_start_source"] == "shared_colab_stage_layout"


@pytest.mark.quick
def test_stage_package_manifest_summarizes_notebook_timing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """阶段 zip manifest 必须携带 Notebook 总耗时摘要, 便于 Drive 侧快速估算用时。"""

    monkeypatch.setenv("SSTW_COLAB_STAGE_IO_MODE", "local_zip")
    drive_root = tmp_path / "drive" / "SSTW"
    run_root = tmp_path / "workspace" / "probe_paper" / "run"
    run_root.mkdir(parents=True)
    (run_root / "records" / "dummy.jsonl").parent.mkdir(parents=True)
    (run_root / "records" / "dummy.jsonl").write_text("{}\n", encoding="utf-8")
    layout = {
        "drive_project_root": str(drive_root),
        "drive_run_root": str(run_root),
        "workflow_profile": "probe_paper",
        "stage_package_id": "generative_video_generation_colab",
        "local_stage_package_cache_root": str(tmp_path / "cache"),
    }
    timer = start_notebook_run_timer(
        layout,
        notebook_role="generative_video_generation",
        workflow_profile="probe_paper",
        repo_root=".",
    )
    timer.finish("completed_before_stage_package_publish")

    package = publish_colab_stage_package(
        layout,
        notebook_role="generative_video_generation",
        include_videos=False,
    )

    manifest_path = Path(package["stage_package_manifest_path"])
    package_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert package_manifest["notebook_run_id"] == timer.notebook_run_id
    assert package_manifest["notebook_elapsed_sec"] >= 0
    assert package_manifest["notebook_timing_status"] == "completed_before_stage_package_publish"
    assert package_manifest["notebook_runtime_report_path"].endswith("notebook_runtime_report.json")
    assert package_manifest["notebook_run_timing_manifest_path"].endswith("notebook_run_timing_manifest.json")
    assert package_manifest["stage_package_publish_elapsed_sec"] >= 0
    with zipfile.ZipFile(package["drive_stage_package_zip"]) as archive:
        names = archive.namelist()
    assert any(name.endswith("artifacts/notebook_runtime_report.json") for name in names)
    assert any(name.endswith("artifacts/notebook_run_timing_manifest.json") for name in names)


@pytest.mark.quick
def test_stage_package_creates_runtime_report_without_explicit_timer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """统一打包入口必须自动生成 notebook_runtime_report.json。"""

    monkeypatch.setenv("SSTW_COLAB_STAGE_IO_MODE", "local_zip")
    drive_root = tmp_path / "drive" / "SSTW"
    run_root = tmp_path / "workspace" / "probe_paper" / "run"
    run_root.mkdir(parents=True)
    layout = initialize_notebook_runtime_session(
        {
            "drive_project_root": str(drive_root),
            "drive_run_root": str(run_root),
            "workflow_profile": "probe_paper",
            "stage_package_id": "generative_video_generation_colab",
            "local_stage_package_cache_root": str(tmp_path / "cache"),
        },
        notebook_role="generative_video_generation",
    )

    package = publish_colab_stage_package(
        layout,
        notebook_role="generative_video_generation",
        include_videos=False,
    )

    report_path = run_root / "artifacts" / "notebook_runtime_report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["manifest_kind"] == "notebook_runtime_report"
    assert report["notebook_timing_start_source"] == "shared_colab_stage_layout"
    with zipfile.ZipFile(package["drive_stage_package_zip"]) as archive:
        names = archive.namelist()
    assert any(name.endswith("artifacts/notebook_runtime_report.json") for name in names)


@pytest.mark.quick
def test_colab_notebooks_do_not_hand_write_runtime_timing_cells() -> None:
    """Notebook 不应逐个手写计时逻辑, 计时应由共享入口与打包层承担。"""

    notebook_dir = Path("paper_workflow/colab_notebooks")
    for notebook_path in notebook_dir.glob("*.ipynb"):
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        source = "".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        assert "SSTW_NOTEBOOK_STARTED_AT_UTC" not in source
        assert "SSTW_NOTEBOOK_STARTED_AT_PERF_COUNTER" not in source
