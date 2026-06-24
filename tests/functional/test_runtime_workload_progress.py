"""验证长耗时 runtime runner 使用真实工作量进度, 而不是 Notebook cell 进度。"""

from __future__ import annotations

import io
from pathlib import Path
import sys

import pytest

from main.core.progress import ProgressReporter
from paper_workflow.notebook_utils.streaming_command import run_streaming_command


@pytest.mark.quick
def test_progress_reporter_prints_runtime_total_from_caller() -> None:
    """进度显示必须使用调用方传入的真实任务总数。"""
    stream = io.StringIO()
    items = ["a", "b", "c", "d"]
    progress = ProgressReporter("test_runtime_stage", len(items), "runtime_video", stream=stream, enabled=True)

    for index, item in enumerate(items, start=1):
        progress.update(index, f"item={item}")
    progress.finish("done")

    output = stream.getvalue()
    assert "SSTW 工作量进度 | test_runtime_stage | start | total=4 runtime_video" in output
    assert "1/4 (25.0%)" in output
    assert "4/4 (100.0%)" in output
    assert "finish | total=4 runtime_video" in output


@pytest.mark.quick
def test_notebook_command_runner_streams_subprocess_output(capsys: pytest.CaptureFixture[str]) -> None:
    """Notebook helper 必须流式转发子进程输出, 否则 Colab 中看不到实时工作量进度。"""
    result = run_streaming_command([
        sys.executable,
        "-c",
        "print('SSTW 工作量进度 | stream_test | 1/1 (100.0%)', flush=True)",
    ])

    captured = capsys.readouterr()
    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stderr == ""
    assert "SSTW 工作量进度 | stream_test | 1/1 (100.0%)" in captured.out


@pytest.mark.quick
def test_colab_notebooks_do_not_use_cell_level_progress_as_runtime_progress() -> None:
    """Notebook 不应再维护 code cell 级进度, 避免误导真实样本进度判断。"""
    for notebook_path in sorted(Path("paper_workflow/colab_utils").glob("*.ipynb")):
        source = notebook_path.read_text(encoding="utf-8")
        assert "SSTW_NOTEBOOK_PROGRESS_TOTAL" not in source, notebook_path
        assert "sstw_show_progress(" not in source, notebook_path


@pytest.mark.quick
def test_runtime_runners_use_dynamic_plan_and_record_counts_for_progress() -> None:
    """长耗时 runner 的进度总数必须来自 plan 或 records 长度, 不能在 Notebook 中硬编码数量。"""
    expected_patterns = {
        "experiments/generative_video_model_probe/colab_runtime.py": [
            'ProgressReporter("wan21_runtime_generation", len(plan), "video")',
        ],
        "experiments/sampling_time_constraint/colab_runtime.py": [
            'ProgressReporter("sampling_time_constraint_generation", len(plan), "constraint_video")',
        ],
        "experiments/generative_video_model_probe/formal_metric_runner.py": [
            'ProgressReporter("formal_metric_runtime_video_scan", len(generation_records), "runtime_video")',
        ],
        "experiments/generative_video_model_probe/attack_runner.py": [
            "total_attack_jobs = len(selection.eligible_generation_records) * len(attack_names)",
            'ProgressReporter("runtime_attack_video_transform", total_attack_jobs, "attack_video")',
        ],
        "experiments/generative_video_model_probe/detection_runner.py": [
            'ProgressReporter("runtime_detection_attacked_video_scan", len(runtime_attack_records), "attacked_video")',
        ],
        "experiments/generative_video_model_probe/external_baseline_runner.py": [
            'ProgressReporter("external_baseline_adapter_matrix", len(baseline_records), "baseline_adapter")',
            "comparable_runtime_video_count={len(comparable_records)}",
        ],
        "external_baseline/modern_command_adapter.py": [
            "len(detection_records)",
            '"runtime_video"',
        ],
        "external_baseline/primary/explicit_dtw_temporal_alignment/adapter/run_sstw_eval.py": [
            "len(detection_records)",
            '"runtime_video"',
        ],
        "external_baseline/primary/explicit_frame_matching_temporal_registration/adapter/run_sstw_eval.py": [
            "len(detection_records)",
            '"runtime_video"',
        ],
    }

    for path_text, patterns in expected_patterns.items():
        source = Path(path_text).read_text(encoding="utf-8")
        for pattern in patterns:
            assert pattern in source, path_text


@pytest.mark.quick
def test_notebook_workflow_helpers_do_not_capture_subprocess_output() -> None:
    """Colab workflow helper 不得使用 capture_output 缓存长耗时 runner 进度。"""
    helper_paths = [
        Path("paper_workflow/notebook_utils/generative_video_model_probe_workflow.py"),
        Path("paper_workflow/notebook_utils/sampling_time_constraint_workflow.py"),
        Path("paper_workflow/notebook_utils/flow_model_adapter_preflight_workflow.py"),
    ]
    for helper_path in helper_paths:
        source = helper_path.read_text(encoding="utf-8")
        assert "run_streaming_command(command)" in source
        assert "capture_output=True" not in source
