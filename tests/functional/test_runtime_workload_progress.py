"""验证长耗时 runtime runner 使用真实工作量进度, 而不是 Notebook cell 进度。"""

from __future__ import annotations

import io
import os
from pathlib import Path
import sys

import pytest

from main.core.progress import (
    ProgressReporter,
    configure_noisy_library_progress,
    configure_pipeline_progress_bar,
    suppress_third_party_progress_output,
)
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
def test_noisy_library_progress_defaults_are_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """Colab 子进程默认应压制第三方下载、加载和 tqdm 进度噪声。"""
    for key in [
        "HF_HUB_DISABLE_PROGRESS_BARS",
        "HF_HUB_DISABLE_TELEMETRY",
        "TQDM_DISABLE",
        "TRANSFORMERS_VERBOSITY",
        "DIFFUSERS_VERBOSITY",
    ]:
        monkeypatch.delenv(key, raising=False)

    configure_noisy_library_progress()

    assert os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"
    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"
    assert os.environ["TQDM_DISABLE"] == "1"
    assert os.environ["TRANSFORMERS_VERBOSITY"] == "error"
    assert os.environ["DIFFUSERS_VERBOSITY"] == "error"


@pytest.mark.quick
def test_pipeline_progress_bar_is_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """单次 Diffusers pipeline 内部进度条默认关闭, 只保留外层工作量进度。"""
    monkeypatch.delenv("SSTW_ENABLE_PIPELINE_PROGRESS_BAR", raising=False)

    class FakePipeline:
        def __init__(self) -> None:
            self.kwargs: dict[str, bool] | None = None

        def set_progress_bar_config(self, **kwargs: bool) -> None:
            self.kwargs = kwargs

    pipeline = FakePipeline()
    status = configure_pipeline_progress_bar(pipeline)

    assert status == "disabled"
    assert pipeline.kwargs == {"disable": True}


@pytest.mark.quick
def test_third_party_progress_output_is_suppressed_by_default(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """第三方库进度噪声默认不应污染 Colab 输出。"""
    monkeypatch.delenv("SSTW_SUPPRESS_THIRD_PARTY_PROGRESS", raising=False)

    with suppress_third_party_progress_output("unit_noise_test"):
        print("Fetching 19 files: 100%")
        print("Loading checkpoint shards: 100%", file=sys.stderr)

    captured = capsys.readouterr()
    assert "Fetching 19 files" not in captured.out
    assert "Loading checkpoint shards" not in captured.err


@pytest.mark.quick
def test_third_party_progress_output_tail_is_visible_on_failure(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """压制第三方输出时, 异常路径仍应保留末尾摘要用于诊断。"""
    monkeypatch.delenv("SSTW_SUPPRESS_THIRD_PARTY_PROGRESS", raising=False)

    with pytest.raises(RuntimeError):
        with suppress_third_party_progress_output("unit_failure_test", tail_chars=120):
            print("Loading checkpoint shards: 100%", file=sys.stderr)
            raise RuntimeError("simulated failure")

    captured = capsys.readouterr()
    assert "third_party_output_suppressed_before_failure" in captured.err
    assert "Loading checkpoint shards" in captured.err


@pytest.mark.quick
def test_colab_notebooks_do_not_use_cell_level_progress_as_runtime_progress() -> None:
    """Notebook 不应再维护 code cell 级进度, 避免误导真实样本进度判断。"""
    assert not list(Path("paper_workflow/colab_utils").glob("*.ipynb"))
    for notebook_path in sorted(Path("paper_workflow/colab_notebooks").glob("*.ipynb")):
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
            "total_attack_jobs = len(selection.eligible_generation_records) * len(selected_attack_names)",
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
def test_runtime_runners_suppress_third_party_pipeline_noise() -> None:
    """真实 GPU runner 应默认压制第三方内部进度条, 避免覆盖 SSTW 工作量进度。"""
    expected_patterns = {
        "experiments/generative_video_model_probe/colab_runtime.py": [
            "configure_noisy_library_progress()",
            "configure_pipeline_progress_bar(pipe)",
            'suppress_third_party_progress_output("wan21_runtime_single_video_generation")',
        ],
        "experiments/sampling_time_constraint/colab_runtime.py": [
            "configure_noisy_library_progress()",
            "configure_pipeline_progress_bar(pipe)",
            'suppress_third_party_progress_output("sampling_constraint_single_video_generation")',
        ],
        "experiments/flow_model_adapter_preflight/wan21_preflight.py": [
            "configure_noisy_library_progress()",
            "configure_pipeline_progress_bar(pipe)",
            'suppress_third_party_progress_output("wan21_preflight_single_video_generation")',
        ],
        "paper_workflow/notebook_utils/streaming_command.py": [
            "NOISY_LIBRARY_ENV_DEFAULTS",
            'env.setdefault("SSTW_SUPPRESS_THIRD_PARTY_PROGRESS", "1")',
            'env.setdefault("SSTW_ENABLE_PIPELINE_PROGRESS_BAR", "0")',
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
