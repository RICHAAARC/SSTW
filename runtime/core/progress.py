"""运行时工作量进度显示与第三方进度噪声控制工具。"""

from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
import os
import sys
import time
from typing import Any, Iterator, TextIO


NOISY_LIBRARY_ENV_DEFAULTS = {
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "TQDM_DISABLE": "1",
    "TRANSFORMERS_VERBOSITY": "error",
    "DIFFUSERS_VERBOSITY": "error",
    "TOKENIZERS_PARALLELISM": "false",
}


def progress_enabled_from_env() -> bool:
    """根据环境变量判断是否输出进度。

    该函数属于通用工程写法。默认启用进度显示, 因为 Colab 长时间运行时需要
    明确知道当前真实工作量已经完成多少。若在自动化环境中需要静默运行, 可设置
    `SSTW_PROGRESS=0`、`false`、`no` 或 `off`。
    """
    value = os.environ.get("SSTW_PROGRESS", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def pipeline_progress_bar_enabled_from_env() -> bool:
    """判断是否显示单次 pipeline 内部进度条。

    该函数属于通用工程写法。SSTW 的 Colab 入口默认只显示外层真实工作量进度,
    例如总共需要生成多少个视频、已经完成多少个视频。Diffusers / tqdm 的单次
    pipeline 内部进度条默认关闭, 避免大量下载、加载和采样进度刷屏。若需要调试
    第三方库内部行为, 可设置 `SSTW_ENABLE_PIPELINE_PROGRESS_BAR=1`。
    """
    value = os.environ.get("SSTW_ENABLE_PIPELINE_PROGRESS_BAR", "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


def third_party_progress_suppression_enabled_from_env() -> bool:
    """判断是否压制第三方库的 stdout / stderr 进度噪声。

    该函数只控制人类可读日志, 不影响正式 records、tables、figures、reports 或
    manifests。默认开启是因为 Colab 流式输出会把 Hugging Face 下载、权重加载和
    tqdm 进度条全部转发出来, 容易掩盖 SSTW 自己的真实工作量进度。
    """
    value = os.environ.get("SSTW_SUPPRESS_THIRD_PARTY_PROGRESS", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def configure_noisy_library_progress() -> None:
    """为 Hugging Face / Diffusers / tqdm 设置默认静默配置。

    该函数属于通用工程写法。它使用 `setdefault`, 因此用户在 Colab 或 shell 中显式
    设置的调试环境变量会优先保留。函数会尽量调用已安装库的静默 API, 但不会把
    这些库变成运行时必需依赖。
    """
    for key, value in NOISY_LIBRARY_ENV_DEFAULTS.items():
        os.environ.setdefault(key, value)

    try:  # pragma: no cover - 仅在安装且已导入 huggingface_hub 的环境中生效
        if "huggingface_hub" not in sys.modules and "huggingface_hub.utils" not in sys.modules:
            raise ImportError("huggingface_hub_not_imported")
        from huggingface_hub.utils import disable_progress_bars

        if os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS", "1").strip().lower() not in {"0", "false", "no", "off"}:
            disable_progress_bars()
    except Exception:
        pass

    try:  # pragma: no cover - 仅在安装且已导入 diffusers 的环境中生效
        if "diffusers" not in sys.modules:
            raise ImportError("diffusers_not_imported")
        from diffusers.utils import logging as diffusers_logging

        if os.environ.get("DIFFUSERS_VERBOSITY", "error").strip().lower() == "error":
            diffusers_logging.set_verbosity_error()
    except Exception:
        pass

    try:  # pragma: no cover - 仅在安装且已导入 transformers 的环境中生效
        if "transformers" not in sys.modules:
            raise ImportError("transformers_not_imported")
        from transformers.utils import logging as transformers_logging

        if os.environ.get("TRANSFORMERS_VERBOSITY", "error").strip().lower() == "error":
            transformers_logging.set_verbosity_error()
    except Exception:
        pass


def configure_pipeline_progress_bar(pipeline: Any) -> str:
    """配置 Diffusers pipeline 内部进度条并返回配置状态。

    通用工程写法是优先调用 pipeline 自带的 `set_progress_bar_config`。项目特定写法是
    默认关闭内部进度条, 只保留外层 `SSTW 工作量进度`, 这样使用者看到的是视频数量
    级别的真实进度, 而不是每个视频内部采样 step 的噪声进度。
    """
    if not hasattr(pipeline, "set_progress_bar_config"):
        return "not_supported"
    enabled = pipeline_progress_bar_enabled_from_env()
    pipeline.set_progress_bar_config(disable=not enabled)
    return "enabled" if enabled else "disabled"


def emit_progress_event(stage_id: str, message: str, stream: TextIO | None = None) -> None:
    """输出单次运行事件。

    该函数用于模型加载开始、模型加载结束等不适合建成计数器的事件。它与
    `ProgressReporter` 使用相同前缀, 便于 Colab 中快速区分 SSTW 自己的进度和
    第三方库内部日志。
    """
    if not progress_enabled_from_env():
        return
    print(f"SSTW 工作量进度 | {stage_id} | {message}", file=stream or sys.stdout, flush=True)


class _BoundedProgressCapture:
    """仅保留第三方输出末尾片段的内存缓冲。

    该类属于通用工程辅助结构。模型下载或加载失败时, 完全丢弃第三方输出不利于诊断;
    但完整保留下载进度又会造成大量噪声。因此这里只保留末尾若干字符, 只在异常时
    输出摘要。
    """

    def __init__(self, max_chars: int) -> None:
        self.max_chars = max(0, int(max_chars))
        self._tail = ""
        self.char_count = 0

    def write(self, text: str) -> int:
        text = str(text)
        self.char_count += len(text)
        if self.max_chars > 0:
            self._tail = (self._tail + text)[-self.max_chars :]
        return len(text)

    def flush(self) -> None:
        """兼容 file-like 接口。"""

    def close(self) -> None:
        """兼容 logging / absl 在解释器退出时关闭 stream 的行为。"""

    def tail(self) -> str:
        """返回已捕获输出的末尾片段。"""
        return self._tail.strip()


@contextmanager
def suppress_third_party_progress_output(stage_id: str = "third_party_progress", tail_chars: int = 4000) -> Iterator[None]:
    """在作用域内压制第三方库 stdout / stderr 进度噪声。

    该上下文管理器默认用于 `from_pretrained` 和单次 pipeline 调用。正常成功时不输出
    第三方库内部日志; 若作用域内抛出异常, 会向原始 stderr 打印末尾摘要后继续抛出,
    从而避免“静默失败”。
    """
    if not third_party_progress_suppression_enabled_from_env():
        yield
        return

    stdout_capture = _BoundedProgressCapture(tail_chars)
    stderr_capture = _BoundedProgressCapture(tail_chars)
    original_stderr = sys.stderr
    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            yield
    except Exception:
        print(
            f"SSTW 工作量进度 | {stage_id} | third_party_output_suppressed_before_failure"
            f" | stdout_chars={stdout_capture.char_count}"
            f" | stderr_chars={stderr_capture.char_count}",
            file=original_stderr,
            flush=True,
        )
        stdout_tail = stdout_capture.tail()
        stderr_tail = stderr_capture.tail()
        if stdout_tail:
            print(f"SSTW 工作量进度 | {stage_id} | stdout_tail:\n{stdout_tail}", file=original_stderr, flush=True)
        if stderr_tail:
            print(f"SSTW 工作量进度 | {stage_id} | stderr_tail:\n{stderr_tail}", file=original_stderr, flush=True)
        raise


@dataclass
class ProgressReporter:
    """按实际任务总数输出进度, 不写入正式研究产物。

    该类只向 stdout 或指定 stream 打印人类可读进度。它不会写 records、tables、
    figures、reports、manifests 或 claim artifacts, 因而不会改变实验协议语义。
    调用方必须把 `total` 设置为运行时实际构造出的 plan 或 records 数量, 例如
    `len(plan)`、`len(generation_records)` 或 `len(runtime_attack_records)`。
    """

    stage_id: str
    total: int
    unit: str
    stream: TextIO | None = None
    enabled: bool = field(default_factory=progress_enabled_from_env)
    started_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.total = max(0, int(self.total))
        self.stream = self.stream or sys.stdout
        self._emit(f"start | total={self.total} {self.unit}")

    def _elapsed_min(self) -> float:
        return (time.time() - self.started_at) / 60.0

    def _eta_min(self, completed: int) -> float | None:
        if completed <= 0 or self.total <= 0:
            return None
        elapsed = self._elapsed_min()
        remaining = max(self.total - completed, 0)
        return elapsed * remaining / completed

    def _emit(self, message: str) -> None:
        if not self.enabled:
            return
        print(f"SSTW 工作量进度 | {self.stage_id} | {message}", file=self.stream, flush=True)

    def update(self, completed: int, label: str = "") -> None:
        """输出一次进度更新。

        `completed` 使用 1-based 已完成数量。总数来自调用方运行时构造的真实任务列表,
        因此可以自动适配 `probe_paper`、`pilot_paper` 和 `full_paper`。
        """
        completed = max(0, min(int(completed), self.total if self.total else int(completed)))
        percent = 100.0 if self.total == 0 else 100.0 * completed / self.total
        eta = self._eta_min(completed)
        eta_text = "unknown" if eta is None else f"{eta:.1f} min"
        label_text = f" | {label}" if label else ""
        self._emit(
            f"{completed}/{self.total} ({percent:.1f}%)"
            f" | elapsed={self._elapsed_min():.1f} min"
            f" | eta={eta_text}"
            f"{label_text}"
        )

    def finish(self, label: str = "") -> None:
        """输出任务结束状态。"""
        label_text = f" | {label}" if label else ""
        self._emit(f"finish | total={self.total} {self.unit} | elapsed={self._elapsed_min():.1f} min{label_text}")
