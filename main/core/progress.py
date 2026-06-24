"""运行时工作量进度显示工具。"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import sys
import time
from typing import TextIO


def progress_enabled_from_env() -> bool:
    """根据环境变量判断是否输出进度。

    该函数属于通用工程写法。默认启用进度显示, 因为 Colab 长时间运行时需要
    明确知道当前真实工作量已经完成多少。若在自动化环境中需要静默运行, 可设置
    `SSTW_PROGRESS=0`、`false`、`no` 或 `off`。
    """
    value = os.environ.get("SSTW_PROGRESS", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


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
        因此可以自动适配 `validation_scale`、`pilot_paper` 和未来 `full_paper`。
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
