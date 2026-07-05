"""external baseline 官方运行器的统一进度显示工具。

该模块只负责人类可读的运行进度输出, 不写正式 records、tables、figures 或
reports。它的主要用途是在 Colab 中让使用者先看到需要处理的样本数量, 再看到
低噪声的阶段级进度和心跳, 避免官方脚本长时间运行时看起来像卡死。
"""

from __future__ import annotations

import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Mapping

from main.core.progress import emit_progress_event


def _heartbeat_seconds_from_env() -> float:
    """读取官方命令心跳间隔。

    该函数属于通用工程写法。默认 60 秒输出一次心跳, 足以证明进程仍在运行,
    同时不会像第三方 tqdm 或逐帧日志那样刷屏。
    """

    value = os.environ.get("SSTW_OFFICIAL_COMMAND_HEARTBEAT_SEC", "").strip()
    if not value:
        return 60.0
    return max(5.0, float(value))


def _elapsed_min(started_at: float) -> float:
    """返回从 started_at 到当前时间的分钟数。"""

    return (time.time() - started_at) / 60.0


def official_record_label(record: Mapping[str, Any]) -> str:
    """把 runtime comparison unit 记录转成人类可读标签。"""

    return (
        f"prompt={record.get('prompt_id')} "
        f"seed={record.get('seed_id')} "
        f"attack={record.get('attack_name')}"
    )


def emit_official_reference_plan(
    baseline_id: str,
    *,
    runtime_detection_record_count: int,
    generated_video_unit_count: int | None = None,
    runtime_attack_count: int | None = None,
    extra: str = "",
) -> None:
    """在官方流程开始前输出待处理样本数量。

    该函数对应用户要求的“检查需要处理样本数量, 再显示进度”。它只输出到
    stdout, 不改变任何实验语义。
    """

    parts = [
        f"baseline={baseline_id}",
        f"runtime_detection_record_count={int(runtime_detection_record_count)}",
    ]
    if generated_video_unit_count is not None:
        parts.append(f"generated_video_unit_count={int(generated_video_unit_count)}")
    if runtime_attack_count is not None:
        parts.append(f"runtime_attack_count={int(runtime_attack_count)}")
    if extra:
        parts.append(str(extra))
    emit_progress_event(f"official_reference_plan:{baseline_id}", " | ".join(parts))


def _format_progress_probe_payload(payload: Mapping[str, Any] | str | None) -> str:
    """把外部进度探针结果格式化为进度行片段。"""

    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload.strip()
    parts: list[str] = []
    for key, value in payload.items():
        parts.append(f"{key}={value}")
    return " | ".join(parts)


def _safe_progress_probe_payload(
    progress_probe: Callable[[], Mapping[str, Any] | str | None] | None,
) -> str:
    """安全读取进度探针。

    进度显示不能影响官方命令本身的执行。若探针因为目录暂未创建等原因失败,
    这里只输出探针失败摘要, 不中断子进程。
    """

    if progress_probe is None:
        return ""
    try:
        return _format_progress_probe_payload(progress_probe())
    except Exception as exc:  # pragma: no cover - 只保护真实 Colab 文件系统竞态
        return f"progress_probe_error={type(exc).__name__}:{exc}"


def _forward_governed_progress_line(line: str) -> None:
    """转发子进程中的 SSTW 自有进度行。

    官方脚本或仓库子命令的普通 stdout / stderr 仍然只被捕获后落盘, 以避免第三方
    tqdm 噪声刷屏。只有 `SSTW 工作量进度` 这种项目自有进度行会实时显示。
    """

    if line.startswith("SSTW 工作量进度 |"):
        print(line, end="", file=sys.stdout, flush=True)


def run_official_subprocess_with_heartbeat(
    command: list[str],
    *,
    cwd: str | Path,
    env: Mapping[str, str] | None = None,
    timeout_seconds: float | None = None,
    stage_id: str,
    progress_probe: Callable[[], Mapping[str, Any] | str | None] | None = None,
) -> subprocess.CompletedProcess[str]:
    """运行官方命令, 捕获日志并定期输出低噪声心跳。

    与 `subprocess.run(capture_output=True)` 相比, 该函数仍然把官方 stdout /
    stderr 完整保存在内存中供调用方落盘, 但运行期间会按固定间隔输出
    `SSTW 工作量进度` 心跳。这样 VideoMark / SIGMark / VidSig 等官方脚本在
    长时间生成或抽取时不会表现为 Notebook 无输出卡住。
    """

    merged_env = dict(os.environ)
    if env:
        merged_env.update({str(key): str(value) for key, value in env.items()})
    started_at = time.time()
    emit_progress_event(stage_id, f"start | cwd={Path(cwd).as_posix()}")
    progress_text = _safe_progress_probe_payload(progress_probe)
    if progress_text:
        emit_progress_event(stage_id, f"progress | {progress_text}")
    try:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=merged_env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
        )
    except OSError as exc:
        message = f"official_command_launch_error:{exc}"
        emit_progress_event(stage_id, f"failed_to_launch | {message}")
        return subprocess.CompletedProcess(command, -1, stdout="", stderr=message)

    output_queue: queue.Queue[tuple[str, str | None]] = queue.Queue()

    def _read_pipe(stream_name: str, pipe: Any) -> None:
        """读取子进程管道, 同时保留完整输出供调用方落盘。"""

        if pipe is None:
            output_queue.put((stream_name, None))
            return
        try:
            for line in pipe:
                output_queue.put((stream_name, line))
        finally:
            output_queue.put((stream_name, None))

    stdout_reader = threading.Thread(target=_read_pipe, args=("stdout", process.stdout), daemon=True)
    stderr_reader = threading.Thread(target=_read_pipe, args=("stderr", process.stderr), daemon=True)
    stdout_reader.start()
    stderr_reader.start()
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    closed_streams: set[str] = set()
    heartbeat_seconds = _heartbeat_seconds_from_env()
    effective_timeout = float(timeout_seconds or 0.0)
    last_heartbeat_at = started_at
    while True:
        elapsed_sec = time.time() - started_at
        if effective_timeout > 0 and elapsed_sec >= effective_timeout:
            process.kill()
            process.wait()
            timeout_message = f"\ncommand_timeout_seconds={effective_timeout}"
            stderr_chunks.append(timeout_message)
            emit_progress_event(stage_id, f"timeout | elapsed={elapsed_sec / 60.0:.1f} min")
            return subprocess.CompletedProcess(
                command,
                -9,
                stdout="".join(stdout_chunks),
                stderr="".join(stderr_chunks),
            )
        try:
            stream_name, line = output_queue.get(timeout=0.2)
            if line is None:
                closed_streams.add(stream_name)
            elif stream_name == "stdout":
                stdout_chunks.append(line)
                _forward_governed_progress_line(line)
            else:
                stderr_chunks.append(line)
                _forward_governed_progress_line(line)
        except queue.Empty:
            pass
        now = time.time()
        if process.poll() is None and now - last_heartbeat_at >= heartbeat_seconds:
            progress_text = _safe_progress_probe_payload(progress_probe)
            suffix = f" | {progress_text}" if progress_text else ""
            emit_progress_event(stage_id, f"running | elapsed={(now - started_at) / 60.0:.1f} min{suffix}")
            last_heartbeat_at = now
        if process.poll() is not None and {"stdout", "stderr"}.issubset(closed_streams):
            stdout_reader.join(timeout=1.0)
            stderr_reader.join(timeout=1.0)
            emit_progress_event(
                stage_id,
                f"finish | return_code={process.returncode} | elapsed={_elapsed_min(started_at):.1f} min",
            )
            return subprocess.CompletedProcess(
                command,
                int(process.returncode or 0),
                stdout="".join(stdout_chunks),
                stderr="".join(stderr_chunks),
            )
