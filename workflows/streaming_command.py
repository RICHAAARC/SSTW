"""Colab Notebook 子进程流式输出工具。"""

from __future__ import annotations

import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

from runtime.core.progress import NOISY_LIBRARY_ENV_DEFAULTS, emit_progress_event


def _resolve_repository_root() -> Path:
    """解析 Colab 子进程应使用的仓库根目录。

    该函数属于 Notebook 入口层的通用工程写法。Colab 中直接执行
    `python scripts/xxx.py` 时, Python 会把 `scripts/` 放到 `sys.path[0]`,
    而不是自动把仓库根目录放到 import 路径中。这里统一从 `SSTW_REPO_DIR`
    或当前 helper 文件位置解析仓库根目录, 让所有 Notebook 子进程都能导入
    `main`、`experiments` 和 `paper_workflow` 等 repository module。
    """

    candidates: list[Path] = []
    env_repo_dir = os.environ.get("SSTW_REPO_DIR")
    if env_repo_dir:
        candidates.append(Path(env_repo_dir).expanduser())
    candidates.append(Path(__file__).resolve().parents[2])
    for candidate in candidates:
        resolved = candidate.resolve()
        if (resolved / "main").is_dir() and (resolved / "paper_workflow").is_dir():
            return resolved
    return Path.cwd().resolve()


def _prepend_env_path(existing: str, path: Path) -> str:
    """把仓库根目录放到环境变量路径列表最前面, 同时避免重复。"""

    path_text = str(path)
    parts = [item for item in existing.split(os.pathsep) if item]
    normalized = {str(Path(item).resolve()) for item in parts if item}
    if str(path.resolve()) not in normalized:
        parts.insert(0, path_text)
    return os.pathsep.join(parts)


def _heartbeat_seconds_from_env() -> float:
    """读取 Notebook 子进程心跳间隔。

    该函数属于通用工程写法。Notebook 子进程的正式样本级进度仍由 runner 自己
    输出, 此处只在子进程长时间没有 stdout 时输出低频心跳, 用于判断进程仍在运行。
    """

    value = os.environ.get("SSTW_NOTEBOOK_COMMAND_HEARTBEAT_SEC", "").strip()
    if not value:
        return 60.0
    return max(5.0, float(value))


def _command_stage_label(command: list[str]) -> str:
    """为 Notebook 子进程生成短标签, 避免在进度行中打印过长命令。"""

    if len(command) >= 2 and command[0].endswith(("python", "python3", "python.exe")):
        return Path(command[1]).name
    if not command:
        return "unknown"
    return Path(command[0]).name or str(command[0])


def run_streaming_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    """执行命令并实时转发 stdout / stderr。

    该函数属于 Notebook 入口层的通用工程工具。Colab 中长时间运行的真实 GPU
    子进程如果使用 `capture_output=True`, 进度会被缓存到进程结束后才显示。
    因此这里改用 `Popen` 逐行转发输出, 让 runner 内部的真实工作量进度可以实时显示。

    该函数只影响屏幕输出方式, 不写正式 records、tables、figures、reports 或
    claim artifacts。
    """
    env = os.environ.copy()
    repo_root = _resolve_repository_root()
    command_label = _command_stage_label(command)
    env["PYTHONPATH"] = _prepend_env_path(env.get("PYTHONPATH", ""), repo_root)
    env.setdefault("PYTHONUNBUFFERED", "1")
    for key, value in NOISY_LIBRARY_ENV_DEFAULTS.items():
        env.setdefault(key, value)
    env.setdefault("SSTW_SUPPRESS_THIRD_PARTY_PROGRESS", "1")
    env.setdefault("SSTW_ENABLE_PIPELINE_PROGRESS_BAR", "0")
    started_at = time.time()
    heartbeat_seconds = _heartbeat_seconds_from_env()
    emit_progress_event(
        "notebook_subprocess_command",
        f"start | command={command_label} | cwd={repo_root.as_posix()}",
    )
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
        cwd=str(repo_root),
    )
    output_queue: queue.Queue[str | None] = queue.Queue()

    def _forward_stdout() -> None:
        """从子进程 stdout 读取行并交给主线程转发。"""

        if process.stdout is None:
            output_queue.put(None)
            return
        try:
            for line in process.stdout:
                output_queue.put(line)
        finally:
            output_queue.put(None)

    reader = threading.Thread(target=_forward_stdout, daemon=True)
    reader.start()
    stdout_closed = False
    last_heartbeat_at = started_at
    no_output_sentinel = object()
    while True:
        try:
            item = output_queue.get(timeout=0.5)
        except queue.Empty:
            item = no_output_sentinel
        if item is None:
            stdout_closed = True
        elif item is not no_output_sentinel:
            print(item, end="", file=sys.stdout, flush=True)
        return_code = process.poll()
        now = time.time()
        if return_code is None and now - last_heartbeat_at >= heartbeat_seconds:
            emit_progress_event(
                "notebook_subprocess_command",
                f"running | command={command_label} | elapsed={(now - started_at) / 60.0:.1f} min",
            )
            last_heartbeat_at = now
        if return_code is not None and stdout_closed:
            break
    reader.join(timeout=1.0)
    return_code = int(process.returncode or 0)
    emit_progress_event(
        "notebook_subprocess_command",
        f"finish | command={command_label} | return_code={return_code} | elapsed={(time.time() - started_at) / 60.0:.1f} min",
    )
    return subprocess.CompletedProcess(command, return_code, stdout="", stderr="")
