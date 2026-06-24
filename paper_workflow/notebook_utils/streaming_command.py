"""Colab Notebook 子进程流式输出工具。"""

from __future__ import annotations

import os
import subprocess
import sys


def run_streaming_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    """执行命令并实时转发 stdout / stderr。

    该函数属于 Notebook 入口层的通用工程工具。Colab 中长时间运行的真实 GPU
    子进程如果使用 `capture_output=True`, 进度会被缓存到进程结束后才显示。
    因此这里改用 `Popen` 逐行转发输出, 让 runner 内部的真实工作量进度可以实时显示。

    该函数只影响屏幕输出方式, 不写正式 records、tables、figures、reports 或
    claim artifacts。
    """
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    if process.stdout is not None:
        for line in process.stdout:
            print(line, end="", file=sys.stdout, flush=True)
    return_code = process.wait()
    return subprocess.CompletedProcess(command, return_code, stdout="", stderr="")
