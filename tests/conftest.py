"""pytest 轻量测试隔离配置。"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def isolate_sstw_runtime_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """清理外部 Colab / 服务器运行残留的 SSTW 环境变量。

    该 fixture 属于通用测试工程写法: 每个测试用例应显式构造自己的输入,
    不能依赖调用 pytest 前 shell 或 Notebook 中残留的环境变量。
    项目特定考虑在于, paper gate Notebook 会在同一 Python 进程中注入
    `SSTW_*` runtime、baseline 和 stage zip handoff 环境变量; 如果不隔离,
    默认 `pytest -q` 会把已经配置好的正式运行环境误当作测试夹具, 导致
    fail-closed preflight、legacy packaging 和 non-run record 测试出现假失败。
    """

    for env_name in list(os.environ):
        if env_name.startswith("SSTW_"):
            monkeypatch.delenv(env_name, raising=False)
