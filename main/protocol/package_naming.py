"""为 Google Drive 落盘包生成可追踪的批次文件名。"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
import subprocess


def current_utc_time_for_filename() -> str:
    """返回适合文件名使用的 UTC 时间。

    该函数属于通用工程写法。统一使用 UTC 可以避免 Colab、Windows 本地和
    Google Drive 同步环境之间的时区差异, 使同一批次产出物按时间排序。
    """
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def current_short_commit(repo_root: str | Path | None = None) -> str:
    """读取当前 Git HEAD 的短 commit。

    该函数属于通用工程写法。Colab 冷启动时仓库由 `git clone` 获得, 因此可以
    用 commit 定位代码版本。若运行环境没有 Git 信息, 返回 `no_git`, 但不阻断打包。
    """
    command = ["git", "rev-parse", "--short=8", "HEAD"]
    completed = subprocess.run(
        command,
        cwd=str(repo_root) if repo_root is not None else None,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return "no_git"
    return sanitize_filename_token(completed.stdout.strip() or "no_git")


def sanitize_filename_token(value: str) -> str:
    """将任意短文本转换为安全的 snake_case 文件名 token。"""
    normalized = re.sub(r"[^0-9A-Za-z_\\-]+", "_", value.strip())
    normalized = normalized.strip("_-").lower()
    return normalized or "unknown"


def build_package_batch_id(utc_time: str | None = None, short_commit: str | None = None) -> str:
    """生成 `<utc_time>_<short_commit>` 形式的包批次 ID。

    该函数属于项目特定写法。Google Drive 中的 zip 和 manifest 会共享同一个
    batch ID, 从而可以快速判断它们是否属于同一批 Colab 运行。
    """
    time_part = sanitize_filename_token(utc_time or current_utc_time_for_filename())
    commit_part = sanitize_filename_token(short_commit or current_short_commit())
    return f"{time_part}_{commit_part}"


def build_package_file_stem(prefix: str, utc_time: str | None = None, short_commit: str | None = None) -> str:
    """生成 `<prefix>_<utc_time>_<short_commit>` 形式的包文件名前缀。"""
    return f"{sanitize_filename_token(prefix)}_{build_package_batch_id(utc_time, short_commit)}"
