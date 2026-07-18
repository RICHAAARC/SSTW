"""从环境变量或 Google Drive 私有文件引导 Hugging Face 认证。"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Callable, MutableMapping


HF_TOKEN_ENV = "HF_TOKEN"
HF_TOKEN_FILE_ENV = "SSTW_HF_TOKEN_FILE"
DEFAULT_HF_TOKEN_RELATIVE_PATH = ".sstw_private/hf_token.txt"


def default_hf_token_path(
    drive_project_root: str | Path,
    *,
    environ: MutableMapping[str, str] | None = None,
) -> Path:
    """解析 Drive 私密 token 文件路径, 并允许服务器环境显式覆盖。"""

    environment = os.environ if environ is None else environ
    override_path = str(environment.get(HF_TOKEN_FILE_ENV, "")).strip()
    if override_path:
        return Path(override_path).expanduser()
    return Path(drive_project_root) / DEFAULT_HF_TOKEN_RELATIVE_PATH


def _read_hf_token(token_path: Path) -> str:
    """读取并做最小格式校验, 错误消息不得包含 token。"""

    if not token_path.is_file():
        raise FileNotFoundError(f"缺少 Hugging Face 私密 token 文件: {token_path}")
    token = token_path.read_text(encoding="utf-8-sig").strip()
    if not token:
        raise ValueError(f"Hugging Face 私密 token 文件为空: {token_path}")
    if re.search(r"\s", token):
        raise ValueError("Hugging Face token 不得包含空白字符")
    return token


def bootstrap_huggingface_authentication(
    drive_project_root: str | Path,
    *,
    token_path: str | Path | None = None,
    environ: MutableMapping[str, str] | None = None,
    login_fn: Callable[..., object] | None = None,
    whoami_fn: Callable[..., object] | None = None,
) -> dict[str, str]:
    """认证 Hugging Face 并返回不包含 token 的安全摘要。

    已存在的 ``HF_TOKEN`` 优先于 Drive 私密文件。若从文件加载, token 会写入当前
    Python 进程环境, 使 Notebook 启动的服务器 CLI 子进程继承相同认证身份。
    """

    environment = os.environ if environ is None else environ
    token = str(environment.get(HF_TOKEN_ENV, "")).strip()
    source = "environment"
    resolved_path: Path | None = None
    if not token:
        resolved_path = (
            Path(token_path).expanduser()
            if token_path is not None
            else default_hf_token_path(drive_project_root, environ=environment)
        )
        token = _read_hf_token(resolved_path)
        environment[HF_TOKEN_ENV] = token
        source = "private_drive_file"

    if login_fn is None or whoami_fn is None:
        from huggingface_hub import login, whoami

        login_fn = login if login_fn is None else login_fn
        whoami_fn = whoami if whoami_fn is None else whoami_fn

    login_fn(token=token, add_to_git_credential=False)
    identity = whoami_fn(token=token)
    if not isinstance(identity, dict) or not str(identity.get("name", "")).strip():
        raise RuntimeError("Hugging Face whoami 未返回有效账号身份")

    summary = {
        "huggingface_authentication_status": "authenticated",
        "huggingface_authentication_source": source,
        "huggingface_account_name": str(identity["name"]),
        "huggingface_secret_exposure_status": "not_returned_not_logged",
    }
    if resolved_path is not None:
        summary["huggingface_authentication_source_path"] = resolved_path.as_posix()
    if token in json.dumps(summary, ensure_ascii=False):
        raise RuntimeError("Hugging Face 认证摘要不得包含 token")
    return summary
