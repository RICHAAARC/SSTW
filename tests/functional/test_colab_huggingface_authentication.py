"""验证 VSCode 连接 Colab 时从 Drive 私密文件引导 Hugging Face 认证。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from paper_workflow.colab_utils.huggingface_authentication import (
    DEFAULT_HF_TOKEN_RELATIVE_PATH,
    HF_TOKEN_ENV,
    HF_TOKEN_FILE_ENV,
    bootstrap_huggingface_authentication,
    default_hf_token_path,
)


FAKE_HF_TOKEN = "hf_" + "a" * 32


def _fake_authentication_calls() -> tuple[list[tuple[str, str]], object, object]:
    calls: list[tuple[str, str]] = []

    def fake_login(*, token: str, add_to_git_credential: bool) -> None:
        assert add_to_git_credential is False
        calls.append(("login", token))

    def fake_whoami(*, token: str) -> dict[str, str]:
        calls.append(("whoami", token))
        return {"name": "sstw-test-account"}

    return calls, fake_login, fake_whoami


@pytest.mark.quick
def test_private_drive_hf_token_authenticates_without_returning_secret(
    tmp_path: Path,
) -> None:
    """Drive token 应进入子进程环境, 但安全摘要不得包含 token。"""

    drive_root = tmp_path / "SSTW"
    token_path = drive_root / DEFAULT_HF_TOKEN_RELATIVE_PATH
    token_path.parent.mkdir(parents=True)
    token_path.write_text(FAKE_HF_TOKEN, encoding="utf-8")
    environment: dict[str, str] = {}
    calls, fake_login, fake_whoami = _fake_authentication_calls()

    summary = bootstrap_huggingface_authentication(
        drive_root,
        environ=environment,
        login_fn=fake_login,
        whoami_fn=fake_whoami,
    )

    assert environment[HF_TOKEN_ENV] == FAKE_HF_TOKEN
    assert calls == [("login", FAKE_HF_TOKEN), ("whoami", FAKE_HF_TOKEN)]
    assert summary["huggingface_authentication_status"] == "authenticated"
    assert summary["huggingface_authentication_source"] == "private_drive_file"
    assert summary["huggingface_account_name"] == "sstw-test-account"
    assert FAKE_HF_TOKEN not in json.dumps(summary, ensure_ascii=False)


@pytest.mark.quick
def test_existing_environment_hf_token_has_precedence(tmp_path: Path) -> None:
    """普通服务器已有环境变量时不应强制依赖 Google Drive。"""

    environment = {HF_TOKEN_ENV: FAKE_HF_TOKEN}
    calls, fake_login, fake_whoami = _fake_authentication_calls()

    summary = bootstrap_huggingface_authentication(
        tmp_path / "missing_drive",
        environ=environment,
        login_fn=fake_login,
        whoami_fn=fake_whoami,
    )

    assert calls == [("login", FAKE_HF_TOKEN), ("whoami", FAKE_HF_TOKEN)]
    assert summary["huggingface_authentication_source"] == "environment"
    assert "huggingface_authentication_source_path" not in summary


@pytest.mark.quick
def test_hf_token_path_supports_explicit_environment_override(tmp_path: Path) -> None:
    """非默认 Drive 布局可以通过环境变量覆盖私密文件路径。"""

    override_path = tmp_path / "private" / "hf_token.txt"
    resolved = default_hf_token_path(
        tmp_path / "unused_drive",
        environ={HF_TOKEN_FILE_ENV: str(override_path)},
    )

    assert resolved == override_path


@pytest.mark.quick
def test_missing_private_drive_hf_token_fails_closed(tmp_path: Path) -> None:
    """既无环境变量也无私密文件时必须在 GPU workflow 前失败。"""

    calls, fake_login, fake_whoami = _fake_authentication_calls()

    with pytest.raises(FileNotFoundError, match="缺少 Hugging Face 私密 token 文件"):
        bootstrap_huggingface_authentication(
            tmp_path / "SSTW",
            environ={},
            login_fn=fake_login,
            whoami_fn=fake_whoami,
        )
    assert calls == []


@pytest.mark.quick
def test_method_notebook_bootstraps_hf_authentication_before_server_cli() -> None:
    """机制验证 Notebook 必须在启动服务器 CLI 前完成共享认证引导。"""

    notebook_path = Path(
        "paper_workflow/colab_notebooks/method_mechanism_validation_colab.ipynb"
    )
    source = notebook_path.read_text(encoding="utf-8")

    assert "bootstrap_huggingface_authentication" in source
    assert source.index("bootstrap_huggingface_authentication") < source.index(
        "result = run_streaming_command(server_command)"
    )
    assert "hf_token.txt" not in source
