"""验证 Colab 从 Google Drive 私有文件加载轨迹认证密钥。"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from paper_workflow.colab_utils.trajectory_authentication import (
    DEFAULT_TRAJECTORY_AUTHENTICATION_RELATIVE_PATH,
    TRAJECTORY_AUTHENTICATION_FILE_ENV,
    TRAJECTORY_AUTHENTICATION_KEY_ENV,
    TRAJECTORY_AUTHENTICATION_KEY_ID_ENV,
    TRAJECTORY_AUTHENTICATION_SECRET_FORMAT,
    default_trajectory_authentication_secret_path,
    load_trajectory_authentication_from_private_drive,
)


def _write_secret(path: Path, *, key_id: str = "sstw-paper-20260710-v1") -> str:
    """写入确定性测试密钥, 测试文件只存在于 pytest 临时目录。"""

    authentication_key = base64.b64encode(bytes(range(32))).decode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "secret_format": TRAJECTORY_AUTHENTICATION_SECRET_FORMAT,
                "authentication_algorithm": "hmac_sha256",
                TRAJECTORY_AUTHENTICATION_KEY_ENV: authentication_key,
                TRAJECTORY_AUTHENTICATION_KEY_ID_ENV: key_id,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return authentication_key


@pytest.mark.quick
def test_private_drive_secret_loads_without_returning_key(tmp_path: Path) -> None:
    """共享 helper 应写入环境变量, 但返回摘要不得暴露实际密钥。"""

    drive_root = tmp_path / "SSTW"
    secret_path = drive_root / DEFAULT_TRAJECTORY_AUTHENTICATION_RELATIVE_PATH
    expected_key = _write_secret(secret_path)
    environment: dict[str, str] = {}

    summary = load_trajectory_authentication_from_private_drive(
        drive_root,
        environ=environment,
    )

    assert environment[TRAJECTORY_AUTHENTICATION_KEY_ENV] == expected_key
    assert environment[TRAJECTORY_AUTHENTICATION_KEY_ID_ENV] == "sstw-paper-20260710-v1"
    assert summary["trajectory_authentication_status"] == "loaded_from_private_drive_file"
    assert summary["trajectory_authentication_source_entropy_bytes"] == 32
    assert expected_key not in json.dumps(summary, ensure_ascii=False)


@pytest.mark.quick
def test_private_drive_secret_path_supports_explicit_environment_override(
    tmp_path: Path,
) -> None:
    """服务器或不同 Drive 布局可以通过环境变量复用同一加载逻辑。"""

    override_path = tmp_path / "private" / "trajectory_authentication.json"
    environment = {TRAJECTORY_AUTHENTICATION_FILE_ENV: str(override_path)}

    resolved = default_trajectory_authentication_secret_path(
        tmp_path / "unused_drive_root",
        environ=environment,
    )

    assert resolved == override_path


@pytest.mark.quick
def test_private_drive_secret_rejects_weak_key(tmp_path: Path) -> None:
    """解码后少于32字节的密钥不能进入正式 Notebook 环境。"""

    secret_path = tmp_path / "trajectory_authentication.json"
    secret_path.write_text(
        json.dumps(
            {
                "secret_format": TRAJECTORY_AUTHENTICATION_SECRET_FORMAT,
                TRAJECTORY_AUTHENTICATION_KEY_ENV: base64.b64encode(b"weak").decode("ascii"),
                TRAJECTORY_AUTHENTICATION_KEY_ID_ENV: "sstw-paper-weak-v1",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="至少包含32字节"):
        load_trajectory_authentication_from_private_drive(
            tmp_path,
            secret_path=secret_path,
            environ={},
        )


@pytest.mark.quick
def test_private_drive_secret_rejects_environment_identity_conflict(
    tmp_path: Path,
) -> None:
    """同一 runtime 已存在不同认证身份时必须失败, 不能静默覆盖。"""

    secret_path = tmp_path / "trajectory_authentication.json"
    _write_secret(secret_path)
    environment = {
        TRAJECTORY_AUTHENTICATION_KEY_ENV: base64.b64encode(b"x" * 32).decode("ascii"),
        TRAJECTORY_AUTHENTICATION_KEY_ID_ENV: "sstw-paper-other-v1",
    }

    with pytest.raises(RuntimeError, match="与私有文件不一致"):
        load_trajectory_authentication_from_private_drive(
            tmp_path,
            secret_path=secret_path,
            environ=environment,
        )


@pytest.mark.quick
def test_paper_profile_notebooks_load_shared_private_drive_secret() -> None:
    """所有 paper profile 阶段都必须在执行仓库 runner 前加载同一认证文件。"""

    notebook_names = (
        "generative_video_generation_colab.ipynb",
        "generative_video_quality_scoring_colab.ipynb",
        "runtime_attack_colab.ipynb",
        "runtime_detection_colab.ipynb",
        "formal_comparison_scoring_colab.ipynb",
        "paper_evidence_postprocess_colab.ipynb",
        "paper_gate_and_package_colab.ipynb",
    )
    for notebook_name in notebook_names:
        notebook_path = Path("paper_workflow/colab_notebooks") / notebook_name
        source = notebook_path.read_text(encoding="utf-8")
        assert "load_trajectory_authentication_from_private_drive" in source
        assert source.index("load_trajectory_authentication_from_private_drive") < source.index(
            "run_configured_colab_stage_plan"
        )
        assert "trajectory_authentication" in source
