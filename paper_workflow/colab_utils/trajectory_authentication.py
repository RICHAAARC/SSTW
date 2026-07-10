"""从 Google Drive 私有文件加载 SSTW 轨迹认证环境变量。"""

from __future__ import annotations

import base64
import binascii
import hmac
import json
import os
from pathlib import Path
import re
from typing import MutableMapping


TRAJECTORY_AUTHENTICATION_KEY_ENV = "SSTW_TRAJECTORY_AUTHENTICATION_KEY"
TRAJECTORY_AUTHENTICATION_KEY_ID_ENV = "SSTW_TRAJECTORY_AUTHENTICATION_KEY_ID"
TRAJECTORY_AUTHENTICATION_FILE_ENV = "SSTW_TRAJECTORY_AUTHENTICATION_FILE"
TRAJECTORY_AUTHENTICATION_SECRET_FORMAT = "sstw_trajectory_authentication_secret"
DEFAULT_TRAJECTORY_AUTHENTICATION_RELATIVE_PATH = (
    ".sstw_private/trajectory_authentication.json"
)
MINIMUM_TRAJECTORY_AUTHENTICATION_ENTROPY_BYTES = 32


def default_trajectory_authentication_secret_path(
    drive_project_root: str | Path,
    *,
    environ: MutableMapping[str, str] | None = None,
) -> Path:
    """解析私有认证文件路径, 并允许服务器环境显式覆盖。

    这一实现属于通用工程写法: 密钥文件与正式实验输出分离, Notebook 只获取路径,
    不把密钥写入配置、records 或阶段包。项目特定约定是默认文件固定放在
    Google Drive 项目根目录的 `.sstw_private` 子目录中。
    """

    environment = os.environ if environ is None else environ
    override_path = str(environment.get(TRAJECTORY_AUTHENTICATION_FILE_ENV, "")).strip()
    if override_path:
        return Path(override_path).expanduser()
    return Path(drive_project_root) / DEFAULT_TRAJECTORY_AUTHENTICATION_RELATIVE_PATH


def _read_secret_payload(secret_path: Path) -> dict[str, object]:
    """读取私有 JSON, 但不在错误消息中包含任何密钥值。"""

    if not secret_path.is_file():
        raise FileNotFoundError(
            "缺少 SSTW trajectory authentication 私有文件: "
            f"{secret_path}"
        )
    try:
        payload = json.loads(secret_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"SSTW trajectory authentication 私有文件不是有效 JSON: {secret_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise TypeError("SSTW trajectory authentication 私有文件顶层必须是 JSON 对象")
    return payload


def _validated_secret_values(payload: dict[str, object]) -> tuple[str, str, int]:
    """校验密钥强度和标识格式, 防止 placeholder 或弱密钥进入正式流程。"""

    secret_format = str(payload.get("secret_format", "")).strip()
    if secret_format != TRAJECTORY_AUTHENTICATION_SECRET_FORMAT:
        raise ValueError(
            "SSTW trajectory authentication 私有文件的 secret_format 不匹配"
        )

    authentication_key = str(payload.get(TRAJECTORY_AUTHENTICATION_KEY_ENV, "")).strip()
    authentication_key_id = str(
        payload.get(TRAJECTORY_AUTHENTICATION_KEY_ID_ENV, "")
    ).strip()
    if not authentication_key:
        raise ValueError(f"私有文件缺少 {TRAJECTORY_AUTHENTICATION_KEY_ENV}")
    if not authentication_key_id:
        raise ValueError(f"私有文件缺少 {TRAJECTORY_AUTHENTICATION_KEY_ID_ENV}")

    try:
        decoded_key = base64.b64decode(
            authentication_key.encode("ascii"),
            validate=True,
        )
    except (UnicodeEncodeError, binascii.Error) as exc:
        raise ValueError("SSTW trajectory authentication key 必须是有效 Base64") from exc
    if len(decoded_key) < MINIMUM_TRAJECTORY_AUTHENTICATION_ENTROPY_BYTES:
        raise ValueError(
            "SSTW trajectory authentication key 解码后必须至少包含"
            f"{MINIMUM_TRAJECTORY_AUTHENTICATION_ENTROPY_BYTES}字节"
        )
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,127}", authentication_key_id):
        raise ValueError(
            "SSTW trajectory authentication key ID 只能包含字母、数字、点、下划线和连字符"
        )
    return authentication_key, authentication_key_id, len(decoded_key)


def load_trajectory_authentication_from_private_drive(
    drive_project_root: str | Path,
    *,
    secret_path: str | Path | None = None,
    environ: MutableMapping[str, str] | None = None,
) -> dict[str, object]:
    """加载认证密钥并返回不包含密钥本体的安全摘要。

    函数会在写入环境变量前完成全部验证。如果当前进程已经存在不同密钥或不同
    key ID, 函数会直接失败, 避免同一 Colab runtime 在不同阶段静默切换认证身份。
    返回值只用于 Notebook 展示加载状态, 不属于论文 claim evidence。
    """

    environment = os.environ if environ is None else environ
    resolved_path = (
        Path(secret_path).expanduser()
        if secret_path is not None
        else default_trajectory_authentication_secret_path(
            drive_project_root,
            environ=environment,
        )
    )
    payload = _read_secret_payload(resolved_path)
    authentication_key, authentication_key_id, entropy_bytes = _validated_secret_values(payload)

    existing_key = str(environment.get(TRAJECTORY_AUTHENTICATION_KEY_ENV, ""))
    existing_key_id = str(environment.get(TRAJECTORY_AUTHENTICATION_KEY_ID_ENV, ""))
    if existing_key and not hmac.compare_digest(existing_key, authentication_key):
        raise RuntimeError(
            f"当前进程中的 {TRAJECTORY_AUTHENTICATION_KEY_ENV} 与私有文件不一致"
        )
    if existing_key_id and existing_key_id != authentication_key_id:
        raise RuntimeError(
            f"当前进程中的 {TRAJECTORY_AUTHENTICATION_KEY_ID_ENV} 与私有文件不一致"
        )

    environment[TRAJECTORY_AUTHENTICATION_KEY_ENV] = authentication_key
    environment[TRAJECTORY_AUTHENTICATION_KEY_ID_ENV] = authentication_key_id
    return {
        "trajectory_authentication_status": "loaded_from_private_drive_file",
        "trajectory_authentication_key_id": authentication_key_id,
        "trajectory_authentication_source_path": resolved_path.as_posix(),
        "trajectory_authentication_source_entropy_bytes": entropy_bytes,
        "trajectory_authentication_secret_exposure_status": "not_returned_not_logged",
    }
