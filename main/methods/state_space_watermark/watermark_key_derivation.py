"""使用所有者秘密密钥派生 SSTW 水印方向上下文。"""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any, Mapping


WATERMARK_KEY_DERIVATION_ID = "hmac_sha256_owner_secret_context_key_v1"


def derive_watermark_key_text(
    authentication_key: bytes,
    *,
    key_id: str,
    generation_model_id: str,
    prompt_id: str,
    seed_id: str,
    extra_context: Mapping[str, Any] | None = None,
) -> str:
    """从所有者秘密和公开生成上下文派生不可预测的水印 key 文本。

    公开的 model、prompt 和 seed 只作为域分离上下文。真正决定 tubelet 方向的
    HMAC key 不进入 records；输出仅包含 key ID 和不可逆摘要, 可由授权检测方复算。
    """

    secret = bytes(authentication_key)
    if len(secret) < 32:
        raise ValueError("SSTW 水印认证密钥至少需要32字节")
    identifier = str(key_id).strip()
    if not identifier:
        raise ValueError("SSTW 水印 key ID 不能为空")
    payload = {
        "derivation_id": WATERMARK_KEY_DERIVATION_ID,
        "generation_model_id": str(generation_model_id),
        "prompt_id": str(prompt_id),
        "seed_id": str(seed_id),
        "extra_context": dict(extra_context or {}),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hmac.new(secret, encoded, hashlib.sha256).hexdigest()
    return f"{WATERMARK_KEY_DERIVATION_ID}:{identifier}:{digest}"


def derive_wrong_key_control_text(
    authentication_key: bytes,
    *,
    key_id: str,
    generation_model_id: str,
    prompt_id: str,
    seed_id: str,
    extra_context: Mapping[str, Any] | None = None,
) -> str:
    """使用域分离的错误所有者秘密构造 wrong-key 对照。"""

    wrong_secret = hmac.new(
        bytes(authentication_key),
        b"sstw_wrong_owner_key_control",
        hashlib.sha256,
    ).digest()
    return derive_watermark_key_text(
        wrong_secret,
        key_id=f"{key_id}:wrong_owner_control",
        generation_model_id=generation_model_id,
        prompt_id=prompt_id,
        seed_id=seed_id,
        extra_context=extra_context,
    )
