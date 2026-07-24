"""使用所有者秘密密钥派生 SSTW 水印方向上下文。"""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any, Mapping


WATERMARK_KEY_DERIVATION_ID = "hmac_sha256_owner_secret_context_key_v1"
PROMPT_ORTHOGONAL_MASTER_KEY_DERIVATION_ID = (
    "hmac_sha256_prompt_independent_owner_master_key_v1"
)


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


def derive_prompt_orthogonal_master_key_text(
    authentication_key: bytes,
    *,
    key_id: str,
) -> str:
    """Derive the new method identity without model, prompt, seed, or grid."""

    secret = bytes(authentication_key)
    if len(secret) < 32:
        raise ValueError("SSTW 水印认证密钥至少需要32字节")
    identifier = str(key_id).strip()
    if not identifier:
        raise ValueError("SSTW 水印 key ID 不能为空")
    payload = json.dumps(
        {
            "derivation_id": (
                PROMPT_ORTHOGONAL_MASTER_KEY_DERIVATION_ID
            ),
            "key_id": identifier,
            "method_domain": "sstw_prompt_orthogonal_state_trajectory",
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hmac.new(secret, payload, hashlib.sha256).hexdigest()
    return (
        f"{PROMPT_ORTHOGONAL_MASTER_KEY_DERIVATION_ID}:"
        f"{identifier}:{digest}"
    )


def derive_wrong_prompt_orthogonal_master_key_text(
    authentication_key: bytes,
    *,
    key_id: str,
) -> str:
    """Derive a domain-separated wrong-owner identity for candidate controls."""

    derive_prompt_orthogonal_master_key_text(
        authentication_key,
        key_id=key_id,
    )
    wrong_secret = hmac.new(
        bytes(authentication_key),
        b"sstw_prompt_orthogonal_wrong_owner_control",
        hashlib.sha256,
    ).digest()
    return derive_prompt_orthogonal_master_key_text(
        wrong_secret,
        key_id=f"{key_id}:wrong_owner_control",
    )


def derive_prompt_orthogonal_wrong_candidate_master_key_text(
    authentication_key: bytes,
    *,
    key_id: str,
    candidate_index: int,
) -> str:
    """Derive one member of the frozen wrong-owner candidate sequence."""

    derive_prompt_orthogonal_master_key_text(
        authentication_key,
        key_id=key_id,
    )
    index = int(candidate_index)
    if index < 0:
        raise ValueError("wrong-owner candidate index 不能为负数")
    wrong_secret = hmac.new(
        bytes(authentication_key),
        (
            "sstw_prompt_orthogonal_wrong_owner_candidate::"
            f"{index}"
        ).encode("utf-8"),
        hashlib.sha256,
    ).digest()
    return derive_prompt_orthogonal_master_key_text(
        wrong_secret,
        key_id=f"{key_id}:wrong_owner_candidate:{index}",
    )
