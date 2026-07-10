"""使用 HMAC 认证 SSTW owner-side trajectory sketch。"""

from __future__ import annotations

import hmac
import json
from hashlib import sha256
from typing import Any, Iterable, Mapping


SKETCH_STEP_FIELDS = (
    "trajectory_step_index",
    "trajectory_timestep",
    "flow_phase",
    "path_projection_normalized",
    "velocity_projection_normalized",
    "path_velocity_consistency",
)


def _stable_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def build_authenticated_trajectory_sketch_payload(
    step_records: Iterable[Mapping[str, Any]],
    *,
    key_id: str,
    prompt_digest: str,
    seed_id: str,
    model_signature: str,
    sampler_signature: str,
    time_grid_id: str,
    generation_nonce_random: str,
) -> dict[str, Any]:
    """构造不包含完整 latent 的压缩轨迹认证载荷。"""

    rows = [
        {field_name: record.get(field_name) for field_name in SKETCH_STEP_FIELDS}
        for record in step_records
    ]
    if not rows:
        raise ValueError("认证 trajectory sketch 不能缺少 step records")
    required_context = {
        "key_id": key_id,
        "prompt_digest": prompt_digest,
        "seed_id": seed_id,
        "model_signature": model_signature,
        "sampler_signature": sampler_signature,
        "time_grid_id": time_grid_id,
        "generation_nonce_random": generation_nonce_random,
    }
    missing = [name for name, value in required_context.items() if not str(value)]
    if missing:
        raise ValueError(f"trajectory sketch 缺少认证上下文: {', '.join(missing)}")
    return {
        "trajectory_sketch_format": "sstw_hmac_path_projection_sketch",
        **required_context,
        "trajectory_step_count": len(rows),
        "trajectory_steps": rows,
    }


def sign_authenticated_trajectory_sketch(
    payload: Mapping[str, Any],
    *,
    authentication_key: bytes,
) -> dict[str, Any]:
    """使用服务端 HMAC-SHA256 对轨迹摘要签名。"""

    if not authentication_key:
        raise ValueError("authentication_key 不能为空")
    signature = hmac.new(authentication_key, _stable_json(payload), sha256).hexdigest()
    return {
        "trajectory_sketch_authentication_algorithm": "hmac_sha256",
        "trajectory_sketch_payload": dict(payload),
        "trajectory_sketch_signature": signature,
        "trajectory_sketch_verification_status": "signed",
    }


def verify_authenticated_trajectory_sketch(
    signed_sketch: Mapping[str, Any],
    *,
    authentication_key: bytes,
) -> bool:
    """验证签名并拒绝 payload、上下文或签名的任何变更。"""

    if signed_sketch.get("trajectory_sketch_authentication_algorithm") != "hmac_sha256":
        return False
    payload = signed_sketch.get("trajectory_sketch_payload")
    signature = signed_sketch.get("trajectory_sketch_signature")
    if not isinstance(payload, Mapping) or not isinstance(signature, str):
        return False
    expected = hmac.new(authentication_key, _stable_json(payload), sha256).hexdigest()
    return hmac.compare_digest(signature, expected)
