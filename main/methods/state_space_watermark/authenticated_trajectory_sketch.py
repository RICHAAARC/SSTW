"""使用 HMAC 认证 SSTW owner-side trajectory sketch。"""

from __future__ import annotations

import hmac
import json
import re
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Iterable, Mapping, MutableSet


SKETCH_STEP_FIELDS = (
    "trajectory_step_index",
    "trajectory_timestep",
    "flow_phase",
    "path_projection_normalized",
    "velocity_projection_normalized",
    "path_velocity_consistency",
)

FORMAL_SKETCH_BINDING_FIELDS = (
    "trajectory_trace_id",
    "method_configuration_id",
    "video_sha256",
    "generation_record_digest",
    "code_commit",
)

_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
_GIT_COMMIT_PATTERN = re.compile(r"^[0-9a-fA-F]{7,64}$")


@dataclass(frozen=True)
class AuthenticatedTrajectorySketchVerification:
    """保存签名、上下文绑定与 nonce 一次性消费的完整验证结果。"""

    verified: bool
    signature_valid: bool
    formal_binding_complete: bool
    binding_matches: bool
    nonce_fresh: bool
    failure_reasons: tuple[str, ...]


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
    trajectory_trace_id: str | None = None,
    method_configuration_id: str | None = None,
    video_sha256: str | None = None,
    generation_record_digest: str | None = None,
    code_commit: str | None = None,
) -> dict[str, Any]:
    """构造不包含完整 latent 的压缩轨迹认证载荷。

    新生成的正式载荷必须同时绑定 trace、方法配置、输出视频、生成记录和代码
    commit。五个字段全部缺失时仅保留旧调用兼容语义，并显式标记正式绑定不完整；
    部分提供会立即失败，防止调用方误以为不完整 HMAC 已形成正式认证证据。
    """

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
    formal_binding = {
        "trajectory_trace_id": trajectory_trace_id,
        "method_configuration_id": method_configuration_id,
        "video_sha256": video_sha256,
        "generation_record_digest": generation_record_digest,
        "code_commit": code_commit,
    }
    provided_binding_count = sum(bool(str(value or "").strip()) for value in formal_binding.values())
    if 0 < provided_binding_count < len(FORMAL_SKETCH_BINDING_FIELDS):
        missing_binding = [
            name
            for name, value in formal_binding.items()
            if not str(value or "").strip()
        ]
        raise ValueError(
            "trajectory sketch 正式绑定字段必须一次性完整提供: "
            + ", ".join(missing_binding)
        )
    formal_binding_complete = provided_binding_count == len(FORMAL_SKETCH_BINDING_FIELDS)
    if formal_binding_complete:
        if not _SHA256_PATTERN.fullmatch(str(video_sha256)):
            raise ValueError("trajectory sketch 的 video_sha256 必须是64位十六进制摘要")
        if not _SHA256_PATTERN.fullmatch(str(generation_record_digest)):
            raise ValueError(
                "trajectory sketch 的 generation_record_digest 必须是64位十六进制摘要"
            )
        if not _GIT_COMMIT_PATTERN.fullmatch(str(code_commit)):
            raise ValueError("trajectory sketch 的 code_commit 必须是合法 Git commit")
        if len(str(generation_nonce_random)) < 32:
            raise ValueError("正式 trajectory sketch 的 generation nonce 至少需要128 bit")
        incomplete_steps = [
            index
            for index, row in enumerate(rows)
            if any(row.get(field_name) is None for field_name in SKETCH_STEP_FIELDS)
        ]
        if incomplete_steps:
            raise ValueError(
                "正式 trajectory sketch 存在不完整 step: "
                + ", ".join(str(index) for index in incomplete_steps)
            )
        step_indices = [int(row["trajectory_step_index"]) for row in rows]
        if len(set(step_indices)) != len(step_indices):
            raise ValueError("正式 trajectory sketch 的 step index 不得重复")
    return {
        "trajectory_sketch_format": (
            "sstw_hmac_path_projection_sketch_formal_binding_v2"
            if formal_binding_complete
            else "sstw_hmac_path_projection_sketch_compatibility_v1"
        ),
        **required_context,
        **formal_binding,
        "trajectory_sketch_formal_binding_complete": formal_binding_complete,
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
    """验证 HMAC 完整性；正式绑定与 nonce 重放应使用一次性验证接口。"""

    if signed_sketch.get("trajectory_sketch_authentication_algorithm") != "hmac_sha256":
        return False
    payload = signed_sketch.get("trajectory_sketch_payload")
    signature = signed_sketch.get("trajectory_sketch_signature")
    if not isinstance(payload, Mapping) or not isinstance(signature, str):
        return False
    expected = hmac.new(authentication_key, _stable_json(payload), sha256).hexdigest()
    return hmac.compare_digest(signature, expected)


def verify_authenticated_trajectory_sketch_once(
    signed_sketch: Mapping[str, Any],
    *,
    authentication_key: bytes,
    expected_binding: Mapping[str, str],
    consumed_nonces: MutableSet[str],
) -> AuthenticatedTrajectorySketchVerification:
    """验证正式绑定，并在成功后按一次性语义消费 generation nonce。

    ``consumed_nonces`` 由部署层提供，可以是进程内集合或持久化注册表适配器。
    本函数只在签名与全部绑定都正确且 nonce 尚未出现时写入该集合；失败验证不会
    污染注册表。调用方可把同一接口复用于数据库唯一索引或分布式 nonce 存储。
    """

    payload = signed_sketch.get("trajectory_sketch_payload")
    signature_valid = verify_authenticated_trajectory_sketch(
        signed_sketch,
        authentication_key=authentication_key,
    )
    if not isinstance(payload, Mapping):
        return AuthenticatedTrajectorySketchVerification(
            verified=False,
            signature_valid=False,
            formal_binding_complete=False,
            binding_matches=False,
            nonce_fresh=False,
            failure_reasons=("invalid_payload",),
        )

    missing_expected = [
        name
        for name in FORMAL_SKETCH_BINDING_FIELDS
        if not str(expected_binding.get(name) or "").strip()
    ]
    if missing_expected:
        raise ValueError(
            "nonce 验证缺少正式 expected binding: " + ", ".join(missing_expected)
        )
    formal_binding_complete = (
        payload.get("trajectory_sketch_formal_binding_complete") is True
        and all(str(payload.get(name) or "").strip() for name in FORMAL_SKETCH_BINDING_FIELDS)
    )
    binding_mismatches = [
        name
        for name, expected_value in expected_binding.items()
        if str(payload.get(name) or "") != str(expected_value)
    ]
    binding_matches = formal_binding_complete and not binding_mismatches
    nonce = str(payload.get("generation_nonce_random") or "").strip()
    nonce_fresh = bool(nonce) and nonce not in consumed_nonces
    failure_reasons: list[str] = []
    if not signature_valid:
        failure_reasons.append("signature_invalid")
    if not formal_binding_complete:
        failure_reasons.append("formal_binding_incomplete")
    if binding_mismatches:
        failure_reasons.extend(
            f"binding_mismatch:{name}" for name in binding_mismatches
        )
    if not nonce:
        failure_reasons.append("nonce_missing")
    elif not nonce_fresh:
        failure_reasons.append("nonce_replayed")
    verified = not failure_reasons
    if verified:
        consumed_nonces.add(nonce)
    return AuthenticatedTrajectorySketchVerification(
        verified=verified,
        signature_valid=signature_valid,
        formal_binding_complete=formal_binding_complete,
        binding_matches=binding_matches,
        nonce_fresh=nonce_fresh,
        failure_reasons=tuple(failure_reasons),
    )
