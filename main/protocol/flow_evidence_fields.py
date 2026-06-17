"""补齐 Flow trajectory 证据 records 的通用协议字段。"""

from __future__ import annotations

from typing import Any


def flow_evidence_protocol_defaults(
    *,
    negative_family: str = "not_applicable",
    trajectory_source_level: str = "not_captured",
    sampler_signature_placeholder: str | None = None,
    flow_state_admissibility_status: str = "not_evaluated",
    claim_support_status: str = "not_supported_until_governed_artifacts_ready",
) -> dict[str, Any]:
    """返回 Flow trajectory 相关 records 的默认字段集合。

    该函数属于项目特定写法。它的作用不是生成实验结论, 而是在真实 GPU records
    到来之前先固定字段边界, 使 Wan2.1 callback、采样器签名、velocity proxy、
    replay uncertainty 和 claim audit 后续可以写入同一套受治理字段。

    在其他阶段复用时, 调用方应只覆盖已经由当前阶段真实记录或可审计后处理得到的字段。
    无证据字段必须保持为 None、placeholder 或 not_supported 状态, 不能伪装成 supported claim。
    """
    return {
        "negative_family": negative_family,
        "sampler_signature_placeholder": sampler_signature_placeholder,
        "trajectory_source_level": trajectory_source_level,
        "S_path_inv": None,
        "S_velocity": None,
        "S_final_conservative": None,
        "path_marginal_gain_at_fixed_fpr": None,
        "replay_uncertainty_mean": None,
        "flow_state_admissibility_status": flow_state_admissibility_status,
        "claim_support_status": claim_support_status,
    }


def conservative_flow_score(record: dict[str, Any]) -> float | None:
    """从 endpoint、path 和 velocity 字段计算保守最终分数。

    该函数属于通用工程写法。保守聚合只使用已经存在的数值字段, 并取最小值,
    目的是防止单一证据层绕过 endpoint、path 或 velocity 的一致性要求。
    """
    candidate_fields = ("S_final", "S_path_inv", "S_velocity")
    values: list[float] = []
    for field_name in candidate_fields:
        value = record.get(field_name)
        if value is not None:
            values.append(float(value))
    if not values:
        return None
    return round(min(values), 6)
