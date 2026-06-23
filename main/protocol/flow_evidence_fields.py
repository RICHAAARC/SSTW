"""补齐 Flow trajectory 证据 records 的通用协议字段。"""

from __future__ import annotations

from typing import Any


FLOW_EVIDENCE_PROTOCOL_FIELDS = (
    "negative_family",
    "sampler_signature_placeholder",
    "trajectory_source_level",
    "S_path_inv",
    "S_velocity",
    "S_final_conservative",
    "path_marginal_gain_at_fixed_fpr",
    "replay_uncertainty_mean",
    "flow_state_admissibility_status",
    "claim_support_status",
)


def flow_evidence_protocol_defaults(
    *,
    negative_family: str | None = "not_applicable",
    trajectory_source_level: str = "not_captured",
    sampler_signature_placeholder: str | None = None,
    flow_state_admissibility_status: str = "not_evaluated",
    claim_support_status: str = "not_supported_until_governed_artifacts_ready",
) -> dict[str, Any]:
    """返回 Flow trajectory 相关 records 的默认协议字段。

    该函数属于项目特定写法。它的作用不是生成实验结论, 而是在真实 GPU records
    到来之前固定字段边界。这样 Wan2.1 callback、采样器签名、velocity proxy、
    replay uncertainty 和 claim audit 后续可以写入同一套受治理字段。

    在其他阶段复用时, 调用方只能覆盖已经由当前阶段真实记录或可审计后处理得到的
    字段。没有证据的字段必须保持为 None、placeholder 或 not_supported 状态,
    不能伪装成 supported claim。
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


def with_flow_evidence_protocol_defaults(
    record: dict[str, Any],
    *,
    negative_family: str | None = "not_applicable",
    trajectory_source_level: str = "not_captured",
    sampler_signature_placeholder: str | None = None,
    flow_state_admissibility_status: str = "not_evaluated",
    claim_support_status: str = "not_supported_until_governed_artifacts_ready",
    compute_conservative_score: bool = False,
) -> dict[str, Any]:
    """为单条 record 补齐 Flow evidence 协议字段。

    该函数属于通用工程写法。调用方先提供当前阶段的默认语义, 再把原始 record
    覆盖进去, 从而保证不会改写已经存在的真实字段。只有在
    compute_conservative_score 为 True 且 record 尚未提供 S_final_conservative 时,
    才会根据现有 endpoint、path 和 velocity 字段补一个保守分数。
    """
    merged = flow_evidence_protocol_defaults(
        negative_family=negative_family,
        trajectory_source_level=trajectory_source_level,
        sampler_signature_placeholder=sampler_signature_placeholder,
        flow_state_admissibility_status=flow_state_admissibility_status,
        claim_support_status=claim_support_status,
    )
    merged.update(record)
    if compute_conservative_score and merged.get("S_final_conservative") is None:
        merged["S_final_conservative"] = conservative_flow_score(merged)
    return merged


def with_flow_evidence_protocol_defaults_many(
    records: list[dict[str, Any]],
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """为一组 records 补齐 Flow evidence 协议字段。

    该函数用于 runner 写出 JSONL 前的最后一步归一化。它只补字段, 不改变记录数量,
    因而不会影响 prompt、seed、attack 或 negative family 的覆盖统计。
    """
    return [with_flow_evidence_protocol_defaults(record, **kwargs) for record in records]


def conservative_flow_score(record: dict[str, Any]) -> float | None:
    """从 endpoint、path 和 velocity 字段计算保守最终分数。

    该函数属于通用工程写法。保守聚合只使用已经存在的数值字段, 并取最小值。
    这样可以防止单一证据层绕过 endpoint、path 或 velocity 的一致性要求。
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
