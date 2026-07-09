"""现代 external baseline 分数语义归一化工具。

该模块属于通用适配层。它只负责从第三方官方输出 JSON 中选择“检测水印存在性”
所需的连续分数, 并把 payload bit accuracy 等辅助分数保留下来。项目特定考虑是:
论文主比较不能直接混用 bit accuracy、confidence 和二值 detected, 因此所有
official adapter 必须显式写出 score semantics, 后续公平比较门禁才能校准阈值。
"""

from __future__ import annotations

from typing import Any, Mapping


DETECTOR_SCORE_FIELDS = (
    "raw_detector_score",
    "external_baseline_raw_detector_score",
    "detection_score",
    "confidence",
    "watermark_score",
    "score",
    "external_baseline_score",
    "bit_accuracy",
    "external_baseline_bit_accuracy",
    "detected",
    "external_baseline_detected",
)
PAYLOAD_BIT_ACCURACY_FIELDS = (
    "payload_bit_accuracy",
    "external_baseline_payload_bit_accuracy",
    "bit_accuracy",
    "external_baseline_bit_accuracy",
)



REQUIRED_OFFICIAL_REFERENCE_PROTOCOL_ANCHOR = "same_prompt_seed_attack_runtime_comparison_unit"
OFFICIAL_SCORE_EXTRACTION_POLICY_FIELDS = (
    "official_score_extraction_policy",
    "official_score_assignment_policy",
    "official_detection_logic",
    "external_baseline_official_score_extraction_policy",
    "external_baseline_official_score_assignment_policy",
)
OFFICIAL_SCORE_GRANULARITY_FIELDS = (
    "official_score_granularity",
    "external_baseline_official_score_granularity",
)
OFFICIAL_SCORE_VALUE_TYPE_FIELDS = (
    "official_score_value_type",
    "external_baseline_official_score_value_type",
)
OFFICIAL_CLEAN_NEGATIVE_SCORE_GRANULARITY_FIELDS = (
    "official_clean_negative_score_granularity",
    "external_baseline_official_clean_negative_score_granularity",
)
OFFICIAL_CLEAN_NEGATIVE_SCORE_VALUE_TYPE_FIELDS = (
    "official_clean_negative_score_value_type",
    "external_baseline_official_clean_negative_score_value_type",
)
FORMAL_POSITIVE_SCORE_GRANULARITIES = {"per_prompt_seed_attack"}
FORMAL_CLEAN_NEGATIVE_SCORE_GRANULARITIES = {
    "per_prompt_seed_attack",
    "per_prompt_seed",
    "per_clean_negative_sample",
}
FORMAL_SCORE_VALUE_TYPES = {
    "continuous_detector_score",
    "payload_bit_accuracy_score",
}


def explicit_score_semantics(payload: Mapping[str, Any]) -> str:
    """读取官方输出显式声明的分数语义。"""

    return str(payload.get("score_semantics") or payload.get("external_baseline_score_semantics") or "").strip()


def official_score_extraction_policy(payload: Mapping[str, Any]) -> str:
    """读取官方输出显式声明的分数抽取策略。"""

    for field_name in OFFICIAL_SCORE_EXTRACTION_POLICY_FIELDS:
        value = str(payload.get(field_name) or "").strip()
        if value:
            return value
    return ""


def _first_text(payload: Mapping[str, Any], field_names: tuple[str, ...]) -> str:
    """按候选字段顺序读取第一个非空文本值。"""

    for field_name in field_names:
        value = str(payload.get(field_name) or "").strip()
        if value:
            return value
    return ""


def _policy_for_score(payload: Mapping[str, Any], *, clean_negative: bool = False) -> str:
    """读取 positive 或 clean negative 分数的官方抽取策略。"""

    if clean_negative:
        value = _first_text(
            payload,
            (
                "official_clean_negative_score_extraction_policy",
                "official_clean_negative_score_assignment_policy",
                "external_baseline_official_clean_negative_score_extraction_policy",
                "external_baseline_official_clean_negative_score_assignment_policy",
            ),
        )
        if value:
            return value
    return official_score_extraction_policy(payload)


def _score_semantics_for_value_type(payload: Mapping[str, Any], *, clean_negative: bool = False) -> str:
    """读取 positive 或 clean negative 分数字段的语义声明。"""

    if clean_negative:
        value = _first_text(
            payload,
            (
                "official_clean_negative_score_semantics",
                "external_baseline_clean_negative_score_semantics",
                "clean_negative_score_semantics",
            ),
        )
        if value:
            return value
    return explicit_score_semantics(payload)


def _policy_uses_aggregate_assignment(policy: str, result_key: str = "") -> bool:
    """判断官方分数是否来自跨样本聚合, 不能进入正式公平比较。"""

    text = f"{policy} {result_key}".lower()
    aggregate_markers = (
        "aggregate_",
        "mean_over_npz_entries",
        "mean_over_temporal_results",
        "mean_over_available_temporal_attacks",
        "official_bit_accuracy_npz_mean",
        "official_decode_acc_temporal_result_mean",
    )
    return any(marker in text for marker in aggregate_markers)


def _result_key_for_score(payload: Mapping[str, Any], *, clean_negative: bool = False) -> str:
    """读取 positive 或 clean negative 分数绑定的官方 result key。"""

    if clean_negative:
        return _first_text(
            payload,
            (
                "official_clean_negative_result_key",
                "external_baseline_official_clean_negative_result_key",
            ),
        )
    return _first_text(
        payload,
        (
            "official_result_key",
            "external_baseline_official_result_key",
        ),
    )


def infer_official_score_granularity(payload: Mapping[str, Any], *, clean_negative: bool = False) -> str:
    """推断官方分数的样本粒度。

    通用工程写法是把“分数数值”和“分数来自哪个 comparison unit”拆开记录。
    项目特定要求是: paper profile 的正式 positive 比较只能使用同一
    prompt / seed / attack anchor 上的分数, 不能把官方 aggregate 均值伪装成
    单条样本分数。
    """

    explicit_fields = (
        OFFICIAL_CLEAN_NEGATIVE_SCORE_GRANULARITY_FIELDS
        if clean_negative
        else OFFICIAL_SCORE_GRANULARITY_FIELDS
    )
    explicit = _first_text(payload, explicit_fields)
    if explicit:
        return explicit
    policy = _policy_for_score(payload, clean_negative=clean_negative)
    result_key = _result_key_for_score(payload, clean_negative=clean_negative)
    if _policy_uses_aggregate_assignment(policy, result_key):
        return "aggregate"
    policy_lower = policy.lower()
    if "per_prompt_seed_runtime_attack" in policy_lower:
        return "per_prompt_seed_attack"
    if clean_negative and "per_prompt_seed" in policy_lower:
        return "per_prompt_seed"
    if "per_prompt_seed" in policy_lower:
        return "per_prompt_seed"
    if any(name in policy_lower for name in ("videoseal", "videoshield", "vidsig")):
        return "per_prompt_seed_attack"
    if policy_lower == "test_official_detector_confidence":
        return "per_prompt_seed_attack"
    anchor = str(payload.get("official_reference_protocol_anchor") or "").strip()
    if anchor == REQUIRED_OFFICIAL_REFERENCE_PROTOCOL_ANCHOR and not clean_negative:
        return "per_prompt_seed_attack"
    if clean_negative and (
        payload.get("external_baseline_clean_negative_score") not in {None, "", "unsupported"}
        or payload.get("clean_negative_score") not in {None, "", "unsupported"}
    ):
        return "per_clean_negative_sample"
    return "unspecified_score_granularity"


def infer_official_score_value_type(
    payload: Mapping[str, Any],
    *,
    clean_negative: bool = False,
    selected_score_field: str | None = None,
) -> str:
    """推断官方分数的值类型。

    该函数不改变分数本身, 只给后续门禁提供可审计口径。连续 detector score
    和 payload bit accuracy 可以在方法自身 clean negative 分布上重新校准阈值;
    固定 FPR 日志结果和二值 decision 不能再次作为连续分数做公平校准。
    """

    explicit_fields = (
        OFFICIAL_CLEAN_NEGATIVE_SCORE_VALUE_TYPE_FIELDS
        if clean_negative
        else OFFICIAL_SCORE_VALUE_TYPE_FIELDS
    )
    explicit = _first_text(payload, explicit_fields)
    if explicit:
        return explicit
    semantics = _score_semantics_for_value_type(payload, clean_negative=clean_negative)
    field_name = selected_score_field or (first_present_field(payload, DETECTOR_SCORE_FIELDS) or ("", None))[0]
    if semantics in {"watermark_presence_confidence", "watermark_presence_detector_score"}:
        return "continuous_detector_score"
    if semantics == "official_tpr_at_fixed_fpr_detection_score":
        return "fixed_fpr_detection_score"
    if semantics in {"payload_bit_accuracy_extraction_score", "payload_bit_accuracy_auxiliary_score"}:
        return "payload_bit_accuracy_score"
    if field_name in {"detected", "external_baseline_detected"}:
        return "binary_decision"
    if field_name in {
        "raw_detector_score",
        "external_baseline_raw_detector_score",
        "detection_score",
        "confidence",
        "watermark_score",
        "score",
        "external_baseline_score",
    }:
        return "continuous_detector_score"
    if field_name in {"bit_accuracy", "external_baseline_bit_accuracy"}:
        return "payload_bit_accuracy_score"
    return "unspecified_score_value_type"


def official_score_formal_comparison_summary(
    payload: Mapping[str, Any],
    *,
    clean_negative: bool = False,
    selected_score_field: str | None = None,
) -> dict[str, str]:
    """生成官方分数进入公平比较前的粒度与资格摘要。"""

    granularity = infer_official_score_granularity(payload, clean_negative=clean_negative)
    value_type = infer_official_score_value_type(
        payload,
        clean_negative=clean_negative,
        selected_score_field=selected_score_field,
    )
    policy = _policy_for_score(payload, clean_negative=clean_negative)
    result_key = _result_key_for_score(payload, clean_negative=clean_negative)
    eligible_granularities = (
        FORMAL_CLEAN_NEGATIVE_SCORE_GRANULARITIES
        if clean_negative
        else FORMAL_POSITIVE_SCORE_GRANULARITIES
    )
    block_reason = "none"
    if _policy_uses_aggregate_assignment(policy, result_key):
        block_reason = "aggregate_score_assignment_not_formal_comparison_eligible"
    elif granularity not in eligible_granularities:
        block_reason = f"score_granularity_not_formal_comparison_eligible:{granularity}"
    elif value_type not in FORMAL_SCORE_VALUE_TYPES:
        block_reason = f"score_value_type_not_formal_comparison_eligible:{value_type}"
    eligibility = "eligible" if block_reason == "none" else "blocked"
    prefix = "official_clean_negative_score" if clean_negative else "official_score"
    return {
        f"{prefix}_granularity": granularity,
        f"{prefix}_value_type": value_type,
        f"{prefix}_formal_comparison_eligibility": eligibility,
        f"{prefix}_formal_comparison_block_reason": block_reason,
    }


def validate_official_formal_comparison_eligibility(payload: Mapping[str, Any]) -> None:
    """校验 official 输出是否可进入 paper profile 正式公平比较。

    该函数与 `validate_official_score_extraction_payload` 的区别在于: 后者只确认
    有可解释的分数口径, 这里进一步确认该分数是逐 comparison unit、可重新用
    clean negative 分布校准阈值的正式分数。aggregate 均值、固定 FPR 日志值和
    二值 decision 都会被 fail closed。
    """

    positive = official_score_formal_comparison_summary(payload)
    if positive["official_score_formal_comparison_eligibility"] != "eligible":
        raise RuntimeError(
            "official_score_formal_comparison_ineligible:"
            f"{positive['official_score_formal_comparison_block_reason']}"
        )
    clean_negative = official_score_formal_comparison_summary(payload, clean_negative=True)
    if clean_negative["official_clean_negative_score_formal_comparison_eligibility"] != "eligible":
        raise RuntimeError(
            "official_clean_negative_score_formal_comparison_ineligible:"
            f"{clean_negative['official_clean_negative_score_formal_comparison_block_reason']}"
        )


def external_official_score_formal_comparison_payload(payload: Mapping[str, Any]) -> dict[str, str]:
    """构造写入 external baseline measured_formal record 的 positive 分数资格字段。"""

    summary = official_score_formal_comparison_summary(payload, clean_negative=False)
    return {
        "external_baseline_official_score_granularity": summary["official_score_granularity"],
        "external_baseline_official_score_value_type": summary["official_score_value_type"],
        "external_baseline_official_score_formal_comparison_eligibility": summary[
            "official_score_formal_comparison_eligibility"
        ],
        "external_baseline_official_score_formal_comparison_block_reason": summary[
            "official_score_formal_comparison_block_reason"
        ],
    }


def external_clean_negative_score_formal_comparison_payload(payload: Mapping[str, Any]) -> dict[str, str]:
    """构造写入 external baseline measured_formal record 的 clean negative 分数资格字段。"""

    summary = official_score_formal_comparison_summary(payload, clean_negative=True)
    return {
        "external_baseline_official_clean_negative_score_granularity": summary[
            "official_clean_negative_score_granularity"
        ],
        "external_baseline_official_clean_negative_score_value_type": summary[
            "official_clean_negative_score_value_type"
        ],
        "external_baseline_official_clean_negative_score_formal_comparison_eligibility": summary[
            "official_clean_negative_score_formal_comparison_eligibility"
        ],
        "external_baseline_official_clean_negative_score_formal_comparison_block_reason": summary[
            "official_clean_negative_score_formal_comparison_block_reason"
        ],
    }


def validate_official_score_extraction_payload(payload: Mapping[str, Any]) -> None:
    """校验 official 输出是否足以进入公平检测校准。

    paper profile 的公平比较要求每个 baseline 明确说明分数从哪个官方检测口径
    抽取、分数方向是什么, 并绑定同一 prompt / seed / attack comparison unit。
    该函数只检查口径证据, 不替代 clean negative 和 official bundle provenance 检查。
    """

    extract_raw_detector_score(payload)
    semantics = explicit_score_semantics(payload)
    if not semantics or semantics == "unspecified_detector_score":
        raise RuntimeError("official_score_extraction_missing_score_semantics")
    orientation = str(
        payload.get("score_orientation")
        or payload.get("external_baseline_score_orientation")
        or ""
    ).strip()
    if orientation != "higher_is_more_watermarked":
        raise RuntimeError(f"official_score_extraction_unsupported_score_orientation:{orientation or 'missing'}")
    policy = official_score_extraction_policy(payload)
    if not policy:
        raise RuntimeError("official_score_extraction_missing_policy")
    anchor = str(payload.get("official_reference_protocol_anchor") or "").strip()
    if anchor != REQUIRED_OFFICIAL_REFERENCE_PROTOCOL_ANCHOR:
        raise RuntimeError(f"official_score_extraction_missing_protocol_anchor:{anchor or 'missing'}")

def safe_float(value: Any, default: float = 0.0) -> float:
    """把官方输出中的数值字段安全转换为 float。"""

    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def first_present_field(payload: Mapping[str, Any], field_names: tuple[str, ...]) -> tuple[str, Any] | None:
    """返回第一个存在且非空的字段名和值。"""

    for field_name in field_names:
        if field_name in payload and payload.get(field_name) not in {None, ""}:
            return field_name, payload.get(field_name)
    return None


def extract_raw_detector_score(payload: Mapping[str, Any]) -> tuple[float, str]:
    """提取主检测分数。

    通用工程规则是优先使用 detector confidence / detection score。只有官方输出不
    提供连续检测分数时, 才退回 payload bit accuracy 或二值 detected。这样可以
    避免把 VideoSeal 的 bit accuracy 误当成 presence detection 分数。
    """

    selected = first_present_field(payload, DETECTOR_SCORE_FIELDS)
    if selected is None:
        raise ValueError("official_output_missing_score")
    field_name, value = selected
    if field_name in {"detected", "external_baseline_detected"}:
        return (1.0 if bool(value) else 0.0), field_name
    return safe_float(value, 0.0), field_name


def extract_payload_bit_accuracy(payload: Mapping[str, Any]) -> float | None:
    """提取 payload bit accuracy 辅助分数, 不存在时返回 None。"""

    selected = first_present_field(payload, PAYLOAD_BIT_ACCURACY_FIELDS)
    if selected is None:
        return None
    return safe_float(selected[1], 0.0)


def infer_score_semantics(payload: Mapping[str, Any], *, selected_score_field: str | None = None) -> str:
    """根据官方字段推断分数语义。

    该推断只作为适配器默认值。若某个 baseline 官方 wrapper 已经提供
    `score_semantics`, 则优先使用 wrapper 的显式声明。
    """

    explicit = str(payload.get("score_semantics") or payload.get("external_baseline_score_semantics") or "").strip()
    if explicit:
        return explicit
    field_name = selected_score_field or (first_present_field(payload, DETECTOR_SCORE_FIELDS) or ("", None))[0]
    if field_name in {"raw_detector_score", "external_baseline_raw_detector_score", "detection_score", "confidence", "watermark_score", "score"}:
        return "watermark_presence_detector_score"
    if field_name in {"bit_accuracy", "external_baseline_bit_accuracy"}:
        return "payload_bit_accuracy_auxiliary_score"
    if field_name in {"detected", "external_baseline_detected"}:
        return "binary_official_decision_only"
    return "unspecified_detector_score"


def normalized_score_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """构造统一的分数语义 payload。

    返回值可以直接合并到 bridge / command adapter 的 governed record 中。主检测
    分数字段为 `external_baseline_raw_detector_score`, payload 解码准确率作为
    `external_baseline_payload_bit_accuracy` 保留。
    """

    raw_score, selected_field = extract_raw_detector_score(payload)
    bit_accuracy = extract_payload_bit_accuracy(payload)
    return {
        "external_baseline_raw_detector_score": round(float(raw_score), 6),
        "external_baseline_score": round(float(raw_score), 6),
        "external_baseline_score_field": selected_field,
        "external_baseline_score_semantics": infer_score_semantics(payload, selected_score_field=selected_field),
        "external_baseline_score_orientation": str(payload.get("score_orientation") or payload.get("external_baseline_score_orientation") or "higher_is_more_watermarked"),
        "external_baseline_payload_bit_accuracy": round(float(bit_accuracy), 6) if bit_accuracy is not None else None,
        **external_official_score_formal_comparison_payload(payload),
    }
