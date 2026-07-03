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
    }
