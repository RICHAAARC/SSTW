"""冻结 probe、pilot 与 full 共用的论文机制配置。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


PAPER_PROFILE_NAMES = {"probe_paper", "pilot_paper", "full_paper"}
COMMON_CONTRACT_PATH_FIELD = "paper_profile_common_contract_path"

# 正式 profile 文件只允许在以下字段上表达层级差异。这里使用显式字段集合而不是
# `minimum_*` 一类宽泛前缀, 主要考虑在于避免未来把新的机制阈值或证据开关伪装成
# “统计规模”后绕过公共契约。新增差异字段时必须先在此处完成分类和代码审阅。
PAPER_PROFILE_ONLY_FIELD_CATEGORIES: dict[str, frozenset[str]] = {
    "profile_identity": frozenset({
        "paper_result_level",
        "paper_profile_names",
        "stage_id",
    }),
    "target_fpr_and_statistical_scale": frozenset({
        "target_fpr",
        "blocked_target_fpr",
        "minimum_prompt_count",
        "minimum_seed_per_prompt",
        "minimum_calibration_seed_per_prompt",
        "minimum_test_seed_per_prompt",
        "minimum_unique_video_count",
        "minimum_calibration_unique_video_count",
        "minimum_test_unique_video_count",
        "minimum_clean_negative_count",
        "minimum_calibration_negative_event_count",
        "minimum_heldout_test_negative_event_count",
        "minimum_heldout_attacked_positive_event_count",
        "minimum_calibration_negative_event_count_per_family",
        "minimum_heldout_negative_event_count_per_family",
        "minimum_negative_event_count_per_family",
        "minimum_attack_event_count_per_attack",
        "minimum_independent_negative_video_count_for_fpr_upper_bound",
        "minimum_external_baseline_trace_count",
        "minimum_internal_ablation_trace_count",
    }),
    "outer_stage_prerequisite": frozenset({
        "require_probe_paper_gate_passed",
        "require_probe_paper_to_pilot_paper_transition_decision",
        "require_pilot_paper_gate_passed",
        "require_pilot_paper_to_full_paper_transition_decision",
        "require_full_paper_to_submission_freeze_transition_decision",
        "require_full_paper_to_submission_freeze_transition_decision_scope",
        "submission_freeze_allowed_after_full_paper_checker_pass",
    }),
    "profile_documentation": frozenset({
        "paper_protocol_difference_from_probe_paper",
        "paper_protocol_difference_from_pilot_paper",
        "paper_protocol_difference_from_full_paper",
        "required_runtime_attack_protocol_note",
    }),
}

PAPER_PROFILE_ONLY_FIELDS = frozenset().union(
    *PAPER_PROFILE_ONLY_FIELD_CATEGORIES.values()
)

PAPER_PROFILE_EXPECTED_IDENTITY: dict[str, dict[str, Any]] = {
    "probe_paper": {
        "paper_profile_names": ["probe_paper"],
        "stage_id": "probe_paper_generative_probe_gate",
    },
    "pilot_paper": {
        "paper_profile_names": ["pilot_paper"],
        "stage_id": "pilot_paper_generative_probe_gate",
    },
    "full_paper": {
        "paper_profile_names": ["full_paper"],
        "stage_id": "full_paper_generative_probe_gate",
    },
}

PAPER_PROFILE_EXPECTED_TARGET_FPR = {
    "probe_paper": 0.1,
    "pilot_paper": 0.01,
    "full_paper": 0.001,
}

PAPER_PROFILE_EXPECTED_BLOCKED_TARGET_FPR = {
    "probe_paper": 0.01,
    "pilot_paper": 0.001,
    "full_paper": None,
}

PAPER_PROFILE_REQUIRED_SCALE_FIELDS = (
    PAPER_PROFILE_ONLY_FIELD_CATEGORIES["target_fpr_and_statistical_scale"]
    - {"blocked_target_fpr"}
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"paper profile JSON 顶层必须是对象: {path}")
    return payload


def enforce_paper_profile_common_contract(
    profile: Mapping[str, Any],
    profile_path: str | Path,
) -> dict[str, Any]:
    """验证 profile 未漂移并返回合并后的只读语义副本。

    profile 文件仍显式保存公共字段, 便于独立审阅。加载时逐字段与公共契约比对,
    任一差异立即失败。公共契约之外只接受本模块显式分类的 profile 身份、目标
    FPR、样本统计规模、外层阶段前置与说明字段; 任意未登记键立即失败。这样可以
    从机制上阻断 pilot 残留旧语义或用新证据开关制造单文件静默漂移。
    """

    merged = dict(profile)
    level = str(merged.get("paper_result_level") or "")
    if level not in PAPER_PROFILE_NAMES:
        return merged
    raw_contract_path = merged.get(COMMON_CONTRACT_PATH_FIELD)
    if not raw_contract_path:
        canonical_names = {
            "probe_paper_generative_probe.json",
            "pilot_paper_generative_probe.json",
            "full_paper_generative_probe.json",
        }
        if Path(profile_path).name in canonical_names:
            raise KeyError(f"{level} 缺少 {COMMON_CONTRACT_PATH_FIELD}")
        # 单元测试或用户自定义的临时 profile 可以只覆盖局部门禁。正式仓库配置
        # 由上面的规范文件名检查和约束测试共同保证不能绕过公共契约。
        return merged
    contract_path = Path(str(raw_contract_path))
    if not contract_path.is_absolute() and not contract_path.exists():
        candidate = Path(profile_path).parent / contract_path
        if candidate.exists():
            contract_path = candidate
    if not contract_path.exists():
        raise FileNotFoundError(f"paper profile 公共契约不存在: {contract_path}")
    contract = _read_json(contract_path)
    unknown_fields = sorted(
        set(merged)
        - set(contract)
        - set(PAPER_PROFILE_ONLY_FIELDS)
        - {COMMON_CONTRACT_PATH_FIELD}
    )
    if unknown_fields:
        raise ValueError(
            "paper profile 包含未登记的层级差异字段: "
            + json.dumps(unknown_fields, ensure_ascii=False)
        )
    drift = {
        key: {"expected": expected, "observed": merged.get(key)}
        for key, expected in contract.items()
        if merged.get(key) != expected
    }
    if drift:
        raise ValueError(
            "paper profile 公共机制配置漂移: "
            + json.dumps(drift, ensure_ascii=False, sort_keys=True)
        )
    identity_drift = {
        key: {"expected": expected, "observed": merged.get(key)}
        for key, expected in PAPER_PROFILE_EXPECTED_IDENTITY[level].items()
        if merged.get(key) != expected
    }
    if identity_drift:
        raise ValueError(
            "paper profile 身份字段漂移: "
            + json.dumps(identity_drift, ensure_ascii=False, sort_keys=True)
        )
    missing_scale_fields = sorted(
        PAPER_PROFILE_REQUIRED_SCALE_FIELDS - set(merged)
    )
    if missing_scale_fields:
        raise ValueError(
            "paper profile 缺少必填统计规模字段: "
            + json.dumps(missing_scale_fields, ensure_ascii=False)
        )
    expected_target_fpr = PAPER_PROFILE_EXPECTED_TARGET_FPR[level]
    if float(merged["target_fpr"]) != expected_target_fpr:
        raise ValueError(
            f"{level} target_fpr 必须为 {expected_target_fpr}, "
            f"实际为 {merged['target_fpr']}"
        )
    expected_blocked_fpr = PAPER_PROFILE_EXPECTED_BLOCKED_TARGET_FPR[level]
    observed_blocked_fpr = merged.get("blocked_target_fpr")
    if (
        expected_blocked_fpr is None
        and observed_blocked_fpr is not None
    ) or (
        expected_blocked_fpr is not None
        and (
            observed_blocked_fpr is None
            or float(observed_blocked_fpr) != expected_blocked_fpr
        )
    ):
        raise ValueError(
            f"{level} blocked_target_fpr 必须为 {expected_blocked_fpr}, "
            f"实际为 {observed_blocked_fpr}"
        )
    invalid_difference_fields = sorted(
        field_name
        for field_name in (
            "paper_protocol_difference_from_probe_paper",
            "paper_protocol_difference_from_pilot_paper",
            "paper_protocol_difference_from_full_paper",
        )
        if field_name in merged
        and merged[field_name] != "sample_scale_and_target_fpr_only"
    )
    if invalid_difference_fields:
        raise ValueError(
            "paper profile 层级差异声明只能是样本规模与目标 FPR: "
            + json.dumps(invalid_difference_fields, ensure_ascii=False)
        )
    invalid_scale_fields = sorted(
        field_name
        for field_name in PAPER_PROFILE_REQUIRED_SCALE_FIELDS
        if field_name != "target_fpr"
        and (
            isinstance(merged.get(field_name), bool)
            or not isinstance(merged.get(field_name), int)
            or int(merged[field_name]) <= 0
        )
    )
    if invalid_scale_fields:
        raise ValueError(
            "paper profile 统计规模字段必须为正整数: "
            + json.dumps(invalid_scale_fields, ensure_ascii=False)
        )
    merged.update(contract)
    merged["paper_profile_common_contract_resolved_path"] = str(contract_path)
    merged["paper_profile_common_contract_status"] = "matched"
    return merged
