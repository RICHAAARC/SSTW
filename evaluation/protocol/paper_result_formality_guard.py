"""论文结果包正式性门禁。

该模块提供三层 paper profile 共用的轻量审计逻辑。它只读取已经落盘的
records、artifacts、thresholds、manifests、tables、figures 和 reports, 不运行实验, 不补造结果。
其职责是防止 proxy、placeholder 或 fallback 证据进入 `probe_paper`、
`pilot_paper` 和 `full_paper` 的正式结果包。
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


PAPER_RESULT_CLAIM_IDS: dict[str, str] = {
    "probe_paper": "probe_claim",
    "pilot_paper": "pilot_claim",
    "full_paper": "full_claim",
}

PAPER_RESULT_CLAIM_PASS_STATUSES: dict[str, str] = {
    "probe_paper": "probe_claim_supported",
    "pilot_paper": "pilot_claim_supported",
    "full_paper": "full_claim_supported",
}

PAPER_RESULT_CLAIM_FAIL_STATUSES: dict[str, str] = {
    "probe_paper": "probe_claim_blocked",
    "pilot_paper": "pilot_claim_blocked",
    "full_paper": "full_claim_blocked",
}

FORMAL_RESULT_PROFILE_TARGET_FPRS: dict[str, float] = {
    "probe_paper": 0.1,
    "pilot_paper": 0.01,
    "full_paper": 0.001,
}

BANNED_FORMALITY_TERMS = ("proxy", "placeholder", "fallback")

# `runtime_attack_proxy_free=true` 是 attack 证据的防 proxy 声明, 不是 proxy 结果本身。
ALLOWED_PROXY_GUARD_FIELDS = {"runtime_attack_proxy_free"}

EXCLUDED_SCAN_FILE_NAMES = {
    "paper_result_formality_guard_decision.json",
    "paper_result_formality_guard_records.jsonl",
    "paper_result_formality_guard_table.csv",
    "paper_profile_gate_decision.json",
    "paper_profile_gate_records.jsonl",
    "paper_profile_gate_table.csv",
    "probe_paper_gate_decision.json",
    "probe_paper_gate_records.jsonl",
    "probe_paper_gate_table.csv",
    "pilot_paper_gate_decision.json",
    "pilot_paper_gate_records.jsonl",
    "pilot_paper_gate_table.csv",
    "full_paper_gate_decision.json",
    "full_paper_gate_records.jsonl",
    "full_paper_gate_table.csv",
    "full_paper_result_checker_decision.json",
    "full_paper_result_decision.json",
    "full_paper_result_checker_records.jsonl",
    "full_paper_result_checker_table.csv",
}

SCAN_DIR_NAMES = ("records", "artifacts", "thresholds", "manifests", "tables", "figures", "reports")
SCAN_SUFFIXES = (".json", ".jsonl", ".csv", ".md")
FORMALITY_SCAN_ARTIFACT_NAMES = {
    "runtime_attack_decision.json",
    "runtime_detection_decision.json",
    "external_baseline_comparison_decision.json",
    "external_baseline_self_containment_decision.json",
    "sstw_measured_formal_decision.json",
    "fair_detection_calibration_decision.json",
    "formal_method_baseline_comparison_decision.json",
    "formal_baseline_difference_interval_decision.json",
    "data_split_and_leakage_guard_decision.json",
    "formal_internal_ablation_summary_decision.json",
    "validation_internal_ablation_decision.json",
    "adaptive_attack_decision.json",
    "replay_and_sketch_gate_decision.json",
    "statistical_confidence_interval_decision.json",
    "low_fpr_formal_statistics_decision.json",
    "paper_result_artifact_skeleton_decision.json",
    "validation_artifact_rebuild_dry_run_decision.json",
    "artifact_rebuild_dry_run_decision.json",
    "motion_consistency_exclusion_decision.json",
    "motion_threshold_calibration_decision.json",
    "formal_adaptive_attack_execution_decision.json",
}

GOVERNANCE_POLICY_FIELD_TOKENS = (
    "prohibited",
    "forbidden",
    "rejected",
    "blocked_terms",
    "fail_closed_policy",
    "ban_rule",
    "fallback_rule",
)


@dataclass(frozen=True)
class FormalityViolation:
    """单个正式性违规项。

    `field_path` 描述 JSON / JSONL / CSV 中的字段位置。该对象只用于审计输出,
    不承载任何正式论文指标。
    """

    relative_path: str
    field_path: str
    violation_kind: str
    observed_value_preview: str

    def as_dict(self) -> dict[str, str]:
        """转换为可序列化 dict。"""

        return {
            "relative_path": self.relative_path,
            "field_path": self.field_path,
            "violation_kind": self.violation_kind,
            "observed_value_preview": self.observed_value_preview,
        }


def paper_claim_id_for_level(paper_result_level: str) -> str:
    """返回当前 paper result level 对应的正式 claim id。"""

    return PAPER_RESULT_CLAIM_IDS.get(str(paper_result_level), "unknown_paper_claim")


def paper_claim_pass_status_for_level(paper_result_level: str) -> str:
    """返回当前 paper result level 通过时的正式 claim 状态。"""

    return PAPER_RESULT_CLAIM_PASS_STATUSES.get(str(paper_result_level), "unknown_paper_claim_supported")


def paper_claim_fail_status_for_level(paper_result_level: str) -> str:
    """返回当前 paper result level 阻断时的正式 claim 状态。"""

    return PAPER_RESULT_CLAIM_FAIL_STATUSES.get(str(paper_result_level), "unknown_paper_claim_blocked")


def _preview(value: Any, *, max_length: int = 160) -> str:
    """生成稳定短预览, 避免把长路径或大对象写入审计结果。"""

    text = str(value)
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def _contains_banned_term(value: str) -> str | None:
    """检查文本是否包含正式结果包禁用的弱证据标记。"""

    lowered = value.lower()
    for term in BANNED_FORMALITY_TERMS:
        if term in lowered:
            return term
    return None


def _is_governance_policy_field_path(field_path: str) -> bool:
    """判断字段路径是否属于禁用来源清单或 fail-closed 策略描述。"""

    lowered_path = field_path.lower()
    return any(token in lowered_path for token in GOVERNANCE_POLICY_FIELD_TOKENS)


def _violation(
    *,
    relative_path: str,
    field_path: str,
    violation_kind: str,
    observed_value: Any,
) -> FormalityViolation:
    """构造单个正式性违规项。"""

    return FormalityViolation(
        relative_path=relative_path,
        field_path=field_path,
        violation_kind=violation_kind,
        observed_value_preview=_preview(observed_value),
    )


def _scan_mapping(
    payload: Mapping[str, Any],
    *,
    relative_path: str,
    field_path: str,
) -> list[FormalityViolation]:
    """递归扫描 dict 中的弱证据标记。

    该函数属于通用治理写法。它区分“说明该 attack 不是 proxy”的布尔防护字段
    与真正的 proxy 结果字段, 从而避免误伤 `runtime_attack_proxy_free=true`。
    """

    violations: list[FormalityViolation] = []
    for key, value in payload.items():
        key_text = str(key)
        child_path = f"{field_path}.{key_text}" if field_path else key_text
        if key_text in ALLOWED_PROXY_GUARD_FIELDS:
            if not (value is True or str(value).lower() == "true"):
                violations.append(_violation(
                    relative_path=relative_path,
                    field_path=child_path,
                    violation_kind="proxy_guard_not_true",
                    observed_value=value,
                ))
            continue
        banned_key_term = _contains_banned_term(key_text)
        if banned_key_term and not _is_governance_policy_field_path(child_path):
            violations.append(_violation(
                relative_path=relative_path,
                field_path=child_path,
                violation_kind=f"{banned_key_term}_field_forbidden_in_formal_result",
                observed_value=key_text,
            ))
        violations.extend(_scan_payload(value, relative_path=relative_path, field_path=child_path))
    return violations


def _scan_sequence(
    payload: Iterable[Any],
    *,
    relative_path: str,
    field_path: str,
) -> list[FormalityViolation]:
    """递归扫描 list / tuple 中的弱证据标记。"""

    violations: list[FormalityViolation] = []
    for index, item in enumerate(payload):
        violations.extend(_scan_payload(item, relative_path=relative_path, field_path=f"{field_path}[{index}]"))
    return violations


def _scan_string(
    value: str,
    *,
    relative_path: str,
    field_path: str,
) -> list[FormalityViolation]:
    """扫描字符串值中的弱证据标记。"""

    term = _contains_banned_term(value)
    if not term:
        return []
    if _is_governance_policy_field_path(field_path):
        return []
    return [_violation(
        relative_path=relative_path,
        field_path=field_path,
        violation_kind=f"{term}_value_forbidden_in_formal_result",
        observed_value=value,
    )]


def _scan_payload(
    payload: Any,
    *,
    relative_path: str,
    field_path: str,
) -> list[FormalityViolation]:
    """递归扫描任意 JSON 兼容对象。"""

    if isinstance(payload, Mapping):
        return _scan_mapping(payload, relative_path=relative_path, field_path=field_path)
    if isinstance(payload, (list, tuple)):
        return _scan_sequence(payload, relative_path=relative_path, field_path=field_path)
    if isinstance(payload, str):
        return _scan_string(payload, relative_path=relative_path, field_path=field_path)
    return []


def _read_json(path: Path) -> Any:
    """读取 JSON 文件, 兼容 UTF-8 BOM。"""

    return json.loads(path.read_text(encoding="utf-8-sig"))


def _scan_json_file(path: Path, *, run_root: Path) -> list[FormalityViolation]:
    """扫描 JSON 文件。"""

    relative_path = path.relative_to(run_root).as_posix()
    return _scan_payload(_read_json(path), relative_path=relative_path, field_path="")


def _scan_jsonl_file(path: Path, *, run_root: Path) -> list[FormalityViolation]:
    """扫描 JSONL 文件。"""

    relative_path = path.relative_to(run_root).as_posix()
    violations: list[FormalityViolation] = []
    for line_index, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        violations.extend(_scan_payload(
            json.loads(line),
            relative_path=relative_path,
            field_path=f"line[{line_index}]",
        ))
    return violations


def _scan_csv_file(path: Path, *, run_root: Path) -> list[FormalityViolation]:
    """扫描 CSV 文件中的字段名和值。"""

    relative_path = path.relative_to(run_root).as_posix()
    violations: list[FormalityViolation] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader, start=1):
            violations.extend(_scan_mapping(
                row,
                relative_path=relative_path,
                field_path=f"row[{row_index}]",
            ))
    return violations


def _scan_markdown_file(path: Path, *, run_root: Path) -> list[FormalityViolation]:
    """扫描 Markdown 报告正文。"""

    relative_path = path.relative_to(run_root).as_posix()
    violations: list[FormalityViolation] = []
    for line_index, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        violations.extend(_scan_string(line, relative_path=relative_path, field_path=f"line[{line_index}]"))
    return violations


def _iter_scan_files(run_root: Path) -> list[Path]:
    """枚举正式结果包需要扫描的结构化文件。"""

    files: list[Path] = []
    for dirname in SCAN_DIR_NAMES:
        root = run_root / dirname
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.name in EXCLUDED_SCAN_FILE_NAMES:
                continue
            if path.suffix.lower() in SCAN_SUFFIXES:
                files.append(path)
    return sorted(files)


def _scan_file(path: Path, *, run_root: Path) -> list[FormalityViolation]:
    """按文件类型扫描结构化结果文件。"""

    suffix = path.suffix.lower()
    if suffix == ".json":
        return _scan_json_file(path, run_root=run_root)
    if suffix == ".jsonl":
        return _scan_jsonl_file(path, run_root=run_root)
    if suffix == ".csv":
        return _scan_csv_file(path, run_root=run_root)
    if suffix == ".md":
        return _scan_markdown_file(path, run_root=run_root)
    return []


def build_paper_result_formality_guard(
    run_root: str | Path,
    *,
    paper_result_level: str,
    target_fpr: float | None,
) -> dict[str, Any]:
    """构建三层 paper result 共用的正式性审计结果。

    probe_paper、pilot_paper 和 full_paper 都是正式流程, 只允许样本规模和
    target FPR 不同。因此只要结果包中出现 proxy、placeholder 或 fallback 证据,
    当前层级的 claim 就必须 fail-closed。
    """

    run_root = Path(run_root)
    paper_result_level = str(paper_result_level)
    expected_claim_id = paper_claim_id_for_level(paper_result_level)
    expected_target_fpr = FORMAL_RESULT_PROFILE_TARGET_FPRS.get(paper_result_level)
    target_fpr_mismatch = (
        target_fpr is None
        or expected_target_fpr is None
        or abs(float(target_fpr) - float(expected_target_fpr)) > 1e-12
    )
    files = _iter_scan_files(run_root)
    violations: list[FormalityViolation] = []
    for path in files:
        violations.extend(_scan_file(path, run_root=run_root))
    if target_fpr_mismatch:
        violations.append(_violation(
            relative_path="<protocol_config>",
            field_path="target_fpr",
            violation_kind="paper_result_target_fpr_mismatch",
            observed_value=target_fpr,
        ))
    decision = "PASS" if not violations else "FAIL"
    blocking_terms = sorted({
        kind
        for violation in violations
        for kind in BANNED_FORMALITY_TERMS
        if kind in violation.violation_kind
    })
    return {
        "stage_id": "paper_result_formality_guard",
        "paper_result_formality_guard_decision": decision,
        "paper_result_formality_guard_status": (
            "formal_result_package_clean"
            if decision == "PASS"
            else "formal_result_package_blocked_by_weak_evidence_markers"
        ),
        "paper_result_formality_guard_violation_count": len(violations),
        "paper_result_formality_guard_scanned_file_count": len(files),
        "paper_result_formality_guard_blocking_terms": blocking_terms,
        "paper_result_formality_guard_violations": [violation.as_dict() for violation in violations],
        "paper_result_level": paper_result_level,
        "paper_claim_id": expected_claim_id,
        "paper_claim_level": paper_result_level,
        "paper_claim_support_status": (
            paper_claim_pass_status_for_level(paper_result_level)
            if decision == "PASS"
            else paper_claim_fail_status_for_level(paper_result_level)
        ),
        "target_fpr": target_fpr,
        "expected_formal_target_fpr": expected_target_fpr,
        "paper_result_target_fpr_matches_profile": not target_fpr_mismatch,
    }
