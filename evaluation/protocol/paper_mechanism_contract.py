"""校验三个 paper profile 是否共享同一完整论文机制。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


REQUIRED_COMPLETE_RESULT_FLAGS = (
    "require_claim1_full_support",
    "require_claim2_full_support",
    "require_claim3_full_support",
    "require_fair_detection_calibration",
    "require_formal_internal_ablation_summary",
    "require_formal_method_baseline_comparison",
    "require_formal_baseline_difference_interval",
    "require_sstw_advantage_claim_ready",
    "require_adaptive_attack_records",
    "require_real_adaptive_attack_records",
    "require_video_quality_metric_records",
    "require_baseline_matched_video_quality_metrics",
    "require_confidence_interval_report",
    "require_data_split_and_leakage_guard",
    "require_claim_audit_report",
    "require_artifact_rebuild_report",
)


@dataclass(frozen=True)
class PaperMechanismContractAudit:
    """保存 profile 机制一致性审计结果。"""

    passed: bool
    contract_id: str
    audited_profile_count: int
    violations: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """转换为可写入 governed artifact 的字典。"""

        return {
            "paper_mechanism_contract_decision": "PASS" if self.passed else "FAIL",
            "formal_mechanism_contract_id": self.contract_id,
            "audited_profile_count": self.audited_profile_count,
            "paper_mechanism_contract_violations": list(self.violations),
        }


def load_paper_mechanism_contract(path: str | Path) -> dict[str, Any]:
    """读取完整机制契约并检查三层主张基础约束。"""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if len(payload.get("required_claims") or []) != 3:
        raise ValueError("完整论文机制必须登记三层主张")
    return payload


def audit_paper_profile_mechanism_contract(
    profile_configs: Iterable[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> PaperMechanismContractAudit:
    """确认各 profile 的结果能力、消融和攻击机制完全一致。"""

    rows = list(profile_configs)
    contract_id = str(contract.get("contract_id") or "")
    required_claim_ids = tuple(
        str(item["claim_id"])
        for item in contract.get("required_claims") or []
    )
    required_capabilities = tuple(str(value) for value in contract.get("required_result_capabilities") or [])
    required_variants = tuple(str(value) for value in contract.get("formal_method_variants") or [])
    violations: list[str] = []
    for row in rows:
        profile = str(row.get("paper_result_level") or "unknown_profile")
        checks = {
            "formal_mechanism_contract_id": row.get("formal_mechanism_contract_id") == contract_id,
            "required_claim_ids": tuple(row.get("required_claim_ids") or []) == required_claim_ids,
            "required_result_capabilities": tuple(row.get("required_result_capabilities") or []) == required_capabilities,
            "required_internal_ablation_variants": tuple(row.get("required_internal_ablation_variants") or []) == required_variants,
            "require_claim3_full_support": row.get("require_claim3_full_support") is True,
            "shared_attack_protocol_id": row.get("shared_attack_protocol_id") == contract.get("shared_attack_protocol_id"),
            "complete_result_flags": all(row.get(field_name) is True for field_name in REQUIRED_COMPLETE_RESULT_FLAGS),
        }
        violations.extend(f"{profile}:{name}" for name, passed in checks.items() if not passed)
    return PaperMechanismContractAudit(
        passed=bool(rows) and not violations,
        contract_id=contract_id,
        audited_profile_count=len(rows),
        violations=tuple(violations),
    )
