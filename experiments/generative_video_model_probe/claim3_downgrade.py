"""Claim-3 降级门禁 runner。

该模块只负责把 Claim-3 的证据边界写成 governed records。它不会伪造
replay/sketch 结果, 也不会把降级路径解释为强 replay verification 已完成。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from main.protocol.flow_evidence_fields import with_flow_evidence_protocol_defaults
from main.protocol.record_writer import write_json, write_jsonl
from main.protocol.table_builder import write_csv


def _read_json(path: Path) -> dict:
    """读取 JSON artifact, 文件缺失时返回空对象。"""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_jsonl(path: Path) -> list[dict]:
    """读取 JSONL records, 文件缺失时返回空列表。"""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _decision_pass(decision: dict, *field_names: str) -> bool:
    """检查任一指定决策字段是否为 PASS。"""
    return any(decision.get(field_name) == "PASS" for field_name in field_names)


def _records_ready(records: list[dict], status_field: str, ready_values: set[str] | None = None) -> bool:
    """判断记录是否非空且存在 ready 状态。

    该函数属于通用工程写法。不同 replay/sketch 子任务的状态字段尚未完全统一,
    因此这里允许调用方传入一组可接受状态值。没有记录时必须返回 false, 不能把
    缺失记录解释为通过。
    """
    if not records:
        return False
    if ready_values is None:
        ready_values = {"ready", "pass", "PASS"}
    return any(str(record.get(status_field)) in ready_values for record in records)


def build_claim3_downgrade_audit(run_root: str | Path) -> dict[str, Any]:
    """构建 Claim-3 降级门禁审计结果。

    该函数只读取已经落盘的 replay/sketch 相关 governed artifacts。若完整
    replay/sketch gate 未通过, 它会显式设置 `claim3_downgraded = true`,
    表示后续 validation-scale 可以继续, 但 Claim-3 不能作为强 supported claim。
    """
    run_root = Path(run_root)
    replay_decision = _read_json(run_root / "artifacts" / "replay_and_sketch_gate_decision.json")
    trajectory_sketch_records = _read_jsonl(run_root / "records" / "trajectory_sketch_verification_records.jsonl")
    replay_uncertainty_records = _read_jsonl(run_root / "records" / "replay_uncertainty_records.jsonl")
    wrong_sampler_records = _read_jsonl(run_root / "records" / "wrong_sampler_replay_records.jsonl")
    wrong_prompt_records = _read_jsonl(run_root / "records" / "wrong_prompt_replay_records.jsonl")

    replay_gate_passed = _decision_pass(replay_decision, "replay_and_sketch_gate_decision")
    authenticated_trajectory_sketch_ready = _records_ready(
        trajectory_sketch_records,
        "trajectory_sketch_verification_status",
        {"pass", "PASS", "verified", "ready"},
    )
    replay_uncertainty_records_ready = bool(replay_uncertainty_records)
    wrong_sampler_replay_records_ready = bool(wrong_sampler_records)
    wrong_prompt_replay_records_ready = bool(wrong_prompt_records)

    missing_replay_requirements = []
    if not authenticated_trajectory_sketch_ready:
        missing_replay_requirements.append("authenticated_trajectory_sketch")
    if not replay_uncertainty_records_ready:
        missing_replay_requirements.append("replay_uncertainty_records")
    if not wrong_sampler_replay_records_ready:
        missing_replay_requirements.append("wrong_sampler_replay_records")
    if not wrong_prompt_replay_records_ready:
        missing_replay_requirements.append("wrong_prompt_replay_records")
    if not replay_gate_passed:
        missing_replay_requirements.append("replay_and_sketch_gate_decision_pass")

    claim3_downgraded = not replay_gate_passed
    claim_support_status = (
        "claim3_full_support_available"
        if replay_gate_passed
        else "claim3_downgraded_validation_scale_only"
    )
    replay_or_sketch_status = (
        "replay_and_sketch_gate_passed"
        if replay_gate_passed
        else "claim3_explicitly_downgraded"
    )

    return {
        "stage_id": "claim3_downgrade_gate",
        "run_root": str(run_root),
        "claim3_downgrade_decision": "PASS",
        "claim3_downgraded": claim3_downgraded,
        "claim3_original_scope": "robust_replay_verification",
        "claim3_allowed_scope": "robust_replay_verification"
        if replay_gate_passed
        else "owner_side_audit_or_exploratory_replay_analysis",
        "claim3_downgrade_reason": "replay_and_sketch_gate_passed"
        if replay_gate_passed
        else "replay_and_sketch_gate_not_yet_implemented_or_not_passed",
        "claim3_full_support_allowed": replay_gate_passed,
        "replay_or_sketch_status": replay_or_sketch_status,
        "replay_and_sketch_gate_decision": replay_decision.get("replay_and_sketch_gate_decision"),
        "authenticated_trajectory_sketch_status": "ready"
        if authenticated_trajectory_sketch_ready
        else "not_ready",
        "trajectory_sketch_verification_status": "pass"
        if authenticated_trajectory_sketch_ready
        else "not_ready",
        "replay_uncertainty_records_ready": replay_uncertainty_records_ready,
        "wrong_sampler_replay_records_ready": wrong_sampler_replay_records_ready,
        "wrong_prompt_replay_records_ready": wrong_prompt_replay_records_ready,
        "claim3_missing_replay_requirement_count": len(missing_replay_requirements),
        "claim3_missing_replay_requirements": missing_replay_requirements,
        "claim_support_status": claim_support_status,
    }


def build_claim3_downgrade_records(run_root: str | Path) -> list[dict[str, Any]]:
    """构建 Claim-3 降级 record。

    该 record 是 validation-scale 的治理边界记录。它可以让 gate 继续运行,
    但不会支持 robust replay verification 的正向论文 claim。
    """
    audit = build_claim3_downgrade_audit(run_root)
    record = with_flow_evidence_protocol_defaults(
        {
            "record_version": "claim3_downgrade_gate_v1",
            **audit,
        },
        trajectory_source_level="claim3_downgrade_governance_record",
        flow_state_admissibility_status="claim3_downgraded"
        if audit["claim3_downgraded"]
        else "replay_and_sketch_gate_passed",
        claim_support_status=audit["claim_support_status"],
    )
    return [record]


def write_claim3_downgrade_outputs(run_root: str | Path) -> dict[str, Any]:
    """写出 Claim-3 降级 records、table、decision 和 report。"""
    run_root = Path(run_root)
    records = build_claim3_downgrade_records(run_root)
    audit = build_claim3_downgrade_audit(run_root)
    write_jsonl(run_root / "records" / "claim3_downgrade_records.jsonl", records)
    write_csv(run_root / "tables" / "claim3_downgrade_table.csv", records)
    write_json(run_root / "artifacts" / "claim3_downgrade_decision.json", audit)
    report = (
        "# Claim-3 Downgrade Gate Report\n\n"
        "该报告只声明 Claim-3 的当前证据边界。若 replay/sketch gate 未通过, "
        "本报告允许 validation-scale 继续运行, 但不允许把 robust replay verification "
        "写成强 supported claim。\n\n"
        f"- claim3_downgrade_decision: {audit['claim3_downgrade_decision']}\n"
        f"- claim3_downgraded: {str(audit['claim3_downgraded']).lower()}\n"
        f"- claim3_full_support_allowed: {str(audit['claim3_full_support_allowed']).lower()}\n"
        f"- replay_or_sketch_status: {audit['replay_or_sketch_status']}\n"
        f"- claim3_missing_replay_requirement_count: {audit['claim3_missing_replay_requirement_count']}\n"
        f"- claim_support_status: {audit['claim_support_status']}\n"
    )
    report_path = run_root / "reports" / "claim3_downgrade_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="写出 Claim-3 downgrade gate 产物。")
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    payload = write_claim3_downgrade_outputs(args.run_root)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
