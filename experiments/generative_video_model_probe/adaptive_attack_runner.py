"""paper profile 正式 non-runtime / adaptive attack 证据整理器。

该 runner 不再根据 runtime detection 分数合成 adaptive attack 结果。它只接收
已经由项目内脚本或人工可复现流程执行完成的正式 adaptive attack records,
然后进行 schema 归一、覆盖检查和门禁输出。缺少真实执行证据时, 本 runner
会写出阻断记录, 防止非正式执行记录或手工数字进入论文结论。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from main.attacks.video_runtime_attack_protocol import FULL_PAPER_NON_RUNTIME_ATTACK_PROTOCOLS
from main.protocol.flow_evidence_fields import with_flow_evidence_protocol_defaults
from main.protocol.record_writer import write_json, write_jsonl
from main.protocol.table_builder import write_csv


FORMAL_ADAPTIVE_ATTACK_EVIDENCE_LEVEL = "formal_adaptive_attack_execution"
FORMAL_ADAPTIVE_ATTACK_READY_STATUS = "formal_adaptive_attack_measured_ready"
FORMAL_ADAPTIVE_ATTACK_BLOCKED_STATUS = "formal_adaptive_attack_missing_execution"
FORMAL_ADAPTIVE_ATTACK_INPUT_FILENAMES = (
    "formal_adaptive_attack_execution_records.jsonl",
    "formal_adaptive_attack_input_records.jsonl",
)


def _adaptive_family_from_protocol(protocol_name: str) -> str:
    """把正式协议名称映射到论文表格中的攻击族。"""

    if protocol_name in {"wrong_sampler_replay_attack", "wrong_prompt_replay_attack"}:
        return "replay_signature_mismatch"
    if protocol_name in {"flow_time_grid_mismatch_attack"}:
        return "time_grid_or_scheduler_mismatch"
    if protocol_name in {"wrong_key_attack"}:
        return "key_mismatch_attack"
    if protocol_name in {"generative_recompression_or_regeneration_attack"}:
        return "generative_recompression_or_regeneration"
    if protocol_name in {"detector_probing_with_public_negatives"}:
        return "detector_threshold_probing"
    if protocol_name in {"watermark_removal_optimization_attack"}:
        return "watermark_removal_optimization"
    if protocol_name in {"watermark_spoofing_or_copy_attack"}:
        return "watermark_spoofing_or_copy"
    if protocol_name in {"collusion_multi_sample_attack"}:
        return "collusion_multi_sample"
    if protocol_name in {"adversarial_detector_evasion_attack"}:
        return "adversarial_detector_evasion"
    return "endpoint_preserving_path_attack"


ADAPTIVE_ATTACK_SPECS: tuple[dict[str, Any], ...] = tuple(
    {
        "adaptive_attack_name": protocol_name,
        "non_runtime_attack_protocol": protocol_name,
        "adaptive_attack_family": _adaptive_family_from_protocol(protocol_name),
    }
    for protocol_name in FULL_PAPER_NON_RUNTIME_ATTACK_PROTOCOLS
)


def _read_jsonl(path: Path) -> list[dict]:
    """读取 JSONL records, 文件不存在时返回空列表。"""

    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _safe_float(value: Any) -> float | None:
    """把可选数值字段转为 float。"""

    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _formal_input_records(run_root: Path) -> list[dict[str, Any]]:
    """读取正式 adaptive attack 执行输入记录。"""

    records: list[dict[str, Any]] = []
    for filename in FORMAL_ADAPTIVE_ATTACK_INPUT_FILENAMES:
        records.extend(_read_jsonl(run_root / "records" / filename))
    return records


def _formal_record_ready(record: dict[str, Any]) -> bool:
    """判断一条 adaptive attack 记录是否具备正式论文证据资格。"""

    return (
        record.get("metric_status") == "measured_formal"
        and record.get("adaptive_attack_evidence_level") == FORMAL_ADAPTIVE_ATTACK_EVIDENCE_LEVEL
        and record.get("adaptive_robustness_claim_allowed") is True
        and record.get("adaptive_attack_status") == "ready"
        and bool(record.get("non_runtime_attack_protocol") or record.get("adaptive_attack_name"))
    )


def _normalize_formal_record(record: dict[str, Any]) -> dict[str, Any]:
    """把外部正式输入记录归一为本项目统一 adaptive_attack_records schema。"""

    protocol = str(record.get("non_runtime_attack_protocol") or record.get("adaptive_attack_name") or "")
    family = str(record.get("adaptive_attack_family") or _adaptive_family_from_protocol(protocol))
    score = _safe_float(
        record.get("adaptive_attack_score")
        if record.get("adaptive_attack_score") is not None
        else record.get("adaptive_attack_tpr_at_target_fpr")
    )
    payload = {
        **record,
        "record_version": "formal_adaptive_attack_execution_v1",
        "non_runtime_attack_protocol": protocol,
        "adaptive_attack_name": str(record.get("adaptive_attack_name") or protocol),
        "adaptive_attack_family": family,
        "adaptive_attack_status": "ready",
        "metric_status": "measured_formal",
        "adaptive_attack_evidence_level": FORMAL_ADAPTIVE_ATTACK_EVIDENCE_LEVEL,
        "adaptive_robustness_claim_allowed": True,
        "adaptive_attack_score": score,
        "claim_support_status": FORMAL_ADAPTIVE_ATTACK_READY_STATUS,
    }
    return with_flow_evidence_protocol_defaults(
        payload,
        trajectory_source_level=FORMAL_ADAPTIVE_ATTACK_EVIDENCE_LEVEL,
        flow_state_admissibility_status="formal_adaptive_attack_execution_ready",
        claim_support_status=FORMAL_ADAPTIVE_ATTACK_READY_STATUS,
    )


def _missing_protocol_record(protocol_name: str) -> dict[str, Any]:
    """构造缺失正式执行证据时的阻断记录。"""

    return with_flow_evidence_protocol_defaults(
        {
            "record_version": "formal_adaptive_attack_execution_v1",
            "non_runtime_attack_protocol": protocol_name,
            "adaptive_attack_name": protocol_name,
            "adaptive_attack_family": _adaptive_family_from_protocol(protocol_name),
            "adaptive_attack_status": "missing",
            "metric_status": "missing",
            "adaptive_attack_evidence_level": "formal_adaptive_attack_execution_required",
            "adaptive_robustness_claim_allowed": False,
            "adaptive_attack_score": None,
            "adaptive_attack_failure_reason": "missing_formal_adaptive_attack_execution_record",
            "claim_support_status": FORMAL_ADAPTIVE_ATTACK_BLOCKED_STATUS,
        },
        trajectory_source_level="formal_adaptive_attack_execution_missing",
        flow_state_admissibility_status="formal_adaptive_attack_execution_missing",
        claim_support_status=FORMAL_ADAPTIVE_ATTACK_BLOCKED_STATUS,
    )


def build_adaptive_attack_records(run_root: str | Path) -> list[dict[str, Any]]:
    """构建正式 adaptive attack records。

    该函数只归一已有正式执行记录。没有真实执行记录的协议会被显式写为 missing,
    以便 paper gate 明确阻断。
    """

    run_root = Path(run_root)
    input_records = _formal_input_records(run_root)
    formal_by_protocol: dict[str, list[dict[str, Any]]] = {}
    for record in input_records:
        protocol = str(record.get("non_runtime_attack_protocol") or record.get("adaptive_attack_name") or "")
        if protocol and _formal_record_ready(record):
            formal_by_protocol.setdefault(protocol, []).append(_normalize_formal_record(record))

    records: list[dict[str, Any]] = []
    for protocol in FULL_PAPER_NON_RUNTIME_ATTACK_PROTOCOLS:
        protocol_records = formal_by_protocol.get(protocol, [])
        if protocol_records:
            records.extend(protocol_records)
        else:
            records.append(_missing_protocol_record(protocol))
    return records


def audit_adaptive_attack_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """审计正式 adaptive attack records 覆盖情况。"""

    required_protocols = {str(item) for item in FULL_PAPER_NON_RUNTIME_ATTACK_PROTOCOLS}
    formal_records = [record for record in records if _formal_record_ready(record)]
    observed_protocols = {
        str(record.get("non_runtime_attack_protocol") or record.get("adaptive_attack_name") or "")
        for record in formal_records
        if record.get("non_runtime_attack_protocol") or record.get("adaptive_attack_name")
    }
    missing_non_runtime_protocols = sorted(required_protocols - observed_protocols)
    non_formal_records = [record for record in records if not _formal_record_ready(record)]
    scores = [
        value
        for value in (_safe_float(record.get("adaptive_attack_score")) for record in formal_records)
        if value is not None
    ]
    decision = "PASS" if records and not missing_non_runtime_protocols and not non_formal_records else "FAIL"
    return {
        "stage_id": "formal_adaptive_attack_execution_gate",
        "adaptive_attack_decision": decision,
        "claim_support_status": "formal_adaptive_attack_execution_ready"
        if decision == "PASS"
        else "formal_adaptive_attack_execution_blocked",
        "adaptive_attack_record_count": len(records),
        "formal_adaptive_attack_record_count": len(formal_records),
        "adaptive_attack_name_count": len({
            str(record.get("adaptive_attack_name"))
            for record in formal_records
            if record.get("adaptive_attack_name")
        }),
        "adaptive_attack_family_count": len({
            str(record.get("adaptive_attack_family"))
            for record in formal_records
            if record.get("adaptive_attack_family")
        }),
        "non_runtime_attack_protocol_count": len(observed_protocols),
        "required_non_runtime_attack_protocols": sorted(required_protocols),
        "observed_non_runtime_attack_protocols": sorted(observed_protocols),
        "missing_non_runtime_attack_protocols": missing_non_runtime_protocols,
        "adaptive_attack_missing_names": missing_non_runtime_protocols,
        "adaptive_attack_score_mean": round(mean(scores), 6) if scores else None,
        "adaptive_robustness_claim_allowed": decision == "PASS",
        "adaptive_attack_evidence_level": FORMAL_ADAPTIVE_ATTACK_EVIDENCE_LEVEL,
        "adaptive_attack_non_formal_record_count": len(non_formal_records),
    }


def run_adaptive_attack_formal_protocol(run_root: str | Path) -> dict[str, Any]:
    """写出正式 adaptive attack records、table、decision 和 report。"""

    run_root = Path(run_root)
    records = build_adaptive_attack_records(run_root)
    audit = audit_adaptive_attack_records(records)
    write_jsonl(run_root / "records" / "adaptive_attack_records.jsonl", records)
    write_csv(run_root / "tables" / "adaptive_attack_table.csv", records)
    write_json(run_root / "artifacts" / "adaptive_attack_decision.json", audit)
    report = (
        "# Formal Adaptive Attack Execution Report\n\n"
        "该报告只汇总正式 adaptive / non-runtime attack 执行记录。缺少真实执行证据时,"
        "对应协议被写为 missing, 不允许以 runtime detection 分数合成替代结果。\n\n"
        f"- adaptive_attack_decision: {audit['adaptive_attack_decision']}\n"
        f"- adaptive_attack_record_count: {audit['adaptive_attack_record_count']}\n"
        f"- formal_adaptive_attack_record_count: {audit['formal_adaptive_attack_record_count']}\n"
        f"- missing_non_runtime_attack_protocols: {', '.join(audit['missing_non_runtime_attack_protocols']) if audit['missing_non_runtime_attack_protocols'] else 'none'}\n"
        f"- adaptive_robustness_claim_allowed: {str(audit['adaptive_robustness_claim_allowed']).lower()}\n"
        f"- claim_support_status: {audit['claim_support_status']}\n"
    )
    report_path = run_root / "reports" / "adaptive_attack_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="整理 paper profile 正式 adaptive attack records。")
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    payload = run_adaptive_attack_formal_protocol(args.run_root)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
