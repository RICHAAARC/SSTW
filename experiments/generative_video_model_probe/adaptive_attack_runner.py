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

from evaluation.attacks.video_runtime_attack_protocol import FULL_PAPER_NON_RUNTIME_ATTACK_PROTOCOLS
from evaluation.attacks.adaptive_video_optimizer import (
    ADAPTIVE_QUERY_BUDGET_CHECKPOINT_PROTOCOL,
    ADAPTIVE_SOURCE_CLUSTER_SELECTION_PROTOCOL,
)
from evaluation.protocol.flow_evidence_fields import with_flow_evidence_protocol_defaults
from evaluation.protocol.record_writer import write_json, write_jsonl
from evaluation.protocol.table_builder import write_csv


FORMAL_ADAPTIVE_ATTACK_EVIDENCE_LEVEL = "formal_adaptive_attack_execution"
FORMAL_ADAPTIVE_ATTACK_EXECUTION_GRANULARITY = (
    "per_video_frozen_flow_detector_adaptive_execution"
)
FORMAL_ADAPTIVE_ATTACK_READY_STATUS = "formal_adaptive_attack_measured_ready"
FORMAL_ADAPTIVE_ATTACK_BLOCKED_STATUS = "formal_adaptive_attack_missing_execution"
FORMAL_ADAPTIVE_ATTACK_INPUT_FILENAMES = (
    "formal_adaptive_attack_execution_records.jsonl",
)
PER_VIDEO_SEARCH_PROTOCOLS = {
    "generative_recompression_or_regeneration_attack",
    "endpoint_preserving_path_perturbation_attack",
    "detector_probing_with_public_negatives",
    "watermark_removal_optimization_attack",
    "adversarial_detector_evasion_attack",
}
CROSS_VIDEO_PROTOCOLS = {
    "watermark_spoofing_or_copy_attack",
    "collusion_multi_sample_attack",
}
COLLUSION_PROTOCOL = "collusion_multi_sample_attack"
SPOOF_PROTOCOL = "watermark_spoofing_or_copy_attack"


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


def _read_json(path: Path) -> dict[str, Any]:
    """读取正式 decision artifact, 文件缺失时返回空对象并触发 fail-closed。"""

    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return payload if isinstance(payload, dict) else {}


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

    protocol = str(record.get("non_runtime_attack_protocol") or record.get("adaptive_attack_name") or "")
    base_ready = (
        record.get("metric_status") == "measured_formal"
        and record.get("adaptive_attack_evidence_level") == FORMAL_ADAPTIVE_ATTACK_EVIDENCE_LEVEL
        and record.get("adaptive_attack_execution_granularity")
        == FORMAL_ADAPTIVE_ATTACK_EXECUTION_GRANULARITY
        and record.get("adaptive_robustness_claim_allowed") is True
        and record.get("adaptive_attack_status") == "ready"
        and int(record.get("adaptive_attack_query_count") or 0) > 0
        and bool(record.get("statistical_cluster_id"))
        and bool(record.get("adaptive_attack_source_statistical_cluster_id"))
        and int(
            record.get(
                "minimum_adaptive_attack_source_video_cluster_count_per_protocol"
            )
            or 0
        )
        > 0
        and record.get("adaptive_attack_source_cluster_selection_protocol")
        == ADAPTIVE_SOURCE_CLUSTER_SELECTION_PROTOCOL
        and record.get("test_time_threshold_update_blocked") is True
        and bool(protocol)
    )
    if not base_ready:
        return False
    if protocol in PER_VIDEO_SEARCH_PROTOCOLS:
        query_count = int(record.get("adaptive_attack_query_count") or 0)
        query_budget = int(record.get("adaptive_attack_query_budget") or 0)
        checkpoints = record.get("adaptive_attack_query_budget_checkpoints")
        checkpoint_records = record.get(
            "adaptive_attack_query_budget_checkpoint_records"
        )
        return (
            query_count == query_budget
            and query_budget >= 3
            and bool(record.get("adaptive_attack_output_video_sha256"))
            and isinstance(record.get("adaptive_attack_candidate_records"), list)
            and len(record["adaptive_attack_candidate_records"])
            == query_count
            and record.get("adaptive_attack_query_budget_checkpoint_protocol")
            == ADAPTIVE_QUERY_BUDGET_CHECKPOINT_PROTOCOL
            and isinstance(checkpoints, list)
            and checkpoints == sorted(set(checkpoints))
            and bool(checkpoints)
            and int(checkpoints[-1]) == query_budget
            and isinstance(checkpoint_records, list)
            and len(checkpoint_records) == len(checkpoints)
            and record.get("adaptive_attack_query_accounting_protocol")
            == "all_target_and_public_negative_frozen_detector_calls"
            and int(
                record.get("adaptive_attack_total_detector_query_count") or 0
            )
            == query_count
            + int(record.get("adaptive_attack_public_negative_probe_count") or 0)
            and all(
                int(
                    row.get("adaptive_attack_query_budget_checkpoint") or 0
                )
                == int(checkpoint)
                and int(
                    row.get("adaptive_attack_checkpoint_observed_query_count")
                    or 0
                )
                == int(checkpoint)
                and row.get("adaptive_attack_checkpoint_has_admissible_candidate")
                is True
                and bool(
                    row.get("adaptive_attack_checkpoint_output_video_sha256")
                )
                for checkpoint, row in zip(checkpoints, checkpoint_records)
            )
        )
    if protocol in CROSS_VIDEO_PROTOCOLS:
        return bool(record.get("adaptive_attack_output_video_sha256"))
    return record.get("adaptive_attack_execution_backend") == "per_video_precomputed_key_independent_replay_control"


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
        "record_version": str(record.get("record_version") or "formal_per_video_adaptive_attack_v1"),
        "non_runtime_attack_protocol": protocol,
        "adaptive_attack_name": str(record.get("adaptive_attack_name") or protocol),
        "adaptive_attack_family": family,
        "adaptive_attack_status": "ready",
        "metric_status": "measured_formal",
        "adaptive_attack_evidence_level": FORMAL_ADAPTIVE_ATTACK_EVIDENCE_LEVEL,
        "adaptive_attack_execution_granularity": (
            FORMAL_ADAPTIVE_ATTACK_EXECUTION_GRANULARITY
        ),
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


def audit_adaptive_attack_records(
    records: list[dict[str, Any]],
    *,
    execution_decision: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """按 source-video cluster 审计正式覆盖、查询曲线与伪重复风险。"""

    required_protocols = {str(item) for item in FULL_PAPER_NON_RUNTIME_ATTACK_PROTOCOLS}
    formal_records = [record for record in records if _formal_record_ready(record)]
    observed_protocols = {
        str(record.get("non_runtime_attack_protocol") or "")
        for record in formal_records
        if record.get("non_runtime_attack_protocol")
    }
    missing_non_runtime_protocols = sorted(required_protocols - observed_protocols)
    non_formal_records = [record for record in records if not _formal_record_ready(record)]
    scores = [
        value
        for value in (
            _safe_float(record.get("adaptive_attack_score"))
            for record in formal_records
        )
        if value is not None
    ]

    decision_artifact = execution_decision or {}
    minimum_cluster_count = int(
        decision_artifact.get(
            "minimum_adaptive_attack_source_video_cluster_count_per_protocol"
        )
        or 0
    )
    minimum_spoof_cluster_count = int(
        decision_artifact.get(
            "minimum_adaptive_spoof_source_video_cluster_count"
        )
        or 0
    )
    execution_decision_ready = bool(
        decision_artifact.get("formal_adaptive_attack_execution_decision") == "PASS"
        and decision_artifact.get("adaptive_attack_source_cluster_coverage_decision")
        == "PASS"
        and decision_artifact.get(
            "adaptive_attack_independent_unit_uniqueness_decision"
        )
        == "PASS"
        and decision_artifact.get(
            "adaptive_attack_query_budget_checkpoint_coverage_decision"
        )
        == "PASS"
        and minimum_cluster_count > 0
        and minimum_spoof_cluster_count >= minimum_cluster_count
        and decision_artifact.get("adaptive_watermark_retention_decision")
        == "PASS"
        and decision_artifact.get("adaptive_spoof_rejection_decision") == "PASS"
        and decision_artifact.get("adaptive_replay_control_rejection_decision")
        == "PASS"
        and decision_artifact.get("adaptive_robustness_claim_allowed") is True
    )

    record_minima = {
        int(
            row.get(
                "minimum_adaptive_attack_source_video_cluster_count_per_protocol"
            )
            or 0
        )
        for row in formal_records
    }
    query_budgets = {
        int(row.get("adaptive_attack_query_budget") or 0)
        for row in formal_records
    }
    checkpoint_sets = {
        tuple(int(value) for value in row.get("adaptive_attack_query_budget_checkpoints", []))
        for row in formal_records
    }
    mechanism_invariant_ready = bool(
        record_minima == {minimum_cluster_count}
        and len(query_budgets) == 1
        and min(query_budgets, default=0) >= 3
        and len(checkpoint_sets) == 1
        and next(iter(checkpoint_sets), ())
        and next(iter(checkpoint_sets))[-1] == next(iter(query_budgets))
    )

    per_protocol_cluster_counts: dict[str, int] = {}
    per_protocol_source_sets: dict[str, set[str]] = {}
    duplicate_independent_units: list[str] = []
    incomplete_protocols: list[str] = []
    for protocol in sorted(required_protocols - {COLLUSION_PROTOCOL}):
        scoped = [
            row for row in formal_records
            if row.get("non_runtime_attack_protocol") == protocol
        ]
        source_ids = [
            str(row.get("adaptive_attack_source_statistical_cluster_id") or "")
            for row in scoped
        ]
        source_set = set(source_ids)
        per_protocol_source_sets[protocol] = source_set
        per_protocol_cluster_counts[protocol] = len(source_set)
        duplicate_independent_units.extend(
            f"{protocol}::{cluster_id}"
            for cluster_id in sorted(source_set)
            if source_ids.count(cluster_id) != 1
        )
        expected_count = (
            minimum_spoof_cluster_count
            if protocol == SPOOF_PROTOCOL
            else minimum_cluster_count
        )
        if not (
            len(scoped) == expected_count
            and len(source_ids) == len(source_set)
            and len(source_set) == expected_count
        ):
            incomplete_protocols.append(protocol)

    retention_source_sets = {
        protocol: source_set
        for protocol, source_set in per_protocol_source_sets.items()
        if protocol != SPOOF_PROTOCOL
    }
    reference_source_clusters = (
        next(iter(retention_source_sets.values()))
        if retention_source_sets
        else set()
    )
    for protocol, source_set in retention_source_sets.items():
        if source_set != reference_source_clusters and protocol not in incomplete_protocols:
            incomplete_protocols.append(protocol)
    spoof_source_clusters = per_protocol_source_sets.get(SPOOF_PROTOCOL, set())
    if (
        len(spoof_source_clusters) != minimum_spoof_cluster_count
        or not reference_source_clusters <= spoof_source_clusters
    ) and SPOOF_PROTOCOL not in incomplete_protocols:
        incomplete_protocols.append(SPOOF_PROTOCOL)

    collusion_rows = [
        row for row in formal_records
        if row.get("non_runtime_attack_protocol") == COLLUSION_PROTOCOL
    ]
    collusion_pair_ids = [
        str(row.get("statistical_cluster_id") or "") for row in collusion_rows
    ]
    collusion_member_ids = [
        str(member)
        for row in collusion_rows
        for member in row.get("adaptive_attack_member_statistical_cluster_ids", [])
    ]
    duplicate_independent_units.extend(
        f"{COLLUSION_PROTOCOL}::{pair_id}"
        for pair_id in sorted(set(collusion_pair_ids))
        if collusion_pair_ids.count(pair_id) != 1
    )
    collusion_ready = bool(
        minimum_cluster_count % 2 == 0
        and len(collusion_rows) == minimum_cluster_count // 2
        and len(collusion_pair_ids) == len(set(collusion_pair_ids))
        and len(collusion_member_ids) == minimum_cluster_count
        and len(collusion_member_ids) == len(set(collusion_member_ids))
        and set(collusion_member_ids) == reference_source_clusters
        and all(
            row.get("statistical_independent_unit") == "disjoint_source_video_pair"
            for row in collusion_rows
        )
    )
    per_protocol_cluster_counts[COLLUSION_PROTOCOL] = len(set(collusion_pair_ids))
    if not collusion_ready:
        incomplete_protocols.append(COLLUSION_PROTOCOL)

    checkpoint_ready = bool(
        mechanism_invariant_ready
        and all(
            isinstance(row.get("adaptive_attack_query_budget_checkpoint_records"), list)
            and len(row["adaptive_attack_query_budget_checkpoint_records"])
            == len(row["adaptive_attack_query_budget_checkpoints"])
            for row in formal_records
            if row.get("non_runtime_attack_protocol") in PER_VIDEO_SEARCH_PROTOCOLS
        )
    )
    pseudoreplication_ready = not duplicate_independent_units
    cluster_coverage_ready = bool(
        len(reference_source_clusters) == minimum_cluster_count
        and not incomplete_protocols
    )
    decision = "PASS" if (
        records
        and execution_decision_ready
        and not missing_non_runtime_protocols
        and not non_formal_records
        and mechanism_invariant_ready
        and cluster_coverage_ready
        and pseudoreplication_ready
        and checkpoint_ready
    ) else "FAIL"
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
        "adaptive_attack_independent_video_count": len(reference_source_clusters),
        "minimum_adaptive_attack_source_video_cluster_count_per_protocol": (
            minimum_cluster_count
        ),
        "minimum_adaptive_spoof_source_video_cluster_count": (
            minimum_spoof_cluster_count
        ),
        "adaptive_attack_spoof_source_video_cluster_count": len(
            spoof_source_clusters
        ),
        "adaptive_attack_protocol_independent_cluster_counts": (
            per_protocol_cluster_counts
        ),
        "adaptive_attack_collusion_independent_pair_count": len(
            set(collusion_pair_ids)
        ),
        "adaptive_attack_incomplete_cluster_protocols": sorted(
            set(incomplete_protocols)
        ),
        "adaptive_attack_duplicate_independent_units": sorted(
            set(duplicate_independent_units)
        ),
        "adaptive_attack_source_cluster_coverage_decision": (
            "PASS" if cluster_coverage_ready else "FAIL"
        ),
        "adaptive_attack_pseudoreplication_decision": (
            "PASS" if pseudoreplication_ready else "FAIL"
        ),
        "adaptive_attack_query_budget_checkpoint_decision": (
            "PASS" if checkpoint_ready else "FAIL"
        ),
        "formal_adaptive_attack_execution_artifact_decision": (
            "PASS" if execution_decision_ready else "FAIL"
        ),
        "adaptive_attack_mechanism_invariance_decision": (
            "PASS" if mechanism_invariant_ready else "FAIL"
        ),
        "adaptive_attack_incomplete_video_clusters": sorted(
            set(incomplete_protocols)
        ),
        "per_video_adaptive_attack_optimization": True,
    }

def run_adaptive_attack_formal_protocol(run_root: str | Path) -> dict[str, Any]:
    """写出正式 adaptive attack records、table、decision 和 report。"""

    run_root = Path(run_root)
    records = build_adaptive_attack_records(run_root)
    execution_decision = _read_json(
        run_root / "artifacts" / "formal_adaptive_attack_execution_decision.json"
    )
    audit = audit_adaptive_attack_records(
        records,
        execution_decision=execution_decision,
    )
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
        f"- adaptive_attack_independent_video_count: {audit['adaptive_attack_independent_video_count']}\n"
        f"- adaptive_attack_source_cluster_coverage_decision: {audit['adaptive_attack_source_cluster_coverage_decision']}\n"
        f"- adaptive_attack_pseudoreplication_decision: {audit['adaptive_attack_pseudoreplication_decision']}\n"
        f"- adaptive_attack_query_budget_checkpoint_decision: {audit['adaptive_attack_query_budget_checkpoint_decision']}\n"
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
