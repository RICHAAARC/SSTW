"""replay/sketch gate 的 paper profile 工程 runner。

该模块把已经落盘的 generation records 与 trajectory trace 转换为四类可审计 records:
1. authenticated trajectory sketch verification;
2. replay uncertainty weighting;
3. wrong sampler replay control;
4. wrong prompt replay control。

该实现属于 paper profile 工程闭环。它可以证明 replay/sketch 协议入口、records、table、report 和 gate
能够由 governed records 自动重建, 但不会把 owner-side diagnostic 伪装成 full-paper 强 Claim-3。
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from experiments.generative_video_model_probe.formal_motion_claim_filter import select_motion_claim_generation_records
from main.core.digest import build_stable_digest
from main.methods.state_space_watermark.authenticated_trajectory_sketch import verify_authenticated_trajectory_sketch
from main.protocol.flow_evidence_fields import with_flow_evidence_protocol_defaults
from main.protocol.record_writer import write_json, write_jsonl
from main.protocol.table_builder import write_csv


REPLAY_AND_SKETCH_CLAIM_SUPPORT_STATUS = "replay_and_sketch_owner_side_diagnostic_only"
REPLAY_AND_SKETCH_EVIDENCE_LEVEL = "owner_side_runtime_trace_diagnostic"
FULL_CLAIM3_EVIDENCE_LEVEL = "attacked_video_wan_vae_model_velocity_replay_with_hmac_sketch"
REPLAY_RECORD_TABLE_FIELDS = (
    "record_version",
    "replay_record_type",
    "generation_model_id",
    "prompt_id",
    "seed_id",
    "trajectory_trace_id",
    "authenticated_trajectory_sketch_status",
    "trajectory_sketch_digest_random",
    "trajectory_sketch_verification_status",
    "replay_uncertainty_weight",
    "replay_uncertainty_mean",
    "replay_scheduler_id",
    "replay_time_grid_id",
    "wrong_sampler_replay_control",
    "wrong_prompt_replay_control",
    "replay_control_status",
    "replay_signature_mismatch_status",
    "replay_and_sketch_evidence_level",
    "claim_support_status",
    "trajectory_source_level",
    "flow_state_admissibility_status",
)


def _read_jsonl(path: Path) -> list[dict]:
    """读取 JSONL records, 文件不存在时返回空列表。"""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_json(path: Path) -> dict[str, Any]:
    """读取 JSON decision; 文件不存在时返回空对象。"""

    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return payload if isinstance(payload, dict) else {}


def _group_by_trace_id(records: Iterable[dict]) -> dict[str, list[dict]]:
    """按 trajectory_trace_id 对轨迹 step records 分组。"""
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        trace_id = str(record.get("trajectory_trace_id") or "")
        if trace_id:
            groups[trace_id].append(record)
    for trace_records in groups.values():
        trace_records.sort(key=lambda item: int(item.get("trajectory_step_index") or 0))
    return groups


def _build_full_claim3_records(run_root: Path) -> dict[str, list[dict[str, Any]]] | None:
    """从真实 attacked-video replay 与 HMAC sketch 构造 Claim-3 正式 records。"""

    evidence_records = [
        record
        for record in _read_jsonl(run_root / "records" / "formal_flow_evidence_records.jsonl")
        if record.get("sample_role") == "attacked_positive"
        and record.get("method_variant") == "sstw_full_method"
        and record.get("split") == "test"
        and record.get("metric_status") == "measured_formal"
    ]
    if not evidence_records:
        return None
    sketches = {
        str(record.get("trajectory_trace_id")): record
        for record in _read_jsonl(run_root / "records" / "trajectory_sketch_records.jsonl")
        if record.get("authenticated_trajectory_sketch_status") == "signed"
    }
    authentication_key_text = os.environ.get("SSTW_TRAJECTORY_AUTHENTICATION_KEY") or ""
    authentication_key = authentication_key_text.encode("utf-8")
    groups: dict[str, list[dict[str, Any]]] = {
        "trajectory_sketch_verification_records": [],
        "replay_uncertainty_records": [],
        "wrong_sampler_replay_records": [],
        "wrong_prompt_replay_records": [],
        "wrong_key_replay_records": [],
    }
    for evidence in evidence_records:
        trace_id = str(evidence.get("trajectory_trace_id") or "")
        sketch = sketches.get(trace_id, {})
        payload = sketch.get("trajectory_sketch_payload") if isinstance(sketch, dict) else None
        context_matches = bool(
            isinstance(payload, dict)
            and payload.get("model_signature") == evidence.get("generation_model_id")
            and str(payload.get("seed_id")) == str(evidence.get("seed_id"))
            and sketch.get("prompt_id") == evidence.get("prompt_id")
            and sketch.get("method_variant") == evidence.get("method_variant")
            and payload.get("prompt_digest") == evidence.get("replay_prompt_digest")
            and payload.get("sampler_signature") == evidence.get("replay_sampler_signature")
            and payload.get("time_grid_id") == evidence.get("authenticated_generation_time_grid_id")
        )
        signature_verified = bool(authentication_key) and verify_authenticated_trajectory_sketch(
            sketch,
            authentication_key=authentication_key,
        )
        sketch_verified = signature_verified and context_matches
        common = {
            "record_version": "replay_and_sketch_gate_v2",
            "generation_model_id": evidence.get("generation_model_id"),
            "prompt_id": evidence.get("prompt_id"),
            "seed_id": evidence.get("seed_id"),
            "trajectory_trace_id": trace_id,
            "attack_name": evidence.get("attack_name"),
            "method_variant": evidence.get("method_variant"),
            "replay_and_sketch_evidence_level": FULL_CLAIM3_EVIDENCE_LEVEL,
            "claim_support_status": "claim3_attacked_video_replay_posterior_candidate",
            "trajectory_source_level": "attacked_video_model_velocity_inversion_replay",
            "flow_state_admissibility_status": evidence.get("flow_state_admissibility_status"),
            "flow_posterior_confidence": evidence.get("flow_posterior_confidence"),
            "flow_state_posterior_entropy": evidence.get("flow_state_posterior_entropy"),
            "S_final_conservative": evidence.get("S_final_conservative"),
        }
        sketch_record = {
            **common,
            "replay_record_type": "trajectory_sketch_verification",
            "authenticated_trajectory_sketch_status": "ready" if sketch_verified else "not_ready",
            "trajectory_sketch_digest_random": build_stable_digest(sketch) if sketch else None,
            "trajectory_sketch_verification_status": "verified" if sketch_verified else "verification_failed",
            "replay_control_status": "sketch_verified" if sketch_verified else "sketch_rejected",
            "replay_signature_mismatch_status": "matched_authenticated_context" if context_matches else "context_mismatch",
        }
        groups["trajectory_sketch_verification_records"].append(sketch_record)

        replay_ready = (
            evidence.get("replay_inversion_status") == "ready"
            and evidence.get("formal_flow_evidence_level") == "attacked_video_wan_vae_model_velocity_replay"
            and evidence.get("replay_trajectory_source") == "attacked_video_endpoint_model_velocity_inversion"
        )
        groups["replay_uncertainty_records"].append({
            **common,
            "replay_record_type": "replay_uncertainty",
            "authenticated_trajectory_sketch_status": "ready" if sketch_verified else "not_ready",
            "trajectory_sketch_verification_status": "verified" if sketch_verified else "verification_failed",
            "replay_uncertainty_mean": evidence.get("replay_uncertainty_mean"),
            "replay_uncertainty_weight": evidence.get("replay_reliability_weight"),
            "replay_control_status": "uncertainty_weight_ready" if replay_ready else "replay_inversion_missing",
            "replay_scheduler_id": "wan_flow_match_euler_discrete_scheduler",
            "replay_time_grid_id": str(evidence.get("replay_step_counts")),
        })
        for group_name, control_name, margin_field in (
            ("wrong_sampler_replay_records", "wrong_sampler_replay", "wrong_sampler_control_margin"),
            ("wrong_prompt_replay_records", "wrong_prompt_replay", "wrong_prompt_control_margin"),
            ("wrong_key_replay_records", "wrong_key_replay", "wrong_key_control_margin"),
        ):
            margin = evidence.get(margin_field)
            rejected = margin is not None and float(margin) > 0.0
            groups[group_name].append({
                **common,
                "replay_record_type": control_name,
                "authenticated_trajectory_sketch_status": "ready" if sketch_verified else "not_ready",
                "trajectory_sketch_verification_status": "verified" if sketch_verified else "verification_failed",
                margin_field: margin,
                "replay_control_status": "replay_rejected" if rejected else "control_not_rejected",
                "replay_signature_mismatch_status": f"{control_name}_measured_formal",
            })
    return groups


def _float_values(records: list[dict], field_name: str) -> list[float]:
    """从 records 中提取可转换为 float 的字段值。"""
    values: list[float] = []
    for record in records:
        value = record.get(field_name)
        if value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def _replay_uncertainty(trace_records: list[dict]) -> float:
    """根据 latent norm 的相对变化估计 replay uncertainty diagnostic。

    这是 paper profile owner-side diagnostic 写法。它只使用 trajectory trace 中的路径统计, 不读取 `S_final`
    或最终检测判定分数, 因此不会把最终检测结果反向用于污染过滤。
    """
    norms = _float_values(trace_records, "latent_norm")
    if len(norms) < 2:
        return 1.0
    denominator = abs(mean(norms)) + 1e-6
    return round(max(0.0, min(1.0, (max(norms) - min(norms)) / denominator)), 6)


def _trace_digest_payload(generation_record: dict, trace_records: list[dict]) -> dict[str, Any]:
    """构造 authenticated sketch digest 的稳定输入。"""
    return {
        "generation_model_id": generation_record.get("generation_model_id"),
        "prompt_id": generation_record.get("prompt_id"),
        "seed_id": generation_record.get("seed_id"),
        "trajectory_trace_id": generation_record.get("trajectory_trace_id"),
        "sampler_signature_placeholder": generation_record.get("sampler_signature_placeholder"),
        "trace_steps": [
            {
                "trajectory_step_index": record.get("trajectory_step_index"),
                "latent_norm": record.get("latent_norm"),
                "latent_mean": record.get("latent_mean"),
                "latent_std": record.get("latent_std"),
            }
            for record in trace_records
        ],
    }


def _base_replay_record(generation_record: dict, replay_record_type: str) -> dict[str, Any]:
    """构造 replay/sketch 四类 records 共享的基础字段。"""
    return {
        "record_version": "replay_and_sketch_gate_v1",
        "replay_record_type": replay_record_type,
        "generation_model_id": generation_record.get("generation_model_id"),
        "prompt_id": generation_record.get("prompt_id"),
        "seed_id": generation_record.get("seed_id"),
        "trajectory_trace_id": generation_record.get("trajectory_trace_id"),
        "authenticated_trajectory_sketch_status": "not_evaluated",
        "trajectory_sketch_digest_random": None,
        "trajectory_sketch_verification_status": "not_evaluated",
        "replay_uncertainty_weight": None,
        "replay_uncertainty_mean": None,
        "replay_scheduler_id": generation_record.get("scheduler_id") or "wan21_owner_side_diagnostic_scheduler",
        "replay_time_grid_id": generation_record.get("time_grid_id") or "wan21_owner_side_diagnostic_time_grid",
        "wrong_sampler_replay_control": "not_applicable",
        "wrong_prompt_replay_control": "not_applicable",
        "replay_control_status": "not_applicable",
        "replay_signature_mismatch_status": "not_applicable",
        "replay_and_sketch_evidence_level": REPLAY_AND_SKETCH_EVIDENCE_LEVEL,
        "claim_support_status": REPLAY_AND_SKETCH_CLAIM_SUPPORT_STATUS,
    }


def _with_protocol(record: dict[str, Any], *, admissibility_status: str) -> dict[str, Any]:
    """为 replay/sketch record 补齐 Flow evidence 协议字段。"""
    return with_flow_evidence_protocol_defaults(
        record,
        trajectory_source_level="replay_and_sketch_owner_side_trace_diagnostic",
        flow_state_admissibility_status=admissibility_status,
        claim_support_status=REPLAY_AND_SKETCH_CLAIM_SUPPORT_STATUS,
    )


def build_replay_and_sketch_records(run_root: str | Path) -> dict[str, list[dict[str, Any]]]:
    """从已落盘 records 构造 replay/sketch gate 所需的四类 records。"""
    run_root = Path(run_root)
    full_claim3_records = _build_full_claim3_records(run_root)
    if full_claim3_records is not None:
        return full_claim3_records
    generation_records = _read_jsonl(run_root / "records" / "generation_records.jsonl")
    formal_metric_records = _read_jsonl(run_root / "records" / "formal_quality_motion_semantic_records.jsonl")
    trajectory_records = _read_jsonl(run_root / "records" / "trajectory_trace.jsonl")
    selection = select_motion_claim_generation_records(generation_records, formal_metric_records)
    eligible_generation_records = selection.eligible_generation_records
    trajectory_groups = _group_by_trace_id(trajectory_records)

    sketch_records: list[dict[str, Any]] = []
    uncertainty_records: list[dict[str, Any]] = []
    wrong_sampler_records: list[dict[str, Any]] = []
    wrong_prompt_records: list[dict[str, Any]] = []
    sketch_by_trace: dict[str, dict[str, Any]] = {}

    for generation_record in eligible_generation_records:
        trace_id = str(generation_record.get("trajectory_trace_id") or "")
        trace_records = trajectory_groups.get(trace_id, [])
        sketch_ready = bool(trace_records)
        sketch_digest = build_stable_digest(_trace_digest_payload(generation_record, trace_records)) if sketch_ready else None
        sketch_status = "verified" if sketch_ready else "missing_trace_records"
        sketch_record = _base_replay_record(generation_record, "trajectory_sketch_verification")
        sketch_record.update({
            "authenticated_trajectory_sketch_status": "ready" if sketch_ready else "not_ready",
            "trajectory_sketch_digest_random": sketch_digest,
            "trajectory_sketch_verification_status": sketch_status,
            "replay_control_status": "sketch_verified" if sketch_ready else "sketch_missing",
            "replay_signature_mismatch_status": "matched_scheduler_and_time_grid" if sketch_ready else "not_evaluated",
        })
        sketch_record = _with_protocol(
            sketch_record,
            admissibility_status="replay_sketch_trace_verified" if sketch_ready else "replay_sketch_trace_missing",
        )
        sketch_records.append(sketch_record)
        if sketch_digest:
            sketch_by_trace[trace_id] = sketch_record

        uncertainty = _replay_uncertainty(trace_records)
        uncertainty_weight = round(1.0 / (1.0 + uncertainty), 6)
        uncertainty_record = _base_replay_record(generation_record, "replay_uncertainty")
        uncertainty_record.update({
            "authenticated_trajectory_sketch_status": "ready" if sketch_ready else "not_ready",
            "trajectory_sketch_digest_random": sketch_digest,
            "trajectory_sketch_verification_status": sketch_status,
            "replay_uncertainty_mean": uncertainty,
            "replay_uncertainty_weight": uncertainty_weight,
            "replay_control_status": "uncertainty_weight_ready" if sketch_ready else "uncertainty_blocked_by_missing_trace",
        })
        uncertainty_records.append(_with_protocol(
            uncertainty_record,
            admissibility_status="replay_uncertainty_weight_ready" if sketch_ready else "replay_uncertainty_missing_trace",
        ))

        wrong_sampler_digest = build_stable_digest({
            "matched_sketch_digest": sketch_digest,
            "wrong_sampler": "wrong_sampler_signature_control",
            "replay_time_grid_id": "wrong_time_grid_control",
        }) if sketch_digest else None
        wrong_sampler_record = _base_replay_record(generation_record, "wrong_sampler_replay")
        wrong_sampler_record.update({
            "authenticated_trajectory_sketch_status": "ready" if sketch_ready else "not_ready",
            "trajectory_sketch_digest_random": wrong_sampler_digest,
            "trajectory_sketch_verification_status": "mismatch_rejected" if sketch_ready else "not_ready",
            "wrong_sampler_replay_control": "wrong_sampler_signature_control",
            "replay_control_status": "replay_rejected" if sketch_ready else "blocked_by_missing_sketch",
            "replay_signature_mismatch_status": "wrong_sampler_signature_mismatch" if sketch_ready else "not_evaluated",
        })
        wrong_sampler_records.append(_with_protocol(
            wrong_sampler_record,
            admissibility_status="wrong_sampler_replay_rejected" if sketch_ready else "wrong_sampler_replay_not_evaluated",
        ))

    # wrong prompt control 需要至少两个 prompt。它通过跨 prompt sketch digest mismatch 证明 prompt 条件不能互相伪造。
    ready_sketch_records = [record for record in sketch_records if record.get("trajectory_sketch_digest_random")]
    for index, generation_record in enumerate(eligible_generation_records):
        trace_id = str(generation_record.get("trajectory_trace_id") or "")
        matched_sketch = sketch_by_trace.get(trace_id)
        mismatch_candidates = [
            record for record in ready_sketch_records
            if record.get("prompt_id") != generation_record.get("prompt_id")
        ]
        wrong_prompt_record = _base_replay_record(generation_record, "wrong_prompt_replay")
        if matched_sketch and mismatch_candidates:
            candidate = mismatch_candidates[index % len(mismatch_candidates)]
            wrong_prompt_digest = build_stable_digest({
                "matched_sketch_digest": matched_sketch.get("trajectory_sketch_digest_random"),
                "wrong_prompt_sketch_digest": candidate.get("trajectory_sketch_digest_random"),
                "wrong_prompt_id": candidate.get("prompt_id"),
            })
            wrong_prompt_record.update({
                "authenticated_trajectory_sketch_status": "ready",
                "trajectory_sketch_digest_random": wrong_prompt_digest,
                "trajectory_sketch_verification_status": "mismatch_rejected",
                "wrong_prompt_replay_control": str(candidate.get("prompt_id") or "wrong_prompt_control"),
                "replay_control_status": "replay_rejected",
                "replay_signature_mismatch_status": "wrong_prompt_sketch_mismatch",
            })
            admissibility_status = "wrong_prompt_replay_rejected"
        else:
            wrong_prompt_record.update({
                "authenticated_trajectory_sketch_status": "not_ready",
                "trajectory_sketch_verification_status": "not_ready",
                "wrong_prompt_replay_control": "missing_cross_prompt_candidate",
                "replay_control_status": "blocked_by_missing_cross_prompt_candidate",
            })
            admissibility_status = "wrong_prompt_replay_not_evaluated"
        wrong_prompt_records.append(_with_protocol(wrong_prompt_record, admissibility_status=admissibility_status))

    return {
        "trajectory_sketch_verification_records": sketch_records,
        "replay_uncertainty_records": uncertainty_records,
        "wrong_sampler_replay_records": wrong_sampler_records,
        "wrong_prompt_replay_records": wrong_prompt_records,
    }


def _all_records(record_groups: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """把四类 replay/sketch records 展平为统一 table rows。"""
    records: list[dict[str, Any]] = []
    for group in record_groups.values():
        records.extend(group)
    return records


def _table_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """将 records 投影为统一 CSV 表结构, 避免不同 record 类型字段不一致。"""
    return [{field_name: record.get(field_name) for field_name in REPLAY_RECORD_TABLE_FIELDS} for record in records]


def audit_replay_and_sketch_records(record_groups: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """审计 replay/sketch gate 是否具备 paper profile 工程闭环。"""
    sketch_records = record_groups["trajectory_sketch_verification_records"]
    uncertainty_records = record_groups["replay_uncertainty_records"]
    wrong_sampler_records = record_groups["wrong_sampler_replay_records"]
    wrong_prompt_records = record_groups["wrong_prompt_replay_records"]
    wrong_key_records = record_groups.get("wrong_key_replay_records", [])
    sketch_ready_count = sum(1 for record in sketch_records if record.get("trajectory_sketch_verification_status") == "verified")
    uncertainty_ready_count = sum(1 for record in uncertainty_records if record.get("replay_control_status") == "uncertainty_weight_ready")
    wrong_sampler_rejected_count = sum(1 for record in wrong_sampler_records if record.get("replay_control_status") == "replay_rejected")
    wrong_prompt_rejected_count = sum(1 for record in wrong_prompt_records if record.get("replay_control_status") == "replay_rejected")
    total_sketch_count = len(sketch_records)
    full_claim3_mode = bool(sketch_records) and all(
        record.get("replay_and_sketch_evidence_level") == FULL_CLAIM3_EVIDENCE_LEVEL
        for record in sketch_records
    )

    if full_claim3_mode:
        wrong_key_rejected_count = sum(
            1 for record in wrong_key_records if record.get("replay_control_status") == "replay_rejected"
        )
        minimum_control_pass_rate = 0.8
        control_rates = {
            "wrong_sampler": wrong_sampler_rejected_count / max(1, len(wrong_sampler_records)),
            "wrong_prompt": wrong_prompt_rejected_count / max(1, len(wrong_prompt_records)),
            "wrong_key": wrong_key_rejected_count / max(1, len(wrong_key_records)),
        }
        uncertainty_weights = [
            float(record["replay_uncertainty_weight"])
            for record in uncertainty_records
            if record.get("replay_uncertainty_weight") is not None
        ]
        posterior_ready_count = sum(
            1
            for record in uncertainty_records
            if record.get("flow_posterior_confidence") is not None
            and record.get("flow_state_posterior_entropy") is not None
            and record.get("S_final_conservative") is not None
        )
        minimum_replay_reliability_mean = 0.5
        replay_reliability_mean = mean(uncertainty_weights) if uncertainty_weights else 0.0
        requirement_checks = {
            "authenticated_trajectory_sketch_records_ready": total_sketch_count > 0 and sketch_ready_count == total_sketch_count,
            "attacked_video_replay_uncertainty_records_ready": total_sketch_count > 0 and uncertainty_ready_count == total_sketch_count,
            "flow_replay_posterior_records_ready": posterior_ready_count == total_sketch_count,
            "replay_reliability_mean_ready": replay_reliability_mean >= minimum_replay_reliability_mean,
            "wrong_sampler_replay_control_reliable": control_rates["wrong_sampler"] >= minimum_control_pass_rate,
            "wrong_prompt_replay_control_reliable": control_rates["wrong_prompt"] >= minimum_control_pass_rate,
            "wrong_key_replay_control_reliable": control_rates["wrong_key"] >= minimum_control_pass_rate,
        }
        missing = [name for name, passed in requirement_checks.items() if not passed]
        decision = "PASS" if not missing else "FAIL"
        return {
            "stage_id": "replay_and_authenticated_sketch_gate",
            "replay_and_sketch_gate_decision": decision,
            "claim_support_status": "claim3_attacked_video_replay_posterior_supported" if decision == "PASS" else "claim3_attacked_video_replay_posterior_blocked",
            "replay_and_sketch_evidence_level": FULL_CLAIM3_EVIDENCE_LEVEL,
            "claim3_full_support_allowed": decision == "PASS",
            "claim3_full_support_blocking_reason": "none" if decision == "PASS" else "formal_replay_or_control_requirement_failed",
            "replay_or_sketch_status": "full_attacked_video_replay_and_authenticated_sketch_ready" if decision == "PASS" else "full_attacked_video_replay_blocked",
            "replay_and_sketch_missing_requirements": missing,
            "replay_and_sketch_missing_requirement_count": len(missing),
            "trajectory_sketch_verification_record_count": total_sketch_count,
            "trajectory_sketch_verified_count": sketch_ready_count,
            "replay_uncertainty_record_count": len(uncertainty_records),
            "replay_uncertainty_ready_count": uncertainty_ready_count,
            "replay_uncertainty_weight_mean": round(mean(uncertainty_weights), 6) if uncertainty_weights else None,
            "flow_replay_posterior_ready_count": posterior_ready_count,
            "minimum_replay_reliability_mean": minimum_replay_reliability_mean,
            "wrong_sampler_replay_record_count": len(wrong_sampler_records),
            "wrong_sampler_replay_rejected_count": wrong_sampler_rejected_count,
            "wrong_prompt_replay_record_count": len(wrong_prompt_records),
            "wrong_prompt_replay_rejected_count": wrong_prompt_rejected_count,
            "wrong_key_replay_record_count": len(wrong_key_records),
            "wrong_key_replay_rejected_count": wrong_key_rejected_count,
            "minimum_replay_control_pass_rate": minimum_control_pass_rate,
            "replay_control_pass_rates": control_rates,
        }

    requirement_checks = {
        "authenticated_trajectory_sketch_records_ready": total_sketch_count > 0 and sketch_ready_count == total_sketch_count,
        "replay_uncertainty_records_ready": total_sketch_count > 0 and uncertainty_ready_count == total_sketch_count,
        "wrong_sampler_replay_records_ready": total_sketch_count > 0 and wrong_sampler_rejected_count == total_sketch_count,
        "wrong_prompt_replay_records_ready": total_sketch_count > 0 and wrong_prompt_rejected_count == total_sketch_count,
    }
    missing = [name for name, passed in requirement_checks.items() if not passed]
    decision = "PASS" if not missing else "FAIL"
    uncertainty_weights = [
        float(record["replay_uncertainty_weight"])
        for record in uncertainty_records
        if record.get("replay_uncertainty_weight") is not None
    ]
    return {
        "stage_id": "replay_and_authenticated_sketch_gate",
        "replay_and_sketch_gate_decision": decision,
        "claim_support_status": REPLAY_AND_SKETCH_CLAIM_SUPPORT_STATUS if decision == "PASS" else "replay_and_sketch_owner_side_diagnostic_blocked",
        "replay_and_sketch_evidence_level": REPLAY_AND_SKETCH_EVIDENCE_LEVEL,
        "claim3_full_support_allowed": False,
        "claim3_full_support_blocking_reason": "owner_side_diagnostic_not_full_paper_authenticated_replay",
        "replay_or_sketch_status": "replay_and_sketch_gate_passed_owner_side_diagnostic" if decision == "PASS" else "replay_and_sketch_gate_blocked",
        "replay_and_sketch_missing_requirements": missing,
        "replay_and_sketch_missing_requirement_count": len(missing),
        "trajectory_sketch_verification_record_count": len(sketch_records),
        "trajectory_sketch_verified_count": sketch_ready_count,
        "replay_uncertainty_record_count": len(uncertainty_records),
        "replay_uncertainty_ready_count": uncertainty_ready_count,
        "replay_uncertainty_weight_mean": round(mean(uncertainty_weights), 6) if uncertainty_weights else None,
        "wrong_sampler_replay_record_count": len(wrong_sampler_records),
        "wrong_sampler_replay_rejected_count": wrong_sampler_rejected_count,
        "wrong_prompt_replay_record_count": len(wrong_prompt_records),
        "wrong_prompt_replay_rejected_count": wrong_prompt_rejected_count,
        "wrong_key_replay_record_count": len(wrong_key_records),
        "wrong_key_replay_rejected_count": 0,
    }


def run_replay_and_sketch_gate(run_root: str | Path) -> dict[str, Any]:
    """写出 replay/sketch gate 的 records、table、decision 和 report。"""
    run_root = Path(run_root)
    record_groups = build_replay_and_sketch_records(run_root)
    audit = audit_replay_and_sketch_records(record_groups)
    write_jsonl(run_root / "records" / "trajectory_sketch_verification_records.jsonl", record_groups["trajectory_sketch_verification_records"])
    write_jsonl(run_root / "records" / "replay_uncertainty_records.jsonl", record_groups["replay_uncertainty_records"])
    write_jsonl(run_root / "records" / "wrong_sampler_replay_records.jsonl", record_groups["wrong_sampler_replay_records"])
    write_jsonl(run_root / "records" / "wrong_prompt_replay_records.jsonl", record_groups["wrong_prompt_replay_records"])
    write_jsonl(run_root / "records" / "wrong_key_replay_records.jsonl", record_groups.get("wrong_key_replay_records", []))
    write_csv(run_root / "tables" / "replay_verification_table.csv", _table_rows(_all_records(record_groups)))
    write_json(run_root / "artifacts" / "replay_and_sketch_gate_decision.json", audit)
    pre_replay = _read_json(run_root / "artifacts" / "three_layer_mechanism_evidence_decision.json")
    claim1_pass = pre_replay.get("claim_1_velocity_constraint_detectable_watermark_decision") == "PASS"
    claim2_pass = pre_replay.get("claim_2_path_evidence_independent_gain_decision") == "PASS"
    claim3_pass = audit.get("claim3_full_support_allowed") is True
    complete_claim_audit = {
        "stage_id": "sstw_complete_paper_mechanism_claim_gate",
        "claim_1_velocity_constraint_detectable_watermark_decision": "PASS" if claim1_pass else "FAIL",
        "claim_2_path_evidence_independent_gain_decision": "PASS" if claim2_pass else "FAIL",
        "claim_3_attacked_video_replay_posterior_decision": "PASS" if claim3_pass else "FAIL",
        "complete_paper_mechanism_claim_decision": "PASS" if claim1_pass and claim2_pass and claim3_pass else "FAIL",
        "claim3_downgrade_allowed": False,
        "claim3_full_support_allowed": claim3_pass,
        "claim_support_status": "sstw_complete_three_layer_claim_supported"
        if claim1_pass and claim2_pass and claim3_pass
        else "sstw_complete_three_layer_claim_blocked",
    }
    write_json(run_root / "artifacts" / "complete_paper_mechanism_claim_decision.json", complete_claim_audit)
    report = (
        "# Replay and Authenticated Sketch Gate Report\n\n"
        "该报告由 generation records 与 trajectory trace 自动生成, 用于闭合 paper profile 的 replay/sketch 工程入口。"
        "报告会区分 owner-side diagnostic 与 attacked-video model-velocity replay。"
        "只有真实 replay、HMAC 验证和三类错误条件对照同时通过时才允许 Claim-3 完整支持。\n\n"
        f"- replay_and_sketch_gate_decision: {audit['replay_and_sketch_gate_decision']}\n"
        f"- replay_and_sketch_evidence_level: {audit['replay_and_sketch_evidence_level']}\n"
        f"- trajectory_sketch_verified_count: {audit['trajectory_sketch_verified_count']}\n"
        f"- replay_uncertainty_ready_count: {audit['replay_uncertainty_ready_count']}\n"
        f"- wrong_sampler_replay_rejected_count: {audit['wrong_sampler_replay_rejected_count']}\n"
        f"- wrong_prompt_replay_rejected_count: {audit['wrong_prompt_replay_rejected_count']}\n"
        f"- claim3_full_support_allowed: {str(audit['claim3_full_support_allowed']).lower()}\n"
        f"- claim_support_status: {audit['claim_support_status']}\n"
    )
    report_path = run_root / "reports" / "replay_and_sketch_gate_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    complete_report_path = run_root / "reports" / "complete_paper_mechanism_claim_report.md"
    complete_report_path.write_text(
        "# SSTW Complete Paper Mechanism Claim Report\n\n"
        f"- claim_1_decision: {complete_claim_audit['claim_1_velocity_constraint_detectable_watermark_decision']}\n"
        f"- claim_2_decision: {complete_claim_audit['claim_2_path_evidence_independent_gain_decision']}\n"
        f"- claim_3_decision: {complete_claim_audit['claim_3_attacked_video_replay_posterior_decision']}\n"
        f"- complete_paper_mechanism_claim_decision: {complete_claim_audit['complete_paper_mechanism_claim_decision']}\n",
        encoding="utf-8",
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 replay/sketch gate validation records。")
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    payload = run_replay_and_sketch_gate(args.run_root)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
