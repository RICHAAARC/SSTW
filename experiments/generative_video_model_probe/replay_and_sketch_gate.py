"""replay/sketch gate 的 paper profile 正式 runner.

该模块消费真实攻击后视频 replay records, 并审计 authenticated sketch, replay
uncertainty, wrong key, wrong sampler, wrong prompt 和 wrong time grid 控制. 只有
校准概率后验, 固定路径假设检验和全部控制同时通过时, 才允许 Claim-3 完整支持.
缺少任一正式输入时直接形成 FAIL 关闭记录, 不生成 owner-side 降级证据.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from statistics import mean
from typing import Any

from runtime.core.digest import build_stable_digest
from main.methods.state_space_watermark.authenticated_trajectory_sketch import (
    verify_authenticated_trajectory_sketch_once,
)
from main.methods.state_space_watermark.formal_detector import FLOW_STATE_POSTERIOR_SCORE_SOURCE
from main.methods.state_space_watermark.replay_inversion import (
    REPLAY_GAUSSIAN_LIKELIHOOD_MODEL_ID,
)
from evaluation.protocol.generation_record_binding import (
    build_generation_record_binding_digest,
)
from evaluation.protocol.record_writer import write_json, write_jsonl
from evaluation.protocol.table_builder import write_csv


FULL_CLAIM3_EVIDENCE_LEVEL = (
    "attacked_video_key_independent_inversion_hypothesis_replay_with_hmac_sketch"
)
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
    "trajectory_sketch_signature_valid",
    "trajectory_sketch_formal_binding_complete",
    "trajectory_sketch_binding_matches",
    "trajectory_sketch_nonce_fresh",
    "trajectory_sketch_generation_record_digest_matches",
    "trajectory_sketch_verification_failure_reasons",
    "replay_uncertainty_weight",
    "replay_uncertainty_mean",
    "replay_log_likelihood_ratio_mean",
    "replay_likelihood_model_id",
    "replay_likelihood_calibration_protocol",
    "replay_likelihood_calibration_cluster_count",
    "replay_relative_observation_noise_standard_deviation",
    "replay_control_fixed_reverse_path_reused",
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
    "flow_state_log_likelihood_ratio",
    "flow_state_filter_step_count",
    "flow_state_filtering_status",
    "flow_state_smoothing_status",
    "flow_tubelet_formal_context_complete",
    "path_quadrature_context_complete",
    "endpoint_formal_context_complete",
    "flow_state_observation_formal_context_complete",
    "replay_control_joint_context_complete",
    "flow_watermark_posterior_probability",
    "flow_watermark_posterior_log_odds",
    "flow_state_posterior_entropy",
)


def _read_jsonl(path: Path) -> list[dict]:
    """读取 JSONL records, 文件不存在时返回空列表。"""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _is_positive_finite_number(value: Any) -> bool:
    """判断治理记录中的数值是否为有限正数, 非法文本直接返回 False。"""

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric) and numeric > 0.0


def _read_json(path: Path) -> dict[str, Any]:
    """读取 JSON decision; 文件不存在时返回空对象。"""

    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return payload if isinstance(payload, dict) else {}


def _empty_claim3_record_groups() -> dict[str, list[dict[str, Any]]]:
    """构造正式 Claim-3 缺失时的空记录组, 禁止回退到诊断证据。"""

    return {
        "trajectory_sketch_verification_records": [],
        "replay_uncertainty_records": [],
        "wrong_sampler_replay_records": [],
        "wrong_prompt_replay_records": [],
        "wrong_key_replay_records": [],
    }


def _build_full_claim3_records(run_root: Path) -> dict[str, list[dict[str, Any]]]:
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
        return _empty_claim3_record_groups()
    sketches = {
        str(record.get("trajectory_trace_id")): record
        for record in _read_jsonl(run_root / "records" / "trajectory_sketch_records.jsonl")
        if record.get("authenticated_trajectory_sketch_status")
        == "signed_formal_binding"
    }
    authentication_key_text = os.environ.get("SSTW_TRAJECTORY_AUTHENTICATION_KEY") or ""
    authentication_key = authentication_key_text.encode("utf-8")
    generation_by_trace = {
        str(record.get("trajectory_trace_id") or ""): record
        for record in _read_jsonl(run_root / "records" / "generation_records.jsonl")
        if record.get("generation_status") == "success"
        and str(record.get("trajectory_trace_id") or "").strip()
    }
    consumed_nonces: set[str] = set()
    sketch_verification_by_trace: dict[str, Any] = {}
    generation_digest_matches_by_trace: dict[str, bool] = {}
    for trace_id in sorted({
        str(record.get("trajectory_trace_id") or "") for record in evidence_records
    }):
        generation_record = generation_by_trace.get(trace_id, {})
        sketch = sketches.get(trace_id, {})
        try:
            recomputed_generation_digest = build_generation_record_binding_digest(
                generation_record
            )
        except (KeyError, TypeError, ValueError):
            recomputed_generation_digest = ""
        stored_generation_digest = str(
            generation_record.get("generation_record_digest") or ""
        )
        generation_digest_matches_by_trace[trace_id] = bool(
            recomputed_generation_digest
            and stored_generation_digest == recomputed_generation_digest
        )
        expected_binding = {
            "trajectory_trace_id": trace_id,
            "method_configuration_id": str(
                generation_record.get("method_variant") or ""
            ),
            "video_sha256": str(generation_record.get("video_sha256") or ""),
            "generation_record_digest": recomputed_generation_digest,
            "code_commit": str(generation_record.get("code_commit") or ""),
        }
        if authentication_key and all(expected_binding.values()):
            sketch_verification_by_trace[trace_id] = (
                verify_authenticated_trajectory_sketch_once(
                    sketch,
                    authentication_key=authentication_key,
                    expected_binding=expected_binding,
                    consumed_nonces=consumed_nonces,
                )
            )
    groups = _empty_claim3_record_groups()
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
        verification = sketch_verification_by_trace.get(trace_id)
        signature_verified = bool(
            verification is not None and verification.signature_valid
        )
        formal_binding_complete = bool(
            verification is not None and verification.formal_binding_complete
        )
        binding_matches = bool(
            verification is not None and verification.binding_matches
        )
        nonce_fresh = bool(
            verification is not None and verification.nonce_fresh
        )
        generation_digest_matches = generation_digest_matches_by_trace.get(
            trace_id,
            False,
        )
        sketch_verified = bool(
            verification is not None
            and verification.verified
            and context_matches
            and generation_digest_matches
        )
        common = {
            "record_version": "replay_and_sketch_gate_v2",
            "generation_model_id": evidence.get("generation_model_id"),
            "prompt_id": evidence.get("prompt_id"),
            "seed_id": evidence.get("seed_id"),
            "trajectory_trace_id": trace_id,
            "attack_name": evidence.get("attack_name"),
            "method_variant": evidence.get("method_variant"),
            "replay_and_sketch_evidence_level": FULL_CLAIM3_EVIDENCE_LEVEL,
            "claim_support_status": "claim3_attacked_video_replay_posterior_evidence",
            "trajectory_source_level": "attacked_video_model_velocity_inversion_replay",
            "flow_state_admissibility_status": evidence.get("flow_state_admissibility_status"),
            "flow_posterior_confidence": evidence.get("flow_posterior_confidence"),
            "flow_watermark_posterior_probability": evidence.get("flow_watermark_posterior_probability"),
            "flow_watermark_posterior_log_odds": evidence.get("flow_watermark_posterior_log_odds"),
            "flow_state_posterior_entropy": evidence.get("flow_state_posterior_entropy"),
            "flow_state_log_likelihood_ratio": evidence.get("flow_state_log_likelihood_ratio"),
            "flow_state_filter_step_count": evidence.get("flow_state_filter_step_count"),
            "flow_state_filtering_status": evidence.get("flow_state_filtering_status"),
            "flow_state_smoothing_status": evidence.get("flow_state_smoothing_status"),
            "flow_detector_score_source": evidence.get("flow_detector_score_source"),
            "replay_log_likelihood_ratio_mean": evidence.get("replay_log_likelihood_ratio_mean"),
            "replay_likelihood_model_id": evidence.get("replay_likelihood_model_id"),
            "replay_likelihood_calibration_protocol": evidence.get(
                "replay_likelihood_calibration_protocol"
            ),
            "replay_likelihood_calibration_cluster_count": evidence.get(
                "replay_likelihood_calibration_cluster_count"
            ),
            "replay_relative_observation_noise_standard_deviation": evidence.get(
                "replay_relative_observation_noise_standard_deviation"
            ),
            "replay_control_fixed_reverse_path_reused": evidence.get(
                "replay_control_fixed_reverse_path_reused"
            ),
            "threshold_source_split": evidence.get("threshold_source_split"),
            "test_time_threshold_update_blocked": evidence.get("test_time_threshold_update_blocked"),
            "S_final_conservative": evidence.get("S_final_conservative"),
            "flow_tubelet_formal_context_complete": evidence.get(
                "flow_tubelet_formal_context_complete"
            ),
            "path_quadrature_context_complete": evidence.get(
                "path_quadrature_context_complete"
            ),
            "endpoint_formal_context_complete": evidence.get(
                "endpoint_formal_context_complete"
            ),
            "flow_state_observation_formal_context_complete": evidence.get(
                "flow_state_observation_formal_context_complete"
            ),
            "replay_control_joint_context_complete": evidence.get(
                "replay_control_joint_context_complete"
            ),
            "trajectory_sketch_signature_valid": signature_verified,
            "trajectory_sketch_formal_binding_complete": formal_binding_complete,
            "trajectory_sketch_binding_matches": binding_matches,
            "trajectory_sketch_nonce_fresh": nonce_fresh,
            "trajectory_sketch_generation_record_digest_matches": (
                generation_digest_matches
            ),
            "trajectory_sketch_verification_failure_reasons": (
                list(verification.failure_reasons)
                if verification is not None
                else ["missing_authentication_key_or_generation_binding"]
            ),
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
            and evidence.get("formal_flow_evidence_level")
            == "attacked_video_key_independent_inversion_hypothesis_replay"
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


def build_replay_and_sketch_records(
    run_root: str | Path,
) -> dict[str, list[dict[str, Any]]]:
    """只从正式 attacked-video replay 构造 Claim-3 记录。

    若正式 Flow evidence 不存在, 返回空记录组并由同一正式门禁判定 FAIL。
    此处不再生成 owner-side diagnostic, 因而任何 profile 都不存在 Claim-3
    降级通过路径。
    """

    return _build_full_claim3_records(Path(run_root))


def _all_records(record_groups: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """把四类 replay/sketch records 展平为统一 table rows。"""
    records: list[dict[str, Any]] = []
    for group in record_groups.values():
        records.extend(group)
    return records


def _table_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """将 records 投影为统一 CSV 表结构, 避免不同 record 类型字段不一致。"""
    return [{field_name: record.get(field_name) for field_name in REPLAY_RECORD_TABLE_FIELDS} for record in records]


def audit_replay_and_sketch_records(
    record_groups: dict[str, list[dict[str, Any]]],
    *,
    heldout_posterior_calibration_decision: str | None = None,
) -> dict[str, Any]:
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
        if record.get("flow_watermark_posterior_probability") is not None
        and 0.0 <= float(record["flow_watermark_posterior_probability"]) <= 1.0
        and record.get("flow_watermark_posterior_log_odds") is not None
        and record.get("flow_state_posterior_entropy") is not None
        and record.get("flow_state_log_likelihood_ratio") is not None
        and int(record.get("flow_state_filter_step_count") or 0) >= 2
        and record.get("flow_state_filtering_status") == "kalman_filter_ready"
        and record.get("flow_state_smoothing_status")
        == "rauch_tung_striebel_smoother_ready"
        and record.get("replay_log_likelihood_ratio_mean") is not None
        and record.get("replay_likelihood_model_id")
        == REPLAY_GAUSSIAN_LIKELIHOOD_MODEL_ID
        and record.get("replay_likelihood_calibration_protocol")
        == "calibration_clean_video_null_residual_cluster_equal_mle"
        and int(
            record.get("replay_likelihood_calibration_cluster_count") or 0
        ) >= 2
        and _is_positive_finite_number(
            record.get(
                "replay_relative_observation_noise_standard_deviation"
            )
        )
        and record.get("replay_control_fixed_reverse_path_reused") is True
        and record.get("flow_detector_score_source")
        == FLOW_STATE_POSTERIOR_SCORE_SOURCE
        and record.get("threshold_source_split") == "calibration"
        and record.get("test_time_threshold_update_blocked") is True
        and record.get("S_final_conservative") is not None
        and record.get("flow_tubelet_formal_context_complete") is True
        and record.get("path_quadrature_context_complete") is True
        and record.get("endpoint_formal_context_complete") is True
        and record.get("flow_state_observation_formal_context_complete")
        is True
        and record.get("replay_control_joint_context_complete") is True
    )
    minimum_replay_reliability_mean = 0.5
    replay_reliability_mean = mean(uncertainty_weights) if uncertainty_weights else 0.0
    requirement_checks = {
        "authenticated_trajectory_sketch_records_ready": total_sketch_count > 0 and sketch_ready_count == total_sketch_count,
        "attacked_video_replay_uncertainty_records_ready": total_sketch_count > 0 and uncertainty_ready_count == total_sketch_count,
        "flow_replay_posterior_records_ready": posterior_ready_count == total_sketch_count,
        "flow_replay_state_space_filtering_smoothing_ready": (
            posterior_ready_count == total_sketch_count
        ),
        "replay_reliability_mean_ready": replay_reliability_mean >= minimum_replay_reliability_mean,
        "wrong_sampler_replay_control_reliable": control_rates["wrong_sampler"] >= minimum_control_pass_rate,
        "wrong_prompt_replay_control_reliable": control_rates["wrong_prompt"] >= minimum_control_pass_rate,
        "wrong_key_replay_control_reliable": control_rates["wrong_key"] >= minimum_control_pass_rate,
        "heldout_attacked_test_posterior_calibration_ready": (
            heldout_posterior_calibration_decision == "PASS"
        ),
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
        "heldout_posterior_calibration_decision": (
            heldout_posterior_calibration_decision or "MISSING"
        ),
    }

def run_replay_and_sketch_gate(run_root: str | Path) -> dict[str, Any]:
    """写出 replay/sketch gate 的 records、table、decision 和 report。"""
    run_root = Path(run_root)
    record_groups = build_replay_and_sketch_records(run_root)
    heldout_posterior_audit = _read_json(
        run_root / "artifacts" / "heldout_posterior_calibration_decision.json"
    )
    audit = audit_replay_and_sketch_records(
        record_groups,
        heldout_posterior_calibration_decision=heldout_posterior_audit.get(
            "heldout_posterior_calibration_decision"
        ),
    )
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
        "claim3_full_support_allowed": claim3_pass,
        "claim_support_status": "sstw_complete_three_layer_claim_supported"
        if claim1_pass and claim2_pass and claim3_pass
        else "sstw_complete_three_layer_claim_blocked",
    }
    write_json(run_root / "artifacts" / "complete_paper_mechanism_claim_decision.json", complete_claim_audit)
    report = (
        "# Replay and Authenticated Sketch Gate Report\n\n"
        "该报告由正式 attacked-video replay 与认证 trajectory sketch 自动生成, 用于闭合 paper profile 的 replay/sketch 工程入口。"
        "缺少正式输入时直接失败关闭, 不提供 owner-side diagnostic 降级路径。"
        "只有真实 replay、HMAC 验证和三类错误条件对照同时通过时才允许 Claim-3 完整支持。\n\n"
        f"- replay_and_sketch_gate_decision: {audit['replay_and_sketch_gate_decision']}\n"
        f"- replay_and_sketch_evidence_level: {audit['replay_and_sketch_evidence_level']}\n"
        f"- trajectory_sketch_verified_count: {audit['trajectory_sketch_verified_count']}\n"
        f"- replay_uncertainty_ready_count: {audit['replay_uncertainty_ready_count']}\n"
        f"- wrong_sampler_replay_rejected_count: {audit['wrong_sampler_replay_rejected_count']}\n"
        f"- wrong_prompt_replay_rejected_count: {audit['wrong_prompt_replay_rejected_count']}\n"
        f"- heldout_posterior_calibration_decision: {audit.get('heldout_posterior_calibration_decision', 'NOT_APPLICABLE')}\n"
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
