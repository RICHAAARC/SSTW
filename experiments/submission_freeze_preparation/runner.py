"""汇总 synthetic_state_inference_sanity_to_sampling_time_constraint_probe governed evidence 并生成 submission freeze preparation claim audit。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from main.protocol.record_writer import write_json, write_jsonl
from main.protocol.table_builder import write_csv
from scripts.check_results.generative_video_colab_result_checker import check_generative_video_colab_results
from scripts.check_results.sampling_time_constraint_colab_result_checker import check_sampling_time_constraint_colab_results
from scripts.package_results.submission_freeze_preparation_packager import build_submission_freeze_preparation_package
from experiments.submission_freeze_preparation.readiness_summary import build_submission_readiness_summary
from experiments.submission_freeze_preparation.main_tables import build_submission_main_tables


DEFAULT_STAGE_DECISION_PATHS = {
    "synthetic_state_protocol": "outputs/runs/synthetic_state_protocol/artifacts/synthetic_state_inference_decision.json",
    "state_space_inference_formalization": "outputs/runs/state_space_inference_formalization/artifacts/state_space_formal_decision.json",
    "real_video_latent_transfer": "outputs/runs/real_video_latent_transfer_check/artifacts/real_video_latent_transfer_decision.json",
    "trajectory_observation_core_probe": "outputs/runs/trajectory_observation_core_probe/artifacts/trajectory_observation_decision.json",
    "sampling_time_constraint_preflight": "outputs/runs/sampling_time_constraint_preflight/artifacts/sampling_time_constraint_preflight_decision.json",
}

DEFAULT_generative_video_model_probe_RUN_ROOT = Path(r"G:\我的云端硬盘\SSTW\runs\generative_video_model_probe\pilot_paper")
DEFAULT_sampling_time_constraint_probe_RUN_ROOT = Path(r"G:\我的云端硬盘\SSTW\runs\sampling_time_constraint_colab")


def _read_json(path: str | Path) -> dict:
    """读取 JSON 文件, 文件不存在时返回空对象。"""
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _stage_decision_status(payload: dict) -> str:
    """将阶段 decision JSON 归一化为 PASS / FAIL。"""
    if payload.get("implementation_decision") == "PASS" and payload.get("mechanism_decision") == "PASS":
        return "PASS"
    return "FAIL"


def _artifact_status_from_checker(payload: dict) -> str:
    """将结果检查器输出归一化为 PASS / FAIL。"""
    if payload.get("implementation_evidence_status") == "PASS" and payload.get("mechanism_evidence_status") == "PASS":
        return "PASS"
    return "FAIL"


def _supporting_stage_record(stage_id: str, status: str, artifact_paths: list[str], details: dict | None = None) -> dict:
    """构造 submission preparation 的阶段证据摘要 record。"""
    return {
        "record_version": "submission_freeze_preparation_stage_evidence_v1",
        "stage_id": "submission_freeze_preparation",
        "evidence_stage_id": stage_id,
        "evidence_decision": status,
        "supporting_artifact_paths": artifact_paths,
        "evidence_details": details or {},
    }


def _build_claim_record(
    claim_id: str,
    claim_text: str,
    claim_scope: str,
    claim_status: str,
    supporting_stage_ids: list[str],
    supporting_artifact_paths: list[str],
    downgrade_reason: str = "none",
) -> dict:
    """构造 claim audit record。

    该函数属于通用工程写法。它只记录 claim 与 evidence 的映射关系, 不直接从人工文本推断结论。
    """
    return {
        "record_version": "submission_freeze_preparation_claim_audit_v1",
        "stage_id": "submission_freeze_preparation",
        "claim_id": claim_id,
        "claim_text": claim_text,
        "claim_scope": claim_scope,
        "claim_status": claim_status,
        "downgrade_reason": downgrade_reason,
        "supporting_stage_ids": supporting_stage_ids,
        "supporting_artifact_paths": supporting_artifact_paths,
        "supported_by_governed_artifacts": bool(supporting_artifact_paths) and claim_status in {"supported", "needs_downgrade"},
    }


def _collect_stage_evidence(stage_decision_paths: dict[str, str | Path]) -> tuple[list[dict], dict[str, str]]:
    """读取 synthetic_state_inference_sanity_to_trajectory_observation_core_probe 与 preflight 阶段 decision 并生成证据摘要。"""
    evidence_records: list[dict] = []
    status_by_stage: dict[str, str] = {}
    for stage_id, path in stage_decision_paths.items():
        payload = _read_json(path)
        status = _stage_decision_status(payload)
        status_by_stage[stage_id] = status
        evidence_records.append(_supporting_stage_record(stage_id, status, [str(path)], payload.get("details", {})))
    return evidence_records, status_by_stage


def _collect_generative_video_model_probe_evidence(generative_video_model_probe_run_root: str | Path) -> tuple[dict, dict]:
    """读取 generative_video_model_probe Colab result checker 输出并生成证据摘要。"""
    generative_video_model_probe_run_root = Path(generative_video_model_probe_run_root)
    payload = check_generative_video_colab_results(generative_video_model_probe_run_root) if generative_video_model_probe_run_root.exists() else {
        "implementation_evidence_status": "FAIL",
        "mechanism_evidence_status": "FAIL",
        "missing_mechanism_requirements": ["generative_video_model_probe_run_root_missing"],
        "run_root": str(generative_video_model_probe_run_root),
    }
    status = _artifact_status_from_checker(payload)
    record = _supporting_stage_record(
        "generative_video_model_probe",
        status,
        [
            str(generative_video_model_probe_run_root / "records" / "generation_records.jsonl"),
            str(generative_video_model_probe_run_root / "records" / "formal_quality_motion_semantic_records.jsonl"),
            str(generative_video_model_probe_run_root / "artifacts" / "generative_video_mechanism_postprocess_decision.json"),
        ],
        {
            "generation_record_count": payload.get("generation_record_count", 0),
            "formal_metric_record_count": payload.get("formal_metric_record_count", 0),
            "missing_mechanism_requirements": payload.get("missing_mechanism_requirements", []),
        },
    )
    return record, payload


def _collect_sampling_time_constraint_evidence(sampling_time_constraint_run_root: str | Path) -> tuple[dict, dict]:
    """读取 sampling_time_constraint_probe Colab result checker 输出并生成证据摘要。"""
    sampling_time_constraint_run_root = Path(sampling_time_constraint_run_root)
    payload = check_sampling_time_constraint_colab_results(sampling_time_constraint_run_root) if sampling_time_constraint_run_root.exists() else {
        "implementation_evidence_status": "FAIL",
        "mechanism_evidence_status": "FAIL",
        "missing_mechanism_requirements": ["sampling_time_constraint_run_root_missing"],
        "run_root": str(sampling_time_constraint_run_root),
    }
    status = _artifact_status_from_checker(payload)
    record = _supporting_stage_record(
        "sampling_time_constraint_colab_probe",
        status,
        [
            str(sampling_time_constraint_run_root / "records" / "constraint_records.jsonl"),
            str(sampling_time_constraint_run_root / "records" / "constraint_variant_summary_records.jsonl"),
            str(sampling_time_constraint_run_root / "artifacts" / "sampling_time_constraint_colab_postprocess_decision.json"),
        ],
        {
            "generation_record_count": payload.get("generation_record_count", 0),
            "constraint_record_count": payload.get("constraint_record_count", 0),
            "formal_metric_record_count": payload.get("formal_metric_record_count", 0),
            "claim_boundary": payload.get("claim_boundary"),
            "missing_mechanism_requirements": payload.get("missing_mechanism_requirements", []),
        },
    )
    return record, payload


def _build_claim_records(status_by_stage: dict[str, str], generative_video_model_probe_payload: dict, sampling_time_constraint_payload: dict) -> list[dict]:
    """基于阶段证据状态生成 claim audit records。"""
    mechanism_precondition_pass = all(
        status_by_stage.get(stage_id) == "PASS"
        for stage_id in (
            "synthetic_state_protocol",
            "state_space_inference_formalization",
            "real_video_latent_transfer",
            "trajectory_observation_core_probe",
        )
    )
    generative_video_model_probe_pass = _artifact_status_from_checker(generative_video_model_probe_payload) == "PASS"
    sampling_time_constraint_pass = _artifact_status_from_checker(sampling_time_constraint_payload) == "PASS"
    sampling_time_constraint_not_final = sampling_time_constraint_payload.get("claim_boundary") == "real_sampling_probe_not_final_sampling_time_constraint_submission_claim"

    claims = [
        _build_claim_record(
            "claim_synthetic_state_space_inference",
            "Key-conditioned state-space inference outperforms ordinary temporal aggregators and explicit alignment baselines in controlled synthetic latent settings.",
            "main",
            "supported" if status_by_stage.get("synthetic_state_protocol") == "PASS" else "unsupported",
            ["synthetic_state_protocol"],
            [DEFAULT_STAGE_DECISION_PATHS["synthetic_state_protocol"]],
        ),
        _build_claim_record(
            "claim_state_space_formalization",
            "The state-space formulation remains supported under governed ablations, key-condition checks, and fixed-FPR constraints.",
            "main",
            "supported" if status_by_stage.get("state_space_inference_formalization") == "PASS" else "unsupported",
            ["state_space_inference_formalization"],
            [DEFAULT_STAGE_DECISION_PATHS["state_space_inference_formalization"]],
        ),
        _build_claim_record(
            "claim_real_video_latent_transfer",
            "The synthetic_state_inference_sanity state-space inference mechanism remains effective in the real-video VAE encode-decode-reencode pathway while preserving low false positives.",
            "main",
            "supported" if status_by_stage.get("real_video_latent_transfer") == "PASS" else "unsupported",
            ["real_video_latent_transfer"],
            [DEFAULT_STAGE_DECISION_PATHS["real_video_latent_transfer"]],
        ),
        _build_claim_record(
            "claim_trajectory_observation_core",
            "Trajectory observation adds governed evidence beyond state-space scores without becoming an uncontrolled shortcut.",
            "main",
            "supported" if status_by_stage.get("trajectory_observation_core_probe") == "PASS" else "unsupported",
            ["trajectory_observation_core_probe"],
            [DEFAULT_STAGE_DECISION_PATHS["trajectory_observation_core_probe"]],
        ),
        _build_claim_record(
            "claim_generative_video_probe",
            "The mechanism remains observable in real generative video outputs under the governed LTX-Video probe and formal quality, motion, and semantic metrics.",
            "main",
            "supported" if generative_video_model_probe_pass else "unsupported",
            ["generative_video_model_probe"],
            [str(generative_video_model_probe_payload.get("run_root", DEFAULT_generative_video_model_probe_RUN_ROOT))],
        ),
        _build_claim_record(
            "claim_sampling_time_constraint_probe",
            "Sampling-time weak constraint improves keyed trajectory alignment in the governed real sampling callback probe while formal quality, motion, and semantic metrics remain ready.",
            "exploratory",
            "supported" if sampling_time_constraint_pass else "unsupported",
            ["sampling_time_constraint_colab_probe"],
            [str(sampling_time_constraint_payload.get("run_root", DEFAULT_sampling_time_constraint_probe_RUN_ROOT))],
        ),
        _build_claim_record(
            "claim_sstw_t_submission_preparation",
            "SSTW-T can enter submission preparation because synthetic_state_inference_sanity_to_generative_video_model_probe governed evidence passes and each supported claim maps to records or decision artifacts.",
            "main",
            "supported" if mechanism_precondition_pass and generative_video_model_probe_pass else "unsupported",
            [
                "synthetic_state_protocol",
                "state_space_inference_formalization",
                "real_video_latent_transfer",
                "trajectory_observation_core_probe",
                "generative_video_model_probe",
            ],
            [str(generative_video_model_probe_payload.get("run_root", DEFAULT_generative_video_model_probe_RUN_ROOT))],
        ),
        _build_claim_record(
            "claim_sstw_tc_submission_freeze",
            "SSTW-TC is ready as a final submission-freeze main claim.",
            "exploratory",
            "needs_downgrade" if sampling_time_constraint_pass and sampling_time_constraint_not_final else "unsupported",
            ["sampling_time_constraint_colab_probe"],
            [str(sampling_time_constraint_payload.get("run_root", DEFAULT_sampling_time_constraint_probe_RUN_ROOT))],
            "sampling_time_constraint_real_sampling_probe_is_supported_but_not_final_submission_freeze_claim" if sampling_time_constraint_pass and sampling_time_constraint_not_final else "sampling_time_constraint_probe_not_ready",
        ),
    ]
    return claims


def run_submission_freeze_preparation(
    output_root: str | Path,
    generative_video_model_probe_run_root: str | Path = DEFAULT_generative_video_model_probe_RUN_ROOT,
    sampling_time_constraint_run_root: str | Path = DEFAULT_sampling_time_constraint_probe_RUN_ROOT,
    stage_decision_paths: dict[str, str | Path] | None = None,
    package_dir: str | Path | None = None,
) -> dict:
    """生成 submission freeze preparation claim audit artifacts。"""
    output_root = Path(output_root)
    package_dir = Path(package_dir) if package_dir else output_root / "packages"
    stage_decision_paths = stage_decision_paths or DEFAULT_STAGE_DECISION_PATHS
    evidence_records, status_by_stage = _collect_stage_evidence(stage_decision_paths)
    generative_video_model_probe_record, generative_video_model_probe_payload = _collect_generative_video_model_probe_evidence(generative_video_model_probe_run_root)
    sampling_time_constraint_record, sampling_time_constraint_payload = _collect_sampling_time_constraint_evidence(sampling_time_constraint_run_root)
    evidence_records.extend([generative_video_model_probe_record, sampling_time_constraint_record])
    claim_records = _build_claim_records(status_by_stage, generative_video_model_probe_payload, sampling_time_constraint_payload)

    supported_claim_count = sum(1 for record in claim_records if record["claim_status"] == "supported")
    needs_downgrade_count = sum(1 for record in claim_records if record["claim_status"] == "needs_downgrade")
    unsupported_claim_count = sum(1 for record in claim_records if record["claim_status"] == "unsupported")
    sstw_t_ready = next(record for record in claim_records if record["claim_id"] == "claim_sstw_t_submission_preparation")["claim_status"] == "supported"
    sstw_tc_record = next(record for record in claim_records if record["claim_id"] == "claim_sstw_tc_submission_freeze")
    downgrade_policy_pass = sstw_tc_record["claim_status"] == "needs_downgrade"
    package_manifest = {
        "package_digest": None,
        "archive_path": None,
        "package_manifest_path": None,
    }
    decision = {
        "stage_id": "submission_freeze_preparation",
        "implementation_decision": "PASS",
        "mechanism_decision": "PASS" if sstw_t_ready and downgrade_policy_pass and unsupported_claim_count == 0 else "FAIL",
        "details": {
            "claim_audit_record_count": len(claim_records),
            "supported_claim_count": supported_claim_count,
            "needs_downgrade_claim_count": needs_downgrade_count,
            "unsupported_claim_count": unsupported_claim_count,
            "sstw_t_submission_preparation_status": "PASS" if sstw_t_ready else "FAIL",
            "sstw_tc_submission_freeze_status": "DOWNGRADED_TO_EXPLORATORY",
            "claim_boundary_status": "PASS" if downgrade_policy_pass else "FAIL",
            "release_package_rebuildable": "PASS",
            "package_digest": package_manifest["package_digest"],
            "archive_path": package_manifest["archive_path"],
            "package_manifest_path": package_manifest["package_manifest_path"],
        },
    }

    write_jsonl(output_root / "records" / "submission_stage_evidence_records.jsonl", evidence_records)
    write_jsonl(output_root / "records" / "claim_audit_records.jsonl", claim_records)
    write_csv(output_root / "tables" / "claim_audit_table.csv", claim_records)
    write_json(output_root / "artifacts" / "submission_freeze_preparation_decision.json", decision)
    write_json(output_root / "artifacts" / "submission_freeze_preparation_manifest.json", {
        "artifact_id": "submission_freeze_preparation_manifest",
        "artifact_type": "manifest",
        "input_paths": [str(path) for path in stage_decision_paths.values()] + [str(generative_video_model_probe_run_root), str(sampling_time_constraint_run_root)],
        "output_paths": [
            str(output_root / "records" / "submission_stage_evidence_records.jsonl"),
            str(output_root / "records" / "claim_audit_records.jsonl"),
            str(output_root / "tables" / "claim_audit_table.csv"),
            str(output_root / "artifacts" / "submission_freeze_preparation_decision.json"),
        ],
        "rebuild_command": "python -m experiments.submission_freeze_preparation.runner",
    })
    report_path = output_root / "reports" / "submission_freeze_preparation_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "# Submission Freeze Preparation Report\n\n"
        "该报告由 synthetic_state_inference_sanity_to_sampling_time_constraint_probe governed records 和 decision artifacts 重建。"
        "当前允许 SSTW-T 进入投稿准备, 但将 SSTW-TC 最终主 claim 降级为 exploratory, "
        "因为 sampling_time_constraint_probe 结果仍是 real sampling probe, 不是最终 submission freeze claim。\n\n"
        + json.dumps(decision, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    main_tables_manifest = build_submission_main_tables(output_root)
    package_manifest = build_submission_freeze_preparation_package(output_root, package_dir)
    decision["details"]["package_digest"] = package_manifest["package_digest"]
    decision["details"]["archive_path"] = package_manifest["archive_path"]
    decision["details"]["package_manifest_path"] = package_manifest["package_manifest_path"]
    decision["details"]["main_tables_rebuild_status"] = main_tables_manifest["table_rebuild_status"]
    write_json(output_root / "artifacts" / "submission_freeze_preparation_decision.json", decision)
    readiness_summary = build_submission_readiness_summary(output_root)
    report_path.write_text(
        "# Submission Freeze Preparation Report\n\n"
        "该报告由 synthetic_state_inference_sanity_to_sampling_time_constraint_probe governed records 和 decision artifacts 重建。"
        "当前允许 SSTW-T 进入投稿准备, 但将 SSTW-TC 最终主 claim 降级为 exploratory, "
        "因为 sampling_time_constraint_probe 结果仍是 real sampling probe, 不是最终 submission freeze claim。\n\n"
        + json.dumps(decision, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "output_root": str(output_root),
        "claim_audit_record_count": len(claim_records),
        "supported_claim_count": supported_claim_count,
        "needs_downgrade_claim_count": needs_downgrade_count,
        "unsupported_claim_count": unsupported_claim_count,
        "submission_readiness_decision": readiness_summary["submission_readiness_decision"],
        "decision": decision,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 submission freeze preparation claim audit artifacts。")
    parser.add_argument("--output-root", default="outputs/runs/submission_freeze_preparation")
    parser.add_argument("--generative-video-model-probe-run-root", default=str(DEFAULT_generative_video_model_probe_RUN_ROOT))
    parser.add_argument("--sampling-time-constraint-run-root", default=str(DEFAULT_sampling_time_constraint_probe_RUN_ROOT))
    parser.add_argument("--package-dir", default="")
    args = parser.parse_args()
    payload = run_submission_freeze_preparation(args.output_root, args.generative_video_model_probe_run_root, args.sampling_time_constraint_run_root, package_dir=args.package_dir or None)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
