"""从已经通过门禁的 governed artifacts 构建审稿证据索引."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from evaluation.protocol.record_writer import write_json, write_jsonl
from evaluation.protocol.table_builder import write_csv


CLAIM_EVIDENCE_PATHS: dict[str, tuple[str, ...]] = {
    "claim_1_velocity_constraint_detectable_watermark": (
        "artifacts/formal_flow_evidence_decision.json",
        "records/formal_flow_evidence_records.jsonl",
        "records/validation_internal_ablation_records.jsonl",
        "artifacts/statistical_confidence_interval_decision.json",
        "reports/formal_flow_evidence_report.md",
    ),
    "claim_2_path_evidence_independent_gain": (
        "artifacts/formal_flow_evidence_decision.json",
        "records/paired_path_evidence_gain_records.jsonl",
        "tables/paired_path_evidence_gain_table.csv",
        "artifacts/statistical_confidence_interval_decision.json",
        "reports/formal_flow_evidence_report.md",
    ),
    "claim_3_attacked_video_replay_posterior": (
        "artifacts/replay_and_sketch_gate_decision.json",
        "records/replay_uncertainty_records.jsonl",
        "records/wrong_key_replay_records.jsonl",
        "records/wrong_sampler_replay_records.jsonl",
        "records/wrong_prompt_replay_records.jsonl",
        "records/wrong_time_grid_replay_records.jsonl",
        "reports/replay_and_sketch_gate_report.md",
    ),
    "supportive_cross_model_generalization": (
        "artifacts/cross_model_generalization_decision.json",
        "tables/cross_model_generalization_table.csv",
        "records/formal_flow_evidence_records.jsonl",
        "reports/formal_flow_evidence_report.md",
    ),
}

CLAIM_DECISION_FIELDS = {
    "claim_1_velocity_constraint_detectable_watermark": "claim_1_velocity_constraint_detectable_watermark_decision",
    "claim_2_path_evidence_independent_gain": "claim_2_path_evidence_independent_gain_decision",
    "claim_3_attacked_video_replay_posterior": "claim_3_attacked_video_replay_posterior_decision",
    "supportive_cross_model_generalization": "cross_model_generalization_decision",
}

CLAIM_DECISION_SOURCE_PATHS = {
    "supportive_cross_model_generalization": "artifacts/cross_model_generalization_decision.json",
}


def _read_json(path: Path) -> dict[str, Any]:
    """读取 JSON 对象, 文件不存在时返回空对象."""

    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON 顶层必须是对象: {path}")
    return payload


def _sha256(path: Path) -> str:
    """计算 artifact 内容摘要, 便于审稿时核对索引与文件是否一致."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _current_profile_gate(run_root: Path) -> tuple[str, str, dict[str, Any]]:
    """按正式阶段顺序读取当前 profile 的最终门禁."""

    candidates = (
        ("full_paper", "artifacts/full_paper_gate_decision.json"),
        ("full_paper", "artifacts/full_paper_result_checker_decision.json"),
        ("pilot_paper", "artifacts/pilot_paper_gate_decision.json"),
        ("probe_paper", "artifacts/probe_paper_gate_decision.json"),
    )
    for profile, relative_path in candidates:
        payload = _read_json(run_root / relative_path)
        if payload:
            return profile, relative_path, payload
    return "unknown", "", {}


def _profile_gate_passed(profile: str, gate: Mapping[str, Any]) -> bool:
    """判断当前 profile 的规范门禁是否为 PASS."""

    fields = {
        "probe_paper": ("probe_paper_gate_decision", "paper_profile_gate_decision"),
        "pilot_paper": ("pilot_paper_gate_decision",),
        "full_paper": (
            "full_paper_gate_decision",
            "paper_profile_gate_decision",
            "full_paper_result_checker_decision",
            "full_paper_result_decision",
        ),
    }.get(profile, ())
    return any(gate.get(field) == "PASS" for field in fields)


def build_reviewer_evidence_index(run_root: str | Path) -> dict[str, Any]:
    """构建三层主 claim 与跨模型支持性 claim 到真实 artifact 的可验证映射."""

    run_root = Path(run_root)
    profile, gate_path, gate = _current_profile_gate(run_root)
    complete_claim = _read_json(run_root / "artifacts/complete_paper_mechanism_claim_decision.json")
    rows: list[dict[str, Any]] = []
    missing_paths: list[str] = []
    failed_claims: list[str] = []

    for claim_id, relative_paths in CLAIM_EVIDENCE_PATHS.items():
        decision_field = CLAIM_DECISION_FIELDS[claim_id]
        decision_source_path = CLAIM_DECISION_SOURCE_PATHS.get(claim_id)
        decision_payload = (
            _read_json(run_root / decision_source_path)
            if decision_source_path
            else complete_claim
        )
        claim_passed = decision_payload.get(decision_field) == "PASS"
        if not claim_passed:
            failed_claims.append(claim_id)
        for relative_path in relative_paths:
            artifact_path = run_root / relative_path
            exists = artifact_path.is_file()
            if not exists:
                missing_paths.append(relative_path)
            rows.append({
                "paper_profile": profile,
                "claim_id": claim_id,
                "claim_decision_field": decision_field,
                "claim_decision": decision_payload.get(decision_field, "MISSING"),
                "claim_decision_source_path": (
                    decision_source_path
                    or "artifacts/complete_paper_mechanism_claim_decision.json"
                ),
                "evidence_path": relative_path,
                "evidence_exists": exists,
                "evidence_sha256": _sha256(artifact_path) if exists else None,
                "evidence_source": "governed_artifact",
            })

    gate_passed = _profile_gate_passed(profile, gate)
    complete_claim_passed = complete_claim.get("complete_paper_mechanism_claim_decision") == "PASS"
    decision = (
        "PASS"
        if gate_passed and complete_claim_passed and not missing_paths and not failed_claims
        else "FAIL"
    )
    return {
        "stage_id": f"{profile}_reviewer_evidence_index" if profile != "unknown" else "reviewer_evidence_index",
        "paper_profile": profile,
        "profile_gate_path": gate_path,
        "profile_gate_passed": gate_passed,
        "complete_paper_mechanism_claim_decision": complete_claim.get(
            "complete_paper_mechanism_claim_decision", "MISSING"
        ),
        "reviewer_evidence_index_decision": decision,
        "claim_support_status": (
            f"{profile}_reviewer_evidence_index_ready"
            if decision == "PASS"
            else f"{profile}_reviewer_evidence_index_blocked"
        ),
        "indexed_claim_count": len(CLAIM_EVIDENCE_PATHS),
        "indexed_artifact_count": len(rows),
        "failed_claim_ids": sorted(set(failed_claims)),
        "missing_evidence_paths": sorted(set(missing_paths)),
        "evidence_rows": rows,
    }


def write_reviewer_evidence_index(run_root: str | Path) -> dict[str, Any]:
    """写出 records、table、decision 和审稿人可读报告."""

    run_root = Path(run_root)
    index = build_reviewer_evidence_index(run_root)
    rows = index["evidence_rows"]
    write_jsonl(run_root / "records" / "reviewer_evidence_index_records.jsonl", rows)
    write_csv(run_root / "tables" / "reviewer_evidence_index_table.csv", rows)
    write_json(
        run_root / "artifacts" / "reviewer_evidence_index_decision.json",
        {key: value for key, value in index.items() if key != "evidence_rows"},
    )
    write_json(run_root / "reports" / "reviewer_evidence_index.json", index)

    lines = [
        "# Reviewer Evidence Index",
        "",
        "该索引只引用已经落盘的 governed artifacts, 不复制或手工填写实验数值.",
        "",
        f"- paper_profile: {index['paper_profile']}",
        f"- profile_gate_passed: {str(index['profile_gate_passed']).lower()}",
        f"- reviewer_evidence_index_decision: {index['reviewer_evidence_index_decision']}",
        f"- failed_claim_ids: {', '.join(index['failed_claim_ids']) if index['failed_claim_ids'] else 'none'}",
        f"- missing_evidence_paths: {', '.join(index['missing_evidence_paths']) if index['missing_evidence_paths'] else 'none'}",
        "",
        "| claim_id | evidence_path | sha256 |",
        "| --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['claim_id']} | {row['evidence_path']} | {row['evidence_sha256'] or 'missing'} |"
        )
    report_path = run_root / "reports" / "reviewer_evidence_index.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return index


def main() -> None:
    """命令行入口."""

    parser = argparse.ArgumentParser(description="构建三层主张与跨模型支持性主张的审稿证据索引.")
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    print(json.dumps(write_reviewer_evidence_index(args.run_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
