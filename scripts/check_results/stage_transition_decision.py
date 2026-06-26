"""生成主干阶段之间的轻量跳转判定。

该脚本只读取已经落盘的 source gate decision artifact, 不运行实验、不生成分数,
也不把上游失败解释为人工放行。它的作用是把
`validation_scale -> pilot_paper -> full_paper -> submission_freeze` 的阶段跳转
写成可审计 records, 供下一阶段 gate 消费。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from main.protocol.flow_evidence_fields import with_flow_evidence_protocol_defaults
from main.protocol.record_writer import write_json, write_jsonl
from main.protocol.table_builder import write_csv


TRANSITION_SPECS: dict[str, dict[str, Any]] = {
    "validation_scale_to_pilot_paper": {
        "decision_field": "validation_scale_to_pilot_paper_transition_decision",
        "source_stage": "validation_scale",
        "target_stage": "pilot_paper",
        "source_gate_path": "artifacts/validation_scale_gate_decision.json",
        "source_gate_fields": ("validation_scale_gate_decision",),
        "allowed_next_result_profiles": ("pilot_paper",),
        "blocked_next_result_profiles": ("full_paper", "submission_freeze"),
        "claim_support_status_pass": "validation_scale_ready_to_enter_pilot_paper",
        "claim_support_status_fail": "validation_scale_to_pilot_paper_blocked",
    },
    "pilot_paper_to_full_paper": {
        "decision_field": "pilot_paper_to_full_paper_transition_decision",
        "source_stage": "pilot_paper",
        "target_stage": "full_paper",
        "source_gate_path": "artifacts/pilot_paper_gate_decision.json",
        "source_gate_fields": ("pilot_paper_gate_decision",),
        "required_upstream_transitions": (
            "artifacts/validation_scale_to_pilot_paper_transition_decision.json",
        ),
        "required_upstream_transition_fields": (
            "validation_scale_to_pilot_paper_transition_decision",
        ),
        "allowed_next_result_profiles": ("full_paper",),
        "blocked_next_result_profiles": ("submission_freeze",),
        "claim_support_status_pass": "pilot_paper_ready_to_enter_full_paper",
        "claim_support_status_fail": "pilot_paper_to_full_paper_blocked",
    },
    "full_paper_to_submission_freeze": {
        "decision_field": "full_paper_to_submission_freeze_transition_decision",
        "source_stage": "full_paper",
        "target_stage": "submission_freeze",
        "source_gate_path": "artifacts/full_paper_result_checker_decision.json",
        "alternate_source_gate_paths": (
            "artifacts/full_paper_result_decision.json",
        ),
        "source_gate_fields": (
            "full_paper_result_checker_decision",
            "full_paper_result_decision",
        ),
        "required_upstream_transitions": (
            "artifacts/pilot_paper_to_full_paper_transition_decision.json",
        ),
        "required_upstream_transition_fields": (
            "pilot_paper_to_full_paper_transition_decision",
        ),
        "allowed_next_result_profiles": ("submission_freeze",),
        "blocked_next_result_profiles": (),
        "claim_support_status_pass": "full_paper_ready_to_enter_submission_freeze",
        "claim_support_status_fail": "full_paper_to_submission_freeze_blocked",
        "extra_pass_predicates": ("full_paper_claim_allowed",),
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    """读取 UTF-8 JSON artifact, 文件不存在时返回空对象。"""
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON 顶层必须是对象: {path}")
    return payload


def _decision_pass(payload: Mapping[str, Any], fields: tuple[str, ...]) -> bool:
    """检查任一指定字段是否显式为 PASS。"""
    return any(payload.get(field) == "PASS" for field in fields)


def _resolve_source_gate_path(run_root: Path, spec: Mapping[str, Any]) -> Path:
    """解析 source gate decision artifact 路径, 支持未来 full_paper 的兼容文件名。"""
    candidates = [run_root / str(spec["source_gate_path"])]
    candidates.extend(run_root / str(path) for path in spec.get("alternate_source_gate_paths", ()))
    return next((path for path in candidates if path.exists()), candidates[0])


def _upstream_transition_status(
    run_root: Path,
    spec: Mapping[str, Any],
) -> tuple[bool, list[dict[str, Any]], list[str]]:
    """检查当前跳转所依赖的更早跳转是否已经 PASS。"""
    paths = tuple(str(item) for item in spec.get("required_upstream_transitions", ()))
    fields = tuple(str(item) for item in spec.get("required_upstream_transition_fields", ()))
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    if not paths:
        return True, rows, missing
    for index, relative_path in enumerate(paths):
        field = fields[index] if index < len(fields) else ""
        path = _resolve_cross_profile_artifact_path(run_root, relative_path, field)
        payload = _read_json(path)
        passed = bool(field and payload.get(field) == "PASS")
        rows.append({
            "upstream_transition_path": relative_path,
            "upstream_transition_field": field,
            "upstream_transition_decision": payload.get(field),
            "upstream_transition_passed": passed,
        })
        if not passed:
            missing.append(f"{field}_passed")
    return not missing, rows, missing


def _resolve_cross_profile_artifact_path(run_root: Path, relative_path: str, field: str) -> Path:
    """解析跨 profile 的上游轻量判定 artifact。

    Colab 的 `validation_scale`、`pilot_paper` 和 `full_paper` 使用相互隔离的
    run_root。目标阶段 gate 可以消费上一阶段 artifact, 但不能要求用户手工复制文件。
    因此这里先查当前 run_root, 再查同级 profile run_root。
    """
    local_path = run_root / relative_path
    if local_path.exists():
        return local_path
    profile_by_field = {
        "validation_scale_to_pilot_paper_transition_decision": "validation_scale",
        "pilot_paper_to_full_paper_transition_decision": "pilot_paper",
    }
    source_profile = profile_by_field.get(field)
    if source_profile:
        sibling_path = run_root.parent / source_profile / relative_path
        if sibling_path.exists():
            return sibling_path
    return local_path


def build_stage_transition_decision(
    run_root: str | Path,
    transition_id: str,
) -> dict[str, Any]:
    """构建某个阶段跳转的轻量判定。

    该函数属于项目特定写法。它把“上游 gate 已经 PASS”和“下一阶段允许进入”
    拆开记录, 防止把 `validation_scale` PASS 误读为可以直接进入 `full_paper`。
    """
    if transition_id not in TRANSITION_SPECS:
        raise KeyError(f"未知阶段跳转: {transition_id}")
    run_root = Path(run_root)
    spec = TRANSITION_SPECS[transition_id]
    decision_field = str(spec["decision_field"])
    source_gate_path = _resolve_source_gate_path(run_root, spec)
    source_payload = _read_json(source_gate_path)
    source_passed = _decision_pass(source_payload, tuple(spec["source_gate_fields"]))
    upstream_passed, upstream_rows, upstream_missing = _upstream_transition_status(run_root, spec)

    missing_requirements: list[str] = []
    if not source_passed:
        missing_requirements.append(f"{spec['source_stage']}_source_gate_passed")
    missing_requirements.extend(upstream_missing)

    for predicate in spec.get("extra_pass_predicates", ()):
        if source_payload.get(str(predicate)) is not True:
            missing_requirements.append(f"{predicate}_true")

    if transition_id == "validation_scale_to_pilot_paper":
        if source_payload.get("full_paper_allowed") is True:
            missing_requirements.append("validation_scale_must_not_directly_allow_full_paper")

    transition_decision = "PASS" if not missing_requirements and upstream_passed else "FAIL"
    claim_support_status = (
        spec["claim_support_status_pass"]
        if transition_decision == "PASS"
        else spec["claim_support_status_fail"]
    )
    return {
        "stage_id": "stage_transition_decision",
        "transition_id": transition_id,
        "transition_decision_field": decision_field,
        decision_field: transition_decision,
        "source_stage": spec["source_stage"],
        "target_stage": spec["target_stage"],
        "source_gate_decision_path": str(source_gate_path),
        "source_gate_fields": list(spec["source_gate_fields"]),
        "source_gate_passed": source_passed,
        "source_gate_decisions": {
            field: source_payload.get(field)
            for field in spec["source_gate_fields"]
        },
        "required_upstream_transition_results": upstream_rows,
        "missing_transition_requirements": missing_requirements,
        "transition_missing_requirement_count": len(missing_requirements),
        "allowed_next_result_profiles": list(spec["allowed_next_result_profiles"]),
        "blocked_next_result_profiles": list(spec["blocked_next_result_profiles"]),
        "full_paper_allowed": transition_id == "pilot_paper_to_full_paper" and transition_decision == "PASS",
        "submission_freeze_allowed": transition_id == "full_paper_to_submission_freeze" and transition_decision == "PASS",
        "claim_support_status": claim_support_status,
    }


def write_stage_transition_decision(
    run_root: str | Path,
    transition_id: str,
) -> dict[str, Any]:
    """写出阶段跳转 decision、record、table 和报告。"""
    run_root = Path(run_root)
    audit = build_stage_transition_decision(run_root, transition_id)
    decision_field = str(audit["transition_decision_field"])
    record = with_flow_evidence_protocol_defaults(
        {"record_version": "stage_transition_decision_v1", **audit},
        trajectory_source_level="stage_transition_governed_decision",
        flow_state_admissibility_status="not_applicable",
        claim_support_status=audit["claim_support_status"],
    )
    write_jsonl(run_root / "records" / f"{decision_field}_records.jsonl", [record])
    write_csv(run_root / "tables" / f"{decision_field}_table.csv", [record])
    write_json(run_root / "artifacts" / f"{decision_field}.json", audit)
    report = (
        "# Stage Transition Decision Report\n\n"
        "该报告由 source gate decision artifact 自动生成, 不运行实验, 不人工放行。\n\n"
        f"- transition_id: {audit['transition_id']}\n"
        f"- {decision_field}: {audit[decision_field]}\n"
        f"- source_stage: {audit['source_stage']}\n"
        f"- target_stage: {audit['target_stage']}\n"
        f"- missing_transition_requirements: "
        f"{', '.join(audit['missing_transition_requirements']) if audit['missing_transition_requirements'] else 'none'}\n"
        f"- allowed_next_result_profiles: {', '.join(audit['allowed_next_result_profiles']) or 'none'}\n"
        f"- blocked_next_result_profiles: {', '.join(audit['blocked_next_result_profiles']) or 'none'}\n"
    )
    report_path = run_root / "reports" / f"{decision_field}_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="生成主干阶段轻量跳转判定。")
    parser.add_argument("--run-root", required=True)
    parser.add_argument(
        "--transition",
        required=True,
        choices=sorted(TRANSITION_SPECS),
    )
    args = parser.parse_args()
    payload = write_stage_transition_decision(args.run_root, args.transition)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
