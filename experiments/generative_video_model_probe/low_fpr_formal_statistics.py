"""低 FPR 正式统计的 validation_scale 阻断记录。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from main.protocol.flow_evidence_fields import with_flow_evidence_protocol_defaults
from main.protocol.record_writer import write_json, write_jsonl
from main.protocol.table_builder import write_csv


DEFAULT_VALIDATION_CONFIG = "configs/protocol/validation_scale_generative_probe.json"
DEFAULT_PILOT_CONFIG = "configs/protocol/pilot_paper_generative_probe.json"
DEFAULT_FULL_CONFIG = "configs/protocol/full_paper_generative_probe.json"


def _read_json(path: Path) -> dict[str, Any]:
    """读取 JSON 配置或 artifact, 文件不存在时返回空对象。"""
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON 顶层必须是对象: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL records, 文件不存在时返回空列表。"""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _negative_event_count(run_root: Path) -> int:
    """估计当前 run_root 中可用于低 FPR 的 negative event 数量。"""
    count = 0
    for record in _read_jsonl(run_root / "records" / "generation_records.jsonl"):
        role_fields = {
            str(record.get("sample_role") or "").lower(),
            str(record.get("negative_family") or "").lower(),
            str(record.get("motion_calibration_role") or "").lower(),
        }
        if any("negative" in value for value in role_fields):
            count += 1
    return count


def _target_rows(validation_config: dict[str, Any], pilot_config: dict[str, Any], full_config: dict[str, Any]) -> list[dict[str, Any]]:
    """从 pilot/full protocol 中抽取 validation_scale 不能声称的低 FPR 目标。"""
    rows: list[dict[str, Any]] = []
    for profile_name, config in (("pilot_paper", pilot_config), ("full_paper", full_config)):
        if "target_fpr" not in config:
            continue
        rows.append({
            "blocked_result_profile": profile_name,
            "blocked_target_fpr": float(config["target_fpr"]),
            "minimum_negative_event_count_required": int(
                config.get("minimum_heldout_test_negative_event_count")
                or config.get("minimum_calibration_negative_event_count")
                or 0
            ),
            "threshold_protocol_required": str(config.get("threshold_protocol") or "calibration_split_to_frozen_threshold_to_heldout_test_split"),
        })
    if "blocked_target_fpr" in validation_config:
        rows.append({
            "blocked_result_profile": "validation_config_blocked_target",
            "blocked_target_fpr": float(validation_config["blocked_target_fpr"]),
            "minimum_negative_event_count_required": 0,
            "threshold_protocol_required": "declared_in_validation_config",
        })
    return rows


def build_low_fpr_formal_statistics_records(
    run_root: str | Path,
    config_path: str | Path = DEFAULT_VALIDATION_CONFIG,
    pilot_config_path: str | Path = DEFAULT_PILOT_CONFIG,
    full_config_path: str | Path = DEFAULT_FULL_CONFIG,
) -> list[dict[str, Any]]:
    """构建低 FPR 正式统计阻断 records。"""
    run_root = Path(run_root)
    validation_config = _read_json(Path(config_path))
    if "target_fpr" not in validation_config:
        raise KeyError(f"validation config 缺少 target_fpr: {config_path}")
    pilot_config = _read_json(Path(pilot_config_path))
    full_config = _read_json(Path(full_config_path))
    current_target_fpr = float(validation_config["target_fpr"])
    observed_negative_count = _negative_event_count(run_root)
    records: list[dict[str, Any]] = []
    for row in _target_rows(validation_config, pilot_config, full_config):
        status = "blocked_by_validation_scale_sample_size_and_result_level"
        records.append(with_flow_evidence_protocol_defaults({
            "record_version": "low_fpr_formal_statistics_blocking_v1",
            "paper_result_level": validation_config.get("paper_result_level", "validation_scale"),
            "current_target_fpr": current_target_fpr,
            "blocked_result_profile": row["blocked_result_profile"],
            "blocked_target_fpr": row["blocked_target_fpr"],
            "low_fpr_formal_statistics_status": status,
            "formal_low_fpr_claim_allowed": False,
            "observed_negative_event_count": observed_negative_count,
            "minimum_negative_event_count_required": row["minimum_negative_event_count_required"],
            "threshold_protocol_required": row["threshold_protocol_required"],
            "low_fpr_blocking_reason": (
                "validation_scale 只验证小样本全流程闭环, 不能替代 pilot_paper 或 full_paper "
                "所需的 calibration split、held-out negative split 和低 FPR 统计。"
            ),
            "claim_support_status": "low_fpr_formal_statistics_blocking_record",
        }, trajectory_source_level="low_fpr_formal_statistics_blocking_record", claim_support_status="low_fpr_formal_statistics_blocking_record"))
    return records


def audit_low_fpr_formal_statistics_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """审计低 FPR 阻断记录是否已明确落盘。"""
    blocked_targets = sorted({float(record["blocked_target_fpr"]) for record in records if record.get("blocked_target_fpr") is not None})
    all_claims_blocked = all(record.get("formal_low_fpr_claim_allowed") is False for record in records)
    decision = "PASS" if records and all_claims_blocked else "FAIL"
    return {
        "stage_id": "low_fpr_formal_statistics",
        "low_fpr_formal_statistics_decision": decision,
        "claim_support_status": "low_fpr_formal_statistics_blocking_record" if decision == "PASS" else "low_fpr_formal_statistics_blocked_record_missing",
        "low_fpr_formal_statistics_record_count": len(records),
        "low_fpr_blocked_target_fprs": blocked_targets,
        "formal_low_fpr_claim_allowed": False,
        "low_fpr_formal_statistics_status": "blocking_record_ready" if decision == "PASS" else "blocking_record_missing",
    }


def run_low_fpr_formal_statistics(
    run_root: str | Path,
    config_path: str | Path = DEFAULT_VALIDATION_CONFIG,
) -> dict[str, Any]:
    """写出低 FPR 正式统计阻断 records、table、decision 和 report。"""
    run_root = Path(run_root)
    records = build_low_fpr_formal_statistics_records(run_root, config_path)
    audit = audit_low_fpr_formal_statistics_records(records)
    write_jsonl(run_root / "records" / "low_fpr_formal_statistics_records.jsonl", records)
    write_csv(run_root / "tables" / "low_fpr_formal_statistics_table.csv", records)
    write_json(run_root / "artifacts" / "low_fpr_formal_statistics_decision.json", audit)
    report = (
        "# Low FPR Formal Statistics Blocking Report\n\n"
        "该报告明确记录 validation_scale 不能支持低 FPR 正式统计主张。低 FPR 主张必须在 "
        "pilot_paper 或 full_paper 的 calibration / held-out negative split 上重新运行并通过门禁。\n\n"
        f"- low_fpr_formal_statistics_decision: {audit['low_fpr_formal_statistics_decision']}\n"
        f"- low_fpr_blocked_target_fprs: {', '.join(str(item) for item in audit['low_fpr_blocked_target_fprs'])}\n"
        f"- formal_low_fpr_claim_allowed: {str(audit['formal_low_fpr_claim_allowed']).lower()}\n"
        f"- claim_support_status: {audit['claim_support_status']}\n"
    )
    report_path = run_root / "reports" / "low_fpr_formal_statistics_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="生成低 FPR 正式统计阻断记录。")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--config-path", default=DEFAULT_VALIDATION_CONFIG)
    args = parser.parse_args()
    payload = run_low_fpr_formal_statistics(args.run_root, args.config_path)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
