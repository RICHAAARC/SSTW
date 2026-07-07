"""低 FPR 正式统计的跨 profile 治理记录。

validation_scale、pilot_paper 和 full_paper 使用同一套构建逻辑。区别只来自
protocol config 中的 `target_fpr`、样本规模和 negative event 要求。这样可以
保证 Notebook 只切换 profile, 不承担低 FPR 统计口径的业务逻辑。
"""

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


def _generation_negative_event_count(run_root: Path) -> int:
    """估计 generation records 中可用于低 FPR 的 negative event 数量。"""

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


def _fair_calibration_negative_event_count(run_root: Path) -> int:
    """从公平校准 records 估计每个方法都具备的 clean negative 下界。"""

    counts = [
        int(record.get("clean_negative_score_count") or 0)
        for record in _read_jsonl(run_root / "records" / "fair_detection_calibration_records.jsonl")
        if record.get("fair_comparison_status") == "ready"
    ]
    return min(counts) if counts else 0


def _negative_event_count(run_root: Path) -> int:
    """返回低 FPR 统计可使用的 negative event 保守估计。"""

    return max(_generation_negative_event_count(run_root), _fair_calibration_negative_event_count(run_root))


def _minimum_negative_event_required(config: dict[str, Any]) -> int:
    """读取某个 profile 的最低 negative event 要求。"""

    return int(
        config.get("minimum_heldout_test_negative_event_count")
        or config.get("minimum_calibration_negative_event_count")
        or config.get("minimum_clean_negative_count")
        or 0
    )


def _target_rows(current_config: dict[str, Any], pilot_config: dict[str, Any], full_config: dict[str, Any]) -> list[dict[str, Any]]:
    """从当前 profile、pilot 和 full protocol 中抽取低 FPR 目标。"""

    rows: list[dict[str, Any]] = []
    current_profile = str(current_config.get("paper_result_level") or "validation_scale")
    if "target_fpr" in current_config:
        rows.append({
            "result_profile": current_profile,
            "target_fpr": float(current_config["target_fpr"]),
            "row_role": "current_profile_target",
            "minimum_negative_event_count_required": _minimum_negative_event_required(current_config),
            "threshold_protocol_required": str(current_config.get("threshold_protocol") or "method_specific_clean_negative_calibration_to_target_fpr"),
        })
    for profile_name, config in (("pilot_paper", pilot_config), ("full_paper", full_config)):
        if profile_name == current_profile:
            continue
        if "target_fpr" not in config:
            continue
        rows.append({
            "result_profile": profile_name,
            "target_fpr": float(config["target_fpr"]),
            "row_role": "future_profile_target",
            "minimum_negative_event_count_required": _minimum_negative_event_required(config),
            "threshold_protocol_required": str(config.get("threshold_protocol") or "calibration_split_to_frozen_threshold_to_heldout_test_split"),
        })
    if "blocked_target_fpr" in current_config:
        rows.append({
            "result_profile": f"{current_profile}_blocked_target",
            "target_fpr": float(current_config["blocked_target_fpr"]),
            "row_role": "blocked_target_declared_by_current_config",
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
    """构建低 FPR 正式统计 records。"""

    run_root = Path(run_root)
    current_config = _read_json(Path(config_path))
    if "target_fpr" not in current_config:
        raise KeyError(f"protocol config 缺少 target_fpr: {config_path}")
    pilot_config = _read_json(Path(pilot_config_path))
    full_config = _read_json(Path(full_config_path))
    current_profile = str(current_config.get("paper_result_level") or "validation_scale")
    current_target_fpr = float(current_config["target_fpr"])
    observed_negative_count = _negative_event_count(run_root)
    records: list[dict[str, Any]] = []
    for row in _target_rows(current_config, pilot_config, full_config):
        is_current_target = row["row_role"] == "current_profile_target"
        enough_negative = observed_negative_count >= int(row["minimum_negative_event_count_required"])
        formal_allowed = bool(is_current_target and enough_negative)
        if formal_allowed:
            status = "current_profile_low_fpr_statistics_ready"
            claim_status = f"{current_profile}_target_fpr_statistics_ready"
            blocking_reason = "none"
        elif is_current_target:
            status = "blocked_by_current_profile_negative_sample_size"
            claim_status = "low_fpr_formal_statistics_current_profile_blocking_record"
            blocking_reason = (
                "当前 profile 的 negative event 数量不足, 不能在当前 target_fpr 下报告正式低 FPR 主张。"
            )
        else:
            status = "blocked_until_matching_future_profile_run"
            claim_status = "low_fpr_formal_statistics_blocking_record"
            blocking_reason = (
                "该 FPR 等级需要切换到对应 workflow profile 后, 使用同一低 FPR 统计逻辑在更大样本上重新运行。"
            )
        records.append(with_flow_evidence_protocol_defaults({
            "record_version": "low_fpr_formal_statistics_v2",
            "paper_result_level": current_profile,
            "current_target_fpr": current_target_fpr,
            "result_profile": row["result_profile"],
            "target_fpr": row["target_fpr"],
            "blocked_result_profile": row["result_profile"],
            "blocked_target_fpr": row["target_fpr"],
            "low_fpr_formal_statistics_status": status,
            "formal_low_fpr_claim_allowed": formal_allowed,
            "observed_negative_event_count": observed_negative_count,
            "minimum_negative_event_count_required": row["minimum_negative_event_count_required"],
            "threshold_protocol_required": row["threshold_protocol_required"],
            "low_fpr_blocking_reason": blocking_reason,
            "claim_support_status": claim_status,
        }, trajectory_source_level="low_fpr_formal_statistics_governed_record", claim_support_status=claim_status))
    return records


def audit_low_fpr_formal_statistics_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """审计低 FPR records 是否已明确落盘。"""

    blocked_targets = sorted({
        float(record["target_fpr"])
        for record in records
        if record.get("formal_low_fpr_claim_allowed") is False and record.get("target_fpr") is not None
    })
    current_rows = [record for record in records if record.get("result_profile") == record.get("paper_result_level")]
    current_allowed = any(record.get("formal_low_fpr_claim_allowed") is True for record in current_rows)
    current_blocked = any(record.get("formal_low_fpr_claim_allowed") is False for record in current_rows)
    decision = "PASS" if records and (current_allowed or current_blocked) else "FAIL"
    if current_allowed:
        claim_status = "low_fpr_formal_statistics_current_profile_ready"
        status = "current_profile_statistics_ready"
    elif decision == "PASS":
        claim_status = "low_fpr_formal_statistics_blocking_record"
        status = "blocking_record_ready"
    else:
        claim_status = "low_fpr_formal_statistics_blocked_record_missing"
        status = "blocking_record_missing"
    return {
        "stage_id": "low_fpr_formal_statistics",
        "low_fpr_formal_statistics_decision": decision,
        "claim_support_status": claim_status,
        "low_fpr_formal_statistics_record_count": len(records),
        "low_fpr_blocked_target_fprs": blocked_targets,
        "formal_low_fpr_claim_allowed": current_allowed,
        "current_profile_low_fpr_claim_allowed": current_allowed,
        "current_profile_low_fpr_claim_blocked": current_blocked,
        "low_fpr_formal_statistics_status": status,
    }


def run_low_fpr_formal_statistics(
    run_root: str | Path,
    config_path: str | Path = DEFAULT_VALIDATION_CONFIG,
) -> dict[str, Any]:
    """写出低 FPR 正式统计 records、table、decision 和 report。"""
    run_root = Path(run_root)
    records = build_low_fpr_formal_statistics_records(run_root, config_path)
    audit = audit_low_fpr_formal_statistics_records(records)
    write_jsonl(run_root / "records" / "low_fpr_formal_statistics_records.jsonl", records)
    write_csv(run_root / "tables" / "low_fpr_formal_statistics_table.csv", records)
    write_json(run_root / "artifacts" / "low_fpr_formal_statistics_decision.json", audit)
    report = (
        "# Low FPR Formal Statistics Report\n\n"
        "该报告按当前 protocol config 的 target_fpr 写出低 FPR 统计状态。"
        "validation_scale、pilot_paper 和 full_paper 使用同一脚本, 只由配置决定 FPR 等级和样本规模。\n\n"
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
