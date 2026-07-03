"""validation-scale generative video probe 的自动门禁审计。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from main.external_baselines.baseline_registry import audit_external_baseline_records
from experiments.generative_video_model_probe.external_baseline_runner import audit_external_baseline_comparison_records
from experiments.generative_video_model_probe.formal_motion_claim_filter import (
    FORMAL_MOTION_CLAIM_READY_STATUSES,
    select_motion_claim_generation_records,
)
from main.protocol.flow_evidence_fields import with_flow_evidence_protocol_defaults
from main.protocol.record_writer import write_json, write_jsonl
from main.protocol.table_builder import write_csv


DEFAULT_VALIDATION_SCALE_CONFIG = "configs/protocol/validation_scale_generative_probe.json"
DEFAULT_VALIDATION_PROFILE_NAMES = {"validation_scale"}
DEFAULT_MINIMUM_PROMPT_COUNT = 8
DEFAULT_MINIMUM_SEED_PER_PROMPT = 3
DEFAULT_MINIMUM_ATTACK_COUNT = 3
DEFAULT_REQUIRED_MODERN_EXTERNAL_BASELINE_ADAPTER_NAMES = (
    "videoshield",
    "sigmark",
    "videomark",
    "vidsig",
    "videoseal",
)
DEFAULT_MINIMUM_EXTERNAL_BASELINE_MEASURED_ADAPTER_COUNT = 7
DEFAULT_MINIMUM_MODERN_EXTERNAL_BASELINE_FORMAL_ADAPTER_COUNT = len(DEFAULT_REQUIRED_MODERN_EXTERNAL_BASELINE_ADAPTER_NAMES)


def _read_json(path: Path) -> dict:
    """读取 JSON 文件, 并兼容 Windows 或 Colab 产生的 UTF-8 BOM。"""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _required_float(raw: dict, field_name: str, config_path: Path) -> float:
    """从 validation-scale protocol config 读取必填 float 字段。"""
    if field_name not in raw:
        raise KeyError(f"validation-scale protocol config 缺少必填字段 {field_name}: {config_path}")
    return float(raw[field_name])


def _format_fpr(value: float | None) -> str:
    """把 FPR 数值格式化为报告中的稳定短文本。"""
    if value is None:
        return "未配置"
    return f"{float(value):g}"


def _read_jsonl(path: Path) -> list[dict]:
    """读取 JSONL records, 文件不存在时返回空列表。"""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_config(config_path: str | Path = DEFAULT_VALIDATION_SCALE_CONFIG) -> dict:
    """读取 validation-scale 门禁配置。"""
    path = Path(config_path)
    config = _read_json(path)
    return {
        "validation_profile_names": config.get("validation_profile_names", sorted(DEFAULT_VALIDATION_PROFILE_NAMES)),
        "target_fpr": _required_float(config, "target_fpr", path),
        "paper_result_level": config.get("paper_result_level", "validation_scale"),
        "minimum_prompt_count": int(config.get("minimum_prompt_count", DEFAULT_MINIMUM_PROMPT_COUNT)),
        "minimum_seed_per_prompt": int(config.get("minimum_seed_per_prompt", DEFAULT_MINIMUM_SEED_PER_PROMPT)),
        "minimum_attack_count": int(config.get("minimum_attack_count", DEFAULT_MINIMUM_ATTACK_COUNT)),
        "require_external_baseline_status_records": bool(config.get("require_external_baseline_status_records", True)),
        "require_external_baseline_comparison_records": bool(config.get("require_external_baseline_comparison_records", True)),
        "require_external_baseline_self_containment_decision": bool(config.get("require_external_baseline_self_containment_decision", True)),
        "require_sstw_measured_formal_records": bool(config.get("require_sstw_measured_formal_records", True)),
        "require_formal_method_baseline_comparison": bool(config.get("require_formal_method_baseline_comparison", True)),
        "require_formal_baseline_difference_interval": bool(config.get("require_formal_baseline_difference_interval", True)),
        "minimum_external_baseline_measured_adapter_count": int(config.get("minimum_external_baseline_measured_adapter_count", DEFAULT_MINIMUM_EXTERNAL_BASELINE_MEASURED_ADAPTER_COUNT)),
        "minimum_modern_external_baseline_formal_adapter_count": int(config.get("minimum_modern_external_baseline_formal_adapter_count", DEFAULT_MINIMUM_MODERN_EXTERNAL_BASELINE_FORMAL_ADAPTER_COUNT)),
        "required_modern_external_baseline_adapter_names": list(config.get("required_modern_external_baseline_adapter_names", DEFAULT_REQUIRED_MODERN_EXTERNAL_BASELINE_ADAPTER_NAMES)),
        "require_motion_threshold_calibration_ready": bool(config.get("require_motion_threshold_calibration_ready", True)),
        "require_formal_motion_claim_ready": bool(config.get("require_formal_motion_claim_ready", True)),
        "require_internal_ablation_records": bool(config.get("require_internal_ablation_records", True)),
        "require_validation_scale_formal_internal_ablation": bool(config.get("require_validation_scale_formal_internal_ablation", True)),
        "require_adaptive_attack_records": bool(config.get("require_adaptive_attack_records", True)),
        "require_replay_or_sketch_records_or_claim3_downgrade": bool(config.get("require_replay_or_sketch_records_or_claim3_downgrade", True)),
        "require_confidence_interval_report": bool(config.get("require_confidence_interval_report", True)),
        "require_low_fpr_formal_statistics_blocking_record": bool(config.get("require_low_fpr_formal_statistics_blocking_record", True)),
        "require_artifact_rebuild_dry_run": bool(config.get("require_artifact_rebuild_dry_run", True)),
        "require_data_split_and_leakage_guard": bool(config.get("require_data_split_and_leakage_guard", True)),
    }


def _unique_nonempty(records: list[dict], field: str) -> set[str]:
    """从 records 中提取非空唯一字段值。"""
    return {str(record.get(field)) for record in records if record.get(field) not in {None, ""}}


def _seed_per_prompt_min(records: list[dict]) -> int:
    """统计每个 prompt 对应的成功 seed 最小数量。"""
    grouped: dict[str, set[str]] = {}
    for record in records:
        prompt_id = str(record.get("prompt_id") or "")
        seed_id = str(record.get("seed_id") or "")
        if prompt_id and seed_id:
            grouped.setdefault(prompt_id, set()).add(seed_id)
    return min((len(seed_ids) for seed_ids in grouped.values()), default=0)


def _decision_pass(decision: dict, *field_names: str) -> bool:
    """检查任一指定决策字段是否为 PASS。"""
    return any(decision.get(field_name) == "PASS" for field_name in field_names)


def _validation_generation_records(generation_records: list[dict], validation_profile_names: set[str]) -> list[dict]:
    """筛选 validation-scale profile 产生的成功生成记录。"""
    return [
        record for record in generation_records
        if record.get("generation_status") == "success"
        and record.get("colab_runtime_profile") in validation_profile_names
    ]



def _external_baseline_comparison_ready(
    run_root: Path,
    minimum_measured_adapter_count: int,
    minimum_modern_formal_adapter_count: int,
    required_modern_adapter_names: list[str],
) -> tuple[bool, int, int, int, list[str], str]:
    """检查 external_baseline/ adapter 是否已经写出 comparison records。"""
    records = _read_jsonl(run_root / "records" / "external_baseline_score_records.jsonl")
    decision = _read_json(run_root / "artifacts" / "external_baseline_comparison_decision.json")
    if not decision and records:
        decision = audit_external_baseline_comparison_records(records)
    measured_adapter_count = int(decision.get("external_baseline_measured_adapter_count") or 0)
    modern_formal_adapter_count = int(decision.get("modern_external_baseline_formal_measured_adapter_count") or 0)
    modern_formal_names = {
        str(name)
        for name in decision.get("modern_external_baseline_formal_measured_adapter_names", [])
        if str(name)
    }
    required_modern_names = {str(name) for name in required_modern_adapter_names if str(name)}
    missing_modern_names = sorted(required_modern_names - modern_formal_names)
    ready = (
        _decision_pass(decision, "external_baseline_comparison_decision")
        and measured_adapter_count >= minimum_measured_adapter_count
        and modern_formal_adapter_count >= minimum_modern_formal_adapter_count
        and not missing_modern_names
    )
    return (
        ready,
        len(records),
        measured_adapter_count,
        modern_formal_adapter_count,
        missing_modern_names,
        decision.get("external_baseline_claim_support_status", "missing_external_baseline_comparison_decision"),
    )

def _internal_ablation_ready(run_root: Path) -> tuple[bool, int, str]:
    """检查 validation-scale 是否已有内部消融记录。"""
    records = _read_jsonl(run_root / "records" / "validation_internal_ablation_records.jsonl")
    if not records:
        records = _read_jsonl(run_root / "records" / "ablation_scores.jsonl")
    decision = _read_json(run_root / "artifacts" / "validation_internal_ablation_decision.json")
    if not decision:
        decision = _read_json(run_root / "artifacts" / "internal_ablation_decision.json")
    decision_ready = not decision or _decision_pass(decision, "validation_internal_ablation_decision", "internal_ablation_decision")
    return bool(records) and decision_ready, len(records), decision.get("claim_support_status", "missing_internal_ablation_decision")


def _validation_scale_formal_internal_ablation_ready(run_root: Path) -> tuple[bool, int, str]:
    """检查 validation_scale 级 formal-compatible 内部消融汇总是否已通过。"""
    records = _read_jsonl(run_root / "records" / "validation_scale_formal_internal_ablation_records.jsonl")
    decision = _read_json(run_root / "artifacts" / "validation_scale_formal_internal_ablation_decision.json")
    ready = bool(records) and _decision_pass(decision, "validation_scale_formal_internal_ablation_decision")
    ready_count = int(decision.get("formal_internal_ablation_variant_count") or 0)
    return ready, ready_count, decision.get("claim_support_status", "missing_validation_scale_formal_internal_ablation_decision")


def _adaptive_attack_ready(run_root: Path) -> tuple[bool, int, str]:
    """检查 validation-scale 是否已有 Flow-specific adaptive attack 记录。"""
    records = _read_jsonl(run_root / "records" / "adaptive_attack_records.jsonl")
    decision = _read_json(run_root / "artifacts" / "adaptive_attack_decision.json")
    ready = bool(records) and _decision_pass(decision, "adaptive_attack_decision")
    return ready, len(records), decision.get("claim_support_status", "missing_adaptive_attack_decision")


def _replay_or_sketch_ready(run_root: Path) -> tuple[bool, str]:
    """检查 replay/sketch gate 是否闭合, 或 Claim-3 是否已经显式降级。"""
    replay_decision = _read_json(run_root / "artifacts" / "replay_and_sketch_gate_decision.json")
    if _decision_pass(replay_decision, "replay_and_sketch_gate_decision"):
        return True, str(replay_decision.get("replay_or_sketch_status") or "replay_and_sketch_gate_passed")
    downgrade_decision = _read_json(run_root / "artifacts" / "claim3_downgrade_decision.json")
    if downgrade_decision.get("claim3_downgraded") is True:
        return True, "claim3_explicitly_downgraded"
    return False, "missing_replay_sketch_gate_or_claim3_downgrade"


def _confidence_interval_ready(run_root: Path) -> tuple[bool, str]:
    """检查统计置信区间报告是否已由 governed artifact 写出。"""
    decision = _read_json(run_root / "artifacts" / "statistical_confidence_interval_decision.json")
    ready = _decision_pass(decision, "statistical_confidence_interval_decision")
    return ready, decision.get("claim_support_status", "missing_confidence_interval_decision")


def _low_fpr_formal_statistics_ready(run_root: Path) -> tuple[bool, int, str]:
    """检查低 FPR 正式统计阻断记录是否已落盘。"""
    records = _read_jsonl(run_root / "records" / "low_fpr_formal_statistics_records.jsonl")
    decision = _read_json(run_root / "artifacts" / "low_fpr_formal_statistics_decision.json")
    ready = bool(records) and _decision_pass(decision, "low_fpr_formal_statistics_decision")
    record_count = int(decision.get("low_fpr_formal_statistics_record_count") or len(records))
    return ready, record_count, decision.get("claim_support_status", "missing_low_fpr_formal_statistics_decision")


def _sstw_measured_formal_ready(run_root: Path) -> tuple[bool, int, str]:
    """检查 SSTW 本方法是否已转写 measured_formal records。"""
    records = _read_jsonl(run_root / "records" / "sstw_measured_formal_records.jsonl")
    decision = _read_json(run_root / "artifacts" / "sstw_measured_formal_decision.json")
    ready_records = [record for record in records if record.get("metric_status") == "measured_formal"]
    ready = bool(ready_records) and _decision_pass(decision, "sstw_measured_formal_decision")
    return ready, len(ready_records), decision.get("claim_support_status", "missing_sstw_measured_formal_decision")


def _formal_method_baseline_comparison_ready(run_root: Path) -> tuple[bool, int, str]:
    """检查 SSTW 与 5 个现代 baseline 的同协议统计表是否已通过。"""
    records = _read_jsonl(run_root / "records" / "formal_method_baseline_comparison_records.jsonl")
    decision = _read_json(run_root / "artifacts" / "formal_method_baseline_comparison_decision.json")
    ready = bool(records) and _decision_pass(decision, "formal_method_baseline_comparison_decision")
    ready_count = int(decision.get("formal_comparison_ready_method_count") or 0)
    return ready, ready_count, decision.get("claim_support_status", "missing_formal_method_baseline_comparison_decision")


def _formal_baseline_difference_interval_ready(run_root: Path) -> tuple[bool, int, str]:
    """检查 SSTW 相对 baseline 的差值置信区间报告是否已通过。"""
    records = _read_jsonl(run_root / "records" / "formal_baseline_difference_interval_records.jsonl")
    decision = _read_json(run_root / "artifacts" / "formal_baseline_difference_interval_decision.json")
    ready = bool(records) and _decision_pass(decision, "formal_baseline_difference_interval_decision")
    ready_count = int(decision.get("difference_interval_ready_count") or 0)
    return ready, ready_count, decision.get("claim_support_status", "missing_formal_baseline_difference_interval_decision")


def _artifact_rebuild_ready(run_root: Path) -> tuple[bool, str]:
    """检查 validation-scale artifact rebuild dry-run 是否通过。"""
    decision = _read_json(run_root / "artifacts" / "validation_artifact_rebuild_dry_run_decision.json")
    if not decision:
        decision = _read_json(run_root / "artifacts" / "artifact_rebuild_dry_run_decision.json")
    ready = _decision_pass(decision, "validation_artifact_rebuild_dry_run_decision", "artifact_rebuild_dry_run_decision")
    return ready, decision.get("claim_support_status", "missing_artifact_rebuild_dry_run_decision")


def build_validation_scale_gate_audit(
    run_root: str | Path,
    config_path: str | Path = DEFAULT_VALIDATION_SCALE_CONFIG,
) -> dict[str, Any]:
    """构建 validation-scale generative probe 门禁审计结果。

    该函数属于项目特定写法。它只读取 run_root 中已经落盘的 governed records、decision artifacts
    和 reports, 不运行 GPU, 不补造 baseline、消融、adaptive attack 或 replay 结果。若某类证据缺失,
    它必须把缺口写入 `missing_validation_requirements`, 防止从 pilot 直接跳到 full_paper。
    """
    run_root = Path(run_root)
    config = _load_config(config_path)
    validation_profile_names = set(config["validation_profile_names"])

    generation_records = _read_jsonl(run_root / "records" / "generation_records.jsonl")
    validation_generation_records = _validation_generation_records(generation_records, validation_profile_names)
    runtime_attack_records = _read_jsonl(run_root / "records" / "runtime_attack_records.jsonl")
    runtime_detection_records = _read_jsonl(run_root / "records" / "runtime_detection_records.jsonl")
    external_baseline_records = _read_jsonl(run_root / "records" / "external_baseline_records.jsonl")
    formal_metric_records = _read_jsonl(run_root / "records" / "formal_quality_motion_semantic_records.jsonl")

    runtime_attack_decision = _read_json(run_root / "artifacts" / "runtime_attack_decision.json")
    runtime_detection_decision = _read_json(run_root / "artifacts" / "runtime_detection_decision.json")
    motion_threshold_decision = _read_json(run_root / "artifacts" / "motion_threshold_calibration_decision.json")
    external_baseline_self_containment_decision = _read_json(run_root / "artifacts" / "external_baseline_self_containment_decision.json")
    data_split_decision = _read_json(run_root / "artifacts" / "data_split_and_leakage_guard_decision.json")

    prompt_count = len(_unique_nonempty(validation_generation_records, "prompt_id"))
    seed_per_prompt_min = _seed_per_prompt_min(validation_generation_records)
    attack_count = int(runtime_attack_decision.get("runtime_attack_count") or len(_unique_nonempty(runtime_attack_records, "attack_name")))
    runtime_attack_ready_count = int(runtime_attack_decision.get("runtime_attack_ready_count") or sum(1 for record in runtime_attack_records if record.get("attack_runtime_status") == "ready"))
    runtime_detection_ready_count = int(runtime_detection_decision.get("runtime_detection_ready_count") or sum(1 for record in runtime_detection_records if record.get("runtime_detection_status") == "ready"))

    external_baseline_audit = audit_external_baseline_records(external_baseline_records) if external_baseline_records else {}
    (
        external_baseline_comparison_ready,
        external_baseline_comparison_record_count,
        external_baseline_measured_adapter_count,
        modern_external_baseline_formal_measured_adapter_count,
        missing_modern_external_baseline_formal_adapter_names,
        external_baseline_comparison_status,
    ) = _external_baseline_comparison_ready(
        run_root,
        config["minimum_external_baseline_measured_adapter_count"],
        config["minimum_modern_external_baseline_formal_adapter_count"],
        config["required_modern_external_baseline_adapter_names"],
    )
    internal_ablation_ready, internal_ablation_record_count, internal_ablation_status = _internal_ablation_ready(run_root)
    formal_internal_ablation_ready, formal_internal_ablation_variant_count, formal_internal_ablation_status = _validation_scale_formal_internal_ablation_ready(run_root)
    adaptive_attack_ready, adaptive_attack_record_count, adaptive_attack_status = _adaptive_attack_ready(run_root)
    replay_or_sketch_ready, replay_or_sketch_status = _replay_or_sketch_ready(run_root)
    confidence_interval_ready, confidence_interval_status = _confidence_interval_ready(run_root)
    low_fpr_ready, low_fpr_record_count, low_fpr_status = _low_fpr_formal_statistics_ready(run_root)
    sstw_measured_formal_ready, sstw_measured_formal_record_count, sstw_measured_formal_status = _sstw_measured_formal_ready(run_root)
    formal_method_comparison_ready, formal_method_comparison_ready_count, formal_method_comparison_status = _formal_method_baseline_comparison_ready(run_root)
    formal_difference_interval_ready, formal_difference_interval_ready_count, formal_difference_interval_status = _formal_baseline_difference_interval_ready(run_root)
    artifact_rebuild_ready, artifact_rebuild_status = _artifact_rebuild_ready(run_root)
    motion_selection = select_motion_claim_generation_records(validation_generation_records, formal_metric_records)
    formal_motion_claim_ready = motion_selection.formal_motion_claim_status in FORMAL_MOTION_CLAIM_READY_STATUSES
    motion_threshold_ready = motion_threshold_decision.get("motion_threshold_calibration_ready") is True

    requirement_checks = {
        "validation_generation_records_ready": prompt_count >= config["minimum_prompt_count"] and seed_per_prompt_min >= config["minimum_seed_per_prompt"],
        "validation_motion_threshold_calibration_ready": (not config["require_motion_threshold_calibration_ready"]) or motion_threshold_ready,
        "validation_formal_motion_claim_ready": (not config["require_formal_motion_claim_ready"]) or formal_motion_claim_ready,
        "validation_attack_records_ready": _decision_pass(runtime_attack_decision, "runtime_attack_decision") and attack_count >= config["minimum_attack_count"],
        "validation_detection_records_ready": _decision_pass(runtime_detection_decision, "runtime_detection_decision") and runtime_detection_ready_count >= runtime_attack_ready_count > 0,
        "validation_external_baseline_status_records_ready": (not config["require_external_baseline_status_records"]) or external_baseline_audit.get("external_baseline_status_decision") == "PASS",
        "validation_external_baseline_comparison_records_ready": (not config["require_external_baseline_comparison_records"]) or external_baseline_comparison_ready,
        "validation_external_baseline_self_containment_ready": (not config["require_external_baseline_self_containment_decision"]) or external_baseline_self_containment_decision.get("external_baseline_self_containment_decision") == "PASS",
        "validation_sstw_measured_formal_records_ready": (not config["require_sstw_measured_formal_records"]) or sstw_measured_formal_ready,
        "validation_formal_method_baseline_comparison_ready": (not config["require_formal_method_baseline_comparison"]) or formal_method_comparison_ready,
        "validation_formal_baseline_difference_interval_ready": (not config["require_formal_baseline_difference_interval"]) or formal_difference_interval_ready,
        "validation_data_split_and_leakage_guard_ready": (not config["require_data_split_and_leakage_guard"]) or data_split_decision.get("data_split_and_leakage_guard_decision") == "PASS",
        "validation_internal_ablation_records_ready": (not config["require_internal_ablation_records"]) or internal_ablation_ready,
        "validation_scale_formal_internal_ablation_ready": (not config["require_validation_scale_formal_internal_ablation"]) or formal_internal_ablation_ready,
        "validation_adaptive_attack_records_ready": (not config["require_adaptive_attack_records"]) or adaptive_attack_ready,
        "validation_replay_or_sketch_records_ready": (not config["require_replay_or_sketch_records_or_claim3_downgrade"]) or replay_or_sketch_ready,
        "validation_confidence_interval_report_ready": (not config["require_confidence_interval_report"]) or confidence_interval_ready,
        "validation_low_fpr_formal_statistics_blocking_record_ready": (not config["require_low_fpr_formal_statistics_blocking_record"]) or low_fpr_ready,
        "validation_artifact_rebuild_dry_run_ready": (not config["require_artifact_rebuild_dry_run"]) or artifact_rebuild_ready,
    }
    missing_requirements = [name for name, passed in requirement_checks.items() if not passed]
    gate_decision = "PASS" if not missing_requirements else "FAIL"
    claim_support_status = "validation_scale_ready_for_pilot_paper" if gate_decision == "PASS" else "validation_scale_blocked"

    return {
        "stage_id": "validation_scale_generative_probe_gate",
        "run_root": str(run_root),
        "validation_scale_gate_decision": gate_decision,
        "claim_support_status": claim_support_status,
        "paper_result_level": config["paper_result_level"],
        "target_fpr": config["target_fpr"],
        "missing_validation_requirements": missing_requirements,
        "validation_missing_requirement_count": len(missing_requirements),
        "validation_profile_names": sorted(validation_profile_names),
        "generation_record_count": len(generation_records),
        "validation_generation_record_count": len(validation_generation_records),
        "validation_prompt_count": prompt_count,
        "validation_seed_per_prompt_min": seed_per_prompt_min,
        "minimum_prompt_count": config["minimum_prompt_count"],
        "minimum_seed_per_prompt": config["minimum_seed_per_prompt"],
        "motion_threshold_calibration_decision": motion_threshold_decision.get("motion_threshold_calibration_decision"),
        "motion_threshold_calibration_ready": motion_threshold_ready,
        "motion_threshold_id": motion_threshold_decision.get("motion_threshold_id"),
        "motion_threshold_source_split": motion_threshold_decision.get("motion_threshold_source_split"),
        "formal_motion_claim_status": motion_selection.formal_motion_claim_status,
        "formal_motion_consistency_ready_count": motion_selection.formal_motion_consistency_ready_count,
        "formal_motion_consistency_blocked_count": motion_selection.formal_motion_consistency_blocked_count,
        "motion_claim_eligible_generation_count": motion_selection.motion_claim_eligible_generation_count,
        "motion_claim_excluded_generation_count": motion_selection.motion_claim_excluded_generation_count,
        "runtime_attack_decision": runtime_attack_decision.get("runtime_attack_decision"),
        "runtime_attack_ready_count": runtime_attack_ready_count,
        "runtime_attack_count": attack_count,
        "runtime_detection_decision": runtime_detection_decision.get("runtime_detection_decision"),
        "runtime_detection_ready_count": runtime_detection_ready_count,
        "external_baseline_status_decision": external_baseline_audit.get("external_baseline_status_decision"),
        "modern_external_baseline_record_count": external_baseline_audit.get("modern_external_baseline_record_count", 0),
        "modern_external_baseline_main_comparison_ready_count": external_baseline_audit.get("modern_external_baseline_main_comparison_ready_count", 0),
        "external_baseline_comparison_record_count": external_baseline_comparison_record_count,
        "external_baseline_measured_adapter_count": external_baseline_measured_adapter_count,
        "modern_external_baseline_formal_measured_adapter_count": modern_external_baseline_formal_measured_adapter_count,
        "minimum_modern_external_baseline_formal_adapter_count": config["minimum_modern_external_baseline_formal_adapter_count"],
        "required_modern_external_baseline_adapter_names": sorted(config["required_modern_external_baseline_adapter_names"]),
        "missing_modern_external_baseline_formal_adapter_names": missing_modern_external_baseline_formal_adapter_names,
        "external_baseline_comparison_status": external_baseline_comparison_status,
        "external_baseline_self_containment_decision": external_baseline_self_containment_decision.get("external_baseline_self_containment_decision"),
        "sstw_measured_formal_record_count": sstw_measured_formal_record_count,
        "sstw_measured_formal_status": sstw_measured_formal_status,
        "formal_method_baseline_comparison_ready_count": formal_method_comparison_ready_count,
        "formal_method_baseline_comparison_status": formal_method_comparison_status,
        "formal_baseline_difference_interval_ready_count": formal_difference_interval_ready_count,
        "formal_baseline_difference_interval_status": formal_difference_interval_status,
        "data_split_and_leakage_guard_decision": data_split_decision.get("data_split_and_leakage_guard_decision"),
        "minimum_external_baseline_measured_adapter_count": config["minimum_external_baseline_measured_adapter_count"],
        "internal_ablation_record_count": internal_ablation_record_count,
        "internal_ablation_status": internal_ablation_status,
        "validation_scale_formal_internal_ablation_variant_count": formal_internal_ablation_variant_count,
        "validation_scale_formal_internal_ablation_status": formal_internal_ablation_status,
        "adaptive_attack_record_count": adaptive_attack_record_count,
        "adaptive_attack_status": adaptive_attack_status,
        "replay_or_sketch_status": replay_or_sketch_status,
        "confidence_interval_status": confidence_interval_status,
        "low_fpr_formal_statistics_record_count": low_fpr_record_count,
        "low_fpr_formal_statistics_status": low_fpr_status,
        "artifact_rebuild_status": artifact_rebuild_status,
        "full_paper_allowed": False,
        "full_paper_next_gate": "pilot_paper_generative_probe_gate" if gate_decision == "PASS" else "complete_missing_validation_requirements",
    }


def write_validation_scale_gate_audit(
    run_root: str | Path,
    config_path: str | Path = DEFAULT_VALIDATION_SCALE_CONFIG,
) -> dict[str, Any]:
    """写出 validation-scale gate records、table、decision 和 report。"""
    run_root = Path(run_root)
    audit = build_validation_scale_gate_audit(run_root, config_path)
    record = with_flow_evidence_protocol_defaults(
        {"record_version": "validation_scale_generative_probe_gate_v1", **audit},
        trajectory_source_level="validation_scale_gate_aggregated_records",
        flow_state_admissibility_status="validation_scale_ready" if audit["validation_scale_gate_decision"] == "PASS" else "validation_scale_blocked",
        claim_support_status=audit["claim_support_status"],
    )
    write_jsonl(run_root / "records" / "validation_scale_gate_records.jsonl", [record])
    write_csv(run_root / "tables" / "validation_scale_gate_table.csv", [record])
    write_json(run_root / "artifacts" / "validation_scale_gate_decision.json", audit)
    target_fpr_text = _format_fpr(audit.get("target_fpr"))
    report = (
        "# Validation-scale Generative Probe Gate Report\n\n"
        "该报告由已落盘的 governed records 与 decision artifacts 自动生成。它只判断 validation-scale "
        "是否已经作为 paper 级前的小样本全流程打通层完成闭环。该层级使用当前 protocol config "
        f"指定的 target_fpr={target_fpr_text} "
        "预演口径验证 records、tables、figures、reports、manifests、baseline、消融、attack、CI "
        "和 artifact rebuild 是否能够完整产出。它不支撑效果主张, 通过后只能生成 "
        "validation_scale_to_pilot_paper_transition_decision 并进入 pilot_paper; "
        "不能直接进入 full_paper。\n\n"
        f"- validation_scale_gate_decision: {audit['validation_scale_gate_decision']}\n"
        f"- claim_support_status: {audit['claim_support_status']}\n"
        f"- paper_result_level: {audit['paper_result_level']}\n"
        f"- target_fpr: {target_fpr_text}\n"
        f"- missing_validation_requirements: {', '.join(audit['missing_validation_requirements']) if audit['missing_validation_requirements'] else 'none'}\n"
        f"- validation_generation_record_count: {audit['validation_generation_record_count']}\n"
        f"- validation_prompt_count: {audit['validation_prompt_count']}\n"
        f"- validation_seed_per_prompt_min: {audit['validation_seed_per_prompt_min']}\n"
        f"- motion_threshold_calibration_decision: {audit['motion_threshold_calibration_decision']}\n"
        f"- formal_motion_claim_status: {audit['formal_motion_claim_status']}\n"
        f"- modern_external_baseline_main_comparison_ready_count: {audit['modern_external_baseline_main_comparison_ready_count']}\n"
        f"- external_baseline_comparison_record_count: {audit['external_baseline_comparison_record_count']}\n"
        f"- external_baseline_measured_adapter_count: {audit['external_baseline_measured_adapter_count']}\n"
        f"- modern_external_baseline_formal_measured_adapter_count: {audit['modern_external_baseline_formal_measured_adapter_count']}\n"
        f"- external_baseline_self_containment_decision: {audit['external_baseline_self_containment_decision']}\n"
        f"- sstw_measured_formal_record_count: {audit['sstw_measured_formal_record_count']}\n"
        f"- sstw_measured_formal_status: {audit['sstw_measured_formal_status']}\n"
        f"- formal_method_baseline_comparison_ready_count: {audit['formal_method_baseline_comparison_ready_count']}\n"
        f"- formal_method_baseline_comparison_status: {audit['formal_method_baseline_comparison_status']}\n"
        f"- formal_baseline_difference_interval_ready_count: {audit['formal_baseline_difference_interval_ready_count']}\n"
        f"- formal_baseline_difference_interval_status: {audit['formal_baseline_difference_interval_status']}\n"
        f"- validation_scale_formal_internal_ablation_variant_count: {audit['validation_scale_formal_internal_ablation_variant_count']}\n"
        f"- validation_scale_formal_internal_ablation_status: {audit['validation_scale_formal_internal_ablation_status']}\n"
        f"- low_fpr_formal_statistics_record_count: {audit['low_fpr_formal_statistics_record_count']}\n"
        f"- low_fpr_formal_statistics_status: {audit['low_fpr_formal_statistics_status']}\n"
        f"- data_split_and_leakage_guard_decision: {audit['data_split_and_leakage_guard_decision']}\n"
        f"- missing_modern_external_baseline_formal_adapter_names: {', '.join(audit['missing_modern_external_baseline_formal_adapter_names']) if audit['missing_modern_external_baseline_formal_adapter_names'] else 'none'}\n"
        f"- full_paper_allowed: {str(audit['full_paper_allowed']).lower()}\n"
    )
    report_path = run_root / "reports" / "validation_scale_gate_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="审计 validation-scale generative video probe gate。")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--config-path", default=DEFAULT_VALIDATION_SCALE_CONFIG)
    parser.add_argument("--write-outputs", action="store_true")
    args = parser.parse_args()
    payload = write_validation_scale_gate_audit(args.run_root, args.config_path) if args.write_outputs else build_validation_scale_gate_audit(args.run_root, args.config_path)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
