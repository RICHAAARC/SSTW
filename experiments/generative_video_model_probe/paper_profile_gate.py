"""paper profile generative video probe 的自动门禁审计。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from external_baseline.baseline_registry import audit_external_baseline_records
from experiments.generative_video_model_probe.external_baseline_runner import audit_external_baseline_comparison_records
from experiments.generative_video_model_probe.sstw_formal_result import (
    formal_sstw_clean_negative_record_ready_for_calibration,
    formal_sstw_score_record_ready_for_claim,
)
from experiments.generative_video_model_probe.formal_motion_claim_filter import (
    FORMAL_MOTION_CLAIM_READY_STATUSES,
    select_motion_claim_generation_records,
)
from evaluation.attacks.video_runtime_attack_protocol import (
    FULL_PAPER_NON_RUNTIME_ATTACK_PROTOCOLS,
    audit_runtime_attack_protocol_config,
    load_protocol_config_with_shared_attack_protocol,
    required_non_runtime_attack_protocols_from_config,
    required_runtime_attack_names_from_config,
)
from evaluation.protocol.flow_evidence_fields import with_flow_evidence_protocol_defaults
from evaluation.protocol.paper_result_formality_guard import build_paper_result_formality_guard
from evaluation.protocol.record_writer import write_json, write_jsonl
from evaluation.protocol.paper_mechanism_contract import (
    audit_paper_profile_mechanism_contract,
    load_paper_mechanism_contract,
)
from evaluation.protocol.paper_profile_evidence_closure import (
    build_paper_profile_evidence_closure_audit,
)
from evaluation.protocol.table_builder import write_csv


DEFAULT_PAPER_PROFILE_CONFIG = "configs/protocol/probe_paper_generative_probe.json"
FORMAL_FLOW_EVIDENCE_LEVEL = "attacked_video_key_independent_inversion_hypothesis_replay"
DEFAULT_PAPER_PROFILE_NAMES = {"probe_paper"}
DEFAULT_MINIMUM_PROMPT_COUNT = 8
DEFAULT_MINIMUM_SEED_PER_PROMPT = 3
DEFAULT_MINIMUM_ATTACK_COUNT = 3
DEFAULT_MINIMUM_NON_RUNTIME_ATTACK_PROTOCOL_COUNT = len(FULL_PAPER_NON_RUNTIME_ATTACK_PROTOCOLS)
DEFAULT_REQUIRED_MODERN_EXTERNAL_BASELINE_ADAPTER_NAMES = (
    "videoshield",
    "vidsig",
    "videoseal",
    "videomark",
    "wam_frame",
)
SSTW_METHOD_ID = "sstw_key_conditioned_flow_trajectory"
DEFAULT_MINIMUM_EXTERNAL_BASELINE_MEASURED_ADAPTER_COUNT = len(DEFAULT_REQUIRED_MODERN_EXTERNAL_BASELINE_ADAPTER_NAMES)
DEFAULT_MINIMUM_MODERN_EXTERNAL_BASELINE_FORMAL_ADAPTER_COUNT = len(DEFAULT_REQUIRED_MODERN_EXTERNAL_BASELINE_ADAPTER_NAMES)
HARD_REQUIRED_PAPER_PROFILE_CONFIG_FLAGS = (
    "require_external_baseline_comparison_records",
    "require_external_baseline_self_containment_decision",
    "require_sstw_measured_formal_records",
    "require_fair_detection_calibration",
    "require_formal_method_baseline_comparison",
    "require_formal_baseline_difference_interval",
    "require_data_split_and_leakage_guard",
)
HARD_REQUIRED_PROBE_PAPER_CONFIG_FLAGS = (
    *HARD_REQUIRED_PAPER_PROFILE_CONFIG_FLAGS,
    "require_sstw_advantage_claim_ready",
)


def _read_json(path: Path) -> dict:
    """读取 JSON 文件, 并兼容 Windows 或 Colab 产生的 UTF-8 BOM。"""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _required_float(raw: dict, field_name: str, config_path: Path) -> float:
    """从 paper profile protocol config 读取必填 float 字段。"""
    if field_name not in raw:
        raise KeyError(f"paper profile protocol config 缺少必填字段 {field_name}: {config_path}")
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


def _load_config(config_path: str | Path = DEFAULT_PAPER_PROFILE_CONFIG) -> dict:
    """读取 paper profile 门禁配置。"""
    path = Path(config_path)
    config = load_protocol_config_with_shared_attack_protocol(path)
    paper_result_level = str(config.get("paper_result_level", "probe_paper"))
    required_runtime_attack_names = list(required_runtime_attack_names_from_config(config))
    required_non_runtime_attack_protocols = list(required_non_runtime_attack_protocols_from_config(config))
    mechanism_contract_path = config.get("formal_mechanism_contract_path")
    mechanism_contract_audit = None
    if mechanism_contract_path:
        mechanism_contract = load_paper_mechanism_contract(mechanism_contract_path)
        mechanism_contract_audit = audit_paper_profile_mechanism_contract([config], mechanism_contract).as_dict()
    return {
        "paper_profile_names": config.get(
            "paper_profile_names",
            config.get("probe_profile_names", [paper_result_level]),
        ),
        "target_fpr": _required_float(config, "target_fpr", path),
        "paper_result_level": paper_result_level,
        "stage_id": config.get("stage_id", "paper_profile_generative_probe_gate"),
        "minimum_prompt_count": int(config.get("minimum_prompt_count", DEFAULT_MINIMUM_PROMPT_COUNT)),
        "minimum_seed_per_prompt": int(config.get("minimum_seed_per_prompt", DEFAULT_MINIMUM_SEED_PER_PROMPT)),
        "minimum_attack_count": int(config.get("minimum_attack_count", max(DEFAULT_MINIMUM_ATTACK_COUNT, len(required_runtime_attack_names)))),
        "required_runtime_attack_names": required_runtime_attack_names,
        "required_non_runtime_attack_protocols": required_non_runtime_attack_protocols,
        "minimum_non_runtime_attack_protocol_count": int(config.get("minimum_non_runtime_attack_protocol_count", max(DEFAULT_MINIMUM_NON_RUNTIME_ATTACK_PROTOCOL_COUNT, len(required_non_runtime_attack_protocols)))),
        "runtime_attack_protocol_audit": audit_runtime_attack_protocol_config(config),
        "minimum_clean_negative_count": int(config.get("minimum_clean_negative_count", 0)),
        "require_external_baseline_status_records": bool(config.get("require_external_baseline_status_records", True)),
        "require_external_baseline_comparison_records": bool(config.get("require_external_baseline_comparison_records", True)),
        "require_external_baseline_self_containment_decision": bool(config.get("require_external_baseline_self_containment_decision", True)),
        "require_sstw_measured_formal_records": bool(config.get("require_sstw_measured_formal_records", True)),
        "require_fair_detection_calibration": bool(config.get("require_fair_detection_calibration", True)),
        "require_formal_method_baseline_comparison": bool(config.get("require_formal_method_baseline_comparison", True)),
        "require_formal_baseline_difference_interval": bool(config.get("require_formal_baseline_difference_interval", True)),
        "minimum_external_baseline_measured_adapter_count": int(config.get("minimum_external_baseline_measured_adapter_count", DEFAULT_MINIMUM_EXTERNAL_BASELINE_MEASURED_ADAPTER_COUNT)),
        "minimum_modern_external_baseline_formal_adapter_count": int(config.get("minimum_modern_external_baseline_formal_adapter_count", DEFAULT_MINIMUM_MODERN_EXTERNAL_BASELINE_FORMAL_ADAPTER_COUNT)),
        "required_modern_external_baseline_adapter_names": list(config.get("required_modern_external_baseline_adapter_names", DEFAULT_REQUIRED_MODERN_EXTERNAL_BASELINE_ADAPTER_NAMES)),
        "require_motion_threshold_calibration_ready": bool(config.get("require_motion_threshold_calibration_ready", True)),
        "require_formal_motion_claim_ready": bool(config.get("require_formal_motion_claim_ready", True)),
        "require_motion_consistency_exclusion_report": bool(config.get("require_motion_consistency_exclusion_report", True)),
        "require_internal_ablation_records": bool(config.get("require_internal_ablation_records", True)),
        "require_formal_internal_ablation_summary": bool(config.get("require_formal_internal_ablation_summary", True)),
        "require_adaptive_attack_records": bool(config.get("require_adaptive_attack_records", True)),
        "require_replay_and_sketch_full_support": bool(
            config.get("require_replay_and_sketch_full_support", True)
        ),
        "require_claim3_full_support": bool(config.get("require_claim3_full_support", False)),
        "require_complete_paper_mechanism_contract": bool(config.get("require_complete_paper_mechanism_contract", False)),
        "paper_mechanism_contract_audit": mechanism_contract_audit,
        "require_confidence_interval_report": bool(config.get("require_confidence_interval_report", True)),
        "require_low_fpr_formal_statistics_blocking_record": bool(config.get("require_low_fpr_formal_statistics_blocking_record", True)),
        "require_paper_result_artifact_skeleton": bool(config.get("require_paper_result_artifact_skeleton", True)),
        "require_artifact_rebuild_dry_run": bool(config.get("require_artifact_rebuild_dry_run", True)),
        "require_data_split_and_leakage_guard": bool(config.get("require_data_split_and_leakage_guard", True)),
        "require_sstw_advantage_claim_ready": bool(config.get("require_sstw_advantage_claim_ready", True)),
        "minimum_sstw_advantage_baseline_count": int(config.get("minimum_sstw_advantage_baseline_count", DEFAULT_MINIMUM_MODERN_EXTERNAL_BASELINE_FORMAL_ADAPTER_COUNT)),
        "minimum_sstw_tpr_at_target_fpr_difference": float(config.get("minimum_sstw_tpr_at_target_fpr_difference", 0.0)),
        "require_sstw_advantage_ci_lower_above_zero": bool(config.get("require_sstw_advantage_ci_lower_above_zero", True)),
    }


def _hard_required_config_missing(config: dict[str, Any]) -> list[str]:
    """检查当前 profile 的公平比较硬前置是否被配置关闭。

    paper_profile 是共享检查实现, 不是独立主干阶段。当前主干首个 paper profile 是
    probe_paper, 它在 target_fpr=0.1 下要求 self-contained baseline、SSTW measured_formal、
    公平 FPR 校准、同协议比较、差值区间、数据切分防泄漏和 SSTW 优势 claim ready
    都能生成。该检查属于项目特定门禁硬化, 用于防止通过配置关闭核心证据链。
    """
    required_flags = (
        HARD_REQUIRED_PROBE_PAPER_CONFIG_FLAGS
        if str(config.get("paper_result_level")) == "probe_paper"
        else HARD_REQUIRED_PAPER_PROFILE_CONFIG_FLAGS
    )
    return [
        f"{field_name}_must_be_true"
        for field_name in required_flags
        if config.get(field_name) is not True
    ]


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


def _paper_profile_generation_records(generation_records: list[dict], paper_profile_names: set[str]) -> list[dict]:
    """筛选当前 paper profile 产生的成功生成记录。"""
    return [
        record for record in generation_records
        if record.get("generation_status") == "success"
        and record.get("colab_runtime_profile") in paper_profile_names
        and str(record.get("sample_role") or record.get("generation_sample_role") or "").lower() != "clean_negative"
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
    records_audit = audit_external_baseline_comparison_records(records) if records else {}
    measured_adapter_count = int(records_audit.get("external_baseline_measured_adapter_count") or 0)
    modern_formal_adapter_count = int(records_audit.get("modern_external_baseline_formal_measured_adapter_count") or 0)
    modern_formal_names = {
        str(name)
        for name in records_audit.get("modern_external_baseline_formal_measured_adapter_names", [])
        if str(name)
    }
    required_modern_names = {str(name) for name in required_modern_adapter_names if str(name)}
    missing_modern_names = sorted(required_modern_names - modern_formal_names)
    ready = (
        _decision_pass(decision, "external_baseline_comparison_decision")
        and records_audit.get("external_baseline_comparison_decision") == "PASS"
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
        records_audit.get(
            "external_baseline_claim_support_status",
            decision.get("external_baseline_claim_support_status", "missing_external_baseline_comparison_decision"),
        ),
    )

def _internal_ablation_ready(run_root: Path) -> tuple[bool, int, str]:
    """检查 paper profile 是否已有正式内部消融原始记录。"""

    records = _read_jsonl(run_root / "records" / "formal_internal_ablation_variant_records.jsonl")
    if not records:
        records = _read_jsonl(run_root / "records" / "validation_internal_ablation_records.jsonl")
    decision = _read_json(run_root / "artifacts" / "validation_internal_ablation_decision.json")
    if not decision:
        decision = _read_json(run_root / "artifacts" / "internal_ablation_decision.json")
    formal_records = [
        record
        for record in records
        if record.get("metric_status") == "measured_formal"
        and record.get("formal_internal_ablation_evidence_level") == "formal_component_removal_video_detector"
    ]
    decision_ready = (
        _decision_pass(decision, "validation_internal_ablation_decision", "internal_ablation_decision")
        and decision.get("validation_internal_ablation_evidence_level") == "formal_component_removal_video_detector"
    )
    return (
        bool(records) and len(formal_records) == len(records) and decision_ready,
        len(records),
        decision.get("claim_support_status", "missing_internal_ablation_decision"),
    )


def _motion_consistency_exclusion_ready(run_root: Path) -> tuple[bool, int, str]:
    """检查 motion consistency 阻断样本处理报告是否已写出。"""
    records = _read_jsonl(run_root / "records" / "motion_consistency_exclusion_records.jsonl")
    decision = _read_json(run_root / "artifacts" / "motion_consistency_exclusion_decision.json")
    ready = bool(records) and _decision_pass(decision, "motion_consistency_exclusion_decision")
    excluded_count = int(decision.get("motion_consistency_excluded_count") or 0)
    return ready, excluded_count, decision.get("claim_support_status", "missing_motion_consistency_exclusion_decision")


def _formal_internal_ablation_summary_ready(run_root: Path) -> tuple[bool, int, str]:
    """检查 paper_profile 级 formal-compatible 内部消融汇总是否已通过。"""
    records = _read_jsonl(run_root / "records" / "formal_internal_ablation_summary_records.jsonl")
    decision = _read_json(run_root / "artifacts" / "formal_internal_ablation_summary_decision.json")
    formal_records = [record for record in records if record.get("metric_status") == "measured_formal"]
    proxy_records = [record for record in records if str(record.get("metric_status") or "") == "measured_proxy" or "proxy" in str(record.get("formal_internal_ablation_evidence_level") or "")]
    ready = (
        bool(records)
        and _decision_pass(decision, "formal_internal_ablation_summary_decision")
        and len(formal_records) == len(records)
        and not proxy_records
    )
    ready_count = len(formal_records)
    status = decision.get("claim_support_status", "missing_formal_internal_ablation_summary_decision")
    if proxy_records:
        status = "formal_internal_ablation_summary_blocked_by_proxy_records"
    return ready, ready_count, status


def _adaptive_attack_ready(
    run_root: Path,
    required_non_runtime_attack_protocols: list[str],
    minimum_non_runtime_attack_protocol_count: int,
) -> tuple[bool, int, int, list[str], list[str], str]:
    """检查 paper profile 是否已有完整 non-runtime/adaptive attack 协议记录。"""
    records = _read_jsonl(run_root / "records" / "adaptive_attack_records.jsonl")
    decision = _read_json(run_root / "artifacts" / "adaptive_attack_decision.json")
    formal_records = [
        record
        for record in records
        if record.get("adaptive_attack_evidence_level") == "formal_adaptive_attack_execution"
        and record.get("adaptive_robustness_claim_allowed") is True
        and record.get("metric_status") == "measured_formal"
    ]
    observed_protocols = {
        str(record.get("non_runtime_attack_protocol") or record.get("adaptive_attack_name") or "")
        for record in formal_records
        if record.get("non_runtime_attack_protocol") or record.get("adaptive_attack_name")
    }
    required_protocols = {str(item) for item in required_non_runtime_attack_protocols if str(item)}
    missing_protocols = sorted(required_protocols - observed_protocols)
    ready = (
        bool(records)
        and _decision_pass(decision, "adaptive_attack_decision")
        and len(formal_records) == len(records)
        and len(observed_protocols) >= minimum_non_runtime_attack_protocol_count
        and not missing_protocols
    )
    return (
        ready,
        len(records),
        len(observed_protocols),
        sorted(observed_protocols),
        missing_protocols,
        decision.get("claim_support_status", "missing_adaptive_attack_decision"),
    )


def _replay_or_sketch_ready(run_root: Path) -> tuple[bool, str]:
    """检查 attacked-video replay 与认证 sketch 是否提供完整 Claim-3 支持。"""
    replay_decision = _read_json(run_root / "artifacts" / "replay_and_sketch_gate_decision.json")
    if _decision_pass(replay_decision, "replay_and_sketch_gate_decision"):
        if replay_decision.get("claim3_full_support_allowed") is not True:
            return False, "replay_gate_passed_without_claim3_full_support"
        return True, str(replay_decision.get("replay_or_sketch_status") or "replay_and_sketch_gate_passed")
    return False, "missing_full_attacked_video_replay_and_authenticated_sketch"


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


def _paper_result_artifact_skeleton_ready(run_root: Path) -> tuple[bool, str]:
    """检查论文级图表和补充实验产物骨架是否已闭合。"""

    decision = _read_json(run_root / "artifacts" / "paper_result_artifact_skeleton_decision.json")
    ready = _decision_pass(decision, "paper_result_artifact_skeleton_decision")
    return ready, decision.get("claim_support_status", "missing_paper_result_artifact_skeleton_decision")


def _paper_profile_sstw_advantage_claim_ready(
    run_root: Path,
    *,
    required_modern_adapter_names: list[str],
    target_fpr: float,
    minimum_advantage_baseline_count: int,
    minimum_difference: float,
    require_ci_lower_above_zero: bool,
) -> tuple[bool, int, list[str], list[str], str]:
    """检查 paper_profile 是否足以支持 SSTW 在 target FPR 下优于 5 个 baseline。

    该函数属于项目特定论文 claim 门禁。它不重新计算指标, 只读取
    `formal_baseline_difference_interval_records.jsonl` 中由公平比较链路生成的
    差值记录。通过条件是: 每个现代 baseline 都有同 target_fpr、同 attack anchor
    的 ready 差值记录, 且 SSTW 的 TPR@FPR 差值大于配置阈值; 若配置要求, 差值
    置信区间下界也必须大于 0。
    """

    records = _read_jsonl(run_root / "records" / "formal_baseline_difference_interval_records.jsonl")
    required_baselines = {str(item) for item in required_modern_adapter_names if str(item)}
    ready_baselines: set[str] = set()
    blocked_reasons: list[str] = []
    for record in records:
        baseline_id = str(record.get("baseline_method_id") or "")
        if baseline_id not in required_baselines:
            continue
        if not _target_fpr_matches(record, target_fpr):
            blocked_reasons.append(f"{baseline_id}:target_fpr_mismatch")
            continue
        if record.get("difference_interval_status") != "ready" or record.get("metric_status") != "measured_formal":
            blocked_reasons.append(f"{baseline_id}:difference_interval_not_ready")
            continue
        difference = record.get("tpr_at_target_fpr_difference")
        try:
            difference_value = float(difference)
        except (TypeError, ValueError):
            blocked_reasons.append(f"{baseline_id}:difference_missing")
            continue
        if difference_value <= minimum_difference:
            blocked_reasons.append(f"{baseline_id}:difference_not_above_minimum")
            continue
        if require_ci_lower_above_zero:
            try:
                ci_lower = float(record.get("difference_ci_lower"))
            except (TypeError, ValueError):
                blocked_reasons.append(f"{baseline_id}:difference_ci_lower_missing")
                continue
            if ci_lower <= 0.0:
                blocked_reasons.append(f"{baseline_id}:difference_ci_lower_not_above_zero")
                continue
        ready_baselines.add(baseline_id)

    missing_baselines = sorted(required_baselines - ready_baselines)
    ready = (
        len(ready_baselines) >= minimum_advantage_baseline_count
        and not missing_baselines
        and not blocked_reasons
    )
    status = (
        "paper_profile_target_fpr_0_1_sstw_advantage_claim_supported"
        if ready
        else "paper_profile_sstw_advantage_claim_blocked"
    )
    return ready, len(ready_baselines), missing_baselines, blocked_reasons, status


def _sstw_measured_formal_ready(run_root: Path) -> tuple[bool, int, str]:
    """检查 SSTW 本方法是否已转写 measured_formal records。"""
    records = _read_jsonl(run_root / "records" / "sstw_measured_formal_records.jsonl")
    decision = _read_json(run_root / "artifacts" / "sstw_measured_formal_decision.json")
    ready_records = [record for record in records if record.get("metric_status") == "measured_formal"]
    positive_records = [
        record
        for record in ready_records
        if str(record.get("sample_role") or "").lower() not in {"clean_negative", "controlled_negative"}
    ]
    clean_negative_records = [
        record
        for record in ready_records
        if str(record.get("sample_role") or "").lower() == "clean_negative"
    ]
    formal_positive_count = sum(1 for record in positive_records if formal_sstw_score_record_ready_for_claim(record))
    formal_clean_negative_count = sum(1 for record in clean_negative_records if formal_sstw_clean_negative_record_ready_for_calibration(record))
    ready = (
        bool(ready_records)
        and _decision_pass(decision, "sstw_measured_formal_decision")
        and formal_positive_count == len(positive_records)
        and formal_clean_negative_count == len(clean_negative_records)
        and bool(positive_records)
        and bool(clean_negative_records)
    )
    status = decision.get("claim_support_status", "missing_sstw_measured_formal_decision")
    if ready_records and not ready:
        status = "sstw_measured_formal_blocked_by_non_formal_video_detector_evidence"
    return ready, len(ready_records), status


def _required_fair_method_ids(required_modern_adapter_names: list[str]) -> set[str]:
    """返回公平比较必须覆盖的 SSTW 与现代 baseline 方法集合。"""

    return {SSTW_METHOD_ID, *{str(name) for name in required_modern_adapter_names if str(name)}}


def _target_fpr_matches(record: dict, expected_target_fpr: float) -> bool:
    """检查 fairness 产物是否来自当前 protocol config 的 target FPR。"""

    try:
        return abs(float(record.get("target_fpr")) - float(expected_target_fpr)) <= 1e-12
    except (TypeError, ValueError):
        return False


def _nonnegative_int(record: dict, field_name: str) -> int:
    """读取 governed record 中的非负计数字段, 缺失或不可解析时按 0 处理。"""

    try:
        return max(int(record.get(field_name) or 0), 0)
    except (TypeError, ValueError):
        return 0


def _attack_names_from_anchor_keys(anchor_keys: Any) -> set[str]:
    """从 prompt / seed / attack anchor key 中解析 attack 名称。

    当前 anchor 规范是 `prompt_id::seed_id::attack_name`。该函数只作为门禁
    兜底解析; 新 records 应优先显式写出 `positive_attack_names`、
    `comparison_attack_names` 或 `paired_attack_names`。
    """

    if not isinstance(anchor_keys, list):
        return set()
    names: set[str] = set()
    for item in anchor_keys:
        parts = str(item or "").split("::")
        if len(parts) >= 3 and parts[-1]:
            names.add(parts[-1])
    return names


def _record_attack_names(record: dict, explicit_field_name: str, anchor_field_name: str | None = None) -> set[str]:
    """读取 record 中声明的 attack 名称集合。"""

    raw_names = record.get(explicit_field_name)
    if isinstance(raw_names, list):
        names = {str(item) for item in raw_names if str(item)}
        if names:
            return names
    if anchor_field_name:
        return _attack_names_from_anchor_keys(record.get(anchor_field_name))
    return set()


def _missing_required_attack_names(record: dict, explicit_field_name: str, required_attack_names: list[str], anchor_field_name: str | None = None) -> list[str]:
    """计算 record 相对当前 profile 必须 attack 集合的缺口。"""

    observed = _record_attack_names(record, explicit_field_name, anchor_field_name)
    required = {str(item) for item in required_attack_names if str(item)}
    return sorted(required - observed)


def _records_cover_required_attacks(records: list[dict], status_field: str, ready_value: str, required_attack_names: list[str]) -> tuple[bool, list[str], list[str]]:
    """检查 runtime attack / detection records 是否覆盖当前 profile 的完整 attack 集合。"""

    ready_records = [record for record in records if record.get(status_field) == ready_value]
    observed = sorted({str(record.get("attack_name")) for record in ready_records if record.get("attack_name")})
    required = {str(item) for item in required_attack_names if str(item)}
    missing = sorted(required - set(observed))
    return not missing, observed, missing


def _runtime_attack_records_formal_ready(records: list[dict]) -> tuple[bool, int]:
    """检查 runtime attack records 是否全部是正式视频文件级变换。"""

    ready_records = [record for record in records if record.get("attack_runtime_status") == "ready"]
    formal_ready = [
        record
        for record in ready_records
        if record.get("runtime_attack_implementation_level") == "formal_runtime_video_transform"
        and record.get("runtime_attack_formal_evidence_level") == "formal_runtime_video_transform"
        and record.get("runtime_attack_proxy_free") is True
    ]
    return bool(ready_records) and len(formal_ready) == len(ready_records), len(formal_ready)


def _runtime_detection_records_formal_ready(records: list[dict]) -> tuple[bool, int]:
    """检查 runtime detection records 是否全部来自固定路径 Flow replay 检测器。"""

    ready_records = [record for record in records if record.get("runtime_detection_status") == "ready"]
    formal_ready = [
        record
        for record in ready_records
        if record.get("sstw_detector_evidence_level") == FORMAL_FLOW_EVIDENCE_LEVEL
        and record.get("trajectory_trace_used_for_score") is False
        and record.get("runtime_detection_claim_level") == "formal_paper_detector"
        and record.get("sstw_raw_detector_score") is not None
    ]
    return bool(ready_records) and len(formal_ready) == len(ready_records), len(formal_ready)


def _fair_detection_anchor_ready(record: dict, minimum_clean_negative_count: int, required_attack_names: list[str]) -> bool:
    """检查 fair calibration record 是否具备完整公平比较证据。

    该门禁不能只相信上游写入的 `fair_comparison_status`, 还需要显式核验
    clean negative 校准数量、positive anchor 和 formal evidence 缺口计数。
    这样可以阻断手工改写或旧版本产物把未校准 baseline 伪装成可进入
    pilot_paper 的公平比较结果。
    """

    return (
        _nonnegative_int(record, "clean_negative_score_count") >= minimum_clean_negative_count
        and _nonnegative_int(record, "positive_anchor_count") > 0
        and _nonnegative_int(record, "positive_anchor_missing_count") == 0
        and _nonnegative_int(record, "positive_formal_evidence_missing_count") == 0
        and _nonnegative_int(record, "negative_formal_evidence_missing_count") == 0
        and not _missing_required_attack_names(record, "positive_attack_names", required_attack_names, "positive_anchor_keys")
    )


def _formal_comparison_anchor_ready(record: dict, required_attack_names: list[str]) -> bool:
    """检查同协议统计行是否与 SSTW reference anchor 集合对齐。"""

    method_role = str(record.get("method_role") or "")
    alignment_status = str(record.get("comparison_anchor_alignment_status") or "")
    if _nonnegative_int(record, "comparison_anchor_count") <= 0:
        return False
    if _nonnegative_int(record, "reference_anchor_count") <= 0:
        return False
    if _missing_required_attack_names(record, "comparison_attack_names", required_attack_names, "comparison_anchor_keys"):
        return False
    if method_role == "proposed_method":
        return alignment_status == "reference_method_anchor_set_ready"
    return (
        alignment_status == "aligned_with_sstw_reference_anchors"
        and _nonnegative_int(record, "missing_reference_anchor_count") == 0
        and _nonnegative_int(record, "extra_anchor_count") == 0
    )


def _difference_interval_anchor_ready(record: dict, required_attack_names: list[str]) -> bool:
    """检查差值区间是否来自 prompt / seed / attack 完全配对的比较单元。"""

    return (
        str(record.get("comparison_anchor_alignment_status") or "") == "aligned_with_sstw_reference_anchors"
        and _nonnegative_int(record, "paired_comparison_unit_count") > 0
        and _nonnegative_int(record, "unpaired_reference_anchor_count") == 0
        and _nonnegative_int(record, "unpaired_baseline_anchor_count") == 0
        and not _missing_required_attack_names(record, "paired_attack_names", required_attack_names, "paired_comparison_anchor_keys")
    )


def _formal_method_baseline_comparison_ready(
    run_root: Path,
    required_modern_adapter_names: list[str],
    target_fpr: float,
    required_attack_names: list[str],
) -> tuple[bool, int, str]:
    """检查 SSTW 与 5 个现代 baseline 的同协议统计表是否已通过。"""
    records = _read_jsonl(run_root / "records" / "formal_method_baseline_comparison_records.jsonl")
    decision = _read_json(run_root / "artifacts" / "formal_method_baseline_comparison_decision.json")
    required_method_ids = _required_fair_method_ids(required_modern_adapter_names)
    ready_method_ids = {
        str(record.get("method_id") or "")
        for record in records
        if record.get("metric_status") == "measured_formal"
        and _target_fpr_matches(record, target_fpr)
        and _formal_comparison_anchor_ready(record, required_attack_names)
    }
    missing_method_ids = sorted(required_method_ids - ready_method_ids)
    ready_count = int(decision.get("formal_comparison_ready_method_count") or len(ready_method_ids))
    ready = (
        bool(records)
        and _decision_pass(decision, "formal_method_baseline_comparison_decision")
        and _target_fpr_matches(decision, target_fpr)
        and ready_count >= len(required_method_ids)
        and not missing_method_ids
    )
    return ready, ready_count, decision.get("claim_support_status", "missing_formal_method_baseline_comparison_decision")


def _fair_detection_calibration_ready(
    run_root: Path,
    required_modern_adapter_names: list[str],
    target_fpr: float,
    minimum_clean_negative_count: int,
    required_attack_names: list[str],
) -> tuple[bool, int, str]:
    """检查 clean negative calibration 公平比较是否已通过。"""
    records = _read_jsonl(run_root / "records" / "fair_detection_calibration_records.jsonl")
    decision = _read_json(run_root / "artifacts" / "fair_detection_calibration_decision.json")
    required_method_ids = _required_fair_method_ids(required_modern_adapter_names)
    ready_method_ids = {
        str(record.get("method_id") or "")
        for record in records
        if record.get("fair_comparison_status") == "ready"
        and record.get("metric_status") == "measured_formal"
        and _target_fpr_matches(record, target_fpr)
        and _fair_detection_anchor_ready(record, minimum_clean_negative_count, required_attack_names)
    }
    missing_method_ids = sorted(required_method_ids - ready_method_ids)
    ready_count = len(ready_method_ids)
    ready = (
        bool(records)
        and _decision_pass(decision, "fair_detection_calibration_decision")
        and _target_fpr_matches(decision, target_fpr)
        and ready_count >= len(required_method_ids)
        and not missing_method_ids
    )
    return ready, ready_count, decision.get("claim_support_status", "missing_fair_detection_calibration_decision")


def _formal_baseline_difference_interval_ready(
    run_root: Path,
    required_modern_adapter_names: list[str],
    target_fpr: float,
    required_attack_names: list[str],
) -> tuple[bool, int, str]:
    """检查 SSTW 相对 baseline 的差值置信区间报告是否已通过。"""
    records = _read_jsonl(run_root / "records" / "formal_baseline_difference_interval_records.jsonl")
    decision = _read_json(run_root / "artifacts" / "formal_baseline_difference_interval_decision.json")
    required_baseline_ids = {str(name) for name in required_modern_adapter_names if str(name)}
    ready_baseline_ids = {
        str(record.get("baseline_method_id") or "")
        for record in records
        if record.get("difference_interval_status") == "ready"
        and record.get("metric_status") == "measured_formal"
        and _target_fpr_matches(record, target_fpr)
        and _difference_interval_anchor_ready(record, required_attack_names)
    }
    missing_baseline_ids = sorted(required_baseline_ids - ready_baseline_ids)
    ready_count = int(decision.get("difference_interval_ready_count") or len(ready_baseline_ids))
    ready = (
        bool(records)
        and _decision_pass(decision, "formal_baseline_difference_interval_decision")
        and _target_fpr_matches(decision, target_fpr)
        and ready_count >= len(required_baseline_ids)
        and not missing_baseline_ids
    )
    return ready, ready_count, decision.get("claim_support_status", "missing_formal_baseline_difference_interval_decision")


def _artifact_rebuild_ready(run_root: Path) -> tuple[bool, str]:
    """检查 paper profile artifact rebuild dry-run 是否通过。"""
    decision = _read_json(run_root / "artifacts" / "validation_artifact_rebuild_dry_run_decision.json")
    if not decision:
        decision = _read_json(run_root / "artifacts" / "artifact_rebuild_dry_run_decision.json")
    ready = _decision_pass(decision, "validation_artifact_rebuild_dry_run_decision", "artifact_rebuild_dry_run_decision")
    return ready, decision.get("claim_support_status", "missing_artifact_rebuild_dry_run_decision")


def _string_list(value: Any) -> list[str]:
    """把 artifact 中的列表字段规整为字符串列表。"""
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]


def _external_baseline_self_containment_ready(
    decision: dict[str, Any],
    required_modern_adapter_names: list[str],
) -> tuple[bool, dict[str, Any]]:
    """检查 external baseline self-containment 是否包含完整公平比较闭环。

    paper_profile 不能只接受一个手写或旧版
    `external_baseline_self_containment_decision: PASS`。该函数要求 5 个现代
    baseline 均有项目内 official bundle、clean negative、分数抽取口径、完整
    prompt / seed / attack anchor 和 official baseline 身份。
    """

    required_names = {str(name) for name in required_modern_adapter_names if str(name)}
    rows = [
        row
        for row in decision.get("baseline_self_containment_rows", [])
        if isinstance(row, dict)
    ]
    row_by_name = {
        str(row.get("baseline_name") or ""): row
        for row in rows
        if str(row.get("baseline_name") or "")
    }
    missing_row_names = sorted(required_names - set(row_by_name))
    not_self_contained_names = sorted(
        name
        for name in required_names
        if row_by_name.get(name, {}).get("external_baseline_self_contained") is not True
    )
    missing_repository_bundle_names = sorted(set(
        _string_list(decision.get("missing_repository_generated_official_bundle_modern_external_baseline_names"))
        + [
            name
            for name in required_names
            if row_by_name.get(name, {}).get("repository_generated_official_bundle_ready") is not True
        ]
    ))
    missing_clean_negative_names = sorted(set(
        _string_list(decision.get("missing_clean_negative_modern_external_baseline_names"))
        + [
            name
            for name in required_names
            if row_by_name.get(name, {}).get("clean_negative_ready") is not True
        ]
    ))
    missing_score_extraction_names = sorted(set(
        _string_list(decision.get("missing_score_extraction_modern_external_baseline_names"))
        + [
            name
            for name in required_names
            if row_by_name.get(name, {}).get("score_extraction_ready") is not True
        ]
    ))
    missing_official_identity_names = sorted(set(
        _string_list(decision.get("missing_official_identity_modern_external_baseline_names"))
        + [
            name
            for name in required_names
            if row_by_name.get(name, {}).get("official_baseline_identity_ready") is not True
        ]
    ))
    missing_anchor_names = sorted(set(
        _string_list(decision.get("missing_anchor_modern_external_baseline_names"))
        + [
            name
            for name in required_names
            if row_by_name.get(name, {}).get("anchor_ready") is not True
        ]
    ))
    missing_formal_names = sorted(set(
        _string_list(decision.get("missing_formal_modern_external_baseline_names"))
        + [
            name
            for name in required_names
            if int(row_by_name.get(name, {}).get("measured_formal_record_count") or 0) <= 0
        ]
    ))
    missing_requirements = _string_list(decision.get("missing_self_containment_requirements"))
    try:
        missing_requirement_count = int(decision.get("self_containment_missing_requirement_count") or 0)
    except (TypeError, ValueError):
        missing_requirement_count = -1
    ready_count = sum(
        1
        for name in required_names
        if row_by_name.get(name, {}).get("external_baseline_self_contained") is True
    )
    summary_missing: list[str] = []
    if decision.get("external_baseline_self_containment_decision") != "PASS":
        summary_missing.append("external_baseline_self_containment_decision_passed")
    if missing_requirements or missing_requirement_count != 0:
        summary_missing.append("external_baseline_self_containment_missing_requirements_empty")
    if missing_row_names:
        summary_missing.append("external_baseline_self_containment_required_rows_present")
    if not_self_contained_names or ready_count < len(required_names):
        summary_missing.append("external_baseline_self_containment_required_baselines_ready")
    if missing_repository_bundle_names:
        summary_missing.append("external_baseline_self_containment_repository_generated_bundles_ready")
    if missing_clean_negative_names:
        summary_missing.append("external_baseline_self_containment_clean_negative_ready")
    if missing_score_extraction_names:
        summary_missing.append("external_baseline_self_containment_score_extraction_ready")
    if missing_official_identity_names:
        summary_missing.append("external_baseline_self_containment_official_identity_ready")
    if missing_anchor_names:
        summary_missing.append("external_baseline_self_containment_anchor_ready")
    if missing_formal_names:
        summary_missing.append("external_baseline_self_containment_measured_formal_ready")
    return not summary_missing, {
        "external_baseline_self_containment_decision": decision.get("external_baseline_self_containment_decision"),
        "external_baseline_self_containment_ready_count": ready_count,
        "external_baseline_self_containment_required_count": len(required_names),
        "external_baseline_self_containment_gate_missing_requirements": summary_missing,
        "missing_self_contained_modern_external_baseline_names": sorted(set(_string_list(decision.get("missing_self_contained_modern_external_baseline_names")) + not_self_contained_names + missing_row_names)),
        "missing_repository_generated_official_bundle_modern_external_baseline_names": missing_repository_bundle_names,
        "missing_clean_negative_modern_external_baseline_names": missing_clean_negative_names,
        "missing_score_extraction_modern_external_baseline_names": missing_score_extraction_names,
        "missing_official_identity_modern_external_baseline_names": missing_official_identity_names,
        "missing_anchor_modern_external_baseline_names": missing_anchor_names,
        "missing_formal_modern_external_baseline_names": missing_formal_names,
    }


def build_paper_profile_gate_audit(
    run_root: str | Path,
    config_path: str | Path = DEFAULT_PAPER_PROFILE_CONFIG,
) -> dict[str, Any]:
    """构建 paper profile generative probe 门禁审计结果。

    该函数属于项目特定写法。它只读取 run_root 中已经落盘的 governed records、decision artifacts
    和 reports, 不运行 GPU, 不补造 baseline、消融、adaptive attack 或 replay 结果。若某类证据缺失,
    它必须把缺口写入 `missing_validation_requirements`, 防止从 pilot 直接跳到 full_paper。
    """
    run_root = Path(run_root)
    config = _load_config(config_path)
    paper_profile_names = set(config["paper_profile_names"])
    hard_config_missing = _hard_required_config_missing(config)

    generation_records = _read_jsonl(run_root / "records" / "generation_records.jsonl")
    validation_generation_records = _paper_profile_generation_records(generation_records, paper_profile_names)
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
    runtime_attack_required_ready, runtime_attack_observed_names, runtime_attack_missing_names = _records_cover_required_attacks(
        runtime_attack_records,
        "attack_runtime_status",
        "ready",
        config["required_runtime_attack_names"],
    )
    runtime_detection_required_ready, runtime_detection_observed_names, runtime_detection_missing_names = _records_cover_required_attacks(
        runtime_detection_records,
        "runtime_detection_status",
        "ready",
        config["required_runtime_attack_names"],
    )
    runtime_attack_formal_ready, runtime_attack_formal_ready_count = _runtime_attack_records_formal_ready(runtime_attack_records)
    runtime_detection_formal_ready, runtime_detection_formal_ready_count = _runtime_detection_records_formal_ready(runtime_detection_records)

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
    formal_internal_ablation_ready, formal_internal_ablation_variant_count, formal_internal_ablation_status = _formal_internal_ablation_summary_ready(run_root)
    (
        adaptive_attack_ready,
        adaptive_attack_record_count,
        non_runtime_attack_protocol_count,
        observed_non_runtime_attack_protocols,
        missing_non_runtime_attack_protocols,
        adaptive_attack_status,
    ) = _adaptive_attack_ready(
        run_root,
        config["required_non_runtime_attack_protocols"],
        config["minimum_non_runtime_attack_protocol_count"],
    )
    replay_or_sketch_ready, replay_or_sketch_status = _replay_or_sketch_ready(run_root)
    mechanism_contract_audit = config.get("paper_mechanism_contract_audit") or {}
    mechanism_contract_ready = mechanism_contract_audit.get("paper_mechanism_contract_decision") == "PASS"
    complete_claim_decision = _read_json(
        run_root / "artifacts" / "complete_paper_mechanism_claim_decision.json"
    )
    complete_claim_ready = complete_claim_decision.get("complete_paper_mechanism_claim_decision") == "PASS"
    confidence_interval_ready, confidence_interval_status = _confidence_interval_ready(run_root)
    low_fpr_ready, low_fpr_record_count, low_fpr_status = _low_fpr_formal_statistics_ready(run_root)
    paper_skeleton_ready, paper_skeleton_status = _paper_result_artifact_skeleton_ready(run_root)
    sstw_measured_formal_ready, sstw_measured_formal_record_count, sstw_measured_formal_status = _sstw_measured_formal_ready(run_root)
    fair_detection_ready, fair_detection_ready_count, fair_detection_status = _fair_detection_calibration_ready(
        run_root,
        config["required_modern_external_baseline_adapter_names"],
        config["target_fpr"],
        config["minimum_clean_negative_count"],
        config["required_runtime_attack_names"],
    )
    formal_method_comparison_ready, formal_method_comparison_ready_count, formal_method_comparison_status = _formal_method_baseline_comparison_ready(
        run_root,
        config["required_modern_external_baseline_adapter_names"],
        config["target_fpr"],
        config["required_runtime_attack_names"],
    )
    formal_difference_interval_ready, formal_difference_interval_ready_count, formal_difference_interval_status = _formal_baseline_difference_interval_ready(
        run_root,
        config["required_modern_external_baseline_adapter_names"],
        config["target_fpr"],
        config["required_runtime_attack_names"],
    )
    (
        sstw_advantage_claim_ready,
        sstw_advantage_ready_baseline_count,
        sstw_advantage_missing_baseline_names,
        sstw_advantage_blocking_reasons,
        sstw_advantage_claim_status,
    ) = _paper_profile_sstw_advantage_claim_ready(
        run_root,
        required_modern_adapter_names=config["required_modern_external_baseline_adapter_names"],
        target_fpr=config["target_fpr"],
        minimum_advantage_baseline_count=config["minimum_sstw_advantage_baseline_count"],
        minimum_difference=config["minimum_sstw_tpr_at_target_fpr_difference"],
        require_ci_lower_above_zero=config["require_sstw_advantage_ci_lower_above_zero"],
    )
    external_baseline_self_containment_ready, external_baseline_self_containment_summary = _external_baseline_self_containment_ready(
        external_baseline_self_containment_decision,
        config["required_modern_external_baseline_adapter_names"],
    )
    artifact_rebuild_ready, artifact_rebuild_status = _artifact_rebuild_ready(run_root)
    motion_selection = select_motion_claim_generation_records(validation_generation_records, formal_metric_records)
    formal_motion_claim_ready = motion_selection.formal_motion_claim_status in FORMAL_MOTION_CLAIM_READY_STATUSES
    motion_threshold_ready = motion_threshold_decision.get("motion_threshold_calibration_ready") is True
    motion_exclusion_ready, motion_exclusion_excluded_count, motion_exclusion_status = _motion_consistency_exclusion_ready(run_root)
    formality_guard = build_paper_result_formality_guard(
        run_root,
        paper_result_level=str(config["paper_result_level"]),
        target_fpr=float(config["target_fpr"]),
    )
    evidence_closure = build_paper_profile_evidence_closure_audit(run_root, config_path)

    requirement_checks = {
        "paper_result_formality_guard_passed": formality_guard["paper_result_formality_guard_decision"] == "PASS",
        "validation_generation_records_ready": prompt_count >= config["minimum_prompt_count"] and seed_per_prompt_min >= config["minimum_seed_per_prompt"],
        "validation_motion_threshold_calibration_ready": (not config["require_motion_threshold_calibration_ready"]) or motion_threshold_ready,
        "validation_formal_motion_claim_ready": (not config["require_formal_motion_claim_ready"]) or formal_motion_claim_ready,
        "validation_motion_consistency_exclusion_report_ready": (not config["require_motion_consistency_exclusion_report"]) or motion_exclusion_ready,
        "validation_runtime_attack_protocol_config_ready": config["runtime_attack_protocol_audit"]["runtime_attack_protocol_decision"] == "PASS",
        "validation_attack_records_ready": _decision_pass(runtime_attack_decision, "runtime_attack_decision")
        and attack_count >= config["minimum_attack_count"]
        and runtime_attack_required_ready
        and runtime_attack_formal_ready,
        "validation_detection_records_ready": _decision_pass(runtime_detection_decision, "runtime_detection_decision")
        and runtime_detection_ready_count >= runtime_attack_ready_count > 0
        and runtime_detection_required_ready
        and runtime_detection_formal_ready,
        "validation_external_baseline_status_records_ready": (not config["require_external_baseline_status_records"]) or external_baseline_audit.get("external_baseline_status_decision") == "PASS",
        "validation_external_baseline_comparison_records_ready": (not config["require_external_baseline_comparison_records"]) or external_baseline_comparison_ready,
        "validation_external_baseline_self_containment_ready": (not config["require_external_baseline_self_containment_decision"]) or external_baseline_self_containment_ready,
        "validation_sstw_measured_formal_records_ready": (not config["require_sstw_measured_formal_records"]) or sstw_measured_formal_ready,
        "validation_fair_detection_calibration_ready": (not config["require_fair_detection_calibration"]) or fair_detection_ready,
        "validation_formal_method_baseline_comparison_ready": (not config["require_formal_method_baseline_comparison"]) or formal_method_comparison_ready,
        "validation_formal_baseline_difference_interval_ready": (not config["require_formal_baseline_difference_interval"]) or formal_difference_interval_ready,
        "paper_profile_sstw_advantage_claim_ready": (not config["require_sstw_advantage_claim_ready"]) or sstw_advantage_claim_ready,
        "validation_data_split_and_leakage_guard_ready": (not config["require_data_split_and_leakage_guard"]) or data_split_decision.get("data_split_and_leakage_guard_decision") == "PASS",
        "validation_internal_ablation_records_ready": (not config["require_internal_ablation_records"]) or internal_ablation_ready,
        "paper_profile_formal_internal_ablation_ready": (not config["require_formal_internal_ablation_summary"]) or formal_internal_ablation_ready,
        "validation_adaptive_attack_records_ready": (not config["require_adaptive_attack_records"]) or adaptive_attack_ready,
        "validation_replay_or_sketch_records_ready": (
            (not config["require_replay_and_sketch_full_support"])
            or replay_or_sketch_ready
        ),
        "validation_claim3_full_support_ready": (not config["require_claim3_full_support"]) or replay_or_sketch_ready,
        "validation_complete_paper_mechanism_contract_ready": (
            (not config["require_complete_paper_mechanism_contract"])
            or (mechanism_contract_ready and complete_claim_ready)
        ),
        "validation_confidence_interval_report_ready": (not config["require_confidence_interval_report"]) or confidence_interval_ready,
        "validation_low_fpr_formal_statistics_blocking_record_ready": (not config["require_low_fpr_formal_statistics_blocking_record"]) or low_fpr_ready,
        "validation_paper_result_artifact_skeleton_ready": (not config["require_paper_result_artifact_skeleton"]) or paper_skeleton_ready,
        "validation_artifact_rebuild_dry_run_ready": (not config["require_artifact_rebuild_dry_run"]) or artifact_rebuild_ready,
        "paper_profile_common_evidence_closure_ready": (
            evidence_closure["paper_profile_evidence_closure_decision"] == "PASS"
        ),
    }
    missing_requirements = list(dict.fromkeys(
        [name for name, passed in requirement_checks.items() if not passed] + hard_config_missing
    ))
    gate_decision = "PASS" if not missing_requirements else "FAIL"
    paper_result_level = str(config["paper_result_level"])
    if gate_decision == "PASS" and paper_result_level == "probe_paper":
        claim_support_status = "probe_paper_target_fpr_0_1_paper_claim_supported"
    elif gate_decision == "PASS":
        claim_support_status = "paper_profile_full_protocol_handoff_ready"
    else:
        claim_support_status = f"{paper_result_level}_blocked"
    profile_gate_field = f"{paper_result_level}_gate_decision"

    return {
        "stage_id": config["stage_id"],
        "run_root": str(run_root),
        "paper_profile_gate_decision": gate_decision,
        profile_gate_field: gate_decision,
        "claim_support_status": claim_support_status,
        "paper_claim_id": formality_guard["paper_claim_id"],
        "paper_claim_level": formality_guard["paper_claim_level"],
        "paper_claim_support_status": (
            formality_guard["paper_claim_support_status"]
            if gate_decision == "PASS"
            else f"{formality_guard['paper_claim_id']}_blocked"
        ),
        "paper_result_formality_guard_decision": formality_guard["paper_result_formality_guard_decision"],
        "paper_result_formality_guard_status": formality_guard["paper_result_formality_guard_status"],
        "paper_result_formality_guard_violation_count": formality_guard["paper_result_formality_guard_violation_count"],
        "paper_result_formality_guard_scanned_file_count": formality_guard["paper_result_formality_guard_scanned_file_count"],
        "paper_result_formality_guard_blocking_terms": formality_guard["paper_result_formality_guard_blocking_terms"],
        "paper_result_formality_guard_violations": formality_guard["paper_result_formality_guard_violations"],
        "paper_result_level": config["paper_result_level"],
        "target_fpr": config["target_fpr"],
        "missing_validation_requirements": missing_requirements,
        "validation_missing_requirement_count": len(missing_requirements),
        "paper_profile_hard_required_config_missing": hard_config_missing,
        "paper_profile_hard_required_config_missing_count": len(hard_config_missing),
        **evidence_closure,
        "paper_profile_names": sorted(paper_profile_names),
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
        "motion_consistency_exclusion_excluded_count": motion_exclusion_excluded_count,
        "motion_consistency_exclusion_status": motion_exclusion_status,
        "runtime_attack_decision": runtime_attack_decision.get("runtime_attack_decision"),
        "runtime_attack_protocol_decision": config["runtime_attack_protocol_audit"]["runtime_attack_protocol_decision"],
        "runtime_attack_family_counts": config["runtime_attack_protocol_audit"]["runtime_attack_family_counts"],
        "runtime_attack_missing_family_minimums": config["runtime_attack_protocol_audit"]["runtime_attack_missing_family_minimums"],
        "required_non_runtime_attack_protocols": sorted(config["required_non_runtime_attack_protocols"]),
        "missing_non_runtime_attack_protocols": config["runtime_attack_protocol_audit"]["missing_non_runtime_attack_protocols"],
        "runtime_attack_ready_count": runtime_attack_ready_count,
        "runtime_attack_formal_ready_count": runtime_attack_formal_ready_count,
        "runtime_attack_count": attack_count,
        "required_runtime_attack_names": sorted(config["required_runtime_attack_names"]),
        "runtime_attack_observed_names": runtime_attack_observed_names,
        "runtime_attack_missing_required_names": runtime_attack_missing_names,
        "runtime_attack_missing_required_count": len(runtime_attack_missing_names),
        "runtime_detection_decision": runtime_detection_decision.get("runtime_detection_decision"),
        "runtime_detection_ready_count": runtime_detection_ready_count,
        "runtime_detection_formal_ready_count": runtime_detection_formal_ready_count,
        "runtime_detection_observed_names": runtime_detection_observed_names,
        "runtime_detection_missing_required_names": runtime_detection_missing_names,
        "runtime_detection_missing_required_count": len(runtime_detection_missing_names),
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
        **external_baseline_self_containment_summary,
        "sstw_measured_formal_record_count": sstw_measured_formal_record_count,
        "sstw_measured_formal_status": sstw_measured_formal_status,
        "fair_detection_calibration_ready_count": fair_detection_ready_count,
        "fair_detection_calibration_status": fair_detection_status,
        "formal_method_baseline_comparison_ready_count": formal_method_comparison_ready_count,
        "formal_method_baseline_comparison_status": formal_method_comparison_status,
        "formal_baseline_difference_interval_ready_count": formal_difference_interval_ready_count,
        "formal_baseline_difference_interval_status": formal_difference_interval_status,
        "paper_profile_sstw_advantage_claim_ready": sstw_advantage_claim_ready,
        "paper_profile_sstw_advantage_ready_baseline_count": sstw_advantage_ready_baseline_count,
        "paper_profile_sstw_advantage_missing_baseline_names": sstw_advantage_missing_baseline_names,
        "paper_profile_sstw_advantage_blocking_reasons": sstw_advantage_blocking_reasons,
        "paper_profile_sstw_advantage_claim_status": sstw_advantage_claim_status,
        "minimum_sstw_advantage_baseline_count": config["minimum_sstw_advantage_baseline_count"],
        "minimum_sstw_tpr_at_target_fpr_difference": config["minimum_sstw_tpr_at_target_fpr_difference"],
        "require_sstw_advantage_ci_lower_above_zero": config["require_sstw_advantage_ci_lower_above_zero"],
        "data_split_and_leakage_guard_decision": data_split_decision.get("data_split_and_leakage_guard_decision"),
        "minimum_external_baseline_measured_adapter_count": config["minimum_external_baseline_measured_adapter_count"],
        "internal_ablation_record_count": internal_ablation_record_count,
        "internal_ablation_status": internal_ablation_status,
        "formal_internal_ablation_summary_variant_count": formal_internal_ablation_variant_count,
        "formal_internal_ablation_summary_status": formal_internal_ablation_status,
        "adaptive_attack_record_count": adaptive_attack_record_count,
        "adaptive_attack_status": adaptive_attack_status,
        "non_runtime_attack_protocol_count": non_runtime_attack_protocol_count,
        "observed_non_runtime_attack_protocols": observed_non_runtime_attack_protocols,
        "adaptive_attack_missing_non_runtime_protocols": missing_non_runtime_attack_protocols,
        "adaptive_attack_missing_non_runtime_protocol_count": len(missing_non_runtime_attack_protocols),
        "replay_or_sketch_status": replay_or_sketch_status,
        "require_claim3_full_support": config["require_claim3_full_support"],
        "paper_mechanism_contract_decision": mechanism_contract_audit.get("paper_mechanism_contract_decision"),
        "formal_mechanism_contract_id": mechanism_contract_audit.get("formal_mechanism_contract_id"),
        "paper_mechanism_contract_violations": mechanism_contract_audit.get("paper_mechanism_contract_violations", []),
        "complete_paper_mechanism_claim_decision": complete_claim_decision.get("complete_paper_mechanism_claim_decision"),
        "confidence_interval_status": confidence_interval_status,
        "low_fpr_formal_statistics_record_count": low_fpr_record_count,
        "low_fpr_formal_statistics_status": low_fpr_status,
        "paper_result_artifact_skeleton_status": paper_skeleton_status,
        "artifact_rebuild_status": artifact_rebuild_status,
        "full_paper_allowed": False,
        "full_paper_next_gate": (
            "pilot_paper_generative_probe_gate"
            if gate_decision == "PASS" and paper_result_level == "probe_paper"
            else ("probe_paper_generative_probe_gate" if gate_decision == "PASS" else "complete_missing_validation_requirements")
        ),
    }


def write_paper_profile_gate_audit(
    run_root: str | Path,
    config_path: str | Path = DEFAULT_PAPER_PROFILE_CONFIG,
) -> dict[str, Any]:
    """写出 paper profile gate records、table、decision 和 report。"""
    run_root = Path(run_root)
    audit = build_paper_profile_gate_audit(run_root, config_path)
    record = with_flow_evidence_protocol_defaults(
        {"record_version": "paper_profile_generative_probe_gate_v1", **audit},
        trajectory_source_level="paper_profile_gate_aggregated_records",
        flow_state_admissibility_status="paper_profile_ready" if audit["paper_profile_gate_decision"] == "PASS" else "paper_profile_blocked",
        claim_support_status=audit["claim_support_status"],
    )
    write_jsonl(run_root / "records" / "paper_profile_gate_records.jsonl", [record])
    write_csv(run_root / "tables" / "paper_profile_gate_table.csv", [record])
    write_json(run_root / "artifacts" / "paper_profile_gate_decision.json", audit)
    target_fpr_text = _format_fpr(audit.get("target_fpr"))
    if audit.get("paper_result_level") == "probe_paper":
        report_scope = (
            "该报告由已落盘的 governed records 与 decision artifacts 自动生成。它判断 probe_paper "
            "是否已经作为 target_fpr=0.1 的小样本论文闭合层完成闭环。该层级使用当前 protocol config "
            f"指定的 target_fpr={target_fpr_text} "
            "验证 records、tables、figures、reports、manifests、baseline、clean negative 公平校准、消融、46 个 runtime attack、"
            "11 个 non-runtime/adaptive 协议、CI、SSTW 对 5 个现代 baseline 的优势证据和 artifact rebuild 是否能够完整产出。"
            "通过后表示如果论文结论限定在 FPR=0.1 设定, 当前 probe_paper 结果可以支撑小样本论文写作; "
            "它仍不能外推到 pilot_paper 的 FPR=0.01 或 full_paper 的 FPR=0.001, 也不能直接进入 full_paper。\n\n"
        )
    else:
        report_scope = (
            "该报告由已落盘的 governed records 与 decision artifacts 自动生成。它只判断 paper_profile "
            "是否已经作为当前 paper profile 的小样本论文协议闭合检查完成闭环。该层级使用当前 protocol config "
            f"指定的 target_fpr={target_fpr_text} "
            "验证 records、tables、figures、reports、manifests、baseline、clean negative 公平校准、消融、46 个 runtime attack、"
            "11 个 non-runtime/adaptive 协议、CI 和 artifact rebuild 是否能够完整产出。"
            "该共享报告本身不定义阶段跳转; 具体跳转必须由 profile-specific transition decision 控制, "
            "也不能外推到 pilot_paper 的 FPR=0.01 或 full_paper 的 FPR=0.001。\n\n"
        )
    report = (
        "# Paper Profile Generative Probe Gate Report\n\n"
        + report_scope
        + f"- paper_profile_gate_decision: {audit['paper_profile_gate_decision']}\n"
        f"- claim_support_status: {audit['claim_support_status']}\n"
        f"- paper_claim_id: {audit['paper_claim_id']}\n"
        f"- paper_claim_support_status: {audit['paper_claim_support_status']}\n"
        f"- paper_result_formality_guard_decision: {audit['paper_result_formality_guard_decision']}\n"
        f"- paper_result_formality_guard_violation_count: {audit['paper_result_formality_guard_violation_count']}\n"
        f"- paper_result_level: {audit['paper_result_level']}\n"
        f"- target_fpr: {target_fpr_text}\n"
        f"- missing_validation_requirements: {', '.join(audit['missing_validation_requirements']) if audit['missing_validation_requirements'] else 'none'}\n"
        f"- paper_profile_hard_required_config_missing: {', '.join(audit['paper_profile_hard_required_config_missing']) if audit['paper_profile_hard_required_config_missing'] else 'none'}\n"
        f"- validation_generation_record_count: {audit['validation_generation_record_count']}\n"
        f"- validation_prompt_count: {audit['validation_prompt_count']}\n"
        f"- validation_seed_per_prompt_min: {audit['validation_seed_per_prompt_min']}\n"
        f"- runtime_attack_protocol_decision: {audit['runtime_attack_protocol_decision']}\n"
        f"- required_runtime_attack_names: {', '.join(audit['required_runtime_attack_names'])}\n"
        f"- runtime_attack_missing_required_names: {', '.join(audit['runtime_attack_missing_required_names']) if audit['runtime_attack_missing_required_names'] else 'none'}\n"
        f"- runtime_detection_missing_required_names: {', '.join(audit['runtime_detection_missing_required_names']) if audit['runtime_detection_missing_required_names'] else 'none'}\n"
        f"- required_non_runtime_attack_protocols: {', '.join(audit['required_non_runtime_attack_protocols'])}\n"
        f"- adaptive_attack_missing_non_runtime_protocols: {', '.join(audit['adaptive_attack_missing_non_runtime_protocols']) if audit['adaptive_attack_missing_non_runtime_protocols'] else 'none'}\n"
        f"- motion_threshold_calibration_decision: {audit['motion_threshold_calibration_decision']}\n"
        f"- formal_motion_claim_status: {audit['formal_motion_claim_status']}\n"
        f"- motion_consistency_exclusion_excluded_count: {audit['motion_consistency_exclusion_excluded_count']}\n"
        f"- motion_consistency_exclusion_status: {audit['motion_consistency_exclusion_status']}\n"
        f"- modern_external_baseline_main_comparison_ready_count: {audit['modern_external_baseline_main_comparison_ready_count']}\n"
        f"- external_baseline_comparison_record_count: {audit['external_baseline_comparison_record_count']}\n"
        f"- external_baseline_measured_adapter_count: {audit['external_baseline_measured_adapter_count']}\n"
        f"- modern_external_baseline_formal_measured_adapter_count: {audit['modern_external_baseline_formal_measured_adapter_count']}\n"
        f"- external_baseline_self_containment_decision: {audit['external_baseline_self_containment_decision']}\n"
        f"- external_baseline_self_containment_ready_count: {audit['external_baseline_self_containment_ready_count']}\n"
        f"- external_baseline_self_containment_gate_missing_requirements: {', '.join(audit['external_baseline_self_containment_gate_missing_requirements']) if audit['external_baseline_self_containment_gate_missing_requirements'] else 'none'}\n"
        f"- sstw_measured_formal_record_count: {audit['sstw_measured_formal_record_count']}\n"
        f"- sstw_measured_formal_status: {audit['sstw_measured_formal_status']}\n"
        f"- fair_detection_calibration_ready_count: {audit['fair_detection_calibration_ready_count']}\n"
        f"- fair_detection_calibration_status: {audit['fair_detection_calibration_status']}\n"
        f"- formal_method_baseline_comparison_ready_count: {audit['formal_method_baseline_comparison_ready_count']}\n"
        f"- formal_method_baseline_comparison_status: {audit['formal_method_baseline_comparison_status']}\n"
        f"- formal_baseline_difference_interval_ready_count: {audit['formal_baseline_difference_interval_ready_count']}\n"
        f"- formal_baseline_difference_interval_status: {audit['formal_baseline_difference_interval_status']}\n"
        f"- paper_profile_sstw_advantage_claim_ready: {str(audit['paper_profile_sstw_advantage_claim_ready']).lower()}\n"
        f"- paper_profile_sstw_advantage_ready_baseline_count: {audit['paper_profile_sstw_advantage_ready_baseline_count']}\n"
        f"- paper_profile_sstw_advantage_missing_baseline_names: {', '.join(audit['paper_profile_sstw_advantage_missing_baseline_names']) if audit['paper_profile_sstw_advantage_missing_baseline_names'] else 'none'}\n"
        f"- paper_profile_sstw_advantage_blocking_reasons: {', '.join(audit['paper_profile_sstw_advantage_blocking_reasons']) if audit['paper_profile_sstw_advantage_blocking_reasons'] else 'none'}\n"
        f"- formal_internal_ablation_summary_variant_count: {audit['formal_internal_ablation_summary_variant_count']}\n"
        f"- formal_internal_ablation_summary_status: {audit['formal_internal_ablation_summary_status']}\n"
        f"- low_fpr_formal_statistics_record_count: {audit['low_fpr_formal_statistics_record_count']}\n"
        f"- low_fpr_formal_statistics_status: {audit['low_fpr_formal_statistics_status']}\n"
        f"- paper_result_artifact_skeleton_status: {audit['paper_result_artifact_skeleton_status']}\n"
        f"- data_split_and_leakage_guard_decision: {audit['data_split_and_leakage_guard_decision']}\n"
        f"- missing_modern_external_baseline_formal_adapter_names: {', '.join(audit['missing_modern_external_baseline_formal_adapter_names']) if audit['missing_modern_external_baseline_formal_adapter_names'] else 'none'}\n"
        f"- full_paper_allowed: {str(audit['full_paper_allowed']).lower()}\n"
    )
    report_path = run_root / "reports" / "paper_profile_gate_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    paper_result_level = str(audit.get("paper_result_level") or "paper_profile")
    if paper_result_level != "paper_profile":
        profile_gate_field = f"{paper_result_level}_gate_decision"
        profile_record = {
            **record,
            "record_version": f"{paper_result_level}_gate_v1",
        }
        write_jsonl(run_root / "records" / f"{paper_result_level}_gate_records.jsonl", [profile_record])
        write_csv(run_root / "tables" / f"{paper_result_level}_gate_table.csv", [profile_record])
        write_json(run_root / "artifacts" / f"{paper_result_level}_gate_decision.json", audit)
        profile_report = report.replace(
            "Paper Profile Generative Probe Gate Report",
            f"{paper_result_level} Generative Probe Gate Report",
        )
        profile_report += f"\n- {profile_gate_field}: {audit.get(profile_gate_field)}\n"
        (run_root / "reports" / f"{paper_result_level}_gate_report.md").write_text(
            profile_report,
            encoding="utf-8",
        )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="审计 paper profile generative video probe gate。")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--config-path", default=DEFAULT_PAPER_PROFILE_CONFIG)
    parser.add_argument("--write-outputs", action="store_true")
    args = parser.parse_args()
    payload = write_paper_profile_gate_audit(args.run_root, args.config_path) if args.write_outputs else build_paper_profile_gate_audit(args.run_root, args.config_path)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
