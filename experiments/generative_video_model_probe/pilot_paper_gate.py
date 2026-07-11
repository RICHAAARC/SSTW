"""pilot_paper fixed-FPR gate 的自动审计入口。

该模块只读取已经落盘的 governed records, 不运行 GPU, 不补造样本。
与早期 workflow pilot 不同, 本 gate 明确采用论文实验同构的低 FPR 流程:
calibration split -> frozen threshold artifact -> held-out test split -> report / claim audit input。

通过该 gate 可以支持当前 protocol config 指定 target_fpr 下的 `pilot_paper`
规模论文级主张。`pilot_paper` 与 `full_paper` 共用同一份 attack 协议清单,
差异只允许来自样本规模、统计置信度和 protocol config 指定的 FPR 口径。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from experiments.generative_video_model_probe.formal_motion_claim_filter import (
    FORMAL_MOTION_CLAIM_READY_STATUSES,
    filter_records_to_motion_claim_eligible,
    record_identity_key,
    select_motion_claim_generation_records,
)
from experiments.generative_video_model_probe.external_baseline_runner import formal_score_record_ready_for_claim
from evaluation.attacks.video_runtime_attack_protocol import (
    audit_runtime_attack_protocol_config,
    load_protocol_config_with_shared_attack_protocol,
    required_runtime_attack_names_from_config,
)
from evaluation.protocol.flow_evidence_fields import with_flow_evidence_protocol_defaults
from evaluation.protocol.paper_result_formality_guard import build_paper_result_formality_guard
from evaluation.protocol.paper_profile_evidence_closure import (
    build_paper_profile_evidence_closure_audit,
)
from evaluation.protocol.record_writer import write_json, write_jsonl
from evaluation.protocol.table_builder import write_csv


DEFAULT_PILOT_PAPER_CONFIG = "configs/protocol/pilot_paper_generative_probe.json"
DEFAULT_PILOT_PROFILE_NAMES = {"pilot_paper"}
DEFAULT_MINIMUM_PROMPT_COUNT = 21
DEFAULT_MINIMUM_SEED_PER_PROMPT = 8
DEFAULT_MINIMUM_SPLIT_SEED_PER_PROMPT = 4
DEFAULT_MINIMUM_UNIQUE_VIDEO_COUNT = 168
DEFAULT_MINIMUM_SPLIT_UNIQUE_VIDEO_COUNT = 84
DEFAULT_MINIMUM_CALIBRATION_NEGATIVE_EVENT_COUNT = 1000
DEFAULT_MINIMUM_HELDOUT_NEGATIVE_EVENT_COUNT = 1000
DEFAULT_MINIMUM_HELDOUT_ATTACKED_POSITIVE_EVENT_COUNT = 200
DEFAULT_MINIMUM_NEGATIVE_FAMILY_COUNT = 4
DEFAULT_MINIMUM_NEGATIVE_EVENT_COUNT_PER_FAMILY = 200
DEFAULT_MINIMUM_ATTACK_EVENT_COUNT_PER_ATTACK = 60
DEFAULT_REQUIRED_MODERN_EXTERNAL_BASELINE_ADAPTER_NAMES = (
    "videoshield",
    "vidsig",
    "videoseal",
    "videomark",
    "wam_frame",
)
DEFAULT_REQUIRED_EXTERNAL_BASELINE_ADAPTER_NAMES = (
    *DEFAULT_REQUIRED_MODERN_EXTERNAL_BASELINE_ADAPTER_NAMES,
)
DEFAULT_MINIMUM_EXTERNAL_BASELINE_MEASURED_ADAPTER_COUNT = len(DEFAULT_REQUIRED_EXTERNAL_BASELINE_ADAPTER_NAMES)
DEFAULT_MINIMUM_MODERN_EXTERNAL_BASELINE_FORMAL_ADAPTER_COUNT = len(DEFAULT_REQUIRED_MODERN_EXTERNAL_BASELINE_ADAPTER_NAMES)
DEFAULT_REQUIRED_INTERNAL_ABLATION_VARIANTS = (
    "sstw_full_method",
    "endpoint_only_control",
    "trajectory_only_score",
    "without_velocity_constraint",
    "without_endpoint_aware_control",
    "without_replay_uncertainty_weighting",
    "without_flow_state_admissibility",
    "generic_ssm_baseline",
)
HARD_REQUIRED_PILOT_PAPER_CONFIG_FLAGS = (
    "require_probe_paper_gate_passed",
    "require_probe_paper_to_pilot_paper_transition_decision",
    "require_external_baseline_comparison_ready",
    "require_external_baseline_self_contained_outputs",
    "require_modern_external_baseline_formal_results",
    "require_fair_detection_calibration",
    "require_formal_method_baseline_comparison",
    "require_formal_baseline_difference_interval",
)
SCORE_FIELDS = (
    "sstw_raw_detector_score",
    "raw_detector_score",
    "sstw_score",
)
IGNORED_NEGATIVE_FAMILIES = {"", "none", "not_applicable", "not_evaluated"}
SSTW_METHOD_ID = "sstw_key_conditioned_flow_trajectory"


def _read_json(path: Path) -> dict:
    """读取 JSON artifact, 兼容 Windows 或 Colab 产生的 UTF-8 BOM。"""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _required_float(raw: dict[str, Any], field_name: str, config_path: Path) -> float:
    """从 protocol config 读取必填 float 字段, 缺失时 fail-closed。"""
    if field_name not in raw:
        raise KeyError(f"pilot_paper protocol config 缺少必填字段 {field_name}: {config_path}")
    return float(raw[field_name])


def _optional_float(raw: dict[str, Any], field_name: str) -> float | None:
    """从 protocol config 读取可选 float 字段。"""
    value = raw.get(field_name)
    if value in {None, ""}:
        return None
    return float(value)


def _format_fpr(value: float | None) -> str:
    """把 FPR 数值格式化为报告中的稳定短文本。"""
    if value is None:
        return "未配置"
    return f"{float(value):g}"


def _read_probe_paper_artifact(run_root: Path, relative_path: str) -> dict:
    """读取 probe_paper 上游 artifact。

    pilot_paper 只能从 probe_paper 进入。Colab / 服务器 workflow 会把 probe_paper
    的 paper gate 阶段包恢复到同级 run_root, 因此这里先查当前目录中的复用副本,
    再查同级 `probe_paper` run_root。
    """

    local_path = run_root / relative_path
    if local_path.exists():
        return _read_json(local_path)
    sibling_path = run_root.parent / "probe_paper" / relative_path
    return _read_json(sibling_path)


def _read_jsonl(path: Path) -> list[dict]:
    """读取 JSONL records, 文件不存在时返回空列表。"""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_config(config_path: str | Path = DEFAULT_PILOT_PAPER_CONFIG) -> dict[str, Any]:
    """读取 pilot_paper gate 配置。

    target_fpr 是 paper gate 的核心语义, 必须来自 protocol config。这里不再提供
    脚本内置默认值, 避免 probe_paper / pilot_paper / full_paper 切换时出现脚本
    默认值覆盖配置的问题。
    """
    path = Path(config_path)
    raw = load_protocol_config_with_shared_attack_protocol(path)
    return {
        "pilot_profile_names": raw.get("pilot_profile_names", sorted(DEFAULT_PILOT_PROFILE_NAMES)),
        "target_fpr": _required_float(raw, "target_fpr", path),
        "blocked_target_fpr": _optional_float(raw, "blocked_target_fpr"),
        "threshold_protocol": raw.get("threshold_protocol", "calibration_split_to_frozen_threshold_to_heldout_test_split"),
        "paper_result_level": raw.get("paper_result_level", "pilot_paper"),
        "paper_protocol_level": raw.get("paper_protocol_level", "paper_grade_protocol"),
        "paper_protocol_difference_from_full_paper": raw.get("paper_protocol_difference_from_full_paper", "sample_scale_only"),
        "minimum_prompt_count": int(raw.get("minimum_prompt_count", DEFAULT_MINIMUM_PROMPT_COUNT)),
        "minimum_seed_per_prompt": int(raw.get("minimum_seed_per_prompt", DEFAULT_MINIMUM_SEED_PER_PROMPT)),
        "minimum_calibration_seed_per_prompt": int(raw.get("minimum_calibration_seed_per_prompt", DEFAULT_MINIMUM_SPLIT_SEED_PER_PROMPT)),
        "minimum_test_seed_per_prompt": int(raw.get("minimum_test_seed_per_prompt", DEFAULT_MINIMUM_SPLIT_SEED_PER_PROMPT)),
        "minimum_unique_video_count": int(raw.get("minimum_unique_video_count", DEFAULT_MINIMUM_UNIQUE_VIDEO_COUNT)),
        "minimum_calibration_unique_video_count": int(raw.get("minimum_calibration_unique_video_count", DEFAULT_MINIMUM_SPLIT_UNIQUE_VIDEO_COUNT)),
        "minimum_test_unique_video_count": int(raw.get("minimum_test_unique_video_count", DEFAULT_MINIMUM_SPLIT_UNIQUE_VIDEO_COUNT)),
        "minimum_calibration_negative_event_count": int(raw.get("minimum_calibration_negative_event_count", DEFAULT_MINIMUM_CALIBRATION_NEGATIVE_EVENT_COUNT)),
        "minimum_heldout_test_negative_event_count": int(raw.get("minimum_heldout_test_negative_event_count", DEFAULT_MINIMUM_HELDOUT_NEGATIVE_EVENT_COUNT)),
        "minimum_heldout_attacked_positive_event_count": int(raw.get("minimum_heldout_attacked_positive_event_count", DEFAULT_MINIMUM_HELDOUT_ATTACKED_POSITIVE_EVENT_COUNT)),
        "minimum_clean_negative_count": int(raw.get("minimum_clean_negative_count", DEFAULT_MINIMUM_CALIBRATION_NEGATIVE_EVENT_COUNT)),
        "minimum_negative_family_count": int(raw.get("minimum_negative_family_count", DEFAULT_MINIMUM_NEGATIVE_FAMILY_COUNT)),
        "minimum_calibration_negative_event_count_per_family": int(raw.get("minimum_calibration_negative_event_count_per_family", DEFAULT_MINIMUM_NEGATIVE_EVENT_COUNT_PER_FAMILY)),
        "minimum_heldout_negative_event_count_per_family": int(raw.get("minimum_heldout_negative_event_count_per_family", DEFAULT_MINIMUM_NEGATIVE_EVENT_COUNT_PER_FAMILY)),
        "minimum_attack_event_count_per_attack": int(raw.get("minimum_attack_event_count_per_attack", DEFAULT_MINIMUM_ATTACK_EVENT_COUNT_PER_ATTACK)),
        "minimum_external_baseline_measured_adapter_count": int(raw.get("minimum_external_baseline_measured_adapter_count", DEFAULT_MINIMUM_EXTERNAL_BASELINE_MEASURED_ADAPTER_COUNT)),
        "minimum_pilot_paper_external_baseline_trace_count": int(raw.get("minimum_pilot_paper_external_baseline_trace_count", DEFAULT_MINIMUM_SPLIT_UNIQUE_VIDEO_COUNT)),
        "minimum_pilot_paper_internal_ablation_trace_count": int(raw.get("minimum_pilot_paper_internal_ablation_trace_count", DEFAULT_MINIMUM_SPLIT_UNIQUE_VIDEO_COUNT)),
        "minimum_internal_ablation_variant_count": int(raw.get("minimum_internal_ablation_variant_count", len(DEFAULT_REQUIRED_INTERNAL_ABLATION_VARIANTS))),
        "minimum_modern_external_baseline_formal_adapter_count": int(raw.get("minimum_modern_external_baseline_formal_adapter_count", DEFAULT_MINIMUM_MODERN_EXTERNAL_BASELINE_FORMAL_ADAPTER_COUNT)),
        "required_external_baseline_adapter_names": raw.get("required_external_baseline_adapter_names", list(DEFAULT_REQUIRED_EXTERNAL_BASELINE_ADAPTER_NAMES)),
        "required_modern_external_baseline_adapter_names": raw.get("required_modern_external_baseline_adapter_names", list(DEFAULT_REQUIRED_MODERN_EXTERNAL_BASELINE_ADAPTER_NAMES)),
        "required_internal_ablation_variants": raw.get("required_internal_ablation_variants", list(DEFAULT_REQUIRED_INTERNAL_ABLATION_VARIANTS)),
        "required_runtime_attack_names": list(required_runtime_attack_names_from_config(raw)),
        "runtime_attack_protocol_audit": audit_runtime_attack_protocol_config(raw),
        "require_external_baseline_comparison_ready": bool(raw.get("require_external_baseline_comparison_ready", True)),
        "require_external_baseline_self_contained_outputs": bool(raw.get("require_external_baseline_self_contained_outputs", True)),
        "require_fair_detection_calibration": bool(raw.get("require_fair_detection_calibration", True)),
        "require_formal_method_baseline_comparison": bool(raw.get("require_formal_method_baseline_comparison", True)),
        "require_formal_baseline_difference_interval": bool(raw.get("require_formal_baseline_difference_interval", True)),
        "require_modern_external_baseline_formal_results": bool(raw.get("require_modern_external_baseline_formal_results", True)),
        "require_internal_ablation_matrix_ready": bool(raw.get("require_internal_ablation_matrix_ready", True)),
        "require_motion_threshold_calibration_ready": bool(raw.get("require_motion_threshold_calibration_ready", True)),
        "require_probe_paper_gate_passed": bool(raw.get("require_probe_paper_gate_passed", False)),
        "require_probe_paper_to_pilot_paper_transition_decision": bool(raw.get("require_probe_paper_to_pilot_paper_transition_decision", False)),
        "require_formal_motion_claim_ready": bool(raw.get("require_formal_motion_claim_ready", True)),
    }


def _hard_required_config_missing(config: dict[str, Any]) -> list[str]:
    """检查 pilot_paper 是否试图关闭 probe_paper 与公平比较硬前置。

    pilot_paper 必须从已完成的 probe_paper 进入。该函数防止通过 protocol config
    跳过上游 probe_paper gate、阶段跳转判定、external baseline 自包含输出、
    measured_formal 结果和同 FPR 公平比较证据链。
    """
    return [
        f"{field_name}_must_be_true"
        for field_name in HARD_REQUIRED_PILOT_PAPER_CONFIG_FLAGS
        if config.get(field_name) is not True
    ]


def _unique_nonempty(records: Iterable[dict], field: str) -> set[str]:
    """从 records 中提取非空唯一字段值。"""
    return {str(record.get(field)) for record in records if record.get(field) not in {None, ""}}


def _seed_per_prompt_min(records: Iterable[dict]) -> int:
    """统计每个 prompt 下成功 seed 的最小数量。"""
    grouped: dict[str, set[str]] = defaultdict(set)
    for record in records:
        prompt_id = str(record.get("prompt_id") or "")
        seed_id = str(record.get("seed_id") or "")
        if prompt_id and seed_id:
            grouped[prompt_id].add(seed_id)
    return min((len(seed_ids) for seed_ids in grouped.values()), default=0)


def _score_value(record: dict) -> float | None:
    """按照保守优先级提取可用于 fixed-FPR 统计的分数。"""
    for field_name in SCORE_FIELDS:
        value = record.get(field_name)
        if value in {None, ""}:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _records_with_scores(records: Iterable[dict]) -> list[dict]:
    """只保留包含可解析分数的 records, 并写入审计用分数字段。"""
    rows: list[dict] = []
    for record in records:
        score = _score_value(record)
        if score is None:
            continue
        rows.append({**record, "pilot_paper_score": score})
    return rows


def _pilot_generation_records(generation_records: list[dict], profile_names: set[str]) -> list[dict]:
    """筛选 pilot_paper profile 产生的成功 generation records。"""
    return [
        record for record in generation_records
        if record.get("generation_status") == "success"
        and record.get("colab_runtime_profile") in profile_names
        and str(record.get("sample_role") or record.get("generation_sample_role") or "").lower() != "clean_negative"
    ]


def _records_by_split(records: Iterable[dict], split_name: str) -> list[dict]:
    """按 split 字段筛选 records。"""
    return [record for record in records if record.get("split") == split_name]


def _decision_pass(decision: dict, *field_names: str) -> bool:
    """检查任一指定决策字段是否为 PASS。"""

    return any(decision.get(field_name) == "PASS" for field_name in field_names)


def _identity_keys(records: Iterable[dict]) -> set[tuple[str, str, str, str]]:
    """提取 records 的稳定身份键集合。"""
    return {record_identity_key(record) for record in records}


def _records_in_keys(records: Iterable[dict], keys: set[tuple[str, str, str, str]]) -> list[dict]:
    """筛选属于给定 generation 身份集合的下游 records。"""
    return [record for record in records if record_identity_key(record) in keys]


def _trace_ids(records: Iterable[dict]) -> set[str]:
    """提取 records 中非空 trajectory trace id, 用于检查 baseline 和消融是否覆盖同一批样本。"""
    return {str(record.get("trajectory_trace_id")) for record in records if record.get("trajectory_trace_id") not in {None, ""}}


def _target_fpr_matches(record: dict, expected_target_fpr: float) -> bool:
    """检查 paper gate 消费的公平比较产物是否来自当前 protocol config。"""

    try:
        return abs(float(record.get("target_fpr")) - float(expected_target_fpr)) <= 1e-12
    except (TypeError, ValueError):
        return False


def _safe_int(value: Any) -> int | None:
    """把上游 gate artifact 中的计数字段安全转换为 int。"""

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    """把上游 frozen threshold 字段安全转换为 float。"""

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _nonnegative_int(record: dict, field_name: str) -> int:
    """读取 governed record 中的非负计数字段, 缺失时按 0 处理。"""

    value = _safe_int(record.get(field_name))
    return max(value or 0, 0)


def _string_list(value: Any) -> list[str]:
    """把上游 artifact 中的列表字段规整为字符串列表。"""

    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]


def _required_fair_method_ids(config: dict[str, Any]) -> set[str]:
    """返回 pilot_paper 公平比较必须覆盖的 SSTW 与 modern baseline 方法集合。"""

    return {
        SSTW_METHOD_ID,
        *{str(name) for name in config["required_modern_external_baseline_adapter_names"] if str(name)},
    }


def _official_sstw_frozen_threshold(
    run_root: Path,
    config: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    """读取公平校准阶段已经冻结的 SSTW 正式阈值。

    pilot gate 只能审计该只读 artifact, 不能再次从 calibration scores 选择阈值。
    这可以保证 probe、pilot、full 的阈值来源语义完全一致, 并避免同一运行内
    出现两套冻结阈值。
    """

    records = _read_jsonl(
        run_root / "records" / "fair_detection_calibration_records.jsonl"
    )
    record = next(
        (
            row
            for row in records
            if row.get("method_id") == SSTW_METHOD_ID
            and row.get("fair_comparison_status") == "ready"
            and row.get("metric_status") == "measured_formal"
            and _target_fpr_matches(row, config["target_fpr"])
        ),
        {},
    )
    threshold = _safe_float(record.get("calibrated_threshold"))
    calibration_fpr = _safe_float(
        record.get("calibration_fpr_at_calibrated_threshold")
    )
    heldout_fpr = _safe_float(record.get("heldout_fpr_at_calibrated_threshold"))
    tpr = _safe_float(record.get("tpr_at_target_fpr"))
    negative_units = [
        row
        for row in record.get("negative_detection_units_at_target_fpr") or []
        if isinstance(row, dict) and row.get("statistical_cluster_id")
    ]
    positive_units = [
        row
        for row in record.get("positive_detection_units_at_target_fpr") or []
        if isinstance(row, dict) and row.get("comparison_anchor_key")
    ]
    ready = (
        bool(record)
        and threshold is not None
        and calibration_fpr is not None
        and heldout_fpr is not None
        and tpr is not None
        and record.get("threshold_protocol") == config["threshold_protocol"]
        and record.get("threshold_source_split") == "calibration"
        and record.get("test_time_threshold_update_blocked") is True
    )
    calibration_count = _nonnegative_int(
        record,
        "calibration_clean_negative_score_count",
    )
    return ready, {
        "formal_frozen_threshold_record_id": record.get(
            "fair_detection_calibration_record_id"
        ),
        "formal_frozen_threshold_source_path": (
            "records/fair_detection_calibration_records.jsonl"
        ),
        "formal_frozen_threshold_value": threshold,
        "formal_frozen_threshold_source_split": record.get(
            "threshold_source_split"
        ),
        "formal_frozen_threshold_protocol": record.get("threshold_protocol"),
        "formal_frozen_threshold_test_time_update_blocked": record.get(
            "test_time_threshold_update_blocked"
        ),
        "formal_frozen_threshold_calibration_negative_count": calibration_count,
        "formal_frozen_threshold_calibration_fpr": calibration_fpr,
        "formal_frozen_threshold_heldout_fpr": heldout_fpr,
        "formal_frozen_threshold_tpr": tpr,
        "formal_frozen_threshold_heldout_false_positive_count": sum(
            1 for row in negative_units if row.get("false_positive_at_target_fpr") is True
        ),
        "formal_frozen_threshold_true_positive_count": sum(
            1 for row in positive_units if row.get("detected_at_target_fpr") is True
        ),
        "formal_frozen_threshold_ready": ready,
    }


def _probe_paper_gate_ready_for_pilot(decision: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """检查 probe_paper gate 是否为进入 pilot_paper 的完整 PASS。"""

    missing_requirements = _string_list(decision.get("missing_validation_requirements"))
    missing: list[str] = []
    if decision.get("probe_paper_gate_decision") != "PASS":
        missing.append("probe_paper_gate_decision_passed")
    if decision.get("claim_support_status") != "probe_paper_target_fpr_0_1_paper_claim_supported":
        missing.append("probe_paper_claim_support_status_ready")
    if decision.get("paper_claim_id") != "probe_claim":
        missing.append("probe_paper_claim_id_ready")
    if decision.get("paper_result_formality_guard_decision") != "PASS":
        missing.append("probe_paper_formality_guard_passed")
    if decision.get("paper_result_level") != "probe_paper":
        missing.append("probe_paper_result_level_current")
    if missing_requirements or _safe_int(decision.get("validation_missing_requirement_count")) != 0:
        missing.append("probe_paper_missing_requirements_empty")
    if decision.get("full_paper_allowed") is not False:
        missing.append("probe_paper_must_not_allow_full_paper")
    return not missing, {
        "probe_paper_gate_decision": decision.get("probe_paper_gate_decision"),
        "probe_paper_claim_support_status": decision.get("claim_support_status"),
        "probe_paper_gate_missing_requirements": missing_requirements,
        "probe_paper_gate_missing_requirement_count": _safe_int(decision.get("validation_missing_requirement_count")),
        "probe_paper_gate_fairness_missing_requirements": missing,
    }


def _probe_paper_transition_ready_for_pilot(decision: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """检查 probe_paper -> pilot_paper 跳转判定是否为完整 governed PASS。"""

    missing_transition_requirements = _string_list(decision.get("missing_transition_requirements"))
    allowed_profiles = _string_list(decision.get("allowed_next_result_profiles"))
    blocked_profiles = _string_list(decision.get("blocked_next_result_profiles"))
    missing: list[str] = []
    if decision.get("probe_paper_to_pilot_paper_transition_decision") != "PASS":
        missing.append("probe_paper_to_pilot_paper_transition_decision_passed")
    if decision.get("claim_support_status") != "probe_paper_ready_to_enter_pilot_paper":
        missing.append("probe_paper_transition_claim_support_status_ready")
    if decision.get("source_stage") != "probe_paper" or decision.get("target_stage") != "pilot_paper":
        missing.append("probe_paper_transition_source_target_registered")
    if decision.get("source_gate_passed") is not True:
        missing.append("probe_paper_transition_source_gate_passed")
    if missing_transition_requirements or _safe_int(decision.get("transition_missing_requirement_count")) != 0:
        missing.append("probe_paper_transition_missing_requirements_empty")
    if "pilot_paper" not in allowed_profiles:
        missing.append("probe_paper_transition_allows_pilot_paper")
    if not {"full_paper", "submission_freeze"}.issubset(set(blocked_profiles)):
        missing.append("probe_paper_transition_blocks_later_profiles")
    if decision.get("full_paper_allowed") is not False:
        missing.append("probe_paper_transition_must_not_allow_full_paper")
    return not missing, {
        "probe_paper_to_pilot_paper_transition_decision": decision.get("probe_paper_to_pilot_paper_transition_decision"),
        "probe_paper_transition_claim_support_status": decision.get("claim_support_status"),
        "probe_paper_transition_source_gate_passed": decision.get("source_gate_passed"),
        "probe_paper_transition_missing_requirements": missing_transition_requirements,
        "probe_paper_transition_missing_requirement_count": _safe_int(decision.get("transition_missing_requirement_count")),
        "probe_paper_transition_allowed_next_result_profiles": allowed_profiles,
        "probe_paper_transition_blocked_next_result_profiles": blocked_profiles,
        "probe_paper_transition_fairness_missing_requirements": missing,
    }


def _fair_detection_anchor_ready(record: dict, minimum_clean_negative_count: int) -> bool:
    """检查 fair calibration record 是否具备完整公平比较证据。

    pilot_paper 不能消费旧版本流程中手工标成 ready 的
    fair calibration 记录。这里显式要求 clean negative 数量、positive anchor
    和 formal evidence 缺口计数全部满足当前 protocol config。
    """

    return (
        _nonnegative_int(record, "clean_negative_score_count") >= minimum_clean_negative_count
        and _nonnegative_int(record, "positive_anchor_count") > 0
        and _nonnegative_int(record, "positive_anchor_missing_count") == 0
        and _nonnegative_int(record, "positive_formal_evidence_missing_count") == 0
        and _nonnegative_int(record, "negative_formal_evidence_missing_count") == 0
    )


def _formal_comparison_anchor_ready(record: dict) -> bool:
    """检查同协议统计行是否与 SSTW reference anchors 对齐。"""

    status = str(record.get("comparison_anchor_alignment_status") or "")
    return (
        status in {"reference_method_anchor_set_ready", "aligned_with_sstw_reference_anchors"}
        and int(record.get("comparison_anchor_count") or 0) > 0
        and int(record.get("missing_reference_anchor_count") or 0) == 0
        and int(record.get("extra_anchor_count") or 0) == 0
    )


def _difference_interval_anchor_ready(record: dict) -> bool:
    """检查差值区间是否来自 prompt / seed / attack 完全配对的比较单元。"""

    return (
        str(record.get("comparison_anchor_alignment_status") or "") == "aligned_with_sstw_reference_anchors"
        and int(record.get("paired_comparison_unit_count") or 0) > 0
        and int(record.get("unpaired_reference_anchor_count") or 0) == 0
        and int(record.get("unpaired_baseline_anchor_count") or 0) == 0
    )


def _external_baseline_self_containment_ready(run_root: Path) -> tuple[bool, dict[str, Any]]:
    """检查 pilot_paper external baseline 是否完成项目内 clone/build/run/adapt/record 闭环。"""

    decision = _read_json(run_root / "artifacts" / "external_baseline_self_containment_decision.json")
    ready = decision.get("external_baseline_self_containment_decision") == "PASS"
    return ready, {
        "external_baseline_self_containment_decision": decision.get("external_baseline_self_containment_decision"),
        "self_contained_modern_external_baseline_count": decision.get("self_contained_modern_external_baseline_count", 0),
        "missing_self_contained_modern_external_baseline_names": decision.get("missing_self_contained_modern_external_baseline_names", []),
        "missing_self_containment_requirements": decision.get("missing_self_containment_requirements", []),
    }


def _fair_detection_calibration_ready(run_root: Path, config: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """检查 pilot_paper 是否已经完成同 target FPR 的公平检测校准。"""

    records = _read_jsonl(run_root / "records" / "fair_detection_calibration_records.jsonl")
    decision = _read_json(run_root / "artifacts" / "fair_detection_calibration_decision.json")
    required_method_ids = _required_fair_method_ids(config)
    ready_method_ids = {
        str(record.get("method_id") or "")
        for record in records
        if record.get("fair_comparison_status") == "ready"
        and record.get("metric_status") == "measured_formal"
        and _target_fpr_matches(record, config["target_fpr"])
        and _fair_detection_anchor_ready(record, config["minimum_clean_negative_count"])
    }
    missing_method_ids = sorted(required_method_ids - ready_method_ids)
    ready_count = len(ready_method_ids)
    ready = (
        bool(records)
        and decision.get("fair_detection_calibration_decision") == "PASS"
        and _target_fpr_matches(decision, config["target_fpr"])
        and ready_count >= len(required_method_ids)
        and not missing_method_ids
    )
    return ready, {
        "fair_detection_calibration_decision": decision.get("fair_detection_calibration_decision"),
        "fair_detection_calibration_ready_count": ready_count,
        "fair_detection_calibration_missing_method_ids": missing_method_ids,
        "fair_detection_calibration_target_fpr": decision.get("target_fpr"),
        "fair_detection_calibration_status": decision.get("claim_support_status", "missing_fair_detection_calibration_decision"),
    }


def _formal_method_baseline_comparison_ready(run_root: Path, config: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """检查 pilot_paper 是否已经产出 SSTW 与 5 个 baseline 的同协议统计表。"""

    records = _read_jsonl(run_root / "records" / "formal_method_baseline_comparison_records.jsonl")
    decision = _read_json(run_root / "artifacts" / "formal_method_baseline_comparison_decision.json")
    required_method_ids = _required_fair_method_ids(config)
    ready_method_ids = {
        str(record.get("method_id") or "")
        for record in records
        if record.get("metric_status") == "measured_formal"
        and _target_fpr_matches(record, config["target_fpr"])
        and _formal_comparison_anchor_ready(record)
    }
    missing_method_ids = sorted(required_method_ids - ready_method_ids)
    ready_count = int(decision.get("formal_comparison_ready_method_count") or len(ready_method_ids))
    ready = (
        bool(records)
        and decision.get("formal_method_baseline_comparison_decision") == "PASS"
        and _target_fpr_matches(decision, config["target_fpr"])
        and ready_count >= len(required_method_ids)
        and not missing_method_ids
    )
    return ready, {
        "formal_method_baseline_comparison_decision": decision.get("formal_method_baseline_comparison_decision"),
        "formal_method_baseline_comparison_ready_count": ready_count,
        "formal_method_baseline_comparison_missing_method_ids": missing_method_ids,
        "formal_method_baseline_comparison_target_fpr": decision.get("target_fpr"),
        "formal_method_baseline_comparison_status": decision.get("claim_support_status", "missing_formal_method_baseline_comparison_decision"),
    }


def _formal_baseline_difference_interval_ready(run_root: Path, config: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """检查 pilot_paper 是否已经产出 SSTW 相对 5 个 baseline 的配对差值区间。"""

    records = _read_jsonl(run_root / "records" / "formal_baseline_difference_interval_records.jsonl")
    decision = _read_json(run_root / "artifacts" / "formal_baseline_difference_interval_decision.json")
    required_baseline_ids = {str(name) for name in config["required_modern_external_baseline_adapter_names"] if str(name)}
    ready_baseline_ids = {
        str(record.get("baseline_method_id") or "")
        for record in records
        if record.get("difference_interval_status") == "ready"
        and record.get("metric_status") == "measured_formal"
        and _target_fpr_matches(record, config["target_fpr"])
        and _difference_interval_anchor_ready(record)
    }
    missing_baseline_ids = sorted(required_baseline_ids - ready_baseline_ids)
    ready_count = int(decision.get("difference_interval_ready_count") or len(ready_baseline_ids))
    ready = (
        bool(records)
        and decision.get("formal_baseline_difference_interval_decision") == "PASS"
        and _target_fpr_matches(decision, config["target_fpr"])
        and ready_count >= len(required_baseline_ids)
        and not missing_baseline_ids
    )
    return ready, {
        "formal_baseline_difference_interval_decision": decision.get("formal_baseline_difference_interval_decision"),
        "formal_baseline_difference_interval_ready_count": ready_count,
        "formal_baseline_difference_interval_missing_baseline_ids": missing_baseline_ids,
        "formal_baseline_difference_interval_target_fpr": decision.get("target_fpr"),
        "formal_baseline_difference_interval_status": decision.get("claim_support_status", "missing_formal_baseline_difference_interval_decision"),
    }


def _external_baseline_readiness(
    run_root: Path,
    config: dict[str, Any],
    required_trace_ids: set[str],
) -> tuple[bool, dict[str, Any]]:
    """审计 pilot_paper 是否已有 external_baseline adapter comparison 结果。

    这一检查属于项目特定写法。它不把 unsupported modern baseline 当作正向比较证据。
    所有进入 pilot_paper 的 external baseline 都必须是完整 measured_formal,
    即必须同时具备 prompt / seed / attack anchor、自身 clean negative 校准分数和项目内 official run 证据,
    并且必须覆盖 pilot_paper held-out test trace。
    """
    decision = _read_json(run_root / "artifacts" / "external_baseline_comparison_decision.json")
    records = _read_jsonl(run_root / "records" / "external_baseline_score_records.jsonl")
    formal_candidate_records = [record for record in records if record.get("metric_status") == "measured_formal"]
    formal_records = [record for record in formal_candidate_records if formal_score_record_ready_for_claim(record)]
    formal_incomplete_records = [
        record
        for record in formal_candidate_records
        if not formal_score_record_ready_for_claim(record)
    ]
    measured_records = [record for record in records if formal_score_record_ready_for_claim(record)]
    measured_adapter_names = {str(record.get("external_baseline_name")) for record in measured_records if record.get("external_baseline_name")}
    formal_adapter_names = {str(record.get("external_baseline_name")) for record in formal_records if record.get("external_baseline_name")}
    required_adapter_names = set(str(name) for name in config["required_external_baseline_adapter_names"])
    required_modern_adapter_names = set(str(name) for name in config["required_modern_external_baseline_adapter_names"])
    covered_trace_ids = _trace_ids(measured_records) & required_trace_ids
    missing_adapter_names = sorted(required_adapter_names - measured_adapter_names)
    missing_modern_formal_adapter_names = sorted(required_modern_adapter_names - formal_adapter_names)
    trace_ids_by_adapter: dict[str, set[str]] = defaultdict(set)
    for record in measured_records:
        adapter_name = str(record.get("external_baseline_name") or "")
        trace_id = str(record.get("trajectory_trace_id") or "")
        if adapter_name and trace_id and adapter_name in required_adapter_names:
            trace_ids_by_adapter[adapter_name].add(trace_id)
    adapter_trace_counts = {
        adapter_name: len((trace_ids_by_adapter.get(adapter_name) or set()) & required_trace_ids)
        for adapter_name in sorted(required_adapter_names)
    }
    adapter_trace_count_min = min(adapter_trace_counts.values(), default=0)
    ready = (
        decision.get("external_baseline_comparison_decision") == "PASS"
        and len(measured_adapter_names) >= config["minimum_external_baseline_measured_adapter_count"]
        and len(formal_adapter_names & required_modern_adapter_names) >= config["minimum_modern_external_baseline_formal_adapter_count"]
        and not missing_adapter_names
        and (not config["require_modern_external_baseline_formal_results"] or not missing_modern_formal_adapter_names)
        and adapter_trace_count_min >= config["minimum_pilot_paper_external_baseline_trace_count"]
    )
    return ready, {
        "external_baseline_comparison_decision": decision.get("external_baseline_comparison_decision"),
        "external_baseline_comparison_table_status": decision.get("external_baseline_comparison_table_status"),
        "external_baseline_measured_adapter_count": len(measured_adapter_names),
        "external_baseline_measured_adapter_names": sorted(measured_adapter_names),
        "external_baseline_formal_measured_adapter_count": len(formal_adapter_names),
        "external_baseline_formal_measured_adapter_names": sorted(formal_adapter_names),
        "external_baseline_formal_candidate_record_count": len(formal_candidate_records),
        "external_baseline_formal_incomplete_record_count": len(formal_incomplete_records),
        "modern_external_baseline_formal_measured_adapter_count": len(formal_adapter_names & required_modern_adapter_names),
        "modern_external_baseline_formal_measured_adapter_names": sorted(formal_adapter_names & required_modern_adapter_names),
        "required_external_baseline_adapter_names": sorted(required_adapter_names),
        "required_modern_external_baseline_adapter_names": sorted(required_modern_adapter_names),
        "missing_external_baseline_adapter_names": missing_adapter_names,
        "missing_modern_external_baseline_formal_adapter_names": missing_modern_formal_adapter_names,
        "pilot_paper_external_baseline_trace_count": len(covered_trace_ids),
        "pilot_paper_external_baseline_trace_count_min": adapter_trace_count_min,
        "pilot_paper_external_baseline_trace_counts": adapter_trace_counts,
        "minimum_pilot_paper_external_baseline_trace_count": config["minimum_pilot_paper_external_baseline_trace_count"],
        "external_baseline_claim_support_status": decision.get("external_baseline_claim_support_status"),
    }


def _internal_ablation_readiness(
    run_root: Path,
    config: dict[str, Any],
    required_trace_ids: set[str],
) -> tuple[bool, dict[str, Any]]:
    """审计 pilot_paper 是否已有正式内部消融矩阵 records。"""
    decision = _read_json(run_root / "artifacts" / "validation_internal_ablation_decision.json")
    records = _read_jsonl(run_root / "records" / "formal_internal_ablation_variant_records.jsonl")
    if not records:
        records = _read_jsonl(run_root / "records" / "validation_internal_ablation_records.jsonl")
    formal_records = [
        record
        for record in records
        if record.get("metric_status") == "measured_formal"
        and record.get("formal_internal_ablation_evidence_level") == "formal_component_removal_video_detector"
    ]
    required_variants = set(str(name) for name in config["required_internal_ablation_variants"])
    variants = {str(record.get("method_variant")) for record in formal_records if record.get("method_variant")}
    missing_variants = sorted(required_variants - variants)
    trace_ids_by_variant: dict[str, set[str]] = defaultdict(set)
    for record in formal_records:
        variant = str(record.get("method_variant") or "")
        trace_id = str(record.get("trajectory_trace_id") or "")
        if variant and trace_id and variant in required_variants:
            trace_ids_by_variant[variant].add(trace_id)
    variant_trace_counts = {
        variant: len((trace_ids_by_variant.get(variant) or set()) & required_trace_ids)
        for variant in sorted(required_variants)
    }
    variant_trace_count_min = min(variant_trace_counts.values(), default=0)
    score_margin = decision.get("validation_internal_ablation_score_margin")
    try:
        score_margin_value = float(score_margin)
    except (TypeError, ValueError):
        score_margin_value = None
    ready = (
        decision.get("validation_internal_ablation_decision") == "PASS"
        and decision.get("validation_internal_ablation_evidence_level") == "formal_component_removal_video_detector"
        and decision.get("detector_only_video_reuse_decision") == "PASS"
        and decision.get("generation_variant_independent_video_decision") == "PASS"
        and bool(records)
        and len(formal_records) == len(records)
        and len(variants) >= config["minimum_internal_ablation_variant_count"]
        and not missing_variants
        and variant_trace_count_min >= config["minimum_pilot_paper_internal_ablation_trace_count"]
        and score_margin_value is not None
        and score_margin_value > 0
    )
    return ready, {
        "validation_internal_ablation_decision": decision.get("validation_internal_ablation_decision"),
        "internal_ablation_record_count": decision.get("internal_ablation_record_count", len(records)),
        "formal_internal_ablation_record_count": len(formal_records),
        "validation_internal_ablation_variant_count": len(variants),
        "required_internal_ablation_variants": sorted(required_variants),
        "missing_internal_ablation_variants": missing_variants,
        "pilot_paper_internal_ablation_trace_count_min": variant_trace_count_min,
        "pilot_paper_internal_ablation_trace_counts": variant_trace_counts,
        "minimum_pilot_paper_internal_ablation_trace_count": config["minimum_pilot_paper_internal_ablation_trace_count"],
        "validation_internal_ablation_score_margin": score_margin_value,
        "detector_only_video_reuse_decision": decision.get(
            "detector_only_video_reuse_decision"
        ),
        "generation_variant_independent_video_decision": decision.get(
            "generation_variant_independent_video_decision"
        ),
        "internal_ablation_claim_support_status": decision.get("claim_support_status"),
    }


def _negative_family_counts(records: Iterable[dict]) -> Counter[str]:
    """按独立 source-video cluster 统计真实 negative family 覆盖。"""

    clusters_by_family: dict[str, set[str]] = {}
    for record in records:
        family = str(record.get("negative_family") or "")
        if family not in IGNORED_NEGATIVE_FAMILIES:
            cluster_id = str(
                record.get("statistical_cluster_id")
                or f"{record.get('prompt_id')}::{record.get('seed_id')}"
            )
            clusters_by_family.setdefault(family, set()).add(cluster_id)
    return Counter({
        family: len(cluster_ids)
        for family, cluster_ids in clusters_by_family.items()
    })


def _cluster_maximum_negative_scores(records: Iterable[dict]) -> list[float]:
    """每个独立视频只保留最高负假设分数, 防止多 trial 稀释 FPR。"""

    scores_by_cluster: dict[str, float] = {}
    for record in records:
        cluster_id = str(
            record.get("statistical_cluster_id")
            or f"{record.get('prompt_id')}::{record.get('seed_id')}"
        )
        score = float(record["pilot_paper_score"])
        scores_by_cluster[cluster_id] = max(
            scores_by_cluster.get(cluster_id, float("-inf")),
            score,
        )
    return list(scores_by_cluster.values())


def _attack_counts(records: Iterable[dict]) -> Counter[str]:
    """统计 attacked positive records 在每个 attack 下的覆盖数。"""
    counter: Counter[str] = Counter()
    for record in records:
        attack_name = str(record.get("attack_name") or "")
        if attack_name and attack_name != "no_attack":
            counter[attack_name] += 1
    return counter


def _wrong_sampler_replay_rejected(records: Iterable[dict]) -> bool:
    """判断 wrong_sampler_replay 是否以受控负样本形式被拒绝。"""
    for record in records:
        joined = " ".join(str(record.get(field) or "") for field in ("negative_family", "control_name", "decision", "wrong_sampler_replay_status"))
        if "wrong_sampler_replay" not in joined:
            continue
        if record.get("wrong_sampler_replay_control_not_equivalent") is True:
            return True
        if record.get("decision") in {"replay_rejected", "controlled_negative_below_threshold"}:
            return True
    return False


def build_pilot_paper_gate_audit(
    run_root: str | Path,
    config_path: str | Path = DEFAULT_PILOT_PAPER_CONFIG,
) -> dict[str, Any]:
    """构建 pilot_paper fixed-FPR gate 审计结果。

    该函数是项目特定写法。它只汇总已落盘 records, 并强制区分 calibration split 与
    held-out test split。只有 calibration negative 用于冻结阈值; held-out test negative
    用于报告 FPR; held-out attacked positive 用于报告 TPR。
    """
    run_root = Path(run_root)
    config = _load_config(config_path)
    profile_names = set(config["pilot_profile_names"])
    hard_config_missing = _hard_required_config_missing(config)

    generation_records = _read_jsonl(run_root / "records" / "generation_records.jsonl")
    formal_metric_records = _read_jsonl(run_root / "records" / "formal_quality_motion_semantic_records.jsonl")
    pilot_generation_records = _pilot_generation_records(generation_records, profile_names)
    motion_selection = select_motion_claim_generation_records(pilot_generation_records, formal_metric_records)
    eligible_generation_records = motion_selection.eligible_generation_records
    calibration_generation_records = _records_by_split(eligible_generation_records, "calibration")
    test_generation_records = _records_by_split(eligible_generation_records, "test")
    calibration_keys = _identity_keys(calibration_generation_records)
    test_keys = _identity_keys(test_generation_records)
    test_trace_ids = _trace_ids(test_generation_records)

    sstw_measured_formal_records = _read_jsonl(run_root / "records" / "sstw_measured_formal_records.jsonl")
    sstw_formal_negative_records = [
        record for record in sstw_measured_formal_records
        if str(record.get("sample_role") or "").lower()
        in {"clean_negative", "controlled_negative"}
        and record.get("metric_status") == "measured_formal"
    ]
    sstw_formal_positive_records = [
        record for record in sstw_measured_formal_records
        if str(record.get("sample_role") or "").lower() not in {"clean_negative", "controlled_negative"}
        and record.get("metric_status") == "measured_formal"
    ]

    calibration_negative_records = _records_with_scores([
        record for record in sstw_formal_negative_records
        if _target_fpr_matches(record, config["target_fpr"]) and record.get("split") == "calibration"
    ])
    heldout_negative_records = _records_with_scores([
        record for record in sstw_formal_negative_records
        if _target_fpr_matches(record, config["target_fpr"]) and record.get("split") in {"test", "heldout", "heldout_test"}
    ])
    heldout_positive_records = _records_with_scores([
        record for record in sstw_formal_positive_records
        if _target_fpr_matches(record, config["target_fpr"]) and record.get("split") in {"test", "heldout", "heldout_test"}
    ])

    calibration_negative_scores = _cluster_maximum_negative_scores(
        calibration_negative_records
    )
    heldout_negative_scores = _cluster_maximum_negative_scores(
        heldout_negative_records
    )
    heldout_positive_scores = [float(record["pilot_paper_score"]) for record in heldout_positive_records]
    formal_frozen_threshold_ready, formal_frozen_threshold = (
        _official_sstw_frozen_threshold(run_root, config)
    )
    threshold = formal_frozen_threshold["formal_frozen_threshold_value"]
    calibration_fpr = formal_frozen_threshold[
        "formal_frozen_threshold_calibration_fpr"
    ]
    heldout_fpr = formal_frozen_threshold["formal_frozen_threshold_heldout_fpr"]
    tpr_at_fpr = formal_frozen_threshold["formal_frozen_threshold_tpr"]
    calibration_false_positive_count = round(
        float(calibration_fpr or 0.0)
        * int(
            formal_frozen_threshold[
                "formal_frozen_threshold_calibration_negative_count"
            ]
            or 0
        )
    )
    heldout_false_positive_count = formal_frozen_threshold[
        "formal_frozen_threshold_heldout_false_positive_count"
    ]
    true_positive_count = formal_frozen_threshold[
        "formal_frozen_threshold_true_positive_count"
    ]

    prompt_count = len(_unique_nonempty(eligible_generation_records, "prompt_id"))
    seed_per_prompt_min = _seed_per_prompt_min(eligible_generation_records)
    calibration_seed_per_prompt_min = _seed_per_prompt_min(calibration_generation_records)
    test_seed_per_prompt_min = _seed_per_prompt_min(test_generation_records)
    unique_video_count = len(_identity_keys(eligible_generation_records))
    calibration_unique_video_count = len(calibration_keys)
    test_unique_video_count = len(test_keys)

    calibration_family_counts = _negative_family_counts(calibration_negative_records)
    heldout_family_counts = _negative_family_counts(heldout_negative_records)
    attack_counts = _attack_counts(heldout_positive_records)
    calibration_negative_event_count_per_family_min = min(calibration_family_counts.values(), default=0)
    heldout_negative_event_count_per_family_min = min(heldout_family_counts.values(), default=0)
    attack_event_count_per_attack_min = min(attack_counts.values(), default=0)
    required_runtime_attack_names = {str(name) for name in config["required_runtime_attack_names"] if str(name)}
    missing_required_runtime_attack_names = sorted(required_runtime_attack_names - set(attack_counts))
    runtime_attack_protocol_ready = config["runtime_attack_protocol_audit"]["runtime_attack_protocol_decision"] == "PASS"

    path_gain_values: list[float] = []
    replay_uncertainty_values: list[float] = []
    negative_tail_statuses = {"not_inflated"} if heldout_fpr is not None and heldout_fpr <= config["target_fpr"] else set()
    replay_gate_decision = _read_json(
        run_root / "artifacts" / "replay_and_sketch_gate_decision.json"
    )
    wrong_sampler_replay_ready = (
        _decision_pass(replay_gate_decision, "replay_and_sketch_gate_decision")
        and int(replay_gate_decision.get("wrong_sampler_replay_record_count") or 0) > 0
        and replay_gate_decision.get("wrong_sampler_replay_record_count")
        == replay_gate_decision.get("wrong_sampler_replay_rejected_count")
    )
    external_baseline_ready, external_baseline_summary = _external_baseline_readiness(run_root, config, test_trace_ids)
    internal_ablation_ready, internal_ablation_summary = _internal_ablation_readiness(run_root, config, test_trace_ids)
    external_baseline_self_containment_ready, external_baseline_self_containment_summary = _external_baseline_self_containment_ready(run_root)
    fair_detection_ready, fair_detection_summary = _fair_detection_calibration_ready(run_root, config)
    formal_method_comparison_ready, formal_method_comparison_summary = _formal_method_baseline_comparison_ready(run_root, config)
    formal_difference_interval_ready, formal_difference_interval_summary = _formal_baseline_difference_interval_ready(run_root, config)

    probe_paper_decision = _read_probe_paper_artifact(run_root, "artifacts/probe_paper_gate_decision.json")
    probe_paper_to_pilot_transition = _read_probe_paper_artifact(run_root, "artifacts/probe_paper_to_pilot_paper_transition_decision.json")
    motion_threshold_decision = _read_json(run_root / "artifacts" / "motion_threshold_calibration_decision.json")
    raw_probe_paper_ready, probe_paper_summary = _probe_paper_gate_ready_for_pilot(probe_paper_decision)
    raw_probe_paper_to_pilot_transition_ready, probe_paper_transition_summary = _probe_paper_transition_ready_for_pilot(
        probe_paper_to_pilot_transition,
    )
    formal_motion_claim_ready = (
        not config["require_formal_motion_claim_ready"]
        or motion_selection.formal_motion_claim_status in FORMAL_MOTION_CLAIM_READY_STATUSES
    )
    motion_threshold_ready = (
        not config["require_motion_threshold_calibration_ready"]
        or motion_threshold_decision.get("motion_threshold_calibration_ready") is True
    )
    probe_paper_ready = (
        not config["require_probe_paper_gate_passed"]
        or raw_probe_paper_ready
    )
    probe_paper_to_pilot_transition_ready = (
        not config["require_probe_paper_to_pilot_paper_transition_decision"]
        or raw_probe_paper_to_pilot_transition_ready
    )
    formality_guard = build_paper_result_formality_guard(
        run_root,
        paper_result_level=str(config["paper_result_level"]),
        target_fpr=float(config["target_fpr"]),
    )
    evidence_closure = build_paper_profile_evidence_closure_audit(run_root, config_path)

    requirement_checks = {
        "paper_result_formality_guard_passed": formality_guard["paper_result_formality_guard_decision"] == "PASS",
        "probe_paper_gate_passed": probe_paper_ready,
        "probe_paper_to_pilot_paper_transition_decision_passed": probe_paper_to_pilot_transition_ready,
        "motion_threshold_calibration_ready": motion_threshold_ready,
        "formal_motion_claim_ready": formal_motion_claim_ready,
        "pilot_paper_profile_generation_records_ready": prompt_count >= config["minimum_prompt_count"]
        and seed_per_prompt_min >= config["minimum_seed_per_prompt"]
        and unique_video_count >= config["minimum_unique_video_count"],
        "pilot_paper_calibration_split_ready": calibration_seed_per_prompt_min >= config["minimum_calibration_seed_per_prompt"]
        and calibration_unique_video_count >= config["minimum_calibration_unique_video_count"],
        "pilot_paper_heldout_test_split_ready": test_seed_per_prompt_min >= config["minimum_test_seed_per_prompt"]
        and test_unique_video_count >= config["minimum_test_unique_video_count"],
        "calibration_negative_event_count_ready": len(calibration_negative_scores) >= config["minimum_calibration_negative_event_count"],
        "heldout_test_negative_event_count_ready": len(heldout_negative_scores) >= config["minimum_heldout_test_negative_event_count"],
        "heldout_attacked_positive_event_count_ready": len(heldout_positive_records) >= config["minimum_heldout_attacked_positive_event_count"],
        "calibration_negative_family_coverage_ready": (
            len(calibration_family_counts) >= config["minimum_negative_family_count"]
            and calibration_negative_event_count_per_family_min
            >= config["minimum_calibration_negative_event_count_per_family"]
        ),
        "heldout_negative_family_coverage_ready": (
            len(heldout_family_counts) >= config["minimum_negative_family_count"]
            and heldout_negative_event_count_per_family_min
            >= config["minimum_heldout_negative_event_count_per_family"]
        ),
        "pilot_paper_runtime_attack_protocol_config_ready": runtime_attack_protocol_ready,
        "attack_event_coverage_ready": bool(attack_counts)
        and not missing_required_runtime_attack_names
        and attack_event_count_per_attack_min >= config["minimum_attack_event_count_per_attack"],
        "formal_frozen_threshold_artifact_ready": formal_frozen_threshold_ready,
        "heldout_fpr_within_target": heldout_fpr is not None and heldout_fpr <= config["target_fpr"],
        "tpr_at_target_fpr_computable": tpr_at_fpr is not None,
        "path_marginal_gain_ready": formal_method_comparison_ready,
        "negative_tail_not_inflated": heldout_fpr is not None and heldout_fpr <= config["target_fpr"],
        "wrong_sampler_replay_rejected": wrong_sampler_replay_ready,
        "pilot_paper_external_baseline_comparison_ready": (not config["require_external_baseline_comparison_ready"]) or external_baseline_ready,
        "pilot_paper_external_baseline_self_containment_ready": (not config["require_external_baseline_self_contained_outputs"]) or external_baseline_self_containment_ready,
        "pilot_paper_fair_detection_calibration_ready": (not config["require_fair_detection_calibration"]) or fair_detection_ready,
        "pilot_paper_formal_method_baseline_comparison_ready": (not config["require_formal_method_baseline_comparison"]) or formal_method_comparison_ready,
        "pilot_paper_formal_baseline_difference_interval_ready": (not config["require_formal_baseline_difference_interval"]) or formal_difference_interval_ready,
        "pilot_paper_internal_ablation_matrix_ready": (not config["require_internal_ablation_matrix_ready"]) or internal_ablation_ready,
        "paper_profile_common_evidence_closure_ready": (
            evidence_closure["paper_profile_evidence_closure_decision"] == "PASS"
        ),
    }
    missing = list(dict.fromkeys(
        [name for name, passed in requirement_checks.items() if not passed] + hard_config_missing
    ))
    gate_decision = "PASS" if not missing else "FAIL"

    if not pilot_generation_records:
        claim_support_status = "blocked_until_pilot_paper_generation_records"
    elif missing:
        claim_support_status = "pilot_paper_blocked"
    else:
        claim_support_status = "pilot_paper_calibrated_heldout_claim_ready"

    return {
        "stage_id": "pilot_paper_generative_probe_gate",
        "run_root": str(run_root),
        "pilot_paper_gate_decision": gate_decision,
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
        "paper_protocol_level": config["paper_protocol_level"],
        "paper_protocol_difference_from_full_paper": config["paper_protocol_difference_from_full_paper"],
        "pilot_paper_protocol_matches_full_paper": config["paper_protocol_difference_from_full_paper"] in {
            "sample_scale_only",
            "sample_scale_and_target_fpr_only",
        },
        "pilot_paper_claim_allowed": gate_decision == "PASS",
        "missing_pilot_paper_requirements": missing,
        "pilot_paper_missing_requirement_count": len(missing),
        "pilot_paper_hard_required_config_missing": hard_config_missing,
        "pilot_paper_hard_required_config_missing_count": len(hard_config_missing),
        **evidence_closure,
        "pilot_profile_names": sorted(profile_names),
        "threshold_protocol": config["threshold_protocol"],
        **probe_paper_summary,
        **probe_paper_transition_summary,
        **external_baseline_summary,
        **external_baseline_self_containment_summary,
        **fair_detection_summary,
        **formal_method_comparison_summary,
        **formal_difference_interval_summary,
        **formal_frozen_threshold,
        **internal_ablation_summary,
        "target_fpr": config["target_fpr"],
        "blocked_target_fpr": config["blocked_target_fpr"],
        "threshold_id": formal_frozen_threshold.get(
            "formal_frozen_threshold_record_id"
        ),
        "threshold_source_split": formal_frozen_threshold.get(
            "formal_frozen_threshold_source_split"
        ),
        "test_time_threshold_update_blocked": True,
        "fpr_threshold_value": threshold,
        "calibration_negative_fpr_at_threshold": calibration_fpr,
        "calibration_negative_false_positive_count_at_threshold": calibration_false_positive_count,
        "heldout_negative_fpr_at_threshold": heldout_fpr,
        "heldout_negative_false_positive_count_at_threshold": heldout_false_positive_count,
        "observed_negative_fpr_at_threshold": heldout_fpr,
        "tpr_at_target_fpr": tpr_at_fpr,
        "tpr_at_fpr_01": tpr_at_fpr,
        "true_positive_count_at_threshold": true_positive_count,
        "target_fpr_claim_allowed": gate_decision == "PASS",
        "tpr_at_fpr_01_pilot_claim_allowed": gate_decision == "PASS",
        "blocked_target_fpr_claim_allowed": False,
        "tpr_at_fpr_001_claim_allowed": False,
        "full_paper_allowed": False,
        "generation_record_count": len(generation_records),
        "pilot_paper_generation_record_count": len(pilot_generation_records),
        "pilot_paper_motion_claim_eligible_generation_count": len(eligible_generation_records),
        "pilot_paper_prompt_count": prompt_count,
        "pilot_paper_seed_per_prompt_min": seed_per_prompt_min,
        "pilot_paper_calibration_seed_per_prompt_min": calibration_seed_per_prompt_min,
        "pilot_paper_test_seed_per_prompt_min": test_seed_per_prompt_min,
        "pilot_paper_unique_video_count": unique_video_count,
        "pilot_paper_calibration_unique_video_count": calibration_unique_video_count,
        "pilot_paper_test_unique_video_count": test_unique_video_count,
        "calibration_negative_event_count": len(calibration_negative_scores),
        "calibration_negative_raw_hypothesis_count": len(calibration_negative_records),
        "heldout_test_negative_event_count": len(heldout_negative_scores),
        "heldout_negative_raw_hypothesis_count": len(heldout_negative_records),
        "heldout_attacked_positive_event_count": len(heldout_positive_records),
        "heldout_negative_event_count": len(heldout_negative_scores),
        "attacked_positive_event_count": len(heldout_positive_records),
        "calibration_negative_family_count": len(calibration_family_counts),
        "heldout_negative_family_count": len(heldout_family_counts),
        "negative_family_count": len(heldout_family_counts),
        "calibration_negative_event_count_per_family_min": calibration_negative_event_count_per_family_min,
        "heldout_negative_event_count_per_family_min": heldout_negative_event_count_per_family_min,
        "negative_event_count_per_family_min": heldout_negative_event_count_per_family_min,
        "attack_count": len(attack_counts),
        "attack_event_count_per_attack_min": attack_event_count_per_attack_min,
        "required_runtime_attack_count": len(config["required_runtime_attack_names"]),
        "required_runtime_attack_names": sorted(config["required_runtime_attack_names"]),
        "missing_required_runtime_attack_names": missing_required_runtime_attack_names,
        "runtime_attack_protocol_decision": config["runtime_attack_protocol_audit"]["runtime_attack_protocol_decision"],
        "runtime_attack_family_counts": config["runtime_attack_protocol_audit"]["runtime_attack_family_counts"],
        "runtime_attack_missing_family_minimums": config["runtime_attack_protocol_audit"]["runtime_attack_missing_family_minimums"],
        "calibration_negative_family_event_counts": dict(sorted(calibration_family_counts.items())),
        "heldout_negative_family_event_counts": dict(sorted(heldout_family_counts.items())),
        "attack_event_counts": dict(sorted(attack_counts.items())),
        "path_marginal_gain_at_fixed_fpr": round(mean(path_gain_values), 6) if path_gain_values else None,
        "replay_uncertainty_mean": round(mean(replay_uncertainty_values), 6) if replay_uncertainty_values else None,
        "negative_tail_status": "not_inflated" if requirement_checks["negative_tail_not_inflated"] else "missing_or_not_ready",
        "wrong_sampler_replay_control_not_equivalent": requirement_checks["wrong_sampler_replay_rejected"],
        "formal_motion_claim_status": motion_selection.formal_motion_claim_status,
        "motion_threshold_calibration_decision": motion_threshold_decision.get("motion_threshold_calibration_decision"),
        "motion_threshold_id": motion_threshold_decision.get("motion_threshold_id"),
        "motion_threshold_source_split": motion_threshold_decision.get("motion_threshold_source_split"),
        "minimum_prompt_count": config["minimum_prompt_count"],
        "minimum_seed_per_prompt": config["minimum_seed_per_prompt"],
        "minimum_unique_video_count": config["minimum_unique_video_count"],
        "minimum_calibration_negative_event_count": config["minimum_calibration_negative_event_count"],
        "minimum_heldout_test_negative_event_count": config["minimum_heldout_test_negative_event_count"],
        "minimum_heldout_attacked_positive_event_count": config["minimum_heldout_attacked_positive_event_count"],
        "minimum_calibration_negative_event_count_per_family": config["minimum_calibration_negative_event_count_per_family"],
        "minimum_heldout_negative_event_count_per_family": config["minimum_heldout_negative_event_count_per_family"],
        "minimum_attack_event_count_per_attack": config["minimum_attack_event_count_per_attack"],
        "minimum_external_baseline_measured_adapter_count": config["minimum_external_baseline_measured_adapter_count"],
        "minimum_modern_external_baseline_formal_adapter_count": config["minimum_modern_external_baseline_formal_adapter_count"],
        "minimum_internal_ablation_variant_count": config["minimum_internal_ablation_variant_count"],
        "next_allowed_action": "report_pilot_paper_result_then_plan_full_paper_scaleup" if gate_decision == "PASS" else "complete_missing_pilot_paper_requirements",
        "next_forbidden_action": "do_not_report_blocked_target_fpr_or_full_paper_scale_claim_from_pilot_paper",
    }


def write_pilot_paper_gate_audit(
    run_root: str | Path,
    config_path: str | Path = DEFAULT_PILOT_PAPER_CONFIG,
) -> dict[str, Any]:
    """写出 pilot_paper fixed-FPR gate records、table、decision 和 report。"""
    run_root = Path(run_root)
    audit = build_pilot_paper_gate_audit(run_root, config_path)
    record = with_flow_evidence_protocol_defaults(
        {"record_version": "pilot_paper_gate_v1", **audit},
        trajectory_source_level="pilot_paper_gate_aggregated_records",
        flow_state_admissibility_status="pilot_paper_ready" if audit["pilot_paper_gate_decision"] == "PASS" else "pilot_paper_blocked",
        claim_support_status=audit["claim_support_status"],
    )
    write_jsonl(run_root / "records" / "pilot_paper_gate_records.jsonl", [record])
    write_csv(run_root / "tables" / "pilot_paper_gate_table.csv", [record])
    write_json(run_root / "artifacts" / "pilot_paper_gate_decision.json", audit)
    target_fpr_text = _format_fpr(audit.get("target_fpr"))
    blocked_target_fpr_text = _format_fpr(audit.get("blocked_target_fpr"))
    report = (
        "# pilot_paper fixed-FPR Paper Gate Report\n\n"
        "该报告由已落盘的 governed records 自动生成, 只读消费 fair detection calibration "
        "阶段已经冻结的阈值, 不在 gate 内再次校准, 并报告对应 held-out FPR 与 TPR。"
        "该报告可支持 pilot_paper 规模的 "
        f"TPR@target_fpr={target_fpr_text} 论文级结论。pilot_paper 是小规模跑完整 full_paper 协议并产出 "
        "pilot 级论文结果的阶段, 因此不再需要单独的前置预演阶段。"
        "pilot_paper 与 full_paper 的差异由 protocol config 显式记录, 包括样本规模、统计功效、target FPR "
        "和 attack coverage; "
        f"因此该报告不支持 blocked_target_fpr={blocked_target_fpr_text}、完整 full_paper attack coverage 或 full-paper 规模结论。\n\n"
        f"- pilot_paper_gate_decision: {audit['pilot_paper_gate_decision']}\n"
        f"- claim_support_status: {audit['claim_support_status']}\n"
        f"- paper_claim_id: {audit['paper_claim_id']}\n"
        f"- paper_claim_support_status: {audit['paper_claim_support_status']}\n"
        f"- paper_result_formality_guard_decision: {audit['paper_result_formality_guard_decision']}\n"
        f"- paper_result_formality_guard_violation_count: {audit['paper_result_formality_guard_violation_count']}\n"
        f"- paper_result_level: {audit['paper_result_level']}\n"
        f"- paper_protocol_difference_from_full_paper: {audit['paper_protocol_difference_from_full_paper']}\n"
        f"- threshold_protocol: {audit['threshold_protocol']}\n"
        f"- pilot_paper_hard_required_config_missing: {', '.join(audit['pilot_paper_hard_required_config_missing']) if audit['pilot_paper_hard_required_config_missing'] else 'none'}\n"
        f"- probe_paper_gate_decision: {audit['probe_paper_gate_decision']}\n"
        f"- probe_paper_gate_fairness_missing_requirements: {', '.join(audit['probe_paper_gate_fairness_missing_requirements']) if audit['probe_paper_gate_fairness_missing_requirements'] else 'none'}\n"
        f"- probe_paper_to_pilot_paper_transition_decision: {audit['probe_paper_to_pilot_paper_transition_decision']}\n"
        f"- probe_paper_transition_fairness_missing_requirements: {', '.join(audit['probe_paper_transition_fairness_missing_requirements']) if audit['probe_paper_transition_fairness_missing_requirements'] else 'none'}\n"
        f"- external_baseline_comparison_decision: {audit['external_baseline_comparison_decision']}\n"
        f"- external_baseline_self_containment_decision: {audit['external_baseline_self_containment_decision']}\n"
        f"- external_baseline_measured_adapter_count: {audit['external_baseline_measured_adapter_count']}\n"
        f"- modern_external_baseline_formal_measured_adapter_count: {audit['modern_external_baseline_formal_measured_adapter_count']}\n"
        f"- fair_detection_calibration_decision: {audit['fair_detection_calibration_decision']}\n"
        f"- fair_detection_calibration_ready_count: {audit['fair_detection_calibration_ready_count']}\n"
        f"- formal_method_baseline_comparison_decision: {audit['formal_method_baseline_comparison_decision']}\n"
        f"- formal_method_baseline_comparison_ready_count: {audit['formal_method_baseline_comparison_ready_count']}\n"
        f"- formal_baseline_difference_interval_decision: {audit['formal_baseline_difference_interval_decision']}\n"
        f"- formal_baseline_difference_interval_ready_count: {audit['formal_baseline_difference_interval_ready_count']}\n"
        f"- pilot_paper_external_baseline_trace_count_min: {audit['pilot_paper_external_baseline_trace_count_min']}\n"
        f"- validation_internal_ablation_decision: {audit['validation_internal_ablation_decision']}\n"
        f"- validation_internal_ablation_variant_count: {audit['validation_internal_ablation_variant_count']}\n"
        f"- missing_pilot_paper_requirements: {', '.join(audit['missing_pilot_paper_requirements']) if audit['missing_pilot_paper_requirements'] else 'none'}\n"
        f"- pilot_paper_generation_record_count: {audit['pilot_paper_generation_record_count']}\n"
        f"- pilot_paper_calibration_unique_video_count: {audit['pilot_paper_calibration_unique_video_count']}\n"
        f"- pilot_paper_test_unique_video_count: {audit['pilot_paper_test_unique_video_count']}\n"
        f"- runtime_attack_protocol_decision: {audit['runtime_attack_protocol_decision']}\n"
        f"- required_runtime_attack_count: {audit['required_runtime_attack_count']}\n"
        f"- missing_required_runtime_attack_names: {', '.join(audit['missing_required_runtime_attack_names']) if audit['missing_required_runtime_attack_names'] else 'none'}\n"
        f"- calibration_negative_event_count: {audit['calibration_negative_event_count']}\n"
        f"- heldout_test_negative_event_count: {audit['heldout_test_negative_event_count']}\n"
        f"- heldout_attacked_positive_event_count: {audit['heldout_attacked_positive_event_count']}\n"
        f"- calibration_negative_fpr_at_threshold: {audit['calibration_negative_fpr_at_threshold']}\n"
        f"- heldout_negative_fpr_at_threshold: {audit['heldout_negative_fpr_at_threshold']}\n"
        f"- target_fpr: {target_fpr_text}\n"
        f"- tpr_at_target_fpr: {audit['tpr_at_target_fpr']}\n"
        f"- target_fpr_claim_allowed: {str(audit['target_fpr_claim_allowed']).lower()}\n"
        f"- blocked_target_fpr: {blocked_target_fpr_text}\n"
        f"- blocked_target_fpr_claim_allowed: {str(audit['blocked_target_fpr_claim_allowed']).lower()}\n"
        f"- tpr_at_fpr_01: {audit['tpr_at_fpr_01']}\n"
        f"- tpr_at_fpr_001_claim_allowed: {str(audit['tpr_at_fpr_001_claim_allowed']).lower()}\n"
        f"- full_paper_allowed: {str(audit['full_paper_allowed']).lower()}\n"
    )
    report_path = run_root / "reports" / "pilot_paper_gate_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="审计 pilot_paper fixed-FPR gate。")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--config-path", default=DEFAULT_PILOT_PAPER_CONFIG)
    parser.add_argument("--write-outputs", action="store_true")
    args = parser.parse_args()
    payload = write_pilot_paper_gate_audit(args.run_root, args.config_path) if args.write_outputs else build_pilot_paper_gate_audit(args.run_root, args.config_path)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
