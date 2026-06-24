"""读取外部 baseline 配置并生成可审计状态记录。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from external_baseline.registry import adapter_status as external_adapter_status


ADAPTER_STATUS_BUILDERS = {
    "explicit_dtw_temporal_alignment": lambda: external_adapter_status("explicit_dtw_temporal_alignment"),
    "explicit_frame_matching_temporal_registration": lambda: external_adapter_status("explicit_frame_matching_temporal_registration"),
    "videoshield": lambda: external_adapter_status("videoshield"),
    "sigmark": lambda: external_adapter_status("sigmark"),
    "spdmark": lambda: external_adapter_status("spdmark"),
    "videomark_or_vidsig": lambda: external_adapter_status("videomark_or_vidsig"),
    "videoseal": lambda: external_adapter_status("videoseal"),
}

DEFAULT_BASELINE_STATUS_FIELDS: dict[str, Any] = {
    "external_baseline_adapter_status": "not_integrated",
    "external_baseline_input_compatibility_status": "not_evaluated",
    "external_baseline_output_record_status": "non_run_record_written",
    "external_baseline_threshold_policy_compatible": False,
    "external_baseline_attack_manifest_compatible": False,
    "external_baseline_result_used_for_claim": False,
    "external_baseline_command_config_status": "not_applicable",
    "external_baseline_command_env_var": "not_applicable",
}

MODERN_BASELINE_FAMILIES = {
    "in_generation_or_diffusion_video_watermark_baseline",
    "blind_extraction_video_watermark_baseline",
    "parameter_or_adapter_video_watermark_baseline",
    "latent_video_signature_baseline",
    "post_hoc_neural_video_watermark_baseline",
}


def load_external_baselines(path: str | Path) -> list[dict]:
    """读取外部 baseline 配置。

    该函数属于通用工程写法, 只负责把 JSON 配置解析成 baseline 条目。具体能否用于
    claim 必须由后续状态补全和 checker 判断, 不能仅凭配置存在就认为 baseline 已运行。
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(data.get("baselines", []))


def _baseline_layer(family: str) -> str:
    """根据 baseline family 判断其在论文比较中的层级。"""
    if family in MODERN_BASELINE_FAMILIES:
        return "modern_external_baseline"
    if family in {"explicit_temporal_alignment", "reference_guided_temporal_registration"}:
        return "explicit_synchronization_control"
    return "external_baseline"


def _main_comparison_ready(record: dict) -> bool:
    """判断 baseline 是否达到可进入主对比表的最低治理条件。

    该函数不运行 baseline, 只检查 record 中已经写出的状态字段。不能运行或协议不兼容的
    baseline 仍会保留 governed non-run record, 但不得支撑正向比较 claim。
    """
    return all([
        record.get("external_baseline_runnable_status") == "runnable",
        record.get("external_baseline_adapter_status") == "ready",
        record.get("external_baseline_output_record_status") == "governed_records_written",
        record.get("external_baseline_threshold_policy_compatible") is True,
        record.get("external_baseline_attack_manifest_compatible") is True,
        record.get("external_baseline_result_used_for_claim") is True,
    ])


def _complete_external_baseline_record(item: dict) -> dict:
    """补齐单个 baseline 的治理状态字段。"""
    status_builder = ADAPTER_STATUS_BUILDERS.get(item["external_baseline_name"])
    adapter_record = status_builder() if status_builder else {}
    record = {
        "record_version": "external_baseline_status_v2",
        **DEFAULT_BASELINE_STATUS_FIELDS,
        **item,
        **adapter_record,
    }
    family = str(record.get("external_baseline_family") or "unknown_external_baseline_family")
    record["external_baseline_layer"] = _baseline_layer(family)
    if record.get("external_baseline_runnable_status") != "runnable":
        record["external_baseline_result_used_for_claim"] = False
    if record["external_baseline_layer"] == "explicit_synchronization_control":
        record["external_baseline_result_used_for_claim"] = False
    record["external_baseline_main_comparison_ready"] = _main_comparison_ready(record)
    if record["external_baseline_main_comparison_ready"]:
        record["external_baseline_claim_support_status"] = "baseline_ready_for_main_comparison"
    elif record["external_baseline_runnable_status"] == "runnable":
        record["external_baseline_claim_support_status"] = "runnable_control_or_not_claim_ready"
    else:
        record["external_baseline_claim_support_status"] = "governed_non_run_record_only"
    return record


def build_external_baseline_records(path: str | Path) -> list[dict]:
    """生成外部 baseline 状态 records。

    该函数的职责是把所有配置中的 baseline 都写成 governed record。无法运行的现代
    baseline 不能静默删除, 必须保留 not-run reason、protocol gap 和 claim 边界。
    真正进入论文主表前, 仍需要 full-paper checker 验证同 split、同 attack manifest
    和同 threshold policy 下的正式 baseline scores。
    """
    return [_complete_external_baseline_record(item) for item in load_external_baselines(path)]


def audit_external_baseline_records(records: list[dict]) -> dict:
    """汇总外部 baseline 状态, 供 validation-scale 与 full-paper gate 使用。"""
    modern_records = [record for record in records if record.get("external_baseline_layer") == "modern_external_baseline"]
    main_ready_records = [record for record in records if record.get("external_baseline_main_comparison_ready") is True]
    non_run_records = [record for record in records if record.get("external_baseline_runnable_status") != "runnable"]
    return {
        "stage_id": "external_baseline_status_audit",
        "external_baseline_record_count": len(records),
        "modern_external_baseline_record_count": len(modern_records),
        "modern_external_baseline_status_records_ready": bool(modern_records),
        "modern_external_baseline_main_comparison_ready_count": sum(
            1 for record in modern_records if record.get("external_baseline_main_comparison_ready") is True
        ),
        "external_baseline_main_comparison_ready_count": len(main_ready_records),
        "external_baseline_non_run_record_count": len(non_run_records),
        "external_baseline_status_decision": "PASS" if records and modern_records else "FAIL",
        "external_baseline_claim_support_status": "governed_status_records_only"
        if modern_records and not main_ready_records
        else "baseline_main_comparison_ready",
    }
