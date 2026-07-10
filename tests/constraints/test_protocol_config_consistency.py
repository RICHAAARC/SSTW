"""约束 paper profile 协议配置中的 baseline 数量语义保持一致。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.attacks.video_runtime_attack_protocol import load_protocol_config_with_shared_attack_protocol


PAPER_PROFILE_PROTOCOL_CONFIGS = (
    Path("configs/protocol/probe_paper_generative_probe.json"),
    Path("configs/protocol/pilot_paper_generative_probe.json"),
    Path("configs/protocol/full_paper_generative_probe.json"),
)


PAPER_PROFILE_ARTIFACT_REQUIREMENT_FIELDS = (
    "require_adaptive_attack_records",
    "require_artifact_rebuild_dry_run",
    "require_artifact_rebuild_report",
    "require_claim_audit_report",
    "require_complete_result_artifact_skeleton",
    "require_confidence_interval_report",
    "require_data_split_and_leakage_guard",
    "require_efficiency_metric_records",
    "require_external_baseline_self_containment_decision",
    "require_low_fpr_curve_records",
    "require_low_fpr_formal_statistics_blocking_record",
    "require_paper_result_artifact_skeleton",
    "require_real_adaptive_attack_records",
    "require_real_world_attack_records",
    "require_replay_and_sketch_full_support",
    "reviewer_evidence_index_required",
    "require_sstw_measured_formal_records",
    "require_statistical_confidence_interval_decision",
    "require_video_quality_metric_records",
)


@pytest.mark.constraint
def test_external_baseline_minimum_count_matches_required_modern_baselines() -> None:
    """主实验 baseline 数量不能保留旧配置导致 probe_paper 被错误阻断。"""

    mismatches: list[dict[str, object]] = []
    for config_path in PAPER_PROFILE_PROTOCOL_CONFIGS:
        config = json.loads(config_path.read_text(encoding="utf-8-sig"))
        required_baselines = [
            str(name)
            for name in config.get("required_modern_external_baseline_adapter_names", [])
            if str(name)
        ]
        minimum_count = int(config.get("minimum_external_baseline_measured_adapter_count", -1))
        if minimum_count != len(required_baselines):
            mismatches.append({
                "config_path": config_path.as_posix(),
                "minimum_external_baseline_measured_adapter_count": minimum_count,
                "required_modern_external_baseline_adapter_count": len(required_baselines),
            })

    assert mismatches == []


@pytest.mark.constraint
def test_paper_profile_split_counts_match_declared_sample_capacity() -> None:
    """probe / pilot / full 的 calibration-test split 必须与样本容量一致。"""

    mismatches: list[dict[str, object]] = []
    for config_path in (
        Path("configs/protocol/probe_paper_generative_probe.json"),
        Path("configs/protocol/pilot_paper_generative_probe.json"),
        Path("configs/protocol/full_paper_generative_probe.json"),
    ):
        config = load_protocol_config_with_shared_attack_protocol(config_path)
        prompt_count = int(config["minimum_prompt_count"])
        seed_per_prompt = int(config["minimum_seed_per_prompt"])
        calibration_seed_per_prompt = int(config["minimum_calibration_seed_per_prompt"])
        test_seed_per_prompt = int(config["minimum_test_seed_per_prompt"])
        calibration_unique_video_count = int(config["minimum_calibration_unique_video_count"])
        test_unique_video_count = int(config["minimum_test_unique_video_count"])
        unique_video_count = int(config["minimum_unique_video_count"])
        required_attack_count = len(config["required_runtime_attack_names"])
        attack_event_count_per_attack = int(config["minimum_attack_event_count_per_attack"])
        heldout_attacked_positive_event_count = int(config["minimum_heldout_attacked_positive_event_count"])

        config_mismatches: list[str] = []
        if calibration_seed_per_prompt + test_seed_per_prompt != seed_per_prompt:
            config_mismatches.append("split_seed_per_prompt_sum_mismatch")
        if calibration_unique_video_count != prompt_count * calibration_seed_per_prompt:
            config_mismatches.append("calibration_unique_video_count_mismatch")
        if test_unique_video_count != prompt_count * test_seed_per_prompt:
            config_mismatches.append("test_unique_video_count_mismatch")
        if calibration_unique_video_count + test_unique_video_count != unique_video_count:
            config_mismatches.append("unique_video_count_sum_mismatch")
        if heldout_attacked_positive_event_count != attack_event_count_per_attack * required_attack_count:
            config_mismatches.append("heldout_attacked_positive_event_count_mismatch")

        if config_mismatches:
            mismatches.append({
                "config_path": config_path.as_posix(),
                "mismatches": config_mismatches,
            })

    assert mismatches == []


@pytest.mark.constraint
def test_probe_pilot_full_require_same_paper_artifact_outputs() -> None:
    """probe / pilot / full 必须声明同构论文产物要求。

    该约束不要求三者样本量相同, 但要求后续 full_paper 会消费的 records、tables、
    figures、reports 和 manifests 类产物在 probe_paper 与 pilot_paper 中已经显式
    登记, 防止昂贵 full_paper 运行时才发现产物链路缺口。
    """

    mismatches: list[dict[str, object]] = []
    for config_path in (
        Path("configs/protocol/probe_paper_generative_probe.json"),
        Path("configs/protocol/pilot_paper_generative_probe.json"),
        Path("configs/protocol/full_paper_generative_probe.json"),
    ):
        config = json.loads(config_path.read_text(encoding="utf-8-sig"))
        missing_or_false = [
            field_name
            for field_name in PAPER_PROFILE_ARTIFACT_REQUIREMENT_FIELDS
            if config.get(field_name) is not True
        ]
        if missing_or_false:
            mismatches.append({
                "config_path": config_path.as_posix(),
                "missing_or_false_artifact_requirement_fields": missing_or_false,
            })

    assert mismatches == []


@pytest.mark.constraint
def test_paper_profiles_match_the_same_common_mechanism_contract() -> None:
    """三个正式 profile 的机制字段必须逐项等于公共契约。"""

    contract_path = Path("configs/protocol/paper_profile_common_contract.json")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    for config_path in PAPER_PROFILE_PROTOCOL_CONFIGS:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        assert config["paper_profile_common_contract_path"] == contract_path.as_posix()
        assert {
            key: config.get(key)
            for key in contract
        } == contract


@pytest.mark.constraint
def test_formal_workflow_contains_no_claim3_alternative_branch() -> None:
    """正式配置与工作流不得重新引入 Claim-3 替代分支。"""

    governed_paths = [
        Path("configs/protocol/paper_profile_common_contract.json"),
        Path("configs/protocol/probe_paper_generative_probe.json"),
        Path("configs/protocol/pilot_paper_generative_probe.json"),
        Path("configs/protocol/full_paper_generative_probe.json"),
        Path("configs/paper_workflow/generative_video_notebook_workflows.json"),
        Path("workflows/generative_video_paper.py"),
    ]
    forbidden_token = "claim3_" + "downgrade"
    assert all(forbidden_token not in path.read_text(encoding="utf-8") for path in governed_paths)


@pytest.mark.constraint
def test_profile_negative_video_counts_support_their_fpr_scale() -> None:
    """低 FPR 样本量必须按独立视频计算, 不能依靠 key trial 扩增。"""

    import math

    for config_path in PAPER_PROFILE_PROTOCOL_CONFIGS:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        target_fpr = float(config["target_fpr"])
        required_for_zero_false_positives = math.ceil(
            math.log(0.05) / math.log(1.0 - target_fpr)
        )
        assert int(config["minimum_calibration_unique_video_count"]) >= required_for_zero_false_positives
        assert int(config["minimum_test_unique_video_count"]) >= required_for_zero_false_positives
        assert int(config["minimum_heldout_test_negative_event_count"]) == int(
            config["minimum_test_unique_video_count"]
        )


@pytest.mark.constraint
def test_workflow_profile_counts_are_derived_from_protocol_scale() -> None:
    """工作流摘要数量必须与规范 protocol config 完全一致, 防止双配置漂移."""

    workflow = json.loads(
        Path("configs/paper_workflow/generative_video_notebook_workflows.json").read_text(encoding="utf-8")
    )
    profiles = workflow["workflow_profiles"]
    for config_path in PAPER_PROFILE_PROTOCOL_CONFIGS:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        profile_name = str(config["paper_result_level"])
        profile = profiles[profile_name]
        total = int(config["minimum_prompt_count"]) * int(config["minimum_seed_per_prompt"])
        assert int(config["minimum_unique_video_count"]) == total
        assert int(profile["method_sample_count"]) == total
        assert int(profile["baseline_sample_count"]) == total
        assert int(profile["minimum_clean_negative_count"]) == int(config["minimum_clean_negative_count"])
        assert float(profile["target_fpr"]) == float(config["target_fpr"])
        assert profile["enabled_for_claim"] is True
