"""约束 paper profile 协议配置中的 baseline 数量语义保持一致。"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from evaluation.attacks.video_runtime_attack_protocol import load_protocol_config_with_shared_attack_protocol
from evaluation.protocol.paper_profile_contract import (
    CANONICAL_COMMON_CONTRACT_ID,
    CANONICAL_COMMON_CONTRACT_PATH,
    CANONICAL_COMMON_CONTRACT_SHA256,
    COMMON_CONTRACT_DIGEST_FIELD,
    COMMON_CONTRACT_PATH_FIELD,
    PAPER_PROFILE_CANONICAL_PATHS,
    PAPER_PROFILE_INVARIANT_METADATA_FIELDS,
    PAPER_PROFILE_ONLY_FIELD_CATEGORIES,
    PAPER_PROFILE_ONLY_FIELDS,
    enforce_paper_profile_common_contract,
)


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
    "require_baseline_matched_video_quality_metrics",
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
    core_method = json.loads(
        Path("configs/methods/sstw_core_method.json").read_text(encoding="utf-8")
    )
    replay_likelihood = core_method["replay_likelihood"]
    assert contract[
        "minimum_replay_likelihood_calibration_clean_video_cluster_count"
    ] == replay_likelihood["minimum_calibration_clean_video_cluster_count"]
    assert contract["replay_likelihood_calibration_step_count"] == (
        replay_likelihood["calibration_replay_step_count"]
    )
    assert contract["paper_profile_common_contract_id"] == CANONICAL_COMMON_CONTRACT_ID
    assert sha256(contract_path.read_bytes()).hexdigest() == (
        CANONICAL_COMMON_CONTRACT_SHA256
    )
    for config_path in PAPER_PROFILE_PROTOCOL_CONFIGS:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        assert config["paper_profile_common_contract_path"] == contract_path.as_posix()
        assert config[COMMON_CONTRACT_DIGEST_FIELD] == CANONICAL_COMMON_CONTRACT_SHA256
        assert {
            key: config.get(key)
            for key in contract
        } == contract


@pytest.mark.constraint
def test_canonical_paper_profiles_load_with_matched_common_contract() -> None:
    """三份 canonical profile 必须能经正式加载入口完成摘要校验。"""

    for config_path in PAPER_PROFILE_PROTOCOL_CONFIGS:
        loaded = load_protocol_config_with_shared_attack_protocol(config_path)
        assert loaded["paper_profile_common_contract_status"] == "matched"
        assert loaded["paper_profile_common_contract_observed_sha256"] == (
            CANONICAL_COMMON_CONTRACT_SHA256
        )


@pytest.mark.constraint
def test_every_profile_only_field_has_an_explicit_non_mechanism_category() -> None:
    """公共契约之外的字段必须进入显式允许列表, 不能使用宽泛前缀绕过。"""

    contract = json.loads(
        Path("configs/protocol/paper_profile_common_contract.json").read_text(
            encoding="utf-8"
        )
    )
    categorized = set().union(*PAPER_PROFILE_ONLY_FIELD_CATEGORIES.values())
    assert categorized == set(PAPER_PROFILE_ONLY_FIELDS)
    for config_path in PAPER_PROFILE_PROTOCOL_CONFIGS:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        profile_only = (
            set(config) - set(contract) - set(PAPER_PROFILE_INVARIANT_METADATA_FIELDS)
        )
        assert profile_only <= set(PAPER_PROFILE_ONLY_FIELDS)


@pytest.mark.constraint
def test_unregistered_profile_evidence_key_is_rejected_fail_closed(
    tmp_path: Path,
) -> None:
    """新增机制或证据键未登记时必须在加载阶段失败。"""

    source = json.loads(PAPER_PROFILE_PROTOCOL_CONFIGS[0].read_text(encoding="utf-8"))
    source["require_unregistered_claim_evidence"] = True
    with pytest.raises(ValueError, match="未登记的层级差异字段"):
        enforce_paper_profile_common_contract(
            source,
            PAPER_PROFILE_CANONICAL_PATHS["probe_paper"],
        )


@pytest.mark.constraint
def test_profile_target_fpr_cannot_drift_within_the_allowed_scale_category(
    tmp_path: Path,
) -> None:
    """profile-only 字段虽允许变化, 但每个正式层级的目标 FPR 仍必须冻结。"""

    source = json.loads(PAPER_PROFILE_PROTOCOL_CONFIGS[0].read_text(encoding="utf-8"))
    source["target_fpr"] = 0.2
    with pytest.raises(ValueError, match="target_fpr 必须为"):
        enforce_paper_profile_common_contract(
            source,
            PAPER_PROFILE_CANONICAL_PATHS["probe_paper"],
        )


@pytest.mark.constraint
def test_noncanonical_formal_profile_is_rejected_by_default(tmp_path: Path) -> None:
    """正式 CLI 不得从临时路径加载自定义 probe/pilot/full 配置。"""

    source = json.loads(PAPER_PROFILE_PROTOCOL_CONFIGS[0].read_text(encoding="utf-8"))
    config_path = tmp_path / "probe_paper_custom.json"
    config_path.write_text(json.dumps(source, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="只接受仓库 canonical profile"):
        load_protocol_config_with_shared_attack_protocol(config_path)


@pytest.mark.constraint
def test_custom_common_contract_path_is_rejected(tmp_path: Path) -> None:
    """profile 不能把任意自定义 JSON 声明为可信公共机制契约。"""

    source = json.loads(PAPER_PROFILE_PROTOCOL_CONFIGS[0].read_text(encoding="utf-8"))
    custom_contract = tmp_path / "paper_profile_common_contract.json"
    custom_contract.write_text("{}\n", encoding="utf-8")
    source[COMMON_CONTRACT_PATH_FIELD] = str(custom_contract)

    with pytest.raises(ValueError, match="只能引用仓库 canonical 公共契约"):
        enforce_paper_profile_common_contract(
            source,
            PAPER_PROFILE_CANONICAL_PATHS["probe_paper"],
        )


@pytest.mark.constraint
def test_common_contract_declared_digest_must_match_frozen_content() -> None:
    """profile 声明摘要必须与代码冻结的 canonical 内容摘要一致。"""

    source = json.loads(PAPER_PROFILE_PROTOCOL_CONFIGS[0].read_text(encoding="utf-8"))
    source[COMMON_CONTRACT_DIGEST_FIELD] = "0" * 64

    with pytest.raises(ValueError, match=COMMON_CONTRACT_DIGEST_FIELD):
        enforce_paper_profile_common_contract(
            source,
            PAPER_PROFILE_CANONICAL_PATHS["probe_paper"],
        )

    assert CANONICAL_COMMON_CONTRACT_PATH == Path(
        "configs/protocol/paper_profile_common_contract.json"
    ).resolve()


@pytest.mark.constraint
def test_three_formal_profiles_use_one_parameterized_gate_entrypoint() -> None:
    """三档 profile 必须运行同一 gate, profile 专属旧 gate 不得进入 stage plan。"""

    workflow = json.loads(
        Path("configs/paper_workflow/generative_video_notebook_workflows.json").read_text(
            encoding="utf-8"
        )
    )
    stage_plan = workflow["notebook_roles"]["paper_gate_and_package"]["stage_plan"]
    assert stage_plan.count("paper_profile_gate") == 1
    assert "pilot_paper_gate" not in stage_plan
    assert "full_paper_result_checker" not in stage_plan
    for profile_name in ("probe_paper", "pilot_paper", "full_paper"):
        profile = workflow["workflow_profiles"][profile_name]
        assert profile["profile_gate_entrypoint"] == (
            "experiments.generative_video_model_probe.paper_profile_gate"
        )
        assert profile["profile_gate_contract"] == (
            "shared_parameterized_gate_target_fpr_and_sample_scale_only"
        )
        assert "paper_profile_gate" not in profile["disabled_stage_names"]

    from workflows.generative_video_paper import build_workflow_stage_plan

    expected_transition = {
        "probe_paper": "probe_paper_to_pilot_paper_transition_decision",
        "pilot_paper": "pilot_paper_to_full_paper_transition_decision",
        "full_paper": "full_paper_to_submission_freeze_transition_decision",
    }
    for profile_name, transition_name in expected_transition.items():
        resolved_plan = build_workflow_stage_plan(
            profile_name,
            "paper_gate_and_package",
        )
        assert "paper_profile_gate" in resolved_plan
        assert transition_name in resolved_plan
        assert sum(name.endswith("transition_decision") for name in resolved_plan) == 1


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
def test_heldout_posterior_requirements_are_profile_invariant_and_scale_derived() -> None:
    """held-out 后验阈值必须同构, 簇数只能由各档统计规模派生。"""

    invariant_expected = {
        "maximum_heldout_posterior_brier_score": 0.25,
        "maximum_heldout_posterior_log_loss": 0.693147,
        "maximum_heldout_posterior_expected_calibration_error": 0.1,
        "heldout_posterior_bootstrap_resample_count": 5000,
        "minimum_sstw_worst_attack_tpr_ci_lower": 0.5,
        "require_heldout_posterior_probability_evaluation": True,
    }
    for config_path in PAPER_PROFILE_PROTOCOL_CONFIGS:
        config = load_protocol_config_with_shared_attack_protocol(config_path)
        for field_name, expected in invariant_expected.items():
            assert config[field_name] == expected
        heldout_count = int(config["minimum_test_unique_video_count"])
        assert int(config["minimum_heldout_posterior_positive_cluster_count"]) == heldout_count
        assert int(config["minimum_heldout_posterior_negative_cluster_count"]) == heldout_count
        attack_count = int(config["minimum_attack_event_count_per_attack"])
        assert int(config["minimum_heldout_posterior_attack_cluster_count"]) == attack_count
        assert attack_count <= heldout_count


@pytest.mark.constraint
def test_adaptive_query_mechanism_is_invariant_and_only_cluster_count_scales() -> None:
    """三个层级必须共享真实查询机制, 只能扩大 independent video 数量。"""

    expected_cluster_counts = {
        "probe_paper": 30,
        "pilot_paper": 100,
        "full_paper": 200,
    }
    for config_path in PAPER_PROFILE_PROTOCOL_CONFIGS:
        config = json.loads(config_path.read_text(encoding="utf-8-sig"))
        level = str(config["paper_result_level"])
        assert config["adaptive_attack_query_budget_per_video"] == 9
        assert config["adaptive_attack_query_budget_checkpoints"] == [1, 3, 5, 9]
        assert config[
            "adaptive_attack_public_negative_probe_query_budget"
        ] == 3
        assert config["adaptive_attack_source_cluster_selection_protocol"] == (
            "stable_sha256_rank_over_heldout_source_cluster_before_attack_scoring_v1"
        )
        assert config[
            "minimum_adaptive_attack_source_video_cluster_count_per_protocol"
        ] == expected_cluster_counts[level]
        assert config[
            "minimum_adaptive_attack_source_video_cluster_count_per_protocol"
        ] == config["minimum_attack_event_count_per_attack"]
        spoof_count = int(
            config[
                "minimum_independent_negative_video_count_for_fpr_upper_bound"
            ]
        )
        zero_false_accept_upper = 1.0 - (0.05 ** (1.0 / spoof_count))
        assert zero_false_accept_upper <= float(config["target_fpr"])


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


@pytest.mark.constraint
def test_stage_transition_occurs_after_complete_profile_package() -> None:
    """三个正式 profile 都必须先完成审稿索引、诊断图和 package 再跳转。"""

    workflow = json.loads(
        Path("configs/paper_workflow/generative_video_notebook_workflows.json").read_text(
            encoding="utf-8"
        )
    )
    plan = workflow["notebook_roles"]["paper_gate_and_package"]["stage_plan"]
    reviewer_index = plan.index("reviewer_evidence_index")
    figure_index = plan.index("paper_profile_gate_figure_builder")
    manifest_index = plan.index("paper_profile_package_manifest_builder")
    assert reviewer_index < figure_index < manifest_index
    for transition_name in (
        "probe_paper_to_pilot_paper_transition_decision",
        "pilot_paper_to_full_paper_transition_decision",
        "full_paper_to_submission_freeze_transition_decision",
    ):
        assert manifest_index < plan.index(transition_name)
    for profile_name in ("probe_paper", "pilot_paper", "full_paper"):
        disabled = set(
            workflow["workflow_profiles"][profile_name]["disabled_stage_names"]
        )
        assert "reviewer_evidence_index" not in disabled
        assert "paper_profile_gate_figure_builder" not in disabled
        assert "paper_profile_package_manifest_builder" not in disabled


@pytest.mark.constraint
def test_internal_ablation_video_reuse_policy_is_profile_invariant() -> None:
    """detector-only 消融必须复用 full 视频, 只有生成机制消融可独立生成。"""

    for config_path in PAPER_PROFILE_PROTOCOL_CONFIGS:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        profile = str(config["paper_result_level"])
        required = set(config["required_internal_ablation_variants"])
        generation = set(config["generation_internal_ablation_variants"])
        detector_only = set(config["detector_only_internal_ablation_variants"])
        assert generation & detector_only == set()
        assert generation | detector_only == required
        assert config["internal_ablation_video_reuse_policy"] == (
            "detector_only_reuses_full_method_video_generation_variants_use_independent_videos"
        )
        assert config["require_internal_ablation_video_reuse_policy"] is True
        assert int(config["minimum_internal_ablation_trace_count"]) == int(
            config["minimum_attack_event_count_per_attack"]
        )
        assert int(config["minimum_internal_ablation_trace_count"]) <= int(
            config["minimum_test_unique_video_count"]
        )
