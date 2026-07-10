"""验证 generative_video_model_probe Colab Notebook 入口、Drive 落盘与 prompt suite 构造。"""

from __future__ import annotations

import json
from pathlib import Path
import re
import shutil
import zipfile

import pytest

from paper_workflow.notebook_utils.generative_video_model_probe_workflow import (
    build_drive_layout,
    build_workflow_stage_plan,
    build_modern_baseline_command_env,
    build_external_baseline_official_bundle_generation_command,
    build_external_baseline_official_resource_bootstrap_command,
    build_external_baseline_official_result_bundle_preflight_command,
    build_external_baseline_self_containment_decision_command,
    build_formal_baseline_difference_interval_command,
    build_formal_method_baseline_comparison_command,
    build_low_fpr_formal_statistics_command,
    build_motion_consistency_exclusion_report_command,
    build_data_split_and_leakage_guard_command,
    build_modern_baseline_official_bridge_command_templates,
    build_modern_baseline_official_bridge_preflight_decision,
    build_repository_official_baseline_eval_command_templates,
    build_sstw_measured_formal_result_command,
    build_statistical_confidence_interval_command,
    build_formal_internal_ablation_summary_command,
    build_pilot_paper_gate_command,
    build_paper_profile_gate_command,
    build_probe_paper_to_pilot_paper_transition_decision_command,
    default_workflow_profile_for_notebook_role,
    ensure_drive_layout,
    resolve_notebook_workflow_profile,
    validate_modern_baseline_commands_for_profile,
    validate_modern_baseline_official_bridge_for_profile,
    validate_motion_threshold_ready_for_profile,
    write_modern_baseline_colab_command_config_summary,
    write_external_baseline_colab_preflight_decision,
    write_motion_threshold_reuse_artifact_for_profile,
)
from paper_workflow.colab_utils.stage_package_sync import (
    _default_required_stage_packages,
    activate_local_stage_layout,
    hydrate_stage_package,
    prepare_colab_stage_layout,
    publish_colab_stage_package,
    stage_package_id_for_notebook,
)
from experiments.generative_video_model_probe.colab_runtime import (
    _build_generation_plan,
    _formalize_paper_trajectory_record,
)
from scripts.package_results.generative_video_drive_packager import package_generative_video_colab_run
from scripts.prepare_generative_video_prompt_suite import write_prompt_suite
from main.attacks.video_runtime_attack_protocol import load_protocol_config_with_shared_attack_protocol
from main.protocol.record_writer import write_json, write_jsonl


@pytest.mark.quick
def test_prepare_generative_video_prompt_suite_is_separate_from_runtime(tmp_path: Path) -> None:
    """prompt suite 构造必须独立于 GPU 模型测试运行。"""
    output_root = tmp_path / "prompt_suite"
    summary = write_prompt_suite(output_root)

    suite_path = Path(summary["prompt_suite_path"])
    manifest_path = Path(summary["manifest_path"])
    assert suite_path.exists()
    assert manifest_path.exists()

    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    assert suite["dataset_construction_status"] == "constructed"
    assert suite["dataset_source"] == "repository_deterministic_prompt_seed_spec"
    assert len(suite["prompts"]) >= 8
    assert len(suite["seeds"]) >= 2


@pytest.mark.quick
def test_motion_calibration_prompt_suite_has_target_design(tmp_path: Path) -> None:
    """motion calibration prompt suite 必须展开为 128 / 64 / 32 的目标规模。"""
    output_root = tmp_path / "prompt_suite"
    summary = write_prompt_suite(output_root)
    suite = json.loads(Path(summary["prompt_suite_path"]).read_text(encoding="utf-8"))
    prompts = suite["prompts"]
    seeds = suite["seeds"]

    negative_prompts = [item for item in prompts if item.get("motion_calibration_role") == "negative_static"]
    positive_prompts = [item for item in prompts if item.get("motion_calibration_role") == "positive_motion"]
    ambiguous_prompts = [item for item in prompts if item.get("motion_calibration_role") == "ambiguous_low_motion"]
    calibration_seeds = [item for item in seeds if item.get("prompt_suite_role") == "motion_calibration"]
    plan = _build_generation_plan(suite, "motion_calibration", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", None)

    assert len(negative_prompts) == 16
    assert len(positive_prompts) == 8
    assert len(ambiguous_prompts) == 4
    assert len(calibration_seeds) == 8
    assert len(plan) == 224
    assert sum(1 for item in plan if item["motion_calibration_role"] == "negative_static") == 128
    assert sum(1 for item in plan if item["motion_calibration_role"] == "positive_motion") == 64
    assert sum(1 for item in plan if item["motion_calibration_role"] == "ambiguous_low_motion") == 32
    assert {item["split"] for item in plan} == {"calibration"}
    assert {item["seed_suite_role"] for item in plan} == {"motion_calibration"}
    assert {item["prompt_suite_role"] for item in plan} == {
        "motion_calibration_negative_static",
        "motion_calibration_positive_motion",
        "motion_calibration_ambiguous_low_motion",
    }


@pytest.mark.quick
def test_motion_calibration_prompt_suite_uses_observability_repair_prompts(tmp_path: Path) -> None:
    """修复后的 calibration prompt 不应继续使用已知弱 positive prompt 或高污染 static prompt。"""
    output_root = tmp_path / "prompt_suite"
    summary = write_prompt_suite(output_root)
    suite = json.loads(Path(summary["prompt_suite_path"]).read_text(encoding="utf-8"))
    prompts_by_id = {item["prompt_id"]: item["prompt_text"].lower() for item in suite["prompts"]}

    assert suite["prompt_suite_id"] == "generative_video_probe_prompt_suite_motion_observability_pilot_paper"

    # 这两个历史 prompt 在真实 Wan2.1 calibration 中分别只有 3 / 8 和 1 / 8 通过, 因此不能回退。
    assert "large red square slides" not in prompts_by_id["motion_calib_positive_motion_00"]
    assert "bright blue circle bounces" not in prompts_by_id["motion_calib_positive_motion_02"]
    assert "person carries" in prompts_by_id["motion_calib_positive_motion_00"]
    assert "beach ball" in prompts_by_id["motion_calib_positive_motion_02"]

    # 这些静态物体或纹理在真实结果中容易触发模型自发运动或纹理闪烁, 不能再作为 clean static 设计。
    for prompt_id in (
        "motion_calib_negative_static_04",
        "motion_calib_negative_static_14",
        "motion_calib_negative_static_15",
    ):
        prompt_text = prompts_by_id[prompt_id]
        assert "checkerboard" not in prompt_text
        assert "chess" not in prompt_text
        assert "clock" not in prompt_text


@pytest.mark.quick
def test_pilot_prompt_suite_replaces_low_motion_heldout_rotation_prompt(tmp_path: Path) -> None:
    """pilot heldout prompt 必须避免回退到真实 Wan2.1 中已失败的低运动旋转设计。"""
    output_root = tmp_path / "prompt_suite"
    summary = write_prompt_suite(output_root)
    suite = json.loads(Path(summary["prompt_suite_path"]).read_text(encoding="utf-8"))
    prompts_by_id = {item["prompt_id"]: item for item in suite["prompts"]}
    heldout = prompts_by_id["heldout_rotation_scene"]
    prompt_text = heldout["prompt_text"].lower()
    plan = _build_generation_plan(suite, "pilot", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", None)
    heldout_plan_items = [item for item in plan if item["prompt_id"] == "heldout_rotation_scene"]

    assert "rotates gently" not in prompt_text
    assert "slides from the far left edge to the far right edge" in prompt_text
    assert "spinning rapidly" in prompt_text
    assert "strong visible displacement" in prompt_text
    assert heldout["motion_pattern_id"] == "large_rotation_translation"
    assert heldout["motion_claim_role"] == "positive_motion"
    assert len(heldout_plan_items) == 2
    assert {item["seed_id"] for item in heldout_plan_items} == {"seed_main_a", "seed_main_b"}
    assert {item["motion_claim_role"] for item in heldout_plan_items} == {"positive_motion"}


@pytest.mark.quick
def test_removed_pre_probe_profile_is_not_in_runtime_plan(tmp_path: Path) -> None:
    """已移除的 pre-probe profile 不再属于主干 runtime profile。"""
    output_root = tmp_path / "prompt_suite"
    summary = write_prompt_suite(output_root)
    suite = json.loads(Path(summary["prompt_suite_path"]).read_text(encoding="utf-8"))

    removed_profile = "validation" + "_scale"
    with pytest.raises(KeyError):
        _build_generation_plan(suite, removed_profile, "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", None)


@pytest.mark.quick
def test_paper_profile_trajectory_records_do_not_emit_proxy_fields() -> None:
    """paper profile 轨迹记录必须使用正式 callback latent displacement 字段。"""

    record = _formalize_paper_trajectory_record(
        {
            "trajectory_trace_id": "trace_a",
            "flow_velocity_proxy_available": True,
            "flow_velocity_proxy_source": "adjacent_callback_latent_displacement",
            "flow_velocity_alignment_gain": 0.12,
        },
        "probe_paper",
    )

    assert "flow_velocity_proxy_available" not in record
    assert "flow_velocity_alignment_gain" not in record
    assert record["callback_latent_displacement_available"] is True
    assert record["callback_latent_displacement_source"] == "adjacent_callback_latent_displacement"
    assert record["callback_latent_displacement_alignment_gain"] == 0.12
    assert record["callback_latent_displacement_evidence_level"] == "adjacent_callback_latent_state_displacement"


@pytest.mark.quick
def test_pilot_paper_profile_constructs_medium_scale_low_fpr_plan(tmp_path: Path) -> None:
    """pilot_paper profile 必须构造 calibration/test split 低 FPR 数据集。"""
    output_root = tmp_path / "prompt_suite"
    summary = write_prompt_suite(output_root)
    suite = json.loads(Path(summary["prompt_suite_path"]).read_text(encoding="utf-8"))
    plan = _build_generation_plan(suite, "pilot_paper", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", None)
    pilot_paper_plan = _build_generation_plan(suite, "pilot_paper", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", None)

    pilot_paper_prompts = [item for item in suite["prompts"] if item.get("prompt_suite_role") == "pilot_paper"]
    pilot_paper_seeds = [item for item in suite["seeds"] if item.get("prompt_suite_role") == "pilot_paper"]

    pilot_protocol = load_protocol_config_with_shared_attack_protocol("configs/protocol/pilot_paper_generative_probe.json")
    assert suite["pilot_paper_design"]["target_fpr"] == pilot_protocol["target_fpr"]
    assert suite["pilot_paper_design"]["blocked_target_fpr"] == pilot_protocol["blocked_target_fpr"]
    assert suite["pilot_paper_design"]["paper_result_level"] == "pilot_paper"
    assert suite["pilot_paper_design"]["paper_protocol_difference_from_full_paper"] == "sample_scale_and_target_fpr_only"
    assert suite["pilot_paper_design"]["recommended_runtime_profile"] == "pilot_paper"
    assert suite["pilot_paper_design"]["threshold_protocol"] == "calibration_split_to_frozen_threshold_to_heldout_test_split"
    assert suite["pilot_paper_design"]["target_generation_video_count"] == 100
    assert suite["pilot_paper_design"]["target_calibration_unique_video_count"] == 50
    assert suite["pilot_paper_design"]["target_test_unique_video_count"] == 50
    expected_attack_count = len(pilot_protocol["required_runtime_attack_names"])
    expected_test_positive_count = 50 * expected_attack_count
    assert suite["pilot_paper_design"]["target_runtime_attack_count"] == expected_attack_count
    assert suite["pilot_paper_design"]["target_runtime_attack_names"] == pilot_protocol["required_runtime_attack_names"]
    assert suite["pilot_paper_design"]["target_calibration_negative_event_count"] == 5000
    assert suite["pilot_paper_design"]["target_heldout_test_negative_event_count"] == 5000
    assert suite["pilot_paper_design"]["target_test_attacked_positive_event_count"] == expected_test_positive_count
    assert len(pilot_paper_prompts) == 25
    assert len(pilot_paper_seeds) == 4
    assert len(plan) == 200
    assert len(pilot_paper_plan) == 200
    assert [(item["prompt_id"], item["seed_id"]) for item in pilot_paper_plan] == [(item["prompt_id"], item["seed_id"]) for item in plan]
    positive_plan = [item for item in plan if item["sample_role"] == "attacked_positive_source"]
    clean_plan = [item for item in plan if item["sample_role"] == "clean_negative"]
    assert len(positive_plan) == 100
    assert len(clean_plan) == 100
    assert len({item["prompt_id"] for item in plan}) == 25
    assert {item["seed_suite_role"] for item in plan} == {"pilot_paper"}
    assert {item["prompt_suite_role"] for item in plan} == {"pilot_paper"}
    assert {item["split"] for item in plan} == {"calibration", "test"}
    assert sum(1 for item in positive_plan if item["split"] == "calibration") == 50
    assert sum(1 for item in positive_plan if item["split"] == "test") == 50
    assert all(item["watermark_embedding_status"] == "clean_unwatermarked_reference" for item in clean_plan)


@pytest.mark.quick
def test_probe_paper_profile_constructs_ten_unit_fixed_fpr_plan(tmp_path: Path) -> None:
    """probe_paper profile 必须构造 10 生成单元与 500 clean negative event 设计。"""
    output_root = tmp_path / "prompt_suite"
    summary = write_prompt_suite(output_root)
    suite = json.loads(Path(summary["prompt_suite_path"]).read_text(encoding="utf-8"))
    plan = _build_generation_plan(suite, "probe_paper", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", None)

    probe_paper_prompts = [item for item in suite["prompts"] if item.get("prompt_suite_role") == "probe_paper"]
    probe_paper_seeds = [item for item in suite["seeds"] if item.get("prompt_suite_role") == "probe_paper"]
    probe_protocol = load_protocol_config_with_shared_attack_protocol("configs/protocol/probe_paper_generative_probe.json")
    expected_attack_count = len(probe_protocol["required_runtime_attack_names"])

    assert suite["probe_paper_design"]["target_fpr"] == 0.1
    assert suite["probe_paper_design"]["paper_result_level"] == "probe_paper"
    assert suite["probe_paper_design"]["target_generation_video_count"] == 10
    assert suite["probe_paper_design"]["target_calibration_unique_video_count"] == 5
    assert suite["probe_paper_design"]["target_test_unique_video_count"] == 5
    assert suite["probe_paper_design"]["target_test_attacked_positive_event_count"] == 5 * expected_attack_count
    assert suite["probe_paper_design"]["target_calibration_negative_event_count"] == 500
    assert suite["probe_paper_design"]["target_heldout_test_negative_event_count"] == 500
    assert len(probe_paper_prompts) == 5
    assert len(probe_paper_seeds) == 2
    positive_plan = [item for item in plan if item["sample_role"] == "attacked_positive_source"]
    clean_plan = [item for item in plan if item["sample_role"] == "clean_negative"]
    assert len(plan) == 20
    assert len(positive_plan) == 10
    assert len(clean_plan) == 10
    assert len({item["prompt_id"] for item in plan}) == 5
    assert len({item["seed_id"] for item in plan}) == 2
    assert {item["seed_suite_role"] for item in plan} == {"probe_paper"}
    assert {item["prompt_suite_role"] for item in plan} == {"probe_paper"}
    assert {item["split"] for item in plan} == {"calibration", "test"}
    assert sum(1 for item in positive_plan if item["split"] == "calibration") == 5
    assert sum(1 for item in positive_plan if item["split"] == "test") == 5
    assert all(item["watermark_embedding_status"] == "clean_unwatermarked_reference" for item in clean_plan)


@pytest.mark.quick
def test_full_paper_profile_constructs_full_scale_split_plan(tmp_path: Path) -> None:
    """full_paper profile 必须一次展开完整 125 prompt × 8 seed 计划。

    该测试属于工程门禁约束, 作用是防止 full_paper 在昂贵 GPU 运行前仍缺少
    prompt / seed / split 配置。Notebook 只切换 profile, 具体样本计划必须由仓库代码生成。
    """
    output_root = tmp_path / "prompt_suite"
    summary = write_prompt_suite(output_root)
    suite = json.loads(Path(summary["prompt_suite_path"]).read_text(encoding="utf-8"))
    plan = _build_generation_plan(suite, "full_paper", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", None)

    full_paper_prompts = [item for item in suite["prompts"] if item.get("prompt_suite_role") == "full_paper"]
    full_paper_seeds = [item for item in suite["seeds"] if item.get("prompt_suite_role") == "full_paper"]
    full_protocol = load_protocol_config_with_shared_attack_protocol("configs/protocol/full_paper_generative_probe.json")
    expected_attack_count = len(full_protocol["required_runtime_attack_names"])

    assert suite["full_paper_design"]["paper_result_level"] == "full_paper"
    assert suite["full_paper_design"]["recommended_runtime_profile"] == "full_paper"
    assert suite["full_paper_design"]["threshold_protocol"] == "calibration_split_to_frozen_threshold_to_heldout_test_split"
    assert suite["full_paper_design"]["target_fpr"] == full_protocol["target_fpr"]
    assert suite["full_paper_design"]["target_generation_video_count"] == 1000
    assert suite["full_paper_design"]["target_calibration_unique_video_count"] == 500
    assert suite["full_paper_design"]["target_test_unique_video_count"] == 500
    assert suite["full_paper_design"]["target_runtime_attack_count"] == expected_attack_count
    assert suite["full_paper_design"]["target_runtime_attack_names"] == full_protocol["required_runtime_attack_names"]
    assert suite["full_paper_design"]["target_test_attacked_positive_event_count"] == 1000 * expected_attack_count
    assert suite["full_paper_design"]["target_calibration_negative_event_count"] == 50000
    assert suite["full_paper_design"]["target_heldout_test_negative_event_count"] == 50000
    assert len(full_paper_prompts) == 125
    assert len(full_paper_seeds) == 8
    positive_plan = [item for item in plan if item["sample_role"] == "attacked_positive_source"]
    clean_plan = [item for item in plan if item["sample_role"] == "clean_negative"]
    assert len(plan) == 2000
    assert len(positive_plan) == 1000
    assert len(clean_plan) == 1000
    assert len({item["prompt_id"] for item in plan}) == 125
    assert len({item["seed_id"] for item in plan}) == 8
    assert {item["seed_suite_role"] for item in plan} == {"full_paper"}
    assert {item["prompt_suite_role"] for item in plan} == {"full_paper"}
    assert {item["split"] for item in plan} == {"calibration", "test"}
    assert sum(1 for item in positive_plan if item["split"] == "calibration") == 500
    assert sum(1 for item in positive_plan if item["split"] == "test") == 500
    assert all(item["watermark_embedding_status"] == "clean_unwatermarked_reference" for item in clean_plan)


@pytest.mark.quick
def test_generative_video_colab_notebook_calls_repository_modules() -> None:
    """generation Notebook 只能作为入口, 必须通过统一 workflow profile 调用仓库模块。"""
    notebook_path = Path("paper_workflow/colab_notebooks/generative_video_generation_colab.ipynb")
    assert notebook_path.exists()
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    helper_text = Path("paper_workflow/notebook_utils/generative_video_model_probe_workflow.py").read_text(encoding="utf-8")

    assert "/content/drive/MyDrive/SSTW" in source
    assert "drive.mount('/content/drive')" in source
    assert "SSTW_WORKFLOW_PROFILE" in source
    assert "NOTEBOOK_ROLE = 'generative_video_generation'" in source
    assert "default_workflow_profile_for_notebook_role" in source
    assert "resolve_notebook_workflow_profile" in source
    assert "ensure_drive_layout(" in source
    assert "SSTW_COLAB_STAGE_IO_MODE" in source
    assert "prepare_colab_stage_layout" in source
    assert "publish_colab_stage_package" not in source
    assert "publish_colab_stage_package" in helper_text
    assert "active_local_layout" in source
    assert "workflow_profile=WORKFLOW_PROFILE" in source
    assert "run_configured_colab_stage_plan" in source
    assert "stage_enabled(" not in source
    assert "workflow_stage_enabled" not in source
    assert "HF_TOKEN" in source
    assert "add_to_git_credential=False" in source
    assert "generative_video_model_probe_workflow" in source
    assert "MODEL_ID = os.environ.get('SSTW_MODEL_ID'" in source
    assert "RUN_EXTERNAL_BASELINE_SOURCE_CLONE" not in source
    assert "EXTERNAL_BASELINE_EVIDENCE_PATHS" not in source
    assert "REQUIRE_MODERN_BASELINE_COMMANDS_FOR_PAPER_GATE" not in source
    assert "SSTW_EXTERNAL_BASELINE_EVIDENCE_PATHS" not in source
    assert "build_modern_baseline_command_env" not in source
    assert "write_modern_baseline_colab_command_config_summary" not in source
    assert "write_external_baseline_colab_preflight_decision" not in source
    assert "validate_modern_baseline_commands_for_profile" not in source
    assert "build_modern_baseline_command_env" in helper_text
    assert "write_modern_baseline_colab_command_config_summary" in helper_text
    assert "write_external_baseline_colab_preflight_decision" in helper_text
    assert "validate_modern_baseline_commands_for_profile" in helper_text
    assert "external_baseline_colab_preflight_decision" in helper_text
    assert "external_baseline_command_template_summary" in helper_text
    assert "write_motion_threshold_reuse_artifact_for_profile" not in source
    assert "build_formal_metric_command" not in source
    assert "build_motion_threshold_calibration_command" not in source
    assert "build_mechanism_postprocess_command" not in source
    assert "build_protocol_evaluation_matrix_postprocess_command" not in source
    assert "build_runtime_attack_command" not in source
    assert "build_runtime_detection_command" not in source
    assert "build_small_scale_claim_pilot_gate_command" not in source
    assert "build_external_baseline_source_intake_command" not in source
    assert "build_external_baseline_comparison_command" not in source
    assert "build_validation_internal_ablation_command" not in source
    assert "build_adaptive_attack_formal_command" not in source
    assert "build_replay_and_sketch_gate_command" not in source
    assert "build_claim3_downgrade_command" not in source
    assert "build_statistical_confidence_interval_command" not in source
    assert "build_pilot_paper_gate_command" not in source
    assert "build_validation_artifact_rebuild_dry_run_command" not in source
    assert "build_paper_profile_gate_command" not in source
    assert "scripts/prepare_generative_video_prompt_suite.py" in helper_text
    assert "experiments.generative_video_model_probe.colab_runtime" in helper_text
    assert "experiments.generative_video_model_probe.formal_metric_runner" in helper_text
    assert "experiments.generative_video_model_probe.motion_threshold_calibration" in helper_text
    assert "experiments.generative_video_model_probe.motion_consistency_exclusion_report" in helper_text
    assert "experiments.generative_video_model_probe.attack_runner" in helper_text
    assert "experiments.generative_video_model_probe.detection_runner" in helper_text
    assert "scripts/build_external_baseline_source_intake.py" in helper_text
    assert "--execute-clone" in helper_text
    assert "experiments.generative_video_model_probe.external_baseline_runner" in helper_text
    removed_pilot_gate_module = ".".join(["experiments", "generative_video_model_probe", "pilot_claim_gate"])
    assert removed_pilot_gate_module not in helper_text
    assert "experiments.generative_video_model_probe.validation_internal_ablation" in helper_text
    assert "experiments.generative_video_model_probe.adaptive_attack_runner" in helper_text
    assert "experiments.generative_video_model_probe.replay_and_sketch_gate" in helper_text
    assert "experiments.generative_video_model_probe.claim3_downgrade" in helper_text
    assert "experiments.generative_video_model_probe.statistical_confidence_interval" in helper_text
    assert "experiments.generative_video_model_probe.low_fpr_formal_statistics" in helper_text
    assert "experiments.generative_video_model_probe.sstw_formal_result" in helper_text
    assert "experiments.generative_video_model_probe.formal_method_baseline_comparison" in helper_text
    assert "experiments.generative_video_model_probe.formal_baseline_difference_interval" in helper_text
    assert "experiments.generative_video_model_probe.formal_internal_ablation_summary" in helper_text
    assert "experiments.generative_video_model_probe.pilot_paper_gate" in helper_text
    assert "experiments.generative_video_model_probe.validation_artifact_rebuild" in helper_text
    assert "experiments.generative_video_model_probe.paper_profile_gate" in helper_text
    assert "scripts/package_results/generative_video_drive_packager.py" in helper_text
    assert "pytest -q" not in source
    assert "tools/harness/run_all_audits.py" not in source
    assert "pytest" in helper_text
    assert "tools/harness/run_all_audits.py" in helper_text
    assert "pilot_paper_results" not in source


@pytest.mark.quick
def test_split_colab_notebooks_are_profile_driven() -> None:
    """拆分后的 Colab Notebook 必须通过统一配置切换 profile, 不能维护独立路径硬编码。"""
    expected_roles = {
        "motion_threshold_calibration_colab.ipynb": "motion_threshold_calibration",
        "generative_video_generation_colab.ipynb": "generative_video_generation",
        "generative_video_quality_scoring_colab.ipynb": "generative_video_quality_scoring",
        "runtime_attack_colab.ipynb": "runtime_attack",
        "runtime_detection_colab.ipynb": "runtime_detection",
        "formal_comparison_scoring_colab.ipynb": "formal_comparison_scoring",
        "paper_evidence_postprocess_colab.ipynb": "paper_evidence_postprocess",
        "paper_gate_and_package_colab.ipynb": "paper_gate_and_package",
    }
    for notebook_name, role in expected_roles.items():
        notebook_path = Path("paper_workflow/colab_notebooks") / notebook_name
        assert notebook_path.exists()
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        source = "".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        first_code_cell = next(cell for cell in notebook["cells"] if cell.get("cell_type") == "code")
        first_code_source = "".join(first_code_cell.get("source", []))
        switch_source = "".join(notebook["cells"][2].get("source", []))
        if role == "motion_threshold_calibration":
            assert first_code_source.startswith("# 1. 挂载 Google Drive 并检查 GPU")
            assert "SSTW_WORKFLOW_PROFILE_VALUE" not in source
            assert "os.environ.pop('SSTW_WORKFLOW_PROFILE', None)" in switch_source
            assert "motion_threshold_calibration 固定使用 motion_calibration" in switch_source
            assert "WORKFLOW_PROFILE = probe_workflow.default_workflow_profile_for_notebook_role(NOTEBOOK_ROLE)" in source
        else:
            assert first_code_source.startswith("SSTW_WORKFLOW_PROFILE_VALUE = 'probe_paper'")
            assert (
                "SSTW_WORKFLOW_PROFILE_VALUE = globals().get('SSTW_WORKFLOW_PROFILE_VALUE', 'probe_paper')"
                in switch_source
            )
            assert "os.environ['SSTW_WORKFLOW_PROFILE']" in switch_source
            assert source.index("SSTW_WORKFLOW_PROFILE_VALUE") < source.index("resolve_notebook_workflow_profile")
        obsolete_numbered_profile_header = "# 1.1 " + "可编辑 workflow " + "profile 切换"
        obsolete_first_cell_hint = "修改第一个代码 cell 第一行的 " + "SSTW_WORKFLOW_PROFILE_VALUE"
        obsolete_generic_profile_header = "可编辑 workflow " + "profile 切换"
        assert obsolete_numbered_profile_header not in switch_source
        assert obsolete_first_cell_hint not in switch_source
        assert obsolete_generic_profile_header not in switch_source
        assert f"NOTEBOOK_ROLE = '{role}'" in source
        assert "SSTW_WORKFLOW_PROFILE" in source
        assert "resolve_notebook_workflow_profile" in source
        assert "workflow_profile=WORKFLOW_PROFILE" in source
        assert "SSTW_COLAB_STAGE_IO_MODE" in source
        assert "prepare_colab_stage_layout" in source
        assert "publish_colab_stage_package" not in source
        assert "run_configured_colab_stage_plan" in source
        assert "layout['drive_package_dir']" not in source
        assert "package_dir = Path(layout['drive_package_dir'])" not in source
        assert "stage_package_dir = Path(layout['stage_package_dir'])" not in source
        assert "drive_stage_package_zip" not in source
        assert "stage_package_manifest_path" not in source
        assert "active_local_layout" in source
        assert "stage_enabled(" not in source
        assert "drive.mount('/content/drive')" in source
        assert "git clone" in source
        assert "tools/harness/run_all_audits.py" not in source
        assert "pilot_paper_results" not in source
        assert "full_paper_results" not in source

    runtime_source = Path("paper_workflow/colab_notebooks/generative_video_generation_colab.ipynb").read_text(encoding="utf-8")
    formal_source = Path("paper_workflow/colab_notebooks/formal_comparison_scoring_colab.ipynb").read_text(encoding="utf-8")
    evidence_source = Path("paper_workflow/colab_notebooks/paper_evidence_postprocess_colab.ipynb").read_text(encoding="utf-8")
    gate_source = Path("paper_workflow/colab_notebooks/paper_gate_and_package_colab.ipynb").read_text(encoding="utf-8")
    helper_text = Path("paper_workflow/notebook_utils/generative_video_model_probe_workflow.py").read_text(encoding="utf-8")
    assert "external_baseline_colab_preflight" not in runtime_source
    assert "external_baseline_colab_preflight" in helper_text
    assert "build_external_baseline_comparison_command" not in runtime_source
    assert "apply_formal_comparison_external_baseline_environment" in formal_source
    assert "apply_paper_gate_external_baseline_environment" not in gate_source
    assert "build_motion_consistency_exclusion_report_command" not in gate_source
    assert "build_external_baseline_official_result_bundle_preflight_command" not in gate_source
    assert "build_external_baseline_comparison_command" not in gate_source
    assert "build_external_baseline_self_containment_decision_command" not in gate_source
    assert "build_low_fpr_formal_statistics_command" not in gate_source
    assert "build_sstw_measured_formal_result_command" not in gate_source
    assert "build_formal_method_baseline_comparison_command" not in gate_source
    assert "build_formal_baseline_difference_interval_command" not in gate_source
    assert "build_formal_internal_ablation_summary_command" not in gate_source
    assert "build_pilot_paper_gate_command" not in gate_source
    assert "build_paper_profile_gate_command" not in gate_source
    assert "build_motion_consistency_exclusion_report_command" not in evidence_source
    assert "build_paper_profile_gate_command" not in evidence_source
    assert "build_motion_consistency_exclusion_report_command" in helper_text
    assert "build_external_baseline_official_result_bundle_preflight_command" in helper_text
    assert "build_external_baseline_comparison_command" in helper_text
    assert "build_external_baseline_self_containment_decision_command" in helper_text
    assert "build_low_fpr_formal_statistics_command" in helper_text
    assert "build_sstw_measured_formal_result_command" in helper_text
    assert "build_formal_method_baseline_comparison_command" in helper_text
    assert "build_formal_baseline_difference_interval_command" in helper_text
    assert "build_formal_internal_ablation_summary_command" in helper_text
    assert "build_pilot_paper_gate_command" in helper_text
    assert "build_paper_profile_gate_command" in helper_text
    assert not Path("paper_workflow/colab_notebooks/probe_paper_formal_gate_colab.ipynb").exists()
    assert not Path("paper_workflow/colab_notebooks/external_baseline_formal_scoring_colab.ipynb").exists()
    assert not list(Path("paper_workflow/colab_utils").glob("*.ipynb"))


@pytest.mark.quick
def test_colab_notebooks_are_separated_from_python_helpers() -> None:
    """Notebook 入口必须与 Python helper 分目录保存。"""

    notebook_dir = Path("paper_workflow/colab_notebooks")
    helper_dir = Path("paper_workflow/colab_utils")

    assert notebook_dir.exists()
    assert helper_dir.exists()
    assert list(notebook_dir.glob("*.ipynb"))
    assert not list(helper_dir.glob("*.ipynb"))
    assert not (notebook_dir / "probe_paper_formal_gate_colab.ipynb").exists()


@pytest.mark.quick
@pytest.mark.quick
@pytest.mark.quick
def test_notebook_workflow_profile_config_supports_profile_switching() -> None:
    """统一配置层必须能区分 probe_paper、pilot_paper 和 full_paper。"""
    assert default_workflow_profile_for_notebook_role("generative_video_generation") == "probe_paper"
    assert default_workflow_profile_for_notebook_role("generative_video_quality_scoring") == "probe_paper"
    assert default_workflow_profile_for_notebook_role("runtime_attack") == "probe_paper"
    assert default_workflow_profile_for_notebook_role("runtime_detection") == "probe_paper"
    assert default_workflow_profile_for_notebook_role("formal_comparison_scoring") == "probe_paper"
    assert default_workflow_profile_for_notebook_role("paper_evidence_postprocess") == "probe_paper"
    external_reference_role = resolve_notebook_workflow_profile("probe_paper", "external_baseline_formal_scoring")
    formal_scoring = resolve_notebook_workflow_profile("probe_paper", "formal_comparison_scoring")
    evidence_postprocess = resolve_notebook_workflow_profile("probe_paper", "paper_evidence_postprocess")
    probe = resolve_notebook_workflow_profile("probe_paper", "paper_gate_and_package")
    pilot = resolve_notebook_workflow_profile("pilot_paper", "paper_gate_and_package")
    full = resolve_notebook_workflow_profile("full_paper", config_path="configs/paper_workflow/generative_video_notebook_workflows.json", allow_disabled=True)
    probe_protocol = json.loads(Path(probe["protocol_config_path"]).read_text(encoding="utf-8"))
    pilot_protocol = json.loads(Path(pilot["protocol_config_path"]).read_text(encoding="utf-8"))
    full_protocol = json.loads(Path(full["protocol_config_path"]).read_text(encoding="utf-8"))

    assert "motion_threshold_reuse_check" in build_workflow_stage_plan("probe_paper", "paper_evidence_postprocess")
    assert "validation_internal_ablation" in build_workflow_stage_plan("probe_paper", "paper_evidence_postprocess")
    assert "formal_adaptive_attack_execution" in build_workflow_stage_plan("probe_paper", "paper_evidence_postprocess")
    assert "adaptive_attack_formal" in build_workflow_stage_plan("probe_paper", "paper_evidence_postprocess")
    assert "motion_threshold_reuse_check" not in build_workflow_stage_plan("probe_paper", "paper_gate_and_package")
    assert "validation_internal_ablation" not in build_workflow_stage_plan("probe_paper", "paper_gate_and_package")
    assert "adaptive_attack_formal" not in build_workflow_stage_plan("probe_paper", "paper_gate_and_package")
    assert "external_baseline_comparison" not in build_workflow_stage_plan("probe_paper", "paper_gate_and_package")
    assert "external_baseline_comparison" in build_workflow_stage_plan("probe_paper", "formal_comparison_scoring")
    assert "fair_detection_calibration" in build_workflow_stage_plan("probe_paper", "formal_comparison_scoring")
    assert "paper_profile_gate" in build_workflow_stage_plan("probe_paper", "paper_gate_and_package")
    assert "removed_pre_probe_transition_decision" not in build_workflow_stage_plan("probe_paper", "paper_gate_and_package")
    assert "probe_paper_to_pilot_paper_transition_decision" in build_workflow_stage_plan("probe_paper", "paper_gate_and_package")
    assert "pilot_paper_gate" not in build_workflow_stage_plan("probe_paper", "paper_gate_and_package")
    assert probe["notebook_path"] == "paper_workflow/colab_notebooks/paper_gate_and_package_colab.ipynb"
    assert evidence_postprocess["notebook_path"] == "paper_workflow/colab_notebooks/paper_evidence_postprocess_colab.ipynb"
    assert formal_scoring["notebook_path"] == "paper_workflow/colab_notebooks/formal_comparison_scoring_colab.ipynb"
    assert external_reference_role["notebook_path"] == ""
    assert external_reference_role["entrypoint_status"] == "no_standalone_notebook_per_baseline_formal_reference_only"
    with pytest.raises(KeyError):
        default_workflow_profile_for_notebook_role("probe_paper_formal_gate")
    with pytest.raises(KeyError):
        resolve_notebook_workflow_profile("probe_paper", "probe_paper_formal_gate")
    removed_profile = "validation" + "_scale"
    with pytest.raises(KeyError):
        resolve_notebook_workflow_profile(removed_profile, "paper_gate_and_package")

    assert probe["requested_workflow_profile"] == "probe_paper"
    assert probe["workflow_profile"] == "probe_paper"
    assert probe["result_tier"] == "probe_paper"
    assert probe["enabled_for_claim"] is True
    assert probe["method_sample_count"] == 10
    assert probe["baseline_sample_count"] == 10
    assert probe["protocol_config_path"] == "configs/protocol/probe_paper_generative_probe.json"
    assert probe["target_fpr"] == probe_protocol["target_fpr"]
    assert probe["protocol_target_fpr"] == probe_protocol["target_fpr"]
    assert "paper_profile_gate" in build_workflow_stage_plan("probe_paper", "paper_gate_and_package")
    assert "removed_pre_probe_transition_decision" not in build_workflow_stage_plan("probe_paper", "paper_gate_and_package")
    assert "probe_paper_to_pilot_paper_transition_decision" in build_workflow_stage_plan("probe_paper", "paper_gate_and_package")
    assert "pilot_paper_gate" not in build_workflow_stage_plan("probe_paper", "paper_gate_and_package")

    assert pilot["requested_workflow_profile"] == "pilot_paper"
    assert pilot["workflow_profile"] == "pilot_paper"
    assert pilot["profile_alias_applied"] is False
    assert pilot["result_tier"] == "pilot_paper"
    assert pilot["enabled_for_claim"] is True
    assert pilot["method_sample_count"] == 100
    assert pilot["baseline_sample_count"] == 100
    assert pilot["protocol_config_path"] == "configs/protocol/pilot_paper_generative_probe.json"
    assert pilot["target_fpr"] == pilot_protocol["target_fpr"]
    assert pilot["protocol_target_fpr"] == pilot_protocol["target_fpr"]
    assert "motion_threshold_reuse_check" in build_workflow_stage_plan("pilot_paper", "paper_evidence_postprocess")
    assert "validation_internal_ablation" in build_workflow_stage_plan("pilot_paper", "paper_evidence_postprocess")
    assert "formal_adaptive_attack_execution" in build_workflow_stage_plan("pilot_paper", "paper_evidence_postprocess")
    assert "motion_threshold_reuse_check" not in build_workflow_stage_plan("pilot_paper", "paper_gate_and_package")
    assert "external_baseline_comparison" not in build_workflow_stage_plan("pilot_paper", "paper_gate_and_package")
    assert "external_baseline_comparison" in build_workflow_stage_plan("pilot_paper", "formal_comparison_scoring")
    assert "pilot_paper_gate" in build_workflow_stage_plan("pilot_paper", "paper_gate_and_package")
    assert "paper_profile_gate" not in build_workflow_stage_plan("pilot_paper", "paper_gate_and_package")
    assert "probe_paper_to_pilot_paper_transition_decision" not in build_workflow_stage_plan("pilot_paper", "paper_gate_and_package")

    assert full["workflow_profile"] == "full_paper"
    assert full["profile_status"] == "implemented_requires_pilot_paper_gate_and_full_scale_resources"
    assert full["enabled_for_run"] is True
    assert full["enabled_for_claim"] is False
    assert full["target_fpr"] == full_protocol["target_fpr"]
    assert full["protocol_target_fpr"] == full_protocol["target_fpr"]
    full_gate = resolve_notebook_workflow_profile("full_paper", "paper_gate_and_package")
    assert full_gate["workflow_profile"] == "full_paper"
    assert "full_paper_result_checker" in build_workflow_stage_plan("full_paper", "paper_gate_and_package")
    assert "paper_profile_gate" not in build_workflow_stage_plan("full_paper", "paper_gate_and_package")


@pytest.mark.quick
def test_profile_specific_drive_layout_prevents_result_mixing(tmp_path: Path) -> None:
    """带 workflow profile 的 Drive layout 必须按结果层级隔离 run 和 package 目录。"""
    validation_layout = build_drive_layout(
        str(tmp_path / "SSTW"),
        workflow_profile="probe_paper",
        notebook_role="generative_video_generation",
    )
    pilot_layout = build_drive_layout(
        str(tmp_path / "SSTW"),
        workflow_profile="pilot_paper",
        notebook_role="generative_video_generation",
    )

    assert validation_layout["drive_run_root"].endswith("/runs/generative_video_model_probe/probe_paper")
    assert validation_layout["drive_package_dir"].replace("\\", "/").endswith("/probe_paper/generative_video_generation_colab")
    assert validation_layout["motion_threshold_artifact_run_root"].endswith("/runs/generative_video_model_probe/motion_calibration")
    assert validation_layout["runtime_profile"] == "probe_paper"
    assert pilot_layout["drive_run_root"].endswith("/runs/generative_video_model_probe/pilot_paper")
    assert pilot_layout["drive_package_dir"].replace("\\", "/").endswith("/pilot_paper/generative_video_generation_colab")
    assert pilot_layout["motion_threshold_artifact_run_root"].endswith("/runs/generative_video_model_probe/motion_calibration")
    assert pilot_layout["runtime_profile"] == "pilot_paper"
    assert validation_layout["drive_run_root"] != pilot_layout["drive_run_root"]


@pytest.mark.quick
def test_split_stage_package_dependencies_match_notebook_responsibility() -> None:
    """阶段包依赖必须体现拆分职责, paper gate 不应直接恢复 5 个 baseline 大包。"""

    validation_layout = {
        "workflow_profile": "probe_paper",
    }
    formal_required = _default_required_stage_packages(validation_layout, "formal_comparison_scoring")
    evidence_required = _default_required_stage_packages(validation_layout, "paper_evidence_postprocess")
    gate_required = _default_required_stage_packages(validation_layout, "paper_gate_and_package")

    assert _default_required_stage_packages(validation_layout, "generative_video_quality_scoring") == [
        "generative_video_generation_colab",
        "motion_threshold_calibration_colab",
    ]
    assert _default_required_stage_packages(validation_layout, "runtime_attack") == [
        "generative_video_generation_colab",
        "generative_video_quality_scoring_colab",
    ]
    assert _default_required_stage_packages(validation_layout, "runtime_detection") == [
        "generative_video_generation_colab",
        "runtime_attack_colab",
    ]
    assert formal_required[0] == "runtime_detection_colab"
    assert "external_baseline_formal_reference_videoseal" in formal_required
    assert "external_baseline_formal_reference_wam_frame" in formal_required
    assert evidence_required == [
        "generative_video_generation_colab",
        "generative_video_quality_scoring_colab",
        "runtime_attack_colab",
        "runtime_detection_colab",
        "motion_threshold_calibration_colab",
        "formal_comparison_scoring_colab",
    ]
    assert gate_required == [
        "generative_video_generation_colab",
        "generative_video_quality_scoring_colab",
        "runtime_attack_colab",
        "runtime_detection_colab",
        "motion_threshold_calibration_colab",
        "formal_comparison_scoring_colab",
        "paper_evidence_postprocess_colab",
    ]


@pytest.mark.quick
def test_stage_package_sync_round_trips_local_run_without_drive_small_file_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """阶段 zip 交接应把本地 run_root 打包成 Drive 单 zip, 后续再复制 zip 本地解压复用。"""

    monkeypatch.setenv("SSTW_COLAB_STAGE_IO_MODE", "local_zip")
    monkeypatch.setenv("SSTW_LOCAL_STAGE_PACKAGE_CACHE_ROOT", str(tmp_path / "local_package_cache"))
    drive_root = tmp_path / "drive" / "SSTW"
    layout = build_drive_layout(
        str(drive_root),
        workflow_profile="probe_paper",
        notebook_role="generative_video_generation",
    )
    local_layout = activate_local_stage_layout(
        layout,
        notebook_role="generative_video_generation",
        local_workspace_root=tmp_path / "local_workspace",
    )
    run_root = Path(local_layout["drive_run_root"])
    write_jsonl(run_root / "records" / "generation_records.jsonl", [
        {"generation_status": "success", "prompt_id": "prompt_a", "seed_id": "seed_a"},
    ])
    write_json(run_root / "artifacts" / "generative_video_colab_runtime_decision.json", {
        "stage_id": "generative_video_generation",
        "implementation_decision": "PASS",
    })
    video_path = run_root / "videos" / "sample.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(b"sample-video")

    published = publish_colab_stage_package(
        local_layout,
        notebook_role="generative_video_generation",
        include_videos=True,
    )

    assert published["stage_package_publish_status"] == "published"
    assert Path(published["drive_stage_package_zip"]).exists()
    assert published["latest_drive_stage_package_zip"] == ""
    stage_manifest = json.loads(Path(published["stage_package_manifest_path"]).read_text(encoding="utf-8"))
    assert stage_manifest["stage_package_publish_status"] == "published"
    assert published["stage_package_entry_count"] >= 3
    assert published["stage_package_id"] == stage_package_id_for_notebook("generative_video_generation")

    shutil.rmtree(run_root)
    restored = hydrate_stage_package(
        local_layout,
        "generative_video_generation_colab",
        required=True,
    )

    assert restored["stage_package_restore_status"] == "restored"
    assert (run_root / "records" / "generation_records.jsonl").exists()
    assert (run_root / "videos" / "sample.mp4").exists()


@pytest.mark.quick
def test_probe_paper_restores_motion_calibration_stage_package_from_calibration_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """probe_paper 应从 motion_calibration profile 恢复冻结阈值阶段包。

    该测试覆盖项目特定规则: motion threshold 是独立 calibration split 的产物,
    后续 evaluation profile 只能复用该冻结包, 不能在当前 profile 下查找同名阶段包。
    """

    drive_root = tmp_path / "drive" / "SSTW"
    monkeypatch.setenv("SSTW_COLAB_STAGE_IO_MODE", "local_zip")
    monkeypatch.setenv("SSTW_LOCAL_STAGE_PACKAGE_CACHE_ROOT", str(tmp_path / "local_package_cache"))

    calibration_drive_layout = build_drive_layout(
        str(drive_root),
        workflow_profile="motion_calibration",
        notebook_role="motion_threshold_calibration",
    )
    calibration_local_layout = activate_local_stage_layout(
        calibration_drive_layout,
        notebook_role="motion_threshold_calibration",
        local_workspace_root=tmp_path / "calibration_workspace",
    )
    calibration_run_root = Path(calibration_local_layout["drive_run_root"])
    write_json(calibration_run_root / "artifacts" / "motion_threshold_calibration_decision.json", {
        "stage_id": "motion_threshold_calibration",
        "motion_threshold_calibration_ready": True,
        "motion_threshold_calibration_decision": "PASS",
        "motion_threshold_id": "motion_delta_calibrated_v1",
        "motion_threshold_source_split": "calibration",
        "motion_delta_threshold": 0.010607,
        "claim_support_status": "motion_threshold_engineering_calibrated",
    })

    published = publish_colab_stage_package(
        calibration_local_layout,
        notebook_role="motion_threshold_calibration",
        include_videos=False,
    )

    assert published["workflow_profile"] == "motion_calibration"
    assert Path(published["drive_stage_package_zip"]).exists()
    assert not (
        drive_root
        / "probe_paper"
        / "motion_threshold_calibration_colab"
    ).exists()

    generation_drive_layout = build_drive_layout(
        str(drive_root),
        workflow_profile="probe_paper",
        notebook_role="generative_video_generation",
    )
    generation_local_layout = activate_local_stage_layout(
        generation_drive_layout,
        notebook_role="generative_video_generation",
        local_workspace_root=tmp_path / "generation_workspace",
    )
    generation_run_root = Path(generation_local_layout["drive_run_root"])
    write_jsonl(generation_run_root / "records" / "generation_records.jsonl", [
        {"generation_status": "success", "prompt_id": "prompt_a", "seed_id": "seed_a"},
    ])
    write_json(generation_run_root / "artifacts" / "generative_video_colab_runtime_decision.json", {
        "stage_id": "generative_video_generation",
        "implementation_decision": "PASS",
    })
    publish_colab_stage_package(
        generation_local_layout,
        notebook_role="generative_video_generation",
        include_videos=False,
    )

    monkeypatch.setenv("SSTW_LOCAL_STAGE_WORKSPACE_ROOT", str(tmp_path / "quality_workspace"))
    validation_drive_layout = build_drive_layout(
        str(drive_root),
        workflow_profile="probe_paper",
        notebook_role="generative_video_quality_scoring",
    )
    validation_local_layout = prepare_colab_stage_layout(
        validation_drive_layout,
        notebook_role="generative_video_quality_scoring",
    )

    restored_decision_path = (
        Path(validation_local_layout["motion_threshold_artifact_run_root"])
        / "artifacts"
        / "motion_threshold_calibration_decision.json"
    )
    restore_manifest = json.loads(Path(validation_local_layout["stage_package_restore_decision_path"]).read_text(encoding="utf-8"))
    restored_decision = json.loads(restored_decision_path.read_text(encoding="utf-8"))

    assert restored_decision_path.exists()
    assert restored_decision["motion_threshold_calibration_decision"] == "PASS"
    assert restore_manifest["required_stage_package_ids"] == [
        "generative_video_generation_colab",
        "motion_threshold_calibration_colab",
    ]
    motion_restore_row = restore_manifest["stage_package_restore_rows"][1]
    assert motion_restore_row["stage_package_source_workflow_profile"] == "motion_calibration"
    assert motion_restore_row["stage_package_target_workflow_profile"] == "probe_paper"
    assert "motion_threshold/motion_calibration_motion_threshold_calibration_colab" in (
        motion_restore_row["drive_stage_package_zip"].replace("\\", "/")
    )


@pytest.mark.quick
def test_profile_specific_commands_pass_protocol_config_path(tmp_path: Path) -> None:
    """Notebook helper 构造的 gate/postprocess 命令必须显式携带当前 profile 的 protocol config。"""
    validation_layout = build_drive_layout(
        str(tmp_path / "SSTW"),
        workflow_profile="probe_paper",
        notebook_role="paper_gate_and_package",
    )
    evidence_layout = build_drive_layout(
        str(tmp_path / "SSTW"),
        workflow_profile="probe_paper",
        notebook_role="paper_evidence_postprocess",
    )
    formal_layout = build_drive_layout(
        str(tmp_path / "SSTW"),
        workflow_profile="probe_paper",
        notebook_role="formal_comparison_scoring",
    )
    pilot_layout = build_drive_layout(
        str(tmp_path / "SSTW"),
        workflow_profile="pilot_paper",
        notebook_role="paper_gate_and_package",
    )
    probe_layout = build_drive_layout(
        str(tmp_path / "SSTW"),
        workflow_profile="probe_paper",
        notebook_role="paper_gate_and_package",
    )

    validation_commands = [
        build_motion_consistency_exclusion_report_command(evidence_layout),
        build_statistical_confidence_interval_command(evidence_layout),
        build_low_fpr_formal_statistics_command(evidence_layout),
        build_sstw_measured_formal_result_command(formal_layout),
        build_formal_method_baseline_comparison_command(formal_layout),
        build_formal_baseline_difference_interval_command(formal_layout),
        build_formal_internal_ablation_summary_command(evidence_layout),
        build_paper_profile_gate_command(validation_layout),
    ]
    pilot_command = build_pilot_paper_gate_command(pilot_layout)
    probe_gate_command = build_paper_profile_gate_command(probe_layout)

    for command in validation_commands:
        assert "--config-path" in command
        assert command[command.index("--config-path") + 1] == "configs/protocol/probe_paper_generative_probe.json"
    assert "--config-path" in pilot_command
    assert pilot_command[pilot_command.index("--config-path") + 1] == "configs/protocol/pilot_paper_generative_probe.json"
    assert "--config-path" in probe_gate_command
    assert probe_gate_command[probe_gate_command.index("--config-path") + 1] == "configs/protocol/probe_paper_generative_probe.json"


@pytest.mark.quick
def test_paper_gate_commands_use_module_mode_for_check_result_scripts(tmp_path: Path) -> None:
    """paper gate 中的 check_results 脚本必须用 `python -m`, 避免 Colab 直接脚本路径丢失 repo root。"""

    gate_layout = build_drive_layout(
        str(tmp_path / "SSTW"),
        workflow_profile="probe_paper",
        notebook_role="paper_gate_and_package",
    )
    evidence_layout = build_drive_layout(
        str(tmp_path / "SSTW"),
        workflow_profile="probe_paper",
        notebook_role="paper_evidence_postprocess",
    )
    formal_layout = build_drive_layout(
        str(tmp_path / "SSTW"),
        workflow_profile="probe_paper",
        notebook_role="formal_comparison_scoring",
    )
    commands = [
        build_external_baseline_self_containment_decision_command(formal_layout),
        build_data_split_and_leakage_guard_command(evidence_layout),
        build_probe_paper_to_pilot_paper_transition_decision_command(gate_layout),
    ]

    for command in commands:
        assert command[1] == "-m"
        assert command[2].startswith("scripts.check_results.")
        assert not any(str(item).startswith("scripts/check_results/") for item in command)


@pytest.mark.quick
@pytest.mark.quick
def test_profile_specific_layout_reuses_shared_motion_threshold_artifact(tmp_path: Path) -> None:
    """profile-specific run_root 隔离后, evaluation profile 仍必须能读取独立 calibration artifact。"""
    layout = build_drive_layout(
        str(tmp_path / "SSTW"),
        workflow_profile="probe_paper",
        notebook_role="generative_video_quality_scoring",
    )
    calibration_root = Path(layout["motion_threshold_artifact_run_root"])
    write_json(calibration_root / "artifacts" / "motion_threshold_calibration_decision.json", {
        "motion_threshold_calibration_ready": True,
        "motion_threshold_calibration_decision": "PASS",
        "motion_threshold_id": "motion_delta_calibrated_v1",
        "motion_threshold_source_split": "calibration",
        "motion_delta_threshold": 0.01,
        "claim_support_status": "motion_threshold_calibration_ready",
    })

    decision = validate_motion_threshold_ready_for_profile(layout, "probe_paper")

    assert decision["motion_threshold_reuse_required"] is True
    assert decision["motion_threshold_reuse_status"] == "ready"
    assert decision["motion_threshold_id"] == "motion_delta_calibrated_v1"

    persisted = write_motion_threshold_reuse_artifact_for_profile(layout, "probe_paper")
    target_artifact = Path(layout["drive_run_root"]) / "artifacts" / "motion_threshold_calibration_decision.json"
    reuse_artifact = Path(layout["drive_run_root"]) / "artifacts" / "motion_threshold_reuse_decision.json"
    copied_decision = json.loads(target_artifact.read_text(encoding="utf-8"))

    assert persisted["motion_threshold_reuse_decision"] == "PASS"
    assert target_artifact.exists()
    assert reuse_artifact.exists()
    assert copied_decision["motion_threshold_id"] == "motion_delta_calibrated_v1"
    assert copied_decision["motion_threshold_reused_by_profile"] == "probe_paper"


@pytest.mark.quick
@pytest.mark.quick
@pytest.mark.quick
@pytest.mark.quick
@pytest.mark.quick
@pytest.mark.quick
@pytest.mark.quick
def test_external_baseline_official_result_bundle_preflight_command_is_profile_driven(tmp_path: Path) -> None:
    """官方结果包 preflight 命令必须由 workflow layout 生成, 不能在 Notebook 中硬写路径。"""
    layout = build_drive_layout(str(tmp_path / "SSTW"), workflow_profile="probe_paper")
    bootstrap_command = build_external_baseline_official_resource_bootstrap_command(layout)
    generation_command = build_external_baseline_official_bundle_generation_command(layout)
    command = build_external_baseline_official_result_bundle_preflight_command(layout)

    assert bootstrap_command[:3] == [bootstrap_command[0], "-m", "external_baseline.official_resource_bootstrap"]
    assert "--resource-root" in bootstrap_command
    assert layout["external_baseline_resource_root"] in bootstrap_command
    assert generation_command[:3] == [generation_command[0], "-m", "external_baseline.official_bundle_generator"]
    assert "--bundle-root" in generation_command
    assert layout["external_baseline_official_result_bundle_root"] in generation_command
    assert "--generate-auto-supported" in generation_command
    assert command[:3] == [command[0], "-m", "external_baseline.official_result_bundle"]
    assert "--run-root" in command
    assert layout["drive_run_root"] in command
    assert "--output-json" in command
    assert "external_baseline_official_result_bundle_preflight_decision.json" in command[-1]


@pytest.mark.quick
def test_probe_paper_run_through_test_keeps_preflight_fail_but_does_not_raise(tmp_path: Path) -> None:
    """run-through test 只能放行 Notebook 工程链路, 不能把 FAIL preflight 改成 PASS。"""
    layout = build_drive_layout(str(tmp_path / "SSTW"))
    bridge_templates = build_modern_baseline_official_bridge_command_templates("probe_paper")
    bridge_decision = build_modern_baseline_official_bridge_preflight_decision(
        layout,
        profile="probe_paper",
        command_env=bridge_templates,
        use_bridge_commands=True,
        require_bridge_official_commands=True,
    )

    assert bridge_decision["external_baseline_official_bridge_preflight_decision"] == "FAIL"
    with pytest.raises(RuntimeError):
        validate_modern_baseline_official_bridge_for_profile(bridge_decision)
    validate_modern_baseline_official_bridge_for_profile(bridge_decision, allow_run_through_test=True)

    external_decision = write_external_baseline_colab_preflight_decision(
        layout,
        profile="probe_paper",
        command_env={},
        require_modern_baseline_commands_for_paper_gate=True,
        run_external_baseline_source_clone=True,
        evidence_paths=[],
    )
    assert external_decision["external_baseline_colab_preflight_decision"] == "FAIL"
    with pytest.raises(RuntimeError):
        validate_modern_baseline_commands_for_profile(external_decision)
    validate_modern_baseline_commands_for_profile(external_decision, allow_run_through_test=True)


@pytest.mark.quick
def test_generative_video_drive_layout_uses_sstw_drive_root() -> None:
    """Colab workflow 的长期输出必须默认落盘到 MyDrive/SSTW 子目录。"""
    layout = build_drive_layout()
    assert layout["drive_project_root"] == "/content/drive/MyDrive/SSTW"
    assert layout["drive_dataset_root"].startswith("/content/drive/MyDrive/SSTW/datasets/")
    assert layout["drive_run_root"].startswith("/content/drive/MyDrive/SSTW/runs/")
    assert layout["drive_package_dir"].startswith("/content/drive/MyDrive/SSTW/probe_paper/")
    assert layout["drive_log_dir"].startswith("/content/drive/MyDrive/SSTW/logs/")


@pytest.mark.quick
def test_generative_video_local_zip_does_not_precreate_drive_hot_dirs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """local_zip 模式下主流程 Notebook 不应在 Drive 上预创建空 run / log / dataset 目录。"""

    monkeypatch.setenv("SSTW_COLAB_STAGE_IO_MODE", "local_zip")
    drive_root = tmp_path / "SSTW"

    layout = ensure_drive_layout(
        str(drive_root),
        workflow_profile="probe_paper",
        notebook_role="generative_video_generation",
    )

    assert Path(layout["drive_project_root"]).exists()
    assert not (drive_root / "runs" / "generative_video_model_probe" / "probe_paper").exists()
    assert not (drive_root / "logs" / "generative_video_model_probe" / "probe_paper").exists()
    assert not (drive_root / "datasets" / "generative_video_prompt_suite").exists()
    assert not (drive_root / "probe_paper" / "generative_video_generation_colab").exists()


@pytest.mark.quick
def test_generative_video_drive_packager_creates_archive_and_manifest(tmp_path: Path) -> None:
    """Drive packager 必须从已有 run outputs 生成 zip 和 package manifest。"""
    run_root = tmp_path / "runs" / "generative_video_runtime"
    package_dir = tmp_path / "packages"
    validation_protocol = json.loads(Path("configs/protocol/probe_paper_generative_probe.json").read_text(encoding="utf-8"))
    pilot_protocol = json.loads(Path("configs/protocol/pilot_paper_generative_probe.json").read_text(encoding="utf-8"))
    write_jsonl(run_root / "records" / "generation_records.jsonl", [{"generation_model_id": "model", "prompt_id": "prompt"}])
    write_json(run_root / "artifacts" / "generative_video_colab_runtime_decision.json", {"stage_id": "generative_video_generation", "implementation_decision": "PASS", "mechanism_decision": "FAIL"})
    write_json(run_root / "artifacts" / "generation_manifest.json", {"artifact_id": "manifest"})
    write_json(run_root / "artifacts" / "runtime_attack_decision.json", {
        "runtime_attack_decision": "PASS",
        "runtime_attack_record_count": 48,
        "runtime_attack_ready_count": 48,
    })
    write_json(run_root / "artifacts" / "runtime_detection_decision.json", {
        "runtime_detection_decision": "PASS",
        "runtime_detection_record_count": 48,
        "runtime_detection_ready_count": 48,
    })
    write_json(run_root / "artifacts" / "motion_consistency_exclusion_decision.json", {
        "motion_consistency_exclusion_decision": "PASS",
        "motion_consistency_included_count": 46,
        "motion_consistency_excluded_count": 2,
    })
    write_json(run_root / "artifacts" / "sstw_measured_formal_decision.json", {
        "sstw_measured_formal_decision": "PASS",
        "sstw_measured_formal_record_count": 48,
        "sstw_measured_formal_score_mean": 0.82,
        "sstw_measured_formal_detectable_rate": 1.0,
    })
    write_json(run_root / "artifacts" / "motion_threshold_calibration_decision.json", {
        "motion_threshold_calibration_decision": "INSUFFICIENT_SAMPLE",
        "motion_threshold_id": "motion_delta_heuristic_v1",
        "motion_threshold_source_split": "heuristic_precalibration",
        "motion_threshold_calibration_required": True,
    })
    write_json(run_root / "artifacts" / "paper_profile_gate_decision.json", {
        "paper_profile_gate_decision": "FAIL",
        "claim_support_status": "paper_profile_blocked",
        "paper_result_level": "probe_paper",
        "target_fpr": validation_protocol["target_fpr"],
        "validation_missing_requirement_count": 5,
        "validation_generation_record_count": 0,
        "validation_prompt_count": 0,
        "validation_seed_per_prompt_min": 0,
    })
    write_json(run_root / "artifacts" / "validation_internal_ablation_decision.json", {
        "validation_internal_ablation_decision": "PASS",
        "internal_ablation_record_count": 12,
    })
    write_json(run_root / "artifacts" / "adaptive_attack_decision.json", {
        "adaptive_attack_decision": "PASS",
        "adaptive_attack_record_count": 72,
        "adaptive_robustness_claim_allowed": False,
    })
    write_json(run_root / "artifacts" / "replay_and_sketch_gate_decision.json", {
        "replay_and_sketch_gate_decision": "PASS",
        "replay_and_sketch_evidence_level": "validation_runtime_trace_proxy",
        "trajectory_sketch_verified_count": 12,
        "replay_uncertainty_ready_count": 12,
        "wrong_sampler_replay_rejected_count": 12,
        "wrong_prompt_replay_rejected_count": 12,
        "claim3_full_support_allowed": False,
    })
    write_json(run_root / "artifacts" / "claim3_downgrade_decision.json", {
        "claim3_downgrade_decision": "PASS",
        "claim3_downgraded": True,
        "claim3_full_support_allowed": False,
        "replay_or_sketch_status": "claim3_explicitly_downgraded",
    })
    write_json(run_root / "artifacts" / "statistical_confidence_interval_decision.json", {
        "statistical_confidence_interval_decision": "PASS",
        "ci_total_count": 12,
    })
    write_json(run_root / "artifacts" / "low_fpr_formal_statistics_decision.json", {
        "low_fpr_formal_statistics_decision": "PASS",
        "low_fpr_formal_statistics_record_count": 2,
        "formal_low_fpr_claim_allowed": False,
    })
    write_json(run_root / "artifacts" / "pilot_paper_gate_decision.json", {
        "pilot_paper_gate_decision": "PASS",
        "claim_support_status": "pilot_paper_calibrated_heldout_claim_ready",
        "paper_result_level": "pilot_paper",
        "paper_protocol_level": "paper_grade_protocol",
        "paper_protocol_difference_from_full_paper": "sample_scale_and_target_fpr_only",
        "pilot_paper_protocol_matches_full_paper": False,
        "pilot_paper_claim_allowed": True,
        "pilot_paper_missing_requirement_count": 0,
        "threshold_protocol": "calibration_split_to_frozen_threshold_to_heldout_test_split",
        "threshold_source_split": "calibration",
        "test_time_threshold_update_blocked": True,
        "target_fpr": pilot_protocol["target_fpr"],
        "tpr_at_target_fpr": 0.91,
        "target_fpr_claim_allowed": True,
        "blocked_target_fpr": pilot_protocol["blocked_target_fpr"],
        "blocked_target_fpr_claim_allowed": False,
        "tpr_at_fpr_01": 0.91,
        "calibration_negative_fpr_at_threshold": 0.008,
        "heldout_negative_fpr_at_threshold": 0.009,
        "observed_negative_fpr_at_threshold": 0.009,
        "calibration_negative_event_count": 5000,
        "heldout_test_negative_event_count": 5000,
        "heldout_negative_event_count": 5000,
        "heldout_attacked_positive_event_count": 2300,
        "attacked_positive_event_count": 2300,
        "tpr_at_fpr_01_pilot_claim_allowed": True,
        "tpr_at_fpr_001_claim_allowed": False,
    })
    write_json(run_root / "artifacts" / "validation_artifact_rebuild_dry_run_decision.json", {
        "validation_artifact_rebuild_dry_run_decision": "PASS",
        "artifact_rebuild_missing_count": 0,
    })
    write_json(run_root / "artifacts" / "external_baseline_comparison_decision.json", {
        "external_baseline_comparison_decision": "PASS",
        "external_baseline_comparison_record_count": 96,
        "external_baseline_comparison_ready_count": 48,
        "external_baseline_measured_adapter_count": 2,
        "external_baseline_comparison_status": "adapter_proxy_records_written",
        "external_baseline_comparison_table_status": "ready",
    })
    write_json(run_root / "artifacts" / "formal_method_baseline_comparison_decision.json", {
        "formal_method_baseline_comparison_decision": "PASS",
        "formal_comparison_ready_method_count": 6,
        "formal_comparison_modern_baseline_ready_count": 5,
        "formal_comparison_missing_method_count": 0,
    })
    write_json(run_root / "artifacts" / "formal_baseline_difference_interval_decision.json", {
        "formal_baseline_difference_interval_decision": "PASS",
        "difference_interval_ready_count": 5,
        "difference_interval_missing_baseline_count": 0,
    })
    write_json(run_root / "artifacts" / "formal_internal_ablation_summary_decision.json", {
        "formal_internal_ablation_summary_decision": "PASS",
        "formal_internal_ablation_variant_count": 8,
        "formal_internal_ablation_full_method_formal_ready": True,
    })

    payload = package_generative_video_colab_run(run_root, package_dir, include_videos=False)
    archive_path = Path(payload["archive_path"])
    manifest_path = Path(payload["package_manifest_path"])
    assert archive_path.exists()
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["generation_manifest_status"] == "present"
    assert manifest["decision_summary"]["implementation_decision"] == "PASS"
    assert manifest["decision_summary"]["runtime_attack_decision"] == "PASS"
    assert manifest["decision_summary"]["runtime_attack_record_count"] == 48
    assert manifest["decision_summary"]["runtime_attack_ready_count"] == 48
    assert manifest["decision_summary"]["runtime_detection_decision"] == "PASS"
    assert manifest["decision_summary"]["runtime_detection_record_count"] == 48
    assert manifest["decision_summary"]["runtime_detection_ready_count"] == 48
    assert manifest["decision_summary"]["motion_consistency_exclusion_decision"] == "PASS"
    assert manifest["decision_summary"]["motion_consistency_included_count"] == 46
    assert manifest["decision_summary"]["motion_consistency_excluded_count"] == 2
    assert manifest["decision_summary"]["sstw_measured_formal_decision"] == "PASS"
    assert manifest["decision_summary"]["sstw_measured_formal_record_count"] == 48
    assert manifest["decision_summary"]["sstw_measured_formal_score_mean"] == 0.82
    assert manifest["decision_summary"]["sstw_measured_formal_detectable_rate"] == 1.0
    assert manifest["decision_summary"]["paper_profile_gate_decision"] == "FAIL"
    assert manifest["decision_summary"]["paper_profile_claim_support_status"] == "paper_profile_blocked"
    assert manifest["decision_summary"]["paper_profile_result_level"] == "probe_paper"
    assert manifest["decision_summary"]["paper_profile_target_fpr"] == validation_protocol["target_fpr"]
    assert manifest["decision_summary"]["paper_profile_missing_requirement_count"] == 5
    assert manifest["decision_summary"]["external_baseline_comparison_decision"] == "PASS"
    assert manifest["decision_summary"]["external_baseline_comparison_record_count"] == 96
    assert manifest["decision_summary"]["external_baseline_comparison_ready_count"] == 48
    assert manifest["decision_summary"]["external_baseline_measured_adapter_count"] == 2
    assert manifest["decision_summary"]["external_baseline_comparison_table_status"] == "ready"
    assert manifest["decision_summary"]["formal_method_baseline_comparison_decision"] == "PASS"
    assert manifest["decision_summary"]["formal_comparison_ready_method_count"] == 6
    assert manifest["decision_summary"]["formal_comparison_modern_baseline_ready_count"] == 5
    assert manifest["decision_summary"]["formal_comparison_missing_method_count"] == 0
    assert manifest["decision_summary"]["formal_baseline_difference_interval_decision"] == "PASS"
    assert manifest["decision_summary"]["difference_interval_ready_count"] == 5
    assert manifest["decision_summary"]["difference_interval_missing_baseline_count"] == 0
    assert manifest["decision_summary"]["formal_internal_ablation_summary_decision"] == "PASS"
    assert manifest["decision_summary"]["formal_internal_ablation_variant_count"] == 8
    assert manifest["decision_summary"]["formal_internal_ablation_full_method_formal_ready"] is True
    assert manifest["decision_summary"]["validation_internal_ablation_decision"] == "PASS"
    assert manifest["decision_summary"]["validation_internal_ablation_record_count"] == 12
    assert manifest["decision_summary"]["adaptive_attack_decision"] == "PASS"
    assert manifest["decision_summary"]["adaptive_attack_record_count"] == 72
    assert manifest["decision_summary"]["adaptive_robustness_claim_allowed"] is False
    assert manifest["decision_summary"]["replay_and_sketch_gate_decision"] == "PASS"
    assert manifest["decision_summary"]["replay_and_sketch_evidence_level"] == "validation_runtime_trace_proxy"
    assert manifest["decision_summary"]["trajectory_sketch_verified_count"] == 12
    assert manifest["decision_summary"]["replay_and_sketch_claim3_full_support_allowed"] is False
    assert manifest["decision_summary"]["claim3_downgrade_decision"] == "PASS"
    assert manifest["decision_summary"]["claim3_downgraded"] is True
    assert manifest["decision_summary"]["claim3_full_support_allowed"] is False
    assert manifest["decision_summary"]["replay_or_sketch_status"] == "claim3_explicitly_downgraded"
    assert manifest["decision_summary"]["statistical_confidence_interval_decision"] == "PASS"
    assert manifest["decision_summary"]["statistical_confidence_interval_total_count"] == 12
    assert manifest["decision_summary"]["low_fpr_formal_statistics_decision"] == "PASS"
    assert manifest["decision_summary"]["low_fpr_formal_statistics_record_count"] == 2
    assert manifest["decision_summary"]["formal_low_fpr_claim_allowed"] is False
    assert manifest["decision_summary"]["pilot_paper_gate_decision"] == "PASS"
    assert manifest["decision_summary"]["pilot_paper_claim_support_status"] == "pilot_paper_calibrated_heldout_claim_ready"
    assert manifest["decision_summary"]["pilot_paper_result_level"] == "pilot_paper"
    assert manifest["decision_summary"]["pilot_paper_protocol_level"] == "paper_grade_protocol"
    assert manifest["decision_summary"]["pilot_paper_protocol_difference_from_full_paper"] == "sample_scale_and_target_fpr_only"
    assert manifest["decision_summary"]["pilot_paper_protocol_matches_full_paper"] is False
    assert manifest["decision_summary"]["pilot_paper_claim_allowed"] is True
    assert manifest["decision_summary"]["pilot_paper_missing_requirement_count"] == 0
    assert manifest["decision_summary"]["pilot_paper_threshold_protocol"] == "calibration_split_to_frozen_threshold_to_heldout_test_split"
    assert manifest["decision_summary"]["pilot_paper_threshold_source_split"] == "calibration"
    assert manifest["decision_summary"]["pilot_paper_test_time_threshold_update_blocked"] is True
    assert manifest["decision_summary"]["pilot_paper_target_fpr"] == pilot_protocol["target_fpr"]
    assert manifest["decision_summary"]["pilot_paper_tpr_at_target_fpr"] == 0.91
    assert manifest["decision_summary"]["pilot_paper_target_fpr_claim_allowed"] is True
    assert manifest["decision_summary"]["pilot_paper_blocked_target_fpr"] == pilot_protocol["blocked_target_fpr"]
    assert manifest["decision_summary"]["pilot_paper_blocked_target_fpr_claim_allowed"] is False
    assert manifest["decision_summary"]["pilot_paper_tpr_at_fpr_01"] == 0.91
    assert manifest["decision_summary"]["pilot_paper_calibration_negative_fpr_at_threshold"] == 0.008
    assert manifest["decision_summary"]["pilot_paper_heldout_negative_fpr_at_threshold"] == 0.009
    assert manifest["decision_summary"]["pilot_paper_observed_negative_fpr_at_threshold"] == 0.009
    assert manifest["decision_summary"]["pilot_paper_calibration_negative_event_count"] == 5000
    assert manifest["decision_summary"]["pilot_paper_heldout_test_negative_event_count"] == 5000
    assert manifest["decision_summary"]["pilot_paper_heldout_negative_event_count"] == 5000
    assert manifest["decision_summary"]["pilot_paper_heldout_attacked_positive_event_count"] == 2300
    assert manifest["decision_summary"]["pilot_paper_attacked_positive_event_count"] == 2300
    assert manifest["decision_summary"]["pilot_paper_tpr_at_fpr_01_pilot_claim_allowed"] is True
    assert manifest["decision_summary"]["pilot_paper_tpr_at_fpr_001_claim_allowed"] is False
    assert manifest["decision_summary"]["validation_artifact_rebuild_dry_run_decision"] == "PASS"
    assert manifest["decision_summary"]["validation_artifact_rebuild_missing_count"] == 0
    assert manifest["decision_summary"]["motion_threshold_calibration_decision"] == "INSUFFICIENT_SAMPLE"
    assert manifest["decision_summary"]["motion_threshold_id"] == "motion_delta_heuristic_v1"
    assert manifest["decision_summary"]["motion_threshold_source_split"] == "heuristic_precalibration"
    assert manifest["decision_summary"]["motion_threshold_calibration_required"] is True
    assert re.match(r"generative_video_runtime_\d{8}_\d{6}_[a-z0-9_\-]+\.zip", archive_path.name)
    assert manifest["package_batch_id"] == f"{manifest['package_utc_time']}_{manifest['package_short_commit']}"
    assert archive_path.stem.endswith(manifest["package_batch_id"])
    assert manifest_path.stem.endswith(f"{manifest['package_batch_id']}_package_manifest")
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
    assert any(name.endswith("records/generation_records.jsonl") for name in names)


@pytest.mark.quick
def test_generative_video_drive_packager_keeps_runtime_mechanism_decision(tmp_path: Path) -> None:
    """package manifest 不应再用已删除的 proxy 后处理结果提升 runtime 机制判定。"""
    run_root = tmp_path / "runs" / "generative_video_runtime"
    package_dir = tmp_path / "packages"
    write_jsonl(run_root / "records" / "generation_records.jsonl", [{"generation_model_id": "model", "prompt_id": "prompt"}])
    write_json(run_root / "artifacts" / "generative_video_colab_runtime_decision.json", {
        "stage_id": "generative_video_generation",
        "implementation_decision": "PASS",
        "mechanism_decision": "FAIL",
    })

    payload = package_generative_video_colab_run(run_root, package_dir, include_videos=False)

    manifest = json.loads(Path(payload["package_manifest_path"]).read_text(encoding="utf-8"))
    summary = manifest["decision_summary"]
    assert summary["runtime_mechanism_decision"] == "FAIL"
    assert summary["mechanism_decision"] == "FAIL"
    assert summary["effective_mechanism_decision"] == "FAIL"
    assert summary["mechanism_decision_source"] == "runtime_mechanism_artifact"
    assert "postprocess_mechanism_decision" not in summary


@pytest.mark.quick
def test_generative_video_colab_runtime_uses_optional_hf_token_without_recording_secret() -> None:
    """模型加载应支持 HF_TOKEN, 但只能记录 provided / not_provided 状态。"""
    runtime_text = Path("experiments/generative_video_model_probe/colab_runtime.py").read_text(encoding="utf-8")
    assert "os.environ.get(\"HF_TOKEN\")" in runtime_text
    assert "token=hf_token" in runtime_text
    assert "WanPipeline" in runtime_text
    assert "WAN21_PRIMARY_MODEL_ID" in runtime_text
    assert "hf_token_status" in runtime_text
    assert "provided" in runtime_text
    assert "not_provided" in runtime_text
    assert '"pilot": {"prompt_limit": 8, "seed_limit": 2' in runtime_text
    assert '"probe_paper": {' in runtime_text
    assert '"seed_suite_roles": ["probe_paper"]' in runtime_text
    assert 'default="pilot"' in runtime_text
    assert 'default=WAN21_PRIMARY_MODEL_ID' in runtime_text

