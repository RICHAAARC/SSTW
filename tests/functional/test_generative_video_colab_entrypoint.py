"""验证 B5 Colab Notebook 入口、Drive 落盘与 prompt suite 构造。"""

from __future__ import annotations

import json
from pathlib import Path
import re
import zipfile

import pytest

from paper_workflow.notebook_utils.generative_video_model_probe_workflow import (
    build_drive_layout,
    build_workflow_stage_plan,
    build_modern_baseline_command_env,
    default_workflow_profile_for_notebook_role,
    resolve_notebook_workflow_profile,
    validate_motion_threshold_ready_for_profile,
    write_external_baseline_colab_preflight_decision,
)
from experiments.generative_video_model_probe.colab_runtime import _build_generation_plan
from scripts.package_results.generative_video_drive_packager import package_generative_video_colab_run
from scripts.prepare_generative_video_prompt_suite import write_prompt_suite
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
def test_validation_scale_profile_expands_pilot_prompts_to_three_seeds(tmp_path: Path) -> None:
    """validation_scale profile 必须使用 pilot 后的 8 个 prompt 和 3 个 seed。"""
    output_root = tmp_path / "prompt_suite"
    summary = write_prompt_suite(output_root)
    suite = json.loads(Path(summary["prompt_suite_path"]).read_text(encoding="utf-8"))
    plan = _build_generation_plan(suite, "validation_scale", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", None)

    assert len(plan) == 24
    assert len({item["prompt_id"] for item in plan}) == 8
    assert {item["seed_id"] for item in plan} == {"seed_main_a", "seed_main_b", "seed_heldout_c"}
    assert {item["prompt_suite_role"] for item in plan} == {"main", "heldout_prompt", "pilot_main"}


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

    assert suite["pilot_paper_design"]["target_fpr"] == 0.01
    assert suite["pilot_paper_design"]["paper_result_level"] == "pilot_paper"
    assert suite["pilot_paper_design"]["paper_protocol_difference_from_full_paper"] == "sample_scale_only"
    assert suite["pilot_paper_design"]["recommended_runtime_profile"] == "pilot_paper"
    assert suite["pilot_paper_design"]["threshold_protocol"] == "calibration_split_to_frozen_threshold_to_heldout_test_split"
    assert suite["pilot_paper_design"]["target_generation_video_count"] == 168
    assert suite["pilot_paper_design"]["target_calibration_negative_event_count"] == 1008
    assert suite["pilot_paper_design"]["target_heldout_test_negative_event_count"] == 1008
    assert suite["pilot_paper_design"]["target_test_attacked_positive_event_count"] == 252
    assert len(pilot_paper_prompts) == 21
    assert len(pilot_paper_seeds) == 8
    assert len(plan) == 168
    assert len(pilot_paper_plan) == 168
    assert [(item["prompt_id"], item["seed_id"]) for item in pilot_paper_plan] == [(item["prompt_id"], item["seed_id"]) for item in plan]
    assert len({item["prompt_id"] for item in plan}) == 21
    assert {item["seed_suite_role"] for item in plan} == {"pilot_paper"}
    assert {item["prompt_suite_role"] for item in plan} == {"pilot_paper"}
    assert {item["split"] for item in plan} == {"calibration", "test"}
    assert sum(1 for item in plan if item["split"] == "calibration") == 84
    assert sum(1 for item in plan if item["split"] == "test") == 84


@pytest.mark.quick
def test_generative_video_colab_notebook_calls_repository_modules() -> None:
    """runtime Notebook 只能作为入口, 必须通过统一 workflow profile 调用仓库模块。"""
    notebook_path = Path("paper_workflow/colab_utils/generative_video_runtime_colab.ipynb")
    assert notebook_path.exists()
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    helper_text = Path("paper_workflow/notebook_utils/generative_video_model_probe_workflow.py").read_text(encoding="utf-8")

    assert "/content/drive/MyDrive/SSTW" in source
    assert "drive.mount('/content/drive')" in source
    assert "SSTW_WORKFLOW_PROFILE" in source
    assert "NOTEBOOK_ROLE = 'generative_video_runtime'" in source
    assert "default_workflow_profile_for_notebook_role" in source
    assert "resolve_notebook_workflow_profile" in source
    assert "ensure_drive_layout(" in source
    assert "workflow_profile=WORKFLOW_PROFILE" in source
    assert "stage_enabled(" in source
    assert "workflow_stage_enabled" in source
    assert "HF_TOKEN" in source
    assert "add_to_git_credential=False" in source
    assert "generative_video_model_probe_workflow" in source
    assert "MODEL_ID = os.environ.get('SSTW_MODEL_ID'" in source
    assert "RUN_EXTERNAL_BASELINE_SOURCE_CLONE" in source
    assert "EXTERNAL_BASELINE_EVIDENCE_PATHS" in source
    assert "REQUIRE_MODERN_BASELINE_COMMANDS_FOR_PAPER_GATE" in source
    assert "SSTW_EXTERNAL_BASELINE_EVIDENCE_PATHS" in source
    assert "build_modern_baseline_command_env" in source
    assert "write_external_baseline_colab_preflight_decision" in source
    assert "validate_modern_baseline_commands_for_profile" in source
    assert "external_baseline_colab_preflight_decision" in helper_text
    assert "validate_motion_threshold_ready_for_profile" in source
    assert "build_formal_metric_command" in source
    assert "build_motion_threshold_calibration_command" in source
    assert "build_mechanism_postprocess_command" in source
    assert "build_pilot_matrix_postprocess_command" in source
    assert "build_runtime_attack_command" in source
    assert "build_runtime_detection_command" in source
    assert "build_small_scale_claim_pilot_gate_command" in source
    assert "build_external_baseline_source_intake_command" not in source
    assert "build_external_baseline_comparison_command" not in source
    assert "build_validation_internal_ablation_command" not in source
    assert "build_adaptive_attack_command" not in source
    assert "build_replay_and_sketch_gate_command" not in source
    assert "build_claim3_downgrade_command" not in source
    assert "build_statistical_confidence_interval_command" not in source
    assert "build_pilot_paper_gate_command" not in source
    assert "build_validation_artifact_rebuild_dry_run_command" not in source
    assert "build_validation_scale_gate_command" not in source
    assert "scripts/prepare_generative_video_prompt_suite.py" in helper_text
    assert "experiments.generative_video_model_probe.colab_runtime" in helper_text
    assert "experiments.generative_video_model_probe.formal_metric_runner" in helper_text
    assert "experiments.generative_video_model_probe.motion_threshold_calibration" in helper_text
    assert "experiments.generative_video_model_probe.postprocess_runner" in helper_text
    assert "experiments.generative_video_model_probe.pilot_matrix_postprocess" in helper_text
    assert "experiments.generative_video_model_probe.attack_runner" in helper_text
    assert "experiments.generative_video_model_probe.detection_runner" in helper_text
    assert "scripts/build_external_baseline_source_intake.py" in helper_text
    assert "--execute-clone" in helper_text
    assert "experiments.generative_video_model_probe.external_baseline_runner" in helper_text
    assert "experiments.generative_video_model_probe.pilot_claim_gate" in helper_text
    assert "experiments.generative_video_model_probe.validation_internal_ablation" in helper_text
    assert "experiments.generative_video_model_probe.adaptive_attack_runner" in helper_text
    assert "experiments.generative_video_model_probe.replay_and_sketch_gate" in helper_text
    assert "experiments.generative_video_model_probe.claim3_downgrade" in helper_text
    assert "experiments.generative_video_model_probe.statistical_confidence_interval" in helper_text
    assert "experiments.generative_video_model_probe.pilot_paper_gate" in helper_text
    assert "experiments.generative_video_model_probe.validation_artifact_rebuild" in helper_text
    assert "experiments.generative_video_model_probe.validation_scale_gate" in helper_text
    assert "scripts/package_results/generative_video_drive_packager.py" in helper_text
    assert "pytest -q" in source
    assert "tools/harness/run_all_audits.py" in source
    assert "pilot_paper_results" not in source


@pytest.mark.quick
def test_split_colab_notebooks_are_profile_driven() -> None:
    """拆分后的 Colab Notebook 必须通过统一配置切换 profile, 不能维护独立路径硬编码。"""
    expected_roles = {
        "motion_threshold_calibration_colab.ipynb": "motion_threshold_calibration",
        "generative_video_runtime_colab.ipynb": "generative_video_runtime",
        "external_baseline_formal_scoring_colab.ipynb": "external_baseline_formal_scoring",
        "paper_gate_and_package_colab.ipynb": "paper_gate_and_package",
    }
    for notebook_name, role in expected_roles.items():
        notebook_path = Path("paper_workflow/colab_utils") / notebook_name
        assert notebook_path.exists()
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        source = "".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        switch_source = "".join(notebook["cells"][2].get("source", []))
        assert "# 1.1 可编辑 workflow profile 切换" in switch_source
        if role == "motion_threshold_calibration":
            assert "SSTW_WORKFLOW_PROFILE_VALUE = ''" in switch_source
        else:
            assert "SSTW_WORKFLOW_PROFILE_VALUE = 'validation_scale'" in switch_source
        assert "os.environ['SSTW_WORKFLOW_PROFILE']" in switch_source
        assert source.index("SSTW_WORKFLOW_PROFILE_VALUE") < source.index("resolve_notebook_workflow_profile")
        assert f"NOTEBOOK_ROLE = '{role}'" in source
        assert "SSTW_WORKFLOW_PROFILE" in source
        assert "resolve_notebook_workflow_profile" in source
        assert "workflow_profile=WORKFLOW_PROFILE" in source
        assert "stage_enabled(" in source
        assert "drive.mount('/content/drive')" in source
        assert "git clone" in source
        assert "tools/harness/run_all_audits.py" in source
        assert "pilot_paper_results" not in source
        assert "full_paper_results" not in source

    runtime_source = Path("paper_workflow/colab_utils/generative_video_runtime_colab.ipynb").read_text(encoding="utf-8")
    baseline_source = Path("paper_workflow/colab_utils/external_baseline_formal_scoring_colab.ipynb").read_text(encoding="utf-8")
    gate_source = Path("paper_workflow/colab_utils/paper_gate_and_package_colab.ipynb").read_text(encoding="utf-8")
    assert "external_baseline_colab_preflight" in runtime_source
    assert "build_external_baseline_comparison_command" not in runtime_source
    assert "build_colab_runtime_command" not in baseline_source
    assert "build_external_baseline_comparison_command" in baseline_source
    assert "build_pilot_paper_gate_command" in gate_source
    assert "build_validation_scale_gate_command" in gate_source


@pytest.mark.quick
def test_colab_workflow_readme_documents_validation_scale_execution() -> None:
    """Colab workflow README 必须说明 validation-scale 执行顺序、profile 切换和 Drive 落盘边界。"""
    readme_path = Path("paper_workflow/colab_utils/README.md")
    assert readme_path.exists()
    text = readme_path.read_text(encoding="utf-8")

    assert "motion_threshold_calibration_colab.ipynb" in text
    assert "generative_video_runtime_colab.ipynb" in text
    assert "external_baseline_formal_scoring_colab.ipynb" in text
    assert "paper_gate_and_package_colab.ipynb" in text
    assert text.index("motion_threshold_calibration_colab.ipynb") < text.index("generative_video_runtime_colab.ipynb")
    assert text.index("generative_video_runtime_colab.ipynb") < text.index("external_baseline_formal_scoring_colab.ipynb")
    assert text.index("external_baseline_formal_scoring_colab.ipynb") < text.index("paper_gate_and_package_colab.ipynb")
    assert "SSTW_WORKFLOW_PROFILE_VALUE = 'validation_scale'" in text
    assert "SSTW_WORKFLOW_PROFILE_VALUE = 'pilot_paper'" in text
    assert "不要把该 Notebook 切换到 `validation_scale` 或 `pilot_paper`" in text
    assert "SSTW_VIDEOSHIELD_EVAL_COMMAND" in text
    assert "SSTW_SIGMARK_EVAL_COMMAND" in text
    assert "SSTW_SPDMARK_EVAL_COMMAND" in text
    assert "SSTW_VIDEOMARK_EVAL_COMMAND" in text
    assert "SSTW_VIDSIG_EVAL_COMMAND" in text
    assert "SSTW_VIDEOSEAL_EVAL_COMMAND" in text
    assert "validation_scale_gate_decision.json" in text
    assert "pilot_paper_gate_decision.json" in text
    assert "/content/drive/MyDrive/SSTW" in text
    assert r"G:\我的云端硬盘\SSTW" in text
    assert "不要手写 records" in text
    assert "<utc_time>_<short_commit>" in text
    assert "SSTW 工作量进度" in text
    assert "Wan2.1 生成: len(plan)" in text
    assert "runtime detection 视频扫描: len(runtime_attack_records)" in text
    assert "单个 baseline 读取 runtime 视频: len(comparable_detection_records)" in text
    assert "不需要在 Notebook 中硬编码 24、168 或 224" in text
    assert "不写入正式 records、tables、figures、reports、manifests 或 claim artifacts" in text


@pytest.mark.quick
def test_notebook_workflow_profile_config_supports_profile_switching() -> None:
    """统一配置层必须能区分 validation_scale、pilot_paper 和未开放的 full_paper。"""
    assert default_workflow_profile_for_notebook_role("generative_video_runtime") == "validation_scale"
    validation = resolve_notebook_workflow_profile("validation_scale", "paper_gate_and_package")
    pilot = resolve_notebook_workflow_profile("pilot_paper", "paper_gate_and_package")
    full = resolve_notebook_workflow_profile("full_paper", config_path="configs/paper_workflow/generative_video_notebook_workflows.json", allow_disabled=True)

    assert validation["workflow_profile"] == "validation_scale"
    assert validation["result_tier"] == "validation_scale"
    assert validation["enabled_for_claim"] is False
    assert validation["target_fpr"] == 0.01
    assert "validation_scale_gate" in build_workflow_stage_plan("validation_scale", "paper_gate_and_package")
    assert "pilot_paper_gate" not in build_workflow_stage_plan("validation_scale", "paper_gate_and_package")

    assert pilot["requested_workflow_profile"] == "pilot_paper"
    assert pilot["workflow_profile"] == "pilot_paper"
    assert pilot["profile_alias_applied"] is False
    assert pilot["result_tier"] == "pilot_paper"
    assert pilot["enabled_for_claim"] is True
    assert pilot["method_sample_count"] == 168
    assert pilot["baseline_sample_count"] == 84
    assert pilot["protocol_config_path"] == "configs/protocol/pilot_paper_generative_probe.json"
    assert "pilot_paper_gate" in build_workflow_stage_plan("pilot_paper", "paper_gate_and_package")
    assert "validation_scale_gate" not in build_workflow_stage_plan("pilot_paper", "paper_gate_and_package")

    assert full["workflow_profile"] == "full_paper"
    assert full["profile_status"] == "design_registered_not_ready"
    assert full["enabled_for_run"] is False
    assert full["enabled_for_claim"] is False
    with pytest.raises(RuntimeError):
        resolve_notebook_workflow_profile("full_paper", "paper_gate_and_package")


@pytest.mark.quick
def test_profile_specific_drive_layout_prevents_result_mixing(tmp_path: Path) -> None:
    """带 workflow profile 的 Drive layout 必须按结果层级隔离 run 和 package 目录。"""
    validation_layout = build_drive_layout(
        str(tmp_path / "SSTW"),
        workflow_profile="validation_scale",
        notebook_role="generative_video_runtime",
    )
    pilot_layout = build_drive_layout(
        str(tmp_path / "SSTW"),
        workflow_profile="pilot_paper",
        notebook_role="generative_video_runtime",
    )

    assert validation_layout["drive_run_root"].endswith("/runs/generative_video_model_probe/validation_scale")
    assert validation_layout["drive_package_dir"].endswith("/packages/generative_video_model_probe/validation_scale")
    assert validation_layout["motion_threshold_artifact_run_root"].endswith("/runs/generative_video_model_probe/motion_calibration")
    assert validation_layout["runtime_profile"] == "validation_scale"
    assert pilot_layout["drive_run_root"].endswith("/runs/generative_video_model_probe/pilot_paper")
    assert pilot_layout["drive_package_dir"].endswith("/packages/generative_video_model_probe/pilot_paper")
    assert pilot_layout["motion_threshold_artifact_run_root"].endswith("/runs/generative_video_model_probe/motion_calibration")
    assert pilot_layout["runtime_profile"] == "pilot_paper"
    assert validation_layout["drive_run_root"] != pilot_layout["drive_run_root"]


@pytest.mark.quick
def test_profile_specific_layout_reuses_shared_motion_threshold_artifact(tmp_path: Path) -> None:
    """profile-specific run_root 隔离后, evaluation profile 仍必须能读取独立 calibration artifact。"""
    layout = build_drive_layout(
        str(tmp_path / "SSTW"),
        workflow_profile="validation_scale",
        notebook_role="generative_video_runtime",
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

    decision = validate_motion_threshold_ready_for_profile(layout, "validation_scale")

    assert decision["motion_threshold_reuse_required"] is True
    assert decision["motion_threshold_reuse_status"] == "ready"
    assert decision["motion_threshold_id"] == "motion_delta_calibrated_v1"


@pytest.mark.quick
def test_external_baseline_colab_preflight_uses_protocol_config(tmp_path: Path) -> None:
    """Notebook preflight 必须从 protocol config 读取 6 个现代 baseline 要求并写入 Drive artifact。"""
    layout = build_drive_layout(str(tmp_path / "SSTW"))
    commands = build_modern_baseline_command_env("validation_scale", {
        "videoshield": "python videoshield.py --output-json {output_json_path}",
        "sigmark": "python sigmark.py --output-json {output_json_path}",
    })

    decision = write_external_baseline_colab_preflight_decision(
        layout,
        profile="validation_scale",
        command_env=commands,
        require_modern_baseline_commands_for_paper_gate=True,
        run_external_baseline_source_clone=True,
        evidence_paths=[str(tmp_path / "baseline.log")],
    )

    assert decision["external_baseline_colab_preflight_decision"] == "FAIL"
    assert decision["external_baseline_colab_preflight_status"] == "commands_missing_for_paper_gate"
    assert decision["external_baseline_colab_preflight_required_env_var_count"] == 6
    assert decision["external_baseline_colab_preflight_configured_env_var_count"] == 2
    assert decision["external_baseline_colab_preflight_missing_env_var_count"] == 4
    assert set(decision["required_modern_external_baseline_adapter_names"]) == {
        "videoshield",
        "sigmark",
        "spdmark",
        "videomark",
        "vidsig",
        "videoseal",
    }
    artifact_path = Path(layout["drive_run_root"]) / "artifacts" / "external_baseline_colab_preflight_decision.json"
    assert artifact_path.exists()
    persisted = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert persisted["run_external_baseline_source_clone"] is True
    assert persisted["claim_support_status"] == "external_baseline_colab_preflight_only_not_claim_evidence"


@pytest.mark.quick
def test_external_baseline_colab_preflight_passes_when_all_commands_configured(tmp_path: Path) -> None:
    """6 个现代 baseline command 均配置时, paper gate preflight 应允许继续进入 GPU 流程。"""
    layout = build_drive_layout(str(tmp_path / "SSTW"))
    commands = build_modern_baseline_command_env("pilot_paper", {
        "videoshield": "python videoshield.py --output-json {output_json_path}",
        "sigmark": "python sigmark.py --output-json {output_json_path}",
        "spdmark": "python spdmark.py --output-json {output_json_path}",
        "videomark": "python videomark.py --output-json {output_json_path}",
        "vidsig": "python vidsig.py --output-json {output_json_path}",
        "videoseal": "python videoseal.py --output-json {output_json_path}",
    })

    decision = write_external_baseline_colab_preflight_decision(
        layout,
        profile="pilot_paper",
        command_env=commands,
        require_modern_baseline_commands_for_paper_gate=True,
        run_external_baseline_source_clone=False,
        evidence_paths=[],
    )

    assert decision["external_baseline_colab_preflight_decision"] == "PASS"
    assert decision["external_baseline_colab_preflight_status"] == "commands_configured_for_paper_gate"
    assert decision["external_baseline_colab_preflight_missing_env_vars"] == []
    assert set(decision["external_baseline_colab_preflight_configured_env_vars"]) == {
        "SSTW_VIDEOSHIELD_EVAL_COMMAND",
        "SSTW_SIGMARK_EVAL_COMMAND",
        "SSTW_SPDMARK_EVAL_COMMAND",
        "SSTW_VIDEOMARK_EVAL_COMMAND",
        "SSTW_VIDSIG_EVAL_COMMAND",
        "SSTW_VIDEOSEAL_EVAL_COMMAND",
    }


@pytest.mark.quick
def test_generative_video_drive_layout_uses_sstw_drive_root() -> None:
    """Colab workflow 的长期输出必须默认落盘到 MyDrive/SSTW 子目录。"""
    layout = build_drive_layout()
    assert layout["drive_project_root"] == "/content/drive/MyDrive/SSTW"
    assert layout["drive_dataset_root"].startswith("/content/drive/MyDrive/SSTW/datasets/")
    assert layout["drive_run_root"].startswith("/content/drive/MyDrive/SSTW/runs/")
    assert layout["drive_package_dir"].startswith("/content/drive/MyDrive/SSTW/packages/")
    assert layout["drive_log_dir"].startswith("/content/drive/MyDrive/SSTW/logs/")


@pytest.mark.quick
def test_generative_video_drive_packager_creates_archive_and_manifest(tmp_path: Path) -> None:
    """Drive packager 必须从已有 run outputs 生成 zip 和 package manifest。"""
    run_root = tmp_path / "runs" / "generative_video_runtime"
    package_dir = tmp_path / "packages"
    write_jsonl(run_root / "records" / "generation_records.jsonl", [{"generation_model_id": "model", "prompt_id": "prompt"}])
    write_json(run_root / "artifacts" / "generative_video_colab_runtime_decision.json", {"stage_id": "generative_video_runtime", "implementation_decision": "PASS", "mechanism_decision": "FAIL"})
    write_json(run_root / "artifacts" / "generation_manifest.json", {"artifact_id": "manifest"})
    write_json(run_root / "artifacts" / "small_scale_claim_pilot_gate_decision.json", {
        "pilot_gate_decision": "FAIL",
        "claim_support_status": "blocked_until_motion_threshold_calibration",
        "pilot_missing_requirement_count": 0,
    })
    write_json(run_root / "artifacts" / "small_scale_claim_pilot_matrix_decision.json", {
        "pilot_matrix_postprocess_decision": "PASS",
        "pilot_matrix_record_count": 480,
    })
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
    write_json(run_root / "artifacts" / "motion_threshold_calibration_decision.json", {
        "motion_threshold_calibration_decision": "INSUFFICIENT_SAMPLE",
        "motion_threshold_id": "motion_delta_heuristic_v1",
        "motion_threshold_source_split": "heuristic_precalibration",
        "motion_threshold_calibration_required": True,
    })
    write_json(run_root / "artifacts" / "validation_scale_gate_decision.json", {
        "validation_scale_gate_decision": "FAIL",
        "claim_support_status": "validation_scale_blocked",
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
    write_json(run_root / "artifacts" / "pilot_paper_gate_decision.json", {
        "pilot_paper_gate_decision": "PASS",
        "claim_support_status": "pilot_paper_calibrated_heldout_claim_ready",
        "paper_result_level": "pilot_paper",
        "paper_protocol_level": "paper_grade_protocol",
        "paper_protocol_difference_from_full_paper": "sample_scale_only",
        "pilot_paper_protocol_matches_full_paper": True,
        "pilot_paper_claim_allowed": True,
        "pilot_paper_missing_requirement_count": 0,
        "threshold_protocol": "calibration_split_to_frozen_threshold_to_heldout_test_split",
        "threshold_source_split": "calibration",
        "test_time_threshold_update_blocked": True,
        "tpr_at_fpr_01": 0.91,
        "calibration_negative_fpr_at_threshold": 0.008,
        "heldout_negative_fpr_at_threshold": 0.009,
        "observed_negative_fpr_at_threshold": 0.009,
        "calibration_negative_event_count": 1008,
        "heldout_test_negative_event_count": 1008,
        "heldout_negative_event_count": 1008,
        "heldout_attacked_positive_event_count": 252,
        "attacked_positive_event_count": 252,
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

    payload = package_generative_video_colab_run(run_root, package_dir, include_videos=False)
    archive_path = Path(payload["archive_path"])
    manifest_path = Path(payload["package_manifest_path"])
    assert archive_path.exists()
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["generation_manifest_status"] == "present"
    assert manifest["decision_summary"]["implementation_decision"] == "PASS"
    assert manifest["decision_summary"]["small_scale_pilot_gate_decision"] == "FAIL"
    assert manifest["decision_summary"]["small_scale_pilot_claim_support_status"] == "blocked_until_motion_threshold_calibration"
    assert manifest["decision_summary"]["small_scale_pilot_matrix_postprocess_decision"] == "PASS"
    assert manifest["decision_summary"]["small_scale_pilot_matrix_record_count"] == 480
    assert manifest["decision_summary"]["runtime_attack_decision"] == "PASS"
    assert manifest["decision_summary"]["runtime_attack_record_count"] == 48
    assert manifest["decision_summary"]["runtime_attack_ready_count"] == 48
    assert manifest["decision_summary"]["runtime_detection_decision"] == "PASS"
    assert manifest["decision_summary"]["runtime_detection_record_count"] == 48
    assert manifest["decision_summary"]["runtime_detection_ready_count"] == 48
    assert manifest["decision_summary"]["validation_scale_gate_decision"] == "FAIL"
    assert manifest["decision_summary"]["validation_scale_claim_support_status"] == "validation_scale_blocked"
    assert manifest["decision_summary"]["validation_missing_requirement_count"] == 5
    assert manifest["decision_summary"]["external_baseline_comparison_decision"] == "PASS"
    assert manifest["decision_summary"]["external_baseline_comparison_record_count"] == 96
    assert manifest["decision_summary"]["external_baseline_comparison_ready_count"] == 48
    assert manifest["decision_summary"]["external_baseline_measured_adapter_count"] == 2
    assert manifest["decision_summary"]["external_baseline_comparison_table_status"] == "ready"
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
    assert manifest["decision_summary"]["pilot_paper_gate_decision"] == "PASS"
    assert manifest["decision_summary"]["pilot_paper_claim_support_status"] == "pilot_paper_calibrated_heldout_claim_ready"
    assert manifest["decision_summary"]["pilot_paper_result_level"] == "pilot_paper"
    assert manifest["decision_summary"]["pilot_paper_protocol_level"] == "paper_grade_protocol"
    assert manifest["decision_summary"]["pilot_paper_protocol_difference_from_full_paper"] == "sample_scale_only"
    assert manifest["decision_summary"]["pilot_paper_protocol_matches_full_paper"] is True
    assert manifest["decision_summary"]["pilot_paper_claim_allowed"] is True
    assert manifest["decision_summary"]["pilot_paper_missing_requirement_count"] == 0
    assert manifest["decision_summary"]["pilot_paper_threshold_protocol"] == "calibration_split_to_frozen_threshold_to_heldout_test_split"
    assert manifest["decision_summary"]["pilot_paper_threshold_source_split"] == "calibration"
    assert manifest["decision_summary"]["pilot_paper_test_time_threshold_update_blocked"] is True
    assert manifest["decision_summary"]["pilot_paper_tpr_at_fpr_01"] == 0.91
    assert manifest["decision_summary"]["pilot_paper_calibration_negative_fpr_at_threshold"] == 0.008
    assert manifest["decision_summary"]["pilot_paper_heldout_negative_fpr_at_threshold"] == 0.009
    assert manifest["decision_summary"]["pilot_paper_observed_negative_fpr_at_threshold"] == 0.009
    assert manifest["decision_summary"]["pilot_paper_calibration_negative_event_count"] == 1008
    assert manifest["decision_summary"]["pilot_paper_heldout_test_negative_event_count"] == 1008
    assert manifest["decision_summary"]["pilot_paper_heldout_negative_event_count"] == 1008
    assert manifest["decision_summary"]["pilot_paper_heldout_attacked_positive_event_count"] == 252
    assert manifest["decision_summary"]["pilot_paper_attacked_positive_event_count"] == 252
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
def test_generative_video_drive_packager_reports_effective_mechanism_decision(tmp_path: Path) -> None:
    """package manifest 必须区分 runtime 原始机制判定与后处理后的有效机制判定。"""
    run_root = tmp_path / "runs" / "generative_video_runtime"
    package_dir = tmp_path / "packages"
    write_jsonl(run_root / "records" / "generation_records.jsonl", [{"generation_model_id": "model", "prompt_id": "prompt"}])
    write_json(run_root / "artifacts" / "generative_video_colab_runtime_decision.json", {
        "stage_id": "generative_video_runtime",
        "implementation_decision": "PASS",
        "mechanism_decision": "FAIL",
    })
    write_json(run_root / "artifacts" / "generative_video_mechanism_postprocess_decision.json", {
        "stage_id": "generative_video_mechanism_postprocess",
        "mechanism_postprocess_decision": "PASS",
        "mechanism_decision": "PASS",
        "details": {"formal_claim_status": "supported_by_governed_generation_records"},
    })
    write_json(run_root / "artifacts" / "small_scale_claim_pilot_gate_decision.json", {
        "pilot_gate_decision": "PASS",
        "claim_support_status": "supported_by_small_scale_claim_pilot_records",
        "pilot_missing_requirement_count": 0,
    })

    payload = package_generative_video_colab_run(run_root, package_dir, include_videos=False)

    manifest = json.loads(Path(payload["package_manifest_path"]).read_text(encoding="utf-8"))
    summary = manifest["decision_summary"]
    assert summary["runtime_mechanism_decision"] == "FAIL"
    assert summary["postprocess_mechanism_decision"] == "PASS"
    assert summary["mechanism_decision"] == "PASS"
    assert summary["effective_mechanism_decision"] == "PASS"
    assert summary["mechanism_decision_source"] == "small_scale_claim_pilot_gate"


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
    assert '"validation_scale": {' in runtime_text
    assert '"seed_suite_roles": ["main", "heldout_seed"]' in runtime_text
    assert 'default="pilot"' in runtime_text
    assert 'default=WAN21_PRIMARY_MODEL_ID' in runtime_text
