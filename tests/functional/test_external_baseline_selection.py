"""验证 B5 external baseline 推荐与 claim 约束。"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from main.external_baselines.baseline_registry import audit_external_baseline_records, build_external_baseline_records
from main.external_baselines.explicit_dtw_temporal_alignment import compute_dtw_alignment_cost
from main.external_baselines.frame_matching_temporal_registration import compute_registration_cost, match_frames
from experiments.generative_video_model_probe.external_baseline_runner import write_external_baseline_comparison_outputs, write_external_baseline_status_outputs
from external_baseline.official_bundle_generator import build_official_bundle_generation_plan
import external_baseline.official_resource_bootstrap as official_resource_bootstrap
from external_baseline.official_resource_bootstrap import bootstrap_official_resources
from external_baseline.official_result_bundle import build_official_result_bundle_preflight
from external_baseline.official_runtime_closure import (
    build_official_runtime_closure_requirements,
    load_official_runtime_closure_requirements,
)
from external_baseline.video_tensor_io import read_video_tchw_uint8, write_video_tchw
from external_baseline.videoseal_official_runtime import (
    ensure_videoseal_official_runtime_layout,
    inspect_videoseal_official_runtime_layout,
    videoseal_official_source_cwd,
)
from main.protocol.record_writer import read_jsonl, write_jsonl
from external_baseline.source_intake import build_source_intake_manifest, write_source_intake_artifacts


@pytest.mark.quick
def test_external_baseline_selection_keeps_modern_non_run_records() -> None:
    """外部 baseline 必须同时保留显式同步 control 和现代视频水印 non-run 记录。"""
    config_path = Path("configs/external_baselines/external_baselines.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    records = build_external_baseline_records(config_path)

    names = [record["external_baseline_name"] for record in records]
    assert "explicit_dtw_temporal_alignment" in names
    assert "explicit_frame_matching_temporal_registration" in names
    assert {"videoshield", "sigmark", "spdmark", "videomark", "vidsig", "videoseal"} <= set(names)
    excluded_related_work_names = {"riva" + "gan", "vid" + "stamp"}
    assert excluded_related_work_names.isdisjoint(names)
    assert config["selection_policy"]["claim_rule"]
    assert "key_conditioned_state_space_inference" in config["internal_mechanism_baselines"]
    assert all(record["external_baseline_result_used_for_claim"] is False for record in records)

    explicit_records = [record for record in records if record["external_baseline_layer"] == "explicit_synchronization_control"]
    modern_records = [record for record in records if record["external_baseline_layer"] == "modern_external_baseline"]
    assert len(explicit_records) == 2
    assert len(modern_records) >= 6
    assert all(record["external_baseline_runnable_status"] == "runnable" for record in explicit_records)
    assert all(record["external_baseline_runnable_status"] == "not_runnable" for record in modern_records)
    assert all(record["external_baseline_adapter_status"] == "adapter_ready_command_not_configured" for record in modern_records)
    assert all(record["external_baseline_claim_support_status"] == "governed_non_run_record_only" for record in modern_records)


@pytest.mark.quick
def test_external_baseline_status_audit_reports_modern_gap() -> None:
    """现代 baseline 已有 governed 状态记录, 但尚未达到主表比较 ready。"""
    records = build_external_baseline_records("configs/external_baselines/external_baselines.json")
    audit = audit_external_baseline_records(records)

    assert audit["external_baseline_status_decision"] == "PASS"
    assert audit["modern_external_baseline_status_records_ready"] is True
    assert audit["modern_external_baseline_record_count"] >= 6
    assert audit["modern_external_baseline_main_comparison_ready_count"] == 0
    assert audit["external_baseline_claim_support_status"] == "governed_status_records_only"


@pytest.mark.quick
def test_external_baseline_source_intake_writes_governed_manifests(tmp_path: Path) -> None:
    """source intake 必须写出源码、inspection、clone plan 和 table plan 治理文件。"""
    manifest = build_source_intake_manifest()
    assert manifest["external_baseline_source_intake_decision"] == "PASS"
    assert manifest["baseline_source_count"] >= 8
    assert manifest["modern_external_baseline_source_count"] >= 6
    modern_rows = [
        row for row in manifest["baseline_sources"]
        if row["baseline_id"] in {"videoshield", "sigmark", "spdmark", "videomark", "vidsig", "videoseal"}
    ]
    assert all(row["source_cloneable"] is True for row in modern_rows)
    assert {row["baseline_id"] for row in modern_rows} == {"videoshield", "sigmark", "spdmark", "videomark", "vidsig", "videoseal"}
    assert manifest["claim_support_status"] == "source_intake_manifest_only_not_claim_evidence"

    summary = write_source_intake_artifacts(tmp_path / "external_baseline_artifacts")
    assert Path(summary["source_intake_manifest_path"]).exists()
    assert Path(summary["source_inspection_manifest_path"]).exists()
    assert Path(summary["clone_results_manifest_path"]).exists()
    assert Path(summary["table_plan_path"]).exists()
    clone_manifest = json.loads(Path(summary["clone_results_manifest_path"]).read_text(encoding="utf-8"))
    clone_rows = {row["baseline_id"]: row for row in clone_manifest["clone_results"]}
    assert clone_rows["spdmark"]["planned_repository_url"] == "https://github.com/Samar-Fares/SPDMark"
    assert clone_rows["spdmark"]["target_repository_commit"] == "4d9a894384a8585734b493301fe9d1a4d6abd07c"


@pytest.mark.quick
def test_modern_baseline_colab_command_config_is_guidance_not_claim_evidence() -> None:
    """联网核验后的 command 配置只能作为 Colab 配置辅助, 不能自动变成正式结果。"""
    config_path = Path("configs/external_baselines/modern_baseline_colab_commands.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rows = {row["baseline_id"]: row for row in config["baseline_command_configs"]}

    required_ids = {"videoshield", "sigmark", "spdmark", "videomark", "vidsig", "videoseal"}
    assert set(rows) == required_ids
    assert config["command_configuration_boundary"]["template_file_auto_applied"] == "optional_when_SSTW_USE_MODERN_BASELINE_BRIDGE_COMMANDS_true"
    assert config["bridge_command_policy"]["bridge_module"] == "external_baseline.official_command_bridge"
    assert config["bridge_command_policy"]["official_inner_command_env_var_pattern"] == "SSTW_<BASELINE>_OFFICIAL_EVAL_COMMAND"
    assert "fail_closed" in config["formal_result_policy"]
    assert rows["videoshield"]["official_repository_url"] == "https://github.com/hurunyi/VideoShield"
    assert rows["sigmark"]["official_repository_url"] == "https://github.com/JeremyZhao1998/SIGMark-release"
    assert rows["spdmark"]["official_repository_url"] == "https://github.com/Samar-Fares/SPDMark"
    assert rows["videomark"]["official_repository_url"] == "https://github.com/KYRIE-LI11/VideoMark"
    assert rows["vidsig"]["official_repository_url"] == "https://github.com/hardenyu21/Video-Signature"
    assert rows["videoseal"]["official_repository_url"] == "https://github.com/facebookresearch/videoseal"
    for baseline_id, row in rows.items():
        assert row["external_baseline_command_env_var"] == f"SSTW_{baseline_id.upper()}_EVAL_COMMAND"
        assert row["official_baseline_command_env_var"] == f"SSTW_{baseline_id.upper()}_OFFICIAL_EVAL_COMMAND"
        assert row["source_verification_status"] == "git_ls_remote_head_verified_2026_06_25"
        if baseline_id in {"videomark", "videoshield"}:
            assert row["sstw_eval_command_template_status"] == "repository_bridge_ready_uses_project_owned_official_bundle_when_available"
        else:
            assert row["sstw_eval_command_template_status"] == "repository_bridge_ready_requires_official_eval_command"
        if baseline_id == "videomark":
            assert row["repository_official_eval_command_template_status"] == (
                "repository_official_wrapper_ready_with_project_owned_videomark_runtime_default"
            )
        elif baseline_id == "videoshield":
            assert row["repository_official_eval_command_template_status"] == (
                "repository_official_wrapper_ready_with_project_owned_videoshield_runtime_default"
            )
            assert row["project_owned_formal_reference_runner_module"] == "external_baseline.videoshield_official_runtime"
        elif baseline_id == "vidsig":
            assert row["repository_official_eval_command_template_status"] == (
                "repository_official_wrapper_ready_with_project_owned_vidsig_runtime_default"
            )
        else:
            assert row["repository_official_eval_command_template_status"] == (
                "repository_official_wrapper_ready_fail_closed_requires_official_source_and_artifacts"
            )
        assert row["repository_official_eval_adapter_module"] == f"external_baseline.official_eval_adapters.{baseline_id}"
        assert f"external_baseline.official_eval_adapters.{baseline_id}" in row["repository_official_eval_command_template"]
        assert "{official_output_json_path}" in row["repository_official_eval_command_template"]
        command = row["sstw_eval_command_template"]
        for token in config["required_command_format_tokens"]:
            assert "{" + token + "}" in command
        assert "external_baseline.official_command_bridge" in command
        assert row["score_output_contract"]["json_object_required"] is True
        assert "external_baseline_score" in row["score_output_contract"]["minimum_required_score_field_any_of"]


@pytest.mark.quick
def test_official_resource_requirements_define_auto_and_manual_boundaries() -> None:
    """官方资源要求配置必须区分可自动补齐项和必须手动提供的官方产物。"""
    config_path = Path("configs/external_baselines/official_resource_requirements.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rows = {row["baseline_id"]: row for row in config["resource_rows"]}

    assert set(rows) == {"videoshield", "sigmark", "spdmark", "videomark", "vidsig", "videoseal"}
    assert rows["videoseal"]["automatic_bundle_generation_supported_by_sstw"] is True
    assert rows["videoseal"]["colab_l4_auto_bundle_status"] == "auto_bundle_supported"
    assert rows["vidsig"]["automatic_bundle_generation_supported_by_sstw"] is True
    assert rows["vidsig"]["colab_l4_auto_bundle_status"] == "auto_bundle_supported_after_public_checkpoint_bootstrap"
    assert rows["videoshield"]["automatic_bundle_generation_supported_by_sstw"] is True
    assert rows["videoshield"]["colab_l4_auto_bundle_status"] == "auto_bundle_supported_after_hf_model_download_and_colab_gpu_success"
    assert rows["videoshield"]["project_owned_runner_module"] == "external_baseline.videoshield_official_runtime"
    assert rows["spdmark"]["automatic_bundle_generation_supported_by_sstw"] is False
    assert rows["spdmark"]["colab_l4_auto_bundle_status"] == "blocked_by_missing_public_trained_weights"
    assert rows["sigmark"]["colab_l4_auto_bundle_status"] == "blocked_by_official_gpu_memory_requirement"
    assert "sstw_proxy" not in json.dumps(config, ensure_ascii=False)


@pytest.mark.quick
def test_official_runtime_closure_requirements_are_first_class_colab_config() -> None:
    """真实运行闭合要求必须有独立配置和每个 baseline 的 requirements 文件。"""
    config = load_official_runtime_closure_requirements()
    rows = {row["baseline_id"]: row for row in config["baseline_runtime_requirements"]}

    assert config["config_kind"] == "modern_external_baseline_official_runtime_closure_requirements"
    assert set(rows) == {"videoshield", "sigmark", "spdmark", "videomark", "vidsig", "videoseal"}
    assert config["self_containment_rule"].startswith("external baseline 必须在项目内 clone")
    assert rows["videoseal"]["automatic_bundle_generation_supported_by_sstw"] is True
    assert rows["videoseal"]["colab_default_can_attempt_without_user_files"] is True
    assert rows["vidsig"]["automatic_bundle_generation_supported_by_sstw"] is True
    assert rows["vidsig"]["colab_default_can_attempt_without_user_files"] is True
    assert rows["vidsig"]["project_owned_vidsig_runner_module"] == "external_baseline.vidsig_official_runtime"
    assert rows["videoshield"]["automatic_bundle_generation_supported_by_sstw"] is True
    assert rows["videoshield"]["colab_default_can_attempt_without_user_files"] is True
    assert rows["videoshield"]["project_owned_videoshield_runner_module"] == "external_baseline.videoshield_official_runtime"
    assert rows["videoshield"]["resource_env_vars"] == []
    videoseal_requirements = Path(rows["videoseal"]["requirements_file"]).read_text(encoding="utf-8")
    assert "ffmpeg-python" in videoseal_requirements
    assert "git+https://github.com/facebookresearch/videoseal" not in videoseal_requirements
    vidsig_requirements = Path(rows["vidsig"]["requirements_file"]).read_text(encoding="utf-8")
    assert "augly==1.0.0" in vidsig_requirements
    assert "omegaconf" in vidsig_requirements
    for baseline_id, row in rows.items():
        assert row["external_supplemental_result_bundle_allowed"] is False
        assert row["official_baseline_command_env_var"] == f"SSTW_{baseline_id.upper()}_OFFICIAL_EVAL_COMMAND"
        assert row["external_baseline_command_env_var"] == f"SSTW_{baseline_id.upper()}_EVAL_COMMAND"
        assert row["native_command_env_var"] == f"SSTW_{baseline_id.upper()}_NATIVE_EVAL_COMMAND"
        requirements_file = Path(row["requirements_file"])
        assert requirements_file.exists()
        requirement_lines = [
            line.strip()
            for line in requirements_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        assert "torch" not in requirement_lines
        assert "torchvision" not in requirement_lines


@pytest.mark.quick
def test_videoseal_official_runtime_layout_requires_source_root_config(tmp_path: Path) -> None:
    """VideoSeal 官方源码加载必须以源码根目录作为临时 cwd 解析官方配置。"""
    source_dir = tmp_path / "videoseal_source"
    (source_dir / "videoseal").mkdir(parents=True)
    (source_dir / "configs").mkdir()
    (source_dir / "configs" / "attenuation.yaml").write_text("jnd_1_1: {}\n", encoding="utf-8")

    audit = ensure_videoseal_official_runtime_layout(source_dir)

    assert audit["layout_decision"] == "PASS"
    assert audit["layout_status"] == "official_source_root_config_ready"
    previous_cwd = Path.cwd()
    with videoseal_official_source_cwd(source_dir):
        assert Path.cwd() == source_dir.resolve()
        assert Path("configs/attenuation.yaml").is_file()
    assert Path.cwd() == previous_cwd


@pytest.mark.quick
def test_videoseal_official_runtime_layout_fails_closed_without_official_config(tmp_path: Path) -> None:
    """缺少官方 attenuation.yaml 时必须阻断, 不能生成临时配置伪装为正式 baseline。"""
    source_dir = tmp_path / "videoseal_source"
    (source_dir / "videoseal").mkdir(parents=True)

    audit = inspect_videoseal_official_runtime_layout(source_dir)

    assert audit["layout_decision"] == "FAIL"
    assert audit["layout_status"] == "official_required_config_missing"
    with pytest.raises(FileNotFoundError, match="videoseal_official_config_missing"):
        ensure_videoseal_official_runtime_layout(source_dir)
    assert not (source_dir / "videoseal" / "configs" / "attenuation.yaml").exists()


@pytest.mark.quick
def test_external_baseline_video_tensor_io_uses_imageio_backend(tmp_path: Path) -> None:
    """VideoSeal bundle I/O 必须避开 Colab 中不稳定的 torchvision 视频接口。"""
    import torch

    video_path = tmp_path / "sample.mp4"
    video = torch.zeros((3, 3, 16, 16), dtype=torch.float32)
    video[1, 0] = 1.0

    write_info = write_video_tchw(video_path, video, fps=8.0)
    decoded, read_info = read_video_tchw_uint8(video_path)

    assert video_path.exists()
    assert write_info["video_io_backend"] == "imageio_v3"
    assert read_info["video_io_backend"] == "imageio_v3"
    assert decoded.ndim == 4
    assert decoded.shape[1] == 3
    assert decoded.shape[0] >= 1


@pytest.mark.quick
def test_official_runtime_closure_preflight_fails_with_actionable_missing_inputs(tmp_path: Path) -> None:
    """缺少真实 runtime 输入时, 新预检必须明确失败而不是伪装为可运行。"""
    run_root = tmp_path / "runs" / "generative_video_model_probe" / "validation_scale"

    audit = build_official_runtime_closure_requirements(
        run_root,
        repo_root=Path("."),
        resource_root=tmp_path / "resources" / "external_baseline",
        official_result_bundle_root=tmp_path / "external_baseline_official_result_bundles" / "validation_scale",
    )

    assert audit["official_runtime_closure_decision"] == "FAIL"
    assert audit["runtime_input_audit"]["runtime_input_ready"] is False
    assert "records/runtime_detection_records.jsonl" in audit["runtime_input_audit"]["missing_runtime_requirements"]
    assert audit["runtime_closure_blocked_count"] == 6
    assert set(audit["runtime_closure_blocked_baselines"]) == {
        "videoshield",
        "sigmark",
        "spdmark",
        "videomark",
        "vidsig",
        "videoseal",
    }


@pytest.mark.quick
def test_official_runtime_closure_preflight_binds_existing_default_drive_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """默认 Drive 资源路径存在时, 预检应自动给 Notebook 父进程提供环境变量更新。"""
    run_root = tmp_path / "runs" / "generative_video_model_probe" / "validation_scale"
    _write_external_baseline_runtime_fixture(run_root)
    resource_root = tmp_path / "resources" / "external_baseline"
    decoder = resource_root / "vidsig" / "nested" / "dec_48b_whit.torchscript.pt"
    decoder.parent.mkdir(parents=True)
    decoder.write_bytes(b"vidsig-decoder")
    for env_name in ("SSTW_VIDSIG_MSG_DECODER_PATH", "SSTW_VIDSIG_VAE_CHECKPOINT_PATH"):
        monkeypatch.delenv(env_name, raising=False)

    audit = build_official_runtime_closure_requirements(
        run_root,
        repo_root=Path("."),
        resource_root=resource_root,
        official_result_bundle_root=tmp_path / "external_baseline_official_result_bundles" / "validation_scale",
        baseline_id="vidsig",
    )

    assert audit["baseline_count"] == 1
    assert audit["environment_updates"]["SSTW_VIDSIG_MSG_DECODER_PATH"] == str(decoder)
    vidsig_row = audit["baseline_runtime_rows"][0]
    assert vidsig_row["baseline_id"] == "vidsig"
    assert vidsig_row["resource_requirement"]["resource_requirements"][0]["effective_resource_path_exists"] is True


@pytest.mark.quick
def test_sigmark_runtime_closure_binds_prefixed_official_bit_accuracy_npz(tmp_path: Path) -> None:
    """SIGMark 官方输出文件名带实验前缀时, 预检仍应自动绑定到环境变量。

    SIGMark 官方 `main.py --mode extract` 写出的文件名不是固定的
    `bit_accuracy.npz`, 而是 `<setting>-bit_accuracy.npz`。该测试防止
    Colab 已经完成官方提取后, runtime closure 因 glob 过窄而误报资源缺失。
    """

    run_root = tmp_path / "runs" / "generative_video_model_probe" / "validation_scale"
    _write_external_baseline_runtime_fixture(run_root)
    resource_root = tmp_path / "resources" / "external_baseline"
    bit_accuracy_npz = (
        resource_root
        / "sigmark"
        / "official_outputs"
        / "HunyuanVideo-I2V-community-VBench2_aug-512x512-65frams-sigmark-128x4bits-bit_accuracy.npz"
    )
    bit_accuracy_npz.parent.mkdir(parents=True)
    bit_accuracy_npz.write_bytes(b"npz-placeholder")

    audit = build_official_runtime_closure_requirements(
        run_root,
        repo_root=Path("."),
        resource_root=resource_root,
        official_result_bundle_root=tmp_path / "external_baseline_official_result_bundles" / "validation_scale",
        baseline_id="sigmark",
    )

    assert audit["baseline_count"] == 1
    assert audit["environment_updates"]["SSTW_SIGMARK_BIT_ACCURACY_NPZ"] == str(bit_accuracy_npz)
    sigmark_row = audit["baseline_runtime_rows"][0]
    assert sigmark_row["baseline_id"] == "sigmark"
    assert sigmark_row["resource_requirement"]["resource_requirements"][0]["effective_resource_path_exists"] is True


@pytest.mark.quick
def test_official_bundle_generation_plan_is_fail_closed_about_auto_blocked_baselines(tmp_path: Path) -> None:
    """official bundle generator 只能自动生成官方可支持的方法, 不能把缺资源 baseline 伪装为可自动生成。"""
    run_root = tmp_path / "generative_video_runtime"
    _write_external_baseline_runtime_fixture(run_root)
    plan = build_official_bundle_generation_plan(run_root, tmp_path / "official_bundles")

    assert plan["runtime_comparison_unit_count"] == 2
    assert plan["baseline_count"] == 6
    assert plan["auto_supported_baselines"] == ["videoshield", "videomark", "vidsig", "videoseal"]
    assert "spdmark" in plan["auto_blocked_baselines"]
    assert "sigmark" in plan["auto_blocked_baselines"]
    assert plan["auto_blocked_baseline_count"] == 2
    spdmark_row = next(row for row in plan["plan_rows"] if row["baseline_id"] == "spdmark")
    assert spdmark_row["automatic_bundle_generation_supported_by_sstw"] is False
    assert spdmark_row["resource_blocker"]


@pytest.mark.quick
def test_official_resource_bootstrap_writes_repair_artifact_without_network(tmp_path: Path) -> None:
    """bootstrap 在禁止网络时仍必须落盘可审计修复计划和环境变量更新。"""
    run_root = tmp_path / "runs" / "validation_scale"
    decision = bootstrap_official_resources(
        run_root,
        resource_root=tmp_path / "resources" / "external_baseline",
        allow_network=False,
        source_root=tmp_path / "external_baseline" / "primary",
    )

    artifact_path = run_root / "artifacts" / "external_baseline_official_resource_bootstrap_decision.json"
    assert artifact_path.exists()
    assert decision["official_resource_bootstrap_decision"] == "PASS"
    assert "videoseal" in decision["ready_baselines"]
    assert "videoshield" in decision["ready_baselines"]
    assert "spdmark" in decision["manual_official_resource_required_baselines"]
    assert decision["manual_official_resource_required_count"] >= 3
    assert decision["strict_gate_auto_resource_closure"] is False
    assert decision["environment_updates"]["SSTW_EXTERNAL_BASELINE_RESOURCE_ROOT"].endswith("external_baseline")


@pytest.mark.quick
def test_official_resource_bootstrap_preserves_colab_torch_stack(tmp_path: Path) -> None:
    """自动 bootstrap 不应通过 pip 安装 torch / torchvision 或官方 git 包破坏 Colab 运行栈。"""
    videoseal_row = official_resource_bootstrap.bootstrap_videoseal(
        tmp_path / "resources" / "external_baseline",
        allow_network=False,
        source_root=tmp_path / "external_baseline" / "primary",
    )
    vidsig_row = official_resource_bootstrap.bootstrap_vidsig(
        tmp_path / "resources" / "external_baseline",
        allow_network=False,
    )
    install_targets = [
        item["install_target"]
        for row in (videoseal_row, vidsig_row)
        for item in row["install_results"]
    ]

    assert videoseal_row["colab_torch_stack_policy"] == "preserve_preinstalled_torch_and_torchvision"
    assert vidsig_row["colab_torch_stack_policy"] == "preserve_preinstalled_torch_and_torchvision"
    assert "augly==1.0.0" in install_targets
    assert "omegaconf" in install_targets
    assert "torch" not in install_targets
    assert "torchvision" not in install_targets
    assert not any(str(target).startswith("git+https://github.com/facebookresearch/videoseal") for target in install_targets)


@pytest.mark.quick
def test_vidsig_bootstrap_uses_gdown_positional_file_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """gdown 新版 Colab 环境不再接受 `--id`, bootstrap 必须使用位置参数。"""
    commands: list[list[str]] = []

    monkeypatch.setattr(
        official_resource_bootstrap,
        "_ensure_gdown",
        lambda: {"tool": "gdown", "status": "already_available"},
    )
    monkeypatch.setattr(
        official_resource_bootstrap,
        "_pip_install_target",
        lambda target, *, allow_network, timeout_sec=1800: {
            "install_target": target,
            "install_status": "installed" if allow_network else "planned_network_disabled",
        },
    )

    def fake_run_command(command: list[str], *, timeout_sec: int = 1800) -> dict[str, object]:
        commands.append(command)
        return {"command": command, "return_code": 2, "stdout_tail": "", "stderr_tail": "test"}

    monkeypatch.setattr(official_resource_bootstrap, "_run_command", fake_run_command)

    row = official_resource_bootstrap.bootstrap_vidsig(tmp_path / "resources" / "external_baseline", allow_network=True)

    assert row["bootstrap_status"] == "manual_official_resource_required"
    assert commands
    assert "--id" not in commands[0]
    assert official_resource_bootstrap.VIDSIG_GOOGLE_DRIVE_FILE_ID in commands[0]


@pytest.mark.quick
def test_external_baseline_runner_writes_governed_status_outputs(tmp_path: Path) -> None:
    """外部 baseline runner 必须写出 records、table、decision 和 report。"""
    run_root = tmp_path / "generative_video_runtime"
    audit = write_external_baseline_status_outputs(run_root)
    records = read_jsonl(run_root / "records" / "external_baseline_records.jsonl")

    assert audit["external_baseline_status_decision"] == "PASS"
    assert len(records) == audit["external_baseline_record_count"]
    assert all("external_baseline_adapter_status" in record for record in records)
    assert all("claim_support_status" in record for record in records)
    assert (run_root / "tables" / "external_baseline_status_table.csv").exists()
    assert (run_root / "artifacts" / "external_baseline_status_decision.json").exists()
    assert (run_root / "artifacts" / "external_baseline_intake_manifest.json").exists()
    assert (run_root / "artifacts" / "external_baseline_source_inspection.json").exists()
    assert (run_root / "artifacts" / "external_baseline_clone_results.json").exists()
    assert (run_root / "reports" / "external_baseline_status_report.md").exists()


@pytest.mark.quick
def test_explicit_synchronization_adapters_run_on_small_sequences() -> None:
    """两个 external synchronization control adapter 必须能在轻量 embedding 序列上运行。"""
    reference = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
    observed = [[0.0, 0.0], [1.1, 0.0], [2.0, 0.0]]

    assert compute_dtw_alignment_cost(reference, observed) >= 0.0
    assert compute_registration_cost(reference, observed) >= 0.0

    matches = match_frames(reference, observed)
    assert [item["reference_index"] for item in matches] == [0, 1, 2]



def _write_external_baseline_runtime_fixture(run_root: Path) -> None:
    """写出 external_baseline adapter 可消费的最小 runtime detection 与 trajectory fixture。"""
    trajectory_records = []
    for step_index in range(4):
        trajectory_records.append({
            "trajectory_trace_id": "trace_0",
            "trajectory_step_index": step_index,
            "latent_norm": 4.0 - step_index * 0.4,
            "latent_mean": 0.1 * step_index,
            "latent_std": 0.2 + step_index * 0.05,
        })
    write_jsonl(run_root / "records" / "trajectory_trace.jsonl", trajectory_records)
    source_video_path = run_root / "videos" / "source.mp4"
    attacked_video_path = run_root / "attacks" / "attacked.mp4"
    source_video_path.parent.mkdir(parents=True, exist_ok=True)
    attacked_video_path.parent.mkdir(parents=True, exist_ok=True)
    source_video_path.write_bytes(b"source-video-placeholder")
    attacked_video_path.write_bytes(b"attacked-video-placeholder")
    write_jsonl(run_root / "records" / "runtime_detection_records.jsonl", [
        {
            "runtime_detection_status": "ready",
            "generation_model_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            "prompt_id": "prompt_0",
            "seed_id": "seed_0",
            "trajectory_trace_id": "trace_0",
            "attack_name": "video_compression_runtime",
            "source_video_path": str(source_video_path),
            "attacked_video_path": str(attacked_video_path),
            "sample_role": "generated_positive",
            "source_frame_count": 4,
            "attacked_frame_count": 4,
            "attacked_video_decoded_frame_count": 4,
            "S_runtime_attack_detection": 0.82,
        },
        {
            "runtime_detection_status": "ready",
            "generation_model_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            "prompt_id": "prompt_0",
            "seed_id": "seed_0",
            "trajectory_trace_id": "trace_0",
            "attack_name": "frame_rate_resampling_runtime",
            "source_video_path": str(source_video_path),
            "attacked_video_path": str(attacked_video_path),
            "sample_role": "generated_positive",
            "source_frame_count": 4,
            "attacked_frame_count": 2,
            "attacked_video_decoded_frame_count": 2,
            "S_runtime_attack_detection": 0.71,
        },
    ])


@pytest.mark.quick
def test_external_baseline_comparison_runner_uses_external_baseline_adapters(tmp_path: Path) -> None:
    """baseline comparison 必须通过 external_baseline/ adapter 产出 records、table、decision 和 report。"""
    run_root = tmp_path / "generative_video_runtime"
    _write_external_baseline_runtime_fixture(run_root)

    audit = write_external_baseline_comparison_outputs(run_root)
    records = read_jsonl(run_root / "records" / "external_baseline_score_records.jsonl")

    assert audit["external_baseline_comparison_decision"] == "PASS"
    assert audit["external_baseline_measured_adapter_count"] == 2
    assert "explicit_dtw_temporal_alignment" in audit["external_baseline_measured_adapter_names"]
    assert "explicit_frame_matching_temporal_registration" in audit["external_baseline_measured_adapter_names"]
    assert any(record["external_baseline_adapter_path"].startswith("external_baseline/") for record in records)
    assert all(record["external_baseline_result_used_for_claim"] is False for record in records)
    assert all(record.get("S_final") is None for record in records)
    assert (run_root / "tables" / "external_baseline_comparison_table.csv").exists()
    assert (run_root / "artifacts" / "external_baseline_comparison_decision.json").exists()
    assert (run_root / "artifacts" / "external_baseline_execution_manifest.json").exists()
    assert (run_root / "reports" / "external_baseline_comparison_report.md").exists()


@pytest.mark.quick
def test_modern_external_baseline_formal_command_adapters_write_measured_records(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """现代视频水印 baseline 必须通过正式 command adapter 产出 measured_formal records。"""
    run_root = tmp_path / "generative_video_runtime"
    _write_external_baseline_runtime_fixture(run_root)
    fake_adapter = tmp_path / "fake_modern_baseline_eval.py"
    fake_adapter.write_text(
        "import argparse, json\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--output-json', required=True)\n"
        "parser.add_argument('--source-video')\n"
        "parser.add_argument('--attacked-video')\n"
        "parser.add_argument('--attack-name')\n"
        "args = parser.parse_args()\n"
        "json.dump({'external_baseline_score': 0.37, 'detected': True, 'bit_accuracy': 0.91, 'threshold': 0.5, "
        "'external_baseline_source_video_path': 'official/source.mp4', "
        "'external_baseline_attacked_video_path': 'official/attacked.mp4', "
        "'external_baseline_generation_model_id': 'official_baseline_model', "
        "'external_baseline_official_execution_mode': 'official_result_bundle'}, open(args.output_json, 'w', encoding='utf-8'))\n",
        encoding="utf-8",
    )
    command = f'{sys.executable} {fake_adapter} --source-video {{source_video_path}} --attacked-video {{attacked_video_path}} --attack-name {{attack_name}} --output-json {{output_json_path}}'
    for env_var in (
        "SSTW_VIDEOSHIELD_EVAL_COMMAND",
        "SSTW_SIGMARK_EVAL_COMMAND",
        "SSTW_SPDMARK_EVAL_COMMAND",
        "SSTW_VIDEOMARK_EVAL_COMMAND",
        "SSTW_VIDSIG_EVAL_COMMAND",
        "SSTW_VIDEOSEAL_EVAL_COMMAND",
    ):
        monkeypatch.setenv(env_var, command)

    audit = write_external_baseline_comparison_outputs(run_root)
    records = read_jsonl(run_root / "records" / "external_baseline_score_records.jsonl")
    formal_records = [record for record in records if record.get("metric_status") == "measured_formal"]

    assert audit["external_baseline_comparison_decision"] == "PASS"
    assert audit["external_baseline_measured_adapter_count"] == 8
    assert audit["modern_external_baseline_formal_measured_adapter_count"] == 6
    assert set(audit["modern_external_baseline_formal_measured_adapter_names"]) == {
        "videoshield",
        "sigmark",
        "spdmark",
        "videomark",
        "vidsig",
        "videoseal",
    }
    assert formal_records
    assert all(record["external_baseline_result_used_for_claim"] is True for record in formal_records)
    assert all(Path(record["external_baseline_official_output_path"]).exists() for record in formal_records)
    assert all(Path(record["external_baseline_official_stdout_path"]).exists() for record in formal_records)
    assert all(Path(record["external_baseline_official_stderr_path"]).exists() for record in formal_records)
    assert all(Path(record["external_baseline_official_command_manifest_path"]).exists() for record in formal_records)
    assert all(record["external_baseline_source_video_path"] == "official/source.mp4" for record in formal_records)
    assert all(record["external_baseline_attacked_video_path"] == "official/attacked.mp4" for record in formal_records)
    assert all(record["external_baseline_generation_model_id"] == "official_baseline_model" for record in formal_records)
    assert all(record.get("S_final") is None for record in records)
    execution_manifest = json.loads((run_root / "artifacts" / "external_baseline_execution_manifest.json").read_text(encoding="utf-8"))
    assert execution_manifest["modern_external_baseline_formal_measured_adapter_count"] == 6
    assert execution_manifest["formal_evidence_status"] == "evidence_paths_bound"
    assert execution_manifest["evidence_path_count"] >= len(formal_records)


@pytest.mark.quick
def test_modern_external_baseline_bridge_commands_require_real_official_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """repository bridge 只能把官方命令输出归一化为 measured_formal records。"""
    run_root = tmp_path / "generative_video_runtime"
    _write_external_baseline_runtime_fixture(run_root)
    fake_official = tmp_path / "fake_official_detector.py"
    fake_official.write_text(
        "import argparse, json\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--official-output-json', required=True)\n"
        "parser.add_argument('--source-video')\n"
        "parser.add_argument('--attacked-video')\n"
        "parser.add_argument('--attack-name')\n"
        "args = parser.parse_args()\n"
        "json.dump({'score': 0.42, 'detected': True, 'bit_accuracy': 0.88}, open(args.official_output_json, 'w', encoding='utf-8'))\n",
        encoding="utf-8",
    )
    official_command = (
        f"{sys.executable} {fake_official} "
        "--source-video {source_video_path} "
        "--attacked-video {attacked_video_path} "
        "--attack-name {attack_name} "
        "--official-output-json {official_output_json_path}"
    )
    for baseline_id in ("videoshield", "sigmark", "spdmark", "videomark", "vidsig", "videoseal"):
        official_source_dir = tmp_path / "official_sources" / baseline_id
        official_source_dir.mkdir(parents=True)
        monkeypatch.setenv(f"SSTW_{baseline_id.upper()}_OFFICIAL_EVAL_COMMAND", official_command)
        bridge_command = (
            f"{sys.executable} -m external_baseline.official_command_bridge "
            f"--baseline-id {baseline_id} "
            f"--official-source-dir {official_source_dir} "
            "--source-video {source_video_path} "
            "--attacked-video {attacked_video_path} "
            "--attack-name {attack_name} "
            "--output-json {output_json_path} "
            "--run-root {run_root} "
            "--prompt-id {prompt_id} "
            "--seed-id {seed_id} "
            "--trajectory-trace-id {trajectory_trace_id}"
        )
        monkeypatch.setenv(f"SSTW_{baseline_id.upper()}_EVAL_COMMAND", bridge_command)

    audit = write_external_baseline_comparison_outputs(run_root)
    records = read_jsonl(run_root / "records" / "external_baseline_score_records.jsonl")
    formal_records = [record for record in records if record.get("metric_status") == "measured_formal"]

    assert audit["modern_external_baseline_formal_measured_adapter_count"] == 6
    assert set(audit["modern_external_baseline_formal_measured_adapter_names"]) == {
        "videoshield",
        "sigmark",
        "spdmark",
        "videomark",
        "vidsig",
        "videoseal",
    }
    assert formal_records
    assert all(record["external_baseline_score"] == 0.42 for record in formal_records)
    raw_paths = [
        Path(record["external_baseline_official_output_path"]).with_name(
            Path(record["external_baseline_official_output_path"]).stem + "_official_raw.json"
        )
        for record in formal_records
    ]
    assert all(path.exists() for path in raw_paths)


@pytest.mark.quick
def test_official_result_bundle_preflight_requires_all_modern_baseline_units(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """官方结果包必须覆盖全部 modern baseline 与 runtime comparison unit。"""
    run_root = tmp_path / "generative_video_runtime"
    _write_external_baseline_runtime_fixture(run_root)
    bundle_root = tmp_path / "official_baseline_bundle"
    monkeypatch.setenv("SSTW_EXTERNAL_BASELINE_OFFICIAL_RESULT_BUNDLE_ROOT", str(bundle_root))
    monkeypatch.delenv("SSTW_VIDEOSHIELD_RESULT_JSON", raising=False)
    monkeypatch.delenv("SSTW_SIGMARK_BIT_ACCURACY_NPZ", raising=False)
    monkeypatch.delenv("SSTW_SPDMARK_EXTRACTOR_PATH", raising=False)
    monkeypatch.delenv("SSTW_SPDMARK_GT_BITS_PATH", raising=False)
    monkeypatch.delenv("SSTW_VIDEOMARK_TEMPORAL_RESULTS_JSON", raising=False)
    monkeypatch.delenv("SSTW_VIDSIG_MSG_DECODER_PATH", raising=False)
    for baseline_id in ("VIDEOSHIELD", "SIGMARK", "SPDMARK", "VIDEOMARK", "VIDSIG", "VIDEOSEAL"):
        monkeypatch.delenv(f"SSTW_{baseline_id}_NATIVE_EVAL_COMMAND", raising=False)

    records = read_jsonl(run_root / "records" / "runtime_detection_records.jsonl")
    for baseline_id in ("videoshield", "sigmark", "spdmark", "videomark", "vidsig", "videoseal"):
        manifest_path = bundle_root / baseline_id / "official_reference_execution_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps({
                "manifest_kind": "test_repository_generated_official_bundle_manifest",
                "baseline_id": baseline_id,
                "claim_support_status": "test_fixture_only",
            }),
            encoding="utf-8",
        )
        for record in records:
            output = (
                bundle_root
                / baseline_id
                / "records"
                / f"{record['prompt_id']}__{record['seed_id']}__{record['attack_name']}.json"
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps({
                    "external_baseline_score": 0.61,
                    "detected": True,
                    "bit_accuracy": 0.93,
                    "official_result_provenance": "repository_generated_from_third_party_official_code",
                    "official_execution_manifest_path": str(manifest_path),
                    "external_baseline_source_video_path": f"{baseline_id}/source.mp4",
                    "external_baseline_attacked_video_path": f"{baseline_id}/attacked.mp4",
                }),
                encoding="utf-8",
            )

    audit = build_official_result_bundle_preflight(run_root)
    assert audit["official_result_bundle_preflight_decision"] == "PASS"
    assert audit["expected_bundle_result_count"] == 12
    assert audit["present_bundle_result_count"] == 12
    assert audit["strict_missing_baselines"] == []


@pytest.mark.quick
def test_official_result_bundle_preflight_fails_when_non_runtime_baseline_bundle_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """除 VideoSeal 这类可直接尝试官方 API 的方法外, 缺 bundle 必须提前暴露。"""
    run_root = tmp_path / "generative_video_runtime"
    _write_external_baseline_runtime_fixture(run_root)
    monkeypatch.setenv("SSTW_EXTERNAL_BASELINE_OFFICIAL_RESULT_BUNDLE_ROOT", str(tmp_path / "missing_bundle"))
    for env_name in (
        "SSTW_VIDEOSHIELD_RESULT_JSON",
        "SSTW_SIGMARK_BIT_ACCURACY_NPZ",
        "SSTW_SPDMARK_EXTRACTOR_PATH",
        "SSTW_SPDMARK_GT_BITS_PATH",
        "SSTW_VIDEOMARK_TEMPORAL_RESULTS_JSON",
        "SSTW_VIDSIG_MSG_DECODER_PATH",
        "SSTW_VIDSIG_VAE_CHECKPOINT_PATH",
    ):
        monkeypatch.delenv(env_name, raising=False)
    for baseline_id in ("VIDEOSHIELD", "SIGMARK", "SPDMARK", "VIDEOMARK", "VIDSIG", "VIDEOSEAL"):
        monkeypatch.delenv(f"SSTW_{baseline_id}_NATIVE_EVAL_COMMAND", raising=False)

    audit = build_official_result_bundle_preflight(run_root)
    assert audit["official_result_bundle_preflight_decision"] == "FAIL"
    assert "spdmark" in audit["strict_missing_baselines"]
    assert audit["missing_bundle_examples"]


@pytest.mark.quick
def test_official_result_bundle_preflight_requires_vidsig_official_bundle_not_only_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VidSig 不能只靠 checkpoint 资源被误判为已经具备正式结果包。"""
    run_root = tmp_path / "generative_video_runtime"
    _write_external_baseline_runtime_fixture(run_root)
    monkeypatch.setenv("SSTW_EXTERNAL_BASELINE_OFFICIAL_RESULT_BUNDLE_ROOT", str(tmp_path / "missing_bundle"))
    monkeypatch.delenv("SSTW_VIDSIG_NATIVE_EVAL_COMMAND", raising=False)

    decoder = tmp_path / "resources" / "vidsig" / "ckpts" / "msg_decoder" / "dec_48b_whit.torchscript.pt"
    decoder.parent.mkdir(parents=True)
    decoder.write_bytes(b"decoder")
    monkeypatch.setenv("SSTW_VIDSIG_MSG_DECODER_PATH", str(decoder))
    vae = tmp_path / "resources" / "vidsig" / "ckpts" / "vae" / "modelscope" / "checkpoint.pth"
    vae.parent.mkdir(parents=True)
    vae.write_bytes(b"vae")
    monkeypatch.setenv("SSTW_VIDSIG_VAE_CHECKPOINT_PATH", str(vae))

    audit = build_official_result_bundle_preflight(run_root, baseline_ids=("vidsig",))
    row = audit["baseline_resource_rows"][0]
    assert row["runtime_resource_ready"] is False
    assert row["runtime_resource_mode"] == "vidsig_requires_project_owned_generate_ms_official_bundle_or_native_command"
    assert audit["strict_missing_baselines"] == ["vidsig"]


@pytest.mark.quick
def test_official_result_bundle_preflight_rejects_external_supplement_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """旧外部补交式 bundle 不能作为 measured_formal 的项目内缓存输入。"""
    run_root = tmp_path / "generative_video_runtime"
    _write_external_baseline_runtime_fixture(run_root)
    bundle_root = tmp_path / "official_baseline_bundle"
    monkeypatch.setenv("SSTW_EXTERNAL_BASELINE_OFFICIAL_RESULT_BUNDLE_ROOT", str(bundle_root))
    for env_name in (
        "SSTW_VIDEOSHIELD_RESULT_JSON",
        "SSTW_SIGMARK_BIT_ACCURACY_NPZ",
        "SSTW_SPDMARK_EXTRACTOR_PATH",
        "SSTW_SPDMARK_GT_BITS_PATH",
        "SSTW_VIDEOMARK_TEMPORAL_RESULTS_JSON",
        "SSTW_VIDSIG_MSG_DECODER_PATH",
        "SSTW_VIDSIG_VAE_CHECKPOINT_PATH",
    ):
        monkeypatch.delenv(env_name, raising=False)
    for baseline_id in ("VIDEOSHIELD", "SIGMARK", "SPDMARK", "VIDEOMARK", "VIDSIG", "VIDEOSEAL"):
        monkeypatch.delenv(f"SSTW_{baseline_id}_NATIVE_EVAL_COMMAND", raising=False)

    record = read_jsonl(run_root / "records" / "runtime_detection_records.jsonl")[0]
    output = bundle_root / "videoshield" / "records" / f"{record['prompt_id']}__{record['seed_id']}__{record['attack_name']}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({
            "external_baseline_score": 0.61,
            "detected": True,
            "official_result_provenance": "third_party_official_code",
        }),
        encoding="utf-8",
    )

    audit = build_official_result_bundle_preflight(run_root)
    invalid_reasons = [
        str(row.get("invalid_bundle_reason") or "")
        for row in audit["missing_bundle_examples"]
    ]
    assert audit["official_result_bundle_preflight_decision"] == "FAIL"
    assert any("official_result_bundle_not_repository_generated" in reason for reason in invalid_reasons)


@pytest.mark.quick
def test_repository_official_eval_adapters_are_tracked_fail_closed_entrypoints() -> None:
    """6 个现代 baseline 必须有可导入的 repository official adapter 入口。"""
    import importlib

    for baseline_id in ("videoshield", "sigmark", "spdmark", "videomark", "vidsig", "videoseal"):
        module = importlib.import_module(f"external_baseline.official_eval_adapters.{baseline_id}")
        assert callable(module.main)
        assert module.BASELINE_ID == baseline_id
        assert module.REQUIRED_SOURCE_FILES


@pytest.mark.quick
@pytest.mark.parametrize("npz_value", ["", "."])
def test_sigmark_adapter_missing_npz_fails_closed_without_loading_current_directory(
    tmp_path: Path,
    npz_value: str,
) -> None:
    """SigMark adapter 不能把空环境变量或目录 `.` 当作官方 bit accuracy npz。

    该测试复现 Colab 中 `SSTW_SIGMARK_BIT_ACCURACY_NPZ` 未正确指向文件时的
    失败路径。期望行为是明确报告官方资源缺失, 而不是让 `numpy.load('.')`
    抛出不易理解的 `IsADirectoryError`。
    """

    source_dir = tmp_path / "sigmark_source"
    (source_dir / "watermarks").mkdir(parents=True)
    for relative in ("main.py", "watermarks/sigmark.py", "apply_disturbances.py"):
        path = source_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# 官方源码占位文件, 用于测试 adapter 资源校验边界。\n", encoding="utf-8")
    source_video = tmp_path / "source.mp4"
    attacked_video = tmp_path / "attacked.mp4"
    source_video.write_bytes(b"source")
    attacked_video.write_bytes(b"attacked")
    output_json = tmp_path / "sigmark_output.json"

    env = dict(os.environ)
    env.pop("SSTW_SIGMARK_NATIVE_EVAL_COMMAND", None)
    if npz_value:
        env["SSTW_SIGMARK_BIT_ACCURACY_NPZ"] = npz_value
    else:
        env.pop("SSTW_SIGMARK_BIT_ACCURACY_NPZ", None)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "external_baseline.official_eval_adapters.sigmark",
            "--official-source-dir",
            str(source_dir),
            "--source-video",
            str(source_video),
            "--attacked-video",
            str(attacked_video),
            "--attack-name",
            "video_compression_runtime",
            "--official-output-json",
            str(output_json),
        ],
        cwd=Path.cwd(),
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert completed.returncode != 0
    assert "sigmark_official_required_artifacts_missing" in completed.stderr
    assert "SSTW_SIGMARK_BIT_ACCURACY_NPZ" in completed.stderr
    assert "IsADirectoryError" not in completed.stderr


@pytest.mark.quick
def test_sigmark_adapter_discovers_prefixed_bit_accuracy_npz_from_output_dir(tmp_path: Path) -> None:
    """SigMark adapter 应支持官方带前缀的 `*-bit_accuracy.npz` 输出文件。"""

    import numpy as np

    source_dir = tmp_path / "sigmark_source"
    (source_dir / "watermarks").mkdir(parents=True)
    for relative in ("main.py", "watermarks/sigmark.py", "apply_disturbances.py"):
        path = source_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# 官方源码占位文件, 用于测试 adapter 资源校验边界。\n", encoding="utf-8")
    source_video = tmp_path / "source.mp4"
    attacked_video = tmp_path / "attacked.mp4"
    source_video.write_bytes(b"source")
    attacked_video.write_bytes(b"attacked")
    official_output_dir = tmp_path / "sigmark_official_outputs"
    official_output_dir.mkdir()
    bit_accuracy_npz = official_output_dir / "HunyuanVideo-I2V-community-sigmark-bit_accuracy.npz"
    np.savez(bit_accuracy_npz, sample_0=np.array([0.75, 0.85], dtype=float))
    output_json = tmp_path / "sigmark_output.json"

    env = dict(os.environ)
    env.pop("SSTW_SIGMARK_NATIVE_EVAL_COMMAND", None)
    env.pop("SSTW_SIGMARK_BIT_ACCURACY_NPZ", None)
    env["SSTW_SIGMARK_OUTPUT_DIR"] = str(official_output_dir)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "external_baseline.official_eval_adapters.sigmark",
            "--official-source-dir",
            str(source_dir),
            "--source-video",
            str(source_video),
            "--attacked-video",
            str(attacked_video),
            "--attack-name",
            "video_compression_runtime",
            "--official-output-json",
            str(output_json),
        ],
        cwd=Path.cwd(),
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["official_bit_accuracy_npz_path"] == str(bit_accuracy_npz)
    assert payload["bit_accuracy"] == 0.8
