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
from experiments.generative_video_model_probe.external_baseline_runner import (
    audit_external_baseline_comparison_records,
    build_external_baseline_comparison_table_rows,
    write_external_baseline_comparison_outputs,
    write_external_baseline_status_outputs,
)
from external_baseline.official_bundle_generator import _apply_video_tensor_attack, build_official_bundle_generation_plan
import external_baseline.official_resource_bootstrap as official_resource_bootstrap
from external_baseline.official_resource_bootstrap import bootstrap_official_resources
from external_baseline.official_result_bundle import build_official_result_bundle_preflight
from external_baseline.modern_command_adapter import ModernBaselineCommandConfig, build_modern_score_records
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
from paper_workflow.notebook_utils.generative_video_model_probe_workflow import (
    build_paper_gate_external_baseline_environment,
)
from main.protocol.record_writer import read_jsonl, write_jsonl
from external_baseline.source_intake import build_execution_manifest, build_source_intake_manifest, write_source_intake_artifacts


@pytest.mark.quick
def test_external_baseline_status_audit_reports_modern_gap() -> None:
    """现代 baseline 已有 governed 状态记录, 但尚未达到主表比较 ready。"""
    records = build_external_baseline_records("configs/external_baselines/external_baselines.json")
    audit = audit_external_baseline_records(records)

    assert audit["external_baseline_status_decision"] == "PASS"
    assert audit["modern_external_baseline_status_records_ready"] is True
    assert audit["modern_external_baseline_record_count"] == 3
    assert audit["modern_external_baseline_main_comparison_ready_count"] == 0
    assert audit["external_baseline_claim_support_status"] == "governed_status_records_only"


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
def test_videoseal_runtime_attack_mapping_is_explicit_and_fail_closed() -> None:
    """VideoSeal official bundle 的 runtime attack 映射必须显式登记并对未知 attack 阻断。"""

    import torch

    video = torch.arange(5, dtype=torch.float32).reshape(5, 1, 1, 1)

    assert _apply_video_tensor_attack(video, "video_compression_runtime").shape[0] == 5
    assert _apply_video_tensor_attack(video, "temporal_crop_runtime").shape[0] == 3
    assert _apply_video_tensor_attack(video, "frame_rate_resampling_runtime").shape[0] == 3
    with pytest.raises(ValueError, match="unsupported_videoseal_runtime_attack"):
        _apply_video_tensor_attack(video, "unexpected_runtime_attack")


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
    assert decision["manual_official_resource_required_count"] >= 1
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
def test_external_baseline_audit_rejects_handwritten_measured_formal_without_evidence() -> None:
    """external baseline audit 不能把手写 measured_formal 行计入正式 baseline 覆盖。"""

    records = [
        {
            "external_baseline_name": "videoseal",
            "external_baseline_layer": "modern_external_baseline",
            "metric_status": "measured_formal",
            "external_baseline_score": 0.61,
            "prompt_id": "prompt_0",
            "seed_id": "seed_0",
            "attack_name": "video_compression_runtime",
        },
    ]
    audit = audit_external_baseline_comparison_records(records)

    assert audit["external_baseline_comparison_decision"] == "FAIL"
    assert audit["external_baseline_formal_ready_count"] == 0
    assert audit["external_baseline_formal_incomplete_record_count"] == 1
    assert audit["modern_external_baseline_formal_measured_adapter_count"] == 0
    assert audit["modern_external_baseline_formal_measured_adapter_names"] == []


@pytest.mark.quick
def test_external_baseline_audit_rejects_formal_row_without_score_extraction_policy() -> None:
    """measured_formal 行缺少官方分数抽取口径时不能计入公平比较覆盖。"""

    records = [
        {
            "external_baseline_name": "videoseal",
            "external_baseline_layer": "modern_external_baseline",
            "metric_status": "measured_formal",
            "external_baseline_score": 0.61,
            "external_baseline_raw_detector_score": 0.61,
            "external_baseline_score_semantics": "watermark_presence_detector_score",
            "external_baseline_score_orientation": "higher_is_more_watermarked",
            "prompt_id": "prompt_0",
            "seed_id": "seed_0",
            "attack_name": "video_compression_runtime",
            "external_baseline_clean_negative_score": 0.08,
            "external_baseline_clean_negative_video_path": "official/videoseal/clean_negative.mp4",
            "external_baseline_official_result_provenance": "repository_generated_from_third_party_official_code",
            "external_baseline_official_result_bundle_path": "official/videoseal/bundle.json",
            "external_baseline_official_execution_manifest_path": "official/videoseal/manifest.json",
        },
    ]
    audit = audit_external_baseline_comparison_records(records)

    assert audit["external_baseline_comparison_decision"] == "FAIL"
    assert audit["external_baseline_formal_ready_count"] == 0
    assert audit["external_baseline_formal_incomplete_record_count"] == 1
    assert audit["modern_external_baseline_formal_measured_adapter_count"] == 0


@pytest.mark.quick
def test_external_baseline_table_and_manifest_reject_incomplete_measured_formal_rows(tmp_path: Path) -> None:
    """表格和 execution manifest 必须与 audit 使用同一 formal evidence 口径。"""

    run_root = tmp_path / "run"
    write_jsonl(run_root / "records" / "runtime_detection_records.jsonl", [])
    records = [
        {
            "external_baseline_name": "videoseal",
            "external_baseline_layer": "modern_external_baseline",
            "external_baseline_family": "video_watermark",
            "metric_status": "measured_formal",
            "external_baseline_score": 0.61,
            "prompt_id": "prompt_0",
            "seed_id": "seed_0",
            "attack_name": "video_compression_runtime",
            "claim_support_status": "modern_external_baseline_formal_measured",
        },
    ]

    table_rows = build_external_baseline_comparison_table_rows(run_root, records)
    manifest = build_execution_manifest(
        records,
        run_root=run_root,
        config_path="configs/external_baselines/external_baselines.json",
    )
    videoseal_row = next(row for row in table_rows if row["method_id"] == "videoseal")

    assert videoseal_row["metric_status"] == "unsupported"
    assert videoseal_row["comparison_scope"] == "external_baseline_result_missing"
    assert videoseal_row["external_baseline_result_used_for_claim"] is False
    assert videoseal_row["external_baseline_formal_incomplete_record_count"] == 1
    assert manifest["external_baseline_formal_measured_adapter_count"] == 0
    assert manifest["modern_external_baseline_formal_measured_adapter_count"] == 0
    assert manifest["external_baseline_formal_incomplete_record_count"] == 1
    assert manifest["formal_result_claim"] is False
    assert manifest["formal_evidence_status"] == "no_formal_rows"


@pytest.mark.quick
def test_modern_external_baseline_formal_command_adapters_require_clean_negative(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """现代 baseline 官方命令若缺少 clean negative 分数, 不能写成 measured_formal。"""

    run_root = tmp_path / "generative_video_runtime"
    _write_external_baseline_runtime_fixture(run_root)
    fake_adapter = tmp_path / "fake_modern_baseline_eval_without_clean_negative.py"
    fake_adapter.write_text(
        "import argparse, json\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--output-json', required=True)\n"
        "parser.add_argument('--source-video')\n"
        "parser.add_argument('--attacked-video')\n"
        "parser.add_argument('--attack-name')\n"
        "args = parser.parse_args()\n"
        "json.dump({'external_baseline_score': 0.37, 'detected': True}, open(args.output_json, 'w', encoding='utf-8'))\n",
        encoding="utf-8",
    )
    command = (
        f"{sys.executable} {fake_adapter} "
        "--source-video {source_video_path} "
        "--attacked-video {attacked_video_path} "
        "--attack-name {attack_name} "
        "--output-json {output_json_path}"
    )
    monkeypatch.setenv("SSTW_UNIT_MODERN_EVAL_COMMAND", command)
    config = ModernBaselineCommandConfig(
        baseline_name="unit_modern",
        baseline_family="unit_modern_video_watermark",
        adapter_path="external_baseline/primary/unit_modern/adapter/run_sstw_eval.py",
        env_var="SSTW_UNIT_MODERN_EVAL_COMMAND",
        default_source_script="external_baseline/primary/unit_modern/adapter/run_sstw_eval.py",
        score_source="official_command_adapter",
    )

    modern_records = build_modern_score_records(
        run_root,
        {
            "external_baseline_name": "unit_modern",
            "external_baseline_family": "unit_modern_video_watermark",
            "external_baseline_layer": "modern_external_baseline",
        },
        config,
    )

    assert modern_records
    assert all(record["metric_status"] == "unsupported" for record in modern_records)
    assert any(
        "clean_negative" in record["external_baseline_score_failure_reason"]
        for record in modern_records
    )


@pytest.mark.quick
def test_modern_external_baseline_formal_command_adapters_require_official_bundle_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """现代 baseline 官方命令若只有 detector 输出, 不能写成正式 comparison 证据。"""

    run_root = tmp_path / "generative_video_runtime"
    _write_external_baseline_runtime_fixture(run_root)
    fake_adapter = tmp_path / "fake_modern_baseline_eval_without_bundle.py"
    fake_adapter.write_text(
        "import argparse, json\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--output-json', required=True)\n"
        "parser.add_argument('--source-video')\n"
        "parser.add_argument('--attacked-video')\n"
        "parser.add_argument('--attack-name')\n"
        "args = parser.parse_args()\n"
        "json.dump({'external_baseline_score': 0.37, 'detected': True, "
        "'score_semantics': 'watermark_presence_confidence', 'score_orientation': 'higher_is_more_watermarked', 'official_score_extraction_policy': 'test_official_detector_confidence', 'official_reference_protocol_anchor': 'same_prompt_seed_attack_runtime_comparison_unit', 'external_baseline_clean_negative_score': 0.08, "
        "'external_baseline_clean_negative_video_path': 'official/clean_negative.mp4', "
        "'official_result_provenance': 'repository_generated_from_third_party_official_code'}, "
        "open(args.output_json, 'w', encoding='utf-8'))\n",
        encoding="utf-8",
    )
    command = (
        f"{sys.executable} {fake_adapter} "
        "--source-video {source_video_path} "
        "--attacked-video {attacked_video_path} "
        "--attack-name {attack_name} "
        "--output-json {output_json_path}"
    )
    monkeypatch.setenv("SSTW_UNIT_MODERN_EVAL_COMMAND", command)
    config = ModernBaselineCommandConfig(
        baseline_name="unit_modern",
        baseline_family="unit_modern_video_watermark",
        adapter_path="external_baseline/primary/unit_modern/adapter/run_sstw_eval.py",
        env_var="SSTW_UNIT_MODERN_EVAL_COMMAND",
        default_source_script="external_baseline/primary/unit_modern/adapter/run_sstw_eval.py",
        score_source="official_command_adapter",
    )

    modern_records = build_modern_score_records(
        run_root,
        {
            "external_baseline_name": "unit_modern",
            "external_baseline_family": "unit_modern_video_watermark",
            "external_baseline_layer": "modern_external_baseline",
        },
        config,
    )

    assert modern_records
    assert all(record["metric_status"] == "unsupported" for record in modern_records)
    assert any(
        "official_result_bundle_evidence_missing" in record["external_baseline_score_failure_reason"]
        for record in modern_records
    )


@pytest.mark.quick
def test_modern_external_baseline_formal_command_adapters_reject_external_bundle_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """现代 baseline 不能把外部补交 bundle 路径伪装成项目内正式证据。"""

    run_root = tmp_path / "generative_video_runtime"
    _write_external_baseline_runtime_fixture(run_root)
    fake_adapter = tmp_path / "fake_modern_baseline_eval_with_external_bundle.py"
    fake_adapter.write_text(
        "import argparse, json\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--output-json', required=True)\n"
        "parser.add_argument('--source-video')\n"
        "parser.add_argument('--attacked-video')\n"
        "parser.add_argument('--attack-name')\n"
        "args = parser.parse_args()\n"
        "json.dump({'external_baseline_score': 0.37, 'detected': True, "
        "'score_semantics': 'watermark_presence_confidence', 'score_orientation': 'higher_is_more_watermarked', 'official_score_extraction_policy': 'test_official_detector_confidence', 'official_reference_protocol_anchor': 'same_prompt_seed_attack_runtime_comparison_unit', 'external_baseline_clean_negative_score': 0.08, "
        "'external_baseline_clean_negative_video_path': 'official/clean_negative.mp4', "
        "'official_result_provenance': 'external_user_supplied_result', "
        "'official_result_bundle_path': 'official/bundle_record.json', "
        "'official_execution_manifest_path': 'official/execution_manifest.json'}, "
        "open(args.output_json, 'w', encoding='utf-8'))\n",
        encoding="utf-8",
    )
    command = (
        f"{sys.executable} {fake_adapter} "
        "--source-video {source_video_path} "
        "--attacked-video {attacked_video_path} "
        "--attack-name {attack_name} "
        "--output-json {output_json_path}"
    )
    monkeypatch.setenv("SSTW_UNIT_MODERN_EVAL_COMMAND", command)
    config = ModernBaselineCommandConfig(
        baseline_name="unit_modern",
        baseline_family="unit_modern_video_watermark",
        adapter_path="external_baseline/primary/unit_modern/adapter/run_sstw_eval.py",
        env_var="SSTW_UNIT_MODERN_EVAL_COMMAND",
        default_source_script="external_baseline/primary/unit_modern/adapter/run_sstw_eval.py",
        score_source="official_command_adapter",
    )

    modern_records = build_modern_score_records(
        run_root,
        {
            "external_baseline_name": "unit_modern",
            "external_baseline_family": "unit_modern_video_watermark",
            "external_baseline_layer": "modern_external_baseline",
        },
        config,
    )

    assert modern_records
    assert all(record["metric_status"] == "unsupported" for record in modern_records)
    assert any(
        "official_result_bundle_provenance_invalid" in record["external_baseline_score_failure_reason"]
        for record in modern_records
    )


@pytest.mark.quick
def test_modern_external_baseline_formal_command_adapters_require_bundle_baseline_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """现代 baseline 官方命令输出必须声明与当前 adapter 一致的 baseline 身份。"""

    run_root = tmp_path / "generative_video_runtime"
    _write_external_baseline_runtime_fixture(run_root)
    fake_adapter = tmp_path / "fake_modern_baseline_eval_without_bundle_identity.py"
    fake_adapter.write_text(
        "import argparse, json\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--output-json', required=True)\n"
        "parser.add_argument('--source-video')\n"
        "parser.add_argument('--attacked-video')\n"
        "parser.add_argument('--attack-name')\n"
        "args = parser.parse_args()\n"
        "json.dump({'external_baseline_score': 0.37, 'detected': True, "
        "'score_semantics': 'watermark_presence_confidence', 'score_orientation': 'higher_is_more_watermarked', 'official_score_extraction_policy': 'test_official_detector_confidence', 'official_reference_protocol_anchor': 'same_prompt_seed_attack_runtime_comparison_unit', 'external_baseline_clean_negative_score': 0.08, "
        "'external_baseline_clean_negative_video_path': 'official/clean_negative.mp4', "
        "'official_result_provenance': 'repository_generated_from_third_party_official_code', "
        "'official_result_bundle_path': 'official/bundle_record.json', "
        "'official_execution_manifest_path': 'official/execution_manifest.json'}, "
        "open(args.output_json, 'w', encoding='utf-8'))\n",
        encoding="utf-8",
    )
    command = (
        f"{sys.executable} {fake_adapter} "
        "--source-video {source_video_path} "
        "--attacked-video {attacked_video_path} "
        "--attack-name {attack_name} "
        "--output-json {output_json_path}"
    )
    monkeypatch.setenv("SSTW_UNIT_MODERN_EVAL_COMMAND", command)
    config = ModernBaselineCommandConfig(
        baseline_name="unit_modern",
        baseline_family="unit_modern_video_watermark",
        adapter_path="external_baseline/primary/unit_modern/adapter/run_sstw_eval.py",
        env_var="SSTW_UNIT_MODERN_EVAL_COMMAND",
        default_source_script="external_baseline/primary/unit_modern/adapter/run_sstw_eval.py",
        score_source="official_command_adapter",
    )

    modern_records = build_modern_score_records(
        run_root,
        {
            "external_baseline_name": "unit_modern",
            "external_baseline_family": "unit_modern_video_watermark",
            "external_baseline_layer": "modern_external_baseline",
        },
        config,
    )

    assert modern_records
    assert all(record["metric_status"] == "unsupported" for record in modern_records)
    assert any(
        "official_result_bundle_missing_baseline_id" in record["external_baseline_score_failure_reason"]
        for record in modern_records
    )


@pytest.mark.quick
def test_modern_external_baseline_formal_command_adapter_normalizes_clean_negative_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """现代 baseline 命令输出 clean_negative_* 短字段时也必须转成正式校准字段。"""

    run_root = tmp_path / "generative_video_runtime"
    _write_external_baseline_runtime_fixture(run_root)
    fake_adapter = tmp_path / "fake_modern_baseline_eval_with_clean_negative_aliases.py"
    fake_adapter.write_text(
        "import argparse, json\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--output-json', required=True)\n"
        "parser.add_argument('--source-video')\n"
        "parser.add_argument('--attacked-video')\n"
        "parser.add_argument('--attack-name')\n"
        "parser.add_argument('--baseline-id', default='unit_modern')\n"
        "args = parser.parse_args()\n"
        "json.dump({'score': 0.44, 'detected': True, "
        "'score_semantics': 'watermark_presence_confidence', 'score_orientation': 'higher_is_more_watermarked', 'official_score_extraction_policy': 'test_official_detector_confidence', 'official_reference_protocol_anchor': 'same_prompt_seed_attack_runtime_comparison_unit', 'clean_negative_score': 0.09, "
        "'clean_negative_score_semantics': 'watermark_presence_confidence', "
        "'clean_negative_video_path': 'official/clean_negative_alias.mp4', "
        "'official_result_provenance': 'repository_generated_from_third_party_official_code', "
        "'official_adapter_baseline_id': args.baseline_id, 'official_baseline_id': args.baseline_id, "
        "'official_result_bundle_path': 'official/bundle_record.json', "
        "'official_execution_manifest_path': 'official/execution_manifest.json'}, "
        "open(args.output_json, 'w', encoding='utf-8'))\n",
        encoding="utf-8",
    )
    command = (
        f"{sys.executable} {fake_adapter} "
        "--source-video {source_video_path} "
        "--attacked-video {attacked_video_path} "
        "--attack-name {attack_name} "
        "--baseline-id {baseline_id} "
        "--output-json {output_json_path}"
    )
    monkeypatch.setenv("SSTW_UNIT_MODERN_EVAL_COMMAND", command)
    config = ModernBaselineCommandConfig(
        baseline_name="unit_modern",
        baseline_family="unit_modern_video_watermark",
        adapter_path="external_baseline/primary/unit_modern/adapter/run_sstw_eval.py",
        env_var="SSTW_UNIT_MODERN_EVAL_COMMAND",
        default_source_script="external_baseline/primary/unit_modern/adapter/run_sstw_eval.py",
        score_source="official_command_adapter",
    )

    modern_records = build_modern_score_records(
        run_root,
        {
            "external_baseline_name": "unit_modern",
            "external_baseline_family": "unit_modern_video_watermark",
            "external_baseline_layer": "modern_external_baseline",
        },
        config,
    )

    assert modern_records
    assert all(record["metric_status"] == "measured_formal" for record in modern_records)
    assert all(record["external_baseline_clean_negative_score"] == 0.09 for record in modern_records)
    assert all(
        record["external_baseline_clean_negative_score_semantics"] == "watermark_presence_confidence"
        for record in modern_records
    )
    assert all(
        record["external_baseline_clean_negative_video_path"] == "official/clean_negative_alias.mp4"
        for record in modern_records
    )


@pytest.mark.quick
def test_modern_external_baseline_bridge_rejects_incomplete_official_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """bridge 不能用当前 baseline_id 回填半结构化 official payload 身份字段。"""

    run_root = tmp_path / "generative_video_runtime"
    _write_external_baseline_runtime_fixture(run_root)
    fake_official = tmp_path / "fake_official_detector_without_complete_identity.py"
    fake_official.write_text(
        "import argparse, json\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--official-output-json', required=True)\n"
        "parser.add_argument('--source-video')\n"
        "parser.add_argument('--attacked-video')\n"
        "parser.add_argument('--attack-name')\n"
        "parser.add_argument('--baseline-id')\n"
        "args = parser.parse_args()\n"
        "json.dump({'score': 0.42, 'detected': True, 'baseline_id': args.baseline_id, "
        "'score_semantics': 'watermark_presence_confidence', 'score_orientation': 'higher_is_more_watermarked', "
        "'official_score_extraction_policy': 'test_official_detector_confidence', "
        "'official_reference_protocol_anchor': 'same_prompt_seed_attack_runtime_comparison_unit', "
        "'external_baseline_clean_negative_score': 0.07, "
        "'external_baseline_clean_negative_score_semantics': 'watermark_presence_confidence', "
        "'external_baseline_clean_negative_video_path': 'official/clean_negative.mp4', "
        "'official_result_provenance': 'repository_generated_from_third_party_official_code', "
        "'official_result_bundle_path': 'official/bundle_record.json', "
        "'official_execution_manifest_path': 'official/execution_manifest.json'}, "
        "open(args.official_output_json, 'w', encoding='utf-8'))\n",
        encoding="utf-8",
    )
    official_command = (
        f"{sys.executable} {fake_official} "
        "--source-video {source_video_path} "
        "--attacked-video {attacked_video_path} "
        "--attack-name {attack_name} "
        "--baseline-id {baseline_id} "
        "--official-output-json {official_output_json_path}"
    )
    official_source_dir = tmp_path / "official_sources" / "unit_modern"
    official_source_dir.mkdir(parents=True)
    monkeypatch.setenv("SSTW_UNIT_MODERN_OFFICIAL_EVAL_COMMAND", official_command)
    bridge_command = (
        f"{sys.executable} -m external_baseline.official_command_bridge "
        "--baseline-id unit_modern "
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
    monkeypatch.setenv("SSTW_UNIT_MODERN_EVAL_COMMAND", bridge_command)
    config = ModernBaselineCommandConfig(
        baseline_name="unit_modern",
        baseline_family="unit_modern_video_watermark",
        adapter_path="external_baseline/primary/unit_modern/adapter/run_sstw_eval.py",
        env_var="SSTW_UNIT_MODERN_EVAL_COMMAND",
        default_source_script="external_baseline/primary/unit_modern/adapter/run_sstw_eval.py",
        score_source="official_command_adapter",
    )

    modern_records = build_modern_score_records(
        run_root,
        {
            "external_baseline_name": "unit_modern",
            "external_baseline_family": "unit_modern_video_watermark",
            "external_baseline_layer": "modern_external_baseline",
        },
        config,
    )

    assert modern_records
    assert all(record["metric_status"] == "unsupported" for record in modern_records)
    assert any(
        "official_result_bundle_missing_complete_baseline_identity" in record["external_baseline_score_failure_reason"]
        for record in modern_records
    )


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
