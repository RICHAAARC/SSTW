from __future__ import annotations

import json
from pathlib import Path
import zipfile

import pytest

from paper_workflow.colab_utils.stage_package_sync import (
    hydrate_external_baseline_resource_packages,
    latest_stage_package_zip,
    publish_colab_stage_package,
)
from paper_workflow.notebook_utils.generative_video_model_probe_workflow import build_drive_packaging_command


def _write_validation_scale_package_pass_manifest(run_root: Path) -> None:
    """写入 validation_scale paper gate 发布所需的最小 PASS manifest。

    该 helper 属于测试内的通用写法, 用于表达 paper gate 阶段 zip 只有在
    validation_scale package manifest 已证明门禁闭环通过后才允许发布。
    """

    manifest_path = run_root / "manifests" / "validation_scale_package_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "validation_scale_package_manifest_decision": "PASS",
                "validation_scale_gate_decision": "PASS",
                "validation_scale_to_pilot_paper_transition_decision": "PASS",
                "missing_artifact_count": 0,
                "missing_artifact_relpaths": [],
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.quick
def test_stage_package_publish_keeps_timestamp_zip_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """阶段 zip 默认保留时间戳包, 不再写 latest 小入口。"""

    monkeypatch.setenv("SSTW_COLAB_STAGE_IO_MODE", "local_zip")
    monkeypatch.delenv("SSTW_STAGE_PACKAGE_KEEP_TIMESTAMP_SNAPSHOT", raising=False)
    drive_root = tmp_path / "drive" / "SSTW"
    run_root = tmp_path / "workspace" / "runs" / "validation_scale"
    run_root.mkdir(parents=True)
    (run_root / "artifacts").mkdir()
    (run_root / "artifacts" / "runtime_decision.json").write_text(
        json.dumps({"decision": "PASS"}),
        encoding="utf-8",
    )
    layout = {
        "drive_project_root": str(drive_root),
        "workflow_profile": "validation_scale",
        "drive_run_root": str(run_root),
        "local_stage_package_cache_root": str(tmp_path / "local_cache"),
        "local_stage_workspace_root": str(tmp_path / "workspace"),
    }

    result = publish_colab_stage_package(layout, notebook_role="generative_video_runtime", include_videos=True)

    package_dir = drive_root / "validation_scale" / "generative_video_runtime_colab"
    assert result["stage_package_publish_status"] == "published"
    assert not (package_dir / "stage_package_latest.zip").exists()
    assert not (package_dir / "stage_package_latest_manifest.json").exists()
    timestamp_zips = list(package_dir.glob("validation_scale_generative_video_runtime_colab_*.zip"))
    assert len(timestamp_zips) == 1
    assert latest_stage_package_zip(drive_root, "validation_scale", "generative_video_runtime_colab") == timestamp_zips[0]
    with zipfile.ZipFile(timestamp_zips[0]) as archive:
        assert any(name.endswith("runtime_decision.json") for name in archive.namelist())


@pytest.mark.quick
def test_formal_comparison_stage_package_contains_only_scoring_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """formal comparison 阶段包只保存公平比较产物, 不重复打包上游视频或 official bundle。"""

    monkeypatch.setenv("SSTW_COLAB_STAGE_IO_MODE", "local_zip")
    drive_root = tmp_path / "drive" / "SSTW"
    run_root = tmp_path / "workspace" / "runs" / "generative_video_model_probe" / "validation_scale"
    write_targets = {
        "records/sstw_measured_formal_records.jsonl": "{}\n",
        "artifacts/sstw_measured_formal_decision.json": json.dumps({"sstw_measured_formal_decision": "PASS"}),
        "records/fair_detection_calibration_records.jsonl": "{}\n",
        "tables/fair_detection_calibration_table.csv": "method_id,status\nsstw,ready\n",
        "reports/fair_detection_calibration_report.md": "# fair calibration\n",
        "artifacts/fair_detection_calibration_decision.json": json.dumps(
            {"fair_detection_calibration_decision": "PASS"}
        ),
        "records/formal_method_baseline_comparison_records.jsonl": "{}\n",
        "tables/formal_method_baseline_comparison_table.csv": "method_id,status\nsstw,ready\n",
        "reports/formal_method_baseline_comparison_report.md": "# formal comparison\n",
        "artifacts/formal_method_baseline_comparison_decision.json": json.dumps(
            {"formal_method_baseline_comparison_decision": "PASS"}
        ),
        "records/formal_baseline_difference_interval_records.jsonl": "{}\n",
        "tables/formal_baseline_difference_interval_table.csv": "baseline_id,status\nvideoseal,ready\n",
        "reports/formal_baseline_difference_interval_report.md": "# difference interval\n",
        "artifacts/formal_baseline_difference_interval_decision.json": json.dumps(
            {"formal_baseline_difference_interval_decision": "PASS"}
        ),
    }
    for relpath, content in write_targets.items():
        path = run_root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    (run_root / "videos").mkdir(parents=True)
    (run_root / "videos" / "source.mp4").write_bytes(b"runtime-video")
    (run_root / "artifacts" / "external_baseline_evidence" / "vidsig" / "unit_000").mkdir(parents=True)
    (run_root / "artifacts" / "external_baseline_evidence" / "vidsig" / "unit_000" / "stdout.txt").write_text(
        "large evidence",
        encoding="utf-8",
    )
    layout = {
        "drive_project_root": str(drive_root),
        "workflow_profile": "validation_scale",
        "drive_run_root": str(run_root),
        "local_stage_package_cache_root": str(tmp_path / "local_cache"),
        "local_stage_workspace_root": str(tmp_path / "workspace"),
    }

    result = publish_colab_stage_package(layout, notebook_role="formal_comparison_scoring", include_videos=True)

    assert result["stage_package_publish_status"] == "published"
    assert "validation_scale/formal_comparison_scoring_colab" in result["drive_stage_package_zip"].replace("\\", "/")
    with zipfile.ZipFile(result["drive_stage_package_zip"]) as archive:
        names = archive.namelist()
    for relpath in write_targets:
        assert any(name.endswith(relpath) for name in names)
    assert not any(name.endswith("source.mp4") for name in names)
    assert not any("external_baseline_evidence/vidsig" in name for name in names)


@pytest.mark.quick
def test_paper_gate_stage_package_includes_fair_comparison_governed_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """paper gate 阶段包必须包含公平比较 records、tables、reports 和 decision。"""

    monkeypatch.setenv("SSTW_COLAB_STAGE_IO_MODE", "local_zip")
    drive_root = tmp_path / "drive" / "SSTW"
    run_root = tmp_path / "workspace" / "runs" / "generative_video_model_probe" / "validation_scale"
    write_targets = {
        "records/fair_detection_calibration_records.jsonl": "{}\n",
        "tables/fair_detection_calibration_table.csv": "method_id,status\nsstw,ready\n",
        "reports/fair_detection_calibration_report.md": "# fair calibration\n",
        "artifacts/fair_detection_calibration_decision.json": json.dumps(
            {"fair_detection_calibration_decision": "PASS"}
        ),
        "records/formal_method_baseline_comparison_records.jsonl": "{}\n",
        "tables/formal_method_baseline_comparison_table.csv": "method_id,status\nsstw,ready\n",
        "reports/formal_method_baseline_comparison_report.md": "# formal comparison\n",
        "artifacts/formal_method_baseline_comparison_decision.json": json.dumps(
            {"formal_method_baseline_comparison_decision": "PASS"}
        ),
        "records/formal_baseline_difference_interval_records.jsonl": "{}\n",
        "tables/formal_baseline_difference_interval_table.csv": "baseline_id,status\nvideoseal,ready\n",
        "reports/formal_baseline_difference_interval_report.md": "# difference interval\n",
        "artifacts/formal_baseline_difference_interval_decision.json": json.dumps(
            {"formal_baseline_difference_interval_decision": "PASS"}
        ),
    }
    for relpath, content in write_targets.items():
        path = run_root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    _write_validation_scale_package_pass_manifest(run_root)
    layout = {
        "drive_project_root": str(drive_root),
        "workflow_profile": "validation_scale",
        "drive_run_root": str(run_root),
        "local_stage_package_cache_root": str(tmp_path / "local_cache"),
        "local_stage_workspace_root": str(tmp_path / "workspace"),
    }

    result = publish_colab_stage_package(layout, notebook_role="paper_gate_and_package", include_videos=False)

    assert result["stage_package_publish_status"] == "published"
    with zipfile.ZipFile(result["drive_stage_package_zip"]) as archive:
        names = archive.namelist()
    for relpath in write_targets:
        assert any(name.endswith(relpath) for name in names)


@pytest.mark.quick
def test_validation_scale_paper_gate_package_blocks_without_package_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """validation_scale paper gate 缺少最终 package manifest 时只能写阻断 manifest。"""

    monkeypatch.setenv("SSTW_COLAB_STAGE_IO_MODE", "local_zip")
    drive_root = tmp_path / "drive" / "SSTW"
    run_root = tmp_path / "workspace" / "runs" / "generative_video_model_probe" / "validation_scale"
    (run_root / "records").mkdir(parents=True)
    (run_root / "records" / "generation_records.jsonl").write_text("{}\n", encoding="utf-8")
    layout = {
        "drive_project_root": str(drive_root),
        "workflow_profile": "validation_scale",
        "drive_run_root": str(run_root),
        "local_stage_package_cache_root": str(tmp_path / "local_cache"),
        "local_stage_workspace_root": str(tmp_path / "workspace"),
    }

    result = publish_colab_stage_package(layout, notebook_role="paper_gate_and_package", include_videos=False)

    package_dir = drive_root / "validation_scale" / "paper_gate_and_package_colab"
    manifests = list(package_dir.glob("validation_scale_paper_gate_and_package_colab_*_manifest.json"))
    assert result["stage_package_publish_status"] == "blocked_missing_validation_scale_package_manifest"
    assert result["drive_stage_package_zip"] == ""
    assert not list(package_dir.glob("validation_scale_paper_gate_and_package_colab_*.zip"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["stage_package_publish_status"] == "blocked_missing_validation_scale_package_manifest"
    assert manifest["claim_support_status"] == "stage_package_blocked_not_claim_evidence"


@pytest.mark.quick
@pytest.mark.quick
@pytest.mark.quick
def test_external_baseline_role_without_baseline_id_can_publish_helper_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """external baseline role 缺少 baseline_id 时应归入 helper, 不误判为单 baseline reference。"""

    monkeypatch.setenv("SSTW_COLAB_STAGE_IO_MODE", "local_zip")
    drive_root = tmp_path / "drive" / "SSTW"
    run_root = tmp_path / "workspace" / "runs" / "validation_scale"
    (run_root / "artifacts").mkdir(parents=True)
    (run_root / "artifacts" / "external_baseline_comparison_decision.json").write_text(
        json.dumps({"external_baseline_comparison_decision": "PASS"}),
        encoding="utf-8",
    )
    bundle_root = tmp_path / "workspace" / "external_baseline_official_result_bundles" / "validation_scale"
    (bundle_root / "videoseal" / "records").mkdir(parents=True)
    (bundle_root / "videoseal" / "records" / "sample.json").write_text("{}", encoding="utf-8")
    layout = {
        "drive_project_root": str(drive_root),
        "workflow_profile": "validation_scale",
        "stage_package_id": "external_baseline_aggregate_diagnostic_colab",
        "drive_run_root": str(run_root),
        "external_baseline_official_result_bundle_root": str(bundle_root),
        "local_stage_package_cache_root": str(tmp_path / "local_cache"),
        "local_stage_workspace_root": str(tmp_path / "workspace"),
    }

    result = publish_colab_stage_package(
        layout,
        notebook_role="external_baseline_formal_scoring",
        baseline_id=None,
        include_videos=False,
    )

    assert result["stage_package_publish_status"] == "published"
    assert "helper" in result["drive_stage_package_zip"].replace("\\", "/")


@pytest.mark.quick
def test_history_drive_packager_is_noop_in_stage_zip_mode() -> None:
    """local_zip 模式下历史 drive packager 只能成为 no-op, 实际落盘由阶段 zip 完成。"""

    command = build_drive_packaging_command(
        {
            "stage_package_handoff_mode": "local_zip",
            "drive_run_root": "/content/SSTW_stage_workspace/runs/validation_scale",
            "drive_package_dir": "/content/drive/MyDrive/SSTW/validation_scale/generative_video_runtime_colab",
        }
    )

    assert command[1] == "-c"
    assert "skip legacy drive packager" in command[2]
    assert "generative_video_drive_packager.py" in command[2]


@pytest.mark.quick
def test_external_baseline_package_contains_only_current_baseline_bundle_and_respects_video_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """baseline 专用 Notebook 只能打包当前 baseline official bundle, 并遵守 include_videos。"""

    monkeypatch.setenv("SSTW_COLAB_STAGE_IO_MODE", "local_zip")
    drive_root = tmp_path / "drive" / "SSTW"
    workspace = tmp_path / "workspace"
    run_root = workspace / "runs" / "generative_video_model_probe" / "validation_scale"
    decision_dir = run_root / "artifacts" / "external_baseline_formal_reference"
    decision_dir.mkdir(parents=True)
    (decision_dir / "videoseal_formal_reference_decision.json").write_text(
        json.dumps(
            {
                "manifest_kind": "modern_external_baseline_formal_reference_decision",
                "baseline_id": "videoseal",
                "formal_reference_decision": "PASS",
                "formal_reference_status": "official_reference_bundle_complete",
            }
        ),
        encoding="utf-8",
    )
    (decision_dir / "vidsig_formal_reference_decision.json").write_text(
        json.dumps(
            {
                "manifest_kind": "modern_external_baseline_formal_reference_decision",
                "baseline_id": "vidsig",
                "formal_reference_decision": "PASS",
                "formal_reference_status": "official_reference_bundle_complete",
            }
        ),
        encoding="utf-8",
    )
    (run_root / "artifacts" / "external_baseline_evidence" / "videoseal" / "unit_000").mkdir(parents=True)
    (run_root / "artifacts" / "external_baseline_evidence" / "videoseal" / "unit_000" / "manifest.json").write_text(
        "{}",
        encoding="utf-8",
    )
    (run_root / "artifacts" / "external_baseline_evidence" / "vidsig" / "unit_000").mkdir(parents=True)
    (run_root / "artifacts" / "external_baseline_evidence" / "vidsig" / "unit_000" / "manifest.json").write_text(
        "{}",
        encoding="utf-8",
    )
    bundle_root = workspace / "external_baseline_official_result_bundles" / "validation_scale"
    (bundle_root / "videoseal" / "records").mkdir(parents=True)
    (bundle_root / "videoseal" / "records" / "sample.json").write_text("{}", encoding="utf-8")
    (bundle_root / "videoseal" / "official_outputs" / "wm" / "frames").mkdir(parents=True)
    (bundle_root / "videoseal" / "official_outputs" / "wm.mp4").write_bytes(b"video")
    (bundle_root / "videoseal" / "official_outputs" / "wm" / "frames" / "000.png").write_bytes(b"frame")
    (bundle_root / "vidsig" / "records").mkdir(parents=True)
    (bundle_root / "vidsig" / "records" / "other.json").write_text("{}", encoding="utf-8")
    layout = {
        "drive_project_root": str(drive_root),
        "workflow_profile": "validation_scale",
        "stage_package_id": "external_baseline_formal_reference_videoseal",
        "drive_run_root": str(run_root),
        "external_baseline_official_result_bundle_root": str(bundle_root),
        "local_stage_package_cache_root": str(tmp_path / "local_cache"),
        "local_stage_workspace_root": str(workspace),
    }

    result = publish_colab_stage_package(
        layout,
        notebook_role="external_baseline_formal_scoring",
        baseline_id="videoseal",
        include_videos=False,
    )

    assert result["stage_package_publish_status"] == "published"
    assert "validation_scale/external_baseline_official_reference" in result["drive_stage_package_zip"].replace("\\", "/")
    with zipfile.ZipFile(result["drive_stage_package_zip"]) as archive:
        names = archive.namelist()
    assert any("videoseal/records/sample.json" in name for name in names)
    assert any("videoseal_formal_reference_decision.json" in name for name in names)
    assert any(name.endswith("artifacts/notebook_runtime_report.json") for name in names)
    assert any("external_baseline_evidence/videoseal/unit_000/manifest.json" in name for name in names)
    assert not any("vidsig/records/other.json" in name for name in names)
    assert not any("vidsig_formal_reference_decision.json" in name for name in names)
    assert not any("external_baseline_evidence/vidsig/unit_000/manifest.json" in name for name in names)
    assert not any(name.endswith(".mp4") for name in names)
    assert not any(name.endswith("000.png") for name in names)


@pytest.mark.quick
def test_stage_package_excludes_obsolete_spdmark_and_small_scale_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """阶段 zip 不应继续发布已退出当前主实验规则的历史 artifact 路径。"""

    monkeypatch.setenv("SSTW_COLAB_STAGE_IO_MODE", "local_zip")
    drive_root = tmp_path / "drive" / "SSTW"
    run_root = tmp_path / "workspace" / "runs" / "generative_video_model_probe" / "validation_scale"
    write_targets = {
        "records/generation_records.jsonl": "{}\n",
        "artifacts/small_scale_claim_pilot_gate_decision.json": "{}",
        "records/small_scale_claim_pilot_gate_records.jsonl": "{}\n",
        "tables/small_scale_claim_pilot_gate_table.csv": "stage_id\nsmall_scale\n",
        "reports/small_scale_claim_pilot_gate_report.md": "# old\n",
        "artifacts/external_baseline_evidence/spdmark/unit_000/official_stdout.txt": "old",
        "artifacts/external_baseline_evidence/videoseal/unit_000/official_stdout.txt": "ok",
    }
    for relpath, content in write_targets.items():
        path = run_root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    _write_validation_scale_package_pass_manifest(run_root)
    layout = {
        "drive_project_root": str(drive_root),
        "workflow_profile": "validation_scale",
        "drive_run_root": str(run_root),
        "local_stage_package_cache_root": str(tmp_path / "local_cache"),
        "local_stage_workspace_root": str(tmp_path / "workspace"),
    }

    result = publish_colab_stage_package(layout, notebook_role="paper_gate_and_package", include_videos=True)

    with zipfile.ZipFile(result["drive_stage_package_zip"]) as archive:
        names = archive.namelist()
    assert any(name.endswith("records/generation_records.jsonl") for name in names)
    assert any("external_baseline_evidence/videoseal" in name for name in names)
    assert not any("small_scale_claim_pilot" in name for name in names)
    assert not any("external_baseline_evidence/spdmark" in name for name in names)


@pytest.mark.quick
def test_external_baseline_resource_zip_is_hydrated_to_local_resource_root(tmp_path: Path) -> None:
    """Drive resources 中的 zip 包应复制并解压到本地资源根目录。"""

    remote_root = tmp_path / "drive" / "SSTW" / "resources" / "external_baseline"
    remote_root.mkdir(parents=True)
    package_zip = remote_root / "vidsig_resources.zip"
    with zipfile.ZipFile(package_zip, mode="w") as archive:
        archive.writestr("resources/external_baseline/vidsig/ckpts/model.bin", b"checkpoint")
    local_root = tmp_path / "workspace" / "resources" / "external_baseline"
    layout = {
        "external_baseline_resource_root_remote": str(remote_root),
        "external_baseline_resource_root_local": str(local_root),
        "local_stage_package_cache_root": str(tmp_path / "local_cache"),
    }

    result = hydrate_external_baseline_resource_packages(layout)

    assert result["external_baseline_resource_package_restore_status"] == "restored"
    assert result["resource_package_count"] == 1
    assert (local_root / "vidsig" / "ckpts" / "model.bin").read_bytes() == b"checkpoint"
