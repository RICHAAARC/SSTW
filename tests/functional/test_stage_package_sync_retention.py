from __future__ import annotations

import json
from pathlib import Path
import zipfile

import pytest

from paper_workflow.colab_utils.stage_package_sync import latest_stage_package_zip, publish_colab_stage_package
from paper_workflow.notebook_utils.generative_video_model_probe_workflow import build_drive_packaging_command


@pytest.mark.quick
def test_stage_package_publish_keeps_latest_zip_only_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """阶段 zip 默认只在 Drive 保留 latest, 避免历史时间戳包重复占用空间。"""

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

    package_dir = drive_root / "stage_packages" / "validation_scale" / "generative_video_runtime_colab"
    assert result["stage_package_publish_status"] == "published"
    assert (package_dir / "stage_package_latest.zip").exists()
    assert (package_dir / "stage_package_latest_manifest.json").exists()
    assert not list(package_dir.glob("stage_package__*.zip"))
    assert latest_stage_package_zip(drive_root, "validation_scale", "generative_video_runtime_colab") == package_dir / "stage_package_latest.zip"
    with zipfile.ZipFile(package_dir / "stage_package_latest.zip") as archive:
        assert any(name.endswith("runtime_decision.json") for name in archive.namelist())


@pytest.mark.quick
def test_failed_external_baseline_reference_writes_manifest_only_and_removes_stale_zip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """失败的 external baseline 只能留下阻断 manifest, 不能留下可恢复 zip。"""

    monkeypatch.setenv("SSTW_COLAB_STAGE_IO_MODE", "local_zip")
    drive_root = tmp_path / "drive" / "SSTW"
    run_root = tmp_path / "workspace" / "runs" / "validation_scale"
    decision_dir = run_root / "artifacts" / "external_baseline_formal_reference"
    decision_dir.mkdir(parents=True)
    (decision_dir / "sigmark_formal_reference_decision.json").write_text(
        json.dumps(
            {
                "manifest_kind": "modern_external_baseline_formal_reference_decision",
                "baseline_id": "sigmark",
                "formal_reference_decision": "FAIL",
                "formal_reference_status": "official_reference_failures_present",
            }
        ),
        encoding="utf-8",
    )
    package_dir = drive_root / "stage_packages" / "validation_scale" / "external_baseline_formal_reference_sigmark"
    package_dir.mkdir(parents=True)
    (package_dir / "stage_package_latest.zip").write_bytes(b"stale")
    (package_dir / "stage_package__old.zip").write_bytes(b"stale")
    (package_dir / "stage_package__old_stage_package_manifest.json").write_text("{}", encoding="utf-8")
    bundle_root = tmp_path / "workspace" / "external_baseline_official_result_bundles" / "validation_scale"
    bundle_root.mkdir(parents=True)
    (bundle_root / "large_failed_output.bin").write_bytes(b"x" * 1024)
    layout = {
        "drive_project_root": str(drive_root),
        "workflow_profile": "validation_scale",
        "stage_package_id": "external_baseline_formal_reference_sigmark",
        "drive_run_root": str(run_root),
        "external_baseline_official_result_bundle_root": str(bundle_root),
        "local_stage_package_cache_root": str(tmp_path / "local_cache"),
        "local_stage_workspace_root": str(tmp_path / "workspace"),
    }

    result = publish_colab_stage_package(
        layout,
        notebook_role="external_baseline_formal_scoring",
        baseline_id="sigmark",
        include_videos=True,
    )

    latest_manifest = package_dir / "stage_package_latest_manifest.json"
    manifest = json.loads(latest_manifest.read_text(encoding="utf-8"))
    assert result["stage_package_publish_status"] == "skipped_failed_external_baseline_reference"
    assert manifest["stage_package_publish_status"] == "skipped_failed_external_baseline_reference"
    assert not (package_dir / "stage_package_latest.zip").exists()
    assert not list(package_dir.glob("stage_package__*.zip"))
    assert latest_stage_package_zip(drive_root, "validation_scale", "external_baseline_formal_reference_sigmark") is None


@pytest.mark.quick
def test_external_baseline_restore_requires_pass_decision_in_stage_manifest(tmp_path: Path) -> None:
    """历史 external baseline zip 若没有 PASS 决策, 不得被后续 Notebook 恢复。"""

    drive_root = tmp_path / "drive" / "SSTW"
    package_dir = drive_root / "stage_packages" / "validation_scale" / "external_baseline_formal_reference_sigmark"
    package_dir.mkdir(parents=True)
    with zipfile.ZipFile(package_dir / "stage_package_latest.zip", mode="w") as archive:
        archive.writestr("runs/validation_scale/records/legacy.json", "{}")
    (package_dir / "stage_package_latest_manifest.json").write_text(
        json.dumps(
            {
                "manifest_kind": "colab_stage_zip_handoff_manifest",
                "stage_package_publish_status": "published",
            }
        ),
        encoding="utf-8",
    )

    assert latest_stage_package_zip(drive_root, "validation_scale", "external_baseline_formal_reference_sigmark") is None

    (package_dir / "stage_package_latest_manifest.json").write_text(
        json.dumps(
            {
                "manifest_kind": "colab_stage_zip_handoff_manifest",
                "stage_package_publish_status": "published",
                "formal_reference_decision": "PASS",
            }
        ),
        encoding="utf-8",
    )
    assert latest_stage_package_zip(drive_root, "validation_scale", "external_baseline_formal_reference_sigmark") == (
        package_dir / "stage_package_latest.zip"
    )


@pytest.mark.quick
def test_legacy_drive_packager_is_noop_in_stage_zip_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """local_zip 模式下旧版 packages/ 打包只能成为 no-op, 实际落盘由阶段 zip 完成。"""

    monkeypatch.delenv("SSTW_WRITE_LEGACY_DRIVE_PACKAGE_IN_STAGE_ZIP_MODE", raising=False)
    command = build_drive_packaging_command(
        {
            "stage_package_handoff_mode": "local_zip",
            "drive_run_root": "/content/SSTW_stage_workspace/runs/validation_scale",
            "drive_package_dir": "/content/drive/MyDrive/SSTW/packages/generative_video_model_probe/validation_scale",
        }
    )

    assert command[1] == "-c"
    assert "skip legacy drive packager" in command[2]
    assert "generative_video_drive_packager.py" in command[2]
