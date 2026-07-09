"""验证 Google Drive package 批次命名规则。"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from main.protocol.package_naming import build_package_batch_id, build_package_file_stem, sanitize_filename_token
from scripts.package_results.drive_package_paths import (
    archive_run_root_for_stage,
    build_packager_file_stem,
    packager_manifest_filename,
    resolve_stage_package_output_dir,
)


@pytest.mark.quick
def test_package_batch_id_uses_utc_time_and_short_commit() -> None:
    """package batch ID 必须使用 `<utc_time>_<short_commit>` 形式。"""
    batch_id = build_package_batch_id("20260618_004044", "abc123ef")

    assert batch_id == "20260618_004044_abc123ef"


@pytest.mark.quick
def test_package_file_stem_prefixes_batch_id() -> None:
    """zip 和 manifest 的文件名前缀必须共享同一个 batch ID。"""
    stem = build_package_file_stem("sampling_time_constraint_colab", "20260618_004044", "abc123ef")

    assert stem == "sampling_time_constraint_colab_20260618_004044_abc123ef"
    assert re.match(r"^[a-z0-9_]+_\d{8}_\d{6}_[a-z0-9_\-]+$", stem)


@pytest.mark.quick
def test_package_filename_token_sanitizer_keeps_snake_case() -> None:
    """文件名 token 必须稳定落到可审计的 snake_case 兼容形式。"""
    assert sanitize_filename_token("Wan2.1 Package/ABC") == "wan2_1_package_abc"


@pytest.mark.quick
@pytest.mark.quick
def test_independent_packager_stage_file_names_match_stage_zip_policy() -> None:
    """独立 packager 在阶段命名模式下必须使用 profile + stage + 时间戳 + commit。"""

    stem = build_packager_file_stem(
        "generative_video_generation",
        "20260701_030405",
        "abc123ef",
        workflow_profile="probe_paper",
        stage_package_id="generative_video_generation_colab",
    )

    assert stem == "probe_paper_generative_video_generation_colab_20260701_030405_abc123ef"
    assert packager_manifest_filename(stem, stage_package_naming=True).endswith("_manifest.json")
    assert packager_manifest_filename(stem, stage_package_naming=False).endswith("_package_manifest.json")
    assert archive_run_root_for_stage(
        "probe_paper",
        workflow_profile="probe_paper",
        stage_package_id="generative_video_generation_colab",
    ) == "runs/generative_video_model_probe/probe_paper"
    assert archive_run_root_for_stage(
        "sampling_time_constraint_colab",
        workflow_profile="sampling_time_constraint",
        stage_package_id="sampling_time_constraint_colab",
    ) == "runs/sampling_time_constraint_colab"


@pytest.mark.quick
def test_independent_packager_cli_defaults_do_not_reference_legacy_packages_dir() -> None:
    """独立 packager CLI 默认值不得再包含旧版 SSTW/packages 入口。"""

    for path in (
        Path("scripts/package_results/generative_video_drive_packager.py"),
        Path("scripts/package_results/sampling_time_constraint_drive_packager.py"),
        Path("scripts/package_results/wan21_flow_adapter_preflight_drive_packager.py"),
    ):
        source = path.read_text(encoding="utf-8")
        assert f"{'{'}DEFAULT_DRIVE_PROJECT_ROOT{'}'}/packages" not in source
        assert "/packages/" not in source
