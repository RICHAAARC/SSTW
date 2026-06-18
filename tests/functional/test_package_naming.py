"""验证 Google Drive package 批次命名规则。"""

from __future__ import annotations

import re

import pytest

from main.protocol.package_naming import build_package_batch_id, build_package_file_stem, sanitize_filename_token


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
