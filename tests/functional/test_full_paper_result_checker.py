"""验证 full_paper 旧 checker 只复用公共 profile gate。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.generative_video_model_probe.paper_profile_gate import (
    build_paper_profile_gate_audit,
)
from scripts.check_results.full_paper_result_checker import (
    build_full_paper_result_checker_audit,
    write_full_paper_result_checker_audit,
)


FULL_CONFIG = Path("configs/protocol/full_paper_generative_probe.json")
PILOT_CONFIG = Path("configs/protocol/pilot_paper_generative_probe.json")


@pytest.mark.quick
def test_full_checker_is_public_gate_with_legacy_aliases(tmp_path: Path) -> None:
    """旧 checker 只能增加字段别名, 不得改变公共 gate 的任何结论。"""

    compatibility = build_full_paper_result_checker_audit(tmp_path, FULL_CONFIG)
    public = build_paper_profile_gate_audit(tmp_path, FULL_CONFIG)

    assert {
        key: value
        for key, value in compatibility.items()
        if key
        not in {
            "full_paper_result_checker_decision",
            "full_paper_result_decision",
        }
    } == public
    assert compatibility["paper_result_level"] == "full_paper"
    assert compatibility["target_fpr"] == pytest.approx(0.001)
    assert compatibility["full_paper_result_checker_decision"] == (
        public["paper_profile_gate_decision"]
    )
    assert compatibility["full_paper_result_decision"] == (
        public["paper_profile_gate_decision"]
    )


@pytest.mark.quick
def test_full_checker_rejects_non_full_profile_config(tmp_path: Path) -> None:
    """旧 checker 名称不得成为 pilot 结果冒充 full_paper 的旁路。"""

    with pytest.raises(ValueError, match="只接受 full_paper"):
        build_full_paper_result_checker_audit(tmp_path, PILOT_CONFIG)


@pytest.mark.quick
def test_full_checker_writer_preserves_legacy_files_without_parallel_logic(
    tmp_path: Path,
) -> None:
    """旧消费者需要的文件应与公共 full gate 决策完全一致。"""

    audit = write_full_paper_result_checker_audit(tmp_path, FULL_CONFIG)
    public = json.loads(
        (tmp_path / "artifacts" / "full_paper_gate_decision.json").read_text(
            encoding="utf-8"
        )
    )
    legacy = json.loads(
        (
            tmp_path
            / "artifacts"
            / "full_paper_result_checker_decision.json"
        ).read_text(encoding="utf-8")
    )

    assert audit == legacy
    assert legacy["paper_profile_gate_decision"] == public[
        "paper_profile_gate_decision"
    ]
    assert (tmp_path / "records" / "full_paper_result_checker_records.jsonl").is_file()
    assert (tmp_path / "artifacts" / "full_paper_result_decision.json").is_file()
    assert (tmp_path / "reports" / "full_paper_result_checker_report.md").is_file()


@pytest.mark.quick
def test_full_checker_compatibility_module_remains_thin() -> None:
    """full 兼容层不得重新引入独立的样本、攻击或 proxy 审计实现。"""

    source = Path("scripts/check_results/full_paper_result_checker.py").read_text(
        encoding="utf-8"
    )

    assert "build_paper_profile_gate_audit" in source
    assert "write_paper_profile_gate_audit" in source
    assert "required_runtime_attack_names_from_config" not in source
    assert len(source.splitlines()) < 140
