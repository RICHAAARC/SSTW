"""验证 pilot_paper 旧模块只保留公共 gate 兼容转发。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.generative_video_model_probe.paper_profile_gate import (
    build_paper_profile_gate_audit,
)
from experiments.generative_video_model_probe.pilot_paper_gate import (
    build_pilot_paper_gate_audit,
    write_pilot_paper_gate_audit,
)


PILOT_CONFIG = Path("configs/protocol/pilot_paper_generative_probe.json")
PROBE_CONFIG = Path("configs/protocol/probe_paper_generative_probe.json")


@pytest.mark.quick
def test_pilot_compatibility_builder_is_exact_public_gate_delegation(
    tmp_path: Path,
) -> None:
    """兼容 builder 不得增加、删除或改写任何 pilot 专属结论条件。"""

    compatibility = build_pilot_paper_gate_audit(tmp_path, PILOT_CONFIG)
    public = build_paper_profile_gate_audit(tmp_path, PILOT_CONFIG)

    assert compatibility == public
    assert compatibility["paper_result_level"] == "pilot_paper"
    assert compatibility["target_fpr"] == pytest.approx(0.01)
    assert compatibility["paper_profile_gate_decision"] == "FAIL"
    assert compatibility["pilot_paper_gate_decision"] == "FAIL"


@pytest.mark.quick
def test_pilot_compatibility_entry_rejects_other_profile_config(
    tmp_path: Path,
) -> None:
    """旧模块名称不得成为用 probe 配置冒充 pilot 结果的旁路。"""

    with pytest.raises(ValueError, match="只接受 pilot_paper"):
        build_pilot_paper_gate_audit(tmp_path, PROBE_CONFIG)


@pytest.mark.quick
def test_pilot_compatibility_writer_emits_public_and_profile_aliases(
    tmp_path: Path,
) -> None:
    """公共 writer 必须同时保留服务器可能消费的 pilot 命名产物。"""

    audit = write_pilot_paper_gate_audit(tmp_path, PILOT_CONFIG)
    common = json.loads(
        (tmp_path / "artifacts" / "paper_profile_gate_decision.json").read_text(
            encoding="utf-8"
        )
    )
    pilot = json.loads(
        (tmp_path / "artifacts" / "pilot_paper_gate_decision.json").read_text(
            encoding="utf-8"
        )
    )

    assert audit == common == pilot
    assert (tmp_path / "records" / "paper_profile_gate_records.jsonl").is_file()
    assert (tmp_path / "records" / "pilot_paper_gate_records.jsonl").is_file()
    assert (tmp_path / "reports" / "pilot_paper_gate_report.md").is_file()


@pytest.mark.quick
def test_pilot_compatibility_module_contains_no_second_gate_implementation() -> None:
    """pilot 兼容层必须保持轻薄, 防止以后重新引入平行门禁语义。"""

    source = Path(
        "experiments/generative_video_model_probe/pilot_paper_gate.py"
    ).read_text(encoding="utf-8")

    assert "build_paper_profile_gate_audit" in source
    assert "write_paper_profile_gate_audit" in source
    assert len(source.splitlines()) < 100
    assert "minimum_pilot_paper_" not in source
