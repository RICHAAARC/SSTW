"""验证 Flow trajectory 协议字段闭合。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from main.protocol.flow_evidence_fields import conservative_flow_score, flow_evidence_protocol_defaults


REQUIRED_FLOW_FIELDS = {
    "negative_family",
    "sampler_signature_placeholder",
    "trajectory_source_level",
    "S_path_inv",
    "S_velocity",
    "S_final_conservative",
    "path_marginal_gain_at_fixed_fpr",
    "replay_uncertainty_mean",
    "flow_state_admissibility_status",
    "claim_support_status",
}


@pytest.mark.quick
def test_flow_evidence_defaults_cover_protocol_fields() -> None:
    """默认字段集合必须覆盖后续真实 GPU records 需要承接的协议字段。"""
    record = flow_evidence_protocol_defaults()
    assert REQUIRED_FLOW_FIELDS <= set(record)
    assert record["sampler_signature_placeholder"] is None
    assert record["claim_support_status"].startswith("not_supported")


@pytest.mark.quick
def test_conservative_flow_score_uses_lowest_available_evidence() -> None:
    """保守分数必须取 endpoint、path 和 velocity 中的最低可用证据。"""
    record = {"S_final": 0.8, "S_path_inv": 0.4, "S_velocity": 0.6}
    assert conservative_flow_score(record) == 0.4


@pytest.mark.constraint
def test_event_schema_requires_flow_protocol_fields() -> None:
    """event record schema 必须显式要求 Flow trajectory 协议字段。"""
    schema = json.loads(Path("configs/records/event_record_schema.json").read_text(encoding="utf-8"))
    assert REQUIRED_FLOW_FIELDS <= set(schema["required_fields"])
