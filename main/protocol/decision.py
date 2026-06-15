"""生成第一阶段 implementation 与 mechanism 决策。"""

from __future__ import annotations


def build_stage_decision(implementation_pass: bool, mechanism_pass: bool, details: dict) -> dict:
    """构造阶段决策对象。"""
    return {"stage_id": "synthetic_state_protocol", "implementation_decision": "PASS" if implementation_pass else "FAIL", "mechanism_decision": "PASS" if mechanism_pass else "FAIL", "details": details}
