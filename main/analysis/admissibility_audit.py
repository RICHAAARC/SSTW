"""分析 state_space_inference_formalization admissibility negative tail。"""

from __future__ import annotations


def admissibility_negative_tail_status(records: list[dict]) -> str:
    """判断启用 admissibility 的负样本是否存在越阈救回。"""
    violations = [record for record in records if record["method_variant"] == "key_conditioned_state_space_inference" and record["sample_role"].endswith("negative") and record["S_final"] >= record["threshold_value"]]
    return "PASS" if not violations else "FAIL"
