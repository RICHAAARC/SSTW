"""审计 B4 trajectory observation 与 static evidence 的非冗余性。"""

from __future__ import annotations

from main.trajectory.trajectory_statistic import centered_correlation


def audit_correlation(records: list[dict], correlation_threshold: float) -> dict:
    """返回 trajectory 与 payload/state 的中心化相关性审计。"""
    main_records = [record for record in records if record["method_variant"] == "key_conditioned_state_space_with_trajectory"]
    payload_corr = centered_correlation(main_records, "S_trajectory_observation", "S_payload_state")
    state_corr = centered_correlation(main_records, "S_trajectory_observation", "S_state_posterior")
    status = "PASS" if abs(payload_corr) < correlation_threshold and abs(state_corr) < correlation_threshold else "FAIL"
    return {
        "trajectory_payload_correlation": payload_corr,
        "trajectory_state_correlation": state_corr,
        "correlation_threshold": correlation_threshold,
        "correlation_status": status,
    }
