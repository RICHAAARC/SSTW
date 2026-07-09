"""审计 trajectory_observation_core_probe trajectory observation core probe。"""

from __future__ import annotations

from statistics import mean

from experiments.trajectory_observation_core.correlation_audit import audit_correlation

ATTACKED_POSITIVE_ROLES = {"attacked_positive"}


def _mean_score(records: list[dict], method_variant: str, sample_role: str = "attacked_positive") -> float:
    values = [float(record["S_final"]) for record in records if record["split"] == "test" and record["sample_role"] == sample_role and record["method_variant"] == method_variant]
    return mean(values) if values else 0.0


def _mean_trajectory(records: list[dict], sample_role: str) -> float:
    values = [float(record["S_trajectory_observation"]) for record in records if record["method_variant"] == "key_conditioned_state_space_with_trajectory" and record["sample_role"] == sample_role and record.get("S_trajectory_observation") is not None]
    return mean(values) if values else 0.0


def audit_mechanism(records: list[dict], control_records: list[dict], config: dict) -> dict:
    """返回 trajectory_observation_core_probe 机制审计结果。"""
    core_score = _mean_score(records, "key_conditioned_state_space_inference")
    trajectory_score = _mean_score(records, "key_conditioned_state_space_with_trajectory")
    trajectory_gain = round(trajectory_score - core_score, 6)
    negative_core_decisions = [1.0 if record["decision"] == "positive" else 0.0 for record in records if record["split"] == "test" and record["sample_role"] == "attacked_negative" and record["method_variant"] == "key_conditioned_state_space_inference"]
    negative_traj_decisions = [1.0 if record["decision"] == "positive" else 0.0 for record in records if record["split"] == "test" and record["sample_role"] == "attacked_negative" and record["method_variant"] == "key_conditioned_state_space_with_trajectory"]
    negative_core = mean(negative_core_decisions) if negative_core_decisions else 0.0
    negative_traj = mean(negative_traj_decisions) if negative_traj_decisions else 0.0
    negative_leakage_delta = round(max(negative_traj - negative_core, 0.0), 6)
    positive_distribution = _mean_trajectory(records, "attacked_positive")
    negative_distribution = _mean_trajectory(records, "attacked_negative")
    statistical_separation = positive_distribution > negative_distribution
    correlation = audit_correlation(records, float(config["correlation_threshold"]))
    control_suppression_status = "PASS" if control_records and all(record["control_status"] == "suppressed" for record in control_records) else "FAIL"
    attacked_negative = [record for record in records if record["split"] == "test" and record["sample_role"] == "attacked_negative"]
    attacked_negative_fpr = mean(1.0 if record["decision"] == "positive" else 0.0 for record in attacked_negative) if attacked_negative else 0.0
    runtime_overhead_status = "PASS" if all(float(record["trajectory_runtime_sec"]) < float(config["runtime_overhead_blocking_sec"]) for record in records if record.get("trajectory_runtime_sec") is not None) else "BLOCKING"
    source_statuses = {record["trajectory_source_status"] for record in records if record["method_variant"] == "key_conditioned_state_space_with_trajectory"}
    source_not_only_surrogate = source_statuses != {"surrogate"}
    mechanism_pass = all([trajectory_gain > 0, negative_leakage_delta <= float(config["negative_leakage_tolerance"]), statistical_separation, correlation["correlation_status"] == "PASS", control_suppression_status == "PASS", attacked_negative_fpr <= 0.02, runtime_overhead_status != "BLOCKING"])
    top_conference_gate = mechanism_pass and source_not_only_surrogate
    return {
        "trajectory_observation_mechanism_decision": "PASS" if mechanism_pass else "FAIL",
        "mechanism_pass": mechanism_pass,
        "top_conference_trajectory_gate": "PASS" if top_conference_gate else "FAIL",
        "trajectory_gain_over_state_space": trajectory_gain,
        "trajectory_negative_leakage_delta": negative_leakage_delta,
        "trajectory_positive_distribution_mean": round(positive_distribution, 6),
        "trajectory_negative_distribution_mean": round(negative_distribution, 6),
        "trajectory_statistical_separation": "PASS" if statistical_separation else "FAIL",
        "control_suppression_status": control_suppression_status,
        "attacked_negative_fpr": round(attacked_negative_fpr, 6),
        "runtime_overhead_status": runtime_overhead_status,
        "trajectory_source_status": "not_only_surrogate" if source_not_only_surrogate else "only_surrogate",
        **correlation,
    }
