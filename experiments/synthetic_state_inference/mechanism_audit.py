"""审计 synthetic state inference 的第一阶段机制条件。"""

from __future__ import annotations

from collections import defaultdict
from statistics import mean

COMPLEX_TEMPORAL_ATTACKS = {"temporal_crop", "local_clip", "irregular_frame_dropping", "frame_duplication", "speed_change", "frame_rate_resampling", "segment_jump"}


def _mean_score(records: list[dict], method_variant: str, attack_name: str, sample_role: str) -> float:
    values = [float(record["S_final"]) for record in records if record["split"] == "test" and record["method_variant"] == method_variant and record["attack_name"] == attack_name and record["sample_role"] == sample_role]
    if not values:
        raise ValueError(f"missing records for {method_variant} {attack_name} {sample_role}")
    return mean(values)


def audit_mechanism(records: list[dict], target_fpr: float) -> dict:
    """返回第一阶段机制审计结果。"""
    tubelet_beats_frame = all(_mean_score(records, "tubelet_only", attack, "attacked_positive") > _mean_score(records, "frame_prc", attack, "attacked_positive") for attack in COMPLEX_TEMPORAL_ATTACKS)
    state_beats_generic_attacks = [attack for attack in COMPLEX_TEMPORAL_ATTACKS if _mean_score(records, "key_conditioned_state_space_inference", attack, "attacked_positive") > _mean_score(records, "generic_state_space_model", attack, "attacked_positive")]
    state_beats_key_agnostic = all(_mean_score(records, "key_conditioned_state_space_inference", attack, "attacked_positive") > _mean_score(records, "key_agnostic_state_space_model", attack, "attacked_positive") for attack in COMPLEX_TEMPORAL_ATTACKS)
    aggregator_methods = ["conv1d_temporal_aggregator", "gru_temporal_aggregator", "transformer_temporal_aggregator"]
    state_beats_aggregators = any(all(_mean_score(records, "key_conditioned_state_space_inference", attack, "attacked_positive") > _mean_score(records, method, attack, "attacked_positive") for method in aggregator_methods) for attack in COMPLEX_TEMPORAL_ATTACKS)
    attacked_negative = [record for record in records if record["split"] == "test" and record["sample_role"] == "attacked_negative"]
    attacked_negative_fpr = mean(1.0 if record["decision"] == "positive" else 0.0 for record in attacked_negative)
    negative_state_over_threshold_count = sum(1 for record in records if record["sample_role"].endswith("negative") and record["method_variant"] == "key_conditioned_state_space_inference" and record["S_final"] >= record["threshold_value"])
    entropy_by_outcome: dict[str, list[float]] = defaultdict(list)
    for record in records:
        if record["split"] == "test" and record["sample_role"].endswith("positive"):
            entropy_by_outcome["success" if record["decision"] == "positive" else "failure"].append(float(record["state_entropy"]))
    entropy_trend_documented = bool(entropy_by_outcome["success"])
    mechanism_pass = all([tubelet_beats_frame, len(state_beats_generic_attacks) >= 2, state_beats_key_agnostic, state_beats_aggregators, attacked_negative_fpr <= max(2 * target_fpr, target_fpr + 1 / max(len(attacked_negative), 1)), negative_state_over_threshold_count == 0, entropy_trend_documented])
    return {"mechanism_pass": mechanism_pass, "tubelet_beats_frame": tubelet_beats_frame, "state_beats_generic_attack_count": len(state_beats_generic_attacks), "state_beats_key_agnostic": state_beats_key_agnostic, "state_beats_aggregators": state_beats_aggregators, "attacked_negative_fpr": round(attacked_negative_fpr, 6), "negative_state_over_threshold_count": negative_state_over_threshold_count, "entropy_trend_documented": entropy_trend_documented}
