"""审计 real_video_latent_transfer_check real video latent transfer check 的机制条件。"""

from __future__ import annotations

from statistics import mean

NON_UNIFORM_TEMPORAL_ATTACKS = {"temporal_crop", "local_clip", "irregular_frame_dropping", "frame_duplication", "speed_change", "frame_rate_resampling"}
TEMPORAL_ATTACKS = NON_UNIFORM_TEMPORAL_ATTACKS | {"regular_frame_dropping"}


def _mean_score(records: list[dict], method_variant: str, attack_name: str, sample_role: str) -> float:
    values = [float(record["S_final"]) for record in records if record["split"] == "test" and record["method_variant"] == method_variant and record["attack_name"] == attack_name and record["sample_role"] == sample_role]
    if not values:
        raise ValueError(f"missing records for {method_variant} {attack_name} {sample_role}")
    return mean(values)


def audit_mechanism(records: list[dict], target_fpr: float, quality_records: list[dict]) -> dict:
    """返回 real_video_latent_transfer_check 机制审计结果。"""
    state_beats_tubelet = all(_mean_score(records, "key_conditioned_state_space_inference", attack, "attacked_positive") > _mean_score(records, "tubelet_only", attack, "attacked_positive") for attack in TEMPORAL_ATTACKS)
    state_beats_explicit_attacks = [attack for attack in NON_UNIFORM_TEMPORAL_ATTACKS if _mean_score(records, "key_conditioned_state_space_inference", attack, "attacked_positive") > _mean_score(records, "explicit_temporal_alignment", attack, "attacked_positive")]
    state_beats_key_agnostic = all(_mean_score(records, "key_conditioned_state_space_inference", attack, "attacked_positive") > _mean_score(records, "key_agnostic_state_space_model", attack, "attacked_positive") for attack in TEMPORAL_ATTACKS)
    attacked_negative = [record for record in records if record["split"] == "test" and record["sample_role"] == "attacked_negative"]
    attacked_negative_fpr = mean(1.0 if record["decision"] == "positive" else 0.0 for record in attacked_negative)
    negative_state_over_threshold_count = sum(1 for record in records if record["sample_role"].endswith("negative") and record["method_variant"] == "key_conditioned_state_space_inference" and record["S_final"] >= record["threshold_value"])
    admissibility_pass = all(record["key_state_admissibility_status"] in {"pass", "blocked_negative_tail"} for record in records if record["method_variant"] == "key_conditioned_state_space_inference")
    quality_not_collapsed_pass = all(record["quality_not_collapsed"] == "PASS" for record in quality_records)
    temporal_consistency_pass = all(record["temporal_consistency_not_collapsed"] == "PASS" for record in quality_records)
    mechanism_pass = all([state_beats_tubelet, len(state_beats_explicit_attacks) >= 1, state_beats_key_agnostic, attacked_negative_fpr <= max(2 * target_fpr, target_fpr + 1 / max(len(attacked_negative), 1)), negative_state_over_threshold_count == 0, admissibility_pass, quality_not_collapsed_pass, temporal_consistency_pass])
    return {
        "mechanism_pass": mechanism_pass,
        "state_beats_tubelet_under_temporal_attacks": state_beats_tubelet,
        "state_beats_explicit_non_uniform_attack_count": len(state_beats_explicit_attacks),
        "state_beats_key_agnostic": state_beats_key_agnostic,
        "attacked_negative_fpr": round(attacked_negative_fpr, 6),
        "negative_state_over_threshold_count": negative_state_over_threshold_count,
        "key_state_admissibility_status": "PASS" if admissibility_pass else "FAIL",
        "quality_not_collapsed": "PASS" if quality_not_collapsed_pass else "FAIL",
        "temporal_consistency_not_collapsed": "PASS" if temporal_consistency_pass else "FAIL",
    }
