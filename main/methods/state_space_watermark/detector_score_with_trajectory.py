"""提供带 trajectory observation 的检测分数计算。"""

from __future__ import annotations

from main.methods.state_space_watermark.formal_interface import run_formal_inference
from main.methods.state_space_watermark.trajectory_state_observation import fuse_trajectory_into_state_score
from main.trajectory.trajectory_observation import compute_trajectory_observation


def score_with_trajectory(sample_role: str, attack_name: str, method_variant: str, sample_index: int) -> dict:
    """计算 B4 方法变体的检测分数。"""
    if method_variant == "key_conditioned_state_space_inference":
        base = run_formal_inference(sample_role, attack_name, "key_conditioned_state_space_inference")
        base["S_trajectory_observation"] = None
        base["S_traj_state"] = None
        return base

    if method_variant == "explicit_temporal_alignment_with_trajectory_fusion":
        base = run_formal_inference(sample_role, attack_name, "no_state_inference")
    elif method_variant == "generic_state_space_with_trajectory":
        base = run_formal_inference(sample_role, attack_name, "generic_state_space_model")
    elif method_variant in {"trajectory_only", "trajectory_random_key_control", "trajectory_time_shuffled_control", "trajectory_direction_shuffled_control", "trajectory_observation_without_key_condition"}:
        base = run_formal_inference(sample_role, attack_name, "key_conditioned_state_space_inference")
        base["S_final"] = 0.0
        base["S_state_posterior"] = 0.0
        base["S_payload_state"] = base["S_payload_raw"]
    else:
        base = run_formal_inference(sample_role, attack_name, "key_conditioned_state_space_inference")

    trajectory_score = compute_trajectory_observation(sample_role, attack_name, method_variant, sample_index)
    traj_state = fuse_trajectory_into_state_score(float(base["S_state_posterior"]), trajectory_score, method_variant)
    result = dict(base)
    result["S_trajectory_observation"] = trajectory_score
    result["S_traj_state"] = traj_state
    result["S_final"] = traj_state
    return result
