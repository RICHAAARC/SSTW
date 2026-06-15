"""提供 B2 real video VAE latent transfer 的轻量 backend。"""

from __future__ import annotations

from main.methods.state_space_watermark.score import score_method

REAL_VIDEO_METHOD_PENALTY = {
    "frame_prc": 0.10,
    "tubelet_only": 0.06,
    "explicit_temporal_alignment": 0.05,
    "generic_temporal_mean_pooling": 0.07,
    "transformer_temporal_aggregator": 0.06,
    "generic_state_space_model": 0.04,
    "key_agnostic_state_space_model": 0.04,
    "key_conditioned_state_space_inference": 0.025,
    "key_conditioned_state_space_without_admissibility": 0.025,
}


def score_real_video_transfer(sample_role: str, attack_name: str, method_variant: str, severity: float):
    """计算真实视频 VAE latent transfer proxy 下的检测结果。

    该实现沿用 B1 的状态空间检测结构, 只加入 VAE 重建和真实视频攻击导致的轻量退化。
    """
    result = score_method(sample_role, attack_name if attack_name in {"no_attack", "temporal_crop", "local_clip", "regular_frame_dropping", "irregular_frame_dropping", "frame_duplication", "speed_change", "frame_rate_resampling"} else "latent_gaussian_noise", method_variant)
    penalty = REAL_VIDEO_METHOD_PENALTY.get(method_variant, 0.05) * severity
    final_score = max(result.final_score - penalty, 0.0)
    return {
        "S_payload_raw": round(max(result.payload_raw - penalty * 0.5, 0.0), 6),
        "S_payload_state": round(max(result.payload_state - penalty * 0.4, 0.0), 6),
        "S_state_posterior": round(max(result.state_posterior - penalty, 0.0), 6),
        "S_final": round(final_score, 6),
        "state_entropy": round(result.state_entropy + severity * 0.15, 6),
        "state_coverage_ratio": round(max(result.state_coverage_ratio - severity * 0.10, 0.0), 6),
        "state_matched_count": result.state_matched_count,
        "state_transition_residual": round(result.state_transition_residual + severity * 0.10, 6),
        "key_state_admissibility_status": result.admissibility_status,
    }
