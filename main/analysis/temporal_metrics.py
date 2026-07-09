"""提供 real_video_latent_transfer_check 时间一致性指标代理计算。"""

from __future__ import annotations


def compute_temporal_metrics(severity: float) -> dict[str, float | str | None]:
    """根据攻击强度生成确定性的时序一致性指标。"""
    return {
        "temporal_flicker_score": round(0.05 + severity * 0.35, 6),
        "motion_consistency_score_placeholder": None,
        "motion_consistency_status": "placeholder_until_generative_video_model_probe",
        "motion_consistency_reason": "optical_flow_metric_is_deferred_to_generation_stage",
    }
