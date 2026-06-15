"""提供 B2 质量指标代理计算。"""

from __future__ import annotations


def compute_quality_metrics(severity: float) -> dict[str, float | str]:
    """根据攻击强度生成确定性的质量指标。"""
    return {
        "quality_psnr": round(36.0 - severity * 18.0, 6),
        "quality_ssim": round(0.96 - severity * 0.25, 6),
        "quality_lpips": None,
        "quality_metric_status": "enabled",
        "quality_metric_failure_reason": "quality_lpips_disabled_in_lightweight_proxy",
    }
