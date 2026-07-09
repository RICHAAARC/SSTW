"""汇总 real_video_latent_transfer_check 质量和时间一致性 gate。"""

from __future__ import annotations


def quality_not_collapsed(quality_psnr: float, quality_ssim: float, psnr_floor: float, ssim_floor: float) -> bool:
    """判断质量指标是否未崩溃。"""
    return quality_psnr >= psnr_floor and quality_ssim >= ssim_floor


def temporal_consistency_not_collapsed(temporal_flicker_score: float, flicker_ceiling: float) -> bool:
    """判断时间一致性指标是否未崩溃。"""
    return temporal_flicker_score <= flicker_ceiling
