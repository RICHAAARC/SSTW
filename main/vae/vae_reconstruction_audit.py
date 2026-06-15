"""提供 B2 VAE 重建质量代理审计。"""

from __future__ import annotations


def vae_reconstruction_metrics(severity: float) -> dict[str, float | str]:
    """根据攻击强度生成确定性的 VAE 重建质量代理指标。"""
    return {
        "vae_reconstruction_psnr": round(38.0 - severity * 16.0, 6),
        "vae_reconstruction_ssim": round(0.97 - severity * 0.22, 6),
        "vae_reconstruction_lpips_status": "disabled",
    }
