"""B5 攻击矩阵状态构建。"""

from __future__ import annotations

ATTACK_NAMES = (
    "no_attack", "h264_compression", "h265_compression", "spatial_resize", "crop_resize",
    "temporal_crop", "local_clip", "regular_frame_dropping", "irregular_frame_dropping",
    "frame_duplication", "speed_change", "frame_rate_resampling", "gaussian_noise", "blur",
)


def build_attack_status_records(runnable_status: str) -> list[dict]:
    """生成攻击矩阵状态记录, 未生成视频时只记录 not_run。"""
    records = []
    for attack_name in ATTACK_NAMES:
        records.append({
            "attack_name": attack_name,
            "attack_failure_status": "not_run" if runnable_status != "runnable" else "pending_runtime",
            "attack_failure_reason": "generation_model_not_runnable" if runnable_status != "runnable" else "none",
        })
    return records
