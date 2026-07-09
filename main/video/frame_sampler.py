"""提供 real_video_latent_transfer_check 视频帧采样接口。"""

from __future__ import annotations


def frame_sample_status(video_num_frames: int) -> str:
    """返回帧采样状态。"""
    return "pass" if video_num_frames > 0 else "failed"
