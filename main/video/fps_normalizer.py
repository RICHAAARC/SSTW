"""提供 B2 fps 归一化接口。"""

from __future__ import annotations


def normalize_fps_status(video_fps: int) -> str:
    """返回 fps 归一化状态。"""
    return "pass" if video_fps > 0 else "failed"
