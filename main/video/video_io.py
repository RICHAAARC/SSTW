"""提供 real_video_latent_transfer_check 阶段真实视频代理样本的读取接口。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VideoSample:
    """表示一个真实视频 transfer check 的轻量元数据样本。"""

    source_video_id: str
    dataset_id: str
    split: str
    sample_role: str
    key_id: str
    content_id: str
    video_fps: int
    video_num_frames: int
    video_resolution: str
    video_duration_sec: float


def build_video_samples(config: dict, splits: tuple[str, ...], sample_roles: tuple[str, ...]) -> list[VideoSample]:
    """构造 real_video_latent_transfer_check 使用的轻量真实视频代理样本清单。

    该函数不读取真实大文件, 只冻结真实视频链路所需的治理字段。后续可以将此接口替换为
    真实视频 reader, 但 records schema 和 split 语义保持不变。
    """
    samples: list[VideoSample] = []
    for split in splits:
        for sample_role in sample_roles:
            for index in range(int(config["sample_count_per_cell"])):
                samples.append(VideoSample(
                    source_video_id=f"real_video_proxy_{index:04d}",
                    dataset_id=config["dataset_id"],
                    split=split,
                    sample_role=sample_role,
                    key_id="negative_key" if sample_role.endswith("negative") else "key_alpha",
                    content_id=f"real_content_proxy_{index:04d}",
                    video_fps=int(config["video_fps"]),
                    video_num_frames=int(config["video_num_frames"]),
                    video_resolution=config["video_resolution"],
                    video_duration_sec=float(config["video_duration_sec"]),
                ))
    return samples
