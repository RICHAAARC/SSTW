"""在模型原生 Flow latent 与 SSTW 五维 tubelet 坐标之间执行可逆转换。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class FlowLatentLayout(Protocol):
    """定义模型 latent 与 SSTW 规范五维 latent 之间的可逆接口。

    通用工程价值在于把不同生成模型的张量排布差异隔离在适配器中。SSTW 的
    tubelet 密钥、endpoint 检测和路径投影始终在 ``[B, C, T, H, W]`` 坐标中
    定义, 不会因为第三方 pipeline 改用 token 序列而改变算法语义。
    """

    layout_id: str

    def to_canonical(self, latent: Any) -> Any:
        """把模型原生 latent 转换为 SSTW 五维规范坐标。"""

    def from_canonical(self, latent: Any) -> Any:
        """把 SSTW 五维规范坐标转换回模型原生坐标。"""

    def as_dict(self) -> dict[str, Any]:
        """返回可写入 governed record 的布局元数据。"""


@dataclass(frozen=True)
class FiveDimensionalFlowLatentLayout:
    """表示模型原生 latent 已经是 ``[B, C, T, H, W]`` 的恒等布局。"""

    layout_id: str = "five_dimensional_flow_latent"

    @staticmethod
    def _validate(latent: Any) -> None:
        if getattr(latent, "ndim", None) != 5:
            raise ValueError("五维 Flow latent 布局要求 [B, C, T, H, W] 张量")

    def to_canonical(self, latent: Any) -> Any:
        self._validate(latent)
        return latent

    def from_canonical(self, latent: Any) -> Any:
        self._validate(latent)
        return latent

    def as_dict(self) -> dict[str, Any]:
        return {
            "flow_latent_layout_id": self.layout_id,
            "flow_latent_native_rank": 5,
            "flow_latent_canonical_rank": 5,
            "flow_latent_layout_roundtrip_exact": True,
        }


@dataclass(frozen=True)
class PackedTokenFlowLatentLayout:
    """实现 LTX 类 pipeline 的三维 token latent 与五维 latent 的可逆变换。

    该实现复现 Diffusers ``LTXPipeline._pack_latents`` 与
    ``LTXPipeline._unpack_latents`` 的维度变换, 但不依赖 Diffusers 私有函数。
    因此核心方法包可以独立测试, 同时不会把第三方 pipeline 代码复制到 Notebook。
    """

    num_frames: int
    height: int
    width: int
    spatial_patch_size: int = 1
    temporal_patch_size: int = 1
    layout_id: str = "packed_token_flow_latent"

    def __post_init__(self) -> None:
        dimensions = (
            self.num_frames,
            self.height,
            self.width,
            self.spatial_patch_size,
            self.temporal_patch_size,
        )
        if any(int(value) <= 0 for value in dimensions):
            raise ValueError("packed token latent 的尺寸和 patch 大小必须为正整数")
        if self.num_frames % self.temporal_patch_size != 0:
            raise ValueError("latent 帧数必须能被 temporal patch 大小整除")
        if self.height % self.spatial_patch_size != 0 or self.width % self.spatial_patch_size != 0:
            raise ValueError("latent 高宽必须能被 spatial patch 大小整除")

    @property
    def token_count(self) -> int:
        return (
            self.num_frames // self.temporal_patch_size
            * (self.height // self.spatial_patch_size)
            * (self.width // self.spatial_patch_size)
        )

    def to_canonical(self, latent: Any) -> Any:
        """把 ``[B, S, D]`` token 序列还原为 ``[B, C, T, H, W]``。"""

        if getattr(latent, "ndim", None) != 3:
            raise ValueError("packed token Flow latent 必须使用 [B, S, D] 三维张量")
        batch_size, token_count, feature_count = (int(value) for value in latent.shape)
        if token_count != self.token_count:
            raise ValueError(
                f"packed token 数量与布局不一致: expected={self.token_count}, actual={token_count}"
            )
        patch_volume = (
            self.temporal_patch_size
            * self.spatial_patch_size
            * self.spatial_patch_size
        )
        if feature_count % patch_volume != 0:
            raise ValueError("packed token 特征数不能还原为完整 channel 维")
        post_patch_frames = self.num_frames // self.temporal_patch_size
        post_patch_height = self.height // self.spatial_patch_size
        post_patch_width = self.width // self.spatial_patch_size
        unpacked = latent.reshape(
            batch_size,
            post_patch_frames,
            post_patch_height,
            post_patch_width,
            -1,
            self.temporal_patch_size,
            self.spatial_patch_size,
            self.spatial_patch_size,
        )
        return (
            unpacked.permute(0, 4, 1, 5, 2, 6, 3, 7)
            .flatten(6, 7)
            .flatten(4, 5)
            .flatten(2, 3)
        )

    def from_canonical(self, latent: Any) -> Any:
        """把 ``[B, C, T, H, W]`` 转换为模型使用的 ``[B, S, D]``。"""

        if getattr(latent, "ndim", None) != 5:
            raise ValueError("SSTW 规范 Flow latent 必须使用 [B, C, T, H, W] 五维张量")
        batch_size, _channels, frames, height, width = (int(value) for value in latent.shape)
        if (frames, height, width) != (self.num_frames, self.height, self.width):
            raise ValueError(
                "SSTW 规范 latent 尺寸与 packed token 布局不一致: "
                f"expected={(self.num_frames, self.height, self.width)}, "
                f"actual={(frames, height, width)}"
            )
        post_patch_frames = frames // self.temporal_patch_size
        post_patch_height = height // self.spatial_patch_size
        post_patch_width = width // self.spatial_patch_size
        packed = latent.reshape(
            batch_size,
            -1,
            post_patch_frames,
            self.temporal_patch_size,
            post_patch_height,
            self.spatial_patch_size,
            post_patch_width,
            self.spatial_patch_size,
        )
        return packed.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)

    def as_dict(self) -> dict[str, Any]:
        return {
            "flow_latent_layout_id": self.layout_id,
            "flow_latent_native_rank": 3,
            "flow_latent_canonical_rank": 5,
            "flow_latent_layout_roundtrip_exact": True,
            "flow_latent_num_frames": self.num_frames,
            "flow_latent_height": self.height,
            "flow_latent_width": self.width,
            "flow_latent_spatial_patch_size": self.spatial_patch_size,
            "flow_latent_temporal_patch_size": self.temporal_patch_size,
            "flow_latent_token_count": self.token_count,
        }
