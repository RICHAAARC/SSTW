"""提供第一阶段使用的轻量 synthetic latent 后端。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SyntheticLatentSample:
    """表示一个可复现实验样本的元数据。"""

    sample_id: str
    split: str
    sample_role: str
    content_id: str
    seed_id: str
    key_id: str
    latent_shape: tuple[int, int, int, int]


def build_synthetic_samples(splits: tuple[str, ...], sample_roles: tuple[str, ...], sample_count_per_cell: int, latent_shape: tuple[int, int, int, int]) -> list[SyntheticLatentSample]:
    """构造确定性的 synthetic 样本清单。

    该函数只生成元数据, 不生成真实大规模 latent 张量。这样可以让默认验证保持轻量,
    同时保留后续替换为真实 synthetic tensor backend 的接口边界。
    """
    samples: list[SyntheticLatentSample] = []
    for split in splits:
        for sample_role in sample_roles:
            for index in range(sample_count_per_cell):
                samples.append(SyntheticLatentSample(
                    sample_id=f"{split}_{sample_role}_{index:04d}",
                    split=split,
                    sample_role=sample_role,
                    content_id=f"synthetic_content_{index:04d}",
                    seed_id=f"synthetic_seed_{index:04d}",
                    key_id="negative_key" if sample_role.endswith("negative") else "key_alpha",
                    latent_shape=latent_shape,
                ))
    return samples
