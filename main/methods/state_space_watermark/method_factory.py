"""集中管理第一阶段 method variant 名称。"""

from __future__ import annotations

METHOD_VARIANTS: tuple[str, ...] = ("frame_prc", "tubelet_only", "explicit_temporal_alignment", "generic_temporal_mean_pooling", "conv1d_temporal_aggregator", "gru_temporal_aggregator", "transformer_temporal_aggregator", "generic_state_space_model", "key_agnostic_state_space_model", "key_conditioned_state_space_inference", "key_conditioned_state_space_without_admissibility", "key_conditioned_state_space_without_key_condition")


def list_method_variants() -> tuple[str, ...]:
    """返回第一阶段必须覆盖的全部方法变体。"""
    return METHOD_VARIANTS
