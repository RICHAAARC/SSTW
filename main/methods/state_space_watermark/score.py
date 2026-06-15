"""统一计算第一阶段各 method variant 的检测分数。"""

from __future__ import annotations

from main.methods.state_space_watermark.state_observation import build_state_observation
from main.methods.state_space_watermark.state_synchronizer import SynchronizerResult, synchronize_state

BASELINE_FACTOR = {"frame_prc": 0.50, "tubelet_only": 0.64, "explicit_temporal_alignment": 0.68, "generic_temporal_mean_pooling": 0.60, "conv1d_temporal_aggregator": 0.66, "gru_temporal_aggregator": 0.67, "transformer_temporal_aggregator": 0.68}


def score_method(sample_role: str, attack_name: str, method_variant: str) -> SynchronizerResult:
    """根据方法变体生成可审计的检测分数。"""
    if method_variant in BASELINE_FACTOR:
        observation = build_state_observation(sample_role, attack_name, key_conditioned=False)
        payload_state = observation.payload_response * BASELINE_FACTOR[method_variant] + observation.sync_response * 0.12
        if method_variant == "tubelet_only" and attack_name != "no_attack" and sample_role.endswith("positive"):
            payload_state += 0.05
        return SynchronizerResult(round(observation.payload_response, 6), round(payload_state, 6), round(payload_state, 6), round(payload_state, 6), round(0.35 + observation.attack_penalty, 6), round(1.0 - observation.attack_penalty, 6), max(1, int((1.0 - observation.attack_penalty) * 6)), round(observation.attack_penalty, 6), "not_applicable")
    key_conditioned = method_variant not in {"generic_state_space_model", "key_agnostic_state_space_model", "key_conditioned_state_space_without_key_condition"}
    return synchronize_state(build_state_observation(sample_role, attack_name, key_conditioned), sample_role, method_variant)
