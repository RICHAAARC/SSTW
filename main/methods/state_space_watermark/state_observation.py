"""构造状态空间推断所需的最小观测量。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StateObservation:
    """表示 payload 与同步证据组成的局部观测。"""

    payload_response: float
    sync_response: float
    attack_penalty: float
    key_match: bool


def build_state_observation(sample_role: str, attack_name: str, key_conditioned: bool) -> StateObservation:
    """根据样本角色和攻击类型生成确定性观测。

    该实现属于第一阶段的机制 sanity proxy, 不是最终论文算法。它用于验证 records、
    threshold、baseline 对照和 admissibility gate 的治理闭环。
    """
    positive = sample_role.endswith("positive")
    temporal_attacks = {"temporal_crop", "local_clip", "irregular_frame_dropping", "frame_duplication", "speed_change", "frame_rate_resampling", "segment_jump"}
    attack_penalty = 0.22 if attack_name in temporal_attacks else 0.08
    payload_response = 0.78 if positive else 0.12
    sync_response = 0.72 if positive else 0.10
    if attack_name != "no_attack":
        payload_response -= attack_penalty * 0.45
        sync_response -= attack_penalty * 0.30
    return StateObservation(max(payload_response, 0.0), max(sync_response, 0.0), attack_penalty, positive and key_conditioned)
