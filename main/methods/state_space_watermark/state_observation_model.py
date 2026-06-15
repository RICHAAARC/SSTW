"""定义 B3 正式化阶段的状态观测模型接口。"""

from __future__ import annotations

from dataclasses import dataclass

from main.methods.state_space_watermark.state_observation import build_state_observation


@dataclass(frozen=True)
class FormalObservationVector:
    """表示状态空间模型的观测向量。"""

    r_payload_t: float
    r_sync_t: float
    q_t: float
    e_key_t: float
    r_traj_t_status: str = "disabled"


def build_formal_observation(sample_role: str, attack_name: str, key_conditioned: bool, quality_confidence: float = 1.0) -> FormalObservationVector:
    """构造 B3 使用的正式观测向量。

    该接口明确 trajectory 在 B3 中禁用, 只使用 payload、sync、质量置信度和 key embedding。
    """
    observation = build_state_observation(sample_role, attack_name, key_conditioned)
    return FormalObservationVector(
        r_payload_t=observation.payload_response,
        r_sync_t=observation.sync_response,
        q_t=quality_confidence,
        e_key_t=1.0 if observation.key_match else 0.0,
    )
