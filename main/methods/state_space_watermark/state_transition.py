"""定义 B3 正式化阶段的状态转移模型接口。"""

from __future__ import annotations

from dataclasses import dataclass

from main.methods.state_space_watermark.state_observation_model import FormalObservationVector


@dataclass(frozen=True)
class FormalHiddenState:
    """表示可解释的水印隐状态。"""

    phase_state_proxy: float
    evidence_state_proxy: float
    confidence_state_proxy: float
    disturbance_state_proxy: float
    state_transition_residual: float

    @property
    def state_hidden_vector(self) -> tuple[float, float, float, float]:
        """返回统一的 hidden state 向量。"""
        return (self.phase_state_proxy, self.evidence_state_proxy, self.confidence_state_proxy, self.disturbance_state_proxy)


def transition_state(observation: FormalObservationVector, method_variant: str) -> FormalHiddenState:
    """根据观测向量和方法变体执行确定性状态转移。"""
    phase = observation.r_sync_t * (0.85 if "without_phase_state" not in method_variant else 0.35)
    evidence = observation.r_payload_t * (0.90 if "without_evidence_state" not in method_variant else 0.45)
    confidence = (observation.q_t + observation.e_key_t) * (0.42 if "without_confidence_state" not in method_variant else 0.20)
    disturbance = (1.0 - observation.r_sync_t) * (0.30 if "without_disturbance_state" not in method_variant else 0.08)
    residual = abs(phase - evidence) * 0.20 + disturbance * 0.10
    return FormalHiddenState(round(phase, 6), round(evidence, 6), round(confidence, 6), round(disturbance, 6), round(residual, 6))
