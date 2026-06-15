"""提供 key-conditioned state-space synchronizer 的轻量实现。"""

from __future__ import annotations

from dataclasses import dataclass

from main.methods.state_space_watermark.key_conditioner import key_condition_gain
from main.methods.state_space_watermark.key_state_admissibility import evaluate_admissibility
from main.methods.state_space_watermark.state_observation import StateObservation


@dataclass(frozen=True)
class SynchronizerResult:
    """保存状态同步后的核心分数和可解释代理状态。"""

    payload_raw: float
    payload_state: float
    state_posterior: float
    final_score: float
    state_entropy: float
    state_coverage_ratio: float
    state_matched_count: int
    state_transition_residual: float
    admissibility_status: str


def synchronize_state(observation: StateObservation, sample_role: str, method_variant: str) -> SynchronizerResult:
    """执行第一阶段的状态同步 proxy。"""
    positive = sample_role.endswith("positive")
    base_state_gain = 0.12 if "state_space" in method_variant else 0.03
    if method_variant == "generic_state_space_model":
        base_state_gain = 0.07
    if method_variant == "key_agnostic_state_space_model":
        base_state_gain = 0.08
    payload_state = observation.payload_response + base_state_gain + (key_condition_gain(method_variant) if positive else 0.0)
    posterior = payload_state + observation.sync_response * 0.20
    admissibility_status = evaluate_admissibility(sample_role, observation.payload_response, method_variant)
    if admissibility_status == "blocked_negative_tail":
        posterior = min(posterior, observation.payload_response + 0.04)
    entropy = 0.25 + observation.attack_penalty - (0.08 if method_variant == "key_conditioned_state_space_inference" and positive else 0.0)
    return SynchronizerResult(round(observation.payload_response, 6), round(payload_state, 6), round(posterior, 6), round(posterior, 6), round(max(entropy, 0.0), 6), round(1.0 - observation.attack_penalty, 6), max(1, int((1.0 - observation.attack_penalty) * 8)), round(observation.attack_penalty * 0.5, 6), admissibility_status)
