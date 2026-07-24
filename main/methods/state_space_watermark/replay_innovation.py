"""Scheduler-aware key-independent replay innovation extraction."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


FLOW_MATCH_EULER_TRANSITION_ADAPTER_ID = (
    "flow_match_euler_first_order_transition_residual"
)


@dataclass(frozen=True)
class ReplayInnovationStep:
    """One key-independent transition residual and bounded diagnostics."""

    innovation: Any
    adapter_id: str
    delta_sigma: float
    innovation_norm: float
    base_transition_norm: float
    innovation_relative_norm: float
    scheduler_transition_context_complete: bool


def compute_flow_match_euler_replay_innovation(
    current_state: Any,
    next_state: Any,
    base_velocity: Any,
    *,
    delta_sigma: float,
) -> ReplayInnovationStep:
    """Compute ``(z_next-z_current)/delta_sigma - base_velocity``."""

    if (
        current_state.shape != next_state.shape
        or current_state.shape != base_velocity.shape
    ):
        raise ValueError("replay innovation state 与 velocity 必须同形")
    interval = float(delta_sigma)
    if not math.isfinite(interval) or abs(interval) <= 1e-12:
        raise ValueError("replay innovation 需要有限非零 delta sigma")
    current = current_state.detach().float()
    following = next_state.detach().float()
    velocity = base_velocity.detach().float()
    observed_transition = following - current
    base_transition = velocity * interval
    innovation = observed_transition / interval - velocity
    innovation_norm = float(innovation.norm().item())
    base_transition_norm = float(base_transition.norm().item())
    if not math.isfinite(innovation_norm) or not math.isfinite(
        base_transition_norm
    ):
        raise ValueError("replay innovation 结果必须有限")
    return ReplayInnovationStep(
        innovation=innovation,
        adapter_id=FLOW_MATCH_EULER_TRANSITION_ADAPTER_ID,
        delta_sigma=interval,
        innovation_norm=innovation_norm,
        base_transition_norm=base_transition_norm,
        innovation_relative_norm=(
            innovation_norm
            / max(float(velocity.norm().item()), 1e-12)
        ),
        scheduler_transition_context_complete=True,
    )
