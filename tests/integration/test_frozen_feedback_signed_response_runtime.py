"""Optional torch/diffusers checks for the frozen Flow scheduler replay."""

from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration
torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")

from diffusers import FlowMatchEulerDiscreteScheduler

from experiments.generative_video_model_probe.frozen_feedback_signed_response_diagnostic import (
    CleanTrace,
    CleanTraceStep,
    _tensor_digest,
    _validate_offline_clean_replay,
)
from experiments.generative_video_model_probe.frozen_feedback_signed_response_diagnostic import (
    _runtime_config,
)
from evaluation.protocol.frozen_feedback_signed_response_contract import (
    load_frozen_feedback_signed_response_config,
)


@pytest.mark.skipif(
    diffusers.__version__ != "0.35.2",
    reason="frozen runtime contract requires diffusers 0.35.2",
)
def test_offline_clean_scheduler_clone_is_exact_array_equal() -> None:
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=3.0,
    )
    scheduler.set_timesteps(8, device="cpu")
    runtime_config = _runtime_config(
        load_frozen_feedback_signed_response_config()
    )
    initial = torch.linspace(
        -0.5,
        0.5,
        16,
        dtype=torch.float32,
    ).reshape(1, 1, 4, 4)
    state = initial.clone()
    steps: list[CleanTraceStep] = []
    for index in range(8):
        base = torch.full_like(state, 0.01 * (index + 1))
        record = {
            "clean_velocity_trace_record_id": f"clean-step-{index}",
        }
        steps.append(
            CleanTraceStep(
                step_index=index,
                timestep=scheduler.timesteps[index].clone(),
                state_before=state.clone(),
                base_velocity=base.clone(),
                record=record,
            )
        )
        state = scheduler.step(
            base,
            scheduler.timesteps[index],
            state,
            return_dict=False,
        )[0]
    trace = CleanTrace(
        initial_latent=initial,
        final_latent=state,
        steps=tuple(steps),
        transformer_call_records=(),
        generator_state_digest_random="integration",
        trace_digest="integration",
    )
    assert _validate_offline_clean_replay(
        scheduler,
        trace,
        runtime_config,
        device=torch.device("cpu"),
    ) == _tensor_digest(state)
