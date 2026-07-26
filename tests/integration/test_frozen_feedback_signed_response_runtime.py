"""Optional torch/diffusers checks for the frozen Flow scheduler replay."""

from __future__ import annotations

from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pytest


pytestmark = pytest.mark.integration
torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")

from diffusers import FlowMatchEulerDiscreteScheduler

from experiments.generative_video_model_probe.frozen_feedback_signed_response_diagnostic import (
    CleanTrace,
    CleanTraceStep,
    _run_wan_decode_no_grad,
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


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
def test_tensor_digest_handles_stride_zero_scalar_and_noncontiguous(
    dtype: torch.dtype,
) -> None:
    contiguous_singleton = torch.tensor([1.25], dtype=dtype)
    stride_zero_singleton = torch.tensor(
        [1.25],
        dtype=dtype,
    ).expand(3)[:1]
    assert stride_zero_singleton.shape == (1,)
    assert stride_zero_singleton.stride() == (0,)
    assert _tensor_digest(stride_zero_singleton) == _tensor_digest(
        contiguous_singleton
    )
    assert _tensor_digest(torch.tensor(1.25, dtype=dtype)) == (
        _tensor_digest(contiguous_singleton)
    )

    logical = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]],
        dtype=dtype,
    )
    noncontiguous = torch.tensor(
        [[1.0, 3.0], [2.0, 4.0]],
        dtype=dtype,
    ).t()
    assert not noncontiguous.is_contiguous()
    assert torch.equal(logical, noncontiguous)
    assert _tensor_digest(noncontiguous) == _tensor_digest(logical)
    assert _tensor_digest(
        torch.tensor([2.25], dtype=dtype)
    ) != _tensor_digest(contiguous_singleton)

    if dtype == torch.float32:
        expected = sha256(
            np.asarray(logical.numpy(), dtype=np.float32).tobytes(order="C")
        ).hexdigest()
        assert _tensor_digest(logical) == expected
    if torch.cuda.is_available():
        assert _tensor_digest(logical.cuda()) == _tensor_digest(logical)


def test_wan_decode_helper_matches_torch_no_grad_semantics() -> None:
    observed: list[tuple[str, bool]] = []
    hook_calls: list[str] = []

    class TorchVae:
        def decode(
            self,
            value: torch.Tensor,
            *,
            return_dict: bool,
        ) -> tuple[torch.Tensor]:
            observed.append(("decode", torch.is_grad_enabled()))
            assert return_dict is False
            return (value * 2.0,)

    class TorchProcessor:
        def postprocess_video(
            self,
            value: torch.Tensor,
            *,
            output_type: str,
        ) -> torch.Tensor:
            observed.append(("postprocess", torch.is_grad_enabled()))
            assert output_type == "np"
            return value + 1.0

    pipe = SimpleNamespace(
        vae=TorchVae(),
        video_processor=TorchProcessor(),
        maybe_free_model_hooks=lambda: hook_calls.append("hooks"),
    )
    source = torch.tensor([1.0], requires_grad=True)
    result = _run_wan_decode_no_grad(
        pipe,
        torch_module=torch,
        normalize_latent=lambda: source * 3.0,
    )
    assert observed == [("decode", False), ("postprocess", False)]
    assert result.requires_grad is False
    assert hook_calls == ["hooks"]
