from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from types import SimpleNamespace

import pytest

from main.methods.state_space_watermark.patch_relation_carrier import (
    ROPE_TUPLE_SHAPE,
    build_public_patch_relation_descriptor,
)
from main.methods.state_space_watermark.patch_relation_wan_runtime import (
    ScopedWanRopeOutputAdapter,
    TRANSFORMER_HIDDEN_SHAPE,
    apply_wan_rotary_phase_runtime,
)


pytestmark = pytest.mark.integration
try:
    import torch
except ImportError:  # pragma: no cover - local environment boundary
    torch = None

try:
    if torch is None:
        raise ImportError
    from diffusers.models.transformers.transformer_wan import (
        WanRotaryPosEmbed,
    )

    DIFFUSERS_VERSION = version("diffusers")
except (ImportError, PackageNotFoundError):  # pragma: no cover
    WanRotaryPosEmbed = None
    DIFFUSERS_VERSION = None


OFFICIAL_RUNTIME_AVAILABLE = (
    torch is not None
    and WanRotaryPosEmbed is not None
    and DIFFUSERS_VERSION == "0.35.2"
)


if torch is not None:

    class _TorchRope(torch.nn.Module):
        def __init__(self, dtype) -> None:
            super().__init__()
            self.register_buffer(
                "cosine",
                torch.ones(ROPE_TUPLE_SHAPE, dtype=dtype),
            )
            self.register_buffer(
                "sine",
                torch.zeros(ROPE_TUPLE_SHAPE, dtype=dtype),
            )

        def forward(self, hidden_states):
            return self.cosine, self.sine


def _scope(transformer, *, coefficient: int, control_role: str):
    return ScopedWanRopeOutputAdapter(
        transformer,
        descriptor=build_public_patch_relation_descriptor(),
        signed_coefficient=coefficient,
        probe_id="patch_relation_probe",
        step_index=0,
        control_role=control_role,
        cfg_branch_role="conditional",
        input_binding_digest="1" * 64,
    )


@pytest.mark.skipif(
    not OFFICIAL_RUNTIME_AVAILABLE,
    reason="exact local torch and diffusers 0.35.2 runtime unavailable",
)
def test_official_wan_rope_float32_output_runs_through_adapter() -> None:
    rope = WanRotaryPosEmbed(
        attention_head_dim=128,
        patch_size=(1, 2, 2),
        max_seq_len=1024,
    )
    transformer = SimpleNamespace(rope=rope)
    hidden_states = torch.zeros(
        TRANSFORMER_HIDDEN_SHAPE,
        dtype=torch.bfloat16,
    )
    with torch.no_grad():
        baseline_cosine, baseline_sine = rope(hidden_states)
    assert baseline_cosine.dtype == torch.float32
    assert baseline_sine.dtype == torch.float32
    assert tuple(baseline_cosine.shape) == ROPE_TUPLE_SHAPE
    assert tuple(baseline_sine.shape) == ROPE_TUPLE_SHAPE
    assert baseline_cosine.device == hidden_states.device
    assert baseline_sine.device == hidden_states.device

    scope = _scope(
        transformer,
        coefficient=1,
        control_role="controlled",
    )
    with torch.no_grad(), scope:
        shifted_cosine, shifted_sine = rope(hidden_states)
    assert shifted_cosine.dtype == torch.float32
    assert shifted_sine.dtype == torch.float32
    assert shifted_cosine.device == baseline_cosine.device
    assert shifted_sine.device == baseline_sine.device
    cosine_changed = shifted_cosine != baseline_cosine
    sine_changed = shifted_sine != baseline_sine
    assert torch.count_nonzero(cosine_changed).item() == 12
    assert torch.count_nonzero(sine_changed).item() == 12
    assert torch.count_nonzero(cosine_changed[..., 2:]).item() == 0
    assert torch.count_nonzero(sine_changed[..., 2:]).item() == 0
    assert scope.record().successful_rope_call_count == 1


@pytest.mark.skipif(
    not OFFICIAL_RUNTIME_AVAILABLE,
    reason="exact local torch and diffusers 0.35.2 runtime unavailable",
)
def test_official_wan_rope_clean_control_returns_exact_tuple_identity() -> None:
    rope = WanRotaryPosEmbed(
        attention_head_dim=128,
        patch_size=(1, 2, 2),
        max_seq_len=1024,
    )
    transformer = SimpleNamespace(rope=rope)
    hidden_states = torch.zeros(
        TRANSFORMER_HIDDEN_SHAPE,
        dtype=torch.bfloat16,
    )
    captured: dict[str, tuple[object, object]] = {}
    official_forward = rope.forward

    def capturing_forward(value):
        result = official_forward(value)
        captured["tuple"] = result
        return result

    rope.forward = capturing_forward
    scope = _scope(transformer, coefficient=0, control_role="base")
    with torch.no_grad(), scope:
        returned = rope(hidden_states)
    assert returned[0] is captured["tuple"][0]
    assert returned[1] is captured["tuple"][1]
    assert returned[0].dtype == torch.float32
    assert returned[1].dtype == torch.float32
    assert rope.forward is capturing_forward
    assert scope.record().clean_exact_noop


@pytest.mark.skipif(torch is None, reason="local torch runtime unavailable")
def test_real_torch_scope_rejects_float64_tuple_with_dtype_details() -> None:
    rope = _TorchRope(torch.float64)
    transformer = SimpleNamespace(rope=rope)
    hidden_states = torch.zeros(
        TRANSFORMER_HIDDEN_SHAPE,
        dtype=torch.bfloat16,
    )
    scope = _scope(
        transformer,
        coefficient=1,
        control_role="controlled",
    )
    with pytest.raises(
        ValueError,
        match=r"expected=torch.float32, observed=torch.float64",
    ):
        with torch.no_grad(), scope:
            rope(hidden_states)
    assert "forward" not in rope.__dict__
    assert not hasattr(rope, "_sstw_patch_relation_rope_scope_active")


@pytest.mark.skipif(torch is None, reason="local torch runtime unavailable")
def test_real_torch_scope_rejects_grad_enabled_and_restores() -> None:
    rope = _TorchRope(torch.float32)
    transformer = SimpleNamespace(rope=rope)
    hidden_states = torch.zeros(
        TRANSFORMER_HIDDEN_SHAPE,
        dtype=torch.bfloat16,
    )
    scope = _scope(
        transformer,
        coefficient=1,
        control_role="controlled",
    )
    with pytest.raises(RuntimeError, match="no-grad"):
        with scope:
            rope(hidden_states)
    assert "forward" not in rope.__dict__
    assert not hasattr(rope, "_sstw_patch_relation_rope_scope_active")


@pytest.mark.skipif(torch is None, reason="local torch runtime unavailable")
def test_torch_clean_runtime_returns_original_float32_tuple() -> None:
    cosine = torch.ones(ROPE_TUPLE_SHAPE, dtype=torch.float32)
    sine = torch.zeros(ROPE_TUPLE_SHAPE, dtype=torch.float32)
    with torch.no_grad():
        returned = apply_wan_rotary_phase_runtime(
            cosine,
            sine,
            descriptor=build_public_patch_relation_descriptor(),
            signed_coefficient=0,
        )
    assert returned[0] is cosine
    assert returned[1] is sine
