from __future__ import annotations

from types import SimpleNamespace

import pytest

from main.methods.state_space_watermark.patch_relation_carrier import (
    ROPE_TUPLE_SHAPE,
    build_public_patch_relation_descriptor,
)
from main.methods.state_space_watermark.patch_relation_wan_runtime import (
    ScopedWanRopeOutputAdapter,
    TRANSFORMER_HIDDEN_SHAPE,
)


pytestmark = pytest.mark.integration
try:
    import torch
except ImportError:  # pragma: no cover - local environment boundary
    torch = None


if torch is not None:

    class _TorchRope(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer(
                "cosine",
                torch.ones(ROPE_TUPLE_SHAPE, dtype=torch.float64),
            )
            self.register_buffer(
                "sine",
                torch.zeros(ROPE_TUPLE_SHAPE, dtype=torch.float64),
            )

        def forward(self, hidden_states):
            return self.cosine, self.sine


@pytest.mark.skipif(torch is None, reason="local torch runtime unavailable")
def test_real_torch_scope_preserves_dtype_device_and_restores_forward() -> None:
    rope = _TorchRope()
    transformer = SimpleNamespace(rope=rope)
    hidden_states = torch.zeros(
        TRANSFORMER_HIDDEN_SHAPE,
        dtype=torch.bfloat16,
    )
    descriptor = build_public_patch_relation_descriptor()
    scope = ScopedWanRopeOutputAdapter(
        transformer,
        descriptor=descriptor,
        signed_coefficient=1,
        probe_id="patch_relation_probe",
        step_index=0,
        control_role="controlled",
        cfg_branch_role="conditional",
        input_binding_digest="1" * 64,
    )
    assert "forward" not in rope.__dict__
    with torch.no_grad(), scope:
        shifted_cosine, shifted_sine = rope(hidden_states)
    assert "forward" not in rope.__dict__
    assert shifted_cosine.dtype == torch.float64
    assert shifted_sine.dtype == torch.float64
    assert shifted_cosine.device == rope.cosine.device
    assert shifted_sine.device == rope.sine.device
    cosine_changed = shifted_cosine != rope.cosine
    sine_changed = shifted_sine != rope.sine
    assert torch.count_nonzero(cosine_changed).item() == 12
    assert torch.count_nonzero(shifted_sine).item() == 12
    assert torch.count_nonzero(sine_changed).item() == 12
    assert (
        torch.count_nonzero(cosine_changed).item()
        + torch.count_nonzero(sine_changed).item()
        == 24
    )
    assert torch.count_nonzero(cosine_changed[..., 2:]).item() == 0
    assert torch.count_nonzero(sine_changed[..., 2:]).item() == 0
    assert torch.equal(rope.cosine, torch.ones_like(rope.cosine))
    assert torch.equal(rope.sine, torch.zeros_like(rope.sine))
    assert scope.record().successful_rope_call_count == 1


@pytest.mark.skipif(torch is None, reason="local torch runtime unavailable")
def test_real_torch_scope_rejects_grad_enabled_and_restores() -> None:
    rope = _TorchRope()
    transformer = SimpleNamespace(rope=rope)
    hidden_states = torch.zeros(
        TRANSFORMER_HIDDEN_SHAPE,
        dtype=torch.bfloat16,
    )
    scope = ScopedWanRopeOutputAdapter(
        transformer,
        descriptor=build_public_patch_relation_descriptor(),
        signed_coefficient=1,
        probe_id="patch_relation_probe",
        step_index=0,
        control_role="controlled",
        cfg_branch_role="conditional",
        input_binding_digest="1" * 64,
    )
    with pytest.raises(RuntimeError, match="no-grad"):
        with scope:
            rope(hidden_states)
    assert "forward" not in rope.__dict__
    assert not hasattr(rope, "_sstw_patch_relation_rope_scope_active")
