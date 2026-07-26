"""Optional torch checks for the frame-state Gate 0 Jacobian adapter."""

from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration
torch = pytest.importorskip("torch")

from experiments.generative_video_model_probe.frame_state_signed_observability_construction import (
    torch_jacobian_gram_product,
)


def test_torch_jacobian_gram_product_matches_linear_reference() -> None:
    matrix = torch.tensor(
        [[2.0, -1.0], [0.5, 3.0]],
        dtype=torch.float32,
    )
    reference = torch.tensor([0.25, -0.75], dtype=torch.float32)
    direction = torch.tensor([1.0, 2.0], dtype=torch.float32)

    product = torch_jacobian_gram_product(
        lambda value: matrix @ value,
        reference,
        direction,
        torch_module=torch,
    )

    assert torch.equal(product, matrix.T @ matrix @ direction)


def test_torch_jacobian_gram_product_rejects_dtype_drift() -> None:
    reference = torch.ones(2, dtype=torch.float64)
    direction = torch.ones(2, dtype=torch.float64)

    with pytest.raises(ValueError, match="float32"):
        torch_jacobian_gram_product(
            lambda value: value,
            reference,
            direction,
            torch_module=torch,
        )
