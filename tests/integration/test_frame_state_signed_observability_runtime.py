"""Optional torch checks for the frame-state Gate 0 Jacobian adapter."""

from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration
torch = pytest.importorskip("torch")

from experiments.generative_video_model_probe.frame_state_signed_observability_construction import (
    _wan_jacobian_math_sdpa,
    _wan_vae_eval_frozen_parameters,
    torch_jacobian_gram_product,
)


def test_torch_jacobian_gram_product_matches_nonlinear_reference(
    monkeypatch,
) -> None:
    matrix = torch.tensor(
        [[2.0, -1.0], [0.5, 3.0]],
        dtype=torch.float32,
    )
    reference = torch.tensor([0.25, -0.75], dtype=torch.float32)
    direction = torch.tensor([1.0, 2.0], dtype=torch.float32)
    save_on_cpu_calls: list[bool] = []
    original_save_on_cpu = torch.autograd.graph.save_on_cpu

    def observed_save_on_cpu(*, pin_memory: bool):
        save_on_cpu_calls.append(pin_memory)
        return original_save_on_cpu(pin_memory=pin_memory)

    monkeypatch.setattr(
        torch.autograd.graph,
        "save_on_cpu",
        observed_save_on_cpu,
    )
    monkeypatch.setattr(
        torch.autograd.functional,
        "jvp",
        lambda *args, **kwargs: pytest.fail(
            "double-backward JVP must not be called"
        ),
    )

    def feature(value):
        return torch.sin(matrix @ value)

    product = torch_jacobian_gram_product(
        feature,
        reference,
        direction,
        torch_module=torch,
    )
    cosine = torch.cos(matrix @ reference)
    expected = matrix.T @ (cosine.square() * (matrix @ direction))

    assert torch.allclose(product, expected, rtol=1e-5, atol=1e-6)
    assert product.grad_fn is None
    assert reference.grad is None
    assert save_on_cpu_calls == [False]


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


def test_torch_jacobian_rejects_nonfinite_input() -> None:
    reference = torch.tensor([0.0, float("nan")], dtype=torch.float32)
    direction = torch.ones(2, dtype=torch.float32)

    with pytest.raises(RuntimeError, match="非有限"):
        torch_jacobian_gram_product(
            lambda value: value.sin(),
            reference,
            direction,
            torch_module=torch,
        )


def test_vae_parameter_state_and_graph_do_not_cross_iterations() -> None:
    module = torch.nn.Linear(2, 2, bias=False).float()
    prior_requires_grad = tuple(
        parameter.requires_grad for parameter in module.parameters()
    )
    reference = torch.tensor([0.1, -0.3], dtype=torch.float32)
    direction = torch.tensor([0.5, 0.25], dtype=torch.float32)

    with _wan_vae_eval_frozen_parameters(module):
        products = [
            torch_jacobian_gram_product(
                lambda value: torch.tanh(module(value)),
                reference,
                direction,
                torch_module=torch,
            )
            for _ in range(2)
        ]
        assert all(
            parameter.requires_grad is False
            and parameter.grad is None
            for parameter in module.parameters()
        )

    assert all(product.grad_fn is None for product in products)
    assert tuple(
        parameter.requires_grad for parameter in module.parameters()
    ) == prior_requires_grad
    assert all(parameter.grad is None for parameter in module.parameters())


def test_math_sdpa_context_supports_true_forward_ad_on_cpu() -> None:
    query = torch.randn(1, 1, 4, 8, dtype=torch.float32)
    direction = torch.randn_like(query)

    def feature(value):
        return torch.nn.functional.scaled_dot_product_attention(
            value,
            value,
            value,
        )

    with _wan_jacobian_math_sdpa(torch):
        _, tangent = torch.func.jvp(
            feature,
            (query,),
            (direction,),
            strict=True,
        )

    assert tangent.shape == query.shape
    assert bool(torch.isfinite(tangent).all().item())


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA integration requires an explicit GPU environment",
)
def test_cuda_true_forward_jvp_and_saved_tensor_offload() -> None:
    reference = torch.tensor(
        [0.1, -0.3],
        dtype=torch.float32,
        device="cuda",
    )
    direction = torch.tensor(
        [0.5, 0.25],
        dtype=torch.float32,
        device="cuda",
    )

    product = torch_jacobian_gram_product(
        lambda value: torch.sin(value.square()),
        reference,
        direction,
        torch_module=torch,
    )

    assert product.device.type == "cuda"
    assert product.grad_fn is None
    assert bool(torch.isfinite(product).all().item())
