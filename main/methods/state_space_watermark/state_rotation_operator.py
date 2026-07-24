"""Prompt-independent key derivation and low-rank state rotation."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import hmac
import json
import math
from typing import Any


PROMPT_ORTHOGONAL_METHOD_DOMAIN = (
    "sstw_prompt_orthogonal_state_trajectory"
)
PROMPT_ORTHOGONAL_LATENT_LAYOUT_ID = (
    "five_dimensional_flow_latent_flattened_per_sample"
)
PROMPT_ORTHOGONAL_OPERATOR_SCHEMA_ID = (
    "rank_two_antisymmetric_state_rotation"
)
PROMPT_ORTHOGONAL_CODE_SCHEMA_ID = (
    "two_channel_first_harmonic_balanced_code"
)
PROMPT_ORTHOGONAL_OPERATOR_RANK = 2
PROMPT_ORTHOGONAL_PROJECTION_TOLERANCE = 1e-5
PROMPT_ORTHOGONAL_MINIMUM_RETAINED_RATIO = 0.01
PROMPT_ORTHOGONAL_CANONICAL_PLANE_CONSTRUCTION_DEVICE = "cpu"


@dataclass(frozen=True)
class PromptOrthogonalKeyDomainConfig:
    """Freeze the prompt-independent identity domain of the candidate method."""

    method_domain: str = PROMPT_ORTHOGONAL_METHOD_DOMAIN
    latent_layout_id: str = PROMPT_ORTHOGONAL_LATENT_LAYOUT_ID
    operator_schema_id: str = PROMPT_ORTHOGONAL_OPERATOR_SCHEMA_ID
    code_schema_id: str = PROMPT_ORTHOGONAL_CODE_SCHEMA_ID

    def __post_init__(self) -> None:
        expected = {
            "method_domain": PROMPT_ORTHOGONAL_METHOD_DOMAIN,
            "latent_layout_id": PROMPT_ORTHOGONAL_LATENT_LAYOUT_ID,
            "operator_schema_id": PROMPT_ORTHOGONAL_OPERATOR_SCHEMA_ID,
            "code_schema_id": PROMPT_ORTHOGONAL_CODE_SCHEMA_ID,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"prompt-orthogonal {name} 未冻结")


@dataclass(frozen=True)
class PromptOrthogonalSubkeys:
    """Hold internal subkeys and the public domain-separation digest."""

    state_operator_subkey: str
    trajectory_code_subkey: str
    domain_separation_digest: str


@dataclass(frozen=True)
class PromptOrthogonalDirection:
    """A prompt-nuisance-orthogonal keyed tangent and its diagnostics."""

    direction: Any
    operator_plane_digest: str
    operator_rank: int
    state_tangent_norm: float
    projected_tangent_norm: float
    projection_retained_ratio: float
    state_orthogonality_residual: float
    velocity_orthogonality_residual: float
    active: bool


def derive_prompt_orthogonal_subkeys(
    master_key_text: str,
    *,
    config: PromptOrthogonalKeyDomainConfig | None = None,
) -> PromptOrthogonalSubkeys:
    """Derive state and time subkeys without prompt, seed, model, or grid."""

    config = config or PromptOrthogonalKeyDomainConfig()
    secret = str(master_key_text).encode("utf-8")
    if len(secret) < 16:
        raise ValueError("prompt-orthogonal master key 至少需要16字节")
    public_domain = {
        "code_schema_id": config.code_schema_id,
        "latent_layout_id": config.latent_layout_id,
        "method_domain": config.method_domain,
        "operator_schema_id": config.operator_schema_id,
    }
    canonical = json.dumps(
        public_domain,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    def derive(label: str) -> str:
        return hmac.new(
            secret,
            f"{canonical}::{label}".encode("utf-8"),
            sha256,
        ).hexdigest()

    state_subkey = derive("state_operator")
    code_subkey = derive("trajectory_code")
    if hmac.compare_digest(state_subkey, code_subkey):
        raise RuntimeError("prompt-orthogonal 子 key 域分离失败")
    return PromptOrthogonalSubkeys(
        state_operator_subkey=state_subkey,
        trajectory_code_subkey=code_subkey,
        domain_separation_digest=sha256(canonical.encode("utf-8")).hexdigest(),
    )


def low_rank_rotation_tangent_values(
    state_values: tuple[float, ...],
    plane_a_values: tuple[float, ...],
    plane_b_values: tuple[float, ...],
) -> tuple[float, ...]:
    """Reference implementation of ``(ab^T-ba^T)z`` for lightweight tests."""

    if not state_values or not (
        len(state_values) == len(plane_a_values) == len(plane_b_values)
    ):
        raise ValueError("state 与低秩 plane 向量必须非空且等长")
    if not all(
        math.isfinite(float(value))
        for values in (state_values, plane_a_values, plane_b_values)
        for value in values
    ):
        raise ValueError("state rotation 输入必须有限")
    b_dot_state = sum(
        float(left) * float(right)
        for left, right in zip(plane_b_values, state_values, strict=True)
    )
    a_dot_state = sum(
        float(left) * float(right)
        for left, right in zip(plane_a_values, state_values, strict=True)
    )
    return tuple(
        float(left) * b_dot_state - float(right) * a_dot_state
        for left, right in zip(
            plane_a_values,
            plane_b_values,
            strict=True,
        )
    )


def _stable_torch_seed(*parts: object) -> int:
    payload = "::".join(str(part) for part in parts)
    return int(sha256(payload.encode("utf-8")).hexdigest()[:16], 16) % (
        2**63 - 1
    )


def _per_sample_flatten(value: Any) -> Any:
    if value.ndim < 2:
        return value.reshape(1, -1)
    return value.reshape(value.shape[0], -1)


def _normalized_rows(value: Any, *, epsilon: float = 1e-12) -> Any:
    return value / value.norm(dim=1, keepdim=True).clamp_min(epsilon)


def build_state_rotation_plane_like(
    reference: Any,
    *,
    state_operator_subkey: str,
    config: PromptOrthogonalKeyDomainConfig | None = None,
) -> tuple[Any, Any, str]:
    """Build an orthonormal rank-two plane with the reference tensor shape."""

    config = config or PromptOrthogonalKeyDomainConfig()
    try:
        import torch
    except ImportError as error:  # pragma: no cover - method runtime boundary
        raise RuntimeError("state rotation runtime 需要 torch") from error
    if not torch.is_tensor(reference):
        raise TypeError("state rotation reference 必须是 torch tensor")
    if reference.numel() < 2:
        raise ValueError("state rotation 至少需要二维 latent")
    shape = tuple(int(value) for value in reference.shape)
    # CPU is the canonical construction device.  Equal seeds are not enough
    # to make PyTorch CPU and CUDA RNG streams numerically identical.
    generator = torch.Generator(
        device=PROMPT_ORTHOGONAL_CANONICAL_PLANE_CONSTRUCTION_DEVICE
    )
    generator.manual_seed(
        _stable_torch_seed(
            config.method_domain,
            config.latent_layout_id,
            config.operator_schema_id,
            state_operator_subkey,
            shape,
        )
    )
    plane_a = torch.randn(
        shape,
        generator=generator,
        device=PROMPT_ORTHOGONAL_CANONICAL_PLANE_CONSTRUCTION_DEVICE,
        dtype=torch.float32,
    )
    plane_b = torch.randn(
        shape,
        generator=generator,
        device=PROMPT_ORTHOGONAL_CANONICAL_PLANE_CONSTRUCTION_DEVICE,
        dtype=torch.float32,
    )
    a_flat = _normalized_rows(_per_sample_flatten(plane_a))
    b_flat = _per_sample_flatten(plane_b)
    b_flat = b_flat - (b_flat * a_flat).sum(
        dim=1,
        keepdim=True,
    ) * a_flat
    b_norm = b_flat.norm(dim=1, keepdim=True)
    if bool((b_norm <= 1e-12).any().item()):
        raise RuntimeError("state rotation keyed plane 线性相关")
    b_flat = b_flat / b_norm
    plane_a = a_flat.reshape(shape)
    plane_b = b_flat.reshape(shape)
    digest_payload = {
        "latent_layout_id": config.latent_layout_id,
        "operator_rank": PROMPT_ORTHOGONAL_OPERATOR_RANK,
        "operator_schema_id": config.operator_schema_id,
        "plane_construction_device": (
            PROMPT_ORTHOGONAL_CANONICAL_PLANE_CONSTRUCTION_DEVICE
        ),
        "shape": list(shape),
        "state_operator_subkey_digest": sha256(
            state_operator_subkey.encode("utf-8")
        ).hexdigest(),
    }
    plane_digest = sha256(
        json.dumps(
            digest_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return (
        plane_a.to(device=reference.device),
        plane_b.to(device=reference.device),
        plane_digest,
    )


def build_prompt_orthogonal_state_direction(
    state: Any,
    base_velocity: Any,
    *,
    state_operator_subkey: str,
    config: PromptOrthogonalKeyDomainConfig | None = None,
    minimum_retained_ratio: float = (
        PROMPT_ORTHOGONAL_MINIMUM_RETAINED_RATIO
    ),
    projection_tolerance: float = PROMPT_ORTHOGONAL_PROJECTION_TOLERANCE,
    state_rotation_plane: tuple[Any, Any, str] | None = None,
) -> PromptOrthogonalDirection:
    """Construct ``A_K z`` and remove state/base-velocity nuisance directions."""

    config = config or PromptOrthogonalKeyDomainConfig()
    try:
        import torch
    except ImportError as error:  # pragma: no cover - method runtime boundary
        raise RuntimeError("prompt-orthogonal direction 需要 torch") from error
    if not torch.is_tensor(state) or not torch.is_tensor(base_velocity):
        raise TypeError("state 与 base velocity 必须是 torch tensor")
    if state.shape != base_velocity.shape:
        raise ValueError("state 与 base velocity 必须同形")
    if not 0.0 <= float(minimum_retained_ratio) <= 1.0:
        raise ValueError("minimum retained ratio 必须位于[0,1]")
    state_float = state.detach().float()
    velocity_float = base_velocity.detach().float()
    if not bool(torch.isfinite(state_float).all().item()) or not bool(
        torch.isfinite(velocity_float).all().item()
    ):
        raise ValueError("state 与 base velocity 必须有限")
    if state_rotation_plane is None:
        plane_a, plane_b, plane_digest = build_state_rotation_plane_like(
            state_float,
            state_operator_subkey=state_operator_subkey,
            config=config,
        )
    else:
        plane_a, plane_b, plane_digest = state_rotation_plane
        if plane_a.shape != state.shape or plane_b.shape != state.shape:
            raise ValueError("cached state rotation plane 与 state 必须同形")
    z_flat = _per_sample_flatten(state_float)
    v_flat = _per_sample_flatten(velocity_float)
    a_flat = _per_sample_flatten(plane_a)
    b_flat = _per_sample_flatten(plane_b)
    tangent = (
        a_flat * (b_flat * z_flat).sum(dim=1, keepdim=True)
        - b_flat * (a_flat * z_flat).sum(dim=1, keepdim=True)
    )
    tangent_norm = tangent.norm(dim=1, keepdim=True)
    z_unit = _normalized_rows(z_flat)
    projected = tangent - (tangent * z_unit).sum(
        dim=1,
        keepdim=True,
    ) * z_unit
    velocity_residual = v_flat - (v_flat * z_unit).sum(
        dim=1,
        keepdim=True,
    ) * z_unit
    velocity_residual_norm = velocity_residual.norm(dim=1, keepdim=True)
    usable_velocity = velocity_residual_norm > 1e-12
    velocity_unit = velocity_residual / velocity_residual_norm.clamp_min(1e-12)
    projected = projected - (
        (projected * velocity_unit).sum(dim=1, keepdim=True)
        * velocity_unit
        * usable_velocity
    )
    projected_norm = projected.norm(dim=1, keepdim=True)
    retained = projected_norm / tangent_norm.clamp_min(1e-12)
    active_rows = (
        (tangent_norm > 1e-12)
        & (projected_norm > 1e-12)
        & (retained >= float(minimum_retained_ratio))
    )
    active = bool(active_rows.all().item())
    if active:
        direction_flat = projected / projected_norm
    else:
        direction_flat = projected * 0.0
    state_residual = (
        (direction_flat * z_unit).sum(dim=1).abs().max().item()
    )
    velocity_unit_for_residual = _normalized_rows(v_flat)
    velocity_residual_value = (
        (direction_flat * velocity_unit_for_residual)
        .sum(dim=1)
        .abs()
        .max()
        .item()
    )
    if active and (
        float(state_residual) > float(projection_tolerance)
        or float(velocity_residual_value) > float(projection_tolerance)
    ):
        raise RuntimeError("prompt nuisance 正交投影数值残差超限")
    return PromptOrthogonalDirection(
        direction=direction_flat.reshape(state.shape),
        operator_plane_digest=plane_digest,
        operator_rank=PROMPT_ORTHOGONAL_OPERATOR_RANK,
        state_tangent_norm=float(tangent_norm.min().item()),
        projected_tangent_norm=float(projected_norm.min().item()),
        projection_retained_ratio=float(retained.min().item()),
        state_orthogonality_residual=float(state_residual),
        velocity_orthogonality_residual=float(velocity_residual_value),
        active=active,
    )
