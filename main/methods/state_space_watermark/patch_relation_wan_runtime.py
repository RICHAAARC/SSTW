"""Local Wan runtime adapter for the first Patch-relation Gate 0 contract.

The adapter is deliberately narrower than an experiment runner.  It provides:

* an exception-safe, one-forward scope around ``transformer.rope.forward``;
* exact conditional/unconditional CFG branch binding; and
* a strict NumPy boundary that recomputes the FP32 CFG velocity and binds the
  realized update to the scheduler's controlled/base next-state difference.

No function in this module authorizes model execution, produces Gate evidence,
or turns caller-provided scheduler provenance into a governed runtime record.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
import re
from types import TracebackType
from typing import Any, Callable
import weakref

import numpy as np

from main.methods.state_space_watermark.patch_relation_carrier import (
    PublicPatchRelationDescriptor,
    PHASE_BUDGET_RADIANS,
    ROPE_TUPLE_SHAPE,
    apply_wan_rotary_phase_numpy,
    build_relation_phase_delta,
    build_public_patch_relation_descriptor,
    validate_public_patch_relation_descriptor,
)
from main.methods.state_space_watermark.state_trajectory_injection import (
    _budget_guard_passed,
)


DIFFUSERS_VERSION = "0.35.2"
TRANSFORMER_HIDDEN_SHAPE = (1, 16, 9, 40, 64)
SCHEDULER_VELOCITY_SHAPE = TRANSFORMER_HIDDEN_SHAPE
CFG_GUIDANCE_SCALE = 5.0
LAMBDA_MAX = 0.12
VELOCITY_NORM_RATIO_BUDGET = 0.02
FLOW_ENERGY_BUDGET_RATIO = 0.000015
MINIMUM_DIRECTION_COSINE = 0.999
PHASE_PROJECTION_MAX_ATTEMPTS = 4
PHASE_PROJECTION_SAFETY_FACTOR = 0.9
PHASE_PROJECTION_MINIMUM_NONZERO_SCALE = 0.000001
EXPECTED_ROPE_CALLS_PER_BRANCH = 1
CFG_BRANCH_ORDER = ("conditional", "unconditional")
RUNTIME_ADAPTER_PROTOCOL_DIGEST = (
    "dfd9d80274f028316eab306c80cb0d2cebfdfe33abd571f0bdb3ecb450d589f1"
)

_LOWER_HEX_64 = re.compile(r"[0-9a-f]{64}\Z")
_PROBE_ID = re.compile(r"[a-z0-9_]+\Z")
_ACTIVE_SCOPE_ATTRIBUTE = "_sstw_patch_relation_rope_scope_active"


@dataclass(frozen=True)
class WanRopeBranchApplicationRecord:
    """Local record produced by one restored RoPE-forward scope."""

    probe_id: str
    step_index: int
    control_role: str
    cfg_branch_role: str
    cfg_branch_order_index: int
    signed_coefficient: int
    maximum_phase_budget_radians: float
    phase_projection_scale: float
    realized_phase_magnitude_radians: float
    input_binding_digest: str
    descriptor_digest: str
    rope_call_attempt_count: int
    successful_rope_call_count: int
    expected_rope_call_count: int
    scope_completed_successfully: bool
    clean_exact_noop: bool
    local_contract_only: bool = True
    execution_evidence_allowed: bool = False
    formal_result: bool = False
    stage_progression_allowed: bool = False


@dataclass(frozen=True)
class CfgRopeApplicationPair:
    """Exact cond/uncond binding for one base or controlled forward."""

    probe_id: str
    step_index: int
    control_role: str
    signed_coefficient: int
    maximum_phase_budget_radians: float
    phase_projection_scale: float
    realized_phase_magnitude_radians: float
    input_binding_digest: str
    descriptor_digest: str
    conditional: WanRopeBranchApplicationRecord
    unconditional: WanRopeBranchApplicationRecord
    local_contract_only: bool = True
    execution_evidence_allowed: bool = False


@dataclass(frozen=True, eq=False)
class CfgStateUpdateMeasurement:
    """Strict measurement bound to the scheduler's actual FP32 next state."""

    probe_id: str
    step_index: int
    signed_coefficient: int
    input_binding_digest: str
    base_raw_velocity_digest: str
    controlled_raw_velocity_digest: str
    base_cfg_velocity: np.ndarray
    controlled_cfg_velocity: np.ndarray
    intended_delta_velocity: np.ndarray
    constrained_velocity: np.ndarray
    actual_delta_velocity: np.ndarray
    scheduler_sample: np.ndarray
    base_next_state: np.ndarray
    controlled_next_state: np.ndarray
    base_state_update_delta: np.ndarray
    intended_state_update_delta: np.ndarray
    actual_state_update_delta: np.ndarray
    delta_sigma: float
    base_velocity_norm: float
    intended_delta_norm: float
    actual_delta_norm: float
    base_state_update_norm: float
    intended_state_update_norm: float
    state_update_delta_norm: float
    state_update_direction_dot: float
    direction_actual_norm: float
    direction_intended_norm: float
    norm_budget: float
    cumulative_reference_energy_before_step: float
    cumulative_control_energy_before_step: float
    remaining_step_count: int
    reference_energy_increment: float
    projected_reference_energy: float
    total_flow_energy_budget: float
    remaining_flow_energy: float
    energy_increment: float
    direction_cosine: float | None
    signed_state_update_exposure: float
    norm_guard_passed: bool
    energy_guard_passed: bool
    direction_guard_passed: bool | None
    clean_exact_noop: bool
    local_contract_only: bool = True
    execution_evidence_allowed: bool = False
    formal_result: bool = False
    stage_progression_allowed: bool = False


@dataclass(frozen=True)
class PhaseProjectionSignEvaluation:
    """Validated scalar summary of one raw-array candidate evaluation."""

    probe_id: str
    step_index: int
    signed_coefficient: int
    input_binding_digest: str
    descriptor_digest: str
    candidate_context_digest: str
    controlled_transition_digest: str
    base_raw_velocity_digest: str
    base_cfg_velocity_digest: str
    controlled_cfg_velocity_digest: str
    scheduler_sample_digest: str
    actual_state_update_digest: str
    base_pair: CfgRopeApplicationPair
    controlled_pair: CfgRopeApplicationPair
    phase_projection_scale: float
    delta_sigma: float
    cumulative_reference_energy_before_step: float
    cumulative_control_energy_before_step: float
    remaining_step_count: int
    base_velocity_norm: float
    actual_delta_norm: float
    state_update_delta_norm: float
    norm_budget: float
    reference_energy_increment: float
    projected_reference_energy: float
    total_flow_energy_budget: float
    energy_increment: float
    remaining_flow_energy: float
    direction_cosine: float | None
    signed_state_update_exposure: float
    norm_guard_passed: bool
    energy_guard_passed: bool
    direction_guard_passed: bool
    feasible: bool


@dataclass(frozen=True)
class SymmetricPhaseProjectionSelection:
    """One deterministic common scale selected from the worst ± response."""

    selected_scale: float
    realized_phase_magnitude_radians: float
    attempt_count: int
    backoff_count: int
    initial_positive: PhaseProjectionSignEvaluation
    initial_negative: PhaseProjectionSignEvaluation
    final_positive: PhaseProjectionSignEvaluation
    final_negative: PhaseProjectionSignEvaluation


_ISSUED_PHASE_PROJECTION_SELECTIONS: weakref.WeakKeyDictionary[
    SymmetricPhaseProjectionSelection,
    tuple[Any, ...],
] = weakref.WeakKeyDictionary()
_ISSUED_PHASE_PROJECTION_EVALUATIONS: weakref.WeakKeyDictionary[
    PhaseProjectionSignEvaluation,
    tuple[Any, ...],
] = weakref.WeakKeyDictionary()
_ISSUED_CFG_STATE_UPDATE_MEASUREMENTS: weakref.WeakKeyDictionary[
    CfgStateUpdateMeasurement,
    tuple[Any, ...],
] = weakref.WeakKeyDictionary()


def _float32_array_digest(*items: tuple[str, np.ndarray]) -> str:
    digest = sha256()
    for label, value in items:
        array = _require_velocity(value, label)
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _phase_projection_candidate_context_digest(
    *,
    base_pair: CfgRopeApplicationPair,
    base_conditional_velocity: np.ndarray,
    base_unconditional_velocity: np.ndarray,
    scheduler_sample: np.ndarray,
    delta_sigma: float,
    cumulative_reference_energy_before_step: float,
    cumulative_control_energy_before_step: float,
    remaining_step_count: int,
) -> str:
    return sha256(
        (
            f"{base_pair.probe_id}\0{base_pair.step_index}\0"
            f"{base_pair.input_binding_digest}\0{base_pair.descriptor_digest}\0"
            f"{float(np.float32(delta_sigma))!r}\0"
            f"{float(cumulative_reference_energy_before_step)!r}\0"
            f"{float(cumulative_control_energy_before_step)!r}\0"
            f"{remaining_step_count}\0"
            + _float32_array_digest(
                (
                    "projection base conditional velocity",
                    base_conditional_velocity,
                ),
                (
                    "projection base unconditional velocity",
                    base_unconditional_velocity,
                ),
                ("projection scheduler sample", scheduler_sample),
            )
        ).encode("utf-8")
    ).hexdigest()


def _cfg_state_update_measurement_payload(
    measurement: CfgStateUpdateMeasurement,
) -> tuple[Any, ...]:
    payload: list[Any] = []
    for field_name in measurement.__dataclass_fields__:
        value = getattr(measurement, field_name)
        if isinstance(value, np.ndarray):
            payload.append(
                (
                    field_name,
                    value.dtype.str,
                    tuple(value.shape),
                    sha256(value.tobytes(order="C")).hexdigest(),
                )
            )
        else:
            payload.append((field_name, value))
    return tuple(payload)


def _issue_cfg_state_update_measurement(
    measurement: CfgStateUpdateMeasurement,
) -> CfgStateUpdateMeasurement:
    _ISSUED_CFG_STATE_UPDATE_MEASUREMENTS[
        measurement
    ] = _cfg_state_update_measurement_payload(measurement)
    return measurement


def _require_issued_cfg_state_update_measurement(
    measurement: CfgStateUpdateMeasurement,
) -> CfgStateUpdateMeasurement:
    if not isinstance(measurement, CfgStateUpdateMeasurement):
        raise TypeError("CFG state-update measurement 类型不匹配")
    payload = _ISSUED_CFG_STATE_UPDATE_MEASUREMENTS.get(measurement)
    if (
        payload is None
        or payload != _cfg_state_update_measurement_payload(measurement)
    ):
        raise ValueError("CFG state-update measurement 未经raw-array factory签发")
    return measurement


def _phase_projection_evaluation_payload(
    evaluation: PhaseProjectionSignEvaluation,
) -> tuple[Any, ...]:
    return tuple(
        getattr(evaluation, field_name)
        for field_name in evaluation.__dataclass_fields__
    )


def require_validated_phase_projection_sign_evaluation(
    evaluation: PhaseProjectionSignEvaluation,
) -> PhaseProjectionSignEvaluation:
    """Require a capability issued only after raw transition validation."""

    if not isinstance(evaluation, PhaseProjectionSignEvaluation):
        raise TypeError("phase projection sign evaluation 类型不匹配")
    payload = _ISSUED_PHASE_PROJECTION_EVALUATIONS.get(evaluation)
    if (
        payload is None
        or payload != _phase_projection_evaluation_payload(evaluation)
    ):
        raise ValueError("phase projection sign evaluation 未经raw-array验证")
    return evaluation


def _phase_projection_selection_payload(
    selection: SymmetricPhaseProjectionSelection,
) -> tuple[Any, ...]:
    return (
        selection.selected_scale,
        selection.realized_phase_magnitude_radians,
        selection.attempt_count,
        selection.backoff_count,
        selection.initial_positive,
        selection.initial_negative,
        selection.final_positive,
        selection.final_negative,
    )


def require_validated_symmetric_phase_projection(
    selection: SymmetricPhaseProjectionSelection,
) -> SymmetricPhaseProjectionSelection:
    """Require the in-process capability issued by the bounded search."""

    if not isinstance(selection, SymmetricPhaseProjectionSelection):
        raise TypeError("phase projection selection 类型不匹配")
    payload = _ISSUED_PHASE_PROJECTION_SELECTIONS.get(selection)
    if (
        payload is None
        or payload != _phase_projection_selection_payload(selection)
    ):
        raise ValueError("phase projection selection 未经真实candidate search验证")
    return selection


def validate_selected_phase_projection_transition(
    selection: SymmetricPhaseProjectionSelection,
    measurement: CfgStateUpdateMeasurement,
    *,
    base_pair: CfgRopeApplicationPair,
    controlled_pair: CfgRopeApplicationPair,
    base_conditional_velocity: np.ndarray,
    base_unconditional_velocity: np.ndarray,
    controlled_conditional_velocity: np.ndarray,
    controlled_unconditional_velocity: np.ndarray,
) -> PhaseProjectionSignEvaluation:
    """Bind the selected raw four-branch evaluation to the real transition."""

    require_validated_symmetric_phase_projection(selection)
    if measurement.signed_coefficient == 1:
        evaluation = selection.final_positive
    elif measurement.signed_coefficient == -1:
        evaluation = selection.final_negative
    else:
        raise ValueError("phase projection transition 只允许active正负符号")
    require_validated_phase_projection_sign_evaluation(evaluation)
    base_cond = _require_velocity(
        base_conditional_velocity,
        "selected base conditional velocity",
    )
    base_uncond = _require_velocity(
        base_unconditional_velocity,
        "selected base unconditional velocity",
    )
    controlled_cond = _require_velocity(
        controlled_conditional_velocity,
        "selected controlled conditional velocity",
    )
    controlled_uncond = _require_velocity(
        controlled_unconditional_velocity,
        "selected controlled unconditional velocity",
    )
    if (
        evaluation.base_pair != base_pair
        or evaluation.controlled_pair != controlled_pair
        or evaluation.probe_id != measurement.probe_id
        or evaluation.step_index != measurement.step_index
        or evaluation.signed_coefficient != measurement.signed_coefficient
        or evaluation.input_binding_digest
        != measurement.input_binding_digest
        or evaluation.phase_projection_scale != selection.selected_scale
        or evaluation.feasible is not True
    ):
        raise ValueError("selected phase candidate 与scheduler transition身份不一致")
    raw_digest_fields = (
        (
            "base_raw_velocity_digest",
            _float32_array_digest(
                ("projection base conditional velocity", base_cond),
                ("projection base unconditional velocity", base_uncond),
            ),
        ),
        (
            "controlled_transition_digest",
            _float32_array_digest(
                (
                    "projection controlled conditional velocity",
                    controlled_cond,
                ),
                (
                    "projection controlled unconditional velocity",
                    controlled_uncond,
                ),
            ),
        ),
        (
            "candidate_context_digest",
            _phase_projection_candidate_context_digest(
                base_pair=base_pair,
                base_conditional_velocity=base_cond,
                base_unconditional_velocity=base_uncond,
                scheduler_sample=measurement.scheduler_sample,
                delta_sigma=measurement.delta_sigma,
                cumulative_reference_energy_before_step=(
                    measurement.cumulative_reference_energy_before_step
                ),
                cumulative_control_energy_before_step=(
                    measurement.cumulative_control_energy_before_step
                ),
                remaining_step_count=measurement.remaining_step_count,
            ),
        ),
    )
    for field_name, observed in raw_digest_fields:
        if getattr(evaluation, field_name) != observed:
            raise ValueError(
                "selected phase candidate 与scheduler raw branch不一致: "
                f"{field_name}"
            )
    exact_fields = (
        "delta_sigma",
        "cumulative_reference_energy_before_step",
        "cumulative_control_energy_before_step",
        "remaining_step_count",
        "base_velocity_norm",
        "actual_delta_norm",
        "state_update_delta_norm",
        "norm_budget",
        "reference_energy_increment",
        "projected_reference_energy",
        "total_flow_energy_budget",
        "energy_increment",
        "remaining_flow_energy",
        "direction_cosine",
        "signed_state_update_exposure",
        "norm_guard_passed",
        "energy_guard_passed",
        "direction_guard_passed",
    )
    for field_name in exact_fields:
        expected = getattr(measurement, field_name)
        if field_name == "direction_guard_passed":
            expected = expected is True
        if getattr(evaluation, field_name) != expected:
            raise ValueError(
                "selected phase candidate 与scheduler transition统计不一致: "
                f"{field_name}"
            )
    digest_fields = (
        (
            "base_cfg_velocity_digest",
            measurement.base_cfg_velocity,
        ),
        (
            "controlled_cfg_velocity_digest",
            measurement.controlled_cfg_velocity,
        ),
        ("scheduler_sample_digest", measurement.scheduler_sample),
        (
            "actual_state_update_digest",
            measurement.actual_state_update_delta,
        ),
    )
    for field_name, array in digest_fields:
        observed = sha256(array.tobytes(order="C")).hexdigest()
        if getattr(evaluation, field_name) != observed:
            raise ValueError(
                "selected phase candidate 与scheduler transition数组不一致: "
                f"{field_name}"
            )
    return evaluation


class PatchRelationRuntimeGuardError(RuntimeError):
    """Fail closed with scalar-only diagnostics."""

    def __init__(self, diagnostics: dict[str, object]) -> None:
        self.diagnostics = dict(diagnostics)
        super().__init__(
            "Patch-relation CFG/state-update runtime guard 失败: "
            + ",".join(
                f"{key}={self.diagnostics[key]!r}"
                for key in sorted(self.diagnostics)
            )
        )


def _require_digest(value: str, label: str) -> str:
    if not isinstance(value, str) or _LOWER_HEX_64.fullmatch(value) is None:
        raise ValueError(f"{label} 必须为64字符小写SHA-256")
    return value


def _require_probe_id(value: str) -> str:
    if not isinstance(value, str) or _PROBE_ID.fullmatch(value) is None:
        raise ValueError("probe_id 必须为非空 snake_case identifier")
    return value


def _require_signed_coefficient(value: int) -> int:
    if type(value) is not int or value not in (-1, 0, 1):
        raise ValueError("signed_coefficient 只允许 -1/0/+1")
    return value


def _require_phase_projection_scale(
    value: float,
    *,
    signed_coefficient: int,
) -> float:
    scale = float(value)
    if (
        not math.isfinite(scale)
        or scale < 0.0
        or scale > 1.0
        or (
            signed_coefficient != 0
            and scale < PHASE_PROJECTION_MINIMUM_NONZERO_SCALE
        )
    ):
        raise ValueError("phase projection scale 越界或低于冻结非零下限")
    return scale


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - exercised without torch
        raise RuntimeError(
            "Patch-relation Wan tensor adapter 需要现有 torch runtime"
        ) from error
    return torch


def _torch_all_finite(torch: Any, value: Any) -> bool:
    return bool(torch.isfinite(value).all().item())


def _validate_torch_hidden_states(torch: Any, hidden_states: Any) -> None:
    if not isinstance(hidden_states, torch.Tensor):
        raise TypeError("Wan hidden_states 必须为 torch.Tensor")
    if tuple(hidden_states.shape) != TRANSFORMER_HIDDEN_SHAPE:
        raise ValueError("Wan hidden_states shape 不匹配")
    if hidden_states.dtype != torch.bfloat16:
        raise ValueError("Wan hidden_states dtype 必须精确为 bfloat16")
    if getattr(hidden_states, "layout", None) != torch.strided:
        raise ValueError("Wan hidden_states 必须为 strided layout")
    if not hidden_states.is_contiguous():
        raise ValueError("Wan hidden_states 必须为 contiguous")
    if not _torch_all_finite(torch, hidden_states):
        raise ValueError("Wan hidden_states 必须全部有限")


def _validate_numpy_hidden_states(hidden_states: np.ndarray) -> None:
    array = np.asarray(hidden_states)
    if array.shape != TRANSFORMER_HIDDEN_SHAPE:
        raise ValueError("fake Wan hidden_states shape 不匹配")
    if array.dtype != np.dtype("<f4"):
        raise ValueError("fake Wan hidden_states dtype 必须精确为 float32")
    if not array.flags.c_contiguous:
        raise ValueError("fake Wan hidden_states 必须为 C-contiguous")
    if not np.all(np.isfinite(array)):
        raise ValueError("fake Wan hidden_states 必须全部有限")


def _validate_numpy_rope_tensor(
    value: np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != ROPE_TUPLE_SHAPE:
        raise ValueError(f"{label} shape 不匹配")
    expected_dtype = np.dtype("<f4")
    if array.dtype != expected_dtype:
        raise ValueError(
            f"{label} dtype 不匹配: "
            f"expected={expected_dtype.name}, observed={array.dtype.name}"
        )
    if not array.flags.c_contiguous:
        raise ValueError(f"{label} 必须为 contiguous")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} 必须全部有限")
    return array


def _validate_torch_rope_tensor(
    torch: Any,
    value: Any,
    *,
    label: str,
    expected_device: Any | None = None,
) -> Any:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} 必须为 torch.Tensor")
    if tuple(value.shape) != ROPE_TUPLE_SHAPE:
        raise ValueError(f"{label} shape 不匹配")
    if value.dtype != torch.float32:
        raise ValueError(
            f"{label} dtype 不匹配: "
            f"expected={torch.float32}, observed={value.dtype}"
        )
    if getattr(value, "layout", None) != torch.strided:
        raise ValueError(f"{label} 必须为 strided layout")
    if not value.is_contiguous():
        raise ValueError(f"{label} 必须为 contiguous")
    if expected_device is not None and value.device != expected_device:
        raise ValueError("Wan RoPE tuple device 不一致")
    if not _torch_all_finite(torch, value):
        raise ValueError(f"{label} 必须全部有限")
    return value


def apply_wan_rotary_phase_runtime(
    freqs_cos: Any,
    freqs_sin: Any,
    *,
    descriptor: PublicPatchRelationDescriptor,
    signed_coefficient: int,
    phase_projection_scale: float = 1.0,
) -> tuple[Any, Any]:
    """Apply the frozen core phase on NumPy fakes or real torch tensors.

    The real path changes 24 tuple scalar elements: entries 0 and 1 in both
    ``freqs_cos`` and ``freqs_sin`` for six active tokens.  Twelve of those
    elements are consumed by ``WanAttnProcessor`` as cosine-even entry 0 and
    sine-odd entry 1.  The exact active token/angle values come from
    ``build_relation_phase_delta`` rather than a second relation formula.
    """

    validate_public_patch_relation_descriptor(descriptor)
    coefficient = _require_signed_coefficient(signed_coefficient)
    scale = _require_phase_projection_scale(
        phase_projection_scale,
        signed_coefficient=coefficient,
    )
    if isinstance(freqs_cos, np.ndarray) or isinstance(freqs_sin, np.ndarray):
        if not isinstance(freqs_cos, np.ndarray) or not isinstance(
            freqs_sin, np.ndarray
        ):
            raise TypeError("fake Wan RoPE tuple backend 必须一致")
        cosine = _validate_numpy_rope_tensor(
            freqs_cos,
            label="fake Wan freqs_cos",
        )
        sine = _validate_numpy_rope_tensor(
            freqs_sin,
            label="fake Wan freqs_sin",
        )
        if coefficient == 0:
            return cosine, sine
        # The governed phase delta remains float64.  Compute the tiny rotation
        # stably in that schema, then restore the official float32 tuple
        # storage before the result reaches Wan attention.
        shifted = apply_wan_rotary_phase_numpy(
            np.ascontiguousarray(cosine, dtype="<f8"),
            np.ascontiguousarray(sine, dtype="<f8"),
            descriptor=descriptor,
            signed_coefficient=coefficient,
            phase_projection_scale=scale,
        )
        return (
            np.ascontiguousarray(shifted[0], dtype="<f4"),
            np.ascontiguousarray(shifted[1], dtype="<f4"),
        )

    torch = _import_torch()
    if torch.is_grad_enabled():
        raise RuntimeError("Wan RoPE runtime adapter 只允许 no-grad inference")
    cosine = _validate_torch_rope_tensor(
        torch,
        freqs_cos,
        label="Wan freqs_cos",
    )
    sine = _validate_torch_rope_tensor(
        torch,
        freqs_sin,
        label="Wan freqs_sin",
        expected_device=cosine.device,
    )
    if coefficient == 0:
        return cosine, sine

    phase_delta = build_relation_phase_delta(
        descriptor,
        signed_coefficient=coefficient,
        phase_projection_scale=scale,
    )
    active_token_indices = np.flatnonzero(phase_delta[0, :, 0, 0])
    if active_token_indices.size != 6:
        raise RuntimeError("冻结 Patch relation active token 数量不匹配")
    if not np.array_equal(
        phase_delta[0, active_token_indices, 0, 0],
        phase_delta[0, active_token_indices, 0, 1],
    ) or np.count_nonzero(phase_delta[..., 2:]) != 0:
        raise RuntimeError("冻结 temporal RoPE pair phase layout 不匹配")

    token_indices = torch.as_tensor(
        active_token_indices,
        dtype=torch.long,
        device=cosine.device,
    )
    angles = torch.as_tensor(
        phase_delta[0, active_token_indices, 0, 0],
        dtype=torch.float64,
        device=cosine.device,
    )
    angle_cosine = torch.cos(angles)
    angle_sine = torch.sin(angles)
    shifted_cosine = cosine.clone()
    shifted_sine = sine.clone()
    for entry in (0, 1):
        original_cosine = cosine[0, token_indices, 0, entry]
        original_sine = sine[0, token_indices, 0, entry]
        shifted_cosine[0, token_indices, 0, entry] = (
            original_cosine * angle_cosine - original_sine * angle_sine
        ).to(dtype=cosine.dtype)
        shifted_sine[0, token_indices, 0, entry] = (
            original_sine * angle_cosine + original_cosine * angle_sine
        ).to(dtype=sine.dtype)
    _validate_torch_rope_tensor(
        torch,
        shifted_cosine,
        label="shifted Wan freqs_cos",
        expected_device=cosine.device,
    )
    _validate_torch_rope_tensor(
        torch,
        shifted_sine,
        label="shifted Wan freqs_sin",
        expected_device=cosine.device,
    )
    return shifted_cosine, shifted_sine


class ScopedWanRopeOutputAdapter:
    """Install one temporary RoPE output transform and restore exact state."""

    def __init__(
        self,
        transformer: Any,
        *,
        descriptor: PublicPatchRelationDescriptor,
        signed_coefficient: int,
        phase_projection_scale: float = 1.0,
        probe_id: str,
        step_index: int,
        control_role: str,
        cfg_branch_role: str,
        input_binding_digest: str,
    ) -> None:
        validate_public_patch_relation_descriptor(descriptor)
        self._transformer = transformer
        self._descriptor = descriptor
        self._signed_coefficient = _require_signed_coefficient(
            signed_coefficient
        )
        self._phase_projection_scale = _require_phase_projection_scale(
            phase_projection_scale,
            signed_coefficient=self._signed_coefficient,
        )
        self._probe_id = _require_probe_id(probe_id)
        if type(step_index) is not int or not 0 <= step_index < 8:
            raise ValueError("step_index 必须位于冻结8-step范围")
        self._step_index = step_index
        if control_role not in ("base", "controlled"):
            raise ValueError("control_role 只允许 base/controlled")
        if control_role == "base" and self._signed_coefficient != 0:
            raise ValueError("base RoPE scope 必须使用 zero control")
        self._control_role = control_role
        if cfg_branch_role not in CFG_BRANCH_ORDER:
            raise ValueError("cfg_branch_role 不匹配")
        self._cfg_branch_role = cfg_branch_role
        self._input_binding_digest = _require_digest(
            input_binding_digest,
            "input_binding_digest",
        )
        self._attempted_call_count = 0
        self._successful_call_count = 0
        self._entered = False
        self._exit_attempted = False
        self._cleanup_completed = False
        self._completed_successfully = False
        self._rope: Any | None = None
        self._had_instance_forward = False
        self._previous_instance_forward: Any | None = None
        self._original_forward: Any | None = None

    def __enter__(self) -> ScopedWanRopeOutputAdapter:
        if self._entered:
            raise RuntimeError("Wan RoPE scope 不得重复进入")
        rope = getattr(self._transformer, "rope", None)
        if rope is None or not callable(getattr(rope, "forward", None)):
            raise TypeError("transformer.rope.forward 不可用")
        if getattr(rope, _ACTIVE_SCOPE_ATTRIBUTE, False):
            raise RuntimeError("Wan RoPE scope 不允许嵌套")
        self._rope = rope
        instance_dict = getattr(rope, "__dict__", {})
        self._had_instance_forward = "forward" in instance_dict
        self._previous_instance_forward = instance_dict.get("forward")
        self._original_forward = rope.forward

        def scoped_forward(hidden_states: Any, *args: Any, **kwargs: Any) -> Any:
            self._attempted_call_count += 1
            if self._successful_call_count >= EXPECTED_ROPE_CALLS_PER_BRANCH:
                raise RuntimeError("Wan RoPE scope 收到超额 forward 调用")
            if isinstance(hidden_states, np.ndarray):
                _validate_numpy_hidden_states(hidden_states)
            else:
                torch = _import_torch()
                _validate_torch_hidden_states(torch, hidden_states)
            result = self._original_forward(hidden_states, *args, **kwargs)
            if not isinstance(result, tuple) or len(result) != 2:
                raise TypeError("transformer.rope 必须返回(freqs_cos,freqs_sin)")
            if not isinstance(hidden_states, np.ndarray):
                if (
                    result[0].device != hidden_states.device
                    or result[1].device != hidden_states.device
                ):
                    raise ValueError(
                        "Wan RoPE tuple 必须与 hidden_states 位于同一 device"
                    )
            shifted = apply_wan_rotary_phase_runtime(
                result[0],
                result[1],
                descriptor=self._descriptor,
                signed_coefficient=self._signed_coefficient,
                phase_projection_scale=self._phase_projection_scale,
            )
            self._successful_call_count += 1
            if self._signed_coefficient == 0:
                return result
            return shifted

        setattr(rope, _ACTIVE_SCOPE_ATTRIBUTE, True)
        try:
            setattr(rope, "forward", scoped_forward)
        except BaseException:
            delattr(rope, _ACTIVE_SCOPE_ATTRIBUTE)
            raise
        self._entered = True
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        if self._exit_attempted:
            self._cleanup_completed = False
            self._completed_successfully = False
            raise RuntimeError("Wan RoPE scope 不得重复退出")
        self._exit_attempted = True
        self._cleanup_completed = False
        self._completed_successfully = False
        cleanup_error: BaseException | None = None
        try:
            if self._rope is not None:
                if self._had_instance_forward:
                    setattr(
                        self._rope,
                        "forward",
                        self._previous_instance_forward,
                    )
                else:
                    delattr(self._rope, "forward")
                delattr(self._rope, _ACTIVE_SCOPE_ATTRIBUTE)
                instance_dict = getattr(self._rope, "__dict__", {})
                if self._had_instance_forward:
                    if (
                        instance_dict.get("forward")
                        is not self._previous_instance_forward
                    ):
                        raise RuntimeError(
                            "Wan RoPE scope 未恢复原 instance forward"
                        )
                elif "forward" in instance_dict:
                    raise RuntimeError(
                        "Wan RoPE scope 未移除临时 instance forward"
                    )
                if hasattr(self._rope, _ACTIVE_SCOPE_ATTRIBUTE):
                    raise RuntimeError("Wan RoPE scope active state 未清除")
                self._cleanup_completed = True
        except BaseException as error:  # pragma: no cover - defensive cleanup
            cleanup_error = error
        if exception is not None:
            if cleanup_error is not None and hasattr(exception, "add_note"):
                exception.add_note(
                    "Wan RoPE scope cleanup also failed: "
                    f"{type(cleanup_error).__name__}"
                )
            return False
        if cleanup_error is not None:
            raise cleanup_error
        if (
            self._attempted_call_count != EXPECTED_ROPE_CALLS_PER_BRANCH
            or self._successful_call_count != EXPECTED_ROPE_CALLS_PER_BRANCH
        ):
            raise RuntimeError("Wan RoPE scope 未收到精确一次 forward")
        self._completed_successfully = True
        return False

    def record(self) -> WanRopeBranchApplicationRecord:
        if not self._exit_attempted:
            raise RuntimeError("Wan RoPE scope record 只能在退出后读取")
        if not self._cleanup_completed or not self._completed_successfully:
            raise RuntimeError(
                "Wan RoPE scope record 仅允许完整clean exit成功后读取"
            )
        if (
            self._attempted_call_count != EXPECTED_ROPE_CALLS_PER_BRANCH
            or self._successful_call_count != EXPECTED_ROPE_CALLS_PER_BRANCH
        ):
            raise RuntimeError("Wan RoPE scope record call count 不完整")
        return WanRopeBranchApplicationRecord(
            probe_id=self._probe_id,
            step_index=self._step_index,
            control_role=self._control_role,
            cfg_branch_role=self._cfg_branch_role,
            cfg_branch_order_index=CFG_BRANCH_ORDER.index(
                self._cfg_branch_role
            ),
            signed_coefficient=self._signed_coefficient,
            maximum_phase_budget_radians=PHASE_BUDGET_RADIANS,
            phase_projection_scale=self._phase_projection_scale,
            realized_phase_magnitude_radians=(
                0.0
                if self._signed_coefficient == 0
                else PHASE_BUDGET_RADIANS * self._phase_projection_scale
            ),
            input_binding_digest=self._input_binding_digest,
            descriptor_digest=self._descriptor.descriptor_digest,
            rope_call_attempt_count=self._attempted_call_count,
            successful_rope_call_count=self._successful_call_count,
            expected_rope_call_count=EXPECTED_ROPE_CALLS_PER_BRANCH,
            scope_completed_successfully=True,
            clean_exact_noop=self._signed_coefficient == 0,
        )


def validate_cfg_rope_application_pair(
    conditional: WanRopeBranchApplicationRecord,
    unconditional: WanRopeBranchApplicationRecord,
) -> CfgRopeApplicationPair:
    """Bind official Wan CFG order and the same relation control on both calls."""

    records = (conditional, unconditional)
    expected_descriptor_digest = (
        build_public_patch_relation_descriptor().descriptor_digest
    )
    for record in records:
        if (
            not record.local_contract_only
            or record.execution_evidence_allowed
            or record.formal_result
            or record.stage_progression_allowed
            or record.rope_call_attempt_count
            != EXPECTED_ROPE_CALLS_PER_BRANCH
            or record.successful_rope_call_count
            != EXPECTED_ROPE_CALLS_PER_BRANCH
            or record.expected_rope_call_count
            != EXPECTED_ROPE_CALLS_PER_BRANCH
            or record.scope_completed_successfully is not True
        ):
            raise ValueError("CFG RoPE branch record boundary 不匹配")
        _require_probe_id(record.probe_id)
        _require_signed_coefficient(record.signed_coefficient)
        _require_digest(record.input_binding_digest, "input_binding_digest")
        _require_digest(record.descriptor_digest, "descriptor_digest")
        if (
            type(record.step_index) is not int
            or not 0 <= record.step_index < 8
            or record.control_role not in ("base", "controlled")
            or record.clean_exact_noop
            is not (record.signed_coefficient == 0)
            or record.descriptor_digest != expected_descriptor_digest
            or record.maximum_phase_budget_radians != PHASE_BUDGET_RADIANS
            or (
                record.signed_coefficient == 0
                and record.phase_projection_scale != 1.0
            )
            or record.phase_projection_scale
            != _require_phase_projection_scale(
                record.phase_projection_scale,
                signed_coefficient=record.signed_coefficient,
            )
            or record.realized_phase_magnitude_radians
            != (
                0.0
                if record.signed_coefficient == 0
                else PHASE_BUDGET_RADIANS * record.phase_projection_scale
            )
        ):
            raise ValueError("CFG RoPE branch record identity/layout 不匹配")
    if (
        conditional.cfg_branch_role != "conditional"
        or conditional.cfg_branch_order_index != 0
        or unconditional.cfg_branch_role != "unconditional"
        or unconditional.cfg_branch_order_index != 1
    ):
        raise ValueError("CFG branch role/order 不匹配")
    common_fields = (
        "probe_id",
        "step_index",
        "control_role",
        "signed_coefficient",
        "maximum_phase_budget_radians",
        "phase_projection_scale",
        "realized_phase_magnitude_radians",
        "input_binding_digest",
        "descriptor_digest",
        "clean_exact_noop",
    )
    if any(
        getattr(conditional, field) != getattr(unconditional, field)
        for field in common_fields
    ):
        raise ValueError("conditional/unconditional relation control 不一致")
    if (
        conditional.control_role == "base"
        and conditional.signed_coefficient != 0
    ):
        raise ValueError("base CFG pair 必须为 zero control")
    return CfgRopeApplicationPair(
        probe_id=conditional.probe_id,
        step_index=conditional.step_index,
        control_role=conditional.control_role,
        signed_coefficient=conditional.signed_coefficient,
        maximum_phase_budget_radians=conditional.maximum_phase_budget_radians,
        phase_projection_scale=conditional.phase_projection_scale,
        realized_phase_magnitude_radians=(
            conditional.realized_phase_magnitude_radians
        ),
        input_binding_digest=conditional.input_binding_digest,
        descriptor_digest=conditional.descriptor_digest,
        conditional=conditional,
        unconditional=unconditional,
    )


def _require_velocity(values: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(values)
    if array.shape != SCHEDULER_VELOCITY_SHAPE:
        raise ValueError(f"{label} shape 不匹配")
    if array.dtype != np.dtype("<f4"):
        raise ValueError(f"{label} dtype 必须精确为 float32")
    if not array.flags.c_contiguous:
        raise ValueError(f"{label} 必须为 C-contiguous")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} 必须全部有限")
    return array


def _float32_norm(values: np.ndarray) -> float:
    return float(np.linalg.norm(values.reshape(-1)))


def _safe_cosine(left: np.ndarray, right: np.ndarray) -> float:
    left64 = left.astype(np.float64, copy=False).reshape(-1)
    right64 = right.astype(np.float64, copy=False).reshape(-1)
    denominator = float(np.linalg.norm(left64) * np.linalg.norm(right64))
    if denominator <= 0.0 or not math.isfinite(denominator):
        raise ValueError("runtime direction cosine denominator 退化")
    raw = float(np.dot(left64, right64) / denominator)
    if (
        not math.isfinite(raw)
        or raw < -1.0 - 1e-12
        or raw > 1.0 + 1e-12
    ):
        raise ValueError("runtime direction cosine 超出Cauchy边界")
    return min(1.0, max(-1.0, raw))


def _cfg_combine(
    conditional: np.ndarray,
    unconditional: np.ndarray,
) -> np.ndarray:
    delta = np.subtract(conditional, unconditional, dtype=np.float32)
    scaled = np.multiply(delta, np.float32(CFG_GUIDANCE_SCALE), dtype=np.float32)
    return np.ascontiguousarray(
        np.add(unconditional, scaled, dtype=np.float32),
        dtype="<f4",
    )


def measure_cfg_state_update_numpy(
    *,
    base_pair: CfgRopeApplicationPair,
    controlled_pair: CfgRopeApplicationPair,
    base_conditional_velocity: np.ndarray,
    base_unconditional_velocity: np.ndarray,
    controlled_conditional_velocity: np.ndarray,
    controlled_unconditional_velocity: np.ndarray,
    scheduler_consumed_velocity: np.ndarray,
    scheduler_sample: np.ndarray,
    scheduler_base_next_state: np.ndarray,
    scheduler_controlled_next_state: np.ndarray,
    delta_sigma: float,
    cumulative_reference_energy_before_step: float,
    cumulative_control_energy_before_step: float,
    remaining_step_count: int,
) -> CfgStateUpdateMeasurement:
    """Recompute FP32 CFG and bind it to the scheduler's returned next state.

    The caller must supply both the actual controlled scheduler result and a
    counterfactual base result computed with the same frozen Euler/dtype
    semantics.  This local function checks both against the formula, then uses
    their realized difference for direction, energy, and signed exposure.  It
    still cannot establish full-schedule provenance and remains ineligible as
    execution evidence until promoted by the governed runner.
    """

    validated_base = validate_cfg_rope_application_pair(
        base_pair.conditional,
        base_pair.unconditional,
    )
    validated_controlled = validate_cfg_rope_application_pair(
        controlled_pair.conditional,
        controlled_pair.unconditional,
    )
    if validated_base != base_pair or validated_controlled != controlled_pair:
        raise ValueError("CFG pair dataclass 与重建结果不一致")
    if (
        base_pair.control_role != "base"
        or base_pair.signed_coefficient != 0
        or controlled_pair.control_role != "controlled"
    ):
        raise ValueError("base/controlled CFG role 不匹配")
    common_fields = (
        "probe_id",
        "step_index",
        "input_binding_digest",
        "descriptor_digest",
    )
    if any(
        getattr(base_pair, field) != getattr(controlled_pair, field)
        for field in common_fields
    ):
        raise ValueError("base/controlled forward input binding 不一致")
    coefficient = _require_signed_coefficient(
        controlled_pair.signed_coefficient
    )
    interval = float(np.float32(delta_sigma))
    if not math.isfinite(interval) or interval >= 0.0:
        raise ValueError("delta_sigma 必须为有限负值")
    cumulative_reference = float(cumulative_reference_energy_before_step)
    cumulative_control = float(cumulative_control_energy_before_step)
    if (
        not math.isfinite(cumulative_reference)
        or cumulative_reference < 0.0
        or not math.isfinite(cumulative_control)
        or cumulative_control < 0.0
        or type(remaining_step_count) is not int
        or remaining_step_count <= 0
    ):
        raise ValueError("Flow energy schedule context 不完整")

    base_cond = _require_velocity(
        base_conditional_velocity,
        "base conditional velocity",
    )
    base_uncond = _require_velocity(
        base_unconditional_velocity,
        "base unconditional velocity",
    )
    controlled_cond = _require_velocity(
        controlled_conditional_velocity,
        "controlled conditional velocity",
    )
    controlled_uncond = _require_velocity(
        controlled_unconditional_velocity,
        "controlled unconditional velocity",
    )
    base_raw_velocity_digest = _float32_array_digest(
        ("measurement base conditional velocity", base_cond),
        ("measurement base unconditional velocity", base_uncond),
    )
    controlled_raw_velocity_digest = _float32_array_digest(
        ("measurement controlled conditional velocity", controlled_cond),
        ("measurement controlled unconditional velocity", controlled_uncond),
    )
    base_cfg = _cfg_combine(base_cond, base_uncond)
    controlled_cfg = _cfg_combine(controlled_cond, controlled_uncond)
    consumed_cfg = _require_velocity(
        scheduler_consumed_velocity,
        "scheduler consumed CFG velocity",
    )
    sample = _require_velocity(scheduler_sample, "scheduler sample")
    base_next = _require_velocity(
        scheduler_base_next_state,
        "scheduler base next state",
    )
    controlled_next = _require_velocity(
        scheduler_controlled_next_state,
        "scheduler controlled next state",
    )
    if not np.array_equal(consumed_cfg, controlled_cfg):
        raise ValueError(
            "scheduler consumed CFG velocity 与 controlled FP32 CFG 不一致"
        )
    intended_delta = np.ascontiguousarray(
        np.subtract(controlled_cfg, base_cfg, dtype=np.float32),
        dtype="<f4",
    )
    # The official scheduler consumes the controlled CFG output itself.  Keep
    # that exact FP32 array as the constrained velocity instead of reconstructing
    # it through base+(controlled-base), which can differ by one ULP.
    constrained = np.ascontiguousarray(consumed_cfg, dtype="<f4")
    actual_delta = np.ascontiguousarray(
        np.subtract(constrained, base_cfg, dtype=np.float32),
        dtype="<f4",
    )
    intended_state_update = np.ascontiguousarray(
        np.multiply(intended_delta, np.float32(interval), dtype=np.float32),
        dtype="<f4",
    )
    expected_base_next = np.ascontiguousarray(
        np.add(
            sample,
            np.multiply(base_cfg, np.float32(interval), dtype=np.float32),
            dtype=np.float32,
        ),
        dtype="<f4",
    )
    expected_controlled_next = np.ascontiguousarray(
        np.add(
            sample,
            np.multiply(
                controlled_cfg,
                np.float32(interval),
                dtype=np.float32,
            ),
            dtype=np.float32,
        ),
        dtype="<f4",
    )
    if (
        not np.array_equal(base_next, expected_base_next)
        or not np.array_equal(controlled_next, expected_controlled_next)
    ):
        raise ValueError("scheduler next-state 与冻结Euler FP32语义不一致")
    base_state_update = np.ascontiguousarray(
        np.subtract(base_next, sample, dtype=np.float32),
        dtype="<f4",
    )
    controlled_state_update = np.ascontiguousarray(
        np.subtract(controlled_next, sample, dtype=np.float32),
        dtype="<f4",
    )
    state_update_delta = np.ascontiguousarray(
        np.subtract(controlled_next, base_next, dtype=np.float32),
        dtype="<f4",
    )
    base_norm = _float32_norm(base_cfg)
    intended_norm = _float32_norm(intended_delta)
    actual_norm = _float32_norm(actual_delta)
    base_state_update_norm = _float32_norm(base_state_update)
    intended_state_update_norm = _float32_norm(intended_state_update)
    state_update_norm = _float32_norm(state_update_delta)
    if not all(
        math.isfinite(value)
        for value in (
            base_norm,
            intended_norm,
            actual_norm,
            base_state_update_norm,
            intended_state_update_norm,
            state_update_norm,
        )
    ):
        raise ValueError("CFG/state-update norm 非有限")
    norm_budget = (
        base_norm * VELOCITY_NORM_RATIO_BUDGET * LAMBDA_MAX
    )
    reference_increment = base_state_update_norm**2
    projected_reference = (
        cumulative_reference
        + reference_increment * remaining_step_count
    )
    total_flow_energy_budget = (
        FLOW_ENERGY_BUDGET_RATIO * projected_reference
    )
    remaining = max(
        0.0,
        total_flow_energy_budget - cumulative_control,
    )
    energy_increment = state_update_norm**2
    norm_guard = _budget_guard_passed(actual_norm, norm_budget)
    energy_guard = _budget_guard_passed(energy_increment, remaining)

    if coefficient == 0:
        if (
            not np.array_equal(base_cond, controlled_cond)
            or not np.array_equal(base_uncond, controlled_uncond)
            or np.count_nonzero(intended_delta) != 0
            or np.count_nonzero(actual_delta) != 0
            or np.count_nonzero(state_update_delta) != 0
            or not np.array_equal(base_next, controlled_next)
        ):
            raise PatchRelationRuntimeGuardError(
                {
                    "clean_exact_noop": False,
                    "probe_id": base_pair.probe_id,
                    "step_index": base_pair.step_index,
                }
            )
        return _issue_cfg_state_update_measurement(CfgStateUpdateMeasurement(
            probe_id=base_pair.probe_id,
            step_index=base_pair.step_index,
            signed_coefficient=0,
            input_binding_digest=base_pair.input_binding_digest,
            base_raw_velocity_digest=base_raw_velocity_digest,
            controlled_raw_velocity_digest=controlled_raw_velocity_digest,
            base_cfg_velocity=base_cfg,
            controlled_cfg_velocity=controlled_cfg,
            intended_delta_velocity=intended_delta,
            constrained_velocity=constrained,
            actual_delta_velocity=actual_delta,
            scheduler_sample=sample,
            base_next_state=base_next,
            controlled_next_state=controlled_next,
            base_state_update_delta=base_state_update,
            intended_state_update_delta=intended_state_update,
            actual_state_update_delta=state_update_delta,
            delta_sigma=interval,
            base_velocity_norm=base_norm,
            intended_delta_norm=0.0,
            actual_delta_norm=0.0,
            base_state_update_norm=base_state_update_norm,
            intended_state_update_norm=0.0,
            state_update_delta_norm=0.0,
            state_update_direction_dot=0.0,
            direction_actual_norm=0.0,
            direction_intended_norm=0.0,
            norm_budget=norm_budget,
            cumulative_reference_energy_before_step=cumulative_reference,
            cumulative_control_energy_before_step=cumulative_control,
            remaining_step_count=remaining_step_count,
            reference_energy_increment=reference_increment,
            projected_reference_energy=projected_reference,
            total_flow_energy_budget=total_flow_energy_budget,
            remaining_flow_energy=remaining,
            energy_increment=0.0,
            direction_cosine=None,
            signed_state_update_exposure=0.0,
            norm_guard_passed=True,
            energy_guard_passed=True,
            direction_guard_passed=None,
            clean_exact_noop=True,
        ))

    if base_norm <= 0.0 or intended_norm <= 0.0 or actual_norm <= 0.0:
        raise PatchRelationRuntimeGuardError(
            {
                "actual_delta_norm": actual_norm,
                "base_velocity_norm": base_norm,
                "intended_delta_norm": intended_norm,
                "probe_id": base_pair.probe_id,
                "step_index": base_pair.step_index,
            }
        )
    actual_direction_values = state_update_delta.astype(
        np.float64,
        copy=False,
    ).reshape(-1)
    intended_direction_values = intended_state_update.astype(
        np.float64,
        copy=False,
    ).reshape(-1)
    direction_dot = float(
        np.dot(actual_direction_values, intended_direction_values)
    )
    direction_actual_norm = float(np.linalg.norm(actual_direction_values))
    direction_intended_norm = float(np.linalg.norm(intended_direction_values))
    direction_cosine = _safe_cosine(
        state_update_delta,
        intended_state_update,
    )
    direction_guard = direction_cosine + 1e-12 >= MINIMUM_DIRECTION_COSINE
    if not norm_guard or not energy_guard or not direction_guard:
        raise PatchRelationRuntimeGuardError(
            {
                "actual_delta_norm": actual_norm,
                "base_velocity_norm": base_norm,
                "cumulative_control_energy_before_step": cumulative_control,
                "cumulative_reference_energy_before_step": cumulative_reference,
                "delta_sigma": interval,
                "direction_cosine": direction_cosine,
                "direction_guard_passed": direction_guard,
                "energy_guard_passed": energy_guard,
                "energy_increment": energy_increment,
                "projected_reference_energy": projected_reference,
                "reference_energy_increment": reference_increment,
                "remaining_step_count": remaining_step_count,
                "norm_budget": norm_budget,
                "norm_guard_passed": norm_guard,
                "probe_id": base_pair.probe_id,
                "remaining_flow_energy": remaining,
                "signed_state_update_exposure": (
                    float(coefficient) * state_update_norm
                ),
                "state_update_delta_norm": state_update_norm,
                "step_index": base_pair.step_index,
                "total_flow_energy_budget": total_flow_energy_budget,
            }
        )
    signed_exposure = float(coefficient) * state_update_norm
    return _issue_cfg_state_update_measurement(CfgStateUpdateMeasurement(
        probe_id=base_pair.probe_id,
        step_index=base_pair.step_index,
        signed_coefficient=coefficient,
        input_binding_digest=base_pair.input_binding_digest,
        base_raw_velocity_digest=base_raw_velocity_digest,
        controlled_raw_velocity_digest=controlled_raw_velocity_digest,
        base_cfg_velocity=base_cfg,
        controlled_cfg_velocity=controlled_cfg,
        intended_delta_velocity=intended_delta,
        constrained_velocity=constrained,
        actual_delta_velocity=actual_delta,
        scheduler_sample=sample,
        base_next_state=base_next,
        controlled_next_state=controlled_next,
        base_state_update_delta=base_state_update,
        intended_state_update_delta=intended_state_update,
        actual_state_update_delta=state_update_delta,
        delta_sigma=interval,
        base_velocity_norm=base_norm,
        intended_delta_norm=intended_norm,
        actual_delta_norm=actual_norm,
        base_state_update_norm=base_state_update_norm,
        intended_state_update_norm=intended_state_update_norm,
        state_update_delta_norm=state_update_norm,
        state_update_direction_dot=direction_dot,
        direction_actual_norm=direction_actual_norm,
        direction_intended_norm=direction_intended_norm,
        norm_budget=norm_budget,
        cumulative_reference_energy_before_step=cumulative_reference,
        cumulative_control_energy_before_step=cumulative_control,
        remaining_step_count=remaining_step_count,
        reference_energy_increment=reference_increment,
        projected_reference_energy=projected_reference,
        total_flow_energy_budget=total_flow_energy_budget,
        remaining_flow_energy=remaining,
        energy_increment=energy_increment,
        direction_cosine=direction_cosine,
        signed_state_update_exposure=signed_exposure,
        norm_guard_passed=norm_guard,
        energy_guard_passed=energy_guard,
        direction_guard_passed=direction_guard,
        clean_exact_noop=False,
    ))


def evaluate_phase_projection_sign_numpy(
    *,
    base_pair: CfgRopeApplicationPair,
    controlled_pair: CfgRopeApplicationPair,
    phase_projection_scale: float,
    base_conditional_velocity: np.ndarray,
    base_unconditional_velocity: np.ndarray,
    controlled_conditional_velocity: np.ndarray,
    controlled_unconditional_velocity: np.ndarray,
    scheduler_sample: np.ndarray,
    delta_sigma: float,
    cumulative_reference_energy_before_step: float,
    cumulative_control_energy_before_step: float,
    remaining_step_count: int,
) -> PhaseProjectionSignEvaluation:
    """Validate one raw-array re-forward without advancing a scheduler.

    The returned object is an in-process consistency capability.  It is not a
    cryptographic authentication primitive, but callers cannot construct or
    replace it from scalar summaries alone.
    """

    coefficient = _require_signed_coefficient(
        controlled_pair.signed_coefficient
    )
    if coefficient == 0:
        raise ValueError("phase projection candidate 只允许active正负符号")
    scale = _require_phase_projection_scale(
        phase_projection_scale,
        signed_coefficient=coefficient,
    )
    if (
        controlled_pair.phase_projection_scale != scale
        or base_pair.phase_projection_scale != 1.0
    ):
        raise ValueError("phase projection candidate pair scale 不匹配")
    base_cond = _require_velocity(
        base_conditional_velocity,
        "projection base conditional velocity",
    )
    base_uncond = _require_velocity(
        base_unconditional_velocity,
        "projection base unconditional velocity",
    )
    controlled_cond = _require_velocity(
        controlled_conditional_velocity,
        "projection controlled conditional velocity",
    )
    controlled_uncond = _require_velocity(
        controlled_unconditional_velocity,
        "projection controlled unconditional velocity",
    )
    sample = _require_velocity(scheduler_sample, "projection scheduler sample")
    interval = float(np.float32(delta_sigma))
    context_digest = _phase_projection_candidate_context_digest(
        base_pair=base_pair,
        base_conditional_velocity=base_cond,
        base_unconditional_velocity=base_uncond,
        scheduler_sample=sample,
        delta_sigma=interval,
        cumulative_reference_energy_before_step=(
            cumulative_reference_energy_before_step
        ),
        cumulative_control_energy_before_step=(
            cumulative_control_energy_before_step
        ),
        remaining_step_count=remaining_step_count,
    )
    controlled_transition_digest = _float32_array_digest(
        ("projection controlled conditional velocity", controlled_cond),
        ("projection controlled unconditional velocity", controlled_uncond),
    )
    base_raw_velocity_digest = _float32_array_digest(
        ("projection base conditional velocity", base_cond),
        ("projection base unconditional velocity", base_uncond),
    )
    base_cfg = _cfg_combine(base_cond, base_uncond)
    controlled_cfg = _cfg_combine(controlled_cond, controlled_uncond)
    base_next = np.ascontiguousarray(
        np.add(
            sample,
            np.multiply(base_cfg, np.float32(interval), dtype=np.float32),
            dtype=np.float32,
        ),
        dtype="<f4",
    )
    controlled_next = np.ascontiguousarray(
        np.add(
            sample,
            np.multiply(
                controlled_cfg,
                np.float32(interval),
                dtype=np.float32,
            ),
            dtype=np.float32,
        ),
        dtype="<f4",
    )
    actual_state_update = np.ascontiguousarray(
        np.subtract(controlled_next, base_next, dtype=np.float32),
        dtype="<f4",
    )
    base_cfg_digest = sha256(base_cfg.tobytes(order="C")).hexdigest()
    controlled_cfg_digest = sha256(
        controlled_cfg.tobytes(order="C")
    ).hexdigest()
    scheduler_sample_digest = sha256(sample.tobytes(order="C")).hexdigest()
    actual_state_update_digest = sha256(
        actual_state_update.tobytes(order="C")
    ).hexdigest()
    try:
        measurement = measure_cfg_state_update_numpy(
            base_pair=base_pair,
            controlled_pair=controlled_pair,
            base_conditional_velocity=base_cond,
            base_unconditional_velocity=base_uncond,
            controlled_conditional_velocity=controlled_cond,
            controlled_unconditional_velocity=controlled_uncond,
            scheduler_consumed_velocity=controlled_cfg,
            scheduler_sample=sample,
            scheduler_base_next_state=base_next,
            scheduler_controlled_next_state=controlled_next,
            delta_sigma=interval,
            cumulative_reference_energy_before_step=(
                cumulative_reference_energy_before_step
            ),
            cumulative_control_energy_before_step=(
                cumulative_control_energy_before_step
            ),
            remaining_step_count=remaining_step_count,
        )
    except PatchRelationRuntimeGuardError as error:
        diagnostics = error.diagnostics
        actual_delta_norm = float(diagnostics.get("actual_delta_norm", 0.0))
        norm_budget = float(diagnostics.get("norm_budget", 0.0))
        energy_increment = float(diagnostics.get("energy_increment", math.inf))
        remaining = float(diagnostics.get("remaining_flow_energy", 0.0))
        direction = diagnostics.get("direction_cosine")
        direction_cosine = None if direction is None else float(direction)
        scalars = (
            actual_delta_norm,
            norm_budget,
            energy_increment,
            remaining,
        )
        if not all(math.isfinite(value) for value in scalars):
            raise PatchRelationRuntimeGuardError(
                {
                    "phase_projection_candidate_nonfinite": True,
                    "phase_projection_scale": scale,
                    "signed_coefficient": coefficient,
                }
            ) from error
        evaluation = PhaseProjectionSignEvaluation(
            probe_id=base_pair.probe_id,
            step_index=base_pair.step_index,
            signed_coefficient=coefficient,
            input_binding_digest=base_pair.input_binding_digest,
            descriptor_digest=base_pair.descriptor_digest,
            candidate_context_digest=context_digest,
            controlled_transition_digest=controlled_transition_digest,
            base_raw_velocity_digest=base_raw_velocity_digest,
            base_cfg_velocity_digest=base_cfg_digest,
            controlled_cfg_velocity_digest=controlled_cfg_digest,
            scheduler_sample_digest=scheduler_sample_digest,
            actual_state_update_digest=actual_state_update_digest,
            base_pair=base_pair,
            controlled_pair=controlled_pair,
            phase_projection_scale=scale,
            delta_sigma=float(diagnostics["delta_sigma"]),
            cumulative_reference_energy_before_step=float(
                diagnostics["cumulative_reference_energy_before_step"]
            ),
            cumulative_control_energy_before_step=float(
                diagnostics["cumulative_control_energy_before_step"]
            ),
            remaining_step_count=int(diagnostics["remaining_step_count"]),
            base_velocity_norm=float(diagnostics["base_velocity_norm"]),
            actual_delta_norm=actual_delta_norm,
            state_update_delta_norm=float(
                diagnostics["state_update_delta_norm"]
            ),
            norm_budget=norm_budget,
            reference_energy_increment=float(
                diagnostics["reference_energy_increment"]
            ),
            projected_reference_energy=float(
                diagnostics["projected_reference_energy"]
            ),
            total_flow_energy_budget=float(
                diagnostics["total_flow_energy_budget"]
            ),
            energy_increment=energy_increment,
            remaining_flow_energy=remaining,
            direction_cosine=direction_cosine,
            signed_state_update_exposure=float(
                diagnostics["signed_state_update_exposure"]
            ),
            norm_guard_passed=bool(
                diagnostics.get("norm_guard_passed", False)
            ),
            energy_guard_passed=bool(
                diagnostics.get("energy_guard_passed", False)
            ),
            direction_guard_passed=bool(
                diagnostics.get("direction_guard_passed", False)
            ),
            feasible=False,
        )
        _ISSUED_PHASE_PROJECTION_EVALUATIONS[
            evaluation
        ] = _phase_projection_evaluation_payload(evaluation)
        return evaluation
    evaluation = PhaseProjectionSignEvaluation(
        probe_id=measurement.probe_id,
        step_index=measurement.step_index,
        signed_coefficient=coefficient,
        input_binding_digest=measurement.input_binding_digest,
        descriptor_digest=controlled_pair.descriptor_digest,
        candidate_context_digest=context_digest,
        controlled_transition_digest=controlled_transition_digest,
        base_raw_velocity_digest=base_raw_velocity_digest,
        base_cfg_velocity_digest=base_cfg_digest,
        controlled_cfg_velocity_digest=controlled_cfg_digest,
        scheduler_sample_digest=scheduler_sample_digest,
        actual_state_update_digest=actual_state_update_digest,
        base_pair=base_pair,
        controlled_pair=controlled_pair,
        phase_projection_scale=scale,
        delta_sigma=measurement.delta_sigma,
        cumulative_reference_energy_before_step=(
            measurement.cumulative_reference_energy_before_step
        ),
        cumulative_control_energy_before_step=(
            measurement.cumulative_control_energy_before_step
        ),
        remaining_step_count=measurement.remaining_step_count,
        base_velocity_norm=measurement.base_velocity_norm,
        actual_delta_norm=measurement.actual_delta_norm,
        state_update_delta_norm=measurement.state_update_delta_norm,
        norm_budget=measurement.norm_budget,
        reference_energy_increment=measurement.reference_energy_increment,
        projected_reference_energy=measurement.projected_reference_energy,
        total_flow_energy_budget=measurement.total_flow_energy_budget,
        energy_increment=measurement.energy_increment,
        remaining_flow_energy=measurement.remaining_flow_energy,
        direction_cosine=measurement.direction_cosine,
        signed_state_update_exposure=(
            measurement.signed_state_update_exposure
        ),
        norm_guard_passed=measurement.norm_guard_passed,
        energy_guard_passed=measurement.energy_guard_passed,
        direction_guard_passed=measurement.direction_guard_passed is True,
        feasible=True,
    )
    _ISSUED_PHASE_PROJECTION_EVALUATIONS[
        evaluation
    ] = _phase_projection_evaluation_payload(evaluation)
    return evaluation


def select_symmetric_phase_projection(
    evaluator: Callable[
        [float],
        tuple[PhaseProjectionSignEvaluation, PhaseProjectionSignEvaluation],
    ],
) -> SymmetricPhaseProjectionSelection:
    """Select one common scale from real positive/negative re-forwards.

    There is no refinement or result-dependent sweep.  The first candidate is
    the frozen maximum phase.  A failed norm/energy candidate deterministically
    proposes one smaller scale from its worst observed ratios and the frozen
    safety factor.  Direction or non-finite failure stops immediately.
    """

    if not callable(evaluator):
        raise TypeError("phase projection evaluator 必须可调用")
    shared_context_fields = (
        "probe_id",
        "step_index",
        "input_binding_digest",
        "descriptor_digest",
        "candidate_context_digest",
        "base_pair",
        "delta_sigma",
        "cumulative_reference_energy_before_step",
        "cumulative_control_energy_before_step",
        "remaining_step_count",
        "base_velocity_norm",
        "norm_budget",
        "reference_energy_increment",
        "projected_reference_energy",
        "total_flow_energy_budget",
        "remaining_flow_energy",
        "base_raw_velocity_digest",
        "base_cfg_velocity_digest",
        "scheduler_sample_digest",
    )
    scale = 1.0
    initial: tuple[
        PhaseProjectionSignEvaluation,
        PhaseProjectionSignEvaluation,
    ] | None = None
    final: tuple[
        PhaseProjectionSignEvaluation,
        PhaseProjectionSignEvaluation,
    ] | None = None
    attempt_count = 0
    last_evaluated_scale: float | None = None
    frozen_attempt_context: tuple[Any, ...] | None = None
    for attempt_index in range(PHASE_PROJECTION_MAX_ATTEMPTS):
        attempt_count = attempt_index + 1
        positive, negative = evaluator(scale)
        require_validated_phase_projection_sign_evaluation(positive)
        require_validated_phase_projection_sign_evaluation(negative)
        last_evaluated_scale = scale
        if (
            positive.signed_coefficient != 1
            or negative.signed_coefficient != -1
            or positive.phase_projection_scale != scale
            or negative.phase_projection_scale != scale
        ):
            raise ValueError("phase projection evaluator sign/scale 不匹配")
        if any(
            getattr(positive, field_name) != getattr(negative, field_name)
            for field_name in shared_context_fields
        ):
            raise ValueError("phase projection ±candidate context 不一致")
        attempt_context = tuple(
            getattr(positive, field_name)
            for field_name in shared_context_fields
        )
        if frozen_attempt_context is None:
            frozen_attempt_context = attempt_context
        elif attempt_context != frozen_attempt_context:
            raise ValueError(
                "phase projection backoff attempt shared context 漂移"
            )
        if (
            positive.controlled_pair.signed_coefficient != 1
            or negative.controlled_pair.signed_coefficient != -1
            or positive.controlled_pair.phase_projection_scale != scale
            or negative.controlled_pair.phase_projection_scale != scale
        ):
            raise ValueError("phase projection controlled pair 绑定不一致")
        evaluations = (positive, negative)
        for evaluation in evaluations:
            scalar_values = (
                evaluation.actual_delta_norm,
                evaluation.norm_budget,
                evaluation.energy_increment,
                evaluation.remaining_flow_energy,
            )
            if (
                not all(math.isfinite(value) for value in scalar_values)
                or evaluation.actual_delta_norm <= 0.0
                or evaluation.norm_budget <= 0.0
                or evaluation.energy_increment <= 0.0
                or evaluation.remaining_flow_energy < 0.0
                or evaluation.direction_cosine is None
                or not math.isfinite(evaluation.direction_cosine)
                or evaluation.direction_guard_passed is not True
                or evaluation.feasible
                is not (
                    evaluation.norm_guard_passed
                    and evaluation.energy_guard_passed
                    and evaluation.direction_guard_passed
                )
            ):
                raise PatchRelationRuntimeGuardError(
                    {
                        "phase_projection_attempt": attempt_index + 1,
                        "phase_projection_direction_or_nonfinite_failure": True,
                        "phase_projection_scale": scale,
                        "signed_coefficient": evaluation.signed_coefficient,
                    }
                )
        if initial is None:
            initial = evaluations
        final = evaluations
        if all(evaluation.feasible for evaluation in evaluations):
            selection = SymmetricPhaseProjectionSelection(
                selected_scale=scale,
                realized_phase_magnitude_radians=(
                    PHASE_BUDGET_RADIANS * scale
                ),
                attempt_count=attempt_index + 1,
                backoff_count=attempt_index,
                initial_positive=initial[0],
                initial_negative=initial[1],
                final_positive=positive,
                final_negative=negative,
            )
            _ISSUED_PHASE_PROJECTION_SELECTIONS[
                selection
            ] = _phase_projection_selection_payload(selection)
            return selection
        worst_norm_ratio = max(
            evaluation.actual_delta_norm / evaluation.norm_budget
            for evaluation in evaluations
        )
        worst_energy_ratio = max(
            (
                math.inf
                if evaluation.remaining_flow_energy <= 0.0
                else evaluation.energy_increment
                / evaluation.remaining_flow_energy
            )
            for evaluation in evaluations
        )
        limiting_factor = min(
            1.0 / worst_norm_ratio,
            1.0 / math.sqrt(worst_energy_ratio),
        )
        next_scale = float(
            scale * limiting_factor * PHASE_PROJECTION_SAFETY_FACTOR
        )
        if (
            not math.isfinite(next_scale)
            or next_scale >= scale
            or next_scale < PHASE_PROJECTION_MINIMUM_NONZERO_SCALE
        ):
            break
        scale = next_scale
    raise PatchRelationRuntimeGuardError(
        {
            "phase_projection_attempt_count": (
                attempt_count
            ),
            "phase_projection_maximum_phase_radians": PHASE_BUDGET_RADIANS,
            "phase_projection_minimum_nonzero_scale": (
                PHASE_PROJECTION_MINIMUM_NONZERO_SCALE
            ),
            "phase_projection_no_feasible_nonzero_scale": True,
            "phase_projection_last_scale": last_evaluated_scale,
            "phase_projection_final_worst_actual_delta_norm": (
                None
                if final is None
                else max(item.actual_delta_norm for item in final)
            ),
            "phase_projection_final_worst_energy_increment": (
                None
                if final is None
                else max(item.energy_increment for item in final)
            ),
        }
    )


def validate_cfg_state_update_measurement_numpy(
    measurement: CfgStateUpdateMeasurement,
    *,
    base_pair: CfgRopeApplicationPair,
    controlled_pair: CfgRopeApplicationPair,
    base_conditional_velocity: np.ndarray,
    base_unconditional_velocity: np.ndarray,
    controlled_conditional_velocity: np.ndarray,
    controlled_unconditional_velocity: np.ndarray,
) -> CfgStateUpdateMeasurement:
    """Rebuild every transition statistic from the retained raw FP32 arrays.

    This validator consumes the still-live raw conditional/unconditional
    branch arrays and independently rebuilds their FP32 CFG values, scheduler
    next states, norms, direction, energy, exposure, and guards.
    """

    _require_issued_cfg_state_update_measurement(measurement)
    base_cond = _require_velocity(
        base_conditional_velocity,
        "measurement validation base conditional velocity",
    )
    base_uncond = _require_velocity(
        base_unconditional_velocity,
        "measurement validation base unconditional velocity",
    )
    controlled_cond = _require_velocity(
        controlled_conditional_velocity,
        "measurement validation controlled conditional velocity",
    )
    controlled_uncond = _require_velocity(
        controlled_unconditional_velocity,
        "measurement validation controlled unconditional velocity",
    )
    observed_base_raw_digest = _float32_array_digest(
        ("measurement base conditional velocity", base_cond),
        ("measurement base unconditional velocity", base_uncond),
    )
    observed_controlled_raw_digest = _float32_array_digest(
        ("measurement controlled conditional velocity", controlled_cond),
        ("measurement controlled unconditional velocity", controlled_uncond),
    )
    if (
        measurement.base_raw_velocity_digest != observed_base_raw_digest
        or measurement.controlled_raw_velocity_digest
        != observed_controlled_raw_digest
    ):
        raise ValueError(
            "CFG state-update measurement 与创建时raw branches不一致"
        )
    rebuilt = measure_cfg_state_update_numpy(
        base_pair=base_pair,
        controlled_pair=controlled_pair,
        base_conditional_velocity=base_cond,
        base_unconditional_velocity=base_uncond,
        controlled_conditional_velocity=controlled_cond,
        controlled_unconditional_velocity=controlled_uncond,
        scheduler_consumed_velocity=measurement.constrained_velocity,
        scheduler_sample=measurement.scheduler_sample,
        scheduler_base_next_state=measurement.base_next_state,
        scheduler_controlled_next_state=measurement.controlled_next_state,
        delta_sigma=measurement.delta_sigma,
        cumulative_reference_energy_before_step=(
            measurement.cumulative_reference_energy_before_step
        ),
        cumulative_control_energy_before_step=(
            measurement.cumulative_control_energy_before_step
        ),
        remaining_step_count=measurement.remaining_step_count,
    )
    array_fields = (
        "base_cfg_velocity",
        "controlled_cfg_velocity",
        "intended_delta_velocity",
        "constrained_velocity",
        "actual_delta_velocity",
        "scheduler_sample",
        "base_next_state",
        "controlled_next_state",
        "base_state_update_delta",
        "intended_state_update_delta",
        "actual_state_update_delta",
    )
    scalar_fields = (
        "probe_id",
        "step_index",
        "signed_coefficient",
        "input_binding_digest",
        "base_raw_velocity_digest",
        "controlled_raw_velocity_digest",
        "delta_sigma",
        "base_velocity_norm",
        "intended_delta_norm",
        "actual_delta_norm",
        "base_state_update_norm",
        "intended_state_update_norm",
        "state_update_delta_norm",
        "state_update_direction_dot",
        "direction_actual_norm",
        "direction_intended_norm",
        "norm_budget",
        "cumulative_reference_energy_before_step",
        "cumulative_control_energy_before_step",
        "remaining_step_count",
        "reference_energy_increment",
        "projected_reference_energy",
        "total_flow_energy_budget",
        "remaining_flow_energy",
        "energy_increment",
        "direction_cosine",
        "signed_state_update_exposure",
        "norm_guard_passed",
        "energy_guard_passed",
        "direction_guard_passed",
        "clean_exact_noop",
        "local_contract_only",
        "execution_evidence_allowed",
        "formal_result",
        "stage_progression_allowed",
    )
    if any(
        not np.array_equal(
            getattr(measurement, field_name),
            getattr(rebuilt, field_name),
        )
        for field_name in array_fields
    ) or any(
        getattr(measurement, field_name) != getattr(rebuilt, field_name)
        for field_name in scalar_fields
    ):
        raise ValueError(
            "CFG state-update measurement 与原始transition数组重建不一致"
        )
    return measurement
