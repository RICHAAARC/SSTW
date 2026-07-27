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
import math
import re
from types import TracebackType
from typing import Any

import numpy as np

from main.methods.state_space_watermark.patch_relation_carrier import (
    PublicPatchRelationDescriptor,
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
EXPECTED_ROPE_CALLS_PER_BRANCH = 1
CFG_BRANCH_ORDER = ("conditional", "unconditional")
RUNTIME_ADAPTER_PROTOCOL_DIGEST = (
    "454e380c2900b9bd989ff8f95c3c0563545037650331f941b33eee650c0a0ddc"
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
    input_binding_digest: str
    descriptor_digest: str
    conditional: WanRopeBranchApplicationRecord
    unconditional: WanRopeBranchApplicationRecord
    local_contract_only: bool = True
    execution_evidence_allowed: bool = False


@dataclass(frozen=True)
class CfgStateUpdateMeasurement:
    """Strict measurement bound to the scheduler's actual FP32 next state."""

    probe_id: str
    step_index: int
    signed_coefficient: int
    input_binding_digest: str
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
    if value.dtype != torch.float64:
        raise ValueError(f"{label} dtype 必须精确为 float64")
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
    if isinstance(freqs_cos, np.ndarray) or isinstance(freqs_sin, np.ndarray):
        if not isinstance(freqs_cos, np.ndarray) or not isinstance(
            freqs_sin, np.ndarray
        ):
            raise TypeError("fake Wan RoPE tuple backend 必须一致")
        shifted = apply_wan_rotary_phase_numpy(
            freqs_cos,
            freqs_sin,
            descriptor=descriptor,
            signed_coefficient=coefficient,
        )
        if coefficient == 0:
            return freqs_cos, freqs_sin
        return shifted

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
        )
        shifted_sine[0, token_indices, 0, entry] = (
            original_sine * angle_cosine + original_cosine * angle_sine
        )
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
        return CfgStateUpdateMeasurement(
            probe_id=base_pair.probe_id,
            step_index=base_pair.step_index,
            signed_coefficient=0,
            input_binding_digest=base_pair.input_binding_digest,
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
        )

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
                "delta_sigma": interval,
                "direction_cosine": direction_cosine,
                "direction_guard_passed": direction_guard,
                "energy_guard_passed": energy_guard,
                "energy_increment": energy_increment,
                "norm_budget": norm_budget,
                "norm_guard_passed": norm_guard,
                "probe_id": base_pair.probe_id,
                "remaining_flow_energy": remaining,
                "step_index": base_pair.step_index,
            }
        )
    signed_exposure = float(coefficient) * state_update_norm
    return CfgStateUpdateMeasurement(
        probe_id=base_pair.probe_id,
        step_index=base_pair.step_index,
        signed_coefficient=coefficient,
        input_binding_digest=base_pair.input_binding_digest,
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
    )


def validate_cfg_state_update_measurement_numpy(
    measurement: CfgStateUpdateMeasurement,
    *,
    base_pair: CfgRopeApplicationPair,
    controlled_pair: CfgRopeApplicationPair,
) -> CfgStateUpdateMeasurement:
    """Rebuild every transition statistic from the retained raw FP32 arrays.

    This validator intentionally routes the already-combined base/controlled
    CFG arrays through the same measurement function as identical cond/uncond
    branches.  That preserves the exact combined values while independently
    rebuilding scheduler next states, velocity/state-update norms, direction,
    energy, exposure, and guards.  A caller cannot make coordinated scalar
    edits pass without also presenting the original transition arrays.
    """

    if not isinstance(measurement, CfgStateUpdateMeasurement):
        raise TypeError("CFG state-update measurement 类型不匹配")
    rebuilt = measure_cfg_state_update_numpy(
        base_pair=base_pair,
        controlled_pair=controlled_pair,
        base_conditional_velocity=measurement.base_cfg_velocity,
        base_unconditional_velocity=measurement.base_cfg_velocity,
        controlled_conditional_velocity=measurement.controlled_cfg_velocity,
        controlled_unconditional_velocity=measurement.controlled_cfg_velocity,
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
