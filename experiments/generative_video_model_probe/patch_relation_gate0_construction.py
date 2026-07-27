"""Run the first real Patch-relation Gate 0 construction.

The runner is deliberately limited to two frozen identities and eight videos.
It measures a zero-relation counterfactual and the controlled RoPE relation on
the same Wan transformer inputs, verifies the exact FP32 CFG velocity consumed
by the scheduler, constructs C0 whitening/T_rel, and evaluates identity A
apply-only.  Every result remains non-formal.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from hashlib import sha256
import gc
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence
import weakref

import numpy as np

from evaluation.protocol.patch_relation_gate0_contract import (
    DEFAULT_CONFIG_PATH,
    PatchRelationProbePlanRecord,
    build_patch_relation_gate0_plan,
    load_patch_relation_gate0_config,
)
from evaluation.protocol.record_writer import write_json, write_jsonl
from main.methods.state_space_watermark.patch_relation_carrier import (
    C0RelationConstruction,
    ConstructionWhitening,
    FEATURE_SCHEMA_ID,
    FEATURE_SHAPE,
    GateZeroRelationEvaluation,
    PublicPatchRelationDescriptor,
    SignedRelationStatistics,
    build_public_patch_relation_descriptor,
    construct_c0_relation_transfer,
    evaluate_gate0_apply_only,
    extract_saved_rgb24_patch_relation_feature,
    signed_gate_ready,
)
from main.methods.state_space_watermark.patch_relation_wan_runtime import (
    CFG_GUIDANCE_SCALE,
    FLOW_ENERGY_BUDGET_RATIO,
    LAMBDA_MAX,
    MINIMUM_DIRECTION_COSINE,
    VELOCITY_NORM_RATIO_BUDGET,
    CfgRopeApplicationPair,
    CfgStateUpdateMeasurement,
    SCHEDULER_VELOCITY_SHAPE,
    ScopedWanRopeOutputAdapter,
    WanRopeBranchApplicationRecord,
    measure_cfg_state_update_numpy,
    validate_cfg_state_update_measurement_numpy,
    validate_cfg_rope_application_pair,
)
from main.methods.state_space_watermark.state_trajectory_injection import (
    _budget_guard_passed,
)
from runtime.core.progress import emit_progress_event


TEST_ID = "patch_relation_gate0_construction"
PROFILE_ID = "sstw_patch_relation_gate0_construction"
PHASE = "gate0"
RECORD_VERSION = "patch_relation_gate0_construction_v1"
CLAIM_SUPPORT_STATUS = (
    "patch_relation_gate0_construction_only_not_method_evidence"
)
DECISION_FILENAME = "patch_relation_gate0_decision.json"
MANIFEST_FILENAME = "patch_relation_gate0_manifest.json"
GENERATION_PLAN_PATH = "records/patch_relation_generation_plan.jsonl"
GENERATION_RECORDS_PATH = "records/patch_relation_generation_records.jsonl"
STEP_RECORDS_PATH = "records/patch_relation_step_records.jsonl"
FEATURE_RECORDS_PATH = "records/patch_relation_feature_records.jsonl"
C0_ARTIFACT_PATH = "artifacts/patch_relation_c0_construction.npz"


class _ValidatedTransitionSeal:
    """In-process consistency capability, not cryptographic authentication."""

    __slots__ = ("__weakref__",)


_ISSUED_TRANSITION_SEALS: weakref.WeakKeyDictionary[
    _ValidatedTransitionSeal,
    tuple[Any, ...],
] = (
    weakref.WeakKeyDictionary()
)


@dataclass(frozen=True)
class GovernedPatchRelationStep:
    """Compact result after the real tensors and scheduler input were checked."""

    probe_id: str
    step_index: int
    signed_coefficient: int
    input_binding_digest: str
    conditional_encoder_digest: str
    unconditional_encoder_digest: str
    base_pair: CfgRopeApplicationPair
    controlled_pair: CfgRopeApplicationPair
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
    actual_delta_digest: str
    scheduler_consumed_velocity_digest: str
    scheduler_sample_digest: str
    scheduler_base_next_state_digest: str
    scheduler_controlled_next_state_digest: str
    actual_state_update_digest: str
    _validated_transition_seal: _ValidatedTransitionSeal | None = field(
        default=None,
        repr=False,
        compare=False,
    )


@dataclass(frozen=True)
class PatchRelationProbeMeasurement:
    plan_index: int
    identity_role: str
    probe_id: str
    signed_coefficient: int
    generator_state_digest_random: str
    initial_hidden_state_digest_random: str
    video_path: str
    video_sha256: str
    saved_rgb24_digest: str
    output_binding_digest: str
    feature: np.ndarray
    steps: tuple[GovernedPatchRelationStep, ...]
    actual_signed_exposure: float


@dataclass(frozen=True)
class PatchRelationRuntimeBatch:
    measurements: tuple[PatchRelationProbeMeasurement, ...]
    generation_records: tuple[Mapping[str, Any], ...]
    step_records: tuple[Mapping[str, Any], ...]
    feature_records: tuple[Mapping[str, Any], ...]


def _transition_seal_payload(
    step: GovernedPatchRelationStep,
) -> tuple[Any, ...]:
    return tuple(
        getattr(step, descriptor.name)
        for descriptor in fields(GovernedPatchRelationStep)
        if descriptor.name != "_validated_transition_seal"
    )


def _require_validated_transition_seal(
    step: GovernedPatchRelationStep,
) -> None:
    seal = step._validated_transition_seal
    try:
        issued_payload = (
            None if seal is None else _ISSUED_TRANSITION_SEALS[seal]
        )
    except KeyError:
        issued_payload = None
    if issued_payload is None or issued_payload != _transition_seal_payload(step):
        raise ValueError(
            "governed step 缺少同进程原始transition数组验证seal"
        )


def _stable_digest(value: Any) -> str:
    return sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_saved_rgb24_file(path: Path) -> np.ndarray:
    from experiments.generative_video_model_probe.frozen_feedback_signed_response_diagnostic import (
        _read_saved_rgb24_array,
    )

    try:
        return _read_saved_rgb24_array(path)
    except Exception as error:
        raise ValueError(
            f"saved RGB24 MP4 回读失败: {type(error).__name__}"
        ) from error


def _rgb24_digest(values: np.ndarray) -> str:
    array = np.asarray(values)
    if (
        array.shape != (33, 320, 512, 3)
        or array.dtype != np.uint8
        or not array.flags.c_contiguous
    ):
        raise ValueError("saved RGB24 digest 输入不匹配")
    return sha256(array.tobytes(order="C")).hexdigest()


def _output_binding_digest(
    plan: PatchRelationProbePlanRecord,
    *,
    video_sha256: str,
    saved_rgb24_digest: str,
    feature_digest: str,
) -> str:
    return _stable_digest(
        {
            "plan_index": plan.plan_index,
            "probe_id": plan.probe_id,
            "identity_role": plan.identity_role,
            "video_sha256": video_sha256,
            "saved_rgb24_digest": saved_rgb24_digest,
            "feature_schema_id": FEATURE_SCHEMA_ID,
            "feature_digest": feature_digest,
        }
    )


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not_installed"


def _repository_commit() -> str:
    root = Path(__file__).resolve().parents[2]
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
    ).strip()


def _identity(
    config: Mapping[str, Any],
    identity_role: str,
) -> Mapping[str, Any]:
    identities = config["protocol_contract"]["execution_identity_contract"]
    if identity_role == "construction_c0":
        return identities["construction_identity_c0"]
    if identity_role == "gate0_identity_a":
        return identities["gate0_identity_a"]
    raise ValueError(f"未知 Patch-relation identity role: {identity_role}")


def _tensor_signature(value: Any) -> dict[str, Any]:
    """Hash logical C-order values without changing dtype."""

    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        if not np.all(np.isfinite(array)):
            raise ValueError("tensor signature 输入包含非有限值")
        logical = np.ascontiguousarray(array)
        return {
            "shape": list(logical.shape),
            "dtype": logical.dtype.str,
            "values_digest": sha256(logical.tobytes(order="C")).hexdigest(),
        }
    detach = getattr(value, "detach", None)
    if not callable(detach):
        raise TypeError("tensor signature 输入必须为 NumPy 或 torch tensor")
    source = detach()
    if not bool(source.isfinite().all().item()):
        raise ValueError("tensor signature 输入包含非有限值")
    flattened = source.reshape(-1)
    cpu = source.new_empty(
        (int(source.numel()),),
        device="cpu",
        dtype=source.dtype,
    )
    cpu.copy_(flattened)
    raw = cpu.view(dtype=__import__("torch").uint8).numpy().tobytes()
    return {
        "shape": [int(value) for value in source.shape],
        "dtype": str(source.dtype),
        "values_digest": sha256(raw).hexdigest(),
    }


def _tensor_values_digest(value: Any) -> str:
    return str(_tensor_signature(value)["values_digest"])


def _timestep_signature(value: Any) -> dict[str, Any]:
    if isinstance(value, np.ndarray):
        array = np.asarray(value, dtype=np.float32).reshape(-1)
    else:
        array = value.detach().float().cpu().numpy().reshape(-1)
    if (
        array.size < 1
        or not np.all(np.isfinite(array))
        or not np.all(array == array[0])
    ):
        raise ValueError("Wan timestep 必须为有限重复scalar")
    return {"float32_value": float(np.float32(array[0]))}


def _to_float32_numpy(value: Any, *, label: str) -> np.ndarray:
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
    else:
        array = value.detach().float().cpu().numpy()
    result = np.ascontiguousarray(array, dtype="<f4")
    expected = SCHEDULER_VELOCITY_SHAPE
    if result.shape != expected or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} 必须为有限 {expected} float32")
    return result


def _runtime_float32_velocity(value: Any, *, label: str) -> Any:
    """Return the exact FP32 tensor/array that the pipeline will consume."""

    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        if (
            array.shape != SCHEDULER_VELOCITY_SHAPE
            or array.dtype != np.dtype("<f4")
            or not array.flags.c_contiguous
            or not np.all(np.isfinite(array))
        ):
            raise ValueError(f"{label} 必须为有限C-contiguous float32")
        return array
    result = value.float().contiguous()
    if tuple(result.shape) != SCHEDULER_VELOCITY_SHAPE:
        raise ValueError(f"{label} shape 不匹配")
    if str(result.dtype) != "torch.float32":
        raise ValueError(f"{label} 必须转换为torch.float32")
    if not bool(result.isfinite().all().item()):
        raise ValueError(f"{label} 包含非有限值")
    return result


def _replace_tuple_velocity(result: Any, velocity: Any, *, label: str) -> tuple[Any, ...]:
    if not isinstance(result, tuple) or len(result) < 1:
        raise TypeError(f"{label} transformer 必须返回 tuple")
    return (velocity, *result[1:])


def _runtime_cfg_combine(conditional: Any, unconditional: Any) -> Any:
    if isinstance(conditional, np.ndarray):
        if not isinstance(unconditional, np.ndarray):
            raise TypeError("CFG runtime backend 必须一致")
        return np.ascontiguousarray(
            np.add(
                unconditional,
                np.multiply(
                    np.subtract(
                        conditional,
                        unconditional,
                        dtype=np.float32,
                    ),
                    np.float32(CFG_GUIDANCE_SCALE),
                    dtype=np.float32,
                ),
                dtype=np.float32,
            ),
            dtype="<f4",
        )
    if isinstance(unconditional, np.ndarray):
        raise TypeError("CFG runtime backend 必须一致")
    return (
        unconditional
        + float(CFG_GUIDANCE_SCALE) * (conditional - unconditional)
    ).float().contiguous()


def _runtime_arrays_equal(left: Any, right: Any) -> bool:
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(right, np.ndarray):
        return False
    torch = __import__("torch")
    return bool(torch.equal(left, right))


def _runtime_euler_next_state(
    sample: Any,
    velocity: Any,
    *,
    delta_sigma: float,
) -> Any:
    if isinstance(sample, np.ndarray):
        if not isinstance(velocity, np.ndarray):
            raise TypeError("Euler runtime backend 必须一致")
        return np.ascontiguousarray(
            np.add(
                sample.astype(np.float32, copy=False),
                np.multiply(
                    velocity,
                    np.float32(delta_sigma),
                    dtype=np.float32,
                ),
                dtype=np.float32,
            ),
            dtype="<f4",
        )
    if isinstance(velocity, np.ndarray):
        raise TypeError("Euler runtime backend 必须一致")
    return (
        sample.float()
        + float(delta_sigma) * velocity.float()
    ).to(dtype=velocity.dtype).contiguous()


def _extract_scheduler_next_state(result: Any) -> Any:
    if not isinstance(result, tuple) or len(result) < 1:
        raise TypeError("Flow scheduler 必须以tuple返回prev_sample")
    return result[0]


def _extract_transformer_velocity(result: Any, *, label: str) -> Any:
    if not isinstance(result, tuple) or len(result) < 1:
        raise TypeError(f"{label} transformer 必须返回 tuple")
    return result[0]


def _scalar_float32(value: Any, *, label: str) -> float:
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError(f"{label} 必须为 scalar")
        scalar = float(value.reshape(-1)[0])
    elif hasattr(value, "detach"):
        detached = value.detach().float().cpu()
        if int(detached.numel()) != 1:
            raise ValueError(f"{label} 必须为 scalar")
        scalar = float(detached.item())
    else:
        scalar = float(value)
    scalar = float(np.float32(scalar))
    if not math.isfinite(scalar):
        raise ValueError(f"{label} 必须有限")
    return scalar


def _frozen_schedule(
    config: Mapping[str, Any],
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    runtime = config["protocol_contract"]["gate0_runtime_execution_contract"]
    sigmas = tuple(
        float(np.float32(value)) for value in runtime["sigma_grid_decimal"]
    )
    deltas = tuple(
        float(np.float32(value))
        for value in runtime["delta_sigma_by_step_decimal"]
    )
    timesteps = tuple(
        float(np.float32(value))
        for value in runtime["timestep_by_step_decimal"]
    )
    if len(sigmas) != 9 or len(deltas) != 8 or len(timesteps) != 8:
        raise ValueError("Patch-relation frozen schedule 长度不匹配")
    for index, delta in enumerate(deltas):
        recomputed = float(np.float32(sigmas[index + 1] - sigmas[index]))
        if recomputed != delta:
            raise ValueError("Patch-relation frozen schedule 内部不一致")
        recomputed_timestep = float(np.float32(sigmas[index] * 1000.0))
        if recomputed_timestep != timesteps[index]:
            raise ValueError("Patch-relation frozen timestep 内部不一致")
    return sigmas, deltas, timesteps


def _restore_instance_attribute(
    target: Any,
    name: str,
    *,
    had_instance_value: bool,
    previous_value: Any,
) -> None:
    if had_instance_value:
        setattr(target, name, previous_value)
    else:
        delattr(target, name)


def _run_cleanup_operations(
    operations: Sequence[Callable[[], None]],
) -> tuple[BaseException, ...]:
    """Run every cleanup operation and preserve all failures."""

    errors: list[BaseException] = []
    for operation in operations:
        try:
            operation()
        except BaseException as error:
            errors.append(error)
    return tuple(errors)


class ScopedPatchRelationWanProbeAdapter:
    """Measure base/control forwards and bind the controlled CFG scheduler step."""

    _ACTIVE_TRANSFORMER_ATTRIBUTE = "_sstw_patch_relation_probe_scope_active"
    _ACTIVE_SCHEDULER_ATTRIBUTE = "_sstw_patch_relation_scheduler_scope_active"

    def __init__(
        self,
        transformer: Any,
        scheduler: Any,
        *,
        descriptor: PublicPatchRelationDescriptor,
        probe_id: str,
        identity_id: str,
        signed_coefficient: int,
        sigma_grid: Sequence[float],
        delta_sigma_by_step: Sequence[float],
        timestep_by_step: Sequence[float],
    ) -> None:
        if signed_coefficient not in (-1, 0, 1):
            raise ValueError("signed_coefficient 只允许 -1/0/+1")
        if (
            len(sigma_grid) != 9
            or len(delta_sigma_by_step) != 8
            or len(timestep_by_step) != 8
        ):
            raise ValueError("probe adapter schedule 长度不匹配")
        self.transformer = transformer
        self.scheduler = scheduler
        self.descriptor = descriptor
        self.probe_id = probe_id
        self.identity_id = identity_id
        self.signed_coefficient = signed_coefficient
        self.sigma_grid = tuple(float(np.float32(v)) for v in sigma_grid)
        self.delta_sigma_by_step = tuple(
            float(np.float32(v)) for v in delta_sigma_by_step
        )
        self.timestep_by_step = tuple(
            float(np.float32(v)) for v in timestep_by_step
        )
        self._transformer_calls = 0
        self._scheduler_calls = 0
        self._cumulative_reference_energy = 0.0
        self._cumulative_control_energy = 0.0
        self._branch_outputs: dict[str, Any] = {}
        self._branch_records: dict[str, WanRopeBranchApplicationRecord] = {}
        self._branch_encoder_digests: dict[str, str] = {}
        self._pending_step: dict[str, Any] | None = None
        self._steps: list[GovernedPatchRelationStep] = []
        self._initial_hidden_state_digest = ""
        self._entered = False
        self._exit_attempted = False
        self._completed = False
        self._transformer_had_forward = False
        self._transformer_previous_forward: Any = None
        self._scheduler_had_step = False
        self._scheduler_previous_step: Any = None
        self._original_forward: Any = None
        self._original_scheduler_step: Any = None

    def _common_binding_digest(
        self,
        hidden_states: Any,
        timestep: Any,
        *,
        step_index: int,
    ) -> str:
        payload = {
            "probe_id": self.probe_id,
            "identity_id": self.identity_id,
            "step_index": step_index,
            "hidden_states": _tensor_signature(hidden_states),
            "timestep": _timestep_signature(timestep),
        }
        return _stable_digest(payload)

    def _branch_encoder_digest(
        self,
        encoder_hidden_states: Any,
        *,
        branch_role: str,
        common_binding_digest: str,
    ) -> str:
        return _stable_digest(
            {
                "common_binding_digest": common_binding_digest,
                "cfg_branch_role": branch_role,
                "encoder_hidden_states": _tensor_signature(
                    encoder_hidden_states
                ),
            }
        )

    def _run_relation_forward(
        self,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
        *,
        step_index: int,
        branch_role: str,
        control_role: str,
        coefficient: int,
        input_binding_digest: str,
    ) -> tuple[Any, WanRopeBranchApplicationRecord]:
        scope = ScopedWanRopeOutputAdapter(
            self.transformer,
            descriptor=self.descriptor,
            signed_coefficient=coefficient,
            probe_id=self.probe_id,
            step_index=step_index,
            control_role=control_role,
            cfg_branch_role=branch_role,
            input_binding_digest=input_binding_digest,
        )
        with scope:
            result = self._original_forward(*args, **dict(kwargs))
        return result, scope.record()

    def _wrapped_forward(self, *args: Any, **kwargs: Any) -> Any:
        if args:
            raise RuntimeError(
                "Patch-relation runner 要求 Wan transformer 使用冻结 keyword API"
            )
        required = {"hidden_states", "timestep", "encoder_hidden_states"}
        if not required.issubset(kwargs):
            raise RuntimeError("Wan transformer forward 输入字段不完整")
        if self._transformer_calls >= 16:
            raise RuntimeError("Wan transformer external branch 调用超额")
        step_index = self._transformer_calls // 2
        branch_role = ("conditional", "unconditional")[
            self._transformer_calls % 2
        ]
        common_digest = self._common_binding_digest(
            kwargs["hidden_states"],
            kwargs["timestep"],
            step_index=step_index,
        )
        encoder_digest = self._branch_encoder_digest(
            kwargs["encoder_hidden_states"],
            branch_role=branch_role,
            common_binding_digest=common_digest,
        )
        if step_index == 0 and branch_role == "conditional":
            self._initial_hidden_state_digest = _tensor_values_digest(
                kwargs["hidden_states"]
            )
        base_result, base_record = self._run_relation_forward(
            args,
            kwargs,
            step_index=step_index,
            branch_role=branch_role,
            control_role="base",
            coefficient=0,
            input_binding_digest=common_digest,
        )
        controlled_result, controlled_record = self._run_relation_forward(
            args,
            kwargs,
            step_index=step_index,
            branch_role=branch_role,
            control_role="controlled",
            coefficient=self.signed_coefficient,
            input_binding_digest=common_digest,
        )
        base_velocity = _runtime_float32_velocity(
            _extract_transformer_velocity(
                base_result,
                label=f"base {branch_role}",
            ),
            label=f"base {branch_role} velocity",
        )
        controlled_velocity = _runtime_float32_velocity(
            _extract_transformer_velocity(
                controlled_result,
                label=f"controlled {branch_role}",
            ),
            label=f"controlled {branch_role} velocity",
        )
        controlled_result = _replace_tuple_velocity(
            controlled_result,
            controlled_velocity,
            label=f"controlled {branch_role}",
        )
        self._branch_outputs[f"base_{branch_role}"] = base_velocity
        self._branch_outputs[f"controlled_{branch_role}"] = controlled_velocity
        self._branch_records[f"base_{branch_role}"] = base_record
        self._branch_records[f"controlled_{branch_role}"] = controlled_record
        self._branch_encoder_digests[branch_role] = encoder_digest
        self._transformer_calls += 1
        if branch_role == "unconditional":
            base_pair = validate_cfg_rope_application_pair(
                self._branch_records["base_conditional"],
                self._branch_records["base_unconditional"],
            )
            controlled_pair = validate_cfg_rope_application_pair(
                self._branch_records["controlled_conditional"],
                self._branch_records["controlled_unconditional"],
            )
            if base_pair.input_binding_digest != common_digest:
                raise RuntimeError("cond/uncond hidden/timestep binding 不一致")
            if (
                self._branch_encoder_digests["conditional"]
                == self._branch_encoder_digests["unconditional"]
            ):
                raise RuntimeError("cond/uncond encoder binding 不得相同")
            self._pending_step = {
                "base_pair": base_pair,
                "controlled_pair": controlled_pair,
                "base_conditional": self._branch_outputs["base_conditional"],
                "base_unconditional": self._branch_outputs[
                    "base_unconditional"
                ],
                "controlled_conditional": self._branch_outputs[
                    "controlled_conditional"
                ],
                "controlled_unconditional": self._branch_outputs[
                    "controlled_unconditional"
                ],
                "conditional_encoder_digest": self._branch_encoder_digests[
                    "conditional"
                ],
                "unconditional_encoder_digest": self._branch_encoder_digests[
                    "unconditional"
                ],
            }
            self._branch_outputs = {}
            self._branch_records = {}
            self._branch_encoder_digests = {}
        del base_result
        return controlled_result

    def _scheduler_sigmas(self, step_index: int) -> tuple[float, float]:
        sigmas = getattr(self.scheduler, "sigmas", None)
        if sigmas is None or len(sigmas) != 9:
            raise RuntimeError("Flow scheduler 必须公开精确9点 sigma grid")
        before = _scalar_float32(
            sigmas[step_index],
            label="scheduler sigma before",
        )
        after = _scalar_float32(
            sigmas[step_index + 1],
            label="scheduler sigma after",
        )
        expected = (
            self.sigma_grid[step_index],
            self.sigma_grid[step_index + 1],
        )
        if (before, after) != expected:
            raise RuntimeError("Flow scheduler sigma grid 与冻结合同不一致")
        return before, after

    def _validate_scheduler_timestep(
        self,
        timestep: Any,
        *,
        step_index: int,
    ) -> None:
        timesteps = getattr(self.scheduler, "timesteps", None)
        if timesteps is None or len(timesteps) != 8:
            raise RuntimeError("Flow scheduler 必须公开精确8项 timesteps")
        actual = _scalar_float32(timestep, label="scheduler timestep")
        scheduler_value = _scalar_float32(
            timesteps[step_index],
            label="scheduler frozen timestep",
        )
        expected = self.timestep_by_step[step_index]
        if actual != scheduler_value or actual != expected:
            raise RuntimeError("scheduler timestep 与冻结step/sigma行不一致")

    def _validate_scheduler_internal_index(
        self,
        *,
        step_index: int,
        after_step: bool,
    ) -> None:
        observed = getattr(self.scheduler, "step_index", None)
        expected = step_index + 1 if after_step else (
            None if step_index == 0 else step_index
        )
        if observed != expected:
            phase = "after" if after_step else "before"
            raise RuntimeError(
                f"scheduler internal step index {phase} 漂移: "
                f"expected={expected!r},observed={observed!r}"
            )

    def _wrapped_scheduler_step(
        self,
        model_output: Any,
        timestep: Any,
        sample: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        step_index = self._scheduler_calls
        if step_index >= 8 or self._pending_step is None:
            raise RuntimeError("scheduler step 缺少完整 cond/uncond measurement")
        self._validate_scheduler_timestep(timestep, step_index=step_index)
        self._validate_scheduler_internal_index(
            step_index=step_index,
            after_step=False,
        )
        sigma_before, sigma_after = self._scheduler_sigmas(step_index)
        interval = float(np.float32(sigma_after - sigma_before))
        if interval != self.delta_sigma_by_step[step_index]:
            raise RuntimeError("scheduler delta_sigma 与冻结合同不一致")
        pending = self._pending_step
        base_cfg_runtime = _runtime_cfg_combine(
            pending["base_conditional"],
            pending["base_unconditional"],
        )
        controlled_cfg_runtime = _runtime_cfg_combine(
            pending["controlled_conditional"],
            pending["controlled_unconditional"],
        )
        consumed_runtime = _runtime_float32_velocity(
            model_output,
            label="scheduler consumed controlled CFG velocity",
        )
        if not _runtime_arrays_equal(consumed_runtime, controlled_cfg_runtime):
            raise RuntimeError(
                "pipeline实际CFG与冻结FP32 branch combine不一致"
            )
        result = self._original_scheduler_step(
            consumed_runtime,
            timestep,
            sample,
            *args,
            **kwargs,
        )
        self._validate_scheduler_internal_index(
            step_index=step_index,
            after_step=True,
        )
        controlled_next_runtime = _extract_scheduler_next_state(result)
        base_next_runtime = _runtime_euler_next_state(
            sample,
            base_cfg_runtime,
            delta_sigma=interval,
        )
        measurement = measure_cfg_state_update_numpy(
            base_pair=pending["base_pair"],
            controlled_pair=pending["controlled_pair"],
            base_conditional_velocity=_to_float32_numpy(
                pending["base_conditional"],
                label="base conditional velocity",
            ),
            base_unconditional_velocity=_to_float32_numpy(
                pending["base_unconditional"],
                label="base unconditional velocity",
            ),
            controlled_conditional_velocity=_to_float32_numpy(
                pending["controlled_conditional"],
                label="controlled conditional velocity",
            ),
            controlled_unconditional_velocity=_to_float32_numpy(
                pending["controlled_unconditional"],
                label="controlled unconditional velocity",
            ),
            scheduler_consumed_velocity=_to_float32_numpy(
                consumed_runtime,
                label="scheduler consumed controlled CFG velocity",
            ),
            scheduler_sample=_to_float32_numpy(
                sample,
                label="scheduler sample",
            ),
            scheduler_base_next_state=_to_float32_numpy(
                base_next_runtime,
                label="scheduler base next state",
            ),
            scheduler_controlled_next_state=_to_float32_numpy(
                controlled_next_runtime,
                label="scheduler controlled next state",
            ),
            delta_sigma=interval,
            cumulative_reference_energy_before_step=(
                self._cumulative_reference_energy
            ),
            cumulative_control_energy_before_step=(
                self._cumulative_control_energy
            ),
            remaining_step_count=8 - step_index,
        )
        hidden_signature = pending["base_pair"].input_binding_digest
        sample_cast = sample
        hidden_states_dtype = getattr(
            getattr(self.transformer, "dtype", None),
            "__str__",
            lambda: "",
        )()
        if not isinstance(sample, np.ndarray) and "bfloat16" in hidden_states_dtype:
            sample_cast = sample.to(dtype=self.transformer.dtype)
        if _stable_digest(
            {
                "probe_id": self.probe_id,
                "identity_id": self.identity_id,
                "step_index": step_index,
                "hidden_states": _tensor_signature(sample_cast),
                "timestep": _timestep_signature(timestep),
            }
        ) != hidden_signature:
            raise RuntimeError("scheduler sample/timestep 与 transformer input 不一致")
        summary = _governed_step_from_measurement(
            measurement,
            base_pair=pending["base_pair"],
            controlled_pair=pending["controlled_pair"],
            conditional_encoder_digest=pending[
                "conditional_encoder_digest"
            ],
            unconditional_encoder_digest=pending[
                "unconditional_encoder_digest"
            ],
        )
        self._cumulative_reference_energy += (
            measurement.reference_energy_increment
        )
        self._cumulative_control_energy += measurement.energy_increment
        self._steps.append(summary)
        self._pending_step = None
        self._scheduler_calls += 1
        del (
            measurement,
            base_cfg_runtime,
            controlled_cfg_runtime,
            consumed_runtime,
            base_next_runtime,
            controlled_next_runtime,
        )
        return result

    def __enter__(self) -> "ScopedPatchRelationWanProbeAdapter":
        if self._entered:
            raise RuntimeError("Patch-relation probe scope 不得重复进入")
        if bool(getattr(self.transformer, "is_cache_enabled", False)):
            raise RuntimeError(
                "Patch-relation base/controlled counterfactual 要求 transformer cache disabled"
            )
        if getattr(self.transformer, "_cache_config", None) is not None:
            raise RuntimeError("Patch-relation transformer cache config 必须为空")
        if getattr(
            self.transformer,
            self._ACTIVE_TRANSFORMER_ATTRIBUTE,
            False,
        ) or getattr(
            self.scheduler,
            self._ACTIVE_SCHEDULER_ATTRIBUTE,
            False,
        ):
            raise RuntimeError("Patch-relation pipeline scope 不允许嵌套")
        transformer_dict = getattr(self.transformer, "__dict__", {})
        scheduler_dict = getattr(self.scheduler, "__dict__", {})
        self._transformer_had_forward = "forward" in transformer_dict
        self._transformer_previous_forward = transformer_dict.get("forward")
        self._scheduler_had_step = "step" in scheduler_dict
        self._scheduler_previous_step = scheduler_dict.get("step")
        self._original_forward = self.transformer.forward
        self._original_scheduler_step = self.scheduler.step
        transformer_active_set = False
        scheduler_active_set = False
        transformer_forward_set = False
        scheduler_step_set = False
        try:
            setattr(
                self.transformer,
                self._ACTIVE_TRANSFORMER_ATTRIBUTE,
                True,
            )
            transformer_active_set = True
            setattr(
                self.scheduler,
                self._ACTIVE_SCHEDULER_ATTRIBUTE,
                True,
            )
            scheduler_active_set = True
            setattr(self.transformer, "forward", self._wrapped_forward)
            transformer_forward_set = True
            setattr(self.scheduler, "step", self._wrapped_scheduler_step)
            scheduler_step_set = True
        except BaseException as error:
            cleanup_operations: list[Callable[[], None]] = []
            if scheduler_step_set:
                cleanup_operations.append(
                    lambda: _restore_instance_attribute(
                        self.scheduler,
                        "step",
                        had_instance_value=self._scheduler_had_step,
                        previous_value=self._scheduler_previous_step,
                    )
                )
            if transformer_forward_set:
                cleanup_operations.append(
                    lambda: _restore_instance_attribute(
                        self.transformer,
                        "forward",
                        had_instance_value=self._transformer_had_forward,
                        previous_value=self._transformer_previous_forward,
                    )
                )
            if scheduler_active_set:
                cleanup_operations.append(
                    lambda: delattr(
                        self.scheduler,
                        self._ACTIVE_SCHEDULER_ATTRIBUTE,
                    )
                )
            if transformer_active_set:
                cleanup_operations.append(
                    lambda: delattr(
                        self.transformer,
                        self._ACTIVE_TRANSFORMER_ATTRIBUTE,
                    )
                )
            cleanup_errors = _run_cleanup_operations(cleanup_operations)
            if cleanup_errors and hasattr(error, "add_note"):
                error.add_note(
                    "Patch-relation scope enter rollback also failed: "
                    + ",".join(type(item).__name__ for item in cleanup_errors)
                )
            raise
        self._entered = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if self._exit_attempted:
            self._completed = False
            raise RuntimeError("Patch-relation probe scope 不得重复退出")
        self._exit_attempted = True
        self._completed = False
        cleanup_errors = _run_cleanup_operations(
            (
                lambda: _restore_instance_attribute(
                    self.transformer,
                    "forward",
                    had_instance_value=self._transformer_had_forward,
                    previous_value=self._transformer_previous_forward,
                ),
                lambda: _restore_instance_attribute(
                    self.scheduler,
                    "step",
                    had_instance_value=self._scheduler_had_step,
                    previous_value=self._scheduler_previous_step,
                ),
                lambda: delattr(
                    self.transformer,
                    self._ACTIVE_TRANSFORMER_ATTRIBUTE,
                ),
                lambda: delattr(
                    self.scheduler,
                    self._ACTIVE_SCHEDULER_ATTRIBUTE,
                ),
            )
        )
        if exc is not None:
            if cleanup_errors and hasattr(exc, "add_note"):
                exc.add_note(
                    "Patch-relation pipeline scope cleanup also failed: "
                    + ",".join(type(item).__name__ for item in cleanup_errors)
                )
            return False
        if cleanup_errors:
            primary = cleanup_errors[0]
            if len(cleanup_errors) > 1 and hasattr(primary, "add_note"):
                primary.add_note(
                    "Additional Patch-relation cleanup failures: "
                    + ",".join(
                        type(item).__name__ for item in cleanup_errors[1:]
                    )
                )
            raise primary
        if (
            self._transformer_calls != 16
            or self._scheduler_calls != 8
            or self._pending_step is not None
            or len(self._steps) != 8
        ):
            raise RuntimeError("Patch-relation probe runtime coverage 不完整")
        self._completed = True
        return False

    def records(self) -> tuple[GovernedPatchRelationStep, ...]:
        if not self._exit_attempted or not self._completed:
            raise RuntimeError("Patch-relation probe records 只允许完整退出后读取")
        return tuple(self._steps)

    @property
    def initial_hidden_state_digest_random(self) -> str:
        if not self._completed or len(self._initial_hidden_state_digest) != 64:
            raise RuntimeError("initial hidden state digest 尚未完成")
        return self._initial_hidden_state_digest


def _governed_step_from_measurement(
    measurement: CfgStateUpdateMeasurement,
    *,
    base_pair: CfgRopeApplicationPair,
    controlled_pair: CfgRopeApplicationPair,
    conditional_encoder_digest: str,
    unconditional_encoder_digest: str,
) -> GovernedPatchRelationStep:
    validate_cfg_state_update_measurement_numpy(
        measurement,
        base_pair=base_pair,
        controlled_pair=controlled_pair,
    )
    if (
        measurement.local_contract_only is not True
        or measurement.execution_evidence_allowed is not False
        or measurement.formal_result
        or measurement.stage_progression_allowed
    ):
        raise ValueError("local measurement 证据边界发生漂移")
    if measurement.signed_coefficient == 0:
        if not measurement.clean_exact_noop:
            raise ValueError("clean measurement 必须 exact no-op")
    elif (
        measurement.clean_exact_noop
        or not measurement.norm_guard_passed
        or not measurement.energy_guard_passed
        or measurement.direction_guard_passed is not True
        or measurement.actual_delta_norm <= 0.0
    ):
        raise ValueError("signed measurement guard 未通过")
    governed = GovernedPatchRelationStep(
        probe_id=measurement.probe_id,
        step_index=measurement.step_index,
        signed_coefficient=measurement.signed_coefficient,
        input_binding_digest=measurement.input_binding_digest,
        conditional_encoder_digest=conditional_encoder_digest,
        unconditional_encoder_digest=unconditional_encoder_digest,
        base_pair=base_pair,
        controlled_pair=controlled_pair,
        delta_sigma=measurement.delta_sigma,
        base_velocity_norm=measurement.base_velocity_norm,
        intended_delta_norm=measurement.intended_delta_norm,
        actual_delta_norm=measurement.actual_delta_norm,
        base_state_update_norm=measurement.base_state_update_norm,
        intended_state_update_norm=measurement.intended_state_update_norm,
        state_update_delta_norm=measurement.state_update_delta_norm,
        state_update_direction_dot=measurement.state_update_direction_dot,
        direction_actual_norm=measurement.direction_actual_norm,
        direction_intended_norm=measurement.direction_intended_norm,
        norm_budget=measurement.norm_budget,
        cumulative_reference_energy_before_step=(
            measurement.cumulative_reference_energy_before_step
        ),
        cumulative_control_energy_before_step=(
            measurement.cumulative_control_energy_before_step
        ),
        remaining_step_count=measurement.remaining_step_count,
        reference_energy_increment=measurement.reference_energy_increment,
        projected_reference_energy=measurement.projected_reference_energy,
        total_flow_energy_budget=measurement.total_flow_energy_budget,
        remaining_flow_energy=measurement.remaining_flow_energy,
        energy_increment=measurement.energy_increment,
        direction_cosine=measurement.direction_cosine,
        signed_state_update_exposure=(
            measurement.signed_state_update_exposure
        ),
        norm_guard_passed=measurement.norm_guard_passed,
        energy_guard_passed=measurement.energy_guard_passed,
        direction_guard_passed=measurement.direction_guard_passed,
        clean_exact_noop=measurement.clean_exact_noop,
        actual_delta_digest=sha256(
            measurement.actual_delta_velocity.tobytes(order="C")
        ).hexdigest(),
        scheduler_consumed_velocity_digest=sha256(
            measurement.controlled_cfg_velocity.tobytes(order="C")
        ).hexdigest(),
        scheduler_sample_digest=sha256(
            measurement.scheduler_sample.tobytes(order="C")
        ).hexdigest(),
        scheduler_base_next_state_digest=sha256(
            measurement.base_next_state.tobytes(order="C")
        ).hexdigest(),
        scheduler_controlled_next_state_digest=sha256(
            measurement.controlled_next_state.tobytes(order="C")
        ).hexdigest(),
        actual_state_update_digest=sha256(
            measurement.actual_state_update_delta.tobytes(order="C")
        ).hexdigest(),
    )
    seal = _ValidatedTransitionSeal()
    governed = replace(governed, _validated_transition_seal=seal)
    _ISSUED_TRANSITION_SEALS[seal] = _transition_seal_payload(governed)
    return governed


def _step_record(
    plan: PatchRelationProbePlanRecord,
    step: GovernedPatchRelationStep,
) -> dict[str, Any]:
    record = {
        "record_version": RECORD_VERSION,
        "profile_id": PROFILE_ID,
        "patch_relation_probe_plan_index": plan.plan_index,
        "patch_relation_identity_role": plan.identity_role,
        "patch_relation_probe_id": plan.probe_id,
        "patch_relation_step_index": step.step_index,
        "patch_relation_signed_coefficient": step.signed_coefficient,
        "patch_relation_input_binding_digest": step.input_binding_digest,
        "patch_relation_conditional_encoder_digest": (
            step.conditional_encoder_digest
        ),
        "patch_relation_unconditional_encoder_digest": (
            step.unconditional_encoder_digest
        ),
        "patch_relation_base_rope_pair_digest": _stable_digest(
            asdict(step.base_pair)
        ),
        "patch_relation_controlled_rope_pair_digest": _stable_digest(
            asdict(step.controlled_pair)
        ),
        "patch_relation_delta_sigma": step.delta_sigma,
        "patch_relation_base_velocity_norm": step.base_velocity_norm,
        "patch_relation_intended_delta_norm": step.intended_delta_norm,
        "patch_relation_actual_delta_norm": step.actual_delta_norm,
        "patch_relation_base_state_update_norm": (
            step.base_state_update_norm
        ),
        "patch_relation_intended_state_update_norm": (
            step.intended_state_update_norm
        ),
        "patch_relation_state_update_delta_norm": (
            step.state_update_delta_norm
        ),
        "patch_relation_state_update_direction_dot": (
            step.state_update_direction_dot
        ),
        "patch_relation_direction_actual_norm": step.direction_actual_norm,
        "patch_relation_direction_intended_norm": (
            step.direction_intended_norm
        ),
        "patch_relation_norm_budget": step.norm_budget,
        "patch_relation_cumulative_reference_energy_before_step": (
            step.cumulative_reference_energy_before_step
        ),
        "patch_relation_cumulative_control_energy_before_step": (
            step.cumulative_control_energy_before_step
        ),
        "patch_relation_remaining_step_count": step.remaining_step_count,
        "patch_relation_reference_energy_increment": (
            step.reference_energy_increment
        ),
        "patch_relation_projected_reference_energy": (
            step.projected_reference_energy
        ),
        "patch_relation_total_flow_energy_budget": (
            step.total_flow_energy_budget
        ),
        "patch_relation_remaining_flow_energy": step.remaining_flow_energy,
        "patch_relation_energy_increment": step.energy_increment,
        "patch_relation_direction_cosine": step.direction_cosine,
        "patch_relation_signed_state_update_exposure": (
            step.signed_state_update_exposure
        ),
        "patch_relation_norm_guard_passed": step.norm_guard_passed,
        "patch_relation_energy_guard_passed": step.energy_guard_passed,
        "patch_relation_direction_guard_passed": (
            step.direction_guard_passed
        ),
        "patch_relation_clean_exact_noop": step.clean_exact_noop,
        "patch_relation_actual_delta_digest": step.actual_delta_digest,
        "patch_relation_scheduler_consumed_velocity_digest": (
            step.scheduler_consumed_velocity_digest
        ),
        "patch_relation_scheduler_sample_digest": step.scheduler_sample_digest,
        "patch_relation_scheduler_base_next_state_digest": (
            step.scheduler_base_next_state_digest
        ),
        "patch_relation_scheduler_controlled_next_state_digest": (
            step.scheduler_controlled_next_state_digest
        ),
        "patch_relation_actual_state_update_digest": (
            step.actual_state_update_digest
        ),
        "formal_result": False,
        "stage_progression_allowed": False,
        "claim_support_status": CLAIM_SUPPORT_STATUS,
    }
    record["patch_relation_step_record_id"] = _stable_digest(record)
    return record


def _feature_record(
    plan: PatchRelationProbePlanRecord,
    feature: np.ndarray,
    *,
    video_sha256: str,
    saved_rgb24_digest: str,
    output_binding_digest: str,
) -> dict[str, Any]:
    values = np.asarray(feature)
    if (
        values.shape != FEATURE_SHAPE
        or values.dtype != np.dtype("<f8")
        or not values.flags.c_contiguous
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("Patch-relation feature record 输入不完整")
    record = {
        "record_version": RECORD_VERSION,
        "profile_id": PROFILE_ID,
        "patch_relation_probe_plan_index": plan.plan_index,
        "patch_relation_identity_role": plan.identity_role,
        "patch_relation_probe_id": plan.probe_id,
        "patch_relation_feature_schema_id": FEATURE_SCHEMA_ID,
        "patch_relation_feature_shape": list(FEATURE_SHAPE),
        "patch_relation_feature_values": values.tolist(),
        "patch_relation_feature_digest": sha256(
            values.tobytes(order="C")
        ).hexdigest(),
        "patch_relation_saved_rgb24_digest": saved_rgb24_digest,
        "patch_relation_output_binding_digest": output_binding_digest,
        "video_sha256": video_sha256,
        "formal_result": False,
        "stage_progression_allowed": False,
        "claim_support_status": CLAIM_SUPPORT_STATUS,
    }
    record["patch_relation_feature_record_id"] = _stable_digest(record)
    return record


def _generation_record(
    config: Mapping[str, Any],
    plan: PatchRelationProbePlanRecord,
    measurement: PatchRelationProbeMeasurement,
    *,
    generation_runtime_sec: float,
) -> dict[str, Any]:
    identity = _identity(config, plan.identity_role)
    common = config["protocol_contract"]["execution_identity_contract"][
        "execution_common"
    ]
    record = {
        "record_version": RECORD_VERSION,
        "profile_id": PROFILE_ID,
        "patch_relation_probe_plan_index": plan.plan_index,
        "patch_relation_identity_role": plan.identity_role,
        "patch_relation_identity_id": plan.identity_id,
        "patch_relation_probe_id": plan.probe_id,
        "patch_relation_probe_role": plan.probe_role,
        "patch_relation_signed_coefficient": plan.signed_state_coefficient,
        "generation_status": "success",
        "generation_failure_reason": "none",
        "generation_model_id": common["model_id"],
        "generation_model_revision": common["model_revision"],
        "scheduler_signature": common["scheduler_signature"],
        "prompt_id": identity["prompt_id"],
        "prompt_text_sha256": identity["prompt_text_sha256"],
        "negative_prompt_text_sha256": identity[
            "negative_prompt_text_sha256"
        ],
        "seed_id": identity["seed_id"],
        "generation_seed_random": identity["seed_value"],
        "generation_generator_state_digest_random": (
            measurement.generator_state_digest_random
        ),
        "generation_initial_hidden_state_digest_random": (
            measurement.initial_hidden_state_digest_random
        ),
        "trajectory_step_count": len(measurement.steps),
        "patch_relation_actual_signed_exposure": (
            measurement.actual_signed_exposure
        ),
        "video_path": measurement.video_path,
        "video_sha256": measurement.video_sha256,
        "patch_relation_saved_rgb24_digest": (
            measurement.saved_rgb24_digest
        ),
        "patch_relation_output_binding_digest": (
            measurement.output_binding_digest
        ),
        "generation_runtime_sec": round(float(generation_runtime_sec), 3),
        "formal_result": False,
        "stage_progression_allowed": False,
        "claim_support_status": CLAIM_SUPPORT_STATUS,
    }
    record["patch_relation_generation_record_id"] = _stable_digest(record)
    return record


def _statistics_payload(statistics: SignedRelationStatistics) -> dict[str, Any]:
    ready = signed_gate_ready(statistics)
    return {
        "clean_noise_norm": statistics.clean_noise_norm,
        "odd_norm": statistics.odd_norm,
        "common_norm": statistics.common_norm,
        "antisymmetry_cosine": statistics.antisymmetry_cosine,
        "antisymmetry_residual": statistics.antisymmetry_residual,
        "common_odd_ratio": statistics.common_odd_ratio,
        "odd_clean_noise_ratio": statistics.odd_clean_noise_ratio,
        "signed_gate_ready": ready,
    }


def _write_c0_artifact(
    path: Path,
    construction: C0RelationConstruction,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    statistics = construction.statistics
    np.savez(
        path,
        center=construction.whitening.center,
        scale=construction.whitening.scale,
        transfer=construction.transfer_values,
        clean_intercept=statistics.clean_intercept,
        observed_odd=statistics.observed_odd,
        observed_common=statistics.observed_common,
        statistic_scalars=np.asarray(
            [
                statistics.clean_noise_norm,
                statistics.odd_norm,
                statistics.common_norm,
                statistics.antisymmetry_cosine,
                statistics.antisymmetry_residual,
                statistics.common_odd_ratio,
                statistics.odd_clean_noise_ratio,
            ],
            dtype="<f8",
        ),
        exposures=np.asarray(
            [
                construction.positive_exposure,
                construction.negative_exposure,
            ],
            dtype="<f8",
        ),
        descriptor_digest=np.asarray(construction.descriptor_digest),
        feature_schema_id=np.asarray(construction.feature_schema_id),
        whitening_digest=np.asarray(
            construction.whitening.whitening_digest
        ),
        construction_ready=np.asarray(
            int(construction.construction_ready),
            dtype="<i8",
        ),
        construction_digest=np.asarray(construction.construction_digest),
    )


def _validate_c0_artifact(
    path: Path,
    expected: C0RelationConstruction,
) -> None:
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "center",
            "scale",
            "transfer",
            "clean_intercept",
            "observed_odd",
            "observed_common",
            "statistic_scalars",
            "exposures",
            "descriptor_digest",
            "feature_schema_id",
            "whitening_digest",
            "construction_ready",
            "construction_digest",
        }
        if set(payload.files) != required:
            raise ValueError("C0 artifact members 不匹配")
        statistics = expected.statistics
        expected_scalars = np.asarray(
            [
                statistics.clean_noise_norm,
                statistics.odd_norm,
                statistics.common_norm,
                statistics.antisymmetry_cosine,
                statistics.antisymmetry_residual,
                statistics.common_odd_ratio,
                statistics.odd_clean_noise_ratio,
            ],
            dtype="<f8",
        )
        expected_exposures = np.asarray(
            [expected.positive_exposure, expected.negative_exposure],
            dtype="<f8",
        )
        array_pairs = (
            (payload["center"], expected.whitening.center),
            (payload["scale"], expected.whitening.scale),
            (payload["transfer"], expected.transfer_values),
            (payload["clean_intercept"], statistics.clean_intercept),
            (payload["observed_odd"], statistics.observed_odd),
            (payload["observed_common"], statistics.observed_common),
            (payload["statistic_scalars"], expected_scalars),
            (payload["exposures"], expected_exposures),
        )
        scalar_pairs = (
            (
                str(payload["descriptor_digest"].item()),
                expected.descriptor_digest,
            ),
            (
                str(payload["feature_schema_id"].item()),
                expected.feature_schema_id,
            ),
            (
                str(payload["whitening_digest"].item()),
                expected.whitening.whitening_digest,
            ),
            (
                str(payload["construction_digest"].item()),
                expected.construction_digest,
            ),
            (
                bool(int(payload["construction_ready"].item())),
                expected.construction_ready,
            ),
        )
        if any(
            not np.array_equal(observed, expected_value)
            for observed, expected_value in array_pairs
        ) or any(observed != expected_value for observed, expected_value in scalar_pairs):
            raise ValueError("C0 artifact readback 与 construction 不一致")


def _validate_runtime_batch(
    config: Mapping[str, Any],
    plan: Sequence[PatchRelationProbePlanRecord],
    batch: PatchRelationRuntimeBatch,
    *,
    output_root: Path,
) -> PatchRelationRuntimeBatch:
    if (
        len(batch.measurements) != 8
        or len(batch.generation_records) != 8
        or len(batch.step_records) != 64
        or len(batch.feature_records) != 8
    ):
        raise ValueError("Patch-relation runtime coverage 必须为8/64/8")
    if tuple(item.probe_id for item in batch.measurements) != tuple(
        item.probe_id for item in plan
    ):
        raise ValueError("Patch-relation probe identity/order 不匹配")
    generator_by_identity: dict[str, set[str]] = {}
    initial_by_identity: dict[str, set[str]] = {}
    _sigma_grid, frozen_deltas, _timesteps = _frozen_schedule(config)
    trusted_output = output_root.resolve()
    for expected, measurement in zip(plan, batch.measurements, strict=True):
        if (
            measurement.plan_index != expected.plan_index
            or measurement.identity_role != expected.identity_role
            or measurement.signed_coefficient
            != expected.signed_state_coefficient
            or len(measurement.steps) != 8
        ):
            raise ValueError(f"{expected.probe_id} measurement identity 不完整")
        cumulative_reference = 0.0
        cumulative_control = 0.0
        for step_index, step in enumerate(measurement.steps):
            _require_validated_transition_seal(step)
            rebuilt_base = validate_cfg_rope_application_pair(
                step.base_pair.conditional,
                step.base_pair.unconditional,
            )
            rebuilt_controlled = validate_cfg_rope_application_pair(
                step.controlled_pair.conditional,
                step.controlled_pair.unconditional,
            )
            expected_norm_budget = (
                step.base_velocity_norm
                * VELOCITY_NORM_RATIO_BUDGET
                * LAMBDA_MAX
            )
            expected_reference_increment = step.base_state_update_norm**2
            expected_projected_reference = (
                cumulative_reference
                + expected_reference_increment * (8 - step_index)
            )
            expected_total_budget = (
                FLOW_ENERGY_BUDGET_RATIO * expected_projected_reference
            )
            expected_remaining = max(
                0.0,
                expected_total_budget - cumulative_control,
            )
            expected_energy_increment = step.state_update_delta_norm**2
            expected_norm_guard = _budget_guard_passed(
                step.actual_delta_norm,
                expected_norm_budget,
            )
            expected_energy_guard = _budget_guard_passed(
                expected_energy_increment,
                expected_remaining,
            )
            expected_exposure = (
                float(expected.signed_state_coefficient)
                * step.state_update_delta_norm
            )
            if expected.signed_state_coefficient == 0:
                expected_direction_cosine = None
                expected_direction_guard = None
            else:
                denominator = (
                    step.direction_actual_norm
                    * step.direction_intended_norm
                )
                if denominator <= 0.0 or not math.isfinite(denominator):
                    raise ValueError(
                        f"{expected.probe_id} direction denominator 退化"
                    )
                raw_cosine = step.state_update_direction_dot / denominator
                if (
                    not math.isfinite(raw_cosine)
                    or raw_cosine < -1.0 - 1e-12
                    or raw_cosine > 1.0 + 1e-12
                ):
                    raise ValueError(
                        f"{expected.probe_id} direction cosine 越界"
                    )
                expected_direction_cosine = min(
                    1.0,
                    max(-1.0, raw_cosine),
                )
                expected_direction_guard = (
                    expected_direction_cosine + 1e-12
                    >= MINIMUM_DIRECTION_COSINE
                )
            scalar_values = (
                step.base_velocity_norm,
                step.intended_delta_norm,
                step.actual_delta_norm,
                step.base_state_update_norm,
                step.intended_state_update_norm,
                step.state_update_delta_norm,
                step.state_update_direction_dot,
                step.direction_actual_norm,
                step.direction_intended_norm,
                step.norm_budget,
                step.cumulative_reference_energy_before_step,
                step.cumulative_control_energy_before_step,
                step.reference_energy_increment,
                step.projected_reference_energy,
                step.total_flow_energy_budget,
                step.remaining_flow_energy,
                step.energy_increment,
                step.signed_state_update_exposure,
            )
            if (
                rebuilt_base != step.base_pair
                or rebuilt_controlled != step.controlled_pair
                or step.probe_id != expected.probe_id
                or step.step_index != step_index
                or step.signed_coefficient
                != expected.signed_state_coefficient
                or step.base_pair.probe_id != expected.probe_id
                or step.controlled_pair.probe_id != expected.probe_id
                or step.base_pair.step_index != step_index
                or step.controlled_pair.step_index != step_index
                or step.base_pair.signed_coefficient != 0
                or step.controlled_pair.signed_coefficient
                != expected.signed_state_coefficient
                or step.base_pair.input_binding_digest
                != step.controlled_pair.input_binding_digest
                or step.input_binding_digest
                != step.base_pair.input_binding_digest
                or step.delta_sigma != frozen_deltas[step_index]
                or step.remaining_step_count != 8 - step_index
                or step.cumulative_reference_energy_before_step
                != cumulative_reference
                or step.cumulative_control_energy_before_step
                != cumulative_control
                or not all(math.isfinite(value) for value in scalar_values)
                or step.norm_budget != expected_norm_budget
                or step.reference_energy_increment
                != expected_reference_increment
                or step.projected_reference_energy
                != expected_projected_reference
                or step.total_flow_energy_budget != expected_total_budget
                or step.remaining_flow_energy != expected_remaining
                or step.energy_increment != expected_energy_increment
                or step.norm_guard_passed is not expected_norm_guard
                or step.energy_guard_passed is not expected_energy_guard
                or step.direction_cosine != expected_direction_cosine
                or step.direction_guard_passed
                is not expected_direction_guard
                or step.signed_state_update_exposure != expected_exposure
                or len(step.conditional_encoder_digest) != 64
                or len(step.unconditional_encoder_digest) != 64
                or step.conditional_encoder_digest
                == step.unconditional_encoder_digest
                or len(step.actual_delta_digest) != 64
                or len(step.scheduler_consumed_velocity_digest) != 64
                or len(step.scheduler_sample_digest) != 64
                or len(step.scheduler_base_next_state_digest) != 64
                or len(step.scheduler_controlled_next_state_digest) != 64
                or len(step.actual_state_update_digest) != 64
            ):
                raise ValueError(
                    f"{expected.probe_id} governed step 重建不一致"
                )
            if expected.signed_state_coefficient == 0:
                if (
                    not step.clean_exact_noop
                    or step.actual_delta_norm != 0.0
                    or step.state_update_delta_norm != 0.0
                    or step.energy_increment != 0.0
                    or step.direction_cosine is not None
                    or step.direction_guard_passed is not None
                ):
                    raise ValueError(
                        f"{expected.probe_id} clean step 非 exact no-op"
                    )
            elif (
                step.clean_exact_noop
                or step.actual_delta_norm <= 0.0
                or step.direction_cosine is None
                or not math.isfinite(step.direction_cosine)
                or not -1.0 <= step.direction_cosine <= 1.0
                or not step.norm_guard_passed
                or not step.energy_guard_passed
                or step.direction_guard_passed is not True
            ):
                raise ValueError(
                    f"{expected.probe_id} signed step guard 不完整"
                )
            cumulative_reference += step.reference_energy_increment
            cumulative_control += step.energy_increment
        recomputed_exposure = math.fsum(
            step.signed_state_update_exposure for step in measurement.steps
        )
        if measurement.actual_signed_exposure != recomputed_exposure:
            raise ValueError(
                f"{expected.probe_id} actual exposure aggregation 不一致"
            )
        path = Path(measurement.video_path)
        expected_path = (
            trusted_output
            / "videos"
            / f"{expected.plan_index:02d}_{expected.probe_id}.mp4"
        )
        resolved_path = path.resolve()
        resolved_expected = expected_path.resolve()
        if (
            path.is_symlink()
            or resolved_path != resolved_expected
            or not resolved_path.is_relative_to(trusted_output)
            or not path.is_file()
            or _sha256_file(path) != measurement.video_sha256
        ):
            raise ValueError(f"{expected.probe_id} video path/SHA 不一致")
        saved_rgb24 = _read_saved_rgb24_file(path)
        if (
            saved_rgb24.shape != (33, 320, 512, 3)
            or saved_rgb24.dtype != np.uint8
            or not saved_rgb24.flags.c_contiguous
        ):
            raise ValueError(f"{expected.probe_id} RGB24 readback 不完整")
        recomputed_rgb_digest = _rgb24_digest(saved_rgb24)
        recomputed_feature = extract_saved_rgb24_patch_relation_feature(
            saved_rgb24
        )
        recomputed_feature_digest = sha256(
            recomputed_feature.tobytes(order="C")
        ).hexdigest()
        recomputed_output_binding = _output_binding_digest(
            expected,
            video_sha256=measurement.video_sha256,
            saved_rgb24_digest=recomputed_rgb_digest,
            feature_digest=recomputed_feature_digest,
        )
        if (
            recomputed_rgb_digest != measurement.saved_rgb24_digest
            or not np.array_equal(recomputed_feature, measurement.feature)
            or recomputed_output_binding != measurement.output_binding_digest
        ):
            raise ValueError(
                f"{expected.probe_id} MP4/RGB24/feature binding 不一致"
            )
        observed_steps = batch.step_records[
            expected.plan_index * 8 : expected.plan_index * 8 + 8
        ]
        if tuple(observed_steps) != tuple(
            _step_record(expected, step) for step in measurement.steps
        ):
            raise ValueError(f"{expected.probe_id} governed step records 不一致")
        expected_feature = _feature_record(
            expected,
            recomputed_feature,
            video_sha256=measurement.video_sha256,
            saved_rgb24_digest=recomputed_rgb_digest,
            output_binding_digest=recomputed_output_binding,
        )
        if batch.feature_records[expected.plan_index] != expected_feature:
            raise ValueError(f"{expected.probe_id} feature record 不一致")
        observed_generation = batch.generation_records[expected.plan_index]
        expected_generation = _generation_record(
            config,
            expected,
            measurement,
            generation_runtime_sec=float(
                observed_generation.get("generation_runtime_sec", 0.0)
            ),
        )
        if observed_generation != expected_generation:
            raise ValueError(f"{expected.probe_id} generation record 不一致")
        generator_by_identity.setdefault(expected.identity_role, set()).add(
            measurement.generator_state_digest_random
        )
        initial_by_identity.setdefault(expected.identity_role, set()).add(
            measurement.initial_hidden_state_digest_random
        )
    if any(len(values) != 1 for values in generator_by_identity.values()):
        raise ValueError("同一 identity 四项 generator 初态必须一致")
    if any(len(values) != 1 for values in initial_by_identity.values()):
        raise ValueError("同一 identity 四项 initial latent 必须一致")
    if (
        generator_by_identity["construction_c0"]
        == generator_by_identity["gate0_identity_a"]
        or initial_by_identity["construction_c0"]
        == initial_by_identity["gate0_identity_a"]
    ):
        raise ValueError("C0/A initial noise identity 必须隔离")
    return batch


def _cuda_progress_fields(torch_module: Any) -> str:
    try:
        gib = float(1024**3)
        allocated = float(torch_module.cuda.memory_allocated()) / gib
        reserved = float(torch_module.cuda.memory_reserved()) / gib
        free_bytes, total_bytes = torch_module.cuda.mem_get_info()
        return (
            f"allocated_gib={allocated:.3f} "
            f"reserved_gib={reserved:.3f} "
            f"free_gib={float(free_bytes) / gib:.3f} "
            f"total_gib={float(total_bytes) / gib:.3f}"
        )
    except Exception as exc:
        return f"cuda_memory_error_type={type(exc).__name__}"


def _require_native_bfloat16_runtime(
    torch_module: Any,
    *,
    selected_dtype: Any,
) -> None:
    """Fail before model load unless CUDA exposes native BF16 and BF16 dtype."""

    try:
        supported = torch_module.cuda.is_bf16_supported(
            including_emulation=False
        )
    except TypeError as error:
        raise RuntimeError(
            "torch.cuda.is_bf16_supported 缺少including_emulation=False兼容签名"
        ) from error
    if supported is not True:
        raise RuntimeError("Patch-relation Gate0 要求CUDA原生bfloat16")
    try:
        capability = tuple(torch_module.cuda.get_device_capability())
    except Exception as error:
        raise RuntimeError("CUDA device capability 无法验证") from error
    if len(capability) != 2 or capability[0] < 8:
        raise RuntimeError(
            "Patch-relation Gate0 要求Ampere或更新CUDA compute capability"
        )
    if selected_dtype is not torch_module.bfloat16:
        raise RuntimeError(
            "Patch-relation Gate0 pipeline selected dtype 必须exact torch.bfloat16"
        )


def execute_real_patch_relation_gate0(
    config: Mapping[str, Any],
    plan: Sequence[PatchRelationProbePlanRecord],
    output_root: Path,
) -> PatchRelationRuntimeBatch:
    """Execute the exact eight-video Wan plan on an explicitly started GPU."""

    import torch

    from experiments.generative_video_model_probe.colab_runtime import (
        _export_video,
        _generation_model_provenance_from_pipeline,
        _load_video_generation_pipeline,
        _scheduler_signature,
        _select_dtype,
    )
    from experiments.generative_video_model_probe.frozen_feedback_signed_response_diagnostic import (
        _decode_wan_final_latent,
        _read_saved_rgb24_array,
        _tensor_digest,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("Patch-relation Gate0 需要用户显式启动 Colab CUDA")
    selected_dtype = _select_dtype(torch)
    _require_native_bfloat16_runtime(
        torch,
        selected_dtype=selected_dtype,
    )
    common = config["protocol_contract"]["execution_identity_contract"][
        "execution_common"
    ]
    if _package_version("diffusers") != common["diffusers_version"]:
        raise RuntimeError("Patch-relation Gate0 diffusers 版本漂移")
    pipe = _load_video_generation_pipeline(
        common["model_id"],
        selected_dtype,
        revision=common["model_revision"],
    )
    try:
        provenance = _generation_model_provenance_from_pipeline(
            pipe,
            expected_model_id=common["model_id"],
        )
        if (
            provenance["generation_model_commit_or_hash"]
            != common["model_revision"]
            or _scheduler_signature(pipe.scheduler)
            != common["scheduler_signature"]
        ):
            raise RuntimeError("Patch-relation model/scheduler provenance 未冻结")
        if (
            getattr(pipe, "transformer_2", None) is not None
            or getattr(pipe.config, "boundary_ratio", None) is not None
        ):
            raise RuntimeError("Patch-relation Gate0 只允许 Wan2.1 单 transformer")
        if bool(getattr(pipe.transformer, "is_cache_enabled", False)):
            raise RuntimeError("Patch-relation Gate0 禁止 transformer cache")
    except BaseException:
        maybe_free = getattr(pipe, "maybe_free_model_hooks", None)
        if callable(maybe_free):
            try:
                maybe_free()
            except BaseException:
                pass
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
        raise

    descriptor = build_public_patch_relation_descriptor()
    sigma_grid, deltas, timesteps = _frozen_schedule(config)
    measurements: list[PatchRelationProbeMeasurement] = []
    generation_records: list[Mapping[str, Any]] = []
    step_records: list[Mapping[str, Any]] = []
    feature_records: list[Mapping[str, Any]] = []
    videos_root = output_root / "videos"
    videos_root.mkdir(parents=True, exist_ok=True)
    try:
        for probe in plan:
            identity = _identity(config, probe.identity_role)
            generator = torch.Generator(device="cuda").manual_seed(
                int(identity["seed_value"])
            )
            generator_digest = _tensor_digest(generator.get_state())
            emit_progress_event(
                "patch_relation_gate0_generation",
                (
                    f"{probe.plan_index + 1}/8 start "
                    f"probe={probe.probe_id} {_cuda_progress_fields(torch)}"
                ),
            )
            started = time.time()
            adapter = ScopedPatchRelationWanProbeAdapter(
                pipe.transformer,
                pipe.scheduler,
                descriptor=descriptor,
                probe_id=probe.probe_id,
                identity_id=probe.identity_id,
                signed_coefficient=probe.signed_state_coefficient,
                sigma_grid=sigma_grid,
                delta_sigma_by_step=deltas,
                timestep_by_step=timesteps,
            )
            with torch.no_grad(), adapter:
                pipeline_result = pipe(
                    prompt=identity["prompt_text"],
                    negative_prompt=identity["negative_prompt_text"],
                    generator=generator,
                    height=common["height"],
                    width=common["width"],
                    num_frames=common["num_frames"],
                    num_inference_steps=common["num_inference_steps"],
                    guidance_scale=float(common["guidance_scale_decimal"]),
                    output_type="latent",
                )
            returned_device = pipeline_result.frames
            if isinstance(returned_device, (list, tuple)):
                returned_device = returned_device[0]
            final_latent = returned_device.detach().float().cpu()
            del pipeline_result, returned_device
            steps = adapter.records()
            decoded = _decode_wan_final_latent(pipe, final_latent)
            video_path = (
                videos_root
                / f"{probe.plan_index:02d}_{probe.probe_id}.mp4"
            )
            _export_video(decoded, video_path, fps=common["fps"])
            video_sha = _sha256_file(video_path)
            saved = _read_saved_rgb24_array(video_path)
            if (
                saved.shape != (33, 320, 512, 3)
                or saved.dtype != np.uint8
            ):
                raise RuntimeError("saved RGB24 readback shape/dtype 漂移")
            feature = extract_saved_rgb24_patch_relation_feature(saved)
            saved_digest = _rgb24_digest(saved)
            feature_digest = sha256(
                feature.tobytes(order="C")
            ).hexdigest()
            output_binding = _output_binding_digest(
                probe,
                video_sha256=video_sha,
                saved_rgb24_digest=saved_digest,
                feature_digest=feature_digest,
            )
            exposure = math.fsum(
                step.signed_state_update_exposure for step in steps
            )
            measurement = PatchRelationProbeMeasurement(
                plan_index=probe.plan_index,
                identity_role=probe.identity_role,
                probe_id=probe.probe_id,
                signed_coefficient=probe.signed_state_coefficient,
                generator_state_digest_random=generator_digest,
                initial_hidden_state_digest_random=(
                    adapter.initial_hidden_state_digest_random
                ),
                video_path=str(video_path),
                video_sha256=video_sha,
                saved_rgb24_digest=saved_digest,
                output_binding_digest=output_binding,
                feature=feature,
                steps=steps,
                actual_signed_exposure=exposure,
            )
            measurements.append(measurement)
            step_records.extend(_step_record(probe, step) for step in steps)
            feature_records.append(
                _feature_record(
                    probe,
                    feature,
                    video_sha256=video_sha,
                    saved_rgb24_digest=saved_digest,
                    output_binding_digest=output_binding,
                )
            )
            generation_records.append(
                _generation_record(
                    config,
                    probe,
                    measurement,
                    generation_runtime_sec=time.time() - started,
                )
            )
            write_jsonl(output_root / GENERATION_RECORDS_PATH, generation_records)
            write_jsonl(output_root / STEP_RECORDS_PATH, step_records)
            write_jsonl(output_root / FEATURE_RECORDS_PATH, feature_records)
            emit_progress_event(
                "patch_relation_gate0_generation",
                (
                    f"{probe.plan_index + 1}/8 finish "
                    f"probe={probe.probe_id} {_cuda_progress_fields(torch)}"
                ),
            )
            final_latent = None
            decoded = None
            saved = None
            feature = None
            gc.collect()
            torch.cuda.empty_cache()
        return PatchRelationRuntimeBatch(
            measurements=tuple(measurements),
            generation_records=tuple(generation_records),
            step_records=tuple(step_records),
            feature_records=tuple(feature_records),
        )
    finally:
        active_error = sys.exc_info()[1]
        cleanup_error: BaseException | None = None
        maybe_free = getattr(pipe, "maybe_free_model_hooks", None)
        if callable(maybe_free):
            try:
                maybe_free()
            except BaseException as error:
                cleanup_error = error
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
        if cleanup_error is not None:
            if active_error is not None and hasattr(active_error, "add_note"):
                active_error.add_note(
                    "Patch-relation pipeline hook cleanup also failed: "
                    f"{type(cleanup_error).__name__}"
                )
            elif active_error is None:
                raise cleanup_error


def run_patch_relation_gate0_construction(
    output_root: str | Path,
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    runtime_executor: Callable[
        [
            Mapping[str, Any],
            Sequence[PatchRelationProbePlanRecord],
            Path,
        ],
        PatchRelationRuntimeBatch,
    ] = execute_real_patch_relation_gate0,
) -> dict[str, Any]:
    """Execute one frozen Gate0 batch and write a non-formal decision."""

    output = Path(output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Patch-relation Gate0 output root 必须为空: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    config = load_patch_relation_gate0_config(config_path)
    plan = build_patch_relation_gate0_plan(config)
    if not all(item.execution_authorized for item in plan):
        raise RuntimeError("Patch-relation Gate0 plan 尚未获 runner 执行授权")
    write_jsonl(
        output / GENERATION_PLAN_PATH,
        [asdict(item) for item in plan],
    )
    repository_commit = _repository_commit()
    try:
        batch = _validate_runtime_batch(
            config,
            plan,
            runtime_executor(config, plan, output),
            output_root=output,
        )
        by_id = {item.probe_id: item for item in batch.measurements}
        c0 = construct_c0_relation_transfer(
            descriptor=build_public_patch_relation_descriptor(),
            clean_a=by_id["construction_c0_clean_a"].feature,
            clean_b=by_id["construction_c0_clean_b"].feature,
            positive=by_id["construction_c0_positive"].feature,
            negative=by_id["construction_c0_negative"].feature,
            positive_exposure=by_id[
                "construction_c0_positive"
            ].actual_signed_exposure,
            negative_exposure=by_id[
                "construction_c0_negative"
            ].actual_signed_exposure,
        )
        c0_path = output / C0_ARTIFACT_PATH
        _write_c0_artifact(c0_path, c0)
        _validate_c0_artifact(c0_path, c0)
        gate: GateZeroRelationEvaluation | None = None
        if c0.construction_ready:
            gate = evaluate_gate0_apply_only(
                descriptor=build_public_patch_relation_descriptor(),
                construction=c0,
                clean_a=by_id["gate0_identity_a_clean_a"].feature,
                clean_b=by_id["gate0_identity_a_clean_b"].feature,
                positive=by_id["gate0_identity_a_positive"].feature,
                negative=by_id["gate0_identity_a_negative"].feature,
                positive_exposure=by_id[
                    "gate0_identity_a_positive"
                ].actual_signed_exposure,
                negative_exposure=by_id[
                    "gate0_identity_a_negative"
                ].actual_signed_exposure,
            )
        gate0_ready = bool(
            c0.construction_ready
            and gate is not None
            and gate.gate_zero_ready
        )
        decision_name = (
            "gate0_pass_double_window_gate_a_design_allowed"
            if gate0_ready
            else "gate0_fail_stop_current_patch_relation_carrier_or_feature"
        )
        decision = {
            "record_version": RECORD_VERSION,
            "profile_id": PROFILE_ID,
            "patch_relation_gate0_decision": decision_name,
            "patch_relation_gate0_ready": gate0_ready,
            "patch_relation_c0_construction_ready": c0.construction_ready,
            "patch_relation_c0_statistics": _statistics_payload(
                c0.statistics
            ),
            "patch_relation_gate0_signed_gate_ready": (
                gate.signed_gate_ready if gate is not None else False
            ),
            "patch_relation_transfer_direction_cosine": (
                gate.transfer_direction_cosine if gate is not None else None
            ),
            "patch_relation_transfer_relative_error": (
                gate.transfer_relative_error if gate is not None else None
            ),
            "next_double_window_gate_a_design_allowed": gate0_ready,
            "next_double_window_gate_a_execution_allowed": False,
            "observer_implementation_allowed": False,
            "wrong_key_execution_allowed": False,
            "attack_execution_allowed": False,
            "fixed_fpr_execution_allowed": False,
            "baseline_execution_allowed": False,
            "paper_claim_allowed": False,
            "formal_result": False,
            "stage_progression_allowed": False,
            "generation_record_count": len(batch.generation_records),
            "trajectory_step_record_count": len(batch.step_records),
            "feature_record_count": len(batch.feature_records),
            "video_count": len(batch.measurements),
            "public_relation_descriptor_digest": (
                build_public_patch_relation_descriptor().descriptor_digest
            ),
            "patch_relation_c0_construction_digest": c0.construction_digest,
            "patch_relation_c0_artifact_sha256": _sha256_file(c0_path),
            "protocol_digest": config["protocol_digest"],
            "repository_commit": repository_commit,
            "claim_support_status": CLAIM_SUPPORT_STATUS,
        }
        write_json(output / DECISION_FILENAME, decision)
        manifest = {
            "manifest_kind": "sstw_patch_relation_gate0_manifest",
            **decision,
            "python_version": sys.version.split()[0],
            "torch_version": _package_version("torch"),
            "diffusers_version": _package_version("diffusers"),
            "method_result_is_gate0_or_paper_evidence": False,
        }
        write_json(output / MANIFEST_FILENAME, manifest)
        return decision
    except Exception as exc:
        failure = {
            "record_version": RECORD_VERSION,
            "profile_id": PROFILE_ID,
            "patch_relation_gate0_decision": (
                "runtime_or_contract_failure_recovery_only"
            ),
            "patch_relation_gate0_ready": False,
            "patch_relation_failure_reason": str(exc),
            "next_double_window_gate_a_design_allowed": False,
            "next_double_window_gate_a_execution_allowed": False,
            "observer_implementation_allowed": False,
            "wrong_key_execution_allowed": False,
            "attack_execution_allowed": False,
            "fixed_fpr_execution_allowed": False,
            "baseline_execution_allowed": False,
            "paper_claim_allowed": False,
            "formal_result": False,
            "stage_progression_allowed": False,
            "repository_commit": repository_commit,
            "claim_support_status": (
                "failure_recovery_only_not_claim_evidence"
            ),
        }
        write_json(output / DECISION_FILENAME, failure)
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-run-root", required=True)
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run_patch_relation_gate0_construction(
                arguments.output_run_root,
                config_path=arguments.config_path,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
