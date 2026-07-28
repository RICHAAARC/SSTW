"""Run one non-Gate Patch-relation phase-response/repeatability preflight.

The preflight executes only C0 clean-A step 0.  It repeats zero-phase and one
predeclared low phase on the exact same transformer inputs, measures real
float32 CFG/state-update responses, and intentionally stops before calling the
real scheduler.  It does not decode or export video and cannot support Gate 0
or a method claim.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import secrets
import stat
import subprocess
import sys
from typing import Any, Callable, Mapping

import numpy as np

from evaluation.protocol.patch_relation_gate0_contract import (
    load_patch_relation_gate0_config,
)
from evaluation.protocol.patch_relation_phase_response_preflight_contract import (
    DEFAULT_CONFIG_PATH,
    NEXT_DETERMINISTIC_SCALE,
    PLATEAU_RATIO_THRESHOLD,
    PROFILE_ID,
    REPEATABILITY_FLOOR_RATIO_THRESHOLD,
    TUPLE_DELTA_RELATIVE_TOLERANCE,
    load_patch_relation_phase_response_preflight_config,
)
from evaluation.protocol.record_writer import write_json
from main.methods.state_space_watermark.patch_relation_carrier import (
    PHASE_BUDGET_RADIANS,
    build_public_patch_relation_descriptor,
)
from main.methods.state_space_watermark.patch_relation_wan_runtime import (
    CFG_GUIDANCE_SCALE,
    CfgRopeApplicationPair,
    FLOW_ENERGY_BUDGET_RATIO,
    LAMBDA_MAX,
    MINIMUM_DIRECTION_COSINE,
    PhaseProjectionSignEvaluation,
    ScopedWanRopeOutputAdapter,
    VELOCITY_NORM_RATIO_BUDGET,
    WanRopeBranchApplicationRecord,
    apply_wan_rotary_phase_runtime,
    evaluate_phase_projection_sign_numpy,
    require_validated_phase_projection_sign_evaluation,
    validate_cfg_rope_application_pair,
)
from main.methods.state_space_watermark.state_trajectory_injection import (
    _budget_guard_passed,
)
from runtime.core.progress import emit_progress_event

from experiments.generative_video_model_probe.patch_relation_gate0_construction import (
    _extract_transformer_velocity,
    _frozen_schedule,
    _identity,
    _package_version,
    _replace_tuple_velocity,
    _require_native_bfloat16_runtime,
    _runtime_arrays_equal,
    _runtime_cfg_combine,
    _runtime_float32_velocity,
    _scalar_float32,
    _scheduler_sample_as_transformer_input,
    _stable_digest,
    _tensor_signature,
    _tensor_values_digest,
    _timestep_signature,
    _to_float32_numpy,
)


TEST_ID = "patch_relation_phase_response_preflight"
PHASE = "phase_response_preflight"
RECORD_VERSION = "patch_relation_phase_response_preflight_v1"
DECISION_FILENAME = "patch_relation_phase_response_preflight_decision.json"
MANIFEST_FILENAME = "patch_relation_phase_response_preflight_manifest.json"
CLAIM_SUPPORT_STATUS = (
    "single_step_runtime_preflight_only_not_gate_or_method_evidence"
)
HISTORICAL_LAST_WORST_ACTUAL_DELTA_NORM = 20.022518157958984
SUCCESS_ARTIFACT_FILENAMES = (
    "phase_response_preflight_record.json",
    DECISION_FILENAME,
    MANIFEST_FILENAME,
)


class _PreflightComplete(RuntimeError):
    pass


def _attempt_runtime_cleanup(
    cleanup_errors: list[tuple[str, BaseException]],
    label: str,
    callback: Callable[[], object],
) -> None:
    try:
        callback()
    except BaseException as error:
        cleanup_errors.append((label, error))


def _finish_runtime_cleanup(
    active_error: BaseException | None,
    cleanup_errors: list[tuple[str, BaseException]],
) -> None:
    if not cleanup_errors:
        return
    if active_error is not None:
        if hasattr(active_error, "add_note"):
            for label, cleanup_error in cleanup_errors:
                active_error.add_note(
                    "phase-response pipeline cleanup also failed: "
                    f"{label}={type(cleanup_error).__name__}"
                )
        return
    raise cleanup_errors[0][1]


@dataclass(frozen=True)
class PhaseResponseForward:
    signed_coefficient: int
    repeat_index: int
    conditional_velocity: np.ndarray
    unconditional_velocity: np.ndarray
    cfg_velocity: np.ndarray
    pair: CfgRopeApplicationPair
    conditional_rope_tuple_delta_norm: float
    unconditional_rope_tuple_delta_norm: float
    conditional_expected_rope_tuple_delta_norm: float
    unconditional_expected_rope_tuple_delta_norm: float
    conditional_rope_tuple_exact_identity: bool
    unconditional_rope_tuple_exact_identity: bool
    conditional_rope_delta_matches_expected: bool
    unconditional_rope_delta_matches_expected: bool
    conditional_rope_actual_delta_digest: str
    unconditional_rope_actual_delta_digest: str
    conditional_rope_expected_delta_digest: str
    unconditional_rope_expected_delta_digest: str


@dataclass(frozen=True)
class RopeTupleDeltaObservation:
    actual_delta_norm: float
    tuple_exact_identity: bool
    expected_delta_norm: float
    actual_matches_expected: bool
    actual_delta_digest: str
    expected_delta_digest: str


@dataclass(frozen=True)
class PhaseResponsePreflightBatch:
    record: Mapping[str, Any]
    candidate_evaluations: tuple[PhaseProjectionSignEvaluation, ...]
    transformer_forward_count: int
    scheduler_step_call_count: int
    initial_hidden_state_digest_random: str
    generator_state_digest_random: str


def _repository_commit() -> str:
    root = Path(__file__).resolve().parents[2]
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
    ).strip()


def _array_digest(value: np.ndarray) -> str:
    array = np.asarray(value)
    if array.dtype != np.dtype("<f4") or not array.flags.c_contiguous:
        raise ValueError("phase-response digest 要求little-endian C-order float32")
    return sha256(array.tobytes(order="C")).hexdigest()


def _float64_l2(value: np.ndarray) -> float:
    array = np.asarray(value)
    if not np.all(np.isfinite(array)):
        raise ValueError("phase-response norm 输入必须有限")
    return float(np.linalg.norm(array.astype(np.float64, copy=False).reshape(-1)))


def _safe_cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    left64 = np.asarray(left).astype(np.float64, copy=False).reshape(-1)
    right64 = np.asarray(right).astype(np.float64, copy=False).reshape(-1)
    if not np.all(np.isfinite(left64)) or not np.all(np.isfinite(right64)):
        raise ValueError("phase-response cosine 输入必须有限")
    left_norm = float(np.linalg.norm(left64))
    right_norm = float(np.linalg.norm(right64))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    value = float(np.dot(left64, right64) / (left_norm * right_norm))
    if value > 1.0:
        if value - 1.0 > 1e-12:
            raise ValueError("phase-response cosine 超出Cauchy边界")
        return 1.0
    if value < -1.0:
        if -1.0 - value > 1e-12:
            raise ValueError("phase-response cosine 超出Cauchy边界")
        return -1.0
    return value


def _public_forward_value_signature(value: Any) -> Any:
    if isinstance(value, np.ndarray) or callable(getattr(value, "detach", None)):
        return {"tensor": _tensor_signature(value)}
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("phase-response forward mapping key 必须为字符串")
        return {
            "mapping": {
                key: _public_forward_value_signature(value[key])
                for key in sorted(value)
            }
        }
    if isinstance(value, (list, tuple)):
        return {
            "sequence_type": type(value).__name__,
            "items": [
                _public_forward_value_signature(item) for item in value
            ],
        }
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("phase-response forward scalar 必须有限")
        return {"float": repr(value)}
    raise TypeError(
        "phase-response forward kwargs 含未冻结类型: "
        f"{type(value).__name__}"
    )


def _branch_forward_context_digest(
    kwargs: Mapping[str, Any],
    *,
    branch: str,
) -> str:
    return _stable_digest(
        {
            "cfg_branch_role": branch,
            "forward_kwargs": {
                key: _public_forward_value_signature(kwargs[key])
                for key in sorted(kwargs)
            },
        }
    )


def _branch_encoder_digest(
    kwargs: Mapping[str, Any],
    *,
    branch: str,
    common_input_binding_digest: str,
) -> str:
    return _stable_digest(
        {
            "common_input_binding_digest": common_input_binding_digest,
            "cfg_branch_role": branch,
            "encoder_hidden_states": _tensor_signature(
                kwargs["encoder_hidden_states"]
            ),
        }
    )


def _raw_encoder_value_digest(kwargs: Mapping[str, Any]) -> str:
    return _stable_digest(
        {
            "encoder_hidden_states": _tensor_signature(
                kwargs["encoder_hidden_states"]
            )
        }
    )


def _same_callable_binding(left: Any, right: Any) -> bool:
    if left is right:
        return True
    return (
        getattr(left, "__self__", None) is getattr(right, "__self__", None)
        and getattr(left, "__func__", None) is getattr(right, "__func__", None)
        and getattr(left, "__func__", None) is not None
    )


def _restore_runtime_method(
    target: Any,
    name: str,
    *,
    had_instance_value: bool,
    previous_instance_value: Any,
    original_binding: Any,
) -> None:
    if had_instance_value:
        setattr(target, name, previous_instance_value)
    else:
        instance_dict = getattr(target, "__dict__", {})
        if name in instance_dict:
            delattr(target, name)
    instance_dict = getattr(target, "__dict__", {})
    if had_instance_value:
        if instance_dict.get(name) is not previous_instance_value:
            raise RuntimeError(f"phase-response 未恢复原instance {name}")
    elif name in instance_dict:
        raise RuntimeError(f"phase-response 未移除临时instance {name}")
    if not _same_callable_binding(getattr(target, name), original_binding):
        raise RuntimeError(f"phase-response {name} callable binding 漂移")


def _rope_storage_numpy(value: Any, *, label: str) -> np.ndarray:
    if isinstance(value, np.ndarray):
        source = np.asarray(value)
    else:
        source = value.detach().cpu().numpy()
    if source.dtype != np.dtype("<f4") or not np.all(np.isfinite(source)):
        raise ValueError(f"{label} 必须是有限little-endian float32")
    return np.ascontiguousarray(source)


def _rope_delta_digest(cosine_delta: np.ndarray, sine_delta: np.ndarray) -> str:
    return sha256(
        b"cosine\0"
        + np.ascontiguousarray(cosine_delta, dtype="<f4").tobytes(order="C")
        + b"sine\0"
        + np.ascontiguousarray(sine_delta, dtype="<f4").tobytes(order="C")
    ).hexdigest()


def _tuple_delta_observer(
    sink: list[RopeTupleDeltaObservation],
    *,
    descriptor: Any,
    signed_coefficient: int,
    phase_projection_scale: float,
) -> Callable[[Any, Any, Any, Any], None]:
    def observe(
        original_cosine: Any,
        original_sine: Any,
        shifted_cosine: Any,
        shifted_sine: Any,
    ) -> None:
        original_cosine_array = _rope_storage_numpy(
            original_cosine,
            label="original RoPE cosine",
        )
        original_sine_array = _rope_storage_numpy(
            original_sine,
            label="original RoPE sine",
        )
        shifted_cosine_array = _rope_storage_numpy(
            shifted_cosine,
            label="shifted RoPE cosine",
        )
        shifted_sine_array = _rope_storage_numpy(
            shifted_sine,
            label="shifted RoPE sine",
        )
        cosine_delta = np.subtract(
            shifted_cosine_array,
            original_cosine_array,
            dtype=np.float32,
        )
        sine_delta = np.subtract(
            shifted_sine_array,
            original_sine_array,
            dtype=np.float32,
        )
        exact = bool(
            np.array_equal(original_cosine_array, shifted_cosine_array)
            and np.array_equal(original_sine_array, shifted_sine_array)
        )
        norm = math.sqrt(
            _float64_l2(cosine_delta) ** 2
            + _float64_l2(sine_delta) ** 2
        )
        if not math.isfinite(norm):
            raise RuntimeError("RoPE tuple delta norm 非有限")
        expected_cosine, expected_sine = apply_wan_rotary_phase_runtime(
            original_cosine,
            original_sine,
            descriptor=descriptor,
            signed_coefficient=signed_coefficient,
            phase_projection_scale=phase_projection_scale,
        )
        expected_cosine_array = _rope_storage_numpy(
            expected_cosine,
            label="expected RoPE cosine",
        )
        expected_sine_array = _rope_storage_numpy(
            expected_sine,
            label="expected RoPE sine",
        )
        expected_cosine_delta = np.subtract(
            expected_cosine_array,
            original_cosine_array,
            dtype=np.float32,
        )
        expected_sine_delta = np.subtract(
            expected_sine_array,
            original_sine_array,
            dtype=np.float32,
        )
        expected_norm = math.sqrt(
            _float64_l2(expected_cosine_delta) ** 2
            + _float64_l2(expected_sine_delta) ** 2
        )
        actual_digest = _rope_delta_digest(cosine_delta, sine_delta)
        expected_digest = _rope_delta_digest(
            expected_cosine_delta,
            expected_sine_delta,
        )
        sink.append(
            RopeTupleDeltaObservation(
                actual_delta_norm=norm,
                tuple_exact_identity=exact,
                expected_delta_norm=expected_norm,
                actual_matches_expected=(
                    np.array_equal(cosine_delta, expected_cosine_delta)
                    and np.array_equal(sine_delta, expected_sine_delta)
                    and actual_digest == expected_digest
                ),
                actual_delta_digest=actual_digest,
                expected_delta_digest=expected_digest,
            )
        )

    return observe


def _evaluation_payload(
    evaluation: PhaseProjectionSignEvaluation,
    *,
    repeat_index: int,
) -> dict[str, Any]:
    return {
        "probe_id": evaluation.probe_id,
        "step_index": evaluation.step_index,
        "signed_coefficient": evaluation.signed_coefficient,
        "repeat_index": repeat_index,
        "input_binding_digest": evaluation.input_binding_digest,
        "descriptor_digest": evaluation.descriptor_digest,
        "candidate_context_digest": evaluation.candidate_context_digest,
        "phase_projection_scale": evaluation.phase_projection_scale,
        "delta_sigma": evaluation.delta_sigma,
        "cumulative_reference_energy_before_step": (
            evaluation.cumulative_reference_energy_before_step
        ),
        "cumulative_control_energy_before_step": (
            evaluation.cumulative_control_energy_before_step
        ),
        "remaining_step_count": evaluation.remaining_step_count,
        "base_velocity_norm": evaluation.base_velocity_norm,
        "actual_delta_norm": evaluation.actual_delta_norm,
        "state_update_delta_norm": evaluation.state_update_delta_norm,
        "norm_budget": evaluation.norm_budget,
        "reference_energy_increment": evaluation.reference_energy_increment,
        "projected_reference_energy": (
            evaluation.projected_reference_energy
        ),
        "total_flow_energy_budget": evaluation.total_flow_energy_budget,
        "energy_increment": evaluation.energy_increment,
        "remaining_flow_energy": evaluation.remaining_flow_energy,
        "direction_cosine": evaluation.direction_cosine,
        "signed_state_update_exposure": (
            evaluation.signed_state_update_exposure
        ),
        "norm_guard_passed": evaluation.norm_guard_passed,
        "energy_guard_passed": evaluation.energy_guard_passed,
        "direction_guard_passed": evaluation.direction_guard_passed,
        "feasible": evaluation.feasible,
        "base_raw_velocity_digest": evaluation.base_raw_velocity_digest,
        "base_cfg_velocity_digest": evaluation.base_cfg_velocity_digest,
        "scheduler_sample_digest": evaluation.scheduler_sample_digest,
        "controlled_transition_digest": (
            evaluation.controlled_transition_digest
        ),
        "controlled_cfg_velocity_digest": (
            evaluation.controlled_cfg_velocity_digest
        ),
        "actual_state_update_digest": (
            evaluation.actual_state_update_digest
        ),
    }


def classify_phase_response_preflight(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify only the frozen runtime diagnostic, never a method result."""

    zero_norms = tuple(float(value) for value in record["zero_rope_delta_norms"])
    active_norms = tuple(
        float(value) for value in record["active_rope_delta_norms"]
    )
    expected_active_norms = tuple(
        float(value) for value in record["expected_active_rope_delta_norms"]
    )
    zero_matches_expected = tuple(
        bool(value)
        for value in record["zero_rope_delta_matches_expected"]
    )
    active_matches_expected = tuple(
        bool(value)
        for value in record["active_rope_delta_matches_expected"]
    )
    if (
        len(zero_norms) != 4
        or len(active_norms) != 8
        or len(expected_active_norms) != 8
        or len(zero_matches_expected) != 4
        or len(active_matches_expected) != 8
    ):
        raise ValueError("phase-response tuple-delta coverage 不完整")
    positive_norms = active_norms[:4]
    negative_norms = active_norms[4:]
    tuple_scale_failure = (
        any(value != 0.0 for value in zero_norms)
        or any(not value for value in zero_matches_expected)
        or any(not value for value in active_matches_expected)
        or any(not math.isfinite(value) or value <= 0.0 for value in active_norms)
    )
    if not tuple_scale_failure:
        for actual, expected in zip(active_norms, expected_active_norms):
            if (
                not math.isfinite(expected)
                or expected <= 0.0
                or abs(actual - expected) / max(actual, expected)
                > TUPLE_DELTA_RELATIVE_TOLERANCE
            ):
                tuple_scale_failure = True
                break
    if not tuple_scale_failure:
        for positive, negative in zip(positive_norms, negative_norms):
            denominator = max(positive, negative)
            if (
                abs(positive - negative) / denominator
                > TUPLE_DELTA_RELATIVE_TOLERANCE
            ):
                tuple_scale_failure = True
                break
    response_norms = tuple(
        float(value)
        for value in record[
            "control_base_cfg_delta_norm_by_sign_and_repeat"
        ]
    )
    control_repeat_norms = tuple(
        float(value) for value in record["control_repeat_delta_norm_by_sign"]
    )
    base_repeat_norm = float(record["base_cfg_repeat_delta_norm"])
    if (
        len(response_norms) != 4
        or len(control_repeat_norms) != 2
        or any(
            not math.isfinite(value) or value < 0.0
            for value in (
                base_repeat_norm,
                *response_norms,
                *control_repeat_norms,
            )
        )
    ):
        raise ValueError("phase-response repeatability coverage 不完整")
    minimum_response = min(response_norms)
    base_repeatability_floor_ratio = base_repeat_norm / max(
        minimum_response,
        1e-12,
    )
    control_repeatability_floor_ratio_by_sign = [
        control_repeat_norms[index]
        / max(min(response_norms[index * 2 : index * 2 + 2]), 1e-12)
        for index in range(2)
    ]
    maximum_repeatability_floor_ratio = max(
        base_repeatability_floor_ratio,
        *control_repeatability_floor_ratio_by_sign,
    )
    repeatability_floor = (
        maximum_repeatability_floor_ratio
        > REPEATABILITY_FLOOR_RATIO_THRESHOLD
    )
    evaluations = tuple(record["candidate_evaluations"])
    if len(evaluations) != 4:
        raise ValueError("phase-response candidate evaluation coverage 不完整")
    feasible = all(
        bool(value["feasible"]) for value in evaluations
    )
    dead_zone = all(
        bool(value)
        for value in record["control_cfg_equals_zero_base_by_sign_and_repeat"]
    )
    minimum_norm = min(
        float(value["actual_delta_norm"]) for value in evaluations
    )
    plateau_ratio = (
        minimum_norm / HISTORICAL_LAST_WORST_ACTUAL_DELTA_NORM
    )
    plateau = (
        not repeatability_floor
        and not dead_zone
        and plateau_ratio >= PLATEAU_RATIO_THRESHOLD
    )
    candidates: list[str] = []
    if tuple_scale_failure:
        candidates.append("scale_application_failure")
    else:
        if repeatability_floor:
            candidates.append("forward_repeatability_floor_candidate")
        if feasible and not repeatability_floor:
            candidates.append("feasible_nonzero_phase_region_candidate")
        if dead_zone and not repeatability_floor:
            candidates.append("quantization_dead_zone_candidate")
        if plateau:
            candidates.append("bf16_or_attention_piecewise_plateau_candidate")
    if not candidates:
        candidates.append("indeterminate")
    classification = (
        candidates[0]
        if len(candidates) == 1
        else "multiple_candidates"
    )
    return {
        "diagnostic_classification": classification,
        "diagnostic_candidates": candidates,
        "historical_last_worst_actual_delta_norm": (
            HISTORICAL_LAST_WORST_ACTUAL_DELTA_NORM
        ),
        "minimum_candidate_actual_delta_norm": minimum_norm,
        "candidate_to_historical_plateau_ratio": plateau_ratio,
        "base_repeatability_floor_ratio": base_repeatability_floor_ratio,
        "control_repeatability_floor_ratio_by_sign": (
            control_repeatability_floor_ratio_by_sign
        ),
        "maximum_repeatability_floor_ratio": (
            maximum_repeatability_floor_ratio
        ),
        "repeatability_floor_ratio_threshold": (
            REPEATABILITY_FLOOR_RATIO_THRESHOLD
        ),
        "repeatability_floor_detected": repeatability_floor,
        "unique_root_cause_claim_allowed": False,
        "gate0_pass": False,
        "full_eight_video_rerun_allowed": False,
        "formal_result": False,
        "stage_progression_allowed": False,
    }


_CANDIDATE_EVALUATION_KEYS = {
    "probe_id",
    "step_index",
    "signed_coefficient",
    "repeat_index",
    "input_binding_digest",
    "descriptor_digest",
    "candidate_context_digest",
    "phase_projection_scale",
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
    "feasible",
    "base_raw_velocity_digest",
    "base_cfg_velocity_digest",
    "scheduler_sample_digest",
    "controlled_transition_digest",
    "controlled_cfg_velocity_digest",
    "actual_state_update_digest",
}
_PHASE_RESPONSE_RECORD_KEYS = {
    "record_version",
    "profile_id",
    "probe_id",
    "step_index",
    "phase_projection_scale",
    "realized_phase_magnitude_radians",
    "transformer_forward_count",
    "real_scheduler_step_call_count",
    "scheduler_internal_step_index_before_and_after",
    "scheduler_sample_digest",
    "common_input_binding_digest",
    "branch_forward_context_digests",
    "branch_encoder_digests",
    "branch_raw_encoder_value_digests",
    "base_cfg_digests",
    "base_cfg_repeat_delta_norm",
    "base_cfg_repeats_byte_exact",
    "control_cfg_digests_by_sign_and_repeat",
    "control_cfg_repeats_byte_exact_by_sign",
    "control_repeat_delta_norm_by_sign",
    "control_base_cfg_delta_norm_by_sign_and_repeat",
    "control_cfg_equals_zero_base_by_sign_and_repeat",
    "positive_negative_velocity_direction_cosine",
    "zero_rope_delta_norms",
    "zero_rope_tuple_exact_identity",
    "zero_rope_delta_matches_expected",
    "active_rope_delta_norms",
    "expected_active_rope_delta_norms",
    "active_rope_tuple_exact_identity",
    "active_rope_delta_matches_expected",
    "active_rope_actual_delta_digests",
    "active_rope_expected_delta_digests",
    "candidate_evaluations",
    "decode_executed",
    "video_export_executed",
    "gate0_executed",
    "diagnostic_classification",
    "diagnostic_candidates",
    "historical_last_worst_actual_delta_norm",
    "minimum_candidate_actual_delta_norm",
    "candidate_to_historical_plateau_ratio",
    "base_repeatability_floor_ratio",
    "control_repeatability_floor_ratio_by_sign",
    "maximum_repeatability_floor_ratio",
    "repeatability_floor_ratio_threshold",
    "repeatability_floor_detected",
    "unique_root_cause_claim_allowed",
    "gate0_pass",
    "full_eight_video_rerun_allowed",
    "formal_result",
    "stage_progression_allowed",
}


def _require_sha256_text(value: Any, *, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{label} 必须是64位小写SHA256")
    return text


def _finite_number(value: Any, *, label: str, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} 不得是bool")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{label} 数值非法")
    return result


def validate_phase_response_preflight_batch(
    batch: PhaseResponsePreflightBatch,
) -> dict[str, Any]:
    """Validate and recompute the complete non-Gate diagnostic record."""

    if not isinstance(batch, PhaseResponsePreflightBatch):
        raise TypeError("phase-response batch 类型不匹配")
    if (
        batch.transformer_forward_count != 12
        or batch.scheduler_step_call_count != 0
    ):
        raise ValueError("phase-response batch 调用计数漂移")
    _require_sha256_text(
        batch.initial_hidden_state_digest_random,
        label="initial hidden state digest",
    )
    _require_sha256_text(
        batch.generator_state_digest_random,
        label="generator state digest",
    )
    record = dict(batch.record)
    if set(record) != _PHASE_RESPONSE_RECORD_KEYS:
        raise ValueError("phase-response record 字段集合漂移")
    if (
        record["record_version"] != RECORD_VERSION
        or record["profile_id"] != PROFILE_ID
        or record["probe_id"] != TEST_ID
        or record["step_index"] != 0
        or record["phase_projection_scale"] != NEXT_DETERMINISTIC_SCALE
        or record["realized_phase_magnitude_radians"]
        != PHASE_BUDGET_RADIANS * NEXT_DETERMINISTIC_SCALE
        or record["transformer_forward_count"] != 12
        or record["real_scheduler_step_call_count"] != 0
        or record["scheduler_internal_step_index_before_and_after"]
        != [None, None]
    ):
        raise ValueError("phase-response record identity/call boundary 漂移")
    for field_name in (
        "scheduler_sample_digest",
        "common_input_binding_digest",
    ):
        _require_sha256_text(record[field_name], label=field_name)
    for field_name, expected_count in (
        ("branch_forward_context_digests", 2),
        ("branch_encoder_digests", 2),
        ("branch_raw_encoder_value_digests", 2),
        ("base_cfg_digests", 2),
        ("control_cfg_digests_by_sign_and_repeat", 4),
        ("active_rope_actual_delta_digests", 8),
        ("active_rope_expected_delta_digests", 8),
    ):
        values = list(record[field_name])
        if len(values) != expected_count:
            raise ValueError(f"{field_name} coverage 漂移")
        for index, value in enumerate(values):
            _require_sha256_text(value, label=f"{field_name}[{index}]")
    if (
        record["branch_raw_encoder_value_digests"][0]
        == record["branch_raw_encoder_value_digests"][1]
    ):
        raise ValueError(
            "phase-response cond/uncond raw encoder value digest 不得相同"
        )
    for field_name, expected_count in (
        ("zero_rope_delta_norms", 4),
        ("active_rope_delta_norms", 8),
        ("expected_active_rope_delta_norms", 8),
        ("control_repeat_delta_norm_by_sign", 2),
        ("control_base_cfg_delta_norm_by_sign_and_repeat", 4),
    ):
        values = list(record[field_name])
        if len(values) != expected_count:
            raise ValueError(f"{field_name} coverage 漂移")
        for index, value in enumerate(values):
            _finite_number(
                value,
                label=f"{field_name}[{index}]",
                minimum=0.0,
            )
    _finite_number(
        record["base_cfg_repeat_delta_norm"],
        label="base_cfg_repeat_delta_norm",
        minimum=0.0,
    )
    cosine = record["positive_negative_velocity_direction_cosine"]
    if cosine is not None:
        cosine_value = _finite_number(
            cosine,
            label="positive_negative_velocity_direction_cosine",
        )
        if cosine_value < -1.0 or cosine_value > 1.0:
            raise ValueError("phase-response direction cosine 越界")
    for field_name, expected_count in (
        ("zero_rope_tuple_exact_identity", 4),
        ("zero_rope_delta_matches_expected", 4),
        ("active_rope_tuple_exact_identity", 8),
        ("active_rope_delta_matches_expected", 8),
        ("control_cfg_repeats_byte_exact_by_sign", 2),
        ("control_cfg_equals_zero_base_by_sign_and_repeat", 4),
    ):
        values = list(record[field_name])
        if len(values) != expected_count or any(
            not isinstance(value, bool) for value in values
        ):
            raise ValueError(f"{field_name} bool coverage 漂移")
    if (
        not isinstance(record["base_cfg_repeats_byte_exact"], bool)
        or any(not value for value in record["zero_rope_tuple_exact_identity"])
        or any(record["active_rope_tuple_exact_identity"])
    ):
        raise ValueError("phase-response RoPE identity semantics 漂移")
    base_digests = list(record["base_cfg_digests"])
    control_digests = list(
        record["control_cfg_digests_by_sign_and_repeat"]
    )
    if record["base_cfg_repeats_byte_exact"] != (
        base_digests[0] == base_digests[1]
    ):
        raise ValueError("phase-response base digest/byte-exact 漂移")
    if (
        base_digests[0] == base_digests[1]
        and record["base_cfg_repeat_delta_norm"] != 0.0
    ):
        raise ValueError("phase-response base digest/repeat norm 漂移")
    expected_control_repeat_exact = [
        control_digests[0] == control_digests[1],
        control_digests[2] == control_digests[3],
    ]
    if (
        record["control_cfg_repeats_byte_exact_by_sign"]
        != expected_control_repeat_exact
    ):
        raise ValueError("phase-response control digest/repeat bool 漂移")
    for digest_equal, delta_norm in zip(
        expected_control_repeat_exact,
        record["control_repeat_delta_norm_by_sign"],
    ):
        if digest_equal and delta_norm != 0.0:
            raise ValueError("phase-response control digest/repeat norm 漂移")
    expected_control_equals_base = [
        digest == base_digests[0] for digest in control_digests
    ]
    if (
        record["control_cfg_equals_zero_base_by_sign_and_repeat"]
        != expected_control_equals_base
    ):
        raise ValueError("phase-response control/base digest bool 漂移")
    for digest_equal, delta_norm in zip(
        expected_control_equals_base,
        record["control_base_cfg_delta_norm_by_sign_and_repeat"],
    ):
        if digest_equal and delta_norm != 0.0:
            raise ValueError("phase-response control/base digest/norm 漂移")
    digest_matches = [
        actual == expected
        for actual, expected in zip(
            record["active_rope_actual_delta_digests"],
            record["active_rope_expected_delta_digests"],
        )
    ]
    if digest_matches != record["active_rope_delta_matches_expected"]:
        raise ValueError("phase-response RoPE actual/expected digest binding 漂移")
    evaluations = list(record["candidate_evaluations"])
    issued_evaluations = tuple(batch.candidate_evaluations)
    if len(evaluations) != 4 or len(issued_evaluations) != 4:
        raise ValueError("phase-response candidate evaluation count 漂移")
    gate0_config = load_patch_relation_gate0_config()
    _, frozen_delta_sigma, _ = _frozen_schedule(gate0_config)
    descriptor_digest = (
        build_public_patch_relation_descriptor().descriptor_digest
    )
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
    frozen_shared_context: tuple[Any, ...] | None = None
    for index, (
        evaluation,
        issued_evaluation,
        expected_sign,
        expected_repeat,
    ) in enumerate(
        zip(
            evaluations,
            issued_evaluations,
            (1, 1, -1, -1),
            (0, 1, 0, 1),
        )
    ):
        require_validated_phase_projection_sign_evaluation(
            issued_evaluation
        )
        if (
            not isinstance(evaluation, Mapping)
            or set(evaluation) != _CANDIDATE_EVALUATION_KEYS
        ):
            raise ValueError(f"candidate_evaluations[{index}] 字段漂移")
        expected_payload = _evaluation_payload(
            issued_evaluation,
            repeat_index=expected_repeat,
        )
        if dict(evaluation) != expected_payload:
            raise ValueError(
                f"candidate_evaluations[{index}] 与runtime-issued来源不一致"
            )
        if (
            evaluation["probe_id"] != TEST_ID
            or evaluation["step_index"] != 0
            or evaluation["signed_coefficient"] != expected_sign
            or evaluation["repeat_index"] != expected_repeat
            or evaluation["input_binding_digest"]
            != record["common_input_binding_digest"]
            or evaluation["descriptor_digest"] != descriptor_digest
            or evaluation["phase_projection_scale"]
            != NEXT_DETERMINISTIC_SCALE
            or evaluation["delta_sigma"] != frozen_delta_sigma[0]
            or evaluation["base_cfg_velocity_digest"] != base_digests[0]
            or evaluation["controlled_cfg_velocity_digest"]
            != control_digests[index]
            or evaluation["scheduler_sample_digest"]
            != record["scheduler_sample_digest"]
        ):
            raise ValueError(f"candidate_evaluations[{index}] identity 漂移")
        observed_shared_context = tuple(
            getattr(issued_evaluation, field_name)
            for field_name in shared_context_fields
        )
        if frozen_shared_context is None:
            frozen_shared_context = observed_shared_context
        elif observed_shared_context != frozen_shared_context:
            raise ValueError("candidate_evaluations shared step0 context 漂移")
        for field_name in (
            "delta_sigma",
            "cumulative_reference_energy_before_step",
            "cumulative_control_energy_before_step",
            "base_velocity_norm",
            "actual_delta_norm",
            "state_update_delta_norm",
            "norm_budget",
            "reference_energy_increment",
            "projected_reference_energy",
            "total_flow_energy_budget",
            "energy_increment",
            "remaining_flow_energy",
        ):
            minimum = None if field_name == "delta_sigma" else 0.0
            _finite_number(
                evaluation[field_name],
                label=f"candidate_evaluations[{index}].{field_name}",
                minimum=minimum,
            )
        if (
            not isinstance(evaluation["remaining_step_count"], int)
            or isinstance(evaluation["remaining_step_count"], bool)
            or evaluation["remaining_step_count"] != 8
            or evaluation["cumulative_reference_energy_before_step"] != 0.0
            or evaluation["cumulative_control_energy_before_step"] != 0.0
        ):
            raise ValueError("candidate step0 energy context 漂移")
        _finite_number(
            evaluation["signed_state_update_exposure"],
            label=f"candidate_evaluations[{index}].signed_state_update_exposure",
        )
        direction = evaluation["direction_cosine"]
        if direction is not None:
            direction_value = _finite_number(
                direction,
                label=f"candidate_evaluations[{index}].direction_cosine",
            )
            if direction_value < -1.0 or direction_value > 1.0:
                raise ValueError("candidate direction cosine 越界")
        for field_name in (
            "norm_guard_passed",
            "energy_guard_passed",
            "direction_guard_passed",
            "feasible",
        ):
            if not isinstance(evaluation[field_name], bool):
                raise TypeError(f"candidate {field_name} 必须是bool")
        expected_norm_budget = (
            evaluation["base_velocity_norm"]
            * VELOCITY_NORM_RATIO_BUDGET
            * LAMBDA_MAX
        )
        expected_projected_reference = (
            evaluation["cumulative_reference_energy_before_step"]
            + evaluation["reference_energy_increment"]
            * evaluation["remaining_step_count"]
        )
        expected_total_flow_energy_budget = (
            FLOW_ENERGY_BUDGET_RATIO * expected_projected_reference
        )
        expected_remaining_flow_energy = max(
            0.0,
            expected_total_flow_energy_budget
            - evaluation["cumulative_control_energy_before_step"],
        )
        expected_energy_increment = (
            evaluation["state_update_delta_norm"] ** 2
        )
        expected_signed_exposure = (
            evaluation["signed_coefficient"]
            * evaluation["state_update_delta_norm"]
        )
        expected_norm_guard = _budget_guard_passed(
            evaluation["actual_delta_norm"],
            expected_norm_budget,
        )
        expected_energy_guard = _budget_guard_passed(
            expected_energy_increment,
            expected_remaining_flow_energy,
        )
        expected_direction_guard = bool(
            direction is not None
            and direction_value + 1e-12 >= MINIMUM_DIRECTION_COSINE
        )
        for field_name, expected in (
            ("norm_budget", expected_norm_budget),
            ("projected_reference_energy", expected_projected_reference),
            (
                "total_flow_energy_budget",
                expected_total_flow_energy_budget,
            ),
            ("remaining_flow_energy", expected_remaining_flow_energy),
            ("energy_increment", expected_energy_increment),
            ("signed_state_update_exposure", expected_signed_exposure),
            ("norm_guard_passed", expected_norm_guard),
            ("energy_guard_passed", expected_energy_guard),
            ("direction_guard_passed", expected_direction_guard),
        ):
            if evaluation[field_name] != expected:
                raise ValueError(
                    "candidate frozen budget/state-update formula 漂移: "
                    f"{field_name}"
                )
        expected_feasible = bool(
            expected_norm_guard
            and expected_energy_guard
            and expected_direction_guard
        )
        if evaluation["feasible"] != expected_feasible:
            raise ValueError("candidate feasible 与guards不一致")
        for field_name in (
            "controlled_transition_digest",
            "controlled_cfg_velocity_digest",
            "actual_state_update_digest",
        ):
            _require_sha256_text(
                evaluation[field_name],
                label=f"candidate_evaluations[{index}].{field_name}",
            )
    for field_name in (
        "decode_executed",
        "video_export_executed",
        "gate0_executed",
        "gate0_pass",
        "full_eight_video_rerun_allowed",
        "formal_result",
        "stage_progression_allowed",
        "unique_root_cause_claim_allowed",
    ):
        if record[field_name] is not False:
            raise ValueError(f"phase-response record 禁止边界被打开: {field_name}")
    recomputed = classify_phase_response_preflight(record)
    for field_name, expected in recomputed.items():
        observed = record.get(field_name)
        if observed != expected:
            raise ValueError(f"phase-response record 派生字段漂移: {field_name}")
    record.update(recomputed)
    return record


class ScopedPatchRelationPhaseResponsePreflight:
    """Capture one real step-0 response and stop before scheduler progression."""

    def __init__(
        self,
        transformer: Any,
        scheduler: Any,
        *,
        descriptor: Any,
        probe_id: str,
        sigma_grid: tuple[float, ...],
        delta_sigma_by_step: tuple[float, ...],
        timestep_by_step: tuple[float, ...],
    ) -> None:
        self.transformer = transformer
        self.scheduler = scheduler
        self.descriptor = descriptor
        self.probe_id = probe_id
        self.sigma_grid = sigma_grid
        self.delta_sigma_by_step = delta_sigma_by_step
        self.timestep_by_step = timestep_by_step
        self._original_forward: Any = None
        self._original_scheduler_step: Any = None
        self._transformer_had_forward = False
        self._transformer_previous_forward: Any = None
        self._scheduler_had_step = False
        self._scheduler_previous_step: Any = None
        self._entered = False
        self._exit_attempted = False
        self._cleanup_completed = False
        self._forward_calls = 0
        self._scheduler_calls = 0
        self._branch_kwargs: dict[str, Mapping[str, Any]] = {}
        self._common_input_binding_digest = ""
        self._branch_forward_context_digests: dict[str, str] = {}
        self._branch_encoder_digests: dict[str, str] = {}
        self._branch_raw_encoder_value_digests: dict[str, str] = {}
        self._base: dict[
            tuple[int, str],
            tuple[
                Any,
                WanRopeBranchApplicationRecord,
                RopeTupleDeltaObservation,
            ],
        ] = {}
        self._initial_hidden_digest = ""
        self._record: Mapping[str, Any] | None = None
        self._candidate_evaluations: tuple[
            PhaseProjectionSignEvaluation,
            ...,
        ] = ()
        self._completion_sentinel: _PreflightComplete | None = None
        self._completed = False

    @property
    def initial_hidden_state_digest_random(self) -> str:
        if not self._completed:
            raise RuntimeError("phase-response preflight 尚未完成")
        return self._initial_hidden_digest

    def _input_binding(self, hidden: Any, timestep: Any) -> str:
        return _stable_digest(
            {
                "probe_id": self.probe_id,
                "step_index": 0,
                "hidden_states": _tensor_signature(hidden),
                "timestep": _timestep_signature(timestep),
            }
        )

    def _run_forward(
        self,
        kwargs: Mapping[str, Any],
        *,
        branch: str,
        coefficient: int,
        repeat_index: int,
        scale: float,
        input_binding: str,
    ) -> tuple[
        Any,
        WanRopeBranchApplicationRecord,
        RopeTupleDeltaObservation,
    ]:
        if self._input_binding(
            kwargs["hidden_states"],
            kwargs["timestep"],
        ) != input_binding:
            raise RuntimeError("phase-response common input binding 漂移")
        if (
            _branch_forward_context_digest(kwargs, branch=branch)
            != self._branch_forward_context_digests[branch]
            or _branch_encoder_digest(
                kwargs,
                branch=branch,
                common_input_binding_digest=input_binding,
            )
            != self._branch_encoder_digests[branch]
            or _raw_encoder_value_digest(kwargs)
            != self._branch_raw_encoder_value_digests[branch]
        ):
            raise RuntimeError("phase-response branch forward context 漂移")
        tuple_observations: list[RopeTupleDeltaObservation] = []
        scope = ScopedWanRopeOutputAdapter(
            self.transformer,
            descriptor=self.descriptor,
            signed_coefficient=coefficient,
            phase_projection_scale=scale,
            probe_id=self.probe_id,
            step_index=0,
            control_role="base" if coefficient == 0 else "controlled",
            cfg_branch_role=branch,
            input_binding_digest=input_binding,
            diagnostic_tuple_observer=_tuple_delta_observer(
                tuple_observations,
                descriptor=self.descriptor,
                signed_coefficient=coefficient,
                phase_projection_scale=scale,
            ),
        )
        with scope:
            result = self._original_forward(**dict(kwargs))
        if len(tuple_observations) != 1:
            raise RuntimeError("phase-response RoPE tuple observation 不完整")
        self._forward_calls += 1
        return result, scope.record(), tuple_observations[0]

    def _wrapped_forward(self, *args: Any, **kwargs: Any) -> Any:
        if args or self._forward_calls not in (0, 2):
            raise RuntimeError("phase-response external transformer 调用漂移")
        required = {"hidden_states", "timestep", "encoder_hidden_states"}
        if not required.issubset(kwargs):
            raise RuntimeError("phase-response transformer kwargs 不完整")
        branch = "conditional" if self._forward_calls == 0 else "unconditional"
        binding = self._input_binding(
            kwargs["hidden_states"],
            kwargs["timestep"],
        )
        if not self._common_input_binding_digest:
            self._common_input_binding_digest = binding
        elif binding != self._common_input_binding_digest:
            raise RuntimeError("phase-response cond/uncond hidden/timestep 漂移")
        context_digest = _branch_forward_context_digest(
            kwargs,
            branch=branch,
        )
        encoder_digest = _branch_encoder_digest(
            kwargs,
            branch=branch,
            common_input_binding_digest=binding,
        )
        if branch in self._branch_forward_context_digests:
            raise RuntimeError("phase-response branch forward context 重复")
        self._branch_forward_context_digests[branch] = context_digest
        self._branch_encoder_digests[branch] = encoder_digest
        self._branch_raw_encoder_value_digests[branch] = (
            _raw_encoder_value_digest(kwargs)
        )
        if (
            branch == "unconditional"
            and self._branch_raw_encoder_value_digests["conditional"]
            == self._branch_raw_encoder_value_digests["unconditional"]
        ):
            raise RuntimeError(
                "phase-response cond/uncond raw encoder values 不得相同"
            )
        if branch == "conditional":
            self._initial_hidden_digest = _tensor_values_digest(
                kwargs["hidden_states"]
            )
        self._branch_kwargs[branch] = dict(kwargs)
        first_result: Any = None
        for repeat_index in (0, 1):
            result = self._run_forward(
                kwargs,
                branch=branch,
                coefficient=0,
                repeat_index=repeat_index,
                scale=1.0,
                input_binding=binding,
            )
            self._base[(repeat_index, branch)] = result
            if repeat_index == 0:
                first_result = result[0]
        velocity = _runtime_float32_velocity(
            _extract_transformer_velocity(
                first_result,
                label=f"phase-response base {branch}",
            ),
            label=f"phase-response base {branch} velocity",
        )
        return _replace_tuple_velocity(
            first_result,
            velocity,
            label=f"phase-response base {branch}",
        )

    def _forward_pair(
        self,
        *,
        coefficient: int,
        repeat_index: int,
        scale: float,
        binding: str,
    ) -> PhaseResponseForward:
        rows: dict[
            str,
            tuple[
                Any,
                WanRopeBranchApplicationRecord,
                RopeTupleDeltaObservation,
            ],
        ] = {}
        for branch in ("conditional", "unconditional"):
            rows[branch] = self._run_forward(
                self._branch_kwargs[branch],
                branch=branch,
                coefficient=coefficient,
                repeat_index=repeat_index,
                scale=scale,
                input_binding=binding,
            )
        pair = validate_cfg_rope_application_pair(
            rows["conditional"][1],
            rows["unconditional"][1],
        )
        conditional = _to_float32_numpy(
            _runtime_float32_velocity(
                _extract_transformer_velocity(
                    rows["conditional"][0],
                    label="phase-response conditional",
                ),
                label="phase-response conditional velocity",
            ),
            label="phase-response conditional velocity",
        )
        unconditional = _to_float32_numpy(
            _runtime_float32_velocity(
                _extract_transformer_velocity(
                    rows["unconditional"][0],
                    label="phase-response unconditional",
                ),
                label="phase-response unconditional velocity",
            ),
            label="phase-response unconditional velocity",
        )
        cfg = _to_float32_numpy(
            _runtime_cfg_combine(conditional, unconditional),
            label="phase-response CFG velocity",
        )
        return PhaseResponseForward(
            signed_coefficient=coefficient,
            repeat_index=repeat_index,
            conditional_velocity=conditional,
            unconditional_velocity=unconditional,
            cfg_velocity=cfg,
            pair=pair,
            conditional_rope_tuple_delta_norm=(
                rows["conditional"][2].actual_delta_norm
            ),
            unconditional_rope_tuple_delta_norm=(
                rows["unconditional"][2].actual_delta_norm
            ),
            conditional_expected_rope_tuple_delta_norm=(
                rows["conditional"][2].expected_delta_norm
            ),
            unconditional_expected_rope_tuple_delta_norm=rows[
                "unconditional"
            ][2].expected_delta_norm,
            conditional_rope_tuple_exact_identity=(
                rows["conditional"][2].tuple_exact_identity
            ),
            unconditional_rope_tuple_exact_identity=(
                rows["unconditional"][2].tuple_exact_identity
            ),
            conditional_rope_delta_matches_expected=(
                rows["conditional"][2].actual_matches_expected
            ),
            unconditional_rope_delta_matches_expected=(
                rows["unconditional"][2].actual_matches_expected
            ),
            conditional_rope_actual_delta_digest=(
                rows["conditional"][2].actual_delta_digest
            ),
            unconditional_rope_actual_delta_digest=(
                rows["unconditional"][2].actual_delta_digest
            ),
            conditional_rope_expected_delta_digest=(
                rows["conditional"][2].expected_delta_digest
            ),
            unconditional_rope_expected_delta_digest=(
                rows["unconditional"][2].expected_delta_digest
            ),
        )

    def _wrapped_scheduler_step(
        self,
        model_output: Any,
        timestep: Any,
        sample: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        del args, kwargs
        if self._scheduler_calls != 0 or len(self._branch_kwargs) != 2:
            raise RuntimeError("phase-response scheduler boundary 漂移")
        actual_timestep = _scalar_float32(
            timestep,
            label="phase-response scheduler timestep",
        )
        scheduler_timesteps = getattr(self.scheduler, "timesteps", None)
        scheduler_sigmas = getattr(self.scheduler, "sigmas", None)
        if (
            scheduler_timesteps is None
            or len(scheduler_timesteps) != 8
            or scheduler_sigmas is None
            or len(scheduler_sigmas) != 9
            or _scalar_float32(
                scheduler_timesteps[0],
                label="phase-response frozen scheduler timestep",
            )
            != self.timestep_by_step[0]
            or _scalar_float32(
                scheduler_sigmas[0],
                label="phase-response sigma before",
            )
            != self.sigma_grid[0]
            or _scalar_float32(
                scheduler_sigmas[1],
                label="phase-response sigma after",
            )
            != self.sigma_grid[1]
            or actual_timestep != self.timestep_by_step[0]
        ):
            raise RuntimeError("phase-response timestep 不是冻结step0")
        if getattr(self.scheduler, "step_index", None) is not None:
            raise RuntimeError("phase-response scheduler 已提前推进")
        base_rows: list[PhaseResponseForward] = []
        binding = self._input_binding(
            self._branch_kwargs["conditional"]["hidden_states"],
            self._branch_kwargs["conditional"]["timestep"],
        )
        for repeat_index in (0, 1):
            conditional_row = self._base[(repeat_index, "conditional")]
            unconditional_row = self._base[(repeat_index, "unconditional")]
            pair = validate_cfg_rope_application_pair(
                conditional_row[1],
                unconditional_row[1],
            )
            conditional = _to_float32_numpy(
                _extract_transformer_velocity(
                    conditional_row[0],
                    label="phase-response base conditional",
                ),
                label="phase-response base conditional",
            )
            unconditional = _to_float32_numpy(
                _extract_transformer_velocity(
                    unconditional_row[0],
                    label="phase-response base unconditional",
                ),
                label="phase-response base unconditional",
            )
            base_rows.append(
                PhaseResponseForward(
                    signed_coefficient=0,
                    repeat_index=repeat_index,
                    conditional_velocity=conditional,
                    unconditional_velocity=unconditional,
                    cfg_velocity=_to_float32_numpy(
                        _runtime_cfg_combine(conditional, unconditional),
                        label="phase-response base CFG",
                    ),
                    pair=pair,
                    conditional_rope_tuple_delta_norm=(
                        conditional_row[2].actual_delta_norm
                    ),
                    unconditional_rope_tuple_delta_norm=(
                        unconditional_row[2].actual_delta_norm
                    ),
                    conditional_expected_rope_tuple_delta_norm=(
                        conditional_row[2].expected_delta_norm
                    ),
                    unconditional_expected_rope_tuple_delta_norm=(
                        unconditional_row[2].expected_delta_norm
                    ),
                    conditional_rope_tuple_exact_identity=(
                        conditional_row[2].tuple_exact_identity
                    ),
                    unconditional_rope_tuple_exact_identity=(
                        unconditional_row[2].tuple_exact_identity
                    ),
                    conditional_rope_delta_matches_expected=(
                        conditional_row[2].actual_matches_expected
                    ),
                    unconditional_rope_delta_matches_expected=(
                        unconditional_row[2].actual_matches_expected
                    ),
                    conditional_rope_actual_delta_digest=(
                        conditional_row[2].actual_delta_digest
                    ),
                    unconditional_rope_actual_delta_digest=(
                        unconditional_row[2].actual_delta_digest
                    ),
                    conditional_rope_expected_delta_digest=(
                        conditional_row[2].expected_delta_digest
                    ),
                    unconditional_rope_expected_delta_digest=(
                        unconditional_row[2].expected_delta_digest
                    ),
                )
            )
        pipeline_cfg = _to_float32_numpy(
            _runtime_float32_velocity(
                model_output,
                label="phase-response pipeline CFG",
            ),
            label="phase-response pipeline CFG",
        )
        if not np.array_equal(pipeline_cfg, base_rows[0].cfg_velocity):
            raise RuntimeError("phase-response pipeline CFG 与zero base不一致")
        scheduler_sample = _to_float32_numpy(
            _runtime_float32_velocity(
                sample,
                label="phase-response scheduler sample",
            ),
            label="phase-response scheduler sample",
        )
        sample_cast = _scheduler_sample_as_transformer_input(
            sample,
            getattr(self.transformer, "dtype", None),
        )
        scheduler_timestep_signature = _timestep_signature(timestep)
        for branch in ("conditional", "unconditional"):
            branch_kwargs = self._branch_kwargs[branch]
            if (
                _tensor_signature(sample_cast)
                != _tensor_signature(branch_kwargs["hidden_states"])
                or scheduler_timestep_signature
                != _timestep_signature(branch_kwargs["timestep"])
                or _branch_forward_context_digest(
                    branch_kwargs,
                    branch=branch,
                )
                != self._branch_forward_context_digests[branch]
                or _branch_encoder_digest(
                    branch_kwargs,
                    branch=branch,
                    common_input_binding_digest=binding,
                )
                != self._branch_encoder_digests[branch]
                or _raw_encoder_value_digest(branch_kwargs)
                != self._branch_raw_encoder_value_digests[branch]
            ):
                raise RuntimeError(
                    "phase-response scheduler/transformer/branch context 漂移"
                )
        controls = {
            (sign, repeat): self._forward_pair(
                coefficient=sign,
                repeat_index=repeat,
                scale=NEXT_DETERMINISTIC_SCALE,
                binding=binding,
            )
            for sign in (1, -1)
            for repeat in (0, 1)
        }
        interval = self.delta_sigma_by_step[0]
        evaluations: list[PhaseProjectionSignEvaluation] = []
        for sign in (1, -1):
            for repeat in (0, 1):
                row = controls[(sign, repeat)]
                evaluations.append(
                    evaluate_phase_projection_sign_numpy(
                        base_pair=base_rows[0].pair,
                        controlled_pair=row.pair,
                        phase_projection_scale=NEXT_DETERMINISTIC_SCALE,
                        base_conditional_velocity=(
                            base_rows[0].conditional_velocity
                        ),
                        base_unconditional_velocity=(
                            base_rows[0].unconditional_velocity
                        ),
                        controlled_conditional_velocity=(
                            row.conditional_velocity
                        ),
                        controlled_unconditional_velocity=(
                            row.unconditional_velocity
                        ),
                        scheduler_sample=scheduler_sample,
                        delta_sigma=interval,
                        cumulative_reference_energy_before_step=0.0,
                        cumulative_control_energy_before_step=0.0,
                        remaining_step_count=8,
                    )
                )
        positive_delta = np.subtract(
            controls[(1, 0)].cfg_velocity,
            base_rows[0].cfg_velocity,
            dtype=np.float32,
        )
        negative_delta = np.subtract(
            controls[(-1, 0)].cfg_velocity,
            base_rows[0].cfg_velocity,
            dtype=np.float32,
        )
        base_cfg_digests = [
            _array_digest(row.cfg_velocity) for row in base_rows
        ]
        control_cfg_digests = [
            _array_digest(controls[(sign, repeat)].cfg_velocity)
            for sign in (1, -1)
            for repeat in (0, 1)
        ]
        record: dict[str, Any] = {
            "record_version": RECORD_VERSION,
            "profile_id": PROFILE_ID,
            "probe_id": self.probe_id,
            "step_index": 0,
            "phase_projection_scale": NEXT_DETERMINISTIC_SCALE,
            "realized_phase_magnitude_radians": (
                PHASE_BUDGET_RADIANS * NEXT_DETERMINISTIC_SCALE
            ),
            "transformer_forward_count": self._forward_calls,
            "real_scheduler_step_call_count": 0,
            "scheduler_internal_step_index_before_and_after": [
                getattr(self.scheduler, "step_index", None),
                getattr(self.scheduler, "step_index", None),
            ],
            "scheduler_sample_digest": _array_digest(scheduler_sample),
            "common_input_binding_digest": binding,
            "branch_forward_context_digests": [
                self._branch_forward_context_digests[branch]
                for branch in ("conditional", "unconditional")
            ],
            "branch_encoder_digests": [
                self._branch_encoder_digests[branch]
                for branch in ("conditional", "unconditional")
            ],
            "branch_raw_encoder_value_digests": [
                self._branch_raw_encoder_value_digests[branch]
                for branch in ("conditional", "unconditional")
            ],
            "base_cfg_digests": base_cfg_digests,
            "base_cfg_repeat_delta_norm": _float64_l2(
                np.subtract(
                    base_rows[1].cfg_velocity,
                    base_rows[0].cfg_velocity,
                    dtype=np.float32,
                )
            ),
            "base_cfg_repeats_byte_exact": (
                base_cfg_digests[0] == base_cfg_digests[1]
            ),
            "control_cfg_digests_by_sign_and_repeat": control_cfg_digests,
            "control_cfg_repeats_byte_exact_by_sign": [
                control_cfg_digests[offset]
                == control_cfg_digests[offset + 1]
                for offset in (0, 2)
            ],
            "control_repeat_delta_norm_by_sign": [
                _float64_l2(
                    np.subtract(
                        controls[(sign, 1)].cfg_velocity,
                        controls[(sign, 0)].cfg_velocity,
                        dtype=np.float32,
                    )
                )
                for sign in (1, -1)
            ],
            "control_base_cfg_delta_norm_by_sign_and_repeat": [
                _float64_l2(
                    np.subtract(
                        controls[(sign, repeat)].cfg_velocity,
                        base_rows[0].cfg_velocity,
                        dtype=np.float32,
                    )
                )
                for sign in (1, -1)
                for repeat in (0, 1)
            ],
            "control_cfg_equals_zero_base_by_sign_and_repeat": [
                digest == base_cfg_digests[0]
                for digest in control_cfg_digests
            ],
            "positive_negative_velocity_direction_cosine": _safe_cosine(
                positive_delta,
                -negative_delta,
            ),
            "zero_rope_delta_norms": [
                value
                for row in base_rows
                for value in (
                    row.conditional_rope_tuple_delta_norm,
                    row.unconditional_rope_tuple_delta_norm,
                )
            ],
            "zero_rope_tuple_exact_identity": [
                value
                for row in base_rows
                for value in (
                    row.conditional_rope_tuple_exact_identity,
                    row.unconditional_rope_tuple_exact_identity,
                )
            ],
            "zero_rope_delta_matches_expected": [
                value
                for row in base_rows
                for value in (
                    row.conditional_rope_delta_matches_expected,
                    row.unconditional_rope_delta_matches_expected,
                )
            ],
            "active_rope_delta_norms": [
                value
                for sign in (1, -1)
                for repeat in (0, 1)
                for value in (
                    controls[(sign, repeat)].conditional_rope_tuple_delta_norm,
                    controls[(sign, repeat)].unconditional_rope_tuple_delta_norm,
                )
            ],
            "expected_active_rope_delta_norms": [
                value
                for sign in (1, -1)
                for repeat in (0, 1)
                for value in (
                    controls[
                        (sign, repeat)
                    ].conditional_expected_rope_tuple_delta_norm,
                    controls[
                        (sign, repeat)
                    ].unconditional_expected_rope_tuple_delta_norm,
                )
            ],
            "active_rope_tuple_exact_identity": [
                value
                for sign in (1, -1)
                for repeat in (0, 1)
                for value in (
                    controls[(sign, repeat)].conditional_rope_tuple_exact_identity,
                    controls[(sign, repeat)].unconditional_rope_tuple_exact_identity,
                )
            ],
            "active_rope_delta_matches_expected": [
                value
                for sign in (1, -1)
                for repeat in (0, 1)
                for value in (
                    controls[
                        (sign, repeat)
                    ].conditional_rope_delta_matches_expected,
                    controls[
                        (sign, repeat)
                    ].unconditional_rope_delta_matches_expected,
                )
            ],
            "active_rope_actual_delta_digests": [
                value
                for sign in (1, -1)
                for repeat in (0, 1)
                for value in (
                    controls[
                        (sign, repeat)
                    ].conditional_rope_actual_delta_digest,
                    controls[
                        (sign, repeat)
                    ].unconditional_rope_actual_delta_digest,
                )
            ],
            "active_rope_expected_delta_digests": [
                value
                for sign in (1, -1)
                for repeat in (0, 1)
                for value in (
                    controls[
                        (sign, repeat)
                    ].conditional_rope_expected_delta_digest,
                    controls[
                        (sign, repeat)
                    ].unconditional_rope_expected_delta_digest,
                )
            ],
            "candidate_evaluations": [
                _evaluation_payload(evaluation, repeat_index=repeat)
                for evaluation, (sign, repeat) in zip(
                    evaluations,
                    (
                        (1, 0),
                        (1, 1),
                        (-1, 0),
                        (-1, 1),
                    ),
                )
            ],
            "decode_executed": False,
            "video_export_executed": False,
            "gate0_executed": False,
            "formal_result": False,
            "stage_progression_allowed": False,
        }
        record.update(classify_phase_response_preflight(record))
        self._record = record
        self._candidate_evaluations = tuple(evaluations)
        self._scheduler_calls = 0
        self._completion_sentinel = _PreflightComplete(
            "phase-response preflight completed"
        )
        raise self._completion_sentinel

    def __enter__(self) -> "ScopedPatchRelationPhaseResponsePreflight":
        if self._entered:
            raise RuntimeError("phase-response scope 不得重复进入")
        transformer_dict = getattr(self.transformer, "__dict__", {})
        scheduler_dict = getattr(self.scheduler, "__dict__", {})
        self._transformer_had_forward = "forward" in transformer_dict
        self._transformer_previous_forward = transformer_dict.get("forward")
        self._scheduler_had_step = "step" in scheduler_dict
        self._scheduler_previous_step = scheduler_dict.get("step")
        self._original_forward = self.transformer.forward
        self._original_scheduler_step = self.scheduler.step
        try:
            self.transformer.forward = self._wrapped_forward
            self.scheduler.step = self._wrapped_scheduler_step
        except BaseException as install_error:
            rollback_errors: list[BaseException] = []
            for target, name, had_value, previous, original in (
                (
                    self.transformer,
                    "forward",
                    self._transformer_had_forward,
                    self._transformer_previous_forward,
                    self._original_forward,
                ),
                (
                    self.scheduler,
                    "step",
                    self._scheduler_had_step,
                    self._scheduler_previous_step,
                    self._original_scheduler_step,
                ),
            ):
                try:
                    _restore_runtime_method(
                        target,
                        name,
                        had_instance_value=had_value,
                        previous_instance_value=previous,
                        original_binding=original,
                    )
                except BaseException as rollback_error:
                    rollback_errors.append(rollback_error)
            if hasattr(install_error, "add_note"):
                for rollback_error in rollback_errors:
                    install_error.add_note(
                        "phase-response enter rollback also failed: "
                        f"{type(rollback_error).__name__}"
                    )
            raise
        self._entered = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if self._exit_attempted:
            self._completed = False
            self._cleanup_completed = False
            self._record = None
            self._candidate_evaluations = ()
            self._completion_sentinel = None
            raise RuntimeError("phase-response scope 不得重复退出")
        self._exit_attempted = True
        self._completed = False
        cleanup_errors: list[BaseException] = []
        for target, name, had_value, previous, original in (
            (
                self.transformer,
                "forward",
                self._transformer_had_forward,
                self._transformer_previous_forward,
                self._original_forward,
            ),
            (
                self.scheduler,
                "step",
                self._scheduler_had_step,
                self._scheduler_previous_step,
                self._original_scheduler_step,
            ),
        ):
            try:
                _restore_runtime_method(
                    target,
                    name,
                    had_instance_value=had_value,
                    previous_instance_value=previous,
                    original_binding=original,
                )
            except BaseException as error:
                cleanup_errors.append(error)
        if cleanup_errors:
            self._record = None
            self._candidate_evaluations = ()
            self._completion_sentinel = None
            if exc is not None and hasattr(exc, "add_note"):
                for cleanup_error in cleanup_errors:
                    exc.add_note(
                        "phase-response scope cleanup also failed: "
                        f"{type(cleanup_error).__name__}"
                    )
                if exc_type is not _PreflightComplete:
                    return False
            raise cleanup_errors[0] from exc
        self._cleanup_completed = True
        exact_completion = (
            exc_type is _PreflightComplete
            and exc is self._completion_sentinel
            and self._record is not None
            and len(self._candidate_evaluations) == 4
            and self._forward_calls == 12
            and self._scheduler_calls == 0
        )
        self._completion_sentinel = None
        if exact_completion:
            self._completed = True
            return True
        self._record = None
        self._candidate_evaluations = ()
        return False

    def batch(self, *, generator_state_digest_random: str) -> PhaseResponsePreflightBatch:
        if (
            not self._completed
            or not self._exit_attempted
            or not self._cleanup_completed
            or self._record is None
            or self._forward_calls != 12
            or self._scheduler_calls != 0
        ):
            raise RuntimeError("phase-response preflight batch 未完整完成")
        return PhaseResponsePreflightBatch(
            record=self._record,
            candidate_evaluations=self._candidate_evaluations,
            transformer_forward_count=self._forward_calls,
            scheduler_step_call_count=self._scheduler_calls,
            initial_hidden_state_digest_random=self._initial_hidden_digest,
            generator_state_digest_random=generator_state_digest_random,
        )

    def release_runtime_references(self) -> None:
        """Release runtime-heavy references after scope cleanup.

        This is idempotent and must run only after a completed/failed scope has
        restored the transformer and scheduler methods.
        """

        if self._entered and not self._exit_attempted:
            raise RuntimeError(
                "phase-response runtime references 不得在active scope释放"
            )
        self._branch_kwargs.clear()
        self._base.clear()
        self._record = None
        self._candidate_evaluations = ()
        self._completion_sentinel = None
        self._original_forward = None
        self._original_scheduler_step = None
        self._transformer_previous_forward = None
        self._scheduler_previous_step = None
        self.descriptor = None
        self.transformer = None
        self.scheduler = None


def execute_real_patch_relation_phase_response_preflight(
    config: Mapping[str, Any],
    gate0_config: Mapping[str, Any],
) -> PhaseResponsePreflightBatch:
    import torch

    from experiments.generative_video_model_probe.colab_runtime import (
        _generation_model_provenance_from_pipeline,
        _load_video_generation_pipeline,
        _scheduler_signature,
        _select_dtype,
    )
    from experiments.generative_video_model_probe.frozen_feedback_signed_response_diagnostic import (
        _tensor_digest,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("phase-response preflight 需要用户显式启动Colab CUDA")
    selected_dtype = _select_dtype(torch)
    _require_native_bfloat16_runtime(torch, selected_dtype=selected_dtype)
    common = gate0_config["protocol_contract"]["execution_identity_contract"][
        "execution_common"
    ]
    identity = _identity(gate0_config, "construction_c0")
    if _package_version("diffusers") != common["diffusers_version"]:
        raise RuntimeError("phase-response preflight diffusers 版本漂移")
    pipe = _load_video_generation_pipeline(
        common["model_id"],
        selected_dtype,
        revision=common["model_revision"],
    )
    adapter: ScopedPatchRelationPhaseResponsePreflight | None = None
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
            raise RuntimeError("phase-response model/scheduler provenance 漂移")
        if bool(getattr(pipe.transformer, "is_cache_enabled", False)):
            raise RuntimeError("phase-response preflight 禁止transformer cache")
        sigmas, deltas, timesteps = _frozen_schedule(gate0_config)
        generator = torch.Generator(device="cuda").manual_seed(
            int(identity["seed_value"])
        )
        generator_digest = _tensor_digest(generator.get_state())
        adapter = ScopedPatchRelationPhaseResponsePreflight(
            pipe.transformer,
            pipe.scheduler,
            descriptor=build_public_patch_relation_descriptor(),
            probe_id=config["protocol_contract"]["execution_binding"][
                "probe_id"
            ],
            sigma_grid=sigmas,
            delta_sigma_by_step=deltas,
            timestep_by_step=timesteps,
        )
        emit_progress_event(
            "patch_relation_phase_response_preflight",
            "step0 start; scheduler/decode/export disabled",
        )
        with torch.no_grad(), adapter:
            pipe(
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
        batch = adapter.batch(
            generator_state_digest_random=generator_digest,
        )
        emit_progress_event(
            "patch_relation_phase_response_preflight",
            "step0 finish; real_scheduler_step_call_count=0",
        )
        return batch
    finally:
        active_error = sys.exc_info()[1]
        cleanup_errors: list[tuple[str, BaseException]] = []
        if adapter is not None:
            adapter_to_release = adapter
            _attempt_runtime_cleanup(
                cleanup_errors,
                "adapter_release",
                adapter_to_release.release_runtime_references,
            )
            adapter = None
        maybe_free = getattr(pipe, "maybe_free_model_hooks", None)
        if callable(maybe_free):
            _attempt_runtime_cleanup(
                cleanup_errors,
                "pipeline_model_hook_cleanup",
                maybe_free,
            )
        del pipe
        _attempt_runtime_cleanup(
            cleanup_errors,
            "gc_collect",
            gc.collect,
        )
        _attempt_runtime_cleanup(
            cleanup_errors,
            "cuda_empty_cache",
            torch.cuda.empty_cache,
        )
        _finish_runtime_cleanup(active_error, cleanup_errors)


def _owned_success_staging_path(
    output: Path,
) -> tuple[Path, tuple[int, int], int]:
    parent = output.parent.resolve(strict=True)
    token = secrets.token_hex(16)
    if len(token) != 32 or any(
        character not in "0123456789abcdef" for character in token
    ):
        raise RuntimeError("phase-response staging token 非法")
    candidate = parent / (
        f".{output.name}.phase_response_success_staging.{token}"
    )
    candidate.mkdir(mode=0o700, parents=False, exist_ok=False)
    metadata = candidate.lstat()
    identity = (metadata.st_dev, metadata.st_ino)
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or candidate.parent != parent
        or candidate.resolve(strict=True).parent != parent
    ):
        current = candidate.lstat()
        if (
            not stat.S_ISLNK(current.st_mode)
            and stat.S_ISDIR(current.st_mode)
            and (current.st_dev, current.st_ino) == identity
        ):
            candidate.rmdir()
        raise RuntimeError("phase-response owned staging 路径边界非法")
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        candidate.rmdir()
        raise RuntimeError(
            "phase-response staging directory-fd no-follow capability 不可用"
        )
    try:
        directory_fd = os.open(
            candidate,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
    except BaseException as error:
        try:
            current = candidate.lstat()
            if (
                not stat.S_ISLNK(current.st_mode)
                and stat.S_ISDIR(current.st_mode)
                and (current.st_dev, current.st_ino) == identity
            ):
                candidate.rmdir()
        except BaseException as cleanup_error:
            if hasattr(error, "add_note"):
                error.add_note(
                    "phase-response staging open cleanup also failed: "
                    f"{type(cleanup_error).__name__}"
                )
        raise
    try:
        descriptor_metadata = os.fstat(directory_fd)
        if (
            not stat.S_ISDIR(descriptor_metadata.st_mode)
            or (
                descriptor_metadata.st_dev,
                descriptor_metadata.st_ino,
            )
            != identity
        ):
            raise RuntimeError(
                "phase-response staging directory-fd ownership 漂移"
            )
    except BaseException as error:
        try:
            os.close(directory_fd)
        except BaseException as cleanup_error:
            if hasattr(error, "add_note"):
                error.add_note(
                    "phase-response staging fd cleanup also failed: "
                    f"{type(cleanup_error).__name__}"
                )
        try:
            current = candidate.lstat()
            if (
                not stat.S_ISLNK(current.st_mode)
                and stat.S_ISDIR(current.st_mode)
                and (current.st_dev, current.st_ino) == identity
            ):
                candidate.rmdir()
        except BaseException as cleanup_error:
            if hasattr(error, "add_note"):
                error.add_note(
                    "phase-response staging path cleanup also failed: "
                    f"{type(cleanup_error).__name__}"
                )
        raise
    return candidate, identity, directory_fd


def _require_owned_success_staging(
    path: Path,
    *,
    expected_parent: Path,
    expected_identity: tuple[int, int],
) -> None:
    metadata = path.lstat()
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or (metadata.st_dev, metadata.st_ino) != expected_identity
        or path.parent != expected_parent
        or path.resolve(strict=True).parent != expected_parent
    ):
        raise RuntimeError("phase-response staging ownership 漂移")


def _cleanup_owned_success_staging(
    path: Path,
    *,
    expected_parent: Path,
    expected_identity: tuple[int, int],
) -> None:
    _require_owned_success_staging(
        path,
        expected_parent=expected_parent,
        expected_identity=expected_identity,
    )
    for child in path.iterdir():
        metadata = child.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(
            metadata.st_mode
        ):
            raise RuntimeError(
                "phase-response owned staging 出现非普通文件"
            )
        child.unlink()
    path.rmdir()


def _cleanup_success_artifacts_by_directory_fd(
    directory_fd: int,
    *,
    expected_identity: tuple[int, int],
) -> None:
    descriptor_metadata = os.fstat(directory_fd)
    if (
        not stat.S_ISDIR(descriptor_metadata.st_mode)
        or (
            descriptor_metadata.st_dev,
            descriptor_metadata.st_ino,
        )
        != expected_identity
    ):
        raise RuntimeError(
            "phase-response staging directory-fd ownership 漂移"
        )
    for filename in SUCCESS_ARTIFACT_FILENAMES:
        try:
            metadata = os.stat(
                filename,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            continue
        if stat.S_ISDIR(metadata.st_mode):
            raise RuntimeError(
                "phase-response success artifact 名称被目录占用"
            )
        os.unlink(filename, dir_fd=directory_fd)
    remaining_names = set(os.listdir(directory_fd))
    if remaining_names.intersection(SUCCESS_ARTIFACT_FILENAMES):
        raise RuntimeError(
            "phase-response directory-fd cleanup 后仍有success artifact"
        )


def _success_artifact_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_success_artifact_by_directory_fd(
    directory_fd: int,
    filename: str,
    payload: Mapping[str, Any],
) -> None:
    if filename not in SUCCESS_ARTIFACT_FILENAMES:
        raise ValueError("phase-response success artifact filename 非法")
    artifact_fd = os.open(
        filename,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
        dir_fd=directory_fd,
    )
    active_error: BaseException | None = None
    try:
        metadata = os.fstat(artifact_fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(
                "phase-response success artifact fd 必须是普通文件"
            )
        remaining = memoryview(_success_artifact_json_bytes(payload))
        while remaining:
            written = os.write(artifact_fd, remaining)
            if written <= 0:
                raise OSError("phase-response success artifact write 未推进")
            remaining = remaining[written:]
    except BaseException as error:
        active_error = error
        raise
    finally:
        try:
            os.close(artifact_fd)
        except BaseException as cleanup_error:
            if active_error is not None:
                if hasattr(active_error, "add_note"):
                    active_error.add_note(
                        "phase-response artifact fd close also failed: "
                        f"{type(cleanup_error).__name__}"
                    )
            else:
                raise


def _read_success_artifact_by_directory_fd(
    directory_fd: int,
    filename: str,
) -> dict[str, Any]:
    if filename not in SUCCESS_ARTIFACT_FILENAMES:
        raise ValueError("phase-response success artifact filename 非法")
    artifact_fd = os.open(
        filename,
        os.O_RDONLY | os.O_NOFOLLOW,
        dir_fd=directory_fd,
    )
    active_error: BaseException | None = None
    try:
        metadata = os.fstat(artifact_fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(
                "phase-response success artifact fd 必须是普通文件"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(artifact_fd, 65536)
            if not chunk:
                break
            chunks.append(chunk)
        observed = json.loads(b"".join(chunks).decode("utf-8"))
        if not isinstance(observed, dict):
            raise RuntimeError(
                "phase-response success artifact payload 必须是mapping"
            )
        return observed
    except BaseException as error:
        active_error = error
        raise
    finally:
        try:
            os.close(artifact_fd)
        except BaseException as cleanup_error:
            if active_error is not None:
                if hasattr(active_error, "add_note"):
                    active_error.add_note(
                        "phase-response artifact fd close also failed: "
                        f"{type(cleanup_error).__name__}"
                    )
            else:
                raise


def _readback_success_artifacts_by_directory_fd(
    directory_fd: int,
    *,
    expected_identity: tuple[int, int],
    success_artifacts: tuple[tuple[str, Mapping[str, Any]], ...],
) -> None:
    descriptor_metadata = os.fstat(directory_fd)
    if (
        not stat.S_ISDIR(descriptor_metadata.st_mode)
        or (
            descriptor_metadata.st_dev,
            descriptor_metadata.st_ino,
        )
        != expected_identity
    ):
        raise RuntimeError(
            "phase-response staging directory-fd ownership 漂移"
        )
    expected_names = {filename for filename, _ in success_artifacts}
    observed_names = set(os.listdir(directory_fd))
    if observed_names != expected_names:
        raise RuntimeError("phase-response success artifact coverage 漂移")
    for filename, payload in success_artifacts:
        observed = _read_success_artifact_by_directory_fd(
            directory_fd,
            filename,
        )
        if observed != payload:
            raise RuntimeError(
                "phase-response success artifact readback 漂移: "
                f"{filename}"
            )


def _require_promoted_success_output(
    output: Path,
    *,
    expected_parent: Path,
    expected_identity: tuple[int, int],
    directory_fd: int,
    success_artifacts: tuple[tuple[str, Mapping[str, Any]], ...],
) -> None:
    metadata = output.lstat()
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or (metadata.st_dev, metadata.st_ino) != expected_identity
        or output.parent != expected_parent
        or output.resolve(strict=True).parent != expected_parent
    ):
        raise RuntimeError("phase-response promoted output ownership 漂移")
    _readback_success_artifacts_by_directory_fd(
        directory_fd,
        expected_identity=expected_identity,
        success_artifacts=success_artifacts,
    )


def _rollback_promoted_success_output(
    output: Path,
    *,
    expected_parent: Path,
    expected_identity: tuple[int, int],
) -> None:
    metadata = output.lstat()
    if stat.S_ISLNK(metadata.st_mode):
        output.unlink()
        return
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or (metadata.st_dev, metadata.st_ino) != expected_identity
    ):
        raise RuntimeError(
            "phase-response promoted output 非本次owned目录，拒绝清理"
        )
    _cleanup_owned_success_staging(
        output,
        expected_parent=expected_parent,
        expected_identity=expected_identity,
    )


def run_patch_relation_phase_response_preflight(
    output_root: str | Path,
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    runtime_executor: Callable[
        [Mapping[str, Any], Mapping[str, Any]],
        PhaseResponsePreflightBatch,
    ] = execute_real_patch_relation_phase_response_preflight,
) -> dict[str, Any]:
    output = Path(output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"phase-response output root 必须为空: {output}")
    output.mkdir(parents=True, exist_ok=True)
    success_staging: Path | None = None
    success_staging_identity: tuple[int, int] | None = None
    success_staging_directory_fd: int | None = None
    success_promoted = False
    repository_commit = "unavailable"
    try:
        repository_commit = _repository_commit()
        config = load_patch_relation_phase_response_preflight_config(
            config_path
        )
        gate0_config = load_patch_relation_gate0_config(
            config["protocol_contract"]["execution_binding"][
                "gate0_config_path"
            ]
        )
        if (
            gate0_config["protocol_digest"]
            != config["protocol_contract"]["source_failure_boundary"][
                "gate0_protocol_digest"
            ]
        ):
            raise RuntimeError(
                "phase-response source Gate0 contract digest 漂移"
            )
        batch = runtime_executor(config, gate0_config)
        record = validate_phase_response_preflight_batch(batch)
        decision = {
            "record_version": RECORD_VERSION,
            "profile_id": PROFILE_ID,
            "phase_response_preflight_decision": (
                "single_step_diagnostic_completed_no_gate_decision"
            ),
            "diagnostic_classification": record[
                "diagnostic_classification"
            ],
            "diagnostic_candidates": record["diagnostic_candidates"],
            "transformer_forward_count": batch.transformer_forward_count,
            "real_scheduler_step_call_count": (
                batch.scheduler_step_call_count
            ),
            "decode_executed": False,
            "video_export_executed": False,
            "gate0_pass": False,
            "full_eight_video_rerun_allowed": False,
            "formal_result": False,
            "stage_progression_allowed": False,
            "observer_implementation_allowed": False,
            "paper_claim_allowed": False,
            "protocol_digest": config["protocol_digest"],
            "source_gate0_protocol_digest": gate0_config["protocol_digest"],
            "repository_commit": repository_commit,
            "claim_support_status": CLAIM_SUPPORT_STATUS,
        }
        manifest = {
            "manifest_kind": (
                "sstw_patch_relation_phase_response_preflight_manifest"
            ),
            **decision,
            "generator_state_digest_random": (
                batch.generator_state_digest_random
            ),
            "initial_hidden_state_digest_random": (
                batch.initial_hidden_state_digest_random
            ),
            "python_version": sys.version.split()[0],
            "torch_version": _package_version("torch"),
            "diffusers_version": _package_version("diffusers"),
            "result_is_gate_or_method_evidence": False,
        }
        (
            success_staging,
            success_staging_identity,
            success_staging_directory_fd,
        ) = _owned_success_staging_path(output)
        success_artifacts = (
            ("phase_response_preflight_record.json", record),
            (DECISION_FILENAME, decision),
            (MANIFEST_FILENAME, manifest),
        )
        for filename, payload in success_artifacts:
            _write_success_artifact_by_directory_fd(
                success_staging_directory_fd,
                filename,
                payload,
            )
        _readback_success_artifacts_by_directory_fd(
            success_staging_directory_fd,
            expected_identity=success_staging_identity,
            success_artifacts=success_artifacts,
        )
        _require_owned_success_staging(
            success_staging,
            expected_parent=output.parent,
            expected_identity=success_staging_identity,
        )
        output.rmdir()
        success_staging.replace(output)
        success_promoted = True
        _require_promoted_success_output(
            output,
            expected_parent=output.parent,
            expected_identity=success_staging_identity,
            directory_fd=success_staging_directory_fd,
            success_artifacts=success_artifacts,
        )
        os.close(success_staging_directory_fd)
        success_staging_directory_fd = None
        success_staging = None
        success_staging_identity = None
        success_promoted = False
        return decision
    except Exception as exc:
        cleanup_errors: list[BaseException] = []
        recovery_output_safe = True
        if (
            success_staging_directory_fd is not None
            and success_staging_identity is not None
        ):
            try:
                _cleanup_success_artifacts_by_directory_fd(
                    success_staging_directory_fd,
                    expected_identity=success_staging_identity,
                )
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        if success_promoted and success_staging_identity is not None:
            try:
                _rollback_promoted_success_output(
                    output,
                    expected_parent=output.parent,
                    expected_identity=success_staging_identity,
                )
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
                recovery_output_safe = False
        elif (
            success_staging is not None
            and success_staging_identity is not None
        ):
            try:
                _cleanup_owned_success_staging(
                    success_staging,
                    expected_parent=output.parent,
                    expected_identity=success_staging_identity,
                )
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        if success_staging_directory_fd is not None:
            try:
                os.close(success_staging_directory_fd)
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
            success_staging_directory_fd = None
        recovery_output_ready = recovery_output_safe
        if recovery_output_safe:
            try:
                output_metadata = output.lstat()
            except FileNotFoundError:
                output.mkdir(parents=True, exist_ok=False)
            else:
                if stat.S_ISLNK(output_metadata.st_mode) or not stat.S_ISDIR(
                    output_metadata.st_mode
                ):
                    recovery_output_ready = False
                    cleanup_errors.append(
                        RuntimeError(
                            "phase-response recovery output 路径不是普通目录"
                        )
                    )
        if recovery_output_ready:
            for filename in (
                "phase_response_preflight_record.json",
                MANIFEST_FILENAME,
                DECISION_FILENAME,
            ):
                path = output / filename
                if path.exists():
                    try:
                        path.unlink()
                    except BaseException as cleanup_error:
                        cleanup_errors.append(cleanup_error)
        if cleanup_errors and hasattr(exc, "add_note"):
            for cleanup_error in cleanup_errors:
                exc.add_note(
                    "phase-response success cleanup also failed: "
                    f"{type(cleanup_error).__name__}"
                )
        if recovery_output_ready:
            write_json(
                output / DECISION_FILENAME,
                {
                    "record_version": RECORD_VERSION,
                    "profile_id": PROFILE_ID,
                    "phase_response_preflight_decision": (
                        "runtime_or_contract_failure_recovery_only"
                    ),
                    "failure_reason": str(exc),
                    "gate0_pass": False,
                    "full_eight_video_rerun_allowed": False,
                    "formal_result": False,
                    "stage_progression_allowed": False,
                    "repository_commit": repository_commit,
                    "claim_support_status": (
                        "failure_recovery_only_not_claim_evidence"
                    ),
                },
            )
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-run-root", required=True)
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run_patch_relation_phase_response_preflight(
                arguments.output_run_root,
                config_path=arguments.config_path,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
