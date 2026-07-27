"""Run the minimal frame-state signed-observability Gate 0 construction.

The runner is deliberately limited to one public atom, two frozen identities,
and eight videos.  C0 constructs the public atom and saved-video transfer T0;
identity A is apply-only.  Every result remains non-formal and cannot advance
the project stage.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from hashlib import sha256
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from evaluation.protocol.frame_state_signed_observability_contract import (
    FrameStateProbePlanRecord,
    build_frame_state_probe_plan,
    build_public_context_record,
    canonical_json_digest,
    frame_state_probe_plan_digest,
    load_frame_state_signed_observability_config,
    validate_public_context_record,
)
from evaluation.protocol.record_writer import write_json, write_jsonl
from main.methods.state_space_watermark.frame_state_observability import (
    FEATURE_DIMENSION,
    LATENT_SHAPE,
    CheckpointResponses,
    FrameStateControlResult,
    GateZeroEvaluation,
    PublicFrameStateAtom,
    accumulate_actual_signed_exposure,
    apply_frame_state_control_numpy,
    build_flow_schedule,
    build_public_frame_state_atom,
    compute_signed_response_statistics,
    estimate_construction_t0,
    evaluate_gate_zero,
    extract_local_temporal_feature,
    final_latent_carrier_projection,
    read_public_frame_state_atom,
    write_public_frame_state_atom,
)
from runtime.core.progress import emit_progress_event


DEFAULT_CONFIG_PATH = (
    "configs/protocol/"
    "sstw_frame_state_signed_observability_construction.json"
)
TEST_ID = "frame_state_signed_observability_construction"
PROFILE_ID = "sstw_frame_state_signed_observability_construction"
PHASE = "gate0"
RECORD_VERSION = "frame_state_signed_observability_construction_v1"
CHECKPOINT_IDS = (
    "final_latent_carrier_projection",
    "decoded_local_temporal_feature",
    "saved_video_local_temporal_feature",
)
CHECKPOINT_SOURCE_BOUNDARIES = {
    "final_latent_carrier_projection": (
        "final_latent_float32_before_vae_decode"
    ),
    "decoded_local_temporal_feature": (
        "pre_save_postprocessed_float32_frames"
    ),
    "saved_video_local_temporal_feature": "saved_video_rgb24_readback",
}
IDENTITY_ROLES = (
    "construction_identity",
    "signed_observability_identity",
)
CLAIM_SUPPORT_STATUS = (
    "frame_state_gate0_construction_only_not_method_evidence"
)


@dataclass(frozen=True)
class ProbeMeasurement:
    """One real or controlled-fake probe returned by the runtime adapter."""

    plan_index: int
    identity_role: str
    probe_id: str
    signed_state_coefficient: int
    generator_state_digest_random: str
    public_context_digest: str
    video_path: str
    video_sha256: str
    final_latent_projection: np.ndarray
    decoded_feature: np.ndarray
    saved_video_feature: np.ndarray
    step_results: tuple[FrameStateControlResult, ...]
    actual_signed_exposure: float


@dataclass(frozen=True)
class FrameStateRuntimeBatch:
    """Complete eight-video runtime result before Gate 0 evaluation."""

    public_atom: PublicFrameStateAtom
    public_atom_path: str
    public_context_records: tuple[Mapping[str, Any], ...]
    measurements: tuple[ProbeMeasurement, ...]
    generation_records: tuple[Mapping[str, Any], ...]
    step_records: tuple[Mapping[str, Any], ...]
    checkpoint_records: tuple[Mapping[str, Any], ...]


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


def _require_authorized_config(config: Mapping[str, Any]) -> None:
    authorization = config["authorization_boundary"]
    required_true = (
        "runtime_implementation_authorized",
        "runner_implementation_allowed",
        "construction_execution_allowed",
        "gpu_execution_allowed",
        "colab_execution_allowed",
    )
    if any(authorization.get(field) is not True for field in required_true):
        raise RuntimeError("frame-state Gate 0 runner 尚未获最小执行授权")
    required_false = (
        "formal_result",
        "stage_progression_allowed",
        "observer_implementation_allowed",
        "attack_execution_allowed",
        "fixed_fpr_execution_allowed",
        "baseline_execution_allowed",
        "paper_claim_allowed",
    )
    if any(authorization.get(field) is not False for field in required_false):
        raise RuntimeError("frame-state Gate 0 禁止的授权边界发生漂移")


def _identity_for_role(
    config: Mapping[str, Any],
    identity_role: str,
) -> Mapping[str, Any]:
    if identity_role == "construction_identity":
        return config["execution_identity_contract"]["construction_identity"]
    if identity_role == "signed_observability_identity":
        return config["execution_identity_contract"][
            "signed_observability_identity"
        ]
    raise ValueError(f"未知 frame-state identity role: {identity_role}")


def _checkpoint_record(
    *,
    plan: FrameStateProbePlanRecord,
    checkpoint_id: str,
    values: np.ndarray,
    source_boundary: str,
) -> dict[str, Any]:
    numeric = np.asarray(values)
    expected_dimension = 1 if checkpoint_id == CHECKPOINT_IDS[0] else 528
    if (
        numeric.dtype != np.dtype("float64")
        or numeric.shape != (expected_dimension,)
        or not np.all(np.isfinite(numeric))
    ):
        raise ValueError(f"{plan.probe_id}/{checkpoint_id} checkpoint 不完整")
    if source_boundary != CHECKPOINT_SOURCE_BOUNDARIES.get(checkpoint_id):
        raise ValueError(f"{checkpoint_id} source boundary 不匹配")
    record = {
        "record_version": RECORD_VERSION,
        "profile_id": PROFILE_ID,
        "frame_state_probe_plan_index": plan.plan_index,
        "frame_state_identity_role": plan.identity_role,
        "frame_state_probe_id": plan.probe_id,
        "frame_state_checkpoint_id": checkpoint_id,
        "frame_state_checkpoint_dimension": expected_dimension,
        "frame_state_checkpoint_values": numeric.tolist(),
        "frame_state_checkpoint_source_boundary": source_boundary,
        "formal_result": False,
        "stage_progression_allowed": False,
        "claim_support_status": CLAIM_SUPPORT_STATUS,
    }
    record["frame_state_checkpoint_record_id"] = _stable_digest(record)
    return record


def _step_record(
    plan: FrameStateProbePlanRecord,
    result: FrameStateControlResult,
) -> dict[str, Any]:
    actual_digest = sha256(
        result.actual_delta_velocity.tobytes(order="C")
    ).hexdigest()
    record = {
        "record_version": RECORD_VERSION,
        "profile_id": PROFILE_ID,
        "frame_state_probe_plan_index": plan.plan_index,
        "frame_state_identity_role": plan.identity_role,
        "frame_state_probe_id": plan.probe_id,
        "frame_state_flow_step_index": result.step_index,
        "frame_state_delta_sigma": result.delta_sigma,
        "frame_state_waveform": result.waveform,
        "frame_state_actual_delta_sha256": actual_digest,
        "frame_state_actual_delta_norm": result.actual_delta_norm,
        "frame_state_intended_delta_norm": result.intended_delta_norm,
        "frame_state_joint_norm_budget": result.joint_norm_budget,
        "frame_state_remaining_energy": result.remaining_energy,
        "frame_state_energy_increment": result.energy_increment,
        "frame_state_direction_cosine": result.direction_cosine,
        "frame_state_projection_scale": result.projection_scale,
        "frame_state_projection_attempt_count": (
            result.projection_attempt_count
        ),
        "frame_state_projection_status": result.projection_status,
        "frame_state_inactive_exact_noop": result.inactive_exact_noop,
        "frame_state_signed_state_update_exposure": (
            result.signed_state_update_exposure
        ),
        "formal_result": False,
        "stage_progression_allowed": False,
        "claim_support_status": CLAIM_SUPPORT_STATUS,
    }
    record["frame_state_step_record_id"] = _stable_digest(record)
    return record


def _generation_record(
    *,
    config: Mapping[str, Any],
    plan: FrameStateProbePlanRecord,
    measurement: ProbeMeasurement,
    public_atom_digest: str,
    generation_runtime_sec: float,
) -> dict[str, Any]:
    identity = _identity_for_role(config, plan.identity_role)
    common = config["execution_identity_contract"]["execution_common"]
    record = {
        "record_version": RECORD_VERSION,
        "profile_id": PROFILE_ID,
        "frame_state_probe_plan_index": plan.plan_index,
        "frame_state_identity_role": plan.identity_role,
        "frame_state_probe_id": plan.probe_id,
        "frame_state_signed_state_coefficient": (
            plan.signed_state_coefficient
        ),
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
        "public_context_digest": measurement.public_context_digest,
        "public_dictionary_artifact_digest": public_atom_digest,
        "frame_state_actual_signed_exposure": (
            measurement.actual_signed_exposure
        ),
        "trajectory_step_count": len(measurement.step_results),
        "video_path": measurement.video_path,
        "video_sha256": measurement.video_sha256,
        "generation_runtime_sec": round(float(generation_runtime_sec), 3),
        "formal_result": False,
        "stage_progression_allowed": False,
        "claim_support_status": CLAIM_SUPPORT_STATUS,
    }
    return record


def _response_map(
    measurements: Sequence[ProbeMeasurement],
    *,
    identity_role: str,
) -> dict[str, CheckpointResponses]:
    selected = [
        value for value in measurements if value.identity_role == identity_role
    ]
    if len(selected) != 4:
        raise ValueError(f"{identity_role} 必须精确包含4项 measurement")
    values = {
        "final_latent_carrier_projection": [
            item.final_latent_projection for item in selected
        ],
        "decoded_local_temporal_feature": [
            item.decoded_feature for item in selected
        ],
        "saved_video_local_temporal_feature": [
            item.saved_video_feature for item in selected
        ],
    }
    return {
        checkpoint_id: CheckpointResponses(
            clean_a=rows[0],
            clean_b=rows[1],
            positive=rows[2],
            negative=rows[3],
        )
        for checkpoint_id, rows in values.items()
    }


def _statistics_record(
    identity_role: str,
    checkpoint_id: str,
    responses: CheckpointResponses,
) -> dict[str, Any]:
    statistics = compute_signed_response_statistics(responses)
    derived = (
        statistics.clean_noise_norm,
        statistics.odd_norm,
        statistics.common_norm,
        statistics.antisymmetry_cosine,
        statistics.antisymmetry_residual,
        statistics.common_odd_ratio,
        statistics.odd_clean_noise_ratio,
    )
    finite = all(math.isfinite(value) for value in derived)
    record = {
        "record_version": RECORD_VERSION,
        "profile_id": PROFILE_ID,
        "frame_state_identity_role": identity_role,
        "frame_state_checkpoint_id": checkpoint_id,
        "frame_state_clean_noise_norm": statistics.clean_noise_norm,
        "frame_state_odd_norm": statistics.odd_norm,
        "frame_state_common_norm": statistics.common_norm,
        "frame_state_antisymmetry_cosine": (
            statistics.antisymmetry_cosine if finite else None
        ),
        "frame_state_antisymmetry_residual": (
            statistics.antisymmetry_residual if finite else None
        ),
        "frame_state_common_odd_ratio": (
            statistics.common_odd_ratio if finite else None
        ),
        "frame_state_odd_clean_noise_ratio": (
            statistics.odd_clean_noise_ratio if finite else None
        ),
        "frame_state_statistics_finite": finite,
        "frame_state_signed_gate_ready": bool(
            finite
            and statistics.antisymmetry_cosine >= 0.9
            and statistics.antisymmetry_residual <= 0.25
            and statistics.common_odd_ratio <= 0.5
            and statistics.odd_clean_noise_ratio >= 3.0
        ),
        "formal_result": False,
        "stage_progression_allowed": False,
        "claim_support_status": CLAIM_SUPPORT_STATUS,
    }
    record["frame_state_statistics_record_id"] = _stable_digest(record)
    return record


def _validate_runtime_batch(
    config: Mapping[str, Any],
    plan: Sequence[FrameStateProbePlanRecord],
    batch: FrameStateRuntimeBatch,
) -> FrameStateRuntimeBatch:
    atom = read_public_frame_state_atom(batch.public_atom_path)
    if atom.array_digest != batch.public_atom.array_digest:
        raise ValueError("public atom artifact 与 runtime atom digest 不一致")
    if tuple(item.probe_id for item in batch.measurements) != tuple(
        item.probe_id for item in plan
    ):
        raise ValueError("runtime measurement probe identity/order 不一致")
    if len(batch.measurements) != 8 or len(batch.generation_records) != 8:
        raise ValueError("frame-state runtime 必须精确完成8个视频")
    if len(batch.step_records) != 64 or len(batch.checkpoint_records) != 24:
        raise ValueError("frame-state runtime step/checkpoint coverage 不完整")
    expected_checkpoint_order = tuple(
        (probe.probe_id, checkpoint_id)
        for probe in plan
        for checkpoint_id in CHECKPOINT_IDS
    )
    observed_checkpoint_order = tuple(
        (
            str(record.get("frame_state_probe_id") or ""),
            str(record.get("frame_state_checkpoint_id") or ""),
        )
        for record in batch.checkpoint_records
    )
    if observed_checkpoint_order != expected_checkpoint_order:
        raise ValueError("frame-state checkpoint identity/order 不一致")

    expected_context_roles = IDENTITY_ROLES
    observed_context_roles = tuple(
        str(record.get("frame_state_identity_role") or "")
        for record in batch.public_context_records
    )
    if observed_context_roles != expected_context_roles:
        raise ValueError("public context 必须精确覆盖 C0/A 且顺序固定")
    context_digest_by_role: dict[str, str] = {}
    for wrapper in batch.public_context_records:
        role = str(wrapper["frame_state_identity_role"])
        context = wrapper["public_context_record"]
        digest = validate_public_context_record(context, config)
        if digest != wrapper.get("context_digest"):
            raise ValueError("public context wrapper digest 不一致")
        context_digest_by_role[role] = digest
    if len(set(context_digest_by_role.values())) != 2:
        raise ValueError("C0/A public context 不得复用同一 nonce")

    schedule = build_flow_schedule()
    generation_by_probe = {
        str(record.get("frame_state_probe_id") or ""): record
        for record in batch.generation_records
    }
    for expected, measurement in zip(plan, batch.measurements, strict=True):
        generation = generation_by_probe.get(expected.probe_id)
        if generation is None or generation.get("generation_status") != "success":
            raise ValueError(f"{expected.probe_id} generation 未成功")
        if (
            measurement.plan_index != expected.plan_index
            or measurement.identity_role != expected.identity_role
            or measurement.signed_state_coefficient
            != expected.signed_state_coefficient
            or measurement.public_context_digest
            != context_digest_by_role[expected.identity_role]
            or not measurement.generator_state_digest_random
            or len(measurement.step_results) != 8
        ):
            raise ValueError(f"{expected.probe_id} runtime identity 未完整绑定")
        video_path = Path(measurement.video_path)
        if (
            not video_path.is_file()
            or _sha256_file(video_path) != measurement.video_sha256
        ):
            raise ValueError(f"{expected.probe_id} video artifact 不完整")
        recomputed_exposure = accumulate_actual_signed_exposure(
            measurement.step_results,
            batch.public_atom.values,
        )
        if recomputed_exposure != measurement.actual_signed_exposure:
            raise ValueError(f"{expected.probe_id} actual exposure 不一致")
        for step_result, schedule_step in zip(
            measurement.step_results,
            schedule,
            strict=True,
        ):
            if step_result.step_index != schedule_step.step_index:
                raise ValueError("runtime step order 与冻结 schedule 不一致")
        expected_values = (
            measurement.final_latent_projection,
            measurement.decoded_feature,
            measurement.saved_video_feature,
        )
        records = batch.checkpoint_records[
            expected.plan_index * 3 : expected.plan_index * 3 + 3
        ]
        for checkpoint_id, values, record in zip(
            CHECKPOINT_IDS,
            expected_values,
            records,
            strict=True,
        ):
            expected_record = _checkpoint_record(
                plan=expected,
                checkpoint_id=checkpoint_id,
                values=values,
                source_boundary=CHECKPOINT_SOURCE_BOUNDARIES[checkpoint_id],
            )
            if record != expected_record:
                raise ValueError(
                    f"{expected.probe_id}/{checkpoint_id} governed record 不一致"
                )
        expected_steps = tuple(
            _step_record(expected, result)
            for result in measurement.step_results
        )
        observed_steps = batch.step_records[
            expected.plan_index * 8 : expected.plan_index * 8 + 8
        ]
        if tuple(observed_steps) != expected_steps:
            raise ValueError(f"{expected.probe_id} governed step records 不一致")
        expected_generation = _generation_record(
            config=config,
            plan=expected,
            measurement=measurement,
            public_atom_digest=batch.public_atom.array_digest,
            generation_runtime_sec=float(
                generation.get("generation_runtime_sec", 0.0)
            ),
        )
        expected_generation["frame_state_generation_record_id"] = (
            _stable_digest(expected_generation)
        )
        if generation != expected_generation:
            raise ValueError(
                f"{expected.probe_id} governed generation record 不一致"
            )
    generator_by_role = {
        role: {
            item.generator_state_digest_random
            for item in batch.measurements
            if item.identity_role == role
        }
        for role in IDENTITY_ROLES
    }
    if any(len(values) != 1 for values in generator_by_role.values()):
        raise ValueError("同一 identity 四项必须重建相同 generator 初态")
    if generator_by_role[IDENTITY_ROLES[0]] == generator_by_role[
        IDENTITY_ROLES[1]
    ]:
        raise ValueError("C0/A 不同 seed 的 generator 初态摘要不得相同")
    return batch


def torch_jacobian_gram_product(
    feature_function: Callable[[Any], Any],
    reference: Any,
    direction: Any,
    *,
    torch_module: Any,
    phase_observer: Callable[[str], None] | None = None,
    phase_cleanup: Callable[[str], None] | None = None,
) -> Any:
    """Compute exact J^T Jv with true forward AD and an offloaded VJP."""

    if reference.dtype != torch_module.float32 or direction.dtype != (
        torch_module.float32
    ):
        raise ValueError("Jacobian reference/direction 必须精确为 float32")
    if reference.shape != direction.shape:
        raise ValueError("Jacobian reference/direction shape 不一致")
    if getattr(reference, "device", None) != getattr(direction, "device", None):
        raise ValueError("Jacobian reference/direction device 不一致")
    if not callable(getattr(getattr(torch_module, "func", None), "jvp", None)):
        raise RuntimeError(
            "frame-state public atom phase=true_forward_jvp "
            "error_type=UnsupportedForwardAD"
        )
    autograd = getattr(torch_module, "autograd", None)
    graph = getattr(autograd, "graph", None)
    if not callable(getattr(autograd, "grad", None)) or not callable(
        getattr(graph, "save_on_cpu", None)
    ):
        raise RuntimeError(
            "frame-state public atom phase=reverse_vjp_saved_tensor_cpu_offload "
            "error_type=UnsupportedSavedTensorOffload"
        )

    def observe(phase: str) -> None:
        if phase_observer is not None:
            phase_observer(phase)

    def cleanup(phase: str) -> None:
        if phase_cleanup is not None:
            phase_cleanup(phase)

    def require_finite_float32(value: Any, *, label: str) -> None:
        if value.dtype != torch_module.float32:
            raise RuntimeError(f"{label} dtype 漂移")
        try:
            finite = bool(torch_module.isfinite(value).all().item())
        except Exception as exc:
            raise RuntimeError(f"{label} finite 检查失败") from exc
        if not finite:
            raise RuntimeError(f"{label} 包含非有限值")

    require_finite_float32(reference, label="Jacobian reference")
    require_finite_float32(direction, label="Jacobian direction")
    primal = None
    tangent = None
    tangent_for_vjp = None
    leaf = None
    feature = None
    product = None
    returned_product = None
    try:
        cleanup("jvp_pre")
        jvp_error: BaseException | None = None
        try:
            with torch_module.no_grad():
                observe("jvp_start")
                primal, tangent = torch_module.func.jvp(
                    feature_function,
                    (reference.detach(),),
                    (direction.detach(),),
                    strict=True,
                )
                observe("jvp_finish")
            if primal.shape != tangent.shape:
                raise RuntimeError(
                    "true forward JVP primal/tangent shape 不一致"
                )
            require_finite_float32(
                primal,
                label="true forward JVP primal",
            )
            require_finite_float32(
                tangent,
                label="true forward JVP tangent",
            )
            tangent_for_vjp = tangent.detach()
        except BaseException as exc:
            wrapped_error = RuntimeError(
                "frame-state public atom phase=true_forward_jvp "
                f"error_type={type(exc).__name__}"
            )
            jvp_error = wrapped_error
            raise wrapped_error from exc
        finally:
            primal = None
            tangent = None
            try:
                cleanup("jvp_post")
            except BaseException as cleanup_error:
                if jvp_error is not None:
                    jvp_error.add_note(
                        "jvp_post cleanup error_type="
                        f"{type(cleanup_error).__name__}"
                    )
                else:
                    raise

        cleanup("vjp_pre")
        vjp_error: BaseException | None = None
        try:
            leaf = reference.detach().clone().requires_grad_(True)
            with torch_module.enable_grad():
                observe("vjp_start")
                with graph.save_on_cpu(pin_memory=False):
                    feature = feature_function(leaf)
                require_finite_float32(feature, label="reverse VJP feature")
                if feature.shape != tangent_for_vjp.shape:
                    raise RuntimeError(
                        "reverse VJP feature/tangent shape 不一致"
                    )
                (product,) = autograd.grad(
                    outputs=feature,
                    inputs=leaf,
                    grad_outputs=tangent_for_vjp,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )
                observe("vjp_finish")
            if product.shape != reference.shape:
                raise RuntimeError("Jacobian Gram product shape 漂移")
            require_finite_float32(product, label="Jacobian Gram product")
            if getattr(product, "grad_fn", None) is not None:
                raise RuntimeError(
                    "Jacobian Gram product 不得保留 autograd graph"
                )
            if getattr(leaf, "grad", None) is not None:
                raise RuntimeError(
                    "autograd.grad 不得写入 reference leaf.grad"
                )
            returned_product = product.detach()
        except BaseException as exc:
            wrapped_error = RuntimeError(
                "frame-state public atom "
                "phase=reverse_vjp_saved_tensor_cpu_offload "
                f"error_type={type(exc).__name__}"
            )
            vjp_error = wrapped_error
            raise wrapped_error from exc
        finally:
            tangent_for_vjp = None
            leaf = None
            feature = None
            product = None
            try:
                cleanup("vjp_post")
            except BaseException as cleanup_error:
                if vjp_error is not None:
                    vjp_error.add_note(
                        "vjp_post cleanup error_type="
                        f"{type(cleanup_error).__name__}"
                    )
                else:
                    raise
        return returned_product
    finally:
        primal = None
        tangent = None
        tangent_for_vjp = None
        leaf = None
        feature = None
        product = None


def _torch_local_temporal_surrogate(decoded: Any) -> Any:
    """Mirror the frozen [0,1] postprocess and 11x4x4xRGB flattening."""

    import torch

    postprocessed = (decoded / 2.0 + 0.5).clamp(0.0, 1.0)
    frames = postprocessed.permute(0, 2, 3, 4, 1)
    values = []
    for frame_index in range(11, 22):
        frame = frames[:, frame_index]
        for row in range(4):
            row_start, row_end = row * 320 // 4, (row + 1) * 320 // 4
            for column in range(4):
                col_start = column * 512 // 4
                col_end = (column + 1) * 512 // 4
                values.append(
                    frame[
                        :,
                        row_start:row_end,
                        col_start:col_end,
                        :,
                    ].mean(dim=(1, 2))
                )
    return torch.stack(values, dim=1).reshape(-1)


def _cuda_public_atom_progress_fields(torch_module: Any) -> str:
    """Return scalar-only CUDA memory fields for non-governed progress."""

    gib = float(1024**3)
    host_fields = _host_memory_progress_fields()
    try:
        allocated = float(torch_module.cuda.memory_allocated()) / gib
        reserved = float(torch_module.cuda.memory_reserved()) / gib
        maximum = float(torch_module.cuda.max_memory_allocated()) / gib
        free_bytes, total_bytes = torch_module.cuda.mem_get_info()
        return (
            f"allocated_gib={allocated:.3f} "
            f"reserved_gib={reserved:.3f} "
            f"max_allocated_gib={maximum:.3f} "
            f"free_gib={float(free_bytes) / gib:.3f} "
            f"total_gib={float(total_bytes) / gib:.3f} "
            f"grad_enabled={str(torch_module.is_grad_enabled()).lower()} "
            f"{host_fields}"
        )
    except Exception as exc:
        return (
            "cuda_memory_status=unavailable "
            f"cuda_memory_error_type={type(exc).__name__} "
            f"grad_enabled={str(torch_module.is_grad_enabled()).lower()} "
            f"{host_fields}"
        )


def _host_memory_progress_fields() -> str:
    """Read process RSS and host availability from Linux procfs."""

    try:
        status_fields: dict[str, int] = {}
        for line in Path("/proc/self/status").read_text(
            encoding="utf-8",
        ).splitlines():
            if line.startswith("VmRSS:"):
                status_fields["rss_kib"] = int(line.split()[1])
                break
        memory_fields: dict[str, int] = {}
        for line in Path("/proc/meminfo").read_text(
            encoding="utf-8",
        ).splitlines():
            if line.startswith("MemAvailable:"):
                memory_fields["available_kib"] = int(line.split()[1])
                break
        if set(status_fields) != {"rss_kib"} or set(memory_fields) != {
            "available_kib"
        }:
            raise ValueError("procfs memory fields missing")
        kib_per_gib = float(1024**2)
        return (
            f"process_rss_gib={status_fields['rss_kib'] / kib_per_gib:.3f} "
            "host_available_gib="
            f"{memory_fields['available_kib'] / kib_per_gib:.3f}"
        )
    except Exception as exc:
        return (
            "host_memory_status=unavailable "
            f"host_memory_error_type={type(exc).__name__}"
        )


def _emit_public_atom_progress(
    torch_module: Any,
    *,
    iteration_index: int,
    phase: str,
) -> None:
    if phase == "jvp_start":
        try:
            torch_module.cuda.reset_peak_memory_stats()
        except Exception:
            pass
    try:
        emit_progress_event(
            "frame_state_public_atom_construction",
            (
                f"iteration={iteration_index}/8 phase={phase} "
                f"{_cuda_public_atom_progress_fields(torch_module)}"
            ),
        )
    except Exception:
        # Progress is deliberately non-governed and cannot alter the
        # mathematical execution or mask its failure.
        pass


def _synchronize_release_cuda(torch_module: Any) -> None:
    """Finish pending CUDA work, then release unreachable cached storage."""

    torch_module.cuda.synchronize()
    gc.collect()
    torch_module.cuda.empty_cache()


def _clear_wan_vae_cache(vae: Any, *, phase: str) -> None:
    clear_cache = getattr(vae, "clear_cache", None)
    if not callable(clear_cache):
        raise RuntimeError(
            f"frame-state public atom phase={phase} "
            "error_type=MissingWanVaeClearCache"
        )
    try:
        clear_cache()
    except Exception as exc:
        raise RuntimeError(
            f"frame-state public atom phase={phase} "
            f"error_type={type(exc).__name__}"
        ) from exc


def _clear_wan_vae_phase_and_release(
    vae: Any,
    torch_module: Any,
    *,
    phase: str,
) -> None:
    """Clear causal caches and release unreachable CUDA storage."""

    cleanup_error: Exception | None = None
    try:
        _clear_wan_vae_cache(vae, phase=phase)
    except Exception as exc:
        cleanup_error = exc
    try:
        _synchronize_release_cuda(torch_module)
    except Exception as exc:
        cleanup_error = cleanup_error or exc
    if cleanup_error is not None:
        raise cleanup_error


@contextmanager
def _wan_jacobian_math_sdpa(torch_module: Any):
    """Limit only the Jacobian boundary to forward-AD-compatible math SDPA."""

    attention = getattr(getattr(torch_module, "nn", None), "attention", None)
    kernel = getattr(attention, "sdpa_kernel", None)
    backend_type = getattr(attention, "SDPBackend", None)
    math_backend = getattr(backend_type, "MATH", None)
    if not callable(kernel) or math_backend is None:
        raise RuntimeError(
            "frame-state public atom phase=math_sdpa_context "
            "error_type=UnsupportedMathSdpaContext"
        )
    try:
        manager = kernel(math_backend)
    except Exception as exc:
        raise RuntimeError(
            "frame-state public atom phase=math_sdpa_context "
            f"error_type={type(exc).__name__}"
        ) from exc
    try:
        manager.__enter__()
    except Exception as exc:
        raise RuntimeError(
            "frame-state public atom phase=math_sdpa_context "
            f"error_type={type(exc).__name__}"
        ) from exc
    active_error: BaseException | None = None
    exit_arguments: tuple[Any, Any, Any] = (None, None, None)
    try:
        yield
    except BaseException as exc:
        active_error = exc
        exit_arguments = (type(exc), exc, exc.__traceback__)
        raise
    finally:
        try:
            manager.__exit__(*exit_arguments)
        except BaseException as cleanup_error:
            if active_error is not None:
                active_error.add_note(
                    "math SDPA restore error_type="
                    f"{type(cleanup_error).__name__}"
                )
            else:
                raise


@contextmanager
def _wan_vae_eval_frozen_parameters(vae: Any):
    """Freeze VAE parameters for input-only AD and restore all prior state."""

    parameters = tuple(vae.parameters())
    prior_training = bool(vae.training)
    prior_requires_grad = tuple(
        bool(parameter.requires_grad) for parameter in parameters
    )
    if any(getattr(parameter, "grad", None) is not None for parameter in parameters):
        raise RuntimeError("Wan VAE parameter.grad 必须在 Jacobian 前为空")
    vae.eval()
    for parameter in parameters:
        parameter.requires_grad_(False)
    if bool(vae.training) or any(
        bool(parameter.requires_grad) for parameter in parameters
    ):
        raise RuntimeError("Wan VAE eval/frozen parameter 边界未建立")
    active_error: BaseException | None = None
    try:
        yield
    except BaseException as exc:
        active_error = exc
        raise
    finally:
        cleanup_error: BaseException | None = None
        if any(
            getattr(parameter, "grad", None) is not None
            for parameter in parameters
        ):
            cleanup_error = RuntimeError(
                "Wan VAE Jacobian 不得产生 parameter.grad"
            )
        try:
            for parameter, requires_grad in zip(
                parameters,
                prior_requires_grad,
                strict=True,
            ):
                parameter.requires_grad_(requires_grad)
            vae.train(prior_training)
            if tuple(
                bool(parameter.requires_grad) for parameter in parameters
            ) != prior_requires_grad or bool(vae.training) is not prior_training:
                raise RuntimeError(
                    "Wan VAE parameter/training 状态恢复失败"
                )
        except BaseException as exc:
            cleanup_error = cleanup_error or exc
        if cleanup_error is not None:
            if active_error is not None:
                active_error.add_note(
                    "VAE parameter/training cleanup error_type="
                    f"{type(cleanup_error).__name__}"
                )
            else:
                raise cleanup_error


@contextmanager
def _wan_pipeline_public_atom_offload_state(pipe: Any):
    """Remove dispatch hooks only for atom AD, then restore CPU offload."""

    remove_hooks = getattr(pipe, "remove_all_hooks", None)
    enable_offload = getattr(pipe, "enable_model_cpu_offload", None)
    if not callable(remove_hooks) or not callable(enable_offload):
        raise RuntimeError(
            "frame-state public atom 缺少 pipeline offload state API"
        )
    remove_hooks()
    active_error: BaseException | None = None
    try:
        yield
    except BaseException as exc:
        active_error = exc
        raise
    finally:
        try:
            enable_offload()
        except BaseException as cleanup_error:
            if active_error is not None:
                active_error.add_note(
                    "pipeline offload restore error_type="
                    f"{type(cleanup_error).__name__}"
                )
            else:
                raise


@contextmanager
def _wan_jacobian_untiled_decode(vae: Any):
    """Freeze one untiled Jacobian decode and restore the VAE memory policy."""

    required_attributes = (
        "use_tiling",
        "tile_sample_min_height",
        "tile_sample_min_width",
        "tile_sample_stride_height",
        "tile_sample_stride_width",
    )
    if any(not hasattr(vae, name) for name in required_attributes):
        raise RuntimeError("Wan VAE 缺少冻结 Jacobian tiling 状态")
    if not hasattr(vae, "disable_tiling") or not hasattr(vae, "enable_tiling"):
        raise RuntimeError("Wan VAE 缺少冻结 tiling API")
    prior_enabled = bool(vae.use_tiling)
    prior_parameters = {
        name: int(getattr(vae, name))
        for name in required_attributes[1:]
    }
    vae.disable_tiling()
    if bool(vae.use_tiling):
        raise RuntimeError("Wan VAE Jacobian spatial tiling 未关闭")
    active_error: BaseException | None = None
    try:
        yield
    except BaseException as exc:
        active_error = exc
        raise
    finally:
        cleanup_error: BaseException | None = None
        try:
            if prior_enabled:
                vae.enable_tiling(**prior_parameters)
            else:
                vae.disable_tiling()
            if bool(vae.use_tiling) is not prior_enabled:
                raise RuntimeError("Wan VAE Jacobian tiling 状态恢复失败")
            for name, value in prior_parameters.items():
                if int(getattr(vae, name)) != value:
                    raise RuntimeError(
                        "Wan VAE Jacobian tiling 参数恢复失败"
                    )
        except BaseException as exc:
            cleanup_error = exc
        if cleanup_error is not None:
            if active_error is not None:
                active_error.add_note(
                    "VAE tiling restore error_type="
                    f"{type(cleanup_error).__name__}"
                )
            else:
                raise cleanup_error


def _construct_real_public_atom(
    pipe: Any,
    clean_final_latent: Any,
) -> PublicFrameStateAtom:
    """Build the sole public atom from C0 clean-A and the frozen FP32 VAE."""

    import torch

    if _package_version("diffusers") != "0.35.2":
        raise RuntimeError("frame-state public atom 要求 diffusers==0.35.2")
    vae = pipe.vae
    reference = None
    mean = None
    std = None
    feature_function = None
    callback = None
    active_error: BaseException | None = None
    with _wan_pipeline_public_atom_offload_state(pipe):
        try:
            vae.to(device="cuda", dtype=torch.float32)
            reference = clean_final_latent.detach().to(
                device="cuda",
                dtype=torch.float32,
            )
            if tuple(reference.shape) != LATENT_SHAPE:
                raise RuntimeError("C0 clean-A final latent shape 不匹配")
            mean = torch.tensor(
                vae.config.latents_mean,
                dtype=torch.float32,
                device=reference.device,
            ).view(1, vae.config.z_dim, 1, 1, 1)
            std = torch.tensor(
                vae.config.latents_std,
                dtype=torch.float32,
                device=reference.device,
            ).view(1, vae.config.z_dim, 1, 1, 1)

            def feature_function(latent: Any) -> Any:
                decoded = vae.decode(
                    latent * std + mean,
                    return_dict=False,
                )[0]
                return _torch_local_temporal_surrogate(decoded)

            iteration_index = 0

            def callback(direction: np.ndarray) -> np.ndarray:
                nonlocal iteration_index
                iteration_index += 1
                if iteration_index > 8:
                    raise RuntimeError("public atom iteration 超过冻结8次")
                direction_tensor = torch.from_numpy(direction).to(
                    device=reference.device,
                    dtype=torch.float32,
                )
                product = None
                active_callback_error: BaseException | None = None
                try:
                    product = torch_jacobian_gram_product(
                        feature_function,
                        reference,
                        direction_tensor,
                        torch_module=torch,
                        phase_observer=lambda phase: (
                            _emit_public_atom_progress(
                                torch,
                                iteration_index=iteration_index,
                                phase=phase,
                            )
                        ),
                        phase_cleanup=lambda phase: (
                            _clear_wan_vae_phase_and_release(
                                vae,
                                torch,
                                phase=f"{phase}_cache_clear",
                            )
                        ),
                    )
                    return np.ascontiguousarray(
                        product.detach().cpu().numpy(),
                        dtype=np.float32,
                    )
                except BaseException as exc:
                    active_callback_error = exc
                    raise
                finally:
                    direction_tensor = None
                    product = None
                    cleanup_error: BaseException | None = None
                    try:
                        _clear_wan_vae_phase_and_release(
                            vae,
                            torch,
                            phase="iteration_cleanup_cache_clear",
                        )
                    except BaseException as exc:
                        cleanup_error = exc
                    _emit_public_atom_progress(
                        torch,
                        iteration_index=iteration_index,
                        phase="cleanup_finish",
                    )
                    if cleanup_error is not None:
                        if active_callback_error is not None:
                            active_callback_error.add_note(
                                "iteration cleanup error_type="
                                f"{type(cleanup_error).__name__}"
                            )
                        else:
                            raise cleanup_error

            with _wan_vae_eval_frozen_parameters(vae):
                with _wan_jacobian_untiled_decode(vae):
                    with _wan_jacobian_math_sdpa(torch):
                        atom = build_public_frame_state_atom(callback)
            if iteration_index != 8:
                raise RuntimeError("public atom 未精确完成冻结8次 iteration")
            return atom
        except BaseException as exc:
            active_error = exc
            raise
        finally:
            feature_function = None
            callback = None
            reference = None
            mean = None
            std = None
            cleanup_error: Exception | None = None
            try:
                _clear_wan_vae_cache(
                    vae,
                    phase="public_atom_outer_finally_cache_clear",
                )
            except Exception as exc:
                cleanup_error = exc
            try:
                vae.to("cpu")
            except Exception as exc:
                cleanup_error = cleanup_error or exc
            try:
                _synchronize_release_cuda(torch)
            except Exception as exc:
                cleanup_error = cleanup_error or exc
            if cleanup_error is not None:
                if active_error is not None and hasattr(
                    active_error,
                    "add_note",
                ):
                    active_error.add_note(
                        "public atom cleanup error_type="
                        f"{type(cleanup_error).__name__}"
                    )
                elif active_error is None:
                    raise cleanup_error


class FrameStateSchedulerAdapter:
    """Patch one real FlowMatch scheduler and call the audited NumPy core."""

    def __init__(
        self,
        scheduler: Any,
        *,
        plan: FrameStateProbePlanRecord,
        atom: PublicFrameStateAtom,
    ) -> None:
        self.scheduler = scheduler
        self.plan = plan
        self.atom = atom
        self.results: list[FrameStateControlResult] = []
        self.records: list[dict[str, Any]] = []
        self.final_latent: Any | None = None
        self._original_step: Any | None = None
        self._cumulative_control_energy = 0.0
        self._cumulative_reference_energy = 0.0
        self._schedule_validated = False

    def __enter__(self) -> "FrameStateSchedulerAdapter":
        self._original_step = self.scheduler.step
        def step(
            model_output: Any,
            timestep: Any,
            sample: Any,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            return self._run_step(
                model_output,
                timestep,
                sample,
                *args,
                **kwargs,
            )

        self.scheduler.step = step
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._original_step is not None:
            self.scheduler.step = self._original_step

    def _run_step(
        self,
        model_output: Any,
        timestep: Any,
        sample: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        import torch

        if self._original_step is None or len(self.results) >= 8:
            raise RuntimeError("frame-state scheduler adapter 状态不合法")
        if not self._schedule_validated:
            observed_sigmas = tuple(
                float(value.detach().float().cpu().item())
                if hasattr(value, "detach")
                else float(value)
                for value in self.scheduler.sigmas
            )
            build_flow_schedule(observed_sigmas)
            self._schedule_validated = True
        step = build_flow_schedule()[len(self.results)]
        base = np.ascontiguousarray(
            model_output.detach().float().cpu().numpy(),
            dtype=np.float32,
        )
        result = apply_frame_state_control_numpy(
            base,
            self.atom.values,
            signed_state_coefficient=self.plan.signed_state_coefficient,
            step=step,
            cumulative_control_energy=self._cumulative_control_energy,
            cumulative_reference_energy=self._cumulative_reference_energy,
            remaining_step_count=8 - step.step_index,
        )
        base_norm = float(np.linalg.norm(base.reshape(-1)))
        reference_increment = step.delta_sigma**2 * base_norm**2
        self._cumulative_reference_energy += reference_increment
        self._cumulative_control_energy += result.energy_increment
        constrained = torch.from_numpy(result.constrained_velocity).to(
            device=model_output.device,
            dtype=torch.float32,
        )
        self.results.append(result)
        self.records.append(_step_record(self.plan, result))
        scheduler_output = self._original_step(
            constrained,
            timestep,
            sample,
            *args,
            **kwargs,
        )
        previous_sample = (
            scheduler_output[0]
            if isinstance(scheduler_output, tuple)
            else getattr(scheduler_output, "prev_sample", None)
        )
        if previous_sample is None:
            raise RuntimeError("FlowMatch scheduler output 缺少 prev_sample")
        self.final_latent = previous_sample.detach().float().cpu()
        return scheduler_output


def execute_real_frame_state_gate0(
    config: Mapping[str, Any],
    plan: Sequence[FrameStateProbePlanRecord],
    output_root: Path,
    *,
    nonce_factory: Callable[[], str] = lambda: secrets.token_hex(16),
    atom_constructor: Callable[[Any, Any], PublicFrameStateAtom] = (
        _construct_real_public_atom
    ),
) -> FrameStateRuntimeBatch:
    """Execute the real eight-video Wan Gate 0 plan."""

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
        raise RuntimeError("frame-state Gate 0 需要可用 Colab CUDA GPU")
    common = config["execution_identity_contract"]["execution_common"]
    if _package_version("diffusers") != common["diffusers_version"]:
        raise RuntimeError("frame-state Gate 0 diffusers runtime 版本漂移")
    pipe = _load_video_generation_pipeline(
        common["model_id"],
        _select_dtype(torch),
        revision=common["model_revision"],
    )
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
        raise RuntimeError("frame-state model/scheduler provenance 未冻结")

    context_wrappers: list[dict[str, Any]] = []
    context_digest_by_role: dict[str, str] = {}
    for role in IDENTITY_ROLES:
        context = build_public_context_record(
            config,
            public_nonce_random=nonce_factory(),
        )
        digest = validate_public_context_record(context, config)
        context_digest_by_role[role] = digest
        context_wrappers.append(
            {
                "frame_state_identity_role": role,
                "public_context_record": context,
                "context_digest": digest,
            }
        )
    if len(set(context_digest_by_role.values())) != 2:
        raise RuntimeError("C0/A public nonce 必须独立")

    bootstrap_atom = build_public_frame_state_atom(lambda value: value)
    public_atom: PublicFrameStateAtom | None = None
    final_latents: dict[str, Any] = {}
    step_results: dict[str, tuple[FrameStateControlResult, ...]] = {}
    step_records: list[Mapping[str, Any]] = []
    generator_digests: dict[str, str] = {}
    generation_runtime_by_probe: dict[str, float] = {}

    try:
        for probe in plan:
            identity = _identity_for_role(config, probe.identity_role)
            generator = torch.Generator(device="cuda").manual_seed(
                int(identity["seed_value"])
            )
            generator_digests[probe.probe_id] = _tensor_digest(
                generator.get_state()
            )
            started = time.time()
            atom = public_atom or bootstrap_atom
            if probe.signed_state_coefficient != 0 and public_atom is None:
                raise RuntimeError("signed probe 不得先于 public atom construction")
            adapter = FrameStateSchedulerAdapter(
                pipe.scheduler,
                plan=probe,
                atom=atom,
            )
            with adapter:
                pipeline_result = pipe(
                    prompt=identity["prompt_text"],
                    negative_prompt=identity["negative_prompt_text"],
                    generator=generator,
                    height=320,
                    width=512,
                    num_frames=33,
                    num_inference_steps=8,
                    guidance_scale=5.0,
                    output_type="latent",
                )
            returned_device = pipeline_result.frames
            if isinstance(returned_device, (list, tuple)):
                returned_device = returned_device[0]
            returned = returned_device.detach().float().cpu()
            del pipeline_result, returned_device
            gc.collect()
            torch.cuda.empty_cache()
            if (
                adapter.final_latent is None
                or not torch.equal(returned, adapter.final_latent)
            ):
                raise RuntimeError("pipeline latent 与 scheduler final latent 不一致")
            final_latents[probe.probe_id] = returned
            generation_runtime_by_probe[probe.probe_id] = (
                time.time() - started
            )
            step_results[probe.probe_id] = tuple(adapter.results)
            step_records.extend(adapter.records)
            if probe.plan_index == 0:
                public_atom = atom_constructor(pipe, returned)
                atom_path = output_root / "artifacts" / "frame_state_public_atom.npz"
                write_public_frame_state_atom(atom_path, public_atom)
        if public_atom is None:
            raise RuntimeError("public atom construction 未完成")

        videos_root = output_root / "videos"
        videos_root.mkdir(parents=True, exist_ok=True)
        generation_records: list[dict[str, Any]] = []
        checkpoint_records: list[dict[str, Any]] = []
        measurements: list[ProbeMeasurement] = []
        for probe in plan:
            decoded = _decode_wan_final_latent(
                pipe,
                final_latents[probe.probe_id],
            )
            video_path = videos_root / f"{probe.plan_index:02d}_{probe.probe_id}.mp4"
            _export_video(decoded, video_path, fps=8)
            video_sha = _sha256_file(video_path)
            saved = _read_saved_rgb24_array(video_path)
            final_latent = np.ascontiguousarray(
                final_latents[probe.probe_id].numpy(),
                dtype=np.float32,
            )
            latent_projection = final_latent_carrier_projection(
                final_latent,
                public_atom.values,
            )
            decoded_feature = extract_local_temporal_feature(
                decoded,
                rgb24=False,
            )
            saved_feature = extract_local_temporal_feature(
                saved,
                rgb24=True,
            )
            exposure = accumulate_actual_signed_exposure(
                step_results[probe.probe_id],
                public_atom.values,
            )
            measurement = ProbeMeasurement(
                plan_index=probe.plan_index,
                identity_role=probe.identity_role,
                probe_id=probe.probe_id,
                signed_state_coefficient=probe.signed_state_coefficient,
                generator_state_digest_random=generator_digests[probe.probe_id],
                public_context_digest=context_digest_by_role[
                    probe.identity_role
                ],
                video_path=str(video_path),
                video_sha256=video_sha,
                final_latent_projection=latent_projection,
                decoded_feature=decoded_feature,
                saved_video_feature=saved_feature,
                step_results=step_results[probe.probe_id],
                actual_signed_exposure=exposure,
            )
            measurements.append(measurement)
            for checkpoint_id, values, boundary in (
                (
                    CHECKPOINT_IDS[0],
                    latent_projection,
                    "final_latent_float32_before_vae_decode",
                ),
                (
                    CHECKPOINT_IDS[1],
                    decoded_feature,
                    "pre_save_postprocessed_float32_frames",
                ),
                (
                    CHECKPOINT_IDS[2],
                    saved_feature,
                    "saved_video_rgb24_readback",
                ),
            ):
                checkpoint_records.append(
                    _checkpoint_record(
                        plan=probe,
                        checkpoint_id=checkpoint_id,
                        values=values,
                        source_boundary=boundary,
                    )
                )
            generation = _generation_record(
                config=config,
                plan=probe,
                measurement=measurement,
                public_atom_digest=public_atom.array_digest,
                generation_runtime_sec=generation_runtime_by_probe[
                    probe.probe_id
                ],
            )
            generation["frame_state_generation_record_id"] = _stable_digest(
                generation
            )
            generation_records.append(generation)
            write_jsonl(
                output_root / "records" / "frame_state_generation_records.jsonl",
                generation_records,
            )
            write_jsonl(
                output_root / "records" / "frame_state_step_records.jsonl",
                step_records,
            )
            write_jsonl(
                output_root / "records" / "frame_state_checkpoint_records.jsonl",
                checkpoint_records,
            )
            decoded = None
            saved = None
            final_latent = None
            gc.collect()
        return FrameStateRuntimeBatch(
            public_atom=public_atom,
            public_atom_path=str(atom_path),
            public_context_records=tuple(context_wrappers),
            measurements=tuple(measurements),
            generation_records=tuple(generation_records),
            step_records=tuple(step_records),
            checkpoint_records=tuple(checkpoint_records),
        )
    finally:
        del pipe
        gc.collect()
        torch.cuda.empty_cache()


def run_frame_state_signed_observability_construction(
    output_root: str | Path,
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    runtime_executor: Callable[
        [Mapping[str, Any], Sequence[FrameStateProbePlanRecord], Path],
        FrameStateRuntimeBatch,
    ] = execute_real_frame_state_gate0,
) -> dict[str, Any]:
    """Run the construction or a controlled fake, then evaluate C0 and Gate 0."""

    output = Path(output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"frame-state Gate 0 output root 必须为空: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    config = load_frame_state_signed_observability_config(config_path)
    _require_authorized_config(config)
    plan = build_frame_state_probe_plan(config)
    write_jsonl(
        output / "records" / "frame_state_generation_plan.jsonl",
        [asdict(record) for record in plan],
    )
    repository_commit = _repository_commit()
    try:
        batch = _validate_runtime_batch(
            config,
            plan,
            runtime_executor(config, plan, output),
        )
        write_jsonl(
            output / "records" / "frame_state_public_context_records.jsonl",
            list(batch.public_context_records),
        )
        c0 = _response_map(
            batch.measurements,
            identity_role="construction_identity",
        )
        identity_a = _response_map(
            batch.measurements,
            identity_role="signed_observability_identity",
        )
        statistics_records = [
            _statistics_record(role, checkpoint_id, responses[checkpoint_id])
            for role, responses in (
                ("construction_identity", c0),
                ("signed_observability_identity", identity_a),
            )
            for checkpoint_id in CHECKPOINT_IDS
        ]
        write_jsonl(
            output / "records" / "frame_state_statistics_records.jsonl",
            statistics_records,
        )
        c0_ready = all(
            record["frame_state_signed_gate_ready"]
            for record in statistics_records
            if record["frame_state_identity_role"] == "construction_identity"
        )
        measurement_by_id = {
            value.probe_id: value for value in batch.measurements
        }
        c0_t0 = estimate_construction_t0(
            c0["saved_video_local_temporal_feature"],
            positive_actual_exposure=measurement_by_id[
                "construction_positive"
            ].actual_signed_exposure,
            negative_actual_exposure=measurement_by_id[
                "construction_negative"
            ].actual_signed_exposure,
        )
        t0_path = output / "artifacts" / "frame_state_construction_t0.npy"
        t0_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(t0_path, c0_t0, allow_pickle=False)
        predicted = c0_t0 * 0.5 * (
            measurement_by_id[
                "signed_observability_positive"
            ].actual_signed_exposure
            - measurement_by_id[
                "signed_observability_negative"
            ].actual_signed_exposure
        )
        all_statistics_finite = all(
            record["frame_state_statistics_finite"]
            for record in statistics_records
        )
        gate: GateZeroEvaluation | None = None
        if all_statistics_finite:
            gate = evaluate_gate_zero(
                identity_a,
                predicted_primary_odd=predicted,
            )
        identity_a_readiness = {
            record["frame_state_checkpoint_id"]: bool(
                record["frame_state_signed_gate_ready"]
            )
            for record in statistics_records
            if record["frame_state_identity_role"]
            == "signed_observability_identity"
        }
        gate0_ready = bool(
            c0_ready and gate is not None and gate.gate_zero_ready
        )
        decision_name = (
            "gate0_pass_double_window_gate_a_design_allowed"
            if gate0_ready
            else "gate0_fail_stop_current_carrier_or_feature"
        )
        decision = {
            "record_version": RECORD_VERSION,
            "profile_id": PROFILE_ID,
            "frame_state_gate0_decision": decision_name,
            "frame_state_gate0_ready": gate0_ready,
            "frame_state_construction_readiness": c0_ready,
            "frame_state_checkpoint_signed_gate_ready": (
                dict(gate.checkpoint_signed_gate_ready)
                if gate is not None
                else identity_a_readiness
            ),
            "frame_state_transfer_direction_cosine": (
                gate.transfer_direction_cosine if gate is not None else None
            ),
            "frame_state_transfer_relative_error": (
                gate.transfer_relative_error if gate is not None else None
            ),
            "frame_state_primary_transfer_gate_ready": (
                gate.primary_transfer_gate_ready if gate is not None else False
            ),
            "next_double_window_gate_a_design_allowed": gate0_ready,
            "next_double_window_gate_a_execution_allowed": False,
            "observer_implementation_allowed": False,
            "attack_execution_allowed": False,
            "fixed_fpr_execution_allowed": False,
            "baseline_execution_allowed": False,
            "paper_claim_allowed": False,
            "formal_result": False,
            "stage_progression_allowed": False,
            "generation_record_count": len(batch.generation_records),
            "trajectory_step_record_count": len(batch.step_records),
            "checkpoint_record_count": len(batch.checkpoint_records),
            "public_dictionary_artifact_digest": (
                batch.public_atom.array_digest
            ),
            "frame_state_t0_sha256": _sha256_file(t0_path),
            "protocol_digest": config["protocol_digest"],
            "frame_state_probe_plan_digest": frame_state_probe_plan_digest(
                plan,
                config,
            ),
            "repository_commit": repository_commit,
            "claim_support_status": CLAIM_SUPPORT_STATUS,
        }
        write_json(
            output / "frame_state_signed_observability_decision.json",
            decision,
        )
        manifest = {
            "manifest_kind": (
                "sstw_frame_state_signed_observability_construction_manifest"
            ),
            **decision,
            "python_version": sys.version.split()[0],
            "torch_version": _package_version("torch"),
            "diffusers_version": _package_version("diffusers"),
            "video_count": len(batch.measurements),
            "public_context_record_count": len(batch.public_context_records),
        }
        write_json(
            output / "frame_state_signed_observability_manifest.json",
            manifest,
        )
        return decision
    except Exception as exc:
        failure = {
            "record_version": RECORD_VERSION,
            "profile_id": PROFILE_ID,
            "frame_state_gate0_decision": (
                "runtime_or_contract_failure_recovery_only"
            ),
            "frame_state_gate0_ready": False,
            "frame_state_failure_reason": str(exc),
            "next_double_window_gate_a_design_allowed": False,
            "next_double_window_gate_a_execution_allowed": False,
            "observer_implementation_allowed": False,
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
        write_json(
            output / "frame_state_signed_observability_decision.json",
            failure,
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
            run_frame_state_signed_observability_construction(
                arguments.output_run_root,
                config_path=arguments.config_path,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
