"""Block-attention primitive-response preflight runner.

The local helper validates the record/classifier contract.  The real Wan
runner executes only C0 clean-A step 0 and stops before the scheduler advances:
zero repeat, predeclared ± candidates, no decode, no video, no Gate 0.
"""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Any
import gc
import sys

import numpy as np

from evaluation.protocol.patch_relation_block_attention_response_preflight_contract import (
    load_patch_relation_block_attention_response_preflight_config,
)
from evaluation.protocol.patch_relation_gate0_contract import (
    load_patch_relation_gate0_config,
)
from evaluation.protocol.record_writer import write_json
from main.methods.state_space_watermark.patch_relation_attention_runtime import (
    CANDIDATE_BIAS_MAGNITUDES,
    CANDIDATE_SIGNS,
    WanBlockAttentionBranchApplicationRecord,
    build_local_block_attention_primitive_response_record,
    evaluate_block_attention_candidate_response,
    ScopedWanBlockLocalAttentionBiasAdapter,
    validate_block_attention_primitive_response_record,
)
from main.methods.state_space_watermark.patch_relation_block_attention import (
    build_block_attention_relation_descriptor,
)
from runtime.core.progress import emit_progress_event

from experiments.generative_video_model_probe.patch_relation_gate0_construction import (
    _extract_transformer_velocity,
    _frozen_schedule,
    _identity,
    _package_version,
    _replace_tuple_velocity,
    _require_native_bfloat16_runtime,
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


DECISION_FILENAME = (
    "patch_relation_block_attention_response_preflight_decision.json"
)
RECORD_FILENAME = "block_attention_primitive_response_record.json"
TEST_ID = "patch_relation_block_attention_response_preflight"
PHASE = "block_attention_response_preflight"
CLAIM_SUPPORT_STATUS = (
    "block_attention_single_step_runtime_preflight_only_not_gate_or_method_"
    "evidence"
)


class _BlockAttentionPreflightComplete(RuntimeError):
    pass


def run_local_block_attention_response_preflight(
    *,
    output_root: str | Path | None = None,
    response_gain: float = 1.0,
    repeat_floor: float = 0.0,
) -> dict[str, Any]:
    """Run the local primitive preflight without model or scheduler calls."""

    record = build_local_block_attention_primitive_response_record(
        response_gain=response_gain,
        repeat_floor=repeat_floor,
    )
    validate_block_attention_primitive_response_record(record)
    decision = {
        "decision_kind": "patch_relation_block_attention_response_preflight",
        "diagnostic_classification": record.diagnostic_classification,
        "formal_result": False,
        "stage_progression_allowed": False,
        "scheduler_step_call_count": record.scheduler_step_call_count,
        "decode_executed": record.decode_executed,
        "video_export_executed": record.video_export_executed,
        "gate0_executed": record.gate0_executed,
        "claim_support_status": record.claim_support_status,
    }
    if output_root is not None:
        root = Path(output_root)
        root.mkdir(parents=True, exist_ok=False)
        write_json(root / RECORD_FILENAME, asdict(record))
        write_json(root / DECISION_FILENAME, decision)
    return {
        "record": record,
        "decision": decision,
    }


def _array_digest(value: np.ndarray) -> str:
    array = np.asarray(value)
    if array.dtype != np.dtype("<f4") or not array.flags.c_contiguous:
        raise ValueError("block-attention digest 要求little-endian C-order float32")
    return sha256(array.tobytes(order="C")).hexdigest()


def _float64_l2(value: np.ndarray) -> float:
    array = np.asarray(value)
    if not np.all(np.isfinite(array)):
        raise ValueError("block-attention norm 输入必须有限")
    return float(np.linalg.norm(array.astype(np.float64, copy=False).reshape(-1)))


def _input_binding(probe_id: str, hidden: Any, timestep: Any) -> str:
    return _stable_digest(
        {
            "probe_id": probe_id,
            "step_index": 0,
            "hidden_states": _tensor_signature(hidden),
            "timestep": _timestep_signature(timestep),
        }
    )


def _sample_descriptor_response_vector(
    delta_velocity: np.ndarray,
) -> np.ndarray:
    descriptor = build_block_attention_relation_descriptor()
    values: list[float] = []
    _, channels, _, rows, columns = delta_velocity.shape
    for entry in descriptor.entries:
        token_in_frame = entry.query_token_index % (rows * columns)
        row = token_in_frame // columns
        column = token_in_frame % columns
        channel = int(entry.head_index) % channels
        values.append(
            float(
                np.float64(entry.coefficient)
                * np.float64(
                    delta_velocity[
                        0,
                        channel,
                        entry.time_index,
                        row,
                        column,
                    ]
                )
            )
        )
    return np.asarray(values, dtype="<f8")


class ScopedWanBlockAttentionResponsePreflight:
    """Capture real Wan step-0 block-local attention responses."""

    def __init__(
        self,
        transformer: Any,
        scheduler: Any,
        *,
        probe_id: str,
        sigma_grid: tuple[float, ...],
        delta_sigma_by_step: tuple[float, ...],
        timestep_by_step: tuple[float, ...],
    ) -> None:
        self.transformer = transformer
        self.scheduler = scheduler
        self.descriptor = build_block_attention_relation_descriptor()
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
        self._completed = False
        self._forward_calls = 0
        self._scheduler_calls = 0
        self._branch_kwargs: dict[str, dict[str, Any]] = {}
        self._base: dict[tuple[int, str], tuple[Any, WanBlockAttentionBranchApplicationRecord]] = {}
        self._record: dict[str, Any] | None = None
        self._sentinel: _BlockAttentionPreflightComplete | None = None
        self._initial_hidden_digest = ""

    def _run_forward(
        self,
        kwargs: dict[str, Any],
        *,
        branch: str,
        coefficient: int,
        magnitude: float,
        input_binding: str,
    ) -> tuple[Any, WanBlockAttentionBranchApplicationRecord]:
        if _input_binding(self.probe_id, kwargs["hidden_states"], kwargs["timestep"]) != input_binding:
            raise RuntimeError("block-attention common input binding 漂移")
        scope = ScopedWanBlockLocalAttentionBiasAdapter(
            self.transformer,
            descriptor=self.descriptor,
            signed_coefficient=coefficient,
            magnitude=magnitude,
            input_binding_digest=input_binding,
            cfg_branch_role=branch,
        )
        with scope:
            result = self._original_forward(**dict(kwargs))
        self._forward_calls += 1
        return result, scope.record()

    def _wrapped_forward(self, *args: Any, **kwargs: Any) -> Any:
        if args or self._forward_calls not in (0, 2):
            raise RuntimeError("block-attention external transformer 调用漂移")
        required = {"hidden_states", "timestep", "encoder_hidden_states"}
        if not required.issubset(kwargs):
            raise RuntimeError("block-attention transformer kwargs 不完整")
        branch = "conditional" if self._forward_calls == 0 else "unconditional"
        binding = _input_binding(self.probe_id, kwargs["hidden_states"], kwargs["timestep"])
        if not self._initial_hidden_digest:
            self._initial_hidden_digest = _tensor_values_digest(kwargs["hidden_states"])
        self._branch_kwargs[branch] = dict(kwargs)
        first_result: Any = None
        for repeat_index in (0, 1):
            row = self._run_forward(
                dict(kwargs),
                branch=branch,
                coefficient=0,
                magnitude=CANDIDATE_BIAS_MAGNITUDES[0],
                input_binding=binding,
            )
            self._base[(repeat_index, branch)] = row
            if repeat_index == 0:
                first_result = row[0]
        velocity = _runtime_float32_velocity(
            _extract_transformer_velocity(
                first_result,
                label=f"block-attention base {branch}",
            ),
            label=f"block-attention base {branch} velocity",
        )
        return _replace_tuple_velocity(
            first_result,
            velocity,
            label=f"block-attention base {branch}",
        )

    def _forward_pair(
        self,
        *,
        sign: int,
        magnitude: float,
        binding: str,
    ) -> tuple[np.ndarray, tuple[WanBlockAttentionBranchApplicationRecord, ...]]:
        rows: list[tuple[Any, WanBlockAttentionBranchApplicationRecord]] = []
        for branch in ("conditional", "unconditional"):
            rows.append(
                self._run_forward(
                    self._branch_kwargs[branch],
                    branch=branch,
                    coefficient=sign,
                    magnitude=magnitude,
                    input_binding=binding,
                )
            )
        conditional = _to_float32_numpy(
            _runtime_float32_velocity(
                _extract_transformer_velocity(
                    rows[0][0],
                    label="block-attention conditional velocity",
                ),
                label="block-attention conditional velocity",
            ),
            label="block-attention conditional velocity",
        )
        unconditional = _to_float32_numpy(
            _runtime_float32_velocity(
                _extract_transformer_velocity(
                    rows[1][0],
                    label="block-attention unconditional velocity",
                ),
                label="block-attention unconditional velocity",
            ),
            label="block-attention unconditional velocity",
        )
        return (
            _to_float32_numpy(
                _runtime_cfg_combine(conditional, unconditional),
                label="block-attention CFG velocity",
            ),
            (rows[0][1], rows[1][1]),
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
            raise RuntimeError("block-attention scheduler boundary 漂移")
        actual_timestep = _scalar_float32(
            timestep,
            label="block-attention scheduler timestep",
        )
        scheduler_timesteps = getattr(self.scheduler, "timesteps", None)
        scheduler_sigmas = getattr(self.scheduler, "sigmas", None)
        if (
            scheduler_timesteps is None
            or len(scheduler_timesteps) != 8
            or scheduler_sigmas is None
            or len(scheduler_sigmas) != 9
            or _scalar_float32(scheduler_timesteps[0], label="block-attention frozen timestep") != self.timestep_by_step[0]
            or _scalar_float32(scheduler_sigmas[0], label="block-attention sigma before") != self.sigma_grid[0]
            or _scalar_float32(scheduler_sigmas[1], label="block-attention sigma after") != self.sigma_grid[1]
            or actual_timestep != self.timestep_by_step[0]
        ):
            raise RuntimeError("block-attention timestep 不是冻结step0")
        if getattr(self.scheduler, "step_index", None) is not None:
            raise RuntimeError("block-attention scheduler 已提前推进")
        binding = _input_binding(
            self.probe_id,
            self._branch_kwargs["conditional"]["hidden_states"],
            self._branch_kwargs["conditional"]["timestep"],
        )
        sample_cast = _scheduler_sample_as_transformer_input(
            sample,
            getattr(self.transformer, "dtype", None),
        )
        for branch in ("conditional", "unconditional"):
            if (
                _tensor_signature(sample_cast)
                != _tensor_signature(self._branch_kwargs[branch]["hidden_states"])
                or _timestep_signature(timestep)
                != _timestep_signature(self._branch_kwargs[branch]["timestep"])
            ):
                raise RuntimeError("block-attention scheduler/transformer context 漂移")
        base_cfg_rows: list[np.ndarray] = []
        for repeat_index in (0, 1):
            conditional = _to_float32_numpy(
                _extract_transformer_velocity(
                    self._base[(repeat_index, "conditional")][0],
                    label="block-attention base conditional",
                ),
                label="block-attention base conditional",
            )
            unconditional = _to_float32_numpy(
                _extract_transformer_velocity(
                    self._base[(repeat_index, "unconditional")][0],
                    label="block-attention base unconditional",
                ),
                label="block-attention base unconditional",
            )
            base_cfg_rows.append(
                _to_float32_numpy(
                    _runtime_cfg_combine(conditional, unconditional),
                    label="block-attention base CFG",
                )
            )
        pipeline_cfg = _to_float32_numpy(
            _runtime_float32_velocity(
                model_output,
                label="block-attention pipeline CFG",
            ),
            label="block-attention pipeline CFG",
        )
        if not np.array_equal(pipeline_cfg, base_cfg_rows[0]):
            raise RuntimeError("block-attention pipeline CFG 与zero base不一致")
        candidate_records = []
        pair_cosines: list[float] = []
        scheduler_sample = _to_float32_numpy(
            _runtime_float32_velocity(sample, label="block-attention sample"),
            label="block-attention sample",
        )
        del scheduler_sample
        candidate_index = 0
        for magnitude in CANDIDATE_BIAS_MAGNITUDES:
            vectors: dict[int, np.ndarray] = {}
            app_records: dict[int, WanBlockAttentionBranchApplicationRecord] = {}
            for sign in CANDIDATE_SIGNS:
                cfg, branch_records = self._forward_pair(
                    sign=sign,
                    magnitude=magnitude,
                    binding=binding,
                )
                delta = np.subtract(cfg, base_cfg_rows[0], dtype=np.float32)
                response_vector = _sample_descriptor_response_vector(delta)
                repeat_vector = _sample_descriptor_response_vector(
                    np.subtract(
                        base_cfg_rows[1],
                        base_cfg_rows[0],
                        dtype=np.float32,
                    )
                )
                candidate = evaluate_block_attention_candidate_response(
                    self.descriptor,
                    candidate_index=candidate_index,
                    signed_coefficient=sign,
                    magnitude=magnitude,
                    response_vector=response_vector,
                    repeat_delta_vector=repeat_vector,
                )
                candidate_records.append(candidate)
                vectors[sign] = response_vector
                app_records[sign] = branch_records[0]
                candidate_index += 1
            from main.methods.state_space_watermark.patch_relation_attention_runtime import _safe_cosine

            pair_cosines.append(_safe_cosine(vectors[1], vectors[-1]))
            if app_records[1].descriptor_digest != app_records[-1].descriptor_digest:
                raise RuntimeError("block-attention branch descriptor digest 漂移")
        from main.methods.state_space_watermark.patch_relation_attention_runtime import (
            BlockAttentionCandidateResponse,
            BlockAttentionPrimitiveResponseRecord,
            classify_block_attention_primitive_response,
        )

        updated_candidates: list[BlockAttentionCandidateResponse] = []
        for candidate in candidate_records:
            pair_index = CANDIDATE_BIAS_MAGNITUDES.index(candidate.magnitude)
            near_pair = pair_cosines[pair_index] <= -0.9
            updated_candidates.append(
                BlockAttentionCandidateResponse(
                    candidate_index=candidate.candidate_index,
                    signed_coefficient=candidate.signed_coefficient,
                    magnitude=candidate.magnitude,
                    response_vector=candidate.response_vector,
                    repeat_delta_vector=candidate.repeat_delta_vector,
                    response_l2_norm=candidate.response_l2_norm,
                    norm_budget=candidate.norm_budget,
                    norm_guard_passed=candidate.norm_guard_passed,
                    repeat_l2_norm=candidate.repeat_l2_norm,
                    repeat_floor_ratio=candidate.repeat_floor_ratio,
                    nonzero_response=candidate.nonzero_response,
                    repeatable_above_floor=candidate.repeatable_above_floor,
                    near_antisymmetric_pair=bool(near_pair),
                    feasible_candidate=bool(
                        near_pair
                        and candidate.norm_guard_passed
                        and candidate.repeatable_above_floor
                    ),
                    application_record=candidate.application_record,
                )
            )
        record_obj = BlockAttentionPrimitiveResponseRecord(
            record_kind="patch_relation_block_attention_primitive_response_preflight",
            descriptor_digest=self.descriptor.descriptor_digest,
            zero_repeat_count=2,
            scheduler_step_call_count=0,
            decode_executed=False,
            video_export_executed=False,
            gate0_executed=False,
            candidate_responses=tuple(updated_candidates),
            positive_negative_response_cosine_by_magnitude=tuple(pair_cosines),
            diagnostic_classification=classify_block_attention_primitive_response(
                tuple(updated_candidates),
                tuple(pair_cosines),
            ),
            formal_result=False,
            stage_progression_allowed=False,
            claim_support_status=(
                "block_attention_local_primitive_response_only_not_gpu_gate_or_"
                "method_evidence"
            ),
        )
        validate_block_attention_primitive_response_record(record_obj)
        self._record = {
            **asdict(record_obj),
            "record_execution_kind": "real_wan_single_step_block_attention_preflight",
            "transformer_forward_count": self._forward_calls,
            "real_scheduler_step_call_count": 0,
            "scheduler_internal_step_index_before_and_after": [
                getattr(self.scheduler, "step_index", None),
                getattr(self.scheduler, "step_index", None),
            ],
            "initial_hidden_state_digest_random": self._initial_hidden_digest,
            "base_cfg_digests": [_array_digest(row) for row in base_cfg_rows],
            "base_cfg_repeat_delta_norm": _float64_l2(
                np.subtract(base_cfg_rows[1], base_cfg_rows[0], dtype=np.float32)
            ),
            "claim_support_status": CLAIM_SUPPORT_STATUS,
        }
        self._sentinel = _BlockAttentionPreflightComplete(
            "block-attention response preflight completed"
        )
        raise self._sentinel

    def __enter__(self) -> "ScopedWanBlockAttentionResponsePreflight":
        transformer_dict = getattr(self.transformer, "__dict__", {})
        scheduler_dict = getattr(self.scheduler, "__dict__", {})
        self._transformer_had_forward = "forward" in transformer_dict
        self._transformer_previous_forward = transformer_dict.get("forward")
        self._scheduler_had_step = "step" in scheduler_dict
        self._scheduler_previous_step = scheduler_dict.get("step")
        self._original_forward = self.transformer.forward
        self._original_scheduler_step = self.scheduler.step
        self.transformer.forward = self._wrapped_forward
        self.scheduler.step = self._wrapped_scheduler_step
        self._entered = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        self._exit_attempted = True
        cleanup_errors: list[BaseException] = []
        for target, name, had_value, previous, original in (
            (self.transformer, "forward", self._transformer_had_forward, self._transformer_previous_forward, self._original_forward),
            (self.scheduler, "step", self._scheduler_had_step, self._scheduler_previous_step, self._original_scheduler_step),
        ):
            try:
                if had_value:
                    setattr(target, name, previous)
                else:
                    current = getattr(target, name)
                    if getattr(current, "__func__", current) is getattr(original, "__func__", original):
                        delattr(target, name)
                    else:
                        setattr(target, name, original)
            except BaseException as error:
                cleanup_errors.append(error)
        self._cleanup_completed = not cleanup_errors
        exact = (
            exc_type is _BlockAttentionPreflightComplete
            and exc is self._sentinel
            and self._record is not None
            and self._forward_calls == 16
            and self._scheduler_calls == 0
            and self._cleanup_completed
        )
        self._sentinel = None
        if exact:
            self._completed = True
            return True
        self._record = None
        if cleanup_errors and exc is not None and hasattr(exc, "add_note"):
            for error in cleanup_errors:
                exc.add_note(
                    "block-attention scope cleanup also failed: "
                    f"{type(error).__name__}"
                )
        if cleanup_errors:
            raise cleanup_errors[0] from exc
        return False

    def batch(self) -> dict[str, Any]:
        if (
            not self._completed
            or not self._cleanup_completed
            or self._record is None
            or self._forward_calls != 16
            or self._scheduler_calls != 0
        ):
            raise RuntimeError("block-attention response preflight 未完整完成")
        return dict(self._record)

    def release_runtime_references(self) -> None:
        if self._entered and not self._exit_attempted:
            raise RuntimeError("active block-attention scope 不得释放引用")
        self._branch_kwargs.clear()
        self._base.clear()
        self._record = None
        self._sentinel = None
        self._original_forward = None
        self._original_scheduler_step = None
        self.transformer = None
        self.scheduler = None


def execute_real_patch_relation_block_attention_response_preflight(
    config: dict[str, Any],
    gate0_config: dict[str, Any],
) -> dict[str, Any]:
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
        raise RuntimeError("block-attention preflight 需要用户显式启动Colab CUDA")
    selected_dtype = _select_dtype(torch)
    _require_native_bfloat16_runtime(torch, selected_dtype=selected_dtype)
    common = gate0_config["protocol_contract"]["execution_identity_contract"][
        "execution_common"
    ]
    identity = _identity(gate0_config, "construction_c0")
    pipe: Any | None = None
    adapter: ScopedWanBlockAttentionResponsePreflight | None = None
    try:
        pipe = _load_video_generation_pipeline(
            common["model_id"],
            selected_dtype,
            revision=common["model_revision"],
        )
        provenance = _generation_model_provenance_from_pipeline(
            pipe,
            expected_model_id=common["model_id"],
        )
        if (
            _package_version("diffusers") != common["diffusers_version"]
            or provenance["generation_model_commit_or_hash"]
            != common["model_revision"]
            or _scheduler_signature(pipe.scheduler)
            != common["scheduler_signature"]
        ):
            raise RuntimeError("block-attention model/scheduler provenance 漂移")
        if bool(getattr(pipe.transformer, "is_cache_enabled", False)):
            raise RuntimeError("block-attention preflight 禁止transformer cache")
        sigmas, deltas, timesteps = _frozen_schedule(gate0_config)
        generator = torch.Generator(device="cuda").manual_seed(
            int(identity["seed_value"])
        )
        generator_digest = _tensor_digest(generator.get_state())
        adapter = ScopedWanBlockAttentionResponsePreflight(
            pipe.transformer,
            pipe.scheduler,
            probe_id=config["protocol_contract"]["runtime_execution_binding"][
                "probe_id"
            ],
            sigma_grid=sigmas,
            delta_sigma_by_step=deltas,
            timestep_by_step=timesteps,
        )
        emit_progress_event(
            "patch_relation_block_attention_response_preflight",
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
        record = adapter.batch()
        record["generator_state_digest_random"] = generator_digest
        emit_progress_event(
            "patch_relation_block_attention_response_preflight",
            "step0 finish; real_scheduler_step_call_count=0",
        )
        return record
    finally:
        active_error = sys.exc_info()[1]
        cleanup_errors: list[tuple[str, BaseException]] = []
        if adapter is not None:
            try:
                adapter.release_runtime_references()
            except BaseException as error:
                cleanup_errors.append(("adapter_release", error))
        if pipe is not None:
            maybe_free = getattr(pipe, "maybe_free_model_hooks", None)
            if callable(maybe_free):
                try:
                    maybe_free()
                except BaseException as error:
                    cleanup_errors.append(("pipeline_model_hook_cleanup", error))
            del pipe
        try:
            gc.collect()
        except BaseException as error:
            cleanup_errors.append(("gc_collect", error))
        try:
            torch.cuda.empty_cache()
        except BaseException as error:
            cleanup_errors.append(("cuda_empty_cache", error))
        if cleanup_errors and active_error is not None and hasattr(active_error, "add_note"):
            for label, error in cleanup_errors:
                active_error.add_note(
                    "block-attention cleanup also failed: "
                    f"{label}={type(error).__name__}"
                )
        elif cleanup_errors:
            raise cleanup_errors[0][1]


def run_patch_relation_block_attention_response_preflight(
    output_root: str | Path,
    *,
    executor: Any = execute_real_patch_relation_block_attention_response_preflight,
) -> dict[str, Any]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    try:
        config = load_patch_relation_block_attention_response_preflight_config()
        gate0_config = load_patch_relation_gate0_config()
        record = executor(config, gate0_config)
        record_obj = build_local_block_attention_primitive_response_record()
        del record_obj
        decision = {
            "decision_kind": "patch_relation_block_attention_response_preflight",
            "diagnostic_classification": record["diagnostic_classification"],
            "formal_result": False,
            "stage_progression_allowed": False,
            "scheduler_step_call_count": record["scheduler_step_call_count"],
            "decode_executed": False,
            "video_export_executed": False,
            "gate0_executed": False,
            "claim_support_status": CLAIM_SUPPORT_STATUS,
        }
        write_json(root / RECORD_FILENAME, record)
        write_json(root / DECISION_FILENAME, decision)
        return decision
    except BaseException as error:
        decision = {
            "decision_kind": "patch_relation_block_attention_response_preflight",
            "diagnostic_classification": "runtime_or_contract_failure",
            "failure_reason": str(error),
            "formal_result": False,
            "stage_progression_allowed": False,
            "scheduler_step_call_count": 0,
            "decode_executed": False,
            "video_export_executed": False,
            "gate0_executed": False,
            "claim_support_status": "failure_recovery_only_not_claim_evidence",
        }
        write_json(root / DECISION_FILENAME, decision)
        raise
