"""Run the five-output frozen-feedback signed-response construction diagnostic.

Only one eight-step clean Wan trajectory evaluates the transformer.  Four
counterfactual trajectories reuse the clean CFG-combined base velocities with
independent FlowMatch scheduler clones and never feed counterfactual state back
into the model.  This is a root-cause construction diagnostic, not Gate A.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from evaluation.protocol.existing_six_video_spatiotemporal_signed_response_contract import (
    load_spatiotemporal_diagnostic_config,
    validate_existing_six_video_source,
)
from evaluation.protocol.frozen_feedback_signed_response_contract import (
    CLAIM_SUPPORT_STATUS,
    DEFAULT_CONFIG_PATH,
    EXPECTED_CLEAN_SCHEDULER_STEP_COUNT,
    EXPECTED_CLEAN_TRANSFORMER_FORWARD_CALL_COUNT,
    EXPECTED_COUNTERFACTUAL_STEP_COUNT,
    EXPECTED_COUNTERFACTUAL_TRANSFORMER_FORWARD_CALL_COUNT,
    FROZEN_CONFIG_DIGEST,
    FROZEN_OUTPUT_IDS,
    FULL_CHECKPOINT_IDS,
    PAIR_IDS,
    PROFILE_ID,
    RECORD_VERSION,
    FrozenFeedbackResponseGate,
    apply_frozen_signed_response_gate,
    build_frozen_feedback_plan,
    classify_frozen_feedback_results,
    compute_single_clean_response_statistics,
    compute_single_clean_response_statistics_from_gram,
    load_frozen_feedback_signed_response_config,
)
from evaluation.protocol.impulse_observability_contract import (
    CONSTRUCTION_FLOW_STEP_COUNT,
    CONSTRUCTION_LATENT_LAYOUT_SHAPE,
    STAGE_BASIS_RANK,
    ActualImpulseExposureTrace,
    ConstructionStageBasis,
    ImpulseProbePlanRecord,
    build_construction_stage_basis,
    canonical_json_digest,
    compute_intended_impulse_control,
    construction_feature_row_binding_digest,
    effective_construction_basis_directions,
    load_impulse_observability_config,
    _validate_trace_schedule_and_budget,
    _validate_trace_shape,
)
from evaluation.protocol.record_writer import write_json, write_jsonl
from experiments.generative_video_model_probe.output_feature_impulse_observability_construction import (
    ConstructionFeatureBatch,
    ImpulseObservabilitySchedulerRuntime,
    _equal_area_rgb_summary,
    _package_version,
    _sha256_file,
    _stable_digest,
    extract_real_output_features,
    read_saved_video_rgb24_summary,
    validate_output_vae_metadata,
)


TEST_ID = "frozen_feedback_signed_response_diagnostic"
MANIFEST_KIND = "frozen_feedback_signed_response_diagnostic_manifest"
FULL_ARRAY_CHUNK_SIZE = 1_048_576


@dataclass(frozen=True)
class CleanTraceStep:
    """One captured clean scheduler input and its governed provenance."""

    step_index: int
    timestep: Any
    state_before: Any
    base_velocity: Any
    record: Mapping[str, Any]


@dataclass(frozen=True)
class CleanTrace:
    """The only trajectory allowed to evaluate the generation transformer."""

    initial_latent: Any
    final_latent: Any
    steps: tuple[CleanTraceStep, ...]
    transformer_call_records: tuple[Mapping[str, Any], ...]
    generator_state_digest_random: str
    trace_digest: str


@dataclass(frozen=True)
class FrozenFeedbackGenerationBatch:
    """Five decoded outputs plus the one clean and four offline traces."""

    generation_records: tuple[Mapping[str, Any], ...]
    clean_trace_records: tuple[Mapping[str, Any], ...]
    transformer_call_records: tuple[Mapping[str, Any], ...]
    counterfactual_step_records: tuple[Mapping[str, Any], ...]
    exposure_traces: tuple[ActualImpulseExposureTrace, ...]
    checkpoint_records: tuple[Mapping[str, Any], ...]
    latent_basis_projection_records: tuple[Mapping[str, Any], ...]
    latent_paths: Mapping[str, Path]
    decoded_paths: Mapping[str, Path]
    saved_rgb_paths: Mapping[str, Path]
    model_call_summary: Mapping[str, Any]
    clean_trace_digest: str
    initial_latent_digest_random: str


def _repository_commit() -> str:
    root = Path(__file__).resolve().parents[2]
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
    ).strip()


def _tensor_digest(value: Any) -> str:
    """Implementation split out for lightweight tests and readable errors."""

    import torch

    tensor = value.detach().contiguous()
    raw = tensor.view(torch.uint8).cpu().numpy().tobytes(order="C")
    return sha256(raw).hexdigest()


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return sha256(array.tobytes(order="C")).hexdigest()


def _stable_record(
    payload: Mapping[str, Any],
    id_field: str,
) -> dict[str, Any]:
    record = dict(payload)
    record[id_field] = _stable_digest(record)
    return record


def _runtime_config(config: Mapping[str, Any]) -> dict[str, Any]:
    base_contract = config["base_construction_contract"]
    base = load_impulse_observability_config(
        base_contract["config_path"]
    )
    if canonical_json_digest(base) != base_contract["config_digest"]:
        raise RuntimeError("frozen-feedback base construction config 漂移")
    runtime = json.loads(json.dumps(base))
    amplitude = float(config["control_contract"]["lambda_max"])
    runtime["actual_exposure_contract"]["lambda_max"] = amplitude
    runtime["impulse_probe"]["nominal_signed_amplitude"] = amplitude
    if (
        runtime["execution_identity"]
        != {
            key: value
            for key, value in config["execution_identity"].items()
            if key
            not in {
                "prompt_text",
                "negative_prompt_text",
            }
        }
        or runtime["flow_schedule_contract"]["waveform_schema_digest"]
        != base_contract["waveform_schema_digest"]
        or runtime["runtime_adapter_contract"]["adapter_schema_digest"]
        != base_contract["runtime_adapter_schema_digest"]
        or runtime["construction_feature_schema"]["feature_schema_digest"]
        != base_contract["construction_feature_schema_digest"]
    ):
        raise RuntimeError("frozen-feedback runtime identity/schema 漂移")
    return runtime


def _validate_historical_source(
    source_root: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the complete f06 normal-feedback FAIL used by the design decision."""

    binding = config["historical_normal_feedback_source"]
    validator_config = load_spatiotemporal_diagnostic_config(
        binding["validator_config_path"]
    )
    if canonical_json_digest(validator_config) != binding[
        "validator_config_digest"
    ]:
        raise RuntimeError("historical normal-feedback validator config 漂移")
    validated = validate_existing_six_video_source(
        source_root,
        validator_config,
    )
    decision = validated.decision
    if (
        validated.source_snapshot_digest
        != binding["source_snapshot_digest"]
        or decision.get("repository_commit")
        != binding["repository_commit"]
        or decision.get("gate_a_root_cause_diagnostic_decision")
        != binding["decision"]
        or decision.get("frozen_feedback_diagnostic_design_allowed")
        is not True
        or decision.get("gate_a_pass") is not False
        or decision.get("formal_result") is not False
        or decision.get("stage_progression_allowed") is not False
    ):
        raise RuntimeError("historical normal-feedback source binding 失败")

    statistic_path = (
        source_root
        / "records"
        / "root_cause_paired_response_statistics.jsonl"
    )
    rows = [
        json.loads(line)
        for line in statistic_path.read_text(
            encoding="utf-8-sig"
        ).splitlines()
        if line.strip()
    ]
    expected_gate = binding["normal_feedback_full_latent_signed_gate_by_pair"]
    observed: dict[str, bool] = {}
    gate = config["signed_response_gate"]
    for pair_id in PAIR_IDS:
        matches = [
            row
            for row in rows
            if row.get("root_cause_pair_id") == pair_id
            and row.get("root_cause_checkpoint_id")
            == "T_final_latent_full"
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"historical normal-feedback {pair_id} latent statistic 不唯一"
            )
        row = matches[0]
        values = (
            row.get("root_cause_odd_norm"),
            row.get("root_cause_common_odd_ratio"),
            row.get("root_cause_antisymmetry_cosine"),
            row.get("root_cause_antisymmetry_residual"),
        )
        if any(
            not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in values
        ):
            raise RuntimeError("historical normal-feedback statistic 非有限")
        observed[pair_id] = bool(
            float(row["root_cause_odd_norm"])
            >= float(gate["minimum_odd_norm"])
            and float(row["root_cause_common_odd_ratio"])
            <= float(gate["maximum_common_odd_ratio"])
            and float(row["root_cause_antisymmetry_cosine"])
            >= float(gate["minimum_antisymmetry_cosine"])
            and float(row["root_cause_antisymmetry_residual"])
            <= float(gate["maximum_antisymmetry_residual"])
        )
    if observed != expected_gate or any(observed.values()):
        raise RuntimeError(
            "historical normal-feedback latent FAIL control 未冻结"
        )
    return {
        "source_snapshot_digest": validated.source_snapshot_digest,
        "source_repository_commit": decision["repository_commit"],
        "historical_normal_feedback_full_latent_gate_by_pair": observed,
        "historical_gate_a_fail_preserved": True,
        "historical_source_apply_only": True,
    }


class FrozenCleanTraceRuntime:
    """Capture one official Wan clean denoising trajectory without alteration."""

    def __init__(
        self,
        pipe: Any,
        *,
        config: Mapping[str, Any],
        generator_state_digest_random: str,
    ) -> None:
        self.pipe = pipe
        self.scheduler = pipe.scheduler
        self.config = config
        self.generator_state_digest_random = generator_state_digest_random
        self._original_step: Any | None = None
        self._original_forward: Any | None = None
        self._step_index = 0
        self._reference_cumulative_energy = 0.0
        self._steps: list[CleanTraceStep] = []
        self._transformer_calls: list[dict[str, Any]] = []
        self.initial_latent: Any | None = None
        self.final_latent: Any | None = None

    def __enter__(self) -> "FrozenCleanTraceRuntime":
        if "FlowMatch" not in type(self.scheduler).__name__:
            raise RuntimeError("frozen-feedback 要求 FlowMatch scheduler")
        if getattr(self.pipe, "transformer_2", None) is not None:
            raise RuntimeError("frozen-feedback 只冻结 Wan2.1 单 transformer")
        self._original_step = self.scheduler.step
        self._original_forward = self.pipe.transformer.forward

        def traced_forward(*args: Any, **kwargs: Any) -> Any:
            call_index = len(self._transformer_calls)
            step_index = call_index // 2
            component = "conditional" if call_index % 2 == 0 else "unconditional"
            hidden = kwargs.get("hidden_states")
            timestep = kwargs.get("timestep")
            if hidden is None or timestep is None:
                raise RuntimeError("Wan transformer call 缺少 hidden/timestep binding")
            record = {
                "record_version": RECORD_VERSION,
                "profile_id": PROFILE_ID,
                "clean_transformer_forward_invocation_index": call_index,
                "clean_logical_denoiser_step_index": step_index,
                "clean_cfg_component": component,
                "clean_transformer_hidden_shape": list(hidden.shape),
                "clean_transformer_hidden_dtype": str(hidden.dtype),
                "clean_transformer_hidden_array_sha256": _tensor_digest(hidden),
                "clean_transformer_timestep_shape": list(timestep.shape),
                "clean_transformer_timestep_dtype": str(timestep.dtype),
                "clean_transformer_timestep_array_sha256": _tensor_digest(
                    timestep
                ),
                "counterfactual_transformer_call": False,
                "formal_result": False,
                "stage_progression_allowed": False,
                "claim_support_status": CLAIM_SUPPORT_STATUS,
            }
            self._transformer_calls.append(
                _stable_record(
                    record,
                    "clean_transformer_call_record_id",
                )
            )
            return self._original_forward(*args, **kwargs)

        def traced_step(
            model_output: Any,
            timestep: Any,
            sample: Any,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            return self._run_clean_step(
                model_output,
                timestep,
                sample,
                *args,
                **kwargs,
            )

        self.pipe.transformer.forward = traced_forward
        self.scheduler.step = traced_step
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._original_step is not None:
            self.scheduler.step = self._original_step
        if self._original_forward is not None:
            self.pipe.transformer.forward = self._original_forward

    def _validate_schedule(self) -> None:
        observed = tuple(
            float(value.detach().float().item())
            if hasattr(value, "detach")
            else float(value)
            for value in self.scheduler.sigmas
        )
        expected = tuple(
            float(value)
            for value in self.config["flow_schedule_contract"]["sigma_grid"]
        )
        if len(observed) != len(expected) or any(
            not math.isclose(
                value,
                target,
                rel_tol=1e-7,
                abs_tol=1e-8,
            )
            for value, target in zip(observed, expected, strict=True)
        ):
            raise RuntimeError("frozen-feedback clean scheduler grid 漂移")

    def _run_clean_step(
        self,
        model_output: Any,
        timestep: Any,
        sample: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if self._original_step is None:
            raise RuntimeError("clean trace runtime 未进入 context")
        if self._step_index == 0:
            self._validate_schedule()
            self.initial_latent = sample.detach().cpu().clone()
        if self._step_index >= CONSTRUCTION_FLOW_STEP_COUNT:
            raise RuntimeError("clean scheduler 调用超过冻结8步")
        index = self._step_index
        schedule = self.config["flow_schedule_contract"]
        delta_sigma = float(schedule["delta_sigma_by_step"][index])
        phase = float(schedule["flow_phase_by_step"][index])
        macro_interval = int(
            schedule["macro_interval_index_by_step"][index]
        )
        base_raw = model_output.detach().cpu().clone()
        base_fp32 = model_output.detach().float()
        state_before_digest = _tensor_digest(sample)
        base_digest = _tensor_digest(model_output)
        base_norm = float(base_fp32.norm().item())
        reference_increment = delta_sigma**2 * base_norm**2
        self._reference_cumulative_energy += reference_increment
        result = self._original_step(
            model_output,
            timestep,
            sample,
            *args,
            **kwargs,
        )
        state_after = (
            result[0]
            if isinstance(result, tuple)
            else getattr(result, "prev_sample", None)
        )
        if state_after is None:
            raise RuntimeError("clean scheduler output 缺少 prev_sample")
        call_start = index * 2
        call_stop = call_start + 2
        if len(self._transformer_calls) != call_stop:
            raise RuntimeError(
                "clean CFG transformer call 与 scheduler step 未一一绑定"
            )
        record = {
            "record_version": RECORD_VERSION,
            "profile_id": PROFILE_ID,
            "clean_trace_step_index": index,
            "clean_trace_flow_phase": phase,
            "clean_trace_delta_sigma": delta_sigma,
            "clean_trace_macro_interval_index": macro_interval,
            "clean_trace_timestep": float(timestep.detach().float().item()),
            "clean_trace_state_before_shape": list(sample.shape),
            "clean_trace_state_before_dtype": str(sample.dtype),
            "clean_trace_state_before_array_sha256": state_before_digest,
            "clean_trace_base_velocity_shape": list(model_output.shape),
            "clean_trace_base_velocity_dtype": str(model_output.dtype),
            "clean_trace_base_velocity_array_sha256": base_digest,
            "clean_trace_base_velocity_norm": base_norm,
            "clean_trace_reference_energy_increment": reference_increment,
            "clean_trace_reference_cumulative_energy": (
                self._reference_cumulative_energy
            ),
            "clean_trace_state_after_shape": list(state_after.shape),
            "clean_trace_state_after_dtype": str(state_after.dtype),
            "clean_trace_state_after_array_sha256": _tensor_digest(
                state_after
            ),
            "clean_trace_transformer_forward_invocation_start": call_start,
            "clean_trace_transformer_forward_invocation_stop": call_stop,
            "clean_trace_cfg_combined_velocity_ready": True,
            "counterfactual_transformer_call": False,
            "formal_result": False,
            "stage_progression_allowed": False,
            "claim_support_status": CLAIM_SUPPORT_STATUS,
        }
        governed = _stable_record(record, "clean_velocity_trace_record_id")
        self._steps.append(
            CleanTraceStep(
                step_index=index,
                timestep=timestep.detach().cpu().clone(),
                state_before=sample.detach().cpu().clone(),
                base_velocity=base_raw,
                record=governed,
            )
        )
        self.final_latent = state_after.detach().cpu().clone()
        self._step_index += 1
        return result

    def build_trace(self) -> CleanTrace:
        if (
            self.initial_latent is None
            or self.final_latent is None
            or len(self._steps) != EXPECTED_CLEAN_SCHEDULER_STEP_COUNT
            or len(self._transformer_calls)
            != EXPECTED_CLEAN_TRANSFORMER_FORWARD_CALL_COUNT
        ):
            raise RuntimeError("clean trace coverage 未精确完成")
        per_step_calls = {
            int(record["clean_logical_denoiser_step_index"]): 0
            for record in self._transformer_calls
        }
        for record in self._transformer_calls:
            per_step_calls[
                int(record["clean_logical_denoiser_step_index"])
            ] += 1
        if per_step_calls != {index: 2 for index in range(8)}:
            raise RuntimeError("clean CFG 每step必须精确cond/uncond两次")
        trace_digest = _stable_digest(
            {
                "initial_latent_digest_random": _tensor_digest(
                    self.initial_latent
                ),
                "final_latent_digest": _tensor_digest(self.final_latent),
                "step_record_ids": [
                    step.record["clean_velocity_trace_record_id"]
                    for step in self._steps
                ],
                "transformer_call_record_ids": [
                    record["clean_transformer_call_record_id"]
                    for record in self._transformer_calls
                ],
            }
        )
        return CleanTrace(
            initial_latent=self.initial_latent,
            final_latent=self.final_latent,
            steps=tuple(self._steps),
            transformer_call_records=tuple(self._transformer_calls),
            generator_state_digest_random=(
                self.generator_state_digest_random
            ),
            trace_digest=trace_digest,
        )


def _clone_flow_scheduler(
    scheduler: Any,
    *,
    device: Any,
    runtime_config: Mapping[str, Any],
) -> Any:
    clone = type(scheduler).from_config(scheduler.config)
    clone.set_timesteps(CONSTRUCTION_FLOW_STEP_COUNT, device=device)
    observed = tuple(
        float(value.detach().float().item()) for value in clone.sigmas
    )
    expected = tuple(
        float(value)
        for value in runtime_config["flow_schedule_contract"]["sigma_grid"]
    )
    if len(observed) != len(expected) or any(
        not math.isclose(
            value,
            target,
            rel_tol=1e-7,
            abs_tol=1e-8,
        )
        for value, target in zip(observed, expected, strict=True)
    ):
        raise RuntimeError("counterfactual scheduler clone sigma grid 漂移")
    return clone


def _scheduler_prev_sample(result: Any) -> Any:
    value = (
        result[0]
        if isinstance(result, tuple)
        else getattr(result, "prev_sample", None)
    )
    if value is None:
        raise RuntimeError("Flow scheduler result 缺少 prev_sample")
    return value


def _validate_offline_clean_replay(
    scheduler: Any,
    clean_trace: CleanTrace,
    runtime_config: Mapping[str, Any],
    *,
    device: Any,
) -> str:
    """Require an exact clone replay before any signed branch is accepted."""

    import torch

    clone = _clone_flow_scheduler(
        scheduler,
        device=device,
        runtime_config=runtime_config,
    )
    state = clean_trace.initial_latent.to(device)
    for index, step in enumerate(clean_trace.steps):
        base = step.base_velocity.to(device)
        result = clone.step(
            base,
            clone.timesteps[index],
            state,
            return_dict=False,
        )
        state = _scheduler_prev_sample(result)
    replay = state.detach().cpu()
    expected = clean_trace.final_latent
    if (
        replay.shape != expected.shape
        or replay.dtype != expected.dtype
        or not torch.equal(replay, expected)
    ):
        raise RuntimeError(
            "offline clean scheduler replay 与 pipeline clean final latent "
            "不是 exact array equal"
        )
    return _tensor_digest(replay)


def _counterfactual_step_record(
    *,
    probe: ImpulseProbePlanRecord,
    step_index: int,
    runtime_config: Mapping[str, Any],
    clean_step: CleanTraceStep,
    state_before: Any,
    state_after: Any,
    waveform: float,
    base_norm: float,
    reference_increment: float,
    reference_cumulative: float,
    cumulative_control_energy: float,
    remaining_energy: float,
    intended_norm: float,
    intended_exposure: float,
    target_coordinate: int,
    coordinates: Sequence[float],
    selection: Any | None,
    update_recomputation_equal: bool,
) -> dict[str, Any]:
    schedule = runtime_config["flow_schedule_contract"]
    delta_sigma = float(schedule["delta_sigma_by_step"][step_index])
    actual_norm = (
        0.0 if selection is None else selection.evaluation.actual_delta_norm
    )
    energy_increment = (
        0.0 if selection is None else selection.evaluation.energy_increment
    )
    direction_cosine = (
        1.0
        if selection is None
        else float(selection.evaluation.direction_cosine)
    )
    active = abs(float(waveform)) > 1e-15
    actual_exposure = tuple(
        delta_sigma * float(value) for value in coordinates
    )
    record = {
        "record_version": RECORD_VERSION,
        "profile_id": PROFILE_ID,
        "impulse_probe_id": probe.probe_id,
        "impulse_stage_index": probe.stage_index,
        "impulse_state_channel_index": probe.channel_index,
        "impulse_polarity": probe.polarity,
        "impulse_flow_step_index": step_index,
        "impulse_flow_phase": float(
            schedule["flow_phase_by_step"][step_index]
        ),
        "impulse_delta_sigma": delta_sigma,
        "impulse_macro_interval_index": int(
            schedule["macro_interval_index_by_step"][step_index]
        ),
        "impulse_intended_velocity_waveform": waveform,
        "impulse_reference_base_velocity_norm": base_norm,
        "impulse_remaining_control_energy_before_step": remaining_energy,
        "impulse_reference_energy_increment": reference_increment,
        "impulse_reference_cumulative_energy": reference_cumulative,
        "impulse_intended_delta_norm": intended_norm,
        "impulse_actual_velocity_basis_coordinate": float(
            coordinates[target_coordinate]
        ),
        "impulse_actual_channel_velocity_coordinate": [
            float(value) for value in coordinates
        ],
        "impulse_intended_signed_exposure": intended_exposure,
        "impulse_actual_signed_exposure": float(
            actual_exposure[target_coordinate]
        ),
        "impulse_actual_channel_exposure": list(actual_exposure),
        "impulse_actual_delta_norm": actual_norm,
        "impulse_actual_projection_scale": (
            0.0 if intended_norm <= 0.0 else actual_norm / intended_norm
        ),
        "impulse_finite_precision_projection_scale": (
            0.0 if selection is None else selection.evaluation.scale
        ),
        "impulse_finite_precision_projection_attempt_count": (
            0 if selection is None else selection.attempt_count
        ),
        "impulse_finite_precision_backoff_count": (
            0 if selection is None else selection.backoff_count
        ),
        "impulse_finite_precision_projection_status": (
            "inactive_exact_noop"
            if selection is None
            else selection.status
        ),
        "impulse_actual_direction_cosine": direction_cosine,
        "impulse_cumulative_control_energy": cumulative_control_energy,
        "impulse_norm_guard_passed": bool(
            actual_norm <= intended_norm
        ),
        "impulse_energy_guard_passed": bool(
            energy_increment <= remaining_energy
        ),
        "impulse_direction_guard_passed": bool(
            not active
            or direction_cosine + 1e-12
            >= float(
                runtime_config["actual_exposure_contract"][
                    "minimum_direction_cosine"
                ]
            )
        ),
        "impulse_inactive_exact_noop": not active,
        "counterfactual_control_active": active,
        "counterfactual_clean_velocity_trace_record_id": clean_step.record[
            "clean_velocity_trace_record_id"
        ],
        "counterfactual_clean_base_velocity_array_sha256": clean_step.record[
            "clean_trace_base_velocity_array_sha256"
        ],
        "counterfactual_state_before_array_sha256": _tensor_digest(
            state_before
        ),
        "counterfactual_state_after_array_sha256": _tensor_digest(
            state_after
        ),
        "counterfactual_update_recomputation_equal": (
            update_recomputation_equal
        ),
        "counterfactual_transformer_forward_call_count": 0,
        "counterfactual_model_feedback_allowed": False,
        "formal_result": False,
        "stage_progression_allowed": False,
        "claim_support_status": CLAIM_SUPPORT_STATUS,
    }
    return _stable_record(record, "counterfactual_step_record_id")


def _run_one_counterfactual(
    scheduler: Any,
    clean_trace: CleanTrace,
    runtime_config: Mapping[str, Any],
    *,
    probe: ImpulseProbePlanRecord,
    basis: ConstructionStageBasis,
    device: Any,
) -> tuple[Any, tuple[Mapping[str, Any], ...], ActualImpulseExposureTrace]:
    """Integrate one signed branch using only the captured clean velocity."""

    import torch

    clone = _clone_flow_scheduler(
        scheduler,
        device=device,
        runtime_config=runtime_config,
    )
    projection_runtime = ImpulseObservabilitySchedulerRuntime(
        clone,
        config=runtime_config,
        plan_record=probe,
        basis=basis,
    )
    state = clean_trace.initial_latent.to(device)
    cumulative_control_energy = 0.0
    cumulative_reference_energy = 0.0
    records: list[dict[str, Any]] = []
    exposure = runtime_config["actual_exposure_contract"]
    schedule = runtime_config["flow_schedule_contract"]
    target_coordinate = int(probe.stage_index) * 2 + int(
        probe.channel_index
    )
    for index, clean_step in enumerate(clean_trace.steps):
        state_before = state
        base_raw = clean_step.base_velocity.to(device)
        base_fp32 = base_raw.detach().float()
        base_norm = float(base_fp32.norm().item())
        delta_sigma = float(schedule["delta_sigma_by_step"][index])
        waveform = float(
            schedule["temporal_waveform_by_macro_interval"][
                int(probe.stage_index)
            ][index]
        )
        control = compute_intended_impulse_control(
            probe_state_update_polarity=probe.polarity,
            temporal_waveform=waveform,
            delta_sigma=delta_sigma,
            base_velocity_norm=base_norm,
            cumulative_control_energy=cumulative_control_energy,
            cumulative_reference_energy=cumulative_reference_energy,
            remaining_step_count=CONSTRUCTION_FLOW_STEP_COUNT - index,
            lambda_max=float(exposure["lambda_max"]),
            velocity_norm_ratio_budget=float(
                exposure["velocity_norm_ratio_budget"]
            ),
            flow_energy_budget_ratio=float(
                exposure["flow_energy_budget_ratio"]
            ),
        )
        reference_increment = delta_sigma**2 * base_norm**2
        remaining_energy = control.remaining_control_energy
        intended_norm = control.intended_delta_norm
        active = abs(waveform) > 1e-15
        selection = None
        coordinates = (0.0,) * STAGE_BASIS_RANK
        constrained = base_raw
        if active:
            if (
                not math.isfinite(intended_norm)
                or intended_norm <= 1e-15
                or not math.isfinite(remaining_energy)
                or remaining_energy <= 0.0
                or base_norm <= 0.0
            ):
                raise RuntimeError(
                    "frozen-feedback active control 不存在可行非零预算: "
                    f"probe={probe.probe_id} step={index} "
                    f"base_norm={base_norm} intended_norm={intended_norm} "
                    f"remaining_energy={remaining_energy}"
                )
            constrained, selection, coordinates = (
                projection_runtime._apply_active_control(
                    base_fp32,
                    target_coordinate=target_coordinate,
                    signed_delta_norm=control.signed_velocity_coordinate,
                    delta_sigma=delta_sigma,
                    norm_budget=intended_norm,
                    remaining_energy=remaining_energy,
                    minimum_direction_cosine=float(
                        exposure["minimum_direction_cosine"]
                    ),
                )
            )
        elif (
            intended_norm != 0.0
            or control.signed_velocity_coordinate != 0.0
            or control.signed_state_update_exposure != 0.0
        ):
            raise RuntimeError("frozen-feedback inactive waveform 不是 exact no-op")

        result = clone.step(
            constrained,
            clone.timesteps[index],
            state,
            return_dict=False,
        )
        state = _scheduler_prev_sample(result)
        dt = clone.sigmas[index + 1] - clone.sigmas[index]
        recomputed = (
            state_before.float() + dt * constrained
        ).to(constrained.dtype)
        update_equal = bool(torch.equal(state, recomputed))
        if not update_equal:
            raise RuntimeError(
                "counterfactual scheduler update 与冻结 Euler 公式不一致"
            )
        energy_increment = (
            0.0
            if selection is None
            else selection.evaluation.energy_increment
        )
        cumulative_reference_energy += reference_increment
        cumulative_control_energy += energy_increment
        if not math.isclose(
            cumulative_reference_energy,
            float(
                clean_step.record[
                    "clean_trace_reference_cumulative_energy"
                ]
            ),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                "counterfactual reference energy 未共享 clean trace"
            )
        record = _counterfactual_step_record(
            probe=probe,
            step_index=index,
            runtime_config=runtime_config,
            clean_step=clean_step,
            state_before=state_before,
            state_after=state,
            waveform=waveform,
            base_norm=base_norm,
            reference_increment=reference_increment,
            reference_cumulative=cumulative_reference_energy,
            cumulative_control_energy=cumulative_control_energy,
            remaining_energy=remaining_energy,
            intended_norm=intended_norm,
            intended_exposure=control.signed_state_update_exposure,
            target_coordinate=target_coordinate,
            coordinates=coordinates,
            selection=selection,
            update_recomputation_equal=update_equal,
        )
        records.append(record)
    projection_runtime.step_records = records
    trace = projection_runtime.build_exposure_trace()
    if trace is None:
        raise AssertionError("signed counterfactual exposure trace 丢失")
    return state.detach().cpu(), tuple(records), trace


def _decode_wan_final_latent(pipe: Any, final_latent: Any) -> np.ndarray:
    """Mirror diffusers 0.35.2 WanPipeline's public VAE decode path."""

    import torch

    if _package_version("diffusers") != "0.35.2":
        raise RuntimeError("frozen-feedback decode 要求 diffusers==0.35.2")
    device = getattr(pipe, "_execution_device", torch.device("cuda"))
    latent = final_latent.to(device=device, dtype=pipe.vae.dtype)
    mean = (
        torch.tensor(pipe.vae.config.latents_mean)
        .view(1, pipe.vae.config.z_dim, 1, 1, 1)
        .to(latent.device, latent.dtype)
    )
    inverse_std = (
        1.0
        / torch.tensor(pipe.vae.config.latents_std)
        .view(1, pipe.vae.config.z_dim, 1, 1, 1)
        .to(latent.device, latent.dtype)
    )
    normalized = latent / inverse_std + mean
    video = pipe.vae.decode(normalized, return_dict=False)[0]
    frames = pipe.video_processor.postprocess_video(
        video,
        output_type="np",
    )
    value = np.asarray(frames[0], dtype=np.float32)
    if (
        value.shape != (33, 320, 512, 3)
        or not np.all(np.isfinite(value))
        or float(np.min(value)) < 0.0
        or float(np.max(value)) > 1.0
    ):
        raise RuntimeError("frozen-feedback decoded output 边界不一致")
    return np.ascontiguousarray(value)


def _read_saved_rgb24_array(video_path: Path) -> np.ndarray:
    import imageio.v3 as iio

    frames = [np.asarray(frame) for frame in iio.imiter(video_path)]
    if not frames:
        raise RuntimeError(f"saved video 没有可读帧: {video_path}")
    value = np.ascontiguousarray(np.stack(frames, axis=0))
    if value.shape != (33, 320, 512, 3) or value.dtype != np.uint8:
        raise RuntimeError("frozen-feedback saved RGB24 shape/dtype 漂移")
    return value


def _artifact_checkpoint_record(
    *,
    probe: ImpulseProbePlanRecord,
    plan_index: int,
    checkpoint_id: str,
    shape: Sequence[int],
    dtype: str,
    array_sha256: str,
    source_path: str,
    source_status: str,
    packaged: bool,
) -> dict[str, Any]:
    record = {
        "record_version": RECORD_VERSION,
        "profile_id": PROFILE_ID,
        "impulse_probe_id": probe.probe_id,
        "impulse_probe_plan_index": plan_index,
        "impulse_transfer_checkpoint_id": checkpoint_id,
        "impulse_transfer_checkpoint_shape": list(shape),
        "impulse_transfer_checkpoint_dtype": dtype,
        "impulse_transfer_checkpoint_array_sha256": array_sha256,
        "impulse_transfer_checkpoint_source_path": source_path,
        "impulse_transfer_checkpoint_source_status": source_status,
        "impulse_transfer_checkpoint_full_array_packaged": packaged,
        "formal_result": False,
        "stage_progression_allowed": False,
        "claim_support_status": CLAIM_SUPPORT_STATUS,
    }
    return _stable_record(
        record,
        "impulse_transfer_checkpoint_record_id",
    )


def _full_array_checkpoint_records(
    *,
    probe: ImpulseProbePlanRecord,
    plan_index: int,
    latent_path: Path,
    latent: np.ndarray,
    decoded_path: Path,
    decoded: np.ndarray,
    saved_rgb24_path: Path,
    saved_rgb24: np.ndarray,
) -> tuple[dict[str, Any], ...]:
    """Bind each full-array digest to the exact array artifact it describes."""

    return (
        _artifact_checkpoint_record(
            probe=probe,
            plan_index=plan_index,
            checkpoint_id="T_final_latent_full",
            shape=latent.shape,
            dtype=str(latent.dtype),
            array_sha256=_array_digest(latent),
            source_path=str(latent_path),
            source_status="ready_full_float32_npz_packaged",
            packaged=True,
        ),
        _artifact_checkpoint_record(
            probe=probe,
            plan_index=plan_index,
            checkpoint_id="T_decoded_full_rgb_float32",
            shape=decoded.shape,
            dtype=str(decoded.dtype),
            array_sha256=_array_digest(decoded),
            source_path=str(decoded_path),
            source_status="ready_full_array_local_until_statistics",
            packaged=False,
        ),
        _artifact_checkpoint_record(
            probe=probe,
            plan_index=plan_index,
            checkpoint_id="T_saved_video_full_rgb24",
            shape=saved_rgb24.shape,
            dtype=str(saved_rgb24.dtype),
            array_sha256=_array_digest(saved_rgb24),
            source_path=str(saved_rgb24_path),
            source_status="ready_actual_rgb24_readback",
            packaged=False,
        ),
    )


def _latent_basis_projection_record(
    *,
    probe: ImpulseProbePlanRecord,
    plan_index: int,
    latent: np.ndarray,
    basis: ConstructionStageBasis,
) -> dict[str, Any]:
    directions = np.asarray(
        effective_construction_basis_directions(basis),
        dtype=np.float64,
    )
    flattened = np.asarray(latent, dtype=np.float64).reshape(-1)
    values = flattened @ directions
    if values.shape != (STAGE_BASIS_RANK,) or not np.all(
        np.isfinite(values)
    ):
        raise RuntimeError("frozen-feedback latent basis6 projection 失败")
    record = {
        "record_version": RECORD_VERSION,
        "profile_id": PROFILE_ID,
        "impulse_probe_id": probe.probe_id,
        "impulse_probe_plan_index": plan_index,
        "frozen_feedback_diagnostic_representation_id": (
            "T_final_latent_six_basis_projection"
        ),
        "frozen_feedback_latent_six_basis_projection_values": (
            values.tolist()
        ),
        "frozen_feedback_latent_l2_norm_in_signed_gate": False,
        "construction_basis_digest": basis.basis_digest,
        "formal_result": False,
        "stage_progression_allowed": False,
        "claim_support_status": CLAIM_SUPPORT_STATUS,
    }
    return _stable_record(
        record,
        "frozen_feedback_latent_basis_projection_record_id",
    )


def _write_generation_snapshots(
    output_root: Path,
    *,
    generation_records: Sequence[Mapping[str, Any]],
    clean_trace_records: Sequence[Mapping[str, Any]],
    transformer_call_records: Sequence[Mapping[str, Any]],
    counterfactual_step_records: Sequence[Mapping[str, Any]],
    exposure_traces: Sequence[ActualImpulseExposureTrace],
    checkpoint_records: Sequence[Mapping[str, Any]],
    latent_basis_projection_records: Sequence[Mapping[str, Any]] = (),
) -> None:
    write_jsonl(
        output_root / "records" / "frozen_feedback_generation_records.jsonl",
        list(generation_records),
    )
    write_jsonl(
        output_root / "records" / "clean_velocity_trace_records.jsonl",
        list(clean_trace_records),
    )
    write_jsonl(
        output_root / "records" / "clean_transformer_call_records.jsonl",
        list(transformer_call_records),
    )
    write_jsonl(
        output_root / "records" / "counterfactual_step_records.jsonl",
        list(counterfactual_step_records),
    )
    write_jsonl(
        output_root / "records" / "counterfactual_exposure_traces.jsonl",
        [asdict(trace) for trace in exposure_traces],
    )
    write_jsonl(
        output_root / "records" / "frozen_feedback_checkpoint_records.jsonl",
        list(checkpoint_records),
    )
    write_jsonl(
        output_root
        / "records"
        / "frozen_feedback_latent_basis_projection_records.jsonl",
        list(latent_basis_projection_records),
    )


def execute_real_frozen_feedback_generation(
    runtime_config: Mapping[str, Any],
    diagnostic_config: Mapping[str, Any],
    *,
    output_root: Path,
    plan: Sequence[ImpulseProbePlanRecord],
    basis: ConstructionStageBasis,
) -> FrozenFeedbackGenerationBatch:
    """Run one clean Wan trajectory and four no-feedback scheduler replays."""

    import torch

    from experiments.generative_video_model_probe.colab_runtime import (
        _export_video,
        _generation_model_provenance_from_pipeline,
        _load_video_generation_pipeline,
        _scheduler_signature,
        _select_dtype,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("frozen-feedback diagnostic 需要 Colab CUDA GPU")
    identity = diagnostic_config["execution_identity"]
    if _package_version("diffusers") != identity["diffusers_version"]:
        raise RuntimeError("frozen-feedback diffusers runtime version 漂移")
    pipe = _load_video_generation_pipeline(
        identity["generation_model_id"],
        _select_dtype(torch),
        revision=identity["generation_model_revision"],
    )
    provenance = _generation_model_provenance_from_pipeline(
        pipe,
        expected_model_id=identity["generation_model_id"],
    )
    if (
        provenance["generation_model_commit_or_hash"]
        != identity["generation_model_revision"]
        or _scheduler_signature(pipe.scheduler)
        != identity["scheduler_signature"]
    ):
        raise RuntimeError("frozen-feedback generation model/scheduler 未冻结")

    generator = torch.Generator(device="cuda").manual_seed(
        int(identity["seed_value"])
    )
    generator_digest = _tensor_digest(generator.get_state())
    runtime = FrozenCleanTraceRuntime(
        pipe,
        config=runtime_config,
        generator_state_digest_random=generator_digest,
    )
    videos_root = output_root / "videos"
    latent_root = output_root / "artifacts" / "final_latents"
    working_root = output_root / ".frozen_feedback_working"
    decoded_root = working_root / "decoded"
    saved_root = working_root / "saved_rgb24"
    clean_trace_root = working_root / "clean_trace"
    for path in (
        videos_root,
        latent_root,
        decoded_root,
        saved_root,
        clean_trace_root,
    ):
        path.mkdir(parents=True, exist_ok=True)

    generation_records: list[dict[str, Any]] = []
    counterfactual_records: list[Mapping[str, Any]] = []
    exposure_traces: list[ActualImpulseExposureTrace] = []
    checkpoints: list[Mapping[str, Any]] = []
    latent_basis_records: list[Mapping[str, Any]] = []
    latent_paths: dict[str, Path] = {}
    decoded_paths: dict[str, Path] = {}
    saved_paths: dict[str, Path] = {}
    final_latents: dict[str, Any] = {}
    started = time.time()
    try:
        with runtime:
            result = pipe(
                prompt=identity["prompt_text"],
                negative_prompt=identity["negative_prompt_text"],
                generator=generator,
                height=int(identity["height"]),
                width=int(identity["width"]),
                num_frames=int(identity["num_frames"]),
                num_inference_steps=int(identity["num_inference_steps"]),
                guidance_scale=float(identity["guidance_scale"]),
                output_type="latent",
            )
            clean_trace = runtime.build_trace()
            returned_clean = result.frames
            if isinstance(returned_clean, (list, tuple)):
                returned_clean = returned_clean[0]
            returned_clean = returned_clean.detach().cpu()
            if (
                returned_clean.shape != clean_trace.final_latent.shape
                or returned_clean.dtype != clean_trace.final_latent.dtype
                or not torch.equal(
                    returned_clean,
                    clean_trace.final_latent,
                )
            ):
                raise RuntimeError(
                    "pipeline returned latent 与 clean scheduler trace 不一致"
                )
            initial_digest = _tensor_digest(clean_trace.initial_latent)
            for step in clean_trace.steps:
                np.save(
                    clean_trace_root
                    / f"{step.step_index:02d}_state_before.npy",
                    step.state_before.float().numpy(),
                    allow_pickle=False,
                )
                np.save(
                    clean_trace_root
                    / f"{step.step_index:02d}_base_velocity.npy",
                    step.base_velocity.float().numpy(),
                    allow_pickle=False,
                )
            clean_replay_digest = _validate_offline_clean_replay(
                pipe.scheduler,
                clean_trace,
                runtime_config,
                device=torch.device("cuda"),
            )
            final_latents["clean"] = clean_trace.final_latent
            for probe in plan[1:]:
                final_latent, branch_records, trace = (
                    _run_one_counterfactual(
                        pipe.scheduler,
                        clean_trace,
                        runtime_config,
                        probe=probe,
                        basis=basis,
                        device=torch.device("cuda"),
                    )
                )
                if _tensor_digest(clean_trace.initial_latent) != initial_digest:
                    raise RuntimeError(
                        "counterfactual shared initial latent 被修改"
                    )
                final_latents[probe.probe_id] = final_latent
                counterfactual_records.extend(branch_records)
                exposure_traces.append(trace)
            if len(runtime._transformer_calls) != (
                EXPECTED_CLEAN_TRANSFORMER_FORWARD_CALL_COUNT
            ):
                raise RuntimeError(
                    "counterfactual 路径意外调用 generation transformer"
                )

            _write_generation_snapshots(
                output_root,
                generation_records=generation_records,
                clean_trace_records=[
                    step.record for step in clean_trace.steps
                ],
                transformer_call_records=(
                    clean_trace.transformer_call_records
                ),
                counterfactual_step_records=counterfactual_records,
                exposure_traces=exposure_traces,
                checkpoint_records=checkpoints,
                latent_basis_projection_records=latent_basis_records,
            )
            for plan_index, probe in enumerate(plan):
                final_latent = final_latents[probe.probe_id]
                latent_array = np.ascontiguousarray(
                    final_latent.float().numpy(),
                    dtype=np.float32,
                )
                if latent_array.shape != CONSTRUCTION_LATENT_LAYOUT_SHAPE:
                    raise RuntimeError("frozen-feedback final latent shape 漂移")
                latent_path = (
                    latent_root / f"{plan_index:02d}_{probe.probe_id}.npz"
                )
                np.savez_compressed(
                    latent_path,
                    final_latent=latent_array,
                )
                latent_paths[probe.probe_id] = latent_path
                latent_basis_records.append(
                    _latent_basis_projection_record(
                        probe=probe,
                        plan_index=plan_index,
                        latent=latent_array,
                        basis=basis,
                    )
                )
                decoded = _decode_wan_final_latent(pipe, final_latent)
                decoded_path = (
                    decoded_root / f"{plan_index:02d}_{probe.probe_id}.npy"
                )
                np.save(decoded_path, decoded, allow_pickle=False)
                decoded_paths[probe.probe_id] = decoded_path
                video_path = (
                    videos_root / f"{plan_index:02d}_{probe.probe_id}.mp4"
                )
                _export_video(decoded, video_path, fps=int(identity["fps"]))
                video_sha = _sha256_file(video_path)
                saved = _read_saved_rgb24_array(video_path)
                saved_path = (
                    saved_root / f"{plan_index:02d}_{probe.probe_id}.npy"
                )
                np.save(saved_path, saved, allow_pickle=False)
                saved_paths[probe.probe_id] = saved_path
                checkpoints.extend(
                    _full_array_checkpoint_records(
                        probe=probe,
                        plan_index=plan_index,
                        latent_path=latent_path,
                        latent=latent_array,
                        decoded_path=decoded_path,
                        decoded=decoded,
                        saved_rgb24_path=saved_path,
                        saved_rgb24=saved,
                    )
                )
                generation_record = {
                    "record_version": RECORD_VERSION,
                    "profile_id": PROFILE_ID,
                    "impulse_probe_id": probe.probe_id,
                    "impulse_probe_plan_index": plan_index,
                    "impulse_probe_role": probe.probe_role,
                    "impulse_stage_index": probe.stage_index,
                    "impulse_state_channel_index": probe.channel_index,
                    "impulse_polarity": probe.polarity,
                    "impulse_nominal_signed_amplitude": (
                        probe.nominal_signed_amplitude
                    ),
                    "generation_status": "success",
                    "generation_failure_reason": "none",
                    "generation_model_id": identity[
                        "generation_model_id"
                    ],
                    "generation_model_revision": identity[
                        "generation_model_revision"
                    ],
                    "scheduler_signature": identity[
                        "scheduler_signature"
                    ],
                    "prompt_id": identity["prompt_id"],
                    "positive_prompt_text_sha256": identity[
                        "positive_prompt_text_sha256"
                    ],
                    "negative_prompt_text_sha256": identity[
                        "negative_prompt_text_sha256"
                    ],
                    "seed_id": identity["seed_id"],
                    "generation_seed_random": identity["seed_value"],
                    "generation_generator_state_digest_random": (
                        generator_digest
                    ),
                    "shared_initial_latent_digest_random": initial_digest,
                    "shared_clean_velocity_trace_digest": (
                        clean_trace.trace_digest
                    ),
                    "clean_logical_denoiser_call_count": (
                        EXPECTED_CLEAN_SCHEDULER_STEP_COUNT
                        if probe.probe_id == "clean"
                        else 0
                    ),
                    "clean_transformer_forward_invocation_count": (
                        EXPECTED_CLEAN_TRANSFORMER_FORWARD_CALL_COUNT
                        if probe.probe_id == "clean"
                        else 0
                    ),
                    "counterfactual_transformer_forward_call_count": 0,
                    "counterfactual_model_feedback_allowed": False,
                    "trajectory_step_count": (
                        EXPECTED_CLEAN_SCHEDULER_STEP_COUNT
                    ),
                    "endpoint_control_enabled": False,
                    "video_path": str(video_path),
                    "video_sha256": video_sha,
                    "final_latent_artifact_path": str(latent_path),
                    "final_latent_artifact_sha256": _sha256_file(
                        latent_path
                    ),
                    "formal_result": False,
                    "stage_progression_allowed": False,
                    "claim_support_status": CLAIM_SUPPORT_STATUS,
                }
                generation_records.append(
                    _stable_record(
                        generation_record,
                        "frozen_feedback_generation_record_id",
                    )
                )
                _write_generation_snapshots(
                    output_root,
                    generation_records=generation_records,
                    clean_trace_records=[
                        step.record for step in clean_trace.steps
                    ],
                    transformer_call_records=(
                        clean_trace.transformer_call_records
                    ),
                    counterfactual_step_records=counterfactual_records,
                    exposure_traces=exposure_traces,
                    checkpoint_records=checkpoints,
                    latent_basis_projection_records=latent_basis_records,
                )
                del decoded
                del saved
                gc.collect()
            if len(runtime._transformer_calls) != (
                EXPECTED_CLEAN_TRANSFORMER_FORWARD_CALL_COUNT
            ):
                raise RuntimeError("VAE decode 路径意外调用 transformer")
        model_call_summary = {
            "clean_logical_denoiser_call_count": (
                EXPECTED_CLEAN_SCHEDULER_STEP_COUNT
            ),
            "clean_transformer_forward_invocation_count": (
                EXPECTED_CLEAN_TRANSFORMER_FORWARD_CALL_COUNT
            ),
            "counterfactual_logical_denoiser_call_count": 0,
            "counterfactual_transformer_forward_call_count": 0,
            "counterfactual_step_record_count": len(
                counterfactual_records
            ),
            "offline_clean_replay_final_latent_sha256": (
                clean_replay_digest
            ),
            "counterfactual_model_feedback_allowed": False,
        }
    except Exception:
        _write_generation_snapshots(
            output_root,
            generation_records=generation_records,
            clean_trace_records=[
                step.record for step in getattr(runtime, "_steps", ())
            ],
            transformer_call_records=getattr(
                runtime,
                "_transformer_calls",
                (),
            ),
            counterfactual_step_records=counterfactual_records,
            exposure_traces=exposure_traces,
            checkpoint_records=checkpoints,
            latent_basis_projection_records=latent_basis_records,
        )
        raise
    finally:
        del pipe
        gc.collect()
        torch.cuda.empty_cache()

    if clean_trace_root.exists():
        shutil.rmtree(clean_trace_root)
    if len(counterfactual_records) != EXPECTED_COUNTERFACTUAL_STEP_COUNT:
        raise RuntimeError("frozen-feedback counterfactual step coverage 漂移")
    return FrozenFeedbackGenerationBatch(
        generation_records=tuple(generation_records),
        clean_trace_records=tuple(
            step.record for step in clean_trace.steps
        ),
        transformer_call_records=clean_trace.transformer_call_records,
        counterfactual_step_records=tuple(counterfactual_records),
        exposure_traces=tuple(exposure_traces),
        checkpoint_records=tuple(checkpoints),
        latent_basis_projection_records=tuple(latent_basis_records),
        latent_paths=latent_paths,
        decoded_paths=decoded_paths,
        saved_rgb_paths=saved_paths,
        model_call_summary=model_call_summary,
        clean_trace_digest=clean_trace.trace_digest,
        initial_latent_digest_random=initial_digest,
    )


def _relabel_feature_batch(
    batch: ConstructionFeatureBatch,
) -> ConstructionFeatureBatch:
    checkpoints: list[dict[str, Any]] = []
    for source in batch.checkpoint_records:
        value = {
            **source,
            "record_version": RECORD_VERSION,
            "profile_id": PROFILE_ID,
            "claim_support_status": CLAIM_SUPPORT_STATUS,
            "formal_result": False,
            "stage_progression_allowed": False,
        }
        value.pop("impulse_transfer_checkpoint_record_id", None)
        value["impulse_transfer_checkpoint_record_id"] = _stable_digest(
            value
        )
        checkpoints.append(value)
    features: list[dict[str, Any]] = []
    for source in batch.feature_records:
        value = {
            **source,
            "record_version": RECORD_VERSION,
            "profile_id": PROFILE_ID,
            "claim_support_status": CLAIM_SUPPORT_STATUS,
            "formal_result": False,
            "stage_progression_allowed": False,
        }
        value.pop("construction_feature_record_id", None)
        value["construction_feature_record_id"] = _stable_digest(value)
        features.append(value)
    return ConstructionFeatureBatch(
        checkpoint_records=tuple(checkpoints),
        feature_records=tuple(features),
    )


def _validate_generation_artifact_checkpoints(
    plan: Sequence[ImpulseProbePlanRecord],
    batch: FrozenFeedbackGenerationBatch,
) -> None:
    """Bind all 15 full-array records to their exact NPZ/NPY artifacts."""

    expected_ids = tuple(probe.probe_id for probe in plan)
    expected_artifacts = {
        "T_final_latent_full": (
            batch.latent_paths,
            CONSTRUCTION_LATENT_LAYOUT_SHAPE,
            "float32",
            "ready_full_float32_npz_packaged",
            True,
        ),
        "T_decoded_full_rgb_float32": (
            batch.decoded_paths,
            (33, 320, 512, 3),
            "float32",
            "ready_full_array_local_until_statistics",
            False,
        ),
        "T_saved_video_full_rgb24": (
            batch.saved_rgb_paths,
            (33, 320, 512, 3),
            "uint8",
            "ready_actual_rgb24_readback",
            False,
        ),
    }
    expected_checkpoint_sequence = tuple(
        (probe.probe_id, checkpoint_id)
        for probe in plan
        for checkpoint_id in expected_artifacts
    )
    observed_checkpoint_sequence = tuple(
        (
            str(record.get("impulse_probe_id") or ""),
            str(record.get("impulse_transfer_checkpoint_id") or ""),
        )
        for record in batch.checkpoint_records
    )
    if observed_checkpoint_sequence != expected_checkpoint_sequence:
        raise RuntimeError(
            "frozen-feedback generation checkpoint identity/order 不一致"
        )
    for record in batch.checkpoint_records:
        probe_id = str(record["impulse_probe_id"])
        checkpoint_id = str(record["impulse_transfer_checkpoint_id"])
        (
            paths,
            expected_shape,
            expected_dtype,
            expected_status,
            expected_packaged,
        ) = expected_artifacts[checkpoint_id]
        source_path = Path(
            str(record.get("impulse_transfer_checkpoint_source_path") or "")
        )
        expected_path = Path(paths[probe_id])
        if checkpoint_id == "T_final_latent_full":
            with np.load(expected_path, allow_pickle=False) as archive:
                array = np.asarray(archive["final_latent"])
        else:
            array = np.load(
                expected_path,
                mmap_mode="r",
                allow_pickle=False,
            )
        expected_record_id = _stable_digest(
            {
                key: value
                for key, value in record.items()
                if key != "impulse_transfer_checkpoint_record_id"
            }
        )
        if (
            record.get("record_version") != RECORD_VERSION
            or record.get("profile_id") != PROFILE_ID
            or record.get("impulse_probe_plan_index")
            != expected_ids.index(probe_id)
            or source_path != expected_path
            or not expected_path.is_file()
            or tuple(record.get("impulse_transfer_checkpoint_shape") or ())
            != expected_shape
            or tuple(array.shape) != expected_shape
            or record.get("impulse_transfer_checkpoint_dtype")
            != expected_dtype
            or str(array.dtype) != expected_dtype
            or not np.all(np.isfinite(array))
            or record.get("impulse_transfer_checkpoint_array_sha256")
            != _array_digest(array)
            or record.get("impulse_transfer_checkpoint_source_status")
            != expected_status
            or record.get(
                "impulse_transfer_checkpoint_full_array_packaged"
            )
            is not expected_packaged
            or record.get("impulse_transfer_checkpoint_record_id")
            != expected_record_id
            or record.get("formal_result") is not False
            or record.get("stage_progression_allowed") is not False
            or record.get("claim_support_status") != CLAIM_SUPPORT_STATUS
        ):
            raise RuntimeError(
                "frozen-feedback generation checkpoint binding 失败: "
                f"{probe_id}/{checkpoint_id}"
            )


def _validate_generation_batch(
    runtime_config: Mapping[str, Any],
    diagnostic_config: Mapping[str, Any],
    plan: Sequence[ImpulseProbePlanRecord],
    batch: FrozenFeedbackGenerationBatch,
    *,
    basis: ConstructionStageBasis,
) -> None:
    expected_ids = tuple(probe.probe_id for probe in plan)
    if expected_ids != FROZEN_OUTPUT_IDS:
        raise RuntimeError("frozen-feedback plan identity 漂移")
    observed_ids = tuple(
        str(record.get("impulse_probe_id") or "")
        for record in batch.generation_records
    )
    if (
        observed_ids != expected_ids
        or len(batch.generation_records) != 5
        or len(batch.clean_trace_records) != 8
        or len(batch.transformer_call_records) != 16
        or len(batch.counterfactual_step_records) != 32
        or len(batch.exposure_traces) != 4
        or len(batch.checkpoint_records) != 15
        or len(batch.latent_basis_projection_records) != 5
    ):
        raise RuntimeError("frozen-feedback generation/count coverage 未就绪")
    identity = diagnostic_config["execution_identity"]
    generator_digests = {
        record.get("generation_generator_state_digest_random")
        for record in batch.generation_records
    }
    initial_digests = {
        record.get("shared_initial_latent_digest_random")
        for record in batch.generation_records
    }
    trace_digests = {
        record.get("shared_clean_velocity_trace_digest")
        for record in batch.generation_records
    }
    if (
        len(generator_digests) != 1
        or generator_digests == {None}
        or initial_digests != {batch.initial_latent_digest_random}
        or trace_digests != {batch.clean_trace_digest}
    ):
        raise RuntimeError("frozen-feedback same initial/clean trace 未绑定")
    for index, (probe, record) in enumerate(
        zip(plan, batch.generation_records, strict=True)
    ):
        video_path = Path(str(record.get("video_path") or ""))
        if (
            record.get("generation_status") != "success"
            or record.get("impulse_probe_plan_index") != index
            or record.get("impulse_probe_role") != probe.probe_role
            or record.get("impulse_stage_index") != probe.stage_index
            or record.get("impulse_state_channel_index")
            != probe.channel_index
            or record.get("impulse_polarity") != probe.polarity
            or record.get("impulse_nominal_signed_amplitude")
            != probe.nominal_signed_amplitude
            or record.get("generation_model_id")
            != identity["generation_model_id"]
            or record.get("generation_model_revision")
            != identity["generation_model_revision"]
            or record.get("scheduler_signature")
            != identity["scheduler_signature"]
            or record.get("prompt_id") != identity["prompt_id"]
            or record.get("positive_prompt_text_sha256")
            != identity["positive_prompt_text_sha256"]
            or record.get("negative_prompt_text_sha256")
            != identity["negative_prompt_text_sha256"]
            or record.get("seed_id") != identity["seed_id"]
            or record.get("generation_seed_random")
            != identity["seed_value"]
            or record.get("trajectory_step_count") != 8
            or record.get("endpoint_control_enabled") is not False
            or record.get("counterfactual_model_feedback_allowed")
            is not False
            or not video_path.is_file()
            or _sha256_file(video_path) != record.get("video_sha256")
        ):
            raise RuntimeError(
                f"frozen-feedback generation identity 失败: {probe.probe_id}"
            )
    _validate_generation_artifact_checkpoints(plan, batch)
    summary = batch.model_call_summary
    if set(summary) != {
        "clean_logical_denoiser_call_count",
        "clean_transformer_forward_invocation_count",
        "counterfactual_logical_denoiser_call_count",
        "counterfactual_transformer_forward_call_count",
        "counterfactual_step_record_count",
        "offline_clean_replay_final_latent_sha256",
        "counterfactual_model_feedback_allowed",
    } or any(
        (
            summary.get("clean_logical_denoiser_call_count") != 8,
            summary.get("clean_transformer_forward_invocation_count") != 16,
            summary.get("counterfactual_logical_denoiser_call_count") != 0,
            summary.get("counterfactual_transformer_forward_call_count") != 0,
            summary.get("counterfactual_step_record_count") != 32,
            summary.get("counterfactual_model_feedback_allowed") is not False,
        )
    ):
        raise RuntimeError("frozen-feedback model call summary 漂移")
    if not str(
        batch.model_call_summary[
            "offline_clean_replay_final_latent_sha256"
        ]
    ):
        raise RuntimeError("offline clean replay digest 缺失")

    clean_sequence = tuple(
        int(record.get("clean_trace_step_index", -1))
        for record in batch.clean_trace_records
    )
    if clean_sequence != tuple(range(8)):
        raise RuntimeError("clean trace step order 不一致")
    expected_calls = tuple(
        (step, component)
        for step in range(8)
        for component in ("conditional", "unconditional")
    )
    observed_calls = tuple(
        (
            int(record.get("clean_logical_denoiser_step_index", -1)),
            str(record.get("clean_cfg_component") or ""),
        )
        for record in batch.transformer_call_records
    )
    if observed_calls != expected_calls or any(
        record.get("counterfactual_transformer_call") is not False
        for record in batch.transformer_call_records
    ):
        raise RuntimeError("clean transformer call binding 不一致")

    expected_steps = tuple(
        (probe.probe_id, index)
        for probe in plan[1:]
        for index in range(8)
    )
    observed_steps = tuple(
        (
            str(record.get("impulse_probe_id") or ""),
            int(record.get("impulse_flow_step_index", -1)),
        )
        for record in batch.counterfactual_step_records
    )
    if observed_steps != expected_steps:
        raise RuntimeError("counterfactual step identity/order 不一致")
    clean_by_step = {
        int(record["clean_trace_step_index"]): record
        for record in batch.clean_trace_records
    }
    for record in batch.counterfactual_step_records:
        index = int(record["impulse_flow_step_index"])
        probe_id = str(record["impulse_probe_id"])
        stage = (
            0 if "early_flow" in probe_id else 2
        )
        expected_active = (
            index in {0, 1, 2, 3}
            if stage == 0
            else index in {6, 7}
        )
        if (
            record.get("counterfactual_transformer_forward_call_count")
            != EXPECTED_COUNTERFACTUAL_TRANSFORMER_FORWARD_CALL_COUNT
            or record.get("counterfactual_model_feedback_allowed")
            is not False
            or record.get("counterfactual_control_active")
            is not expected_active
            or record.get("impulse_inactive_exact_noop")
            is not (not expected_active)
            or record.get("counterfactual_update_recomputation_equal")
            is not True
            or record.get("counterfactual_clean_velocity_trace_record_id")
            != clean_by_step[index]["clean_velocity_trace_record_id"]
            or record.get(
                "counterfactual_clean_base_velocity_array_sha256"
            )
            != clean_by_step[index][
                "clean_trace_base_velocity_array_sha256"
            ]
            or record.get("impulse_norm_guard_passed") is not True
            or record.get("impulse_energy_guard_passed") is not True
            or record.get("impulse_direction_guard_passed") is not True
            or (
                expected_active
                and float(record.get("impulse_actual_delta_norm") or 0.0)
                <= 0.0
            )
            or (
                not expected_active
                and float(record.get("impulse_actual_delta_norm") or 0.0)
                != 0.0
            )
        ):
            raise RuntimeError(
                f"counterfactual guard/trace binding 失败: {probe_id}/{index}"
            )
    expected_trace_ids = tuple(probe.probe_id for probe in plan[1:])
    if (
        tuple(trace.probe_id for trace in batch.exposure_traces)
        != expected_trace_ids
    ):
        raise RuntimeError("counterfactual exposure trace order 不一致")
    for trace in batch.exposure_traces:
        _validate_trace_shape(trace)
        _validate_trace_schedule_and_budget(runtime_config, trace)
        if trace.basis_digest != basis.basis_digest:
            raise RuntimeError("counterfactual exposure basis digest 不一致")


def _validate_feature_batch(
    diagnostic_config: Mapping[str, Any],
    plan: Sequence[ImpulseProbePlanRecord],
    generation: FrozenFeedbackGenerationBatch,
    feature_batch: ConstructionFeatureBatch,
) -> None:
    expected = tuple(probe.probe_id for probe in plan)
    checkpoint_sequence = tuple(
        (
            str(record.get("impulse_probe_id") or ""),
            str(record.get("impulse_transfer_checkpoint_id") or ""),
        )
        for record in feature_batch.checkpoint_records
    )
    expected_checkpoints = tuple(
        (probe_id, checkpoint_id)
        for probe_id in expected
        for checkpoint_id in ("T_reencoded", "T_output_feature")
    )
    feature_ids = tuple(
        str(record.get("impulse_probe_id") or "")
        for record in feature_batch.feature_records
    )
    if (
        checkpoint_sequence != expected_checkpoints
        or feature_ids != expected
        or len(feature_batch.feature_records) != 5
    ):
        raise RuntimeError("frozen-feedback output feature identity/order 不一致")
    schema_digest = diagnostic_config["base_construction_contract"][
        "construction_feature_schema_digest"
    ]
    runtime_config = _runtime_config(diagnostic_config)
    generation_by_id = {
        str(record["impulse_probe_id"]): record
        for record in generation.generation_records
    }
    checkpoint_values: dict[tuple[str, str], np.ndarray] = {}
    checkpoint_status = {
        "T_reencoded": "ready_streaming_vae_reencode",
        "T_output_feature": (
            "ready_governed_per_video_feature_record"
        ),
    }
    for index, record in enumerate(feature_batch.checkpoint_records):
        probe_index, checkpoint_index = divmod(index, 2)
        probe_id = expected[probe_index]
        checkpoint_id = expected_checkpoints[index][1]
        values = np.asarray(
            record.get("impulse_transfer_checkpoint_values"),
            dtype=np.float64,
        )
        expected_record_id = _stable_digest(
            {
                key: value
                for key, value in record.items()
                if key != "impulse_transfer_checkpoint_record_id"
            }
        )
        if (
            checkpoint_index not in (0, 1)
            or record.get("record_version") != RECORD_VERSION
            or record.get("profile_id") != PROFILE_ID
            or record.get("impulse_probe_plan_index") != probe_index
            or record.get("impulse_transfer_checkpoint_dimension") != 256
            or values.shape != (256,)
            or not np.all(np.isfinite(values))
            or record.get("impulse_transfer_checkpoint_source_path")
            != generation_by_id[probe_id]["video_path"]
            or record.get("impulse_transfer_checkpoint_source_status")
            != checkpoint_status[checkpoint_id]
            or record.get("impulse_transfer_checkpoint_record_id")
            != expected_record_id
            or record.get("formal_result") is not False
            or record.get("stage_progression_allowed") is not False
            or record.get("claim_support_status") != CLAIM_SUPPORT_STATUS
        ):
            raise RuntimeError(
                "frozen-feedback feature checkpoint binding 失败: "
                f"{probe_id}/{checkpoint_id}"
            )
        checkpoint_values[(probe_id, checkpoint_id)] = values
    for index, record in enumerate(feature_batch.feature_records):
        probe_id = expected[index]
        values = np.asarray(
            record.get("construction_feature_values"),
            dtype=np.float64,
        )
        expected_row_binding = construction_feature_row_binding_digest(
            probe_id=probe_id,
            feature_schema_digest=schema_digest,
            feature_values=values,
        )
        expected_record_id = _stable_digest(
            {
                key: value
                for key, value in record.items()
                if key != "construction_feature_record_id"
            }
        )
        if (
            record.get("record_version") != RECORD_VERSION
            or record.get("profile_id") != PROFILE_ID
            or record.get("impulse_probe_plan_index") != index
            or record.get("construction_feature_schema_digest")
            != schema_digest
            or record.get("construction_feature_row_identity_binding_status")
            != "ready"
            or record.get("construction_feature_row_binding_digest")
            != expected_row_binding
            or record.get("construction_feature_record_id")
            != expected_record_id
            or record.get("video_path")
            != generation_by_id[probe_id]["video_path"]
            or record.get("video_sha256")
            != generation_by_id[probe_id]["video_sha256"]
            or values.shape != (256,)
            or not np.all(np.isfinite(values))
            or not np.array_equal(
                values,
                checkpoint_values[(probe_id, "T_output_feature")],
            )
            or not math.isclose(
                float(np.linalg.norm(values)),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            or record.get("formal_result") is not False
            or record.get("stage_progression_allowed") is not False
            or record.get("claim_support_status") != CLAIM_SUPPORT_STATUS
        ):
            raise RuntimeError(
                f"frozen-feedback feature binding 失败: {probe_id}"
            )
        validate_output_vae_metadata(
            runtime_config,
            record.get("encoder_metadata") or {},
        )


def _load_npz_latent(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        value = np.asarray(archive["final_latent"], dtype=np.float32)
    if value.shape != CONSTRUCTION_LATENT_LAYOUT_SHAPE:
        raise ValueError("frozen-feedback latent artifact shape 漂移")
    return value


def _load_npy(path: Path) -> np.ndarray:
    return np.load(path, mmap_mode="r", allow_pickle=False)


def _gram_from_paths(
    paths: Mapping[str, Path],
    *,
    loader: Callable[[Path], np.ndarray],
) -> np.ndarray:
    if tuple(paths) != FROZEN_OUTPUT_IDS:
        raise ValueError("frozen-feedback full array row order 不一致")
    gram = np.zeros((5, 5), dtype=np.float64)
    for left_index, left_id in enumerate(FROZEN_OUTPUT_IDS):
        left = np.asarray(loader(paths[left_id])).reshape(-1)
        if not np.all(np.isfinite(left)):
            raise ValueError(f"{left_id} full representation 非有限")
        for right_index in range(left_index, 5):
            right_id = FROZEN_OUTPUT_IDS[right_index]
            right = np.asarray(loader(paths[right_id])).reshape(-1)
            if left.shape != right.shape:
                raise ValueError("frozen-feedback full representation shape 漂移")
            total = 0.0
            for start in range(0, left.size, FULL_ARRAY_CHUNK_SIZE):
                stop = min(start + FULL_ARRAY_CHUNK_SIZE, left.size)
                total += float(
                    np.dot(
                        left[start:stop].astype(np.float64, copy=False),
                        right[start:stop].astype(np.float64, copy=False),
                    )
                )
            gram[left_index, right_index] = total
            gram[right_index, left_index] = total
            del right
        del left
    if not np.array_equal(gram, gram.T):
        raise AssertionError("frozen-feedback Gram 必须 exact symmetric")
    return gram


def _gram_record(
    *,
    checkpoint_id: str,
    gram: np.ndarray,
    checkpoint_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    relevant = [
        record
        for record in checkpoint_records
        if record.get("impulse_transfer_checkpoint_id") == checkpoint_id
    ]
    if tuple(
        str(record.get("impulse_probe_id") or "")
        for record in relevant
    ) != FROZEN_OUTPUT_IDS:
        raise RuntimeError(f"{checkpoint_id} Gram provenance order 不一致")
    record = {
        "record_version": RECORD_VERSION,
        "profile_id": PROFILE_ID,
        "frozen_feedback_full_representation_checkpoint_id": checkpoint_id,
        "frozen_feedback_full_representation_row_ids": list(
            FROZEN_OUTPUT_IDS
        ),
        "frozen_feedback_full_representation_gram_values": gram.tolist(),
        "frozen_feedback_full_representation_gram_shape": [5, 5],
        "frozen_feedback_full_representation_gram_dtype": "float64",
        "frozen_feedback_full_representation_source_array_sha256": {
            str(record["impulse_probe_id"]): str(
                record["impulse_transfer_checkpoint_array_sha256"]
            )
            for record in relevant
        },
        "formal_result": False,
        "stage_progression_allowed": False,
        "claim_support_status": CLAIM_SUPPORT_STATUS,
    }
    return _stable_record(
        record,
        "frozen_feedback_full_representation_gram_record_id",
    )


def _response_record(gate: FrozenFeedbackResponseGate) -> dict[str, Any]:
    stats = gate.statistics
    record = {
        "record_version": RECORD_VERSION,
        "profile_id": PROFILE_ID,
        "frozen_feedback_pair_id": gate.pair_id,
        "frozen_feedback_checkpoint_id": gate.checkpoint_id,
        "frozen_feedback_clean_distance": stats.clean_distance,
        "frozen_feedback_positive_centered_norm": (
            stats.positive_centered_norm
        ),
        "frozen_feedback_negative_centered_norm": (
            stats.negative_centered_norm
        ),
        "frozen_feedback_odd_norm": stats.odd_norm,
        "frozen_feedback_common_norm": stats.common_norm,
        "frozen_feedback_common_odd_ratio": stats.common_odd_ratio,
        "frozen_feedback_antisymmetry_cosine": (
            stats.antisymmetry_cosine
        ),
        "frozen_feedback_antisymmetry_residual": (
            stats.antisymmetry_residual
        ),
        "frozen_feedback_statistics_finite": stats.finite,
        "frozen_feedback_signed_response_gate_passed": (
            gate.signed_response_ready
        ),
        "formal_result": False,
        "stage_progression_allowed": False,
        "claim_support_status": CLAIM_SUPPORT_STATUS,
    }
    return _stable_record(record, "frozen_feedback_response_record_id")


def _ordered_checkpoint_records(
    plan: Sequence[ImpulseProbePlanRecord],
    generation: FrozenFeedbackGenerationBatch,
    features: ConstructionFeatureBatch,
) -> tuple[Mapping[str, Any], ...]:
    generation_map = {
        (
            str(record["impulse_probe_id"]),
            str(record["impulse_transfer_checkpoint_id"]),
        ): record
        for record in generation.checkpoint_records
    }
    feature_map = {
        (
            str(record["impulse_probe_id"]),
            str(record["impulse_transfer_checkpoint_id"]),
        ): record
        for record in features.checkpoint_records
    }
    rows: list[Mapping[str, Any]] = []
    for probe in plan:
        for checkpoint_id in FULL_CHECKPOINT_IDS:
            mapping = (
                generation_map
                if checkpoint_id
                in {
                    "T_final_latent_full",
                    "T_decoded_full_rgb_float32",
                    "T_saved_video_full_rgb24",
                }
                else feature_map
            )
            try:
                rows.append(mapping[(probe.probe_id, checkpoint_id)])
            except KeyError as exc:
                raise RuntimeError(
                    f"frozen-feedback checkpoint 缺失: "
                    f"{probe.probe_id}/{checkpoint_id}"
                ) from exc
    if len(rows) != 25:
        raise AssertionError("frozen-feedback checkpoint count 漂移")
    return tuple(rows)


def _compute_response_gates(
    config: Mapping[str, Any],
    generation: FrozenFeedbackGenerationBatch,
    features: ConstructionFeatureBatch,
    ordered_checkpoints: Sequence[Mapping[str, Any]],
) -> tuple[
    dict[tuple[str, str], FrozenFeedbackResponseGate],
    tuple[Mapping[str, Any], ...],
]:
    epsilon = float(
        config["checkpoint_contract"]["norm_denominator_epsilon"]
    )
    latent_gram = _gram_from_paths(
        generation.latent_paths,
        loader=_load_npz_latent,
    )
    decoded_gram = _gram_from_paths(
        generation.decoded_paths,
        loader=_load_npy,
    )
    saved_gram = _gram_from_paths(
        generation.saved_rgb_paths,
        loader=_load_npy,
    )
    gram_by_id = {
        "T_final_latent_full": latent_gram,
        "T_decoded_full_rgb_float32": decoded_gram,
        "T_saved_video_full_rgb24": saved_gram,
    }
    gram_records = tuple(
        _gram_record(
            checkpoint_id=checkpoint_id,
            gram=gram,
            checkpoint_records=ordered_checkpoints,
        )
        for checkpoint_id, gram in gram_by_id.items()
    )
    feature_checkpoint_values = {
        (
            str(record["impulse_probe_id"]),
            str(record["impulse_transfer_checkpoint_id"]),
        ): np.asarray(
            record["impulse_transfer_checkpoint_values"],
            dtype=np.float64,
        )
        for record in features.checkpoint_records
    }
    gates: dict[tuple[str, str], FrozenFeedbackResponseGate] = {}
    for pair_id in PAIR_IDS:
        for checkpoint_id, gram in gram_by_id.items():
            stats = compute_single_clean_response_statistics_from_gram(
                pair_id=pair_id,
                checkpoint_id=checkpoint_id,
                gram_matrix=gram,
                row_ids=FROZEN_OUTPUT_IDS,
                denominator_epsilon=epsilon,
            )
            gates[(pair_id, checkpoint_id)] = (
                apply_frozen_signed_response_gate(config, stats)
            )
        for checkpoint_id in ("T_reencoded", "T_output_feature"):
            stats = compute_single_clean_response_statistics(
                pair_id=pair_id,
                checkpoint_id=checkpoint_id,
                clean=feature_checkpoint_values[
                    ("clean", checkpoint_id)
                ],
                positive=feature_checkpoint_values[
                    (f"positive_{pair_id}", checkpoint_id)
                ],
                negative=feature_checkpoint_values[
                    (f"negative_{pair_id}", checkpoint_id)
                ],
                denominator_epsilon=epsilon,
            )
            gates[(pair_id, checkpoint_id)] = (
                apply_frozen_signed_response_gate(config, stats)
            )
    return gates, gram_records


def _authorization_false_fields() -> dict[str, bool]:
    return {
        "gate_a_retry": False,
        "gate_a_pass": False,
        "cross_identity_confirmation_allowed": False,
        "gate_b_execution_allowed": False,
        "gate_c_execution_allowed": False,
        "key_selectivity_execution_allowed": False,
        "wrong_key_execution_allowed": False,
        "observer_execution_allowed": False,
        "state_dynamics_design_allowed": False,
        "state_dynamics_execution_allowed": False,
        "f_k_g_k_design_allowed": False,
        "llr_execution_allowed": False,
        "composite_execution_allowed": False,
        "attack_execution_allowed": False,
        "pilot_execution_allowed": False,
        "fixed_fpr_execution_allowed": False,
        "external_baseline_execution_allowed": False,
        "paper_claim_allowed": False,
        "training_or_finetuning_allowed": False,
        "automatic_feature_selection_allowed": False,
        "strength_sweep_allowed": False,
        "grid_sweep_allowed": False,
        "identity_sweep_allowed": False,
        "channel_sweep_allowed": False,
        "automatic_followup_execution_allowed": False,
    }


def run_frozen_feedback_signed_response_diagnostic(
    source_root: str | Path,
    output_root: str | Path,
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    generation_executor: Callable[..., FrozenFeedbackGenerationBatch] = (
        execute_real_frozen_feedback_generation
    ),
    feature_executor: Callable[..., ConstructionFeatureBatch] = (
        extract_real_output_features
    ),
    basis_builder: Callable[[str], ConstructionStageBasis] = (
        build_construction_stage_basis
    ),
) -> dict[str, Any]:
    """Execute the independently authorized construction-only diagnostic."""

    source = Path(source_root).resolve()
    output = Path(output_root).resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output == source or output.is_relative_to(source) or source.is_relative_to(
        output
    ):
        raise ValueError("frozen-feedback source/output 必须双向隔离")
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise ValueError("frozen-feedback output root 必须为空")
    else:
        output.mkdir(parents=True)
    config = load_frozen_feedback_signed_response_config(config_path)
    if config.get("execution_authorized") is not True:
        raise RuntimeError("frozen-feedback diagnostic 尚未获执行授权")
    runtime_config = _runtime_config(config)
    historical_binding = _validate_historical_source(source, config)
    identity = config["execution_identity"]
    if (
        sha256(identity["prompt_text"].encode("utf-8")).hexdigest()
        != identity["positive_prompt_text_sha256"]
        or sha256(identity["negative_prompt_text"].encode("utf-8")).hexdigest()
        != identity["negative_prompt_text_sha256"]
    ):
        raise RuntimeError("frozen-feedback prompt text digest 漂移")
    master_key = os.environ.get("SSTW_TRAJECTORY_AUTHENTICATION_KEY") or ""
    if len(master_key.encode("utf-8")) < int(
        runtime_config["construction_basis"][
            "master_key_minimum_utf8_bytes"
        ]
    ):
        raise RuntimeError("frozen-feedback owner master key 缺失或过短")
    basis = basis_builder(master_key)
    historical_decision = json.loads(
        (
            source
            / "artifacts"
            / "gate_a_root_cause_amplitude_feedback_decision.json"
        ).read_text(encoding="utf-8-sig")
    )
    if basis.basis_digest != historical_decision[
        "construction_basis_digest"
    ]:
        raise RuntimeError("frozen-feedback owner basis 与 historical source 不一致")
    plan = build_frozen_feedback_plan(config)
    write_jsonl(
        output / "records" / "frozen_feedback_generation_plan.jsonl",
        [asdict(record) for record in plan],
    )
    repository_commit = _repository_commit()
    runtime_versions = {
        "python_version": sys.version.split()[0],
        "torch_version": _package_version("torch"),
        "diffusers_version": _package_version("diffusers"),
    }
    generation: FrozenFeedbackGenerationBatch | None = None
    try:
        generation = generation_executor(
            runtime_config,
            config,
            output_root=output,
            plan=plan,
            basis=basis,
        )
        _validate_generation_batch(
            runtime_config,
            config,
            plan,
            generation,
            basis=basis,
        )
        raw_features = feature_executor(
            runtime_config,
            output_root=output,
            plan=plan,
            generation_records=generation.generation_records,
        )
        features = _relabel_feature_batch(raw_features)
        legacy_feature_path = (
            output / "records" / "impulse_feature_records.jsonl"
        )
        legacy_feature_path.unlink(missing_ok=True)
        _validate_feature_batch(config, plan, generation, features)
        ordered_checkpoints = _ordered_checkpoint_records(
            plan,
            generation,
            features,
        )
        gates, gram_records = _compute_response_gates(
            config,
            generation,
            features,
            ordered_checkpoints,
        )
        classification = classify_frozen_feedback_results(
            clean_coverage_and_guards_ready=True,
            gates=gates,
        )
    except Exception as exc:
        working = output / ".frozen_feedback_working"
        if working.exists():
            shutil.rmtree(working)
        failure = {
            "record_version": RECORD_VERSION,
            "profile_id": PROFILE_ID,
            "frozen_feedback_signed_response_diagnostic_decision": (
                "runtime_or_input_validation_failure_stop"
            ),
            "frozen_feedback_failure_reason": str(exc),
            "historical_normal_feedback_gate_a_fail_preserved": True,
            "historical_source_snapshot_digest": historical_binding[
                "source_snapshot_digest"
            ],
            "construction_basis_digest": basis.basis_digest,
            "config_digest": FROZEN_CONFIG_DIGEST,
            "repository_commit": repository_commit,
            "formal_result": False,
            "stage_progression_allowed": False,
            "unique_root_cause_claim_allowed": False,
            "claim_support_status": (
                "failure_recovery_only_not_claim_evidence"
            ),
            **_authorization_false_fields(),
            **runtime_versions,
        }
        write_json(
            output
            / "artifacts"
            / "frozen_feedback_signed_response_decision.json",
            failure,
        )
        write_jsonl(
            output
            / "records"
            / "frozen_feedback_failure_records.jsonl",
            [failure],
        )
        raise

    response_records = tuple(
        _response_record(gates[(pair_id, checkpoint_id)])
        for pair_id in PAIR_IDS
        for checkpoint_id in FULL_CHECKPOINT_IDS
    )
    write_jsonl(
        output / "records" / "frozen_feedback_response_statistics.jsonl",
        response_records,
    )
    write_jsonl(
        output / "records" / "frozen_feedback_full_gram_records.jsonl",
        gram_records,
    )
    write_jsonl(
        output / "records" / "frozen_feedback_checkpoint_records.jsonl",
        ordered_checkpoints,
    )
    write_jsonl(
        output / "records" / "frozen_feedback_feature_records.jsonl",
        features.feature_records,
    )
    write_jsonl(
        output
        / "records"
        / "frozen_feedback_latent_basis_projection_records.jsonl",
        generation.latent_basis_projection_records,
    )
    working = output / ".frozen_feedback_working"
    if working.exists():
        shutil.rmtree(working)
    latent_gates = {
        pair_id: gates[
            (pair_id, "T_final_latent_full")
        ].signed_response_ready
        for pair_id in PAIR_IDS
    }
    post_latent_ready = all(
        gates[(pair_id, checkpoint_id)].signed_response_ready
        for pair_id in PAIR_IDS
        for checkpoint_id in FULL_CHECKPOINT_IDS[1:]
    )
    decision = {
        "record_version": RECORD_VERSION,
        "profile_id": PROFILE_ID,
        "frozen_feedback_signed_response_diagnostic_decision": (
            "construction_candidate_classification_recorded_no_gate_change"
        ),
        "candidate_classification": classification,
        "clean_coverage_and_guards_ready": True,
        "early_full_final_latent_signed": latent_gates[
            "early_flow_channel_0"
        ],
        "late_full_final_latent_signed": latent_gates[
            "late_flow_channel_0"
        ],
        "all_post_latent_checkpoints_signed": post_latent_ready,
        "historical_normal_feedback_full_latent_gate_by_pair": (
            historical_binding[
                "historical_normal_feedback_full_latent_gate_by_pair"
            ]
        ),
        "historical_normal_feedback_gate_a_fail_preserved": True,
        "historical_source_apply_only": True,
        "frozen_feedback_output_count": len(
            generation.generation_records
        ),
        "clean_scheduler_step_count": len(
            generation.clean_trace_records
        ),
        "clean_logical_denoiser_call_count": (
            generation.model_call_summary[
                "clean_logical_denoiser_call_count"
            ]
        ),
        "clean_transformer_forward_invocation_count": (
            generation.model_call_summary[
                "clean_transformer_forward_invocation_count"
            ]
        ),
        "counterfactual_branch_count": len(
            generation.exposure_traces
        ),
        "counterfactual_transformer_forward_call_count": 0,
        "counterfactual_step_record_count": len(
            generation.counterfactual_step_records
        ),
        "shared_clean_velocity_trace_verified": True,
        "same_initial_latent_verified": True,
        "offline_clean_replay_exact_match_verified": True,
        "checkpoint_record_count": len(ordered_checkpoints),
        "feature_record_count": len(features.feature_records),
        "response_statistic_record_count": len(response_records),
        "construction_basis_digest": basis.basis_digest,
        "construction_feature_schema_digest": config[
            "base_construction_contract"
        ]["construction_feature_schema_digest"],
        "impulse_waveform_schema_digest": config[
            "base_construction_contract"
        ]["waveform_schema_digest"],
        "impulse_runtime_adapter_schema_digest": config[
            "base_construction_contract"
        ]["runtime_adapter_schema_digest"],
        "historical_source_snapshot_digest": historical_binding[
            "source_snapshot_digest"
        ],
        "config_digest": FROZEN_CONFIG_DIGEST,
        "repository_commit": repository_commit,
        "unique_root_cause_claim_allowed": False,
        "formal_result": False,
        "stage_progression_allowed": False,
        "claim_support_status": CLAIM_SUPPORT_STATUS,
        **_authorization_false_fields(),
        **runtime_versions,
    }
    write_json(
        output
        / "artifacts"
        / "frozen_feedback_signed_response_decision.json",
        decision,
    )
    manifest = {
        "manifest_kind": MANIFEST_KIND,
        "profile_id": PROFILE_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit": repository_commit,
        "config_path": str(Path(config_path)),
        "config_digest": FROZEN_CONFIG_DIGEST,
        "historical_source_snapshot_digest": historical_binding[
            "source_snapshot_digest"
        ],
        "historical_source_repository_commit": historical_binding[
            "source_repository_commit"
        ],
        "frozen_feedback_plan_ids": list(FROZEN_OUTPUT_IDS),
        "video_sha256_by_probe": {
            str(record["impulse_probe_id"]): str(record["video_sha256"])
            for record in generation.generation_records
        },
        "shared_initial_latent_digest_random": (
            generation.initial_latent_digest_random
        ),
        "shared_clean_velocity_trace_digest": (
            generation.clean_trace_digest
        ),
        "model_call_summary": dict(generation.model_call_summary),
        "full_final_latent_packaged": True,
        "clean_trace_full_tensors_packaged": False,
        "temporary_decoded_and_rgb24_arrays_packaged": False,
        "videos_packaged": True,
        "output_paths": [
            "artifacts/frozen_feedback_signed_response_decision.json",
            "artifacts/final_latents",
            "records/frozen_feedback_generation_plan.jsonl",
            "records/frozen_feedback_generation_records.jsonl",
            "records/clean_velocity_trace_records.jsonl",
            "records/clean_transformer_call_records.jsonl",
            "records/counterfactual_step_records.jsonl",
            "records/counterfactual_exposure_traces.jsonl",
            "records/frozen_feedback_checkpoint_records.jsonl",
            "records/frozen_feedback_feature_records.jsonl",
            "records/frozen_feedback_latent_basis_projection_records.jsonl",
            "records/frozen_feedback_full_gram_records.jsonl",
            "records/frozen_feedback_response_statistics.jsonl",
            "videos",
        ],
        "gate_a_retry": False,
        "gate_a_pass": False,
        "formal_result": False,
        "stage_progression_allowed": False,
        "claim_support_status": CLAIM_SUPPORT_STATUS,
        **runtime_versions,
    }
    write_json(
        output
        / "artifacts"
        / "frozen_feedback_signed_response_manifest.json",
        manifest,
    )
    return decision


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run-root", required=True)
    parser.add_argument("--output-run-root", required=True)
    parser.add_argument(
        "--config-path",
        default=str(DEFAULT_CONFIG_PATH),
    )
    args = parser.parse_args()
    decision = run_frozen_feedback_signed_response_diagnostic(
        args.source_run_root,
        args.output_run_root,
        config_path=args.config_path,
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
