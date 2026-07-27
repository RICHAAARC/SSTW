from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from hashlib import sha256
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from evaluation.protocol.frame_state_signed_observability_contract import (
    build_public_context_record,
    canonical_json_digest,
    load_frame_state_signed_observability_config,
)
from experiments.generative_video_model_probe.frame_state_signed_observability_construction import (
    CHECKPOINT_IDS,
    FrameStateRuntimeBatch,
    ProbeMeasurement,
    _checkpoint_record,
    _clear_wan_vae_phase_and_release,
    _construct_real_public_atom,
    _generation_record,
    _host_memory_progress_fields,
    _step_record,
    _wan_jacobian_math_sdpa,
    _wan_jacobian_untiled_decode,
    _wan_pipeline_public_atom_offload_state,
    _wan_vae_eval_frozen_parameters,
    run_frame_state_signed_observability_construction,
    torch_jacobian_gram_product,
)
from experiments.generative_video_model_probe.frozen_feedback_signed_response_diagnostic import (
    _run_wan_decode_no_grad,
)
from main.methods.state_space_watermark.frame_state_observability import (
    LATENT_SHAPE,
    apply_frame_state_control_numpy,
    build_flow_schedule,
    build_public_frame_state_atom,
    write_public_frame_state_atom,
)


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


class _FakeWanVaeTiling:
    def __init__(self, *, enabled: bool = True) -> None:
        self.use_tiling = enabled
        self.tile_sample_min_height = 128
        self.tile_sample_min_width = 160
        self.tile_sample_stride_height = 96
        self.tile_sample_stride_width = 112

    def disable_tiling(self) -> None:
        self.use_tiling = False

    def enable_tiling(self, **parameters: int) -> None:
        self.use_tiling = True
        for name, value in parameters.items():
            setattr(self, name, value)


class _FakeParameter:
    def __init__(self, requires_grad: bool) -> None:
        self.requires_grad = requires_grad
        self.grad = None

    def requires_grad_(self, value: bool):
        self.requires_grad = bool(value)
        return self


class _FakeWanVaeState(_FakeWanVaeTiling):
    def __init__(self) -> None:
        super().__init__()
        self.training = True
        self.clear_cache_count = 0
        self._parameters = (
            _FakeParameter(True),
            _FakeParameter(False),
        )

    def parameters(self):
        return iter(self._parameters)

    def eval(self):
        self.training = False
        return self

    def train(self, value: bool = True):
        self.training = bool(value)
        return self

    def clear_cache(self) -> None:
        self.clear_cache_count += 1


class _FakeCuda:
    def __init__(self) -> None:
        self.synchronize_count = 0
        self.empty_cache_count = 0

    def synchronize(self) -> None:
        self.synchronize_count += 1

    def empty_cache(self) -> None:
        self.empty_cache_count += 1


@pytest.mark.quick
def test_jacobian_decode_disables_and_restores_exact_vae_tiling() -> None:
    vae = _FakeWanVaeTiling()
    expected = vars(vae).copy()

    with _wan_jacobian_untiled_decode(vae):
        assert vae.use_tiling is False
        vae.tile_sample_min_height = 999

    assert vars(vae) == expected


@pytest.mark.quick
def test_jacobian_decode_restores_tiling_after_exception() -> None:
    vae = _FakeWanVaeTiling(enabled=False)
    expected = vars(vae).copy()

    with pytest.raises(RuntimeError, match="synthetic"):
        with _wan_jacobian_untiled_decode(vae):
            raise RuntimeError("synthetic")

    assert vars(vae) == expected


@pytest.mark.quick
def test_jacobian_tiling_restore_error_does_not_mask_primary() -> None:
    class _FailingRestoreVae(_FakeWanVaeTiling):
        restore_armed = False

        def disable_tiling(self) -> None:
            super().disable_tiling()
            self.restore_armed = True

        def enable_tiling(self, **kwargs: int) -> None:
            if self.restore_armed:
                raise LookupError("restore failed")
            super().enable_tiling(**kwargs)

    vae = _FailingRestoreVae()

    with pytest.raises(RuntimeError, match="primary") as captured:
        with _wan_jacobian_untiled_decode(vae):
            raise RuntimeError("primary")

    assert any(
        "VAE tiling restore error_type=LookupError" in note
        for note in captured.value.__notes__
    )


@pytest.mark.quick
def test_jacobian_math_sdpa_restores_backend_after_exception() -> None:
    state = {"backend": "flash", "enter": 0, "exit": 0}
    math_backend = object()

    class _Manager:
        def __enter__(self):
            state["enter"] += 1
            state["backend"] = "math"

        def __exit__(self, exc_type, exc, traceback):
            state["exit"] += 1
            state["backend"] = "flash"

    attention = SimpleNamespace(
        SDPBackend=SimpleNamespace(MATH=math_backend),
        sdpa_kernel=lambda backend: (
            _Manager()
            if backend is math_backend
            else pytest.fail("wrong backend")
        ),
    )
    torch_module = SimpleNamespace(
        nn=SimpleNamespace(attention=attention),
    )

    with pytest.raises(RuntimeError, match="synthetic"):
        with _wan_jacobian_math_sdpa(torch_module):
            assert state["backend"] == "math"
            raise RuntimeError("synthetic")

    assert state == {"backend": "flash", "enter": 1, "exit": 1}


@pytest.mark.quick
def test_jacobian_math_sdpa_restore_error_does_not_mask_primary() -> None:
    class _Manager:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, traceback):
            raise LookupError("restore failed")

    torch_module = SimpleNamespace(
        nn=SimpleNamespace(
            attention=SimpleNamespace(
                SDPBackend=SimpleNamespace(MATH="math"),
                sdpa_kernel=lambda backend: _Manager(),
            )
        )
    )

    with pytest.raises(RuntimeError, match="primary") as captured:
        with _wan_jacobian_math_sdpa(torch_module):
            raise RuntimeError("primary")

    assert any(
        "math SDPA restore error_type=LookupError" in note
        for note in captured.value.__notes__
    )


@pytest.mark.quick
def test_vae_eval_parameter_state_restores_after_exception() -> None:
    vae = _FakeWanVaeState()
    expected_requires_grad = tuple(
        parameter.requires_grad for parameter in vae._parameters
    )

    with pytest.raises(RuntimeError, match="synthetic"):
        with _wan_vae_eval_frozen_parameters(vae):
            assert vae.training is False
            assert all(
                parameter.requires_grad is False
                for parameter in vae._parameters
            )
            raise RuntimeError("synthetic")

    assert vae.training is True
    assert tuple(
        parameter.requires_grad for parameter in vae._parameters
    ) == expected_requires_grad
    assert all(parameter.grad is None for parameter in vae._parameters)


@pytest.mark.quick
def test_pipeline_offload_hooks_restore_after_exception() -> None:
    calls: list[str] = []
    pipe = SimpleNamespace(
        remove_all_hooks=lambda: calls.append("remove"),
        enable_model_cpu_offload=lambda: calls.append("enable"),
    )

    with pytest.raises(RuntimeError, match="synthetic"):
        with _wan_pipeline_public_atom_offload_state(pipe):
            raise RuntimeError("synthetic")

    assert calls == ["remove", "enable"]


@pytest.mark.quick
def test_pipeline_restore_error_does_not_mask_primary() -> None:
    def fail_restore() -> None:
        raise LookupError("restore failed")

    pipe = SimpleNamespace(
        remove_all_hooks=lambda: None,
        enable_model_cpu_offload=fail_restore,
    )

    with pytest.raises(RuntimeError, match="primary") as captured:
        with _wan_pipeline_public_atom_offload_state(pipe):
            raise RuntimeError("primary")

    assert any(
        "pipeline offload restore error_type=LookupError" in note
        for note in captured.value.__notes__
    )


@pytest.mark.quick
def test_vae_phase_cleanup_clears_cache_and_releases_cuda() -> None:
    vae = _FakeWanVaeState()
    cuda = _FakeCuda()
    torch_module = SimpleNamespace(cuda=cuda)

    _clear_wan_vae_phase_and_release(
        vae,
        torch_module,
        phase="test",
    )

    assert vae.clear_cache_count == 1
    assert cuda.synchronize_count == 1
    assert cuda.empty_cache_count == 1


@pytest.mark.quick
def test_vae_phase_cleanup_releases_cuda_after_cache_clear_failure() -> None:
    class _FailingCacheVae:
        @staticmethod
        def clear_cache() -> None:
            raise LookupError("cache failed")

    cuda = _FakeCuda()
    torch_module = SimpleNamespace(cuda=cuda)

    with pytest.raises(RuntimeError, match="LookupError"):
        _clear_wan_vae_phase_and_release(
            _FailingCacheVae(),
            torch_module,
            phase="test",
        )

    assert cuda.synchronize_count == 1
    assert cuda.empty_cache_count == 1


@pytest.mark.quick
def test_jacobian_runtime_has_no_double_backward_fallback() -> None:
    source = inspect.getsource(torch_jacobian_gram_product)
    construction_source = inspect.getsource(_construct_real_public_atom)
    decode_source = inspect.getsource(_run_wan_decode_no_grad)

    assert "torch_module.func.jvp" in source
    assert "autograd.functional.jvp" not in source
    assert "save_on_cpu(pin_memory=False)" in source
    assert "_wan_jacobian_math_sdpa(torch)" in construction_source
    assert "iteration_cleanup_cache_clear" in construction_source
    assert "public_atom_outer_finally_cache_clear" in construction_source
    assert "with torch_module.no_grad()" in decode_source
    assert "pipe.maybe_free_model_hooks()" in decode_source


@pytest.mark.quick
def test_jvp_cleanup_error_does_not_mask_forward_ad_failure() -> None:
    class _Tensor:
        dtype = "float32"
        shape = (2,)
        device = "cpu"

        def detach(self):
            return self

    class _Finite:
        def all(self):
            return self

        @staticmethod
        def item() -> bool:
            return True

    def fail_jvp(*args, **kwargs):
        raise ValueError("forward AD unavailable")

    def phase_cleanup(phase: str) -> None:
        if phase == "jvp_post":
            raise LookupError("cache cleanup failed")

    torch_module = SimpleNamespace(
        float32="float32",
        func=SimpleNamespace(jvp=fail_jvp),
        autograd=SimpleNamespace(
            grad=lambda *args, **kwargs: (),
            graph=SimpleNamespace(save_on_cpu=lambda **kwargs: nullcontext()),
        ),
        isfinite=lambda value: _Finite(),
        no_grad=nullcontext,
    )

    with pytest.raises(
        RuntimeError,
        match="phase=true_forward_jvp error_type=ValueError",
    ) as captured:
        torch_jacobian_gram_product(
            lambda value: value,
            _Tensor(),
            _Tensor(),
            torch_module=torch_module,
            phase_cleanup=phase_cleanup,
        )

    assert isinstance(captured.value.__cause__, ValueError)
    assert any(
        "jvp_post cleanup error_type=LookupError" in note
        for note in captured.value.__notes__
    )


@pytest.mark.quick
def test_missing_true_forward_ad_fails_closed_before_execution() -> None:
    tensor = SimpleNamespace(
        dtype="float32",
        shape=(2,),
        device="cpu",
    )
    torch_module = SimpleNamespace(
        float32="float32",
        func=SimpleNamespace(),
    )

    with pytest.raises(RuntimeError, match="UnsupportedForwardAD"):
        torch_jacobian_gram_product(
            lambda value: value,
            tensor,
            tensor,
            torch_module=torch_module,
        )


@pytest.mark.quick
def test_host_memory_progress_is_scalar_only_procfs_diagnostic() -> None:
    fields = _host_memory_progress_fields()

    assert (
        "process_rss_gib=" in fields
        and "host_available_gib=" in fields
    ) or "host_memory_status=unavailable" in fields


def _step_results(coefficient: int, atom: np.ndarray):
    base = np.ones(LATENT_SHAPE, dtype=np.float32)
    shared_zero = np.zeros(LATENT_SHAPE, dtype=np.float32)
    cumulative_control = 0.0
    cumulative_reference = 0.0
    rows = []
    for step in build_flow_schedule():
        result = apply_frame_state_control_numpy(
            base,
            atom,
            signed_state_coefficient=coefficient,
            step=step,
            cumulative_control_energy=cumulative_control,
            cumulative_reference_energy=cumulative_reference,
            remaining_step_count=8 - step.step_index,
        )
        base_norm = float(np.linalg.norm(base.reshape(-1)))
        cumulative_reference += step.delta_sigma**2 * base_norm**2
        cumulative_control += result.energy_increment
        rows.append(
            replace(
                result,
                constrained_velocity=base,
                actual_delta_velocity=(
                    result.actual_delta_velocity
                    if result.actual_delta_norm > 0.0
                    else shared_zero
                ),
            )
        )
    return tuple(rows)


def _fake_runtime(
    config,
    plan,
    output_root: Path,
    *,
    fail_gate: bool = False,
    zero_response: bool = False,
) -> FrameStateRuntimeBatch:
    atom = build_public_frame_state_atom(lambda value: value)
    atom_path = output_root / "artifacts" / "frame_state_public_atom.npz"
    write_public_frame_state_atom(atom_path, atom)
    contexts = []
    context_digests = {}
    for role, nonce in (
        ("construction_identity", "01" * 16),
        ("signed_observability_identity", "02" * 16),
    ):
        context = build_public_context_record(
            config,
            public_nonce_random=nonce,
        )
        digest = canonical_json_digest(context)
        context_digests[role] = digest
        contexts.append(
            {
                "frame_state_identity_role": role,
                "public_context_record": context,
                "context_digest": digest,
            }
        )
    result_by_coefficient = {
        value: _step_results(value, atom.values) for value in (0, 1, -1)
    }
    generation_records = []
    step_records = []
    checkpoint_records = []
    measurements = []
    vector = np.ones(528, dtype=np.float64)
    for probe in plan:
        video = output_root / "videos" / f"{probe.plan_index:02d}.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(f"probe={probe.probe_id}".encode())
        coefficient = probe.signed_state_coefficient
        signed = float(coefficient)
        if zero_response and probe.identity_role == (
            "signed_observability_identity"
        ):
            saved = vector * 0.0
        elif fail_gate and probe.identity_role == "signed_observability_identity":
            saved = vector * (
                1.0 if coefficient > 0 else -0.1 if coefficient < 0 else 0.0
            )
        else:
            saved = vector * signed
        scalar = np.asarray([signed], dtype=np.float64)
        decoded = vector * signed
        steps = result_by_coefficient[coefficient]
        exposure = sum(
            row.signed_state_update_exposure for row in steps
        )
        measurement = ProbeMeasurement(
            plan_index=probe.plan_index,
            identity_role=probe.identity_role,
            probe_id=probe.probe_id,
            signed_state_coefficient=coefficient,
            generator_state_digest_random=(
                "a" * 64
                if probe.identity_role == "construction_identity"
                else "b" * 64
            ),
            public_context_digest=context_digests[probe.identity_role],
            video_path=str(video),
            video_sha256=_sha256_file(video),
            final_latent_projection=scalar,
            decoded_feature=decoded,
            saved_video_feature=saved,
            step_results=steps,
            actual_signed_exposure=exposure,
        )
        measurements.append(measurement)
        generation = _generation_record(
            config=config,
            plan=probe,
            measurement=measurement,
            public_atom_digest=atom.array_digest,
            generation_runtime_sec=0.0,
        )
        generation["frame_state_generation_record_id"] = sha256(
            json.dumps(
                generation,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        generation_records.append(generation)
        step_records.extend(_step_record(probe, row) for row in steps)
        for checkpoint_id, values, boundary in (
            (CHECKPOINT_IDS[0], scalar, "final_latent_float32_before_vae_decode"),
            (
                CHECKPOINT_IDS[1],
                decoded,
                "pre_save_postprocessed_float32_frames",
            ),
            (
                CHECKPOINT_IDS[2],
                saved,
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
    return FrameStateRuntimeBatch(
        public_atom=atom,
        public_atom_path=str(atom_path),
        public_context_records=tuple(contexts),
        measurements=tuple(measurements),
        generation_records=tuple(generation_records),
        step_records=tuple(step_records),
        checkpoint_records=tuple(checkpoint_records),
    )


@pytest.mark.quick
def test_fake_eight_video_runtime_passes_only_design_authorization(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    decision = run_frame_state_signed_observability_construction(
        output,
        runtime_executor=lambda config, plan, root: _fake_runtime(
            config,
            plan,
            root,
        ),
    )

    assert decision["frame_state_gate0_ready"] is True
    assert decision["next_double_window_gate_a_design_allowed"] is True
    assert decision["next_double_window_gate_a_execution_allowed"] is False
    assert decision["formal_result"] is False
    assert decision["stage_progression_allowed"] is False
    assert decision["generation_record_count"] == 8
    assert decision["trajectory_step_record_count"] == 64
    assert decision["checkpoint_record_count"] == 24
    assert (
        output / "artifacts" / "frame_state_public_atom.npz"
    ).is_file()
    assert (
        output / "artifacts" / "frame_state_construction_t0.npy"
    ).is_file()


@pytest.mark.quick
def test_method_gate_failure_is_packaged_as_nonformal_stop(
    tmp_path: Path,
) -> None:
    decision = run_frame_state_signed_observability_construction(
        tmp_path / "run",
        runtime_executor=lambda config, plan, root: _fake_runtime(
            config,
            plan,
            root,
            fail_gate=True,
        ),
    )

    assert decision["frame_state_gate0_ready"] is False
    assert decision["frame_state_gate0_decision"] == (
        "gate0_fail_stop_current_carrier_or_feature"
    )
    assert decision["next_double_window_gate_a_design_allowed"] is False


@pytest.mark.quick
def test_runtime_rejects_measurement_identity_reorder(tmp_path: Path) -> None:
    def reordered(config, plan, output):
        batch = _fake_runtime(config, plan, output)
        return replace(
            batch,
            measurements=tuple(reversed(batch.measurements)),
        )

    with pytest.raises(ValueError, match="identity/order"):
        run_frame_state_signed_observability_construction(
            tmp_path / "run",
            runtime_executor=reordered,
        )


@pytest.mark.quick
def test_runtime_rejects_tampered_governed_step_record(
    tmp_path: Path,
) -> None:
    def tampered(config, plan, output):
        batch = _fake_runtime(config, plan, output)
        rows = [dict(record) for record in batch.step_records]
        rows[0]["frame_state_flow_phase_weight"] = 0.5
        return replace(batch, step_records=tuple(rows))

    with pytest.raises(ValueError, match="governed step records"):
        run_frame_state_signed_observability_construction(
            tmp_path / "run",
            runtime_executor=tampered,
        )


@pytest.mark.quick
def test_runtime_rejects_tampered_checkpoint_source_boundary(
    tmp_path: Path,
) -> None:
    def tampered(config, plan, output):
        batch = _fake_runtime(config, plan, output)
        rows = [dict(record) for record in batch.checkpoint_records]
        rows[0]["frame_state_checkpoint_source_boundary"] = (
            "caller_selected_boundary"
        )
        return replace(batch, checkpoint_records=tuple(rows))

    with pytest.raises(ValueError, match="governed record"):
        run_frame_state_signed_observability_construction(
            tmp_path / "run",
            runtime_executor=tampered,
        )


@pytest.mark.quick
def test_runtime_rejects_generator_state_drift_within_identity(
    tmp_path: Path,
) -> None:
    def tampered(config, plan, output):
        batch = _fake_runtime(config, plan, output)
        measurements = list(batch.measurements)
        measurements[1] = replace(
            measurements[1],
            generator_state_digest_random="f" * 64,
        )
        generation_records = list(batch.generation_records)
        runtime_sec = float(
            generation_records[1]["generation_runtime_sec"]
        )
        generation_records[1] = _generation_record(
            config=config,
            plan=plan[1],
            measurement=measurements[1],
            public_atom_digest=batch.public_atom.array_digest,
            generation_runtime_sec=runtime_sec,
        )
        generation_records[1]["frame_state_generation_record_id"] = (
            sha256(
                json.dumps(
                    generation_records[1],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                ).encode("utf-8")
            ).hexdigest()
        )
        return replace(
            batch,
            measurements=tuple(measurements),
            generation_records=tuple(generation_records),
        )

    with pytest.raises(ValueError, match="相同 generator 初态"):
        run_frame_state_signed_observability_construction(
            tmp_path / "run",
            runtime_executor=tampered,
        )


@pytest.mark.quick
def test_zero_signed_response_is_normal_method_stop_not_runtime_failure(
    tmp_path: Path,
) -> None:
    decision = run_frame_state_signed_observability_construction(
        tmp_path / "run",
        runtime_executor=lambda config, plan, root: _fake_runtime(
            config,
            plan,
            root,
            zero_response=True,
        ),
    )

    assert decision["frame_state_gate0_ready"] is False
    assert decision["frame_state_gate0_decision"] == (
        "gate0_fail_stop_current_carrier_or_feature"
    )
    assert decision["frame_state_transfer_direction_cosine"] is None


@pytest.mark.quick
def test_config_freezes_runner_pending_user_colab_state() -> None:
    config = load_frame_state_signed_observability_config()
    authorization = config["authorization_boundary"]

    assert config["contract_state"] == (
        "gate0_runner_implemented_execution_pending_explicit_user_colab_run"
    )
    assert authorization["runner_implementation_allowed"] is True
    assert authorization["construction_execution_allowed"] is True
    assert authorization["gpu_execution_allowed"] is True
    assert authorization["colab_execution_allowed"] is True
    assert authorization["formal_result"] is False
    assert authorization["stage_progression_allowed"] is False
