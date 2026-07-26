"""Lightweight contract tests for the five-output frozen-feedback diagnostic."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import inspect
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest

from evaluation.protocol.frozen_feedback_signed_response_contract import (
    FROZEN_OUTPUT_IDS,
    FULL_CHECKPOINT_IDS,
    PAIR_IDS,
    FrozenFeedbackResponseGate,
    apply_frozen_signed_response_gate,
    build_frozen_feedback_plan,
    classify_frozen_feedback_results,
    compute_single_clean_response_statistics,
    compute_single_clean_response_statistics_from_gram,
    load_frozen_feedback_signed_response_config,
)
from evaluation.protocol.impulse_observability_contract import (
    build_construction_stage_basis,
    load_impulse_observability_config,
)
from experiments.generative_video_model_probe.frozen_feedback_signed_response_diagnostic import (
    _full_array_checkpoint_records,
    _relabel_feature_batch,
    _stable_digest,
    _tensor_digest,
    _validate_feature_batch,
    _validate_historical_source,
    execute_real_frozen_feedback_generation,
)
from experiments.generative_video_model_probe.output_feature_impulse_observability_construction import (
    ConstructionFeatureBatch,
    apply_numpy_float32_impulse,
    build_output_feature_records,
)
from workflows.colab_test_request import (
    FROZEN_FEEDBACK_SIGNED_RESPONSE_DIAGNOSTIC_TEST_ID,
    REQUEST_SCHEMA_VERSION,
    build_colab_test_dry_run_plan,
    load_colab_test_request,
    run_colab_test_request,
)


REAL_SOURCE = Path("/tmp/sstw_gate_a_root_cause_f06a0934.9OQY2p")


def _config() -> dict:
    return load_frozen_feedback_signed_response_config()


@pytest.mark.quick
def test_frozen_feedback_config_and_plan_are_exact() -> None:
    config = _config()
    plan = build_frozen_feedback_plan(config)
    assert tuple(item.probe_id for item in plan) == FROZEN_OUTPUT_IDS
    assert len(plan) == 5
    assert plan[0].polarity == 0
    assert [(item.stage_index, item.channel_index, item.polarity) for item in plan[1:]] == [
        (0, 0, 1),
        (0, 0, -1),
        (2, 0, 1),
        (2, 0, -1),
    ]
    assert [item.nominal_signed_amplitude for item in plan] == [
        0.0,
        0.06,
        -0.06,
        0.06,
        -0.06,
    ]
    assert config["clean_trace_contract"][
        "clean_transformer_forward_call_count"
    ] == 16
    assert config["clean_trace_contract"]["scheduler_step_count"] == 8
    assert config["clean_trace_contract"][
        "counterfactual_transformer_forward_call_count"
    ] == 0


@pytest.mark.quick
@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("signed_response_gate", "minimum_antisymmetry_cosine"), -999.0),
        (("five_output_plan", "lambda_max"), 0.12),
        (
            ("clean_trace_contract", "counterfactual_transformer_forward_call_count"),
            1,
        ),
        (
            ("authorization_boundary", "unique_root_cause_claim_allowed"),
            True,
        ),
        (
            ("historical_normal_feedback_source", "source_snapshot_digest"),
            "0" * 64,
        ),
        (("execution_identity", "prompt_text"), "mutated prompt"),
    ],
)
def test_frozen_feedback_config_mutations_fail_closed(
    tmp_path: Path,
    path: tuple[str, str],
    value: object,
) -> None:
    config = _config()
    config[path[0]][path[1]] = value
    mutated = tmp_path / "mutated.json"
    mutated.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="预冻结合同"):
        load_frozen_feedback_signed_response_config(mutated)


@pytest.mark.quick
def test_single_clean_statistics_match_five_row_gram() -> None:
    rng = np.random.default_rng(23)
    rows = rng.normal(size=(5, 31))
    gram = rows @ rows.T
    for pair_id in PAIR_IDS:
        plus_index = FROZEN_OUTPUT_IDS.index(f"positive_{pair_id}")
        minus_index = FROZEN_OUTPUT_IDS.index(f"negative_{pair_id}")
        direct = compute_single_clean_response_statistics(
            pair_id=pair_id,
            checkpoint_id="T_final_latent_full",
            clean=rows[0],
            positive=rows[plus_index],
            negative=rows[minus_index],
            denominator_epsilon=1e-15,
        )
        from_gram = compute_single_clean_response_statistics_from_gram(
            pair_id=pair_id,
            checkpoint_id="T_final_latent_full",
            gram_matrix=gram,
            row_ids=FROZEN_OUTPUT_IDS,
            denominator_epsilon=1e-15,
        )
        assert from_gram.clean_distance == 0.0
        assert from_gram.odd_norm == pytest.approx(direct.odd_norm)
        assert from_gram.common_norm == pytest.approx(direct.common_norm)
        assert from_gram.antisymmetry_cosine == pytest.approx(
            direct.antisymmetry_cosine
        )
        assert from_gram.antisymmetry_residual == pytest.approx(
            direct.antisymmetry_residual
        )


def _gate_map(
    *,
    early_latent: bool,
    late_latent: bool,
    post: bool,
) -> dict[tuple[str, str], FrozenFeedbackResponseGate]:
    config = _config()
    passing = compute_single_clean_response_statistics(
        pair_id="placeholder",
        checkpoint_id="placeholder",
        clean=np.zeros(4),
        positive=np.array([1.0, 0.0, 0.0, 0.0]),
        negative=np.array([-1.0, 0.0, 0.0, 0.0]),
        denominator_epsilon=1e-15,
    )
    failing = compute_single_clean_response_statistics(
        pair_id="placeholder",
        checkpoint_id="placeholder",
        clean=np.zeros(4),
        positive=np.ones(4),
        negative=np.ones(4),
        denominator_epsilon=1e-15,
    )
    result: dict[tuple[str, str], FrozenFeedbackResponseGate] = {}
    for pair_id in PAIR_IDS:
        for checkpoint_id in FULL_CHECKPOINT_IDS:
            ready = (
                early_latent
                if pair_id == "early_flow_channel_0"
                and checkpoint_id == "T_final_latent_full"
                else late_latent
                if pair_id == "late_flow_channel_0"
                and checkpoint_id == "T_final_latent_full"
                else post
            )
            source = passing if ready else failing
            stats = replace(
                source,
                pair_id=pair_id,
                checkpoint_id=checkpoint_id,
            )
            result[(pair_id, checkpoint_id)] = (
                apply_frozen_signed_response_gate(config, stats)
            )
    return result


@pytest.mark.quick
@pytest.mark.parametrize(
    ("coverage", "early", "late", "post", "status"),
    [
        (False, True, True, True, "indeterminate_stop"),
        (True, True, True, True, "feedback_isolation_candidate"),
        (True, True, True, False, "multiple_candidates"),
        (True, False, False, False, "stop_current_additive_random_carrier"),
        (True, True, False, False, "indeterminate_stop"),
        (True, False, False, True, "indeterminate_stop"),
    ],
)
def test_frozen_feedback_truth_table(
    coverage: bool,
    early: bool,
    late: bool,
    post: bool,
    status: str,
) -> None:
    result = classify_frozen_feedback_results(
        clean_coverage_and_guards_ready=coverage,
        gates=_gate_map(
            early_latent=early,
            late_latent=late,
            post=post,
        ),
    )
    assert result["classification_status"] == status
    assert result["formal_result"] is False
    assert result["stage_progression_allowed"] is False
    assert result["unique_root_cause_claim_allowed"] is False


@pytest.mark.quick
def test_classifier_recomputes_supplied_gate_boolean() -> None:
    gates = _gate_map(early_latent=True, late_latent=True, post=True)
    identity = ("early_flow_channel_0", "T_final_latent_full")
    gates[identity] = replace(
        gates[identity],
        signed_response_ready=False,
    )
    with pytest.raises(ValueError, match="caller boolean"):
        classify_frozen_feedback_results(
            clean_coverage_and_guards_ready=True,
            gates=gates,
        )


@pytest.mark.quick
def test_real_float32_projection_keeps_frozen_guards() -> None:
    basis = build_construction_stage_basis(
        "frozen-feedback-functional-owner-key"
    )
    direction = np.asarray(basis.values[:, 0], dtype=np.float32)
    base = np.linspace(
        -0.001,
        0.001,
        direction.size,
        dtype=np.float32,
    )
    result = apply_numpy_float32_impulse(
        base,
        direction,
        signed_delta_norm=2.5e-5,
        delta_sigma=-0.052457451820373535,
        norm_budget=2.5e-5,
        remaining_energy=2e-12,
        minimum_direction_cosine=0.999,
    )
    assert result.inactive_noop is False
    assert result.selection is not None
    assert result.selection.evaluation.all_guards_passed
    assert result.selection.evaluation.actual_delta_norm > 0.0


@pytest.mark.quick
def test_real_executor_has_one_clean_pipeline_call_and_no_branch_model_call() -> None:
    source = inspect.getsource(execute_real_frozen_feedback_generation)
    assert source.count("result = pipe(") == 1
    assert "_run_one_counterfactual(" in source
    branch_source = inspect.getsource(
        __import__(
            "experiments.generative_video_model_probe."
            "frozen_feedback_signed_response_diagnostic",
            fromlist=["_run_one_counterfactual"],
        )._run_one_counterfactual
    )
    assert ".transformer" not in branch_source
    assert "clone.step(" in branch_source
    assert "clean_step.base_velocity" in branch_source


@pytest.mark.quick
def test_tensor_digest_materializes_logical_c_order_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTensor:
        def __init__(self, value: np.ndarray) -> None:
            self.value = value
            self.dtype = value.dtype

        def detach(self) -> "FakeTensor":
            return self

        def to(self, *, device: str) -> "FakeTensor":
            assert device == "cpu"
            return self

        def numel(self) -> int:
            return int(self.value.size)

        def reshape(self, *shape: int) -> "FakeTensor":
            return FakeTensor(self.value.reshape(*shape))

        def copy_(self, other: "FakeTensor") -> "FakeTensor":
            np.copyto(self.value, other.value)
            return self

        def view(self, dtype: np.dtype) -> "FakeTensor":
            return FakeTensor(self.value.view(dtype))

        def numpy(self) -> np.ndarray:
            return self.value

    fake_torch = SimpleNamespace(
        uint8=np.dtype(np.uint8),
        empty=lambda size, dtype, device: FakeTensor(
            np.empty(size, dtype=dtype)
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    logical = np.array(
        [[1.0, 2.0], [3.0, 4.0]],
        dtype=np.float32,
    )
    noncontiguous = np.array(
        [[1.0, 3.0], [2.0, 4.0]],
        dtype=np.float32,
    ).T
    assert not noncontiguous.flags.c_contiguous
    expected = sha256(logical.tobytes(order="C")).hexdigest()
    assert _tensor_digest(FakeTensor(logical)) == expected
    assert _tensor_digest(FakeTensor(noncontiguous)) == expected

    singleton_base = np.array([7.0], dtype=np.float32)
    stride_zero_singleton = np.lib.stride_tricks.as_strided(
        singleton_base,
        shape=(1,),
        strides=(0,),
    )
    assert stride_zero_singleton.strides == (0,)
    assert _tensor_digest(FakeTensor(stride_zero_singleton)) == (
        _tensor_digest(
            FakeTensor(np.array([7.0], dtype=np.float32))
        )
    )
    assert _tensor_digest(
        FakeTensor(np.array(7.0, dtype=np.float32))
    ) == _tensor_digest(
        FakeTensor(np.array([7.0], dtype=np.float32))
    )
    assert _tensor_digest(
        FakeTensor(np.array([8.0], dtype=np.float32))
    ) != _tensor_digest(
        FakeTensor(np.array([7.0], dtype=np.float32))
    )


@pytest.mark.quick
def test_full_array_checkpoint_builder_binds_saved_rgb24_npy() -> None:
    probe = build_frozen_feedback_plan(_config())[0]
    records = _full_array_checkpoint_records(
        probe=probe,
        plan_index=0,
        latent_path=Path("/content/artifacts/final_latents/clean.npz"),
        latent=np.zeros((1,), dtype=np.float32),
        decoded_path=Path("/content/work/decoded/clean.npy"),
        decoded=np.zeros((1,), dtype=np.float32),
        saved_rgb24_path=Path("/content/work/saved_rgb24/clean.npy"),
        saved_rgb24=np.zeros((1,), dtype=np.uint8),
    )
    assert tuple(
        record["impulse_transfer_checkpoint_source_path"]
        for record in records
    ) == (
        "/content/artifacts/final_latents/clean.npz",
        "/content/work/decoded/clean.npy",
        "/content/work/saved_rgb24/clean.npy",
    )
    assert records[2]["impulse_transfer_checkpoint_source_path"] != (
        "/content/videos/clean.mp4"
    )


def _valid_feature_batch() -> tuple[
    tuple,
    SimpleNamespace,
    ConstructionFeatureBatch,
]:
    config = _config()
    runtime_config = load_impulse_observability_config(
        config["base_construction_contract"]["config_path"]
    )
    schema = runtime_config["construction_feature_schema"]
    memory = schema["streaming_memory_config"]
    metadata = {
        "endpoint_vae_encode_strategy": (
            "cpu_resident_spatiotemporal_streaming"
        ),
        "endpoint_vae_temporal_chunk_frame_count": memory[
            "temporal_chunk_frame_count"
        ],
        "endpoint_vae_tile_sample_height": memory["tile_sample_height"],
        "endpoint_vae_tile_sample_width": memory["tile_sample_width"],
        "endpoint_vae_tile_sample_stride_height": memory[
            "tile_sample_stride_height"
        ],
        "endpoint_vae_tile_sample_stride_width": memory[
            "tile_sample_stride_width"
        ],
        "endpoint_vae_maximum_incremental_cuda_peak_gib": memory[
            "maximum_incremental_cuda_peak_gib"
        ],
        "endpoint_vae_minimum_cuda_free_gib": memory[
            "minimum_cuda_free_gib"
        ],
        "endpoint_video_frame_count": schema["input_frame_count"],
        "endpoint_vae_model_class": schema["encoder_class"],
        "endpoint_vae_encode_status": "ready",
        "endpoint_latent_shape": [1, 16, 9, 40, 64],
    }
    plan = build_frozen_feedback_plan(config)
    checkpoints: list[dict] = []
    features: list[dict] = []
    generation_records: list[dict] = []
    for index, probe in enumerate(plan):
        video_path = f"/content/videos/{index:02d}_{probe.probe_id}.mp4"
        video_sha = f"{index + 1:064x}"
        latent = np.zeros((1, 16, 9, 40, 64), dtype=np.float32)
        latent.reshape(-1)[index] = 1.0
        reencoded, output, feature = build_output_feature_records(
            runtime_config,
            probe_id=probe.probe_id,
            plan_index=index,
            video_path=video_path,
            video_sha256=video_sha,
            normalized_latent=latent,
            encoder_metadata=metadata,
        )
        checkpoints.extend((reencoded, output))
        features.append(feature)
        generation_records.append(
            {
                "impulse_probe_id": probe.probe_id,
                "video_path": video_path,
                "video_sha256": video_sha,
            }
        )
    batch = _relabel_feature_batch(
        ConstructionFeatureBatch(
            checkpoint_records=tuple(checkpoints),
            feature_records=tuple(features),
        )
    )
    return (
        plan,
        SimpleNamespace(generation_records=tuple(generation_records)),
        batch,
    )


@pytest.mark.quick
def test_feature_checkpoint_values_are_finite_and_governed() -> None:
    plan, generation, batch = _valid_feature_batch()
    _validate_feature_batch(_config(), plan, generation, batch)

    mutated_records = [dict(record) for record in batch.checkpoint_records]
    mutated_records[0]["impulse_transfer_checkpoint_values"][0] = float(
        "nan"
    )
    mutated_records[0]["impulse_transfer_checkpoint_record_id"] = (
        _stable_digest(
            {
                key: value
                for key, value in mutated_records[0].items()
                if key != "impulse_transfer_checkpoint_record_id"
            }
        )
    )
    with pytest.raises(RuntimeError, match="checkpoint binding"):
        _validate_feature_batch(
            _config(),
            plan,
            generation,
            ConstructionFeatureBatch(
                checkpoint_records=tuple(mutated_records),
                feature_records=batch.feature_records,
            ),
        )


@pytest.mark.quick
@pytest.mark.parametrize(
    "mutation",
    ("wrong_dimension", "finite_output_mismatch", "row_binding", "record_id"),
)
def test_feature_checkpoint_and_row_bindings_fail_closed(
    mutation: str,
) -> None:
    plan, generation, batch = _valid_feature_batch()
    checkpoints = [dict(record) for record in batch.checkpoint_records]
    features = [dict(record) for record in batch.feature_records]
    if mutation == "wrong_dimension":
        checkpoints[0]["impulse_transfer_checkpoint_values"] = (
            checkpoints[0]["impulse_transfer_checkpoint_values"][:-1]
        )
        checkpoints[0]["impulse_transfer_checkpoint_dimension"] = 255
        target = checkpoints[0]
        id_field = "impulse_transfer_checkpoint_record_id"
    elif mutation == "finite_output_mismatch":
        checkpoints[1]["impulse_transfer_checkpoint_values"] = list(
            checkpoints[1]["impulse_transfer_checkpoint_values"]
        )
        checkpoints[1]["impulse_transfer_checkpoint_values"][0] += 0.01
        target = checkpoints[1]
        id_field = "impulse_transfer_checkpoint_record_id"
    elif mutation == "row_binding":
        features[0]["construction_feature_row_binding_digest"] = "0" * 64
        target = features[0]
        id_field = "construction_feature_record_id"
    else:
        checkpoints[0]["impulse_transfer_checkpoint_record_id"] = "0" * 64
        target = None
        id_field = ""
    if target is not None:
        target[id_field] = _stable_digest(
            {key: value for key, value in target.items() if key != id_field}
        )
    with pytest.raises(
        (RuntimeError, ValueError),
        match="binding|row binding",
    ):
        _validate_feature_batch(
            _config(),
            plan,
            generation,
            ConstructionFeatureBatch(
                checkpoint_records=tuple(checkpoints),
                feature_records=tuple(features),
            ),
        )


def _write_source_zip(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "generation_status": "success",
        "generation_model_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    }
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(
            "records/impulse_generation_records.jsonl",
            json.dumps(row) + "\n",
        )
        archive.writestr(
            "artifacts/gate_a_root_cause_amplitude_feedback_decision.json",
            "{}\n",
        )


def _request(source_zip: Path) -> dict[str, object]:
    return {
        "request_schema_version": REQUEST_SCHEMA_VERSION,
        "test_id": FROZEN_FEEDBACK_SIGNED_RESPONSE_DIAGNOSTIC_TEST_ID,
        "repository": {
            "url": "https://github.com/RICHAAARC/SSTW.git",
            "ref": "8704717fd498752b99aa3073526721532d6ebbcb",
        },
        "parameters": {
            "phase": "frozen_feedback_diagnostic",
            "run_series_id": "frozen_feedback_signed_response",
            "source_package_path": str(source_zip),
            "resume_package_path": "",
        },
    }


@pytest.mark.quick
def test_colab_allowlist_packages_one_result_and_rejects_resume(
    tmp_path: Path,
) -> None:
    project = tmp_path / "drive" / "SSTW"
    source_zip = project / "inputs" / "historical.zip"
    request_path = project / "requests" / "colab_test_request.json"
    _write_source_zip(source_zip)
    request_path.parent.mkdir(parents=True)
    request_path.write_text(
        json.dumps(_request(source_zip)),
        encoding="utf-8",
    )
    resolved = load_colab_test_request(
        request_path,
        project_root=project,
    )
    assert resolved["phase"] == "frozen_feedback_diagnostic"
    assert build_colab_test_dry_run_plan(
        request_path,
        project_root=project,
    )["test_id"] == FROZEN_FEEDBACK_SIGNED_RESPONSE_DIAGNOSTIC_TEST_ID

    def fake_runner(source_root: Path, output_root: Path) -> dict:
        assert (
            source_root
            / "records"
            / "impulse_generation_records.jsonl"
        ).is_file()
        artifact = output_root / "artifacts" / "decision.json"
        artifact.parent.mkdir(parents=True)
        artifact.write_text('{"formal_result":false}\n', encoding="utf-8")
        return {
            "gate_a_pass": False,
            "formal_result": False,
            "stage_progression_allowed": False,
        }

    result = run_colab_test_request(
        request_path,
        project_root=project,
        repo_root=Path.cwd(),
        local_workspace_root=tmp_path / "content" / "workspace",
        local_package_cache_root=tmp_path / "content" / "packages",
        frozen_feedback_signed_response_runner=fake_runner,
    )
    assert Path(result["drive_result_zip"]).is_file()
    assert Path(result["drive_result_manifest"]).is_file()
    with ZipFile(result["drive_result_zip"]) as archive:
        assert archive.namelist() == ["artifacts/decision.json"]

    payload = _request(source_zip)
    payload["parameters"]["resume_package_path"] = str(source_zip)
    request_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="normal-feedback FAIL"):
        load_colab_test_request(request_path, project_root=project)


@pytest.mark.quick
def test_colab_failure_does_not_publish_frozen_feedback_result(
    tmp_path: Path,
) -> None:
    project = tmp_path / "drive" / "SSTW"
    source_zip = project / "inputs" / "historical.zip"
    request_path = project / "requests" / "colab_test_request.json"
    _write_source_zip(source_zip)
    request_path.parent.mkdir(parents=True)
    request_path.write_text(
        json.dumps(_request(source_zip)),
        encoding="utf-8",
    )

    def failing_runner(source_root: Path, output_root: Path) -> dict:
        partial = output_root / "records" / "partial.jsonl"
        partial.parent.mkdir(parents=True)
        partial.write_text('{"formal_result":false}\n', encoding="utf-8")
        raise RuntimeError("planned frozen-feedback failure")

    with pytest.raises(
        RuntimeError,
        match="planned frozen-feedback failure",
    ):
        run_colab_test_request(
            request_path,
            project_root=project,
            repo_root=Path.cwd(),
            local_workspace_root=tmp_path / "content" / "workspace",
            local_package_cache_root=tmp_path / "content" / "packages",
            frozen_feedback_signed_response_runner=failing_runner,
        )
    result_root = (
        project
        / "diagnostic_tests"
        / FROZEN_FEEDBACK_SIGNED_RESPONSE_DIAGNOSTIC_TEST_ID
    )
    assert not result_root.exists()

    payload = _request(source_zip)
    payload["parameters"]["phase"] = "no_attack"
    request_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="phase 不受支持"):
        load_colab_test_request(request_path, project_root=project)


@pytest.mark.quick
def test_server_cli_dry_run_routes_frozen_feedback_without_gpu(
    tmp_path: Path,
) -> None:
    project = tmp_path / "drive" / "SSTW"
    source_zip = project / "inputs" / "historical.zip"
    request_path = project / "requests" / "colab_test_request.json"
    _write_source_zip(source_zip)
    request_path.parent.mkdir(parents=True)
    request_path.write_text(
        json.dumps(_request(source_zip)),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_generative_video_server_workflow.py",
            "--project-root",
            str(project),
            "--workflow-profile",
            "colab_test",
            "--pipeline",
            "colab_test",
            "--colab-test-request-path",
            str(request_path),
            "--dry-run",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    decision = json.loads(completed.stdout)
    assert decision["server_workflow_decision"] == "DRY_RUN"
    assert decision["pipeline_results"][0]["test_id"] == (
        FROZEN_FEEDBACK_SIGNED_RESPONSE_DIAGNOSTIC_TEST_ID
    )
    assert decision["pipeline_results"][0]["phase"] == (
        "frozen_feedback_diagnostic"
    )


@pytest.mark.integration
@pytest.mark.skipif(
    not REAL_SOURCE.is_dir(),
    reason="真实 f06a0934 complete source 未提供",
)
def test_real_historical_source_is_exactly_bound() -> None:
    result = _validate_historical_source(REAL_SOURCE, _config())
    assert result["source_snapshot_digest"] == (
        "4d5ccdc50394b017948ab10bb677d8d4ddf81cbb21fbc42ab71f36d5ba675b55"
    )
    assert result["historical_normal_feedback_full_latent_gate_by_pair"] == {
        "early_flow_channel_0": False,
        "late_flow_channel_0": False,
    }


@pytest.mark.quick
def test_fixed_notebook_contains_no_frozen_feedback_logic() -> None:
    notebook = Path(
        "paper_workflow/colab_notebooks/colab_test_runner.ipynb"
    ).read_text(encoding="utf-8")
    assert FROZEN_FEEDBACK_SIGNED_RESPONSE_DIAGNOSTIC_TEST_ID not in notebook
    assert "minimum_antisymmetry_cosine" not in notebook
