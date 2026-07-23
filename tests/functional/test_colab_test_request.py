"""验证固定 Colab 测试入口的白名单、续跑和 Drive 打包边界。"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
from zipfile import ZipFile

import pytest

from workflows.colab_test_request import (
    REQUEST_SCHEMA_VERSION,
    TRAJECTORY_REPLAY_SOURCE_BUILD_TEST_ID,
    _safe_extract_zip,
    build_colab_test_dry_run_plan,
    build_colab_test_runtime_preflight_decision,
    load_colab_test_request,
    run_colab_test_request,
)


def _write_source_package(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w") as archive:
        archive.writestr(
            "frozen_run/records/generation_records.jsonl",
            json.dumps({
                "generation_status": "success",
                "generation_model_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            }) + "\n" + json.dumps({
                "generation_status": "success",
                "generation_model_id": "Lightricks/LTX-Video",
            }) + "\n",
        )
        archive.writestr("frozen_run/datasets/prompt_seed_suite.json", "{}\n")


def _request_payload(source_package: Path) -> dict[str, object]:
    return {
        "request_schema_version": REQUEST_SCHEMA_VERSION,
        "test_id": "trajectory_signal_localization_diagnostic",
        "repository": {
            "url": "https://github.com/RICHAAARC/SSTW.git",
            "ref": "main",
        },
        "parameters": {
            "phase": "no_attack",
            "run_series_id": "stage0d_test_001",
            "source_package_path": str(source_package),
            "resume_package_path": "",
        },
    }


def _source_build_request_payload(
    upstream_package: Path,
) -> dict[str, object]:
    return {
        "request_schema_version": REQUEST_SCHEMA_VERSION,
        "test_id": TRAJECTORY_REPLAY_SOURCE_BUILD_TEST_ID,
        "repository": {
            "url": "https://github.com/RICHAAARC/SSTW.git",
            "ref": "main",
        },
        "parameters": {
            "phase": "source_build",
            "run_series_id": "stage0d_source_build_001",
            "source_package_path": str(upstream_package),
            "resume_package_path": "",
        },
    }


def _write_request(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


@pytest.mark.quick
def test_colab_test_request_rejects_non_allowlisted_execution_fields(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "SSTW"
    source_package = project_root / "inputs" / "source.zip"
    request_path = project_root / "requests" / "colab_test_request.json"
    _write_source_package(source_package)
    payload = _request_payload(source_package)
    payload["command"] = ["python", "unreviewed.py"]
    _write_request(request_path, payload)

    with pytest.raises(ValueError, match="未授权字段"):
        load_colab_test_request(request_path, project_root=project_root)

    payload.pop("command")
    payload["test_id"] = "arbitrary_python_module"
    _write_request(request_path, payload)
    with pytest.raises(ValueError, match="不在仓库白名单"):
        load_colab_test_request(request_path, project_root=project_root)

    payload["test_id"] = "trajectory_signal_localization_diagnostic"
    payload["parameters"]["phase"] = "credential_preflight"
    _write_request(request_path, payload)
    with pytest.raises(ValueError, match="phase 不受支持"):
        load_colab_test_request(request_path, project_root=project_root)


@pytest.mark.quick
def test_colab_source_build_request_is_allowlisted_and_has_no_resume(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "SSTW"
    upstream_package = project_root / "method_mechanism_validation" / "source.zip"
    request_path = project_root / "requests" / "colab_test_request.json"
    _write_source_package(upstream_package)
    payload = _source_build_request_payload(upstream_package)
    _write_request(request_path, payload)

    resolved = load_colab_test_request(request_path, project_root=project_root)
    assert resolved["test_id"] == TRAJECTORY_REPLAY_SOURCE_BUILD_TEST_ID
    assert resolved["phase"] == "source_build"
    plan = build_colab_test_dry_run_plan(
        request_path,
        project_root=project_root,
    )
    assert plan["phase"] == "source_build"
    assert plan["claim_support_status"] == (
        "trajectory_replay_source_build_only_not_paper_evidence"
    )

    payload["parameters"]["phase"] = "no_attack"
    _write_request(request_path, payload)
    with pytest.raises(ValueError, match="phase 不受支持"):
        load_colab_test_request(request_path, project_root=project_root)

    payload["parameters"]["phase"] = "source_build"
    payload["parameters"]["resume_package_path"] = str(upstream_package)
    _write_request(request_path, payload)
    with pytest.raises(ValueError, match="不接受 resume package"):
        load_colab_test_request(request_path, project_root=project_root)


@pytest.mark.quick
def test_colab_test_request_rejects_zip_path_traversal(tmp_path: Path) -> None:
    package_path = tmp_path / "unsafe.zip"
    with ZipFile(package_path, "w") as archive:
        archive.writestr("../escape.txt", "blocked")

    with pytest.raises(ValueError, match="不安全路径"):
        _safe_extract_zip(package_path, tmp_path / "extract")
    assert not (tmp_path / "escape.txt").exists()


@pytest.mark.quick
def test_colab_test_lightweight_preflight_checks_only_gpu_and_local_paths(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "drive" / "SSTW"
    decision = build_colab_test_runtime_preflight_decision(
        project_root=project_root,
        local_workspace_root=tmp_path / "content" / "workspace",
        local_package_cache_root=tmp_path / "content" / "packages",
        cuda_available=True,
        hf_home=tmp_path / "content" / "model_cache",
        hf_hub_cache=tmp_path / "content" / "model_cache" / "hub",
    )

    assert decision["runtime_environment_preflight_decision"] == "PASS"
    assert decision["runtime_environment_preflight_failures"] == []
    assert decision["formal_runtime_lock_checked"] is False
    assert "runtime_environment_lock_path" not in decision

    blocked = build_colab_test_runtime_preflight_decision(
        project_root=project_root,
        local_workspace_root=project_root / "workspace",
        local_package_cache_root=project_root / "packages",
        cuda_available=False,
        hf_home=project_root / "model_cache",
        hf_hub_cache=project_root / "model_cache" / "hub",
    )
    assert blocked["runtime_environment_preflight_decision"] == "FAIL"
    assert set(blocked["runtime_environment_preflight_failures"]) == {
        "cuda_unavailable",
        "local_workspace_root_on_drive",
        "local_package_cache_root_on_drive",
        "hf_home_on_drive",
        "hf_hub_cache_on_drive",
    }


@pytest.mark.quick
def test_colab_test_request_packages_and_resumes_without_notebook_changes(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "SSTW"
    source_package = project_root / "inputs" / "source.zip"
    request_path = project_root / "requests" / "colab_test_request.json"
    _write_source_package(source_package)
    payload = _request_payload(source_package)
    _write_request(request_path, payload)

    observed_phases: list[str] = []

    def fake_runner(
        source_root: Path,
        output_root: Path,
        *,
        phase: str,
    ) -> dict[str, object]:
        assert (source_root / "records" / "generation_records.jsonl").is_file()
        assert project_root.resolve() not in output_root.resolve().parents
        drive_result_root = project_root / "diagnostic_tests"
        if phase == "no_attack":
            assert not drive_result_root.exists()
        else:
            manifests = list(
                drive_result_root.rglob("colab_test_package_manifest.json")
            )
            assert len(manifests) == len(observed_phases)
        if phase == "attacked":
            assert (output_root / "artifacts" / "no_attack.json").is_file()
        artifact_path = output_root / "artifacts" / f"{phase}.json"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(json.dumps({"phase": phase}), encoding="utf-8")
        observed_phases.append(phase)
        return {"diagnostic_classification": f"{phase}_complete"}

    first = run_colab_test_request(
        request_path,
        project_root=project_root,
        repo_root=Path.cwd(),
        local_workspace_root=tmp_path / "workspace",
        local_package_cache_root=tmp_path / "cache",
        trajectory_runner=fake_runner,
    )
    first_zip = Path(first["drive_result_zip"])
    first_manifest = Path(first["drive_result_manifest"])
    assert first_zip.is_file()
    assert first_manifest.is_file()
    assert project_root.resolve() in first_zip.resolve().parents

    payload["parameters"]["phase"] = "attacked"
    payload["parameters"]["resume_package_path"] = str(first_zip)
    _write_request(request_path, payload)
    shutil.rmtree(tmp_path / "workspace" / "runs")
    second = run_colab_test_request(
        request_path,
        project_root=project_root,
        repo_root=Path.cwd(),
        local_workspace_root=tmp_path / "workspace",
        local_package_cache_root=tmp_path / "cache",
        trajectory_runner=fake_runner,
    )

    assert observed_phases == ["no_attack", "attacked"]
    assert Path(second["drive_result_zip"]).is_file()
    second_manifest = json.loads(
        Path(second["drive_result_manifest"]).read_text(encoding="utf-8")
    )
    assert second_manifest["resume_package_path"] == str(first_zip)
    assert second_manifest["claim_support_status"] == (
        "diagnostic_only_not_paper_evidence"
    )
    assert second_manifest["generation_model_ids"] == [
        "Lightricks/LTX-Video",
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    ]
    assert not any(
        "hash" in key or "sha" in key or "revision" in key
        for key in second_manifest
    )
    with ZipFile(second["drive_result_zip"]) as archive:
        assert "artifacts/no_attack.json" in archive.namelist()
        assert "artifacts/attacked.json" in archive.namelist()

    payload["parameters"]["phase"] = "decision"
    payload["parameters"]["resume_package_path"] = second["drive_result_zip"]
    _write_request(request_path, payload)
    shutil.rmtree(tmp_path / "workspace" / "runs")
    third = run_colab_test_request(
        request_path,
        project_root=project_root,
        repo_root=Path.cwd(),
        local_workspace_root=tmp_path / "workspace",
        local_package_cache_root=tmp_path / "cache",
        trajectory_runner=fake_runner,
    )
    assert observed_phases == ["no_attack", "attacked", "decision"]
    with ZipFile(third["drive_result_zip"]) as archive:
        assert "artifacts/no_attack.json" in archive.namelist()
        assert "artifacts/attacked.json" in archive.namelist()
        assert "artifacts/decision.json" in archive.namelist()


@pytest.mark.quick
def test_colab_test_failure_does_not_create_drive_result_directory(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "SSTW"
    source_package = project_root / "inputs" / "source.zip"
    request_path = project_root / "requests" / "colab_test_request.json"
    _write_source_package(source_package)
    _write_request(request_path, _request_payload(source_package))

    def failing_runner(
        source_root: Path,
        output_root: Path,
        *,
        phase: str,
    ) -> dict[str, object]:
        raise RuntimeError("planned diagnostic failure")

    with pytest.raises(RuntimeError, match="planned diagnostic failure"):
        run_colab_test_request(
            request_path,
            project_root=project_root,
            repo_root=Path.cwd(),
            local_workspace_root=tmp_path / "workspace",
            local_package_cache_root=tmp_path / "cache",
            trajectory_runner=failing_runner,
        )
    assert not (project_root / "diagnostic_tests").exists()


@pytest.mark.quick
def test_colab_source_build_publishes_only_after_immutable_input_ready(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "SSTW"
    upstream_package = project_root / "method_mechanism_validation" / "source.zip"
    request_path = project_root / "requests" / "colab_test_request.json"
    _write_source_package(upstream_package)
    _write_request(
        request_path,
        _source_build_request_payload(upstream_package),
    )

    def fake_builder(
        cached_upstream: Path,
        output_root: Path,
    ) -> dict[str, object]:
        assert cached_upstream.is_file()
        assert project_root.resolve() not in cached_upstream.resolve().parents
        assert project_root.resolve() not in output_root.resolve().parents
        assert not (project_root / "inputs").exists()
        generation_path = output_root / "records" / "generation_records.jsonl"
        generation_path.parent.mkdir(parents=True, exist_ok=True)
        generation_path.write_text(
            json.dumps({
                "generation_status": "success",
                "generation_model_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            })
            + "\n"
            + json.dumps({
                "generation_status": "success",
                "generation_model_id": "Lightricks/LTX-Video",
            })
            + "\n",
            encoding="utf-8",
        )
        marker = output_root / "artifacts" / "source_build_complete.json"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text('{"status":"complete"}\n', encoding="utf-8")
        return {
            "go_no_go_decision": "NO_GO",
            "go_no_go_reason_category": "trajectory_smoke_signal_not_supported",
        }

    def ready_validator(
        source_root: Path,
        validation_output_root: Path,
    ) -> dict[str, object]:
        assert (source_root / "artifacts" / "source_build_complete.json").is_file()
        assert project_root.resolve() not in validation_output_root.resolve().parents
        assert not (project_root / "inputs").exists()
        return {
            "immutable_input_preflight_status": "ready",
            "immutable_input_scope": "full_replay_diagnostic_inputs",
            "generation_record_count": 12,
            "attack_record_count": 24,
            "likelihood_calibration_input_status": "ready",
        }

    result = run_colab_test_request(
        request_path,
        project_root=project_root,
        repo_root=Path.cwd(),
        local_workspace_root=tmp_path / "workspace",
        local_package_cache_root=tmp_path / "cache",
        trajectory_source_builder=fake_builder,
        trajectory_source_validator=ready_validator,
    )

    expected_root = project_root / "inputs" / "trajectory_signal_localization"
    assert Path(result["drive_result_zip"]) == expected_root / "stage0d_source.zip"
    assert Path(result["drive_result_manifest"]) == (
        expected_root / "stage0d_source_manifest.json"
    )
    with ZipFile(result["drive_result_zip"]) as archive:
        assert "artifacts/source_build_complete.json" in archive.namelist()
    manifest = json.loads(
        Path(result["drive_result_manifest"]).read_text(encoding="utf-8")
    )
    assert manifest["generation_record_count"] == 12
    assert manifest["attack_record_count"] == 24
    assert manifest["likelihood_calibration_input_status"] == "ready"
    assert manifest["generation_model_ids"] == [
        "Lightricks/LTX-Video",
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    ]
    assert manifest["diagnostic_decision"]["go_no_go_decision"] == "NO_GO"
    assert not any(
        "hash" in key or "sha" in key or "revision" in key
        for key in manifest
    )


@pytest.mark.quick
def test_colab_source_build_validation_failure_does_not_publish_to_drive(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "SSTW"
    upstream_package = project_root / "method_mechanism_validation" / "source.zip"
    request_path = project_root / "requests" / "colab_test_request.json"
    _write_source_package(upstream_package)
    _write_request(
        request_path,
        _source_build_request_payload(upstream_package),
    )

    def fake_builder(
        cached_upstream: Path,
        output_root: Path,
    ) -> dict[str, object]:
        generation_path = output_root / "records" / "generation_records.jsonl"
        generation_path.parent.mkdir(parents=True, exist_ok=True)
        generation_path.write_text("\n", encoding="utf-8")
        return {"go_no_go_decision": "NO_GO"}

    def incomplete_validator(
        source_root: Path,
        validation_output_root: Path,
    ) -> dict[str, object]:
        return {
            "immutable_input_preflight_status": "ready",
            "immutable_input_scope": "full_replay_diagnostic_inputs",
            "generation_record_count": 12,
            "attack_record_count": 23,
            "likelihood_calibration_input_status": "ready",
        }

    with pytest.raises(RuntimeError, match="输入快照未就绪"):
        run_colab_test_request(
            request_path,
            project_root=project_root,
            repo_root=Path.cwd(),
            local_workspace_root=tmp_path / "workspace",
            local_package_cache_root=tmp_path / "cache",
            trajectory_source_builder=fake_builder,
            trajectory_source_validator=incomplete_validator,
        )
    assert not (project_root / "inputs").exists()


@pytest.mark.quick
def test_colab_test_server_cli_dry_run_uses_request_without_gpu(tmp_path: Path) -> None:
    project_root = tmp_path / "SSTW"
    source_package = project_root / "inputs" / "source.zip"
    request_path = project_root / "requests" / "colab_test_request.json"
    _write_source_package(source_package)
    _write_request(request_path, _request_payload(source_package))

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_generative_video_server_workflow.py",
            "--project-root",
            str(project_root),
            "--workflow-profile",
            "colab_test",
            "--pipeline",
            "colab_test",
            "--colab-test-request-path",
            str(request_path),
            "--model-revision",
            "1" * 40,
            "--dry-run",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    decision = json.loads(completed.stdout)
    assert decision["server_workflow_decision"] == "DRY_RUN"
    assert decision["pipeline"] == "colab_test"
    assert decision["workflow_profile"] == "colab_test"
    assert decision["runtime_environment_preflight"][
        "runtime_environment_preflight_kind"
    ] == "colab_test_lightweight"
    assert decision["resolved_main_generation_model_revision"] is None
    assert decision["resolved_cross_generation_model_revision"] is None
    assert decision["claim_support_status"] == "diagnostic_only_not_paper_evidence"
    assert decision["pipeline_results"][0]["phase"] == "no_attack"


@pytest.mark.quick
def test_colab_source_build_server_cli_dry_run_uses_same_notebook_handler(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "SSTW"
    upstream_package = project_root / "method_mechanism_validation" / "source.zip"
    request_path = project_root / "requests" / "colab_test_request.json"
    _write_source_package(upstream_package)
    _write_request(
        request_path,
        _source_build_request_payload(upstream_package),
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_generative_video_server_workflow.py",
            "--project-root",
            str(project_root),
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
    assert decision["claim_support_status"] == (
        "trajectory_replay_source_build_only_not_paper_evidence"
    )
    assert decision["pipeline_results"][0]["test_id"] == (
        TRAJECTORY_REPLAY_SOURCE_BUILD_TEST_ID
    )
    assert decision["pipeline_results"][0]["phase"] == "source_build"


@pytest.mark.quick
def test_colab_test_notebook_is_stable_thin_server_cli_entrypoint() -> None:
    notebook_path = Path(
        "paper_workflow/colab_notebooks/colab_test_runner.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert "requests/colab_test_request.json" in source
    assert "SERVER_PIPELINE = 'colab_test'" in source
    assert "WORKFLOW_PROFILE = 'colab_test'" in source
    assert "WORKFLOW_PROFILE = 'method_mechanism_validation'" not in source
    assert "--colab-test-request-path" in source
    assert "scripts/run_generative_video_server_workflow.py" in source
    assert "result = run_streaming_command(server_command)" in source
    assert "from workflows.streaming_command import run_streaming_command" in source
    assert "%pip install --requirement requirements/paper_runtime_lock.txt" in source
    assert "drive.mount('/content/drive')" in source
    assert "MODEL_CACHE_ROOT = '/content/SSTW_model_cache'" in source
    assert "os.environ['HF_HOME'] = MODEL_CACHE_ROOT" in source
    assert "os.environ['HF_HUB_CACHE']" in source
    assert "from experiments" not in source
    assert "def " not in source
    assert "write_json(" not in source
    assert "write_jsonl(" not in source
