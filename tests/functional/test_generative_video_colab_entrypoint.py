"""验证 B5 Colab Notebook 入口、Drive 落盘与 prompt suite 构造。"""

from __future__ import annotations

import json
from pathlib import Path
import zipfile

import pytest

from paper_workflow.notebook_utils.generative_video_model_probe_workflow import build_drive_layout
from scripts.package_results.generative_video_drive_packager import package_generative_video_colab_run
from scripts.prepare_generative_video_prompt_suite import write_prompt_suite
from main.protocol.record_writer import write_json, write_jsonl


@pytest.mark.quick
def test_prepare_generative_video_prompt_suite_is_separate_from_runtime(tmp_path: Path) -> None:
    """prompt suite 构造必须独立于 GPU 模型测试运行。"""
    output_root = tmp_path / "prompt_suite"
    summary = write_prompt_suite(output_root)

    suite_path = Path(summary["prompt_suite_path"])
    manifest_path = Path(summary["manifest_path"])
    assert suite_path.exists()
    assert manifest_path.exists()

    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    assert suite["dataset_construction_status"] == "constructed"
    assert suite["dataset_source"] == "repository_deterministic_prompt_seed_spec"
    assert suite["prompts"]
    assert suite["seeds"]


@pytest.mark.quick
def test_generative_video_colab_notebook_calls_repository_modules() -> None:
    """Notebook 只能作为入口, 必须调用仓库脚本或 experiments 模块生成正式输出。"""
    notebook_path = Path("paper_workflow/colab_utils/generative_video_model_probe_colab.ipynb")
    assert notebook_path.exists()
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "/content/drive/MyDrive/SSTW" in source
    assert "drive.mount('/content/drive')" in source
    assert "HF_TOKEN" in source
    assert "getpass" in source
    assert "add_to_git_credential=False" in source
    assert "generative_video_model_probe_workflow" in source
    assert "PROFILE = 'recommended'" in source
    assert "build_formal_metric_command" in source
    assert "build_mechanism_postprocess_command" in source
    assert "scripts/prepare_generative_video_prompt_suite.py" in Path("paper_workflow/notebook_utils/generative_video_model_probe_workflow.py").read_text(encoding="utf-8")
    assert "experiments.generative_video_model_probe.colab_runtime" in Path("paper_workflow/notebook_utils/generative_video_model_probe_workflow.py").read_text(encoding="utf-8")
    assert "experiments.generative_video_model_probe.formal_metric_runner" in Path("paper_workflow/notebook_utils/generative_video_model_probe_workflow.py").read_text(encoding="utf-8")
    assert "experiments.generative_video_model_probe.postprocess_runner" in Path("paper_workflow/notebook_utils/generative_video_model_probe_workflow.py").read_text(encoding="utf-8")
    assert "scripts/package_results/generative_video_drive_packager.py" in Path("paper_workflow/notebook_utils/generative_video_model_probe_workflow.py").read_text(encoding="utf-8")
    assert "pytest -q" in source
    assert "tools/harness/run_all_audits.py" in source


@pytest.mark.quick
def test_generative_video_drive_layout_uses_sstw_drive_root() -> None:
    """Colab workflow 的长期输出必须默认落盘到 MyDrive/SSTW 子目录。"""
    layout = build_drive_layout()
    assert layout["drive_project_root"] == "/content/drive/MyDrive/SSTW"
    assert layout["drive_dataset_root"].startswith("/content/drive/MyDrive/SSTW/datasets/")
    assert layout["drive_run_root"].startswith("/content/drive/MyDrive/SSTW/runs/")
    assert layout["drive_package_dir"].startswith("/content/drive/MyDrive/SSTW/packages/")
    assert layout["drive_log_dir"].startswith("/content/drive/MyDrive/SSTW/logs/")


@pytest.mark.quick
def test_generative_video_drive_packager_creates_archive_and_manifest(tmp_path: Path) -> None:
    """Drive packager 必须从已有 run outputs 生成 zip 和 package manifest。"""
    run_root = tmp_path / "runs" / "generative_video_model_probe_colab"
    package_dir = tmp_path / "packages"
    write_jsonl(run_root / "records" / "generation_records.jsonl", [{"generation_model_id": "model", "prompt_id": "prompt"}])
    write_json(run_root / "artifacts" / "generative_video_colab_runtime_decision.json", {"stage_id": "generative_video_model_probe_colab_runtime", "implementation_decision": "PASS", "mechanism_decision": "FAIL"})
    write_json(run_root / "artifacts" / "generation_manifest.json", {"artifact_id": "manifest"})

    payload = package_generative_video_colab_run(run_root, package_dir, include_videos=False)
    archive_path = Path(payload["archive_path"])
    manifest_path = Path(payload["package_manifest_path"])
    assert archive_path.exists()
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["generation_manifest_status"] == "present"
    assert manifest["decision_summary"]["implementation_decision"] == "PASS"
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
    assert any(name.endswith("records/generation_records.jsonl") for name in names)


@pytest.mark.quick
def test_generative_video_colab_runtime_uses_optional_hf_token_without_recording_secret() -> None:
    """模型加载应支持 HF_TOKEN, 但只能记录 provided / not_provided 状态。"""
    runtime_text = Path("experiments/generative_video_model_probe/colab_runtime.py").read_text(encoding="utf-8")
    assert "os.environ.get(\"HF_TOKEN\")" in runtime_text
    assert "token=hf_token" in runtime_text
    assert "hf_token_status" in runtime_text
    assert "provided" in runtime_text
    assert "not_provided" in runtime_text
    assert 'default="recommended"' in runtime_text
