"""验证 B6 sampling-time constraint Colab 入口和 callback adapter。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from main.generation.sampling_constraint_adapter import apply_latent_sampling_constraint
from paper_workflow.notebook_utils.sampling_time_constraint_workflow import (
    build_drive_layout,
    build_drive_packaging_command,
    build_formal_metric_command,
    build_postprocess_command,
    build_result_check_command,
    build_sampling_constraint_colab_runtime_command,
)
from scripts.check_results.sampling_time_constraint_colab_result_checker import check_sampling_time_constraint_colab_results


@pytest.mark.quick
def test_sampling_constraint_adapter_updates_latents_on_active_step() -> None:
    """sampling constraint adapter 在 active lambda step 应提升 latent 对齐度。"""
    import torch

    latents = torch.zeros((1, 2, 4, 4), dtype=torch.float32)
    latents[:, :, :, :] = 0.01
    constraint_config = {
        "constraint_norm_budget": 0.06,
        "constraint_key_id": "test_key",
    }
    schedule_config = {
        "lambda_schedule_id": "constant_weak_constraint",
        "lambda_max": 0.1,
        "lambda_time_window": [0.0, 1.0],
    }

    constrained, record = apply_latent_sampling_constraint(
        latents,
        step_index=1,
        num_steps=4,
        constraint_config=constraint_config,
        schedule_config=schedule_config,
        method_variant="keyed_state_trajectory_constraint",
        key_text="test_key::prompt::seed",
    )

    assert record["constraint_apply_status"] == "applied"
    assert record["latent_alignment_gain"] > 0
    assert not torch.equal(latents, constrained)


@pytest.mark.quick
def test_sampling_time_constraint_colab_workflow_uses_drive_layout() -> None:
    """B6 Colab workflow 必须落盘到 MyDrive/SSTW 的 sampling_time_constraint 子目录。"""
    layout = build_drive_layout()

    assert layout["drive_project_root"] == "/content/drive/MyDrive/SSTW"
    assert layout["drive_run_root"] == "/content/drive/MyDrive/SSTW/runs/sampling_time_constraint_colab"
    assert layout["drive_package_dir"] == "/content/drive/MyDrive/SSTW/packages/sampling_time_constraint"

    runtime_command = build_sampling_constraint_colab_runtime_command(layout, "smoke", "Lightricks/LTX-Video")
    formal_command = build_formal_metric_command(layout)
    postprocess_command = build_postprocess_command(layout)
    result_check_command = build_result_check_command(layout)
    package_command = build_drive_packaging_command(layout)

    assert "experiments.sampling_time_constraint.colab_runtime" in runtime_command
    assert "experiments.generative_video_model_probe.formal_metric_runner" in formal_command
    assert "experiments.sampling_time_constraint.postprocess_runner" in postprocess_command
    assert "scripts/check_results/sampling_time_constraint_colab_result_checker.py" in result_check_command
    assert "scripts/package_results/sampling_time_constraint_drive_packager.py" in package_command


@pytest.mark.quick
def test_sampling_time_constraint_colab_notebook_calls_repository_modules() -> None:
    """B6 Notebook 只能作为入口, 必须调用仓库模块生成正式输出。"""
    notebook_path = Path("paper_workflow/colab_utils/sampling_time_constraint_colab.ipynb")
    assert notebook_path.exists()
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "/content/drive/MyDrive/SSTW" in source
    assert "drive.mount('/content/drive')" in source
    assert "sampling_time_constraint_workflow" in source
    assert "PROFILE = 'smoke'" in source
    assert "HF_TOKEN" in source
    assert "pytest -q" in source
    assert "tools/harness/run_all_audits.py" in source
    assert "build_result_check_command" in source

    helper_text = Path("paper_workflow/notebook_utils/sampling_time_constraint_workflow.py").read_text(encoding="utf-8")
    assert "experiments.sampling_time_constraint.colab_runtime" in helper_text
    assert "experiments.sampling_time_constraint.postprocess_runner" in helper_text
    assert "sampling_time_constraint_colab_result_checker.py" in helper_text
    assert "sampling_time_constraint_drive_packager.py" in helper_text


@pytest.mark.quick
def test_sampling_time_constraint_result_checker_accepts_governed_probe_records(tmp_path: Path) -> None:
    """B6 结果检查器应能基于 governed records 判定 real sampling probe 证据状态。"""
    run_root = tmp_path / "sampling_time_constraint_colab"
    for subdir in ("records", "artifacts", "videos"):
        (run_root / subdir).mkdir(parents=True, exist_ok=True)
    video_path = run_root / "videos" / "sample.mp4"
    video_path.write_bytes(b"sample-video")
    import hashlib

    video_sha256 = hashlib.sha256(video_path.read_bytes()).hexdigest()
    variants = [
        "key_conditioned_state_space_with_trajectory",
        "keyed_state_trajectory_constraint",
        "trajectory_constraint_without_admissibility",
        "trajectory_constraint_without_key_condition",
    ]
    generation_records = []
    trajectory_records = []
    constraint_records = []
    formal_records = []
    summary_records = []
    for index, variant in enumerate(variants):
        trace_id = f"trace_{index}"
        constraint_trace_id = f"constraint_{index}"
        generation_records.append({
            "generation_status": "success",
            "generation_model_id": "test_model",
            "method_variant": variant,
            "prompt_id": "prompt_001",
            "seed_id": "seed_001",
            "trajectory_trace_id": trace_id,
            "constraint_trace_id": constraint_trace_id,
            "video_path": str(video_path),
            "video_sha256": video_sha256,
        })
        trajectory_records.append({"trajectory_trace_id": trace_id, "trajectory_step_index": 0})
        applied = variant == "keyed_state_trajectory_constraint"
        constraint_records.append({
            "constraint_trace_id": constraint_trace_id,
            "trajectory_trace_id": trace_id,
            "method_variant": variant,
            "constraint_apply_status": "applied" if applied else "not_applied",
            "latent_alignment_gain": 0.1 if applied else 0.0,
        })
        formal_records.append({
            "method_variant": variant,
            "formal_visual_quality_ready": True,
            "formal_motion_consistency_ready": True,
            "formal_semantic_consistency_ready": True,
        })
        summary_records.append({
            "method_variant": variant,
            "constraint_record_count": 1,
            "formal_metric_record_count": 1,
        })

    for name, records in {
        "generation_records.jsonl": generation_records,
        "trajectory_trace.jsonl": trajectory_records,
        "constraint_records.jsonl": constraint_records,
        "formal_quality_motion_semantic_records.jsonl": formal_records,
        "constraint_variant_summary_records.jsonl": summary_records,
    }.items():
        (run_root / "records" / name).write_text(
            "\n".join(json.dumps(record) for record in records) + "\n",
            encoding="utf-8",
        )
    (run_root / "artifacts" / "generation_manifest.json").write_text(json.dumps({"artifact_id": "manifest"}), encoding="utf-8")
    (run_root / "artifacts" / "sampling_time_constraint_colab_runtime_decision.json").write_text(
        json.dumps({"implementation_decision": "PASS", "stage_id": "sampling_time_constraint_colab_probe"}),
        encoding="utf-8",
    )
    (run_root / "artifacts" / "sampling_time_constraint_colab_postprocess_decision.json").write_text(
        json.dumps({
            "mechanism_decision": "PASS",
            "mechanism_postprocess_decision": "PASS",
            "details": {
                "keyed_constraint_alignment_gain_mean": 0.1,
                "baseline_alignment_gain_mean": 0.0,
                "formal_quality_semantic_ready": True,
            },
        }),
        encoding="utf-8",
    )
    (run_root / "artifacts" / "formal_quality_motion_semantic_decision.json").write_text(
        json.dumps({"formal_quality_motion_semantic_ready": True, "formal_metric_claim_status": "ready"}),
        encoding="utf-8",
    )

    payload = check_sampling_time_constraint_colab_results(run_root)

    assert payload["implementation_evidence_status"] == "PASS"
    assert payload["mechanism_evidence_status"] == "PASS"
    assert payload["claim_boundary"] == "real_sampling_probe_not_final_b6_submission_claim"
