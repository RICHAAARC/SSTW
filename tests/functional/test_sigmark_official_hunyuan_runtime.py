"""验证 SIGMark 官方 Hunyuan gen->extract 项目内运行器的轻量逻辑。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from external_baseline.sigmark_official_hunyuan_runtime import (
    SigmarkOfficialHunyuanRuntimeConfig,
    normalize_sigmark_precision,
    run_sigmark_official_hunyuan_runtime,
    validate_sigmark_watermark_geometry,
    write_sigmark_official_bundle_records,
)


def _write_json(path: Path, payload: object) -> None:
    """写出测试 JSON 文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_runtime_records(run_root: Path) -> None:
    """构造最小 runtime detection records fixture。"""

    records_path = run_root / "records" / "runtime_detection_records.jsonl"
    records_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "runtime_detection_status": "ready",
            "generation_model_id": "wan21_runtime",
            "prompt_id": "prompt_a",
            "seed_id": "seed_0",
            "trajectory_trace_id": "trace_a",
            "attack_name": "clean",
            "source_video_path": str(run_root / "videos" / "source.mp4"),
            "attacked_video_path": str(run_root / "videos" / "attacked.mp4"),
        }
    ]
    records_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def _write_prompt_suite(path: Path) -> None:
    """构造包含 prompt_text 的最小 prompt suite。"""

    _write_json(
        path,
        {
            "prompt_suite_id": "test_prompt_suite",
            "prompts": [
                {
                    "prompt_id": "prompt_a",
                    "prompt_text": "A small red toy car moves across a table with clear motion.",
                }
            ],
        },
    )


def _write_fake_sigmark_source(source_dir: Path) -> None:
    """构造只用于 dry-run 文本改写的伪 SIGMark 官方源码结构。"""

    (source_dir / "watermarks").mkdir(parents=True, exist_ok=True)
    (source_dir / "watermarks" / "sigmark.py").write_text("# fake sigmark watermark\n", encoding="utf-8")
    (source_dir / "apply_disturbances.py").write_text("# fake disturbance script\n", encoding="utf-8")
    (source_dir / "main.py").write_text(
        "\n".join(
            [
                "def generate_videos(args, dimension, prompt):",
                '        image_prompt = load_image(os.path.join(args.image_prompt_dir, dimension, prompt[:180] + "-0.png")) \\',
                "            if args.image_prompt_dir is not None else None",
                "    return image_prompt",
                "def extract(args):",
                "        if args.disturbance_info:",
                "            valid_index = ['existing']",
                "        else:",
                "            valid_index = None",
                "        return valid_index",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


@pytest.mark.quick
def test_sigmark_hunyuan_runtime_dry_run_builds_prompt_set_and_commands(tmp_path: Path) -> None:
    """dry-run 必须只写运行计划, 不触发重型 Hunyuan 生成。"""

    run_root = tmp_path / "runs" / "generative_video_model_probe" / "validation_scale"
    bundle_root = tmp_path / "bundles" / "validation_scale"
    source_dir = tmp_path / "official_source"
    prompt_suite_path = tmp_path / "datasets" / "generative_video_prompt_suite" / "prompt_seed_suite.json"
    _write_runtime_records(run_root)
    _write_prompt_suite(prompt_suite_path)
    _write_fake_sigmark_source(source_dir)

    config = SigmarkOfficialHunyuanRuntimeConfig(
        run_root=str(run_root),
        bundle_root=str(bundle_root),
        source_dir=str(source_dir),
        output_root=str(tmp_path / "sigmark_runtime"),
        resource_root=str(tmp_path / "resources" / "external_baseline"),
        prompt_suite_path=str(prompt_suite_path),
        model_base_path=str(tmp_path / "resources" / "external_baseline" / "sigmark" / "models"),
        dry_run=True,
    )

    manifest = run_sigmark_official_hunyuan_runtime(config)

    assert manifest["execution_status"] == "dry_run_planned"
    assert manifest["generated_bundle_record_count"] == 0
    assert manifest["geometry_manifest"]["geometry_status"] == "ready"
    assert manifest["geometry_manifest"]["width"] == 512
    assert manifest["geometry_manifest"]["height"] == 512
    assert manifest["geometry_manifest"]["latent_w"] == 64
    assert manifest["prompt_manifest"]["prompt_count"] == 1
    assert manifest["patch_manifest"]["patch_status"] == "patched_runtime_copy"
    assert "--mode=gen" in manifest["gen_command"]
    assert "--mode=extract" in manifest["extract_command"]
    assert "--precision=bf16" in manifest["gen_command"]
    assert "--precision=bfloat16" not in manifest["gen_command"]
    prompt_file = Path(manifest["prompt_manifest"]["prompt_file"])
    assert "toy car moves" in prompt_file.read_text(encoding="utf-8")
    runtime_main = Path(manifest["runtime_source_dir"]) / "main.py"
    runtime_main_text = runtime_main.read_text(encoding="utf-8")
    assert '"I2V" in args.model_name' in runtime_main_text
    assert "valid_index = [None] * len(sample_names)" in runtime_main_text
    assert {
        row["patch_name"]: row["patch_status"] for row in manifest["patch_manifest"]["patch_results"]
    } == {
        "t2v_image_prompt_load_guard": "patched_runtime_copy",
        "extract_valid_index_none_guard": "patched_runtime_copy",
    }


@pytest.mark.quick
def test_sigmark_bundle_writer_records_project_owned_provenance(tmp_path: Path) -> None:
    """SIGMark bit accuracy npz 必须转成 project-owned official bundle, 而非外部补交结果。"""

    run_root = tmp_path / "runs" / "generative_video_model_probe" / "validation_scale"
    bundle_root = tmp_path / "bundles" / "validation_scale"
    manifest_path = bundle_root / "sigmark" / "official_reference_execution_manifest.json"
    npz_path = tmp_path / "official_outputs" / "HunyuanVideo-community-sigmark-bit_accuracy.npz"
    _write_runtime_records(run_root)
    _write_json(manifest_path, {"manifest_kind": "test_sigmark_execution_manifest"})
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, **{"sstw_runtime_prompt/example": np.array([0.75, 1.0])})

    result = write_sigmark_official_bundle_records(
        run_root=run_root,
        bundle_root=bundle_root,
        manifest_path=manifest_path,
        bit_accuracy_npz_path=npz_path,
        model_name="HunyuanVideo-community",
    )

    assert result["generated_bundle_record_count"] == 1
    record_path = bundle_root / "sigmark" / "records" / "prompt_a__seed_0__clean.json"
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    assert payload["official_result_provenance"] == "repository_generated_from_third_party_official_code"
    assert payload["external_baseline_official_execution_mode"] == "sigmark_hunyuan_gen_extract"
    assert "metric_status" not in payload
    assert payload["bit_accuracy"] == 0.875
    assert payload["official_execution_manifest_path"] == str(manifest_path)


@pytest.mark.quick
def test_sigmark_hunyuan_runtime_writes_governed_failure_manifest_when_model_missing(tmp_path: Path) -> None:
    """模型缺失时运行器必须写出失败 manifest, 而不是伪造 baseline 分数。"""

    run_root = tmp_path / "runs" / "generative_video_model_probe" / "validation_scale"
    bundle_root = tmp_path / "bundles" / "validation_scale"
    source_dir = tmp_path / "official_source"
    prompt_suite_path = tmp_path / "datasets" / "generative_video_prompt_suite" / "prompt_seed_suite.json"
    _write_runtime_records(run_root)
    _write_prompt_suite(prompt_suite_path)
    _write_fake_sigmark_source(source_dir)

    config = SigmarkOfficialHunyuanRuntimeConfig(
        run_root=str(run_root),
        bundle_root=str(bundle_root),
        source_dir=str(source_dir),
        output_root=str(tmp_path / "sigmark_runtime"),
        resource_root=str(tmp_path / "resources" / "external_baseline"),
        prompt_suite_path=str(prompt_suite_path),
        model_base_path=str(tmp_path / "resources" / "external_baseline" / "sigmark" / "models"),
        auto_download_hf_model=False,
        dry_run=False,
    )

    manifest = run_sigmark_official_hunyuan_runtime(config)

    assert manifest["execution_status"] == "failed"
    assert manifest["generated_bundle_record_count"] == 0
    assert manifest["failed_bundle_record_count"] == 1
    assert "sigmark_hunyuan_model_missing" in manifest["execution_failure_reason"]
    record_path = bundle_root / "sigmark" / "records" / "prompt_a__seed_0__clean.json"
    assert not record_path.exists()


@pytest.mark.quick
def test_sigmark_precision_normalization_matches_official_cli_choices() -> None:
    """项目运行器必须把 PyTorch dtype 名称转换为官方 CLI precision token。"""

    assert normalize_sigmark_precision("bfloat16") == "bf16"
    assert normalize_sigmark_precision("float16") == "fp16"
    assert normalize_sigmark_precision("float32") == "fp32"
    assert normalize_sigmark_precision("bf16") == "bf16"
    with pytest.raises(ValueError, match="sigmark_precision_invalid"):
        normalize_sigmark_precision("torch.bfloat16")


@pytest.mark.quick
def test_sigmark_watermark_geometry_rejects_invalid_720_width() -> None:
    """运行器必须在加载 Hunyuan 前阻断 720x1280 + hw_factor=8 这类官方非法组合。"""

    with pytest.raises(ValueError, match="latent_hw_not_divisible_by_hw_factor"):
        validate_sigmark_watermark_geometry(
            width=720,
            height=1280,
            num_frames=61,
            ch_factor=1,
            hw_factor=8,
            fr_factor=4,
        )

    manifest = validate_sigmark_watermark_geometry(
        width=512,
        height=512,
        num_frames=65,
        ch_factor=2,
        hw_factor=8,
        fr_factor=1,
    )
    assert manifest["geometry_status"] == "ready"
    assert manifest["latent_w"] == 64
    assert manifest["latent_h"] == 64
