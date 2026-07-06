"""验证 SIGMark 官方 Hunyuan gen->extract 项目内运行器的轻量逻辑。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import external_baseline.sigmark_official_hunyuan_runtime as sigmark_runtime
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


def _write_runtime_records_with_prompt_count(run_root: Path, prompt_count: int) -> None:
    """构造多个 prompt 的 runtime detection records fixture。"""

    records_path = run_root / "records" / "runtime_detection_records.jsonl"
    records_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "runtime_detection_status": "ready",
            "generation_model_id": "wan21_runtime",
            "prompt_id": f"prompt_{index}",
            "seed_id": "seed_0",
            "trajectory_trace_id": f"trace_{index}",
            "attack_name": "clean",
            "source_video_path": str(run_root / "videos" / f"source_{index}.mp4"),
            "attacked_video_path": str(run_root / "videos" / f"attacked_{index}.mp4"),
        }
        for index in range(prompt_count)
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


def _write_prompt_suite_with_prompt_count(path: Path, prompt_count: int) -> None:
    """构造多个 prompt_text 的 prompt suite。"""

    _write_json(
        path,
        {
            "prompt_suite_id": "test_prompt_suite",
            "prompts": [
                {
                    "prompt_id": f"prompt_{index}",
                    "prompt_text": f"Prompt {index} contains clear visible motion for SIGMark testing.",
                }
                for index in range(prompt_count)
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
    assert "--watermark_method=none" in manifest["clean_negative_gen_command"]
    assert "--watermark_method=sigmark" in manifest["clean_negative_extract_command"]
    assert "official_hunyuan_clean_negative_outputs" in " ".join(manifest["clean_negative_gen_command"])
    assert "--precision=bf16" in manifest["gen_command"]
    assert "--precision=bfloat16" not in manifest["gen_command"]
    assert "--num_prompts_per_dimension=1" in manifest["gen_command"]
    assert "--num_prompts_diversity=1" in manifest["gen_command"]
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
def test_sigmark_hunyuan_runtime_sets_official_prompt_limit_to_runtime_prompt_count(tmp_path: Path) -> None:
    """SIGMark 命令必须覆盖官方默认 5 prompt 截断, 否则后段 prompt 会缺少视频。"""

    run_root = tmp_path / "runs" / "generative_video_model_probe" / "validation_scale"
    bundle_root = tmp_path / "bundles" / "validation_scale"
    source_dir = tmp_path / "official_source"
    prompt_suite_path = tmp_path / "datasets" / "generative_video_prompt_suite" / "prompt_seed_suite.json"
    _write_runtime_records_with_prompt_count(run_root, prompt_count=8)
    _write_prompt_suite_with_prompt_count(prompt_suite_path, prompt_count=8)
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

    assert manifest["prompt_manifest"]["prompt_count"] == 8
    assert manifest["official_num_prompts_per_dimension"] == 8
    for command_name in (
        "gen_command",
        "extract_command",
        "clean_negative_gen_command",
        "clean_negative_extract_command",
    ):
        assert "--num_prompts_per_dimension=8" in manifest[command_name]
        assert "--num_prompts_diversity=8" in manifest[command_name]


@pytest.mark.quick
def test_sigmark_bundle_writer_records_project_owned_provenance(tmp_path: Path) -> None:
    """SIGMark bit accuracy npz 必须转成 project-owned official bundle, 而非外部补交结果。"""

    run_root = tmp_path / "runs" / "generative_video_model_probe" / "validation_scale"
    bundle_root = tmp_path / "bundles" / "validation_scale"
    manifest_path = bundle_root / "sigmark" / "official_reference_execution_manifest.json"
    npz_path = tmp_path / "official_outputs" / "HunyuanVideo-community-sigmark-bit_accuracy.npz"
    clean_npz_path = tmp_path / "official_outputs" / "HunyuanVideo-community-sigmark-clean-negative-bit_accuracy.npz"
    _write_runtime_records(run_root)
    _write_json(manifest_path, {"manifest_kind": "test_sigmark_execution_manifest"})
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, **{"sstw_runtime_prompt/example": np.array([0.75, 1.0])})
    np.savez(clean_npz_path, **{"sstw_runtime_prompt/example_clean": np.array([0.25, 0.5])})

    result = write_sigmark_official_bundle_records(
        run_root=run_root,
        bundle_root=bundle_root,
        manifest_path=manifest_path,
        bit_accuracy_npz_path=npz_path,
        model_name="HunyuanVideo-community",
        clean_negative_bit_accuracy_npz_path=clean_npz_path,
    )

    assert result["generated_bundle_record_count"] == 1
    record_path = bundle_root / "sigmark" / "records" / "prompt_a__seed_0__clean.json"
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    assert payload["official_result_provenance"] == "repository_generated_from_third_party_official_code"
    assert payload["official_adapter_baseline_id"] == "sigmark"
    assert payload["official_baseline_id"] == "sigmark"
    assert payload["external_baseline_official_execution_mode"] == "sigmark_hunyuan_gen_extract"
    assert "metric_status" not in payload
    assert payload["bit_accuracy"] == 0.875
    assert payload["external_baseline_clean_negative_score"] == 0.375
    assert payload["official_clean_negative_bit_accuracy_npz_path"] == str(clean_npz_path)
    assert payload["official_execution_manifest_path"] == str(manifest_path)
    assert payload["official_score_granularity"] == "aggregate"
    assert payload["official_score_formal_comparison_eligibility"] == "blocked"
    assert payload["official_score_formal_comparison_block_reason"] == (
        "aggregate_score_assignment_not_formal_comparison_eligible"
    )


@pytest.mark.quick
def test_sigmark_bundle_writer_uses_prompt_seed_specific_npz_key(tmp_path: Path) -> None:
    """SIGMark official bundle 需要按 prompt / seed 抽取单条分数, 不能只复用全局均值。"""

    run_root = tmp_path / "runs" / "generative_video_model_probe" / "validation_scale"
    bundle_root = tmp_path / "bundles" / "validation_scale"
    manifest_path = bundle_root / "sigmark" / "official_reference_execution_manifest.json"
    prompt_suite_path = tmp_path / "datasets" / "generative_video_prompt_suite" / "prompt_seed_suite.json"
    npz_path = tmp_path / "official_outputs" / "HunyuanVideo-community-sigmark-bit_accuracy.npz"
    clean_npz_path = tmp_path / "official_outputs" / "HunyuanVideo-community-sigmark-clean-negative-bit_accuracy.npz"
    _write_runtime_records(run_root)
    _write_prompt_suite(prompt_suite_path)
    _write_json(manifest_path, {"manifest_kind": "test_sigmark_execution_manifest"})
    result_key = "sstw_runtime_prompt/A small red toy car moves across a table with clear motion.-0"
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        npz_path,
        **{
            result_key: np.array([0.8]),
            "sstw_runtime_prompt/unrelated-0": np.array([0.1]),
        },
    )
    np.savez(
        clean_npz_path,
        **{
            result_key: np.array([0.2]),
            "sstw_runtime_prompt/unrelated-0": np.array([0.9]),
        },
    )

    result = write_sigmark_official_bundle_records(
        run_root=run_root,
        bundle_root=bundle_root,
        manifest_path=manifest_path,
        bit_accuracy_npz_path=npz_path,
        model_name="HunyuanVideo-community",
        clean_negative_bit_accuracy_npz_path=clean_npz_path,
        prompt_suite_path=prompt_suite_path,
    )

    assert result["generated_bundle_record_count"] == 1
    record_path = bundle_root / "sigmark" / "records" / "prompt_a__seed_0__clean.json"
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    assert payload["official_adapter_baseline_id"] == "sigmark"
    assert payload["official_baseline_id"] == "sigmark"
    assert payload["external_baseline_score"] == 0.8
    assert payload["external_baseline_clean_negative_score"] == 0.2
    assert payload["official_result_key"] == result_key
    assert payload["official_clean_negative_result_key"] == result_key
    assert payload["official_score_assignment_policy"] == "per_prompt_seed_sigmark_bit_accuracy_npz_key"
    assert payload["official_score_granularity"] == "per_prompt_seed"
    assert payload["official_score_formal_comparison_eligibility"] == "blocked"
    assert payload["official_score_formal_comparison_block_reason"] == (
        "score_granularity_not_formal_comparison_eligible:per_prompt_seed"
    )
    assert payload["official_clean_negative_score_granularity"] == "per_prompt_seed"
    assert payload["official_clean_negative_score_formal_comparison_eligibility"] == "eligible"


@pytest.mark.quick
def test_sigmark_bundle_writer_uses_runtime_attack_specific_npz_key(tmp_path: Path) -> None:
    """SIGMark 只有使用逐 runtime attack 的 official extract 结果时才能进入公平比较。"""

    run_root = tmp_path / "runs" / "generative_video_model_probe" / "validation_scale"
    bundle_root = tmp_path / "bundles" / "validation_scale"
    manifest_path = bundle_root / "sigmark" / "official_reference_execution_manifest.json"
    prompt_suite_path = tmp_path / "datasets" / "generative_video_prompt_suite" / "prompt_seed_suite.json"
    attack_name = "video_compression_runtime"
    npz_path = tmp_path / "official_outputs" / "positive" / attack_name / "bit_accuracy.npz"
    clean_npz_path = tmp_path / "official_outputs" / "clean_negative" / attack_name / "bit_accuracy.npz"
    _write_runtime_records(run_root)
    records_path = run_root / "records" / "runtime_detection_records.jsonl"
    record = json.loads(records_path.read_text(encoding="utf-8").splitlines()[0])
    record["attack_name"] = attack_name
    records_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_prompt_suite(prompt_suite_path)
    _write_json(manifest_path, {"manifest_kind": "test_sigmark_execution_manifest"})
    result_key = "sstw_runtime_prompt/A small red toy car moves across a table with clear motion.-0"
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    clean_npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz_path, **{result_key: np.array([0.82])})
    np.savez(clean_npz_path, **{result_key: np.array([0.22])})

    result = write_sigmark_official_bundle_records(
        run_root=run_root,
        bundle_root=bundle_root,
        manifest_path=manifest_path,
        bit_accuracy_npz_path=npz_path,
        model_name="HunyuanVideo-community",
        clean_negative_bit_accuracy_npz_path=clean_npz_path,
        attack_bit_accuracy_npz_paths={attack_name: npz_path},
        clean_negative_attack_bit_accuracy_npz_paths={attack_name: clean_npz_path},
        prompt_suite_path=prompt_suite_path,
    )

    assert result["generated_bundle_record_count"] == 1
    record_path = bundle_root / "sigmark" / "records" / f"prompt_a__seed_0__{attack_name}.json"
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    assert payload["external_baseline_score"] == 0.82
    assert payload["external_baseline_clean_negative_score"] == 0.22
    assert payload["official_score_assignment_policy"] == "per_prompt_seed_runtime_attack_sigmark_bit_accuracy_npz_key"
    assert payload["official_clean_negative_score_assignment_policy"] == (
        "per_prompt_seed_runtime_attack_sigmark_clean_negative_bit_accuracy_npz_key"
    )
    assert payload["attack_protocol_status"] == "project_runtime_attack_applied_to_sigmark_watermarked_video"
    assert payload["official_score_granularity"] == "per_prompt_seed_attack"
    assert payload["official_score_value_type"] == "payload_bit_accuracy_score"
    assert payload["official_score_formal_comparison_eligibility"] == "eligible"
    assert payload["official_clean_negative_score_formal_comparison_eligibility"] == "eligible"


@pytest.mark.quick
def test_sigmark_attack_extract_output_preparation_copies_state_and_attacks_video(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SIGMark per-attack official extract 目录必须复用官方状态并写入攻击后视频。"""

    source_output = tmp_path / "official_positive_outputs"
    attack_output_root = tmp_path / "official_attack_outputs"
    dimension = "sstw_runtime_prompt"
    prompt_text = "A small red toy car moves across a table with clear motion."
    sample_name = f"{prompt_text[:180]}-0"
    source_video = source_output / dimension / f"{sample_name}.mp4"
    source_video.parent.mkdir(parents=True, exist_ok=True)
    source_video.write_bytes(b"video")
    (source_output / "HunyuanVideo-community-VBench2_aug-512x512-65frams-sigmark-1024bits-maintained_info.pkl").write_bytes(b"state")
    (source_output / "HunyuanVideo-community-VBench2_aug-512x512-65frams-sigmark-1024bits-gt_watermark_messages.npz").write_bytes(b"gt")
    records = [
        {
            "prompt_id": "prompt_a",
            "seed_id": "seed_0",
            "attack_name": "temporal_crop_runtime",
        }
    ]

    monkeypatch.setattr(sigmark_runtime, "_read_video_frames", lambda _path: ["f0", "f1", "f2", "f3"])
    monkeypatch.setattr(
        sigmark_runtime,
        "_write_video_frames",
        lambda path, _frames, *, fps: Path(path).parent.mkdir(parents=True, exist_ok=True) or Path(path).write_bytes(b"attacked"),
    )

    manifest = sigmark_runtime._prepare_sigmark_attack_extract_outputs(
        records=records,
        source_output_path=source_output,
        attack_output_root=attack_output_root,
        state_source_output_path=source_output,
        prompt_text_by_id={"prompt_a": prompt_text},
        seed_indices={"prompt_a": {"seed_0": 0}},
        runtime_dimension=dimension,
        allow_prompt_id_fallback=False,
        fps=8,
        role="positive",
    )

    attack_dir = Path(manifest["attack_output_paths"]["temporal_crop_runtime"])
    assert manifest["attack_extract_output_prepare_status"] == "ready"
    assert (attack_dir / source_video.relative_to(source_output)).exists()
    assert list(attack_dir.glob("*-maintained_info.pkl"))
    assert list(attack_dir.glob("*-gt_watermark_messages.npz"))
    assert manifest["prepared_rows"][0]["attack_transform"] == "drop_first_and_last_frame_when_possible"


@pytest.mark.quick
def test_sigmark_attack_extract_output_preparation_adds_supplemental_generated_videos(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """attack-specific extract 目录必须补齐官方 prompt/seed 网格中的未用视频。"""

    source_output = tmp_path / "official_positive_outputs"
    attack_output_root = tmp_path / "official_attack_outputs"
    dimension = "sstw_runtime_prompt"
    prompt_text = "A small red toy car moves across a table with clear motion."
    record_video = source_output / dimension / f"{prompt_text[:180]}-0.mp4"
    supplemental_video = source_output / dimension / f"{prompt_text[:180]}-1.mp4"
    record_video.parent.mkdir(parents=True, exist_ok=True)
    record_video.write_bytes(b"video0")
    supplemental_video.write_bytes(b"video1")
    (source_output / "HunyuanVideo-community-VBench2_aug-512x512-65frams-sigmark-1024bits-maintained_info.pkl").write_bytes(b"state")
    (source_output / "HunyuanVideo-community-VBench2_aug-512x512-65frams-sigmark-1024bits-gt_watermark_messages.npz").write_bytes(b"gt")

    monkeypatch.setattr(sigmark_runtime, "_read_video_frames", lambda _path: ["f0", "f1"])
    monkeypatch.setattr(
        sigmark_runtime,
        "_write_video_frames",
        lambda path, _frames, *, fps: Path(path).parent.mkdir(parents=True, exist_ok=True) or Path(path).write_bytes(b"attacked"),
    )

    manifest = sigmark_runtime._prepare_sigmark_attack_extract_outputs(
        records=[
            {
                "prompt_id": "prompt_a",
                "seed_id": "seed_0",
                "attack_name": "video_compression_runtime",
            }
        ],
        source_output_path=source_output,
        attack_output_root=attack_output_root,
        state_source_output_path=source_output,
        prompt_text_by_id={"prompt_a": prompt_text},
        seed_indices={"prompt_a": {"seed_0": 0}},
        runtime_dimension=dimension,
        allow_prompt_id_fallback=False,
        fps=8,
        role="positive",
    )

    attack_dir = Path(manifest["attack_output_paths"]["video_compression_runtime"])
    assert manifest["prepared_video_count"] == 1
    assert manifest["supplemental_official_extract_video_count"] == 1
    assert (attack_dir / record_video.relative_to(source_output)).exists()
    assert (attack_dir / supplemental_video.relative_to(source_output)).exists()
    assert any(row.get("supplemental_official_extract_video") for row in manifest["prepared_rows"])


@pytest.mark.quick
def test_sigmark_generation_progress_probe_counts_paper_sample_videos(tmp_path: Path) -> None:
    """SIGMark gen 进度探针必须统计论文样本视频落盘数量。"""

    output_path = tmp_path / "official_outputs"
    dimension_dir = output_path / "sstw_runtime_prompt"
    dimension_dir.mkdir(parents=True)
    (dimension_dir / "sample-0.mp4").write_bytes(b"video")
    (dimension_dir / "sample-1.mp4").write_bytes(b"video")

    probe = sigmark_runtime._build_sigmark_generation_progress_probe(
        output_path=output_path,
        runtime_dimension="sstw_runtime_prompt",
        expected_video_unit_count=4,
        role="positive_gen",
    )
    progress = probe()

    assert progress["role"] == "positive_gen"
    assert progress["generated_video_units"] == "2/4"
    assert progress["progress_percent"] == "50.0"


@pytest.mark.quick
def test_sigmark_extract_progress_probe_counts_bit_accuracy_keys(tmp_path: Path) -> None:
    """SIGMark extract 进度探针必须统计 bit accuracy npz 中的样本 key 数量。"""

    output_path = tmp_path / "official_attack_output"
    output_path.mkdir(parents=True)
    np.savez(
        output_path / "HunyuanVideo-community-sigmark-bit_accuracy.npz",
        **{
            "sstw_runtime_prompt/sample-0": np.array([0.8]),
            "sstw_runtime_prompt/sample-1": np.array([0.7]),
        },
    )

    probe = sigmark_runtime._build_sigmark_extract_progress_probe(
        output_path=output_path,
        expected_video_unit_count=4,
        role="positive_extract:video_compression_runtime",
    )
    progress = probe()

    assert progress["role"] == "positive_extract:video_compression_runtime"
    assert progress["extracted_video_units"] == "2/4"
    assert progress["progress_percent"] == "50.0"
    assert progress["bit_accuracy_npz_path"].endswith("bit_accuracy.npz")


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
