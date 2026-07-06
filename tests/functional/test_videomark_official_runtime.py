"""验证 VideoMark 官方运行器的轻量逻辑。"""

from __future__ import annotations

import json
from pathlib import Path
import hashlib

import pytest

import external_baseline.videomark_official_runtime as videomark_runtime
from external_baseline.videomark_official_runtime import (
    VideoMarkOfficialRuntimeConfig,
    build_default_videomark_official_config_from_env,
    run_videomark_official_runtime,
    write_videomark_official_bundle_records,
)


def _write_json(path: Path, payload: object) -> None:
    """写出测试 JSON 文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_runtime_records(
    run_root: Path,
    attack_names: tuple[str, ...] = ("temporal_crop_runtime",),
) -> None:
    """构造最小 runtime detection records fixture。"""

    records_path = run_root / "records" / "runtime_detection_records.jsonl"
    records_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, attack_name in enumerate(attack_names):
        rows.append({
            "runtime_detection_status": "ready",
            "generation_model_id": "wan21_runtime",
            "prompt_id": "prompt_a",
            "seed_id": "seed_0",
            "trajectory_trace_id": f"trace_a_{index}",
            "attack_name": attack_name,
            "source_video_path": str(run_root / "videos" / "source.mp4"),
            "attacked_video_path": str(run_root / "videos" / f"attacked_{attack_name}.mp4"),
        })
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


def _write_fake_videomark_source(source_dir: Path) -> None:
    """构造只用于 dry-run 文本改写的伪 VideoMark 官方源码结构。"""

    (source_dir / "src").mkdir(parents=True, exist_ok=True)
    (source_dir / "src" / "prc.py").write_text("# fake prc\n", encoding="utf-8")
    (source_dir / "temporal_tamper.py").write_text(
        "\n".join(
            [
                "if __name__ == '__main__':",
                "    parser.add_argument('--keys_path', default=\"./keys\")",
                "def temporal_tamper(video_frames, tampering_type_list, shift_value, message_bits_sequence):",
                "    tampered_videos = {}",
                "    return tampered_videos",
                "def simulate_one_round(args):",
                "    return 0",
                "def main(args):",
                "    video_frames_dirs = os.path.join(args.video_frames_dir,\"videomark\",model_name,f\"{num_bit}bit\")",
                "    for dirname in os.listdir(video_frames_dirs):",
                "",
                "        video_frames_dir = os.path.join(video_frames_dirs, dirname,'wm','frames')",
                "        shift_value = np.load(os.path.join(video_frames_dirs, dirname,\"shift_value.npy\"))",
                "",
                "        if not os.path.exists(video_frames_dir):",
                "            continue",
                "        temporal_tampering_type = ['frame insert','frame drop','frame swap']",
                "        video_frames_tampered = temporal_tamper(video_frames, temporal_tampering_type, shift_value, message_bits_sequence)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (source_dir / "embedding_and_extraction.py").write_text(
        "\n".join(
            [
                "import os",
                "import numpy as np",
                "def process_frame(detection_result, message_bits):",
                "    message_placeholder = '<message_placeholder>'",
                "    if not detection_result:",
                "        decode_message_str = message_placeholder",
                "    else:",
                "        decode_message = Decode(decoding_key, reversed_prc)",
                "        decode_message_str = bits_to_string(decode_message)",
                "    return decode_message_str",
                "def main():",
                "    use_watermark = args.use_watermark",
                "    for item in tqdm(range(4)):",
                "        for i, row in enumerate(data):",
                "            current_prompt = row",
                "            video_id = current_prompt.replace(' ', '_')",
                "            if use_watermark:",
                "                pass",
                "            else:",
                "                init_latents_w = torch.randn(num_frames, 1, 4, height, width).to(device)",
                "if __name__ == '__main__':",
                "    parser.add_argument('--model_name', default='i2vgen-xl')",
                "    parser.add_argument('--use_watermark', default=True)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (source_dir / "keys").mkdir(parents=True, exist_ok=True)
    (source_dir / "keys" / "64_64_512bit.pkl").write_bytes(b"fake-key")


@pytest.mark.quick
def test_videomark_runtime_dry_run_builds_prompt_set_and_commands(tmp_path: Path) -> None:
    """dry-run 必须只写运行计划, 不触发重型 VideoMark 生成。"""

    run_root = tmp_path / "runs" / "generative_video_model_probe" / "validation_scale"
    bundle_root = tmp_path / "bundles" / "validation_scale"
    source_dir = tmp_path / "official_source"
    prompt_suite_path = tmp_path / "datasets" / "generative_video_prompt_suite" / "prompt_seed_suite.json"
    _write_runtime_records(run_root)
    _write_prompt_suite(prompt_suite_path)
    _write_fake_videomark_source(source_dir)

    config = VideoMarkOfficialRuntimeConfig(
        run_root=str(run_root),
        bundle_root=str(bundle_root),
        source_dir=str(source_dir),
        output_root=str(tmp_path / "videomark_runtime"),
        resource_root=str(tmp_path / "resources" / "external_baseline"),
        prompt_suite_path=str(prompt_suite_path),
        output_path=str(tmp_path / "official_outputs"),
        dry_run=True,
    )

    manifest = run_videomark_official_runtime(config)

    assert manifest["execution_status"] == "dry_run_planned"
    assert manifest["generated_bundle_record_count"] == 0
    assert manifest["prompt_manifest"]["prompt_count"] == 1
    assert "--model_name=modelscope" in manifest["embedding_command"]
    assert "--use_watermark=true" in manifest["embedding_command"]
    assert "--video_frames_dir=" + config.output_path in manifest["temporal_tamper_command"]
    assert "--threshold=0.5" in manifest["temporal_tamper_command"]
    assert "--resample_num=1" in manifest["temporal_tamper_command"]
    assert "--video_family=videomark" in manifest["temporal_tamper_command"]
    assert "--use_watermark=false" in manifest["clean_negative_embedding_command"]
    assert "--video_family=without_watermark" in manifest["clean_negative_temporal_tamper_command"]
    runtime_embedding = Path(manifest["runtime_source_dir"]) / "embedding_and_extraction.py"
    runtime_temporal = Path(manifest["runtime_source_dir"]) / "temporal_tamper.py"
    runtime_text = runtime_embedding.read_text(encoding="utf-8")
    runtime_temporal_text = runtime_temporal.read_text(encoding="utf-8")
    assert "SSTW_VIDEOMARK_PROMPT_VARIANTS" in runtime_text
    assert "video_id = f\"prompt_{i:04d}_{video_id_digest}\"" in runtime_text
    assert "decode_message = np.full((len(message_bits[0]),), -1)" in runtime_text
    assert "parser.add_argument('--model_path', default=None)" in runtime_text
    assert "use_watermark.strip().lower()" in runtime_text
    assert "_encoding_key, decoding_key = pickle.load(f)" in runtime_text
    assert "sstw_latent_dtype = next(video_pipe.unet.parameters()).dtype" in runtime_text
    assert "1, 4, num_frames, height, width, device=device, dtype=sstw_latent_dtype" in runtime_text
    assert "num_frames, 1, 4, height, width, device=device, dtype=sstw_latent_dtype" not in runtime_text
    assert "parser.add_argument('--threshold', default=0.5, type=float)" in runtime_temporal_text
    assert "parser.add_argument('--resample_num', default=1, type=int)" in runtime_temporal_text
    assert "parser.add_argument('--video_family', default='videomark')" in runtime_temporal_text
    assert "video_family = getattr(args, 'video_family', 'videomark')" in runtime_temporal_text
    assert "if not os.path.isdir(video_output_dir):" in runtime_temporal_text
    assert "shift_value_path = os.path.join(video_output_dir,\"shift_value.npy\")" in runtime_temporal_text
    assert "if not os.path.isdir(video_frames_dir) or not os.path.isfile(shift_value_path):" in runtime_temporal_text
    assert "def sstw_videomark_runtime_temporal_tamper(" in runtime_temporal_text
    assert "'video_compression_runtime'" in runtime_temporal_text
    assert "'temporal_crop_runtime'" in runtime_temporal_text
    assert "'frame_rate_resampling_runtime'" in runtime_temporal_text
    assert "SSTW_VIDEOMARK_RUNTIME_ATTACK_NAMES" in runtime_temporal_text
    assert "spatial_crop_resize_runtime" in runtime_temporal_text
    assert "brightness_contrast_runtime" in runtime_temporal_text
    assert "sstw_videomark_runtime_temporal_tamper(video_frames, temporal_tampering_type" in runtime_temporal_text
    assert {
        row["patch_name"]: row["patch_status"] for row in manifest["patch_manifest"]["patch_results"]
    } == {
        "prompt_variant_count_env_guard": "patched_runtime_copy",
        "safe_prompt_digest_video_id_guard": "patched_runtime_copy",
        "undetected_decode_message_guard": "patched_runtime_copy",
        "embedding_model_path_cli_arg_guard": "patched_runtime_copy",
        "embedding_use_watermark_bool_guard": "patched_runtime_copy",
        "embedding_clean_negative_decode_key_guard": "patched_runtime_copy",
        "temporal_threshold_resample_cli_arg_guard": "patched_runtime_copy",
        "temporal_video_family_guard": "patched_runtime_copy",
        "temporal_output_file_skip_guard": "patched_runtime_copy",
        "temporal_runtime_attack_protocol_guard": "patched_runtime_copy",
    }


@pytest.mark.quick
def test_videomark_patch_repairs_clean_negative_wrong_latent_shape(tmp_path: Path) -> None:
    """已有错误补丁副本也必须被修正为 diffusers 期望的 latent 形状。"""

    source_dir = tmp_path / "official_source"
    _write_fake_videomark_source(source_dir)
    embedding_path = source_dir / "embedding_and_extraction.py"
    embedding_text = embedding_path.read_text(encoding="utf-8")
    embedding_path.write_text(
        embedding_text.replace(
            videomark_runtime.EMBEDDING_CLEAN_BRANCH_TARGET,
            videomark_runtime.EMBEDDING_CLEAN_BRANCH_WRONG_SHAPE_PATCH_TARGET,
        ),
        encoding="utf-8",
    )

    patch_manifest = videomark_runtime._patch_videomark_runtime_source(source_dir)
    patched_text = embedding_path.read_text(encoding="utf-8")
    patch_rows = {row["patch_name"]: row["patch_status"] for row in patch_manifest["patch_results"]}

    assert patch_rows["embedding_clean_negative_decode_key_guard"] == "patched_runtime_copy"
    assert "1, 4, num_frames, height, width, device=device, dtype=sstw_latent_dtype" in patched_text
    assert "num_frames, 1, 4, height, width, device=device, dtype=sstw_latent_dtype" not in patched_text


@pytest.mark.quick
def test_videomark_default_output_path_uses_safe_prompt_digest_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """默认输出目录必须避开旧的长 prompt 目录缓存。"""

    bundle_root = tmp_path / "bundles" / "validation_scale"
    monkeypatch.setenv("SSTW_VIDEOMARK_OFFICIAL_OUTPUT_DIR", str(bundle_root / "videomark" / "official_outputs"))

    config = build_default_videomark_official_config_from_env(
        run_root=tmp_path / "runs" / "generative_video_model_probe" / "validation_scale",
        bundle_root=bundle_root,
        source_dir=tmp_path / "official_source",
    )

    assert Path(config.output_path).name == "official_outputs_safe_prompt_digest_v1"


@pytest.mark.quick
def test_videomark_bundle_writer_records_project_owned_provenance(tmp_path: Path) -> None:
    """VideoMark temporal_results 必须转成 project-owned official bundle, 而非外部补交结果。"""

    run_root = tmp_path / "runs" / "generative_video_model_probe" / "validation_scale"
    bundle_root = tmp_path / "bundles" / "validation_scale"
    manifest_path = bundle_root / "videomark" / "official_reference_execution_manifest.json"
    temporal_path = tmp_path / "official_outputs" / "videomark" / "modelscope" / "512bit" / "temporal_results.json"
    video_path = temporal_path.with_name("video_results.json")
    clean_path = temporal_path.with_name("clean_negative_results.json")
    _write_runtime_records(run_root)
    _write_json(manifest_path, {"manifest_kind": "test_videomark_execution_manifest"})
    _write_json(
        temporal_path,
        {
            "toy_car_0": {
                "frame drop": {"decode_acc": 0.75, "frames_acc": 0.5},
                "frame swap": {"decode_acc": 1.0, "frames_acc": 0.8},
            }
        },
    )
    _write_json(video_path, {"toy_car_0": {"decode_acc": 0.9}})
    _write_json(clean_path, {"toy_car_clean": {"decode_acc": 0.25}})

    result = write_videomark_official_bundle_records(
        run_root=run_root,
        bundle_root=bundle_root,
        manifest_path=manifest_path,
        temporal_results_json_path=temporal_path,
        video_results_json_path=video_path,
        model_name="modelscope",
        clean_negative_results_json_path=clean_path,
    )

    assert result["generated_bundle_record_count"] == 1
    record_path = bundle_root / "videomark" / "records" / "prompt_a__seed_0__temporal_crop_runtime.json"
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    assert payload["official_result_provenance"] == "repository_generated_from_third_party_official_code"
    assert payload["official_adapter_baseline_id"] == "videomark"
    assert payload["official_baseline_id"] == "videomark"
    assert payload["external_baseline_official_execution_mode"] == "videomark_embedding_extraction_temporal_tamper"
    assert payload["external_baseline_score"] == 0.875
    assert payload["external_baseline_clean_negative_score"] == 0.25
    assert payload["official_clean_negative_results_json_path"] == str(clean_path)
    assert payload["official_frames_acc_mean"] == 0.65
    assert payload["official_temporal_attack_names"] == ["frame drop", "frame swap"]
    assert "metric_status" not in payload
    assert payload["official_score_granularity"] == "aggregate"
    assert payload["official_score_formal_comparison_eligibility"] == "blocked"
    assert payload["official_score_formal_comparison_block_reason"] == (
        "aggregate_score_assignment_not_formal_comparison_eligible"
    )


@pytest.mark.quick
def test_videomark_bundle_writer_uses_prompt_seed_attack_specific_key(tmp_path: Path) -> None:
    """VideoMark official bundle 需要按 prompt / seed / attack 抽取单条官方分数。"""

    run_root = tmp_path / "runs" / "generative_video_model_probe" / "validation_scale"
    bundle_root = tmp_path / "bundles" / "validation_scale"
    manifest_path = bundle_root / "videomark" / "official_reference_execution_manifest.json"
    prompt_suite_path = tmp_path / "datasets" / "generative_video_prompt_suite" / "prompt_seed_suite.json"
    temporal_path = tmp_path / "official_outputs" / "videomark" / "modelscope" / "512bit" / "temporal_results.json"
    video_path = temporal_path.with_name("video_results.json")
    clean_path = tmp_path / "official_clean_negative_outputs" / "without_watermark" / "modelscope" / "temporal_results.json"
    _write_runtime_records(run_root)
    _write_prompt_suite(prompt_suite_path)
    _write_json(manifest_path, {"manifest_kind": "test_videomark_execution_manifest"})
    prompt_text = "A small red toy car moves across a table with clear motion."
    digest = hashlib.sha1(prompt_text.encode("utf-8")).hexdigest()[:12]
    video_key = f"prompt_0000_{digest}_0"
    unrelated_key = f"prompt_0000_{digest}_1"
    _write_json(
        temporal_path,
        {
            video_key: {
                "video_compression_runtime": {"decode_acc": 0.66, "frames_acc": 1.0},
                "temporal_crop_runtime": {"decode_acc": 0.7, "frames_acc": 0.5},
                "frame_rate_resampling_runtime": {"decode_acc": 0.62, "frames_acc": 0.4},
            },
            unrelated_key: {"temporal_crop_runtime": {"decode_acc": 0.1, "frames_acc": 0.1}},
        },
    )
    _write_json(video_path, {video_key: {"decode_acc": 0.9}})
    _write_json(
        clean_path,
        {
            video_key: {
                "video_compression_runtime": {"decode_acc": 0.2, "frames_acc": 1.0},
                "temporal_crop_runtime": {"decode_acc": 0.25, "frames_acc": 0.4},
                "frame_rate_resampling_runtime": {"decode_acc": 0.22, "frames_acc": 0.3},
            },
            unrelated_key: {"temporal_crop_runtime": {"decode_acc": 0.9, "frames_acc": 0.9}},
        },
    )

    result = write_videomark_official_bundle_records(
        run_root=run_root,
        bundle_root=bundle_root,
        manifest_path=manifest_path,
        temporal_results_json_path=temporal_path,
        video_results_json_path=video_path,
        model_name="modelscope",
        clean_negative_results_json_path=clean_path,
        prompt_suite_path=prompt_suite_path,
    )

    assert result["generated_bundle_record_count"] == 1
    record_path = bundle_root / "videomark" / "records" / "prompt_a__seed_0__temporal_crop_runtime.json"
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    assert payload["official_adapter_baseline_id"] == "videomark"
    assert payload["official_baseline_id"] == "videomark"
    assert payload["external_baseline_score"] == 0.7
    assert payload["external_baseline_clean_negative_score"] == 0.25
    assert payload["official_result_key"] == video_key
    assert payload["official_temporal_attack_key"] == "temporal_crop_runtime"
    assert payload["official_score_assignment_policy"] == "per_prompt_seed_runtime_attack_mapped_to_videomark_temporal_attack"
    assert payload["official_clean_negative_result_key"] == video_key
    assert payload["official_score_granularity"] == "per_prompt_seed_attack"
    assert payload["official_score_formal_comparison_eligibility"] == "eligible"
    assert payload["official_clean_negative_score_granularity"] == "per_prompt_seed_attack"
    assert payload["official_clean_negative_score_formal_comparison_eligibility"] == "eligible"


@pytest.mark.quick
def test_videomark_bundle_writer_aligns_declared_runtime_attacks_without_aggregate_fallback(
    tmp_path: Path,
) -> None:
    """VideoMark 的正式 bundle 必须覆盖输入 records 中声明的 runtime attack 集合。"""

    run_root = tmp_path / "runs" / "generative_video_model_probe" / "validation_scale"
    bundle_root = tmp_path / "bundles" / "validation_scale"
    manifest_path = bundle_root / "videomark" / "official_reference_execution_manifest.json"
    prompt_suite_path = tmp_path / "datasets" / "generative_video_prompt_suite" / "prompt_seed_suite.json"
    temporal_path = tmp_path / "official_outputs" / "videomark" / "modelscope" / "512bit" / "temporal_results.json"
    video_path = temporal_path.with_name("video_results.json")
    clean_path = tmp_path / "official_clean_negative_outputs" / "without_watermark" / "modelscope" / "temporal_results.json"
    _write_runtime_records(
        run_root,
        attack_names=(
            "video_compression_runtime",
            "temporal_crop_runtime",
            "frame_rate_resampling_runtime",
        ),
    )
    _write_prompt_suite(prompt_suite_path)
    _write_json(manifest_path, {"manifest_kind": "test_videomark_execution_manifest"})
    prompt_text = "A small red toy car moves across a table with clear motion."
    digest = hashlib.sha1(prompt_text.encode("utf-8")).hexdigest()[:12]
    video_key = f"prompt_0000_{digest}_0"
    _write_json(
        temporal_path,
        {
            video_key: {
                "video_compression_runtime": {"decode_acc": 0.61, "frames_acc": 1.0},
                "temporal_crop_runtime": {"decode_acc": 0.72, "frames_acc": 0.5},
                "frame_rate_resampling_runtime": {"decode_acc": 0.58, "frames_acc": 0.4},
            }
        },
    )
    _write_json(video_path, {video_key: {"decode_acc": 0.9}})
    _write_json(
        clean_path,
        {
            video_key: {
                "video_compression_runtime": {"decode_acc": 0.21, "frames_acc": 1.0},
                "temporal_crop_runtime": {"decode_acc": 0.24, "frames_acc": 0.4},
                "frame_rate_resampling_runtime": {"decode_acc": 0.19, "frames_acc": 0.3},
            }
        },
    )

    result = write_videomark_official_bundle_records(
        run_root=run_root,
        bundle_root=bundle_root,
        manifest_path=manifest_path,
        temporal_results_json_path=temporal_path,
        video_results_json_path=video_path,
        model_name="modelscope",
        clean_negative_results_json_path=clean_path,
        prompt_suite_path=prompt_suite_path,
    )

    assert result["generated_bundle_record_count"] == 3
    assert result["failed_bundle_record_count"] == 0
    expected_scores = {
        "video_compression_runtime": 0.61,
        "temporal_crop_runtime": 0.72,
        "frame_rate_resampling_runtime": 0.58,
    }
    for attack_name, expected_score in expected_scores.items():
        record_path = bundle_root / "videomark" / "records" / f"prompt_a__seed_0__{attack_name}.json"
        payload = json.loads(record_path.read_text(encoding="utf-8"))
        assert payload["external_baseline_score"] == expected_score
        assert payload["official_temporal_attack_key"] == attack_name
        assert payload["official_score_granularity"] == "per_prompt_seed_attack"
        assert payload["official_score_formal_comparison_eligibility"] == "eligible"
        assert payload["official_clean_negative_score_granularity"] == "per_prompt_seed_attack"


@pytest.mark.quick
def test_videomark_bundle_writer_aligns_expanded_runtime_attacks_without_aggregate_fallback(
    tmp_path: Path,
) -> None:
    """VideoMark bundle writer 必须支持 pilot/full 扩展 runtime attack 的逐 attack 分数抽取。"""

    run_root = tmp_path / "runs" / "generative_video_model_probe" / "pilot_paper"
    bundle_root = tmp_path / "bundles" / "pilot_paper"
    manifest_path = bundle_root / "videomark" / "official_reference_execution_manifest.json"
    prompt_suite_path = tmp_path / "datasets" / "generative_video_prompt_suite" / "prompt_seed_suite.json"
    temporal_path = tmp_path / "official_outputs" / "videomark" / "modelscope" / "512bit" / "temporal_results.json"
    video_path = temporal_path.with_name("video_results.json")
    clean_path = tmp_path / "official_clean_negative_outputs" / "without_watermark" / "modelscope" / "temporal_results.json"
    expanded_attacks = (
        "frame_drop_uniform_runtime",
        "spatial_crop_resize_runtime",
        "gaussian_noise_runtime",
        "brightness_contrast_runtime",
        "compression_temporal_combined_runtime",
    )
    _write_runtime_records(run_root, attack_names=expanded_attacks)
    _write_prompt_suite(prompt_suite_path)
    _write_json(manifest_path, {"manifest_kind": "test_videomark_execution_manifest"})
    prompt_text = "A small red toy car moves across a table with clear motion."
    digest = hashlib.sha1(prompt_text.encode("utf-8")).hexdigest()[:12]
    video_key = f"prompt_0000_{digest}_0"
    _write_json(
        temporal_path,
        {
            video_key: {
                attack_name: {"decode_acc": round(0.50 + index * 0.03, 3), "frames_acc": 0.5}
                for index, attack_name in enumerate(expanded_attacks)
            }
        },
    )
    _write_json(video_path, {video_key: {"decode_acc": 0.9}})
    _write_json(
        clean_path,
        {
            video_key: {
                attack_name: {"decode_acc": round(0.20 + index * 0.01, 3), "frames_acc": 0.3}
                for index, attack_name in enumerate(expanded_attacks)
            }
        },
    )

    result = write_videomark_official_bundle_records(
        run_root=run_root,
        bundle_root=bundle_root,
        manifest_path=manifest_path,
        temporal_results_json_path=temporal_path,
        video_results_json_path=video_path,
        model_name="modelscope",
        clean_negative_results_json_path=clean_path,
        prompt_suite_path=prompt_suite_path,
    )

    assert result["generated_bundle_record_count"] == len(expanded_attacks)
    assert result["failed_bundle_record_count"] == 0
    for index, attack_name in enumerate(expanded_attacks):
        record_path = bundle_root / "videomark" / "records" / f"prompt_a__seed_0__{attack_name}.json"
        payload = json.loads(record_path.read_text(encoding="utf-8"))
        assert payload["official_temporal_attack_key"] == attack_name
        assert payload["external_baseline_score"] == round(0.50 + index * 0.03, 3)
        assert payload["official_score_assignment_policy"] == "per_prompt_seed_runtime_attack_mapped_to_videomark_temporal_attack"


@pytest.mark.quick
def test_videomark_bundle_writer_rejects_unknown_runtime_attack_without_aggregate_fallback(
    tmp_path: Path,
) -> None:
    """未知 runtime attack 仍必须 fail-closed, 不能退化为 temporal 均值。"""

    run_root = tmp_path / "runs" / "generative_video_model_probe" / "validation_scale"
    bundle_root = tmp_path / "bundles" / "validation_scale"
    manifest_path = bundle_root / "videomark" / "official_reference_execution_manifest.json"
    prompt_suite_path = tmp_path / "datasets" / "generative_video_prompt_suite" / "prompt_seed_suite.json"
    temporal_path = tmp_path / "official_outputs" / "videomark" / "modelscope" / "512bit" / "temporal_results.json"
    video_path = temporal_path.with_name("video_results.json")
    clean_path = tmp_path / "official_clean_negative_outputs" / "without_watermark" / "modelscope" / "temporal_results.json"
    _write_runtime_records(run_root, attack_names=("unsupported_custom_runtime_attack",))
    _write_prompt_suite(prompt_suite_path)
    _write_json(manifest_path, {"manifest_kind": "test_videomark_execution_manifest"})
    prompt_text = "A small red toy car moves across a table with clear motion."
    digest = hashlib.sha1(prompt_text.encode("utf-8")).hexdigest()[:12]
    video_key = f"prompt_0000_{digest}_0"
    _write_json(
        temporal_path,
        {video_key: {"video_compression_runtime": {"decode_acc": 0.7, "frames_acc": 0.5}}},
    )
    _write_json(video_path, {video_key: {"decode_acc": 0.9}})
    _write_json(
        clean_path,
        {video_key: {"video_compression_runtime": {"decode_acc": 0.25, "frames_acc": 0.4}}},
    )

    result = write_videomark_official_bundle_records(
        run_root=run_root,
        bundle_root=bundle_root,
        manifest_path=manifest_path,
        temporal_results_json_path=temporal_path,
        video_results_json_path=video_path,
        model_name="modelscope",
        clean_negative_results_json_path=clean_path,
        prompt_suite_path=prompt_suite_path,
    )

    assert result["generated_bundle_record_count"] == 0
    assert result["failed_bundle_record_count"] == 1
    assert "videomark_runtime_attack_not_supported_by_official_temporal_protocol" in result["failures"][0]["failure_reason"]
    record_path = bundle_root / "videomark" / "records" / "prompt_a__seed_0__unsupported_custom_runtime_attack.json"
    assert not record_path.exists()


@pytest.mark.quick
def test_videomark_embedding_progress_probe_counts_paper_sample_outputs(tmp_path: Path) -> None:
    """VideoMark embedding 进度探针必须统计论文样本视频与 extraction 完成数量。"""

    result_dir = tmp_path / "official_outputs" / "videomark" / "modelscope" / "512bit"
    (result_dir / "prompt_0000_a_0").mkdir(parents=True)
    (result_dir / "prompt_0000_a_0" / "wm.mp4").write_bytes(b"video")
    (result_dir / "prompt_0001_b_0").mkdir(parents=True)
    (result_dir / "prompt_0001_b_0" / "wm.mp4").write_bytes(b"video")
    video_results_path = result_dir / "video_results.json"
    _write_json(video_results_path, {"prompt_0000_a_0": {"decode_acc": 0.8}})

    probe = videomark_runtime._build_videomark_embedding_progress_probe(
        video_results_path=video_results_path,
        result_dir=result_dir,
        expected_video_unit_count=4,
        role="positive_embedding",
    )
    progress = probe()

    assert progress["role"] == "positive_embedding"
    assert progress["generated_video_units"] == "2/4"
    assert progress["extracted_video_units"] == "1/4"
    assert progress["progress_percent"] == "25.0"


@pytest.mark.quick
def test_videomark_temporal_progress_probe_counts_sample_attack_outputs(tmp_path: Path) -> None:
    """VideoMark temporal 进度探针必须统计 prompt / seed / attack 级完成数量。"""

    temporal_results_path = tmp_path / "official_outputs" / "videomark" / "modelscope" / "512bit" / "temporal_results.json"
    _write_json(
        temporal_results_path,
        {
            "prompt_0000_a_0": {
                "video_compression_runtime": {"decode_acc": 0.7, "frames_acc": 0.9},
                "temporal_crop_runtime": {"decode_acc": 0.6},
            },
            "prompt_0001_b_0": {
                "video_compression_runtime": {"decode_acc": 0.5, "frames_acc": 0.8},
            },
        },
    )

    probe = videomark_runtime._build_videomark_temporal_progress_probe(
        temporal_results_path=temporal_results_path,
        expected_video_unit_count=2,
        runtime_attack_count=2,
        role="positive_temporal_tamper",
    )
    progress = probe()

    assert progress["role"] == "positive_temporal_tamper"
    assert progress["completed_video_units"] == "2/2"
    assert progress["completed_sample_attacks"] == "2/4"
    assert progress["runtime_attack_count"] == 2
    assert progress["progress_percent"] == "50.0"
