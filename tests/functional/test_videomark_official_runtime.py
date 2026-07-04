"""验证 VideoMark 官方运行器的轻量逻辑。"""

from __future__ import annotations

import json
from pathlib import Path
import hashlib

import pytest

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
            "attack_name": "temporal_crop_runtime",
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


def _write_fake_videomark_source(source_dir: Path) -> None:
    """构造只用于 dry-run 文本改写的伪 VideoMark 官方源码结构。"""

    (source_dir / "src").mkdir(parents=True, exist_ok=True)
    (source_dir / "src" / "prc.py").write_text("# fake prc\n", encoding="utf-8")
    (source_dir / "temporal_tamper.py").write_text(
        "\n".join(
            [
                "if __name__ == '__main__':",
                "    parser.add_argument('--keys_path', default=\"./keys\")",
                "def main(args):",
                "    video_frames_dirs = os.path.join(args.video_frames_dir,\"videomark\",model_name,f\"{num_bit}bit\")",
                "    for dirname in os.listdir(video_frames_dirs):",
                "",
                "        video_frames_dir = os.path.join(video_frames_dirs, dirname,'wm','frames')",
                "        shift_value = np.load(os.path.join(video_frames_dirs, dirname,\"shift_value.npy\"))",
                "",
                "        if not os.path.exists(video_frames_dir):",
                "            continue",
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
    assert "parser.add_argument('--threshold', default=0.5, type=float)" in runtime_temporal_text
    assert "parser.add_argument('--resample_num', default=1, type=int)" in runtime_temporal_text
    assert "parser.add_argument('--video_family', default='videomark')" in runtime_temporal_text
    assert "video_family = getattr(args, 'video_family', 'videomark')" in runtime_temporal_text
    assert "if not os.path.isdir(video_output_dir):" in runtime_temporal_text
    assert "shift_value_path = os.path.join(video_output_dir,\"shift_value.npy\")" in runtime_temporal_text
    assert "if not os.path.isdir(video_frames_dir) or not os.path.isfile(shift_value_path):" in runtime_temporal_text
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
    }


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
                "frame drop": {"decode_acc": 0.7, "frames_acc": 0.5},
                "frame swap": {"decode_acc": 0.95, "frames_acc": 0.8},
            },
            unrelated_key: {"frame drop": {"decode_acc": 0.1, "frames_acc": 0.1}},
        },
    )
    _write_json(video_path, {video_key: {"decode_acc": 0.9}})
    _write_json(
        clean_path,
        {
            video_key: {
                "frame drop": {"decode_acc": 0.25, "frames_acc": 0.4},
                "frame swap": {"decode_acc": 0.8, "frames_acc": 0.6},
            },
            unrelated_key: {"frame drop": {"decode_acc": 0.9, "frames_acc": 0.9}},
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
    assert payload["official_temporal_attack_key"] == "frame drop"
    assert payload["official_score_assignment_policy"] == "per_prompt_seed_runtime_attack_mapped_to_videomark_temporal_attack"
    assert payload["official_clean_negative_result_key"] == video_key
    assert payload["official_score_granularity"] == "per_prompt_seed_attack"
    assert payload["official_score_formal_comparison_eligibility"] == "eligible"
    assert payload["official_clean_negative_score_granularity"] == "per_prompt_seed"
    assert payload["official_clean_negative_score_formal_comparison_eligibility"] == "eligible"


@pytest.mark.quick
def test_videomark_bundle_writer_rejects_unmapped_runtime_attack_without_aggregate_fallback(
    tmp_path: Path,
) -> None:
    """VideoMark 不能把不支持的 runtime attack 退化为 temporal attack 均值。"""

    run_root = tmp_path / "runs" / "generative_video_model_probe" / "validation_scale"
    bundle_root = tmp_path / "bundles" / "validation_scale"
    manifest_path = bundle_root / "videomark" / "official_reference_execution_manifest.json"
    prompt_suite_path = tmp_path / "datasets" / "generative_video_prompt_suite" / "prompt_seed_suite.json"
    temporal_path = tmp_path / "official_outputs" / "videomark" / "modelscope" / "512bit" / "temporal_results.json"
    video_path = temporal_path.with_name("video_results.json")
    clean_path = tmp_path / "official_clean_negative_outputs" / "without_watermark" / "modelscope" / "temporal_results.json"
    _write_runtime_records(run_root)
    records_path = run_root / "records" / "runtime_detection_records.jsonl"
    record = json.loads(records_path.read_text(encoding="utf-8").splitlines()[0])
    record["attack_name"] = "video_compression_runtime"
    records_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_prompt_suite(prompt_suite_path)
    _write_json(manifest_path, {"manifest_kind": "test_videomark_execution_manifest"})
    prompt_text = "A small red toy car moves across a table with clear motion."
    digest = hashlib.sha1(prompt_text.encode("utf-8")).hexdigest()[:12]
    video_key = f"prompt_0000_{digest}_0"
    _write_json(temporal_path, {video_key: {"frame drop": {"decode_acc": 0.7, "frames_acc": 0.5}}})
    _write_json(video_path, {video_key: {"decode_acc": 0.9}})
    _write_json(clean_path, {video_key: {"frame drop": {"decode_acc": 0.25, "frames_acc": 0.4}}})

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
    record_path = bundle_root / "videomark" / "records" / "prompt_a__seed_0__video_compression_runtime.json"
    assert not record_path.exists()
