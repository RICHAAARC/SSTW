"""验证 VideoShield 官方 runtime 的轻量 dry-run 闭环。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from external_baseline.videoshield_official_runtime import (
    VideoShieldOfficialRuntimeConfig,
    build_default_videoshield_official_config_from_env,
    run_videoshield_official_runtime,
)
from main.protocol.record_writer import write_jsonl


def _write_fake_videoshield_source(source_dir: Path) -> None:
    """写出 dry-run 只需检查存在性的 VideoShield 官方源码结构。"""

    source_dir.mkdir(parents=True, exist_ok=True)
    for relative_path in (
        "watermark.py",
        "utils.py",
        "watermark_embedding_and_extraction.py",
        "temporal_tamper_localization.py",
    ):
        (source_dir / relative_path).write_text("# fake official source for dry-run\n", encoding="utf-8")


def _write_runtime_records(run_root: Path) -> None:
    """写出两个 attack record, 用于验证同一 prompt / seed 只生成一个 VideoShield 单元。"""

    write_jsonl(run_root / "records" / "runtime_detection_records.jsonl", [
        {
            "runtime_detection_status": "ready",
            "generation_model_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            "prompt_id": "prompt_0",
            "seed_id": "seed_0",
            "trajectory_trace_id": "trace_0",
            "attack_name": "video_compression_runtime",
            "source_video_path": str(run_root / "videos" / "source.mp4"),
            "attacked_video_path": str(run_root / "attacked_videos" / "attacked_a.mp4"),
        },
        {
            "runtime_detection_status": "ready",
            "generation_model_id": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            "prompt_id": "prompt_0",
            "seed_id": "seed_0",
            "trajectory_trace_id": "trace_0",
            "attack_name": "frame_rate_resampling_runtime",
            "source_video_path": str(run_root / "videos" / "source.mp4"),
            "attacked_video_path": str(run_root / "attacked_videos" / "attacked_b.mp4"),
        },
    ])


def _write_prompt_suite(path: Path) -> None:
    """写出 VideoShield runtime 读取的 prompt / seed suite。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "prompts": [{"prompt_id": "prompt_0", "prompt_text": "A red panda eating leaves"}],
                "seeds": [{"seed_id": "seed_0", "seed_value": 123}],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


@pytest.mark.quick
def test_videoshield_official_runtime_dry_run_writes_governed_plan(tmp_path: Path) -> None:
    """dry-run 必须完成源码检查、unit plan 和执行 manifest, 且不能写伪分数。"""

    run_root = tmp_path / "runs" / "generative_video_model_probe" / "validation_scale"
    bundle_root = tmp_path / "bundles"
    output_root = tmp_path / "runtime"
    source_dir = tmp_path / "official_source" / "videoshield"
    prompt_suite_path = tmp_path / "datasets" / "generative_video_prompt_suite" / "prompt_seed_suite.json"
    _write_fake_videoshield_source(source_dir)
    _write_runtime_records(run_root)
    _write_prompt_suite(prompt_suite_path)

    config = VideoShieldOfficialRuntimeConfig(
        run_root=str(run_root),
        bundle_root=str(bundle_root),
        source_dir=str(source_dir),
        output_root=str(output_root),
        resource_root=str(tmp_path / "resources"),
        prompt_suite_path=str(prompt_suite_path),
        dry_run=True,
    )

    manifest = run_videoshield_official_runtime(config)

    assert manifest["execution_status"] == "dry_run_planned"
    assert manifest["input_runtime_detection_record_count"] == 2
    assert manifest["generated_video_unit_count"] == 1
    assert manifest["generated_bundle_record_count"] == 0
    assert manifest["failed_bundle_record_count"] == 0
    assert manifest["unit_plan"]["unit_count"] == 1
    assert Path(manifest["unit_plan"]["unit_plan_path"]).exists()
    assert (bundle_root / "videoshield" / "official_reference_execution_manifest.json").exists()
    assert not list((bundle_root / "videoshield").glob("records/*.json"))


@pytest.mark.quick
def test_videoshield_runtime_payload_stamps_official_adapter_identity() -> None:
    """VideoShield 非 dry-run bundle payload 必须自带官方 adapter 身份, 便于统一分数抽取。"""

    runtime_text = Path("external_baseline/videoshield_official_runtime.py").read_text(encoding="utf-8")

    assert '"official_adapter_baseline_id": BASELINE_ID' in runtime_text
    assert '"official_baseline_id": BASELINE_ID' in runtime_text


@pytest.mark.quick
def test_videoshield_default_config_uses_project_owned_runtime_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """默认配置必须使用 ModelScope 官方路径, 且支持通过环境变量切换 dry-run。"""

    monkeypatch.setenv("SSTW_VIDEOSHIELD_OFFICIAL_DRY_RUN", "true")
    monkeypatch.setenv("SSTW_VIDEOSHIELD_REFERENCE_MAX_RECORDS", "2")

    config = build_default_videoshield_official_config_from_env(
        run_root=tmp_path / "runs" / "generative_video_model_probe" / "validation_scale",
        bundle_root=tmp_path / "bundles",
        source_dir=tmp_path / "source",
        repo_root=tmp_path,
    )

    assert config.model_name == "modelscope"
    assert config.model_path == "damo-vilab/text-to-video-ms-1.7b"
    assert config.dry_run is True
    assert config.max_records == 2
    assert Path(config.output_root).as_posix().endswith("videoshield/official_runtime")
