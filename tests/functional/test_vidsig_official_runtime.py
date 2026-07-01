"""验证 VidSig 官方运行器的轻量逻辑。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from external_baseline.official_eval_adapters.vidsig import _run_default
from external_baseline.vidsig_official_runtime import (
    VidSigOfficialRuntimeConfig,
    build_default_vidsig_official_config_from_env,
    run_vidsig_official_runtime,
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
            "seed_id": "seed_main_a",
            "trajectory_trace_id": "trace_a",
            "attack_name": "video_compression_runtime",
            "source_video_path": str(run_root / "videos" / "source.mp4"),
            "attacked_video_path": str(run_root / "videos" / "attacked.mp4"),
        }
    ]
    records_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def _write_prompt_suite(path: Path) -> None:
    """构造包含 prompt_text 和 seed_value 的最小 prompt suite。"""

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
            "seeds": [
                {
                    "seed_id": "seed_main_a",
                    "seed_value": 101,
                }
            ],
        },
    )


def _write_fake_vidsig_source(source_dir: Path) -> None:
    """构造只用于 dry-run 的伪 VidSig 官方源码结构。"""

    (source_dir / "src").mkdir(parents=True, exist_ok=True)
    (source_dir / "yamls").mkdir(parents=True, exist_ok=True)
    (source_dir / "src" / "generate_ms.py").write_text(
        "\n".join([
            "import torch",
            "from torchvision import transforms",
            "normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],  ",
            "                                std=[0.229, 0.224, 0.225])",
            "def main():",
            "    msg_decoder = torch.jit.load('decoder.pt')",
            "    msg_decoder.eval()",
            "    w_frames = []",
            "            decoded = msg_decoder(w_frames)",
            "",
        ]),
        encoding="utf-8",
    )
    (source_dir / "src" / "attack.py").write_text("# fake attack\n", encoding="utf-8")
    (source_dir / "yamls" / "generate_ms.yml").write_text("prompt_file: prompt.txt\n", encoding="utf-8")


@pytest.mark.quick
def test_vidsig_runtime_dry_run_builds_prompt_seed_yaml_and_command(tmp_path: Path) -> None:
    """dry-run 必须只写运行计划, 不触发重型 VidSig 生成。"""

    run_root = tmp_path / "runs" / "generative_video_model_probe" / "validation_scale"
    bundle_root = tmp_path / "bundles" / "validation_scale"
    source_dir = tmp_path / "official_source"
    prompt_suite_path = tmp_path / "datasets" / "generative_video_prompt_suite" / "prompt_seed_suite.json"
    msg_decoder_path = tmp_path / "resources" / "vidsig" / "ckpts" / "msg_decoder" / "dec_48b_whit.torchscript.pt"
    vae_checkpoint_path = tmp_path / "resources" / "vidsig" / "ckpts" / "vae" / "modelscope" / "checkpoint.pth"
    _write_runtime_records(run_root)
    _write_prompt_suite(prompt_suite_path)
    _write_fake_vidsig_source(source_dir)
    msg_decoder_path.parent.mkdir(parents=True, exist_ok=True)
    msg_decoder_path.write_bytes(b"fake-decoder")
    vae_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    vae_checkpoint_path.write_bytes(b"fake-vae")

    config = VidSigOfficialRuntimeConfig(
        run_root=str(run_root),
        bundle_root=str(bundle_root),
        source_dir=str(source_dir),
        output_root=str(tmp_path / "vidsig_runtime"),
        resource_root=str(tmp_path / "resources"),
        prompt_suite_path=str(prompt_suite_path),
        msg_decoder_path=str(msg_decoder_path),
        vae_checkpoint_path=str(vae_checkpoint_path),
        dry_run=True,
    )

    manifest = run_vidsig_official_runtime(config)

    assert manifest["execution_status"] == "dry_run_planned"
    assert manifest["generated_bundle_record_count"] == 0
    assert manifest["generated_video_unit_count"] == 1
    assert manifest["prompt_manifest"]["prompt_count"] == 1
    assert manifest["prompt_manifest"]["seed_count"] == 1
    assert manifest["runtime_patch_audit"]["patch_status"] == "patched"
    assert "convert_diffusers_frames_before_decoder" in manifest["runtime_patch_audit"]["patched_steps"]
    assert manifest["generate_command"][-1] == "src/generate_ms.py"
    patched_source_text = Path(manifest["runtime_source_dir"]) / "src" / "generate_ms.py"
    assert "sstw_prepare_vidsig_decoder_input" in patched_source_text.read_text(encoding="utf-8")
    yaml_text = Path(manifest["prompt_manifest"]["yaml_path"]).read_text(encoding="utf-8")
    assert "damo-vilab/text-to-video-ms-1.7b" in yaml_text
    assert str(msg_decoder_path).replace("\\", "\\\\") in yaml_text
    assert str(vae_checkpoint_path).replace("\\", "\\\\") in yaml_text
    assert "  - 101" in yaml_text


@pytest.mark.quick
def test_vidsig_default_config_prefers_modelscope_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """默认配置应优先选择 ModelScope VAE checkpoint, 避免错配 generate_ms.py。"""

    resource_root = tmp_path / "resources" / "external_baseline"
    decoder = resource_root / "vidsig" / "ckpts" / "msg_decoder" / "dec_48b_whit.torchscript.pt"
    modelscope = resource_root / "vidsig" / "ckpts" / "vae" / "modelscope" / "checkpoint.pth"
    svd = resource_root / "vidsig" / "ckpts" / "vae" / "svd" / "checkpoint.pth"
    decoder.parent.mkdir(parents=True, exist_ok=True)
    decoder.write_bytes(b"decoder")
    svd.parent.mkdir(parents=True, exist_ok=True)
    svd.write_bytes(b"svd")
    modelscope.parent.mkdir(parents=True, exist_ok=True)
    modelscope.write_bytes(b"modelscope")
    monkeypatch.delenv("SSTW_VIDSIG_VAE_CHECKPOINT_PATH", raising=False)
    monkeypatch.delenv("SSTW_VIDSIG_MSG_DECODER_PATH", raising=False)

    config = build_default_vidsig_official_config_from_env(
        run_root=tmp_path / "runs" / "generative_video_model_probe" / "validation_scale",
        bundle_root=tmp_path / "bundles",
        source_dir=tmp_path / "source",
        resource_root=resource_root,
    )

    assert config.msg_decoder_path == str(decoder)
    assert config.vae_checkpoint_path == str(modelscope)


@pytest.mark.quick
def test_vidsig_direct_detector_adapter_fails_closed_without_official_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VidSig adapter 不能直接把 SSTW/Wan 视频送入 detector 伪造成 baseline 结果。"""

    source_dir = tmp_path / "source"
    _write_fake_vidsig_source(source_dir)
    source_video = tmp_path / "source.mp4"
    attacked_video = tmp_path / "attacked.mp4"
    source_video.write_bytes(b"fake")
    attacked_video.write_bytes(b"fake")
    monkeypatch.delenv("SSTW_VIDSIG_ALLOW_DIRECT_DETECTION_ON_SUPPLIED_VIDEO", raising=False)
    monkeypatch.delenv("SSTW_EXTERNAL_BASELINE_OFFICIAL_RESULT_BUNDLE_ROOT", raising=False)

    args = type(
        "Args",
        (),
        {
            "source_video": str(source_video),
            "attacked_video": str(attacked_video),
            "attack_name": "video_compression_runtime",
            "prompt_id": "prompt_a",
            "seed_id": "seed_main_a",
            "trajectory_trace_id": "trace_a",
        },
    )()

    with pytest.raises(RuntimeError, match="vidsig_official_bundle_required"):
        _run_default(args, source_dir, tmp_path / "out.json")
