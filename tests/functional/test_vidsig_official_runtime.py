"""验证 VidSig 官方运行器的轻量逻辑。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from external_baseline.official_eval_adapters.vidsig import _run_default
import external_baseline.vidsig_official_runtime as vidsig_runtime
from external_baseline.vidsig_official_runtime import (
    VidSigOfficialRuntimeConfig,
    build_default_vidsig_official_config_from_env,
    run_vidsig_official_runtime,
    write_vidsig_official_bundle_records,
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
    (source_dir / "src" / "attack.py").write_text(
        "\n".join([
            "import os",
            "import numpy as np",
            "def main():",
            "    params = type('P', (), {'output_dir': '.', 'attack_type': 'clean', 'factor': 2.0})()",
            "    accuracys = [0.8]",
            "    fprs = ['1e-2']",
            "    tprs = {'1e-2': [1]}",
            "    with open(os.path.join(params.output_dir, 'log.txt'), 'a') as f:",
            "        f.write(f'{params.attack_type} {params.factor}\\n')     ",
            "        for fpr in fprs:",
            "            f.write(f'{params.attack_type} fpr = {fpr}: {np.mean(tprs[fpr])}\\n')",
            "",
        ]),
        encoding="utf-8",
    )
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
    assert "write_attack_log_bit_accuracy_for_formal_score" in manifest["runtime_patch_audit"]["patched_steps"]
    assert manifest["generate_command"][-1] == "src/generate_ms.py"
    patched_source_text = Path(manifest["runtime_source_dir"]) / "src" / "generate_ms.py"
    patched_generate_source = patched_source_text.read_text(encoding="utf-8")
    assert "sstw_prepare_vidsig_decoder_input" in patched_generate_source
    assert "sstw_write_vidsig_generate_progress" in patched_generate_source
    patched_attack_text = Path(manifest["runtime_source_dir"]) / "src" / "attack.py"
    assert "sstw formal bit accuracy" in patched_attack_text.read_text(encoding="utf-8")
    yaml_text = Path(manifest["prompt_manifest"]["yaml_path"]).read_text(encoding="utf-8")
    assert "damo-vilab/text-to-video-ms-1.7b" in yaml_text
    assert str(msg_decoder_path).replace("\\", "\\\\") in yaml_text
    assert str(vae_checkpoint_path).replace("\\", "\\\\") in yaml_text
    assert "  - 101" in yaml_text
    assert "sstw_progress_path:" in yaml_text
    assert manifest["prompt_manifest"]["expected_generated_video_unit_count"] == 1
    assert manifest["prompt_manifest"]["expected_generate_ms_video_file_count"] == 2


@pytest.mark.quick
def test_vidsig_generate_ms_progress_probe_counts_clean_and_watermarked_outputs(tmp_path: Path) -> None:
    """VidSig generate_ms 长命令进度应来自官方输出文件数量。"""

    clean_dir = tmp_path / "official_generate_ms_outputs" / "original" / "videos"
    watermarked_dir = tmp_path / "official_generate_ms_outputs" / "video_signature" / "videos"
    clean_dir.mkdir(parents=True)
    watermarked_dir.mkdir(parents=True)
    (clean_dir / "0.mp4").write_bytes(b"clean")
    (clean_dir / "1.mp4").write_bytes(b"clean")
    (watermarked_dir / "0.mp4").write_bytes(b"watermarked")
    prompt_manifest = {
        "clean_video_dir": str(clean_dir),
        "watermarked_video_dir": str(watermarked_dir),
        "generate_ms_progress_path": str(tmp_path / "official_generate_ms_outputs" / "logs" / "sstw_generate_ms_progress.jsonl"),
        "prompt_rows": [{"prompt_id": "p0"}, {"prompt_id": "p1"}],
        "seed_rows": [{"seed_id": "s0"}],
    }
    progress_path = Path(prompt_manifest["generate_ms_progress_path"])
    progress_path.parent.mkdir(parents=True)
    progress_path.write_text(
        json.dumps({
            "phase": "watermarked_video_generation_start",
            "completed": 3,
            "total": 4,
            "current_unit": 2,
            "unit_total": 2,
        }, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    probe = vidsig_runtime._build_vidsig_generate_ms_progress_probe(prompt_manifest)
    progress = probe()

    assert progress["phase"] == "watermarked_video_generation_start"
    assert progress["generated_video_files"] == "3/4"
    assert progress["clean_video_files"] == "2/2"
    assert progress["watermarked_video_files"] == "1/2"
    assert progress["progress_percent"] == "75.0"
    assert progress["official_reported_progress"] == "3/4"
    assert progress["current_video_unit"] == "2/2"


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


@pytest.mark.quick
def test_vidsig_bundle_writer_records_clean_negative_score(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VidSig official bundle 必须包含 clean negative 官方检测分数。"""

    run_root = tmp_path / "runs" / "generative_video_model_probe" / "validation_scale"
    bundle_root = tmp_path / "bundles"
    output_root = tmp_path / "vidsig_runtime"
    source_dir = tmp_path / "source"
    prompt_suite_path = tmp_path / "datasets" / "generative_video_prompt_suite" / "prompt_seed_suite.json"
    manifest_path = bundle_root / "vidsig" / "official_reference_execution_manifest.json"
    _write_runtime_records(run_root)
    _write_prompt_suite(prompt_suite_path)
    _write_fake_vidsig_source(source_dir)
    clean_dir = output_root / "official_generate_ms_outputs" / "original"
    watermarked_dir = output_root / "official_generate_ms_outputs" / "video_signature"
    clean_dir.mkdir(parents=True, exist_ok=True)
    watermarked_dir.mkdir(parents=True, exist_ok=True)
    (clean_dir / "0.mp4").write_bytes(b"clean")
    (watermarked_dir / "0.mp4").write_bytes(b"watermarked")
    prompt_manifest = {
        "prompt_rows": [{"prompt_id": "prompt_a", "prompt_text": "prompt"}],
        "seed_rows": [{"seed_id": "seed_main_a", "seed_value": 101}],
        "clean_video_dir": str(clean_dir),
        "watermarked_video_dir": str(watermarked_dir),
    }
    config = VidSigOfficialRuntimeConfig(
        run_root=str(run_root),
        bundle_root=str(bundle_root),
        source_dir=str(source_dir),
        output_root=str(output_root),
        resource_root=str(tmp_path / "resources"),
        prompt_suite_path=str(prompt_suite_path),
        msg_decoder_path=str(tmp_path / "decoder.pt"),
        vae_checkpoint_path=str(tmp_path / "vae.pt"),
    )

    def fake_read_video_frames(path: Path) -> list[str]:
        """区分原始官方视频和写出后重读的视频, 验证检测帧数组来自文件级路径。"""

        path = Path(path)
        if path.name.endswith("_clean_negative.mp4"):
            return ["decoded_clean_negative_0", "decoded_clean_negative_1"]
        if path.name.endswith("_attacked.mp4"):
            return ["decoded_attacked_0", "decoded_attacked_1", "decoded_attacked_2"]
        return ["frame_0", "frame_1", "frame_2", "frame_3"]

    saved_frame_arrays: dict[str, list[str]] = {}

    def fake_save_frame_array(path: Path, frames: list[str]) -> None:
        """记录传给 VidSig 官方 attack.py 的帧数组来源。"""

        saved_frame_arrays[Path(path).name] = list(frames)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"npy")

    monkeypatch.setattr(vidsig_runtime, "_read_video_frames", fake_read_video_frames)
    monkeypatch.setattr(vidsig_runtime, "_write_video_frames", lambda path, _frames, *, fps: Path(path).parent.mkdir(parents=True, exist_ok=True) or Path(path).write_bytes(b"video"))
    monkeypatch.setattr(vidsig_runtime, "_save_frame_array", fake_save_frame_array)

    call_state = {"count": 0}

    def fake_attack(**kwargs: object) -> dict[str, object]:
        call_state["count"] += 1
        output_dir = Path(kwargs["output_dir"])  # type: ignore[index]
        output_dir.mkdir(parents=True, exist_ok=True)
        score = 0.8 if call_state["count"] == 1 else 0.1
        fixed_fpr_score = 1.0 if call_state["count"] == 1 else 0.0
        (output_dir / "log.txt").write_text(
            f"clean bit accuracy: {score}\nclean fpr = 1e-2: {fixed_fpr_score}\n",
            encoding="utf-8",
        )
        return {"return_code": 0, "stdout_path": str(output_dir / "stdout.txt"), "stderr_path": str(output_dir / "stderr.txt")}

    monkeypatch.setattr(vidsig_runtime, "_run_vidsig_attack_py", fake_attack)

    result = write_vidsig_official_bundle_records(
        run_root=run_root,
        bundle_root=bundle_root,
        manifest_path=manifest_path,
        runtime_source_dir=source_dir,
        prompt_manifest=prompt_manifest,
        config=config,
    )

    assert result["generated_bundle_record_count"] == 1
    payload = json.loads((bundle_root / "vidsig" / "records" / "prompt_a__seed_main_a__video_compression_runtime.json").read_text(encoding="utf-8"))
    assert payload["official_adapter_baseline_id"] == "vidsig"
    assert payload["official_baseline_id"] == "vidsig"
    assert payload["external_baseline_score"] == 0.8
    assert payload["external_baseline_clean_negative_score"] == 0.1
    assert payload["external_baseline_clean_negative_score_semantics"] == "payload_bit_accuracy_extraction_score"
    assert payload["external_baseline_clean_negative_video_path"].endswith("_clean_negative.mp4")
    assert payload["official_vidsig_tpr_at_fpr_1e_2"] == 1.0
    assert payload["official_clean_negative_vidsig_tpr_at_fpr_1e_2"] == 0.0
    assert payload["official_score_granularity"] == "per_prompt_seed_attack"
    assert payload["official_score_value_type"] == "payload_bit_accuracy_score"
    assert payload["official_score_formal_comparison_eligibility"] == "eligible"
    assert payload["official_score_formal_comparison_block_reason"] == "none"
    assert payload["attacked_video_decoded_frame_count"] == 3
    assert payload["clean_negative_video_decoded_frame_count"] == 2
    assert saved_frame_arrays["sstw_attacked_video.npy"] == [
        "decoded_attacked_0",
        "decoded_attacked_1",
        "decoded_attacked_2",
    ]
    assert saved_frame_arrays["sstw_clean_negative_video.npy"] == [
        "decoded_clean_negative_0",
        "decoded_clean_negative_1",
    ]
