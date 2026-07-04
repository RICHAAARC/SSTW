"""验证生成视频 runtime attack runner。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.generative_video_model_probe.attack_runner import run_runtime_attacks
from experiments.generative_video_model_probe.detection_runner import run_runtime_detection
from main.attacks.video_runtime_attack_protocol import (
    FULL_PAPER_NON_RUNTIME_ATTACK_PROTOCOLS,
    FULL_PAPER_RUNTIME_ATTACKS,
    PILOT_PAPER_RUNTIME_ATTACKS,
    RUNTIME_ATTACK_FAMILY_MINIMUMS_BY_PROFILE,
    apply_runtime_attack_to_frames,
    audit_runtime_attack_protocol_config,
)
from main.protocol.record_writer import write_jsonl


def _write_tiny_video(path: Path) -> None:
    """写出一个极小 mp4, 用于验证 runtime attack 文件级链路。"""
    import imageio.v3 as iio
    import numpy as np

    frames = []
    for index in range(6):
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        frame[:, :, 0] = 40 + index * 8
        frame[8:24, 8 + index:16 + index, 1] = 180
        frames.append(frame)
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, frames, fps=4)


@pytest.mark.quick
def test_runtime_attack_runner_writes_attacked_videos_and_records(tmp_path: Path) -> None:
    """runtime attack runner 必须对真实 mp4 生成 attacked video 与 governed records。"""
    run_root = tmp_path / "generative_video_runtime"
    video_path = run_root / "videos" / "tiny.mp4"
    _write_tiny_video(video_path)
    digest = hashlib.sha256(video_path.read_bytes()).hexdigest()
    write_jsonl(run_root / "records" / "generation_records.jsonl", [{
        "generation_model_id": "model",
        "prompt_id": "prompt",
        "seed_id": "seed",
        "generation_status": "success",
        "video_path": str(video_path),
        "video_sha256": digest,
        "trajectory_trace_id": "trace_0000",
    }])

    summary = run_runtime_attacks(run_root)
    records = [json.loads(line) for line in (run_root / "records" / "runtime_attack_records.jsonl").read_text(encoding="utf-8").splitlines()]

    assert summary["runtime_attack_decision"] == "PASS"
    assert summary["runtime_attack_record_count"] == 3
    assert summary["runtime_attack_ready_count"] == 3
    assert summary["runtime_attack_count"] == 3
    assert len(records) == 3
    assert all(record["attack_runtime_status"] == "ready" for record in records)
    assert all(record["attack_matrix_evidence_level"] == "runtime_video_file" for record in records)
    assert all("sampler_signature_placeholder" in record for record in records)
    assert all(record["trajectory_source_level"] == "runtime_video_file_attack" for record in records)
    assert all(record["flow_state_admissibility_status"] == "not_evaluated" for record in records)
    assert all(Path(record["attacked_video_path"]).exists() for record in records)
    assert all(record["attacked_video_sha256"] for record in records)
    assert (run_root / "tables" / "runtime_attack_table.csv").exists()
    assert (run_root / "artifacts" / "runtime_attack_decision.json").exists()
    assert (run_root / "reports" / "runtime_attack_report.md").exists()


@pytest.mark.quick
def test_runtime_detection_runner_scores_attacked_videos(tmp_path: Path) -> None:
    """runtime detection runner 必须把 attacked videos 接入 governed detection records。"""
    run_root = tmp_path / "generative_video_runtime"
    video_path = run_root / "videos" / "tiny.mp4"
    _write_tiny_video(video_path)
    digest = hashlib.sha256(video_path.read_bytes()).hexdigest()
    write_jsonl(run_root / "records" / "generation_records.jsonl", [{
        "generation_model_id": "model",
        "prompt_id": "prompt",
        "seed_id": "seed",
        "generation_status": "success",
        "video_path": str(video_path),
        "video_sha256": digest,
        "trajectory_trace_id": "trace_0000",
    }])
    write_jsonl(run_root / "records" / "trajectory_trace.jsonl", [
        {"trajectory_trace_id": "trace_0000", "trajectory_step_index": 0, "latent_norm": 100.0, "latent_std": 0.8},
        {"trajectory_trace_id": "trace_0000", "trajectory_step_index": 1, "latent_norm": 85.0, "latent_std": 0.7},
        {"trajectory_trace_id": "trace_0000", "trajectory_step_index": 2, "latent_norm": 70.0, "latent_std": 0.6},
    ])

    attack_summary = run_runtime_attacks(run_root)
    detection_summary = run_runtime_detection(run_root)
    records = [json.loads(line) for line in (run_root / "records" / "runtime_detection_records.jsonl").read_text(encoding="utf-8").splitlines()]

    assert attack_summary["runtime_attack_decision"] == "PASS"
    assert detection_summary["runtime_detection_decision"] == "PASS"
    assert detection_summary["runtime_detection_record_count"] == 3
    assert detection_summary["runtime_detection_ready_count"] == 3
    assert detection_summary["runtime_detection_detectable_count"] == 3
    assert len(records) == 3
    assert all(record["runtime_detection_status"] == "ready" for record in records)
    assert all(record["runtime_detection_evidence_level"] == "runtime_attacked_video_file" for record in records)
    assert all(record["S_runtime_attack_detection"] is not None for record in records)
    assert all(record["S_path_inv"] is not None for record in records)
    assert all(record["S_velocity"] is not None for record in records)
    assert all(record["S_final_conservative"] is not None for record in records)
    assert all(record["flow_state_admissibility_status"] == "proxy_admissible" for record in records)
    assert (run_root / "tables" / "runtime_detection_table.csv").exists()
    assert (run_root / "artifacts" / "runtime_detection_decision.json").exists()
    assert (run_root / "reports" / "runtime_detection_report.md").exists()


@pytest.mark.quick
def test_pilot_and_full_paper_attack_protocol_registers_top_tier_coverage() -> None:
    """pilot/full paper 必须登记分层 attack manifest, 不能退回三类最小攻击。"""

    pilot_audit = audit_runtime_attack_protocol_config(
        {
            "paper_result_level": "pilot_paper",
            "required_runtime_attack_names": list(PILOT_PAPER_RUNTIME_ATTACKS),
        }
    )
    full_audit = audit_runtime_attack_protocol_config(
        {
            "paper_result_level": "full_paper",
            "required_runtime_attack_names": list(FULL_PAPER_RUNTIME_ATTACKS),
            "required_non_runtime_attack_protocols": list(FULL_PAPER_NON_RUNTIME_ATTACK_PROTOCOLS),
        }
    )

    assert pilot_audit["runtime_attack_protocol_decision"] == "PASS"
    assert full_audit["runtime_attack_protocol_decision"] == "PASS"
    assert set(PILOT_PAPER_RUNTIME_ATTACKS) < set(FULL_PAPER_RUNTIME_ATTACKS)
    assert {
        "platform_transcode_proxy_runtime",
        "irregular_frame_drop_runtime",
        "frame_insert_noise_runtime",
        "speed_change_runtime",
        "denoise_proxy_runtime",
        "gamma_correction_runtime",
        "sharpen_runtime",
        "compression_color_jitter_combined_runtime",
        "crop_rotation_combined_runtime",
    }.issubset(set(FULL_PAPER_RUNTIME_ATTACKS))
    assert {
        "watermark_removal_optimization_attack",
        "watermark_spoofing_or_copy_attack",
        "collusion_multi_sample_attack",
        "adversarial_detector_evasion_attack",
    }.issubset(set(FULL_PAPER_NON_RUNTIME_ATTACK_PROTOCOLS))
    for family, minimum_count in RUNTIME_ATTACK_FAMILY_MINIMUMS_BY_PROFILE["full_paper"].items():
        assert full_audit["runtime_attack_family_counts"][family] >= minimum_count
    assert full_audit["missing_non_runtime_attack_protocols"] == []


@pytest.mark.quick
def test_new_top_tier_runtime_attack_transforms_are_executable_on_frames() -> None:
    """新增顶会顶刊级轻量 attack 必须能在帧级协议入口执行。"""

    import numpy as np

    frames = []
    for index in range(8):
        frame = np.zeros((24, 24, 3), dtype=np.uint8)
        frame[:, :, 0] = 20 + index
        frame[4:20, 4:20, 1] = 120
        frames.append(frame)

    for attack_name in (
        "temporal_clip_middle_runtime",
        "frame_duplicate_runtime",
        "spatial_corner_crop_resize_runtime",
        "spatial_mask_runtime",
        "salt_pepper_noise_runtime",
        "color_jitter_runtime",
        "jpeg_frame_compression_runtime",
        "compression_noise_combined_runtime",
        "platform_transcode_proxy_runtime",
        "irregular_frame_drop_runtime",
        "frame_insert_noise_runtime",
        "speed_change_runtime",
        "denoise_proxy_runtime",
        "gamma_correction_runtime",
        "sharpen_runtime",
        "compression_color_jitter_combined_runtime",
        "crop_rotation_combined_runtime",
    ):
        attacked_frames, metadata = apply_runtime_attack_to_frames(frames, attack_name)
        assert attacked_frames
        assert metadata["attack_family"] in {
            "temporal",
            "spatial_geometry",
            "visual_degradation",
            "compression",
            "combined",
        }
        assert metadata["runtime_attack_implementation_level"] == "repository_lightweight_runtime_transform"
