"""验证生成视频 runtime attack runner。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.generative_video_model_probe.attack_runner import run_runtime_attacks
from experiments.generative_video_model_probe.detection_runner import (
    build_sstw_clean_negative_score_records,
    run_runtime_detection,
)
from experiments.generative_video_model_probe.formal_adaptive_attack_executor import (
    run_formal_adaptive_attack_execution,
)
from main.attacks.video_runtime_attack_protocol import (
    FULL_PAPER_NON_RUNTIME_ATTACK_PROTOCOLS,
    FULL_PAPER_RUNTIME_ATTACKS,
    PILOT_PAPER_RUNTIME_ATTACKS,
    RUNTIME_ATTACK_FAMILY_MINIMUMS_BY_PROFILE,
    apply_runtime_attack_to_frames,
    audit_runtime_attack_protocol_config,
    load_protocol_config_with_shared_attack_protocol,
)
from main.protocol.record_writer import write_jsonl

SMOKE_RUNTIME_ATTACKS = (
    "video_compression_runtime",
    "temporal_crop_runtime",
    "frame_rate_resampling_runtime",
)


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

    summary = run_runtime_attacks(run_root, attack_names=SMOKE_RUNTIME_ATTACKS)
    records = [json.loads(line) for line in (run_root / "records" / "runtime_attack_records.jsonl").read_text(encoding="utf-8").splitlines()]

    assert summary["runtime_attack_decision"] == "PASS"
    assert summary["runtime_attack_record_count"] == 3
    assert summary["runtime_attack_ready_count"] == 3
    assert summary["runtime_attack_count"] == 3
    assert len(records) == 3
    assert all(record["attack_runtime_status"] == "ready" for record in records)
    assert all(record["attack_matrix_evidence_level"] == "runtime_video_file" for record in records)
    assert all("sampler_signature_placeholder" not in record for record in records)
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

    attack_summary = run_runtime_attacks(run_root, attack_names=SMOKE_RUNTIME_ATTACKS)
    detection_summary = run_runtime_detection(run_root)
    records = [json.loads(line) for line in (run_root / "records" / "runtime_detection_records.jsonl").read_text(encoding="utf-8").splitlines()]

    assert attack_summary["runtime_attack_decision"] == "PASS"
    assert detection_summary["runtime_detection_decision"] == "PASS"
    assert detection_summary["runtime_detection_record_count"] == 3
    assert detection_summary["runtime_detection_ready_count"] == 3
    assert detection_summary["runtime_detection_detectable_count"] == 3
    assert len(records) == 3
    assert all(record["runtime_detection_status"] == "ready" for record in records)
    assert all(record["runtime_detection_evidence_level"] == "attacked_video_content_detector" for record in records)
    assert all(record["sstw_detector_evidence_level"] == "attacked_video_content_detector" for record in records)
    assert all(record["trajectory_trace_used_for_score"] is False for record in records)
    assert all(record["S_runtime_attack_detection"] is not None for record in records)
    assert all(record["sstw_raw_detector_score"] is not None for record in records)
    assert all(record["S_final_conservative"] is not None for record in records)
    assert all(record["flow_state_admissibility_status"] == "formal_video_detector_admissible" for record in records)
    assert (run_root / "tables" / "runtime_detection_table.csv").exists()
    assert (run_root / "artifacts" / "runtime_detection_decision.json").exists()
    assert (run_root / "reports" / "runtime_detection_report.md").exists()


@pytest.mark.quick
def test_runtime_detection_requires_clean_negative_when_protocol_config_demands_it(tmp_path: Path) -> None:
    """paper profile 配置要求 clean negative 时, runtime detection 不能在缺失 clean negative 下放行。"""

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
    config_path = tmp_path / "paper_protocol.json"
    config_path.write_text(json.dumps({
        "minimum_clean_negative_count": 4,
        "minimum_calibration_negative_event_count": 2,
        "minimum_heldout_test_negative_event_count": 2,
    }), encoding="utf-8")

    run_runtime_attacks(run_root, attack_names=SMOKE_RUNTIME_ATTACKS)
    detection_summary = run_runtime_detection(run_root, config_path=config_path)

    assert detection_summary["runtime_detection_decision"] == "FAIL"
    assert detection_summary["sstw_clean_negative_required"] is True
    assert detection_summary["sstw_clean_negative_record_count"] == 0
    assert detection_summary["sstw_clean_negative_requirement_met"] is False


@pytest.mark.quick
def test_clean_negative_score_builder_expands_formal_key_trials(tmp_path: Path) -> None:
    """clean negative 必须从真实 clean video 扩展出 fixed-FPR 校准事件。"""
    run_root = tmp_path / "generative_video_runtime"
    calibration_video = run_root / "videos" / "clean_calibration.mp4"
    test_video = run_root / "videos" / "clean_test.mp4"
    _write_tiny_video(calibration_video)
    _write_tiny_video(test_video)
    write_jsonl(run_root / "records" / "generation_records.jsonl", [
        {
            "generation_model_id": "model",
            "prompt_id": "prompt_a",
            "seed_id": "seed_calib",
            "generation_status": "success",
            "video_path": str(calibration_video),
            "trajectory_trace_id": "trace_clean_calib",
            "sample_role": "clean_negative",
            "watermark_embedding_status": "clean_unwatermarked_reference",
            "split": "calibration",
        },
        {
            "generation_model_id": "model",
            "prompt_id": "prompt_a",
            "seed_id": "seed_test",
            "generation_status": "success",
            "video_path": str(test_video),
            "trajectory_trace_id": "trace_clean_test",
            "sample_role": "clean_negative",
            "watermark_embedding_status": "clean_unwatermarked_reference",
            "split": "test",
        },
    ])
    config_path = tmp_path / "protocol.json"
    config_path.write_text(json.dumps({
        "minimum_clean_negative_count": 8,
        "minimum_calibration_negative_event_count": 4,
        "minimum_heldout_test_negative_event_count": 4,
    }), encoding="utf-8")

    records = build_sstw_clean_negative_score_records(run_root, config_path=config_path)

    assert len(records) == 8
    assert sum(1 for record in records if record["split"] == "calibration") == 4
    assert sum(1 for record in records if record["split"] == "test") == 4
    assert all(record["metric_status"] == "measured_formal" for record in records)
    assert all(record["clean_negative_evidence_level"] == "project_owned_clean_video_content_detector" for record in records)
    assert all(record["trajectory_trace_used_for_score"] is False for record in records)
    assert len({record["clean_negative_unit_id"] for record in records}) == 8


@pytest.mark.quick
def test_formal_adaptive_attack_executor_writes_measured_records(tmp_path: Path) -> None:
    """non-runtime / adaptive 协议必须由真实视频文件重新检测后进入正式记录。"""
    run_root = tmp_path / "generative_video_runtime"
    video_path = run_root / "videos" / "positive.mp4"
    _write_tiny_video(video_path)
    write_jsonl(run_root / "records" / "generation_records.jsonl", [{
        "generation_model_id": "model",
        "prompt_id": "prompt",
        "seed_id": "seed",
        "generation_status": "success",
        "video_path": str(video_path),
        "trajectory_trace_id": "trace_positive",
        "sample_role": "attacked_positive_source",
        "watermark_embedding_status": "sampling_time_key_conditioned_latent_constraint",
        "split": "test",
    }])
    config_path = tmp_path / "protocol.json"
    config_path.write_text(json.dumps({
        "paper_result_level": "probe_paper",
        "target_fpr": 0.1,
        "required_non_runtime_attack_protocols": [
            "wrong_key_attack",
            "detector_probing_with_public_negatives",
        ],
    }), encoding="utf-8")

    summary = run_formal_adaptive_attack_execution(run_root, config_path)
    records = [
        json.loads(line)
        for line in (run_root / "records" / "formal_adaptive_attack_execution_records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert summary["formal_adaptive_attack_execution_decision"] == "PASS"
    assert summary["formal_adaptive_attack_execution_record_count"] == 2
    assert {record["non_runtime_attack_protocol"] for record in records} == {
        "wrong_key_attack",
        "detector_probing_with_public_negatives",
    }
    assert all(record["metric_status"] == "measured_formal" for record in records)
    assert all(record["adaptive_attack_evidence_level"] == "formal_adaptive_attack_execution" for record in records)
    assert all(Path(record["adaptive_attack_input_video_path"]).exists() for record in records)
    assert (run_root / "artifacts" / "formal_adaptive_attack_execution_decision.json").exists()


@pytest.mark.quick
def test_probe_pilot_and_full_paper_attack_protocol_registers_top_tier_coverage() -> None:
    """probe/pilot/full paper 必须登记分层 attack manifest, 不能退回三类最小攻击。"""

    probe_audit = audit_runtime_attack_protocol_config(
        {
            "paper_result_level": "probe_paper",
            "required_runtime_attack_names": list(FULL_PAPER_RUNTIME_ATTACKS),
            "required_non_runtime_attack_protocols": list(FULL_PAPER_NON_RUNTIME_ATTACK_PROTOCOLS),
        }
    )
    pilot_audit = audit_runtime_attack_protocol_config(
        {
            "paper_result_level": "pilot_paper",
            "required_runtime_attack_names": list(PILOT_PAPER_RUNTIME_ATTACKS),
            "required_non_runtime_attack_protocols": list(FULL_PAPER_NON_RUNTIME_ATTACK_PROTOCOLS),
        }
    )
    full_audit = audit_runtime_attack_protocol_config(
        {
            "paper_result_level": "full_paper",
            "required_runtime_attack_names": list(FULL_PAPER_RUNTIME_ATTACKS),
            "required_non_runtime_attack_protocols": list(FULL_PAPER_NON_RUNTIME_ATTACK_PROTOCOLS),
        }
    )

    assert probe_audit["runtime_attack_protocol_decision"] == "PASS"
    assert pilot_audit["runtime_attack_protocol_decision"] == "PASS"
    assert full_audit["runtime_attack_protocol_decision"] == "PASS"
    assert probe_audit["required_runtime_attack_count"] == len(FULL_PAPER_RUNTIME_ATTACKS)
    assert probe_audit["missing_non_runtime_attack_protocols"] == []
    assert set(PILOT_PAPER_RUNTIME_ATTACKS) == set(FULL_PAPER_RUNTIME_ATTACKS)
    assert pilot_audit["missing_non_runtime_attack_protocols"] == []
    assert {
        "platform_transcode_runtime",
        "irregular_frame_drop_runtime",
        "frame_insert_noise_runtime",
        "speed_change_runtime",
        "denoise_runtime",
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
def test_profile_protocols_resolve_shared_attack_manifest() -> None:
    """三个论文 profile 必须通过共享配置解析出同一份 attack 协议清单。"""

    profile_paths = (
        "configs/protocol/probe_paper_generative_probe.json",
        "configs/protocol/pilot_paper_generative_probe.json",
        "configs/protocol/full_paper_generative_probe.json",
    )
    for config_path in profile_paths:
        raw_config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        resolved = load_protocol_config_with_shared_attack_protocol(config_path)
        assert "required_runtime_attack_names" not in raw_config
        assert "required_non_runtime_attack_protocols" not in raw_config
        assert resolved["shared_attack_protocol_id"] == "top_tier_video_watermark_attack_protocol_v1"
        assert resolved["required_runtime_attack_names"] == list(FULL_PAPER_RUNTIME_ATTACKS)
        assert resolved["required_non_runtime_attack_protocols"] == list(FULL_PAPER_NON_RUNTIME_ATTACK_PROTOCOLS)
        assert resolved["minimum_attack_count"] == len(FULL_PAPER_RUNTIME_ATTACKS)
        assert resolved["minimum_non_runtime_attack_protocol_count"] == len(FULL_PAPER_NON_RUNTIME_ATTACK_PROTOCOLS)


@pytest.mark.quick
def test_paper_profile_attack_event_minimums_match_declared_sample_capacity() -> None:
    """paper profile 的 per-attack 事件下限不能超过配置声明的样本容量。"""

    pilot_config = load_protocol_config_with_shared_attack_protocol(
        "configs/protocol/pilot_paper_generative_probe.json"
    )
    probe_config = load_protocol_config_with_shared_attack_protocol(
        "configs/protocol/probe_paper_generative_probe.json"
    )
    full_config = load_protocol_config_with_shared_attack_protocol(
        "configs/protocol/full_paper_generative_probe.json"
    )

    probe_positive_capacity_per_attack = int(probe_config["minimum_unique_video_count"])
    probe_required_attack_count = len(probe_config["required_runtime_attack_names"])
    assert int(probe_config["minimum_attack_event_count_per_attack"]) == 5
    assert int(probe_config["minimum_attack_event_count_per_attack"]) <= probe_positive_capacity_per_attack
    assert int(probe_config["minimum_heldout_attacked_positive_event_count"]) == (
        int(probe_config["minimum_attack_event_count_per_attack"]) * probe_required_attack_count
    )

    pilot_positive_capacity_per_attack = int(pilot_config["minimum_unique_video_count"])
    pilot_required_attack_count = len(pilot_config["required_runtime_attack_names"])
    assert int(pilot_config["minimum_attack_event_count_per_attack"]) == 50
    assert int(pilot_config["minimum_attack_event_count_per_attack"]) <= pilot_positive_capacity_per_attack
    assert int(pilot_config["minimum_heldout_attacked_positive_event_count"]) == (
        int(pilot_config["minimum_attack_event_count_per_attack"]) * pilot_required_attack_count
    )

    full_positive_capacity_per_attack = int(full_config["minimum_unique_video_count"])
    full_required_attack_count = len(full_config["required_runtime_attack_names"])
    assert int(full_config["minimum_calibration_seed_per_prompt"]) == 4
    assert int(full_config["minimum_test_seed_per_prompt"]) == 4
    assert int(full_config["minimum_calibration_seed_per_prompt"]) + int(full_config["minimum_test_seed_per_prompt"]) == int(
        full_config["minimum_seed_per_prompt"]
    )
    assert int(full_config["minimum_calibration_unique_video_count"]) == (
        int(full_config["minimum_prompt_count"]) * int(full_config["minimum_calibration_seed_per_prompt"])
    )
    assert int(full_config["minimum_test_unique_video_count"]) == (
        int(full_config["minimum_prompt_count"]) * int(full_config["minimum_test_seed_per_prompt"])
    )
    assert int(full_config["minimum_calibration_unique_video_count"]) + int(full_config["minimum_test_unique_video_count"]) == int(
        full_config["minimum_unique_video_count"]
    )
    assert int(full_config["minimum_attack_event_count_per_attack"]) == 1000
    assert int(full_config["minimum_attack_event_count_per_attack"]) <= full_positive_capacity_per_attack
    assert int(full_config["minimum_heldout_attacked_positive_event_count"]) == (
        int(full_config["minimum_attack_event_count_per_attack"]) * full_required_attack_count
    )


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
        "platform_transcode_runtime",
        "irregular_frame_drop_runtime",
        "frame_insert_noise_runtime",
        "speed_change_runtime",
        "denoise_runtime",
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
        assert metadata["runtime_attack_implementation_level"] == "formal_runtime_video_transform"
        assert metadata["runtime_attack_proxy_free"] is True
