"""验证 generative_video_model_probe 生成视频正式质量与运动 metric runner。"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import experiments.generative_video_model_probe.formal_metric_runner as formal_metric_runner
import evaluation.metrics.semantic_video_metrics as semantic_video_metrics
from evaluation.metrics.video_file_metrics import compute_paired_video_quality_metrics
from experiments.generative_video_model_probe.formal_metric_runner import run_formal_metric_audit
from evaluation.protocol.record_writer import read_jsonl, write_json, write_jsonl
from scripts.check_results.generative_video_colab_result_checker import check_generative_video_colab_results


def _write_tiny_video(path: Path) -> None:
    """写出一个极小 mp4, 用于 quick 测试真实视频解码链路。"""
    import imageio.v3 as iio
    import numpy as np

    frames = []
    for index in range(6):
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        frame[:, :, 0] = 40 + index * 10
        frame[8:24, 8 + index:16 + index, 1] = 180
        frames.append(frame)
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, frames, fps=4)


def _write_low_motion_video(path: Path) -> None:
    """写出低运动 mp4, 用于验证 formal motion gate 的失败原因可审计。"""
    import imageio.v3 as iio
    import numpy as np

    frame = np.full((32, 32, 3), 120, dtype=np.uint8)
    frame[8:24, 8:24, :] = 180
    frames = [frame.copy() for _ in range(6)]
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, frames, fps=4)


@pytest.mark.quick
def test_paired_video_quality_uses_clean_reference_frames(tmp_path: Path) -> None:
    """水印质量必须由同源 clean-reference 配对失真计算。"""

    reference = tmp_path / "reference.mp4"
    candidate = tmp_path / "candidate.mp4"
    _write_tiny_video(reference)
    _write_tiny_video(candidate)

    metrics = compute_paired_video_quality_metrics(reference, candidate)

    assert metrics["paired_video_quality_status"] == "ready"
    assert metrics["paired_quality_frame_count"] == 6
    assert metrics["paired_watermark_psnr"] > 40.0
    assert metrics["paired_watermark_ssim"] > 0.99
    assert metrics["paired_temporal_delta_error"] < 0.01


@pytest.mark.quick
def test_formal_metric_runner_builds_video_file_metrics(tmp_path: Path) -> None:
    """formal metric runner 必须读取真实 mp4 并记录语义 metric 未配置。"""
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

    audit = run_formal_metric_audit(run_root)
    records = read_jsonl(run_root / "records" / "formal_quality_motion_semantic_records.jsonl")

    assert audit["formal_metric_record_count"] == 1
    assert audit["formal_visual_motion_ready"] is True
    assert audit["formal_semantic_ready"] is False
    assert audit["formal_metric_claim_status"] == "blocked_until_semantic_metric_ready"
    assert records[0]["video_decode_status"] == "ready"
    assert records[0]["motion_delta_threshold"] == 0.0005
    assert records[0]["motion_consistency_failure_reason"] == "none"
    assert records[0]["formal_visual_quality_ready"] is True
    assert records[0]["formal_motion_consistency_ready"] is True
    assert records[0]["formal_semantic_consistency_ready"] is False
    assert records[0]["semantic_metric_status"] == "prompt_text_missing"


@pytest.mark.quick
def test_formal_metric_runner_reports_motion_gate_blocking_reason(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """低运动视频应明确记录 motion gate 阻塞原因, 不能笼统归因到 semantic metric。"""
    run_root = tmp_path / "generative_video_runtime"
    dataset_root = tmp_path / "datasets" / "generative_video_prompt_suite"
    prompt_suite_path = dataset_root / "prompt_seed_suite.json"
    video_path = run_root / "videos" / "low_motion.mp4"
    _write_low_motion_video(video_path)
    digest = hashlib.sha256(video_path.read_bytes()).hexdigest()
    write_json(prompt_suite_path, {
        "prompts": [{
            "prompt_id": "prompt",
            "prompt_text": "A mostly static object with subtle rotation.",
        }],
    })
    write_jsonl(run_root / "records" / "generation_records.jsonl", [{
        "generation_model_id": "model",
        "prompt_id": "prompt",
        "seed_id": "seed",
        "generation_status": "success",
        "video_path": str(video_path),
        "video_sha256": digest,
        "trajectory_trace_id": "trace_0000",
    }])

    def fake_clip_metric(video_path: Path, prompt_text: str, model_id: str, frame_limit: int) -> dict:
        return {
            "semantic_metric_name": "clip_text_video_similarity",
            "semantic_model_id": model_id,
            "semantic_metric_status": "ready",
            "semantic_metric_failure_reason": "none",
            "semantic_consistency_score": 0.31,
            "semantic_consistency_mean_score": 0.31,
            "semantic_consistency_max_score": 0.31,
            "semantic_sampled_frame_count": frame_limit,
            "semantic_frame_limit": frame_limit,
            "semantic_metric_device": "cpu",
        }

    monkeypatch.setattr(formal_metric_runner, "compute_clip_text_video_similarity", fake_clip_metric)

    audit = run_formal_metric_audit(run_root, prompt_suite_path=prompt_suite_path)
    records = read_jsonl(run_root / "records" / "formal_quality_motion_semantic_records.jsonl")

    assert audit["formal_metric_claim_status"] == "blocked_by_formal_motion_consistency"
    assert audit["formal_motion_consistency_blocked_count"] == 1
    assert records[0]["formal_metric_blocking_reason"] == "formal_motion_consistency_not_ready"
    assert records[0]["motion_consistency_failure_reason"] == "motion_delta_below_min"
    assert records[0]["motion_claim_role"] == "positive_motion"
    assert records[0]["formal_motion_gate_policy"] == "positive_motion_requires_min_delta"
    assert records[0]["formal_semantic_consistency_ready"] is True


@pytest.mark.quick
def test_formal_metric_runner_allows_low_motion_for_negative_static_boundary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """negative_static 样本太静止是预期现象, 不应阻断 formal motion gate。"""
    run_root = tmp_path / "generative_video_runtime"
    dataset_root = tmp_path / "datasets" / "generative_video_prompt_suite"
    prompt_suite_path = dataset_root / "prompt_seed_suite.json"
    video_path = run_root / "videos" / "static.mp4"
    _write_low_motion_video(video_path)
    digest = hashlib.sha256(video_path.read_bytes()).hexdigest()
    write_json(prompt_suite_path, {
        "prompts": [{
            "prompt_id": "static_prompt",
            "prompt_text": "A candle remains completely still on a table.",
            "motion_calibration_role": "negative_static",
            "prompt_suite_role": "motion_calibration_negative_static",
        }],
    })
    write_jsonl(run_root / "records" / "generation_records.jsonl", [{
        "generation_model_id": "model",
        "prompt_id": "static_prompt",
        "seed_id": "seed",
        "generation_status": "success",
        "video_path": str(video_path),
        "video_sha256": digest,
        "trajectory_trace_id": "trace_0000",
        "motion_calibration_role": "negative_static",
        "prompt_suite_role": "motion_calibration_negative_static",
    }])

    def fake_clip_metric(video_path: Path, prompt_text: str, model_id: str, frame_limit: int) -> dict:
        return {
            "semantic_metric_name": "clip_text_video_similarity",
            "semantic_model_id": model_id,
            "semantic_metric_status": "ready",
            "semantic_metric_failure_reason": "none",
            "semantic_consistency_score": 0.31,
            "semantic_consistency_mean_score": 0.31,
            "semantic_consistency_max_score": 0.31,
            "semantic_sampled_frame_count": frame_limit,
            "semantic_frame_limit": frame_limit,
            "semantic_metric_device": "cpu",
        }

    monkeypatch.setattr(formal_metric_runner, "compute_clip_text_video_similarity", fake_clip_metric)

    audit = run_formal_metric_audit(run_root, prompt_suite_path=prompt_suite_path)
    records = read_jsonl(run_root / "records" / "formal_quality_motion_semantic_records.jsonl")

    assert audit["formal_metric_claim_status"] == "ready"
    assert audit["formal_motion_consistency_blocked_count"] == 0
    assert records[0]["motion_consistency_failure_reason"] == "motion_delta_below_min"
    assert records[0]["motion_claim_role"] == "negative_static"
    assert records[0]["formal_motion_consistency_ready"] is True
    assert records[0]["formal_motion_gate_policy"] == "low_motion_allowed_for_boundary_role"
    assert records[0]["low_motion_expected_for_role"] is True


@pytest.mark.quick
def test_formal_metric_runner_accepts_clip_semantic_metric(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """formal metric runner 在 CLIP 语义分数达阈值时应解除语义 metric 阻断。"""
    run_root = tmp_path / "generative_video_runtime"
    dataset_root = tmp_path / "datasets" / "generative_video_prompt_suite"
    prompt_suite_path = dataset_root / "prompt_seed_suite.json"
    video_path = run_root / "videos" / "tiny.mp4"
    _write_tiny_video(video_path)
    digest = hashlib.sha256(video_path.read_bytes()).hexdigest()
    write_json(prompt_suite_path, {
        "prompts": [{
            "prompt_id": "prompt",
            "prompt_text": "A small colored square moves across the frame.",
        }],
    })
    write_jsonl(run_root / "records" / "generation_records.jsonl", [{
        "generation_model_id": "model",
        "prompt_id": "prompt",
        "seed_id": "seed",
        "generation_status": "success",
        "video_path": str(video_path),
        "video_sha256": digest,
        "trajectory_trace_id": "trace_0000",
    }])

    def fake_clip_metric(video_path: Path, prompt_text: str, model_id: str, frame_limit: int) -> dict:
        return {
            "semantic_metric_name": "clip_text_video_similarity",
            "semantic_model_id": model_id,
            "semantic_metric_status": "ready",
            "semantic_metric_failure_reason": "none",
            "semantic_consistency_score": 0.31,
            "semantic_consistency_mean_score": 0.31,
            "semantic_consistency_max_score": 0.34,
            "semantic_sampled_frame_count": frame_limit,
            "semantic_frame_limit": frame_limit,
            "semantic_metric_device": "cpu",
        }

    monkeypatch.setattr(formal_metric_runner, "compute_clip_text_video_similarity", fake_clip_metric)

    audit = run_formal_metric_audit(run_root, prompt_suite_path=prompt_suite_path)
    records = read_jsonl(run_root / "records" / "formal_quality_motion_semantic_records.jsonl")

    assert audit["formal_semantic_ready"] is True
    assert audit["formal_quality_motion_semantic_ready"] is True
    assert audit["formal_metric_claim_status"] == "ready"
    assert records[0]["semantic_metric_status"] == "ready"
    assert records[0]["semantic_model_id"] == "openai/clip-vit-base-patch32"
    assert records[0]["formal_semantic_consistency_ready"] is True
    assert records[0]["formal_metric_result_used_for_claim"] is True


@pytest.mark.quick
def test_clip_semantic_metric_accepts_plain_processor_dict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CLIP 语义 metric 不应依赖 processor batch 自带 `.to` 方法。"""
    import numpy as np
    torch = pytest.importorskip("torch", reason="requires optional method-runtime dependency")

    video_path = tmp_path / "placeholder.mp4"
    video_path.write_bytes(b"not_used_by_monkeypatch")

    class FakeProcessor:
        """模拟较旧或差异化 Transformers 环境中返回普通 dict 的 processor。"""

        def __call__(self, text=None, images=None, **kwargs):
            if text is not None:
                return {
                    "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
                    "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
                }
            return {
                "pixel_values": torch.zeros((2, 3, 4, 4), dtype=torch.float32),
            }

    class FakeModel:
        """模拟 CLIP 模型的最小 embedding 接口。"""

        def get_text_features(self, **kwargs):
            return torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)

        def get_image_features(self, **kwargs):
            return torch.tensor([[1.0, 0.0, 0.0], [0.8, 0.2, 0.0]], dtype=torch.float32)

    monkeypatch.setattr(
        semantic_video_metrics,
        "_load_sampled_rgb_frames",
        lambda path, frame_limit: [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(2)],
    )
    monkeypatch.setattr(
        semantic_video_metrics,
        "_load_clip_model_and_processor",
        lambda model_id, device: (FakeModel(), FakeProcessor()),
    )

    result = semantic_video_metrics.compute_clip_text_video_similarity(
        video_path,
        "A small object moves across the frame.",
        device="cpu",
    )

    assert result["semantic_metric_status"] == "ready"
    assert result["semantic_consistency_score"] is not None
    assert result["semantic_sampled_frame_count"] == 2


@pytest.mark.quick
def test_clip_semantic_metric_accepts_pooling_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CLIP 语义 metric 应兼容带 pooler_output 的模型输出对象。"""
    import numpy as np
    torch = pytest.importorskip("torch", reason="requires optional method-runtime dependency")

    video_path = tmp_path / "placeholder.mp4"
    video_path.write_bytes(b"not_used_by_monkeypatch")

    class FakeOutputWithPooling:
        """模拟 Transformers 中 BaseModelOutputWithPooling 的关键字段。"""

        def __init__(self, values: torch.Tensor) -> None:
            self.pooler_output = values

    class FakeProcessor:
        """返回普通 dict, 与真实 processor 的 tensor 字段形态一致。"""

        def __call__(self, text=None, images=None, **kwargs):
            if text is not None:
                return {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)}
            return {"pixel_values": torch.zeros((2, 3, 4, 4), dtype=torch.float32)}

    class FakeModel:
        """模拟返回带 pooler_output 对象的 CLIP 变体。"""

        def get_text_features(self, **kwargs):
            return FakeOutputWithPooling(torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32))

        def get_image_features(self, **kwargs):
            return FakeOutputWithPooling(torch.tensor([[1.0, 0.0, 0.0], [0.7, 0.3, 0.0]], dtype=torch.float32))

    monkeypatch.setattr(
        semantic_video_metrics,
        "_load_sampled_rgb_frames",
        lambda path, frame_limit: [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(2)],
    )
    monkeypatch.setattr(
        semantic_video_metrics,
        "_load_clip_model_and_processor",
        lambda model_id, device: (FakeModel(), FakeProcessor()),
    )

    result = semantic_video_metrics.compute_clip_text_video_similarity(
        video_path,
        "A small object moves across the frame.",
        device="cpu",
    )

    assert result["semantic_metric_status"] == "ready"
    assert result["semantic_metric_failure_reason"] == "none"
    assert result["semantic_consistency_score"] is not None


@pytest.mark.quick
def test_checker_reports_semantic_only_block_after_formal_visual_motion_metrics(tmp_path: Path) -> None:
    """补齐正式质量/运动 metric 后, checker 应只保留正式语义 metric 阻断。"""
    run_root = tmp_path / "generative_video_runtime"
    video_path = run_root / "videos" / "tiny.mp4"
    _write_tiny_video(video_path)
    digest = hashlib.sha256(video_path.read_bytes()).hexdigest()
    write_jsonl(run_root / "records" / "generation_records.jsonl", [{
        "generation_model_id": "model",
        "prompt_id": f"prompt_{index // 2}",
        "seed_id": f"seed_{index % 2}",
        "generation_status": "success",
        "video_path": str(video_path),
        "video_sha256": digest,
        "trajectory_trace_id": f"trace_{index:04d}",
    } for index in range(4)])
    write_jsonl(run_root / "records" / "trajectory_trace.jsonl", [{
        "trajectory_trace_id": f"trace_{trace_index:04d}",
        "trajectory_step_index": step_index,
        "latent_norm": 100.0 - 8.0 * step_index,
        "latent_mean": 0.01 * step_index,
        "latent_std": 0.9 - 0.05 * step_index,
    } for trace_index in range(4) for step_index in range(4)])
    write_jsonl(run_root / "records" / "external_baseline_records.jsonl", [{"external_baseline_runnable_status": "runnable"}])
    write_json(run_root / "artifacts" / "generation_manifest.json", {"artifact_id": "manifest"})
    write_json(run_root / "artifacts" / "generative_video_colab_runtime_decision.json", {
        "stage_id": "generative_video_generation",
        "implementation_decision": "PASS",
        "mechanism_decision": "FAIL",
        "details": {
            "fixed_low_fpr_audit_pass": True,
            "trajectory_observation_gain_confirmed": True,
        },
    })

    run_formal_metric_audit(run_root)
    summary = check_generative_video_colab_results(run_root)

    assert summary["formal_visual_motion_ready_count"] == 4
    assert summary["formal_semantic_ready_count"] == 0
    assert summary["missing_mechanism_requirements"] == ["formal_semantic_metric_missing"]
