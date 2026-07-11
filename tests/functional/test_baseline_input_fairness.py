"""验证现代 baseline 使用自身水印与匹配 clean reference。"""

from __future__ import annotations

from pathlib import Path

import pytest

from evaluation.protocol.record_writer import write_jsonl
from external_baseline.runtime_trace_io import comparable_detection_records


@pytest.mark.quick
def test_formal_baseline_units_use_matched_clean_reference_and_full_method_only(
    tmp_path: Path,
) -> None:
    """内部消融视频不得进入 baseline 主比较，baseline 输入不得是 SSTW 视频。"""

    run_root = tmp_path / "run"
    clean_path = run_root / "videos" / "clean.mp4"
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    clean_path.write_bytes(b"clean")
    write_jsonl(run_root / "records" / "generation_records.jsonl", [{
        "generation_status": "success",
        "sample_role": "clean_negative",
        "generation_model_id": "wan",
        "prompt_id": "prompt-a",
        "seed_id": "seed-a",
        "trajectory_trace_id": "clean-trace",
        "video_path": str(clean_path),
    }])
    shared = {
        "runtime_detection_status": "ready",
        "runtime_detection_claim_level": "formal_paper_detector",
        "generation_model_id": "wan",
        "prompt_id": "prompt-a",
        "seed_id": "seed-a",
        "attack_name": "h264_crf23_runtime",
        "attacked_video_path": "sstw-attacked.mp4",
        "cross_model_role": "main_generation_model",
    }
    write_jsonl(run_root / "records" / "runtime_detection_records.jsonl", [
        {**shared, "method_variant": "sstw_full_method"},
        {**shared, "method_variant": "trajectory_only_score"},
        {
            **shared,
            "method_variant": "sstw_full_method",
            "cross_model_role": "cross_model_validation_model",
        },
    ])

    [record] = comparable_detection_records(run_root)

    assert record["source_video_path"] == str(clean_path)
    assert record["baseline_clean_reference_trajectory_trace_id"] == "clean-trace"
    assert record["baseline_clean_reference_status"] == (
        "matched_same_model_prompt_seed_clean_reference"
    )
    assert record["baseline_input_source_policy"] == (
        "baseline_embeds_own_watermark_into_clean_reference"
    )


@pytest.mark.quick
def test_formal_baseline_missing_clean_reference_remains_visible_and_blocking(
    tmp_path: Path,
) -> None:
    """缺少匹配 clean reference 时不得回退使用 SSTW watermarked source。"""

    run_root = tmp_path / "run"
    write_jsonl(run_root / "records" / "runtime_detection_records.jsonl", [{
        "runtime_detection_status": "ready",
        "runtime_detection_claim_level": "formal_paper_detector",
        "method_variant": "sstw_full_method",
        "generation_model_id": "wan",
        "prompt_id": "prompt-a",
        "seed_id": "seed-a",
        "attack_name": "h264_crf23_runtime",
        "source_video_path": "sstw-watermarked.mp4",
        "attacked_video_path": "sstw-attacked.mp4",
        "cross_model_role": "main_generation_model",
    }])

    [record] = comparable_detection_records(run_root)

    assert record["source_video_path"] == ""
    assert record["baseline_clean_reference_status"] == (
        "missing_same_model_prompt_seed_clean_reference"
    )

