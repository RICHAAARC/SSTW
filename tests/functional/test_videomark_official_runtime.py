"""验证 VideoMark 官方运行器的轻量闭合语义。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="requires optional method-runtime dependency")

from evaluation.protocol.record_writer import write_jsonl
from external_baseline.videomark_official_runtime import (
    VideoMarkOfficialRuntimeConfig,
    _message_shift,
    run_videomark_official_runtime,
)


@pytest.mark.unit
def test_videomark_message_shift_is_stable_and_in_range() -> None:
    """同一 prompt / seed 必须复用稳定且合法的消息窗口。"""

    record = {"prompt_id": "prompt_a", "seed_id": "seed_a"}
    first = _message_shift(record, 500, 16)
    second = _message_shift(record, 500, 16)

    assert first == second
    assert 0 <= first < 484


@pytest.mark.quick
def test_videomark_runtime_generates_per_attack_official_bundle_without_sstw_scores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VideoMark 必须按 attack 写分数并按 prompt / seed 复用官方生成视频。"""

    run_root = tmp_path / "run"
    bundle_root = tmp_path / "bundles"
    prompt_suite = tmp_path / "prompt_suite.json"
    prompt_suite.write_text(
        json.dumps(
            {
                "prompts": [{"prompt_id": "prompt_a", "prompt_text": "A moving river."}],
                "seeds": [{"seed_id": "seed_a", "seed_value": 17}],
            }
        ),
        encoding="utf-8",
    )
    write_jsonl(
        run_root / "records" / "runtime_detection_records.jsonl",
        [
            {
                "runtime_detection_status": "ready",
                "prompt_id": "prompt_a",
                "seed_id": "seed_a",
                "trajectory_trace_id": "trace_a",
                "attack_name": attack_name,
                "S_final_conservative": 0.99,
            }
            for attack_name in ("video_compression_runtime", "temporal_crop_runtime")
        ],
    )
    backend = {"torch": torch, "numpy": np}
    generation_calls: list[tuple[str, str]] = []

    def fake_generate(
        backend_arg: object,
        config_arg: object,
        record: dict,
        prompt_text: str,
        seed_value: int,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        generation_calls.append((prompt_text, str(seed_value)))
        return (
            np.ones((2, 4, 4, 3), dtype=np.float32),
            np.zeros((2, 4, 4, 3), dtype=np.float32),
            7,
        )

    def fake_attack(
        source_video_path: Path,
        attack_name: str,
        output_path: Path,
        fps: float,
    ) -> tuple[list[np.ndarray], dict[str, object]]:
        """模拟共享文件级 executor, 并保持 clean / watermarked 路径可区分。"""

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"verified-attacked-video")
        value = 0.0 if source_video_path.stem.endswith("_clean") else 1.0
        frames = [np.full((4, 4, 3), value, dtype=np.float32) for _ in range(2)]
        return frames, {
            "runtime_attack_name": attack_name,
            "runtime_attack_implementation_level": "formal_runtime_video_transform",
            "runtime_attack_formal_evidence_level": "formal_runtime_video_transform_verified",
            "runtime_attack_proxy_free": True,
            "runtime_attack_effect_verified": True,
            "runtime_attack_effect_verification_status": "verified",
            "runtime_attack_decoded_effect_verified": True,
            "runtime_attack_effect_verification_basis": "test_file_executor",
            "runtime_attack_output_file_changed": True,
            "runtime_attack_writer_parameters_applied": True,
            "video_fps": fps,
        }

    def fake_detect(
        backend_arg: object,
        config_arg: object,
        frames: list[np.ndarray],
    ) -> dict[str, object]:
        score = float(np.asarray(frames).mean())
        return {
            "external_baseline_score": score,
            "raw_detector_score": score,
            "confidence": score,
            "detected": score >= 0.5,
            "threshold": 0.5,
            "score_semantics": "official_prc_detection_gated_temporal_matching_similarity",
            "score_orientation": "higher_is_more_watermarked",
        }

    monkeypatch.setattr(
        "external_baseline.videomark_official_runtime._load_official_backend",
        lambda config: backend,
    )
    monkeypatch.setattr(
        "external_baseline.videomark_official_runtime._generate_reference_pair",
        fake_generate,
    )
    monkeypatch.setattr(
        "external_baseline.videomark_official_runtime._attack_roundtrip",
        fake_attack,
    )
    monkeypatch.setattr(
        "external_baseline.videomark_official_runtime._detect_payload",
        fake_detect,
    )
    monkeypatch.setattr(
        "external_baseline.videomark_official_runtime.write_video_tchw",
        lambda path, *_args, **_kwargs: Path(path).parent.mkdir(parents=True, exist_ok=True)
        or Path(path).write_bytes(b"source-video")
        or {},
    )

    manifest = run_videomark_official_runtime(
        VideoMarkOfficialRuntimeConfig(
            run_root=str(run_root),
            bundle_root=str(bundle_root),
            source_dir=str(tmp_path / "source"),
            prompt_suite_path=str(prompt_suite),
        )
    )

    assert manifest["execution_status"] == "official_reference_bundle_complete"
    assert manifest["generated_bundle_record_count"] == 2
    assert manifest["generated_prompt_seed_pair_count"] == 1
    assert len(generation_calls) == 1
    payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((bundle_root / "videomark" / "records").glob("*.json"))
    ]
    assert len(payloads) == 2
    assert all(payload["external_baseline_score"] == 1.0 for payload in payloads)
    assert all(payload["external_baseline_clean_negative_score"] == 0.0 for payload in payloads)
    assert all(payload["runtime_attack_effect_verified"] is True for payload in payloads)
    assert all(
        payload["clean_negative_runtime_attack_effect_verified"] is True
        for payload in payloads
    )
    assert all("S_final_conservative" not in payload for payload in payloads)
    assert all(
        payload["official_result_provenance"]
        == "repository_generated_from_third_party_official_code"
        for payload in payloads
    )
