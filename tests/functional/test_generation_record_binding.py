"""验证 generation record 与正式 trajectory sketch 的不可替换绑定。"""

from __future__ import annotations

import pytest

from evaluation.protocol.generation_record_binding import (
    GENERATION_RECORD_BINDING_FIELDS,
    build_generation_record_binding_digest,
)


def _record() -> dict:
    return {
        "generation_model_id": "model",
        "generation_model_family": "flow_model",
        "generation_model_commit_or_hash": "a" * 40,
        "prompt_id": "prompt-a",
        "prompt_text_hash": "b" * 16,
        "seed_id": "seed-a",
        "generation_seed_random": 7,
        "generation_generator_state_digest_random": "c" * 64,
        "sample_role": "attacked_positive_source",
        "method_variant": "sstw_full_method",
        "watermark_key_derivation_id": "hmac-key-v1",
        "watermark_key_id": "owner-key",
        "scheduler_id": "flow-scheduler",
        "trajectory_time_grid_id": "grid-a",
        "num_inference_steps": 16,
        "guidance_scale": 5.0,
        "video_length_frames": 49,
        "video_resolution": "256x256",
        "fps": 8,
        "video_sha256": "d" * 64,
        "trajectory_trace_id": "trace-a",
        "velocity_causal_pair_id": "e" * 64,
        "flow_tubelet_key_context_digest": "f" * 64,
        "code_commit": "1234567",
    }


@pytest.mark.quick
def test_generation_record_binding_digest_changes_with_video_or_method() -> None:
    record = _record()
    original = build_generation_record_binding_digest(record)

    changed_video = {**record, "video_sha256": "0" * 64}
    changed_method = {**record, "method_variant": "without_velocity_constraint"}

    assert build_generation_record_binding_digest(changed_video) != original
    assert build_generation_record_binding_digest(changed_method) != original


@pytest.mark.quick
def test_generation_record_binding_rejects_any_missing_governed_field() -> None:
    record = _record()
    for field_name in GENERATION_RECORD_BINDING_FIELDS:
        incomplete = dict(record)
        incomplete.pop(field_name)
        with pytest.raises(ValueError, match=field_name):
            build_generation_record_binding_digest(incomplete)
