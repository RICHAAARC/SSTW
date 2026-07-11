"""定义认证 trajectory sketch 与生成记录之间的稳定绑定协议。"""

from __future__ import annotations

from typing import Any, Mapping

from runtime.core.digest import build_stable_digest


GENERATION_RECORD_BINDING_FIELDS = (
    "generation_model_id",
    "generation_model_family",
    "generation_model_commit_or_hash",
    "prompt_id",
    "prompt_text_hash",
    "seed_id",
    "generation_seed_random",
    "generation_generator_state_digest_random",
    "sample_role",
    "method_variant",
    "watermark_key_derivation_id",
    "watermark_key_id",
    "scheduler_id",
    "trajectory_time_grid_id",
    "num_inference_steps",
    "guidance_scale",
    "video_length_frames",
    "video_resolution",
    "fps",
    "video_sha256",
    "trajectory_trace_id",
    "velocity_causal_pair_id",
    "flow_tubelet_key_context_digest",
    "code_commit",
)


def generation_record_binding_payload(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """提取必须被 HMAC 间接绑定的生成事实，缺失任一字段立即失败。"""

    missing = [
        field_name
        for field_name in GENERATION_RECORD_BINDING_FIELDS
        if record.get(field_name) is None
        or (isinstance(record.get(field_name), str) and not str(record[field_name]).strip())
    ]
    if missing:
        raise ValueError(
            "正式 generation record binding 缺少字段: " + ", ".join(missing)
        )
    return {
        field_name: record[field_name]
        for field_name in GENERATION_RECORD_BINDING_FIELDS
    }


def build_generation_record_binding_digest(
    record: Mapping[str, Any],
) -> str:
    """生成跨文件可复算的 SHA-256，使 sketch 不能替换到其他视频记录。"""

    return build_stable_digest(generation_record_binding_payload(record))
