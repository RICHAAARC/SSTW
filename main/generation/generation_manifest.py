"""生成 B5 generation manifest。"""

from __future__ import annotations

from datetime import datetime, timezone


def build_generation_manifest(config_digest: str, input_paths: list[str], output_paths: list[str], status: dict) -> dict:
    """构造生成式视频模型探测的 manifest。"""
    return {
        "artifact_id": "generative_video_generation_manifest",
        "artifact_type": "manifest",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config_digest": config_digest,
        "input_paths": input_paths,
        "output_paths": output_paths,
        "rebuild_command": "python -m experiments.generative_video_model_probe.runner --output-root outputs/runs/generative_video_model_probe",
        "status": status,
    }
