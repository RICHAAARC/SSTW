"""构造 B5 Colab 运行所需的 prompt / seed / motion 数据集。"""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path


PROMPT_ITEMS = [
    {
        "prompt_id": "motion_object_pan",
        "prompt_text": "A small red toy car moves from left to right on a wooden table while the camera slowly pans.",
        "prompt_negative_text": "worst quality, inconsistent motion, blurry, jittery, distorted",
        "prompt_category": "object_motion",
        "motion_pattern_id": "left_to_right_pan",
        "prompt_suite_role": "main",
    },
    {
        "prompt_id": "camera_zoom_scene",
        "prompt_text": "A ceramic bird figurine stands near a window while the camera slowly zooms in with stable lighting.",
        "prompt_negative_text": "worst quality, inconsistent motion, blurry, jittery, distorted",
        "prompt_category": "camera_motion",
        "motion_pattern_id": "slow_zoom",
        "prompt_suite_role": "main",
    },
    {
        "prompt_id": "heldout_rotation_scene",
        "prompt_text": "A blue cube rotates gently on a plain gray surface with soft shadows and smooth motion.",
        "prompt_negative_text": "worst quality, inconsistent motion, blurry, jittery, distorted",
        "prompt_category": "heldout_motion",
        "motion_pattern_id": "gentle_rotation",
        "prompt_suite_role": "heldout_prompt",
    },
]

SEED_ITEMS = [
    {"seed_id": "seed_main_a", "seed_value": 101, "prompt_suite_role": "main"},
    {"seed_id": "seed_main_b", "seed_value": 202, "prompt_suite_role": "main"},
    {"seed_id": "seed_heldout_c", "seed_value": 303, "prompt_suite_role": "heldout_seed"},
]


def build_prompt_suite() -> dict:
    """构造独立于测试运行的 prompt suite, 便于 Colab 重复使用。"""
    suite = {
        "prompt_suite_id": "generative_video_probe_prompt_suite_v1",
        "dataset_construction_status": "constructed",
        "dataset_source": "repository_deterministic_prompt_seed_spec",
        "prompts": PROMPT_ITEMS,
        "seeds": SEED_ITEMS,
    }
    payload = json.dumps(suite, ensure_ascii=False, sort_keys=True).encode("utf-8")
    suite["prompt_suite_digest"] = sha256(payload).hexdigest()
    return suite


def write_prompt_suite(output_root: str | Path) -> dict:
    """写出 prompt suite 数据集, 不执行任何模型测试。"""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    suite = build_prompt_suite()
    path = output_root / "prompt_seed_suite.json"
    path.write_text(json.dumps(suite, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "artifact_id": "generative_video_prompt_suite_manifest",
        "artifact_type": "manifest",
        "dataset_construction_status": suite["dataset_construction_status"],
        "dataset_source": suite["dataset_source"],
        "output_paths": [str(path)],
        "rebuild_command": f"python scripts/prepare_generative_video_prompt_suite.py --output-root {output_root.as_posix()}",
    }
    manifest_path = output_root / "prompt_seed_suite_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"prompt_suite_path": str(path), "manifest_path": str(manifest_path), "prompt_count": len(suite["prompts"]), "seed_count": len(suite["seeds"])}


def main() -> None:
    parser = argparse.ArgumentParser(description="构造 B5 Colab prompt / seed 数据集。")
    parser.add_argument("--output-root", default="outputs/datasets/generative_video_prompt_suite")
    args = parser.parse_args()
    print(json.dumps(write_prompt_suite(args.output_root), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
