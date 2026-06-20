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
    {
        "prompt_id": "liquid_pour_closeup",
        "prompt_text": "Clear water pours from a glass pitcher into a small cup on a kitchen counter with steady camera framing.",
        "prompt_negative_text": "worst quality, inconsistent motion, blurry, jittery, distorted",
        "prompt_category": "fluid_motion",
        "motion_pattern_id": "continuous_pour",
        "prompt_suite_role": "pilot_main",
    },
    {
        "prompt_id": "walking_robot_sideview",
        "prompt_text": "A small toy robot walks slowly from right to left across a clean desk while the camera remains fixed.",
        "prompt_negative_text": "worst quality, inconsistent motion, blurry, jittery, distorted",
        "prompt_category": "articulated_motion",
        "motion_pattern_id": "sideways_walk",
        "prompt_suite_role": "pilot_main",
    },
    {
        "prompt_id": "falling_leaves_static_camera",
        "prompt_text": "Several yellow leaves fall gently in front of a park bench while the camera stays still.",
        "prompt_negative_text": "worst quality, inconsistent motion, blurry, jittery, distorted",
        "prompt_category": "natural_motion",
        "motion_pattern_id": "downward_fall",
        "prompt_suite_role": "pilot_main",
    },
    {
        "prompt_id": "turntable_product_orbit",
        "prompt_text": "A white ceramic mug rotates on a small turntable under soft studio lighting with a fixed camera.",
        "prompt_negative_text": "worst quality, inconsistent motion, blurry, jittery, distorted",
        "prompt_category": "object_rotation",
        "motion_pattern_id": "turntable_rotation",
        "prompt_suite_role": "pilot_main",
    },
    {
        "prompt_id": "train_window_lateral_motion",
        "prompt_text": "A landscape moves laterally outside a train window while the interior frame remains stable.",
        "prompt_negative_text": "worst quality, inconsistent motion, blurry, jittery, distorted",
        "prompt_category": "background_motion",
        "motion_pattern_id": "lateral_background_shift",
        "prompt_suite_role": "pilot_main",
    },
]

SEED_ITEMS = [
    {"seed_id": "seed_main_a", "seed_value": 101, "prompt_suite_role": "main"},
    {"seed_id": "seed_main_b", "seed_value": 202, "prompt_suite_role": "main"},
    {"seed_id": "seed_heldout_c", "seed_value": 303, "prompt_suite_role": "heldout_seed"},
]


MOTION_CALIBRATION_SEED_ITEMS = [
    {"seed_id": f"seed_motion_calib_{index:02d}", "seed_value": 1001 + index * 37, "prompt_suite_role": "motion_calibration"}
    for index in range(8)
]


def _build_motion_calibration_prompts() -> list[dict]:
    """构造 motion threshold calibration 专用 prompt split。

    该函数属于项目特定写法。它通过 16 个 negative_static prompt、8 个 positive_motion prompt 和
    4 个 ambiguous_low_motion prompt, 配合 8 个 calibration seed, 形成 128 / 64 / 32 的校准规模。
    这些样本只允许用于 threshold calibration, 不得与 pilot main 或 evaluation split 混用。
    """
    prompts: list[dict] = []
    negative_templates = [
        "A locked-off static photograph of a ceramic mug on a wooden desk, frozen frame, no camera movement, no object motion, no lighting change.",
        "A stationary book rests on a clean table as a single static photograph, fixed camera, no pan, no zoom, no moving shadows.",
        "A quiet indoor plant stands beside a window in a frozen still frame, no breeze, no camera movement, stable lighting.",
        "A framed picture hangs on a plain wall in a locked-off static shot, no object motion, no exposure change, no camera shake.",
        "A bowl of fruit sits motionless on a kitchen counter as a still photo, fixed camera, no zoom, no flicker, no background movement.",
        "A small toy house remains still on a shelf in a frozen frame, no camera movement, no object movement, constant illumination.",
        "A pair of shoes is placed on a floor mat in a single static composition, no camera pan, no zoom, no lighting variation.",
        "A white candle stands unlit on a table in a locked-off still image, no flame, no motion, no changing shadows.",
        "A folded towel lies on a chair in a frozen frame, fixed camera, no object motion, no camera motion, no exposure change.",
        "A glass bottle stands on a windowsill as a still photograph, no reflections moving, no camera movement, stable lighting.",
        "A notebook and pencil rest on a desk in a static close-up, no hand movement, no camera motion, no lighting change.",
        "A decorative vase remains still on a side table in a frozen frame, fixed camera, no zoom, no pan, no shadow movement.",
        "A plush bear sits on a sofa in a single still photograph, no camera motion, no object movement, no flicker.",
        "A closed laptop rests on a table in a locked-off static shot, no screen animation, no movement, stable exposure.",
        "A chess board remains motionless under a fixed overhead camera, frozen frame, no moving pieces, no lighting change.",
        "A wall clock is shown as a still object with no visible hand movement, fixed camera, no zoom, no pan, no illumination change.",
    ]
    positive_templates = [
        "A bright red toy car travels from the far left edge to the far right edge across a wooden table, large visible displacement, fixed camera.",
        "A blue cube spins rapidly for multiple full rotations on a turntable, clearly changing orientation in every frame, fixed camera.",
        "Many yellow leaves fall from the top of the frame to the bottom across the whole scene, large visible downward motion, fixed camera.",
        "Clear water continuously pours from a raised glass pitcher into a cup, visible stream motion across many frames, fixed camera.",
        "A toy robot walks from the right edge to the left edge across a clean desk, large visible body displacement, fixed camera.",
        "A colored ball rolls diagonally from the lower left corner to the upper right corner across a flat floor, large visible displacement.",
        "A paper airplane glides across the entire frame from left to right, changing position strongly between frames, fixed camera.",
        "A hand waves a small flag widely from side to side across the frame, repeated large amplitude motion, fixed camera.",
    ]
    ambiguous_templates = [
        "A flower stem sways very slightly in a weak indoor breeze while the camera remains fixed.",
        "A small object rotates extremely slowly on a turntable with subtle visible motion.",
        "A camera performs a barely perceptible slow zoom toward a static ceramic figurine.",
        "Soft shadows shift subtly across a still table scene with very low apparent motion.",
    ]
    for index, prompt_text in enumerate(negative_templates):
        prompts.append({
            "prompt_id": f"motion_calib_negative_static_{index:02d}",
            "prompt_text": prompt_text,
            "prompt_negative_text": "motion blur, camera shake, object movement, flicker, jitter, distorted",
            "prompt_category": "motion_threshold_calibration",
            "motion_pattern_id": "negative_static",
            "prompt_suite_role": "motion_calibration_negative_static",
            "motion_calibration_role": "negative_static",
            "split": "calibration",
        })
    for index, prompt_text in enumerate(positive_templates):
        prompts.append({
            "prompt_id": f"motion_calib_positive_motion_{index:02d}",
            "prompt_text": prompt_text,
            "prompt_negative_text": "static image, frozen frame, no motion, flicker, jitter, distorted",
            "prompt_category": "motion_threshold_calibration",
            "motion_pattern_id": "positive_motion",
            "prompt_suite_role": "motion_calibration_positive_motion",
            "motion_calibration_role": "positive_motion",
            "split": "calibration",
        })
    for index, prompt_text in enumerate(ambiguous_templates):
        prompts.append({
            "prompt_id": f"motion_calib_ambiguous_low_motion_{index:02d}",
            "prompt_text": prompt_text,
            "prompt_negative_text": "large motion, fast camera movement, heavy flicker, jitter, distorted",
            "prompt_category": "motion_threshold_calibration",
            "motion_pattern_id": "ambiguous_low_motion",
            "prompt_suite_role": "motion_calibration_ambiguous_low_motion",
            "motion_calibration_role": "ambiguous_low_motion",
            "split": "calibration",
        })
    return prompts


def build_prompt_suite() -> dict:
    """构造独立于测试运行的 prompt suite, 便于 Colab 重复使用。"""
    suite = {
        "prompt_suite_id": "generative_video_probe_prompt_suite_v1",
        "dataset_construction_status": "constructed",
        "dataset_source": "repository_deterministic_prompt_seed_spec",
        "motion_calibration_design": {
            "negative_static_prompt_count": 16,
            "positive_motion_prompt_count": 8,
            "ambiguous_low_motion_prompt_count": 4,
            "motion_calibration_seed_count": 8,
            "negative_static_target_video_count": 128,
            "positive_motion_target_video_count": 64,
            "ambiguous_low_motion_target_video_count": 32,
            "split": "calibration"
        },
        "prompts": PROMPT_ITEMS + _build_motion_calibration_prompts(),
        "seeds": SEED_ITEMS + MOTION_CALIBRATION_SEED_ITEMS,
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
