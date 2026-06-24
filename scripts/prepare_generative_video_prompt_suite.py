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
        "motion_claim_role": "positive_motion",
        "prompt_suite_role": "main",
    },
    {
        "prompt_id": "camera_zoom_scene",
        "prompt_text": "A ceramic bird figurine stands near a window while the camera slowly zooms in with stable lighting.",
        "prompt_negative_text": "worst quality, inconsistent motion, blurry, jittery, distorted",
        "prompt_category": "camera_motion",
        "motion_pattern_id": "slow_zoom",
        "motion_claim_role": "positive_motion",
        "prompt_suite_role": "main",
    },
    {
        "prompt_id": "heldout_rotation_scene",
        "prompt_text": "A large blue cube with bright orange arrow markings slides from the far left edge to the far right edge while spinning rapidly for a full rotation on a plain gray floor, fixed camera, the cube fills at least one third of the image, strong visible displacement in every frame.",
        "prompt_negative_text": "static image, frozen frame, subtle motion, tiny object, weak rotation, blurry, jittery, distorted",
        "prompt_category": "heldout_motion",
        "motion_pattern_id": "large_rotation_translation",
        "motion_claim_role": "positive_motion",
        "prompt_suite_role": "heldout_prompt",
    },
    {
        "prompt_id": "liquid_pour_closeup",
        "prompt_text": "Clear water pours from a glass pitcher into a small cup on a kitchen counter with steady camera framing.",
        "prompt_negative_text": "worst quality, inconsistent motion, blurry, jittery, distorted",
        "prompt_category": "fluid_motion",
        "motion_pattern_id": "continuous_pour",
        "motion_claim_role": "positive_motion",
        "prompt_suite_role": "pilot_main",
    },
    {
        "prompt_id": "walking_robot_sideview",
        "prompt_text": "A small toy robot walks slowly from right to left across a clean desk while the camera remains fixed.",
        "prompt_negative_text": "worst quality, inconsistent motion, blurry, jittery, distorted",
        "prompt_category": "articulated_motion",
        "motion_pattern_id": "sideways_walk",
        "motion_claim_role": "positive_motion",
        "prompt_suite_role": "pilot_main",
    },
    {
        "prompt_id": "falling_leaves_static_camera",
        "prompt_text": "Several yellow leaves fall gently in front of a park bench while the camera stays still.",
        "prompt_negative_text": "worst quality, inconsistent motion, blurry, jittery, distorted",
        "prompt_category": "natural_motion",
        "motion_pattern_id": "downward_fall",
        "motion_claim_role": "positive_motion",
        "prompt_suite_role": "pilot_main",
    },
    {
        "prompt_id": "turntable_product_orbit",
        "prompt_text": "A white ceramic mug rotates on a small turntable under soft studio lighting with a fixed camera.",
        "prompt_negative_text": "worst quality, inconsistent motion, blurry, jittery, distorted",
        "prompt_category": "object_rotation",
        "motion_pattern_id": "turntable_rotation",
        "motion_claim_role": "positive_motion",
        "prompt_suite_role": "pilot_main",
    },
    {
        "prompt_id": "train_window_lateral_motion",
        "prompt_text": "A landscape moves laterally outside a train window while the interior frame remains stable.",
        "prompt_negative_text": "worst quality, inconsistent motion, blurry, jittery, distorted",
        "prompt_category": "background_motion",
        "motion_pattern_id": "lateral_background_shift",
        "motion_claim_role": "positive_motion",
        "prompt_suite_role": "pilot_main",
    },
]


FPR01_PILOT_PROMPT_ITEMS = [
    {
        "prompt_id": "fpr01_pilot_drone_orbit_marker",
        "prompt_text": "A small drone carrying a bright green marker orbits a large red cube in a clean studio, fixed camera, the marker visibly changes position across the full frame in every frame.",
        "prompt_negative_text": "static image, frozen frame, subtle motion, tiny object, blurry, jittery, distorted",
        "prompt_category": "fpr01_pilot_object_motion",
        "motion_pattern_id": "orbiting_marker_motion",
        "motion_claim_role": "positive_motion",
        "prompt_suite_role": "fpr01_pilot",
        "split": "pilot",
    },
    {
        "prompt_id": "fpr01_pilot_bouncing_ball_grid",
        "prompt_text": "A large orange ball bounces between clearly marked floor grid lines, fixed camera, high contrast background, strong vertical and horizontal displacement in every frame.",
        "prompt_negative_text": "static image, frozen frame, weak motion, tiny object, blurry, jittery, distorted",
        "prompt_category": "fpr01_pilot_object_motion",
        "motion_pattern_id": "bouncing_grid_motion",
        "motion_claim_role": "positive_motion",
        "prompt_suite_role": "fpr01_pilot",
        "split": "pilot",
    },
    {
        "prompt_id": "fpr01_pilot_conveyor_boxes",
        "prompt_text": "Three brightly colored boxes travel on a conveyor belt from left to right across the entire frame, fixed camera, each box remains large and easy to track.",
        "prompt_negative_text": "static image, frozen conveyor, subtle motion, blurry, jittery, distorted",
        "prompt_category": "fpr01_pilot_structured_motion",
        "motion_pattern_id": "conveyor_translation",
        "motion_claim_role": "positive_motion",
        "prompt_suite_role": "fpr01_pilot",
        "split": "pilot",
    },
    {
        "prompt_id": "fpr01_pilot_rotating_arrow_sign",
        "prompt_text": "A large black arrow sign rotates quickly for multiple full turns in the center of a plain white room, fixed camera, orientation changes clearly in every frame.",
        "prompt_negative_text": "static image, frozen arrow, slow rotation, motion blur, jittery, distorted",
        "prompt_category": "fpr01_pilot_rotation",
        "motion_pattern_id": "high_contrast_rotation",
        "motion_claim_role": "positive_motion",
        "prompt_suite_role": "fpr01_pilot",
        "split": "pilot",
    },
    {
        "prompt_id": "fpr01_pilot_flag_sweep_closeup",
        "prompt_text": "A large blue flag sweeps from the lower left to the upper right across most of the image, fixed camera, wide amplitude motion, high contrast indoor background.",
        "prompt_negative_text": "static image, frozen cloth, tiny flag, weak motion, blurry, jittery, distorted",
        "prompt_category": "fpr01_pilot_articulated_motion",
        "motion_pattern_id": "large_amplitude_sweep",
        "motion_claim_role": "positive_motion",
        "prompt_suite_role": "fpr01_pilot",
        "split": "pilot",
    },
    {
        "prompt_id": "fpr01_pilot_train_toy_track",
        "prompt_text": "A red toy train moves along a circular track around a yellow tower, fixed camera, the train occupies a large foreground area and visibly changes position every frame.",
        "prompt_negative_text": "static image, frozen train, subtle motion, tiny object, blurry, jittery, distorted",
        "prompt_category": "fpr01_pilot_orbit_motion",
        "motion_pattern_id": "circular_track_motion",
        "motion_claim_role": "positive_motion",
        "prompt_suite_role": "fpr01_pilot",
        "split": "pilot",
    },
    {
        "prompt_id": "fpr01_pilot_slider_panel",
        "prompt_text": "A high contrast striped rectangular panel slides diagonally from the bottom left corner to the top right corner across the entire frame, fixed camera, no other moving objects.",
        "prompt_negative_text": "static image, frozen panel, subtle motion, blurry, jittery, distorted",
        "prompt_category": "fpr01_pilot_translation",
        "motion_pattern_id": "diagonal_panel_translation",
        "motion_claim_role": "positive_motion",
        "prompt_suite_role": "fpr01_pilot",
        "split": "pilot",
    },
    {
        "prompt_id": "fpr01_pilot_clock_hand_fast",
        "prompt_text": "A large clock face fills the image while a bright red second hand spins rapidly through a full circle, fixed camera, strong angular motion in every frame.",
        "prompt_negative_text": "static image, frozen clock, slow hand, tiny object, blurry, jittery, distorted",
        "prompt_category": "fpr01_pilot_rotation",
        "motion_pattern_id": "fast_clock_hand_rotation",
        "motion_claim_role": "positive_motion",
        "prompt_suite_role": "fpr01_pilot",
        "split": "pilot",
    },
    {
        "prompt_id": "fpr01_pilot_camera_pan_markers",
        "prompt_text": "The camera pans smoothly from left to right across five large colored markers on a plain wall, markers pass through the frame with clear lateral displacement.",
        "prompt_negative_text": "static image, no pan, weak camera motion, blurry, jittery, distorted",
        "prompt_category": "fpr01_pilot_camera_motion",
        "motion_pattern_id": "controlled_camera_pan",
        "motion_claim_role": "positive_motion",
        "prompt_suite_role": "fpr01_pilot",
        "split": "pilot",
    },
    {
        "prompt_id": "fpr01_pilot_zoom_checker_target",
        "prompt_text": "A large checker target on a wall grows from small to large as the camera zooms in quickly and smoothly, strong scale change in every frame.",
        "prompt_negative_text": "static image, no zoom, weak scale change, blurry, jittery, distorted",
        "prompt_category": "fpr01_pilot_camera_motion",
        "motion_pattern_id": "strong_camera_zoom",
        "motion_claim_role": "positive_motion",
        "prompt_suite_role": "fpr01_pilot",
        "split": "pilot",
    },
    {
        "prompt_id": "fpr01_pilot_paper_airplane_crossing",
        "prompt_text": "A white paper airplane flies close to the camera from the far right edge to the far left edge across a dark background, fixed camera, large visible displacement.",
        "prompt_negative_text": "static image, frozen airplane, tiny object, weak motion, blurry, jittery, distorted",
        "prompt_category": "fpr01_pilot_translation",
        "motion_pattern_id": "foreground_crossing_motion",
        "motion_claim_role": "positive_motion",
        "prompt_suite_role": "fpr01_pilot",
        "split": "pilot",
    },
    {
        "prompt_id": "fpr01_pilot_rotating_product_box",
        "prompt_text": "A large product box with red, blue, and yellow faces rotates on a turntable for multiple full rotations, fixed camera, each colored face appears repeatedly.",
        "prompt_negative_text": "static image, frozen product, slow rotation, blurry, jittery, distorted",
        "prompt_category": "fpr01_pilot_rotation",
        "motion_pattern_id": "multi_face_turntable_rotation",
        "motion_claim_role": "positive_motion",
        "prompt_suite_role": "fpr01_pilot",
        "split": "pilot",
    },
    {
        "prompt_id": "fpr01_pilot_robot_arm_sweep",
        "prompt_text": "A bright yellow robot arm sweeps a red block from the left side of a workbench to the right side, fixed camera, large visible displacement and articulated motion in every frame.",
        "prompt_negative_text": "static image, frozen robot arm, weak motion, tiny object, blurry, jittery, distorted",
        "prompt_category": "fpr01_pilot_articulated_motion",
        "motion_pattern_id": "robot_arm_sweep",
        "motion_claim_role": "positive_motion",
        "prompt_suite_role": "fpr01_pilot",
        "split": "pilot",
    },
    {
        "prompt_id": "fpr01_pilot_spinning_fan_marker",
        "prompt_text": "A large white fan with one red blade marker spins rapidly in front of a plain wall, fixed camera, the red marker changes angle clearly in every frame.",
        "prompt_negative_text": "static image, frozen fan, slow rotation, weak marker, blurry, jittery, distorted",
        "prompt_category": "fpr01_pilot_rotation",
        "motion_pattern_id": "marked_fan_rotation",
        "motion_claim_role": "positive_motion",
        "prompt_suite_role": "fpr01_pilot",
        "split": "pilot",
    },
    {
        "prompt_id": "fpr01_pilot_marble_track_descent",
        "prompt_text": "A shiny blue marble rolls down a zigzag track from the upper left to the lower right, fixed camera, high contrast track, clear position change in every frame.",
        "prompt_negative_text": "static image, frozen marble, subtle motion, tiny object, blurry, jittery, distorted",
        "prompt_category": "fpr01_pilot_translation",
        "motion_pattern_id": "zigzag_descent_motion",
        "motion_claim_role": "positive_motion",
        "prompt_suite_role": "fpr01_pilot",
        "split": "pilot",
    },
    {
        "prompt_id": "fpr01_pilot_sliding_door_pattern",
        "prompt_text": "A patterned sliding door moves quickly from closed to open across a dark doorway, fixed camera, strong horizontal displacement and changing visible background.",
        "prompt_negative_text": "static image, frozen door, weak motion, blurry, jittery, distorted",
        "prompt_category": "fpr01_pilot_structured_motion",
        "motion_pattern_id": "sliding_door_translation",
        "motion_claim_role": "positive_motion",
        "prompt_suite_role": "fpr01_pilot",
        "split": "pilot",
    },
    {
        "prompt_id": "fpr01_pilot_swinging_pendulum_card",
        "prompt_text": "A large red card attached to a pendulum swings from far left to far right across a white background, fixed camera, repeated wide arc motion.",
        "prompt_negative_text": "static image, frozen pendulum, tiny card, weak motion, blurry, jittery, distorted",
        "prompt_category": "fpr01_pilot_periodic_motion",
        "motion_pattern_id": "wide_pendulum_swing",
        "motion_claim_role": "positive_motion",
        "prompt_suite_role": "fpr01_pilot",
        "split": "pilot",
    },
    {
        "prompt_id": "fpr01_pilot_escalator_colored_steps",
        "prompt_text": "Colored steps on a small toy escalator move upward continuously, fixed camera, each colored stripe advances clearly between frames.",
        "prompt_negative_text": "static image, frozen steps, weak motion, blurry, jittery, distorted",
        "prompt_category": "fpr01_pilot_structured_motion",
        "motion_pattern_id": "repeating_step_motion",
        "motion_claim_role": "positive_motion",
        "prompt_suite_role": "fpr01_pilot",
        "split": "pilot",
    },
    {
        "prompt_id": "fpr01_pilot_rotating_text_drum",
        "prompt_text": "A large cylinder covered with bold black letters rotates quickly on a stand, fixed camera, the letters move across the visible surface in every frame.",
        "prompt_negative_text": "static image, frozen cylinder, slow rotation, blurry, jittery, distorted",
        "prompt_category": "fpr01_pilot_rotation",
        "motion_pattern_id": "textured_cylinder_rotation",
        "motion_claim_role": "positive_motion",
        "prompt_suite_role": "fpr01_pilot",
        "split": "pilot",
    },
    {
        "prompt_id": "fpr01_pilot_hand_moves_marker_board",
        "prompt_text": "A hand moves a large green marker board diagonally across a plain table from near bottom left to far top right, fixed camera, high contrast foreground motion.",
        "prompt_negative_text": "static image, frozen hand, tiny marker, weak motion, blurry, jittery, distorted",
        "prompt_category": "fpr01_pilot_foreground_motion",
        "motion_pattern_id": "hand_carried_diagonal_motion",
        "motion_claim_role": "positive_motion",
        "prompt_suite_role": "fpr01_pilot",
        "split": "pilot",
    },
    {
        "prompt_id": "fpr01_pilot_waterwheel_markers",
        "prompt_text": "A small waterwheel with red and yellow markers rotates steadily in front of a plain blue background, fixed camera, each marker follows a clear circular path.",
        "prompt_negative_text": "static image, frozen waterwheel, weak rotation, blurry, jittery, distorted",
        "prompt_category": "fpr01_pilot_rotation",
        "motion_pattern_id": "marked_wheel_rotation",
        "motion_claim_role": "positive_motion",
        "prompt_suite_role": "fpr01_pilot",
        "split": "pilot",
    },

]

SEED_ITEMS = [
    {"seed_id": "seed_main_a", "seed_value": 101, "prompt_suite_role": "main"},
    {"seed_id": "seed_main_b", "seed_value": 202, "prompt_suite_role": "main"},
    {"seed_id": "seed_heldout_c", "seed_value": 303, "prompt_suite_role": "heldout_seed"},
]


FPR01_PILOT_SEED_ITEMS = [
    {"seed_id": "seed_fpr01_calib_a", "seed_value": 1401, "prompt_suite_role": "fpr01_pilot", "split": "calibration"},
    {"seed_id": "seed_fpr01_calib_b", "seed_value": 1502, "prompt_suite_role": "fpr01_pilot", "split": "calibration"},
    {"seed_id": "seed_fpr01_calib_c", "seed_value": 1603, "prompt_suite_role": "fpr01_pilot", "split": "calibration"},
    {"seed_id": "seed_fpr01_calib_d", "seed_value": 1704, "prompt_suite_role": "fpr01_pilot", "split": "calibration"},
    {"seed_id": "seed_fpr01_test_a", "seed_value": 1805, "prompt_suite_role": "fpr01_pilot", "split": "test"},
    {"seed_id": "seed_fpr01_test_b", "seed_value": 1906, "prompt_suite_role": "fpr01_pilot", "split": "test"},
    {"seed_id": "seed_fpr01_test_c", "seed_value": 2007, "prompt_suite_role": "fpr01_pilot", "split": "test"},
    {"seed_id": "seed_fpr01_test_d", "seed_value": 2108, "prompt_suite_role": "fpr01_pilot", "split": "test"},
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
        "A locked-off still photograph of a black square printed on white paper on a desk, frozen frame, no camera movement, no object motion, no lighting change.",
        "A single static product photo of a red cube centered on a plain gray background, fixed camera, no pan, no zoom, no moving shadows.",
        "A still photograph of a blue circle printed on a white card, frozen frame, no breeze, no camera movement, stable lighting.",
        "A locked-off still product photo of a matte yellow triangle block resting on a plain gray table, frozen frame, no object motion, no exposure change, no camera shake.",
        "A single matte white index card lies flat on a plain wooden table in a still photograph, fixed camera, no zoom, no flicker, no background movement.",
        "A small toy block remains still on a plain shelf in a frozen frame, no camera movement, no object movement, constant illumination.",
        "A static photo of a white cup centered on a plain blue background, no camera pan, no zoom, no lighting variation.",
        "A simple black arrow printed on paper lies flat on a table in a locked-off still image, no motion, no changing shadows.",
        "A folded white cloth lies on a plain table in a frozen frame, fixed camera, no object motion, no camera motion, no exposure change.",
        "A glass bottle stands on a matte gray surface as a still photograph, no reflection movement, no camera movement, stable lighting.",
        "A notebook and pencil rest on a desk in a static close-up, no hand movement, no camera motion, no lighting change.",
        "A decorative vase remains still against a plain wall in a frozen frame, fixed camera, no zoom, no pan, no shadow movement.",
        "A solid red ball sits motionless on a plain white table in a single still photograph, no camera motion, no object movement, no flicker.",
        "A closed laptop rests on a table in a locked-off static shot, no screen animation, no movement, stable exposure.",
        "A stack of three plain cardboard squares remains motionless under a fixed overhead camera, frozen frame, no moving objects, no lighting change.",
        "A sealed cardboard box with a printed label is shown as a still object on a plain table, fixed camera, no zoom, no pan, no illumination change.",
    ]
    positive_templates = [
        "A person carries a large bright red rectangular board across the entire frame from far left to far right, the board fills nearly half of the image, fixed camera, plain white wall background, strong visible displacement in every frame.",
        "A high-contrast black and white striped panel scrolls continuously from left to right across the entire frame, strong visible motion in every frame, fixed camera.",
        "A close-up bright blue beach ball is thrown upward and downward repeatedly across most of the frame, the ball stays large in the foreground, fixed camera, plain background, strong position change between consecutive frames.",
        "A yellow arrow sweeps diagonally from the lower left corner to the upper right corner across the whole frame, large visible displacement, fixed camera.",
        "A large red exercise ball rolls quickly from the right edge to the left edge across a high-contrast plain floor, occupying a large area of the foreground, fixed camera, clear position change in every frame.",
        "A high-contrast checkerboard board rotates quickly for multiple full rotations, clearly changing orientation in every frame, fixed camera.",
        "A hand sweeps a large white paper airplane close to the camera from left to right across the entire frame, large foreground object size, repeated visible displacement, fixed camera.",
        "A hand waves a large red flag widely from side to side across most of the frame, repeated large amplitude motion, fixed camera.",
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
    fpr01_prompt_count = len(FPR01_PILOT_PROMPT_ITEMS)
    fpr01_seed_count = len(FPR01_PILOT_SEED_ITEMS)
    suite = {
        "prompt_suite_id": "generative_video_probe_prompt_suite_motion_observability_fpr01_pilot",
        "dataset_construction_status": "constructed",
        "dataset_source": "repository_deterministic_prompt_seed_spec",
        "fpr01_pilot_design": {
            "paper_result_level": "pilot_paper",
            "paper_protocol_level": "paper_grade_protocol",
            "paper_protocol_difference_from_full_paper": "sample_scale_only",
            "recommended_runtime_profile": "pilot_paper",
            "compatibility_runtime_profile": "fpr01_pilot",
            "target_fpr": 0.01,
            "blocked_target_fpr": 0.001,
            "prompt_count": fpr01_prompt_count,
            "seed_count": fpr01_seed_count,
            "calibration_seed_count": sum(1 for item in FPR01_PILOT_SEED_ITEMS if item.get("split") == "calibration"),
            "test_seed_count": sum(1 for item in FPR01_PILOT_SEED_ITEMS if item.get("split") == "test"),
            "target_generation_video_count": fpr01_prompt_count * fpr01_seed_count,
            "target_runtime_attack_count": 3,
            "target_test_attacked_positive_event_count": fpr01_prompt_count * 4 * 3,
            "target_negative_family_count": 4,
            "target_calibration_negative_event_count": fpr01_prompt_count * 4 * 3 * 4,
            "target_heldout_test_negative_event_count": fpr01_prompt_count * 4 * 3 * 4,
            "threshold_protocol": "calibration_split_to_frozen_threshold_to_heldout_test_split",
            "claim_support_status": "pilot_paper_dataset_constructed_not_generated",
            "split": "calibration_and_test"
        },
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
        "prompts": PROMPT_ITEMS + FPR01_PILOT_PROMPT_ITEMS + _build_motion_calibration_prompts(),
        "seeds": SEED_ITEMS + FPR01_PILOT_SEED_ITEMS + MOTION_CALIBRATION_SEED_ITEMS,
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
