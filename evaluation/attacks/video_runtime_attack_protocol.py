"""视频水印 runtime attack 协议与正式文件级实现。

该模块的职责是把论文协议中的 attack 名称、分层覆盖要求和实际帧级变换
集中管理。这样 Notebook、SSTW 主流程和 external baseline official reference
可以共享同一套 attack 语义, 避免各文件各自硬写 attack 列表。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from evaluation.protocol.paper_profile_contract import enforce_paper_profile_common_contract


@dataclass(frozen=True)
class RuntimeAttackSpec:
    """描述一个 runtime attack 的协议语义。

    `implementation_level` 用于标记该 attack 是否是可以进入论文主结果的正式
    视频文件级变换。paper profile 只能接受 `formal_runtime_video_transform`,
    防止轻量替代或 proxy 攻击被误写成顶会论文级攻击证据。
    """

    attack_name: str
    attack_family: str
    attack_transform: str
    attack_strength: str
    runtime_attack_expected_effect: str
    implementation_level: str = "formal_runtime_video_transform"
    video_writer_codec: str | None = None
    video_writer_output_params: tuple[str, ...] = ()


FULL_PAPER_RUNTIME_ATTACKS = (
    "video_compression_runtime",
    "h264_crf18_runtime",
    "h264_crf23_runtime",
    "h264_crf28_runtime",
    "h264_crf33_runtime",
    "h264_crf38_runtime",
    "h265_crf23_runtime",
    "h265_crf28_runtime",
    "h265_crf33_runtime",
    "mpeg4_crf28_runtime",
    "mpeg4_q2_runtime",
    "mpeg4_q8_runtime",
    "platform_transcode_runtime",
    "jpeg_frame_compression_runtime",
    "temporal_crop_runtime",
    "temporal_clip_middle_runtime",
    "frame_rate_resampling_runtime",
    "frame_drop_uniform_runtime",
    "irregular_frame_drop_runtime",
    "frame_insert_duplicate_runtime",
    "frame_insert_noise_runtime",
    "frame_duplicate_runtime",
    "speed_change_runtime",
    "frame_swap_adjacent_runtime",
    "frame_average_runtime",
    "spatial_resize_runtime",
    "spatial_crop_resize_runtime",
    "spatial_corner_crop_resize_runtime",
    "rotation_runtime",
    "perspective_runtime",
    "spatial_mask_runtime",
    "gaussian_noise_runtime",
    "salt_pepper_noise_runtime",
    "gaussian_blur_runtime",
    "median_blur_runtime",
    "denoise_runtime",
    "brightness_contrast_runtime",
    "gamma_correction_runtime",
    "color_jitter_runtime",
    "sharpen_runtime",
    "compression_crop_combined_runtime",
    "compression_brightness_combined_runtime",
    "compression_temporal_combined_runtime",
    "compression_noise_combined_runtime",
    "compression_color_jitter_combined_runtime",
    "crop_rotation_combined_runtime",
)

PAPER_PROFILE_RUNTIME_ATTACKS = FULL_PAPER_RUNTIME_ATTACKS

PILOT_PAPER_RUNTIME_ATTACKS = FULL_PAPER_RUNTIME_ATTACKS

RUNTIME_ATTACK_FAMILY_MINIMUMS_BY_PROFILE: dict[str, dict[str, int]] = {
    "probe_paper": {
        "compression": 9,
        "temporal": 8,
        "spatial_geometry": 5,
        "visual_degradation": 8,
        "combined": 5,
    },
    "pilot_paper": {
        "compression": 9,
        "temporal": 8,
        "spatial_geometry": 5,
        "visual_degradation": 8,
        "combined": 5,
    },
    "full_paper": {
        "compression": 9,
        "temporal": 8,
        "spatial_geometry": 5,
        "visual_degradation": 8,
        "combined": 5,
    },
}

FULL_PAPER_NON_RUNTIME_ATTACK_PROTOCOLS = (
    "generative_recompression_or_regeneration_attack",
    "endpoint_preserving_path_perturbation_attack",
    "flow_time_grid_mismatch_attack",
    "wrong_sampler_replay_attack",
    "wrong_prompt_replay_attack",
    "wrong_key_attack",
    "detector_probing_with_public_negatives",
    "watermark_removal_optimization_attack",
    "watermark_spoofing_or_copy_attack",
    "collusion_multi_sample_attack",
    "adversarial_detector_evasion_attack",
)

DEFAULT_TARGET_FPR_LEVELS = (0.1, 0.01, 0.001)
DEFAULT_SHARED_ATTACK_PROTOCOL_CONFIG_PATH = "configs/protocol/shared_generative_video_attack_protocol.json"
SHARED_ATTACK_PROTOCOL_CONFIG_PATH_FIELD = "shared_attack_protocol_config_path"
SHARED_ATTACK_PROTOCOL_FIELDS = (
    "required_runtime_attack_names",
    "runtime_attack_family_minimums",
    "required_non_runtime_attack_protocols",
    "minimum_attack_count",
    "minimum_non_runtime_attack_protocol_count",
    "target_fpr_levels",
)


RUNTIME_ATTACK_SPECS: dict[str, RuntimeAttackSpec] = {
    "video_compression_runtime": RuntimeAttackSpec(
        "video_compression_runtime",
        "compression",
        "decode_reencode",
        "runtime_reencode_default_quality",
        "codec_quantization_or_container_rewrite",
        video_writer_codec="libx264",
        video_writer_output_params=("-crf", "28"),
    ),
    "h264_crf18_runtime": RuntimeAttackSpec(
        "h264_crf18_runtime",
        "compression",
        "h264_reencode",
        "crf_18",
        "h264_light_quantization",
        video_writer_codec="libx264",
        video_writer_output_params=("-crf", "18"),
    ),
    "h264_crf23_runtime": RuntimeAttackSpec(
        "h264_crf23_runtime",
        "compression",
        "h264_reencode",
        "crf_23",
        "h264_codec_quantization",
        video_writer_codec="libx264",
        video_writer_output_params=("-crf", "23"),
    ),
    "h264_crf28_runtime": RuntimeAttackSpec(
        "h264_crf28_runtime",
        "compression",
        "h264_reencode",
        "crf_28",
        "h264_medium_quantization",
        video_writer_codec="libx264",
        video_writer_output_params=("-crf", "28"),
    ),
    "h264_crf33_runtime": RuntimeAttackSpec(
        "h264_crf33_runtime",
        "compression",
        "h264_reencode",
        "crf_33",
        "h264_codec_quantization",
        video_writer_codec="libx264",
        video_writer_output_params=("-crf", "33"),
    ),
    "h264_crf38_runtime": RuntimeAttackSpec(
        "h264_crf38_runtime",
        "compression",
        "h264_reencode",
        "crf_38",
        "h264_heavy_quantization",
        video_writer_codec="libx264",
        video_writer_output_params=("-crf", "38"),
    ),
    "h265_crf23_runtime": RuntimeAttackSpec(
        "h265_crf23_runtime",
        "compression",
        "h265_reencode",
        "crf_23",
        "h265_light_quantization",
        video_writer_codec="libx265",
        video_writer_output_params=("-crf", "23"),
    ),
    "h265_crf28_runtime": RuntimeAttackSpec(
        "h265_crf28_runtime",
        "compression",
        "h265_reencode",
        "crf_28",
        "h265_codec_quantization",
        video_writer_codec="libx265",
        video_writer_output_params=("-crf", "28"),
    ),
    "h265_crf33_runtime": RuntimeAttackSpec(
        "h265_crf33_runtime",
        "compression",
        "h265_reencode",
        "crf_33",
        "h265_heavy_quantization",
        video_writer_codec="libx265",
        video_writer_output_params=("-crf", "33"),
    ),
    "mpeg4_crf28_runtime": RuntimeAttackSpec(
        "mpeg4_crf28_runtime",
        "compression",
        "mpeg4_reencode",
        "crf_28",
        "mpeg4_codec_quantization",
        video_writer_codec="mpeg4",
        video_writer_output_params=("-q:v", "5"),
    ),
    "mpeg4_q2_runtime": RuntimeAttackSpec(
        "mpeg4_q2_runtime",
        "compression",
        "mpeg4_reencode",
        "qscale_2",
        "mpeg4_light_quantization",
        video_writer_codec="mpeg4",
        video_writer_output_params=("-q:v", "2"),
    ),
    "mpeg4_q8_runtime": RuntimeAttackSpec(
        "mpeg4_q8_runtime",
        "compression",
        "mpeg4_reencode",
        "qscale_8",
        "mpeg4_heavy_quantization",
        video_writer_codec="mpeg4",
        video_writer_output_params=("-q:v", "8"),
    ),
    "platform_transcode_runtime": RuntimeAttackSpec(
        "platform_transcode_runtime",
        "compression",
        "platform_like_h264_reencode",
        "h264_crf_32_yuv420p",
        "social_media_or_platform_transcoding",
        video_writer_codec="libx264",
        video_writer_output_params=("-crf", "32", "-pix_fmt", "yuv420p"),
    ),
    "jpeg_frame_compression_runtime": RuntimeAttackSpec(
        "jpeg_frame_compression_runtime",
        "compression",
        "per_frame_jpeg_roundtrip",
        "jpeg_quality_55",
        "intra_frame_quantization",
    ),
    "temporal_crop_runtime": RuntimeAttackSpec(
        "temporal_crop_runtime",
        "temporal",
        "drop_first_and_last_frame_when_possible",
        "crop_boundary_frames",
        "temporal_boundary_shift",
    ),
    "temporal_clip_middle_runtime": RuntimeAttackSpec(
        "temporal_clip_middle_runtime",
        "temporal",
        "remove_middle_clip_window_when_possible",
        "middle_clip_ratio_0_20",
        "temporal_clip_and_context_loss",
    ),
    "frame_rate_resampling_runtime": RuntimeAttackSpec(
        "frame_rate_resampling_runtime",
        "temporal",
        "keep_every_second_frame_when_possible",
        "fps_downsample_by_2",
        "time_grid_resampling",
    ),
    "frame_drop_uniform_runtime": RuntimeAttackSpec(
        "frame_drop_uniform_runtime",
        "temporal",
        "drop_every_third_frame_when_possible",
        "uniform_frame_drop_1_over_3",
        "temporal_frame_loss",
    ),
    "irregular_frame_drop_runtime": RuntimeAttackSpec(
        "irregular_frame_drop_runtime",
        "temporal",
        "drop_deterministic_irregular_frame_positions",
        "drop_positions_mod_5_equals_1_or_4",
        "irregular_temporal_frame_loss",
    ),
    "frame_insert_duplicate_runtime": RuntimeAttackSpec(
        "frame_insert_duplicate_runtime",
        "temporal",
        "duplicate_middle_frame_when_possible",
        "single_frame_insert_duplicate",
        "temporal_frame_insertion",
    ),
    "frame_insert_noise_runtime": RuntimeAttackSpec(
        "frame_insert_noise_runtime",
        "temporal",
        "insert_deterministic_noisy_frame_when_possible",
        "single_noisy_frame_insert",
        "temporal_disturbance_with_out_of_distribution_frame",
    ),
    "frame_duplicate_runtime": RuntimeAttackSpec(
        "frame_duplicate_runtime",
        "temporal",
        "duplicate_every_fourth_frame_when_possible",
        "periodic_frame_duplication",
        "temporal_hold_and_duration_shift",
    ),
    "speed_change_runtime": RuntimeAttackSpec(
        "speed_change_runtime",
        "temporal",
        "drop_and_duplicate_deterministic_frames",
        "playback_speed_1_25x_then_hold",
        "temporal_speed_and_duration_shift",
    ),
    "frame_swap_adjacent_runtime": RuntimeAttackSpec(
        "frame_swap_adjacent_runtime",
        "temporal",
        "swap_middle_adjacent_frames_when_possible",
        "single_adjacent_swap",
        "temporal_order_disturbance",
    ),
    "frame_average_runtime": RuntimeAttackSpec(
        "frame_average_runtime",
        "temporal",
        "average_adjacent_frames",
        "adjacent_frame_average",
        "temporal_smoothing",
    ),
    "spatial_resize_runtime": RuntimeAttackSpec(
        "spatial_resize_runtime",
        "spatial_geometry",
        "downsample_then_restore_resolution",
        "resize_ratio_0_75",
        "spatial_resampling",
    ),
    "spatial_crop_resize_runtime": RuntimeAttackSpec(
        "spatial_crop_resize_runtime",
        "spatial_geometry",
        "center_crop_then_restore_resolution",
        "center_crop_ratio_0_80",
        "spatial_crop_and_rescale",
    ),
    "spatial_corner_crop_resize_runtime": RuntimeAttackSpec(
        "spatial_corner_crop_resize_runtime",
        "spatial_geometry",
        "corner_crop_then_restore_resolution",
        "corner_crop_ratio_0_85",
        "off_center_spatial_crop_and_rescale",
    ),
    "rotation_runtime": RuntimeAttackSpec(
        "rotation_runtime",
        "spatial_geometry",
        "small_angle_rotate_then_restore_canvas",
        "rotation_5_degrees",
        "spatial_geometric_desynchronization",
    ),
    "perspective_runtime": RuntimeAttackSpec(
        "perspective_runtime",
        "spatial_geometry",
        "perspective_warp_then_restore_canvas",
        "corner_shift_ratio_0_04",
        "projective_spatial_desynchronization",
    ),
    "spatial_mask_runtime": RuntimeAttackSpec(
        "spatial_mask_runtime",
        "spatial_geometry",
        "center_rectangle_mask",
        "mask_ratio_0_20",
        "spatial_occlusion_or_crop_drop",
    ),
    "gaussian_noise_runtime": RuntimeAttackSpec(
        "gaussian_noise_runtime",
        "visual_degradation",
        "deterministic_gaussian_like_noise",
        "sigma_4_uint8",
        "pixel_value_noise",
    ),
    "salt_pepper_noise_runtime": RuntimeAttackSpec(
        "salt_pepper_noise_runtime",
        "visual_degradation",
        "deterministic_salt_pepper_noise",
        "probability_0_01",
        "impulse_noise",
    ),
    "gaussian_blur_runtime": RuntimeAttackSpec(
        "gaussian_blur_runtime",
        "visual_degradation",
        "gaussian_blur",
        "radius_1",
        "local_low_pass_filtering",
    ),
    "median_blur_runtime": RuntimeAttackSpec(
        "median_blur_runtime",
        "visual_degradation",
        "median_filter",
        "kernel_3",
        "impulse_noise_suppression_like_filtering",
    ),
    "denoise_runtime": RuntimeAttackSpec(
        "denoise_runtime",
        "visual_degradation",
        "deterministic_median_gaussian_denoise_filter",
        "median3_gaussian_radius_0_6",
        "denoising_or_platform_preprocessing",
    ),
    "brightness_contrast_runtime": RuntimeAttackSpec(
        "brightness_contrast_runtime",
        "visual_degradation",
        "brightness_contrast_adjustment",
        "brightness_1_08_contrast_1_10",
        "global_color_value_shift",
    ),
    "gamma_correction_runtime": RuntimeAttackSpec(
        "gamma_correction_runtime",
        "visual_degradation",
        "gamma_correction",
        "gamma_0_85",
        "nonlinear_luma_value_shift",
    ),
    "color_jitter_runtime": RuntimeAttackSpec(
        "color_jitter_runtime",
        "visual_degradation",
        "deterministic_channel_gain_and_offset",
        "rgb_gain_1_06_0_96_1_02_offset_2",
        "color_balance_shift",
    ),
    "sharpen_runtime": RuntimeAttackSpec(
        "sharpen_runtime",
        "visual_degradation",
        "sharpen_filter",
        "single_pil_sharpen_pass",
        "edge_enhancement_or_platform_postprocessing",
    ),
    "compression_crop_combined_runtime": RuntimeAttackSpec(
        "compression_crop_combined_runtime",
        "combined",
        "decode_reencode_plus_center_crop_resize",
        "h264_crf_28_and_crop_ratio_0_80",
        "combined_codec_and_spatial_desynchronization",
        video_writer_codec="libx264",
        video_writer_output_params=("-crf", "28"),
    ),
    "compression_brightness_combined_runtime": RuntimeAttackSpec(
        "compression_brightness_combined_runtime",
        "combined",
        "decode_reencode_plus_brightness_contrast",
        "h264_crf_28_and_brightness_contrast",
        "combined_codec_and_value_shift",
        video_writer_codec="libx264",
        video_writer_output_params=("-crf", "28"),
    ),
    "compression_temporal_combined_runtime": RuntimeAttackSpec(
        "compression_temporal_combined_runtime",
        "combined",
        "decode_reencode_plus_frame_drop",
        "h264_crf_28_and_uniform_frame_drop",
        "combined_codec_and_temporal_frame_loss",
        video_writer_codec="libx264",
        video_writer_output_params=("-crf", "28"),
    ),
    "compression_noise_combined_runtime": RuntimeAttackSpec(
        "compression_noise_combined_runtime",
        "combined",
        "decode_reencode_plus_gaussian_noise",
        "h264_crf_28_and_sigma_4_noise",
        "combined_codec_and_pixel_noise",
        video_writer_codec="libx264",
        video_writer_output_params=("-crf", "28"),
    ),
    "compression_color_jitter_combined_runtime": RuntimeAttackSpec(
        "compression_color_jitter_combined_runtime",
        "combined",
        "decode_reencode_plus_color_jitter",
        "h264_crf_28_and_color_jitter",
        "combined_codec_and_color_value_shift",
        video_writer_codec="libx264",
        video_writer_output_params=("-crf", "28"),
    ),
    "crop_rotation_combined_runtime": RuntimeAttackSpec(
        "crop_rotation_combined_runtime",
        "combined",
        "center_crop_resize_plus_rotation",
        "crop_ratio_0_80_and_rotation_5_degrees",
        "combined_spatial_crop_and_rotation_desynchronization",
    ),
}


def _read_json_object(path: Path) -> dict[str, Any]:
    """读取 JSON 对象配置, 并兼容 Colab / Windows 常见的 UTF-8 BOM。"""

    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON 顶层必须是对象: {path}")
    return payload


def _resolve_shared_attack_protocol_path(
    config: Mapping[str, Any],
    config_path: str | Path | None = None,
) -> Path | None:
    """解析 profile config 中登记的共享 attack 协议配置路径。

    该函数属于项目特定写法。SSTW 的三个论文 profile 必须共享同一份 attack
    manifest, 因此这里允许 profile config 只声明共享配置路径, 由运行时统一
    还原出 `required_runtime_attack_names` 等字段。
    """

    raw_path = config.get(SHARED_ATTACK_PROTOCOL_CONFIG_PATH_FIELD)
    if raw_path in {None, ""}:
        return None
    candidate = Path(str(raw_path))
    if candidate.is_absolute():
        return candidate
    cwd_candidate = Path.cwd() / candidate
    if cwd_candidate.exists():
        return cwd_candidate
    if config_path is not None:
        profile_path = Path(config_path)
        sibling_candidate = profile_path.parent / candidate
        if sibling_candidate.exists():
            return sibling_candidate
        repo_relative_candidate = profile_path.parent.parent.parent / candidate
        if repo_relative_candidate.exists():
            return repo_relative_candidate
    return candidate


def merge_shared_attack_protocol_config(
    config: Mapping[str, Any],
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """把共享 attack 协议配置合并到单个 profile config 中。

    通用工程写法通常会在每个配置文件中直接写完整字段。这里采用项目特定写法:
    三个论文层级只保存样本规模、FPR 和门禁差异, runtime attack 列表、
    non-runtime/adaptive 协议列表和 family minimums 均来自共享配置。
    """

    merged = dict(config)
    shared_path = _resolve_shared_attack_protocol_path(merged, config_path)
    if shared_path is None:
        return merged
    if not shared_path.exists():
        raise FileNotFoundError(f"缺少共享 attack 协议配置: {shared_path}")
    shared = _read_json_object(shared_path)
    for field_name in SHARED_ATTACK_PROTOCOL_FIELDS:
        current_value = merged.get(field_name)
        if field_name not in merged or current_value is None or current_value == "" or current_value == []:
            if field_name in shared:
                merged[field_name] = shared[field_name]
    if "shared_attack_protocol_id" not in merged and "shared_attack_protocol_id" in shared:
        merged["shared_attack_protocol_id"] = shared["shared_attack_protocol_id"]
    merged[SHARED_ATTACK_PROTOCOL_CONFIG_PATH_FIELD] = str(config.get(SHARED_ATTACK_PROTOCOL_CONFIG_PATH_FIELD))
    merged["shared_attack_protocol_resolved_path"] = str(shared_path)
    merged["shared_attack_protocol_resolution_status"] = "shared_attack_protocol_config_merged"
    return merged


def load_protocol_config_with_shared_attack_protocol(config_path: str | Path) -> dict[str, Any]:
    """读取 protocol config, 并自动合并共享 attack 协议字段。"""

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"缺少 protocol config: {path}")
    profile = enforce_paper_profile_common_contract(_read_json_object(path), path)
    return merge_shared_attack_protocol_config(profile, path)


def runtime_attack_names_for_profile(profile_name: str) -> tuple[str, ...]:
    """按 workflow profile 返回默认 runtime attack 列表。"""

    normalized = str(profile_name or "").strip().lower()
    if normalized in {"probe_paper", "full_paper"}:
        return FULL_PAPER_RUNTIME_ATTACKS
    if normalized == "pilot_paper":
        return PILOT_PAPER_RUNTIME_ATTACKS
    return PAPER_PROFILE_RUNTIME_ATTACKS


def required_runtime_attack_names_from_config(config: Mapping[str, Any]) -> tuple[str, ...]:
    """从 protocol config 读取 required runtime attack 名称集合。"""

    config = merge_shared_attack_protocol_config(config)
    explicit = config.get("required_runtime_attack_names")
    if isinstance(explicit, list) and explicit:
        return tuple(str(item) for item in explicit if str(item))
    return runtime_attack_names_for_profile(str(config.get("paper_result_level") or "probe_paper"))


def required_non_runtime_attack_protocols_from_config(config: Mapping[str, Any]) -> tuple[str, ...]:
    """从 protocol config 读取 non-runtime / adaptive attack 协议集合。"""

    config = merge_shared_attack_protocol_config(config)
    explicit = config.get("required_non_runtime_attack_protocols")
    if isinstance(explicit, list) and explicit:
        return tuple(str(item) for item in explicit if str(item))
    profile = str(config.get("paper_result_level") or "probe_paper").strip()
    if profile in {"probe_paper", "pilot_paper", "full_paper"}:
        return FULL_PAPER_NON_RUNTIME_ATTACK_PROTOCOLS
    return ()


def runtime_attack_family_minimums_from_config(config: Mapping[str, Any]) -> dict[str, int]:
    """从 protocol config 读取 runtime attack family 最低覆盖要求。"""

    config = merge_shared_attack_protocol_config(config)
    profile = str(config.get("paper_result_level") or "probe_paper").strip() or "probe_paper"
    raw_family_minimums = config.get("runtime_attack_family_minimums")
    if isinstance(raw_family_minimums, Mapping):
        return {str(key): int(value) for key, value in raw_family_minimums.items()}
    return dict(
        RUNTIME_ATTACK_FAMILY_MINIMUMS_BY_PROFILE.get(
            profile,
            RUNTIME_ATTACK_FAMILY_MINIMUMS_BY_PROFILE["probe_paper"],
        )
    )


def target_fpr_levels_from_config(config: Mapping[str, Any]) -> tuple[float, ...]:
    """读取当前协议族登记的 FPR 等级集合。"""

    config = merge_shared_attack_protocol_config(config)
    explicit = config.get("target_fpr_levels")
    if isinstance(explicit, list) and explicit:
        return tuple(float(item) for item in explicit)
    return DEFAULT_TARGET_FPR_LEVELS


def audit_runtime_attack_protocol_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """审计 protocol config 中的 runtime attack 覆盖是否符合当前阶段要求。

    该函数属于项目特定治理写法。普通工程只需要检查 attack 名称能否执行, 但
    SSTW 的论文流程还需要防止 `full_paper` 或 `pilot_paper` 在配置层退化为
    少量 easy attacks。因此这里按 profile 检查注册完整性、family 覆盖和
    full-paper 非 runtime 自适应攻击协议登记。
    """

    config = merge_shared_attack_protocol_config(config)
    profile = str(config.get("paper_result_level") or "probe_paper").strip() or "probe_paper"
    attack_names = required_runtime_attack_names_from_config(config)
    missing_registered_names = [name for name in attack_names if name not in RUNTIME_ATTACK_SPECS]
    proxy_named_attacks = [name for name in attack_names if "proxy" in str(name).lower()]
    non_formal_implementation_names = [
        name
        for name in attack_names
        if name in RUNTIME_ATTACK_SPECS
        and RUNTIME_ATTACK_SPECS[name].implementation_level != "formal_runtime_video_transform"
    ]
    family_counts: dict[str, int] = {}
    for name in attack_names:
        if name not in RUNTIME_ATTACK_SPECS:
            continue
        family = RUNTIME_ATTACK_SPECS[name].attack_family
        family_counts[family] = family_counts.get(family, 0) + 1

    family_minimums = runtime_attack_family_minimums_from_config(config)
    missing_family_minimums = [
        {
            "attack_family": family,
            "observed_count": int(family_counts.get(family, 0)),
            "required_minimum_count": int(required_count),
        }
        for family, required_count in sorted(family_minimums.items())
        if int(family_counts.get(family, 0)) < int(required_count)
    ]

    non_runtime_required = required_non_runtime_attack_protocols_from_config(config)
    missing_non_runtime = []
    if profile in {"probe_paper", "pilot_paper", "full_paper"}:
        missing_non_runtime = sorted(set(FULL_PAPER_NON_RUNTIME_ATTACK_PROTOCOLS) - set(non_runtime_required))

    decision = "PASS" if not missing_registered_names and not missing_family_minimums and not missing_non_runtime and not proxy_named_attacks and not non_formal_implementation_names else "FAIL"
    return {
        "runtime_attack_protocol_decision": decision,
        "paper_result_level": profile,
        "required_runtime_attack_count": len(attack_names),
        "required_runtime_attack_names": list(attack_names),
        "runtime_attack_family_counts": dict(sorted(family_counts.items())),
        "runtime_attack_family_minimums": family_minimums,
        "runtime_attack_missing_registered_names": missing_registered_names,
        "runtime_attack_proxy_named_attacks": proxy_named_attacks,
        "runtime_attack_non_formal_implementation_names": non_formal_implementation_names,
        "runtime_attack_missing_family_minimums": missing_family_minimums,
        "required_non_runtime_attack_protocols": list(non_runtime_required),
        "missing_non_runtime_attack_protocols": missing_non_runtime,
        "top_tier_attack_protocol_status": "top_tier_runtime_and_adaptive_protocol_registered"
        if decision == "PASS" and profile in {"probe_paper", "pilot_paper", "full_paper"}
        else "runtime_attack_protocol_needs_completion",
    }


def runtime_attack_spec(attack_name: str) -> RuntimeAttackSpec:
    """读取单个 attack 的规范定义, 未注册名称直接失败。"""

    normalized = str(attack_name or "").strip()
    if normalized not in RUNTIME_ATTACK_SPECS:
        raise ValueError(f"unsupported_runtime_attack:{attack_name}")
    return RUNTIME_ATTACK_SPECS[normalized]


def _to_numpy(frame: Any) -> Any:
    """把一帧转换为 numpy array, 避免在模块导入时强依赖 numpy。"""

    import numpy as np

    return np.asarray(frame)


def _clip_like_uint8(frame: Any) -> Any:
    """将帧裁剪到 uint8 图像范围。"""

    import numpy as np

    return np.clip(frame, 0, 255).astype(np.uint8)


def _resize_frame(frame: Any, size: tuple[int, int]) -> Any:
    """使用 PIL 对单帧做 resize。"""

    import numpy as np
    from PIL import Image

    array = _to_numpy(frame)
    image = Image.fromarray(array.astype(np.uint8))
    resized = image.resize(size, Image.BILINEAR)
    return np.asarray(resized).astype(array.dtype)


def _blur_frame(frame: Any, attack_name: str) -> Any:
    """对单帧做 blur 或 median filter。"""

    import numpy as np
    from PIL import Image, ImageFilter

    array = _to_numpy(frame)
    image = Image.fromarray(array.astype(np.uint8))
    if attack_name == "median_blur_runtime":
        filtered = image.filter(ImageFilter.MedianFilter(size=3))
    else:
        filtered = image.filter(ImageFilter.GaussianBlur(radius=1))
    return np.asarray(filtered).astype(array.dtype)


def _denoise_frame(frame: Any) -> Any:
    """执行确定性视频去噪变换, 覆盖平台预处理或 watermark removal 前置滤波。"""

    import numpy as np

    array = _to_numpy(frame)
    from PIL import Image, ImageFilter

    image = Image.fromarray(array.astype(np.uint8))
    filtered = image.filter(ImageFilter.MedianFilter(size=3)).filter(ImageFilter.GaussianBlur(radius=0.6))
    return np.asarray(filtered).astype(array.dtype)


def _sharpen_frame(frame: Any) -> Any:
    """执行确定性 sharpen, 覆盖平台锐化或再编码后处理。"""

    import numpy as np
    from PIL import Image, ImageFilter

    array = _to_numpy(frame)
    image = Image.fromarray(array.astype(np.uint8))
    filtered = image.filter(ImageFilter.SHARPEN)
    return np.asarray(filtered).astype(array.dtype)


def _spatial_resize(frames: list[Any], ratio: float = 0.75) -> list[Any]:
    """先缩小再恢复到原始分辨率, 模拟 resize 类攻击。"""

    if not frames:
        return []
    first = _to_numpy(frames[0])
    height, width = int(first.shape[0]), int(first.shape[1])
    small = (max(1, int(width * ratio)), max(1, int(height * ratio)))
    restored = (width, height)
    return [_resize_frame(_resize_frame(frame, small), restored) for frame in frames]


def _spatial_crop_resize(frames: list[Any], crop_ratio: float = 0.80) -> list[Any]:
    """中心裁剪后恢复到原始分辨率。"""

    import numpy as np

    attacked: list[Any] = []
    for frame in frames:
        array = _to_numpy(frame)
        height, width = int(array.shape[0]), int(array.shape[1])
        crop_h = max(1, int(height * crop_ratio))
        crop_w = max(1, int(width * crop_ratio))
        top = max(0, (height - crop_h) // 2)
        left = max(0, (width - crop_w) // 2)
        cropped = array[top : top + crop_h, left : left + crop_w]
        attacked.append(_resize_frame(cropped, (width, height)).astype(array.dtype))
    return attacked


def _spatial_corner_crop_resize(frames: list[Any], crop_ratio: float = 0.85) -> list[Any]:
    """左上角裁剪后恢复分辨率, 用于覆盖 off-center crop 类攻击。"""

    attacked: list[Any] = []
    for frame in frames:
        array = _to_numpy(frame)
        height, width = int(array.shape[0]), int(array.shape[1])
        crop_h = max(1, int(height * crop_ratio))
        crop_w = max(1, int(width * crop_ratio))
        cropped = array[:crop_h, :crop_w]
        attacked.append(_resize_frame(cropped, (width, height)).astype(array.dtype))
    return attacked


def _spatial_mask(frames: list[Any]) -> list[Any]:
    """在中心区域加入确定性遮挡, 近似 crop-and-drop / spatial mask 攻击。"""

    attacked: list[Any] = []
    for frame in frames:
        array = _to_numpy(frame).copy()
        height, width = int(array.shape[0]), int(array.shape[1])
        mask_h = max(1, int(height * 0.20))
        mask_w = max(1, int(width * 0.20))
        top = max(0, (height - mask_h) // 2)
        left = max(0, (width - mask_w) // 2)
        array[top : top + mask_h, left : left + mask_w] = 0
        attacked.append(array)
    return attacked


def _rotate_frames(frames: list[Any]) -> list[Any]:
    """小角度旋转帧, 用于空间同步扰动。"""

    import numpy as np
    from PIL import Image

    attacked: list[Any] = []
    for frame in frames:
        array = _to_numpy(frame)
        image = Image.fromarray(array.astype(np.uint8))
        rotated = image.rotate(5, resample=Image.BILINEAR)
        attacked.append(np.asarray(rotated).astype(array.dtype))
    return attacked


def _perspective_coefficients(destination_points: list[tuple[float, float]], source_points: list[tuple[float, float]]) -> list[float]:
    """计算 PIL perspective transform 所需的8个反向映射系数。"""

    import numpy as np

    matrix: list[list[float]] = []
    vector: list[float] = []
    for (dst_x, dst_y), (src_x, src_y) in zip(destination_points, source_points, strict=True):
        matrix.append([dst_x, dst_y, 1.0, 0.0, 0.0, 0.0, -src_x * dst_x, -src_x * dst_y])
        matrix.append([0.0, 0.0, 0.0, dst_x, dst_y, 1.0, -src_y * dst_x, -src_y * dst_y])
        vector.append(src_x)
        vector.append(src_y)
    coefficients = np.linalg.solve(np.asarray(matrix, dtype=np.float64), np.asarray(vector, dtype=np.float64))
    return [float(item) for item in coefficients]


def _perspective_warp_frames(frames: list[Any]) -> list[Any]:
    """对每帧执行真实 projective warp, 用于覆盖透视类空间失同步攻击。"""

    import numpy as np
    from PIL import Image

    attacked: list[Any] = []
    for frame in frames:
        array = _to_numpy(frame)
        height, width = int(array.shape[0]), int(array.shape[1])
        shift_x = max(1.0, width * 0.04)
        shift_y = max(1.0, height * 0.04)
        destination_points = [
            (0.0, 0.0),
            (float(width - 1), 0.0),
            (float(width - 1), float(height - 1)),
            (0.0, float(height - 1)),
        ]
        source_points = [
            (shift_x, shift_y),
            (float(width - 1) - shift_x, 0.0),
            (float(width - 1), float(height - 1) - shift_y),
            (0.0, float(height - 1)),
        ]
        image = Image.fromarray(array.astype(np.uint8))
        transform_mode = getattr(Image, "Transform", Image).PERSPECTIVE
        resample_mode = getattr(Image, "Resampling", Image).BILINEAR
        warped = image.transform(
            (width, height),
            transform_mode,
            _perspective_coefficients(destination_points, source_points),
            resample=resample_mode,
        )
        attacked.append(np.asarray(warped).astype(array.dtype))
    return attacked


def _brightness_contrast(frames: list[Any]) -> list[Any]:
    """调整亮度和对比度。"""

    import numpy as np

    attacked: list[Any] = []
    for frame in frames:
        array = _to_numpy(frame).astype(np.float32)
        mean = array.mean(axis=(0, 1), keepdims=True)
        adjusted = (array - mean) * 1.10 + mean
        adjusted = adjusted * 1.08
        attacked.append(_clip_like_uint8(adjusted).astype(_to_numpy(frame).dtype))
    return attacked


def _gamma_correction(frames: list[Any], gamma: float = 0.85) -> list[Any]:
    """执行确定性 gamma 校正, 用于覆盖非线性亮度变换。"""

    import numpy as np

    attacked: list[Any] = []
    gamma = max(0.05, float(gamma))
    for frame in frames:
        array = _to_numpy(frame).astype(np.float32)
        normalized = np.clip(array / 255.0, 0.0, 1.0)
        corrected = np.power(normalized, gamma) * 255.0
        attacked.append(_clip_like_uint8(corrected).astype(_to_numpy(frame).dtype))
    return attacked


def _gaussian_noise(frames: list[Any]) -> list[Any]:
    """添加确定性高斯近似噪声, 保证测试可复现。"""

    import numpy as np

    rng = np.random.default_rng(20260704)
    attacked: list[Any] = []
    for frame in frames:
        array = _to_numpy(frame)
        noise = rng.normal(loc=0.0, scale=4.0, size=array.shape)
        attacked.append(_clip_like_uint8(array.astype(np.float32) + noise).astype(array.dtype))
    return attacked


def _salt_pepper_noise(frames: list[Any]) -> list[Any]:
    """添加确定性 salt-and-pepper 噪声, 用于覆盖 impulse noise 攻击。"""

    import numpy as np

    rng = np.random.default_rng(20260704)
    attacked: list[Any] = []
    for frame in frames:
        array = _to_numpy(frame).copy()
        mask = rng.random(array.shape[:2])
        salt = mask < 0.005
        pepper = (mask >= 0.005) & (mask < 0.010)
        array[salt] = 255
        array[pepper] = 0
        attacked.append(array.astype(_to_numpy(frame).dtype))
    return attacked


def _color_jitter(frames: list[Any]) -> list[Any]:
    """执行确定性通道增益和偏移, 避免随机增强影响可复现性。"""

    import numpy as np

    gains = np.asarray([1.06, 0.96, 1.02], dtype=np.float32)
    attacked: list[Any] = []
    for frame in frames:
        array = _to_numpy(frame).astype(np.float32)
        if array.ndim >= 3 and array.shape[-1] >= 3:
            array[..., :3] = array[..., :3] * gains + 2.0
        else:
            array = array * 1.03 + 2.0
        attacked.append(_clip_like_uint8(array).astype(_to_numpy(frame).dtype))
    return attacked


def _jpeg_frame_compression(frames: list[Any], quality: int = 55) -> list[Any]:
    """对每帧执行 JPEG 内存往返压缩, 模拟平台抽帧压缩。"""

    import io
    import numpy as np
    from PIL import Image

    attacked: list[Any] = []
    for frame in frames:
        array = _to_numpy(frame)
        image = Image.fromarray(array.astype(np.uint8))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        restored = Image.open(buffer).convert(image.mode)
        attacked.append(np.asarray(restored).astype(array.dtype))
    return attacked


def apply_runtime_attack_to_frames(frames: list[Any], attack_name: str) -> tuple[list[Any], dict[str, Any]]:
    """对解码帧执行指定 runtime attack。

    返回值中的 metadata 是 governed records 的协议证据字段。该函数只执行
    repository 内可复现的视频文件级变换; 真实平台上传下载、screen recording
    和生成式重生成属于 non-runtime/adaptive 协议, 不在该函数中用 proxy 替代。
    """

    if not frames:
        raise ValueError("no_decodable_frames")
    spec = runtime_attack_spec(attack_name)
    normalized = spec.attack_name
    attacked = list(frames)

    if normalized in {
        "video_compression_runtime",
        "h264_crf18_runtime",
        "h264_crf23_runtime",
        "h264_crf28_runtime",
        "h264_crf33_runtime",
        "h264_crf38_runtime",
        "h265_crf23_runtime",
        "h265_crf28_runtime",
        "h265_crf33_runtime",
        "mpeg4_crf28_runtime",
        "mpeg4_q2_runtime",
        "mpeg4_q8_runtime",
        "platform_transcode_runtime",
    }:
        attacked = list(frames)
    elif normalized == "jpeg_frame_compression_runtime":
        attacked = _jpeg_frame_compression(frames)
    elif normalized == "temporal_crop_runtime":
        attacked = frames[1:-1] if len(frames) >= 4 else list(frames)
    elif normalized == "temporal_clip_middle_runtime":
        if len(frames) >= 6:
            clip_len = max(1, int(len(frames) * 0.20))
            start = max(1, (len(frames) - clip_len) // 2)
            attacked = list(frames[:start]) + list(frames[start + clip_len :])
        else:
            attacked = list(frames)
    elif normalized == "frame_rate_resampling_runtime":
        attacked = frames[::2] if len(frames) >= 3 else list(frames)
    elif normalized == "frame_drop_uniform_runtime":
        attacked = [frame for index, frame in enumerate(frames) if (index + 1) % 3 != 0] or list(frames)
    elif normalized == "irregular_frame_drop_runtime":
        attacked = [
            frame for index, frame in enumerate(frames)
            if index % 5 not in {1, 4}
        ] or list(frames)
    elif normalized == "frame_insert_duplicate_runtime":
        midpoint = max(0, len(frames) // 2)
        attacked = list(frames[:midpoint]) + [frames[midpoint]] + list(frames[midpoint:])
    elif normalized == "frame_insert_noise_runtime":
        midpoint = max(0, len(frames) // 2)
        noisy_frame = _gaussian_noise([frames[midpoint]])[0]
        attacked = list(frames[:midpoint]) + [noisy_frame] + list(frames[midpoint:])
    elif normalized == "frame_duplicate_runtime":
        attacked = []
        for index, frame in enumerate(frames):
            attacked.append(frame)
            if (index + 1) % 4 == 0:
                attacked.append(frame)
    elif normalized == "speed_change_runtime":
        attacked = []
        for index, frame in enumerate(frames):
            if (index + 1) % 4 == 0:
                continue
            attacked.append(frame)
            if (index + 1) % 6 == 0:
                attacked.append(frame)
        attacked = attacked or list(frames)
    elif normalized == "frame_swap_adjacent_runtime":
        attacked = list(frames)
        if len(attacked) >= 4:
            midpoint = len(attacked) // 2
            attacked[midpoint - 1], attacked[midpoint] = attacked[midpoint], attacked[midpoint - 1]
    elif normalized == "frame_average_runtime":
        import numpy as np

        attacked = []
        for index, frame in enumerate(frames):
            if index == 0:
                attacked.append(frame)
            else:
                averaged = (_to_numpy(frames[index - 1]).astype(np.float32) + _to_numpy(frame).astype(np.float32)) / 2.0
                attacked.append(_clip_like_uint8(averaged).astype(_to_numpy(frame).dtype))
    elif normalized == "spatial_resize_runtime":
        attacked = _spatial_resize(frames)
    elif normalized == "spatial_crop_resize_runtime":
        attacked = _spatial_crop_resize(frames)
    elif normalized == "spatial_corner_crop_resize_runtime":
        attacked = _spatial_corner_crop_resize(frames)
    elif normalized == "rotation_runtime":
        attacked = _rotate_frames(frames)
    elif normalized == "perspective_runtime":
        attacked = _perspective_warp_frames(frames)
    elif normalized == "spatial_mask_runtime":
        attacked = _spatial_mask(frames)
    elif normalized == "gaussian_noise_runtime":
        attacked = _gaussian_noise(frames)
    elif normalized == "salt_pepper_noise_runtime":
        attacked = _salt_pepper_noise(frames)
    elif normalized in {"gaussian_blur_runtime", "median_blur_runtime"}:
        attacked = [_blur_frame(frame, normalized) for frame in frames]
    elif normalized == "denoise_runtime":
        attacked = [_denoise_frame(frame) for frame in frames]
    elif normalized == "brightness_contrast_runtime":
        attacked = _brightness_contrast(frames)
    elif normalized == "gamma_correction_runtime":
        attacked = _gamma_correction(frames)
    elif normalized == "color_jitter_runtime":
        attacked = _color_jitter(frames)
    elif normalized == "sharpen_runtime":
        attacked = [_sharpen_frame(frame) for frame in frames]
    elif normalized == "compression_crop_combined_runtime":
        attacked = _spatial_crop_resize(frames)
    elif normalized == "compression_brightness_combined_runtime":
        attacked = _brightness_contrast(frames)
    elif normalized == "compression_temporal_combined_runtime":
        attacked = [frame for index, frame in enumerate(frames) if (index + 1) % 3 != 0] or list(frames)
    elif normalized == "compression_noise_combined_runtime":
        attacked = _gaussian_noise(frames)
    elif normalized == "compression_color_jitter_combined_runtime":
        attacked = _color_jitter(frames)
    elif normalized == "crop_rotation_combined_runtime":
        attacked = _rotate_frames(_spatial_crop_resize(frames))
    else:  # pragma: no cover - runtime_attack_spec 已经先行阻断未知名称。
        raise ValueError(f"unsupported_runtime_attack:{attack_name}")

    metadata: dict[str, Any] = {
        "attack_family": spec.attack_family,
        "attack_transform": spec.attack_transform,
        "attack_strength": spec.attack_strength,
        "runtime_attack_expected_effect": spec.runtime_attack_expected_effect,
        "runtime_attack_implementation_level": spec.implementation_level,
        "runtime_attack_formal_evidence_level": "formal_runtime_video_transform",
        "runtime_attack_claim_level": "paper_runtime_attack_protocol",
        "runtime_attack_proxy_free": True,
    }
    if spec.video_writer_codec:
        metadata["video_writer_codec"] = spec.video_writer_codec
    if spec.video_writer_output_params:
        metadata["video_writer_output_params"] = list(spec.video_writer_output_params)
    return attacked, metadata


def apply_runtime_attack_to_video_tensor(video: Any, attack_name: str) -> Any:
    """对 T-first 视频张量执行轻量 runtime attack。

    该函数用于 VideoSeal 这类以张量形式返回视频的 official runtime。若遇到
    空间或像素值攻击, 会尽量保持输入类型不变; 复杂几何攻击采用轻量近似。
    """

    spec = runtime_attack_spec(attack_name)
    normalized = spec.attack_name
    if normalized in {
        "video_compression_runtime",
        "h264_crf18_runtime",
        "h264_crf23_runtime",
        "h264_crf28_runtime",
        "h264_crf33_runtime",
        "h264_crf38_runtime",
        "h265_crf23_runtime",
        "h265_crf28_runtime",
        "h265_crf33_runtime",
        "mpeg4_crf28_runtime",
        "mpeg4_q2_runtime",
        "mpeg4_q8_runtime",
        "platform_transcode_runtime",
    }:
        return video
    if normalized == "jpeg_frame_compression_runtime":
        return _tensor_quantize(video, levels=64)
    if normalized == "temporal_crop_runtime":
        return video[1:-1] if video.shape[0] >= 4 else video
    if normalized == "temporal_clip_middle_runtime":
        if video.shape[0] >= 6:
            clip_len = max(1, int(video.shape[0] * 0.20))
            start = max(1, (int(video.shape[0]) - clip_len) // 2)
            return _tensor_concat([video[:start], video[start + clip_len :]])
        return video
    if normalized in {"frame_rate_resampling_runtime", "compression_temporal_combined_runtime"}:
        return video[::2] if video.shape[0] >= 3 else video
    if normalized == "frame_drop_uniform_runtime":
        indices = [index for index in range(int(video.shape[0])) if (index + 1) % 3 != 0]
        return video[indices] if indices else video
    if normalized == "irregular_frame_drop_runtime":
        indices = [index for index in range(int(video.shape[0])) if index % 5 not in {1, 4}]
        return video[indices] if indices else video
    if normalized == "frame_insert_duplicate_runtime":
        midpoint = max(0, int(video.shape[0]) // 2)
        return _tensor_concat([video[:midpoint], video[midpoint : midpoint + 1], video[midpoint:]])
    if normalized == "frame_insert_noise_runtime":
        midpoint = max(0, int(video.shape[0]) // 2)
        noisy_frame = _tensor_add_noise(video[midpoint : midpoint + 1])
        return _tensor_concat([video[:midpoint], noisy_frame, video[midpoint:]])
    if normalized == "frame_duplicate_runtime":
        parts = []
        for index in range(int(video.shape[0])):
            parts.append(video[index : index + 1])
            if (index + 1) % 4 == 0:
                parts.append(video[index : index + 1])
        return _tensor_concat(parts) if parts else video
    if normalized == "speed_change_runtime":
        parts = []
        for index in range(int(video.shape[0])):
            if (index + 1) % 4 == 0:
                continue
            parts.append(video[index : index + 1])
            if (index + 1) % 6 == 0:
                parts.append(video[index : index + 1])
        return _tensor_concat(parts) if parts else video
    if normalized == "frame_swap_adjacent_runtime":
        output = video.clone() if hasattr(video, "clone") else video.copy()
        if output.shape[0] >= 4:
            midpoint = int(output.shape[0]) // 2
            before = output[midpoint - 1].clone() if hasattr(output[midpoint - 1], "clone") else output[midpoint - 1].copy()
            output[midpoint - 1] = output[midpoint]
            output[midpoint] = before
        return output
    if normalized == "frame_average_runtime":
        output = video.clone() if hasattr(video, "clone") else video.copy()
        if output.shape[0] >= 2:
            output[1:] = (output[:-1] + output[1:]) / 2
        return output
    if normalized in {"spatial_resize_runtime", "spatial_crop_resize_runtime", "compression_crop_combined_runtime"}:
        return _tensor_center_crop(video, crop_ratio=0.80 if "crop" in normalized else 0.75)
    if normalized == "spatial_corner_crop_resize_runtime":
        return _tensor_corner_crop(video, crop_ratio=0.85)
    if normalized in {"rotation_runtime", "perspective_runtime"}:
        return _tensor_roll(video, shift=2)
    if normalized == "spatial_mask_runtime":
        return _tensor_spatial_mask(video)
    if normalized == "gaussian_noise_runtime":
        return _tensor_add_noise(video)
    if normalized == "salt_pepper_noise_runtime":
        return _tensor_salt_pepper(video)
    if normalized in {"gaussian_blur_runtime", "median_blur_runtime", "denoise_runtime"}:
        return _tensor_average_blur(video)
    if normalized in {"brightness_contrast_runtime", "compression_brightness_combined_runtime"}:
        return _tensor_brightness_contrast(video)
    if normalized == "gamma_correction_runtime":
        return _tensor_gamma(video)
    if normalized in {"color_jitter_runtime", "compression_color_jitter_combined_runtime"}:
        return _tensor_color_jitter(video)
    if normalized == "sharpen_runtime":
        return _tensor_sharpen_filter(video)
    if normalized == "compression_noise_combined_runtime":
        return _tensor_add_noise(video)
    if normalized == "crop_rotation_combined_runtime":
        return _tensor_roll(_tensor_center_crop(video, crop_ratio=0.80), shift=2)
    raise ValueError(f"unsupported_runtime_attack:{attack_name}")


def _tensor_concat(parts: list[Any]) -> Any:
    """按输入类型拼接张量或数组。"""

    if not parts:
        return parts
    if hasattr(parts[0], "new_empty"):
        import torch

        return torch.cat(parts, dim=0)
    import numpy as np

    return np.concatenate(parts, axis=0)


def _tensor_center_crop(video: Any, crop_ratio: float) -> Any:
    """对 T-first 张量做中心裁剪并恢复原分辨率。"""

    shape = video.shape
    if len(shape) < 4:
        return video
    height_axis, width_axis = (-2, -1) if shape[1] in {1, 3, 4} else (1, 2)
    height, width = int(shape[height_axis]), int(shape[width_axis])
    crop_h = max(1, int(height * crop_ratio))
    crop_w = max(1, int(width * crop_ratio))
    top = max(0, (height - crop_h) // 2)
    left = max(0, (width - crop_w) // 2)
    if height_axis == -2:
        cropped = video[..., top : top + crop_h, left : left + crop_w]
    else:
        cropped = video[:, top : top + crop_h, left : left + crop_w, ...]
    if hasattr(video, "new_empty"):
        import torch.nn.functional as F

        channels_first = cropped if height_axis == -2 else cropped.permute(0, 3, 1, 2)
        resized = F.interpolate(channels_first.float(), size=(height, width), mode="bilinear", align_corners=False)
        resized = resized.to(dtype=video.dtype)
        return resized if height_axis == -2 else resized.permute(0, 2, 3, 1)
    import numpy as np

    frames = [frame for frame in cropped]
    restored = [_resize_frame(frame, (width, height)) for frame in frames]
    return np.asarray(restored).astype(video.dtype)


def _tensor_corner_crop(video: Any, crop_ratio: float) -> Any:
    """对 T-first 张量做左上角裁剪并恢复原分辨率。"""

    shape = video.shape
    if len(shape) < 4:
        return video
    height_axis, width_axis = (-2, -1) if shape[1] in {1, 3, 4} else (1, 2)
    height, width = int(shape[height_axis]), int(shape[width_axis])
    crop_h = max(1, int(height * crop_ratio))
    crop_w = max(1, int(width * crop_ratio))
    if height_axis == -2:
        cropped = video[..., :crop_h, :crop_w]
    else:
        cropped = video[:, :crop_h, :crop_w, ...]
    if hasattr(video, "new_empty"):
        import torch.nn.functional as F

        channels_first = cropped if height_axis == -2 else cropped.permute(0, 3, 1, 2)
        resized = F.interpolate(channels_first.float(), size=(height, width), mode="bilinear", align_corners=False)
        resized = resized.to(dtype=video.dtype)
        return resized if height_axis == -2 else resized.permute(0, 2, 3, 1)
    import numpy as np

    frames = [frame for frame in cropped]
    restored = [_resize_frame(frame, (width, height)) for frame in frames]
    return np.asarray(restored).astype(video.dtype)


def _tensor_spatial_mask(video: Any) -> Any:
    """对 T-first 张量中心区域加遮挡。"""

    output = video.clone() if hasattr(video, "clone") else video.copy()
    shape = output.shape
    if len(shape) < 4:
        return output
    height_axis, width_axis = (-2, -1) if shape[1] in {1, 3, 4} else (1, 2)
    height, width = int(shape[height_axis]), int(shape[width_axis])
    mask_h = max(1, int(height * 0.20))
    mask_w = max(1, int(width * 0.20))
    top = max(0, (height - mask_h) // 2)
    left = max(0, (width - mask_w) // 2)
    if height_axis == -2:
        output[..., top : top + mask_h, left : left + mask_w] = 0
    else:
        output[:, top : top + mask_h, left : left + mask_w, ...] = 0
    return output


def _tensor_roll(video: Any, shift: int) -> Any:
    """对张量做水平平移, 用于官方 baseline 张量路径的几何扰动。"""

    if hasattr(video, "roll"):
        return video.roll(shifts=shift, dims=-1)
    import numpy as np

    return np.roll(video, shift=shift, axis=-2 if video.shape[-1] in {1, 3, 4} else -1)


def _tensor_add_noise(video: Any) -> Any:
    """对张量添加小幅噪声。"""

    if hasattr(video, "new_empty"):
        import torch

        generator = torch.Generator(device=video.device)
        generator.manual_seed(20260704)
        noise = torch.randn(video.shape, generator=generator, device=video.device, dtype=video.dtype) * 0.015
        return (video + noise).clamp(0.0, 1.0)
    import numpy as np

    rng = np.random.default_rng(20260704)
    return np.clip(video.astype("float32") + rng.normal(0.0, 4.0, size=video.shape), 0, 255).astype(video.dtype)


def _tensor_salt_pepper(video: Any) -> Any:
    """对张量添加确定性 impulse noise。"""

    if hasattr(video, "new_empty"):
        import torch

        generator = torch.Generator(device=video.device)
        generator.manual_seed(20260704)
        mask = torch.rand(video.shape, generator=generator, device=video.device)
        output = video.clone()
        output = torch.where(mask < 0.005, torch.ones_like(output), output)
        output = torch.where((mask >= 0.005) & (mask < 0.010), torch.zeros_like(output), output)
        return output
    import numpy as np

    rng = np.random.default_rng(20260704)
    mask = rng.random(video.shape)
    output = video.copy()
    output[mask < 0.005] = 255
    output[(mask >= 0.005) & (mask < 0.010)] = 0
    return output.astype(video.dtype)


def _tensor_average_blur(video: Any) -> Any:
    """使用平均池化近似 blur。"""

    if hasattr(video, "new_empty"):
        import torch.nn.functional as F

        channels_first = video if video.shape[1] in {1, 3, 4} else video.permute(0, 3, 1, 2)
        blurred = F.avg_pool2d(channels_first.float(), kernel_size=3, stride=1, padding=1).to(dtype=video.dtype)
        return blurred if video.shape[1] in {1, 3, 4} else blurred.permute(0, 2, 3, 1)
    return video


def _tensor_brightness_contrast(video: Any) -> Any:
    """调整张量亮度和对比度。"""

    if hasattr(video, "new_empty"):
        mean = video.mean()
        return (((video - mean) * 1.10 + mean) * 1.08).clamp(0.0, 1.0)
    import numpy as np

    mean = video.mean()
    return np.clip(((video.astype("float32") - mean) * 1.10 + mean) * 1.08, 0, 255).astype(video.dtype)


def _tensor_gamma(video: Any, gamma: float = 0.85) -> Any:
    """对张量执行 gamma 校正。"""

    gamma = max(0.05, float(gamma))
    if hasattr(video, "new_empty"):
        return video.clamp(0.0, 1.0).pow(gamma)
    import numpy as np

    array = video.astype("float32")
    max_value = 255.0 if float(np.max(array)) > 1.5 else 1.0
    corrected = np.power(np.clip(array / max_value, 0.0, 1.0), gamma) * max_value
    return corrected.astype(video.dtype)


def _tensor_color_jitter(video: Any) -> Any:
    """对张量执行确定性颜色扰动。"""

    if hasattr(video, "new_empty"):
        output = video.clone()
        if output.shape[1] in {3, 4}:
            output[:, 0] = output[:, 0] * 1.06 + 0.01
            output[:, 1] = output[:, 1] * 0.96 + 0.01
            output[:, 2] = output[:, 2] * 1.02 + 0.01
        elif output.shape[-1] in {3, 4}:
            output[..., 0] = output[..., 0] * 1.06 + 0.01
            output[..., 1] = output[..., 1] * 0.96 + 0.01
            output[..., 2] = output[..., 2] * 1.02 + 0.01
        else:
            output = output * 1.03 + 0.01
        return output.clamp(0.0, 1.0)
    import numpy as np

    output = video.astype("float32").copy()
    if output.shape[-1] in {3, 4}:
        output[..., 0] = output[..., 0] * 1.06 + 2.0
        output[..., 1] = output[..., 1] * 0.96 + 2.0
        output[..., 2] = output[..., 2] * 1.02 + 2.0
    else:
        output = output * 1.03 + 2.0
    return np.clip(output, 0, 255).astype(video.dtype)


def _tensor_sharpen_filter(video: Any) -> Any:
    """使用原始张量与低通张量差值构造确定性 sharpen 滤波。"""

    if hasattr(video, "new_empty"):
        blurred = _tensor_average_blur(video)
        return (video + (video - blurred) * 0.5).clamp(0.0, 1.0)
    import numpy as np

    array = video.astype("float32")
    height_axis, width_axis = (-2, -1) if array.ndim >= 4 and array.shape[1] in {1, 3, 4} else (1, 2)
    shifted = (
        array
        + np.roll(array, shift=1, axis=height_axis)
        + np.roll(array, shift=-1, axis=height_axis)
        + np.roll(array, shift=1, axis=width_axis)
        + np.roll(array, shift=-1, axis=width_axis)
    ) / 5.0
    max_value = 255.0 if float(np.max(array)) > 1.5 else 1.0
    return np.clip(array + (array - shifted) * 0.5, 0, max_value).astype(video.dtype)


def _tensor_quantize(video: Any, levels: int) -> Any:
    """使用离散量化近似 JPEG 压缩带来的像素值损失。"""

    levels = max(2, int(levels))
    if hasattr(video, "new_empty"):
        return (video * levels).round().div(levels).clamp(0.0, 1.0)
    import numpy as np

    step = max(1, 256 // levels)
    return (np.round(video.astype("float32") / step) * step).clip(0, 255).astype(video.dtype)
