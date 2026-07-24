"""Construction protocol primitives for output-feature impulse observability."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256, shake_256
import hmac
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


IMPULSE_OBSERVABILITY_PROFILE_ID = (
    "sstw_output_feature_impulse_observability_construction"
)
OBSERVER_SYNCHRONIZED_METHOD_ID = (
    "observer_synchronized_state_space_trajectory_watermarking"
)
CONSTRUCTION_FEATURE_EXTRACTOR_SYMBOL = "phi_construction"
CONSTRUCTION_FEATURE_ENCODER_ID = (
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers::vae"
)
CONSTRUCTION_FEATURE_ENCODER_REVISION = (
    "0fad780a534b6463e45facd96134c9f345acfa5b"
)
CONSTRUCTION_FEATURE_OUTPUT_DIMENSION = 256
CONSTRUCTION_FLOW_STEP_COUNT = 8
FLOW_MACRO_INTERVAL_COUNT = 3
WATERMARK_STATE_DIMENSION = 2
STAGE_BASIS_RANK = 6
IMPULSE_TRIAGE_VIDEO_COUNT = 14
IMPULSE_PROBE_COUNT = 12
IMPULSE_CLEAN_REPEAT_COUNT = 2
STAGE_NAMES = ("early_flow", "middle_flow", "late_flow")
POLARITY_NAMES = ("positive", "negative")
CONSTRUCTION_BASIS_SUBKEY_DOMAIN = (
    "sstw_observer_synchronized_impulse_construction_basis"
)
CONSTRUCTION_BASIS_OWNER_LABEL = "construction_owner_stage_basis"
CONSTRUCTION_BASIS_WRONG_KEY_INDEX = 0
CONSTRUCTION_LATENT_LAYOUT_SHAPE = (1, 16, 9, 40, 64)
CONSTRUCTION_RUNTIME_ADAPTER_SCHEMA_ID = (
    "impulse_runtime_actual_delta_exposure_adapter"
)
CONSTRUCTION_SIGMA_GRID = (
    1.0,
    0.9475425481796265,
    0.8827877640724182,
    0.8008373379707336,
    0.6937931180000305,
    0.5480455756187439,
    0.33797216415405273,
    0.008928571827709675,
    0.0,
)
CONSTRUCTION_DELTA_SIGMA = (
    -0.052457451820373535,
    -0.06475478410720825,
    -0.08195042610168457,
    -0.10704421997070312,
    -0.14574754238128662,
    -0.21007341146469116,
    -0.32904359698295593,
    -0.008928571827709675,
)
CONSTRUCTION_FLOW_PHASES = tuple(
    (
        0.5
        * (
            CONSTRUCTION_SIGMA_GRID[index]
            + CONSTRUCTION_SIGMA_GRID[index + 1]
        )
        - CONSTRUCTION_SIGMA_GRID[0]
    )
    / (CONSTRUCTION_SIGMA_GRID[-1] - CONSTRUCTION_SIGMA_GRID[0])
    for index in range(CONSTRUCTION_FLOW_STEP_COUNT)
)
CONSTRUCTION_MACRO_INTERVAL_INDICES = (0, 0, 0, 0, 1, 1, 2, 2)
CONSTRUCTION_TEMPORAL_WAVEFORMS = (
    (1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0),
)


@dataclass(frozen=True)
class ImpulseProbePlanRecord:
    """One construction-only video identity in the frozen 14-video plan."""

    probe_id: str
    probe_role: str
    stage_index: int | None
    stage_name: str | None
    channel_index: int | None
    polarity: int
    nominal_signed_amplitude: float
    formal_result: bool = False
    stage_progression_allowed: bool = False


@dataclass(frozen=True)
class ActualImpulseExposureTrace:
    """Stepwise actual exposure for one signed impulse video."""

    probe_id: str
    stage_index: int
    channel_index: int
    polarity: int
    step_indices: tuple[int, ...]
    flow_phase_by_step: tuple[float, ...]
    delta_sigma_by_step: tuple[float, ...]
    macro_interval_index_by_step: tuple[int, ...]
    intended_velocity_waveform_by_step: tuple[float, ...]
    reference_base_velocity_norm_by_step: tuple[float, ...]
    remaining_control_energy_before_step_by_step: tuple[float, ...]
    reference_energy_increment_by_step: tuple[float, ...]
    reference_cumulative_energy_by_step: tuple[float, ...]
    intended_delta_norm_by_step: tuple[float, ...]
    actual_velocity_basis_coordinate_by_step: tuple[float, ...]
    actual_channel_velocity_coordinate_by_step: tuple[
        tuple[float, ...], ...
    ]
    intended_signed_exposure_by_step: tuple[float, ...]
    actual_signed_exposure_by_step: tuple[float, ...]
    actual_channel_exposure_by_step: tuple[tuple[float, ...], ...]
    actual_exposure_vector: tuple[float, ...]
    delta_norm_by_step: tuple[float, ...]
    projection_scale_by_step: tuple[float, ...]
    cumulative_energy_by_step: tuple[float, ...]
    direction_cosine_by_step: tuple[float, ...]
    norm_guard_passed_by_step: tuple[bool, ...]
    energy_guard_passed_by_step: tuple[bool, ...]
    waveform_schema_digest: str
    runtime_adapter_schema_digest: str
    basis_digest: str


@dataclass(frozen=True)
class ActualDesignMatrix:
    """Compressed six-dimensional actual design after all waveform gates."""

    probe_ids: tuple[str, ...]
    values: np.ndarray
    rank: int
    condition_number: float
    waveform_cosine_by_probe: Mapping[str, float]
    intended_actual_ratio_by_probe: Mapping[str, tuple[float, ...]]
    positive_negative_symmetry_by_channel: Mapping[str, float]
    positive_negative_amplitude_asymmetry_by_channel: Mapping[str, float]
    cross_channel_leakage_by_probe: Mapping[str, float]
    compression_allowed: bool = True


@dataclass(frozen=True)
class ConstructionTransferEstimate:
    """Clean-intercept transfer estimate; this is not an observer."""

    clean_intercept: np.ndarray
    transfer_matrix: np.ndarray
    fitted_output_features: np.ndarray
    residual_matrix: np.ndarray
    signed_center_difference_transfer_matrix: np.ndarray


@dataclass(frozen=True)
class ConstructionStageBasis:
    """In-memory owner/wrong construction basis; values are never a record."""

    values: np.ndarray
    basis_digest: str
    latent_layout_shape: tuple[int, ...]
    wrong_key_candidate_index: int | None


@dataclass(frozen=True)
class IntendedImpulseControl:
    """Frozen scalar control budget for one Flow interval."""

    intended_delta_norm: float
    norm_limited_delta_norm: float
    energy_limited_delta_norm: float
    reference_energy_increment: float
    projected_reference_energy: float
    total_flow_energy_budget: float
    remaining_control_energy: float
    signed_velocity_coordinate: float
    signed_state_update_exposure: float


@dataclass(frozen=True)
class ValidatedConstructionFeatures:
    """Schema-bound, finite, row-wise L2 construction output features."""

    values: np.ndarray
    feature_schema_digest: str
    probe_ids: tuple[str, ...]
    row_binding_digests: tuple[str, ...]


@dataclass(frozen=True)
class GateAStatistics:
    """Executable construction-only Gate A statistics, never stage evidence."""

    clean_repeat_distance: float
    finite_noise_floor: float
    clean_distance_below_numerical_resolution: bool
    minimum_absolute_response: float
    minimum_response_singular_value: float
    noise_normalized_minimum_singular_value: float
    effective_rank: int
    output_transfer_condition_number: float
    minimum_output_feature_snr: float
    minimum_positive_negative_antisymmetry_cosine: float
    maximum_positive_negative_antisymmetry_residual_ratio: float
    primary_checkpoint_chain_ready: bool
    gate_a_ready: bool
    formal_result: bool = False
    stage_progression_allowed: bool = False


def canonical_json_digest(payload: Mapping[str, Any]) -> str:
    return sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def construction_feature_schema_digest(
    feature_schema: Mapping[str, Any],
) -> str:
    payload = {
        key: value
        for key, value in feature_schema.items()
        if key != "feature_schema_digest"
    }
    return canonical_json_digest(payload)


def construction_feature_row_binding_digest(
    *,
    probe_id: str,
    feature_schema_digest: str,
    feature_values: Sequence[float],
) -> str:
    """Bind one immutable feature row to its governed probe identity."""

    identity = str(probe_id)
    schema_digest = str(feature_schema_digest)
    if not identity:
        raise ValueError("feature row binding 缺少 probe_id")
    if (
        len(schema_digest) != 64
        or any(
            character not in "0123456789abcdef"
            for character in schema_digest
        )
    ):
        raise ValueError("feature row binding schema digest 非法")
    row = np.asarray(feature_values, dtype="<f8")
    if row.shape != (CONSTRUCTION_FEATURE_OUTPUT_DIMENSION,):
        raise ValueError("feature row binding 要求精确256维")
    if not np.all(np.isfinite(row)):
        raise ValueError("feature row binding 含非有限值")
    header = json.dumps(
        {
            "algorithm": (
                "sha256_probe_id_feature_schema_digest_and_"
                "little_endian_float64_feature_bytes"
            ),
            "feature_schema_digest": schema_digest,
            "probe_id": identity,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(
        header + b"\x00" + np.ascontiguousarray(row).tobytes(order="C")
    ).hexdigest()


def _schema_digest(
    schema: Mapping[str, Any],
    *,
    digest_key: str,
) -> str:
    return canonical_json_digest(
        {key: value for key, value in schema.items() if key != digest_key}
    )


def flow_schedule_waveform_schema_digest(
    schedule: Mapping[str, Any],
) -> str:
    return _schema_digest(schedule, digest_key="waveform_schema_digest")


def runtime_adapter_schema_digest(
    adapter: Mapping[str, Any],
) -> str:
    return _schema_digest(adapter, digest_key="adapter_schema_digest")


def derive_construction_basis_subkey(
    master_key_text: str,
    *,
    wrong_key_candidate_index: int | None = None,
) -> str:
    """Derive one prompt/seed/model/grid-independent construction subkey."""

    secret = str(master_key_text).encode("utf-8")
    if len(secret) < 16:
        raise ValueError("construction master key 至少需要16个 UTF-8 bytes")
    if wrong_key_candidate_index not in (
        None,
        CONSTRUCTION_BASIS_WRONG_KEY_INDEX,
    ):
        raise ValueError("construction wrong key index 未冻结")
    label = (
        CONSTRUCTION_BASIS_OWNER_LABEL
        if wrong_key_candidate_index is None
        else f"construction_wrong_stage_basis_index_{wrong_key_candidate_index}"
    )
    message = (
        f"{CONSTRUCTION_BASIS_SUBKEY_DOMAIN}::{label}"
    ).encode("utf-8")
    return hmac.new(secret, message, sha256).hexdigest()


def _deterministic_normal_matrix(
    *,
    subkey_hex: str,
    row_count: int,
    column_count: int,
) -> np.ndarray:
    """Expand a subkey with SHAKE256 and deterministic Box-Muller normals."""

    if row_count < column_count or column_count <= 0:
        raise ValueError("construction basis shape 非法")
    value_count = row_count * column_count
    pair_count = (value_count + 1) // 2
    domain = (
        f"{CONSTRUCTION_BASIS_SUBKEY_DOMAIN}::"
        "shake256_box_muller_float64_big_endian_uint64"
    ).encode("utf-8")
    raw = shake_256(bytes.fromhex(subkey_hex) + domain).digest(
        pair_count * 16
    )
    integers = np.frombuffer(raw, dtype=">u8").astype(np.float64)
    uniforms = (integers + 0.5) / float(2**64)
    first = uniforms[0::2]
    second = uniforms[1::2]
    radius = np.sqrt(-2.0 * np.log(first))
    angle = 2.0 * math.pi * second
    normal = np.empty(pair_count * 2, dtype=np.float64)
    normal[0::2] = radius * np.cos(angle)
    normal[1::2] = radius * np.sin(angle)
    return normal[:value_count].reshape(
        row_count,
        column_count,
        order="C",
    )


def build_construction_stage_basis(
    master_key_text: str,
    *,
    wrong_key_candidate_index: int | None = None,
    latent_layout_shape: Sequence[int] = CONSTRUCTION_LATENT_LAYOUT_SHAPE,
) -> ConstructionStageBasis:
    """Build deterministic CPU U_K and freeze B_K,j = U_K E_j."""

    shape = tuple(int(value) for value in latent_layout_shape)
    if shape != CONSTRUCTION_LATENT_LAYOUT_SHAPE:
        raise ValueError("construction Wan latent layout 未冻结")
    row_count = math.prod(shape)
    subkey = derive_construction_basis_subkey(
        master_key_text,
        wrong_key_candidate_index=wrong_key_candidate_index,
    )
    raw = _deterministic_normal_matrix(
        subkey_hex=subkey,
        row_count=row_count,
        column_count=STAGE_BASIS_RANK,
    )
    orthonormal = np.zeros_like(raw, dtype=np.float64)
    for column_index in range(STAGE_BASIS_RANK):
        column = raw[:, column_index].copy()
        for previous_index in range(column_index):
            previous = orthonormal[:, previous_index]
            column -= float(previous @ column) * previous
        norm = float(np.linalg.norm(column))
        if not math.isfinite(norm) or norm <= 1e-12:
            raise RuntimeError("construction basis modified Gram-Schmidt 退化")
        column /= norm
        sign_index = int(np.argmax(np.abs(column)))
        if float(column[sign_index]) < 0.0:
            column *= -1.0
        orthonormal[:, column_index] = column
    values = np.ascontiguousarray(orthonormal, dtype="<f4")
    gram_error = float(
        np.max(
            np.abs(
                values.astype(np.float64).T
                @ values.astype(np.float64)
                - np.eye(STAGE_BASIS_RANK)
            )
        )
    )
    if gram_error > 1e-5:
        raise RuntimeError("construction basis float32 Gram guard 失败")
    digest = sha256(values.tobytes(order="C")).hexdigest()
    return ConstructionStageBasis(
        values=values,
        basis_digest=digest,
        latent_layout_shape=shape,
        wrong_key_candidate_index=wrong_key_candidate_index,
    )


def compute_intended_impulse_control(
    *,
    probe_state_update_polarity: int,
    temporal_waveform: float,
    delta_sigma: float,
    base_velocity_norm: float,
    cumulative_control_energy: float,
    cumulative_reference_energy: float,
    remaining_step_count: int,
    lambda_max: float = 0.12,
    velocity_norm_ratio_budget: float = 0.02,
    flow_energy_budget_ratio: float = 0.000015,
) -> IntendedImpulseControl:
    """Compute the exact frozen norm/energy control in state-update polarity."""

    polarity = int(probe_state_update_polarity)
    waveform = float(temporal_waveform)
    interval = float(delta_sigma)
    base_norm = float(base_velocity_norm)
    control_energy = float(cumulative_control_energy)
    reference_energy = float(cumulative_reference_energy)
    remaining = int(remaining_step_count)
    numeric = (
        waveform,
        interval,
        base_norm,
        control_energy,
        reference_energy,
        float(lambda_max),
        float(velocity_norm_ratio_budget),
        float(flow_energy_budget_ratio),
    )
    if polarity not in (-1, 1):
        raise ValueError("probe polarity 必须是 state-update coordinate 的正负1")
    if (
        any(not math.isfinite(value) for value in numeric)
        or abs(interval) <= 1e-12
        or base_norm < 0.0
        or control_energy < 0.0
        or reference_energy < 0.0
        or remaining <= 0
        or not 0.0 <= abs(waveform) <= 1.0
    ):
        raise ValueError("intended impulse control context 不完整")
    reference_increment = interval**2 * base_norm**2
    projected_reference = (
        reference_energy + reference_increment * remaining
    )
    total_energy_budget = (
        float(flow_energy_budget_ratio) * projected_reference
    )
    remaining_energy = max(0.0, total_energy_budget - control_energy)
    norm_limited = (
        base_norm
        * float(velocity_norm_ratio_budget)
        * float(lambda_max)
        * abs(waveform)
    )
    energy_limited = math.sqrt(remaining_energy) / abs(interval)
    intended_norm = min(norm_limited, energy_limited)
    velocity_sign = polarity * (1 if interval > 0.0 else -1)
    velocity_coordinate = velocity_sign * intended_norm
    state_exposure = interval * velocity_coordinate
    return IntendedImpulseControl(
        intended_delta_norm=intended_norm,
        norm_limited_delta_norm=norm_limited,
        energy_limited_delta_norm=energy_limited,
        reference_energy_increment=reference_increment,
        projected_reference_energy=projected_reference,
        total_flow_energy_budget=total_energy_budget,
        remaining_control_energy=remaining_energy,
        signed_velocity_coordinate=velocity_coordinate,
        signed_state_update_exposure=state_exposure,
    )


def extract_construction_output_feature_from_normalized_latent(
    normalized_wan_latent: np.ndarray,
    *,
    zero_rejection_epsilon: float = 1e-12,
) -> np.ndarray:
    """Reference NumPy phi_construction pooling after frozen Wan VAE encode."""

    latent = np.asarray(normalized_wan_latent)
    if latent.shape != CONSTRUCTION_LATENT_LAYOUT_SHAPE:
        raise ValueError(
            "phi_construction normalized latent 必须为 [1,16,9,40,64]"
        )
    if latent.dtype != np.float32:
        raise ValueError("phi_construction normalized latent 必须为 float32")
    if not np.all(np.isfinite(latent)):
        raise ValueError("phi_construction normalized latent 含非有限值")
    _, channel_count, _, height, width = latent.shape
    pooled: list[float] = []
    for channel_index in range(channel_count):
        for row_index in range(4):
            row_start = math.floor(row_index * height / 4)
            row_end = math.floor((row_index + 1) * height / 4)
            for column_index in range(4):
                column_start = math.floor(column_index * width / 4)
                column_end = math.floor((column_index + 1) * width / 4)
                cell = latent[
                    0,
                    channel_index,
                    :,
                    row_start:row_end,
                    column_start:column_end,
                ]
                pooled.append(float(np.mean(cell, dtype=np.float64)))
    vector = np.asarray(pooled, dtype=np.float64)
    if vector.shape != (CONSTRUCTION_FEATURE_OUTPUT_DIMENSION,):
        raise AssertionError("phi_construction pooling dimension 组装失败")
    norm = float(np.linalg.norm(vector))
    epsilon = float(zero_rejection_epsilon)
    if (
        not math.isfinite(epsilon)
        or epsilon <= 0.0
        or not math.isfinite(norm)
        or norm <= epsilon
    ):
        raise ValueError("phi_construction L2 zero rejection 未通过")
    feature = vector / norm
    if not np.all(np.isfinite(feature)):
        raise ValueError("phi_construction L2 normalization 非有限")
    return feature


def load_impulse_observability_config(
    path: str | Path,
) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_impulse_observability_config(config)
    return config


def validate_impulse_observability_config(
    config: Mapping[str, Any],
) -> None:
    """Fail closed unless the upstream construction contract stays frozen."""

    exact = {
        "profile_id": IMPULSE_OBSERVABILITY_PROFILE_ID,
        "method_id": OBSERVER_SYNCHRONIZED_METHOD_ID,
        "claim_support_status": (
            "construction_contract_only_not_method_evidence"
        ),
        "formal_result": False,
        "stage_progression_allowed": False,
        "primary_detection_input": (
            "saved_video_plus_key_plus_frozen_public_feature_extractor"
        ),
        "prompt_conditioned_replay_role": "construction_diagnostic_only",
        "flow_macro_interval_count": FLOW_MACRO_INTERVAL_COUNT,
        "watermark_state_dimension": WATERMARK_STATE_DIMENSION,
        "stage_basis_rank": STAGE_BASIS_RANK,
        "impulse_triage_video_count": IMPULSE_TRIAGE_VIDEO_COUNT,
        "impulse_probe_count": IMPULSE_PROBE_COUNT,
        "clean_repeat_count": IMPULSE_CLEAN_REPEAT_COUNT,
        "impulse_identity_count": 1,
        "observer_execution_allowed": False,
        "state_dynamics_construction_allowed": False,
        "formal_llr_execution_allowed": False,
        "owner_wrong_method_smoke_allowed": False,
        "attack_execution_allowed": False,
        "pilot_execution_allowed": False,
        "fixed_fpr_execution_allowed": False,
        "external_baseline_execution_allowed": False,
        "notebook_change_allowed": False,
        "pretrained_generation_model_parameter_update_allowed": False,
        "training_or_finetuning_allowed": False,
        "inference_time_sampler_control_only": True,
    }
    required_top_level_keys = set(exact) | {
        "actual_exposure_contract",
        "authorization_state_machine",
        "construction_basis",
        "construction_feature_schema",
        "execution_identity",
        "flow_schedule_contract",
        "future_observer_boundary",
        "gate_a_sample_internal_causal_observability",
        "gate_b_cross_identity_identifiability",
        "gate_c_composite_trajectory_order_identifiability",
        "impulse_probe",
        "key_selectivity_construction",
        "primary_gate_checkpoint",
        "runtime_adapter_contract",
        "stage_basis",
        "time_axes",
        "transfer_checkpoint_contract",
        "transfer_checkpoints",
        "transfer_model",
    }
    if set(config) != required_top_level_keys:
        raise ValueError(
            "impulse observability 顶层字段集合未冻结: "
            + json.dumps(
                {
                    "missing": sorted(required_top_level_keys - set(config)),
                    "unexpected": sorted(set(config) - required_top_level_keys),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    mismatches = {
        key: {"expected": expected, "observed": config.get(key)}
        for key, expected in exact.items()
        if config.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            "impulse observability 顶层合同未冻结: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )

    time_axes = config.get("time_axes")
    if not isinstance(time_axes, Mapping) or time_axes != {
        "active_axis": "flow_diffusion_time",
        "flow_macro_interval_boundaries": [0.0, 1 / 3, 2 / 3, 1.0],
        "flow_macro_interval_boundary_rule": (
            "left_closed_right_open_except_final"
        ),
        "video_frame_time_analysis_allowed": False,
        "frame_deletion_analysis_allowed": False,
        "dtw_analysis_allowed": False,
        "variable_frame_rate_analysis_allowed": False,
        "attack_analysis_allowed": False,
    }:
        raise ValueError("Flow time 与 video frame time 合同未冻结")

    basis = config.get("stage_basis")
    if not isinstance(basis, Mapping) or basis != {
        "construction": "B_K_j_equals_U_K_E_j",
        "shared_rotated_basis_claim_allowed": False,
        "stage_selector_coordinate_pairs": [[0, 1], [2, 3], [4, 5]],
        "maximum_within_block_gram_error": 1e-5,
        "maximum_cross_stage_coherence": 0.05,
        "maximum_wrong_key_latent_coherence": 0.25,
        "latent_orthogonality_sufficient_for_output_observability": False,
    }:
        raise ValueError("三阶段低相干 block 合同未冻结")

    construction_basis = config.get("construction_basis")
    if not isinstance(construction_basis, Mapping) or construction_basis != {
        "basis_column_count": 6,
        "basis_digest_algorithm": (
            "sha256_canonical_float32_little_endian_bytes"
        ),
        "basis_serialization_in_results_allowed": False,
        "canonical_generation_device": "cpu",
        "canonical_generation_dtype": "float64_then_float32",
        "column_sign_rule": "largest_absolute_coordinate_positive",
        "flatten_order": (
            "c_contiguous_batch_channel_time_height_width"
        ),
        "latent_layout_shape": [1, 16, 9, 40, 64],
        "master_key_minimum_utf8_bytes": 16,
        "orthonormalization": (
            "modified_gram_schmidt_float64_column_order_0_to_5"
        ),
        "owner_subkey_label": CONSTRUCTION_BASIS_OWNER_LABEL,
        "prf_expansion": (
            "shake256_subkey_and_domain_to_big_endian_uint64_box_muller"
        ),
        "subkey_derivation": "hmac_sha256_domain_separated",
        "subkey_domain": CONSTRUCTION_BASIS_SUBKEY_DOMAIN,
        "wrong_key_derivation": (
            "hmac_sha256_owner_master_key_domain_wrong_key_index_0"
        ),
        "wrong_key_index": CONSTRUCTION_BASIS_WRONG_KEY_INDEX,
    }:
        raise ValueError("construction U_K KDF/latent/QR 合同未冻结")

    feature = config.get("construction_feature_schema")
    if not isinstance(feature, Mapping):
        raise ValueError("construction feature schema 缺失")
    required_feature = {
        "extractor_symbol": CONSTRUCTION_FEATURE_EXTRACTOR_SYMBOL,
        "encoder_id": CONSTRUCTION_FEATURE_ENCODER_ID,
        "encoder_revision": CONSTRUCTION_FEATURE_ENCODER_REVISION,
        "candidate_key_independent": True,
        "public_specification": True,
        "freeze_before_impulse_generation": True,
        "impulse_sample_training_allowed": False,
        "owner_label_access_allowed": False,
        "postrun_feature_selection_allowed": False,
        "decode_pixel_format": "rgb24",
        "diffusers_version": "0.35.2",
        "encode_execution_dtype": "bfloat16",
        "encode_patch_size_required": None,
        "encode_strategy": (
            "encode_video_to_wan_endpoint_latent_"
            "cpu_resident_spatiotemporal_streaming"
        ),
        "video_tensor_loader": "imageio_v3_all_frames_file_order",
        "video_normalization": "float32_rgb_divide_127_5_minus_1",
        "frame_order": "all_decoded_frames_in_file_order",
        "input_frame_count": 33,
        "input_height": 320,
        "input_resize_allowed": False,
        "input_width": 512,
        "encoder_class": "AutoencoderKLWan",
        "encoder_subfolder": "vae",
        "posterior_statistic": "deterministic_mode_or_first_moment",
        "latent_normalization": (
            "subtract_config_latents_mean_multiply_inverse_latents_std"
        ),
        "latent_layout": "batch_channel_latent_time_height_width",
        "latent_channel_count": 16,
        "temporal_segmentation": (
            "single_global_latent_time_mean_no_frame_time_claim"
        ),
        "layer_tensor": "normalized_vae_endpoint_latent",
        "spatial_pooling": "equal_area_4_by_4_spatial_grid_cell_mean",
        "pooling_accumulator_dtype": "float64",
        "pooling_cell_boundary_rule": (
            "floor_i_size_div_4_to_floor_i_plus_1_size_div_4"
        ),
        "channel_statistics": ["mean"],
        "output_dimension": CONSTRUCTION_FEATURE_OUTPUT_DIMENSION,
        "output_normalization": "feature_vector_l2_with_zero_rejection",
        "output_normalization_epsilon": 1e-12,
        "output_normalization_norm_tolerance": 1e-6,
        "output_row_identity_binding": (
            "exact_frozen_probe_ids_from_per_video_feature_records"
        ),
        "output_row_binding_digest_algorithm": (
            "sha256_probe_id_feature_schema_digest_and_"
            "little_endian_float64_feature_bytes"
        ),
        "output_row_identity_posthoc_matrix_labeling_allowed": False,
        "streaming_memory_config": {
            "maximum_incremental_cuda_peak_gib": 16.0,
            "minimum_cuda_free_gib": 12.0,
            "temporal_chunk_frame_count": 4,
            "tile_sample_height": 128,
            "tile_sample_stride_height": 96,
            "tile_sample_stride_width": 96,
            "tile_sample_width": 128,
        },
        "torch_version_policy": (
            "colab_native_runtime_recorded_not_feature_selected"
        ),
    }
    observed_feature = {
        key: value
        for key, value in feature.items()
        if key != "feature_schema_digest"
    }
    if observed_feature != required_feature:
        raise ValueError("construction feature extractor 未完全冻结")
    if feature.get("feature_schema_digest") != (
        construction_feature_schema_digest(feature)
    ):
        raise ValueError("construction feature schema digest 不匹配")

    exposure = config.get("actual_exposure_contract")
    required_exposure = {
        "actual_channel_state_exposure_formula": (
            "delta_sigma_times_actual_velocity_basis_coordinate"
        ),
        "actual_intended_ratio_denominator_epsilon": 1e-15,
        "actual_state_update_exposure_signed": True,
        "stepwise_waveform_frozen_before_execution": True,
        "minimum_actual_intended_waveform_cosine": 0.999,
        "minimum_positive_negative_waveform_symmetry_cosine": 0.999,
        "maximum_positive_negative_amplitude_asymmetry_ratio": 0.05,
        "maximum_cross_channel_leakage_ratio": 0.05,
        "maximum_actual_design_condition_number": 20.0,
        "minimum_actual_design_rank": 6,
        "signed_exposure_required": True,
        "unsigned_norm_only_exposure_allowed": False,
        "stepwise_fallback_required_on_compression_failure": True,
        "lambda_max": 0.12,
        "velocity_norm_ratio_budget": 0.02,
        "flow_energy_budget_ratio": 0.000015,
        "minimum_direction_cosine": 0.999,
        "intended_delta_norm_formula": (
            "min(base_velocity_norm_times_velocity_ratio_times_lambda_times_"
            "abs_temporal_waveform,sqrt_remaining_flow_energy_div_abs_delta_sigma)"
        ),
        "intended_velocity_sign_formula": (
            "probe_state_update_polarity_times_sign_delta_sigma"
        ),
    }
    if not isinstance(exposure, Mapping) or exposure != required_exposure:
        raise ValueError("actual exposure 压缩与预算合同未冻结")

    probe = config.get("impulse_probe")
    if not isinstance(probe, Mapping) or probe != {
        "plan_order": [
            "clean_a",
            "clean_b",
            "six_positive_impulses",
            "six_negative_impulses",
        ],
        "same_prompt": True,
        "same_seed": True,
        "same_initial_noise": True,
        "same_model_revision": True,
        "same_scheduler": True,
        "same_inference_step_count": True,
        "same_video_save_parameters": True,
        "positive_negative_equal_nominal_budget": True,
        "nominal_signed_amplitude": 0.12,
        "clean_repeat_role": "minimum_cost_runtime_noise_reference_only",
        "clean_repeat_is_formal_noise_distribution": False,
    }:
        raise ValueError("14-video impulse triage 合同未冻结")

    execution_identity = config.get("execution_identity")
    if not isinstance(execution_identity, Mapping) or execution_identity != {
        "diffusers_version": "0.35.2",
        "fps": 8,
        "generation_model_id": (
            "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
        ),
        "generation_model_revision": (
            "0fad780a534b6463e45facd96134c9f345acfa5b"
        ),
        "guidance_scale": 5.0,
        "height": 320,
        "initial_noise_binding": "single_torch_generator_manual_seed",
        "negative_prompt_text_sha256": (
            "798a65de2dd61dffee2b6d5229d1167e4c0aa7053948562ae53dff3d1c0d0d11"
        ),
        "num_frames": 33,
        "num_inference_steps": 8,
        "prompt_id": "probe_paper_paper_master_prompt_003",
        "prompt_suite_id": "sstw_nested_paper_prompt_seed_universe_v1",
        "scheduler_id": "wan21_flow_match_euler_discrete_scheduler_shift_3",
        "scheduler_signature": (
            "FlowMatchEulerDiscreteScheduler:"
            "a63b40d76d729371591d03526e14d24359c732866c07f51e4cc5f918f4941d2b"
        ),
        "seed_id": "probe_paper_paper_master_test_seed_01",
        "seed_value": 2201,
        "video_encoder_backend": "imageio_ffmpeg",
        "video_exporter": "diffusers.utils.export_to_video",
        "width": 512,
    }:
        raise ValueError("14-video 单 identity execution binding 未冻结")

    schedule = config.get("flow_schedule_contract")
    required_schedule_keys = {
        "delta_sigma_by_step",
        "flow_phase_by_step",
        "macro_interval_index_by_step",
        "probe_polarity_coordinate",
        "scheduler_library",
        "scheduler_library_version",
        "scheduler_signature",
        "sigma_grid",
        "step_count",
        "step_indices",
        "temporal_waveform_by_macro_interval",
        "waveform_coordinate",
        "waveform_schema_digest",
        "waveform_schema_id",
    }
    if (
        not isinstance(schedule, Mapping)
        or set(schedule) != required_schedule_keys
        or schedule.get("step_count") != CONSTRUCTION_FLOW_STEP_COUNT
        or schedule.get("step_indices")
        != list(range(CONSTRUCTION_FLOW_STEP_COUNT))
        or schedule.get("scheduler_library") != "diffusers"
        or schedule.get("scheduler_library_version") != "0.35.2"
        or schedule.get("scheduler_signature")
        != execution_identity["scheduler_signature"]
        or schedule.get("waveform_schema_id")
        != "wan_flowmatch_shift3_eight_step_three_macro_unit_impulse"
        or schedule.get("waveform_coordinate")
        != (
            "unit_velocity_norm_multiplier_before_frozen_norm_and_energy_budgets"
        )
        or schedule.get("probe_polarity_coordinate")
        != "state_update_delta_z_equals_delta_sigma_times_delta_v"
        or schedule.get("waveform_schema_digest")
        != flow_schedule_waveform_schema_digest(schedule)
    ):
        raise ValueError("8-step Flow waveform schema 未冻结")
    if not np.array_equal(
        np.asarray(schedule["sigma_grid"], dtype=np.float64),
        np.asarray(CONSTRUCTION_SIGMA_GRID, dtype=np.float64),
    ):
        raise ValueError("8-step Flow sigma grid 未冻结")
    if not np.array_equal(
        np.asarray(schedule["delta_sigma_by_step"], dtype=np.float64),
        np.asarray(CONSTRUCTION_DELTA_SIGMA, dtype=np.float64),
    ):
        raise ValueError("8-step Flow delta sigma 未冻结")
    if not np.array_equal(
        np.asarray(schedule["flow_phase_by_step"], dtype=np.float64),
        np.asarray(CONSTRUCTION_FLOW_PHASES, dtype=np.float64),
    ):
        raise ValueError("8-step Flow phase 未冻结")
    if tuple(schedule["macro_interval_index_by_step"]) != (
        CONSTRUCTION_MACRO_INTERVAL_INDICES
    ):
        raise ValueError("8-step macro interval assignment 未冻结")
    if tuple(
        tuple(float(value) for value in waveform)
        for waveform in schedule["temporal_waveform_by_macro_interval"]
    ) != CONSTRUCTION_TEMPORAL_WAVEFORMS:
        raise ValueError("三宏区间 temporal waveform 未冻结")

    adapter = config.get("runtime_adapter_contract")
    required_adapter = {
        "actual_delta_recomputed_from_float32_constrained_minus_base": True,
        "actual_velocity_basis_coordinates_recomputed_from_actual_delta": True,
        "adapter_schema_digest": (
            "b35d8a9b4f268ee13e0a5686c320acd6e9db0f3a47f0c6a26481cfe2d40513ee"
        ),
        "adapter_schema_id": CONSTRUCTION_RUNTIME_ADAPTER_SCHEMA_ID,
        "caller_guard_boolean_trusted_without_recompute": False,
        "cumulative_energy_recomputed_from_delta_sigma_and_actual_delta_norm": True,
        "reference_base_velocity_norm_required": True,
        "remaining_flow_energy_required": True,
    }
    if (
        not isinstance(adapter, Mapping)
        or adapter != required_adapter
        or adapter.get("adapter_schema_digest")
        != runtime_adapter_schema_digest(adapter)
    ):
        raise ValueError("actual delta runtime adapter 合同未冻结")

    checkpoints = config.get("transfer_checkpoints")
    if checkpoints != [
        "T_latent",
        "T_decoded",
        "T_saved_video",
        "T_reencoded",
        "T_output_feature",
        "T_replay_diagnostic",
    ]:
        raise ValueError("端到端传递 checkpoint 合同未冻结")
    if config.get("primary_gate_checkpoint") != "T_output_feature":
        raise ValueError("primary gate 必须仅由 output feature 支持")
    checkpoint_contract = config.get("transfer_checkpoint_contract")
    if not isinstance(checkpoint_contract, Mapping) or checkpoint_contract != {
        "T_latent": {
            "dimension": 7,
            "normalization": "none",
            "representation": (
                "six_float32_basis_projections_plus_float32_l2_norm"
            ),
        },
        "T_decoded": {
            "dimension": 48,
            "normalization": "none",
            "representation": (
                "float64_accumulated_time_global_rgb_float32_"
                "equal_area_4_by_4_cell_means_before_save"
            ),
        },
        "T_saved_video": {
            "dimension": 48,
            "normalization": "rgb24_divide_255",
            "representation": (
                "float64_accumulated_time_global_decoded_rgb24_"
                "equal_area_4_by_4_cell_means"
            ),
        },
        "T_reencoded": {
            "dimension": 256,
            "normalization": "none",
            "representation": (
                "float64_accumulated_normalized_wan_latent_"
                "time_global_4_by_4_cell_means"
            ),
        },
        "T_output_feature": {
            "dimension": 256,
            "normalization": "l2_with_frozen_zero_rejection",
            "representation": "phi_construction",
        },
        "T_replay_diagnostic": {
            "dimension": 6,
            "normalization": "none",
            "primary_gate_support_allowed": False,
            "representation": (
                "optional_prompt_conditioned_six_basis_projection_summary"
            ),
            "required": False,
        },
    }:
        raise ValueError("transfer checkpoint representation 合同未冻结")

    model = config.get("transfer_model")
    if not isinstance(model, Mapping) or model != {
        "equation": "Y_equals_mu0_1T_plus_T_A_actual_plus_E",
        "clean_intercept_source": "predeclared_clean_a_clean_b_mean",
        "primary_column_estimator": (
            "positive_negative_center_difference_div_"
            "actual_signed_amplitude_difference"
        ),
        "leakage_aware_estimator": (
            "clean_intercept_actual_design_matrix_pseudoinverse"
        ),
        "public_content_as_transfer_column_allowed": False,
    }:
        raise ValueError("clean-intercept transfer model 未冻结")

    gate_a = config.get("gate_a_sample_internal_causal_observability")
    if not isinstance(gate_a, Mapping) or gate_a != {
        "absolute_response_formula": (
            "minimum_l2_norm_of_positive_negative_center_response_columns"
        ),
        "antisymmetry_cosine_formula": (
            "minimum_cosine_positive_centered_vs_negative_negative_centered"
        ),
        "antisymmetry_residual_formula": (
            "maximum_l2_positive_plus_negative_minus_two_mu0_"
            "div_sum_centered_norms"
        ),
        "clean_distance_formula": "l2_clean_a_minus_clean_b",
        "clean_noise_floor_formula": (
            "max(clean_distance_div_sqrt_2,minimum_finite_noise_floor)"
        ),
        "clean_noise_floor_source": (
            "clean_a_clean_b_output_feature_difference"
        ),
        "effective_rank_formula": (
            "count_response_singular_values_at_least_max("
            "absolute_singular_tolerance,"
            "minimum_noise_normalized_singular_value_times_noise_floor)"
        ),
        "high_dimensional_full_rank_alone_sufficient": False,
        "maximum_clean_distance_for_numerical_resolution": 1e-12,
        "maximum_output_transfer_condition_number": 20.0,
        "maximum_positive_negative_antisymmetry_residual_ratio": 0.25,
        "minimum_absolute_output_response": 1e-4,
        "minimum_absolute_singular_value": 1e-6,
        "minimum_effective_rank": 6,
        "minimum_finite_noise_floor": 1e-6,
        "minimum_noise_normalized_singular_value": 2.0,
        "minimum_output_feature_snr": 3.0,
        "minimum_positive_negative_antisymmetry_cosine": 0.9,
        "noise_normalized_singular_formula": (
            "minimum_response_singular_value_div_finite_noise_floor"
        ),
        "output_feature_snr_formula": (
            "minimum_center_response_l2_norm_div_finite_noise_floor"
        ),
        "output_transfer_condition_formula": (
            "condition_number_of_actual_amplitude_normalized_"
            "center_difference_transfer"
        ),
        "primary_required_transfer_checkpoints": [
            "T_latent",
            "T_decoded",
            "T_saved_video",
            "T_reencoded",
            "T_output_feature",
        ],
        "primary_checkpoint": "T_output_feature",
        "replay_diagnostic_required": False,
        "zero_clean_distance_auto_pass_allowed": False,
        "stop_current_carrier_or_feature_on_failure": True,
    }:
        raise ValueError("Gate A 样本内因果可观测合同未冻结")

    gate_b = config.get("gate_b_cross_identity_identifiability")
    if not isinstance(gate_b, Mapping) or gate_b != {
        "calibration_identity_defines_public_coordinates": True,
        "confirmation_identity_apply_only": True,
        "confirmation_identity_excluded_from_observer_calibration": True,
        "construction_confirmation_only_not_generalization": True,
        "maximum_heldout_transfer_prediction_error_ratio": 0.5,
        "maximum_normalized_gram_difference": 0.25,
        "maximum_principal_angle_degrees": 30.0,
        "minimum_stage_prediction_fraction": 5 / 6,
        "minimum_sign_prediction_fraction": 5 / 6,
        "owner_wrong_output_selectivity_required": True,
        "second_independent_identity_required": True,
        "test_identity_adaptive_alignment_allowed": False,
        "test_identity_procrustes_allowed": False,
        "test_identity_sign_flip_allowed": False,
    }:
        raise ValueError("Gate B 跨 identity 可识别合同未冻结")

    key_selectivity = config.get("key_selectivity_construction")
    if not isinstance(key_selectivity, Mapping) or key_selectivity != {
        "adaptive_wrong_key_selection_allowed": False,
        "first_impulse_triage_inclusion_allowed": False,
        "owner_wrong_output_feature_transfer_required": True,
        "representative_stage_channel_pairs": [[0, 0], [1, 0], [2, 0]],
        "same_budget_as_impulse_probe": True,
        "total_norm_or_quality_only_selectivity_allowed": False,
        "wrong_key_candidate_index": 0,
        "wrong_key_count": 1,
        "wrong_key_derivation_id": (
            "observer_synchronized_construction_wrong_key_index_0"
        ),
        "wrong_key_frozen_before_execution": True,
    }:
        raise ValueError("key-selectivity construction 合同未冻结")

    gate_c = config.get("gate_c_composite_trajectory_order_identifiability")
    if not isinstance(gate_c, Mapping) or gate_c != {
        "dynamics_contribution_claim_allowed": False,
        "minimum_order_difference_clean_noise_ratio": 3.0,
        "ordered_composite_required": True,
        "same_energy_permuted_composite_required": True,
        "single_impulse_superposition_check_required": True,
        "total_norm_or_quality_only_explanation_allowed": False,
    }:
        raise ValueError("Gate C 组合轨迹与阶段顺序合同未冻结")

    future = config.get("future_observer_boundary")
    if not isinstance(future, Mapping) or future != {
        "batch_observer_design_requires_all_gates": [
            "sample_internal_causal_observability",
            "cross_identity_identifiability",
            "composite_trajectory_order_identifiability",
        ],
        "formal_llr_name_allowed": False,
        "initial_score_name": "prediction_error_or_matched_dynamic_score",
        "required_future_ablations": [
            "complete_observer",
            "independent_interval_template",
            "static_endpoint",
        ],
        "state_dynamics_f_k_g_k_construction_allowed": False,
    }:
        raise ValueError("future observer 授权边界未冻结")

    gates = config.get("authorization_state_machine")
    required_gates = {
        "current_state": "construction_contract_local_audit",
        "states": [
            "construction_contract_local_audit",
            "independent_readonly_audit",
            "commit_push_authorization_pending",
            "impulse_triage_execution_authorization_pending",
            "sample_internal_causal_observability_gate",
            "cross_identity_construction_confirmation_pending",
            "key_selectivity_construction_pending",
            "composite_trajectory_order_identifiability_pending",
            "state_dynamics_and_batch_observer_design_pending",
        ],
        "impulse_triage_execution_allowed": False,
        "cross_identity_confirmation_allowed": False,
        "key_selectivity_construction_allowed": False,
        "composite_order_gate_execution_allowed": False,
        "state_dynamics_design_allowed": False,
        "batch_observer_design_allowed": False,
    }
    if not isinstance(gates, Mapping) or gates != required_gates:
        raise ValueError("construction 授权状态机未冻结")


def build_stage_selector_blocks() -> tuple[np.ndarray, ...]:
    """Return E_j selectors for B_K,j = U_K E_j, never U_K R_j."""

    blocks: list[np.ndarray] = []
    for stage_index in range(FLOW_MACRO_INTERVAL_COUNT):
        selector = np.zeros(
            (STAGE_BASIS_RANK, WATERMARK_STATE_DIMENSION),
            dtype=np.float64,
        )
        start = stage_index * WATERMARK_STATE_DIMENSION
        selector[start, 0] = 1.0
        selector[start + 1, 1] = 1.0
        blocks.append(selector)
    return tuple(blocks)


def build_impulse_triage_plan(
    config: Mapping[str, Any],
) -> tuple[ImpulseProbePlanRecord, ...]:
    validate_impulse_observability_config(config)
    amplitude = float(config["impulse_probe"]["nominal_signed_amplitude"])
    plan: list[ImpulseProbePlanRecord] = [
        ImpulseProbePlanRecord(
            probe_id="clean_a",
            probe_role="clean_runtime_repeat",
            stage_index=None,
            stage_name=None,
            channel_index=None,
            polarity=0,
            nominal_signed_amplitude=0.0,
        ),
        ImpulseProbePlanRecord(
            probe_id="clean_b",
            probe_role="clean_runtime_repeat",
            stage_index=None,
            stage_name=None,
            channel_index=None,
            polarity=0,
            nominal_signed_amplitude=0.0,
        ),
    ]
    for polarity_name, polarity in zip(
        POLARITY_NAMES,
        (1, -1),
        strict=True,
    ):
        for stage_index, stage_name in enumerate(STAGE_NAMES):
            for channel_index in range(WATERMARK_STATE_DIMENSION):
                plan.append(
                    ImpulseProbePlanRecord(
                        probe_id=(
                            f"{polarity_name}_{stage_name}_"
                            f"channel_{channel_index}"
                        ),
                        probe_role="signed_interval_impulse",
                        stage_index=stage_index,
                        stage_name=stage_name,
                        channel_index=channel_index,
                        polarity=polarity,
                        nominal_signed_amplitude=polarity * amplitude,
                    )
                )
    if len(plan) != IMPULSE_TRIAGE_VIDEO_COUNT:
        raise AssertionError("14-video construction plan 组装失败")
    return tuple(plan)


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    denominator = float(
        np.linalg.norm(left_array) * np.linalg.norm(right_array)
    )
    if denominator <= 0.0:
        return float("nan")
    return float(left_array @ right_array / denominator)


def _validate_trace_shape(trace: ActualImpulseExposureTrace) -> None:
    step_count = len(trace.intended_signed_exposure_by_step)
    vectors = (
        trace.step_indices,
        trace.flow_phase_by_step,
        trace.delta_sigma_by_step,
        trace.macro_interval_index_by_step,
        trace.intended_velocity_waveform_by_step,
        trace.reference_base_velocity_norm_by_step,
        trace.remaining_control_energy_before_step_by_step,
        trace.reference_energy_increment_by_step,
        trace.reference_cumulative_energy_by_step,
        trace.intended_delta_norm_by_step,
        trace.actual_velocity_basis_coordinate_by_step,
        trace.actual_channel_velocity_coordinate_by_step,
        trace.actual_signed_exposure_by_step,
        trace.actual_channel_exposure_by_step,
        trace.delta_norm_by_step,
        trace.projection_scale_by_step,
        trace.cumulative_energy_by_step,
        trace.direction_cosine_by_step,
        trace.norm_guard_passed_by_step,
        trace.energy_guard_passed_by_step,
    )
    if (
        step_count != CONSTRUCTION_FLOW_STEP_COUNT
        or any(len(values) != step_count for values in vectors)
    ):
        raise ValueError(
            f"{trace.probe_id} stepwise exposure 必须精确覆盖8步"
        )
    if len(trace.actual_exposure_vector) != STAGE_BASIS_RANK:
        raise ValueError(f"{trace.probe_id} actual exposure 必须为六维")
    if any(
        len(values) != STAGE_BASIS_RANK
        for values in trace.actual_channel_exposure_by_step
    ):
        raise ValueError(
            f"{trace.probe_id} stepwise channel exposure 必须逐步为六维"
        )
    if any(
        len(values) != STAGE_BASIS_RANK
        for values in trace.actual_channel_velocity_coordinate_by_step
    ):
        raise ValueError(
            f"{trace.probe_id} stepwise velocity coordinate 必须逐步为六维"
        )
    numeric = (
        trace.flow_phase_by_step
        + trace.delta_sigma_by_step
        + trace.intended_velocity_waveform_by_step
        + trace.reference_base_velocity_norm_by_step
        + trace.remaining_control_energy_before_step_by_step
        + trace.reference_energy_increment_by_step
        + trace.reference_cumulative_energy_by_step
        + trace.intended_delta_norm_by_step
        + trace.actual_velocity_basis_coordinate_by_step
        + trace.intended_signed_exposure_by_step
        + trace.actual_signed_exposure_by_step
        + trace.actual_exposure_vector
        + tuple(
            value
            for values in trace.actual_channel_exposure_by_step
            for value in values
        )
        + tuple(
            value
            for values in trace.actual_channel_velocity_coordinate_by_step
            for value in values
        )
        + trace.delta_norm_by_step
        + trace.projection_scale_by_step
        + trace.cumulative_energy_by_step
        + trace.direction_cosine_by_step
    )
    if any(not math.isfinite(float(value)) for value in numeric):
        raise ValueError(f"{trace.probe_id} exposure 含非有限值")
    if any(value < 0.0 for value in trace.delta_norm_by_step):
        raise ValueError(f"{trace.probe_id} delta norm 非法")
    if any(
        value < 0.0 or value > 1.0
        for value in trace.projection_scale_by_step
    ):
        raise ValueError(f"{trace.probe_id} projection scale 非法")
    if any(value < 0.0 for value in trace.cumulative_energy_by_step):
        raise ValueError(f"{trace.probe_id} cumulative energy 非法")
    if tuple(sorted(trace.cumulative_energy_by_step)) != (
        trace.cumulative_energy_by_step
    ):
        raise ValueError(f"{trace.probe_id} cumulative energy 非单调")
    if not all(trace.norm_guard_passed_by_step):
        raise ValueError(f"{trace.probe_id} norm guard 未全部通过")
    if not all(trace.energy_guard_passed_by_step):
        raise ValueError(f"{trace.probe_id} energy guard 未全部通过")
    if any(
        not -1.0 <= float(value) <= 1.0
        for value in trace.direction_cosine_by_step
    ):
        raise ValueError(f"{trace.probe_id} direction cosine 必须位于[-1,1]")
    for name, digest in (
        ("waveform", trace.waveform_schema_digest),
        ("runtime adapter", trace.runtime_adapter_schema_digest),
        ("basis", trace.basis_digest),
    ):
        if (
            len(str(digest)) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"{trace.probe_id} {name} digest 非法")


def _strict_scalar_close(
    observed: float,
    expected: float,
    *,
    absolute_tolerance: float = 1e-12,
) -> bool:
    return math.isclose(
        float(observed),
        float(expected),
        rel_tol=1e-9,
        abs_tol=absolute_tolerance,
    )


def _validate_trace_schedule_and_budget(
    config: Mapping[str, Any],
    trace: ActualImpulseExposureTrace,
) -> None:
    schedule = config["flow_schedule_contract"]
    exposure = config["actual_exposure_contract"]
    adapter = config["runtime_adapter_contract"]
    if trace.step_indices != tuple(schedule["step_indices"]):
        raise ValueError(f"{trace.probe_id} step order 与冻结 schedule 不一致")
    expected_sequences = (
        (
            trace.flow_phase_by_step,
            tuple(schedule["flow_phase_by_step"]),
            "flow phase",
        ),
        (
            trace.delta_sigma_by_step,
            tuple(schedule["delta_sigma_by_step"]),
            "delta sigma",
        ),
    )
    for observed, expected, name in expected_sequences:
        if any(
            not _strict_scalar_close(left, right)
            for left, right in zip(observed, expected, strict=True)
        ):
            raise ValueError(
                f"{trace.probe_id} {name} 与冻结 schedule 不一致"
            )
    if trace.macro_interval_index_by_step != tuple(
        schedule["macro_interval_index_by_step"]
    ):
        raise ValueError(
            f"{trace.probe_id} macro interval assignment 不一致"
        )
    expected_waveform = tuple(
        float(value)
        for value in schedule["temporal_waveform_by_macro_interval"][
            trace.stage_index
        ]
    )
    if trace.intended_velocity_waveform_by_step != expected_waveform:
        raise ValueError(
            f"{trace.probe_id} temporal waveform 与 config 不一致"
        )
    if trace.waveform_schema_digest != schedule["waveform_schema_digest"]:
        raise ValueError(
            f"{trace.probe_id} waveform digest 未绑定 config"
        )
    if (
        trace.runtime_adapter_schema_digest
        != adapter["adapter_schema_digest"]
    ):
        raise ValueError(
            f"{trace.probe_id} runtime adapter digest 未绑定 config"
        )

    previous_reference_energy = 0.0
    previous_control_energy = 0.0
    target_coordinate = (
        trace.stage_index * WATERMARK_STATE_DIMENSION
        + trace.channel_index
    )
    for step_index in range(CONSTRUCTION_FLOW_STEP_COUNT):
        interval = trace.delta_sigma_by_step[step_index]
        control = compute_intended_impulse_control(
            probe_state_update_polarity=trace.polarity,
            temporal_waveform=expected_waveform[step_index],
            delta_sigma=interval,
            base_velocity_norm=(
                trace.reference_base_velocity_norm_by_step[step_index]
            ),
            cumulative_control_energy=previous_control_energy,
            cumulative_reference_energy=previous_reference_energy,
            remaining_step_count=CONSTRUCTION_FLOW_STEP_COUNT - step_index,
            lambda_max=float(exposure["lambda_max"]),
            velocity_norm_ratio_budget=float(
                exposure["velocity_norm_ratio_budget"]
            ),
            flow_energy_budget_ratio=float(
                exposure["flow_energy_budget_ratio"]
            ),
        )
        observed_reference_increment = (
            trace.reference_energy_increment_by_step[step_index]
        )
        if not _strict_scalar_close(
            observed_reference_increment,
            control.reference_energy_increment,
        ):
            raise ValueError(
                f"{trace.probe_id} step {step_index} reference energy 未重算"
            )
        expected_reference_cumulative = (
            previous_reference_energy + control.reference_energy_increment
        )
        if not _strict_scalar_close(
            trace.reference_cumulative_energy_by_step[step_index],
            expected_reference_cumulative,
        ):
            raise ValueError(
                f"{trace.probe_id} step {step_index} reference cumulative energy 不一致"
            )
        if not _strict_scalar_close(
            trace.remaining_control_energy_before_step_by_step[step_index],
            control.remaining_control_energy,
        ):
            raise ValueError(
                f"{trace.probe_id} step {step_index} remaining energy 未重算"
            )
        if not _strict_scalar_close(
            trace.intended_delta_norm_by_step[step_index],
            control.intended_delta_norm,
        ):
            raise ValueError(
                f"{trace.probe_id} step {step_index} intended delta norm 不一致"
            )
        if not _strict_scalar_close(
            trace.intended_signed_exposure_by_step[step_index],
            control.signed_state_update_exposure,
        ):
            raise ValueError(
                f"{trace.probe_id} step {step_index} intended exposure 未按 delta sigma 重算"
            )

        actual_norm = float(trace.delta_norm_by_step[step_index])
        actual_velocity_coordinates = np.asarray(
            trace.actual_channel_velocity_coordinate_by_step[step_index],
            dtype=np.float64,
        )
        actual_target_velocity = float(
            actual_velocity_coordinates[target_coordinate]
        )
        if not _strict_scalar_close(
            trace.actual_velocity_basis_coordinate_by_step[step_index],
            actual_target_velocity,
        ):
            raise ValueError(
                f"{trace.probe_id} step {step_index} target velocity coordinate 不一致"
            )
        recomputed_state_exposure = (
            float(interval) * actual_velocity_coordinates
        )
        if not np.allclose(
            recomputed_state_exposure,
            np.asarray(
                trace.actual_channel_exposure_by_step[step_index],
                dtype=np.float64,
            ),
            rtol=1e-9,
            atol=1e-12,
        ):
            raise ValueError(
                f"{trace.probe_id} step {step_index} state exposure 未按 delta sigma 重算"
            )
        if not _strict_scalar_close(
            trace.actual_signed_exposure_by_step[step_index],
            recomputed_state_exposure[target_coordinate],
        ):
            raise ValueError(
                f"{trace.probe_id} step {step_index} signed exposure 不一致"
            )
        expected_increment = float(interval) ** 2 * actual_norm**2
        expected_control_cumulative = (
            previous_control_energy + expected_increment
        )
        if not _strict_scalar_close(
            trace.cumulative_energy_by_step[step_index],
            expected_control_cumulative,
        ):
            raise ValueError(
                f"{trace.probe_id} step {step_index} control cumulative energy 未重算"
            )
        actual_active = control.intended_delta_norm > 1e-15
        if actual_active:
            if (
                actual_norm <= 0.0
                or actual_norm
                > control.intended_delta_norm + 1e-12
                or expected_increment
                > control.remaining_control_energy + 1e-12
            ):
                raise ValueError(
                    f"{trace.probe_id} step {step_index} actual norm/energy budget 未通过"
                )
            expected_scale = actual_norm / control.intended_delta_norm
            if not _strict_scalar_close(
                trace.projection_scale_by_step[step_index],
                expected_scale,
            ):
                raise ValueError(
                    f"{trace.probe_id} step {step_index} projection scale 不一致"
                )
            desired_velocity_sign = (
                trace.polarity * (1 if interval > 0.0 else -1)
            )
            recomputed_cosine = (
                desired_velocity_sign
                * actual_target_velocity
                / actual_norm
            )
            if not _strict_scalar_close(
                trace.direction_cosine_by_step[step_index],
                recomputed_cosine,
            ):
                raise ValueError(
                    f"{trace.probe_id} step {step_index} direction cosine 未重算"
                )
            direction_ready = (
                math.isfinite(recomputed_cosine)
                and -1.0 <= recomputed_cosine <= 1.0
                and recomputed_cosine
                >= float(exposure["minimum_direction_cosine"])
            )
            if not direction_ready:
                raise ValueError(
                    f"{trace.probe_id} step {step_index} direction guard 未通过"
                )
        else:
            if (
                actual_norm != 0.0
                or np.any(actual_velocity_coordinates != 0.0)
                or trace.projection_scale_by_step[step_index] != 0.0
            ):
                raise ValueError(
                    f"{trace.probe_id} step {step_index} inactive step 非 no-op"
                )
        norm_guard = actual_norm <= control.intended_delta_norm + 1e-12
        energy_guard = (
            expected_increment
            <= control.remaining_control_energy + 1e-12
        )
        if (
            trace.norm_guard_passed_by_step[step_index] is not norm_guard
            or trace.energy_guard_passed_by_step[step_index]
            is not energy_guard
        ):
            raise ValueError(
                f"{trace.probe_id} step {step_index} caller guard bool 与重算不一致"
            )
        previous_reference_energy = expected_reference_cumulative
        previous_control_energy = expected_control_cumulative


def assemble_actual_design_matrix(
    config: Mapping[str, Any],
    traces: Sequence[ActualImpulseExposureTrace],
) -> ActualDesignMatrix:
    """Compress stepwise traces only after every frozen sufficiency gate."""

    validate_impulse_observability_config(config)
    plan = build_impulse_triage_plan(config)
    expected = {
        record.probe_id: record
        for record in plan
        if record.probe_role == "signed_interval_impulse"
    }
    observed = {trace.probe_id: trace for trace in traces}
    if set(observed) != set(expected) or len(observed) != len(traces):
        raise ValueError("actual exposure trace 必须精确覆盖12个 signed impulses")

    exposure = config["actual_exposure_contract"]
    minimum_waveform_cosine = float(
        exposure["minimum_actual_intended_waveform_cosine"]
    )
    minimum_symmetry = float(
        exposure["minimum_positive_negative_waveform_symmetry_cosine"]
    )
    maximum_asymmetry = float(
        exposure["maximum_positive_negative_amplitude_asymmetry_ratio"]
    )
    maximum_leakage = float(
        exposure["maximum_cross_channel_leakage_ratio"]
    )
    minimum_direction_cosine = float(exposure["minimum_direction_cosine"])
    basis_digest: str | None = None
    waveform_cosines: dict[str, float] = {}
    ratios: dict[str, tuple[float, ...]] = {}
    leakage_by_probe: dict[str, float] = {}
    for probe_id, trace in observed.items():
        _validate_trace_shape(trace)
        _validate_trace_schedule_and_budget(config, trace)
        plan_record = expected[probe_id]
        if (
            trace.stage_index != plan_record.stage_index
            or trace.channel_index != plan_record.channel_index
            or trace.polarity != plan_record.polarity
        ):
            raise ValueError(f"{probe_id} 与冻结 plan identity 不一致")
        if basis_digest is None:
            basis_digest = trace.basis_digest
        if trace.basis_digest != basis_digest:
            raise ValueError("12个 impulse 必须使用同一 construction basis digest")
        cosine = _cosine(
            trace.intended_signed_exposure_by_step,
            trace.actual_signed_exposure_by_step,
        )
        waveform_cosines[probe_id] = cosine
        if not math.isfinite(cosine) or cosine < minimum_waveform_cosine:
            raise ValueError(
                f"{probe_id} actual/intended waveform cosine 未通过"
            )
        step_ratios: list[float] = []
        for intended, actual in zip(
            trace.intended_signed_exposure_by_step,
            trace.actual_signed_exposure_by_step,
            strict=True,
        ):
            if abs(intended) <= 1e-15:
                if abs(actual) > 1e-15:
                    raise ValueError(f"{probe_id} inactive step 出现 actual exposure")
                step_ratios.append(0.0)
            else:
                step_ratios.append(float(actual / intended))
        ratios[probe_id] = tuple(step_ratios)
        coordinate = (
            trace.stage_index * WATERMARK_STATE_DIMENSION
            + trace.channel_index
        )
        step_leakages: list[float] = []
        for step_index, (
            signed_exposure,
            channel_exposure,
            delta_norm,
            projection_scale,
            direction_cosine,
        ) in enumerate(
            zip(
                trace.actual_signed_exposure_by_step,
                trace.actual_channel_exposure_by_step,
                trace.delta_norm_by_step,
                trace.projection_scale_by_step,
                trace.direction_cosine_by_step,
                strict=True,
            )
        ):
            if not math.isclose(
                float(channel_exposure[coordinate]),
                float(signed_exposure),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"{probe_id} step {step_index} signed exposure "
                    "与 channel exposure 不一致"
                )
            off_target = float(
                np.linalg.norm(
                    np.delete(
                        np.asarray(channel_exposure, dtype=np.float64),
                        coordinate,
                    )
                )
            )
            if abs(signed_exposure) <= 1e-15:
                step_leakage = 0.0 if off_target <= 1e-15 else float("inf")
            else:
                step_leakage = off_target / abs(float(signed_exposure))
            step_leakages.append(step_leakage)
            if (
                not math.isfinite(step_leakage)
                or step_leakage > maximum_leakage
            ):
                raise ValueError(
                    f"{probe_id} step {step_index} cross-channel leakage 未通过"
                )
            if delta_norm > 0.0 and (
                projection_scale <= 0.0
                or direction_cosine < minimum_direction_cosine
            ):
                raise ValueError(
                    f"{probe_id} step {step_index} actual delta guard 未通过"
                )
        aggregated = np.asarray(
            trace.actual_channel_exposure_by_step,
            dtype=np.float64,
        ).sum(axis=0)
        if not np.allclose(
            aggregated,
            np.asarray(trace.actual_exposure_vector, dtype=np.float64),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"{probe_id} stepwise exposure 聚合不一致")
        signed_target = float(trace.actual_exposure_vector[coordinate])
        if signed_target * trace.polarity <= 0.0:
            raise ValueError(f"{probe_id} actual exposure 符号与 polarity 不一致")
        target = abs(signed_target)
        leakage = float(
            np.linalg.norm(
                np.delete(
                    np.asarray(trace.actual_exposure_vector, dtype=np.float64),
                    coordinate,
                )
            )
            / max(target, 1e-15)
        )
        leakage_by_probe[probe_id] = max([leakage, *step_leakages])
        if not math.isfinite(leakage) or leakage > maximum_leakage:
            raise ValueError(f"{probe_id} cross-channel leakage 未通过")

    symmetry_by_channel: dict[str, float] = {}
    asymmetry_by_channel: dict[str, float] = {}
    for stage_index, stage_name in enumerate(STAGE_NAMES):
        for channel_index in range(WATERMARK_STATE_DIMENSION):
            coordinate_name = f"{stage_name}_channel_{channel_index}"
            positive = observed[f"positive_{coordinate_name}"]
            negative = observed[f"negative_{coordinate_name}"]
            symmetry = _cosine(
                positive.actual_signed_exposure_by_step,
                tuple(
                    -value
                    for value in negative.actual_signed_exposure_by_step
                ),
            )
            symmetry_by_channel[coordinate_name] = symmetry
            if not math.isfinite(symmetry) or symmetry < minimum_symmetry:
                raise ValueError(
                    f"{coordinate_name} positive/negative waveform symmetry 未通过"
                )
            positive_norm = float(
                np.linalg.norm(positive.actual_exposure_vector)
            )
            negative_norm = float(
                np.linalg.norm(negative.actual_exposure_vector)
            )
            asymmetry = abs(positive_norm - negative_norm) / max(
                positive_norm,
                negative_norm,
                1e-15,
            )
            asymmetry_by_channel[coordinate_name] = asymmetry
            if asymmetry > maximum_asymmetry:
                raise ValueError(
                    f"{coordinate_name} positive/negative amplitude symmetry 未通过"
                )

    columns: list[np.ndarray] = []
    for plan_record in plan:
        if plan_record.probe_role == "clean_runtime_repeat":
            columns.append(np.zeros(STAGE_BASIS_RANK, dtype=np.float64))
        else:
            columns.append(
                np.asarray(
                    observed[plan_record.probe_id].actual_exposure_vector,
                    dtype=np.float64,
                )
            )
    matrix = np.stack(columns, axis=1)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    rank = int(np.linalg.matrix_rank(matrix))
    condition = (
        float("inf")
        if singular_values[-1] <= 1e-15
        else float(singular_values[0] / singular_values[-1])
    )
    if rank < int(exposure["minimum_actual_design_rank"]):
        raise ValueError("A_actual rank 未通过")
    if (
        not math.isfinite(condition)
        or condition
        > float(exposure["maximum_actual_design_condition_number"])
    ):
        raise ValueError("A_actual condition number 未通过")
    return ActualDesignMatrix(
        probe_ids=tuple(record.probe_id for record in plan),
        values=matrix,
        rank=rank,
        condition_number=condition,
        waveform_cosine_by_probe=waveform_cosines,
        intended_actual_ratio_by_probe=ratios,
        positive_negative_symmetry_by_channel=symmetry_by_channel,
        positive_negative_amplitude_asymmetry_by_channel=(
            asymmetry_by_channel
        ),
        cross_channel_leakage_by_probe=leakage_by_probe,
    )


def validate_construction_output_features(
    config: Mapping[str, Any],
    output_features: Sequence[Sequence[float]],
    *,
    feature_schema_digest: str,
    probe_ids: Sequence[str],
    row_binding_digests: Sequence[str],
) -> ValidatedConstructionFeatures:
    """Bind 14x256 output features to the frozen public L2 schema."""

    validate_impulse_observability_config(config)
    feature = config["construction_feature_schema"]
    if feature_schema_digest != feature["feature_schema_digest"]:
        raise ValueError("output feature schema digest 未绑定 config")
    observed_probe_ids = tuple(str(value) for value in probe_ids)
    expected_probe_ids = tuple(
        record.probe_id for record in build_impulse_triage_plan(config)
    )
    if (
        len(observed_probe_ids) != IMPULSE_TRIAGE_VIDEO_COUNT
        or len(set(observed_probe_ids)) != len(observed_probe_ids)
        or observed_probe_ids != expected_probe_ids
    ):
        raise ValueError(
            "output feature probe_ids 必须无重复且精确匹配冻结14-video顺序"
        )
    values = np.asarray(output_features, dtype=np.float64)
    if values.shape != (
        IMPULSE_TRIAGE_VIDEO_COUNT,
        CONSTRUCTION_FEATURE_OUTPUT_DIMENSION,
    ):
        raise ValueError("output features 必须按14-video×256维冻结 schema 排列")
    if not np.all(np.isfinite(values)):
        raise ValueError("output features 含非有限值")
    norms = np.linalg.norm(values, axis=1)
    epsilon = float(feature["output_normalization_epsilon"])
    tolerance = float(feature["output_normalization_norm_tolerance"])
    if np.any(norms <= epsilon):
        raise ValueError("output feature L2 zero rejection 未通过")
    if not np.allclose(norms, 1.0, rtol=0.0, atol=tolerance):
        raise ValueError("output feature 未按冻结 L2 schema 归一化")
    observed_row_bindings = tuple(
        str(value) for value in row_binding_digests
    )
    expected_row_bindings = tuple(
        construction_feature_row_binding_digest(
            probe_id=probe_id,
            feature_schema_digest=feature_schema_digest,
            feature_values=values[row_index],
        )
        for row_index, probe_id in enumerate(observed_probe_ids)
    )
    if (
        len(observed_row_bindings) != IMPULSE_TRIAGE_VIDEO_COUNT
        or observed_row_bindings != expected_row_bindings
    ):
        raise ValueError(
            "output feature row binding digest 与 probe_id/value 不一致"
        )
    values = np.array(values, dtype=np.float64, copy=True, order="C")
    values.setflags(write=False)
    return ValidatedConstructionFeatures(
        values=values,
        feature_schema_digest=feature_schema_digest,
        probe_ids=observed_probe_ids,
        row_binding_digests=observed_row_bindings,
    )


def _validate_actual_design_for_estimation(
    config: Mapping[str, Any],
    actual_design: ActualDesignMatrix,
) -> np.ndarray:
    expected_probe_ids = tuple(
        record.probe_id for record in build_impulse_triage_plan(config)
    )
    values = np.asarray(actual_design.values, dtype=np.float64)
    if (
        values.shape
        != (STAGE_BASIS_RANK, IMPULSE_TRIAGE_VIDEO_COUNT)
        or actual_design.probe_ids != expected_probe_ids
        or actual_design.compression_allowed is not True
        or not np.all(np.isfinite(values))
        or not np.array_equal(
            values[:, :IMPULSE_CLEAN_REPEAT_COUNT],
            np.zeros(
                (STAGE_BASIS_RANK, IMPULSE_CLEAN_REPEAT_COUNT),
                dtype=np.float64,
            ),
        )
    ):
        raise ValueError("A_actual identity/shape/clean/compression 不完整")
    singular_values = np.linalg.svd(values, compute_uv=False)
    recomputed_rank = int(np.linalg.matrix_rank(values))
    recomputed_condition = (
        float("inf")
        if singular_values[-1] <= 1e-15
        else float(singular_values[0] / singular_values[-1])
    )
    exposure = config["actual_exposure_contract"]
    if (
        recomputed_rank != actual_design.rank
        or recomputed_rank
        < int(exposure["minimum_actual_design_rank"])
        or not math.isfinite(recomputed_condition)
        or not math.isclose(
            recomputed_condition,
            float(actual_design.condition_number),
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
        or recomputed_condition
        > float(exposure["maximum_actual_design_condition_number"])
    ):
        raise ValueError("A_actual rank/condition metadata 与重算不一致")
    return values


def estimate_construction_transfer(
    config: Mapping[str, Any],
    output_features: ValidatedConstructionFeatures,
    actual_design: ActualDesignMatrix,
) -> ConstructionTransferEstimate:
    """Estimate clean-intercept transfer; this remains construction-only."""

    validate_impulse_observability_config(config)
    expected_probe_ids = tuple(
        record.probe_id for record in build_impulse_triage_plan(config)
    )
    if (
        not isinstance(output_features, ValidatedConstructionFeatures)
        or output_features.feature_schema_digest
        != config["construction_feature_schema"]["feature_schema_digest"]
        or output_features.probe_ids != expected_probe_ids
        or output_features.probe_ids != actual_design.probe_ids
    ):
        raise ValueError(
            "estimator 要求 feature/design/frozen plan probe identity 完全一致"
        )
    features = np.asarray(output_features.values, dtype=np.float64)
    # Do not trust callers to retain validation after dataclass construction.
    validated = validate_construction_output_features(
        config,
        features,
        feature_schema_digest=output_features.feature_schema_digest,
        probe_ids=output_features.probe_ids,
        row_binding_digests=output_features.row_binding_digests,
    )
    features = validated.values
    design_values = _validate_actual_design_for_estimation(
        config,
        actual_design,
    )
    clean_intercept = features[:2].mean(axis=0)
    centered = features.T - clean_intercept[:, None]
    transfer = centered @ np.linalg.pinv(design_values)
    fitted = clean_intercept[:, None] + transfer @ design_values
    residual = features.T - fitted

    center_difference_columns: list[np.ndarray] = []
    for coordinate in range(STAGE_BASIS_RANK):
        positive_index = IMPULSE_CLEAN_REPEAT_COUNT + coordinate
        negative_index = (
            IMPULSE_CLEAN_REPEAT_COUNT
            + STAGE_BASIS_RANK
            + coordinate
        )
        amplitude_difference = (
            design_values[coordinate, positive_index]
            - design_values[coordinate, negative_index]
        )
        if (
            not math.isfinite(float(amplitude_difference))
            or abs(float(amplitude_difference)) <= 1e-15
        ):
            raise ValueError(
                f"A_actual coordinate {coordinate} signed amplitude difference 退化"
            )
        center_difference_columns.append(
            (
                features[positive_index]
                - features[negative_index]
            )
            / float(amplitude_difference)
        )
    signed_center_difference_transfer = np.stack(
        center_difference_columns,
        axis=1,
    )
    return ConstructionTransferEstimate(
        clean_intercept=clean_intercept,
        transfer_matrix=transfer,
        fitted_output_features=fitted,
        residual_matrix=residual,
        signed_center_difference_transfer_matrix=(
            signed_center_difference_transfer
        ),
    )


def evaluate_gate_a_statistics(
    config: Mapping[str, Any],
    output_features: ValidatedConstructionFeatures,
    actual_design: ActualDesignMatrix,
    *,
    primary_checkpoint_ready: Mapping[str, bool],
) -> GateAStatistics:
    """Evaluate frozen Gate A formulas without authorizing execution/staging."""

    expected_probe_ids = tuple(
        record.probe_id for record in build_impulse_triage_plan(config)
    )
    if (
        output_features.probe_ids != expected_probe_ids
        or output_features.probe_ids != actual_design.probe_ids
    ):
        raise ValueError(
            "Gate A feature/design/frozen plan probe identity 不一致"
        )
    estimate = estimate_construction_transfer(
        config,
        output_features,
        actual_design,
    )
    features = output_features.values
    gate = config["gate_a_sample_internal_causal_observability"]
    required_checkpoints = tuple(
        gate["primary_required_transfer_checkpoints"]
    )
    if set(primary_checkpoint_ready) != set(required_checkpoints) or any(
        not isinstance(value, bool)
        for value in primary_checkpoint_ready.values()
    ):
        raise ValueError("Gate A primary checkpoint readiness 集合不完整")
    checkpoint_chain_ready = all(
        primary_checkpoint_ready[name] for name in required_checkpoints
    )
    clean_distance = float(np.linalg.norm(features[0] - features[1]))
    finite_noise_floor = max(
        clean_distance / math.sqrt(2.0),
        float(gate["minimum_finite_noise_floor"]),
    )
    response_columns = 0.5 * (features[2:8] - features[8:14]).T
    response_norms = np.linalg.norm(response_columns, axis=0)
    singular_values = np.linalg.svd(
        response_columns,
        compute_uv=False,
    )
    transfer_singular_values = np.linalg.svd(
        estimate.signed_center_difference_transfer_matrix,
        compute_uv=False,
    )
    minimum_response = float(np.min(response_norms))
    minimum_singular = float(singular_values[-1])
    singular_threshold = max(
        float(gate["minimum_absolute_singular_value"]),
        float(gate["minimum_noise_normalized_singular_value"])
        * finite_noise_floor,
    )
    effective_rank = int(np.count_nonzero(singular_values >= singular_threshold))
    output_condition = (
        float("inf")
        if transfer_singular_values[-1] <= 0.0
        else float(
            transfer_singular_values[0]
            / transfer_singular_values[-1]
        )
    )
    snr = minimum_response / finite_noise_floor
    noise_normalized_singular = minimum_singular / finite_noise_floor

    antisymmetry_cosines: list[float] = []
    antisymmetry_residuals: list[float] = []
    for coordinate in range(STAGE_BASIS_RANK):
        positive = features[2 + coordinate] - estimate.clean_intercept
        negative = features[8 + coordinate] - estimate.clean_intercept
        cosine = _cosine(positive, -negative)
        antisymmetry_cosines.append(cosine)
        denominator = max(
            float(np.linalg.norm(positive) + np.linalg.norm(negative)),
            1e-15,
        )
        antisymmetry_residuals.append(
            float(np.linalg.norm(positive + negative)) / denominator
        )
    minimum_antisymmetry_cosine = min(antisymmetry_cosines)
    maximum_antisymmetry_residual = max(antisymmetry_residuals)
    finite_statistics = all(
        math.isfinite(value)
        for value in (
            clean_distance,
            finite_noise_floor,
            minimum_response,
            minimum_singular,
            output_condition,
            snr,
            noise_normalized_singular,
            minimum_antisymmetry_cosine,
            maximum_antisymmetry_residual,
        )
    )
    ready = bool(
        finite_statistics
        and checkpoint_chain_ready
        and minimum_response
        >= float(gate["minimum_absolute_output_response"])
        and minimum_singular
        >= float(gate["minimum_absolute_singular_value"])
        and effective_rank >= int(gate["minimum_effective_rank"])
        and output_condition
        <= float(gate["maximum_output_transfer_condition_number"])
        and snr >= float(gate["minimum_output_feature_snr"])
        and noise_normalized_singular
        >= float(gate["minimum_noise_normalized_singular_value"])
        and minimum_antisymmetry_cosine
        >= float(gate["minimum_positive_negative_antisymmetry_cosine"])
        and maximum_antisymmetry_residual
        <= float(
            gate[
                "maximum_positive_negative_antisymmetry_residual_ratio"
            ]
        )
    )
    return GateAStatistics(
        clean_repeat_distance=clean_distance,
        finite_noise_floor=finite_noise_floor,
        clean_distance_below_numerical_resolution=(
            clean_distance
            <= float(
                gate["maximum_clean_distance_for_numerical_resolution"]
            )
        ),
        minimum_absolute_response=minimum_response,
        minimum_response_singular_value=minimum_singular,
        noise_normalized_minimum_singular_value=(
            noise_normalized_singular
        ),
        effective_rank=effective_rank,
        output_transfer_condition_number=output_condition,
        minimum_output_feature_snr=snr,
        minimum_positive_negative_antisymmetry_cosine=(
            minimum_antisymmetry_cosine
        ),
        maximum_positive_negative_antisymmetry_residual_ratio=(
            maximum_antisymmetry_residual
        ),
        primary_checkpoint_chain_ready=checkpoint_chain_ready,
        gate_a_ready=ready,
    )
