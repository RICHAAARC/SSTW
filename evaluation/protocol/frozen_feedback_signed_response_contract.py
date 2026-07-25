"""Frozen contract primitives for the clean-trace feedback-isolation diagnostic.

This module contains only construction-time plan, response statistics, and
fail-closed validation.  It is not a Gate A evaluator or an observer.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from evaluation.protocol.existing_six_video_spatiotemporal_signed_response_contract import (
    classify_frozen_feedback_signed_response_design,
)
from evaluation.protocol.gate_a_root_cause_amplitude_feedback_contract import (
    PairedResponseStatistics,
)
from evaluation.protocol.impulse_observability_contract import (
    ImpulseProbePlanRecord,
    canonical_json_digest,
)


PROFILE_ID = "sstw_frozen_feedback_signed_response_construction_diagnostic"
RECORD_VERSION = "frozen_feedback_signed_response_construction_diagnostic_v1"
DEFAULT_CONFIG_PATH = Path(
    "configs/protocol/sstw_frozen_feedback_signed_response_diagnostic.json"
)
FROZEN_CONFIG_DIGEST = (
    "3378abdbff3a4a2f335993e042e3bf7a0310a6b791b189bb01e5fdfbe99e303f"
)
FROZEN_OUTPUT_IDS = (
    "clean",
    "positive_early_flow_channel_0",
    "negative_early_flow_channel_0",
    "positive_late_flow_channel_0",
    "negative_late_flow_channel_0",
)
PAIR_IDS = ("early_flow_channel_0", "late_flow_channel_0")
FULL_CHECKPOINT_IDS = (
    "T_final_latent_full",
    "T_decoded_full_rgb_float32",
    "T_saved_video_full_rgb24",
    "T_reencoded",
    "T_output_feature",
)
POST_LATENT_CHECKPOINT_IDS = FULL_CHECKPOINT_IDS[1:]
EXPECTED_CLEAN_SCHEDULER_STEP_COUNT = 8
EXPECTED_CLEAN_TRANSFORMER_FORWARD_CALL_COUNT = 16
EXPECTED_COUNTERFACTUAL_TRANSFORMER_FORWARD_CALL_COUNT = 0
EXPECTED_COUNTERFACTUAL_STEP_COUNT = 32
CLAIM_SUPPORT_STATUS = (
    "frozen_feedback_signed_response_construction_diagnostic_only_"
    "not_method_evidence"
)


@dataclass(frozen=True)
class FrozenFeedbackResponseGate:
    """One pair/checkpoint signed-response decision."""

    pair_id: str
    checkpoint_id: str
    statistics: PairedResponseStatistics
    signed_response_ready: bool


def _clamp_cosine_machine_roundoff(value: float | None) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("signed-response cosine 非有限")
    if numeric > 1.0:
        if numeric <= 1.0 + 1e-12:
            return 1.0
        raise ValueError("signed-response cosine 超过1")
    if numeric < -1.0:
        if numeric >= -1.0 - 1e-12:
            return -1.0
        raise ValueError("signed-response cosine 小于-1")
    return numeric


def load_frozen_feedback_signed_response_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Load the one exact predeclared diagnostic configuration."""

    config_path = Path(path)
    value = json.loads(config_path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise TypeError("frozen-feedback config 必须是 JSON object")
    digest = canonical_json_digest(value)
    if digest != FROZEN_CONFIG_DIGEST:
        raise ValueError(
            "frozen-feedback config 与预冻结合同不一致: "
            f"expected={FROZEN_CONFIG_DIGEST} observed={digest}"
        )
    if value.get("profile_id") != PROFILE_ID:
        raise ValueError("frozen-feedback profile_id 漂移")
    if (
        tuple(value["five_output_plan"]["ordered_probe_ids"])
        != FROZEN_OUTPUT_IDS
        or tuple(value["checkpoint_contract"]["ordered_checkpoint_ids"])
        != FULL_CHECKPOINT_IDS
        or tuple(
            value["classification_contract"]["post_latent_checkpoint_ids"]
        )
        != POST_LATENT_CHECKPOINT_IDS
    ):
        raise ValueError("frozen-feedback output/checkpoint order 漂移")
    authorization = value["authorization_boundary"]
    if authorization != {
        "gate_a_retry": False,
        "gate_a_pass": False,
        "gate_b_execution_allowed": False,
        "gate_c_execution_allowed": False,
        "cross_identity_confirmation_allowed": False,
        "key_selectivity_execution_allowed": False,
        "wrong_key_execution_allowed": False,
        "observer_execution_allowed": False,
        "state_dynamics_design_allowed": False,
        "state_dynamics_execution_allowed": False,
        "f_k_g_k_design_allowed": False,
        "llr_execution_allowed": False,
        "composite_execution_allowed": False,
        "attack_execution_allowed": False,
        "pilot_execution_allowed": False,
        "fixed_fpr_execution_allowed": False,
        "external_baseline_execution_allowed": False,
        "training_or_finetuning_allowed": False,
        "automatic_feature_selection_allowed": False,
        "strength_sweep_allowed": False,
        "grid_sweep_allowed": False,
        "identity_sweep_allowed": False,
        "channel_sweep_allowed": False,
        "automatic_followup_execution_allowed": False,
        "paper_claim_allowed": False,
        "formal_result": False,
        "stage_progression_allowed": False,
    }:
        raise ValueError("frozen-feedback authorization boundary 漂移")
    return value


def build_frozen_feedback_plan(
    config: Mapping[str, Any],
) -> tuple[ImpulseProbePlanRecord, ...]:
    """Build the exact clean plus four signed counterfactual identities."""

    if canonical_json_digest(config) != FROZEN_CONFIG_DIGEST:
        raise ValueError("frozen-feedback plan 只接受预冻结 config")
    amplitude = float(config["five_output_plan"]["lambda_max"])
    plan = (
        ImpulseProbePlanRecord(
            probe_id="clean",
            probe_role="shared_clean_base_velocity_trace",
            stage_index=None,
            stage_name=None,
            channel_index=None,
            polarity=0,
            nominal_signed_amplitude=0.0,
        ),
        ImpulseProbePlanRecord(
            probe_id="positive_early_flow_channel_0",
            probe_role="frozen_feedback_signed_counterfactual",
            stage_index=0,
            stage_name="early_flow",
            channel_index=0,
            polarity=1,
            nominal_signed_amplitude=amplitude,
        ),
        ImpulseProbePlanRecord(
            probe_id="negative_early_flow_channel_0",
            probe_role="frozen_feedback_signed_counterfactual",
            stage_index=0,
            stage_name="early_flow",
            channel_index=0,
            polarity=-1,
            nominal_signed_amplitude=-amplitude,
        ),
        ImpulseProbePlanRecord(
            probe_id="positive_late_flow_channel_0",
            probe_role="frozen_feedback_signed_counterfactual",
            stage_index=2,
            stage_name="late_flow",
            channel_index=0,
            polarity=1,
            nominal_signed_amplitude=amplitude,
        ),
        ImpulseProbePlanRecord(
            probe_id="negative_late_flow_channel_0",
            probe_role="frozen_feedback_signed_counterfactual",
            stage_index=2,
            stage_name="late_flow",
            channel_index=0,
            polarity=-1,
            nominal_signed_amplitude=-amplitude,
        ),
    )
    if tuple(item.probe_id for item in plan) != FROZEN_OUTPUT_IDS:
        raise AssertionError("frozen-feedback plan order 漂移")
    return plan


def compute_single_clean_response_statistics(
    *,
    pair_id: str,
    checkpoint_id: str,
    clean: Sequence[float] | np.ndarray,
    positive: Sequence[float] | np.ndarray,
    negative: Sequence[float] | np.ndarray,
    denominator_epsilon: float,
) -> PairedResponseStatistics:
    """Apply the frozen odd/common formulas with one shared clean output."""

    arrays = tuple(
        np.asarray(value, dtype=np.float64).reshape(-1)
        for value in (clean, positive, negative)
    )
    if len({item.shape for item in arrays}) != 1:
        raise ValueError(f"{pair_id}/{checkpoint_id} response shapes 不一致")
    if not all(np.all(np.isfinite(item)) for item in arrays):
        raise ValueError(f"{pair_id}/{checkpoint_id} response 含非有限值")
    clean_value, plus, minus = arrays
    delta_plus = plus - clean_value
    delta_minus = minus - clean_value
    odd = 0.5 * (delta_plus - delta_minus)
    common = 0.5 * (delta_plus + delta_minus)
    plus_norm = float(np.linalg.norm(delta_plus))
    minus_norm = float(np.linalg.norm(delta_minus))
    odd_norm = float(np.linalg.norm(odd))
    common_norm = float(np.linalg.norm(common))
    common_ratio = (
        None
        if odd_norm <= denominator_epsilon
        else common_norm / odd_norm
    )
    cosine_denominator = plus_norm * minus_norm
    cosine = _clamp_cosine_machine_roundoff(
        None
        if cosine_denominator <= denominator_epsilon
        else float(delta_plus @ (-delta_minus) / cosine_denominator)
    )
    residual_denominator = plus_norm + minus_norm
    residual = (
        None
        if residual_denominator <= denominator_epsilon
        else float(np.linalg.norm(delta_plus + delta_minus))
        / residual_denominator
    )
    finite_values = (
        plus_norm,
        minus_norm,
        odd_norm,
        common_norm,
    ) + tuple(
        value
        for value in (common_ratio, cosine, residual)
        if value is not None
    )
    return PairedResponseStatistics(
        pair_id=pair_id,
        checkpoint_id=checkpoint_id,
        clean_distance=0.0,
        positive_centered_norm=plus_norm,
        negative_centered_norm=minus_norm,
        odd_norm=odd_norm,
        common_norm=common_norm,
        common_odd_ratio=common_ratio,
        antisymmetry_cosine=cosine,
        antisymmetry_residual=residual,
        finite=all(math.isfinite(value) for value in finite_values),
    )


def compute_single_clean_response_statistics_from_gram(
    *,
    pair_id: str,
    checkpoint_id: str,
    gram_matrix: Sequence[Sequence[float]] | np.ndarray,
    row_ids: Sequence[str],
    denominator_epsilon: float,
) -> PairedResponseStatistics:
    """Compute exact full-array statistics from the five-row Gram matrix."""

    ids = tuple(str(value) for value in row_ids)
    gram = np.asarray(gram_matrix, dtype=np.float64)
    if (
        ids != FROZEN_OUTPUT_IDS
        or gram.shape != (len(ids), len(ids))
        or not np.all(np.isfinite(gram))
        or not np.allclose(gram, gram.T, rtol=1e-10, atol=1e-8)
    ):
        raise ValueError(f"{checkpoint_id} five-output Gram 未冻结")
    index = {probe_id: offset for offset, probe_id in enumerate(ids)}
    plus_id = f"positive_{pair_id}"
    minus_id = f"negative_{pair_id}"
    plus = np.zeros(len(ids), dtype=np.float64)
    minus = np.zeros(len(ids), dtype=np.float64)
    plus[index[plus_id]] = 1.0
    plus[index["clean"]] = -1.0
    minus[index[minus_id]] = 1.0
    minus[index["clean"]] = -1.0
    odd = 0.5 * (plus - minus)
    common = 0.5 * (plus + minus)

    def squared_norm(vector: np.ndarray) -> float:
        value = float(vector @ gram @ vector)
        if value < 0.0 and abs(value) <= 1e-7:
            return 0.0
        if value < 0.0:
            raise ValueError(f"{checkpoint_id} Gram 二次型为负")
        return value

    plus_norm = math.sqrt(squared_norm(plus))
    minus_norm = math.sqrt(squared_norm(minus))
    odd_norm = math.sqrt(squared_norm(odd))
    common_norm = math.sqrt(squared_norm(common))
    common_ratio = (
        None
        if odd_norm <= denominator_epsilon
        else common_norm / odd_norm
    )
    denominator = plus_norm * minus_norm
    cosine = _clamp_cosine_machine_roundoff(
        None
        if denominator <= denominator_epsilon
        else float(plus @ gram @ (-minus) / denominator)
    )
    residual_denominator = plus_norm + minus_norm
    residual = (
        None
        if residual_denominator <= denominator_epsilon
        else 2.0 * common_norm / residual_denominator
    )
    finite_values = (
        plus_norm,
        minus_norm,
        odd_norm,
        common_norm,
    ) + tuple(
        value
        for value in (common_ratio, cosine, residual)
        if value is not None
    )
    return PairedResponseStatistics(
        pair_id=pair_id,
        checkpoint_id=checkpoint_id,
        clean_distance=0.0,
        positive_centered_norm=plus_norm,
        negative_centered_norm=minus_norm,
        odd_norm=odd_norm,
        common_norm=common_norm,
        common_odd_ratio=common_ratio,
        antisymmetry_cosine=cosine,
        antisymmetry_residual=residual,
        finite=all(math.isfinite(value) for value in finite_values),
    )


def apply_frozen_signed_response_gate(
    config: Mapping[str, Any],
    statistics: PairedResponseStatistics,
) -> FrozenFeedbackResponseGate:
    """Apply only the four predeclared signed-response thresholds."""

    if canonical_json_digest(config) != FROZEN_CONFIG_DIGEST:
        raise ValueError("signed-response gate 只接受预冻结 config")
    gate = config["signed_response_gate"]
    ready = bool(
        statistics.finite
        and statistics.antisymmetry_cosine is not None
        and statistics.antisymmetry_cosine
        >= float(gate["minimum_antisymmetry_cosine"])
        and statistics.antisymmetry_residual is not None
        and statistics.antisymmetry_residual
        <= float(gate["maximum_antisymmetry_residual"])
        and statistics.common_odd_ratio is not None
        and statistics.common_odd_ratio
        <= float(gate["maximum_common_odd_ratio"])
        and statistics.odd_norm >= float(gate["minimum_odd_norm"])
    )
    return FrozenFeedbackResponseGate(
        pair_id=statistics.pair_id,
        checkpoint_id=statistics.checkpoint_id,
        statistics=statistics,
        signed_response_ready=ready,
    )


def classify_frozen_feedback_results(
    *,
    clean_coverage_and_guards_ready: bool,
    gates: Mapping[tuple[str, str], FrozenFeedbackResponseGate],
) -> dict[str, Any]:
    """Derive the design-only branch through the committed truth table."""

    expected = {
        (pair_id, checkpoint_id)
        for pair_id in PAIR_IDS
        for checkpoint_id in FULL_CHECKPOINT_IDS
    }
    if set(gates) != expected:
        raise ValueError("frozen-feedback pair/checkpoint gate coverage 不完整")
    verified: dict[
        tuple[str, str], FrozenFeedbackResponseGate
    ] = {}
    for identity, supplied in gates.items():
        if (
            supplied.pair_id != identity[0]
            or supplied.checkpoint_id != identity[1]
        ):
            raise ValueError("frozen-feedback gate identity 不一致")
        recomputed = apply_frozen_signed_response_gate(
            load_frozen_feedback_signed_response_config(),
            supplied.statistics,
        )
        if recomputed != supplied:
            raise ValueError("frozen-feedback gate 不能信任 caller boolean")
        verified[identity] = recomputed
    early_latent = verified[
        ("early_flow_channel_0", "T_final_latent_full")
    ].signed_response_ready
    late_latent = verified[
        ("late_flow_channel_0", "T_final_latent_full")
    ].signed_response_ready
    post_latent = all(
        verified[(pair_id, checkpoint_id)].signed_response_ready
        for pair_id in PAIR_IDS
        for checkpoint_id in POST_LATENT_CHECKPOINT_IDS
    )
    result = classify_frozen_feedback_signed_response_design(
        clean_coverage_and_guards_ready=clean_coverage_and_guards_ready,
        early_full_final_latent_signed=early_latent,
        late_full_final_latent_signed=late_latent,
        all_post_latent_checkpoints_signed=post_latent,
    )
    if (
        result.get("formal_result") is not False
        or result.get("stage_progression_allowed") is not False
        or result.get("unique_root_cause_claim_allowed") is not False
    ):
        raise AssertionError("frozen-feedback classifier 越过非正式边界")
    return result
