"""Runtime-light tests for block-local attention relation control."""

from __future__ import annotations

from pathlib import Path
import inspect

import numpy as np
import pytest

from experiments.generative_video_model_probe.patch_relation_block_attention_response_preflight import (
    DECISION_FILENAME,
    RECORD_FILENAME,
    run_local_block_attention_response_preflight,
)
from main.methods.state_space_watermark.patch_relation_attention_runtime import (
    CANDIDATE_BIAS_MAGNITUDES,
    CANDIDATE_SIGNS,
    SCHEDULER_STEP_CALL_COUNT,
    ScopedWanBlockLocalAttentionBiasAdapter,
    WAN_SELF_ATTENTION_HEAD_COUNT,
    ZERO_CONTROL_REPEAT_COUNT,
    _WanBlockLocalAttentionBiasProcessor,
    apply_block_attention_sparse_bias_runtime,
    build_local_block_attention_primitive_response_record,
    validate_block_attention_primitive_response_record,
)
from main.methods.state_space_watermark.patch_relation_block_attention import (
    build_block_attention_relation_descriptor,
)


@pytest.mark.quick
def test_wan_runtime_head_count_matches_frozen_1_3b_config() -> None:
    assert WAN_SELF_ATTENTION_HEAD_COUNT == 12


@pytest.mark.quick
def test_sparse_attention_bias_runtime_clean_and_signed_antipodes() -> None:
    descriptor = build_block_attention_relation_descriptor()
    clean_values, clean_record = apply_block_attention_sparse_bias_runtime(
        descriptor,
        signed_coefficient=0,
        magnitude=CANDIDATE_BIAS_MAGNITUDES[0],
    )
    positive_values, positive_record = apply_block_attention_sparse_bias_runtime(
        descriptor,
        signed_coefficient=1,
        magnitude=CANDIDATE_BIAS_MAGNITUDES[0],
    )
    negative_values, negative_record = apply_block_attention_sparse_bias_runtime(
        descriptor,
        signed_coefficient=-1,
        magnitude=CANDIDATE_BIAS_MAGNITUDES[0],
    )
    assert clean_record.clean_exact_no_op is True
    assert clean_record.changed_entry_count == 0
    assert np.all(clean_values == 0.0)
    assert positive_record.changed_entry_count == len(descriptor.entries)
    assert negative_record.changed_entry_count == len(descriptor.entries)
    assert np.array_equal(positive_values, -negative_values)
    assert positive_record.bias_digest != negative_record.bias_digest


@pytest.mark.quick
def test_sparse_attention_bias_runtime_rejects_unfrozen_strength() -> None:
    descriptor = build_block_attention_relation_descriptor()
    with pytest.raises(ValueError, match="冻结候选集合"):
        apply_block_attention_sparse_bias_runtime(
            descriptor,
            signed_coefficient=1,
            magnitude=CANDIDATE_BIAS_MAGNITUDES[0] * 0.5,
        )


@pytest.mark.quick
def test_wan_adapter_uses_sparse_row_recompute_not_dense_attention_mask() -> None:
    source = inspect.getsource(_WanBlockLocalAttentionBiasProcessor)
    assert "_apply_sparse_row_bias" in source
    assert "dispatch_attention_fn" in source
    assert "scaled_dot_product_attention" not in source
    assert "torch.zeros" not in source
    assert "attn_mask=mask" not in source


@pytest.mark.quick
def test_wan_attention_scope_rolls_back_enter_failure() -> None:
    class FailingAttention:
        def __init__(self) -> None:
            self.processor = object()
            self.restored_processor = None

        def set_processor(self, processor: object) -> None:
            if processor is not self.processor:
                raise RuntimeError("synthetic install failure")
            self.restored_processor = processor

    class Block:
        def __init__(self, attention: FailingAttention) -> None:
            self.attn1 = attention

    class Transformer:
        def __init__(self, attention: FailingAttention) -> None:
            self.blocks = [Block(FailingAttention()) for _ in range(18)] + [
                Block(attention)
            ]

    attention = FailingAttention()
    transformer = Transformer(attention)
    original_processor = attention.processor
    with pytest.raises(RuntimeError, match="synthetic install failure"):
        with ScopedWanBlockLocalAttentionBiasAdapter(
            transformer,
            signed_coefficient=1,
            magnitude=CANDIDATE_BIAS_MAGNITUDES[0],
            input_binding_digest="0" * 64,
            cfg_branch_role="conditional",
        ):
            raise AssertionError("scope body must not run")
    assert attention.processor is original_processor
    assert attention.restored_processor is original_processor
    assert not hasattr(attention, "_sstw_block_attention_scope_active")


@pytest.mark.quick
def test_local_primitive_response_record_has_no_execution_side_effects() -> None:
    record = build_local_block_attention_primitive_response_record()
    validate_block_attention_primitive_response_record(record)
    assert record.zero_repeat_count == ZERO_CONTROL_REPEAT_COUNT
    assert record.scheduler_step_call_count == SCHEDULER_STEP_CALL_COUNT
    assert record.decode_executed is False
    assert record.video_export_executed is False
    assert record.gate0_executed is False
    assert record.formal_result is False
    assert record.stage_progression_allowed is False
    assert len(record.candidate_responses) == (
        len(CANDIDATE_BIAS_MAGNITUDES) * len(CANDIDATE_SIGNS)
    )
    assert record.diagnostic_classification == (
        "feasible_nonzero_near_antisymmetric_response_candidate"
    )
    assert all(
        cosine == -1.0
        for cosine in record.positive_negative_response_cosine_by_magnitude
    )


@pytest.mark.quick
def test_repeatability_floor_classification_is_fail_closed() -> None:
    record = build_local_block_attention_primitive_response_record(
        repeat_floor=1.0,
    )
    validate_block_attention_primitive_response_record(record)
    assert record.diagnostic_classification == "repeatability_floor_candidate"


@pytest.mark.quick
def test_record_forged_classification_rejected() -> None:
    record = build_local_block_attention_primitive_response_record()
    forged = record.__class__(
        record_kind=record.record_kind,
        descriptor_digest=record.descriptor_digest,
        zero_repeat_count=record.zero_repeat_count,
        scheduler_step_call_count=record.scheduler_step_call_count,
        decode_executed=record.decode_executed,
        video_export_executed=record.video_export_executed,
        gate0_executed=record.gate0_executed,
        candidate_responses=record.candidate_responses,
        positive_negative_response_cosine_by_magnitude=(
            record.positive_negative_response_cosine_by_magnitude
        ),
        diagnostic_classification="indeterminate",
        formal_result=record.formal_result,
        stage_progression_allowed=record.stage_progression_allowed,
        claim_support_status=record.claim_support_status,
    )
    with pytest.raises(ValueError, match="classification"):
        validate_block_attention_primitive_response_record(forged)


@pytest.mark.quick
def test_record_forged_candidate_guard_rejected() -> None:
    record = build_local_block_attention_primitive_response_record()
    first = record.candidate_responses[0]
    forged_first = first.__class__(
        candidate_index=first.candidate_index,
        signed_coefficient=first.signed_coefficient,
        magnitude=first.magnitude,
        response_vector=first.response_vector,
        repeat_delta_vector=first.repeat_delta_vector,
        response_l2_norm=first.norm_budget * 10.0,
        norm_budget=first.norm_budget,
        norm_guard_passed=True,
        repeat_l2_norm=first.repeat_l2_norm,
        repeat_floor_ratio=first.repeat_floor_ratio,
        nonzero_response=first.nonzero_response,
        repeatable_above_floor=first.repeatable_above_floor,
        near_antisymmetric_pair=first.near_antisymmetric_pair,
        feasible_candidate=first.feasible_candidate,
        application_record=first.application_record,
    )
    forged = record.__class__(
        record_kind=record.record_kind,
        descriptor_digest=record.descriptor_digest,
        zero_repeat_count=record.zero_repeat_count,
        scheduler_step_call_count=record.scheduler_step_call_count,
        decode_executed=record.decode_executed,
        video_export_executed=record.video_export_executed,
        gate0_executed=record.gate0_executed,
        candidate_responses=(forged_first,) + record.candidate_responses[1:],
        positive_negative_response_cosine_by_magnitude=(
            record.positive_negative_response_cosine_by_magnitude
        ),
        diagnostic_classification=record.diagnostic_classification,
        formal_result=record.formal_result,
        stage_progression_allowed=record.stage_progression_allowed,
        claim_support_status=record.claim_support_status,
    )
    with pytest.raises(ValueError, match="norm|feasible"):
        validate_block_attention_primitive_response_record(forged)


@pytest.mark.quick
def test_record_forged_antipodal_cosine_and_feasible_rejected() -> None:
    record = build_local_block_attention_primitive_response_record()
    positive = record.candidate_responses[0]
    negative = record.candidate_responses[1]
    forged_negative = negative.__class__(
        candidate_index=negative.candidate_index,
        signed_coefficient=negative.signed_coefficient,
        magnitude=negative.magnitude,
        response_vector=positive.response_vector,
        repeat_delta_vector=negative.repeat_delta_vector,
        response_l2_norm=negative.response_l2_norm,
        norm_budget=negative.norm_budget,
        norm_guard_passed=negative.norm_guard_passed,
        repeat_l2_norm=negative.repeat_l2_norm,
        repeat_floor_ratio=negative.repeat_floor_ratio,
        nonzero_response=negative.nonzero_response,
        repeatable_above_floor=negative.repeatable_above_floor,
        near_antisymmetric_pair=True,
        feasible_candidate=True,
        application_record=negative.application_record,
    )
    forged = record.__class__(
        record_kind=record.record_kind,
        descriptor_digest=record.descriptor_digest,
        zero_repeat_count=record.zero_repeat_count,
        scheduler_step_call_count=record.scheduler_step_call_count,
        decode_executed=record.decode_executed,
        video_export_executed=record.video_export_executed,
        gate0_executed=record.gate0_executed,
        candidate_responses=(
            positive,
            forged_negative,
        )
        + record.candidate_responses[2:],
        positive_negative_response_cosine_by_magnitude=(
            record.positive_negative_response_cosine_by_magnitude
        ),
        diagnostic_classification=(
            "feasible_nonzero_near_antisymmetric_response_candidate"
        ),
        formal_result=record.formal_result,
        stage_progression_allowed=record.stage_progression_allowed,
        claim_support_status=record.claim_support_status,
    )
    with pytest.raises(ValueError, match="antisymmetry cosine"):
        validate_block_attention_primitive_response_record(forged)


@pytest.mark.quick
def test_local_preflight_runner_writes_only_nonformal_artifacts(
    tmp_path: Path,
) -> None:
    result = run_local_block_attention_response_preflight(
        output_root=tmp_path / "preflight",
    )
    decision = result["decision"]
    assert decision["formal_result"] is False
    assert decision["stage_progression_allowed"] is False
    assert decision["scheduler_step_call_count"] == 0
    assert decision["decode_executed"] is False
    assert decision["video_export_executed"] is False
    assert decision["gate0_executed"] is False
    assert (tmp_path / "preflight" / RECORD_FILENAME).is_file()
    assert (tmp_path / "preflight" / DECISION_FILENAME).is_file()
