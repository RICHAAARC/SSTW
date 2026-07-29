"""Block-local Patch-relation attention control primitives.

This module implements both the local, backend-neutral contract and the narrow
Wan runtime scope for the block-local carrier family.  The Wan scope patches
one predeclared self-attention processor on one transformer instance and adds a
sparse pre-softmax QK-logit bias.  It does not call a scheduler, decode frames,
export video, or produce Gate/method evidence by itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
from typing import Any

import numpy as np

from main.methods.state_space_watermark.patch_relation_block_attention import (
    BlockAttentionRelationDescriptor,
    MAXIMUM_LOGIT_BIAS_MAGNITUDE,
    build_block_attention_relation_descriptor,
    signed_sparse_bias_values,
    validate_block_attention_relation_descriptor,
)


ZERO_CONTROL_REPEAT_COUNT = 2
CANDIDATE_BIAS_MAGNITUDES = (
    MAXIMUM_LOGIT_BIAS_MAGNITUDE,
    MAXIMUM_LOGIT_BIAS_MAGNITUDE / 4.0,
    MAXIMUM_LOGIT_BIAS_MAGNITUDE / 16.0,
)
CANDIDATE_SIGNS = (1, -1)
REPEATABILITY_FLOOR_RATIO_THRESHOLD = 0.1
NEAR_ANTISYMMETRY_COSINE_THRESHOLD = -0.9
NONZERO_RESPONSE_FLOOR = 1.0e-12
SCHEDULER_STEP_CALL_COUNT = 0
STEP0_NORM_BUDGET = 2.937237890625
WAN_SELF_ATTENTION_SEQUENCE_LENGTH = 5760
WAN_SELF_ATTENTION_HEAD_COUNT = 12
WAN_RUNTIME_EXPECTED_PROCESSOR_CALLS = 1


def _require_frozen_candidate_magnitude(magnitude: float) -> float:
    observed = float(magnitude)
    if not math.isfinite(observed):
        raise ValueError("block-attention candidate magnitude 必须 finite")
    if observed not in CANDIDATE_BIAS_MAGNITUDES:
        raise ValueError(
            "block-attention candidate magnitude 必须来自冻结候选集合"
        )
    return observed


@dataclass(frozen=True)
class BlockAttentionBiasApplicationRecord:
    """Scalar-only record for one sparse block-attention bias application."""

    descriptor_digest: str
    signed_coefficient: int
    magnitude: float
    entry_count: int
    changed_entry_count: int
    bias_l2_norm: float
    bias_digest: str
    clean_exact_no_op: bool


@dataclass(frozen=True)
class BlockAttentionScopeRecord:
    """Successful local scope record for one adapter application."""

    descriptor_digest: str
    target_block_index: int
    signed_coefficient: int
    magnitude: float
    callback_invocation_count: int
    scope_completed_successfully: bool
    scheduler_step_call_count: int
    decode_executed: bool
    video_export_executed: bool
    gate0_executed: bool
    application_record: BlockAttentionBiasApplicationRecord


@dataclass(frozen=True)
class WanBlockAttentionBranchApplicationRecord:
    """Successful real Wan block-local attention application."""

    descriptor_digest: str
    target_block_index: int
    signed_coefficient: int
    magnitude: float
    input_binding_digest: str
    cfg_branch_role: str
    processor_call_count: int
    scope_completed_successfully: bool
    attention_mask_shape: tuple[int, int, int, int]
    attention_mask_dtype: str
    changed_entry_count: int
    bias_l2_norm: float
    bias_digest: str
    clean_exact_no_op: bool
    local_single_step_preflight_only: bool = True
    scheduler_step_call_count: int = 0
    decode_executed: bool = False
    video_export_executed: bool = False
    gate0_executed: bool = False
    formal_result: bool = False
    stage_progression_allowed: bool = False


@dataclass(frozen=True)
class BlockAttentionCandidateResponse:
    """One local candidate response measurement."""

    candidate_index: int
    signed_coefficient: int
    magnitude: float
    response_vector: tuple[float, ...]
    repeat_delta_vector: tuple[float, ...]
    response_l2_norm: float
    norm_budget: float
    norm_guard_passed: bool
    repeat_l2_norm: float
    repeat_floor_ratio: float
    nonzero_response: bool
    repeatable_above_floor: bool
    near_antisymmetric_pair: bool
    feasible_candidate: bool
    application_record: BlockAttentionBiasApplicationRecord


@dataclass(frozen=True)
class BlockAttentionPrimitiveResponseRecord:
    """Nonformal local primitive response record."""

    record_kind: str
    descriptor_digest: str
    zero_repeat_count: int
    scheduler_step_call_count: int
    decode_executed: bool
    video_export_executed: bool
    gate0_executed: bool
    candidate_responses: tuple[BlockAttentionCandidateResponse, ...]
    positive_negative_response_cosine_by_magnitude: tuple[float, ...]
    diagnostic_classification: str
    formal_result: bool
    stage_progression_allowed: bool
    claim_support_status: str


class ScopedBlockLocalAttentionBiasAdapter:
    """Install a local block-attention relation-bias callback on one object.

    The future Wan adapter must preserve these semantics: one scoped object,
    exact target block binding, clean no-op, signed antipodes, callback coverage
    tracked inside the scope, and fail-closed record formation.
    """

    _active_marker_name = "_sstw_block_attention_bias_scope_active"
    _callback_name = "sstw_block_attention_relation_bias_callback"

    def __init__(
        self,
        attention_block: object,
        *,
        descriptor: BlockAttentionRelationDescriptor | None = None,
        signed_coefficient: int,
        magnitude: float,
        expected_callback_invocations: int = 1,
    ) -> None:
        self._attention_block = attention_block
        self._descriptor = descriptor or build_block_attention_relation_descriptor()
        validate_block_attention_relation_descriptor(self._descriptor)
        self._signed_coefficient = signed_coefficient
        self._magnitude = float(magnitude)
        self._expected_callback_invocations = int(expected_callback_invocations)
        if self._expected_callback_invocations <= 0:
            raise ValueError("expected_callback_invocations 必须为正数")
        self._previous_callback_present = False
        self._previous_callback: object | None = None
        self._entered = False
        self._exit_attempted = False
        self._completed_successfully = False
        self._callback_invocation_count = 0
        self._application_record: BlockAttentionBiasApplicationRecord | None = None

    def __enter__(self) -> "ScopedBlockLocalAttentionBiasAdapter":
        if self._entered:
            raise RuntimeError("block-attention bias scope 不允许重复进入")
        if getattr(self._attention_block, self._active_marker_name, False):
            raise RuntimeError("block-attention bias scope 不允许嵌套")
        self._previous_callback_present = hasattr(
            self._attention_block,
            self._callback_name,
        )
        self._previous_callback = getattr(
            self._attention_block,
            self._callback_name,
            None,
        )
        setattr(self._attention_block, self._active_marker_name, True)
        setattr(
            self._attention_block,
            self._callback_name,
            self._build_callback(),
        )
        self._entered = True
        return self

    def _build_callback(self):
        def _callback(
            *,
            block_index: int,
        ) -> tuple[np.ndarray, BlockAttentionBiasApplicationRecord]:
            if block_index != self._descriptor.target_block_index:
                raise ValueError("block-attention callback block_index 漂移")
            values, record = apply_block_attention_sparse_bias_runtime(
                self._descriptor,
                signed_coefficient=self._signed_coefficient,
                magnitude=self._magnitude,
            )
            self._callback_invocation_count += 1
            self._application_record = record
            return values, record

        return _callback

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if self._exit_attempted:
            self._completed_successfully = False
            raise RuntimeError("block-attention bias scope 不允许重复退出")
        self._exit_attempted = True
        cleanup_errors: list[str] = []
        try:
            if self._previous_callback_present:
                setattr(
                    self._attention_block,
                    self._callback_name,
                    self._previous_callback,
                )
            elif hasattr(self._attention_block, self._callback_name):
                delattr(self._attention_block, self._callback_name)
        except Exception as error:  # pragma: no cover - defensive branch
            cleanup_errors.append(type(error).__name__)
        try:
            if hasattr(self._attention_block, self._active_marker_name):
                delattr(self._attention_block, self._active_marker_name)
        except Exception as error:  # pragma: no cover - defensive branch
            cleanup_errors.append(type(error).__name__)
        if cleanup_errors and exc is not None:
            exc.add_note(
                "block_attention_cleanup_errors="
                + ",".join(cleanup_errors)
            )
        elif cleanup_errors:
            raise RuntimeError(
                "block-attention bias scope cleanup 失败: "
                + ",".join(cleanup_errors)
            )
        if exc_type is not None:
            self._completed_successfully = False
            return False
        self._completed_successfully = (
            self._callback_invocation_count
            == self._expected_callback_invocations
            and self._application_record is not None
        )
        if not self._completed_successfully:
            raise RuntimeError(
                "block-attention bias callback coverage 与冻结合同不一致"
            )
        return False

    def record(self) -> BlockAttentionScopeRecord:
        if not self._completed_successfully or self._application_record is None:
            raise RuntimeError("block-attention bias scope 未成功完成")
        return BlockAttentionScopeRecord(
            descriptor_digest=self._descriptor.descriptor_digest,
            target_block_index=self._descriptor.target_block_index,
            signed_coefficient=self._signed_coefficient,
            magnitude=self._magnitude,
            callback_invocation_count=self._callback_invocation_count,
            scope_completed_successfully=True,
            scheduler_step_call_count=SCHEDULER_STEP_CALL_COUNT,
            decode_executed=False,
            video_export_executed=False,
            gate0_executed=False,
            application_record=self._application_record,
        )


class _WanBlockLocalAttentionBiasProcessor:
    """Minimal Wan self-attention processor with a sparse additive QK mask."""

    def __init__(
        self,
        *,
        original_processor: Any,
        target_attention: Any,
        descriptor: BlockAttentionRelationDescriptor,
        signed_coefficient: int,
        magnitude: float,
        input_binding_digest: str,
        cfg_branch_role: str,
        owner: "ScopedWanBlockLocalAttentionBiasAdapter",
    ) -> None:
        self.original_processor = original_processor
        self.target_attention = target_attention
        self.descriptor = descriptor
        self.signed_coefficient = signed_coefficient
        self.magnitude = magnitude
        self.input_binding_digest = input_binding_digest
        self.cfg_branch_role = cfg_branch_role
        self.owner = owner

    def _sparse_bias_record(self) -> BlockAttentionBiasApplicationRecord:
        values, record = apply_block_attention_sparse_bias_runtime(
            self.descriptor,
            signed_coefficient=self.signed_coefficient,
            magnitude=self.magnitude,
        )
        del values
        return record

    def _sparse_bias_by_query_head(
        self,
    ) -> tuple[dict[tuple[int, int], dict[int, float]], BlockAttentionBiasApplicationRecord]:
        values, record = apply_block_attention_sparse_bias_runtime(
            self.descriptor,
            signed_coefficient=self.signed_coefficient,
            magnitude=self.magnitude,
        )
        bias_by_query_head: dict[tuple[int, int], dict[int, float]] = {}
        for value, entry in zip(values.tolist(), self.descriptor.entries):
            if value == 0.0:
                continue
            row_key = (int(entry.query_token_index), int(entry.head_index))
            key_biases = bias_by_query_head.setdefault(row_key, {})
            key_biases[int(entry.key_token_index)] = (
                key_biases.get(int(entry.key_token_index), 0.0) + float(value)
            )
        return bias_by_query_head, record

    def _validate_query_layout(self, query: Any) -> None:
        if int(query.shape[1]) != WAN_SELF_ATTENTION_SEQUENCE_LENGTH:
            raise RuntimeError("Wan block-attention seq length 与冻结合同不一致")
        if int(query.shape[2]) != WAN_SELF_ATTENTION_HEAD_COUNT:
            raise RuntimeError("Wan block-attention head count 与冻结合同不一致")

    def _apply_sparse_row_bias(
        self,
        *,
        attention_rows: Any,
        query: Any,
        key: Any,
        value: Any,
        bias_by_query_head: dict[tuple[int, int], dict[int, float]],
    ) -> Any:
        if not bias_by_query_head:
            return attention_rows
        torch = _torch_module_from_tensor(query)
        patched = attention_rows.clone()
        scale = 1.0 / math.sqrt(float(query.shape[-1]))
        for (query_index, head_index), key_biases in bias_by_query_head.items():
            query_row = query[0, query_index, head_index, :].float()
            key_rows = key[0, :, head_index, :].float()
            logits = torch.matmul(key_rows, query_row) * scale
            for key_index, bias in key_biases.items():
                logits[key_index] = logits[key_index] + float(bias)
            weights = torch.softmax(logits, dim=-1)
            value_rows = value[0, :, head_index, :].float()
            patched[0, query_index, head_index, :] = torch.matmul(
                weights,
                value_rows,
            ).to(dtype=patched.dtype)
        return patched

    def __call__(
        self,
        attn: Any,
        hidden_states: Any,
        encoder_hidden_states: Any = None,
        attention_mask: Any = None,
        rotary_emb: Any = None,
        **kwargs: Any,
    ) -> Any:
        if kwargs:
            raise RuntimeError("Wan block-attention processor 收到未冻结kwargs")
        if attn is not self.target_attention:
            raise RuntimeError("Wan block-attention target attention 漂移")
        if encoder_hidden_states is not None or attention_mask is not None:
            raise RuntimeError("Wan block-attention 只允许self-attention无mask路径")
        if rotary_emb is None:
            raise RuntimeError("Wan block-attention self-attention 缺少RoPE")
        torch = _torch_module_from_tensor(hidden_states)
        wan_mod = _diffusers_wan_transformer_module()
        query, key, value = wan_mod._get_qkv_projections(
            attn,
            hidden_states,
            None,
        )
        query = attn.norm_q(query)
        key = attn.norm_k(key)
        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))
        query = _apply_wan_rotary_emb_torch(query, rotary_emb)
        key = _apply_wan_rotary_emb_torch(key, rotary_emb)
        self._validate_query_layout(query)
        bias_by_query_head, record = self._sparse_bias_by_query_head()
        attention_rows = wan_mod.dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            backend=getattr(self.original_processor, "_attention_backend", None),
        )
        attention_rows = self._apply_sparse_row_bias(
            attention_rows=attention_rows,
            query=query,
            key=key,
            value=value,
            bias_by_query_head=bias_by_query_head,
        )
        hidden_states = attention_rows.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        self.owner._register_processor_call(
            attention_mask_shape=(
                1,
                WAN_SELF_ATTENTION_HEAD_COUNT,
                WAN_SELF_ATTENTION_SEQUENCE_LENGTH,
                WAN_SELF_ATTENTION_SEQUENCE_LENGTH,
            ),
            attention_mask_dtype=(
                "sparse_row_bias_no_dense_attention_mask_allocated:"
                + str(query.dtype)
            ),
            application_record=record,
        )
        return hidden_states


class ScopedWanBlockLocalAttentionBiasAdapter:
    """Patch Wan ``blocks[target].attn1.processor`` for one transformer call."""

    def __init__(
        self,
        transformer: Any,
        *,
        descriptor: BlockAttentionRelationDescriptor | None = None,
        signed_coefficient: int,
        magnitude: float,
        input_binding_digest: str,
        cfg_branch_role: str,
    ) -> None:
        self.transformer = transformer
        self.descriptor = descriptor or build_block_attention_relation_descriptor()
        validate_block_attention_relation_descriptor(self.descriptor)
        self.signed_coefficient = int(signed_coefficient)
        self.magnitude = _require_frozen_candidate_magnitude(magnitude)
        self.input_binding_digest = str(input_binding_digest)
        self.cfg_branch_role = str(cfg_branch_role)
        self._entered = False
        self._exit_attempted = False
        self._completed_successfully = False
        self._target_attention: Any = None
        self._original_processor: Any = None
        self._processor_call_count = 0
        self._attention_mask_shape: tuple[int, int, int, int] | None = None
        self._attention_mask_dtype = ""
        self._application_record: BlockAttentionBiasApplicationRecord | None = None

    def __enter__(self) -> "ScopedWanBlockLocalAttentionBiasAdapter":
        if self._entered:
            raise RuntimeError("Wan block-attention scope 不得重复进入")
        blocks = getattr(self.transformer, "blocks", None)
        if blocks is None or len(blocks) <= self.descriptor.target_block_index:
            raise RuntimeError("Wan transformer blocks 与冻结target block不一致")
        target_block = blocks[self.descriptor.target_block_index]
        target_attention = getattr(target_block, "attn1", None)
        if target_attention is None:
            raise RuntimeError("Wan target block 缺少attn1 self-attention")
        if getattr(target_attention, "_sstw_block_attention_scope_active", False):
            raise RuntimeError("Wan block-attention scope 不允许嵌套")
        self._target_attention = target_attention
        self._original_processor = getattr(target_attention, "processor", None)
        if self._original_processor is None:
            raise RuntimeError("Wan target attention 缺少processor")
        processor = _WanBlockLocalAttentionBiasProcessor(
            original_processor=self._original_processor,
            target_attention=target_attention,
            descriptor=self.descriptor,
            signed_coefficient=self.signed_coefficient,
            magnitude=self.magnitude,
            input_binding_digest=self.input_binding_digest,
            cfg_branch_role=self.cfg_branch_role,
            owner=self,
        )
        setattr(target_attention, "_sstw_block_attention_scope_active", True)
        try:
            target_attention.set_processor(processor)
        except BaseException:
            try:
                if self._original_processor is not None:
                    target_attention.set_processor(self._original_processor)
            finally:
                if hasattr(
                    target_attention,
                    "_sstw_block_attention_scope_active",
                ):
                    delattr(
                        target_attention,
                        "_sstw_block_attention_scope_active",
                    )
                self._target_attention = None
                self._original_processor = None
            raise
        self._entered = True
        return self

    def _register_processor_call(
        self,
        *,
        attention_mask_shape: tuple[int, int, int, int],
        attention_mask_dtype: str,
        application_record: BlockAttentionBiasApplicationRecord,
    ) -> None:
        self._processor_call_count += 1
        self._attention_mask_shape = attention_mask_shape
        self._attention_mask_dtype = attention_mask_dtype
        self._application_record = application_record

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if self._exit_attempted:
            self._completed_successfully = False
            raise RuntimeError("Wan block-attention scope 不得重复退出")
        self._exit_attempted = True
        cleanup_errors: list[str] = []
        try:
            if self._target_attention is not None and self._original_processor is not None:
                self._target_attention.set_processor(self._original_processor)
        except Exception as error:  # pragma: no cover - defensive branch
            cleanup_errors.append(type(error).__name__)
        try:
            if self._target_attention is not None and hasattr(
                self._target_attention,
                "_sstw_block_attention_scope_active",
            ):
                delattr(
                    self._target_attention,
                    "_sstw_block_attention_scope_active",
                )
        except Exception as error:  # pragma: no cover - defensive branch
            cleanup_errors.append(type(error).__name__)
        if cleanup_errors and exc is not None:
            exc.add_note(
                "wan_block_attention_cleanup_errors="
                + ",".join(cleanup_errors)
            )
        elif cleanup_errors:
            raise RuntimeError(
                "Wan block-attention scope cleanup 失败: "
                + ",".join(cleanup_errors)
            )
        if exc_type is not None:
            self._completed_successfully = False
            return False
        self._completed_successfully = (
            self._processor_call_count == WAN_RUNTIME_EXPECTED_PROCESSOR_CALLS
            and self._application_record is not None
            and self._attention_mask_shape
            == (
                1,
                WAN_SELF_ATTENTION_HEAD_COUNT,
                WAN_SELF_ATTENTION_SEQUENCE_LENGTH,
                WAN_SELF_ATTENTION_SEQUENCE_LENGTH,
            )
        )
        if not self._completed_successfully:
            raise RuntimeError("Wan block-attention processor coverage 不完整")
        return False

    def record(self) -> WanBlockAttentionBranchApplicationRecord:
        if (
            not self._completed_successfully
            or self._application_record is None
            or self._attention_mask_shape is None
        ):
            raise RuntimeError("Wan block-attention scope 未成功完成")
        return WanBlockAttentionBranchApplicationRecord(
            descriptor_digest=self.descriptor.descriptor_digest,
            target_block_index=self.descriptor.target_block_index,
            signed_coefficient=self.signed_coefficient,
            magnitude=self.magnitude,
            input_binding_digest=self.input_binding_digest,
            cfg_branch_role=self.cfg_branch_role,
            processor_call_count=self._processor_call_count,
            scope_completed_successfully=True,
            attention_mask_shape=self._attention_mask_shape,
            attention_mask_dtype=self._attention_mask_dtype,
            changed_entry_count=self._application_record.changed_entry_count,
            bias_l2_norm=self._application_record.bias_l2_norm,
            bias_digest=self._application_record.bias_digest,
            clean_exact_no_op=self._application_record.clean_exact_no_op,
        )


def _torch_module_from_tensor(tensor: Any) -> Any:
    module = type(tensor).__module__.split(".", 1)[0]
    if module != "torch":
        raise TypeError("Wan runtime tensor 必须来自torch")
    import torch

    return torch


def _diffusers_wan_transformer_module() -> Any:
    from diffusers.models.transformers import transformer_wan

    return transformer_wan


def _apply_wan_rotary_emb_torch(hidden_states: Any, rotary_emb: Any) -> Any:
    freqs_cos, freqs_sin = rotary_emb
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    out = hidden_states.new_empty(hidden_states.shape)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.type_as(hidden_states)


def _require_float64_vector(value: np.ndarray, label: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{label} 必须是 numpy.ndarray")
    if value.dtype != np.dtype("<f8"):
        raise TypeError(f"{label} dtype 必须精确为 little-endian float64")
    if value.ndim != 1:
        raise ValueError(f"{label} 必须是一维向量")
    if not value.flags.c_contiguous:
        raise ValueError(f"{label} 必须是 C-contiguous")
    if not np.isfinite(value).all():
        raise ValueError(f"{label} 必须全部 finite")
    return value


def _digest_float64_vector(value: np.ndarray) -> str:
    array = _require_float64_vector(value, "digest vector")
    return sha256(array.tobytes(order="C")).hexdigest()


def _tuple_to_float64_vector(
    value: tuple[float, ...],
    *,
    expected_length: int,
    label: str,
) -> np.ndarray:
    if not isinstance(value, tuple):
        raise TypeError(f"{label} 必须是tuple")
    array = np.asarray(value, dtype="<f8")
    _require_float64_vector(array, label)
    if array.shape != (expected_length,):
        raise ValueError(f"{label} length 与冻结entry count 不一致")
    return array


def apply_block_attention_sparse_bias_runtime(
    descriptor: BlockAttentionRelationDescriptor,
    *,
    signed_coefficient: int,
    magnitude: float,
) -> tuple[np.ndarray, BlockAttentionBiasApplicationRecord]:
    """Build the sparse pre-softmax QK bias vector in descriptor order."""

    validate_block_attention_relation_descriptor(descriptor)
    frozen_magnitude = _require_frozen_candidate_magnitude(magnitude)
    values = signed_sparse_bias_values(
        descriptor,
        signed_coefficient=signed_coefficient,
        magnitude=frozen_magnitude,
    )
    changed_count = int(np.count_nonzero(values))
    record = BlockAttentionBiasApplicationRecord(
        descriptor_digest=descriptor.descriptor_digest,
        signed_coefficient=signed_coefficient,
        magnitude=frozen_magnitude,
        entry_count=len(descriptor.entries),
        changed_entry_count=changed_count,
        bias_l2_norm=float(np.linalg.norm(values.astype(np.float64))),
        bias_digest=_digest_float64_vector(values),
        clean_exact_no_op=bool(signed_coefficient == 0 and changed_count == 0),
    )
    return values, record


def _safe_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a64 = _require_float64_vector(a, "cosine a").astype(np.float64)
    b64 = _require_float64_vector(b, "cosine b").astype(np.float64)
    denom = float(np.linalg.norm(a64) * np.linalg.norm(b64))
    if denom <= 0.0:
        return 0.0
    observed = float(np.dot(a64, b64) / denom)
    if not math.isfinite(observed):
        raise ValueError("cosine 必须 finite")
    if observed < -1.0 - 1.0e-12 or observed > 1.0 + 1.0e-12:
        raise ValueError("cosine 超出机器舍入容许范围")
    return max(-1.0, min(1.0, observed))


def evaluate_block_attention_candidate_response(
    descriptor: BlockAttentionRelationDescriptor,
    *,
    candidate_index: int,
    signed_coefficient: int,
    magnitude: float,
    response_vector: np.ndarray,
    repeat_delta_vector: np.ndarray,
) -> BlockAttentionCandidateResponse:
    """Evaluate one local candidate without trusting caller statistics."""

    frozen_magnitude = _require_frozen_candidate_magnitude(magnitude)
    response = _require_float64_vector(response_vector, "response_vector")
    repeat_delta = _require_float64_vector(
        repeat_delta_vector,
        "repeat_delta_vector",
    )
    if response.shape != (len(descriptor.entries),):
        raise ValueError("response_vector shape 与 descriptor entry count 不一致")
    if repeat_delta.shape != response.shape:
        raise ValueError("repeat_delta_vector shape 与 response_vector 不一致")
    _, application = apply_block_attention_sparse_bias_runtime(
        descriptor,
        signed_coefficient=signed_coefficient,
        magnitude=frozen_magnitude,
    )
    response_norm = float(np.linalg.norm(response.astype(np.float64)))
    repeat_norm = float(np.linalg.norm(repeat_delta.astype(np.float64)))
    denominator = max(response_norm, NONZERO_RESPONSE_FLOOR)
    repeat_ratio = repeat_norm / denominator
    norm_guard = response_norm <= STEP0_NORM_BUDGET
    repeatable = bool(
        response_norm > NONZERO_RESPONSE_FLOOR
        and repeat_ratio <= REPEATABILITY_FLOOR_RATIO_THRESHOLD
    )
    return BlockAttentionCandidateResponse(
        candidate_index=int(candidate_index),
        signed_coefficient=int(signed_coefficient),
        magnitude=frozen_magnitude,
        response_vector=tuple(float(item) for item in response.tolist()),
        repeat_delta_vector=tuple(float(item) for item in repeat_delta.tolist()),
        response_l2_norm=response_norm,
        norm_budget=STEP0_NORM_BUDGET,
        norm_guard_passed=bool(norm_guard),
        repeat_l2_norm=repeat_norm,
        repeat_floor_ratio=float(repeat_ratio),
        nonzero_response=bool(response_norm > NONZERO_RESPONSE_FLOOR),
        repeatable_above_floor=repeatable,
        near_antisymmetric_pair=False,
        feasible_candidate=False,
        application_record=application,
    )


def build_local_block_attention_primitive_response_record(
    *,
    descriptor: BlockAttentionRelationDescriptor | None = None,
    response_gain: float = 1.0,
    repeat_floor: float = 0.0,
) -> BlockAttentionPrimitiveResponseRecord:
    """Build a deterministic local primitive-response record.

    ``response_gain`` and ``repeat_floor`` are test-time local primitives for
    validating the record/classifier plumbing.  They are not Wan measurements
    and are never evidence for method effectiveness.
    """

    if descriptor is None:
        descriptor = build_block_attention_relation_descriptor()
    validate_block_attention_relation_descriptor(descriptor)
    if not math.isfinite(response_gain) or response_gain < 0.0:
        raise ValueError("response_gain 必须是有限非负数")
    if not math.isfinite(repeat_floor) or repeat_floor < 0.0:
        raise ValueError("repeat_floor 必须是有限非负数")

    candidate_records: list[BlockAttentionCandidateResponse] = []
    response_vectors: dict[tuple[float, int], np.ndarray] = {}
    candidate_index = 0
    for magnitude in CANDIDATE_BIAS_MAGNITUDES:
        for sign in CANDIDATE_SIGNS:
            sparse_bias, _ = apply_block_attention_sparse_bias_runtime(
                descriptor,
                signed_coefficient=sign,
                magnitude=magnitude,
            )
            response = (response_gain * sparse_bias).astype("<f8", copy=True)
            repeat_delta = np.full(
                response.shape,
                repeat_floor,
                dtype="<f8",
            )
            candidate = evaluate_block_attention_candidate_response(
                descriptor,
                candidate_index=candidate_index,
                signed_coefficient=sign,
                magnitude=magnitude,
                response_vector=response,
                repeat_delta_vector=repeat_delta,
            )
            candidate_records.append(candidate)
            response_vectors[(magnitude, sign)] = response
            candidate_index += 1

    antipodal_cosines: list[float] = []
    for magnitude in CANDIDATE_BIAS_MAGNITUDES:
        cosine = _safe_cosine(
            response_vectors[(magnitude, 1)],
            response_vectors[(magnitude, -1)],
        )
        antipodal_cosines.append(cosine)
        near_pair = cosine <= NEAR_ANTISYMMETRY_COSINE_THRESHOLD
        for index, candidate in enumerate(candidate_records):
            if candidate.magnitude == magnitude:
                candidate_records[index] = BlockAttentionCandidateResponse(
                    candidate_index=candidate.candidate_index,
                    signed_coefficient=candidate.signed_coefficient,
                    magnitude=candidate.magnitude,
                    response_vector=candidate.response_vector,
                    repeat_delta_vector=candidate.repeat_delta_vector,
                    response_l2_norm=candidate.response_l2_norm,
                    norm_budget=candidate.norm_budget,
                    norm_guard_passed=candidate.norm_guard_passed,
                    repeat_l2_norm=candidate.repeat_l2_norm,
                    repeat_floor_ratio=candidate.repeat_floor_ratio,
                    nonzero_response=candidate.nonzero_response,
                    repeatable_above_floor=candidate.repeatable_above_floor,
                    near_antisymmetric_pair=bool(near_pair),
                    feasible_candidate=bool(
                        near_pair
                        and candidate.norm_guard_passed
                        and candidate.repeatable_above_floor
                    ),
                    application_record=candidate.application_record,
                )
    classification = classify_block_attention_primitive_response(
        tuple(candidate_records),
        tuple(antipodal_cosines),
    )
    return BlockAttentionPrimitiveResponseRecord(
        record_kind="patch_relation_block_attention_primitive_response_preflight",
        descriptor_digest=descriptor.descriptor_digest,
        zero_repeat_count=ZERO_CONTROL_REPEAT_COUNT,
        scheduler_step_call_count=SCHEDULER_STEP_CALL_COUNT,
        decode_executed=False,
        video_export_executed=False,
        gate0_executed=False,
        candidate_responses=tuple(candidate_records),
        positive_negative_response_cosine_by_magnitude=tuple(
            antipodal_cosines
        ),
        diagnostic_classification=classification,
        formal_result=False,
        stage_progression_allowed=False,
        claim_support_status=(
            "block_attention_local_primitive_response_only_not_gpu_gate_or_"
            "method_evidence"
        ),
    )


def classify_block_attention_primitive_response(
    candidates: tuple[BlockAttentionCandidateResponse, ...],
    antipodal_cosines: tuple[float, ...],
) -> str:
    if len(candidates) != len(CANDIDATE_BIAS_MAGNITUDES) * len(CANDIDATE_SIGNS):
        raise ValueError("candidate coverage 与冻结合同不一致")
    if len(antipodal_cosines) != len(CANDIDATE_BIAS_MAGNITUDES):
        raise ValueError("antipodal cosine coverage 与冻结合同不一致")
    if any(candidate.repeat_floor_ratio > REPEATABILITY_FLOOR_RATIO_THRESHOLD for candidate in candidates if candidate.nonzero_response):
        return "repeatability_floor_candidate"
    if not any(candidate.nonzero_response for candidate in candidates):
        return "no_budgeted_response_candidate"
    if any(candidate.feasible_candidate for candidate in candidates) and all(
        cos <= NEAR_ANTISYMMETRY_COSINE_THRESHOLD
        for cos in antipodal_cosines
    ):
        return "feasible_nonzero_near_antisymmetric_response_candidate"
    if not any(candidate.norm_guard_passed for candidate in candidates):
        return "no_budgeted_response_candidate"
    return "indeterminate"


def validate_block_attention_primitive_response_record(
    record: BlockAttentionPrimitiveResponseRecord,
) -> None:
    descriptor = build_block_attention_relation_descriptor()
    if record.descriptor_digest != descriptor.descriptor_digest:
        raise ValueError("primitive response descriptor digest 漂移")
    if (
        record.zero_repeat_count != ZERO_CONTROL_REPEAT_COUNT
        or record.scheduler_step_call_count != 0
        or record.decode_executed
        or record.video_export_executed
        or record.gate0_executed
        or record.formal_result
        or record.stage_progression_allowed
    ):
        raise ValueError("primitive response execution boundary 漂移")
    expected = classify_block_attention_primitive_response(
        record.candidate_responses,
        record.positive_negative_response_cosine_by_magnitude,
    )
    if record.diagnostic_classification != expected:
        raise ValueError("primitive response classification 自报值不一致")
    if record.claim_support_status != (
        "block_attention_local_primitive_response_only_not_gpu_gate_or_"
        "method_evidence"
    ):
        raise ValueError("primitive response claim boundary 漂移")
    expected_candidate_count = len(CANDIDATE_BIAS_MAGNITUDES) * len(CANDIDATE_SIGNS)
    if len(record.candidate_responses) != expected_candidate_count:
        raise ValueError("primitive response candidate coverage 漂移")
    recomputed_cosines_by_magnitude: list[float] = []
    response_by_magnitude_and_sign: dict[tuple[float, int], np.ndarray] = {}
    expected_index = 0
    for magnitude in CANDIDATE_BIAS_MAGNITUDES:
        magnitude_candidates = [
            candidate
            for candidate in record.candidate_responses
            if candidate.magnitude == magnitude
        ]
        if len(magnitude_candidates) != len(CANDIDATE_SIGNS):
            raise ValueError("primitive response magnitude/sign coverage 漂移")
        for sign in CANDIDATE_SIGNS:
            candidate = record.candidate_responses[expected_index]
            if (
                candidate.candidate_index != expected_index
                or candidate.signed_coefficient != sign
                or candidate.magnitude != magnitude
            ):
                raise ValueError("primitive response candidate order 漂移")
            _, expected_application = apply_block_attention_sparse_bias_runtime(
                descriptor,
                signed_coefficient=sign,
                magnitude=magnitude,
            )
            if candidate.application_record != expected_application:
                raise ValueError("primitive response application record 漂移")
            response = _tuple_to_float64_vector(
                candidate.response_vector,
                expected_length=len(descriptor.entries),
                label="candidate response_vector",
            )
            repeat_delta = _tuple_to_float64_vector(
                candidate.repeat_delta_vector,
                expected_length=len(descriptor.entries),
                label="candidate repeat_delta_vector",
            )
            response_by_magnitude_and_sign[(magnitude, sign)] = response
            recomputed_response_norm = float(
                np.linalg.norm(response.astype(np.float64))
            )
            if not math.isclose(
                candidate.response_l2_norm,
                recomputed_response_norm,
                rel_tol=0.0,
                abs_tol=1.0e-15,
            ):
                raise ValueError("primitive response norm 自报值不一致")
            recomputed_repeat_norm = float(
                np.linalg.norm(repeat_delta.astype(np.float64))
            )
            if not math.isclose(
                candidate.repeat_l2_norm,
                recomputed_repeat_norm,
                rel_tol=0.0,
                abs_tol=1.0e-15,
            ):
                raise ValueError("primitive response repeat norm 自报值不一致")
            if candidate.norm_budget != STEP0_NORM_BUDGET:
                raise ValueError("primitive response norm budget 漂移")
            expected_norm_guard = candidate.response_l2_norm <= STEP0_NORM_BUDGET
            if candidate.norm_guard_passed is not expected_norm_guard:
                raise ValueError("primitive response norm guard 自报值不一致")
            if candidate.response_l2_norm < 0.0 or candidate.repeat_l2_norm < 0.0:
                raise ValueError("primitive response norms 必须非负")
            expected_nonzero = candidate.response_l2_norm > NONZERO_RESPONSE_FLOOR
            if candidate.nonzero_response is not expected_nonzero:
                raise ValueError("primitive response nonzero 自报值不一致")
            expected_repeat_ratio = candidate.repeat_l2_norm / max(
                candidate.response_l2_norm,
                NONZERO_RESPONSE_FLOOR,
            )
            if not math.isclose(
                candidate.repeat_floor_ratio,
                expected_repeat_ratio,
                rel_tol=0.0,
                abs_tol=1.0e-15,
            ):
                raise ValueError("primitive response repeat ratio 自报值不一致")
            expected_repeatable = (
                expected_nonzero
                and expected_repeat_ratio <= REPEATABILITY_FLOOR_RATIO_THRESHOLD
            )
            if candidate.repeatable_above_floor is not expected_repeatable:
                raise ValueError("primitive response repeatability 自报值不一致")
            pair_index = CANDIDATE_BIAS_MAGNITUDES.index(magnitude)
            expected_near_pair = (
                record.positive_negative_response_cosine_by_magnitude[pair_index]
                <= NEAR_ANTISYMMETRY_COSINE_THRESHOLD
            )
            if candidate.near_antisymmetric_pair is not expected_near_pair:
                raise ValueError("primitive response antisymmetry 自报值不一致")
            expected_feasible = (
                expected_near_pair
                and expected_norm_guard
                and expected_repeatable
            )
            if candidate.feasible_candidate is not expected_feasible:
                raise ValueError("primitive response feasible 自报值不一致")
            expected_index += 1
        recomputed_cosines_by_magnitude.append(
            _safe_cosine(
                response_by_magnitude_and_sign[(magnitude, 1)],
                response_by_magnitude_and_sign[(magnitude, -1)],
            )
        )
    if tuple(recomputed_cosines_by_magnitude) != (
        record.positive_negative_response_cosine_by_magnitude
    ):
        raise ValueError("primitive response antisymmetry cosine 自报值不一致")
