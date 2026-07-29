"""Local primitives for the block-local Patch-relation attention carrier.

The primitives here are deliberately runtime-light. They freeze the sparse
relation descriptor and its deterministic entry order for a future Wan
attention adapter, but they do not hook diffusers, run a model, or produce
method evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Iterable

import numpy as np


TOKEN_GRID_SHAPE = (9, 20, 32)
ACTIVE_TOKEN_TIME_INDICES = (3, 4, 5)
PATCH_TOKEN_A = (9, 13)
PATCH_TOKEN_B = (9, 18)
HEAD_GROUP_INDICES = (0, 1, 2, 3)
TARGET_BLOCK_INDEX = 18
MAXIMUM_LOGIT_BIAS_MAGNITUDE = 0.015625


@dataclass(frozen=True)
class SparseAttentionBiasEntry:
    """One pre-softmax QK-logit edit for a frozen local relation."""

    time_index: int
    head_index: int
    query_token_index: int
    key_token_index: int
    coefficient: float


@dataclass(frozen=True)
class BlockAttentionRelationDescriptor:
    """Candidate-key-independent descriptor for the next carrier family."""

    descriptor_id: str
    target_block_index: int
    token_grid_shape: tuple[int, int, int]
    active_token_time_indices: tuple[int, ...]
    patch_token_a: tuple[int, int]
    patch_token_b: tuple[int, int]
    head_group_indices: tuple[int, ...]
    entries: tuple[SparseAttentionBiasEntry, ...]
    descriptor_digest: str


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def token_index(
    time_index: int,
    row: int,
    column: int,
    *,
    token_grid_shape: tuple[int, int, int] = TOKEN_GRID_SHAPE,
) -> int:
    """Return Wan patch-token flat index using width-fastest C order."""

    time_count, row_count, column_count = token_grid_shape
    if not (0 <= time_index < time_count):
        raise ValueError("time_index 超出冻结token grid")
    if not (0 <= row < row_count):
        raise ValueError("row 超出冻结token grid")
    if not (0 <= column < column_count):
        raise ValueError("column 超出冻结token grid")
    return (time_index * row_count + row) * column_count + column


def _entry_payload(entry: SparseAttentionBiasEntry) -> dict[str, object]:
    return {
        "time_index": entry.time_index,
        "head_index": entry.head_index,
        "query_token_index": entry.query_token_index,
        "key_token_index": entry.key_token_index,
        "coefficient": entry.coefficient,
    }


def _descriptor_payload(
    *,
    entries: Iterable[SparseAttentionBiasEntry],
) -> dict[str, object]:
    return {
        "descriptor_id": (
            "wan21_block18_patch_pair_head_group_qk_logit_bias_relation_001"
        ),
        "target_block_index": TARGET_BLOCK_INDEX,
        "token_grid_shape": list(TOKEN_GRID_SHAPE),
        "active_token_time_indices": list(ACTIVE_TOKEN_TIME_INDICES),
        "patch_token_a": {"row": PATCH_TOKEN_A[0], "column": PATCH_TOKEN_A[1]},
        "patch_token_b": {"row": PATCH_TOKEN_B[0], "column": PATCH_TOKEN_B[1]},
        "head_group_indices": list(HEAD_GROUP_INDICES),
        "entry_order": [
            "time_index_ascending",
            "head_index_ascending",
            "directed_pair_order",
        ],
        "entries": [_entry_payload(entry) for entry in entries],
    }


def build_block_attention_relation_descriptor() -> (
    BlockAttentionRelationDescriptor
):
    """Build the frozen sparse descriptor without reading key/prompt/seed."""

    entries: list[SparseAttentionBiasEntry] = []
    for time_index in ACTIVE_TOKEN_TIME_INDICES:
        token_a = token_index(time_index, *PATCH_TOKEN_A)
        token_b = token_index(time_index, *PATCH_TOKEN_B)
        for head_index in HEAD_GROUP_INDICES:
            entries.append(
                SparseAttentionBiasEntry(
                    time_index=time_index,
                    head_index=head_index,
                    query_token_index=token_a,
                    key_token_index=token_b,
                    coefficient=1.0,
                )
            )
            entries.append(
                SparseAttentionBiasEntry(
                    time_index=time_index,
                    head_index=head_index,
                    query_token_index=token_b,
                    key_token_index=token_a,
                    coefficient=-1.0,
                )
            )
    entry_tuple = tuple(entries)
    digest = sha256(
        canonical_json_bytes(_descriptor_payload(entries=entry_tuple))
    ).hexdigest()
    return BlockAttentionRelationDescriptor(
        descriptor_id=(
            "wan21_block18_patch_pair_head_group_qk_logit_bias_relation_001"
        ),
        target_block_index=TARGET_BLOCK_INDEX,
        token_grid_shape=TOKEN_GRID_SHAPE,
        active_token_time_indices=ACTIVE_TOKEN_TIME_INDICES,
        patch_token_a=PATCH_TOKEN_A,
        patch_token_b=PATCH_TOKEN_B,
        head_group_indices=HEAD_GROUP_INDICES,
        entries=entry_tuple,
        descriptor_digest=digest,
    )


def signed_sparse_bias_values(
    descriptor: BlockAttentionRelationDescriptor,
    *,
    signed_coefficient: int,
    magnitude: float = MAXIMUM_LOGIT_BIAS_MAGNITUDE,
) -> np.ndarray:
    """Return little-endian float64 sparse values in descriptor entry order."""

    if signed_coefficient not in {-1, 0, 1}:
        raise ValueError("signed_coefficient 必须是 -1, 0 或 1")
    if not math.isfinite(float(magnitude)) or float(magnitude) <= 0.0:
        raise ValueError("magnitude 必须是有限正数")
    values = np.asarray(
        [
            signed_coefficient * float(magnitude) * entry.coefficient
            for entry in descriptor.entries
        ],
        dtype="<f8",
    )
    if not values.flags.c_contiguous:
        raise ValueError("sparse bias values 必须是C-contiguous")
    return values


def validate_block_attention_relation_descriptor(
    descriptor: BlockAttentionRelationDescriptor,
) -> None:
    """Fail closed on descriptor drift or hidden dense/global carrier changes."""

    expected = build_block_attention_relation_descriptor()
    if descriptor != expected:
        raise ValueError("block-local attention descriptor 与冻结合同不一致")
    per_time_head: dict[tuple[int, int], float] = {}
    for entry in descriptor.entries:
        key = (entry.time_index, entry.head_index)
        per_time_head[key] = per_time_head.get(key, 0.0) + entry.coefficient
    if any(total != 0.0 for total in per_time_head.values()):
        raise ValueError("block-local attention relation 必须逐time/head zero-sum")
