"""Tests for the block-local Patch-relation attention preflight contract."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from evaluation.protocol.patch_relation_block_attention_response_preflight_contract import (
    DEFAULT_CONFIG_PATH,
    FROZEN_PROTOCOL_DIGEST,
    load_patch_relation_block_attention_response_preflight_config,
    protocol_digest,
)
from main.methods.state_space_watermark.patch_relation_block_attention import (
    ACTIVE_TOKEN_TIME_INDICES,
    HEAD_GROUP_INDICES,
    MAXIMUM_LOGIT_BIAS_MAGNITUDE,
    PATCH_TOKEN_A,
    PATCH_TOKEN_B,
    TOKEN_GRID_SHAPE,
    build_block_attention_relation_descriptor,
    signed_sparse_bias_values,
    token_index,
    validate_block_attention_relation_descriptor,
)
from workflows.colab_test_request import (
    PATCH_RELATION_BLOCK_ATTENTION_RESPONSE_PREFLIGHT_TEST_ID,
    build_colab_test_dry_run_plan,
    load_colab_test_request,
)


@pytest.mark.quick
def test_block_attention_config_loads_with_frozen_digest() -> None:
    config = load_patch_relation_block_attention_response_preflight_config()
    assert config["protocol_digest"] == FROZEN_PROTOCOL_DIGEST
    assert protocol_digest(config["protocol_contract"]) == FROZEN_PROTOCOL_DIGEST
    assert config["authorization_boundary"]["colab_execution_allowed"] is False
    assert (
        config["protocol_contract"]["result_boundary"][
            "claim_support_status"
        ]
        == "block_attention_local_contract_only_not_runtime_gate_or_method_evidence"
    )


@pytest.mark.quick
def test_block_attention_config_mutations_are_rejected(tmp_path: Path) -> None:
    payload = json.loads(Path(DEFAULT_CONFIG_PATH).read_text(encoding="utf-8"))
    payload["protocol_contract"]["block_local_attention_control_contract"][
        "target_block_index"
    ] = 19
    payload["protocol_digest"] = protocol_digest(payload["protocol_contract"])
    mutated = tmp_path / "mutated.json"
    mutated.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen digest|control contract"):
        load_patch_relation_block_attention_response_preflight_config(mutated)

    payload = json.loads(Path(DEFAULT_CONFIG_PATH).read_text(encoding="utf-8"))
    payload["authorization_boundary"]["colab_execution_allowed"] = True
    mutated.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="authorization boundary"):
        load_patch_relation_block_attention_response_preflight_config(mutated)


@pytest.mark.quick
def test_descriptor_freezes_sparse_zero_sum_entry_order() -> None:
    descriptor = build_block_attention_relation_descriptor()
    validate_block_attention_relation_descriptor(descriptor)
    assert descriptor.target_block_index == 18
    assert descriptor.token_grid_shape == TOKEN_GRID_SHAPE
    assert len(descriptor.entries) == (
        len(ACTIVE_TOKEN_TIME_INDICES) * len(HEAD_GROUP_INDICES) * 2
    )
    first = descriptor.entries[0]
    assert first.time_index == ACTIVE_TOKEN_TIME_INDICES[0]
    assert first.head_index == HEAD_GROUP_INDICES[0]
    assert first.query_token_index == token_index(
        ACTIVE_TOKEN_TIME_INDICES[0],
        *PATCH_TOKEN_A,
    )
    assert first.key_token_index == token_index(
        ACTIVE_TOKEN_TIME_INDICES[0],
        *PATCH_TOKEN_B,
    )
    assert first.coefficient == 1.0
    for offset in range(0, len(descriptor.entries), 2):
        positive = descriptor.entries[offset]
        negative = descriptor.entries[offset + 1]
        assert positive.time_index == negative.time_index
        assert positive.head_index == negative.head_index
        assert positive.coefficient == 1.0
        assert negative.coefficient == -1.0
        assert positive.query_token_index == negative.key_token_index
        assert positive.key_token_index == negative.query_token_index


@pytest.mark.quick
def test_signed_sparse_values_are_exact_antipodes() -> None:
    descriptor = build_block_attention_relation_descriptor()
    positive = signed_sparse_bias_values(
        descriptor,
        signed_coefficient=1,
    )
    negative = signed_sparse_bias_values(
        descriptor,
        signed_coefficient=-1,
    )
    clean = signed_sparse_bias_values(
        descriptor,
        signed_coefficient=0,
    )
    assert positive.dtype == np.dtype("<f8")
    assert positive.flags.c_contiguous
    assert np.all(positive == -negative)
    assert np.all(clean == 0.0)
    assert set(np.unique(np.abs(positive))) == {
        MAXIMUM_LOGIT_BIAS_MAGNITUDE
    }


@pytest.mark.quick
def test_descriptor_mutation_rejected() -> None:
    descriptor = build_block_attention_relation_descriptor()
    entries = list(descriptor.entries)
    entries[0] = copy.copy(entries[1])
    mutated = descriptor.__class__(
        descriptor_id=descriptor.descriptor_id,
        target_block_index=descriptor.target_block_index,
        token_grid_shape=descriptor.token_grid_shape,
        active_token_time_indices=descriptor.active_token_time_indices,
        patch_token_a=descriptor.patch_token_a,
        patch_token_b=descriptor.patch_token_b,
        head_group_indices=descriptor.head_group_indices,
        entries=tuple(entries),
        descriptor_digest=descriptor.descriptor_digest,
    )
    with pytest.raises(ValueError, match="descriptor"):
        validate_block_attention_relation_descriptor(mutated)


def _write_request(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


@pytest.mark.quick
def test_block_attention_request_is_parseable_but_not_executable(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "SSTW"
    request_path = project_root / "requests" / "colab_test_request.json"
    payload: dict[str, object] = {
        "request_schema_version": "sstw_colab_test_request_v1",
        "test_id": PATCH_RELATION_BLOCK_ATTENTION_RESPONSE_PREFLIGHT_TEST_ID,
        "repository": {
            "url": "https://github.com/RICHAAARC/SSTW.git",
            "ref": "a" * 40,
        },
        "parameters": {
            "phase": "block_attention_response_preflight",
            "run_series_id": "patch_relation_block_attention_response_preflight",
            "source_package_path": "",
            "resume_package_path": "",
        },
    }
    _write_request(request_path, payload)
    resolved = load_colab_test_request(
        request_path,
        project_root=project_root,
    )
    assert resolved["test_id"] == (
        PATCH_RELATION_BLOCK_ATTENTION_RESPONSE_PREFLIGHT_TEST_ID
    )
    plan = build_colab_test_dry_run_plan(
        request_path,
        project_root=project_root,
    )
    assert plan["current_stage_execution_allowed"] is False
    assert plan["paused_historical"] is True
    assert plan["claim_support_status"] == (
        "block_attention_response_preflight_contract_only_not_runtime_or_"
        "method_evidence"
    )
