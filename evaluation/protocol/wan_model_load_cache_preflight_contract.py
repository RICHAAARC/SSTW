"""Fail-closed contract for the isolated Wan model-load/cache preflight.

This contract authorizes only a future explicit Colab infrastructure preflight.
It does not authorize a transformer forward, scheduler step, decode, video,
Gate 0 decision, formal result, or stage progression.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Mapping


DEFAULT_CONFIG_PATH = (
    "configs/protocol/sstw_wan_model_load_cache_preflight.json"
)
PROFILE_ID = "sstw_wan_model_load_cache_preflight"
CONTRACT_STATE = (
    "model_load_cache_preflight_runner_implemented_pending_user_colab_run"
)
FROZEN_PROTOCOL_DIGEST = (
    "b938f80e7c2bd12add7ebd96efa099c8c3843ccebf89e7103c4bd0695d949caa"
)
MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
MODEL_REVISION = "0fad780a534b6463e45facd96134c9f345acfa5b"
DIFFUSERS_VERSION = "0.35.2"
TEST_ID = "wan_model_load_cache_preflight"
PHASE = "model_load_cache_preflight"
RUN_SERIES_ID = "wan_model_load_cache_preflight"

OVERALL_TIMEOUT_SECONDS = 2700
NO_PROGRESS_TIMEOUT_SECONDS = 600
POLL_INTERVAL_SECONDS = 30
PROGRESS_EMIT_INTERVAL_SECONDS = 60
TERMINATION_GRACE_SECONDS = 30
HF_HUB_ETAG_TIMEOUT_SECONDS = 60
HF_HUB_DOWNLOAD_TIMEOUT_SECONDS = 600

_LOWER_HEX_64 = re.compile(r"[0-9a-f]{64}\Z")
EXPECTED_TOP_LEVEL_KEYS = {
    "profile_id",
    "contract_state",
    "authorization_boundary",
    "protocol_contract",
    "protocol_digest",
}
EXPECTED_AUTHORIZATION_BOUNDARY = {
    "runtime_implementation_authorized": True,
    "model_load_cache_preflight_execution_allowed": True,
    "gpu_execution_allowed": True,
    "colab_execution_allowed": True,
    "runner_implementation_allowed": True,
    "notebook_handler_implementation_allowed": False,
    "drive_update_allowed": False,
    "gate0_execution_allowed": False,
    "full_eight_video_rerun_allowed": False,
    "transformer_forward_allowed": False,
    "scheduler_step_allowed": False,
    "decode_allowed": False,
    "video_export_allowed": False,
    "observer_implementation_allowed": False,
    "attack_execution_allowed": False,
    "fixed_fpr_execution_allowed": False,
    "baseline_execution_allowed": False,
    "paper_claim_allowed": False,
    "formal_result": False,
    "stage_progression_allowed": False,
    "claim_support_status": (
        "model_load_cache_preflight_only_not_method_evidence"
    ),
}
EXPECTED_PROTOCOL_SECTIONS = {
    "execution_identity",
    "model_contract",
    "cache_contract",
    "worker_contract",
    "loader_phase_contract",
    "result_boundary",
}
EXPECTED_LOADER_PHASES = (
    "snapshot_download",
    "immutable_revision_resolve",
    "wan_import",
    "vae_from_pretrained",
    "pipeline_from_pretrained",
    "scheduler_configuration",
    "cpu_offload",
    "vae_tiling",
)
EXPECTED_LOADER_PHASE_LEDGER = tuple(
    event
    for phase in EXPECTED_LOADER_PHASES
    for event in (f"{phase}:start", f"{phase}:finish")
)


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def protocol_digest(protocol_contract: Mapping[str, Any]) -> str:
    return sha256(canonical_json_bytes(dict(protocol_contract))).hexdigest()


@dataclass(frozen=True)
class WanModelLoadCachePreflightConfig:
    profile_id: str
    contract_state: str
    protocol_digest: str
    authorization_boundary: Mapping[str, Any]
    protocol_contract: Mapping[str, Any]

    @property
    def worker_contract(self) -> Mapping[str, Any]:
        return self.protocol_contract["worker_contract"]

    @property
    def model_contract(self) -> Mapping[str, Any]:
        return self.protocol_contract["model_contract"]

    @property
    def cache_contract(self) -> Mapping[str, Any]:
        return self.protocol_contract["cache_contract"]


def _require_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} 必须是 JSON 对象")
    return dict(value)


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    label: str,
) -> None:
    observed = set(value)
    if observed != expected:
        raise ValueError(
            f"{label} 字段必须精确冻结; missing={sorted(expected-observed)}, "
            f"extra={sorted(observed-expected)}"
        )


def load_wan_model_load_cache_preflight_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
) -> WanModelLoadCachePreflightConfig:
    payload = _require_mapping(
        json.loads(Path(path).read_text(encoding="utf-8")),
        "Wan model-load/cache preflight config",
    )
    _require_exact_keys(payload, EXPECTED_TOP_LEVEL_KEYS, "config")
    if payload["profile_id"] != PROFILE_ID:
        raise ValueError("Wan model-load/cache preflight profile_id 漂移")
    if payload["contract_state"] != CONTRACT_STATE:
        raise ValueError("Wan model-load/cache preflight contract_state 漂移")
    authorization = _require_mapping(
        payload["authorization_boundary"],
        "authorization_boundary",
    )
    if authorization != EXPECTED_AUTHORIZATION_BOUNDARY:
        raise ValueError("Wan model-load/cache preflight authorization 漂移")
    contract = _require_mapping(payload["protocol_contract"], "protocol_contract")
    _require_exact_keys(contract, EXPECTED_PROTOCOL_SECTIONS, "protocol_contract")
    observed_digest = str(payload["protocol_digest"])
    if not _LOWER_HEX_64.fullmatch(observed_digest):
        raise ValueError("Wan model-load/cache preflight protocol_digest 非法")
    recomputed = protocol_digest(contract)
    if observed_digest != recomputed:
        raise ValueError("Wan model-load/cache preflight self digest 不匹配")
    if observed_digest != FROZEN_PROTOCOL_DIGEST:
        raise ValueError("Wan model-load/cache preflight frozen digest 不匹配")

    identity = _require_mapping(
        contract["execution_identity"], "execution_identity"
    )
    if identity != {
        "test_id": TEST_ID,
        "phase": PHASE,
        "run_series_id": RUN_SERIES_ID,
        "self_contained": True,
        "source_package_path": "",
        "resume_package_path": "",
    }:
        raise ValueError("Wan model-load/cache preflight execution identity 漂移")
    model = _require_mapping(contract["model_contract"], "model_contract")
    expected_model = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "diffusers_version": DIFFUSERS_VERSION,
        "transformer_dtype": "bfloat16",
        "vae_dtype": "float32",
        "scheduler_class": "FlowMatchEulerDiscreteScheduler",
        "scheduler_shift": "3.0",
        "cpu_offload_required": True,
        "vae_tiling_required": True,
    }
    if model != expected_model:
        raise ValueError("Wan model-load/cache preflight model contract 漂移")
    worker = _require_mapping(contract["worker_contract"], "worker_contract")
    expected_timing = {
        "overall_timeout_seconds": OVERALL_TIMEOUT_SECONDS,
        "no_progress_timeout_seconds": NO_PROGRESS_TIMEOUT_SECONDS,
        "poll_interval_seconds": POLL_INTERVAL_SECONDS,
        "progress_emit_interval_seconds": PROGRESS_EMIT_INTERVAL_SECONDS,
        "termination_grace_seconds": TERMINATION_GRACE_SECONDS,
    }
    for key, expected in expected_timing.items():
        if worker.get(key) != expected:
            raise ValueError(f"worker_contract.{key} 漂移")
    if worker.get("worker_start_new_session") is not False:
        raise ValueError("model-load worker 不得创建独立 session")
    if worker.get("progress_deadline_reset_fields") != [
        "cache_regular_file_count",
        "cache_regular_file_bytes",
    ]:
        raise ValueError("worker progress deadline reset fields 漂移")
    if worker.get("diagnostic_only_scalar_fields") != [
        "loader_phase_ledger",
        "cache_incomplete_file_count",
        "cache_lock_file_count",
        "worker_rss_bytes",
        "worker_cpu_seconds",
    ]:
        raise ValueError("worker diagnostic-only scalar fields 漂移")
    if (
        worker.get(
            "diagnostic_only_scalars_must_not_reset_progress_deadline"
        )
        is not True
    ):
        raise ValueError("worker diagnostic-only progress boundary 漂移")
    loader = _require_mapping(
        contract["loader_phase_contract"], "loader_phase_contract"
    )
    if tuple(loader.get("ordered_phases") or ()) != EXPECTED_LOADER_PHASES:
        raise ValueError("loader phase order 漂移")
    if loader.get("each_phase_requires_start_and_finish") is not True:
        raise ValueError("loader phase start/finish contract 漂移")
    if (
        loader.get("complete_ordered_phase_ledger_required_for_success")
        is not True
    ):
        raise ValueError("loader phase success ledger contract 漂移")
    if (
        loader.get("duplicate_missing_or_out_of_order_phase_event_allowed")
        is not False
    ):
        raise ValueError("loader phase ordering fail-closed contract 漂移")
    for key in (
        "snapshot_download_precedes_local_only_model_load",
        "snapshot_path_must_resolve_within_frozen_hub_cache",
        "snapshot_directory_name_must_equal_model_revision",
        "all_from_pretrained_calls_require_local_files_only",
    ):
        if loader.get(key) is not True:
            raise ValueError(f"loader_phase_contract.{key} 漂移")
    cache = _require_mapping(contract["cache_contract"], "cache_contract")
    if cache.get("expected_hf_home") != "/content/SSTW_model_cache":
        raise ValueError("HF_HOME frozen local path 漂移")
    if cache.get("expected_hf_hub_cache") != "/content/SSTW_model_cache/hub":
        raise ValueError("HF_HUB_CACHE frozen local path 漂移")
    if cache.get("drive_mount_root") != "/content/drive":
        raise ValueError("Drive mount frozen boundary 漂移")
    expected_cache_mutation_boundary = {
        "hf_snapshot_download_controlled_download_and_cache_fill_allowed": True,
        "manual_cache_file_deletion_allowed": False,
        "manual_lock_or_incomplete_deletion_allowed": False,
        "manual_existing_cache_rewrite_allowed": False,
    }
    for key, expected in expected_cache_mutation_boundary.items():
        if cache.get(key) is not expected:
            raise ValueError(f"cache_contract.{key} 漂移")
    if cache.get("hub_environment") != {
        "HF_HUB_ETAG_TIMEOUT": str(HF_HUB_ETAG_TIMEOUT_SECONDS),
        "HF_HUB_DOWNLOAD_TIMEOUT": str(HF_HUB_DOWNLOAD_TIMEOUT_SECONDS),
    }:
        raise ValueError("Hugging Face timeout contract 漂移")
    result = _require_mapping(contract["result_boundary"], "result_boundary")
    if result != {
        "worker_success_means_cache_and_load_path_preflight_only": True,
        "worker_failure_is_nonformal_runtime_diagnostic": True,
        "automatic_phase_response_or_gate0_execution_allowed": False,
        "formal_result": False,
        "stage_progression_allowed": False,
        "claim_support_status": (
            "model_load_cache_preflight_only_not_method_evidence"
        ),
    }:
        raise ValueError("Wan model-load/cache preflight result boundary 漂移")
    return WanModelLoadCachePreflightConfig(
        profile_id=PROFILE_ID,
        contract_state=CONTRACT_STATE,
        protocol_digest=observed_digest,
        authorization_boundary=authorization,
        protocol_contract=contract,
    )
