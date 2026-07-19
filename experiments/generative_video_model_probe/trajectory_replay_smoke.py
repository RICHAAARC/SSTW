"""对已有4-source机制包执行最小 trajectory replay smoke。

该入口只产生诊断 records、manifest、report 与 go/no-go。它不拟合 fixed-FPR
阈值、不使用 test split、不执行大规模生成或外部 baseline，因此任何输出都不能
作为 paper evidence。真实 replay 必须使用锁定 GPU 环境和所有者密钥；缺少任一
条件时会生成环境阻断型 NO_GO，而不会回退到 proxy score。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import zipfile
from collections import defaultdict
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

from evaluation.attacks.video_runtime_attack_protocol import (
    load_protocol_config_with_shared_attack_protocol,
)
from evaluation.protocol.record_writer import write_json, write_jsonl
from experiments.generative_video_model_probe.attack_runner import (
    build_runtime_attack_records,
)
from experiments.generative_video_model_probe.colab_runtime import (
    validate_generation_model_provenance,
)
from experiments.generative_video_model_probe.formal_flow_evidence_runner import (
    _compute_replay_endpoint_evidence_for_key,
    _evaluate_fixed_replay_hypothesis_for_key,
    _fit_model_specific_replay_likelihood_configs,
    _generation_key,
    _invoke_pipeline_loader,
    _load_pipeline,
    _run_attacked_video_replay_for_model,
    _validated_flow_key_context,
    _wrong_owner_generation_key,
)


DEFAULT_CONFIG_PATH = "configs/protocol/sstw_minimal_trajectory_paper.json"
SMOKE_RECORD_VERSION = "trajectory_replay_smoke_v1"
SMOKE_CLAIM_SUPPORT_STATUS = "trajectory_replay_smoke_only_not_paper_evidence"


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON 顶层必须是对象: {path}")
    return payload


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    return [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_digest(payload: Mapping[str, Any]) -> str:
    return sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _unique_archive_member(members: Iterable[str], suffix: str) -> str:
    matches = [member for member in members if member.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(
            f"source package 中 {suffix} 必须唯一, observed={len(matches)}"
        )
    return matches[0]


def _safe_member_basename(member: str) -> str:
    path = PurePosixPath(member)
    if path.is_absolute() or ".." in path.parts or not path.name:
        raise ValueError(f"source package 含不安全路径: {member}")
    return path.name


def validate_minimal_trajectory_profile(config: Mapping[str, Any]) -> None:
    """确认 smoke profile 不会静默升级为 full-paper 或跨项目执行。"""

    required_false = (
        "large_scale_generation_allowed",
        "external_baseline_execution_allowed",
        "cross_project_integration_allowed",
        "fixed_fpr_evaluation_allowed",
        "test_split_claims_allowed",
    )
    invalid = [field for field in required_false if config.get(field) is not False]
    if invalid:
        raise ValueError("minimal trajectory profile 的禁止项未冻结: " + ", ".join(invalid))
    if config.get("paper_result_level") != "trajectory_paper_smoke":
        raise ValueError("minimal trajectory profile 必须保持 smoke result level")
    required_variants = {
        "sstw_full_method",
        "endpoint_only_control",
        "sstw_clean_unwatermarked_reference",
    }
    observed_variants = {
        str(item) for item in config.get("required_source_method_variants") or []
    }
    if observed_variants != required_variants:
        raise ValueError("minimal trajectory profile 的 source variants 不完整")
    attack_variants = {
        str(item) for item in config.get("runtime_attack_source_method_variants") or []
    }
    if attack_variants != required_variants:
        raise ValueError("标准攻击必须同时覆盖 full、endpoint-only 与 clean")
    attack_names = [
        str(item) for item in config.get("required_runtime_attack_names") or []
    ]
    if attack_names != ["h264_crf28_runtime", "temporal_crop_runtime"]:
        raise ValueError("Stage 0 smoke 只允许一种压缩和一种时间攻击")
    if [int(value) for value in config.get("replay_step_counts") or []] != [20]:
        raise ValueError("Stage 0 smoke 必须使用单一预注册20步 replay grid")


def materialize_source_package(
    package_path: str | Path,
    run_root: str | Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """安全提取已有4-source包中的 records、prompt suite 与必要视频。"""

    source_package = Path(package_path).resolve()
    target_root = Path(run_root).resolve()
    if not source_package.is_file():
        raise FileNotFoundError(f"缺少 source package: {source_package}")
    target_root.mkdir(parents=True, exist_ok=True)
    (target_root / "records").mkdir(parents=True, exist_ok=True)
    (target_root / "videos").mkdir(parents=True, exist_ok=True)
    (target_root / "datasets").mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(source_package) as archive:
        members = archive.namelist()
        generation_member = _unique_archive_member(
            members,
            "/records/generation_records.jsonl",
        )
        prompt_suite_member = _unique_archive_member(
            members,
            "/prompt_seed_suite.json",
        )
        generation_records = [
            json.loads(line)
            for line in archive.read(generation_member).decode("utf-8").splitlines()
            if line.strip()
        ]
        required_variants = {
            str(item) for item in config["required_source_method_variants"]
        }
        selected_records = [
            record
            for record in generation_records
            if record.get("generation_status") == "success"
            and str(record.get("method_variant") or "") in required_variants
        ]
        source_groups: dict[tuple[str, str, str], set[str]] = defaultdict(set)
        for record in selected_records:
            identity = (
                str(record.get("generation_model_id") or ""),
                str(record.get("prompt_id") or ""),
                str(record.get("seed_id") or ""),
            )
            source_groups[identity].add(str(record.get("method_variant") or ""))
        expected_source_count = int(config["required_source_video_count"])
        if len(source_groups) != expected_source_count:
            raise RuntimeError(
                "source package 独立 source 数不匹配: "
                f"expected={expected_source_count}, observed={len(source_groups)}"
            )
        incomplete = {
            "::".join(identity): sorted(required_variants - variants)
            for identity, variants in source_groups.items()
            if variants != required_variants
        }
        if incomplete:
            raise RuntimeError(
                "source package 缺少必要变体: "
                + json.dumps(incomplete, ensure_ascii=False, sort_keys=True)
            )
        source_profiles = {
            str(record.get("colab_runtime_profile") or "")
            for record in selected_records
        }
        if source_profiles != {str(config["source_package_workflow_profile"])}:
            raise RuntimeError(f"source package workflow profile 不匹配: {source_profiles}")
        source_splits = {str(record.get("split") or "") for record in selected_records}
        if source_splits != {"calibration"}:
            raise RuntimeError("Stage 0 必须只消费已有 calibration sources")

        video_members = {
            _safe_member_basename(member): member
            for member in members
            if "/videos/" in member and member.endswith(".mp4")
        }
        required_video_names = {
            Path(str(record.get("video_path") or "")).name
            for record in selected_records
        }
        missing_videos = sorted(required_video_names - set(video_members))
        if missing_videos:
            raise RuntimeError("source package 缺少视频: " + ", ".join(missing_videos))
        for video_name in sorted(required_video_names):
            (target_root / "videos" / video_name).write_bytes(
                archive.read(video_members[video_name])
            )
        prompt_suite_path = target_root / "datasets" / "prompt_seed_suite.json"
        prompt_suite_path.write_bytes(archive.read(prompt_suite_member))

    write_jsonl(target_root / "records" / "generation_records.jsonl", selected_records)
    manifest = {
        "artifact_id": "trajectory_replay_smoke_source_manifest",
        "artifact_type": "manifest",
        "claim_support_status": SMOKE_CLAIM_SUPPORT_STATUS,
        "profile_id": config["profile_id"],
        "source_package_path": str(source_package),
        "source_package_sha256": _sha256_file(source_package),
        "source_package_workflow_profile": config["source_package_workflow_profile"],
        "source_split": "calibration",
        "source_video_count": len(source_groups),
        "source_generation_record_count": len(selected_records),
        "source_method_variants": sorted(required_variants),
        "source_video_file_count": len(required_video_names),
        "source_code_commits": sorted({
            str(record.get("code_commit") or "")
            for record in selected_records
            if str(record.get("code_commit") or "")
        }),
        "prompt_suite_path": str(prompt_suite_path),
        "test_split_claims_allowed": False,
    }
    write_json(
        target_root / "artifacts" / "trajectory_replay_smoke_source_manifest.json",
        manifest,
    )
    return manifest


def replay_runtime_preflight() -> dict[str, Any]:
    """检查真实 replay 环境，只报告凭据是否存在，不读取或输出密钥。"""

    package_status = {
        package_name: importlib.util.find_spec(package_name) is not None
        for package_name in ("torch", "diffusers", "transformers")
    }
    blockers = [
        f"missing_python_package:{package_name}"
        for package_name, ready in package_status.items()
        if not ready
    ]
    cuda_available = False
    if package_status["torch"]:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        if not cuda_available:
            blockers.append("cuda_unavailable")
    key_present = bool(os.environ.get("SSTW_TRAJECTORY_AUTHENTICATION_KEY"))
    key_id_present = bool(os.environ.get("SSTW_TRAJECTORY_AUTHENTICATION_KEY_ID"))
    if not key_present:
        blockers.append("missing_trajectory_authentication_key")
    if not key_id_present:
        blockers.append("missing_trajectory_authentication_key_id")
    return {
        "replay_runtime_preflight_status": "ready" if not blockers else "blocked",
        "replay_runtime_python_package_status": package_status,
        "replay_runtime_cuda_available": cuda_available,
        "replay_runtime_authentication_key_present": key_present,
        "replay_runtime_authentication_key_id_present": key_id_present,
        "replay_runtime_blockers": blockers,
    }


def _prompt_text_by_id(prompt_suite: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(row["prompt_id"]): str(row["prompt_text"])
        for row in prompt_suite.get("prompts") or []
        if row.get("prompt_id") and row.get("prompt_text")
    }


def _record_identity(record: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(record.get("generation_model_id") or ""),
        str(record.get("prompt_id") or ""),
        str(record.get("seed_id") or ""),
        str(record.get("attack_name") or ""),
    )


def _finite_values(*values: Any) -> bool:
    try:
        return all(math.isfinite(float(value)) for value in values)
    except (TypeError, ValueError):
        return False


def execute_replay_smoke(
    run_root: str | Path,
    config: Mapping[str, Any],
    *,
    pipeline_loader: Any = _load_pipeline,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """串行执行 clean-noise calibration 与攻击视频 correct/wrong-key replay。"""

    root = Path(run_root)
    generation_records = _read_jsonl(root / "records" / "generation_records.jsonl")
    attack_records = _read_jsonl(
        root / "records" / "trajectory_replay_smoke_attack_records.jsonl"
    )
    prompt_map = _prompt_text_by_id(_read_json(root / "datasets" / "prompt_seed_suite.json"))
    clean_records = [
        record
        for record in generation_records
        if record.get("method_variant") == "sstw_clean_unwatermarked_reference"
    ]
    model_revisions: dict[str, str] = {}
    for record in generation_records:
        revision = validate_generation_model_provenance(record)
        model_id = str(record.get("generation_model_id") or "")
        previous = model_revisions.setdefault(model_id, revision)
        if previous != revision:
            raise RuntimeError(f"同一模型混用了不同 revision: {model_id}")
    pipelines = {
        model_id: _invoke_pipeline_loader(
            pipeline_loader,
            model_id=model_id,
            revision=revision,
        )
        for model_id, revision in sorted(model_revisions.items())
    }
    likelihood_configs, calibration_records = (
        _fit_model_specific_replay_likelihood_configs(
            root,
            clean_records,
            prompt_map,
            pipelines,
            minimum_clean_video_cluster_count=int(
                config["minimum_replay_likelihood_calibration_clean_video_cluster_count"]
            ),
            calibration_replay_step_count=int(
                config["replay_likelihood_calibration_step_count"]
            ),
        )
    )
    replay_records: list[dict[str, Any]] = []
    failure_records: list[dict[str, Any]] = []
    ready_attacks = [
        record
        for record in attack_records
        if record.get("attack_runtime_status") == "ready"
    ]
    for source in ready_attacks:
        try:
            model_id = str(source.get("generation_model_id") or "")
            prompt = prompt_map[str(source.get("prompt_id") or "")]
            pipeline = pipelines[model_id]
            correct_key = _generation_key(source)
            wrong_key = _wrong_owner_generation_key(source)
            key_context = _validated_flow_key_context(
                source,
                prompt=prompt,
                scheduler=pipeline.scheduler,
            )
            replay = _run_attacked_video_replay_for_model(
                pipeline,
                str(source.get("attacked_video_path") or ""),
                prompt=prompt,
                key_text=correct_key,
                key_context=key_context,
                likelihood_config=likelihood_configs[model_id],
                replay_step_counts=tuple(int(value) for value in config["replay_step_counts"]),
            )
            wrong_endpoint = _compute_replay_endpoint_evidence_for_key(
                replay,
                key_text=wrong_key,
                key_context=key_context,
            )
            wrong_trajectory, wrong_path = _evaluate_fixed_replay_hypothesis_for_key(
                pipeline,
                replay,
                prompt=prompt,
                key_text=wrong_key,
                key_context=key_context,
            )
            correct_trajectory = replay.replay_trajectories[replay.primary_replay_index]
            correct_path_score = float(replay.path_evidence.get("S_path_inv") or 0.0)
            wrong_path_score = float(wrong_path.get("S_path_inv") or 0.0)
            correct_llr = float(correct_trajectory.replay_log_likelihood_ratio)
            wrong_llr = float(wrong_trajectory.replay_log_likelihood_ratio)
            correct_endpoint_score = float(replay.endpoint_evidence.score)
            wrong_endpoint_score = float(wrong_endpoint.score)
            reliability = float(replay.replay_uncertainty.replay_reliability)
            finite = _finite_values(
                correct_path_score,
                wrong_path_score,
                correct_llr,
                wrong_llr,
                correct_endpoint_score,
                wrong_endpoint_score,
                reliability,
            )
            replay_records.append({
                "record_version": SMOKE_RECORD_VERSION,
                "profile_id": config["profile_id"],
                "generation_model_id": model_id,
                "prompt_id": source.get("prompt_id"),
                "seed_id": source.get("seed_id"),
                "trajectory_trace_id": source.get("trajectory_trace_id"),
                "split": source.get("split"),
                "method_variant": source.get("method_variant"),
                "sample_role": source.get("sample_role"),
                "negative_family": source.get("negative_family"),
                "attack_name": source.get("attack_name"),
                "source_video_cluster_id": _stable_digest({
                    "generation_model_id": model_id,
                    "prompt_id": source.get("prompt_id"),
                    "seed_id": source.get("seed_id"),
                }),
                "attacked_video_path": source.get("attacked_video_path"),
                "attacked_video_sha256": source.get("attacked_video_sha256"),
                "replay_step_counts": list(config["replay_step_counts"]),
                "replay_primary_step_count": int(
                    replay.replay_step_counts[replay.primary_replay_index]
                ),
                "replay_reliability": reliability,
                "correct_key_endpoint_score": correct_endpoint_score,
                "wrong_key_endpoint_score": wrong_endpoint_score,
                "correct_key_endpoint_margin": correct_endpoint_score - wrong_endpoint_score,
                "correct_key_path_score": correct_path_score,
                "wrong_key_path_score": wrong_path_score,
                "correct_key_path_margin": correct_path_score - wrong_path_score,
                "correct_key_replay_log_likelihood_ratio": correct_llr,
                "wrong_key_replay_log_likelihood_ratio": wrong_llr,
                "correct_key_replay_likelihood_margin": correct_llr - wrong_llr,
                "wrong_key_fixed_reverse_path_reused": (
                    wrong_trajectory.reverse_states is correct_trajectory.reverse_states
                ),
                "replay_numeric_finite": finite,
                "metric_status": "measured_smoke_diagnostic",
                "claim_support_status": SMOKE_CLAIM_SUPPORT_STATUS,
            })
        except Exception as exc:  # pragma: no cover - 依赖真实GPU与模型
            failure_records.append({
                "record_version": SMOKE_RECORD_VERSION,
                "profile_id": config["profile_id"],
                "generation_model_id": source.get("generation_model_id"),
                "prompt_id": source.get("prompt_id"),
                "seed_id": source.get("seed_id"),
                "trajectory_trace_id": source.get("trajectory_trace_id"),
                "method_variant": source.get("method_variant"),
                "attack_name": source.get("attack_name"),
                "replay_smoke_status": "failed",
                "replay_smoke_failure_reason": str(exc),
                "claim_support_status": SMOKE_CLAIM_SUPPORT_STATUS,
            })
    return replay_records, failure_records, calibration_records


def build_smoke_decision(
    config: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    attack_records: Iterable[Mapping[str, Any]],
    replay_records: Iterable[Mapping[str, Any]],
    failure_records: Iterable[Mapping[str, Any]],
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    """由 smoke records 重建 go/no-go，不读取任何论文 test 结果。"""

    attacks = [dict(record) for record in attack_records]
    replays = [dict(record) for record in replay_records]
    failures = [dict(record) for record in failure_records]
    required_variants = [str(item) for item in config["required_source_method_variants"]]
    required_attacks = [str(item) for item in config["required_runtime_attack_names"]]
    expected_attack_count = (
        int(config["required_source_video_count"])
        * len(required_variants)
        * len(required_attacks)
    )
    ready_attacks = [row for row in attacks if row.get("attack_runtime_status") == "ready"]
    attack_coverage_ready = (
        len(attacks) == expected_attack_count
        and len(ready_attacks) == expected_attack_count
    )
    replay_coverage_ready = len(replays) == expected_attack_count and not failures

    full = [row for row in replays if row.get("method_variant") == "sstw_full_method"]
    endpoint = {
        _record_identity(row): row
        for row in replays
        if row.get("method_variant") == "endpoint_only_control"
    }
    clean = {
        _record_identity(row): row
        for row in replays
        if row.get("method_variant") == "sstw_clean_unwatermarked_reference"
    }

    def fraction(values: list[bool]) -> float:
        return sum(bool(value) for value in values) / len(values) if values else 0.0

    full_path_fraction = fraction([
        float(row.get("correct_key_path_margin") or 0.0) > 0.0 for row in full
    ])
    full_replay_fraction = fraction([
        float(row.get("correct_key_replay_likelihood_margin") or 0.0) > 0.0
        for row in full
    ])
    full_over_endpoint_fraction = fraction([
        float(row.get("correct_key_path_margin") or 0.0)
        > float(endpoint[_record_identity(row)].get("correct_key_path_margin") or 0.0)
        for row in full
        if _record_identity(row) in endpoint
    ])
    full_over_clean_fraction = fraction([
        float(row.get("correct_key_path_margin") or 0.0)
        > float(clean[_record_identity(row)].get("correct_key_path_margin") or 0.0)
        for row in full
        if _record_identity(row) in clean
    ])
    minimum_reliability = min(
        (float(row.get("replay_reliability") or 0.0) for row in replays),
        default=0.0,
    )
    numeric_finite = bool(replays) and all(
        row.get("replay_numeric_finite") is True for row in replays
    )
    fixed_path_reuse = bool(replays) and all(
        row.get("wrong_key_fixed_reverse_path_reused") is True for row in replays
    )
    criteria = {
        "source_package_ready": int(source_manifest.get("source_video_count") or 0)
        == int(config["required_source_video_count"]),
        "standard_attack_coverage_ready": attack_coverage_ready,
        "replay_runtime_preflight_ready": preflight.get("replay_runtime_preflight_status")
        == "ready",
        "replay_coverage_ready": replay_coverage_ready,
        "replay_numeric_finite": numeric_finite,
        "wrong_key_fixed_reverse_path_reused": fixed_path_reuse,
        "minimum_replay_reliability_ready": minimum_reliability
        >= float(config["minimum_replay_reliability"]),
        "full_correct_over_wrong_path_ready": full_path_fraction
        >= float(config["minimum_full_correct_over_wrong_fraction"]),
        "full_correct_over_wrong_replay_ready": full_replay_fraction
        >= float(config["minimum_full_correct_over_wrong_fraction"]),
        "full_path_margin_over_endpoint_ready": full_over_endpoint_fraction
        >= float(config["minimum_full_path_margin_over_endpoint_fraction"]),
        "full_path_margin_over_clean_ready": full_over_clean_fraction
        >= float(config["minimum_full_path_margin_over_clean_fraction"]),
    }
    go = all(criteria.values())
    if preflight.get("replay_runtime_preflight_status") != "ready":
        reason_category = "runtime_environment_blocked"
    elif failures or not replay_coverage_ready:
        reason_category = "replay_execution_failed"
    elif not go:
        reason_category = "trajectory_smoke_signal_not_supported"
    else:
        reason_category = "trajectory_smoke_gate_passed"
    return {
        "stage_id": config["stage_id"],
        "profile_id": config["profile_id"],
        "go_no_go_decision": "GO" if go else "NO_GO",
        "go_no_go_reason_category": reason_category,
        "stage_progression": (
            config["stage_progression_on_go"]
            if go
            else config["stage_progression_on_no_go"]
        ),
        "claim_support_status": SMOKE_CLAIM_SUPPORT_STATUS,
        "fixed_fpr_evaluated": False,
        "test_split_used": False,
        "paper_claim_allowed": False,
        "large_scale_generation_allowed": False,
        "external_baseline_executed": False,
        "cross_project_integration_executed": False,
        "expected_attack_record_count": expected_attack_count,
        "attack_record_count": len(attacks),
        "ready_attack_record_count": len(ready_attacks),
        "replay_record_count": len(replays),
        "replay_failure_record_count": len(failures),
        "replay_skipped_due_preflight_count": (
            expected_attack_count
            if preflight.get("replay_runtime_preflight_status") != "ready"
            else 0
        ),
        "full_replay_record_count": len(full),
        "full_correct_over_wrong_path_fraction": full_path_fraction,
        "full_correct_over_wrong_replay_fraction": full_replay_fraction,
        "full_path_margin_over_endpoint_fraction": full_over_endpoint_fraction,
        "full_path_margin_over_clean_fraction": full_over_clean_fraction,
        "minimum_observed_replay_reliability": minimum_reliability,
        "smoke_gate_criteria": criteria,
        **dict(preflight),
    }


def _write_report(path: Path, decision: Mapping[str, Any]) -> None:
    criteria = decision.get("smoke_gate_criteria") or {}
    criterion_lines = "\n".join(
        f"- {name}: {'PASS' if passed else 'FAIL'}"
        for name, passed in criteria.items()
    )
    blockers = decision.get("replay_runtime_blockers") or []
    blocker_text = ", ".join(str(item) for item in blockers) if blockers else "none"
    report = (
        "# SSTW Minimal Trajectory Replay Smoke Report\n\n"
        f"- go_no_go_decision: {decision['go_no_go_decision']}\n"
        f"- go_no_go_reason_category: {decision['go_no_go_reason_category']}\n"
        f"- stage_progression: {decision['stage_progression']}\n"
        f"- source scope: 现有4-source、仅 calibration 的机制包\n"
        f"- attack_record_count: {decision['attack_record_count']} / "
        f"{decision['expected_attack_record_count']}\n"
        f"- replay_record_count: {decision['replay_record_count']}\n"
        f"- replay_failure_record_count: {decision['replay_failure_record_count']}\n"
        f"- replay_skipped_due_preflight_count: "
        f"{decision['replay_skipped_due_preflight_count']}\n"
        f"- runtime blockers: {blocker_text}\n\n"
        "## 门禁明细\n\n"
        f"{criterion_lines}\n\n"
        "## 证据边界\n\n"
        "本 smoke 不评估 fixed FPR、不使用 held-out test split、不运行外部 baseline，"
        "也不能支持论文主张。GO 只允许继续构建下一阶段 minimal trajectory-paper "
        "协议；NO_GO 会阻止大规模生成，直到记录的 blocker 被解决。\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def run_trajectory_replay_smoke(
    package_path: str | Path,
    run_root: str | Path,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    pipeline_loader: Any = _load_pipeline,
) -> dict[str, Any]:
    """执行 Stage 0，并始终生成来源明确的 report 与 decision。"""

    root = Path(run_root).resolve()
    config = load_protocol_config_with_shared_attack_protocol(config_path)
    validate_minimal_trajectory_profile(config)
    source_manifest = materialize_source_package(package_path, root, config)
    attack_records = build_runtime_attack_records(
        root,
        attack_names=tuple(str(item) for item in config["required_runtime_attack_names"]),
        config_path=config_path,
    )
    for record in attack_records:
        record["profile_id"] = config["profile_id"]
        record["claim_support_status"] = config["runtime_attack_claim_support_status"]
    write_jsonl(
        root / "records" / "trajectory_replay_smoke_attack_records.jsonl",
        attack_records,
    )
    preflight = replay_runtime_preflight()
    replay_records: list[dict[str, Any]] = []
    failure_records: list[dict[str, Any]] = []
    calibration_records: list[dict[str, Any]] = []
    if preflight["replay_runtime_preflight_status"] == "ready":
        try:
            replay_records, failure_records, calibration_records = execute_replay_smoke(
                root,
                config,
                pipeline_loader=pipeline_loader,
            )
        except Exception as exc:  # pragma: no cover - 真实模型初始化失败
            failure_records.append({
                "record_version": SMOKE_RECORD_VERSION,
                "profile_id": config["profile_id"],
                "replay_smoke_status": "failed",
                "replay_smoke_failure_reason": str(exc),
                "claim_support_status": SMOKE_CLAIM_SUPPORT_STATUS,
            })
    write_jsonl(root / "records" / "trajectory_replay_smoke_records.jsonl", replay_records)
    write_jsonl(
        root / "records" / "trajectory_replay_smoke_failure_records.jsonl",
        failure_records,
    )
    write_jsonl(
        root / "records" / "trajectory_replay_smoke_likelihood_calibrations.jsonl",
        calibration_records,
    )
    decision = build_smoke_decision(
        config,
        source_manifest,
        attack_records,
        replay_records,
        failure_records,
        preflight,
    )
    decision_path = root / "artifacts" / "trajectory_replay_smoke_decision.json"
    report_path = root / "reports" / "trajectory_replay_smoke_report.md"
    write_json(decision_path, decision)
    _write_report(report_path, decision)
    record_paths = [
        root / "records" / "trajectory_replay_smoke_attack_records.jsonl",
        root / "records" / "trajectory_replay_smoke_records.jsonl",
        root / "records" / "trajectory_replay_smoke_failure_records.jsonl",
        root / "records" / "trajectory_replay_smoke_likelihood_calibrations.jsonl",
    ]
    manifest = {
        "artifact_id": "trajectory_replay_smoke_manifest",
        "artifact_type": "manifest",
        "profile_id": config["profile_id"],
        "claim_support_status": SMOKE_CLAIM_SUPPORT_STATUS,
        "source_package_path": str(Path(package_path).resolve()),
        "source_package_sha256": source_manifest["source_package_sha256"],
        "protocol_config_path": str(Path(config_path).resolve()),
        "protocol_config_sha256": _sha256_file(config_path),
        "smoke_runner_source_path": str(Path(__file__).resolve()),
        "smoke_runner_source_sha256": _sha256_file(__file__),
        "decision_path": str(decision_path),
        "report_path": str(report_path),
        "record_paths": [str(path) for path in record_paths],
        "output_sha256": {
            str(path): _sha256_file(path)
            for path in [decision_path, report_path, *record_paths]
        },
        "rebuild_command": (
            "python -m experiments.generative_video_model_probe.trajectory_replay_smoke "
            f"--package-path {Path(package_path).resolve()} --run-root {root} "
            f"--config-path {Path(config_path).resolve()}"
        ),
    }
    write_json(root / "artifacts" / "trajectory_replay_smoke_manifest.json", manifest)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(
        description="对已有4-source SSTW包执行最小 trajectory replay smoke。"
    )
    parser.add_argument("--package-path", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    decision = run_trajectory_replay_smoke(
        args.package_path,
        args.run_root,
        args.config_path,
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
