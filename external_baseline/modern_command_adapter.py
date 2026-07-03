"""现代视频水印 baseline 官方命令适配工具。"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Mapping

from external_baseline.runtime_trace_io import build_comparison_unit_id, comparable_detection_records, safe_float
from external_baseline.score_semantics import normalized_score_payload
from external_baseline.official_eval_adapters.common import validate_clean_negative_payload
from main.core.digest import build_stable_digest
from main.core.progress import ProgressReporter
from main.protocol.flow_evidence_fields import with_flow_evidence_protocol_defaults


@dataclass(frozen=True)
class ModernBaselineCommandConfig:
    """描述一个现代 baseline 的外部命令适配契约。

    通用工程写法是用命令模板把第三方官方实现隔离在 `external_baseline/primary/<id>/source/`
    或用户显式配置的路径中。项目特定要求是: adapter 只读取 run_root 中已经落盘的 governed
    records, 并把官方 detector 输出映射为统一 comparison records。
    """

    baseline_name: str
    baseline_family: str
    adapter_path: str
    env_var: str
    default_source_script: str
    score_source: str
    claim_support_status: str = "modern_external_baseline_formal_measured"


def _command_template(config: ModernBaselineCommandConfig) -> str | None:
    """解析现代 baseline 命令模板。

    优先读取环境变量。若环境变量不存在, 再检查默认 source script 是否已由 Colab 或用户下载。
    该函数不尝试联网下载第三方代码, 避免 Notebook 之外出现隐式外部状态。
    """
    configured = os.environ.get(config.env_var)
    if configured:
        return configured
    source_script = Path(config.default_source_script)
    if source_script.exists():
        return (
            f"{sys.executable} {source_script.as_posix()} "
            "--source-video {source_video_path} "
            "--attacked-video {attacked_video_path} "
            "--attack-name {attack_name} "
            "--output-json {output_json_path}"
        )
    return None


def adapter_status_for(config: ModernBaselineCommandConfig) -> dict[str, Any]:
    """返回现代 baseline adapter 的受治理状态。"""
    command = _command_template(config)
    command_configured = command is not None
    return {
        "external_baseline_runnable_status": "runnable" if command_configured else "not_runnable",
        "external_baseline_adapter_status": "ready" if command_configured else "adapter_ready_command_not_configured",
        "external_baseline_adapter_path": config.adapter_path,
        "external_baseline_input_compatibility_status": "runtime_video_files_and_attack_manifest" if command_configured else "requires_official_command_configuration",
        "external_baseline_output_record_status": "governed_records_written" if command_configured else "non_run_record_written",
        "external_baseline_threshold_policy_compatible": command_configured,
        "external_baseline_attack_manifest_compatible": command_configured,
        "external_baseline_command_config_status": "configured" if command_configured else "missing",
        "external_baseline_command_env_var": config.env_var,
        "external_baseline_not_run_reason": "none" if command_configured else "official_command_not_configured",
        "external_baseline_result_used_for_claim": command_configured,
    }


def _format_command(
    template: str,
    run_root: Path,
    detection_record: Mapping[str, Any],
    output_json_path: Path,
) -> list[str]:
    """把命令模板格式化为 argv, 不通过 shell 执行。"""
    values = {
        "run_root": str(run_root),
        "source_video_path": str(detection_record.get("source_video_path") or ""),
        "attacked_video_path": str(detection_record.get("attacked_video_path") or ""),
        "output_json_path": str(output_json_path),
        "prompt_id": str(detection_record.get("prompt_id") or ""),
        "seed_id": str(detection_record.get("seed_id") or ""),
        "attack_name": str(detection_record.get("attack_name") or ""),
        "trajectory_trace_id": str(detection_record.get("trajectory_trace_id") or ""),
    }
    command_text = template.format(**values)
    return shlex.split(command_text, posix=os.name != "nt")


def _read_official_output(path: Path) -> dict[str, Any]:
    """读取官方 baseline 输出 JSON。"""
    if not path.exists():
        raise FileNotFoundError("official_output_json_missing")
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("official_output_json_must_be_object")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """写出外部 baseline 官方命令证据 JSON。

    该函数属于通用工程写法, 目的是让 Colab 冷启动中的第三方命令输出可以随
    Google Drive package 一起落盘, 后续审计时不只依赖聚合分数。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, value: str) -> None:
    """写出官方命令 stdout / stderr 文本证据。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _official_command_evidence_paths(
    run_root: Path,
    config: ModernBaselineCommandConfig,
    score_record_id: str,
) -> dict[str, str]:
    """构造单条现代 baseline 官方命令的持久化证据路径。

    项目特定要求是: `measured_formal` 不能只保留分数, 还必须保留官方命令输出
    JSON、stdout / stderr 和受治理 manifest, 以便 `external_baseline_execution_manifest`
    能证明该结果不是手工表格。
    """
    score_suffix = score_record_id.replace("external_baseline_score_", "")
    evidence_dir = run_root / "artifacts" / "external_baseline_evidence" / config.baseline_name / score_suffix
    return {
        "external_baseline_official_output_path": str(evidence_dir / "official_output.json"),
        "external_baseline_official_stdout_path": str(evidence_dir / "official_stdout.txt"),
        "external_baseline_official_stderr_path": str(evidence_dir / "official_stderr.txt"),
        "external_baseline_official_command_manifest_path": str(evidence_dir / "official_command_manifest.json"),
    }


def _extract_score(payload: Mapping[str, Any]) -> float:
    """从常见官方输出字段中提取 baseline score。"""
    return float(normalized_score_payload(payload)["external_baseline_raw_detector_score"])


def _unsupported_record(
    config: ModernBaselineCommandConfig,
    baseline_record: Mapping[str, Any],
    detection_record: Mapping[str, Any],
    reason: str,
    evidence_paths: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """构造现代 baseline 无法正式评分时的 governed unsupported record。"""
    persisted_evidence_paths = dict(evidence_paths or {})
    return with_flow_evidence_protocol_defaults({
        "record_version": "external_baseline_score_v2",
        "external_baseline_score_record_id": build_comparison_unit_id(config.baseline_name, detection_record),
        "external_baseline_name": config.baseline_name,
        "external_baseline_family": baseline_record.get("external_baseline_family") or config.baseline_family,
        "external_baseline_layer": baseline_record.get("external_baseline_layer"),
        "external_baseline_adapter_path": config.adapter_path,
        "external_baseline_command_env_var": config.env_var,
        "external_baseline_command_config_status": "configured" if _command_template(config) else "missing",
        "generation_model_id": detection_record.get("generation_model_id"),
        "prompt_id": detection_record.get("prompt_id"),
        "seed_id": detection_record.get("seed_id"),
        "trajectory_trace_id": detection_record.get("trajectory_trace_id"),
        "attack_name": detection_record.get("attack_name"),
        "sample_role": detection_record.get("sample_role"),
        "negative_family": detection_record.get("negative_family"),
        "metric_status": "unsupported",
        "external_baseline_score_status": "unsupported",
        "external_baseline_score_source": config.score_source,
        "external_baseline_score_failure_reason": reason,
        "external_baseline_reference_sequence_length": None,
        "external_baseline_observed_sequence_length": None,
        "external_baseline_distance": None,
        "external_baseline_score": None,
        "external_baseline_raw_detector_score": None,
        "external_baseline_score_field": None,
        "external_baseline_score_semantics": None,
        "external_baseline_score_orientation": "higher_is_more_watermarked",
        "external_baseline_detected": None,
        "external_baseline_bit_accuracy": None,
        "external_baseline_payload_bit_accuracy": None,
        "external_baseline_clean_negative_score": None,
        "external_baseline_clean_negative_score_semantics": None,
        "external_baseline_clean_negative_video_path": None,
        "external_baseline_threshold": None,
        "baseline_score_margin": None,
        "external_baseline_result_used_for_claim": False,
        "claim_support_status": "modern_external_baseline_formal_adapter_not_measured",
        **persisted_evidence_paths,
    }, trajectory_source_level="external_modern_baseline_official_adapter", claim_support_status="modern_external_baseline_formal_adapter_not_measured")


def build_modern_score_records(
    run_root: str | Path,
    baseline_record: Mapping[str, Any],
    config: ModernBaselineCommandConfig,
) -> list[dict[str, Any]]:
    """调用现代视频水印官方命令并生成统一 comparison records。

    该函数不实现第三方论文方法本体。它只定义 SSTW 与官方实现之间的稳定 I/O 契约:
    官方命令读取 source / attacked video, 输出 JSON score, adapter 负责写入 governed records。
    如果官方代码或权重未配置, 该函数会写 unsupported record, 使 gate 明确失败。
    """
    root = Path(run_root)
    command_template = _command_template(config)
    detection_records = comparable_detection_records(root)
    records: list[dict[str, Any]] = []
    progress = ProgressReporter(
        f"external_baseline_formal_scoring:{config.baseline_name}",
        len(detection_records),
        "runtime_video",
    )
    if command_template is None:
        unsupported = []
        for index, record in enumerate(detection_records):
            progress.update(
                index + 1,
                f"baseline={config.baseline_name} prompt={record.get('prompt_id')} seed={record.get('seed_id')} attack={record.get('attack_name')}",
            )
            unsupported.append(_unsupported_record(config, baseline_record, record, "official_command_not_configured"))
        progress.finish("official_command_not_configured")
        return unsupported

    timeout_sec = safe_float(os.environ.get("SSTW_EXTERNAL_BASELINE_TIMEOUT_SEC"), 1800.0)
    for index, detection_record in enumerate(detection_records):
        progress.update(
            index + 1,
            (
                f"baseline={config.baseline_name} prompt={detection_record.get('prompt_id')} "
                f"seed={detection_record.get('seed_id')} attack={detection_record.get('attack_name')}"
            ),
        )
        score_record_id = build_comparison_unit_id(config.baseline_name, detection_record)
        evidence_paths: dict[str, str] = {}
        try:
            source_path = Path(str(detection_record.get("source_video_path") or ""))
            attacked_path = Path(str(detection_record.get("attacked_video_path") or ""))
            if not source_path.exists():
                raise FileNotFoundError("source_video_path_missing")
            if not attacked_path.exists():
                raise FileNotFoundError("attacked_video_path_missing")
            evidence_paths = _official_command_evidence_paths(root, config, score_record_id)
            output_json_path = Path(evidence_paths["external_baseline_official_output_path"])
            stdout_path = Path(evidence_paths["external_baseline_official_stdout_path"])
            stderr_path = Path(evidence_paths["external_baseline_official_stderr_path"])
            command_manifest_path = Path(evidence_paths["external_baseline_official_command_manifest_path"])
            output_json_path.parent.mkdir(parents=True, exist_ok=True)
            argv = _format_command(command_template, root, detection_record, output_json_path)
            completed = subprocess.run(argv, text=True, capture_output=True, timeout=timeout_sec)
            _write_text(stdout_path, completed.stdout)
            _write_text(stderr_path, completed.stderr)
            _write_json(command_manifest_path, {
                "manifest_kind": "external_baseline_official_command_evidence",
                "baseline_name": config.baseline_name,
                "external_baseline_score_record_id": score_record_id,
                "trajectory_trace_id": detection_record.get("trajectory_trace_id"),
                "attack_name": detection_record.get("attack_name"),
                "command_argv_digest": build_stable_digest(argv),
                "command_executable": argv[0] if argv else "",
                "command_argument_count": len(argv),
                "command_return_code": completed.returncode,
                "official_output_json_path": str(output_json_path),
                "official_stdout_path": str(stdout_path),
                "official_stderr_path": str(stderr_path),
                "claim_support_status": "external_baseline_official_command_evidence",
            })
            if completed.returncode != 0:
                raise RuntimeError(f"official_command_failed:{completed.returncode}:{completed.stderr[-500:]}")
            payload = _read_official_output(output_json_path)
            validate_clean_negative_payload(payload)
            score_payload = normalized_score_payload(payload)
            score = round(float(score_payload["external_baseline_raw_detector_score"]), 6)
            method_score = safe_float(detection_record.get("S_runtime_attack_detection"), 0.0)
            records.append(with_flow_evidence_protocol_defaults({
                "record_version": "external_baseline_score_v2",
                "external_baseline_score_record_id": score_record_id,
                "external_baseline_name": config.baseline_name,
                "external_baseline_family": baseline_record.get("external_baseline_family") or config.baseline_family,
                "external_baseline_layer": baseline_record.get("external_baseline_layer"),
                "external_baseline_adapter_path": config.adapter_path,
                "external_baseline_command_env_var": config.env_var,
                "external_baseline_command_config_status": "configured",
                "generation_model_id": detection_record.get("generation_model_id"),
                "prompt_id": detection_record.get("prompt_id"),
                "seed_id": detection_record.get("seed_id"),
                "trajectory_trace_id": detection_record.get("trajectory_trace_id"),
                "attack_name": detection_record.get("attack_name"),
                "sample_role": detection_record.get("sample_role"),
                "negative_family": detection_record.get("negative_family"),
                "source_video_path": detection_record.get("source_video_path"),
                "attacked_video_path": detection_record.get("attacked_video_path"),
                "external_baseline_source_video_path": payload.get("external_baseline_source_video_path", payload.get("baseline_source_video_path")),
                "external_baseline_attacked_video_path": payload.get("external_baseline_attacked_video_path", payload.get("baseline_attacked_video_path")),
                "external_baseline_generation_model_id": payload.get("external_baseline_generation_model_id"),
                "external_baseline_official_execution_mode": payload.get(
                    "external_baseline_official_execution_mode",
                    payload.get("official_adapter_status", payload.get("official_bridge_status")),
                ),
                "external_baseline_official_result_bundle_path": payload.get("official_result_bundle_path"),
                "external_baseline_official_execution_manifest_path": payload.get("official_execution_manifest_path"),
                "metric_status": "measured_formal",
                "external_baseline_score_status": "measured_formal",
                "external_baseline_score_source": config.score_source,
                "external_baseline_score_failure_reason": "none",
                "external_baseline_reference_sequence_length": None,
                "external_baseline_observed_sequence_length": None,
                "external_baseline_distance": payload.get("external_baseline_distance"),
                **score_payload,
                "external_baseline_score": score,
                "external_baseline_detected": payload.get("external_baseline_detected", payload.get("detected")),
                "external_baseline_bit_accuracy": payload.get("external_baseline_bit_accuracy", payload.get("bit_accuracy")),
                "external_baseline_clean_negative_score": payload.get("external_baseline_clean_negative_score"),
                "external_baseline_clean_negative_score_semantics": payload.get("external_baseline_clean_negative_score_semantics"),
                "external_baseline_clean_negative_video_path": payload.get("external_baseline_clean_negative_video_path"),
                "external_baseline_threshold": payload.get("external_baseline_threshold", payload.get("threshold")),
                "baseline_score_margin": round(method_score - score, 6),
                "external_baseline_result_used_for_claim": True,
                "claim_support_status": config.claim_support_status,
                **evidence_paths,
            }, trajectory_source_level="external_modern_baseline_official_adapter", claim_support_status=config.claim_support_status))
        except Exception as exc:
            records.append(_unsupported_record(config, baseline_record, detection_record, str(exc), evidence_paths=evidence_paths))
    measured_count = sum(1 for record in records if record.get("external_baseline_score_status") == "measured_formal")
    progress.finish(f"measured={measured_count} unsupported={len(records) - measured_count}")
    return records
