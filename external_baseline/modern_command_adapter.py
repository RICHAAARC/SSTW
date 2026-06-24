"""现代视频水印 baseline 官方命令适配工具。"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from typing import Any, Mapping

from external_baseline.runtime_trace_io import build_comparison_unit_id, comparable_detection_records, safe_float
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


def _extract_score(payload: Mapping[str, Any]) -> float:
    """从常见官方输出字段中提取 baseline score。"""
    for field in (
        "external_baseline_score",
        "watermark_score",
        "detection_score",
        "score",
        "bit_accuracy",
        "confidence",
    ):
        if field in payload:
            return safe_float(payload.get(field), 0.0)
    if "detected" in payload:
        return 1.0 if bool(payload.get("detected")) else 0.0
    raise ValueError("official_output_missing_score")


def _unsupported_record(
    config: ModernBaselineCommandConfig,
    baseline_record: Mapping[str, Any],
    detection_record: Mapping[str, Any],
    reason: str,
) -> dict[str, Any]:
    """构造现代 baseline 无法正式评分时的 governed unsupported record。"""
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
        "external_baseline_detected": None,
        "external_baseline_bit_accuracy": None,
        "external_baseline_threshold": None,
        "baseline_score_margin": None,
        "external_baseline_result_used_for_claim": False,
        "claim_support_status": "modern_external_baseline_formal_adapter_not_measured",
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
    if command_template is None:
        return [_unsupported_record(config, baseline_record, record, "official_command_not_configured") for record in detection_records]

    timeout_sec = safe_float(os.environ.get("SSTW_EXTERNAL_BASELINE_TIMEOUT_SEC"), 1800.0)
    for detection_record in detection_records:
        try:
            source_path = Path(str(detection_record.get("source_video_path") or ""))
            attacked_path = Path(str(detection_record.get("attacked_video_path") or ""))
            if not source_path.exists():
                raise FileNotFoundError("source_video_path_missing")
            if not attacked_path.exists():
                raise FileNotFoundError("attacked_video_path_missing")
            with tempfile.TemporaryDirectory(prefix=f"sstw_{config.baseline_name}_") as temp_dir:
                output_json_path = Path(temp_dir) / "official_output.json"
                argv = _format_command(command_template, root, detection_record, output_json_path)
                completed = subprocess.run(argv, text=True, capture_output=True, timeout=timeout_sec)
                if completed.returncode != 0:
                    raise RuntimeError(f"official_command_failed:{completed.returncode}:{completed.stderr[-500:]}")
                payload = _read_official_output(output_json_path)
            score = round(_extract_score(payload), 6)
            method_score = safe_float(detection_record.get("S_runtime_attack_detection"), 0.0)
            records.append(with_flow_evidence_protocol_defaults({
                "record_version": "external_baseline_score_v2",
                "external_baseline_score_record_id": build_comparison_unit_id(config.baseline_name, detection_record),
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
                "metric_status": "measured_formal",
                "external_baseline_score_status": "measured_formal",
                "external_baseline_score_source": config.score_source,
                "external_baseline_score_failure_reason": "none",
                "external_baseline_reference_sequence_length": None,
                "external_baseline_observed_sequence_length": None,
                "external_baseline_distance": payload.get("external_baseline_distance"),
                "external_baseline_score": score,
                "external_baseline_detected": payload.get("external_baseline_detected", payload.get("detected")),
                "external_baseline_bit_accuracy": payload.get("external_baseline_bit_accuracy", payload.get("bit_accuracy")),
                "external_baseline_threshold": payload.get("external_baseline_threshold", payload.get("threshold")),
                "baseline_score_margin": round(method_score - score, 6),
                "external_baseline_result_used_for_claim": True,
                "claim_support_status": config.claim_support_status,
            }, trajectory_source_level="external_modern_baseline_official_adapter", claim_support_status=config.claim_support_status))
        except Exception as exc:
            records.append(_unsupported_record(config, baseline_record, detection_record, str(exc)))
    return records
