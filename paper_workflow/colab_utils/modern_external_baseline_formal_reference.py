"""现代 external baseline 官方参考运行的 Colab 辅助函数。

该模块的职责是为每个现代视频水印 baseline 提供独立 Notebook 可调用的
clone / build / run / adapt / bundle 闭环。Notebook 只调用这里的函数, 不直接
手写正式 records。official bundle 生成后, 仍由
`experiments.generative_video_model_probe.external_baseline_runner` 统一转写为
`measured_formal` records。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping

from external_baseline.official_runtime_closure import write_official_runtime_closure_requirements
from external_baseline.runtime_trace_io import build_comparison_unit_id, comparable_detection_records
from paper_workflow.colab_utils.stage_package_sync import (
    prepare_colab_stage_layout,
    publish_colab_stage_package,
)
from paper_workflow.notebook_utils import generative_video_model_probe_workflow as probe_workflow


MODERN_EXTERNAL_BASELINE_BUILD_ORDER = (
    "videoseal",
    "vidsig",
    "videomark",
    "videoshield",
    "spdmark",
    "sigmark",
)
REPOSITORY_OFFICIAL_ADAPTER_BASELINES = {
    "videoshield",
    "sigmark",
    "spdmark",
    "videomark",
    "vidsig",
    "videoseal",
}
DEFAULT_NOTEBOOK_ROLE_FOR_LAYOUT = "external_baseline_formal_scoring"
DEFAULT_DRIVE_PROJECT_ROOT = "/content/drive/MyDrive/SSTW"
DEFAULT_WORKFLOW_PROFILE = "validation_scale"
COMMAND_CONFIG_PATH = Path("configs/external_baselines/modern_baseline_colab_commands.json")
SOURCE_REGISTRY_PATH = Path("external_baseline/source_registry.json")


@dataclass(frozen=True)
class ModernExternalBaselineFormalReferenceConfig:
    """描述单个 modern baseline 官方参考运行的最小配置。

    通用工程写法是将 Colab 中可编辑的环境变量收敛到 dataclass, 由 helper
    统一解析路径、命令和产物位置。项目特定要求是 Notebook 只触发仓库 helper:
    先生成 official bundle 与 manifest, 再调用统一 runner 转写 `measured_formal`
    records, 不能在 Notebook cell 中手工拼接正式结果。
    """

    baseline_id: str
    drive_project_root: str
    workflow_profile: str
    repo_root: str
    execute_source_clone: bool
    run_source_intake: bool
    allow_network: bool
    run_official_resource_bootstrap: bool
    generate_auto_supported_bundle: bool
    allow_existing_official_bundle_as_reference_input: bool
    max_records: int | None
    run_official_runtime_closure_preflight: bool
    run_official_result_bundle_preflight: bool
    run_external_baseline_comparison_after_reference: bool
    run_self_containment_after_reference: bool

    def __post_init__(self) -> None:
        """在构造边界校验 baseline 身份和样本上限。"""

        if self.baseline_id not in REPOSITORY_OFFICIAL_ADAPTER_BASELINES:
            raise ValueError(f"未知 modern external baseline: {self.baseline_id}")
        if self.max_records is not None and int(self.max_records) <= 0:
            raise ValueError("max_records 必须为空或正整数")

    def to_dict(self) -> dict[str, Any]:
        """转换为 JSON 兼容字典, 便于写入 manifest。"""

        return asdict(self)


def _utc_now() -> str:
    """返回稳定 UTC 时间戳。"""

    return datetime.now(timezone.utc).isoformat()


def _read_json(path: str | Path) -> dict[str, Any]:
    """读取 JSON 对象, 并兼容 Colab / Windows 常见 UTF-8 BOM。"""

    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON 顶层必须是对象: {path}")
    return payload


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """写出稳定 JSON, 用于跨 Notebook 与脚本审计。"""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def _write_text(path: str | Path, value: str) -> Path:
    """写出命令 stdout / stderr 文本。"""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(value, encoding="utf-8")
    return output_path


def _safe_path_token(value: Any) -> str:
    """将 prompt、seed、attack 等字段转换为 official bundle 路径 token。"""

    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "unknown"))
    return text.strip("_") or "unknown"


def build_official_bundle_record_path(
    bundle_root: str | Path,
    baseline_id: str,
    detection_record: Mapping[str, Any],
) -> Path:
    """构造单条 runtime comparison unit 对应的 official bundle JSON 路径。

    该路径与 `external_baseline.official_eval_adapters.common` 中的读取约定保持一致,
    因而后续 `external_baseline_runner` 可以通过 repository bridge 自动消费该结果。
    """

    prompt = _safe_path_token(detection_record.get("prompt_id"))
    seed = _safe_path_token(detection_record.get("seed_id"))
    attack = _safe_path_token(detection_record.get("attack_name"))
    return Path(bundle_root) / baseline_id / "records" / f"{prompt}__{seed}__{attack}.json"


def build_default_config_from_env(baseline_id: str, repo_root: str | Path = ".") -> ModernExternalBaselineFormalReferenceConfig:
    """从 Colab 环境变量构造单个 baseline 的默认运行配置。"""

    max_records_text = os.environ.get("SSTW_EXTERNAL_BASELINE_REFERENCE_MAX_RECORDS", "").strip()
    max_records = int(max_records_text) if max_records_text else None
    return ModernExternalBaselineFormalReferenceConfig(
        baseline_id=baseline_id,
        drive_project_root=os.environ.get("SSTW_DRIVE_PROJECT_ROOT", DEFAULT_DRIVE_PROJECT_ROOT),
        workflow_profile=os.environ.get("SSTW_WORKFLOW_PROFILE", DEFAULT_WORKFLOW_PROFILE),
        repo_root=str(repo_root),
        execute_source_clone=os.environ.get("SSTW_RUN_EXTERNAL_BASELINE_SOURCE_CLONE", "true").lower() == "true",
        run_source_intake=os.environ.get("SSTW_RUN_EXTERNAL_BASELINE_SOURCE_INTAKE", "true").lower() == "true",
        allow_network=os.environ.get("SSTW_ALLOW_EXTERNAL_BASELINE_RESOURCE_NETWORK", "true").lower() == "true",
        run_official_resource_bootstrap=os.environ.get("SSTW_RUN_OFFICIAL_RESOURCE_BOOTSTRAP_AFTER_REFERENCE", "true").lower() == "true",
        generate_auto_supported_bundle=os.environ.get("SSTW_GENERATE_AUTO_SUPPORTED_OFFICIAL_BUNDLES", "true").lower() == "true",
        allow_existing_official_bundle_as_reference_input=os.environ.get(
            "SSTW_ALLOW_EXISTING_OFFICIAL_BUNDLE_AS_REFERENCE_INPUT",
            "false",
        ).lower() == "true",
        max_records=max_records,
        run_official_runtime_closure_preflight=os.environ.get("SSTW_RUN_OFFICIAL_RUNTIME_CLOSURE_PREFLIGHT", "true").lower() == "true",
        run_official_result_bundle_preflight=os.environ.get("SSTW_RUN_OFFICIAL_RESULT_BUNDLE_PREFLIGHT_AFTER_REFERENCE", "true").lower() == "true",
        run_external_baseline_comparison_after_reference=os.environ.get("SSTW_RUN_EXTERNAL_BASELINE_COMPARISON_AFTER_REFERENCE", "true").lower() == "true",
        run_self_containment_after_reference=os.environ.get("SSTW_RUN_SELF_CONTAINMENT_AFTER_REFERENCE", "true").lower() == "true",
    )


def load_modern_baseline_command_config(config_path: str | Path = COMMAND_CONFIG_PATH) -> dict[str, dict[str, Any]]:
    """读取现代 baseline Colab command 配置, 并按 baseline_id 建索引。"""

    payload = _read_json(config_path)
    rows = payload.get("baseline_command_configs", [])
    if not isinstance(rows, list):
        raise TypeError("baseline_command_configs 必须是列表")
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if isinstance(row, Mapping) and row.get("baseline_id"):
            indexed[str(row["baseline_id"])] = dict(row)
    return indexed


def load_source_registry_index(registry_path: str | Path = SOURCE_REGISTRY_PATH) -> dict[str, dict[str, Any]]:
    """读取 source registry, 并按 baseline_id 建索引。"""

    payload = _read_json(registry_path)
    rows = payload.get("baseline_sources", [])
    if not isinstance(rows, list):
        raise TypeError("baseline_sources 必须是列表")
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if isinstance(row, Mapping) and row.get("baseline_id"):
            indexed[str(row["baseline_id"])] = dict(row)
    return indexed


def official_source_dir_for_baseline(
    baseline_id: str,
    *,
    repo_root: str | Path = ".",
    command_config_path: str | Path = COMMAND_CONFIG_PATH,
    source_registry_path: str | Path = SOURCE_REGISTRY_PATH,
) -> Path:
    """解析当前环境下某个 baseline 的官方源码目录。

    Colab 中优先使用 command config 的 `/content/SSTW/...` 路径; 本地测试或 Windows
    审计时若该路径不存在, 则回退到 source registry 的仓库相对路径。
    """

    command_rows = load_modern_baseline_command_config(command_config_path)
    configured = Path(str(command_rows.get(baseline_id, {}).get("colab_source_dir") or ""))
    if configured.exists():
        return configured
    registry_rows = load_source_registry_index(source_registry_path)
    source_dir = str(registry_rows.get(baseline_id, {}).get("source_dir") or "")
    if not source_dir:
        raise KeyError(f"source registry 缺少 baseline: {baseline_id}")
    return Path(repo_root) / source_dir


def _run_command(
    command: list[str],
    *,
    cwd: str | Path,
    env: Mapping[str, str] | None = None,
    timeout_seconds: float | None = None,
) -> subprocess.CompletedProcess[str]:
    """执行外部命令并捕获输出。

    这里用于 baseline 参考 Notebook 的短命令或官方 adapter 命令。长耗时主流程仍由
    现有 Notebook workflow helper 使用 streaming runner 展示进度。
    """

    merged_env = dict(os.environ)
    if env:
        merged_env.update({str(key): str(value) for key, value in env.items()})
    return subprocess.run(
        command,
        cwd=str(cwd),
        env=merged_env,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
    )


def _run_source_intake(layout: Mapping[str, str], config: ModernExternalBaselineFormalReferenceConfig) -> dict[str, Any]:
    """调用仓库已有 source intake 脚本, 写出 clone / inspection governed artifacts。"""

    command = probe_workflow.build_external_baseline_source_intake_command(
        dict(layout),
        execute_clone=config.execute_source_clone,
    )
    completed = _run_command(command, cwd=config.repo_root)
    return {
        "stage_id": "external_baseline_source_intake",
        "command": command,
        "return_code": completed.returncode,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
        "stage_status": "PASS" if completed.returncode == 0 else "FAIL",
    }


def _run_official_resource_bootstrap(
    layout: Mapping[str, str],
    config: ModernExternalBaselineFormalReferenceConfig,
) -> dict[str, Any]:
    """在单 baseline Notebook 内执行官方资源准备阶段。

    该阶段属于项目特定的 build 边界: 它只尝试准备公开可获得的官方依赖、checkpoint
    或资源路径, 并把无法自动准备的 baseline 写成受治理阻断原因。它不能生成或伪造
    baseline 分数。
    """

    command = probe_workflow.build_external_baseline_official_resource_bootstrap_command(
        dict(layout),
        allow_network=config.allow_network,
    )
    completed = _run_command(command, cwd=config.repo_root)
    decision_path = Path(layout["drive_run_root"]) / "artifacts" / "external_baseline_official_resource_bootstrap_decision.json"
    payload = _read_json(decision_path) if decision_path.exists() else {}
    environment_updates = payload.get("environment_updates", {})
    if isinstance(environment_updates, Mapping):
        for key, value in environment_updates.items():
            if value:
                os.environ[str(key)] = str(value)
    return {
        "stage_id": "external_baseline_official_resource_bootstrap",
        "command": command,
        "return_code": completed.returncode,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
        "stage_status": "PASS" if completed.returncode == 0 else "FAIL",
        "decision_path": str(decision_path),
        "applied_environment_update_keys": sorted(str(key) for key in environment_updates)
        if isinstance(environment_updates, Mapping)
        else [],
    }


def _run_official_runtime_closure_preflight(
    layout: Mapping[str, str],
    config: ModernExternalBaselineFormalReferenceConfig,
) -> dict[str, Any]:
    """写出真实运行闭合要求 artifact, 并自动绑定已存在的默认 Drive 资源。

    该阶段只检查 source、requirements、runtime inputs、官方资源与 official bundle
    cache 是否齐备。它不运行第三方 baseline, 也不生成 `measured_formal` 分数。
    若 Google Drive 中已存在配置文件声明的默认资源路径, 该函数会把对应
    `SSTW_*` 环境变量注入当前 Notebook 进程, 避免用户在每个 baseline cell 中
    反复手动填路径。
    """

    payload = write_official_runtime_closure_requirements(
        layout["drive_run_root"],
        repo_root=config.repo_root,
        resource_root=layout["external_baseline_resource_root"],
        official_result_bundle_root=layout["external_baseline_official_result_bundle_root"],
        baseline_id=config.baseline_id,
    )
    environment_updates = payload.get("environment_updates", {})
    if isinstance(environment_updates, Mapping):
        for key, value in environment_updates.items():
            if value:
                os.environ[str(key)] = str(value)
    return {
        "stage_id": "external_baseline_official_runtime_closure_requirements",
        "stage_status": payload.get("official_runtime_closure_decision"),
        "decision_path": str(
            Path(layout["drive_run_root"])
            / "artifacts"
            / "external_baseline_official_runtime_closure_requirements.json"
        ),
        "runtime_closure_ready_count": payload.get("runtime_closure_ready_count"),
        "runtime_closure_blocked_count": payload.get("runtime_closure_blocked_count"),
        "runtime_closure_blocked_baselines": payload.get("runtime_closure_blocked_baselines", []),
        "runtime_closure_missing_requirement_summary": payload.get("missing_requirement_summary", {}),
        "applied_environment_update_keys": sorted(str(key) for key in environment_updates)
        if isinstance(environment_updates, Mapping)
        else [],
        "claim_support_status": "official_runtime_requirements_preflight_only_not_claim_evidence",
    }


def _runtime_closure_blocks_reference_attempt(
    preflight_result: Mapping[str, Any],
    baseline_id: str,
) -> bool:
    """判断 runtime closure 预检是否已经足以阻断正式参考运行。

    该函数属于项目特定门禁写法。对于 SPDMark 这类需要官方 extractor、
    ground-truth bits 或原生命令的 baseline, 若预检已明确缺少资源, 后续不应再
    逐条调用官方 adapter。这样可以避免生成 69 条重复失败记录, 也能把阻断原因
    保持为 governed preflight artifact。
    """

    if str(preflight_result.get("stage_status") or "") != "FAIL":
        return False
    blocked = [str(item) for item in preflight_result.get("runtime_closure_blocked_baselines", [])]
    return baseline_id in blocked


def _build_runtime_closure_blocked_reference_manifest(
    *,
    baseline_id: str,
    run_root: Path,
    bundle_root: Path,
    official_source_dir: Path,
    selected_record_count: int,
    runtime_closure_preflight_result: Mapping[str, Any],
) -> dict[str, Any]:
    """写出 runtime closure 阻断状态下的 official reference manifest。

    该 manifest 不是论文结果, 也不会伪造分数。它只记录当前 baseline 因缺少
    官方运行资源而无法进入 official bundle 生成阶段, 供 Notebook 和后续审计
    给出明确阻断原因。
    """

    baseline_root = bundle_root / baseline_id
    manifest_path = baseline_root / "official_reference_execution_manifest.json"
    manifest = {
        "manifest_kind": "modern_external_baseline_formal_reference_execution_manifest",
        "baseline_id": baseline_id,
        "run_root": str(run_root),
        "bundle_root": str(bundle_root),
        "official_source_dir": str(official_source_dir),
        "execution_status": "blocked_missing_official_runtime_requirements",
        "official_reference_blocker": "runtime_closure_preflight_failed",
        "runtime_closure_preflight_result": dict(runtime_closure_preflight_result),
        "input_runtime_detection_record_count": int(selected_record_count),
        "generated_bundle_record_count": 0,
        "failed_bundle_record_count": int(selected_record_count),
        "successes": [],
        "failures": [],
        "created_at_utc": _utc_now(),
        "claim_support_status": "official_reference_runtime_requirements_blocked_not_claim_evidence",
    }
    _write_json(manifest_path, manifest)
    return manifest


def _adapter_module_for_baseline(baseline_id: str) -> str:
    """返回 repository official eval adapter 的 Python module 名称。"""

    if baseline_id not in REPOSITORY_OFFICIAL_ADAPTER_BASELINES:
        raise ValueError(f"不支持的 baseline: {baseline_id}")
    return f"external_baseline.official_eval_adapters.{baseline_id}"


def build_official_adapter_command(
    *,
    baseline_id: str,
    official_source_dir: str | Path,
    detection_record: Mapping[str, Any],
    output_json_path: str | Path,
) -> list[str]:
    """构造单条 comparison unit 的 repository official adapter 命令。"""

    return [
        sys.executable,
        "-m",
        _adapter_module_for_baseline(baseline_id),
        "--official-source-dir",
        str(official_source_dir),
        "--source-video",
        str(detection_record.get("source_video_path") or ""),
        "--attacked-video",
        str(detection_record.get("attacked_video_path") or ""),
        "--attack-name",
        str(detection_record.get("attack_name") or ""),
        "--official-output-json",
        str(output_json_path),
        "--run-root",
        str(detection_record.get("run_root") or ""),
        "--prompt-id",
        str(detection_record.get("prompt_id") or ""),
        "--seed-id",
        str(detection_record.get("seed_id") or ""),
        "--trajectory-trace-id",
        str(detection_record.get("trajectory_trace_id") or ""),
    ]


def _enrich_official_bundle_payload(
    path: Path,
    manifest_path: Path,
    baseline_id: str,
    detection_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """为官方 bundle JSON 补充本项目运行 manifest provenance。

    该函数只补充 provenance 字段, 不改写官方分数。这样 official bundle 可以被后续
    bridge 读取, 同时保留本次 Notebook 运行的可审计入口。
    """

    payload = _read_json(path)
    protocol_fields: dict[str, Any] = {}
    if detection_record is not None:
        protocol_fields = {
            "official_reference_protocol_anchor": "same_prompt_seed_attack_runtime_comparison_unit",
            "runtime_comparison_unit_id": build_comparison_unit_id(baseline_id, detection_record),
            "prompt_id": detection_record.get("prompt_id"),
            "seed_id": detection_record.get("seed_id"),
            "attack_name": detection_record.get("attack_name"),
            "trajectory_trace_id": detection_record.get("trajectory_trace_id"),
            "source_video_path": detection_record.get("source_video_path"),
            "attacked_video_path": detection_record.get("attacked_video_path"),
        }
    existing_provenance = str(payload.get("official_result_provenance") or "")
    repository_provenance = "repository_generated_from_third_party_official_code"
    enriched = {
        **payload,
        **{
            key: value
            for key, value in protocol_fields.items()
            if payload.get(key) is None or payload.get(key) == ""
        },
        "official_baseline_id": payload.get("official_baseline_id") or baseline_id,
        "official_result_provenance": repository_provenance
        if existing_provenance in {"", "third_party_official_code"}
        else existing_provenance,
        "official_execution_manifest_path": payload.get("official_execution_manifest_path") or str(manifest_path),
    }
    _write_json(path, enriched)
    return enriched


def _run_generic_repository_official_adapter(
    *,
    baseline_id: str,
    run_root: Path,
    bundle_root: Path,
    official_source_dir: Path,
    repo_root: Path,
    max_records: int | None,
    allow_existing_official_bundle_as_reference_input: bool,
) -> dict[str, Any]:
    """对非 VideoSeal baseline 逐条调用 repository official adapter。

    该路径会 fail closed: 若官方源码、权重、key、message 或用户 native command 缺失,
    对应记录会进入 failures, 不会写出伪造分数。
    """

    records = comparable_detection_records(run_root)
    if max_records is not None:
        records = records[: int(max_records)]
    baseline_root = bundle_root / baseline_id
    log_root = baseline_root / "logs"
    manifest_path = baseline_root / "official_reference_execution_manifest.json"
    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    timeout_seconds = float(os.environ.get("SSTW_OFFICIAL_BASELINE_NATIVE_TIMEOUT_SEC", "3600"))
    env = {
        "PYTHONPATH": str(repo_root) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        "SSTW_EXTERNAL_BASELINE_OFFICIAL_RESULT_BUNDLE_ROOT": str(bundle_root),
    }
    if not allow_existing_official_bundle_as_reference_input:
        env["SSTW_DISABLE_OFFICIAL_RESULT_BUNDLE_READ"] = "true"

    for index, record in enumerate(records, start=1):
        output_json_path = build_official_bundle_record_path(bundle_root, baseline_id, record)
        output_stem = output_json_path.stem
        command = build_official_adapter_command(
            baseline_id=baseline_id,
            official_source_dir=official_source_dir,
            detection_record={**record, "run_root": str(run_root)},
            output_json_path=output_json_path,
        )
        completed = _run_command(command, cwd=repo_root, env=env, timeout_seconds=timeout_seconds)
        stdout_path = log_root / f"{output_stem}_stdout.txt"
        stderr_path = log_root / f"{output_stem}_stderr.txt"
        _write_text(stdout_path, completed.stdout)
        _write_text(stderr_path, completed.stderr)
        row = {
            "baseline_id": baseline_id,
            "record_index": index,
            "prompt_id": record.get("prompt_id"),
            "seed_id": record.get("seed_id"),
            "attack_name": record.get("attack_name"),
            "trajectory_trace_id": record.get("trajectory_trace_id"),
            "official_output_json_path": str(output_json_path),
            "official_stdout_path": str(stdout_path),
            "official_stderr_path": str(stderr_path),
            "command_return_code": completed.returncode,
        }
        if completed.returncode == 0 and output_json_path.exists():
            payload = _enrich_official_bundle_payload(output_json_path, manifest_path, baseline_id, record)
            successes.append({
                **row,
                "official_adapter_status": payload.get("official_adapter_status"),
                "official_result_provenance": payload.get("official_result_provenance"),
            })
        else:
            failures.append({
                **row,
                "failure_reason": completed.stderr[-1000:] or "official_output_json_missing_or_command_failed",
            })

    manifest = {
        "manifest_kind": "modern_external_baseline_formal_reference_execution_manifest",
        "baseline_id": baseline_id,
        "run_root": str(run_root),
        "bundle_root": str(bundle_root),
        "official_source_dir": str(official_source_dir),
        "allow_existing_official_bundle_as_reference_input": bool(allow_existing_official_bundle_as_reference_input),
        "input_runtime_detection_record_count": len(records),
        "generated_bundle_record_count": len(successes),
        "failed_bundle_record_count": len(failures),
        "successes": successes,
        "failures": failures[:20],
        "created_at_utc": _utc_now(),
        "claim_support_status": "official_reference_execution_evidence_not_measured_formal_record",
    }
    _write_json(manifest_path, manifest)
    for success in successes:
        path = Path(success["official_output_json_path"])
        if path.exists():
            matching_record = next(
                (
                    record
                    for record in records
                    if str(record.get("prompt_id") or "") == str(success.get("prompt_id") or "")
                    and str(record.get("seed_id") or "") == str(success.get("seed_id") or "")
                    and str(record.get("attack_name") or "") == str(success.get("attack_name") or "")
                ),
                None,
            )
            _enrich_official_bundle_payload(path, manifest_path, baseline_id, matching_record)
    return manifest


def _run_videoseal_reference(
    *,
    run_root: Path,
    bundle_root: Path,
    official_source_dir: Path,
    max_records: int | None,
    generate_auto_supported_bundle: bool,
) -> dict[str, Any]:
    """运行 VideoSeal 官方 API bundle 生成路径。"""

    if not generate_auto_supported_bundle:
        return {
            "manifest_kind": "modern_external_baseline_formal_reference_execution_manifest",
            "baseline_id": "videoseal",
            "run_root": str(run_root),
            "bundle_root": str(bundle_root),
            "official_source_dir": str(official_source_dir),
            "input_runtime_detection_record_count": len(comparable_detection_records(run_root)),
            "generated_bundle_record_count": 0,
            "failed_bundle_record_count": 0,
            "reference_status": "skipped_generate_auto_supported_bundle_false",
            "claim_support_status": "official_reference_execution_skipped_not_claim_evidence",
        }

    from external_baseline.official_bundle_generator import generate_videoseal_official_bundle

    manifest = generate_videoseal_official_bundle(
        run_root,
        bundle_root,
        source_dir=official_source_dir,
        max_records=max_records,
    )
    manifest_path = bundle_root / "videoseal" / "official_bundle_generation_manifest.json"
    for record in comparable_detection_records(run_root)[: max_records or None]:
        output_json_path = build_official_bundle_record_path(bundle_root, "videoseal", record)
        if output_json_path.exists():
            _enrich_official_bundle_payload(output_json_path, manifest_path, "videoseal", record)
    return {
        **manifest,
        "manifest_kind": "modern_external_baseline_formal_reference_execution_manifest",
        "official_source_dir": str(official_source_dir),
    }


def _run_sigmark_hunyuan_reference(
    *,
    run_root: Path,
    bundle_root: Path,
    official_source_dir: Path,
    repo_root: Path,
    resource_root: str | Path,
    max_records: int | None,
) -> dict[str, Any]:
    """运行 SIGMark 官方 Hunyuan gen->extract official bundle 生成路径。

    该路径属于项目特定的 external baseline 自包含实现: Notebook 在项目内调用
    SIGMark 官方 `main.py --mode=gen` 和 `--mode=extract`, 再把官方 bit accuracy
    转成 official bundle。它不直接写 `measured_formal`, 后续仍由统一 runner 转写。
    """

    from external_baseline.sigmark_official_hunyuan_runtime import (
        build_default_sigmark_official_hunyuan_config_from_env,
        run_sigmark_official_hunyuan_runtime,
    )

    sigmark_config = build_default_sigmark_official_hunyuan_config_from_env(
        run_root=run_root,
        bundle_root=bundle_root,
        source_dir=official_source_dir,
        repo_root=repo_root,
        resource_root=resource_root,
        max_records=max_records,
    )
    return run_sigmark_official_hunyuan_runtime(sigmark_config)


def _run_videomark_official_reference(
    *,
    run_root: Path,
    bundle_root: Path,
    official_source_dir: Path,
    repo_root: Path,
    resource_root: str | Path,
    max_records: int | None,
) -> dict[str, Any]:
    """运行 VideoMark 官方 embedding / extraction / temporal tamper official bundle 生成路径。

    该路径属于项目特定的 external baseline 自包含实现: Notebook 在项目内调用
    VideoMark 官方 `embedding_and_extraction.py` 和 `temporal_tamper.py`, 再把官方
    `temporal_results.json` 转成 official bundle。它不直接写 `measured_formal`,
    后续仍由统一 runner 转写。
    """

    from external_baseline.videomark_official_runtime import (
        build_default_videomark_official_config_from_env,
        run_videomark_official_runtime,
    )

    videomark_config = build_default_videomark_official_config_from_env(
        run_root=run_root,
        bundle_root=bundle_root,
        source_dir=official_source_dir,
        repo_root=repo_root,
        resource_root=resource_root,
        max_records=max_records,
    )
    return run_videomark_official_runtime(videomark_config)


def _run_videoshield_official_reference(
    *,
    run_root: Path,
    bundle_root: Path,
    official_source_dir: Path,
    repo_root: Path,
    resource_root: str | Path,
    max_records: int | None,
) -> dict[str, Any]:
    """运行 VideoShield 官方 generation / inversion official bundle 生成路径。

    VideoShield 属于生成过程中嵌入 latent watermark 的方法。项目特定要求是:
    不能把 SSTW / Wan 生成的视频直接送入 VideoShield 反演逻辑当作 baseline,
    而必须在项目内调用 VideoShield 官方 watermark 生成流程得到自己的
    watermarked 视频, 再按相同 prompt / seed / attack comparison unit 写出
    official bundle。
    """

    from external_baseline.videoshield_official_runtime import (
        build_default_videoshield_official_config_from_env,
        run_videoshield_official_runtime,
    )

    videoshield_config = build_default_videoshield_official_config_from_env(
        run_root=run_root,
        bundle_root=bundle_root,
        source_dir=official_source_dir,
        repo_root=repo_root,
        resource_root=resource_root,
        max_records=max_records,
    )
    return run_videoshield_official_runtime(videoshield_config)


def _run_vidsig_official_reference(
    *,
    run_root: Path,
    bundle_root: Path,
    official_source_dir: Path,
    repo_root: Path,
    resource_root: str,
    max_records: int | None,
) -> dict[str, Any]:
    """运行 VidSig 官方 generate_ms -> attack.py bundle 生成路径。

    VidSig 属于生成过程中嵌入水印的方法。项目特定要求是: 不能把 SSTW / Wan
    生成的视频直接送入 VidSig detector 后当作 baseline 结果, 而必须在项目内先
    调用 VidSig 官方生成流程得到自己的 clean / watermarked 视频, 再按相同
    prompt / seed / attack comparison unit 写出 official bundle。
    """

    from external_baseline.vidsig_official_runtime import (
        build_default_vidsig_official_config_from_env,
        run_vidsig_official_runtime,
    )

    vidsig_config = build_default_vidsig_official_config_from_env(
        run_root=run_root,
        bundle_root=bundle_root,
        source_dir=official_source_dir,
        repo_root=repo_root,
        resource_root=resource_root,
        max_records=max_records,
    )
    return run_vidsig_official_runtime(vidsig_config)


def _build_unified_formal_scoring_environment(
    layout: Mapping[str, str],
    config: ModernExternalBaselineFormalReferenceConfig,
) -> dict[str, str]:
    """构造统一 measured_formal 转写阶段需要的环境变量。

    各 baseline Notebook 只生成自己的 official bundle。统一转写仍复用
    `external_baseline_runner` 和现代 command adapter, 因此这里必须为 6 个现代
    baseline 同时注入外层 bridge command 和内层 repository official adapter command。
    已由用户显式设置的环境变量保持最高优先级。
    """

    bridge_templates = probe_workflow.build_modern_baseline_official_bridge_command_templates(config.workflow_profile)
    repository_official_templates = probe_workflow.build_repository_official_baseline_eval_command_templates(
        config.workflow_profile,
    )
    command_template_source: dict[str, str] = {}
    command_template_source.update(bridge_templates)
    command_template_source.update(repository_official_templates)
    command_template_source.update({key: value for key, value in os.environ.items() if value})
    outer_command_env = probe_workflow.build_modern_baseline_command_env(
        config.workflow_profile,
        command_template_source,
    )
    env: dict[str, str] = {
        "PYTHONPATH": str(config.repo_root) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        "SSTW_EXTERNAL_BASELINE_OFFICIAL_RESULT_BUNDLE_ROOT": layout["external_baseline_official_result_bundle_root"],
        "SSTW_EXTERNAL_BASELINE_EVIDENCE_PATHS": os.pathsep.join([
            layout["external_baseline_official_result_bundle_root"],
            *[
                item
                for item in os.environ.get("SSTW_EXTERNAL_BASELINE_EVIDENCE_PATHS", "").split(os.pathsep)
                if item
            ],
        ]),
    }
    env.update(repository_official_templates)
    env.update(outer_command_env)
    for key, value in os.environ.items():
        if key.startswith("SSTW_") and value:
            env[key] = value
    return env


def _write_unified_formal_scoring_preflight_artifacts(
    layout: Mapping[str, str],
    config: ModernExternalBaselineFormalReferenceConfig,
    scoring_env: Mapping[str, str],
) -> dict[str, Any]:
    """为单 baseline Notebook 写出统一转写前的 command preflight artifacts。"""

    summary = probe_workflow.write_modern_baseline_colab_command_config_summary(
        layout,
        profile=config.workflow_profile,
    )
    bridge_decision = probe_workflow.write_modern_baseline_official_bridge_preflight_decision(
        layout,
        profile=config.workflow_profile,
        command_env=scoring_env,
        use_bridge_commands=True,
        require_bridge_official_commands=True,
    )
    command_env = {
        key: value
        for key, value in scoring_env.items()
        if key.startswith("SSTW_") and key.endswith("_EVAL_COMMAND") and "_OFFICIAL_" not in key
    }
    baseline_preflight = probe_workflow.write_external_baseline_colab_preflight_decision(
        layout,
        profile=config.workflow_profile,
        command_env=command_env,
        require_modern_baseline_commands_for_paper_gate=True,
        run_external_baseline_source_clone=config.execute_source_clone,
        evidence_paths=[layout["external_baseline_official_result_bundle_root"]],
    )
    return {
        "stage_id": "external_baseline_unified_formal_scoring_preflight",
        "command_template_summary_path": str(Path(layout["drive_run_root"]) / "artifacts" / "external_baseline_command_template_summary.json"),
        "bridge_preflight_decision": bridge_decision.get("external_baseline_official_bridge_preflight_decision"),
        "external_baseline_colab_preflight_decision": baseline_preflight.get("external_baseline_colab_preflight_decision"),
        "required_modern_external_baseline_adapter_names": summary.get("required_modern_external_baseline_adapter_names", []),
        "configured_command_env_var_count": baseline_preflight.get("external_baseline_colab_preflight_configured_env_var_count"),
        "claim_support_status": "formal_scoring_preflight_only_not_claim_evidence",
    }


def _run_optional_followup_commands(
    layout: Mapping[str, str],
    config: ModernExternalBaselineFormalReferenceConfig,
) -> list[dict[str, Any]]:
    """按需运行 official bundle preflight、comparison 和 self-containment。"""

    commands: list[tuple[str, list[str]]] = []
    if config.run_official_result_bundle_preflight:
        commands.append((
            "external_baseline_official_result_bundle_preflight",
            probe_workflow.build_external_baseline_official_result_bundle_preflight_command(dict(layout)),
        ))
    if config.run_external_baseline_comparison_after_reference:
        commands.append((
            "external_baseline_unified_measured_formal_scoring",
            probe_workflow.build_external_baseline_comparison_command(dict(layout)),
        ))
    if config.run_self_containment_after_reference:
        commands.append((
            "external_baseline_self_containment_decision",
            probe_workflow.build_external_baseline_self_containment_decision_command(dict(layout)),
        ))

    rows: list[dict[str, Any]] = []
    env = _build_unified_formal_scoring_environment(layout, config)
    rows.append(_write_unified_formal_scoring_preflight_artifacts(layout, config, env))
    for stage_id, command in commands:
        completed = _run_command(command, cwd=config.repo_root, env=env)
        rows.append({
            "stage_id": stage_id,
            "command": command,
            "return_code": completed.returncode,
            "stdout_tail": completed.stdout[-2000:],
            "stderr_tail": completed.stderr[-2000:],
            "stage_status": "PASS" if completed.returncode == 0 else "FAIL",
        })
    return rows


def run_modern_external_baseline_formal_reference_plan(
    config: ModernExternalBaselineFormalReferenceConfig,
) -> dict[str, Any]:
    """执行单个 modern external baseline 的官方参考 bundle 生成计划。

    该函数按同一 prompt / seed / attack comparison unit 生成官方结果缓存和执行
    manifest。默认会继续调用 `external_baseline_runner`, 由统一 command adapter
    读取这些 bundle 并转写为 `measured_formal` records。若其它 baseline 的 bundle
    尚未完成, 统一转写会为它们保留 governed unsupported rows。
    """

    repo_root = Path(config.repo_root).resolve()
    layout = probe_workflow.ensure_drive_layout(
        config.drive_project_root,
        workflow_profile=config.workflow_profile,
        notebook_role=DEFAULT_NOTEBOOK_ROLE_FOR_LAYOUT,
    )
    layout = prepare_colab_stage_layout(
        layout,
        notebook_role=DEFAULT_NOTEBOOK_ROLE_FOR_LAYOUT,
        baseline_id=config.baseline_id,
    )
    run_root = Path(layout["drive_run_root"])
    bundle_root = Path(layout["external_baseline_official_result_bundle_root"])
    os.environ["SSTW_EXTERNAL_BASELINE_OFFICIAL_RESULT_BUNDLE_ROOT"] = str(bundle_root)
    os.environ.setdefault("SSTW_EXTERNAL_BASELINE_RESOURCE_ROOT", layout["external_baseline_resource_root"])

    source_intake_result = (
        _run_source_intake(layout, config)
        if config.run_source_intake
        else {"stage_id": "external_baseline_source_intake", "stage_status": "SKIPPED"}
    )
    resource_bootstrap_result = (
        _run_official_resource_bootstrap(layout, config)
        if config.run_official_resource_bootstrap
        else {"stage_id": "external_baseline_official_resource_bootstrap", "stage_status": "SKIPPED"}
    )
    runtime_closure_preflight_result = (
        _run_official_runtime_closure_preflight(layout, config)
        if config.run_official_runtime_closure_preflight
        else {"stage_id": "external_baseline_official_runtime_closure_requirements", "stage_status": "SKIPPED"}
    )
    official_source_dir = official_source_dir_for_baseline(config.baseline_id, repo_root=repo_root)
    records = comparable_detection_records(run_root)
    if config.max_records is not None:
        selected_record_count = min(len(records), int(config.max_records))
    else:
        selected_record_count = len(records)

    runtime_closure_blocked = _runtime_closure_blocks_reference_attempt(
        runtime_closure_preflight_result,
        config.baseline_id,
    )

    if runtime_closure_blocked:
        reference_manifest = _build_runtime_closure_blocked_reference_manifest(
            baseline_id=config.baseline_id,
            run_root=run_root,
            bundle_root=bundle_root,
            official_source_dir=official_source_dir,
            selected_record_count=selected_record_count,
            runtime_closure_preflight_result=runtime_closure_preflight_result,
        )
    elif config.baseline_id == "videoseal":
        reference_manifest = _run_videoseal_reference(
            run_root=run_root,
            bundle_root=bundle_root,
            official_source_dir=official_source_dir,
            max_records=config.max_records,
            generate_auto_supported_bundle=config.generate_auto_supported_bundle,
        )
    elif config.baseline_id == "sigmark" and os.environ.get(
        "SSTW_RUN_SIGMARK_OFFICIAL_HUNYUAN_PIPELINE",
        "true",
    ).lower() == "true":
        reference_manifest = _run_sigmark_hunyuan_reference(
            run_root=run_root,
            bundle_root=bundle_root,
            official_source_dir=official_source_dir,
            repo_root=repo_root,
            resource_root=layout["external_baseline_resource_root"],
            max_records=config.max_records,
        )
    elif config.baseline_id == "videomark" and os.environ.get(
        "SSTW_RUN_VIDEOMARK_OFFICIAL_PIPELINE",
        "true",
    ).lower() == "true":
        reference_manifest = _run_videomark_official_reference(
            run_root=run_root,
            bundle_root=bundle_root,
            official_source_dir=official_source_dir,
            repo_root=repo_root,
            resource_root=layout["external_baseline_resource_root"],
            max_records=config.max_records,
        )
    elif config.baseline_id == "videoshield" and os.environ.get(
        "SSTW_RUN_VIDEOSHIELD_OFFICIAL_PIPELINE",
        "true",
    ).lower() == "true":
        reference_manifest = _run_videoshield_official_reference(
            run_root=run_root,
            bundle_root=bundle_root,
            official_source_dir=official_source_dir,
            repo_root=repo_root,
            resource_root=layout["external_baseline_resource_root"],
            max_records=config.max_records,
        )
    elif config.baseline_id == "vidsig" and os.environ.get(
        "SSTW_RUN_VIDSIG_OFFICIAL_PIPELINE",
        "true",
    ).lower() == "true":
        reference_manifest = _run_vidsig_official_reference(
            run_root=run_root,
            bundle_root=bundle_root,
            official_source_dir=official_source_dir,
            repo_root=repo_root,
            resource_root=layout["external_baseline_resource_root"],
            max_records=config.max_records,
        )
    else:
        reference_manifest = _run_generic_repository_official_adapter(
            baseline_id=config.baseline_id,
            run_root=run_root,
            bundle_root=bundle_root,
            official_source_dir=official_source_dir,
            repo_root=repo_root,
            max_records=config.max_records,
            allow_existing_official_bundle_as_reference_input=config.allow_existing_official_bundle_as_reference_input,
        )

    followup_results = (
        [
            {
                "stage_id": "external_baseline_followup_commands",
                "stage_status": "SKIPPED",
                "skip_reason": "runtime_closure_preflight_failed_for_selected_baseline",
                "claim_support_status": "followup_skipped_not_claim_evidence",
            }
        ]
        if runtime_closure_blocked
        else _run_optional_followup_commands(layout, config)
    )
    generated_count = int(reference_manifest.get("generated_bundle_record_count") or 0)
    failed_count = int(reference_manifest.get("failed_bundle_record_count") or 0)
    reference_decision = "PASS" if selected_record_count > 0 and generated_count == selected_record_count and failed_count == 0 else "FAIL"
    if runtime_closure_blocked:
        reference_status = "blocked_missing_official_runtime_requirements"
    elif selected_record_count == 0:
        reference_status = "missing_runtime_detection_records"
    elif generated_count != selected_record_count:
        reference_status = "bundle_record_coverage_incomplete"
    elif failed_count:
        reference_status = "official_reference_failures_present"
    else:
        reference_status = "official_reference_bundle_complete"
    decision = {
        "artifact_name": f"{config.baseline_id}_formal_reference_decision.json",
        "manifest_kind": "modern_external_baseline_formal_reference_decision",
        "baseline_id": config.baseline_id,
        "build_order": list(MODERN_EXTERNAL_BASELINE_BUILD_ORDER),
        "config": config.to_dict(),
        "layout": dict(layout),
        "run_root": str(run_root),
        "official_result_bundle_root": str(bundle_root),
        "official_source_dir": str(official_source_dir),
        "runtime_detection_record_count": len(records),
        "selected_runtime_detection_record_count": selected_record_count,
        "generated_bundle_record_count": generated_count,
        "failed_bundle_record_count": failed_count,
        "formal_reference_decision": reference_decision,
        "formal_reference_status": reference_status,
        "source_intake_result": source_intake_result,
        "resource_bootstrap_result": resource_bootstrap_result,
        "runtime_closure_preflight_result": runtime_closure_preflight_result,
        "reference_manifest": reference_manifest,
        "followup_results": followup_results,
        "created_at_utc": _utc_now(),
        "claim_support_status": "official_reference_bundle_ready_not_measured_formal_record"
        if reference_decision == "PASS"
        else "official_reference_bundle_blocked_not_claim_evidence",
    }
    decision_path = run_root / "artifacts" / "external_baseline_formal_reference" / f"{config.baseline_id}_formal_reference_decision.json"
    _write_json(decision_path, decision)
    package_manifest = publish_colab_stage_package(
        layout,
        notebook_role=DEFAULT_NOTEBOOK_ROLE_FOR_LAYOUT,
        baseline_id=config.baseline_id,
        include_videos=os.environ.get("SSTW_INCLUDE_VIDEOS_IN_PACKAGE", "true").lower() == "true",
    )
    decision["stage_package_publish_result"] = package_manifest
    _write_json(decision_path, decision)
    return decision


def run_default_modern_external_baseline_formal_reference_plan(
    baseline_id: str,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    """使用环境变量默认值执行单个 baseline 的官方参考计划。"""

    return run_modern_external_baseline_formal_reference_plan(
        build_default_config_from_env(baseline_id, repo_root=repo_root)
    )
