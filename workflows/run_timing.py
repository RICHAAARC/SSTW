"""记录 Colab Notebook 入口的总耗时与阶段耗时。

该模块的职责是给 Notebook 入口层提供统一、可复用的计时能力。Notebook 本身
仍只负责选择 workflow profile 并调用 repository helper, 不直接写正式 records、
tables、figures 或 reports。这里写出的 timing manifest 仅用于运行成本估算,
不能作为论文效果 claim 的证据。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any, Callable, Mapping, TypeVar

from evaluation.protocol.package_naming import current_short_commit, sanitize_filename_token
from evaluation.protocol.record_writer import write_json, write_jsonl

NOTEBOOK_STARTED_AT_UTC_ENV = "SSTW_NOTEBOOK_STARTED_AT_UTC"
NOTEBOOK_STARTED_AT_PERF_COUNTER_ENV = "SSTW_NOTEBOOK_STARTED_AT_PERF_COUNTER"
NOTEBOOK_TIMING_SCOPE_ENV = "SSTW_NOTEBOOK_TIMING_SCOPE"
NOTEBOOK_RUNTIME_REPORT_RELATIVE_PATH = "artifacts/notebook_runtime_report.json"
NOTEBOOK_RUN_TIMING_MANIFEST_RELATIVE_PATH = "artifacts/notebook_run_timing_manifest.json"
NOTEBOOK_STAGE_TIMING_RECORDS_RELATIVE_PATH = "records/notebook_stage_timing_records.jsonl"
NOTEBOOK_TIMING_RECORD_VERSION = "notebook_run_timing_v1"
NOTEBOOK_TIMING_CLAIM_SUPPORT_STATUS = "notebook_runtime_estimation_only_not_claim_evidence"

T = TypeVar("T")


def utc_now_iso() -> str:
    """返回带时区的 UTC 时间字符串。

    该函数属于通用工程写法。统一使用 UTC 可以避免 Colab、Windows 本地和
    Google Drive 同步环境之间的时区差异。
    """

    return datetime.now(timezone.utc).isoformat()


def _utc_time_for_id() -> str:
    """返回适合 run id 使用的 UTC 时间片段。"""

    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def default_notebook_started_at_utc() -> str:
    """优先读取共享 Colab 入口层写入的开始时间。"""

    return os.environ.get(NOTEBOOK_STARTED_AT_UTC_ENV) or utc_now_iso()


def default_notebook_start_perf_counter() -> float:
    """优先读取共享 Colab 入口层写入的单调时钟起点。"""

    value = os.environ.get(NOTEBOOK_STARTED_AT_PERF_COUNTER_ENV)
    if value:
        try:
            return float(value)
        except ValueError:
            return time.perf_counter()
    return time.perf_counter()


def _round_seconds(value: float) -> float:
    """把秒数统一保留到毫秒, 便于人工比较不同配置用时。"""

    return round(float(value), 3)


def _failure_summary(exc: BaseException | None) -> dict[str, str]:
    """把异常转换为可落盘的最小失败摘要。"""

    if exc is None:
        return {}
    return {
        "stage_failure_type": type(exc).__name__,
        "stage_failure_message": str(exc),
    }


def _normalize_stage_status(raw_status: str) -> str:
    """将不同 helper 的 PASS / SKIPPED 等状态归一到计时层语义。"""

    normalized = str(raw_status or "").strip().lower()
    if normalized in {"pass", "completed", "success", "succeeded"}:
        return "completed"
    if normalized in {"skip", "skipped"}:
        return "skipped"
    if normalized in {"fail", "failed", "error"}:
        return "failed"
    return normalized or "completed"


@dataclass
class NotebookRunTimer:
    """记录单次 Notebook repository stage plan 的总耗时。

    该类属于 Notebook 入口复用层。主流程和 external baseline Notebook 都通过
    repository helper 创建该对象, 因而后续新增阶段或调整命令时不需要修改
    Notebook cell。
    """

    layout: Mapping[str, str]
    notebook_role: str
    workflow_profile: str
    baseline_id: str = ""
    timing_scope: str = "repository_stage_plan"
    enabled_stage_plan: list[str] = field(default_factory=list)
    repo_root: str | Path | None = None
    started_at_utc: str = field(default_factory=default_notebook_started_at_utc)
    start_perf_counter: float = field(default_factory=default_notebook_start_perf_counter)
    stage_records: list[dict[str, Any]] = field(default_factory=list)
    notebook_run_id: str = ""
    git_short_commit: str = ""
    finished: bool = False
    last_manifest: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """补齐稳定 run id 与代码版本。"""

        if not self.git_short_commit:
            self.git_short_commit = current_short_commit(self.repo_root)
        if not self.notebook_run_id:
            parts = [
                self.workflow_profile,
                self.notebook_role,
                self.baseline_id or "main",
                _utc_time_for_id(),
                self.git_short_commit,
            ]
            self.notebook_run_id = sanitize_filename_token("_".join(parts))
        self.write_manifest("running")

    @property
    def notebook_timing_start_source(self) -> str:
        """说明总耗时计时起点来自共享入口层还是 helper 初始化。"""

        if os.environ.get(NOTEBOOK_STARTED_AT_UTC_ENV) and os.environ.get(NOTEBOOK_STARTED_AT_PERF_COUNTER_ENV):
            return "shared_colab_stage_layout_environment"
        return "repository_helper_initialization"

    @property
    def notebook_timing_coverage_status(self) -> str:
        """返回计时覆盖范围说明。"""

        if self.notebook_timing_start_source == "shared_colab_stage_layout_environment":
            return "shared_colab_stage_layout_to_repository_stage_plan_finish"
        return "repository_stage_plan_only_excludes_manual_colab_setup"

    @property
    def run_root(self) -> Path:
        """返回当前 Notebook stage 写入 records/artifacts 的根目录。"""

        return Path(str(self.layout.get("drive_run_root") or self.layout.get("run_root") or "."))

    @property
    def manifest_path(self) -> Path:
        """返回 Notebook 总耗时 manifest 路径。"""

        return self.run_root / NOTEBOOK_RUN_TIMING_MANIFEST_RELATIVE_PATH

    @property
    def runtime_report_path(self) -> Path:
        """返回每个阶段包必须包含的 Notebook 运行时间报告路径。"""

        return self.run_root / NOTEBOOK_RUNTIME_REPORT_RELATIVE_PATH

    @property
    def stage_records_path(self) -> Path:
        """返回 Notebook 阶段耗时 JSONL 路径。"""

        return self.run_root / NOTEBOOK_STAGE_TIMING_RECORDS_RELATIVE_PATH

    def run_stage(
        self,
        stage_name: str,
        stage_execution_kind: str,
        callback: Callable[[], T],
    ) -> T:
        """执行并记录一个阶段的耗时。

        该函数是通用工程写法。它不理解具体 stage 的业务语义, 只负责在执行前后
        记录时间、状态和失败摘要。业务产物仍由被调用的 repository stage 生成。
        """

        stage_started_at_utc = utc_now_iso()
        stage_start_perf = time.perf_counter()
        try:
            result = callback()
        except BaseException as exc:
            self._append_stage_record(
                stage_name=stage_name,
                stage_execution_kind=stage_execution_kind,
                stage_execution_status="failed",
                stage_started_at_utc=stage_started_at_utc,
                stage_elapsed_sec=time.perf_counter() - stage_start_perf,
                failure_summary=_failure_summary(exc),
            )
            self.finish("failed", failure_summary=_failure_summary(exc))
            raise

        stage_status = "completed"
        if isinstance(result, Mapping):
            stage_status = _normalize_stage_status(
                str(result.get("stage_execution_status") or result.get("stage_status") or stage_status)
            )
        self._append_stage_record(
            stage_name=stage_name,
            stage_execution_kind=stage_execution_kind,
            stage_execution_status=stage_status,
            stage_started_at_utc=stage_started_at_utc,
            stage_elapsed_sec=time.perf_counter() - stage_start_perf,
            failure_summary={},
        )
        return result

    def _append_stage_record(
        self,
        *,
        stage_name: str,
        stage_execution_kind: str,
        stage_execution_status: str,
        stage_started_at_utc: str,
        stage_elapsed_sec: float,
        failure_summary: Mapping[str, str],
    ) -> dict[str, Any]:
        """追加一条阶段耗时记录并刷新 JSONL。"""

        stage_elapsed = _round_seconds(stage_elapsed_sec)
        record = {
            "record_version": NOTEBOOK_TIMING_RECORD_VERSION,
            "record_id": f"{self.notebook_run_id}:stage:{len(self.stage_records) + 1}",
            "stage_id": "notebook_stage_timing",
            "notebook_run_id": self.notebook_run_id,
            "notebook_role": self.notebook_role,
            "workflow_profile": self.workflow_profile,
            "baseline_id": self.baseline_id,
            "stage_name": stage_name,
            "stage_execution_kind": stage_execution_kind,
            "stage_execution_status": stage_execution_status,
            "stage_started_at_utc": stage_started_at_utc,
            "stage_finished_at_utc": utc_now_iso(),
            "stage_elapsed_sec": stage_elapsed,
            "stage_elapsed_min": round(stage_elapsed / 60.0, 3),
            "notebook_timing_scope": self.timing_scope,
            "claim_support_status": NOTEBOOK_TIMING_CLAIM_SUPPORT_STATUS,
            **dict(failure_summary),
        }
        self.stage_records.append(record)
        write_jsonl(self.stage_records_path, self.stage_records)
        self.write_manifest("running")
        return record

    def write_manifest(
        self,
        notebook_timing_status: str,
        *,
        failure_summary: Mapping[str, str] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """写出当前 Notebook 总耗时 manifest。"""

        elapsed = _round_seconds(time.perf_counter() - self.start_perf_counter)
        failed_count = sum(1 for item in self.stage_records if item.get("stage_execution_status") == "failed")
        completed_count = sum(1 for item in self.stage_records if item.get("stage_execution_status") == "completed")
        manifest = {
            "manifest_kind": "notebook_runtime_report",
            "record_version": NOTEBOOK_TIMING_RECORD_VERSION,
            "stage_id": "notebook_run_timing",
            "notebook_run_id": self.notebook_run_id,
            "notebook_role": self.notebook_role,
            "workflow_profile": self.workflow_profile,
            "baseline_id": self.baseline_id,
            "run_root": str(self.run_root),
            "repo_root": str(self.repo_root or ""),
            "git_short_commit": self.git_short_commit,
            "notebook_started_at_utc": self.started_at_utc,
            "notebook_finished_at_utc": utc_now_iso() if notebook_timing_status != "running" else "",
            "notebook_elapsed_sec": elapsed,
            "notebook_elapsed_min": round(elapsed / 60.0, 3),
            "notebook_timing_status": notebook_timing_status,
            "notebook_timing_scope": self.timing_scope,
            "notebook_timing_start_source": self.notebook_timing_start_source,
            "notebook_timing_coverage_status": self.notebook_timing_coverage_status,
            "notebook_stage_timing_record_count": len(self.stage_records),
            "notebook_stage_timing_records_path": str(self.stage_records_path),
            "enabled_stage_plan": list(self.enabled_stage_plan),
            "completed_stage_count": completed_count,
            "failed_stage_count": failed_count,
            "claim_support_status": NOTEBOOK_TIMING_CLAIM_SUPPORT_STATUS,
            **dict(failure_summary or {}),
            **dict(extra or {}),
        }
        write_jsonl(self.stage_records_path, self.stage_records)
        write_json(self.runtime_report_path, manifest)
        write_json(self.manifest_path, manifest)
        self.last_manifest = manifest
        return manifest

    def finish(
        self,
        notebook_timing_status: str = "completed",
        *,
        failure_summary: Mapping[str, str] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """结束本次 Notebook 计时并写出最终 manifest。"""

        if self.finished:
            return self.last_manifest
        self.finished = True
        return self.write_manifest(
            notebook_timing_status,
            failure_summary=failure_summary,
            extra=extra,
        )


def start_notebook_run_timer(
    layout: Mapping[str, str],
    *,
    notebook_role: str,
    workflow_profile: str,
    baseline_id: str | None = None,
    timing_scope: str = "repository_stage_plan",
    enabled_stage_plan: list[str] | None = None,
    repo_root: str | Path | None = None,
) -> NotebookRunTimer:
    """创建并启动 Notebook 总耗时计时器。"""

    return NotebookRunTimer(
        layout=layout,
        notebook_role=notebook_role,
        workflow_profile=workflow_profile,
        baseline_id=baseline_id or "",
        timing_scope=timing_scope,
        enabled_stage_plan=list(enabled_stage_plan or []),
        repo_root=repo_root,
    )


def initialize_notebook_runtime_session(
    layout: Mapping[str, str],
    *,
    notebook_role: str,
    baseline_id: str | None = None,
    timing_scope: str = "shared_colab_stage_layout_to_stage_package_publish",
    repo_root: str | Path | None = None,
) -> dict[str, str]:
    """在共享 Colab layout 入口层初始化 Notebook 运行时间会话。

    该函数是项目特定写法。所有正式 Colab Notebook 都会调用
    `prepare_colab_stage_layout`, 因此把计时起点放在这里可以避免逐个修改
    Notebook cell。该计时只覆盖共享 layout 初始化之后的仓库流程, 不把人工打开
    Notebook 或手动编辑参数的时间计入结果包。
    """

    started_at_utc = utc_now_iso()
    start_perf_counter = str(time.perf_counter())
    os.environ[NOTEBOOK_STARTED_AT_UTC_ENV] = started_at_utc
    os.environ[NOTEBOOK_STARTED_AT_PERF_COUNTER_ENV] = start_perf_counter
    os.environ[NOTEBOOK_TIMING_SCOPE_ENV] = timing_scope
    workflow_profile = str(layout.get("workflow_profile") or layout.get("runtime_profile") or "default")
    updated = dict(layout)
    updated["notebook_runtime_started_at_utc"] = started_at_utc
    updated["notebook_runtime_start_perf_counter"] = start_perf_counter
    updated["notebook_runtime_start_source"] = "shared_colab_stage_layout"
    updated["notebook_runtime_timing_scope"] = timing_scope
    updated["notebook_runtime_workflow_profile"] = workflow_profile
    updated["notebook_runtime_notebook_role"] = notebook_role
    updated["notebook_runtime_baseline_id"] = baseline_id or ""
    updated["notebook_runtime_repo_root"] = str(repo_root or "")
    return updated


def _runtime_report_path_from_layout(layout: Mapping[str, str]) -> Path:
    """返回统一 runtime report 路径。"""

    return Path(str(layout.get("drive_run_root") or layout.get("run_root") or ".")) / NOTEBOOK_RUNTIME_REPORT_RELATIVE_PATH


def _legacy_manifest_path_from_layout(layout: Mapping[str, str]) -> Path:
    """返回兼容旧 timing manifest 路径。"""

    return Path(str(layout.get("drive_run_root") or layout.get("run_root") or ".")) / NOTEBOOK_RUN_TIMING_MANIFEST_RELATIVE_PATH


def _stage_records_path_from_layout(layout: Mapping[str, str]) -> Path:
    """返回阶段耗时 records 路径。"""

    return Path(str(layout.get("drive_run_root") or layout.get("run_root") or ".")) / NOTEBOOK_STAGE_TIMING_RECORDS_RELATIVE_PATH


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    """读取可选 JSON 对象。"""

    if not path.exists():
        return {}
    try:
        payload = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        value = json.loads(payload)
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _read_stage_records(path: Path) -> list[dict[str, Any]]:
    """读取可选阶段耗时 records。"""

    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except Exception:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def finalize_notebook_runtime_report_for_package(
    layout: Mapping[str, str],
    *,
    notebook_role: str,
    baseline_id: str | None = None,
    notebook_timing_status: str = "completed_before_stage_package_publish",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """在统一打包入口生成 `notebook_runtime_report.json`。

    此函数由 `publish_colab_stage_package` 调用, 因此每个阶段包都会自动包含
    Notebook 运行时间报告。若上游 stage plan 已经写出报告, 这里会复用该报告;
    若 helper Notebook 没有显式计时器, 这里会基于共享 layout 入口时间生成最小报告。
    """

    report_path = _runtime_report_path_from_layout(layout)
    existing = _read_json_if_exists(report_path)
    if existing and str(existing.get("notebook_timing_status") or "") != "running":
        return existing

    legacy = _read_json_if_exists(_legacy_manifest_path_from_layout(layout))
    source = existing or legacy
    stage_records_path = _stage_records_path_from_layout(layout)
    stage_records = _read_stage_records(stage_records_path)
    workflow_profile = str(
        source.get("workflow_profile")
        or layout.get("notebook_runtime_workflow_profile")
        or layout.get("workflow_profile")
        or layout.get("runtime_profile")
        or "default"
    )
    start_perf_counter_text = str(
        layout.get("notebook_runtime_start_perf_counter")
        or os.environ.get(NOTEBOOK_STARTED_AT_PERF_COUNTER_ENV)
        or ""
    )
    try:
        start_perf_counter = float(start_perf_counter_text)
    except ValueError:
        start_perf_counter = time.perf_counter()
    elapsed = _round_seconds(time.perf_counter() - start_perf_counter)
    started_at_utc = str(
        source.get("notebook_started_at_utc")
        or layout.get("notebook_runtime_started_at_utc")
        or os.environ.get(NOTEBOOK_STARTED_AT_UTC_ENV)
        or utc_now_iso()
    )
    git_short_commit = str(source.get("git_short_commit") or current_short_commit())
    notebook_run_id = str(source.get("notebook_run_id") or sanitize_filename_token(
        "_".join([
            workflow_profile,
            notebook_role,
            baseline_id or "main",
            _utc_time_for_id(),
            git_short_commit,
        ])
    ))
    failed_count = sum(1 for item in stage_records if item.get("stage_execution_status") == "failed")
    completed_count = sum(1 for item in stage_records if item.get("stage_execution_status") == "completed")
    timing_scope = str(
        source.get("notebook_timing_scope")
        or layout.get("notebook_runtime_timing_scope")
        or os.environ.get(NOTEBOOK_TIMING_SCOPE_ENV)
        or "shared_colab_stage_layout_to_stage_package_publish"
    )
    report = {
        "manifest_kind": "notebook_runtime_report",
        "record_version": NOTEBOOK_TIMING_RECORD_VERSION,
        "stage_id": "notebook_runtime_report",
        "notebook_run_id": notebook_run_id,
        "notebook_role": str(source.get("notebook_role") or notebook_role),
        "workflow_profile": workflow_profile,
        "baseline_id": str(source.get("baseline_id") or baseline_id or ""),
        "run_root": str(layout.get("drive_run_root") or layout.get("run_root") or ""),
        "repo_root": str(source.get("repo_root") or layout.get("notebook_runtime_repo_root") or ""),
        "git_short_commit": git_short_commit,
        "notebook_started_at_utc": started_at_utc,
        "notebook_finished_at_utc": utc_now_iso(),
        "notebook_elapsed_sec": elapsed,
        "notebook_elapsed_min": round(elapsed / 60.0, 3),
        "notebook_timing_status": notebook_timing_status,
        "notebook_timing_scope": timing_scope,
        "notebook_timing_start_source": "shared_colab_stage_layout",
        "notebook_timing_coverage_status": "shared_colab_stage_layout_to_stage_package_publish",
        "notebook_stage_timing_record_count": len(stage_records),
        "notebook_stage_timing_records_path": str(stage_records_path),
        "enabled_stage_plan": list(source.get("enabled_stage_plan") or []),
        "completed_stage_count": completed_count,
        "failed_stage_count": failed_count,
        "claim_support_status": NOTEBOOK_TIMING_CLAIM_SUPPORT_STATUS,
        **dict(extra or {}),
    }
    write_jsonl(stage_records_path, stage_records)
    write_json(report_path, report)
    write_json(_legacy_manifest_path_from_layout(layout), report)
    return report
