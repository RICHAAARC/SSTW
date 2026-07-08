"""在普通 GPU 服务器上运行 SSTW 生成式视频论文 workflow。

该脚本是 Notebook 的命令行等价入口。Notebook 只适用于 Colab 交互场景,
本脚本用于无 Notebook 的 GPU 服务器, 但仍复用同一套 workflow profile、
stage plan、阶段 zip 交接和 governed artifact 生成逻辑。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main.protocol.package_naming import current_short_commit, current_utc_time_for_filename
from paper_workflow.colab_utils.modern_external_baseline_formal_reference import (
    MODERN_EXTERNAL_BASELINE_BUILD_ORDER,
    run_default_modern_external_baseline_formal_reference_plan,
)
from paper_workflow.colab_utils.stage_package_sync import prepare_colab_stage_layout
from paper_workflow.notebook_utils import generative_video_model_probe_workflow as probe_workflow


SERVER_PIPELINES = (
    "motion_threshold_calibration",
    "generative_video_generation",
    "generative_video_quality_scoring",
    "sstw_mechanism_postprocess",
    "runtime_attack",
    "runtime_detection",
    "external_baseline_references",
    "formal_comparison_scoring",
    "paper_evidence_postprocess",
    "paper_gate_and_package",
    "paper_protocol_complete",
    "validation_scale_complete",
)

GENERATIVE_VIDEO_SPLIT_ROLE_ORDER = (
    "generative_video_generation",
    "generative_video_quality_scoring",
    "sstw_mechanism_postprocess",
    "runtime_attack",
    "runtime_detection",
)

PIPELINE_ROLE_ORDER = {
    "motion_threshold_calibration": ("motion_threshold_calibration",),
    "generative_video_generation": ("generative_video_generation",),
    "generative_video_quality_scoring": ("generative_video_quality_scoring",),
    "sstw_mechanism_postprocess": ("sstw_mechanism_postprocess",),
    "runtime_attack": ("runtime_attack",),
    "runtime_detection": ("runtime_detection",),
    "external_baseline_references": ("external_baseline_formal_scoring",),
    "formal_comparison_scoring": ("formal_comparison_scoring",),
    "paper_evidence_postprocess": ("paper_evidence_postprocess",),
    "paper_gate_and_package": ("paper_gate_and_package",),
    "paper_protocol_complete": (
        "motion_threshold_calibration",
        *GENERATIVE_VIDEO_SPLIT_ROLE_ORDER,
        "external_baseline_formal_scoring",
        "formal_comparison_scoring",
        "paper_evidence_postprocess",
        "paper_gate_and_package",
    ),
    "validation_scale_complete": (
        "motion_threshold_calibration",
        *GENERATIVE_VIDEO_SPLIT_ROLE_ORDER,
        "external_baseline_formal_scoring",
        "formal_comparison_scoring",
        "paper_evidence_postprocess",
        "paper_gate_and_package",
    ),
}


def _bool_text(value: bool) -> str:
    """把布尔值转换为环境变量友好的文本。"""

    return "true" if value else "false"


def _set_default_env(name: str, value: str) -> None:
    """只在用户未显式设置时写入默认环境变量。"""

    os.environ.setdefault(name, value)


def _apply_server_environment(args: argparse.Namespace) -> None:
    """为服务器命令行运行设置与 Colab stage zip 交接兼容的环境变量。

    该函数不改变实验协议本身, 只把原先由 Notebook cell 设置的入口型环境变量
    转移到命令行脚本中。正式阶段顺序仍由统一 workflow config 控制。
    """

    project_root = Path(args.project_root).expanduser().resolve()
    local_workspace_root = Path(args.local_workspace_root).expanduser().resolve() if args.local_workspace_root else project_root / "_local_stage_workspace"
    local_package_cache_root = (
        Path(args.local_package_cache_root).expanduser().resolve()
        if args.local_package_cache_root
        else project_root / "_local_stage_packages"
    )
    args.project_root = str(project_root)
    args.repo_root = str(Path(args.repo_root).expanduser().resolve())
    args.local_workspace_root = str(local_workspace_root)
    args.local_package_cache_root = str(local_package_cache_root)

    os.environ["SSTW_DRIVE_PROJECT_ROOT"] = str(project_root)
    os.environ["SSTW_WORKFLOW_PROFILE"] = args.workflow_profile
    os.environ["SSTW_REPO_DIR"] = args.repo_root
    os.environ["SSTW_COLAB_STAGE_IO_MODE"] = "local_zip"
    os.environ["SSTW_LOCAL_STAGE_WORKSPACE_ROOT"] = str(local_workspace_root)
    os.environ["SSTW_LOCAL_STAGE_PACKAGE_CACHE_ROOT"] = str(local_package_cache_root)
    os.environ["SSTW_INCLUDE_VIDEOS_IN_PACKAGE"] = _bool_text(args.include_videos)

    if args.model_id:
        os.environ["SSTW_MODEL_ID"] = args.model_id
    if args.cross_model_id:
        os.environ["SSTW_CROSS_MODEL_ID"] = args.cross_model_id
    if args.semantic_model_id:
        os.environ["SSTW_SEMANTIC_MODEL_ID"] = args.semantic_model_id
    if args.semantic_frame_limit is not None:
        os.environ["SSTW_SEMANTIC_FRAME_LIMIT"] = str(args.semantic_frame_limit)
    if args.disable_semantic_metric:
        os.environ["SSTW_DISABLE_SEMANTIC_METRIC"] = "true"
    if args.external_baseline_reference_max_records is not None:
        os.environ["SSTW_EXTERNAL_BASELINE_REFERENCE_MAX_RECORDS"] = str(args.external_baseline_reference_max_records)

    _set_default_env("SSTW_RUN_EXTERNAL_BASELINE_SOURCE_CLONE", "true")
    _set_default_env("SSTW_REQUIRE_MODERN_BASELINE_COMMANDS_FOR_PAPER_GATE", "true")
    _set_default_env("SSTW_USE_MODERN_BASELINE_BRIDGE_COMMANDS", "true")
    _set_default_env("SSTW_REQUIRE_MODERN_BASELINE_BRIDGE_OFFICIAL_COMMANDS", "true")
    _set_default_env("SSTW_USE_REPOSITORY_OFFICIAL_BASELINE_ADAPTERS", "true")


def _assert_safe_reset_path(target: Path, project_root: Path) -> None:
    """校验待清理目录必须位于本次 project_root 内。"""

    resolved_target = target.expanduser().resolve()
    resolved_root = project_root.expanduser().resolve()
    try:
        resolved_target.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"拒绝清理 project_root 外部目录: {resolved_target}") from exc


def _reset_local_workspace_if_requested(args: argparse.Namespace) -> None:
    """按显式开关清理本地热路径缓存, 避免旧工作区污染新运行。"""

    if not args.reset_local_workspace:
        return
    project_root = Path(args.project_root).expanduser().resolve()
    candidates = [
        Path(args.local_workspace_root).expanduser().resolve() if args.local_workspace_root else project_root / "_local_stage_workspace",
        Path(args.local_package_cache_root).expanduser().resolve() if args.local_package_cache_root else project_root / "_local_stage_packages",
    ]
    for target in candidates:
        _assert_safe_reset_path(target, project_root)
        if target.exists():
            shutil.rmtree(target)


def _runtime_options(args: argparse.Namespace) -> dict[str, Any]:
    """构造传给 repository workflow helper 的入口参数。"""

    return {
        "model_id": args.model_id,
        "cross_model_id": args.cross_model_id,
        "semantic_model_id": args.semantic_model_id,
        "semantic_frame_limit": args.semantic_frame_limit,
        "disable_semantic_metric": args.disable_semantic_metric,
    }


def _workflow_profile_for_role(args: argparse.Namespace, notebook_role: str) -> str:
    """返回某个 role 实际使用的 workflow profile。

    motion threshold calibration 使用独立 calibration split, 不应误用
    validation_scale / pilot_paper / full_paper 的 evaluation profile。
    """

    if notebook_role == "motion_threshold_calibration":
        return "motion_calibration"
    return args.workflow_profile


def _build_layout_for_role(args: argparse.Namespace, notebook_role: str, *, hydrate: bool) -> dict[str, str]:
    """构造某个 role 的 layout, 真实运行时恢复前置阶段包。"""

    role_profile = _workflow_profile_for_role(args, notebook_role)
    layout = probe_workflow.ensure_drive_layout(
        args.project_root,
        workflow_profile=role_profile,
        notebook_role=notebook_role,
    )
    if not hydrate:
        return dict(layout)
    return prepare_colab_stage_layout(layout, notebook_role=notebook_role)


def _dry_run_role_plan(args: argparse.Namespace, notebook_role: str) -> dict[str, Any]:
    """生成单个 role 的命令行 dry-run 计划, 不读取前置 zip, 不执行 GPU 任务。"""

    layout = _build_layout_for_role(args, notebook_role, hydrate=False)
    role_profile = _workflow_profile_for_role(args, notebook_role)
    stage_plan = probe_workflow.build_workflow_stage_plan(role_profile, notebook_role)
    stage_rows: list[dict[str, Any]] = []
    for stage_name in stage_plan:
        if stage_name in {
            "external_baseline_colab_preflight",
            "motion_threshold_reuse_check",
            "motion_threshold_calibration_or_reuse_check",
            "quick_tests_and_harness",
        }:
            stage_rows.append({
                "stage_name": stage_name,
                "stage_execution_kind": "python_helper",
            })
            continue
        command = probe_workflow.build_configured_colab_stage_command(
            stage_name,
            layout,
            workflow_profile=role_profile,
            runtime_options={**_runtime_options(args), "include_videos": args.include_videos},
        )
        stage_rows.append({
            "stage_name": stage_name,
            "stage_execution_kind": "command",
            "command": command,
        })
    return {
        "notebook_role": notebook_role,
        "workflow_profile": role_profile,
        "requested_result_workflow_profile": args.workflow_profile,
        "layout": layout,
        "stage_plan": stage_rows,
    }


def _run_role(args: argparse.Namespace, notebook_role: str) -> dict[str, Any]:
    """运行单个非 baseline-reference role。"""

    layout = _build_layout_for_role(args, notebook_role, hydrate=True)
    role_profile = _workflow_profile_for_role(args, notebook_role)
    if notebook_role == "formal_comparison_scoring":
        environment = probe_workflow.apply_formal_comparison_external_baseline_environment(
            layout,
            profile=role_profile,
            repo_root=Path(args.repo_root).expanduser().resolve(),
        )
        print(json.dumps(environment, ensure_ascii=False, indent=2))
    return probe_workflow.run_configured_colab_stage_plan(
        layout,
        workflow_profile=role_profile,
        notebook_role=notebook_role,
        runtime_options=_runtime_options(args),
        include_videos=args.include_videos,
    )


def _baseline_ids_from_args(args: argparse.Namespace) -> list[str]:
    """返回本次需要运行的 external baseline 列表。"""

    return list(args.baseline_id or MODERN_EXTERNAL_BASELINE_BUILD_ORDER)


def _run_external_baseline_references(args: argparse.Namespace) -> dict[str, Any]:
    """按配置运行 5 个 modern external baseline 官方参考结果闭环。"""

    rows: list[dict[str, Any]] = []
    for baseline_id in _baseline_ids_from_args(args):
        print(f"\n===== external baseline formal reference: {baseline_id} =====")
        result = run_default_modern_external_baseline_formal_reference_plan(
            baseline_id,
            repo_root=Path(args.repo_root).expanduser().resolve(),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        rows.append({
            "baseline_id": baseline_id,
            "formal_reference_decision": result.get("formal_reference_decision"),
            "formal_reference_status": result.get("formal_reference_status"),
            "stage_package_publish_result": result.get("stage_package_publish_result", {}),
        })
        if result.get("formal_reference_decision") != "PASS" and not args.continue_on_baseline_failure:
            raise RuntimeError(f"{baseline_id} formal reference 未通过: {result.get('formal_reference_status')}")
    return {
        "notebook_role": "external_baseline_formal_scoring",
        "workflow_profile": args.workflow_profile,
        "baseline_results": rows,
        "external_baseline_reference_decision": "PASS"
        if all(row.get("formal_reference_decision") == "PASS" for row in rows)
        else "FAIL",
    }


def _run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    """按 pipeline 顺序运行服务器 workflow。"""

    role_order = PIPELINE_ROLE_ORDER[args.pipeline]
    rows: list[dict[str, Any]] = []
    for role in role_order:
        if args.dry_run:
            if role == "external_baseline_formal_scoring":
                rows.append({
                    "notebook_role": role,
                    "workflow_profile": args.workflow_profile,
                    "baseline_ids": _baseline_ids_from_args(args),
                    "stage_execution_kind": "external_baseline_formal_reference_helper",
                })
            else:
                rows.append(_dry_run_role_plan(args, role))
            continue
        if role == "external_baseline_formal_scoring":
            rows.append(_run_external_baseline_references(args))
        else:
            rows.append(_run_role(args, role))
    return {
        "manifest_kind": "generative_video_server_workflow_decision",
        "workflow_profile": args.workflow_profile,
        "pipeline": args.pipeline,
        "project_root": str(Path(args.project_root).expanduser().resolve()),
        "repo_root": str(Path(args.repo_root).expanduser().resolve()),
        "dry_run": bool(args.dry_run),
        "include_videos": bool(args.include_videos),
        "created_at_utc_filename": current_utc_time_for_filename(),
        "git_short_commit": current_short_commit(),
        "pipeline_results": rows,
        "server_workflow_decision": "DRY_RUN" if args.dry_run else "PASS",
        "claim_support_status": "server_workflow_runner_not_claim_evidence",
    }


def _write_decision_if_requested(args: argparse.Namespace, decision: dict[str, Any]) -> Path | None:
    """按需写出服务器 workflow 运行清单。"""

    if not args.decision_output:
        return None
    output_path = Path(args.decision_output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""

    parser = argparse.ArgumentParser(description="在无 Notebook 的 GPU 服务器上运行 SSTW 生成式视频 workflow")
    parser.add_argument("--project-root", required=True, help="服务器上的 SSTW 结果根目录, 等价于 Colab 的 Drive 项目根")
    parser.add_argument("--workflow-profile", default="validation_scale", choices=["validation_scale", "pilot_paper", "full_paper"], help="运行层级")
    parser.add_argument("--pipeline", default="validation_scale_complete", choices=SERVER_PIPELINES, help="要执行的语义 pipeline")
    parser.add_argument("--repo-root", default=".", help="SSTW 仓库根目录")
    parser.add_argument("--baseline-id", action="append", choices=MODERN_EXTERNAL_BASELINE_BUILD_ORDER, help="只运行指定 external baseline, 可重复传入")
    parser.add_argument("--model-id", default=os.environ.get("SSTW_MODEL_ID", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"), help="主生成模型 ID")
    parser.add_argument("--cross-model-id", default=os.environ.get("SSTW_CROSS_MODEL_ID", ""), help="可选 cross model ID")
    parser.add_argument("--semantic-model-id", default=os.environ.get("SSTW_SEMANTIC_MODEL_ID", "openai/clip-vit-base-patch32"), help="语义指标模型 ID")
    parser.add_argument("--semantic-frame-limit", type=int, default=int(os.environ.get("SSTW_SEMANTIC_FRAME_LIMIT", "8")), help="语义指标最多抽帧数")
    parser.add_argument("--disable-semantic-metric", action="store_true", help="禁用语义指标")
    parser.add_argument("--exclude-videos", dest="include_videos", action="store_false", help="阶段 zip 中不包含 mp4 等视频大文件")
    parser.set_defaults(include_videos=True)
    parser.add_argument("--external-baseline-reference-max-records", type=int, default=None, help="限制 external baseline official reference 记录数")
    parser.add_argument("--local-workspace-root", default="", help="本地热路径 workspace, 默认位于 project-root/_local_stage_workspace")
    parser.add_argument("--local-package-cache-root", default="", help="本地阶段包 cache, 默认位于 project-root/_local_stage_packages")
    parser.add_argument("--reset-local-workspace", action="store_true", help="运行前清理 project-root 内的本地 workspace/cache")
    parser.add_argument("--continue-on-baseline-failure", action="store_true", help="单个 baseline 失败时继续运行其它 baseline")
    parser.add_argument("--dry-run", action="store_true", help="只打印 stage plan 和命令, 不执行 GPU 或打包任务")
    parser.add_argument("--decision-output", default="", help="可选运行清单 JSON 输出路径")
    return parser


def main(argv: list[str] | None = None) -> int:
    """命令行入口。"""

    parser = build_parser()
    args = parser.parse_args(argv)
    _apply_server_environment(args)
    _reset_local_workspace_if_requested(args)
    try:
        decision = _run_pipeline(args)
    except Exception as exc:
        failure = {
            "manifest_kind": "generative_video_server_workflow_decision",
            "workflow_profile": args.workflow_profile,
            "pipeline": args.pipeline,
            "project_root": str(Path(args.project_root).expanduser().resolve()),
            "repo_root": str(Path(args.repo_root).expanduser().resolve()),
            "server_workflow_decision": "FAIL",
            "failure_reason": str(exc),
            "claim_support_status": "server_workflow_runner_blocked_not_claim_evidence",
        }
        _write_decision_if_requested(args, failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
    _write_decision_if_requested(args, decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
