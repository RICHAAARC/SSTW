"""B5 生成式视频 Colab Notebook 的路径和命令编排工具。"""

from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
from typing import Any, Mapping

from paper_workflow.colab_utils.stage_package_sync import (
    publish_colab_stage_package,
    stage_package_dir,
    stage_package_id_for_notebook,
    stage_zip_handoff_enabled,
)
from paper_workflow.notebook_utils.streaming_command import run_streaming_command


DEFAULT_DRIVE_PROJECT_ROOT = "/content/drive/MyDrive/SSTW"
DEFAULT_VALIDATION_SCALE_CONFIG = "configs/protocol/validation_scale_generative_probe.json"
DEFAULT_PILOT_PAPER_CONFIG = "configs/protocol/pilot_paper_generative_probe.json"
DEFAULT_FULL_PAPER_CONFIG = "configs/protocol/full_paper_generative_probe.json"
DEFAULT_NOTEBOOK_WORKFLOW_CONFIG = "configs/paper_workflow/generative_video_notebook_workflows.json"
DEFAULT_MODERN_BASELINE_COLAB_COMMAND_CONFIG = "configs/external_baselines/modern_baseline_colab_commands.json"
DEFAULT_NOTEBOOK_ROLE = "generative_video_runtime"
PAPER_GATE_PROFILES = {"validation_scale", "pilot_paper"}
EXTERNAL_BASELINE_COLAB_PREFLIGHT_DECISION = "artifacts/external_baseline_colab_preflight_decision.json"
EXTERNAL_BASELINE_COMMAND_TEMPLATE_SUMMARY = "artifacts/external_baseline_command_template_summary.json"
EXTERNAL_BASELINE_OFFICIAL_BRIDGE_PREFLIGHT_DECISION = "artifacts/external_baseline_official_bridge_preflight_decision.json"
EXTERNAL_BASELINE_OFFICIAL_RESOURCE_BOOTSTRAP_DECISION = "artifacts/external_baseline_official_resource_bootstrap_decision.json"
EXTERNAL_BASELINE_OFFICIAL_BUNDLE_GENERATION_DECISION = "artifacts/external_baseline_official_bundle_generation_decision.json"
EXTERNAL_BASELINE_OFFICIAL_RUNTIME_CLOSURE_REQUIREMENTS = "artifacts/external_baseline_official_runtime_closure_requirements.json"
EXTERNAL_BASELINE_OFFICIAL_RESULT_BUNDLE_PREFLIGHT_DECISION = "artifacts/external_baseline_official_result_bundle_preflight_decision.json"
PAPER_GATE_EXTERNAL_BASELINE_ENVIRONMENT_DECISION = "artifacts/paper_gate_external_baseline_environment_decision.json"


def _join_drive_path(root: PurePosixPath, relative_path: str) -> str:
    """把配置中的相对路径拼接到 Drive 根目录下。

    该函数属于通用工程写法, 用于避免 Notebook cell 中硬写多个 Google Drive
    输出目录。项目特定约束是 profile 之间必须隔离 run / package / log 目录,
    防止 `validation_scale`、`pilot_paper` 和未来 `full_paper` 产物混写。
    """
    return (root / PurePosixPath(relative_path)).as_posix()


def load_notebook_workflow_config(
    config_path: str | Path = DEFAULT_NOTEBOOK_WORKFLOW_CONFIG,
) -> dict[str, Any]:
    """读取 Colab Notebook 统一 workflow profile 配置。

    Notebook 只负责入口编排。不同结果层级、Drive 路径、样本规模和默认 profile
    均由该配置控制, 避免把 `pilot_paper` 或 `validation_scale` 写死在 Notebook
    的多处 cell 中。
    """
    config = _read_json(config_path)
    if not config:
        raise FileNotFoundError(f"缺少 Notebook workflow 配置: {config_path}")
    return config


def canonical_workflow_profile(
    profile: str,
    config_path: str | Path = DEFAULT_NOTEBOOK_WORKFLOW_CONFIG,
) -> str:
    """返回 profile 的规范名称。

    当前正式 Colab workflow 不再保留历史 profile alias。该函数仍读取配置中的
    alias 映射, 主要用于 harness 在发现误配置时给出明确错误。
    """
    config = load_notebook_workflow_config(config_path)
    aliases = config.get("workflow_profile_aliases", {})
    canonical = str(aliases.get(profile, profile))
    profiles = config.get("workflow_profiles", {})
    if canonical not in profiles:
        raise KeyError(f"未知 workflow profile: {profile}")
    return canonical


def default_workflow_profile_for_notebook_role(
    notebook_role: str,
    config_path: str | Path = DEFAULT_NOTEBOOK_WORKFLOW_CONFIG,
) -> str:
    """从统一配置中读取某类 Notebook 的默认 profile。"""
    config = load_notebook_workflow_config(config_path)
    defaults = config.get("default_workflow_profile_by_notebook_role", {})
    if notebook_role not in defaults:
        raise KeyError(f"Notebook role 缺少默认 workflow profile: {notebook_role}")
    return str(defaults[notebook_role])


def resolve_notebook_workflow_profile(
    profile: str,
    notebook_role: str | None = None,
    config_path: str | Path = DEFAULT_NOTEBOOK_WORKFLOW_CONFIG,
    *,
    allow_disabled: bool = False,
) -> dict[str, Any]:
    """解析 Notebook workflow profile 与 role 的组合。

    返回值会显式包含 requested / canonical profile、runtime profile、result tier、
    Drive 相对路径、protocol config path 和 stage plan。Notebook 应只消费该返回值,
    不再根据 profile 名称手写分支。
    """
    config = load_notebook_workflow_config(config_path)
    aliases = config.get("workflow_profile_aliases", {})
    canonical_profile = str(aliases.get(profile, profile))
    profiles = config.get("workflow_profiles", {})
    if canonical_profile not in profiles:
        raise KeyError(f"未知 workflow profile: {profile}")
    profile_config = _merge_protocol_target_fpr(canonical_profile, dict(profiles[canonical_profile]))
    role_config: dict[str, Any] = {}
    if notebook_role:
        roles = config.get("notebook_roles", {})
        if notebook_role not in roles:
            raise KeyError(f"未知 Notebook role: {notebook_role}")
        role_config = dict(roles[notebook_role])
        allowed_profiles = {str(item) for item in role_config.get("allowed_workflow_profiles", [])}
        if allowed_profiles and canonical_profile not in allowed_profiles:
            raise ValueError(
                f"Notebook role {notebook_role} 不允许 workflow profile {profile}。"
                f" 允许值: {sorted(allowed_profiles)}"
            )
    if not allow_disabled and profile_config.get("enabled_for_run") is False:
        raise RuntimeError(
            f"workflow profile {canonical_profile} 当前不可运行: "
            f"{profile_config.get('profile_status')}"
        )
    return {
        **profile_config,
        "requested_workflow_profile": profile,
        "canonical_workflow_profile": canonical_profile,
        "workflow_profile": canonical_profile,
        "profile_alias_applied": canonical_profile != profile,
        "notebook_role": notebook_role or "",
        "notebook_path": role_config.get("notebook_path", ""),
        "notebook_path_examples": list(role_config.get("notebook_path_examples", [])),
        "entrypoint_status": str(role_config.get("entrypoint_status", "")),
        "workflow_stage_plan": list(role_config.get("stage_plan", [])),
        "allowed_workflow_profiles": list(role_config.get("allowed_workflow_profiles", [])),
        "config_path": str(config_path),
    }


def build_workflow_stage_plan(
    profile: str,
    notebook_role: str,
    config_path: str | Path = DEFAULT_NOTEBOOK_WORKFLOW_CONFIG,
) -> list[str]:
    """返回指定 profile / Notebook role 的阶段计划。"""
    resolved = resolve_notebook_workflow_profile(profile, notebook_role, config_path)
    disabled_stage_names = {str(item) for item in resolved.get("disabled_stage_names", [])}
    return [
        str(item)
        for item in resolved.get("workflow_stage_plan", [])
        if str(item) not in disabled_stage_names
    ]


def workflow_stage_enabled(
    profile: str,
    notebook_role: str,
    stage_name: str,
    config_path: str | Path = DEFAULT_NOTEBOOK_WORKFLOW_CONFIG,
) -> bool:
    """判断某个语义阶段是否属于当前 Notebook 计划。"""
    return stage_name in build_workflow_stage_plan(profile, notebook_role, config_path)


def workflow_profile_is_paper_gate(
    profile: str,
    config_path: str | Path = DEFAULT_NOTEBOOK_WORKFLOW_CONFIG,
) -> bool:
    """从统一配置判断 profile 是否属于 paper gate 相关运行。"""
    resolved = resolve_notebook_workflow_profile(profile, config_path=config_path, allow_disabled=True)
    return bool(resolved.get("paper_gate_profile"))


def protocol_config_path_for_profile(
    profile: str,
    config_path: str | Path = DEFAULT_NOTEBOOK_WORKFLOW_CONFIG,
) -> str:
    """从统一 workflow 配置中读取 profile 对应的 protocol config。"""
    resolved = resolve_notebook_workflow_profile(profile, config_path=config_path, allow_disabled=True)
    return str(resolved.get("protocol_config_path") or _config_path_for_profile(profile))


def build_drive_layout(
    drive_project_root: str = DEFAULT_DRIVE_PROJECT_ROOT,
    workflow_profile: str | None = None,
    notebook_role: str | None = None,
    workflow_config_path: str | Path = DEFAULT_NOTEBOOK_WORKFLOW_CONFIG,
) -> dict[str, str]:
    """构造 Colab 与 Google Drive 共享的 SSTW 输出目录布局。

    run / package / log 目录由统一配置决定, 从而支持在 Colab 中安全切换
    `validation_scale`、`pilot_paper` 和未来 `full_paper`。不传 `workflow_profile`
    时使用 `generative_video_runtime` 的默认 profile, 不再回退到旧综合 Notebook
    的历史目录。
    """
    root = PurePosixPath(drive_project_root)
    if workflow_profile is None:
        notebook_role = notebook_role or DEFAULT_NOTEBOOK_ROLE
        workflow_profile = default_workflow_profile_for_notebook_role(notebook_role, workflow_config_path)
    workflow = resolve_notebook_workflow_profile(workflow_profile, notebook_role, workflow_config_path)
    config = load_notebook_workflow_config(workflow_config_path)
    dataset_root_relative = str(
        workflow.get("drive_dataset_root_relative")
        or config.get("default_dataset_root_relative")
        or "datasets/generative_video_prompt_suite"
    )
    prompt_suite_path_relative = str(
        workflow.get("prompt_suite_path_relative")
        or config.get("default_prompt_suite_path_relative")
        or "datasets/generative_video_prompt_suite/prompt_seed_suite.json"
    )
    resolved_notebook_role = str(workflow.get("notebook_role") or notebook_role or DEFAULT_NOTEBOOK_ROLE)
    resolved_stage_package_id = stage_package_id_for_notebook(resolved_notebook_role)
    return {
        "drive_project_root": root.as_posix(),
        "drive_dataset_root": _join_drive_path(root, dataset_root_relative),
        "drive_run_root": _join_drive_path(root, str(workflow["drive_run_root_relative"])),
        "drive_package_dir": str(
            stage_package_dir(root.as_posix(), str(workflow["workflow_profile"]), resolved_stage_package_id)
        ).replace("\\", "/"),
        "drive_log_dir": _join_drive_path(root, str(workflow["drive_log_dir_relative"])),
        "external_baseline_resource_root": _join_drive_path(root, "resources/external_baseline"),
        "external_baseline_official_result_bundle_root": _join_drive_path(
            root,
            f"external_baseline_official_result_bundles/{workflow['workflow_profile']}",
        ),
        "motion_threshold_artifact_run_root": _join_drive_path(
            root,
            str(workflow.get("motion_threshold_artifact_run_root_relative") or workflow["drive_run_root_relative"]),
        ),
        "prompt_suite_path": _join_drive_path(root, prompt_suite_path_relative),
        "workflow_profile": str(workflow["workflow_profile"]),
        "canonical_workflow_profile": str(workflow["canonical_workflow_profile"]),
        "requested_workflow_profile": str(workflow["requested_workflow_profile"]),
        "runtime_profile": str(workflow["runtime_profile"]),
        "result_tier": str(workflow["result_tier"]),
        "notebook_role": resolved_notebook_role,
        "protocol_config_path": str(workflow.get("protocol_config_path") or ""),
    }


def ensure_drive_layout(
    drive_project_root: str = DEFAULT_DRIVE_PROJECT_ROOT,
    workflow_profile: str | None = None,
    notebook_role: str | None = None,
    workflow_config_path: str | Path = DEFAULT_NOTEBOOK_WORKFLOW_CONFIG,
) -> dict[str, str]:
    """创建 Google Drive 目标目录并返回路径布局。"""
    layout = build_drive_layout(
        drive_project_root,
        workflow_profile=workflow_profile,
        notebook_role=notebook_role,
        workflow_config_path=workflow_config_path,
    )
    if stage_zip_handoff_enabled():
        # local_zip 模式下, run / log / dataset 都会被本地化到 /content。
        # 此处只确保 Drive 项目根存在, 最终阶段目录由 publish_colab_stage_package 在发布时创建。
        Path(layout["drive_project_root"]).mkdir(parents=True, exist_ok=True)
        return layout
    for key, value in layout.items():
        if key.endswith("_dir") or key.endswith("_root"):
            Path(value).mkdir(parents=True, exist_ok=True)
    return layout


def _read_json(path: str | Path) -> dict:
    """读取 Notebook helper 需要的轻量 JSON 配置, 并兼容 UTF-8 BOM。"""
    input_path = Path(path)
    if not input_path.exists():
        return {}
    payload = json.loads(input_path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON 顶层必须是对象: {input_path}")
    return payload


def _merge_protocol_target_fpr(profile: str, profile_config: dict[str, Any]) -> dict[str, Any]:
    """把 paper gate profile 的 target_fpr 绑定到 protocol config。

    该函数属于项目特定治理写法。它保留 workflow profile 对 Drive 路径、样本规模和
    阶段计划的管理职责, 但把 `validation_scale`、`pilot_paper` 和 `full_paper`
    的 fixed-FPR 口径统一交给各自 protocol config。若 workflow config 与 protocol
    config 同时声明了不同的 `target_fpr`, 直接失败, 防止后续 Notebook 切换 profile
    时出现不同 fixed-FPR 语义漂移。
    """
    should_bind_protocol_fpr = bool(profile_config.get("paper_gate_profile")) or profile in {
        "validation_scale",
        "pilot_paper",
        "full_paper",
    }
    protocol_config_path = str(profile_config.get("protocol_config_path") or "")
    if not should_bind_protocol_fpr or not protocol_config_path:
        return profile_config
    protocol_config = _read_json(protocol_config_path)
    if "target_fpr" not in protocol_config:
        return profile_config
    protocol_target_fpr = float(protocol_config["target_fpr"])
    workflow_target_fpr = profile_config.get("target_fpr")
    if workflow_target_fpr is not None and abs(float(workflow_target_fpr) - protocol_target_fpr) > 1e-12:
        raise ValueError(
            f"workflow profile {profile} 的 target_fpr={workflow_target_fpr} 与 "
            f"protocol config {protocol_config_path} 的 target_fpr={protocol_target_fpr} 不一致。"
        )
    merged = dict(profile_config)
    merged["target_fpr"] = protocol_target_fpr
    merged["protocol_target_fpr"] = protocol_target_fpr
    merged["target_fpr_source_config_path"] = protocol_config_path
    return merged


def _write_json(path: str | Path, payload: Mapping[str, object]) -> None:
    """写出 Colab preflight artifact, 使冷启动失败也能在 Google Drive 中审计原因。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _config_path_for_profile(profile: str) -> str:
    """根据运行 profile 选择现代 baseline 要求来源。"""
    if profile == "pilot_paper":
        return DEFAULT_PILOT_PAPER_CONFIG
    if profile == "full_paper":
        return DEFAULT_FULL_PAPER_CONFIG
    return DEFAULT_VALIDATION_SCALE_CONFIG


def external_baseline_command_env_var_for(baseline_id: str) -> str:
    """由 baseline_id 推导 Colab 中对应的官方命令环境变量名。

    该函数属于通用工程写法。项目特定约定是所有现代视频水印 baseline
    都通过 `SSTW_<BASELINE_ID>_EVAL_COMMAND` 注入官方 detector / scorer 命令。
    """
    return f"SSTW_{baseline_id.upper()}_EVAL_COMMAND"


def external_baseline_official_command_env_var_for(baseline_id: str) -> str:
    """由 baseline_id 推导 bridge 内部官方原生命令环境变量名。

    外层 `SSTW_<BASELINE>_EVAL_COMMAND` 负责满足 SSTW 统一 I/O 契约。
    内层 `SSTW_<BASELINE>_OFFICIAL_EVAL_COMMAND` 负责真正调用第三方官方实现。
    两层分离可以避免把 wrapper 壳层误判为已经完成正式 baseline 运行。
    """
    return f"SSTW_{baseline_id.upper()}_OFFICIAL_EVAL_COMMAND"


def required_modern_external_baseline_command_requirements(
    profile: str,
    config_path: str | Path | None = None,
) -> list[dict[str, str]]:
    """从 protocol config 中读取现代 baseline command 要求。

    Notebook 不应手写 baseline 清单。该函数把 `validation_scale` 和
    `pilot_paper` 的 hard gate 要求统一收敛到 helper, 防止配置已更新而
    Notebook cell 仍保留旧 baseline 列表。
    """
    config = _read_json(config_path or protocol_config_path_for_profile(profile))
    baseline_ids = [
        str(name)
        for name in config.get("required_modern_external_baseline_adapter_names", [])
        if str(name)
    ]
    return [
        {
            "baseline_id": baseline_id,
            "external_baseline_command_env_var": external_baseline_command_env_var_for(baseline_id),
            "official_baseline_command_env_var": external_baseline_official_command_env_var_for(baseline_id),
        }
        for baseline_id in baseline_ids
    ]


def load_modern_baseline_colab_command_config(
    config_path: str | Path = DEFAULT_MODERN_BASELINE_COLAB_COMMAND_CONFIG,
) -> dict[str, Any]:
    """读取现代 baseline Colab command 配置辅助文件。

    该配置只保存联网核验后的源码位置、Colab clone 目标和用户 wrapper command 建议。
    它不等价于正式 baseline 已配置, 也不会自动让 preflight 通过。正式 claim 仍必须由
    `SSTW_<BASELINE>_EVAL_COMMAND` 指向真实可执行命令, 并产生官方输出 JSON。
    """
    config = _read_json(config_path)
    if not config:
        raise FileNotFoundError(f"缺少现代 baseline Colab command 配置: {config_path}")
    return config


def _modern_baseline_command_config_rows(
    config_path: str | Path = DEFAULT_MODERN_BASELINE_COLAB_COMMAND_CONFIG,
) -> dict[str, dict[str, Any]]:
    """按 baseline_id 索引 Colab command 配置行。"""
    config = load_modern_baseline_colab_command_config(config_path)
    rows = config.get("baseline_command_configs", [])
    if not isinstance(rows, list):
        raise TypeError("baseline_command_configs 必须是列表")
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        baseline_id = str(row.get("baseline_id") or "")
        if baseline_id:
            indexed[baseline_id] = dict(row)
    return indexed


def build_modern_baseline_colab_command_config_summary(
    layout: Mapping[str, str],
    *,
    profile: str,
    protocol_config_path: str | Path | None = None,
    command_config_path: str | Path = DEFAULT_MODERN_BASELINE_COLAB_COMMAND_CONFIG,
    command_templates_auto_applied: bool = False,
) -> dict[str, object]:
    """构造现代 baseline command 配置摘要。

    通用工程写法是把“可查看的配置建议”和“已注入的正式命令”分开。项目特定要求是:
    validation-scale 与 pilot-paper 必须 fail closed, 因此该摘要只能帮助用户在 Colab
    中填写 command, 不能替代 `SSTW_<BASELINE>_EVAL_COMMAND`。
    """
    config = load_modern_baseline_colab_command_config(command_config_path)
    config_rows = _modern_baseline_command_config_rows(command_config_path)
    requirements = required_modern_external_baseline_command_requirements(profile, protocol_config_path)
    required_baseline_ids = [item["baseline_id"] for item in requirements]
    rows: list[dict[str, Any]] = []
    for requirement in requirements:
        baseline_id = requirement["baseline_id"]
        env_var = requirement["external_baseline_command_env_var"]
        row = dict(config_rows.get(baseline_id, {}))
        rows.append({
            "baseline_id": baseline_id,
            "external_baseline_command_env_var": env_var,
            "command_config_row_status": "configured_in_template_file" if row else "missing_from_template_file",
            "official_repository_url": row.get("official_repository_url"),
            "official_repository_branch": row.get("official_repository_branch"),
            "verified_head_commit": row.get("verified_head_commit"),
            "source_verification_status": row.get("source_verification_status"),
            "colab_source_dir": row.get("colab_source_dir"),
            "official_baseline_command_env_var": row.get("official_baseline_command_env_var") or external_baseline_official_command_env_var_for(baseline_id),
            "source_clone_command": row.get("source_clone_command"),
            "official_entrypoint_candidates": row.get("official_entrypoint_candidates", []),
            "repository_official_eval_adapter_module": row.get("repository_official_eval_adapter_module"),
            "repository_official_eval_command_template_status": row.get("repository_official_eval_command_template_status"),
            "repository_official_eval_command_template": row.get("repository_official_eval_command_template"),
            "sstw_eval_command_template_status": row.get("sstw_eval_command_template_status"),
            "sstw_eval_command_template": row.get("sstw_eval_command_template"),
            "score_output_contract": row.get("score_output_contract"),
        })
    missing_template_ids = [
        row["baseline_id"]
        for row in rows
        if row["command_config_row_status"] == "missing_from_template_file"
    ]
    summary_path = Path(layout["drive_run_root"]) / EXTERNAL_BASELINE_COMMAND_TEMPLATE_SUMMARY
    return {
        "artifact_name": "external_baseline_command_template_summary.json",
        "manifest_kind": "external_baseline_command_template_summary",
        "profile": profile,
        "run_root": layout["drive_run_root"],
        "command_config_path": str(command_config_path),
        "command_config_kind": config.get("config_kind"),
        "command_config_version": config.get("config_version"),
        "verified_at_utc": config.get("verified_at_utc"),
        "source_verification_method": config.get("source_verification_method"),
        "formal_result_policy": config.get("formal_result_policy"),
        "command_templates_auto_applied": bool(command_templates_auto_applied),
        "required_modern_external_baseline_adapter_names": required_baseline_ids,
        "required_modern_external_baseline_adapter_count": len(required_baseline_ids),
        "configured_template_row_count": len(rows) - len(missing_template_ids),
        "missing_template_row_count": len(missing_template_ids),
        "missing_template_baseline_ids": missing_template_ids,
        "accepted_output_score_fields": config.get("accepted_output_score_fields", []),
        "official_result_bundle_policy": config.get("official_result_bundle_policy", {}),
        "required_command_format_tokens": config.get("required_command_format_tokens", []),
        "optional_command_format_tokens": config.get("optional_command_format_tokens", []),
        "colab_user_action_required": (
            "安装或克隆官方 baseline 源码, 提供官方权重、key/message/maintained info, 或运行项目内 official bundle cache 生成流程。"
            "默认 repository official adapter 会 fail closed; 若需要覆盖, 可配置 SSTW_<BASELINE>_NATIVE_EVAL_COMMAND "
            "或直接配置 SSTW_<BASELINE>_EVAL_COMMAND。"
        ),
        "summary_path": str(summary_path),
        "baseline_command_configs": rows,
        "claim_support_status": "external_baseline_command_template_summary_only_not_claim_evidence",
    }


def write_modern_baseline_colab_command_config_summary(
    layout: Mapping[str, str],
    *,
    profile: str,
    protocol_config_path: str | Path | None = None,
    command_config_path: str | Path = DEFAULT_MODERN_BASELINE_COLAB_COMMAND_CONFIG,
    command_templates_auto_applied: bool = False,
) -> dict[str, object]:
    """写出 command 配置摘要, 便于 Colab 失败后在 Google Drive 中审计。"""
    summary = build_modern_baseline_colab_command_config_summary(
        layout,
        profile=profile,
        protocol_config_path=protocol_config_path,
        command_config_path=command_config_path,
        command_templates_auto_applied=command_templates_auto_applied,
    )
    _write_json(Path(layout["drive_run_root"]) / EXTERNAL_BASELINE_COMMAND_TEMPLATE_SUMMARY, summary)
    return summary


def build_modern_baseline_official_bridge_command_templates(
    profile: str,
    *,
    protocol_config_path: str | Path | None = None,
    command_config_path: str | Path = DEFAULT_MODERN_BASELINE_COLAB_COMMAND_CONFIG,
) -> dict[str, str]:
    """从配置文件构造 SSTW 外层 bridge command 模板。

    返回值的 key 使用 baseline_id, 可以直接传入 `build_modern_baseline_command_env`。
    该函数只构造外层 bridge 命令; 真正的官方命令仍必须通过
    `SSTW_<BASELINE>_OFFICIAL_EVAL_COMMAND` 提供。
    """
    rows = _modern_baseline_command_config_rows(command_config_path)
    templates: dict[str, str] = {}
    for requirement in required_modern_external_baseline_command_requirements(profile, protocol_config_path):
        baseline_id = requirement["baseline_id"]
        template = str(rows.get(baseline_id, {}).get("sstw_eval_command_template") or "").strip()
        if template:
            templates[baseline_id] = template
    return templates


def build_repository_official_baseline_eval_command_templates(
    profile: str,
    *,
    protocol_config_path: str | Path | None = None,
    command_config_path: str | Path = DEFAULT_MODERN_BASELINE_COLAB_COMMAND_CONFIG,
) -> dict[str, str]:
    """从配置文件构造 repository official adapter 命令模板。

    返回值的 key 是 `SSTW_<BASELINE>_OFFICIAL_EVAL_COMMAND`。这些命令是 bridge
    内部命令, 会调用 `external_baseline/official_eval_adapters/` 下的 fail-closed
    wrapper。项目特定约束是: wrapper 只允许调用第三方官方源码/API或读取项目内 official bundle cache,
    缺少官方依赖时必须失败。
    """
    rows = _modern_baseline_command_config_rows(command_config_path)
    templates: dict[str, str] = {}
    for requirement in required_modern_external_baseline_command_requirements(profile, protocol_config_path):
        baseline_id = requirement["baseline_id"]
        env_var = requirement["official_baseline_command_env_var"]
        template = str(rows.get(baseline_id, {}).get("repository_official_eval_command_template") or "").strip()
        if template:
            templates[env_var] = template
    return templates


def _is_modern_baseline_official_bridge_command(command_template: str) -> bool:
    """判断外层 command 是否调用 repository bridge。

    该判断只用于 preflight 路由: 如果用户已经提供直接调用第三方 baseline 的
    `SSTW_<BASELINE>_EVAL_COMMAND`, 则不应再强制要求内部
    `SSTW_<BASELINE>_OFFICIAL_EVAL_COMMAND`。
    """
    return "external_baseline.official_command_bridge" in command_template


def build_modern_baseline_official_bridge_preflight_decision(
    layout: Mapping[str, str],
    *,
    profile: str,
    command_env: Mapping[str, str] | None = None,
    use_bridge_commands: bool,
    require_bridge_official_commands: bool,
    protocol_config_path: str | Path | None = None,
    command_config_path: str | Path = DEFAULT_MODERN_BASELINE_COLAB_COMMAND_CONFIG,
) -> dict[str, object]:
    """构造官方 bridge 命令预检决策。

    该预检不运行第三方 baseline, 只确认如果 Notebook 使用 bridge 外层命令,
    对应的官方原生命令是否已经配置。这样可以避免只配置了 bridge 壳层却在
    external baseline comparison 阶段才发现没有真正官方命令。
    """
    requirements = required_modern_external_baseline_command_requirements(profile, protocol_config_path)
    env_source = dict(os.environ)
    if command_env:
        env_source.update({str(key): str(value) for key, value in command_env.items()})
    rows = _modern_baseline_command_config_rows(command_config_path)
    required_env_vars: list[str] = []
    configured_env_vars: list[str] = []
    missing_env_vars: list[str] = []
    bridge_baseline_ids: list[str] = []
    direct_eval_baseline_ids: list[str] = []
    no_outer_command_baseline_ids: list[str] = []
    effective_outer_command_source: dict[str, str] = {}
    for item in requirements:
        baseline_id = item["baseline_id"]
        outer_env_var = item["external_baseline_command_env_var"]
        official_env_var = item["official_baseline_command_env_var"]
        outer_from_env = str(env_source.get(outer_env_var) or "").strip()
        outer_from_baseline_id = str(env_source.get(baseline_id) or "").strip()
        if not outer_from_env and not outer_from_baseline_id and use_bridge_commands:
            outer_from_baseline_id = str(rows.get(baseline_id, {}).get("sstw_eval_command_template") or "").strip()
        effective_outer_command = outer_from_env or outer_from_baseline_id
        if outer_from_env:
            effective_outer_command_source[baseline_id] = outer_env_var
        elif outer_from_baseline_id:
            effective_outer_command_source[baseline_id] = "baseline_id_template"
        else:
            effective_outer_command_source[baseline_id] = "missing"
            no_outer_command_baseline_ids.append(baseline_id)
            continue
        if _is_modern_baseline_official_bridge_command(effective_outer_command):
            bridge_baseline_ids.append(baseline_id)
            required_env_vars.append(official_env_var)
            if str(env_source.get(official_env_var) or "").strip():
                configured_env_vars.append(official_env_var)
            else:
                missing_env_vars.append(official_env_var)
        else:
            direct_eval_baseline_ids.append(baseline_id)
    try:
        paper_gate_profile = workflow_profile_is_paper_gate(profile)
    except Exception:
        paper_gate_profile = profile in PAPER_GATE_PROFILES
    hard_required = paper_gate_profile and use_bridge_commands and require_bridge_official_commands
    decision = "FAIL" if hard_required and missing_env_vars else "PASS"
    if not use_bridge_commands:
        status = "bridge_commands_disabled_direct_eval_commands_expected"
    elif not paper_gate_profile:
        status = "not_required_for_profile"
    elif not require_bridge_official_commands:
        status = "requirement_disabled"
    elif not bridge_baseline_ids:
        status = "official_bridge_not_required_for_direct_eval_commands"
    elif missing_env_vars:
        status = "official_bridge_commands_missing_for_paper_gate"
    else:
        status = "official_bridge_commands_configured_for_paper_gate"
    return {
        "artifact_name": "external_baseline_official_bridge_preflight_decision.json",
        "manifest_kind": "external_baseline_official_bridge_preflight",
        "profile": profile,
        "run_root": layout["drive_run_root"],
        "use_modern_baseline_bridge_commands": bool(use_bridge_commands),
        "require_bridge_official_commands": bool(require_bridge_official_commands),
        "external_baseline_official_bridge_preflight_decision": decision,
        "external_baseline_official_bridge_preflight_status": status,
        "paper_gate_profile": paper_gate_profile,
        "required_modern_external_baseline_adapter_names": [item["baseline_id"] for item in requirements],
        "official_bridge_planned_bridge_baseline_ids": bridge_baseline_ids,
        "official_bridge_direct_eval_baseline_ids": direct_eval_baseline_ids,
        "official_bridge_no_outer_command_baseline_ids": no_outer_command_baseline_ids,
        "official_bridge_effective_outer_command_source": effective_outer_command_source,
        "official_bridge_required_env_vars": required_env_vars,
        "official_bridge_configured_env_vars": configured_env_vars,
        "official_bridge_missing_env_vars": missing_env_vars,
        "official_bridge_required_env_var_count": len(required_env_vars),
        "official_bridge_configured_env_var_count": len(configured_env_vars),
        "official_bridge_missing_env_var_count": len(missing_env_vars),
        "official_bridge_source_dirs": {
            item["baseline_id"]: str(rows.get(item["baseline_id"], {}).get("colab_source_dir") or "")
            for item in requirements
        },
        "official_bridge_source_dir_check_status": "deferred_to_external_baseline_source_intake",
        "claim_support_status": "external_baseline_official_bridge_preflight_only_not_claim_evidence",
    }


def write_modern_baseline_official_bridge_preflight_decision(
    layout: Mapping[str, str],
    *,
    profile: str,
    command_env: Mapping[str, str] | None = None,
    use_bridge_commands: bool,
    require_bridge_official_commands: bool,
    protocol_config_path: str | Path | None = None,
    command_config_path: str | Path = DEFAULT_MODERN_BASELINE_COLAB_COMMAND_CONFIG,
) -> dict[str, object]:
    """写出官方 bridge 命令预检 artifact。"""
    decision = build_modern_baseline_official_bridge_preflight_decision(
        layout,
        profile=profile,
        command_env=command_env,
        use_bridge_commands=use_bridge_commands,
        require_bridge_official_commands=require_bridge_official_commands,
        protocol_config_path=protocol_config_path,
        command_config_path=command_config_path,
    )
    _write_json(Path(layout["drive_run_root"]) / EXTERNAL_BASELINE_OFFICIAL_BRIDGE_PREFLIGHT_DECISION, decision)
    return decision


def validate_modern_baseline_official_bridge_for_profile(
    preflight_decision: Mapping[str, object],
    *,
    allow_run_through_test: bool = False,
) -> None:
    """在 bridge 模式缺少官方原生命令时提前阻断。

    `allow_run_through_test` 只用于 Colab 工程链路跑通测试。开启后仍会保留 FAIL
    preflight artifact, 后续 external baseline records 也不能升级为正式论文 claim。
    该参数不能用于 paper 结果冻结。
    """
    if preflight_decision.get("external_baseline_official_bridge_preflight_decision") == "FAIL":
        if allow_run_through_test:
            return
        missing = preflight_decision.get("official_bridge_missing_env_vars")
        raise RuntimeError(
            "当前启用了现代视频水印 baseline bridge command, 但 bridge 内部缺少真正调用官方实现的命令。"
            f" 缺失: {missing}。请配置 SSTW_<BASELINE>_OFFICIAL_EVAL_COMMAND, "
            "其输出必须写入 {official_output_json_path}。"
        )


def build_modern_baseline_command_env(
    profile: str,
    command_templates: Mapping[str, str],
    config_path: str | Path | None = None,
) -> dict[str, str]:
    """构造现代 baseline command 环境变量映射。

    `command_templates` 可以使用 baseline_id 作为 key, 也可以直接使用环境变量名
    作为 key。环境变量名优先级更高, 因此用户在 Colab 中显式设置的
    `SSTW_<BASELINE>_EVAL_COMMAND` 可以覆盖默认 bridge 模板。这样 Notebook 只需要
    维护用户可编辑的短变量, 具体 hard gate 清单始终来自 protocol config。
    """
    env: dict[str, str] = {}
    for requirement in required_modern_external_baseline_command_requirements(profile, config_path):
        baseline_id = requirement["baseline_id"]
        env_var = requirement["external_baseline_command_env_var"]
        env[env_var] = str(command_templates.get(env_var) or command_templates.get(baseline_id) or "")
    return env


def build_external_baseline_colab_preflight_decision(
    layout: dict[str, str],
    *,
    profile: str,
    command_env: Mapping[str, str],
    require_modern_baseline_commands_for_paper_gate: bool,
    run_external_baseline_source_clone: bool,
    evidence_paths: list[str] | tuple[str, ...] | None = None,
    config_path: str | Path | None = None,
) -> dict[str, object]:
    """构造 external baseline Colab preflight 决策。

    该决策只检查真实 GPU 运行前是否具备现代 baseline command 配置, 不运行第三方
    baseline, 也不把配置存在解释为论文 claim。其价值在于: Colab 冷启动失败时,
    Google Drive 中仍保留可审计的阻断原因。
    """
    try:
        resolved_profile = resolve_notebook_workflow_profile(profile, allow_disabled=True)
    except Exception:
        resolved_profile = {
            "requested_workflow_profile": profile,
            "canonical_workflow_profile": profile,
            "workflow_profile": profile,
            "result_tier": profile,
        }
    requirements = required_modern_external_baseline_command_requirements(profile, config_path)
    required_env_vars = [item["external_baseline_command_env_var"] for item in requirements]
    configured_env_vars = [
        env_var for env_var in required_env_vars
        if str(command_env.get(env_var) or "").strip()
    ]
    missing_env_vars = [env_var for env_var in required_env_vars if env_var not in configured_env_vars]
    try:
        paper_gate_profile = workflow_profile_is_paper_gate(profile)
    except Exception:
        paper_gate_profile = profile in PAPER_GATE_PROFILES
    hard_required = paper_gate_profile and require_modern_baseline_commands_for_paper_gate
    decision = "FAIL" if hard_required and missing_env_vars else "PASS"
    if not paper_gate_profile:
        status = "not_required_for_profile"
    elif not require_modern_baseline_commands_for_paper_gate:
        status = "requirement_disabled"
    elif missing_env_vars:
        status = "commands_missing_for_paper_gate"
    else:
        status = "commands_configured_for_paper_gate"
    return {
        "artifact_name": "external_baseline_colab_preflight_decision.json",
        "manifest_kind": "external_baseline_colab_preflight",
        "profile": profile,
        "requested_workflow_profile": resolved_profile["requested_workflow_profile"],
        "canonical_workflow_profile": resolved_profile["canonical_workflow_profile"],
        "workflow_profile": resolved_profile["workflow_profile"],
        "result_tier": resolved_profile["result_tier"],
        "run_root": layout["drive_run_root"],
        "external_baseline_colab_preflight_decision": decision,
        "external_baseline_colab_preflight_status": status,
        "paper_gate_profile": paper_gate_profile,
        "require_modern_baseline_commands_for_paper_gate": bool(require_modern_baseline_commands_for_paper_gate),
        "run_external_baseline_source_clone": bool(run_external_baseline_source_clone),
        "external_baseline_command_template_config_path": DEFAULT_MODERN_BASELINE_COLAB_COMMAND_CONFIG,
        "external_baseline_command_template_summary_path": str(
            Path(layout["drive_run_root"]) / EXTERNAL_BASELINE_COMMAND_TEMPLATE_SUMMARY
        ),
        "required_modern_external_baseline_adapter_names": [item["baseline_id"] for item in requirements],
        "external_baseline_colab_preflight_required_env_vars": required_env_vars,
        "external_baseline_colab_preflight_configured_env_vars": configured_env_vars,
        "external_baseline_colab_preflight_missing_env_vars": missing_env_vars,
        "external_baseline_colab_preflight_missing_env_var_count": len(missing_env_vars),
        "external_baseline_colab_preflight_required_env_var_count": len(required_env_vars),
        "external_baseline_colab_preflight_configured_env_var_count": len(configured_env_vars),
        "external_baseline_evidence_path_count": len(evidence_paths or []),
        "evidence_paths": list(evidence_paths or []),
        "claim_support_status": "external_baseline_colab_preflight_only_not_claim_evidence",
    }


def write_external_baseline_colab_preflight_decision(
    layout: dict[str, str],
    *,
    profile: str,
    command_env: Mapping[str, str],
    require_modern_baseline_commands_for_paper_gate: bool,
    run_external_baseline_source_clone: bool,
    evidence_paths: list[str] | tuple[str, ...] | None = None,
    config_path: str | Path | None = None,
) -> dict[str, object]:
    """写出 external baseline Colab preflight 决策 artifact。"""
    decision = build_external_baseline_colab_preflight_decision(
        layout,
        profile=profile,
        command_env=command_env,
        require_modern_baseline_commands_for_paper_gate=require_modern_baseline_commands_for_paper_gate,
        run_external_baseline_source_clone=run_external_baseline_source_clone,
        evidence_paths=evidence_paths,
        config_path=config_path,
    )
    _write_json(Path(layout["drive_run_root"]) / EXTERNAL_BASELINE_COLAB_PREFLIGHT_DECISION, decision)
    return decision


def validate_modern_baseline_commands_for_profile(
    preflight_decision: Mapping[str, object],
    *,
    allow_run_through_test: bool = False,
) -> None:
    """在 paper gate profile 缺少现代 baseline command 时抛出明确错误。

    `allow_run_through_test` 只允许 Notebook 完成工程链路测试, 不改变 preflight
    decision, 不把缺失 baseline command 解释为正式 claim 支持。
    """
    if preflight_decision.get("external_baseline_colab_preflight_decision") == "FAIL":
        if allow_run_through_test:
            return
        missing = preflight_decision.get("external_baseline_colab_preflight_missing_env_vars")
        summary_path = preflight_decision.get("external_baseline_command_template_summary_path")
        raise RuntimeError(
            "当前 workflow profile 是 paper gate 或 paper gate 前最后门禁, 必须先在 Colab 配置现代视频水印 baseline command。"
            f" 缺失: {missing}。可先查看命令配置摘要: {summary_path}"
        )


def _dedupe_non_empty_strings(values: list[str] | tuple[str, ...]) -> list[str]:
    """按出现顺序去重非空字符串。

    该函数属于通用工程写法。这里用于合并环境变量路径列表, 防止 Colab
    多次重跑同一 cell 后把相同 bundle root 反复追加到 `os.pathsep`
    分隔的环境变量中。
    """

    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        result.append(text)
        seen.add(text)
    return result


def build_paper_gate_external_baseline_environment(
    layout: Mapping[str, str],
    *,
    profile: str,
    repo_root: str | Path | None = None,
) -> dict[str, str]:
    """构造 paper gate 聚合阶段所需的 modern baseline 环境变量。

    该函数是项目特定写法。baseline 专用 Notebook 只负责生成各自的
    official bundle; `paper_gate_and_package_colab` 需要再次调用统一
    `external_baseline_runner`, 因此必须在该 Notebook 中显式注入:

    1. 外层 `SSTW_<BASELINE>_EVAL_COMMAND`, 用于让现代 baseline adapter
       从 `not_runnable` 升级为 `runnable`。
    2. 内层 `SSTW_<BASELINE>_OFFICIAL_EVAL_COMMAND`, 用于让 bridge 读取
       项目内 official bundle 或调用 fail-closed 官方 wrapper。
    3. `SSTW_EXTERNAL_BASELINE_OFFICIAL_RESULT_BUNDLE_ROOT`, 用于指向已从
       阶段 zip 恢复到本地 workspace 的 official bundle 根目录。

    该函数只准备执行环境, 不写正式 comparison records。
    """

    bridge_templates = build_modern_baseline_official_bridge_command_templates(profile)
    repository_official_templates = build_repository_official_baseline_eval_command_templates(profile)
    command_template_source: dict[str, str] = {}
    command_template_source.update(bridge_templates)
    command_template_source.update(repository_official_templates)
    command_template_source.update({key: value for key, value in os.environ.items() if value})
    outer_command_env = build_modern_baseline_command_env(profile, command_template_source)

    bundle_root = str(layout.get("external_baseline_official_result_bundle_root") or "").strip()
    existing_bundle_roots = [
        item
        for item in os.environ.get("SSTW_EXTERNAL_BASELINE_OFFICIAL_RESULT_BUNDLE_ROOTS", "").split(os.pathsep)
        if item
    ]
    existing_evidence_paths = [
        item
        for item in os.environ.get("SSTW_EXTERNAL_BASELINE_EVIDENCE_PATHS", "").split(os.pathsep)
        if item
    ]

    env: dict[str, str] = {}
    if repo_root:
        env["PYTHONPATH"] = str(repo_root) + os.pathsep + os.environ.get("PYTHONPATH", "")
    env.update(repository_official_templates)
    env.update(outer_command_env)
    for key, value in os.environ.items():
        if key.startswith("SSTW_") and value:
            env[key] = value
    if bundle_root:
        env["SSTW_EXTERNAL_BASELINE_OFFICIAL_RESULT_BUNDLE_ROOT"] = bundle_root
        env["SSTW_EXTERNAL_BASELINE_OFFICIAL_RESULT_BUNDLE_ROOTS"] = os.pathsep.join(
            _dedupe_non_empty_strings([bundle_root, *existing_bundle_roots])
        )
        env["SSTW_EXTERNAL_BASELINE_EVIDENCE_PATHS"] = os.pathsep.join(
            _dedupe_non_empty_strings([bundle_root, *existing_evidence_paths])
        )
    return env


def apply_paper_gate_external_baseline_environment(
    layout: Mapping[str, str],
    *,
    profile: str,
    repo_root: str | Path | None = None,
    run_external_baseline_source_clone: bool = False,
) -> dict[str, object]:
    """应用 paper gate 聚合阶段的 modern baseline 环境变量并写出预检 artifact。

    Notebook 入口调用该函数后, 后续 `external_baseline_comparison` 子进程会继承
    同一组环境变量, 从已恢复的 official bundle 生成正式 `measured_formal`
    records。该函数写出的 artifact 只说明环境和命令模板已准备, 不直接支持论文
    效果主张。
    """

    env = build_paper_gate_external_baseline_environment(
        layout,
        profile=profile,
        repo_root=repo_root,
    )
    os.environ.update({key: value for key, value in env.items() if value})

    summary = write_modern_baseline_colab_command_config_summary(
        layout,
        profile=profile,
        command_templates_auto_applied=True,
    )
    bridge_decision = write_modern_baseline_official_bridge_preflight_decision(
        layout,
        profile=profile,
        command_env=env,
        use_bridge_commands=True,
        require_bridge_official_commands=True,
    )
    outer_command_env = {
        key: value
        for key, value in env.items()
        if key.startswith("SSTW_") and key.endswith("_EVAL_COMMAND") and "_OFFICIAL_" not in key
    }
    evidence_paths = [env["SSTW_EXTERNAL_BASELINE_OFFICIAL_RESULT_BUNDLE_ROOT"]] if env.get("SSTW_EXTERNAL_BASELINE_OFFICIAL_RESULT_BUNDLE_ROOT") else []
    baseline_preflight = write_external_baseline_colab_preflight_decision(
        dict(layout),
        profile=profile,
        command_env=outer_command_env,
        require_modern_baseline_commands_for_paper_gate=True,
        run_external_baseline_source_clone=run_external_baseline_source_clone,
        evidence_paths=evidence_paths,
    )
    decision = "PASS" if (
        bridge_decision.get("external_baseline_official_bridge_preflight_decision") == "PASS"
        and baseline_preflight.get("external_baseline_colab_preflight_decision") == "PASS"
    ) else "FAIL"
    payload: dict[str, object] = {
        "artifact_name": "paper_gate_external_baseline_environment_decision.json",
        "manifest_kind": "paper_gate_external_baseline_environment_decision",
        "profile": profile,
        "run_root": layout["drive_run_root"],
        "paper_gate_external_baseline_environment_decision": decision,
        "paper_gate_external_baseline_environment_status": (
            "repository_official_bundle_environment_ready"
            if decision == "PASS"
            else "repository_official_bundle_environment_blocked"
        ),
        "command_templates_auto_applied": True,
        "external_baseline_official_result_bundle_root": env.get("SSTW_EXTERNAL_BASELINE_OFFICIAL_RESULT_BUNDLE_ROOT", ""),
        "external_baseline_official_result_bundle_roots": env.get("SSTW_EXTERNAL_BASELINE_OFFICIAL_RESULT_BUNDLE_ROOTS", ""),
        "configured_outer_command_env_vars": sorted(outer_command_env),
        "configured_outer_command_env_var_count": len(outer_command_env),
        "configured_repository_official_command_env_vars": sorted(
            key for key in env if key.startswith("SSTW_") and key.endswith("_OFFICIAL_EVAL_COMMAND")
        ),
        "configured_repository_official_command_env_var_count": sum(
            1
            for key in env
            if key.startswith("SSTW_") and key.endswith("_OFFICIAL_EVAL_COMMAND")
        ),
        "external_baseline_command_template_summary_path": summary.get("summary_path"),
        "external_baseline_colab_preflight_decision": baseline_preflight.get("external_baseline_colab_preflight_decision"),
        "external_baseline_official_bridge_preflight_decision": bridge_decision.get("external_baseline_official_bridge_preflight_decision"),
        "required_modern_external_baseline_adapter_names": summary.get("required_modern_external_baseline_adapter_names", []),
        "claim_support_status": "paper_gate_environment_preflight_only_not_claim_evidence",
    }
    _write_json(Path(layout["drive_run_root"]) / PAPER_GATE_EXTERNAL_BASELINE_ENVIRONMENT_DECISION, payload)
    return payload


def read_motion_threshold_calibration_decision(layout: Mapping[str, str]) -> dict[str, Any]:
    """读取已落盘的 motion threshold calibration 决策, 并兼容 UTF-8 BOM。"""
    candidate_roots = [
        Path(layout.get("motion_threshold_artifact_run_root") or layout["drive_run_root"]),
        Path(layout["drive_run_root"]),
    ]
    candidate_paths = [
        root / "artifacts" / "motion_threshold_calibration_decision.json"
        for root in candidate_roots
    ]
    decision_path = next((path for path in candidate_paths if path.exists()), None)
    if decision_path is None:
        raise FileNotFoundError(
            "缺少 motion_threshold_calibration_decision.json。"
            " 请先运行 motion_threshold_calibration Notebook, 或确认当前 workflow profile 配置的阈值 artifact run_root 中已有阈值 artifact。"
        )
    return json.loads(decision_path.read_text(encoding="utf-8-sig"))


def write_motion_threshold_reuse_artifact_for_profile(
    layout: Mapping[str, str],
    profile: str,
) -> dict[str, Any]:
    """校验并复制当前 profile 复用的 motion threshold artifact。

    该函数属于项目特定写法。`motion_threshold_calibration_colab` 的输出目录与
    `validation_scale` / `pilot_paper` 的运行目录相互隔离, 但 paper gate 只应读取当前
    run_root 中的 governed artifacts。因此非 calibration profile 在复用阈值时, 必须把
    已通过的阈值决策复制到当前 run_root, 并额外写出 reuse decision 说明来源。
    """
    reuse = validate_motion_threshold_ready_for_profile(layout, profile)
    if not reuse.get("motion_threshold_reuse_required"):
        return reuse

    decision = read_motion_threshold_calibration_decision(layout)
    run_root = Path(layout["drive_run_root"])
    artifact_path = run_root / "artifacts" / "motion_threshold_calibration_decision.json"
    reuse_path = run_root / "artifacts" / "motion_threshold_reuse_decision.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    artifact_payload = {
        **decision,
        "motion_threshold_reused_by_profile": canonical_workflow_profile(profile),
        "motion_threshold_reuse_source_run_root": str(layout.get("motion_threshold_artifact_run_root") or ""),
        "motion_threshold_reuse_target_run_root": str(layout["drive_run_root"]),
        "claim_support_status": decision.get("claim_support_status", "motion_threshold_calibration_ready"),
    }
    reuse_payload = {
        "stage_id": "motion_threshold_reuse_check",
        "motion_threshold_reuse_decision": "PASS",
        "motion_threshold_reuse_required": True,
        "motion_threshold_reuse_status": "ready",
        "workflow_profile": canonical_workflow_profile(profile),
        "source_artifact_run_root": str(layout.get("motion_threshold_artifact_run_root") or ""),
        "target_run_root": str(layout["drive_run_root"]),
        "persisted_motion_threshold_artifact_path": str(artifact_path),
        **reuse,
    }
    _write_json(artifact_path, artifact_payload)
    _write_json(reuse_path, reuse_payload)
    return reuse_payload


def validate_motion_threshold_ready_for_profile(
    layout: Mapping[str, str],
    profile: str,
) -> dict[str, Any]:
    """校验非 calibration profile 只能复用已冻结的 motion threshold。

    该函数避免 Notebook 在 `validation_scale` 或 `pilot_paper` 中根据当前测试样本重新估计
    motion threshold。阈值必须来自独立 calibration split, 否则 fixed-FPR 协议会被污染。
    """
    if canonical_workflow_profile(profile) == "motion_calibration":
        return {
            "motion_threshold_reuse_required": False,
            "motion_threshold_reuse_status": "not_required_for_motion_calibration",
        }
    decision = read_motion_threshold_calibration_decision(layout)
    if decision.get("motion_threshold_calibration_ready") is not True:
        raise RuntimeError(
            "当前 workflow profile 需要复用已通过的 motion threshold calibration artifact, "
            "但 artifact 未通过: "
            + str(decision.get("motion_threshold_calibration_decision"))
        )
    return {
        "motion_threshold_reuse_required": True,
        "motion_threshold_reuse_status": "ready",
        "motion_threshold_calibration_decision": decision.get("motion_threshold_calibration_decision"),
        "motion_threshold_id": decision.get("motion_threshold_id"),
        "motion_threshold_source_split": decision.get("motion_threshold_source_split"),
        "motion_delta_threshold": decision.get("motion_delta_threshold"),
        "claim_support_status": decision.get("claim_support_status"),
    }


def build_prompt_suite_command(layout: dict[str, str]) -> list[str]:
    """构造 prompt suite 数据集命令, 该命令不执行 GPU 模型测试。"""
    return [sys.executable, "scripts/prepare_generative_video_prompt_suite.py", "--output-root", layout["drive_dataset_root"]]


def build_colab_runtime_command(layout: dict[str, str], profile: str, model_id: str, cross_model_id: str = "") -> list[str]:
    """构造 B5 Colab GPU 运行命令。"""
    command = [
        sys.executable,
        "-m",
        "experiments.generative_video_model_probe.colab_runtime",
        "--output-root",
        layout["drive_run_root"],
        "--prompt-suite-path",
        layout["prompt_suite_path"],
        "--profile",
        profile,
        "--model-id",
        model_id,
    ]
    if cross_model_id:
        command.extend(["--cross-model-id", cross_model_id])
    return command


def build_formal_metric_command(
    layout: dict[str, str],
    semantic_model_id: str = "openai/clip-vit-base-patch32",
    semantic_frame_limit: int = 8,
    disable_semantic_metric: bool = False,
) -> list[str]:
    """构造 B5 正式质量、运动与语义 metric 命令, 从实际 mp4 文件生成 governed records。"""
    command = [
        sys.executable,
        "-m",
        "experiments.generative_video_model_probe.formal_metric_runner",
        "--run-root",
        layout["drive_run_root"],
        "--prompt-suite-path",
        layout["prompt_suite_path"],
        "--semantic-model-id",
        semantic_model_id,
        "--semantic-frame-limit",
        str(semantic_frame_limit),
    ]
    if disable_semantic_metric:
        command.append("--disable-semantic-metric")
    return command


def build_motion_threshold_calibration_command(layout: dict[str, str]) -> list[str]:
    """构造 formal motion threshold calibration 命令, 从 formal motion records 冻结或报告阈值状态。"""
    return [
        sys.executable,
        "-m",
        "experiments.generative_video_model_probe.motion_threshold_calibration",
        "--run-root",
        layout["drive_run_root"],
    ]


def build_mechanism_postprocess_command(layout: dict[str, str]) -> list[str]:
    """构造 B5 Colab 机制后处理命令, target_fpr 由当前 protocol config 决定。"""
    return [
        sys.executable,
        "-m",
        "experiments.generative_video_model_probe.postprocess_runner",
        "--run-root",
        layout["drive_run_root"],
        "--config-path",
        layout["protocol_config_path"],
    ]


def build_pilot_matrix_postprocess_command(layout: dict[str, str]) -> list[str]:
    """构造 small-scale pilot matrix postprocess 命令, 从 generation 与 trajectory records 补齐 pilot 矩阵。"""
    return [
        sys.executable,
        "-m",
        "experiments.generative_video_model_probe.pilot_matrix_postprocess",
        "--run-root",
        layout["drive_run_root"],
    ]


def build_protocol_evaluation_matrix_postprocess_command(layout: dict[str, str]) -> list[str]:
    """构造主干协议评估矩阵 postprocess 命令, 避免继续写出历史 small-scale gate 文件名。"""
    return [
        sys.executable,
        "-m",
        "experiments.generative_video_model_probe.pilot_matrix_postprocess",
        "--run-root",
        layout["drive_run_root"],
        "--output-family",
        "protocol_evaluation_matrix",
    ]


def build_runtime_attack_command(layout: dict[str, str]) -> list[str]:
    """构造 runtime video-file attack 命令, 对真实 mp4 生成 attacked videos 与 governed records。"""
    return [
        sys.executable,
        "-m",
        "experiments.generative_video_model_probe.attack_runner",
        "--run-root",
        layout["drive_run_root"],
    ]


def build_runtime_detection_command(layout: dict[str, str]) -> list[str]:
    """构造 runtime attacked video detection 命令, 把 attacked videos 接入检测评分 records。"""
    return [
        sys.executable,
        "-m",
        "experiments.generative_video_model_probe.detection_runner",
        "--run-root",
        layout["drive_run_root"],
    ]


def build_external_baseline_source_intake_command(layout: dict[str, str], execute_clone: bool = False) -> list[str]:
    """构造 external baseline source intake 命令, 写出源码、adapter 和命令配置治理清单。

    `execute_clone` 仅应在 Colab 冷启动且用户已经确认第三方源码 URL 可访问时启用。
    默认不访问网络, 这样本地测试和 harness 审计不会被外部仓库状态影响。
    """
    command = [
        sys.executable,
        "scripts/build_external_baseline_source_intake.py",
        "--output-root",
        f"{layout['drive_run_root']}/artifacts",
        "--repo-root",
        ".",
    ]
    if execute_clone:
        command.append("--execute-clone")
    return command


def build_external_baseline_official_resource_bootstrap_command(
    layout: dict[str, str],
    *,
    allow_network: bool = True,
) -> list[str]:
    """构造现代 baseline 官方资源自动准备命令。

    该阶段属于 Colab 冷启动修复层。它会尝试下载公开 checkpoint、安装可公开安装的
    官方依赖, 并把能回写到 Notebook 父进程的环境变量落盘到 bootstrap artifact。
    对没有公开权重或超出当前 GPU 资源的 baseline, 它只能写出明确阻断原因, 不能
    生成 proxy 分数。
    """
    command = [
        sys.executable,
        "-m",
        "external_baseline.official_resource_bootstrap",
        "--run-root",
        layout["drive_run_root"],
        "--resource-root",
        layout["external_baseline_resource_root"],
    ]
    if not allow_network:
        command.append("--disable-network")
    return command


def build_external_baseline_official_bundle_generation_command(
    layout: dict[str, str],
    *,
    generate_auto_supported: bool = True,
) -> list[str]:
    """构造现代 baseline official bundle 自动生成命令。

    该命令只对仓库能真实调用官方 API 或项目内官方流程运行器的 baseline 生成 bundle。
    当前可自动尝试 VideoSeal、VideoMark 与 VidSig。其它需要未公开训练 extractor、
    PRC key、maintained info 或超出当前 GPU 资源的 baseline 会写入计划和阻断说明,
    不会被伪造成 measured_formal。
    """
    command = [
        sys.executable,
        "-m",
        "external_baseline.official_bundle_generator",
        "--run-root",
        layout["drive_run_root"],
        "--bundle-root",
        layout["external_baseline_official_result_bundle_root"],
    ]
    if generate_auto_supported:
        command.append("--generate-auto-supported")
    return command


def build_external_baseline_official_runtime_closure_command(
    layout: dict[str, str],
    *,
    baseline_id: str | None = None,
) -> list[str]:
    """构造现代 baseline 真实运行闭合要求预检命令。

    该命令只写出 source、requirements、runtime inputs、官方资源和 official bundle
    cache 的缺口清单。它不会运行第三方 baseline, 因此可以在 Colab 正式 scoring
    前安全执行, 用于把“缺哪个文件或配置”明确落盘。
    """
    command = [
        sys.executable,
        "-m",
        "external_baseline.official_runtime_closure",
        "--run-root",
        layout["drive_run_root"],
        "--repo-root",
        ".",
        "--resource-root",
        layout["external_baseline_resource_root"],
        "--official-result-bundle-root",
        layout["external_baseline_official_result_bundle_root"],
        "--output-json",
        str(Path(layout["drive_run_root"]) / EXTERNAL_BASELINE_OFFICIAL_RUNTIME_CLOSURE_REQUIREMENTS),
    ]
    if baseline_id:
        command.extend(["--baseline-id", baseline_id])
    return command


def build_external_baseline_official_result_bundle_preflight_command(layout: dict[str, str]) -> list[str]:
    """构造现代 baseline 官方结果包完整性检查命令。

    该阶段不生成分数, 只检查当前 run_root 中的 runtime comparison unit 是否都能在
    `SSTW_EXTERNAL_BASELINE_OFFICIAL_RESULT_BUNDLE_ROOT(S)` 中找到官方结果, 或者当前
    Colab 会话是否已经具备可直接运行的官方资源。它的作用是把严格门禁的 baseline
    资源阻断前移到 comparison 前。
    """
    return [
        sys.executable,
        "-m",
        "external_baseline.official_result_bundle",
        "--run-root",
        layout["drive_run_root"],
        "--output-json",
        str(Path(layout["drive_run_root"]) / EXTERNAL_BASELINE_OFFICIAL_RESULT_BUNDLE_PREFLIGHT_DECISION),
    ]



def build_external_baseline_comparison_command(layout: dict[str, str]) -> list[str]:
    """构建 external_baseline adapter comparison 命令, 从 runtime detection records 生成 baseline 对比结果。"""
    return [
        sys.executable,
        "-m",
        "experiments.generative_video_model_probe.external_baseline_runner",
        "--run-root",
        layout["drive_run_root"],
        "--mode",
        "comparison",
    ]


def build_external_baseline_self_containment_decision_command(layout: dict[str, str]) -> list[str]:
    """构造 external baseline 自包含产出判定命令。

    该命令只读取 comparison records、source intake manifests 和官方命令 evidence,
    检查 5 个主实验现代 baseline 是否由项目内 clone / build / run / adapt / record 产出
    `measured_formal` 结果。non-run record 会被保留为阻断原因, 不能替代正式结果。
    """
    return [
        sys.executable,
        "-m",
        "scripts.check_results.external_baseline_self_containment_decision",
        "--run-root",
        layout["drive_run_root"],
        "--config-path",
        layout["protocol_config_path"],
    ]


def build_data_split_and_leakage_guard_command(layout: dict[str, str]) -> list[str]:
    """构造数据切分与泄漏检查命令。"""
    return [
        sys.executable,
        "-m",
        "scripts.check_results.data_split_and_leakage_guard",
        "--run-root",
        layout["drive_run_root"],
    ]


def build_validation_internal_ablation_command(layout: dict[str, str]) -> list[str]:
    """构造 validation-scale 内部消融矩阵后处理命令。"""
    return [
        sys.executable,
        "-m",
        "experiments.generative_video_model_probe.validation_internal_ablation",
        "--run-root",
        layout["drive_run_root"],
    ]


def build_motion_consistency_exclusion_report_command(layout: dict[str, str]) -> list[str]:
    """构造 motion consistency 阻断样本处理报告命令。"""
    return [
        sys.executable,
        "-m",
        "experiments.generative_video_model_probe.motion_consistency_exclusion_report",
        "--run-root",
        layout["drive_run_root"],
        "--config-path",
        layout["protocol_config_path"],
    ]


def build_adaptive_attack_command(layout: dict[str, str]) -> list[str]:
    """构造 validation-scale adaptive attack proxy 命令。"""
    return [
        sys.executable,
        "-m",
        "experiments.generative_video_model_probe.adaptive_attack_runner",
        "--run-root",
        layout["drive_run_root"],
    ]


def build_replay_and_sketch_gate_command(layout: dict[str, str]) -> list[str]:
    """构造 replay/sketch gate validation proxy 命令。"""
    return [
        sys.executable,
        "-m",
        "experiments.generative_video_model_probe.replay_and_sketch_gate",
        "--run-root",
        layout["drive_run_root"],
    ]


def build_claim3_downgrade_command(layout: dict[str, str]) -> list[str]:
    """构造 Claim-3 downgrade gate 命令, 明确 replay/sketch 未闭合时的 claim 边界。"""
    return [
        sys.executable,
        "-m",
        "experiments.generative_video_model_probe.claim3_downgrade",
        "--run-root",
        layout["drive_run_root"],
    ]


def build_statistical_confidence_interval_command(layout: dict[str, str]) -> list[str]:
    """构造当前 profile 的统计置信区间报告命令。"""
    return [
        sys.executable,
        "-m",
        "experiments.generative_video_model_probe.statistical_confidence_interval",
        "--run-root",
        layout["drive_run_root"],
        "--config-path",
        layout["protocol_config_path"],
    ]


def build_sstw_measured_formal_result_command(layout: dict[str, str]) -> list[str]:
    """构造 SSTW 本方法 measured_formal 结果转写命令。"""
    return [
        sys.executable,
        "-m",
        "experiments.generative_video_model_probe.sstw_formal_result",
        "--run-root",
        layout["drive_run_root"],
        "--config-path",
        layout["protocol_config_path"],
    ]


def build_fair_detection_calibration_command(layout: dict[str, str]) -> list[str]:
    """构造 clean negative calibration 公平比较命令。"""
    return [
        sys.executable,
        "-m",
        "experiments.generative_video_model_probe.fair_detection_calibration",
        "--run-root",
        layout["drive_run_root"],
        "--config-path",
        layout["protocol_config_path"],
    ]


def build_formal_method_baseline_comparison_command(layout: dict[str, str]) -> list[str]:
    """构造 SSTW 与现代 external baseline 的同协议 measured_formal 比较表命令。"""
    return [
        sys.executable,
        "-m",
        "experiments.generative_video_model_probe.formal_method_baseline_comparison",
        "--run-root",
        layout["drive_run_root"],
        "--config-path",
        layout["protocol_config_path"],
    ]


def build_formal_baseline_difference_interval_command(layout: dict[str, str]) -> list[str]:
    """构造 SSTW 相对现代 external baseline 的差值置信区间报告命令。"""
    return [
        sys.executable,
        "-m",
        "experiments.generative_video_model_probe.formal_baseline_difference_interval",
        "--run-root",
        layout["drive_run_root"],
        "--config-path",
        layout["protocol_config_path"],
    ]


def build_validation_scale_formal_internal_ablation_command(layout: dict[str, str]) -> list[str]:
    """构造 validation_scale 级内部消融汇总命令。"""
    return [
        sys.executable,
        "-m",
        "experiments.generative_video_model_probe.validation_scale_formal_internal_ablation",
        "--run-root",
        layout["drive_run_root"],
        "--config-path",
        layout["protocol_config_path"],
    ]


def build_low_fpr_formal_statistics_command(layout: dict[str, str]) -> list[str]:
    """构造 validation_scale 低 FPR 正式统计阻断记录命令。"""
    return [
        sys.executable,
        "-m",
        "experiments.generative_video_model_probe.low_fpr_formal_statistics",
        "--run-root",
        layout["drive_run_root"],
        "--config-path",
        layout["protocol_config_path"],
    ]


def build_pilot_paper_gate_command(layout: dict[str, str]) -> list[str]:
    """构建当前 profile 的 fixed-FPR gate 命令, fixed-FPR 口径来自 protocol config。"""
    return [
        sys.executable,
        "-m",
        "experiments.generative_video_model_probe.pilot_paper_gate",
        "--run-root",
        layout["drive_run_root"],
        "--config-path",
        layout["protocol_config_path"],
        "--write-outputs",
    ]

def build_validation_artifact_rebuild_dry_run_command(layout: dict[str, str]) -> list[str]:
    """构造 validation-scale artifact rebuild dry-run 命令。"""
    return [
        sys.executable,
        "-m",
        "experiments.generative_video_model_probe.validation_artifact_rebuild",
        "--run-root",
        layout["drive_run_root"],
    ]


def build_small_scale_claim_pilot_gate_command(layout: dict[str, str]) -> list[str]:
    """构造 small-scale claim pilot gate 命令, 从 governed records 汇总 pilot 状态。"""
    return [
        sys.executable,
        "-m",
        "experiments.generative_video_model_probe.pilot_claim_gate",
        "--run-root",
        layout["drive_run_root"],
        "--write-outputs",
    ]


def build_validation_scale_gate_command(layout: dict[str, str]) -> list[str]:
    """构造 validation-scale gate 命令, target_fpr 口径来自 validation protocol config。"""
    return [
        sys.executable,
        "-m",
        "experiments.generative_video_model_probe.validation_scale_gate",
        "--run-root",
        layout["drive_run_root"],
        "--config-path",
        layout["protocol_config_path"],
        "--write-outputs",
    ]


def build_stage_transition_decision_command(layout: dict[str, str], transition_id: str) -> list[str]:
    """构造主干阶段跳转判定命令。"""
    return [
        sys.executable,
        "-m",
        "scripts.check_results.stage_transition_decision",
        "--run-root",
        layout["drive_run_root"],
        "--transition",
        transition_id,
    ]


def build_validation_scale_to_pilot_paper_transition_decision_command(layout: dict[str, str]) -> list[str]:
    """构造 validation_scale -> pilot_paper 轻量跳转判定命令。"""
    return build_stage_transition_decision_command(layout, "validation_scale_to_pilot_paper")


def build_pilot_paper_to_full_paper_transition_decision_command(layout: dict[str, str]) -> list[str]:
    """构造 pilot_paper -> full_paper 轻量跳转判定命令。"""
    return build_stage_transition_decision_command(layout, "pilot_paper_to_full_paper")


def build_full_paper_to_submission_freeze_transition_decision_command(layout: dict[str, str]) -> list[str]:
    """构造 full_paper -> submission_freeze 轻量跳转判定命令。"""
    return build_stage_transition_decision_command(layout, "full_paper_to_submission_freeze")


def build_validation_scale_gate_figure_builder_command(layout: dict[str, str]) -> list[str]:
    """构造 validation_scale gate 诊断图 manifest 重建命令。"""
    return [
        sys.executable,
        "-m",
        "experiments.generative_video_model_probe.validation_scale_artifact_package",
        "--run-root",
        layout["drive_run_root"],
        "--mode",
        "figure",
    ]


def build_validation_scale_package_manifest_builder_command(layout: dict[str, str]) -> list[str]:
    """构造 validation_scale package manifest 重建命令。"""
    return [
        sys.executable,
        "-m",
        "experiments.generative_video_model_probe.validation_scale_artifact_package",
        "--run-root",
        layout["drive_run_root"],
        "--mode",
        "manifest",
    ]


def _stage_zip_mode_uses_unified_package(layout: Mapping[str, str]) -> bool:
    """判断当前 Notebook 是否应跳过历史 drive packager。

    Colab 阶段 zip 交接模式下, `publish_colab_stage_package` 已负责把完整阶段包
    写回新目录结构。继续运行历史 drive packager 会生成重复 zip, 增加 Google Drive
    占用并制造后续读取歧义。
    """

    mode = str(layout.get("stage_package_handoff_mode") or os.environ.get("SSTW_COLAB_STAGE_IO_MODE", ""))
    return mode.strip().lower() in {"local_zip", "stage_zip", "zip_handoff"}


def _build_legacy_drive_packaging_noop_command(packager_name: str) -> list[str]:
    """构造可执行的 no-op 命令, 让 Notebook 编排保持单一路径。"""

    message = (
        "SSTW stage zip handoff is active; skip legacy drive packager "
        f"{packager_name}. Unified output is written by publish_colab_stage_package."
    )
    return [sys.executable, "-c", f"print({message!r})"]


def build_drive_packaging_command(layout: dict[str, str], include_videos: bool = True) -> list[str]:
    """构造 Google Drive 打包命令。"""
    if _stage_zip_mode_uses_unified_package(layout):
        return _build_legacy_drive_packaging_noop_command("generative_video_drive_packager.py")

    command = [
        sys.executable,
        "scripts/package_results/generative_video_drive_packager.py",
        "--run-root",
        layout["drive_run_root"],
        "--drive-package-dir",
        layout["drive_package_dir"],
    ]
    if not include_videos:
        command.append("--exclude-videos")
    return command


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    """执行 Notebook 编排命令, 并实时显示 repository runner 进度。"""
    return run_streaming_command(command)


def _env_flag(name: str, default: bool) -> bool:
    """从环境变量读取布尔开关。

    该函数属于通用工程写法, 用于把 Colab 入口中易变的用户开关集中到
    repository helper。Notebook cell 不再维护阶段流程或产物语义, 只负责把
    profile 和少量运行入口参数传入本模块。
    """

    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_path_list(name: str) -> list[str]:
    """按当前系统路径分隔符读取环境变量中的 evidence 路径列表。"""

    return [item for item in os.environ.get(name, "").split(os.pathsep) if item]


def _run_stage_command_or_raise(stage_name: str, command: list[str]) -> dict[str, Any]:
    """执行单个 governed stage 命令, 非零退出时立即阻断。

    这一实现属于 Notebook 入口复用层, 目的是避免每个 Notebook cell 都复制
    `run_or_raise` 逻辑。具体命令仍由 repository command builder 生成。
    """

    print("\n===== stage:", stage_name, "=====")
    print(" ".join(command))
    result = run_command(command)
    print(result.stdout)
    print(result.stderr)
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    return {
        "stage_name": stage_name,
        "stage_execution_status": "completed",
        "stage_execution_kind": "command",
        "command": command,
        "returncode": result.returncode,
    }


def write_external_baseline_colab_preflight_for_profile(
    layout: dict[str, str],
    *,
    workflow_profile: str,
) -> dict[str, Any]:
    """执行 external baseline Colab 预检并写入 governed preflight artifacts。

    该函数把此前散落在 Notebook cell 中的 baseline command 配置、bridge 预检、
    source clone 开关和 evidence path 回写集中到 Python helper 中。Notebook 不再
    维护 baseline 运行流程或产物闭环规则。
    """

    resolved = resolve_notebook_workflow_profile(
        workflow_profile,
        "external_baseline_formal_scoring"
        if layout.get("notebook_role") == "external_baseline_formal_scoring"
        else layout.get("notebook_role", ""),
        allow_disabled=True,
    ) if layout.get("notebook_role") else resolve_notebook_workflow_profile(workflow_profile, allow_disabled=True)
    runtime_profile = str(resolved.get("runtime_profile") or workflow_profile)
    run_external_baseline_source_clone = _env_flag("SSTW_RUN_EXTERNAL_BASELINE_SOURCE_CLONE", True)
    require_modern_baseline_commands_for_paper_gate = _env_flag(
        "SSTW_REQUIRE_MODERN_BASELINE_COMMANDS_FOR_PAPER_GATE",
        True,
    )
    use_modern_baseline_bridge_commands = _env_flag("SSTW_USE_MODERN_BASELINE_BRIDGE_COMMANDS", True)
    require_modern_baseline_bridge_official_commands = _env_flag(
        "SSTW_REQUIRE_MODERN_BASELINE_BRIDGE_OFFICIAL_COMMANDS",
        True,
    )
    use_repository_official_baseline_adapters = _env_flag(
        "SSTW_USE_REPOSITORY_OFFICIAL_BASELINE_ADAPTERS",
        True,
    )
    evidence_paths = _env_path_list("SSTW_EXTERNAL_BASELINE_EVIDENCE_PATHS")

    modern_command_config_summary = write_modern_baseline_colab_command_config_summary(
        layout,
        profile=workflow_profile,
    )
    print(json.dumps(modern_command_config_summary, ensure_ascii=False, indent=2))

    command_template_source: dict[str, str] = {}
    if use_modern_baseline_bridge_commands:
        command_template_source.update(build_modern_baseline_official_bridge_command_templates(runtime_profile))
    if use_repository_official_baseline_adapters:
        command_template_source.update(build_repository_official_baseline_eval_command_templates(runtime_profile))
    command_template_source.update({str(key): str(value) for key, value in os.environ.items()})

    bridge_preflight_decision = write_modern_baseline_official_bridge_preflight_decision(
        layout,
        profile=workflow_profile,
        command_env=command_template_source,
        use_bridge_commands=use_modern_baseline_bridge_commands,
        require_bridge_official_commands=require_modern_baseline_bridge_official_commands,
    )
    print(json.dumps(bridge_preflight_decision, ensure_ascii=False, indent=2))
    validate_modern_baseline_official_bridge_for_profile(bridge_preflight_decision)

    modern_command_env = build_modern_baseline_command_env(runtime_profile, command_template_source)
    external_baseline_preflight_decision = write_external_baseline_colab_preflight_decision(
        layout,
        profile=workflow_profile,
        command_env=modern_command_env,
        require_modern_baseline_commands_for_paper_gate=require_modern_baseline_commands_for_paper_gate,
        run_external_baseline_source_clone=run_external_baseline_source_clone,
        evidence_paths=evidence_paths,
    )
    print(json.dumps(external_baseline_preflight_decision, ensure_ascii=False, indent=2))
    validate_modern_baseline_commands_for_profile(external_baseline_preflight_decision)

    for env_name, command_template in modern_command_env.items():
        if command_template:
            os.environ[env_name] = command_template
    os.environ["SSTW_EXTERNAL_BASELINE_EVIDENCE_PATHS"] = os.pathsep.join(evidence_paths)

    return {
        "stage_name": "external_baseline_colab_preflight",
        "stage_execution_status": "completed",
        "stage_execution_kind": "python_helper",
        "modern_command_config_summary": modern_command_config_summary,
        "bridge_preflight_decision": bridge_preflight_decision,
        "external_baseline_preflight_decision": external_baseline_preflight_decision,
        "modern_command_env_names": sorted(modern_command_env),
    }


def build_configured_colab_stage_command(
    stage_name: str,
    layout: dict[str, str],
    *,
    workflow_profile: str,
    runtime_options: Mapping[str, Any] | None = None,
) -> list[str]:
    """根据统一 stage plan 构造单个 Colab stage 命令。

    该函数是 Notebook 瘦入口的核心。Notebook 不再维护 `stage_name -> command`
    字典, 只把 `workflow_profile` 交给配置和本 helper。新增或删除阶段时应改
    `configs/paper_workflow/generative_video_notebook_workflows.json` 与本函数,
    而不是改 Notebook cell。
    """

    options = dict(runtime_options or {})
    resolved = resolve_notebook_workflow_profile(workflow_profile, allow_disabled=True)
    runtime_profile = str(options.get("profile") or resolved.get("runtime_profile") or workflow_profile)

    simple_builders = {
        "prepare_prompt_suite": build_prompt_suite_command,
        "motion_threshold_calibration": build_motion_threshold_calibration_command,
        "mechanism_postprocess": build_mechanism_postprocess_command,
        "protocol_evaluation_matrix_postprocess": build_protocol_evaluation_matrix_postprocess_command,
        "runtime_attack": build_runtime_attack_command,
        "runtime_detection": build_runtime_detection_command,
        "external_baseline_official_result_bundle_preflight": build_external_baseline_official_result_bundle_preflight_command,
        "external_baseline_comparison": build_external_baseline_comparison_command,
        "external_baseline_self_containment_decision": build_external_baseline_self_containment_decision_command,
        "motion_consistency_exclusion_report": build_motion_consistency_exclusion_report_command,
        "validation_internal_ablation": build_validation_internal_ablation_command,
        "adaptive_attack_proxy": build_adaptive_attack_command,
        "replay_and_sketch_gate": build_replay_and_sketch_gate_command,
        "claim3_downgrade_gate": build_claim3_downgrade_command,
        "statistical_confidence_interval": build_statistical_confidence_interval_command,
        "low_fpr_formal_statistics": build_low_fpr_formal_statistics_command,
        "sstw_measured_formal_result": build_sstw_measured_formal_result_command,
        "fair_detection_calibration": build_fair_detection_calibration_command,
        "formal_method_baseline_comparison": build_formal_method_baseline_comparison_command,
        "formal_baseline_difference_interval": build_formal_baseline_difference_interval_command,
        "validation_scale_formal_internal_ablation": build_validation_scale_formal_internal_ablation_command,
        "data_split_and_leakage_guard": build_data_split_and_leakage_guard_command,
        "pilot_paper_gate": build_pilot_paper_gate_command,
        "pilot_paper_to_full_paper_transition_decision": build_pilot_paper_to_full_paper_transition_decision_command,
        "validation_artifact_rebuild_dry_run": build_validation_artifact_rebuild_dry_run_command,
        "validation_scale_gate": build_validation_scale_gate_command,
        "validation_scale_to_pilot_paper_transition_decision": build_validation_scale_to_pilot_paper_transition_decision_command,
        "validation_scale_gate_figure_builder": build_validation_scale_gate_figure_builder_command,
        "validation_scale_package_manifest_builder": build_validation_scale_package_manifest_builder_command,
    }

    if stage_name == "wan21_runtime_generation":
        return build_colab_runtime_command(
            layout,
            runtime_profile,
            str(options.get("model_id") or "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"),
            str(options.get("cross_model_id") or ""),
        )
    if stage_name == "formal_metric_scoring":
        return build_formal_metric_command(
            layout,
            semantic_model_id=str(options.get("semantic_model_id") or "openai/clip-vit-base-patch32"),
            semantic_frame_limit=int(options.get("semantic_frame_limit") or 8),
            disable_semantic_metric=bool(options.get("disable_semantic_metric", False)),
        )
    if stage_name == "external_baseline_source_intake":
        return build_external_baseline_source_intake_command(
            layout,
            execute_clone=_env_flag("SSTW_RUN_EXTERNAL_BASELINE_SOURCE_CLONE", True),
        )
    if stage_name == "external_baseline_official_resource_bootstrap":
        return build_external_baseline_official_resource_bootstrap_command(
            layout,
            allow_network=_env_flag("SSTW_EXTERNAL_BASELINE_ALLOW_NETWORK_BOOTSTRAP", True),
        )
    if stage_name == "external_baseline_official_bundle_generation":
        return build_external_baseline_official_bundle_generation_command(
            layout,
            generate_auto_supported=_env_flag("SSTW_EXTERNAL_BASELINE_GENERATE_AUTO_SUPPORTED_BUNDLE", True),
        )
    if stage_name == "drive_packaging":
        return build_drive_packaging_command(
            layout,
            include_videos=bool(options.get("include_videos", True)),
        )
    if stage_name in simple_builders:
        return simple_builders[stage_name](layout)
    raise KeyError(f"未登记的 Colab workflow stage: {stage_name}")


def run_configured_colab_stage_plan(
    layout: dict[str, str],
    *,
    workflow_profile: str,
    notebook_role: str,
    runtime_options: Mapping[str, Any] | None = None,
    include_videos: bool = True,
) -> dict[str, Any]:
    """按统一配置执行当前 Notebook role 的完整 stage plan。

    Notebook 只应调用该函数作为 Colab 入口。阶段顺序、是否启用、命令构造、
    governed records / tables / figures / reports / manifests 的产出逻辑均由
    repository 配置和 Python 模块维护。
    """

    resolved = resolve_notebook_workflow_profile(workflow_profile, notebook_role)
    stage_plan = build_workflow_stage_plan(workflow_profile, notebook_role)
    options = {**dict(runtime_options or {}), "include_videos": include_videos}
    stage_results: list[dict[str, Any]] = []
    stage_package: dict[str, Any] = {}

    for stage_name in stage_plan:
        if stage_name == "external_baseline_colab_preflight":
            stage_results.append(
                write_external_baseline_colab_preflight_for_profile(
                    layout,
                    workflow_profile=workflow_profile,
                )
            )
            continue

        if stage_name == "motion_threshold_reuse_check":
            if resolved.get("workflow_profile") == "motion_calibration":
                stage_results.append({
                    "stage_name": stage_name,
                    "stage_execution_status": "skipped",
                    "stage_execution_kind": "python_helper",
                    "skip_reason": "motion_calibration_profile_owns_threshold_calibration",
                })
            else:
                payload = write_motion_threshold_reuse_artifact_for_profile(layout, workflow_profile)
                print(json.dumps(payload, ensure_ascii=False, indent=2))
                stage_results.append({
                    "stage_name": stage_name,
                    "stage_execution_status": "completed",
                    "stage_execution_kind": "python_helper",
                    "motion_threshold_reuse": payload,
                })
            continue

        if stage_name == "motion_threshold_calibration_or_reuse_check":
            if resolved.get("workflow_profile") == "motion_calibration":
                stage_results.append(
                    _run_stage_command_or_raise(
                        "motion_threshold_calibration",
                        build_motion_threshold_calibration_command(layout),
                    )
                )
            else:
                payload = validate_motion_threshold_ready_for_profile(layout, workflow_profile)
                print(json.dumps(payload, ensure_ascii=False, indent=2))
                stage_results.append({
                    "stage_name": stage_name,
                    "stage_execution_status": "completed",
                    "stage_execution_kind": "python_helper",
                    "motion_threshold_reuse": payload,
                })
            continue

        if stage_name == "quick_tests_and_harness":
            stage_results.append(
                _run_stage_command_or_raise(
                    "quick_tests",
                    [sys.executable, "-m", "pytest", "-q"],
                )
            )
            stage_results.append(
                _run_stage_command_or_raise(
                    "harness_audits",
                    [sys.executable, "tools/harness/run_all_audits.py"],
                )
            )
            continue

        command = build_configured_colab_stage_command(
            stage_name,
            layout,
            workflow_profile=workflow_profile,
            runtime_options=options,
        )
        stage_results.append(_run_stage_command_or_raise(stage_name, command))

        if stage_name == "drive_packaging":
            stage_package = publish_colab_stage_package(
                layout,
                notebook_role=notebook_role,
                include_videos=include_videos,
            )
            print(json.dumps(stage_package, ensure_ascii=False, indent=2))

    stage_package_dir_path = Path(layout["stage_package_dir"])
    package_path_checks: dict[str, Any] = {}
    for key in ("drive_stage_package_zip", "stage_package_manifest_path"):
        path_text = str(stage_package.get(key, ""))
        path = Path(path_text) if path_text else None
        package_path_checks[key] = {
            "path": path_text,
            "exists": bool(path and path.exists()),
        }

    return {
        "workflow_profile": resolved,
        "notebook_role": notebook_role,
        "enabled_stage_plan": stage_plan,
        "stage_results": stage_results,
        "stage_package": stage_package,
        "stage_package_dir": str(stage_package_dir_path),
        "package_path_checks": package_path_checks,
    }
