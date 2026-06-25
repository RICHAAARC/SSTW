"""B5 生成式视频 Colab Notebook 的路径和命令编排工具。"""

from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
from typing import Any, Mapping

from paper_workflow.notebook_utils.streaming_command import run_streaming_command


DEFAULT_DRIVE_PROJECT_ROOT = "/content/drive/MyDrive/SSTW"
DEFAULT_VALIDATION_SCALE_CONFIG = "configs/protocol/validation_scale_generative_probe.json"
DEFAULT_PILOT_PAPER_CONFIG = "configs/protocol/pilot_paper_generative_probe.json"
DEFAULT_NOTEBOOK_WORKFLOW_CONFIG = "configs/paper_workflow/generative_video_notebook_workflows.json"
DEFAULT_MODERN_BASELINE_COLAB_COMMAND_CONFIG = "configs/external_baselines/modern_baseline_colab_commands.json"
DEFAULT_NOTEBOOK_ROLE = "generative_video_runtime"
PAPER_GATE_PROFILES = {"validation_scale", "pilot_paper"}
EXTERNAL_BASELINE_COLAB_PREFLIGHT_DECISION = "artifacts/external_baseline_colab_preflight_decision.json"
EXTERNAL_BASELINE_COMMAND_TEMPLATE_SUMMARY = "artifacts/external_baseline_command_template_summary.json"
EXTERNAL_BASELINE_OFFICIAL_BRIDGE_PREFLIGHT_DECISION = "artifacts/external_baseline_official_bridge_preflight_decision.json"


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
    profile_config = dict(profiles[canonical_profile])
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
    return {
        "drive_project_root": root.as_posix(),
        "drive_dataset_root": _join_drive_path(root, dataset_root_relative),
        "drive_run_root": _join_drive_path(root, str(workflow["drive_run_root_relative"])),
        "drive_package_dir": _join_drive_path(root, str(workflow["drive_package_dir_relative"])),
        "drive_log_dir": _join_drive_path(root, str(workflow["drive_log_dir_relative"])),
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
        "notebook_role": str(workflow.get("notebook_role") or ""),
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


def _write_json(path: str | Path, payload: Mapping[str, object]) -> None:
    """写出 Colab preflight artifact, 使冷启动失败也能在 Google Drive 中审计原因。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _config_path_for_profile(profile: str) -> str:
    """根据运行 profile 选择现代 baseline 要求来源。"""
    try:
        return protocol_config_path_for_profile(profile)
    except Exception:
        pass
    if profile == "pilot_paper":
        return DEFAULT_PILOT_PAPER_CONFIG
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
        "command_templates_auto_applied": False,
        "required_modern_external_baseline_adapter_names": required_baseline_ids,
        "required_modern_external_baseline_adapter_count": len(required_baseline_ids),
        "configured_template_row_count": len(rows) - len(missing_template_ids),
        "missing_template_row_count": len(missing_template_ids),
        "missing_template_baseline_ids": missing_template_ids,
        "accepted_output_score_fields": config.get("accepted_output_score_fields", []),
        "required_command_format_tokens": config.get("required_command_format_tokens", []),
        "optional_command_format_tokens": config.get("optional_command_format_tokens", []),
        "colab_user_action_required": (
            "安装或克隆官方 baseline 源码, 编写能输出 JSON score 的 wrapper, "
            "再把 command 写入对应 SSTW_<BASELINE>_EVAL_COMMAND。"
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
) -> dict[str, object]:
    """写出 command 配置摘要, 便于 Colab 失败后在 Google Drive 中审计。"""
    summary = build_modern_baseline_colab_command_config_summary(
        layout,
        profile=profile,
        protocol_config_path=protocol_config_path,
        command_config_path=command_config_path,
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
    required_env_vars = [item["official_baseline_command_env_var"] for item in requirements]
    configured_env_vars = [env_var for env_var in required_env_vars if str(env_source.get(env_var) or "").strip()]
    missing_env_vars = [env_var for env_var in required_env_vars if env_var not in configured_env_vars]
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


def validate_modern_baseline_official_bridge_for_profile(preflight_decision: Mapping[str, object]) -> None:
    """在 bridge 模式缺少官方原生命令时提前阻断。"""
    if preflight_decision.get("external_baseline_official_bridge_preflight_decision") == "FAIL":
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
    作为 key。这样 Notebook 只需要维护用户可编辑的短变量, 具体 hard gate 清单
    始终来自 protocol config。
    """
    env: dict[str, str] = {}
    for requirement in required_modern_external_baseline_command_requirements(profile, config_path):
        baseline_id = requirement["baseline_id"]
        env_var = requirement["external_baseline_command_env_var"]
        env[env_var] = str(command_templates.get(baseline_id) or command_templates.get(env_var) or "")
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


def validate_modern_baseline_commands_for_profile(preflight_decision: Mapping[str, object]) -> None:
    """在 paper gate profile 缺少现代 baseline command 时抛出明确错误。"""
    if preflight_decision.get("external_baseline_colab_preflight_decision") == "FAIL":
        missing = preflight_decision.get("external_baseline_colab_preflight_missing_env_vars")
        summary_path = preflight_decision.get("external_baseline_command_template_summary_path")
        raise RuntimeError(
            "当前 workflow profile 是 paper gate 或 paper gate 前最后门禁, 必须先在 Colab 配置现代视频水印 baseline command。"
            f" 缺失: {missing}。可先查看命令配置摘要: {summary_path}"
        )


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
    """构造 B5 Colab 机制后处理命令, 从已有 governed records 重建后处理 artifacts。"""
    return [
        sys.executable,
        "-m",
        "experiments.generative_video_model_probe.postprocess_runner",
        "--run-root",
        layout["drive_run_root"],
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

def build_validation_internal_ablation_command(layout: dict[str, str]) -> list[str]:
    """构造 validation-scale 内部消融矩阵后处理命令。"""
    return [
        sys.executable,
        "-m",
        "experiments.generative_video_model_probe.validation_internal_ablation",
        "--run-root",
        layout["drive_run_root"],
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
    """构造 validation-scale 统计置信区间报告命令。"""
    return [
        sys.executable,
        "-m",
        "experiments.generative_video_model_probe.statistical_confidence_interval",
        "--run-root",
        layout["drive_run_root"],
    ]


def build_pilot_paper_gate_command(layout: dict[str, str]) -> list[str]:
    """构建 pilot_paper FPR=0.01 gate 命令, 只汇总已落盘 records 并写出冻结阈值。"""
    return [
        sys.executable,
        "-m",
        "experiments.generative_video_model_probe.pilot_paper_gate",
        "--run-root",
        layout["drive_run_root"],
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
    """构造 validation-scale gate 命令, 防止从 small-scale pilot 直接跳到 pilot_paper 或 full_paper。"""
    return [
        sys.executable,
        "-m",
        "experiments.generative_video_model_probe.validation_scale_gate",
        "--run-root",
        layout["drive_run_root"],
        "--write-outputs",
    ]


def build_drive_packaging_command(layout: dict[str, str], include_videos: bool = True) -> list[str]:
    """构造 Google Drive 打包命令。"""
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
