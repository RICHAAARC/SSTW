"""VideoSeal 官方源码运行布局检查。

该模块只处理 VideoSeal 官方源码在 Colab / Notebook 中的路径解析问题。
它不生成水印分数, 不改写官方模型配置, 也不伪造任何 external baseline 结果。
"""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
from typing import Any, Iterator


VIDEOSEAL_REQUIRED_SOURCE_CONFIGS = ("configs/attenuation.yaml",)


def inspect_videoseal_official_runtime_layout(source_dir: str | Path) -> dict[str, Any]:
    """检查 VideoSeal 官方源码是否满足 `videoseal.load(...)` 的相对路径要求。

    通用工程写法是把第三方源码的运行前置条件显式登记为 audit payload。
    本项目特定问题在于: VideoSeal 官方 `resolve_config_path(...)` 会先从当前工作目录
    查找 `configs/attenuation.yaml`, 再从 `videoseal/` 包目录查找同名路径。因此在
    Colab 中只把源码加入 `sys.path` 不够, 还必须在加载模型时临时切换到官方源码根目录。
    """
    source_path = Path(source_dir)
    source_root_configs: list[dict[str, Any]] = []
    package_fallback_configs: list[dict[str, Any]] = []
    for relative_path in VIDEOSEAL_REQUIRED_SOURCE_CONFIGS:
        root_config_path = source_path / relative_path
        package_config_path = source_path / "videoseal" / relative_path
        source_root_configs.append({
            "config_relative_path": relative_path,
            "config_path": str(root_config_path),
            "config_file_exists": root_config_path.is_file(),
        })
        package_fallback_configs.append({
            "config_relative_path": f"videoseal/{relative_path}",
            "config_path": str(package_config_path),
            "config_file_exists": package_config_path.is_file(),
        })

    source_dir_exists = source_path.exists()
    source_root_ready = source_dir_exists and all(row["config_file_exists"] for row in source_root_configs)
    package_fallback_ready = source_dir_exists and all(row["config_file_exists"] for row in package_fallback_configs)
    if source_root_ready:
        layout_decision = "PASS"
        layout_status = "official_source_root_config_ready"
    elif package_fallback_ready:
        layout_decision = "PASS"
        layout_status = "package_fallback_config_ready"
    elif not source_dir_exists:
        layout_decision = "FAIL"
        layout_status = "official_source_dir_missing"
    else:
        layout_decision = "FAIL"
        layout_status = "official_required_config_missing"

    missing_required_config_paths = [
        row["config_path"]
        for row in source_root_configs
        if not row["config_file_exists"]
    ]
    if package_fallback_ready:
        missing_required_config_paths = []

    return {
        "baseline_id": "videoseal",
        "official_source_dir": str(source_path),
        "official_source_dir_exists": source_dir_exists,
        "layout_decision": layout_decision,
        "layout_status": layout_status,
        "required_working_directory": str(source_path),
        "runtime_cwd_policy": "temporarily_chdir_to_official_source_root_during_videoseal_load",
        "source_root_config_paths": source_root_configs,
        "package_fallback_config_paths": package_fallback_configs,
        "missing_required_config_paths": missing_required_config_paths,
        "claim_support_status": "runtime_layout_audit_only_not_claim_evidence",
    }


def ensure_videoseal_official_runtime_layout(source_dir: str | Path) -> dict[str, Any]:
    """返回 VideoSeal 运行布局审计, 若缺少官方配置则 fail closed。

    这里的 fail closed 是项目特定门禁策略: 缺少官方 `attenuation.yaml` 时必须显式阻断,
    不能用 SSTW 侧默认值或临时 YAML 代替官方配置。
    """
    audit = inspect_videoseal_official_runtime_layout(source_dir)
    if audit["layout_decision"] != "PASS":
        missing = ",".join(audit["missing_required_config_paths"])
        raise FileNotFoundError(
            "videoseal_official_config_missing:"
            f"source_dir={audit['official_source_dir']}:"
            f"layout_status={audit['layout_status']}:"
            f"missing={missing}"
        )
    return audit


@contextmanager
def videoseal_official_source_cwd(source_dir: str | Path) -> Iterator[None]:
    """在加载 VideoSeal 官方模型期间临时切换到官方源码根目录。

    该上下文只影响内部 `with` 块, 退出后恢复调用者原始工作目录。这样 Notebook 仍然可以
    以 SSTW 仓库根目录作为主工作目录, 同时满足 VideoSeal 官方配置解析规则。
    """
    previous_cwd = Path.cwd()
    os.chdir(Path(source_dir))
    try:
        yield
    finally:
        os.chdir(previous_cwd)
