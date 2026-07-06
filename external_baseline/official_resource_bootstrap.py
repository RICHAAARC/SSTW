"""现代外部 baseline 官方资源自动准备器。

该模块在 Colab 冷启动时执行“能自动补齐的就自动补齐”。它不同于 preflight:

1. preflight 只检查资源是否存在。
2. bootstrap 会尝试安装下载工具、下载公开 checkpoint、创建统一资源目录。

无法自动补齐的资源会被写成明确的 `manual_official_resource_required`。这不是放弃,
而是防止把没有公开权重或超出 Colab L4 的官方训练流程伪装成可运行 baseline。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping
import urllib.error
import urllib.request
import zipfile


DEFAULT_RESOURCE_REQUIREMENTS = "configs/external_baselines/official_resource_requirements.json"
VIDSIG_GOOGLE_DRIVE_FILE_ID = "1XFyzeX6T0iHgcxSN_DxvjLFy1EXZye-Q"
DEFAULT_PRIMARY_SOURCE_ROOT = "external_baseline/primary"
VIDEOSEAL_COLAB_LIGHTWEIGHT_DEPENDENCIES = (
    "av",
    "ffmpeg-python",
    "imageio",
    "imageio-ffmpeg",
    "opencv-python",
    "omegaconf",
    "einops",
    "timm==0.9.16",
    "decord",
    "PyWavelets",
    "pytorch-msssim",
    "safetensors",
    "scikit-image",
    "scipy",
    "transformers",
    "pandas",
    "pycocotools",
    "lpips",
    "calflops",
)
VIDSIG_COLAB_LIGHTWEIGHT_DEPENDENCIES = (
    "numpy",
    "imageio",
    "imageio-ffmpeg",
    "opencv-python",
    "augly==1.0.0",
    "omegaconf",
    "pandas",
    "Pillow",
    "tqdm",
    "diffusers",
    "transformers",
    "accelerate",
    "safetensors",
)
VIDEOSHIELD_COLAB_LIGHTWEIGHT_DEPENDENCIES = (
    "diffusers",
    "transformers",
    "accelerate",
    "numpy",
    "scipy",
    "opencv-python",
    "imageio",
    "imageio-ffmpeg",
    "pycryptodome",
    "tqdm",
)


def _read_json(path: str | Path) -> dict[str, Any]:
    """读取 JSON 对象。"""
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"json_payload_must_be_object:{path}")
    return payload


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """写出 JSON artifact。"""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _drive_project_root_from_run_root(run_root: str | Path) -> Path:
    """从 profile run_root 推导 Google Drive 项目根目录。"""
    root = Path(run_root)
    try:
        return root.parents[2]
    except IndexError:
        return root


def _run_command(command: list[str], *, timeout_sec: int = 1800) -> dict[str, Any]:
    """运行轻量 bootstrap 命令并返回摘要。"""
    completed = subprocess.run(command, text=True, capture_output=True, timeout=timeout_sec)
    return {
        "command": command,
        "return_code": completed.returncode,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def _run_command_inherit_env(command: list[str], *, timeout_sec: int = 1800) -> dict[str, Any]:
    """运行安装类命令并继承当前环境。

    该函数与 `_run_command` 一样只返回摘要。它用于 Colab 冷启动阶段安装公开依赖,
    不写入正式 records, 也不把安装成功解释为 baseline 已经测量。
    """
    completed = subprocess.run(command, text=True, capture_output=True, timeout=timeout_sec, env=dict(os.environ))
    return {
        "command": command,
        "return_code": completed.returncode,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def _ensure_gdown() -> dict[str, Any]:
    """确保当前环境有 gdown。"""
    if shutil.which("gdown"):
        return {"tool": "gdown", "status": "already_available"}
    result = _run_command([sys.executable, "-m", "pip", "install", "-q", "gdown"], timeout_sec=600)
    return {"tool": "gdown", "status": "installed" if result["return_code"] == 0 else "install_failed", "install_result": result}


def _pip_install_target(target: str, *, allow_network: bool, timeout_sec: int = 1800) -> dict[str, Any]:
    """安装公开 Python 包或本地官方源码包。

    `allow_network=False` 时只记录 planned 状态, 用于本地轻量测试。Colab 正式运行默认
    允许网络, 以便在前置检查失败后自动补齐可公开安装的依赖。
    """
    if not allow_network:
        return {"install_target": target, "install_status": "planned_network_disabled"}
    result = _run_command_inherit_env([sys.executable, "-m", "pip", "install", "-q", target], timeout_sec=timeout_sec)
    return {
        "install_target": target,
        "install_status": "installed" if result["return_code"] == 0 else "install_failed",
        "install_result": result,
    }


def _find_file(root: Path, names: set[str]) -> Path | None:
    """在资源目录中查找指定文件名。"""
    if not root.exists():
        return None
    for path in root.rglob("*"):
        if path.is_file() and path.name in names:
            return path
    return None


def _find_vidsig_vae_checkpoint(root: Path) -> Path | None:
    """优先定位 VidSig Text-to-Video ModelScope 官方 VAE checkpoint。"""

    preferred = root / "ckpts" / "vae" / "modelscope" / "checkpoint.pth"
    if preferred.exists():
        return preferred
    candidates = sorted(path for path in root.rglob("checkpoint.pth") if path.is_file()) if root.exists() else []
    for path in candidates:
        if "modelscope" in path.as_posix().lower():
            return path
    return candidates[0] if candidates else None


def _unpack_if_archive(path: Path, output_dir: Path) -> dict[str, Any]:
    """如果下载结果是 zip, 则解压到资源目录。"""
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            archive.extractall(output_dir)
        return {"archive_path": str(path), "unpack_status": "zip_unpacked", "unpack_dir": str(output_dir)}
    return {"archive_path": str(path), "unpack_status": "not_zip"}


def _download_url_if_missing(url: str, output_path: Path, *, allow_network: bool) -> dict[str, Any]:
    """下载公开 URL 资源, 已存在时直接复用。

    该函数只用于官方仓库 README 中明确公开的小型辅助资源。它不会下载大模型,
    也不会生成任何 baseline 分数。这样可以补齐 Colab 冷启动所需的公开文件,
    同时保持正式 `measured_formal` 结果必须来自后续官方运行链路。
    """

    if output_path.is_file():
        return {
            "download_status": "already_available",
            "url": url,
            "output_path": str(output_path),
            "output_size_bytes": output_path.stat().st_size,
        }
    if not allow_network:
        return {
            "download_status": "planned_network_disabled",
            "url": url,
            "output_path": str(output_path),
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, output_path)
    except (urllib.error.URLError, OSError) as exc:
        return {
            "download_status": "download_failed",
            "url": url,
            "output_path": str(output_path),
            "error": str(exc)[-1000:],
        }
    return {
        "download_status": "downloaded",
        "url": url,
        "output_path": str(output_path),
        "output_size_bytes": output_path.stat().st_size,
    }


def bootstrap_vidsig(resource_root: Path, *, allow_network: bool) -> dict[str, Any]:
    """下载或定位 VidSig 官方公开 checkpoint, 并补齐 Colab 轻量依赖。

    VidSig 官方 `requirements.txt` 会固定 torch / torchvision 版本。Colab 中这两个包
    与 CUDA 运行栈强绑定, 直接按官方 requirements 重装会引入新的不确定性。因此这里
    只安装 `src/attack.py` 和 `src.utils` 导入链需要的轻量依赖, 让正式 Notebook 能在
    保持预装 torch / torchvision 的前提下运行官方攻击检测脚本。
    """
    baseline_root = resource_root / "vidsig"
    baseline_root.mkdir(parents=True, exist_ok=True)
    install_results: list[dict[str, Any]] = []
    for dependency in VIDSIG_COLAB_LIGHTWEIGHT_DEPENDENCIES:
        install_results.append(_pip_install_target(dependency, allow_network=allow_network, timeout_sec=900))
    failed_dependencies = [item for item in install_results if item.get("install_status") == "install_failed"]
    decoder = _find_file(baseline_root, {"dec_48b_whit.torchscript.pt"})
    vae_checkpoint = _find_vidsig_vae_checkpoint(baseline_root)
    if decoder and vae_checkpoint:
        return {
            "baseline_id": "vidsig",
            "bootstrap_status": "ready" if not failed_dependencies else "manual_official_resource_required",
            "resource_mode": "existing_public_checkpoint",
            "SSTW_VIDSIG_MSG_DECODER_PATH": str(decoder),
            "SSTW_VIDSIG_VAE_CHECKPOINT_PATH": str(vae_checkpoint),
            "colab_torch_stack_policy": "preserve_preinstalled_torch_and_torchvision",
            "install_results": install_results,
            "missing_after_bootstrap": [] if not failed_dependencies else ["vidsig_lightweight_python_dependencies"],
        }
    if not allow_network:
        return {
            "baseline_id": "vidsig",
            "bootstrap_status": "manual_official_resource_required",
            "resource_mode": "network_disabled",
            "required_resource": "Google Drive checkpoint id 1XFyzeX6T0iHgcxSN_DxvjLFy1EXZye-Q",
            "colab_torch_stack_policy": "preserve_preinstalled_torch_and_torchvision",
            "install_results": install_results,
        }
    gdown_status = _ensure_gdown()
    archive_path = baseline_root / "vidsig_official_checkpoint_download"
    download = _run_command([
        sys.executable,
        "-m",
        "gdown",
        VIDSIG_GOOGLE_DRIVE_FILE_ID,
        "-O",
        str(archive_path),
    ], timeout_sec=3600)
    unpack = _unpack_if_archive(archive_path, baseline_root) if archive_path.exists() else {"unpack_status": "download_missing"}
    decoder = _find_file(baseline_root, {"dec_48b_whit.torchscript.pt"})
    vae_checkpoint = _find_vidsig_vae_checkpoint(baseline_root)
    ready = decoder is not None and vae_checkpoint is not None and not failed_dependencies
    missing_after_download = [] if decoder is not None and vae_checkpoint is not None else ["dec_48b_whit.torchscript.pt", "checkpoint.pth"]
    if failed_dependencies:
        missing_after_download.append("vidsig_lightweight_python_dependencies")
    return {
        "baseline_id": "vidsig",
        "bootstrap_status": "ready" if ready else "manual_official_resource_required",
        "resource_mode": "public_google_drive_checkpoint",
        "gdown_status": gdown_status,
        "download_result": download,
        "unpack_result": unpack,
        "SSTW_VIDSIG_MSG_DECODER_PATH": str(decoder) if decoder else "",
        "SSTW_VIDSIG_VAE_CHECKPOINT_PATH": str(vae_checkpoint) if vae_checkpoint else "",
        "colab_torch_stack_policy": "preserve_preinstalled_torch_and_torchvision",
        "install_results": install_results,
        "missing_after_download": missing_after_download,
    }


def bootstrap_videoseal(resource_root: Path, *, allow_network: bool, source_root: Path) -> dict[str, Any]:
    """准备 VideoSeal 官方 API 运行环境。

    VideoSeal 官方 `videoseal.load(...)` 会自动下载 checkpoint。因此这里不强制下载。
    为适配 Colab 预装 CUDA 栈, 这里只安装轻量依赖, 不 pip 安装 torch、torchvision
    或 VideoSeal 官方 git 包。官方源码由 source intake clone 后通过 `sys.path` 使用。
    """
    baseline_root = resource_root / "videoseal"
    baseline_root.mkdir(parents=True, exist_ok=True)
    source_dir = source_root / "videoseal" / "source"
    install_results: list[dict[str, Any]] = []
    for dependency in VIDEOSEAL_COLAB_LIGHTWEIGHT_DEPENDENCIES:
        install_results.append(_pip_install_target(dependency, allow_network=allow_network, timeout_sec=900))
    failed = [item for item in install_results if item.get("install_status") == "install_failed"]
    return {
        "baseline_id": "videoseal",
        "bootstrap_status": "ready" if not failed else "manual_official_resource_required",
        "resource_mode": "official_api_auto_download_on_first_use",
        "resource_dir": str(baseline_root),
        "official_source_dir": str(source_dir),
        "colab_torch_stack_policy": "preserve_preinstalled_torch_and_torchvision",
        "install_results": install_results,
        "missing_after_bootstrap": [] if not failed else ["videoseal_lightweight_python_dependencies"],
    }


def bootstrap_videoshield(resource_root: Path, *, allow_network: bool, source_root: Path) -> dict[str, Any]:
    """准备 VideoShield 官方 runtime 的公开 Python 依赖。

    VideoShield 正式结果由 `external_baseline.videoshield_official_runtime` 在项目内
    调用官方 watermark、diffusers ModelScope 生成和官方反演逻辑生成。这里仅安装
    可公开获得的轻量依赖并记录 runner 可尝试状态, 不生成任何 baseline 分数。
    """

    baseline_root = resource_root / "videoshield"
    baseline_root.mkdir(parents=True, exist_ok=True)
    source_dir = source_root / "videoshield" / "source"
    install_results: list[dict[str, Any]] = []
    for dependency in VIDEOSHIELD_COLAB_LIGHTWEIGHT_DEPENDENCIES:
        install_results.append(_pip_install_target(dependency, allow_network=allow_network, timeout_sec=1200))
    failed = [item for item in install_results if item.get("install_status") == "install_failed"]
    return {
        "baseline_id": "videoshield",
        "bootstrap_status": "ready" if not failed else "manual_official_resource_required",
        "resource_mode": "project_owned_official_runtime_downloads_public_model_on_first_use",
        "resource_dir": str(baseline_root),
        "official_source_dir": str(source_dir),
        "project_owned_runner_module": "external_baseline.videoshield_official_runtime",
        "colab_torch_stack_policy": "preserve_preinstalled_torch_and_torchvision",
        "install_results": install_results,
        "missing_after_bootstrap": [] if not failed else ["videoshield_lightweight_python_dependencies"],
    }


def manual_resource_row(baseline_id: str, reason: str, strict_gate_resolution: str) -> dict[str, Any]:
    """构造需要官方离线资源的 baseline 行。"""
    return {
        "baseline_id": baseline_id,
        "bootstrap_status": "manual_official_resource_required",
        "resource_mode": "not_publicly_auto_resolvable_in_colab_l4",
        "manual_reason": reason,
        "strict_gate_resolution": strict_gate_resolution,
    }


def bootstrap_official_resources(
    run_root: str | Path,
    *,
    resource_root: str | Path | None = None,
    allow_network: bool = True,
    resource_config_path: str | Path = DEFAULT_RESOURCE_REQUIREMENTS,
    source_root: str | Path = DEFAULT_PRIMARY_SOURCE_ROOT,
) -> dict[str, Any]:
    """自动补齐可公开获得的官方资源, 并写出缺口。"""
    drive_root = _drive_project_root_from_run_root(run_root)
    resolved_resource_root = Path(resource_root) if resource_root else drive_root / "resources" / "external_baseline"
    resolved_resource_root.mkdir(parents=True, exist_ok=True)
    config = _read_json(resource_config_path)
    rows = {str(row.get("baseline_id")): dict(row) for row in config.get("resource_rows", [])}
    resolved_source_root = Path(source_root)

    baseline_rows: list[dict[str, Any]] = []
    baseline_rows.append(bootstrap_videoseal(resolved_resource_root, allow_network=allow_network, source_root=resolved_source_root))
    baseline_rows.append(bootstrap_vidsig(resolved_resource_root, allow_network=allow_network))
    baseline_rows.append(bootstrap_videoshield(resolved_resource_root, allow_network=allow_network, source_root=resolved_source_root))
    ready = [row["baseline_id"] for row in baseline_rows if row.get("bootstrap_status") == "ready"]
    manual = [row["baseline_id"] for row in baseline_rows if row.get("bootstrap_status") != "ready"]
    env_updates = {
        "SSTW_EXTERNAL_BASELINE_RESOURCE_ROOT": str(resolved_resource_root),
    }
    for row in baseline_rows:
        for key, value in row.items():
            if key.startswith("SSTW_") and value:
                env_updates[key] = str(value)
    decision = {
        "artifact_name": "external_baseline_official_resource_bootstrap_decision.json",
        "manifest_kind": "external_baseline_official_resource_bootstrap",
        "run_root": str(run_root),
        "resource_root": str(resolved_resource_root),
        "source_root": str(resolved_source_root),
        "allow_network": bool(allow_network),
        "official_resource_bootstrap_decision": "PASS" if baseline_rows else "FAIL",
        "strict_gate_auto_resource_closure": not manual,
        "strict_gate_auto_resource_status": "all_resources_auto_ready" if not manual else "manual_official_resources_still_required",
        "ready_baselines": ready,
        "manual_official_resource_required_baselines": manual,
        "manual_official_resource_required_count": len(manual),
        "baseline_resource_rows": sorted(baseline_rows, key=lambda item: item["baseline_id"]),
        "environment_updates": env_updates,
        "claim_support_status": "official_resource_bootstrap_not_claim_evidence",
    }
    _write_json(Path(run_root) / "artifacts" / "external_baseline_official_resource_bootstrap_decision.json", decision)
    return decision


def main() -> None:
    """CLI 入口。"""
    parser = argparse.ArgumentParser(description="自动准备可公开获得的现代 baseline 官方资源。")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--resource-root", default="")
    parser.add_argument("--resource-config-path", default=DEFAULT_RESOURCE_REQUIREMENTS)
    parser.add_argument("--source-root", default=DEFAULT_PRIMARY_SOURCE_ROOT)
    parser.add_argument("--disable-network", action="store_true")
    args = parser.parse_args()
    payload = bootstrap_official_resources(
        args.run_root,
        resource_root=args.resource_root or None,
        allow_network=not args.disable_network,
        resource_config_path=args.resource_config_path,
        source_root=args.source_root,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
