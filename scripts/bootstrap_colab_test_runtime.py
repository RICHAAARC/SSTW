"""在不替换 Colab torch/NumPy 的前提下补齐非正式测试依赖。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REQUIREMENTS_PATH = (
    REPO_ROOT / "requirements" / "colab_test_runtime_compatibility.txt"
)
PROTECTED_DISTRIBUTIONS = ("torch", "numpy")
OBSERVED_DISTRIBUTIONS = (
    "torch",
    "torchvision",
    "numpy",
    "accelerate",
    "diffusers",
    "huggingface-hub",
    "imageio",
    "imageio-ffmpeg",
    "pillow",
    "protobuf",
    "safetensors",
    "sentencepiece",
    "transformers",
)
RUNTIME_PROBE_SOURCE = """
import accelerate
import diffusers
import huggingface_hub
import imageio
import imageio_ffmpeg
import numpy
from PIL import Image
import safetensors
import sentencepiece
import torch
import transformers
import google.protobuf
from diffusers import (
    AutoencoderKLWan,
    LTXPipeline,
    UniPCMultistepScheduler,
    WanPipeline,
)
"""


def _distribution_versions(
    distribution_names: Sequence[str] = OBSERVED_DISTRIBUTIONS,
) -> dict[str, str | None]:
    """读取已安装版本，不导入可能带二进制状态的运行库。"""

    versions: dict[str, str | None] = {}
    for name in distribution_names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _probe_runtime() -> dict[str, Any]:
    """在独立子进程探测关键 import 与 Wan/LTX pipeline 能力。"""

    completed = subprocess.run(
        [sys.executable, "-c", RUNTIME_PROBE_SOURCE],
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "ready": completed.returncode == 0,
        "return_code": completed.returncode,
        "stderr_tail": completed.stderr.strip()[-4000:],
    }


def _install_compatibility_requirements(
    requirements_path: Path,
    protected_versions: Mapping[str, str],
) -> dict[str, Any]:
    """补齐依赖，同时用 constraints 禁止替换 Colab 核心计算栈。"""

    constraint_lines = [
        f"{name}=={version}"
        for name, version in protected_versions.items()
    ]
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".txt",
    ) as constraint_file:
        constraint_file.write("\n".join(constraint_lines) + "\n")
        constraint_file.flush()
        command = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--requirement",
            str(requirements_path),
            "--constraint",
            constraint_file.name,
            "--upgrade-strategy",
            "only-if-needed",
        ]
        completed = subprocess.run(
            command,
            text=True,
            capture_output=False,
            check=False,
        )
    return {
        "return_code": completed.returncode,
        "protected_constraints": constraint_lines,
    }


def bootstrap_colab_test_runtime(
    *,
    requirements_path: str | Path = DEFAULT_REQUIREMENTS_PATH,
    version_reader: Callable[[], dict[str, str | None]] = _distribution_versions,
    probe_runner: Callable[[], dict[str, Any]] = _probe_runtime,
    install_runner: Callable[
        [Path, Mapping[str, str]], dict[str, Any]
    ] = _install_compatibility_requirements,
) -> dict[str, Any]:
    """使用 Colab 原生核心环境，并仅在能力不足时安装兼容依赖。"""

    resolved_requirements = Path(requirements_path).expanduser().resolve()
    if not resolved_requirements.is_file():
        raise FileNotFoundError(
            f"缺少 colab_test compatibility requirements: {resolved_requirements}"
        )
    before_versions = version_reader()
    missing_protected = [
        name
        for name in PROTECTED_DISTRIBUTIONS
        if not before_versions.get(name)
    ]
    if missing_protected:
        raise RuntimeError(
            "Colab 原生核心计算栈不完整，拒绝由轻量 bootstrap 安装: "
            + ", ".join(missing_protected)
        )

    initial_probe = probe_runner()
    install_result: dict[str, Any] | None = None
    if not initial_probe.get("ready"):
        protected_versions = {
            name: str(before_versions[name])
            for name in PROTECTED_DISTRIBUTIONS
        }
        install_result = install_runner(
            resolved_requirements,
            protected_versions,
        )
        if install_result.get("return_code") != 0:
            raise RuntimeError(
                "colab_test compatibility dependency installation failed"
            )

    final_probe = probe_runner()
    after_versions = version_reader()
    changed_protected = [
        name
        for name in PROTECTED_DISTRIBUTIONS
        if before_versions.get(name) != after_versions.get(name)
    ]
    if changed_protected:
        raise RuntimeError(
            "轻量 bootstrap 意外改变 Colab 核心计算栈: "
            + ", ".join(changed_protected)
        )
    if not final_probe.get("ready"):
        raise RuntimeError(
            "Colab 当前环境仍不满足 SSTW colab_test runtime imports: "
            + str(final_probe.get("stderr_tail") or "unknown import failure")
        )

    return {
        "manifest_kind": "sstw_colab_test_runtime_compatibility_decision",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_compatibility_decision": "PASS",
        "runtime_source": (
            "native_colab_core_with_compatibility_overlay"
            if install_result is not None
            else "native_colab_environment"
        ),
        "compatibility_install_executed": install_result is not None,
        "formal_runtime_lock_checked": False,
        "protected_core_distributions": list(PROTECTED_DISTRIBUTIONS),
        "protected_core_versions": {
            name: after_versions[name] for name in PROTECTED_DISTRIBUTIONS
        },
        "observed_distribution_versions": after_versions,
        "initial_probe": initial_probe,
        "final_probe": final_probe,
        "claim_support_status": (
            "colab_test_runtime_compatibility_only_not_claim_evidence"
        ),
    }


def _write_decision(path: str | Path, decision: Mapping[str, Any]) -> None:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="保留 Colab 原生 torch/NumPy，仅补齐 colab_test 兼容依赖"
    )
    parser.add_argument(
        "--requirements-path",
        default=str(DEFAULT_REQUIREMENTS_PATH),
    )
    parser.add_argument("--decision-output", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        decision = bootstrap_colab_test_runtime(
            requirements_path=args.requirements_path,
        )
    except Exception as exc:
        failure = {
            "manifest_kind": "sstw_colab_test_runtime_compatibility_decision",
            "runtime_compatibility_decision": "FAIL",
            "failure_reason": str(exc),
            "formal_runtime_lock_checked": False,
            "claim_support_status": (
                "colab_test_runtime_compatibility_only_not_claim_evidence"
            ),
        }
        if args.decision_output:
            _write_decision(args.decision_output, failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
    if args.decision_output:
        _write_decision(args.decision_output, decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
