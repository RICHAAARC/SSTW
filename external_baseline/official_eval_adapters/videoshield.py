"""VideoShield 官方源码的 SSTW 评测 wrapper。

VideoShield 的官方脚本以生成时 latent watermark、反演和 `wm_info.bin` 为中心。
因此默认 wrapper 只能读取用户提供的官方结果 JSON, 或通过
`SSTW_VIDEOSHIELD_NATIVE_EVAL_COMMAND` 调用用户配置的官方完整评测命令。
缺少这些官方产物时必须失败。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from external_baseline.official_eval_adapters.common import (
    extract_score,
    raise_missing_official_artifacts,
    read_json,
    run_adapter_main,
)


BASELINE_ID = "videoshield"
REQUIRED_SOURCE_FILES = (
    "watermark.py",
    "watermark_embedding_and_extraction.py",
    "temporal_tamper_localization.py",
)


def _run_default(args: argparse.Namespace, source_dir: Path, output_json_path: Path) -> dict[str, Any]:
    """读取 VideoShield 官方评测产物 JSON。

    该分支不自行实现 latent 反演, 因为那会把官方生成/反演协议隐式改写到 SSTW
    仓库中。若需要自动运行, 应配置 `SSTW_VIDEOSHIELD_NATIVE_EVAL_COMMAND` 指向
    用户维护的官方完整命令。
    """
    result_json = os.environ.get("SSTW_VIDEOSHIELD_RESULT_JSON", "").strip()
    if not result_json:
        raise_missing_official_artifacts(
            BASELINE_ID,
            "missing SSTW_VIDEOSHIELD_RESULT_JSON or SSTW_VIDEOSHIELD_NATIVE_EVAL_COMMAND",
        )
    payload = read_json(result_json)
    score = extract_score(payload)
    return {
        **payload,
        "external_baseline_score": round(float(score), 6),
        "official_adapter_status": "measured_from_videoshield_official_result_json",
        "official_adapter_baseline_id": BASELINE_ID,
        "official_source_dir": str(source_dir),
        "official_result_json_path": str(result_json),
        "official_output_json_path": str(output_json_path),
    }


def main() -> None:
    """CLI 入口。"""
    run_adapter_main(
        baseline_id=BASELINE_ID,
        description="VideoShield 官方评测产物 wrapper。",
        required_source_files=REQUIRED_SOURCE_FILES,
        default_runner=_run_default,
    )


if __name__ == "__main__":
    main()

