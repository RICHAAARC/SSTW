"""为旧调用方保留的 pilot_paper 公共门禁兼容入口。

pilot_paper、probe_paper 与 full_paper 的主张检查必须由同一个参数化实现完成。
该模块不再保存第二套门禁逻辑，只把 pilot 配置传给 `paper_profile_gate`。这样可
避免旧 pilot 专属条件继续漂移，同时保留服务器脚本可能依赖的公开函数名称。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluation.attacks.video_runtime_attack_protocol import (
    load_protocol_config_with_shared_attack_protocol,
)
from experiments.generative_video_model_probe.paper_profile_gate import (
    build_paper_profile_gate_audit,
    write_paper_profile_gate_audit,
)


DEFAULT_PILOT_PAPER_CONFIG = "configs/protocol/pilot_paper_generative_probe.json"


def _validate_pilot_profile_config(config_path: str | Path) -> None:
    """阻止兼容入口被用于其他 profile 或绕过公共契约。"""

    config = load_protocol_config_with_shared_attack_protocol(config_path)
    if config.get("paper_result_level") != "pilot_paper":
        raise ValueError("pilot_paper 兼容入口只接受 pilot_paper protocol config")
    if config.get("paper_profile_common_contract_status") != "matched":
        raise ValueError("pilot_paper protocol config 未通过公共机制契约校验")


def build_pilot_paper_gate_audit(
    run_root: str | Path,
    config_path: str | Path = DEFAULT_PILOT_PAPER_CONFIG,
) -> dict[str, Any]:
    """调用公共参数化 gate 构建 pilot_paper 审计, 不增加专属主张条件。"""

    _validate_pilot_profile_config(config_path)
    return build_paper_profile_gate_audit(run_root, config_path)


def write_pilot_paper_gate_audit(
    run_root: str | Path,
    config_path: str | Path = DEFAULT_PILOT_PAPER_CONFIG,
) -> dict[str, Any]:
    """调用公共 writer 同时写出公共与 pilot_paper 兼容命名产物。"""

    _validate_pilot_profile_config(config_path)
    return write_paper_profile_gate_audit(run_root, config_path)


def main() -> None:
    """提供可脱离 Notebook 运行的兼容 CLI。"""

    parser = argparse.ArgumentParser(
        description="通过公共参数化门禁审计 pilot_paper 结果。"
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--config-path", default=DEFAULT_PILOT_PAPER_CONFIG)
    parser.add_argument("--write-outputs", action="store_true")
    args = parser.parse_args()
    payload = (
        write_pilot_paper_gate_audit(args.run_root, args.config_path)
        if args.write_outputs
        else build_pilot_paper_gate_audit(args.run_root, args.config_path)
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
