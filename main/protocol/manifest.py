"""生成第一阶段运行 manifest。"""

from __future__ import annotations

from main.core.digest import build_stable_digest


def build_run_manifest(run_id: str, config: dict, output_paths: list[str]) -> dict:
    """记录第一阶段运行的输入配置、输出路径和重建命令。"""
    return {"run_id": run_id, "stage_id": "synthetic_state_protocol", "config_digest": build_stable_digest(config), "output_paths": output_paths, "rebuild_command": "python -m experiments.synthetic_state_inference.runner --output-root outputs/runs/synthetic_state_protocol", "code_version": "workspace"}
