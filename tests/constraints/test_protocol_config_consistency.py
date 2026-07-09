"""约束 paper profile 协议配置中的 baseline 数量语义保持一致。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


PAPER_PROFILE_PROTOCOL_CONFIGS = (
    Path("configs/protocol/validation_scale_generative_probe.json"),
    Path("configs/protocol/pilot_paper_generative_probe.json"),
    Path("configs/protocol/full_paper_generative_probe.json"),
)


@pytest.mark.constraint
def test_external_baseline_minimum_count_matches_required_modern_baselines() -> None:
    """主实验 baseline 数量不能保留旧配置导致 validation_scale 被错误阻断。"""

    mismatches: list[dict[str, object]] = []
    for config_path in PAPER_PROFILE_PROTOCOL_CONFIGS:
        config = json.loads(config_path.read_text(encoding="utf-8-sig"))
        required_baselines = [
            str(name)
            for name in config.get("required_modern_external_baseline_adapter_names", [])
            if str(name)
        ]
        minimum_count = int(config.get("minimum_external_baseline_measured_adapter_count", -1))
        if minimum_count != len(required_baselines):
            mismatches.append({
                "config_path": config_path.as_posix(),
                "minimum_external_baseline_measured_adapter_count": minimum_count,
                "required_modern_external_baseline_adapter_count": len(required_baselines),
            })

    assert mismatches == []
