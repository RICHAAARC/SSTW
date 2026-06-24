"""外部 baseline adapter 注册表。"""

from __future__ import annotations

from types import ModuleType
from typing import Any

from external_baseline.primary.explicit_dtw_temporal_alignment.adapter import run_sstw_eval as explicit_dtw_temporal_alignment
from external_baseline.primary.explicit_frame_matching_temporal_registration.adapter import run_sstw_eval as explicit_frame_matching_temporal_registration
from external_baseline.primary.sigmark.adapter import run_sstw_eval as sigmark
from external_baseline.primary.spdmark.adapter import run_sstw_eval as spdmark
from external_baseline.primary.videomark.adapter import run_sstw_eval as videomark
from external_baseline.primary.videoseal.adapter import run_sstw_eval as videoseal
from external_baseline.primary.vidsig.adapter import run_sstw_eval as vidsig
from external_baseline.primary.videoshield.adapter import run_sstw_eval as videoshield


ADAPTER_MODULES: dict[str, ModuleType] = {
    "explicit_dtw_temporal_alignment": explicit_dtw_temporal_alignment,
    "explicit_frame_matching_temporal_registration": explicit_frame_matching_temporal_registration,
    "videoshield": videoshield,
    "sigmark": sigmark,
    "spdmark": spdmark,
    "videomark": videomark,
    "vidsig": vidsig,
    "videoseal": videoseal,
}


def get_adapter(external_baseline_name: str) -> ModuleType | None:
    """按配置中的 baseline name 返回 adapter 模块。"""
    return ADAPTER_MODULES.get(external_baseline_name)


def adapter_status(external_baseline_name: str) -> dict[str, Any]:
    """返回单个 adapter 的受治理状态字段。"""
    adapter = get_adapter(external_baseline_name)
    if adapter is None:
        return {}
    return adapter.adapter_status()


def list_adapter_statuses() -> dict[str, dict[str, Any]]:
    """返回所有已注册 adapter 的状态摘要。"""
    return {name: module.adapter_status() for name, module in ADAPTER_MODULES.items()}
