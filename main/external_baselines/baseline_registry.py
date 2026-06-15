"""读取外部 baseline 配置并生成可审计状态记录。"""

from __future__ import annotations

from pathlib import Path
import json

from main.external_baselines.explicit_dtw_temporal_alignment import adapter_status as dtw_adapter_status
from main.external_baselines.frame_matching_temporal_registration import adapter_status as frame_matching_adapter_status


ADAPTER_STATUS_BUILDERS = {
    "explicit_dtw_temporal_alignment": dtw_adapter_status,
    "explicit_frame_matching_temporal_registration": frame_matching_adapter_status,
}


def load_external_baselines(path: str | Path) -> list[dict]:
    """读取外部 baseline 配置。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(data.get("baselines", []))


def build_external_baseline_records(path: str | Path) -> list[dict]:
    """生成外部 baseline 的状态 records。

    此处只记录 baseline 是否具备本地可运行适配入口, 不把该状态自动升级为论文正向 claim。
    真正用于 claim 之前, 仍需要由正式 records、tables 和 reports 重建出完整对比证据。
    """
    records = []
    for item in load_external_baselines(path):
        status_builder = ADAPTER_STATUS_BUILDERS.get(item["external_baseline_name"])
        adapter_record = status_builder() if status_builder else {}
        record = {**item, **adapter_record}
        if record.get("external_baseline_runnable_status") != "runnable":
            record["external_baseline_result_used_for_claim"] = False
        records.append({
            **record,
        })
    return records
