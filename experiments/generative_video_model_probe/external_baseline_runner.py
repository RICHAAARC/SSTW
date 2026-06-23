"""构建 B5 外部 baseline 状态记录。"""

from __future__ import annotations

from main.external_baselines.baseline_registry import build_external_baseline_records
from main.protocol.flow_evidence_fields import with_flow_evidence_protocol_defaults_many


def run_external_baseline_status(config_path: str) -> list[dict]:
    """返回外部 baseline limitation records。

    该函数只记录外部 baseline 是否可运行, 不把 unavailable baseline 伪装成正式比较结果。
    同时补齐 Flow evidence 协议字段, 便于 package 审计时统一读取 records。
    """
    return with_flow_evidence_protocol_defaults_many(
        build_external_baseline_records(config_path),
        trajectory_source_level="not_applicable",
        claim_support_status="external_baseline_limitation_record",
    )
