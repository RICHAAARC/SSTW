"""根据 formal motion gate 结果筛选可支撑 motion claim 的生成样本。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


RecordKey = tuple[str, str, str, str]

BOUNDARY_MOTION_ROLES = {"negative_static", "ambiguous_low_motion"}


def record_identity_key(record: dict) -> RecordKey:
    """构造跨 generation、formal、attack 和 detection records 复用的样本键。

    该函数属于通用工程写法。它只使用受治理 records 中已经存在的稳定字段,
    避免依赖视频文件名或最终检测分数来决定样本是否可用于 claim。
    """
    return (
        str(record.get("generation_model_id") or ""),
        str(record.get("prompt_id") or ""),
        str(record.get("seed_id") or ""),
        str(record.get("trajectory_trace_id") or ""),
    )


def _motion_claim_role(record: dict) -> str:
    """从 record 中推断 motion claim 角色。

    项目特定写法是把 negative_static 与 ambiguous_low_motion 视为边界样本。
    它们可以用于阈值或边界审计, 但不能作为正向 velocity / trajectory claim 的样本。
    """
    explicit_role = record.get("motion_claim_role") or record.get("motion_calibration_role")
    if explicit_role:
        return str(explicit_role)
    prompt_suite_role = str(record.get("prompt_suite_role") or "")
    if "negative_static" in prompt_suite_role:
        return "negative_static"
    if "ambiguous_low_motion" in prompt_suite_role:
        return "ambiguous_low_motion"
    return "positive_motion"


def is_positive_motion_claim_record(record: dict) -> bool:
    """判断 record 是否应承担正向 motion / trajectory claim 职责。"""
    return _motion_claim_role(record) not in BOUNDARY_MOTION_ROLES


def formal_record_supports_motion_claim(record: dict) -> bool:
    """判断 formal metric record 是否可支撑正向 motion claim。

    这里故意只读取 visual、motion、semantic readiness 与样本角色, 不读取 `S_final`
    或任何最终判定分数。这样可以避免把最终检测结果反向用于污染过滤或样本筛选。
    """
    if not is_positive_motion_claim_record(record):
        return False
    ready_by_flags = (
        record.get("formal_visual_quality_ready") is True
        and record.get("formal_motion_consistency_ready") is True
        and record.get("formal_semantic_consistency_ready") is True
    )
    if record.get("formal_metric_result_used_for_claim") is False:
        return False
    return ready_by_flags


def _formal_motion_claim_status(formal_metric_records: list[dict], excluded_count: int) -> str | None:
    """根据 formal records 生成 motion claim readiness 状态。"""
    if not formal_metric_records:
        return None

    positive_records = [record for record in formal_metric_records if is_positive_motion_claim_record(record)]
    if not positive_records:
        return "blocked_until_positive_motion_formal_records"

    visual_blocked = any(record.get("formal_visual_quality_ready") is not True for record in positive_records)
    motion_blocked = any(record.get("formal_motion_consistency_ready") is not True for record in positive_records)
    semantic_blocked = any(record.get("formal_semantic_consistency_ready") is not True for record in positive_records)
    if visual_blocked:
        return "blocked_by_formal_visual_quality"
    if motion_blocked:
        return "blocked_by_formal_motion_consistency"
    if semantic_blocked:
        return "blocked_until_semantic_metric_ready"
    if excluded_count > 0:
        return "blocked_until_formal_quality_motion_semantic_metrics"
    return "ready"


@dataclass(frozen=True)
class MotionClaimSelection:
    """记录 motion claim 样本筛选结果。"""

    generation_records: list[dict]
    eligible_generation_records: list[dict]
    eligible_generation_keys: frozenset[RecordKey]
    formal_metric_record_count: int
    formal_motion_consistency_ready_count: int
    formal_motion_consistency_blocked_count: int
    motion_claim_eligible_generation_count: int
    motion_claim_excluded_generation_count: int
    formal_motion_claim_status: str | None

    @property
    def formal_records_available(self) -> bool:
        """判断当前筛选是否使用了 formal metric records。"""
        return self.formal_metric_record_count > 0

    def audit_fields(self) -> dict:
        """返回可写入 governed audit artifacts 的筛选摘要字段。"""
        return {
            "formal_metric_record_count": self.formal_metric_record_count,
            "formal_motion_consistency_ready_count": self.formal_motion_consistency_ready_count,
            "formal_motion_consistency_blocked_count": self.formal_motion_consistency_blocked_count,
            "motion_claim_eligible_generation_count": self.motion_claim_eligible_generation_count,
            "motion_claim_excluded_generation_count": self.motion_claim_excluded_generation_count,
            "formal_motion_claim_status": self.formal_motion_claim_status,
        }


def select_motion_claim_generation_records(
    generation_records: Iterable[dict],
    formal_metric_records: Iterable[dict],
) -> MotionClaimSelection:
    """选择可进入 motion / trajectory claim 的 generation records。

    当 formal records 尚不存在时, 该函数保持向后兼容, 暂时返回所有成功 generation records。
    当 formal records 已存在时, 只有通过 formal visual、motion 和 semantic gate 的正向运动样本
    才能进入下游 pilot matrix、runtime attack 统计和 small-scale claim gate 覆盖率计算。
    """
    successful_generation_records = [
        record for record in generation_records
        if record.get("generation_status") == "success"
    ]
    formal_records = list(formal_metric_records)
    formal_ready_count = sum(1 for record in formal_records if record.get("formal_motion_consistency_ready") is True)
    formal_blocked_count = len(formal_records) - formal_ready_count

    if not formal_records:
        eligible_keys = frozenset(record_identity_key(record) for record in successful_generation_records)
        return MotionClaimSelection(
            generation_records=successful_generation_records,
            eligible_generation_records=successful_generation_records,
            eligible_generation_keys=eligible_keys,
            formal_metric_record_count=0,
            formal_motion_consistency_ready_count=0,
            formal_motion_consistency_blocked_count=0,
            motion_claim_eligible_generation_count=len(successful_generation_records),
            motion_claim_excluded_generation_count=0,
            formal_motion_claim_status=None,
        )

    eligible_formal_keys = {
        record_identity_key(record)
        for record in formal_records
        if formal_record_supports_motion_claim(record)
    }
    eligible_generation_records = [
        record for record in successful_generation_records
        if record_identity_key(record) in eligible_formal_keys
    ]
    excluded_count = len(successful_generation_records) - len(eligible_generation_records)
    status = _formal_motion_claim_status(formal_records, excluded_count)
    return MotionClaimSelection(
        generation_records=successful_generation_records,
        eligible_generation_records=eligible_generation_records,
        eligible_generation_keys=frozenset(eligible_formal_keys),
        formal_metric_record_count=len(formal_records),
        formal_motion_consistency_ready_count=formal_ready_count,
        formal_motion_consistency_blocked_count=formal_blocked_count,
        motion_claim_eligible_generation_count=len(eligible_generation_records),
        motion_claim_excluded_generation_count=excluded_count,
        formal_motion_claim_status=status,
    )


def filter_records_to_motion_claim_eligible(records: Iterable[dict], selection: MotionClaimSelection) -> list[dict]:
    """按 formal motion claim eligibility 过滤下游 records。

    该函数用于 gate 统计, 防止旧批次中已经生成的 matrix、attack 或 detection records
    继续把 formal motion 失败样本计入 claim 覆盖率。没有 formal records 时保持原样返回。
    """
    rows = list(records)
    if not selection.formal_records_available:
        return rows
    return [record for record in rows if record_identity_key(record) in selection.eligible_generation_keys]
