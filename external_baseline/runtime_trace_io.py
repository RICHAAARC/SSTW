"""外部 baseline adapter 共享的运行记录读取与轨迹序列构造工具。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from main.core.digest import build_stable_digest


RUNTIME_DETECTION_RECORD_PATH = Path("records/runtime_detection_records.jsonl")
TRAJECTORY_TRACE_RECORD_PATH = Path("records/trajectory_trace.jsonl")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """读取 JSONL 文件, 并兼容 Google Drive 或 Windows 工具写入的 UTF-8 BOM。

    该函数属于通用工程写法。外部 baseline adapter 只依赖已经落盘的 governed records,
    不直接访问 Notebook cell 中的临时变量, 因而可以在 Colab、本地 Windows 和 CI 中复用。
    """
    input_path = Path(path)
    if not input_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in input_path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def safe_float(value: Any, default: float = 0.0) -> float:
    """把可能为空的记录字段转换为 float, 失败时返回默认值。"""
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def mean(values: Iterable[float]) -> float | None:
    """计算均值, 空序列返回 None。"""
    materialized = list(values)
    if not materialized:
        return None
    return sum(materialized) / len(materialized)


def group_trajectory_records(trajectory_records: Iterable[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """按 trajectory_trace_id 组织 callback 轨迹记录。"""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in trajectory_records:
        trace_id = str(record.get("trajectory_trace_id") or "")
        if trace_id:
            grouped.setdefault(trace_id, []).append(dict(record))
    for trace_id, rows in grouped.items():
        grouped[trace_id] = sorted(rows, key=lambda item: safe_float(item.get("trajectory_step_index"), 0.0))
    return grouped


def _normalize_columns(rows: list[list[float]]) -> list[list[float]]:
    """把轨迹特征列归一化到稳定数值区间, 避免 latent norm 的量纲支配 DTW cost。"""
    if not rows:
        return []
    column_count = len(rows[0])
    columns = [[row[index] for row in rows] for index in range(column_count)]
    normalized_columns: list[list[float]] = []
    for column in columns:
        lower = min(column)
        upper = max(column)
        if upper == lower:
            normalized_columns.append([0.0 for _ in column])
        else:
            normalized_columns.append([(value - lower) / (upper - lower) for value in column])
    return [
        [round(normalized_columns[column_index][row_index], 6) for column_index in range(column_count)]
        for row_index in range(len(rows))
    ]


def build_reference_sequence(trace_rows: list[Mapping[str, Any]]) -> list[list[float]]:
    """从 callback trajectory records 构造外部同步 baseline 可消费的参考序列。

    这一实现属于 SSTW 项目特定写法。当前 Wan2.1 preflight 已能记录 latent 的 norm、mean 和 std,
    但外部显式同步 baseline 并不直接理解这些字段。因此此处把每个 callback step 转成短向量,
    使 DTW 和 frame matching control 可以在同一个 run_root 中形成可审计的比较记录。
    """
    ordered = sorted(trace_rows, key=lambda item: safe_float(item.get("trajectory_step_index"), 0.0))
    if len(ordered) < 2:
        return []
    last_index = max(len(ordered) - 1, 1)
    raw_rows: list[list[float]] = []
    for index, record in enumerate(ordered):
        raw_rows.append([
            index / last_index,
            safe_float(record.get("latent_norm")),
            safe_float(record.get("latent_mean")),
            safe_float(record.get("latent_std")),
        ])
    return _normalize_columns(raw_rows)


def build_observed_sequence(reference_sequence: list[list[float]], detection_record: Mapping[str, Any]) -> list[list[float]]:
    """基于 runtime video metadata 构造显式同步 baseline 的观测序列 proxy。

    此处刻意不读取 `S_final` 或最终判定分数。观测序列只使用 attack name、源视频帧数、攻击后帧数和解码帧数,
    目的是避免外部 baseline comparison 被 SSTW 最终检测分数污染。该实现是工程闭环 proxy,
    后续接入真正 baseline 检测器时可以替换为官方输出的 embedding 或 detector score。
    """
    if not reference_sequence:
        return []
    attack_name = str(detection_record.get("attack_name") or "")
    source_count = int(safe_float(detection_record.get("source_frame_count"), 0.0))
    attacked_count = int(safe_float(detection_record.get("attacked_frame_count"), 0.0))
    decoded_count = int(safe_float(detection_record.get("attacked_video_decoded_frame_count"), 0.0))
    effective_count = attacked_count or decoded_count or source_count or len(reference_sequence)
    ratio = effective_count / source_count if source_count else 1.0

    if "frame_rate_resampling" in attack_name or ratio <= 0.7:
        observed = reference_sequence[::2]
    elif "temporal_crop" in attack_name or ratio < 0.95:
        observed = reference_sequence[1:-1] if len(reference_sequence) > 3 else reference_sequence[:]
    else:
        observed = [row[:] for row in reference_sequence]

    if not observed:
        observed = reference_sequence[:]

    if "compression" in attack_name:
        adjusted: list[list[float]] = []
        for row in observed:
            copied = row[:]
            if len(copied) >= 4:
                copied[3] = round(max(0.0, min(1.0, copied[3] * 0.98 + 0.01)), 6)
            adjusted.append(copied)
        observed = adjusted
    return observed


def build_comparison_unit_id(baseline_name: str, detection_record: Mapping[str, Any]) -> str:
    """生成单条 baseline comparison record 的稳定标识。"""
    payload = {
        "external_baseline_name": baseline_name,
        "generation_model_id": detection_record.get("generation_model_id"),
        "prompt_id": detection_record.get("prompt_id"),
        "seed_id": detection_record.get("seed_id"),
        "trajectory_trace_id": detection_record.get("trajectory_trace_id"),
        "attack_name": detection_record.get("attack_name"),
    }
    digest = build_stable_digest(payload)
    return f"external_baseline_score_{digest[:16]}"


def comparable_detection_records(run_root: str | Path) -> list[dict[str, Any]]:
    """读取可进入 external baseline comparison 的 runtime detection records。"""
    root = Path(run_root)
    records = read_jsonl(root / RUNTIME_DETECTION_RECORD_PATH)
    return [record for record in records if record.get("runtime_detection_status") == "ready"]


def load_trace_groups(run_root: str | Path) -> dict[str, list[dict[str, Any]]]:
    """读取并分组 run_root 中的 callback trajectory records。"""
    root = Path(run_root)
    return group_trajectory_records(read_jsonl(root / TRAJECTORY_TRACE_RECORD_PATH))
