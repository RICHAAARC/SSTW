"""提供 trajectory 统计量工具。"""

from __future__ import annotations

from statistics import mean


def mean_by_role(records: list[dict], sample_role: str) -> float:
    """计算指定 sample role 的平均 trajectory observation。"""
    values = [float(record["S_trajectory_observation"]) for record in records if record["sample_role"] == sample_role and record.get("S_trajectory_observation") is not None]
    return mean(values) if values else 0.0


def centered_correlation(records: list[dict], x_field: str, y_field: str, group_field: str = "sample_role") -> float:
    """计算按组中心化后的相关系数。

    中心化用于检查 trajectory 是否携带 static evidence 之外的条件变化, 避免 H0/H1 标签本身
    造成的虚假高相关。
    """
    groups: dict[str, list[dict]] = {}
    for record in records:
        if record.get(x_field) is None or record.get(y_field) is None:
            continue
        groups.setdefault(str(record[group_field]), []).append(record)
    xs: list[float] = []
    ys: list[float] = []
    for group in groups.values():
        mean_x = mean(float(record[x_field]) for record in group)
        mean_y = mean(float(record[y_field]) for record in group)
        for record in group:
            xs.append(float(record[x_field]) - mean_x)
            ys.append(float(record[y_field]) - mean_y)
    if not xs or not ys:
        return 0.0
    mean_x = mean(xs)
    mean_y = mean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denom_x = sum((x - mean_x) ** 2 for x in xs) ** 0.5
    denom_y = sum((y - mean_y) ** 2 for y in ys) ** 0.5
    if denom_x == 0 or denom_y == 0:
        return 0.0
    return round(numerator / (denom_x * denom_y), 6)
