"""从 records 重建第一阶段 CSV 表格。"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean


def build_method_attack_table(records: list[dict]) -> list[dict]:
    """按方法、攻击和样本角色聚合平均分数与 positive rate。"""
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for record in records:
        if record["split"] == "test":
            grouped[(record["method_variant"], record["attack_name"], record["sample_role"])].append(record)
    rows: list[dict] = []
    for (method_variant, attack_name, sample_role), group in sorted(grouped.items()):
        rows.append({"method_variant": method_variant, "attack_name": attack_name, "sample_role": sample_role, "mean_s_final": round(mean(float(item["S_final"]) for item in group), 6), "positive_rate": round(mean(1.0 if item["decision"] == "positive" else 0.0 for item in group), 6), "record_count": len(group)})
    return rows


def write_csv(path: str | Path, rows: list[dict]) -> None:
    """将聚合表写为 CSV。

    通用工程写法是用所有行字段的并集构造表头, 而不是只使用第一行字段。
    项目特定原因在于部分 governed records 会同时包含正式曲线点和范围说明行,
    后者可能拥有额外说明字段。CSV 表格应保留这些字段并用空值补齐其它行,
    不能因为记录类型存在轻微扩展而阻断 Notebook 阶段。
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
