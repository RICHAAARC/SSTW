"""冻结 probe、pilot 与 full 共用的论文机制配置。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


PAPER_PROFILE_NAMES = {"probe_paper", "pilot_paper", "full_paper"}
COMMON_CONTRACT_PATH_FIELD = "paper_profile_common_contract_path"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"paper profile JSON 顶层必须是对象: {path}")
    return payload


def enforce_paper_profile_common_contract(
    profile: Mapping[str, Any],
    profile_path: str | Path,
) -> dict[str, Any]:
    """验证 profile 未漂移并返回合并后的只读语义副本。

    profile 文件仍显式保存公共字段, 便于独立审阅。加载时逐字段与公共契约比对,
    任一差异立即失败。只有公共契约未登记的样本数、FPR 与运行分片字段可以按
    profile 变化, 从机制上阻断 pilot 残留旧语义或单文件静默漂移。
    """

    merged = dict(profile)
    level = str(merged.get("paper_result_level") or "")
    if level not in PAPER_PROFILE_NAMES:
        return merged
    raw_contract_path = merged.get(COMMON_CONTRACT_PATH_FIELD)
    if not raw_contract_path:
        canonical_names = {
            "probe_paper_generative_probe.json",
            "pilot_paper_generative_probe.json",
            "full_paper_generative_probe.json",
        }
        if Path(profile_path).name in canonical_names:
            raise KeyError(f"{level} 缺少 {COMMON_CONTRACT_PATH_FIELD}")
        # 单元测试或用户自定义的临时 profile 可以只覆盖局部门禁。正式仓库配置
        # 由上面的规范文件名检查和约束测试共同保证不能绕过公共契约。
        return merged
    contract_path = Path(str(raw_contract_path))
    if not contract_path.is_absolute() and not contract_path.exists():
        candidate = Path(profile_path).parent / contract_path
        if candidate.exists():
            contract_path = candidate
    if not contract_path.exists():
        raise FileNotFoundError(f"paper profile 公共契约不存在: {contract_path}")
    contract = _read_json(contract_path)
    drift = {
        key: {"expected": expected, "observed": merged.get(key)}
        for key, expected in contract.items()
        if merged.get(key) != expected
    }
    if drift:
        raise ValueError(
            "paper profile 公共机制配置漂移: "
            + json.dumps(drift, ensure_ascii=False, sort_keys=True)
        )
    merged.update(contract)
    merged["paper_profile_common_contract_resolved_path"] = str(contract_path)
    merged["paper_profile_common_contract_status"] = "matched"
    return merged
