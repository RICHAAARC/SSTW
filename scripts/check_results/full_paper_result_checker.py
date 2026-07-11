"""为旧结果包保留的 full_paper 公共门禁兼容检查入口。

full_paper 不再维护独立于 probe/pilot 的结论检查器。该模块只调用参数化
`paper_profile_gate`, 并额外写出历史文件名和决策字段，供旧打包脚本读取。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluation.attacks.video_runtime_attack_protocol import (
    load_protocol_config_with_shared_attack_protocol,
)
from evaluation.protocol.flow_evidence_fields import (
    with_flow_evidence_protocol_defaults,
)
from evaluation.protocol.record_writer import write_json, write_jsonl
from evaluation.protocol.table_builder import write_csv
from experiments.generative_video_model_probe.paper_profile_gate import (
    build_paper_profile_gate_audit,
    write_paper_profile_gate_audit,
)


DEFAULT_FULL_PAPER_CONFIG = "configs/protocol/full_paper_generative_probe.json"


def _validate_full_profile_config(config_path: str | Path) -> None:
    """阻止旧 full checker 被用于其他 profile 或绕过公共契约。"""

    config = load_protocol_config_with_shared_attack_protocol(config_path)
    if config.get("paper_result_level") != "full_paper":
        raise ValueError("full_paper 兼容检查器只接受 full_paper protocol config")
    if config.get("paper_profile_common_contract_status") != "matched":
        raise ValueError("full_paper protocol config 未通过公共机制契约校验")


def _with_legacy_decision_aliases(audit: dict[str, Any]) -> dict[str, Any]:
    """只添加旧字段别名, 不增加新的主张判定条件。"""

    decision = str(audit.get("paper_profile_gate_decision") or "FAIL")
    return {
        **audit,
        "full_paper_result_checker_decision": decision,
        "full_paper_result_decision": decision,
    }


def build_full_paper_result_checker_audit(
    run_root: str | Path,
    config_path: str | Path = DEFAULT_FULL_PAPER_CONFIG,
) -> dict[str, Any]:
    """调用公共参数化 gate, 并返回旧决策字段别名。"""

    _validate_full_profile_config(config_path)
    return _with_legacy_decision_aliases(
        build_paper_profile_gate_audit(run_root, config_path)
    )


def write_full_paper_result_checker_audit(
    run_root: str | Path,
    config_path: str | Path = DEFAULT_FULL_PAPER_CONFIG,
) -> dict[str, Any]:
    """写出公共 full gate 后, 同步历史 checker 文件名。"""

    _validate_full_profile_config(config_path)
    root = Path(run_root)
    public_audit = write_paper_profile_gate_audit(root, config_path)
    audit = _with_legacy_decision_aliases(public_audit)
    record = with_flow_evidence_protocol_defaults(
        {"record_version": "full_paper_result_checker_compatibility_v1", **audit},
        trajectory_source_level="full_paper_public_gate_compatibility_alias",
        flow_state_admissibility_status=(
            "full_paper_ready"
            if audit["full_paper_result_checker_decision"] == "PASS"
            else "full_paper_blocked"
        ),
        claim_support_status=str(audit.get("claim_support_status") or "blocked"),
    )
    write_jsonl(root / "records" / "full_paper_result_checker_records.jsonl", [record])
    write_csv(root / "tables" / "full_paper_result_checker_table.csv", [record])
    write_json(root / "artifacts" / "full_paper_result_checker_decision.json", audit)
    write_json(root / "artifacts" / "full_paper_result_decision.json", audit)
    report_path = root / "reports" / "full_paper_result_checker_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "# Full Paper Result Checker Compatibility Report\n\n"
        "该文件只复用公共参数化 paper profile gate 的结论, 不执行第二套检查。\n\n"
        f"- full_paper_result_checker_decision: {audit['full_paper_result_checker_decision']}\n"
        f"- paper_profile_gate_decision: {audit['paper_profile_gate_decision']}\n",
        encoding="utf-8",
    )
    return audit


def main() -> None:
    """提供可脱离 Notebook 运行的旧 CLI。"""

    parser = argparse.ArgumentParser(
        description="通过公共参数化门禁检查 full_paper 结果。"
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--config-path", default=DEFAULT_FULL_PAPER_CONFIG)
    parser.add_argument("--write-outputs", action="store_true")
    args = parser.parse_args()
    payload = (
        write_full_paper_result_checker_audit(args.run_root, args.config_path)
        if args.write_outputs
        else build_full_paper_result_checker_audit(args.run_root, args.config_path)
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
