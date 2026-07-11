"""评估冻结 SSTW 后验在 held-out attacked videos 上的概率可靠性。"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from evaluation.protocol.record_writer import write_json, write_jsonl
from evaluation.protocol.table_builder import write_csv
from evaluation.statistics.probability_calibration import (
    clustered_probability_calibration_interval,
)
from runtime.core.digest import build_stable_digest


HELDOUT_POSTERIOR_CALIBRATION_PROTOCOL = (
    "frozen_detector_heldout_attacked_test_class_balanced_source_video_cluster_bootstrap"
)


def _required_config_number(config: Mapping[str, Any], field_name: str) -> float:
    """读取不可省略的正式阈值，避免配置缺失时静默回退到宽松默认值。"""

    if field_name not in config:
        raise KeyError(f"正式 profile 缺少 held-out posterior 配置字段: {field_name}")
    value = float(config[field_name])
    if value < 0.0:
        raise ValueError(f"{field_name} 不能为负数")
    return value


def _record_label(record: Mapping[str, Any]) -> int:
    """将正式样本角色转换为 Claim-3 二元假设标签。"""

    sample_role = str(record.get("sample_role") or "")
    if sample_role == "attacked_positive":
        return 1
    if sample_role in {"clean_negative", "controlled_negative"}:
        return 0
    raise ValueError(f"held-out posterior 出现未知 sample_role: {sample_role or 'missing'}")


def _scope_rows(
    rows: list[dict[str, Any]],
    *,
    attack_name: str | None,
) -> list[dict[str, Any]]:
    """为全局或单攻击可靠性评测选择正负样本。

    单攻击记录只保留该攻击的 positives 和 controls，同时复用预注册 held-out
    clean negatives。这样不会让其他攻击的 positives 稀释当前攻击的可靠性结论。
    """

    if attack_name is None:
        return list(rows)
    selected: list[dict[str, Any]] = []
    for row in rows:
        sample_role = str(row.get("sample_role") or "")
        if sample_role == "clean_negative":
            selected.append(row)
        elif str(row.get("attack_name") or "") == attack_name:
            selected.append(row)
    return selected


def _build_scope_record(
    rows: list[dict[str, Any]],
    *,
    config: Mapping[str, Any],
    generation_model_id: str,
    cross_model_role: str | None,
    method_variant: str,
    attack_name: str | None,
) -> dict[str, Any]:
    """构建一个冻结模型、方法变体和攻击范围的概率可靠性记录。"""

    scope = "global_attack_set" if attack_name is None else "single_preregistered_attack"
    scoped_rows = _scope_rows(rows, attack_name=attack_name)
    invalid_rows = [
        row
        for row in scoped_rows
        if row.get("split") != "test"
        or row.get("threshold_source_split") != "calibration"
        or row.get("test_time_threshold_update_blocked") is not True
        or row.get("metric_status") != "measured_formal"
        or row.get("flow_watermark_posterior_probability") is None
        or not str(row.get("statistical_cluster_id") or "").strip()
    ]
    if invalid_rows:
        raise ValueError(
            "held-out posterior 只能消费 test split 上由 calibration 冻结检测器产生的 governed records"
        )
    labels = [_record_label(row) for row in scoped_rows]
    probabilities = [
        float(row["flow_watermark_posterior_probability"])
        for row in scoped_rows
    ]
    cluster_ids = [str(row["statistical_cluster_id"]) for row in scoped_rows]
    estimate = clustered_probability_calibration_interval(
        probabilities,
        labels,
        cluster_ids,
        bootstrap_resample_count=int(
            _required_config_number(
                config,
                "heldout_posterior_bootstrap_resample_count",
            )
        ),
        purpose=(
            f"heldout_posterior::{generation_model_id}::{method_variant}::"
            f"{attack_name or 'global'}"
        ),
    )
    minimum_positive_count = int(_required_config_number(
        config,
        "minimum_heldout_posterior_positive_cluster_count",
    ))
    if attack_name is not None:
        minimum_positive_count = int(_required_config_number(
            config,
            "minimum_heldout_posterior_attack_cluster_count",
        ))
    minimum_negative_count = int(_required_config_number(
        config,
        "minimum_heldout_posterior_negative_cluster_count",
    ))
    failure_reasons: list[str] = []
    if estimate.positive_cluster_count < minimum_positive_count:
        failure_reasons.append("heldout_positive_cluster_count_below_minimum")
    if estimate.negative_cluster_count < minimum_negative_count:
        failure_reasons.append("heldout_negative_cluster_count_below_minimum")
    if estimate.brier_score_ci_upper > _required_config_number(
        config,
        "maximum_heldout_posterior_brier_score",
    ):
        failure_reasons.append("heldout_brier_score_upper_bound_above_maximum")
    if estimate.log_loss_ci_upper > _required_config_number(
        config,
        "maximum_heldout_posterior_log_loss",
    ):
        failure_reasons.append("heldout_log_loss_upper_bound_above_maximum")
    if estimate.expected_calibration_error_ci_upper > _required_config_number(
        config,
        "maximum_heldout_posterior_expected_calibration_error",
    ):
        failure_reasons.append("heldout_expected_calibration_error_upper_bound_above_maximum")
    payload: dict[str, Any] = {
        "record_version": "heldout_posterior_calibration_v1",
        "generation_model_id": generation_model_id,
        "cross_model_role": cross_model_role,
        "method_variant": method_variant,
        "attack_name": attack_name,
        "split": "test",
        "heldout_posterior_calibration_scope": scope,
        "heldout_posterior_calibration_protocol": (
            HELDOUT_POSTERIOR_CALIBRATION_PROTOCOL
        ),
        **estimate.as_dict(),
        "heldout_posterior_attack_cluster_count": (
            estimate.positive_cluster_count if attack_name is not None else None
        ),
        "heldout_posterior_calibration_ready": not failure_reasons,
        "heldout_posterior_calibration_failure_reasons": failure_reasons,
        "threshold_source_split": "calibration",
        "test_time_threshold_update_blocked": True,
        "metric_status": "measured_formal" if not failure_reasons else "blocked",
        "claim_support_status": (
            "claim3_heldout_posterior_calibration_ready"
            if not failure_reasons
            else "claim3_heldout_posterior_calibration_blocked"
        ),
    }
    payload["heldout_posterior_calibration_record_id"] = build_stable_digest({
        "generation_model_id": generation_model_id,
        "method_variant": method_variant,
        "attack_name": attack_name,
        "protocol": HELDOUT_POSTERIOR_CALIBRATION_PROTOCOL,
        "record_ids": sorted(
            str(row.get("formal_flow_evidence_unit_id") or "")
            for row in scoped_rows
        ),
    })
    return payload


def build_heldout_posterior_calibration_records(
    scored_records: Iterable[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """从冻结检测器的 held-out records 重建全局和逐攻击可靠性证据。"""

    heldout_rows = [
        dict(record)
        for record in scored_records
        if record.get("split") == "test"
        and record.get("sample_role")
        in {"attacked_positive", "clean_negative", "controlled_negative"}
    ]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in heldout_rows:
        if str(row.get("method_variant") or "") != "sstw_full_method":
            continue
        grouped[(
            str(row.get("generation_model_id") or ""),
            str(row.get("method_variant") or ""),
        )].append(row)
    required_attacks = [
        str(value)
        for value in config.get("required_runtime_attack_names", [])
        if str(value).strip()
    ]
    records: list[dict[str, Any]] = []
    for (generation_model_id, method_variant), rows in sorted(grouped.items()):
        if not generation_model_id or not method_variant:
            raise ValueError("held-out posterior records 缺少模型或方法变体标识")
        cross_model_role = next(
            (str(row.get("cross_model_role")) for row in rows if row.get("cross_model_role")),
            None,
        )
        records.append(_build_scope_record(
            rows,
            config=config,
            generation_model_id=generation_model_id,
            cross_model_role=cross_model_role,
            method_variant=method_variant,
            attack_name=None,
        ))
        # Claim-3 的逐攻击强结论只由完整方法的主验证模型支持。跨模型子集仍
        # 生成全局记录，但不会因没有覆盖全部主攻击而被伪装为逐攻击闭合。
        if (
            method_variant == "sstw_full_method"
            and cross_model_role != "cross_model_validation_model"
        ):
            for attack_name in required_attacks:
                records.append(_build_scope_record(
                    rows,
                    config=config,
                    generation_model_id=generation_model_id,
                    cross_model_role=cross_model_role,
                    method_variant=method_variant,
                    attack_name=attack_name,
                ))
    return records


def audit_heldout_posterior_calibration_records(
    records: Iterable[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """要求主模型完整方法同时具有全局和每个预注册攻击的可靠性证据。"""

    rows = [dict(record) for record in records]
    required_attacks = {
        str(value)
        for value in config.get("required_runtime_attack_names", [])
        if str(value).strip()
    }
    primary_full_rows = [
        row
        for row in rows
        if row.get("method_variant") == "sstw_full_method"
        and row.get("cross_model_role") != "cross_model_validation_model"
    ]
    model_ids = sorted({
        str(row.get("generation_model_id") or "") for row in primary_full_rows
    } - {""})
    missing_scopes: list[str] = []
    blocked_scopes: list[str] = []
    for model_id in model_ids:
        model_rows = [
            row for row in primary_full_rows
            if str(row.get("generation_model_id") or "") == model_id
        ]
        global_rows = [
            row for row in model_rows
            if row.get("heldout_posterior_calibration_scope") == "global_attack_set"
        ]
        if len(global_rows) != 1:
            missing_scopes.append(f"{model_id}::global_attack_set")
        elif global_rows[0].get("heldout_posterior_calibration_ready") is not True:
            blocked_scopes.append(f"{model_id}::global_attack_set")
        by_attack = {
            str(row.get("attack_name")): row
            for row in model_rows
            if row.get("heldout_posterior_calibration_scope")
            == "single_preregistered_attack"
        }
        for attack_name in sorted(required_attacks):
            if attack_name not in by_attack:
                missing_scopes.append(f"{model_id}::{attack_name}")
            elif by_attack[attack_name].get("heldout_posterior_calibration_ready") is not True:
                blocked_scopes.append(f"{model_id}::{attack_name}")
    decision = (
        "PASS"
        if model_ids and not missing_scopes and not blocked_scopes
        else "FAIL"
    )
    return {
        "stage_id": "heldout_posterior_calibration",
        "heldout_posterior_calibration_decision": decision,
        "heldout_posterior_calibration_record_count": len(rows),
        "heldout_posterior_primary_model_ids": model_ids,
        "heldout_posterior_required_attack_names": sorted(required_attacks),
        "heldout_posterior_missing_scopes": missing_scopes,
        "heldout_posterior_blocked_scopes": blocked_scopes,
        "claim_support_status": (
            "claim3_heldout_posterior_calibration_supported"
            if decision == "PASS"
            else "claim3_heldout_posterior_calibration_blocked"
        ),
    }


def write_heldout_posterior_calibration_artifacts(
    run_root: str | Path,
    records: Iterable[Mapping[str, Any]],
    audit: Mapping[str, Any],
) -> None:
    """将同一批 governed records 重建为 record、table、artifact 和 report。"""

    root = Path(run_root)
    rows = [dict(record) for record in records]
    write_jsonl(
        root / "records" / "heldout_posterior_calibration_records.jsonl",
        rows,
    )
    write_csv(
        root / "tables" / "heldout_posterior_calibration_table.csv",
        rows,
    )
    write_json(
        root / "artifacts" / "heldout_posterior_calibration_decision.json",
        dict(audit),
    )
    report_path = root / "reports" / "heldout_posterior_calibration_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "# Held-out Posterior Calibration Report\n\n"
        "该报告只评价 calibration split 冻结后的检测器在 held-out attacked test "
        "上的概率可靠性，并按 source-video cluster 执行 bootstrap。\n\n"
        f"- decision: {audit.get('heldout_posterior_calibration_decision')}\n"
        f"- record_count: {audit.get('heldout_posterior_calibration_record_count')}\n"
        f"- missing_scopes: {audit.get('heldout_posterior_missing_scopes')}\n"
        f"- blocked_scopes: {audit.get('heldout_posterior_blocked_scopes')}\n",
        encoding="utf-8",
    )
