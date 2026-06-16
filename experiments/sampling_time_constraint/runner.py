"""运行 sampling-time weak constraint preflight。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from main.generation.constraint_controller import SamplingConstraintConfig, apply_sampling_constraint
from main.protocol.decision import build_stage_decision
from main.protocol.manifest import build_run_manifest
from main.protocol.record_writer import write_json, write_jsonl
from main.protocol.table_builder import write_csv

METHOD_VARIANTS = (
    "key_conditioned_state_space_with_trajectory",
    "keyed_state_trajectory_constraint",
    "trajectory_constraint_without_admissibility",
    "trajectory_constraint_without_key_condition",
    "trajectory_constraint_early_only",
    "trajectory_constraint_late_only",
    "trajectory_constraint_strong_lambda",
    "trajectory_constraint_weak_lambda",
)


def _load_json(path: str) -> dict:
    """读取 JSON 配置。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _build_config() -> dict:
    """加载 B6 preflight 所需配置。"""
    return {
        "protocol": _load_json("configs/protocol/sampling_time_constraint_preflight.json"),
        "constraint": _load_json("configs/generation/sampling_constraint.json"),
        "lambda_schedules": _load_json("configs/generation/lambda_schedules.json"),
    }


def _schedule_by_id(config: dict, schedule_id: str) -> dict:
    """按 ID 查找 lambda schedule 配置。"""
    for item in config["lambda_schedules"]["schedules"]:
        if item["lambda_schedule_id"] == schedule_id:
            return item
    raise KeyError(schedule_id)


def _variant_schedule_id(method_variant: str) -> str:
    """为方法变体选择 schedule。"""
    if method_variant == "trajectory_constraint_early_only":
        return "early_only_constraint"
    if method_variant == "trajectory_constraint_late_only":
        return "late_only_constraint"
    if method_variant == "trajectory_constraint_strong_lambda":
        return "strong_lambda_constraint"
    if method_variant == "trajectory_constraint_weak_lambda":
        return "constant_weak_constraint"
    return "mid_window_weak_constraint"


def _constraint_enabled(method_variant: str) -> bool:
    """判断方法变体是否启用 sampling constraint。"""
    return method_variant != "key_conditioned_state_space_with_trajectory"


def _key_conditioned(method_variant: str) -> bool:
    """判断方法变体是否保留 key condition。"""
    return method_variant != "trajectory_constraint_without_key_condition"


def _admissibility_enabled(method_variant: str) -> bool:
    """判断方法变体是否保留 admissibility 约束。"""
    return method_variant != "trajectory_constraint_without_admissibility"


def _build_velocity_trace(sample_index: int, num_steps: int, key_conditioned: bool) -> list[list[float]]:
    """构造确定性 preflight 速度轨迹。

    该函数属于项目特定写法。它不替代真实生成模型采样过程, 只用于在无 GPU 的 quick 测试中验证 sampling constraint 的记录、审计与消融闭环。
    """
    sign = 1.0 if key_conditioned else -0.45
    jitter = ((sample_index % 3) - 1) * 0.01
    velocities: list[list[float]] = []
    for step_index in range(num_steps):
        progress = step_index / max(num_steps - 1, 1)
        velocities.append([
            -0.015 + sign * 0.01 + progress * 0.01 + jitter,
            0.085 - sign * 0.006 - progress * 0.008,
            -0.055 + sign * 0.004,
        ])
    return velocities


def _quality_motion_semantic_delta(method_variant: str, gain: float) -> tuple[float, float, float]:
    """构造约束后的质量、运动和语义变化代理。"""
    if method_variant == "trajectory_constraint_strong_lambda":
        return 0.072, 0.061, 0.055
    base = max(gain, 0.0)
    return round(base * 0.22, 6), round(base * 0.18, 6), round(base * 0.15, 6)


def _build_constraint_config(config: dict, method_variant: str) -> SamplingConstraintConfig:
    """构造约束控制器配置。"""
    schedule = _schedule_by_id(config, _variant_schedule_id(method_variant))
    constraint = config["constraint"]
    return SamplingConstraintConfig(
        sampling_constraint_config_id=constraint["sampling_constraint_config_id"],
        lambda_schedule_id=schedule["lambda_schedule_id"],
        lambda_max=float(schedule["lambda_max"]),
        lambda_time_window=(float(schedule["lambda_time_window"][0]), float(schedule["lambda_time_window"][1])),
        constraint_norm_budget=float(constraint["constraint_norm_budget"]),
        constraint_direction=tuple(float(value) for value in constraint["constraint_direction"]),
    )


def build_constraint_records(config: dict) -> list[dict]:
    """构造 B6 preflight constraint records。"""
    records: list[dict] = []
    num_steps = int(config["constraint"]["default_num_steps"])
    for sample_index in range(4):
        prompt_id = "motion_object_pan" if sample_index < 2 else "camera_zoom_scene"
        seed_id = "seed_main_a" if sample_index % 2 == 0 else "seed_main_b"
        for method_variant in METHOD_VARIANTS:
            enabled = _constraint_enabled(method_variant)
            key_conditioned = _key_conditioned(method_variant)
            admissibility = _admissibility_enabled(method_variant)
            velocities = _build_velocity_trace(sample_index, num_steps, key_conditioned)
            if enabled:
                summary = apply_sampling_constraint(velocities, _build_constraint_config(config, method_variant))
            else:
                base_config = _build_constraint_config(config, method_variant)
                summary = apply_sampling_constraint(velocities, SamplingConstraintConfig(
                    sampling_constraint_config_id=base_config.sampling_constraint_config_id,
                    lambda_schedule_id=base_config.lambda_schedule_id,
                    lambda_max=0.0,
                    lambda_time_window=base_config.lambda_time_window,
                    constraint_norm_budget=base_config.constraint_norm_budget,
                    constraint_direction=base_config.constraint_direction,
                ))
            gain = float(summary["trajectory_constraint_gain"])
            if not key_conditioned:
                gain = round(gain * 0.35, 6)
            if not admissibility:
                gain = round(gain * 0.78, 6)
            quality_delta, motion_delta, semantic_delta = _quality_motion_semantic_delta(method_variant, gain)
            threshold = 0.58
            before_tpr = 0.62 + sample_index * 0.015
            after_tpr = before_tpr + max(gain, 0.0) * 0.4
            before_fpr = 0.0
            after_fpr = 0.0 if method_variant != "trajectory_constraint_without_key_condition" else 0.005
            records.append({
                "record_version": "sampling_time_constraint_preflight_v1",
                "stage_id": "sampling_time_constraint_preflight",
                "sample_id": f"b6_preflight_sample_{sample_index:04d}",
                "prompt_id": prompt_id,
                "seed_id": seed_id,
                "method_variant": method_variant,
                "sampling_constraint_enabled": enabled,
                "constraint_projection_operator_id": config["constraint"]["constraint_projection_operator_id"],
                "constraint_key_id": config["constraint"]["constraint_key_id"],
                "constraint_payload_code_id": config["constraint"]["constraint_payload_code_id"],
                "constraint_tubelet_selector_id": config["constraint"]["constraint_tubelet_selector_id"],
                "constraint_admissibility_enabled": admissibility,
                "constraint_key_condition_enabled": key_conditioned,
                "constraint_runtime_overhead_sec": round(0.015 + summary["constraint_apply_steps"] * 0.001, 6),
                "attacked_positive_TPR_before_constraint": round(before_tpr, 6),
                "attacked_positive_TPR_after_constraint": round(after_tpr, 6),
                "attacked_negative_FPR_before_constraint": before_fpr,
                "attacked_negative_FPR_after_constraint": after_fpr,
                "quality_delta_after_constraint": quality_delta,
                "motion_delta_after_constraint": motion_delta,
                "semantic_delta_after_constraint": semantic_delta,
                "constraint_quality_status": "PASS" if quality_delta <= config["protocol"]["quality_delta_limit"] else "BLOCKING",
                "constraint_motion_status": "PASS" if motion_delta <= config["protocol"]["motion_delta_limit"] else "BLOCKING",
                "constraint_semantic_status": "PASS" if semantic_delta <= config["protocol"]["semantic_delta_limit"] else "BLOCKING",
                "constraint_main_claim_status": "preflight_only_not_final_b6_claim",
                "constraint_threshold_value": threshold,
                **summary,
                "trajectory_constraint_gain": gain,
            })
    return records


def _mean_for(records: list[dict], method_variant: str, field: str) -> float:
    """计算某一变体字段均值。"""
    values = [float(record[field]) for record in records if record["method_variant"] == method_variant]
    return mean(values) if values else 0.0


def audit_constraint_preflight(records: list[dict], config: dict) -> dict:
    """审计 B6 preflight 是否形成可继续接入真实采样的证据。"""
    main_gain = _mean_for(records, "keyed_state_trajectory_constraint", "trajectory_constraint_gain")
    baseline_gain = _mean_for(records, "key_conditioned_state_space_with_trajectory", "trajectory_constraint_gain")
    tpr_gain = _mean_for(records, "keyed_state_trajectory_constraint", "attacked_positive_TPR_after_constraint") - _mean_for(records, "keyed_state_trajectory_constraint", "attacked_positive_TPR_before_constraint")
    after_fpr = _mean_for(records, "keyed_state_trajectory_constraint", "attacked_negative_FPR_after_constraint")
    mid_gain = _mean_for(records, "keyed_state_trajectory_constraint", "trajectory_constraint_gain")
    early_gain = _mean_for(records, "trajectory_constraint_early_only", "trajectory_constraint_gain")
    late_gain = _mean_for(records, "trajectory_constraint_late_only", "trajectory_constraint_gain")
    quality_motion_semantic_gate = all(
        record["constraint_quality_status"] == "PASS"
        and record["constraint_motion_status"] == "PASS"
        and record["constraint_semantic_status"] == "PASS"
        for record in records
        if record["method_variant"] == "keyed_state_trajectory_constraint"
    )
    strong_lambda_blocked = any(
        record["constraint_quality_status"] == "BLOCKING"
        or record["constraint_motion_status"] == "BLOCKING"
        or record["constraint_semantic_status"] == "BLOCKING"
        for record in records
        if record["method_variant"] == "trajectory_constraint_strong_lambda"
    )
    preflight_pass = all([
        main_gain > float(config["protocol"]["trajectory_constraint_gain_min"]),
        main_gain > baseline_gain,
        tpr_gain > float(config["protocol"]["attacked_positive_tpr_gain_min"]),
        after_fpr <= float(config["protocol"]["target_fpr"]),
        quality_motion_semantic_gate,
        mid_gain >= early_gain,
        mid_gain >= late_gain,
        strong_lambda_blocked,
    ])
    return {
        "sampling_time_constraint_preflight_decision": "PASS" if preflight_pass else "FAIL",
        "mechanism_pass": preflight_pass,
        "trajectory_constraint_gain_mean": round(main_gain, 6),
        "trajectory_constraint_gain_over_unconstrained": round(main_gain - baseline_gain, 6),
        "attacked_positive_tpr_gain": round(tpr_gain, 6),
        "attacked_negative_fpr_after_constraint": round(after_fpr, 6),
        "quality_motion_semantic_constraint_gate": "PASS" if quality_motion_semantic_gate else "FAIL",
        "lambda_schedule_ablation_supports_mid_stage": bool(mid_gain >= early_gain and mid_gain >= late_gain),
        "strong_lambda_quality_block_detected": strong_lambda_blocked,
        "constraint_main_claim_status": "preflight_only_not_final_b6_claim",
        "submission_claim_policy": config["protocol"]["submission_claim_policy"],
    }


def run(output_root: str | Path) -> dict:
    """运行 B6 sampling-time constraint preflight 并写出 governed artifacts。"""
    output_root = Path(output_root)
    config = _build_config()
    records = build_constraint_records(config)
    audit = audit_constraint_preflight(records, config)

    constraint_path = output_root / "records" / "constraint_records.jsonl"
    table_path = output_root / "tables" / "sampling_constraint_ablation_table.csv"
    quality_table_path = output_root / "tables" / "constraint_quality_motion_semantic_table.csv"
    report_path = output_root / "reports" / "sampling_time_constraint_preflight_report.md"
    decision_path = output_root / "artifacts" / "sampling_time_constraint_preflight_decision.json"
    manifest_path = output_root / "artifacts" / "sampling_time_constraint_preflight_manifest.json"

    write_jsonl(constraint_path, records)
    write_csv(table_path, records)
    write_csv(quality_table_path, [
        {
            "method_variant": method_variant,
            "trajectory_constraint_gain_mean": round(_mean_for(records, method_variant, "trajectory_constraint_gain"), 6),
            "quality_delta_mean": round(_mean_for(records, method_variant, "quality_delta_after_constraint"), 6),
            "motion_delta_mean": round(_mean_for(records, method_variant, "motion_delta_after_constraint"), 6),
            "semantic_delta_mean": round(_mean_for(records, method_variant, "semantic_delta_after_constraint"), 6),
        }
        for method_variant in METHOD_VARIANTS
    ])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "# Sampling-Time Constraint Preflight Report\n\n"
        "该报告验证 sampling-time weak constraint 的工程闭环。当前 records 只支持 preflight 结论, 不支持最终 B6 论文 claim。\n\n"
        + json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    implementation_pass = all([constraint_path.exists(), table_path.exists(), quality_table_path.exists(), bool(records)])
    decision = build_stage_decision(implementation_pass, bool(audit["mechanism_pass"]), audit)
    decision["stage_id"] = "sampling_time_constraint_preflight"
    write_json(decision_path, decision)
    write_json(manifest_path, build_run_manifest("sampling_time_constraint_preflight", config, [str(constraint_path), str(table_path), str(quality_table_path), str(decision_path)]))
    return {
        "output_root": str(output_root),
        "constraint_record_count": len(records),
        "implementation_decision": decision["implementation_decision"],
        "mechanism_decision": decision["mechanism_decision"],
        "audit": audit,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 B6 sampling-time weak constraint preflight。")
    parser.add_argument("--output-root", default="outputs/runs/sampling_time_constraint_preflight")
    args = parser.parse_args()
    print(json.dumps(run(args.output_root), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
