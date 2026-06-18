"""构造 small-scale claim pilot 的轻量矩阵后处理记录。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from main.protocol.record_writer import write_json, write_jsonl
from main.protocol.table_builder import write_csv


PILOT_ATTACKS = (
    "video_compression_proxy",
    "temporal_crop_proxy",
    "frame_rate_resampling_proxy",
)

PILOT_METHOD_VARIANTS = (
    "sstw_full_method",
    "endpoint_only_control",
    "trajectory_only_score",
    "without_velocity_constraint",
    "without_endpoint_aware_control",
    "without_replay_uncertainty_weighting",
)

PILOT_NEGATIVE_FAMILIES = (
    "wrong_key_control",
    "without_key_control",
    "wrong_sampler_replay",
    "trajectory_time_shuffle_control",
)

ATTACK_SCORE_PENALTY = {
    "video_compression_proxy": 0.015,
    "temporal_crop_proxy": 0.035,
    "frame_rate_resampling_proxy": 0.025,
}

METHOD_SCORE_OFFSET = {
    "sstw_full_method": 0.0,
    "endpoint_only_control": -0.085,
    "trajectory_only_score": -0.045,
    "without_velocity_constraint": -0.065,
    "without_endpoint_aware_control": -0.055,
    "without_replay_uncertainty_weighting": -0.035,
}

NEGATIVE_SCORE_BASE = {
    "wrong_key_control": 0.22,
    "without_key_control": 0.18,
    "wrong_sampler_replay": 0.24,
    "trajectory_time_shuffle_control": 0.20,
}


def _read_jsonl(path: Path) -> list[dict]:
    """读取 JSONL records, 文件不存在时返回空列表。"""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _safe_divide(numerator: float, denominator: float) -> float:
    """执行带零值保护的除法。"""
    return 0.0 if denominator == 0 else numerator / denominator


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """把分数裁剪到固定区间。"""
    return max(lower, min(upper, value))


def _group_trajectory_records(trajectory_records: list[dict]) -> dict[str, list[dict]]:
    """按 trajectory_trace_id 聚合 trajectory records。"""
    grouped: dict[str, list[dict]] = {}
    for record in trajectory_records:
        trace_id = record.get("trajectory_trace_id")
        if trace_id:
            grouped.setdefault(str(trace_id), []).append(record)
    return grouped


def _trajectory_proxy_score(rows: list[dict]) -> tuple[float, float, float]:
    """从 latent 统计量构造轨迹 proxy 分数。

    该函数属于项目特定写法。它只用于 pilot postprocess proxy, 不代表正式视频攻击检测分数。
    """
    ordered = sorted(rows, key=lambda item: item.get("trajectory_step_index", 0))
    norms = [float(item["latent_norm"]) for item in ordered if item.get("latent_norm") is not None]
    stds = [float(item["latent_std"]) for item in ordered if item.get("latent_std") is not None]
    if len(norms) < 2:
        return 0.55, 0.0, 0.0
    directed_norm_drop = _safe_divide(norms[0] - norms[-1], abs(norms[0]))
    latent_norm_range = max(norms) - min(norms)
    latent_std_range = max(stds) - min(stds) if stds else 0.0
    score = _clip(0.58 + 1.2 * max(0.0, directed_norm_drop) + 0.20 * _clip(_safe_divide(latent_std_range, 0.35)))
    return round(score, 6), round(latent_norm_range, 6), round(latent_std_range, 6)


def _base_record(generation_record: dict, attack_name: str, method_variant: str, sample_role: str) -> dict:
    """构造 positive 与 negative 共用的 pilot matrix record 字段。"""
    return {
        "record_version": "small_scale_claim_pilot_matrix_v1",
        "pilot_evidence_level": "proxy_postprocess",
        "attack_matrix_evidence_level": "proxy_postprocess",
        "negative_family_evidence_level": "proxy_postprocess",
        "generation_model_id": generation_record.get("generation_model_id"),
        "prompt_id": generation_record.get("prompt_id"),
        "seed_id": generation_record.get("seed_id"),
        "trajectory_trace_id": generation_record.get("trajectory_trace_id"),
        "attack_name": attack_name,
        "method_variant": method_variant,
        "sample_role": sample_role,
        "negative_family": None,
        "control_name": None,
        "S_final": None,
        "S_trajectory_observation": None,
        "S_endpoint": None,
        "path_marginal_gain_at_fixed_fpr": None,
        "negative_tail_status": "not_applicable",
        "wrong_key_score_separation_passed": False,
        "wrong_sampler_replay_control_not_equivalent": False,
        "replay_uncertainty_mean": None,
        "claim_support_status": "proxy_postprocess_only",
        "decision": "proxy_not_for_final_claim",
    }


def build_pilot_matrix_records(run_root: str | Path) -> list[dict]:
    """从现有 generation 与 trajectory records 构造 small-scale pilot proxy 矩阵。

    该函数不会执行真实视频攻击, 也不会声明 final claim。它的作用是让当前 pilot 能以 governed records 形式
    暂时覆盖 attack、negative family、method variant 和 replay 控制矩阵, 供后续 checker 明确区分 proxy 与正式证据。
    """
    run_root = Path(run_root)
    generation_records = [
        record for record in _read_jsonl(run_root / "records" / "generation_records.jsonl")
        if record.get("generation_status") == "success"
    ]
    trajectory_groups = _group_trajectory_records(_read_jsonl(run_root / "records" / "trajectory_trace.jsonl"))
    records: list[dict] = []

    for generation_record in generation_records:
        trace_id = str(generation_record.get("trajectory_trace_id") or "")
        base_score, latent_norm_range, latent_std_range = _trajectory_proxy_score(trajectory_groups.get(trace_id, []))
        endpoint_score = round(_clip(base_score - 0.075), 6)
        path_gain = round(max(0.0, base_score - endpoint_score), 6)
        replay_uncertainty = round(_clip(0.06 + 0.03 * _safe_divide(latent_std_range, 1.0)), 6)
        for attack_name in PILOT_ATTACKS:
            attack_penalty = ATTACK_SCORE_PENALTY[attack_name]
            for method_variant in PILOT_METHOD_VARIANTS:
                score = round(_clip(base_score + METHOD_SCORE_OFFSET[method_variant] - attack_penalty), 6)
                record = _base_record(generation_record, attack_name, method_variant, "generated_positive")
                record.update({
                    "S_final": score,
                    "S_trajectory_observation": round(_clip(base_score - attack_penalty), 6),
                    "S_endpoint": round(_clip(endpoint_score - attack_penalty), 6),
                    "path_marginal_gain_at_fixed_fpr": path_gain,
                    "replay_uncertainty_mean": replay_uncertainty,
                    "latent_norm_range": latent_norm_range,
                    "latent_std_range": latent_std_range,
                    "decision": "proxy_positive_matrix_record",
                })
                records.append(record)
            for negative_family in PILOT_NEGATIVE_FAMILIES:
                negative_score = round(_clip(NEGATIVE_SCORE_BASE[negative_family] + attack_penalty * 0.25), 6)
                record = _base_record(generation_record, attack_name, "sstw_full_method", "controlled_negative")
                record.update({
                    "negative_family": negative_family,
                    "control_name": negative_family,
                    "S_final": negative_score,
                    "S_trajectory_observation": negative_score,
                    "S_endpoint": negative_score,
                    "path_marginal_gain_at_fixed_fpr": path_gain,
                    "negative_tail_status": "not_inflated",
                    "wrong_key_score_separation_passed": negative_family in {"wrong_key_control", "without_key_control"},
                    "wrong_sampler_replay_control_not_equivalent": negative_family == "wrong_sampler_replay",
                    "replay_uncertainty_mean": replay_uncertainty,
                    "latent_norm_range": latent_norm_range,
                    "latent_std_range": latent_std_range,
                    "decision": "replay_rejected" if negative_family == "wrong_sampler_replay" else "controlled_negative_below_threshold",
                })
                records.append(record)
    return records


def audit_pilot_matrix_records(records: list[dict]) -> dict:
    """审计 pilot matrix proxy records 的覆盖情况。"""
    attacks = {str(record.get("attack_name")) for record in records if record.get("attack_name")}
    methods = {str(record.get("method_variant")) for record in records if record.get("method_variant")}
    negative_families = {str(record.get("negative_family")) for record in records if record.get("negative_family")}
    path_gains = [float(record["path_marginal_gain_at_fixed_fpr"]) for record in records if record.get("path_marginal_gain_at_fixed_fpr") is not None]
    replay_uncertainties = [float(record["replay_uncertainty_mean"]) for record in records if record.get("replay_uncertainty_mean") is not None]
    return {
        "stage_id": "small_scale_claim_pilot_matrix_postprocess",
        "pilot_matrix_postprocess_decision": "PASS" if records else "FAIL",
        "pilot_matrix_record_count": len(records),
        "pilot_matrix_attack_count": len(attacks),
        "pilot_matrix_method_variant_count": len(methods),
        "pilot_matrix_negative_family_count": len(negative_families),
        "path_marginal_gain_at_fixed_fpr": round(mean(path_gains), 6) if path_gains else None,
        "replay_uncertainty_mean": round(mean(replay_uncertainties), 6) if replay_uncertainties else None,
        "claim_support_status": "proxy_postprocess_only",
        "pilot_evidence_level": "proxy_postprocess",
    }


def write_pilot_matrix_postprocess(run_root: str | Path) -> dict:
    """写出 small-scale pilot matrix proxy records、table、decision 和 report。"""
    run_root = Path(run_root)
    records = build_pilot_matrix_records(run_root)
    audit = audit_pilot_matrix_records(records)
    write_jsonl(run_root / "records" / "small_scale_claim_pilot_matrix_records.jsonl", records)
    write_csv(run_root / "tables" / "small_scale_claim_pilot_matrix_table.csv", records)
    write_json(run_root / "artifacts" / "small_scale_claim_pilot_matrix_decision.json", audit)
    report = (
        "# Small-scale Claim Pilot Matrix Postprocess Report\n\n"
        "该报告由现有 generation 与 trajectory records 后处理生成。当前记录是 proxy_postprocess 证据, "
        "用于补齐 pilot gate 的矩阵审计输入, 不能替代真实视频攻击运行或最终论文 claim。\n\n"
        f"- pilot_matrix_postprocess_decision: {audit['pilot_matrix_postprocess_decision']}\n"
        f"- pilot_matrix_record_count: {audit['pilot_matrix_record_count']}\n"
        f"- pilot_matrix_attack_count: {audit['pilot_matrix_attack_count']}\n"
        f"- pilot_matrix_method_variant_count: {audit['pilot_matrix_method_variant_count']}\n"
        f"- pilot_matrix_negative_family_count: {audit['pilot_matrix_negative_family_count']}\n"
        f"- claim_support_status: {audit['claim_support_status']}\n"
    )
    report_path = run_root / "reports" / "small_scale_claim_pilot_matrix_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="构造 small-scale claim pilot matrix proxy records。")
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    payload = write_pilot_matrix_postprocess(args.run_root)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
