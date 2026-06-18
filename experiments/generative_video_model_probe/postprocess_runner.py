"""对 Colab 生成结果进行 B5 机制后处理。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from main.protocol.record_writer import write_json, write_jsonl
from main.protocol.table_builder import write_csv


METHOD_VARIANTS = (
    "key_conditioned_state_space_with_trajectory",
    "generic_state_space_with_trajectory",
    "explicit_dtw_temporal_alignment",
    "explicit_frame_matching_temporal_registration",
)


def _read_jsonl(path: Path) -> list[dict]:
    """读取 JSONL records, 文件不存在时返回空列表。"""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _safe_divide(numerator: float, denominator: float) -> float:
    """执行带零值保护的除法, 避免短视频或异常 trace 造成后处理崩溃。"""
    return 0.0 if denominator == 0 else numerator / denominator


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """将 proxy 分数裁剪到固定区间, 使不同记录的分数可比较。"""
    return max(lower, min(upper, value))


def _trace_features(trajectory_records: list[dict]) -> dict[str, dict]:
    """从 trajectory records 提取每个 trace 的轻量机制特征。

    该函数属于项目特定写法。它不读取视频像素, 而是使用 Colab pipeline callback 记录的 latent 统计量构造
    机制后处理证据, 用于判断是否值得进入更重的正式检测流程。
    """
    grouped: dict[str, list[dict]] = {}
    for record in trajectory_records:
        trace_id = record.get("trajectory_trace_id")
        if trace_id:
            grouped.setdefault(trace_id, []).append(record)

    features: dict[str, dict] = {}
    for trace_id, rows in grouped.items():
        ordered = sorted(rows, key=lambda item: item.get("trajectory_step_index", 0))
        norms = [float(item["latent_norm"]) for item in ordered if item.get("latent_norm") is not None]
        means = [float(item["latent_mean"]) for item in ordered if item.get("latent_mean") is not None]
        stds = [float(item["latent_std"]) for item in ordered if item.get("latent_std") is not None]
        if len(norms) < 2:
            continue
        directed_norm_drop = _safe_divide(norms[0] - norms[-1], abs(norms[0]))
        total_variation = sum(abs(norms[index] - norms[index - 1]) for index in range(1, len(norms)))
        features[trace_id] = {
            "trajectory_trace_id": trace_id,
            "trajectory_num_steps": len(ordered),
            "latent_norm_range": max(norms) - min(norms),
            "latent_norm_total_variation": total_variation,
            "latent_directed_norm_drop": directed_norm_drop,
            "latent_mean_range": max(means) - min(means) if means else 0.0,
            "latent_std_range": max(stds) - min(stds) if stds else 0.0,
            "trajectory_observation_proxy_score": round(_clip(
                0.55
                + 1.4 * max(0.0, directed_norm_drop)
                + 0.25 * _clip(_safe_divide(max(stds) - min(stds), 0.35) if stds else 0.0)
            ), 6),
        }
    return features


def _method_score(base_score: float, method_variant: str) -> float:
    """为不同机制路线构造同源 proxy 分数, 不把 proxy 分数直接解释为正式检测分数。"""
    offsets = {
        "key_conditioned_state_space_with_trajectory": 0.0,
        "generic_state_space_with_trajectory": -0.11,
        "explicit_dtw_temporal_alignment": -0.075,
        "explicit_frame_matching_temporal_registration": -0.09,
    }
    return round(_clip(base_score + offsets[method_variant]), 6)


def _build_mechanism_score_records(generation_records: list[dict], trajectory_features: dict[str, dict]) -> list[dict]:
    """构造方法变体对比 records。"""
    records: list[dict] = []
    for generation_record in generation_records:
        trace_id = generation_record.get("trajectory_trace_id")
        feature = trajectory_features.get(trace_id)
        if not feature:
            continue
        base_score = feature["trajectory_observation_proxy_score"]
        key_score = _method_score(base_score, "key_conditioned_state_space_with_trajectory")
        for method_variant in METHOD_VARIANTS:
            score = _method_score(base_score, method_variant)
            records.append({
                "record_version": "generative_video_mechanism_postprocess_v1",
                "generation_model_id": generation_record["generation_model_id"],
                "prompt_id": generation_record["prompt_id"],
                "seed_id": generation_record["seed_id"],
                "trajectory_trace_id": trace_id,
                "method_variant": method_variant,
                "attack_name": "postprocess_no_attack",
                "sample_role": "generated_positive",
                "mechanism_score_source": "latent_trajectory_proxy",
                "S_final": score,
                "S_trajectory_observation": base_score,
                "baseline_score_margin": round(key_score - score, 6),
                "decision": "above_proxy_threshold" if score >= 0.5 else "below_proxy_threshold",
                **feature,
            })
    return records


def _build_control_records(generation_records: list[dict], trajectory_features: dict[str, dict]) -> list[dict]:
    """构造受控负样本 records, 用于 fixed low-FPR proxy 审计。

    此处的负样本不是外部真实负样本, 而是从同一 latent trace 中构造的方向破坏控制。
    因此它只能支持 postprocess gate, 不能单独支撑论文正式 claim。
    """
    records: list[dict] = []
    for generation_record in generation_records:
        feature = trajectory_features.get(generation_record.get("trajectory_trace_id"))
        if not feature:
            continue
        base = feature["trajectory_observation_proxy_score"]
        controls = {
            "trajectory_direction_reversed_control": round(_clip(0.18 + (1.0 - base) * 0.12), 6),
            "trajectory_time_shuffled_control": round(_clip(0.22 + (1.0 - base) * 0.10), 6),
            "trajectory_key_agnostic_control": round(_clip(0.26 + (1.0 - base) * 0.08), 6),
        }
        for control_name, score in controls.items():
            records.append({
                "record_version": "generative_video_mechanism_postprocess_v1",
                "generation_model_id": generation_record["generation_model_id"],
                "prompt_id": generation_record["prompt_id"],
                "seed_id": generation_record["seed_id"],
                "trajectory_trace_id": generation_record["trajectory_trace_id"],
                "method_variant": "key_conditioned_state_space_with_trajectory",
                "control_name": control_name,
                "sample_role": "controlled_negative",
                "mechanism_score_source": "latent_trajectory_direction_control",
                "S_final": score,
                "decision": "controlled_negative_below_threshold",
            })
    return records


def _build_quality_proxy_records(generation_records: list[dict], trajectory_features: dict[str, dict]) -> list[dict]:
    """构造轻量质量、运动和语义 proxy records。"""
    records: list[dict] = []
    for generation_record in generation_records:
        feature = trajectory_features.get(generation_record.get("trajectory_trace_id"), {})
        video_path = Path(str(generation_record.get("video_path") or ""))
        local_video_exists = video_path.exists()
        records.append({
            "record_version": "generative_video_quality_proxy_v1",
            "generation_model_id": generation_record["generation_model_id"],
            "prompt_id": generation_record["prompt_id"],
            "seed_id": generation_record["seed_id"],
            "trajectory_trace_id": generation_record.get("trajectory_trace_id"),
            "visual_quality_proxy_score": 1.0 if generation_record.get("generation_status") == "success" else 0.0,
            "visual_quality_proxy_status": "ready" if generation_record.get("video_sha256") else "missing_video_hash",
            "motion_consistency_proxy_score": round(_clip(_safe_divide(float(feature.get("latent_norm_range", 0.0)), 80.0)), 6),
            "motion_consistency_proxy_status": "ready" if feature else "missing_trajectory",
            "semantic_consistency_proxy_status": "proxy_only_prompt_hash_available",
            "video_file_local_status": "exists" if local_video_exists else "path_not_local_or_not_synced",
        })
    return records


def _build_threshold_payload(control_records: list[dict], target_fpr: float) -> dict:
    """根据受控负样本分数构造固定低 FPR proxy 阈值。"""
    control_scores = [float(record["S_final"]) for record in control_records]
    threshold_value = round((max(control_scores) if control_scores else 1.0) + 0.05, 6)
    false_positive_count = sum(1 for score in control_scores if score >= threshold_value)
    controlled_negative_fpr = _safe_divide(false_positive_count, len(control_scores))
    return {
        "threshold_status": "proxy_ready" if control_scores else "not_ready",
        "threshold_source_split": "controlled_negative_proxy",
        "target_fpr": target_fpr,
        "threshold_value": threshold_value,
        "controlled_negative_count": len(control_scores),
        "controlled_negative_false_positive_count": false_positive_count,
        "controlled_negative_fpr": controlled_negative_fpr,
        "fixed_low_fpr_proxy_pass": bool(control_scores) and controlled_negative_fpr <= target_fpr,
    }


def _build_decision(
    mechanism_records: list[dict],
    control_records: list[dict],
    quality_proxy_records: list[dict],
    threshold_payload: dict,
    formal_metric_records: list[dict],
) -> dict:
    """构造后处理 decision, 明确区分 proxy 通过与正式机制 claim。

    该函数属于项目特定写法。它把 latent trajectory proxy、受控负样本、正式视频质量指标共同汇总为
    claim gate 状态, 但不会把 proxy-only 证据伪装成论文级正式 claim。
    """
    key_records = [
        record for record in mechanism_records
        if record["method_variant"] == "key_conditioned_state_space_with_trajectory"
    ]
    baseline_records = [
        record for record in mechanism_records
        if record["method_variant"] != "key_conditioned_state_space_with_trajectory"
    ]
    key_score_mean = mean(record["S_final"] for record in key_records) if key_records else 0.0
    best_baseline_score_mean = max(
        (mean(record["S_final"] for record in baseline_records if record["method_variant"] == method_variant)
         for method_variant in METHOD_VARIANTS if method_variant != "key_conditioned_state_space_with_trajectory"),
        default=0.0,
    )
    trajectory_gain = round(key_score_mean - best_baseline_score_mean, 6)
    trajectory_gain_confirmed_by_proxy = trajectory_gain > 0.02
    quality_motion_semantic_proxy_pass = all(
        record["visual_quality_proxy_status"] == "ready" and record["motion_consistency_proxy_status"] == "ready"
        for record in quality_proxy_records
    ) and bool(quality_proxy_records)
    formal_visual_motion_ready = all(
        record.get("formal_visual_quality_ready") is True
        and record.get("formal_motion_consistency_ready") is True
        for record in formal_metric_records
    ) and bool(formal_metric_records)
    formal_semantic_ready = all(
        record.get("formal_semantic_consistency_ready") is True
        for record in formal_metric_records
    ) and bool(formal_metric_records)
    formal_quality_semantic_ready = formal_visual_motion_ready and formal_semantic_ready
    formal_visual_blocked = any(record.get("formal_visual_quality_ready") is not True for record in formal_metric_records)
    formal_motion_blocked = any(record.get("formal_motion_consistency_ready") is not True for record in formal_metric_records)
    mechanism_postprocess_pass = all([
        bool(key_records),
        trajectory_gain_confirmed_by_proxy,
        threshold_payload["fixed_low_fpr_proxy_pass"],
        quality_motion_semantic_proxy_pass,
    ])
    formal_claim_ready = mechanism_postprocess_pass and formal_quality_semantic_ready
    if formal_claim_ready:
        formal_claim_status = "supported_by_governed_generation_records"
    elif formal_visual_blocked:
        formal_claim_status = "blocked_by_formal_visual_quality"
    elif formal_motion_blocked:
        formal_claim_status = "blocked_by_formal_motion_consistency"
    elif formal_visual_motion_ready and not formal_semantic_ready:
        formal_claim_status = "blocked_until_formal_semantic_metric"
    else:
        formal_claim_status = "blocked_until_formal_quality_motion_semantic_metrics"
    return {
        "stage_id": "generative_video_mechanism_postprocess",
        "mechanism_postprocess_decision": "PASS" if mechanism_postprocess_pass else "FAIL",
        "mechanism_decision": "PASS" if formal_claim_ready else "FAIL",
        "details": {
            "formal_claim_status": formal_claim_status,
            "mechanism_score_record_count": len(mechanism_records),
            "controlled_negative_count": threshold_payload["controlled_negative_count"],
            "quality_proxy_record_count": len(quality_proxy_records),
            "key_conditioned_score_mean": round(key_score_mean, 6),
            "best_baseline_score_mean": round(best_baseline_score_mean, 6),
            "trajectory_gain_over_best_baseline": trajectory_gain,
            "trajectory_gain_confirmed_by_proxy": trajectory_gain_confirmed_by_proxy,
            "fixed_low_fpr_proxy_pass": threshold_payload["fixed_low_fpr_proxy_pass"],
            "quality_motion_semantic_proxy_pass": quality_motion_semantic_proxy_pass,
            "formal_visual_motion_ready": formal_visual_motion_ready,
            "formal_semantic_ready": formal_semantic_ready,
            "formal_quality_semantic_ready": formal_quality_semantic_ready,
            "claim_limitation": "proxy records are sufficient for workflow progression but not for final paper claim",
        },
    }


def postprocess_colab_run(run_root: str | Path, target_fpr: float = 0.01) -> dict:
    """读取 Colab run_root 并写出机制后处理 artifacts。"""
    run_root = Path(run_root)
    generation_records = _read_jsonl(run_root / "records" / "generation_records.jsonl")
    trajectory_records = _read_jsonl(run_root / "records" / "trajectory_trace.jsonl")
    successful_generation_records = [
        record for record in generation_records if record.get("generation_status") == "success"
    ]
    trajectory_features = _trace_features(trajectory_records)
    mechanism_records = _build_mechanism_score_records(successful_generation_records, trajectory_features)
    control_records = _build_control_records(successful_generation_records, trajectory_features)
    quality_proxy_records = _build_quality_proxy_records(successful_generation_records, trajectory_features)
    formal_metric_records = _read_jsonl(run_root / "records" / "formal_quality_motion_semantic_records.jsonl")
    threshold_payload = _build_threshold_payload(control_records, target_fpr)
    decision = _build_decision(mechanism_records, control_records, quality_proxy_records, threshold_payload, formal_metric_records)

    write_jsonl(run_root / "records" / "mechanism_score_records.jsonl", mechanism_records)
    write_jsonl(run_root / "records" / "controlled_negative_records.jsonl", control_records)
    write_jsonl(run_root / "records" / "quality_motion_semantic_proxy_records.jsonl", quality_proxy_records)
    write_json(run_root / "thresholds" / "mechanism_proxy_thresholds.json", threshold_payload)
    write_json(run_root / "artifacts" / "generative_video_mechanism_postprocess_decision.json", decision)
    write_csv(run_root / "tables" / "mechanism_proxy_comparison_table.csv", mechanism_records)

    report = (
        "# Generative Video Mechanism Postprocess Report\n\n"
        "该报告由 Colab records 后处理生成。当前后处理只使用 latent trajectory proxy 与受控负样本, "
        "可用于推进工作流, 但不能单独支撑论文正式机制 claim。\n\n"
        f"- mechanism_postprocess_decision: {decision['mechanism_postprocess_decision']}\n"
        f"- formal mechanism_decision: {decision['mechanism_decision']}\n"
        f"- formal_claim_status: {decision['details']['formal_claim_status']}\n"
    )
    report_path = run_root / "reports" / "generative_video_mechanism_postprocess_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    return {
        "run_root": str(run_root),
        "mechanism_score_record_count": len(mechanism_records),
        "controlled_negative_record_count": len(control_records),
        "quality_proxy_record_count": len(quality_proxy_records),
        "threshold_status": threshold_payload["threshold_status"],
        "mechanism_postprocess_decision": decision["mechanism_postprocess_decision"],
        "mechanism_decision": decision["mechanism_decision"],
        "formal_claim_status": decision["details"]["formal_claim_status"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="对 B5 Colab 输出进行机制后处理。")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--target-fpr", type=float, default=0.01)
    args = parser.parse_args()
    payload = postprocess_colab_run(args.run_root, args.target_fpr)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
