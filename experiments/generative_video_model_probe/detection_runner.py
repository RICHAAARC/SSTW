"""B5 检测记录状态构建与 runtime attacked video 评分。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any

from main.protocol.flow_evidence_fields import with_flow_evidence_protocol_defaults
from main.protocol.record_writer import write_json, write_jsonl
from main.protocol.table_builder import write_csv


def build_detection_records(generation_records: list[dict], attack_records: list[dict]) -> list[dict]:
    """把生成记录与攻击矩阵合并为 detection records, 未运行时不产生正向分数。

    该函数属于早期 readiness 路径的通用占位写法, 用于在真实生成模型不可运行时保留完整矩阵形状。
    它不会读取视频文件, 也不会生成可支持论文 claim 的检测分数。
    """
    records = []
    for generation_record in generation_records:
        for attack_record in attack_records:
            records.append(with_flow_evidence_protocol_defaults({
                "record_version": "generative_video_model_probe_v1",
                "generation_model_id": generation_record["generation_model_id"],
                "prompt_id": generation_record["prompt_id"],
                "seed_id": generation_record["seed_id"],
                "method_variant": "key_conditioned_state_space_with_trajectory",
                "attack_name": attack_record["attack_name"],
                "decision": "not_run",
                "decision_reason": "generation_model_not_runnable",
                "S_final": None,
                "S_trajectory_observation": None,
                "trajectory_gain_over_state_space": None,
                "trajectory_negative_leakage_delta": None,
                "negative_state_over_threshold_count": None,
            },
                trajectory_source_level="not_captured",
                flow_state_admissibility_status="not_evaluated",
                claim_support_status="not_supported_generation_model_not_runnable",
            ))
    return records


def _read_jsonl(path: Path) -> list[dict]:
    """读取 JSONL records, 文件不存在时返回空列表。"""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256_file(path: Path) -> str:
    """计算文件 sha256, 用于确认 detection 输入与 attack 输出没有断链。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """把轻量 proxy 分数裁剪到稳定区间, 避免异常视频导致下游表格不可比较。"""
    return max(lower, min(upper, value))


def _safe_divide(numerator: float, denominator: float) -> float:
    """执行带零值保护的除法。"""
    return 0.0 if denominator == 0 else numerator / denominator


def _load_video_frame_count(video_path: Path) -> tuple[str, int, str]:
    """读取视频帧数。

    该函数只做文件级可解码性检查, 不执行重型神经网络检测。这样可以在默认 pytest 和 Colab 后处理阶段快速验证
    runtime attack 输出是否真的进入 detection 链路。
    """
    try:
        import imageio.v3 as iio

        frame_count = sum(1 for _frame in iio.imiter(video_path))
        return "decoded", frame_count, "none"
    except Exception as exc:  # pragma: no cover - 依赖具体视频编解码后端
        return "decode_failed", 0, str(exc)


def _trajectory_features(trajectory_records: list[dict]) -> dict[str, dict[str, float]]:
    """从 callback trajectory records 中提取可复用的轻量轨迹特征。

    这是项目特定写法: 当前真实 Wan2.1 GPU preflight 能记录 latent 统计量, 但尚未完成正式水印检测器的全量 GPU 复算。
    因此这里仅把 trajectory proxy 作为 runtime detection evidence, 并通过 claim_support_status 明确限制其 claim 边界。
    """
    grouped: dict[str, list[dict]] = {}
    for record in trajectory_records:
        trace_id = record.get("trajectory_trace_id")
        if trace_id:
            grouped.setdefault(str(trace_id), []).append(record)

    features: dict[str, dict[str, float]] = {}
    for trace_id, rows in grouped.items():
        ordered = sorted(rows, key=lambda item: item.get("trajectory_step_index", 0))
        norms = [float(item["latent_norm"]) for item in ordered if item.get("latent_norm") is not None]
        stds = [float(item["latent_std"]) for item in ordered if item.get("latent_std") is not None]
        if len(norms) < 2:
            continue
        directed_norm_drop = _safe_divide(norms[0] - norms[-1], abs(norms[0]))
        latent_norm_range = max(norms) - min(norms)
        latent_std_range = max(stds) - min(stds) if stds else 0.0
        features[trace_id] = {
            "latent_directed_norm_drop": directed_norm_drop,
            "latent_norm_range": latent_norm_range,
            "latent_std_range": latent_std_range,
            "S_trajectory_observation": round(_clip(0.55 + 1.4 * max(0.0, directed_norm_drop) + 0.25 * _clip(_safe_divide(latent_std_range, 0.35))), 6),
        }
    return features


def _runtime_detection_score(runtime_attack_record: dict, feature: dict[str, float] | None, decoded_frame_count: int) -> dict[str, Any]:
    """为单个 attacked video 构造 runtime detection proxy 分数。

    这一实现属于工程闭环层, 不是最终论文级检测算法。它把轨迹 evidence 与攻击后视频文件状态绑定, 用于证明
    attacked video 已经进入检测路径, 并为后续替换成正式检测器保留 records 结构。
    """
    source_frame_count = int(runtime_attack_record.get("source_frame_count") or 0)
    attacked_frame_count = int(runtime_attack_record.get("attacked_frame_count") or decoded_frame_count or 0)
    frame_ratio = round(_safe_divide(attacked_frame_count, source_frame_count), 6)
    decoded_ratio = round(_safe_divide(decoded_frame_count, source_frame_count), 6)
    trajectory_score = float((feature or {}).get("S_trajectory_observation", 0.0))
    temporal_penalty = abs(1.0 - frame_ratio) * 0.12
    decode_penalty = 0.0 if decoded_frame_count > 0 else 0.35
    attack_score = round(_clip(trajectory_score - temporal_penalty - decode_penalty), 6)
    attack_delta = round(attack_score - trajectory_score, 6)
    return {
        "S_trajectory_observation": round(trajectory_score, 6) if feature else None,
        "S_path_inv": round(trajectory_score, 6) if feature else None,
        "S_velocity": round(trajectory_score, 6) if feature else None,
        "S_runtime_attack_detection": attack_score,
        "S_final_conservative": attack_score,
        "source_to_attack_frame_ratio": frame_ratio,
        "decoded_to_source_frame_ratio": decoded_ratio,
        "attack_score_delta": attack_delta,
        "attacked_video_detectable": decoded_frame_count > 0 and attack_score > 0.0,
    }


def build_runtime_detection_records(run_root: str | Path) -> list[dict]:
    """读取 runtime attack outputs 并构造 attacked video detection records。"""
    run_root = Path(run_root)
    runtime_attack_records = _read_jsonl(run_root / "records" / "runtime_attack_records.jsonl")
    trajectory_records = _read_jsonl(run_root / "records" / "trajectory_trace.jsonl")
    features_by_trace = _trajectory_features(trajectory_records)
    records: list[dict] = []

    for attack_record in runtime_attack_records:
        detection_record = with_flow_evidence_protocol_defaults({
            "record_version": "generative_video_runtime_detection_v1",
            "generation_model_id": attack_record.get("generation_model_id"),
            "prompt_id": attack_record.get("prompt_id"),
            "seed_id": attack_record.get("seed_id"),
            "trajectory_trace_id": attack_record.get("trajectory_trace_id"),
            "method_variant": "key_conditioned_state_space_with_trajectory",
            "attack_name": attack_record.get("attack_name"),
            "runtime_detection_evidence_level": "runtime_attacked_video_file",
            "runtime_detection_status": "failed",
            "runtime_detection_failure_reason": "not_run",
            "source_video_path": attack_record.get("source_video_path"),
            "source_video_sha256": attack_record.get("source_video_sha256"),
            "attacked_video_path": attack_record.get("attacked_video_path"),
            "attacked_video_sha256": attack_record.get("attacked_video_sha256"),
            "attacked_video_decode_status": "not_run",
            "attacked_video_decode_failure_reason": "not_run",
            "source_frame_count": attack_record.get("source_frame_count", 0),
            "attacked_frame_count": attack_record.get("attacked_frame_count", 0),
            "attacked_video_decoded_frame_count": 0,
            "decision": "not_run",
            "decision_reason": "runtime_attack_not_ready",
            "claim_support_status": "runtime_detection_evidence_only",
        },
            negative_family=attack_record.get("negative_family"),
            trajectory_source_level="runtime_attacked_video_file_with_callback_trace_proxy",
            flow_state_admissibility_status="not_evaluated",
            claim_support_status="runtime_detection_evidence_only",
        )
        try:
            if attack_record.get("attack_runtime_status") != "ready":
                raise RuntimeError(str(attack_record.get("attack_runtime_failure_reason") or "runtime_attack_not_ready"))
            attacked_video_path = Path(str(attack_record.get("attacked_video_path") or ""))
            if not attacked_video_path.exists():
                raise FileNotFoundError("attacked_video_not_found")
            actual_digest = _sha256_file(attacked_video_path)
            if attack_record.get("attacked_video_sha256") and actual_digest != attack_record.get("attacked_video_sha256"):
                raise RuntimeError("attacked_video_sha256_mismatch")
            decode_status, decoded_frame_count, decode_reason = _load_video_frame_count(attacked_video_path)
            feature = features_by_trace.get(str(attack_record.get("trajectory_trace_id") or ""))
            score_payload = _runtime_detection_score(attack_record, feature, decoded_frame_count)
            detection_record.update({
                "runtime_detection_status": "ready" if decode_status == "decoded" else "failed",
                "runtime_detection_failure_reason": "none" if decode_status == "decoded" else decode_reason,
                "attacked_video_decode_status": decode_status,
                "attacked_video_decode_failure_reason": decode_reason,
                "attacked_video_decoded_frame_count": decoded_frame_count,
                "decision": "runtime_detectable_proxy" if score_payload["attacked_video_detectable"] else "runtime_detection_proxy_below_threshold",
                "decision_reason": "runtime_attacked_video_scored_with_trajectory_proxy",
                "flow_state_admissibility_status": "proxy_admissible"
                if score_payload["attacked_video_detectable"]
                else "proxy_not_admissible",
                **score_payload,
            })
        except Exception as exc:  # pragma: no cover - 依赖实际落盘文件和编解码后端
            detection_record.update({
                "runtime_detection_status": "failed",
                "runtime_detection_failure_reason": str(exc),
                "decision": "runtime_detection_failed",
                "decision_reason": str(exc),
            })
        records.append(detection_record)
    return records


def audit_runtime_detection_records(records: list[dict]) -> dict:
    """审计 runtime detection records 是否完成工程闭环。"""
    ready_records = [record for record in records if record.get("runtime_detection_status") == "ready"]
    detectable_records = [record for record in ready_records if record.get("attacked_video_detectable") is True]
    attack_names = {str(record.get("attack_name")) for record in ready_records if record.get("attack_name")}
    score_values = [float(record["S_runtime_attack_detection"]) for record in ready_records if record.get("S_runtime_attack_detection") is not None]
    return {
        "stage_id": "generative_video_runtime_detection_runner",
        "runtime_detection_decision": "PASS" if records and len(ready_records) == len(records) else "FAIL",
        "runtime_detection_record_count": len(records),
        "runtime_detection_ready_count": len(ready_records),
        "runtime_detection_detectable_count": len(detectable_records),
        "runtime_detection_attack_count": len(attack_names),
        "runtime_detection_score_mean": round(mean(score_values), 6) if score_values else None,
        "runtime_detection_evidence_level": "runtime_attacked_video_file",
        "claim_support_status": "runtime_detection_evidence_only",
    }


def run_runtime_detection(run_root: str | Path) -> dict:
    """执行 runtime attacked video detection 并写出 governed artifacts。"""
    run_root = Path(run_root)
    records = build_runtime_detection_records(run_root)
    audit = audit_runtime_detection_records(records)
    write_jsonl(run_root / "records" / "runtime_detection_records.jsonl", records)
    write_csv(run_root / "tables" / "runtime_detection_table.csv", records)
    write_json(run_root / "artifacts" / "runtime_detection_decision.json", audit)
    report = (
        "# Runtime Detection Runner Report\n\n"
        "该报告记录 attacked videos 进入 runtime detection scoring 链路的工程证据。当前分数是 trajectory proxy 与文件级可解码性绑定后的工程闭环分数, "
        "不能单独支撑最终论文 claim。\n\n"
        f"- runtime_detection_decision: {audit['runtime_detection_decision']}\n"
        f"- runtime_detection_record_count: {audit['runtime_detection_record_count']}\n"
        f"- runtime_detection_ready_count: {audit['runtime_detection_ready_count']}\n"
        f"- runtime_detection_detectable_count: {audit['runtime_detection_detectable_count']}\n"
        f"- runtime_detection_score_mean: {audit['runtime_detection_score_mean']}\n"
        f"- claim_support_status: {audit['claim_support_status']}\n"
    )
    report_path = run_root / "reports" / "runtime_detection_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="对 runtime attacked videos 执行轻量检测评分。")
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    payload = run_runtime_detection(args.run_root)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
