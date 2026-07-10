"""逐视频执行完整论文 non-runtime 与 adaptive attack 协议。"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping

from evaluation.attacks.adaptive_video_optimizer import (
    optimize_adaptive_attack_for_video,
    write_cross_video_blend,
)
from evaluation.attacks.video_runtime_attack_protocol import (
    load_protocol_config_with_shared_attack_protocol,
    required_non_runtime_attack_protocols_from_config,
)
from evaluation.protocol.flow_evidence_fields import with_flow_evidence_protocol_defaults
from evaluation.protocol.record_writer import write_json, write_jsonl
from evaluation.protocol.table_builder import write_csv
from experiments.generative_video_model_probe.formal_flow_evidence_runner import (
    _load_pipeline,
    _prompt_text_by_id,
    build_flow_evidence_payload,
)
from main.methods.state_space_watermark.formal_detector import (
    apply_frozen_flow_detector,
    frozen_flow_detector_calibration_from_dict,
)
from main.methods.state_space_watermark.wan_flow_replay_backend import (
    run_wan_attacked_video_replay,
)
from runtime.core.digest import build_stable_digest
from runtime.core.progress import ProgressReporter


DEFAULT_PROTOCOL_CONFIG = "configs/protocol/probe_paper_generative_probe.json"
FORMAL_ADAPTIVE_ATTACK_EVIDENCE_LEVEL = "per_video_frozen_flow_detector_adaptive_execution"

ADAPTIVE_SEARCH_PROTOCOLS: dict[str, tuple[str, tuple[str, ...]]] = {
    "generative_recompression_or_regeneration_attack": (
        "minimize_detector_score",
        ("h264_crf28_runtime", "h265_crf28_runtime", "jpeg_frame_compression_runtime"),
    ),
    "endpoint_preserving_path_perturbation_attack": (
        "minimize_path_with_fixed_endpoint",
        ("frame_swap_adjacent_runtime", "frame_drop_uniform_runtime", "speed_change_runtime", "frame_average_runtime"),
    ),
    "detector_probing_with_public_negatives": (
        "minimize_detector_score",
        ("brightness_contrast_runtime", "color_jitter_runtime", "gaussian_blur_runtime", "jpeg_frame_compression_runtime"),
    ),
    "watermark_removal_optimization_attack": (
        "minimize_detector_score",
        ("denoise_runtime", "gaussian_blur_runtime", "median_blur_runtime", "gaussian_noise_runtime", "jpeg_frame_compression_runtime"),
    ),
    "adversarial_detector_evasion_attack": (
        "minimize_detector_score",
        ("compression_noise_combined_runtime", "compression_crop_combined_runtime", "compression_color_jitter_combined_runtime", "crop_rotation_combined_runtime"),
    ),
}

CONTROL_FIELDS = {
    "flow_time_grid_mismatch_attack": ("time_grid_reliability", "one_minus_reliability"),
    "wrong_sampler_replay_attack": ("wrong_sampler_replay_log_likelihood_ratio", "direct"),
    "wrong_prompt_replay_attack": ("wrong_prompt_replay_log_likelihood_ratio", "direct"),
    "wrong_key_attack": ("wrong_key_replay_log_likelihood_ratio", "direct"),
}


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON 顶层必须是对象: {path}")
    return payload


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def _resolve_video(run_root: Path, raw_path: Any, fallback_dir: str) -> Path:
    path = Path(str(raw_path or ""))
    if path.exists():
        return path
    candidate = run_root / fallback_dir / path.name
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"adaptive attack 输入视频不存在: {raw_path}")


def _profile_context(config_path: str | Path) -> dict[str, Any]:
    config = load_protocol_config_with_shared_attack_protocol(config_path)
    return {
        "config": config,
        "paper_result_level": str(config["paper_result_level"]),
        "target_fpr": float(config["target_fpr"]),
        "required_protocols": tuple(required_non_runtime_attack_protocols_from_config(config)),
        "query_budget": int(config.get("adaptive_attack_query_budget_per_video") or 5),
        "minimum_quality_psnr": float(config.get("adaptive_attack_minimum_quality_psnr") or 24.0),
        "endpoint_tolerance": float(config.get("adaptive_attack_endpoint_tolerance") or 0.08),
    }


def _calibration(run_root: Path) -> Any:
    rows = _read_jsonl(run_root / "thresholds" / "formal_flow_detector_thresholds.jsonl")
    row = next((item for item in rows if item.get("method_variant") == "sstw_full_method"), None)
    if row is None:
        raise RuntimeError("缺少 sstw_full_method 冻结概率后验 threshold artifact")
    if row.get("threshold_source_split") != "calibration" or row.get("test_time_threshold_update_blocked") is not True:
        raise RuntimeError("adaptive attack 只能使用 calibration split 冻结的检测器")
    return frozen_flow_detector_calibration_from_dict(row)


def _one_source_per_video(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """每个独立生成视频只选择一个固定输入, 避免把 runtime attack 重复当新视频。"""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("statistical_cluster_id") or row.get("trajectory_trace_id"))].append(row)
    preference = {"h264_crf18_runtime": 0, "h264_crf23_runtime": 1, "platform_transcode_runtime": 2}
    return [
        min(group, key=lambda row: (preference.get(str(row.get("attack_name")), 100), str(row.get("attack_name"))))
        for _cluster, group in sorted(grouped.items())
    ]


def _build_scorer(
    pipeline: Any,
    calibration: Any,
    *,
    prompt: str,
    key_text: str,
) -> Callable[[Path], Mapping[str, Any]]:
    """把真实视频映射为同一个冻结 Flow 后验, 查询期间不重新拟合。"""

    def score(video_path: Path) -> Mapping[str, Any]:
        replay = run_wan_attacked_video_replay(
            pipeline,
            video_path,
            prompt=prompt,
            key_text=key_text,
        )
        evidence = build_flow_evidence_payload(
            replay,
            key_text=key_text,
            method_variant="sstw_full_method",
        )
        return {
            **evidence,
            **apply_frozen_flow_detector(evidence, calibration),
        }

    return score


def _base_record(
    protocol: str,
    source: Mapping[str, Any],
    context: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "record_version": "formal_per_video_adaptive_attack_v1",
        "paper_result_level": context["paper_result_level"],
        "target_fpr": context["target_fpr"],
        "non_runtime_attack_protocol": protocol,
        "adaptive_attack_name": protocol,
        "generation_model_id": source.get("generation_model_id"),
        "prompt_id": source.get("prompt_id"),
        "seed_id": source.get("seed_id"),
        "trajectory_trace_id": source.get("trajectory_trace_id"),
        "statistical_cluster_id": source.get("statistical_cluster_id"),
        "statistical_independent_unit": "source_video_prompt_seed",
        "split": source.get("split"),
        "method_variant": "sstw_full_method",
        "adaptive_attack_status": "ready",
        "metric_status": "measured_formal",
        "adaptive_attack_evidence_level": FORMAL_ADAPTIVE_ATTACK_EVIDENCE_LEVEL,
        "adaptive_attack_score_orientation": "higher_is_more_watermarked",
        "test_time_threshold_update_blocked": True,
        "adaptive_robustness_claim_allowed": True,
        "claim_support_status": "per_video_adaptive_attack_measured_formal",
    }


def _finalize_record(payload: dict[str, Any]) -> dict[str, Any]:
    digest = build_stable_digest(payload)
    return with_flow_evidence_protocol_defaults(
        {"formal_adaptive_attack_execution_record_id": f"formal_adaptive_attack_{digest[:16]}", **payload},
        trajectory_source_level=FORMAL_ADAPTIVE_ATTACK_EVIDENCE_LEVEL,
        flow_state_admissibility_status="frozen_detector_applied",
        claim_support_status="per_video_adaptive_attack_measured_formal",
    )


def run_formal_adaptive_attack_execution(
    run_root: str | Path,
    config_path: str | Path = DEFAULT_PROTOCOL_CONFIG,
    prompt_suite_path: str | Path | None = None,
    *,
    pipeline_loader: Callable[[str], Any] = _load_pipeline,
) -> dict[str, Any]:
    """对 held-out test 的每个独立视频执行全部预注册协议。"""

    if prompt_suite_path is None:
        raise ValueError("正式 adaptive attack 必须提供 prompt suite")
    root = Path(run_root)
    context = _profile_context(config_path)
    calibration = _calibration(root)
    prompt_map = _prompt_text_by_id(_read_json(prompt_suite_path))
    evidence = _read_jsonl(root / "records" / "formal_flow_evidence_records.jsonl")
    positives = [
        row for row in evidence
        if row.get("sample_role") == "attacked_positive"
        and row.get("method_variant") == "sstw_full_method"
        and row.get("split") == "test"
        and row.get("metric_status") == "measured_formal"
    ]
    clean = [
        row for row in evidence
        if row.get("sample_role") == "clean_negative"
        and row.get("method_variant") == "sstw_full_method"
        and row.get("split") == "test"
    ]
    public_calibration_negatives = [
        row for row in evidence
        if row.get("sample_role") == "clean_negative"
        and row.get("method_variant") == "sstw_full_method"
        and row.get("split") == "calibration"
    ]
    sources = _one_source_per_video(positives)
    if not sources:
        raise RuntimeError("缺少 held-out test full-method 视频, 无法执行 adaptive attack")
    models = sorted({str(row["generation_model_id"]) for row in sources})
    pipelines = {model_id: pipeline_loader(model_id) for model_id in models}
    records: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    progress = ProgressReporter(
        "formal_per_video_adaptive_attack",
        len(sources) * len(context["required_protocols"]),
        "video_protocol",
    )
    progress_index = 0
    for source_index, source in enumerate(sources):
        model_id = str(source["generation_model_id"])
        prompt = prompt_map[str(source["prompt_id"])]
        key_text = f"{model_id}::{source.get('prompt_id')}::{source.get('seed_id')}"
        pipeline = pipelines[model_id]
        scorer = _build_scorer(pipeline, calibration, prompt=prompt, key_text=key_text)
        source_video = _resolve_video(root, source.get("attacked_video_path"), "attacked_videos")
        for protocol in context["required_protocols"]:
            progress_index += 1
            progress.update(progress_index, f"video={source_index} protocol={protocol}")
            base = _base_record(protocol, source, context)
            base["adaptive_attack_input_video_path"] = str(source_video)
            base["adaptive_attack_input_video_sha256"] = source.get("attacked_video_sha256")
            try:
                if protocol in ADAPTIVE_SEARCH_PROTOCOLS:
                    objective, candidate_names = ADAPTIVE_SEARCH_PROTOCOLS[protocol]
                    public_probe_summary: dict[str, Any] = {}
                    if protocol == "detector_probing_with_public_negatives":
                        if not public_calibration_negatives:
                            raise RuntimeError("detector probing 缺少 calibration public negative")
                        public_row = public_calibration_negatives[
                            source_index % len(public_calibration_negatives)
                        ]
                        public_model_id = str(public_row["generation_model_id"])
                        public_prompt = prompt_map[str(public_row["prompt_id"])]
                        public_trial_index = int(public_row.get("clean_negative_trial_index") or 0)
                        public_key = (
                            f"{public_model_id}::{public_row.get('prompt_id')}::{public_row.get('seed_id')}"
                            f"::clean_negative::sstw_full_method::{public_trial_index:06d}"
                        )
                        public_scorer = _build_scorer(
                            pipelines[public_model_id],
                            calibration,
                            prompt=public_prompt,
                            key_text=public_key,
                        )
                        public_video = _resolve_video(
                            root,
                            public_row.get("clean_negative_video_path"),
                            "videos",
                        )
                        public_result = optimize_adaptive_attack_for_video(
                            public_video,
                            root / "adaptive_public_negative_probes" / str(source.get("statistical_cluster_id")),
                            candidate_attack_names=candidate_names,
                            scorer=public_scorer,
                            objective="minimize_detector_score",
                            endpoint_reference=float(public_row["endpoint_score"]),
                            endpoint_tolerance=context["endpoint_tolerance"],
                            minimum_quality_psnr=context["minimum_quality_psnr"],
                            query_budget=context["query_budget"],
                        )
                        candidate_names = tuple(
                            row.attack_name
                            for row in sorted(
                                public_result.candidates,
                                key=lambda row: (row.detector_score, row.candidate_index),
                            )
                        )
                        public_probe_summary = {
                            "adaptive_attack_public_negative_probe_count": len(public_result.candidates),
                            "adaptive_attack_public_negative_cluster_id": public_row.get("statistical_cluster_id"),
                            "adaptive_attack_public_negative_candidate_records": [
                                row.as_dict() for row in public_result.candidates
                            ],
                            "adaptive_attack_public_negative_informed_order": list(candidate_names),
                        }
                    result = optimize_adaptive_attack_for_video(
                        source_video,
                        root / "adaptive_attacked_videos" / str(source.get("statistical_cluster_id")) / protocol,
                        candidate_attack_names=candidate_names,
                        scorer=scorer,
                        objective=objective,
                        endpoint_reference=float(source["endpoint_score"]),
                        endpoint_tolerance=context["endpoint_tolerance"],
                        minimum_quality_psnr=context["minimum_quality_psnr"],
                        query_budget=context["query_budget"],
                    )
                    selected = result.selected
                    payload = {
                        **base,
                        **result.as_dict(),
                        **public_probe_summary,
                        "adaptive_attack_execution_backend": "actual_video_candidate_generation_and_frozen_flow_queries",
                        "adaptive_attack_score": selected.detector_score,
                        "adaptive_attack_path_score": selected.path_score,
                        "adaptive_attack_endpoint_score": selected.endpoint_score,
                        "adaptive_attack_detected_by_sstw": selected.decision,
                        "adaptive_attack_score_semantics": "frozen_calibrated_flow_probability_posterior",
                    }
                    for candidate in result.candidates:
                        query_rows.append({
                            **{key: base.get(key) for key in ("paper_result_level", "non_runtime_attack_protocol", "generation_model_id", "prompt_id", "seed_id", "statistical_cluster_id")},
                            **candidate.as_dict(),
                        })
                elif protocol in CONTROL_FIELDS:
                    field_name, transform = CONTROL_FIELDS[protocol]
                    value = float(source[field_name])
                    score = 1.0 - value if transform == "one_minus_reliability" else value
                    payload = {
                        **base,
                        "adaptive_attack_execution_backend": "per_video_precomputed_key_independent_replay_control",
                        "adaptive_attack_query_count": 1,
                        "adaptive_attack_score": score,
                        "adaptive_attack_score_semantics": field_name,
                        "adaptive_attack_detected_by_sstw": False,
                        "adaptive_attack_output_video_path": str(source_video),
                        "adaptive_attack_output_video_sha256": source.get("attacked_video_sha256"),
                    }
                elif protocol in {"watermark_spoofing_or_copy_attack", "collusion_multi_sample_attack"}:
                    if protocol == "watermark_spoofing_or_copy_attack":
                        if not clean:
                            raise RuntimeError("copy/spoof attack 缺少 held-out clean recipient video")
                        primary = _resolve_video(root, clean[source_index % len(clean)].get("clean_negative_video_path"), "videos")
                        secondary = source_video
                        weight = 0.15
                    else:
                        peer = sources[(source_index + 1) % len(sources)]
                        if peer.get("statistical_cluster_id") == source.get("statistical_cluster_id"):
                            raise RuntimeError("collusion attack 至少需要2个独立视频簇")
                        primary = source_video
                        secondary = _resolve_video(root, peer.get("attacked_video_path"), "attacked_videos")
                        weight = 0.5
                    output_path = root / "adaptive_attacked_videos" / str(source.get("statistical_cluster_id")) / protocol / "cross_video_blend.mp4"
                    blend = write_cross_video_blend(primary, secondary, output_path, secondary_weight=weight)
                    score_payload = dict(scorer(output_path))
                    payload = {
                        **base,
                        **blend,
                        "adaptive_attack_execution_backend": "actual_cross_video_frame_blend_then_frozen_flow_query",
                        "adaptive_attack_query_count": 1,
                        "adaptive_attack_score": float(score_payload["S_final_conservative"]),
                        "adaptive_attack_path_score": float(score_payload["S_path_inv"]),
                        "adaptive_attack_endpoint_score": float(score_payload["endpoint_score"]),
                        "adaptive_attack_detected_by_sstw": bool(score_payload["decision"]),
                        "adaptive_attack_score_semantics": "frozen_calibrated_flow_probability_posterior",
                    }
                else:
                    raise RuntimeError(f"未实现的 formal non-runtime protocol: {protocol}")
                records.append(_finalize_record(payload))
            except Exception as exc:  # pragma: no cover - 依赖真实 GPU、codec 与视频文件
                failure_rows.append({
                    **base,
                    "adaptive_attack_status": "failed",
                    "metric_status": "missing",
                    "adaptive_robustness_claim_allowed": False,
                    "adaptive_attack_failure_reason": str(exc),
                })
    expected_count = len(sources) * len(context["required_protocols"])
    audit = {
        "stage_id": "formal_adaptive_attack_execution",
        "formal_adaptive_attack_execution_decision": "PASS" if len(records) == expected_count and not failure_rows else "FAIL",
        "paper_result_level": context["paper_result_level"],
        "independent_video_count": len(sources),
        "required_non_runtime_attack_protocols": list(context["required_protocols"]),
        "formal_adaptive_attack_execution_record_count": len(records),
        "formal_adaptive_attack_expected_record_count": expected_count,
        "formal_adaptive_attack_query_record_count": len(query_rows),
        "formal_adaptive_attack_failure_count": len(failure_rows),
        "per_video_adaptive_attack_optimization": True,
        "test_time_threshold_update_blocked": True,
        "adaptive_robustness_claim_allowed": len(records) == expected_count and not failure_rows,
    }
    write_jsonl(root / "records" / "formal_adaptive_attack_execution_records.jsonl", records)
    write_jsonl(root / "records" / "formal_adaptive_attack_query_records.jsonl", query_rows)
    write_jsonl(root / "records" / "formal_adaptive_attack_failure_records.jsonl", failure_rows)
    write_csv(root / "tables" / "formal_adaptive_attack_execution_table.csv", records)
    write_csv(root / "tables" / "formal_adaptive_attack_query_table.csv", query_rows)
    write_json(root / "artifacts" / "formal_adaptive_attack_execution_decision.json", audit)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="逐视频执行完整论文 adaptive attack 优化。")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--config-path", default=DEFAULT_PROTOCOL_CONFIG)
    parser.add_argument("--prompt-suite-path", required=True)
    args = parser.parse_args()
    payload = run_formal_adaptive_attack_execution(
        args.run_root,
        args.config_path,
        args.prompt_suite_path,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
