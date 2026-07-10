"""逐视频执行完整论文 non-runtime 与 adaptive attack 协议。"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from hashlib import sha256
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
from evaluation.statistics.clustered_inference import (
    clustered_binary_rate_interval,
    one_sided_binomial_upper_bound,
)
from experiments.generative_video_model_probe.formal_flow_evidence_runner import (
    _load_pipeline,
    _prompt_text_by_id,
    _run_attacked_video_replay_for_model,
    build_flow_evidence_payload,
)
from main.methods.state_space_watermark.formal_detector import (
    FLOW_STATE_POSTERIOR_SCORE_SOURCE,
    apply_frozen_flow_detector,
    frozen_flow_detector_calibration_from_dict,
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

WATERMARK_RETENTION_PROTOCOLS = frozenset(ADAPTIVE_SEARCH_PROTOCOLS) | {
    "collusion_multi_sample_attack",
}
SPOOF_REJECTION_PROTOCOLS = frozenset({"watermark_spoofing_or_copy_attack"})

CONTROL_FIELDS = {
    "flow_time_grid_mismatch_attack": (
        "time_grid_reliability",
        "one_minus_reliability",
        "wrong_sampler_control_margin",
    ),
    "wrong_sampler_replay_attack": (
        "wrong_sampler_replay_log_likelihood_ratio",
        "direct",
        "wrong_sampler_control_margin",
    ),
    "wrong_prompt_replay_attack": (
        "wrong_prompt_replay_log_likelihood_ratio",
        "direct",
        "wrong_prompt_control_margin",
    ),
    "wrong_key_attack": (
        "wrong_key_replay_log_likelihood_ratio",
        "direct",
        "wrong_key_control_margin",
    ),
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


def _sha256_file(path: Path) -> str:
    """计算 adaptive 输入视频摘要, 使每次 detector 查询都可追溯。"""

    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        "minimum_retention_rate": float(
            config.get("minimum_adaptive_watermark_retention_rate") or 0.5
        ),
    }


def _calibrations_by_model(run_root: Path) -> dict[str, Any]:
    """按生成模型隔离加载完整方法冻结后验, 禁止跨模型共用阈值。"""

    rows = _read_jsonl(run_root / "thresholds" / "formal_flow_detector_thresholds.jsonl")
    selected = [item for item in rows if item.get("method_variant") == "sstw_full_method"]
    if not selected:
        raise RuntimeError("缺少 sstw_full_method 冻结概率后验 threshold artifact")
    calibrations: dict[str, Any] = {}
    for row in selected:
        if row.get("threshold_source_split") != "calibration" or row.get("test_time_threshold_update_blocked") is not True:
            raise RuntimeError("adaptive attack 只能使用 calibration split 冻结的检测器")
        model_id = str(row.get("generation_model_id") or "")
        if not model_id or model_id in calibrations:
            raise RuntimeError("adaptive attack 的模型专属冻结检测器标识缺失或重复")
        calibrations[model_id] = frozen_flow_detector_calibration_from_dict(row)
    return calibrations


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


def _disjoint_collusion_peer_index(source_index: int, source_count: int) -> int:
    """返回不重叠两两分组中的另一视频索引, 防止环形配对造成伪重复。"""

    count = int(source_count)
    index = int(source_index)
    if count < 4 or count % 2 != 0:
        raise ValueError("collusion 正式统计至少需要4个视频并按偶数个视频形成不重叠配对")
    if not 0 <= index < count:
        raise IndexError("collusion source index 超出视频范围")
    return index + 1 if index % 2 == 0 else index - 1


def _build_scorer(
    pipeline: Any,
    calibration: Any,
    *,
    prompt: str,
    key_text: str,
) -> Callable[[Path], Mapping[str, Any]]:
    """把真实视频映射为同一个冻结 Flow 后验, 查询期间不重新拟合。"""

    def score(video_path: Path) -> Mapping[str, Any]:
        replay = _run_attacked_video_replay_for_model(
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
    calibrations = _calibrations_by_model(root)
    prompt_map = _prompt_text_by_id(_read_json(prompt_suite_path))
    evidence = _read_jsonl(root / "records" / "formal_flow_evidence_records.jsonl")
    positives = [
        row for row in evidence
        if row.get("sample_role") == "attacked_positive"
        and row.get("method_variant") == "sstw_full_method"
        and row.get("split") == "test"
        and row.get("metric_status") == "measured_formal"
        and row.get("cross_model_role") != "cross_model_validation_model"
    ]
    clean = _one_source_per_video([
        row for row in evidence
        if row.get("sample_role") == "clean_negative"
        and row.get("method_variant") == "sstw_full_method"
        and row.get("split") == "test"
        and row.get("cross_model_role") != "cross_model_validation_model"
    ])
    public_calibration_negatives = _one_source_per_video([
        row for row in evidence
        if row.get("sample_role") == "clean_negative"
        and row.get("method_variant") == "sstw_full_method"
        and row.get("split") == "calibration"
        and row.get("cross_model_role") != "cross_model_validation_model"
    ])
    sources = _one_source_per_video(positives)
    if not sources:
        raise RuntimeError("缺少 held-out test full-method 视频, 无法执行 adaptive attack")
    if (
        "collusion_multi_sample_attack" in context["required_protocols"]
        and (len(sources) < 4 or len(sources) % 2 != 0)
    ):
        _disjoint_collusion_peer_index(0, len(sources))
    clean_by_identity = {
        (
            str(row.get("generation_model_id") or ""),
            str(row.get("prompt_id") or ""),
            str(row.get("seed_id") or ""),
        ): row
        for row in clean
    }
    models = sorted({str(row["generation_model_id"]) for row in sources})
    missing_calibration_models = sorted(set(models) - set(calibrations))
    if missing_calibration_models:
        raise RuntimeError(
            f"adaptive attack 缺少模型专属冻结检测器: {missing_calibration_models}"
        )
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
        scorer = _build_scorer(
            pipeline,
            calibrations[model_id],
            prompt=prompt,
            key_text=key_text,
        )
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
                        if public_model_id not in calibrations:
                            raise RuntimeError("public negative 缺少模型专属冻结检测器")
                        if public_model_id not in pipelines:
                            pipelines[public_model_id] = pipeline_loader(public_model_id)
                        public_prompt = prompt_map[str(public_row["prompt_id"])]
                        public_trial_index = int(public_row.get("clean_negative_trial_index") or 0)
                        public_key = (
                            f"{public_model_id}::{public_row.get('prompt_id')}::{public_row.get('seed_id')}"
                            f"::clean_negative::sstw_full_method::{public_trial_index:06d}"
                        )
                        public_scorer = _build_scorer(
                            pipelines[public_model_id],
                            calibrations[public_model_id],
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
                        for candidate in public_result.candidates:
                            query_rows.append({
                                "paper_result_level": base.get("paper_result_level"),
                                "target_fpr": base.get("target_fpr"),
                                "non_runtime_attack_protocol": protocol,
                                "generation_model_id": public_model_id,
                                "prompt_id": public_row.get("prompt_id"),
                                "seed_id": public_row.get("seed_id"),
                                "statistical_cluster_id": public_row.get(
                                    "statistical_cluster_id"
                                ),
                                "test_time_threshold_update_blocked": True,
                                "adaptive_query_role": "calibration_public_negative",
                                "adaptive_attack_input_video_path": str(public_video),
                                "adaptive_attack_input_video_sha256": _sha256_file(
                                    public_video
                                ),
                                **candidate.as_dict(),
                            })
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
                            **{key: base.get(key) for key in ("paper_result_level", "target_fpr", "non_runtime_attack_protocol", "generation_model_id", "prompt_id", "seed_id", "statistical_cluster_id", "test_time_threshold_update_blocked")},
                            "adaptive_query_role": "heldout_test_video",
                            "adaptive_attack_input_video_path": str(source_video),
                            "adaptive_attack_input_video_sha256": source.get(
                                "attacked_video_sha256"
                            ),
                            **candidate.as_dict(),
                        })
                elif protocol in CONTROL_FIELDS:
                    field_name, transform, margin_field = CONTROL_FIELDS[protocol]
                    value = float(source[field_name])
                    score = 1.0 - value if transform == "one_minus_reliability" else value
                    control_margin = float(source[margin_field])
                    payload = {
                        **base,
                        "adaptive_attack_execution_backend": "per_video_precomputed_key_independent_replay_control",
                        "adaptive_attack_query_count": 1,
                        "adaptive_attack_score": score,
                        "adaptive_attack_score_semantics": field_name,
                        "adaptive_attack_detected_by_sstw": None,
                        "adaptive_attack_control_margin": control_margin,
                        "adaptive_attack_control_rejected": control_margin > 0.0,
                        "adaptive_attack_output_video_path": str(source_video),
                        "adaptive_attack_output_video_sha256": source.get("attacked_video_sha256"),
                    }
                elif protocol in {"watermark_spoofing_or_copy_attack", "collusion_multi_sample_attack"}:
                    if protocol == "watermark_spoofing_or_copy_attack":
                        if not clean:
                            raise RuntimeError("copy/spoof attack 缺少 held-out clean recipient video")
                        recipient = clean_by_identity.get((
                            model_id,
                            str(source.get("prompt_id") or ""),
                            str(source.get("seed_id") or ""),
                        ))
                        if recipient is None:
                            raise RuntimeError("copy/spoof attack 缺少同模型、prompt、seed 的 clean recipient")
                        primary = _resolve_video(
                            root,
                            recipient.get("clean_negative_video_path"),
                            "videos",
                        )
                        secondary = source_video
                        weight = 0.15
                        cross_video_statistics = {
                            "adaptive_attack_donor_statistical_cluster_id": source.get(
                                "statistical_cluster_id"
                            ),
                            "statistical_cluster_id": recipient.get(
                                "statistical_cluster_id"
                            ),
                            "statistical_independent_unit": (
                                "clean_recipient_source_video_prompt_seed"
                            ),
                        }
                    else:
                        peer_index = _disjoint_collusion_peer_index(
                            source_index,
                            len(sources),
                        )
                        peer = sources[peer_index]
                        if peer.get("statistical_cluster_id") == source.get("statistical_cluster_id"):
                            raise RuntimeError("collusion attack 至少需要2个独立视频簇")
                        primary = source_video
                        secondary = _resolve_video(root, peer.get("attacked_video_path"), "attacked_videos")
                        weight = 0.5
                        member_cluster_ids = sorted((
                            str(source.get("statistical_cluster_id") or ""),
                            str(peer.get("statistical_cluster_id") or ""),
                        ))
                        cross_video_statistics = {
                            "adaptive_attack_member_statistical_cluster_ids": member_cluster_ids,
                            "statistical_cluster_id": build_stable_digest({
                                "non_runtime_attack_protocol": protocol,
                                "member_statistical_cluster_ids": member_cluster_ids,
                            }),
                            "statistical_independent_unit": "disjoint_source_video_pair",
                        }
                    output_path = root / "adaptive_attacked_videos" / str(source.get("statistical_cluster_id")) / protocol / "cross_video_blend.mp4"
                    blend = write_cross_video_blend(primary, secondary, output_path, secondary_weight=weight)
                    score_payload = dict(scorer(output_path))
                    query_rows.append({
                        **{
                            key: base.get(key)
                            for key in (
                                "paper_result_level",
                                "target_fpr",
                                "non_runtime_attack_protocol",
                                "generation_model_id",
                                "prompt_id",
                                "seed_id",
                                "statistical_cluster_id",
                                "test_time_threshold_update_blocked",
                            )
                        },
                        **cross_video_statistics,
                        "adaptive_query_role": "heldout_cross_video",
                        "adaptive_attack_input_video_path": str(primary),
                        "adaptive_attack_input_video_sha256": _sha256_file(primary),
                        "adaptive_attack_secondary_input_video_path": str(secondary),
                        "adaptive_attack_secondary_input_video_sha256": _sha256_file(secondary),
                        "video_path": str(output_path),
                        "video_sha256": blend["adaptive_attack_output_video_sha256"],
                        "decoded_frame_count": blend[
                            "adaptive_attack_output_decoded_frame_count"
                        ],
                        "quality_psnr": blend["adaptive_attack_output_quality_psnr"],
                        "detector_score": float(score_payload["S_final_conservative"]),
                        "detector_score_source": str(
                            score_payload.get("flow_detector_score_source")
                            or "unspecified_test_scorer"
                        ),
                        "frozen_final_score_threshold": (
                            float(score_payload["frozen_final_score_threshold"])
                            if score_payload.get("frozen_final_score_threshold") is not None
                            else None
                        ),
                        "threshold_source_split": score_payload.get(
                            "threshold_source_split"
                        ),
                        "test_time_threshold_update_blocked": (
                            score_payload.get("test_time_threshold_update_blocked") is True
                        ),
                        "endpoint_score": float(score_payload["endpoint_score"]),
                        "path_score": float(score_payload["S_path_inv"]),
                        "decision": bool(score_payload["decision"]),
                        "admissible": True,
                    })
                    payload = {
                        **base,
                        **cross_video_statistics,
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
    adaptive_candidate_records = [
        candidate
        for record in records
        for candidate in record.get("adaptive_attack_candidate_records", [])
    ]
    public_probe_candidate_records = [
        candidate
        for record in records
        for candidate in record.get(
            "adaptive_attack_public_negative_candidate_records",
            [],
        )
    ]
    adaptive_query_provenance_ready = bool(query_rows) and all(
        query.get("video_sha256")
        and int(query.get("decoded_frame_count") or 0) > 0
        and query.get("detector_score_source") == FLOW_STATE_POSTERIOR_SCORE_SOURCE
        and query.get("frozen_final_score_threshold") is not None
        and query.get("threshold_source_split") == "calibration"
        and query.get("test_time_threshold_update_blocked") is True
        for query in query_rows
    )
    expected_query_count = sum(
        int(record.get("adaptive_attack_query_count") or 0)
        for record in records
        if record.get("non_runtime_attack_protocol") not in CONTROL_FIELDS
    ) + len(public_probe_candidate_records)
    adaptive_execution_complete = (
        len(records) == expected_count
        and not failure_rows
        and adaptive_query_provenance_ready
        and len(query_rows) == expected_query_count
    )
    retention_rows_by_protocol = {
        protocol: [
            record for record in records
            if record.get("non_runtime_attack_protocol") == protocol
            and record.get("adaptive_attack_detected_by_sstw") is not None
        ]
        for protocol in sorted(WATERMARK_RETENTION_PROTOCOLS)
    }
    retention_statistics: list[dict[str, Any]] = []
    retention_supported = True
    for protocol, protocol_rows in retention_rows_by_protocol.items():
        if not protocol_rows:
            retention_supported = False
            continue
        estimate = clustered_binary_rate_interval(
            protocol_rows,
            outcome_field="adaptive_attack_detected_by_sstw",
            purpose=f"adaptive_retention::{protocol}",
        )
        protocol_supported = (
            estimate.estimate >= context["minimum_retention_rate"]
            and estimate.confidence_interval_lower > context["target_fpr"]
        )
        retention_supported = retention_supported and protocol_supported
        retention_statistics.append({
            "non_runtime_attack_protocol": protocol,
            "adaptive_watermark_retention_decision": (
                "PASS" if protocol_supported else "FAIL"
            ),
            **estimate.as_dict("adaptive_watermark_retention_rate"),
        })
    spoof_rows = [
        record for record in records
        if record.get("non_runtime_attack_protocol") in SPOOF_REJECTION_PROTOCOLS
        and record.get("adaptive_attack_detected_by_sstw") is not None
    ]
    spoof_cluster_outcomes: dict[str, bool] = {}
    for record in spoof_rows:
        cluster_id = str(record.get("statistical_cluster_id") or "")
        if not cluster_id:
            raise RuntimeError("copy/spoof 统计记录缺少 source-video cluster")
        spoof_cluster_outcomes[cluster_id] = (
            spoof_cluster_outcomes.get(cluster_id, False)
            or bool(record.get("adaptive_attack_detected_by_sstw"))
        )
    spoof_false_accept_count = sum(spoof_cluster_outcomes.values())
    spoof_cluster_count = len(spoof_cluster_outcomes)
    spoof_fpr_upper = (
        one_sided_binomial_upper_bound(
            spoof_false_accept_count,
            spoof_cluster_count,
            confidence_level=0.95,
        )
        if spoof_cluster_count
        else 1.0
    )
    spoof_rejection_supported = (
        spoof_cluster_count > 0 and spoof_fpr_upper <= context["target_fpr"]
    )
    control_rows = [
        record for record in records
        if record.get("non_runtime_attack_protocol") in CONTROL_FIELDS
    ]
    replay_controls_supported = bool(control_rows) and all(
        record.get("adaptive_attack_control_rejected") is True
        for record in control_rows
    )
    adaptive_robustness_supported = (
        adaptive_execution_complete
        and retention_supported
        and spoof_rejection_supported
        and replay_controls_supported
    )
    audit = {
        "stage_id": "formal_adaptive_attack_execution",
        "formal_adaptive_attack_execution_decision": (
            "PASS" if adaptive_execution_complete else "FAIL"
        ),
        "paper_result_level": context["paper_result_level"],
        "independent_video_count": len(sources),
        "required_non_runtime_attack_protocols": list(context["required_protocols"]),
        "formal_adaptive_attack_execution_record_count": len(records),
        "formal_adaptive_attack_expected_record_count": expected_count,
        "formal_adaptive_attack_query_record_count": len(query_rows),
        "formal_adaptive_attack_expected_query_record_count": expected_query_count,
        "formal_adaptive_attack_failure_count": len(failure_rows),
        "adaptive_attack_candidate_query_count": len(adaptive_candidate_records),
        "adaptive_attack_public_negative_query_count": len(public_probe_candidate_records),
        "adaptive_attack_query_provenance_decision": (
            "PASS" if adaptive_query_provenance_ready else "FAIL"
        ),
        "adaptive_watermark_retention_minimum_rate": context["minimum_retention_rate"],
        "adaptive_watermark_retention_statistics": retention_statistics,
        "adaptive_watermark_retention_decision": (
            "PASS" if retention_supported else "FAIL"
        ),
        "adaptive_spoof_false_accept_count": spoof_false_accept_count,
        "adaptive_spoof_cluster_count": spoof_cluster_count,
        "adaptive_spoof_fpr_ci_95_upper": round(spoof_fpr_upper, 8),
        "adaptive_spoof_rejection_decision": (
            "PASS" if spoof_rejection_supported else "FAIL"
        ),
        "adaptive_replay_control_rejection_decision": (
            "PASS" if replay_controls_supported else "FAIL"
        ),
        "per_video_adaptive_attack_optimization": True,
        "test_time_threshold_update_blocked": True,
        "adaptive_robustness_claim_allowed": adaptive_robustness_supported,
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
